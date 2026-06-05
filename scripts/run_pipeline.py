#!/usr/bin/env python3
"""CLI entry point: fetch → clean → store pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime

from config.settings import Settings
from data_pipeline.cleaner import DataCleaner
from data_pipeline.db_manager import DatabaseManager
from data_pipeline.fetcher import DataFetcherFactory
try:
    from data_pipeline.vector_store import VectorStoreManager
except ImportError:
    VectorStoreManager = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_pipeline")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ETF data pipeline — fetch, clean, and store")
    parser.add_argument("--tickers", default="SPY,QQQ,IWM,XLK,XLF,XLV", help="Comma-separated ETF ticker codes")
    parser.add_argument("--start", default="2023-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--macro", action="store_true", help="Also fetch macro indicators")
    parser.add_argument("--market", default="US", choices=["US", "A"], help="Market: US (yfinance) or A (akshare)")
    parser.add_argument("--drop-tables", action="store_true", help="Drop and recreate all tables")
    args = parser.parse_args(argv)

    settings = Settings()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end) if args.end else date.today()

    # ── Database setup ────────────────────────
    db = DatabaseManager(settings)
    if args.drop_tables:
        db.drop_all()
    db.create_all()

    # ── Fetch ETF prices ─────────────────────
    logger.info("Fetching %s ETF prices for %d tickers: %s", args.market, len(tickers), tickers)
    fetcher = DataFetcherFactory.create(args.market)
    prices = fetcher.fetch(tickers, start_date, end_date)
    if prices.empty:
        logger.error("No ETF price data fetched. Check tickers and date range, or try a different market.")
        sys.exit(1)

    # ── Clean ────────────────────────────────
    cleaner = DataCleaner(*settings.winsorize_bounds)
    prices_clean = cleaner.clean_etf_prices(prices)
    logger.info("Cleaned prices: %d rows", len(prices_clean))

    # ── Store to MySQL ───────────────────────
    db.upsert_prices(prices_clean)
    logger.info("Stored ETF prices to MySQL.")

    # ── Macro indicators (optional) ──────────
    if args.macro:
        macro = fetcher.fetch_macro(start=start_date, end=end_date)
        if not macro.empty:
            macro_clean = cleaner.clean_macro(macro)
            db.upsert_macro(macro_clean)
            logger.info("Stored %d macro rows.", len(macro_clean))

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
