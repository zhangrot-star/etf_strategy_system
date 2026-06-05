#!/usr/bin/env python3
"""CLI entry point: run a backtest with the ETF strategy system."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime

import backtrader as bt
import pandas as pd

from engine.commissions import CorrectBilateralCommission

from backtest.bt_strategy import ETFStrategy
from backtest.attribution import PerformanceAttribution
from config.settings import Settings
from core.ensemble import XGBoostEnsemble
from core.risk_controller import RiskController
from core.strategy import CoreStrategy
from data_pipeline.db_manager import DatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_backtest")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run ETF strategy backtest")
    parser.add_argument("--tickers", default="SPY,QQQ,IWM,XLK,XLF,XLV", help="Comma-separated ETF tickers")
    parser.add_argument("--start", default="2023-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=1_000_000, help="Initial capital")
    parser.add_argument("--commission", type=float, default=0.0003, help="Commission rate per side")
    parser.add_argument("--freq", default="monthly", choices=["daily", "weekly", "monthly"], help="Rebalance frequency")
    parser.add_argument("--model-path", default=None, help="Path to pre-trained XGBoost model")
    args = parser.parse_args(argv)

    settings = Settings()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end) if args.end else date.today()

    # ── Load data ────────────────────────────
    db = DatabaseManager(settings)
    prices = db.load_prices(tickers, start_date, end_date)
    if prices.empty:
        logger.error("No price data found in database. Run run_pipeline.py first.")
        sys.exit(1)

    logger.info("Loaded %d price rows for %d tickers.", len(prices), len(tickers))

    # ── Setup components ─────────────────────
    ensemble = XGBoostEnsemble(settings)
    if args.model_path:
        ensemble.load(args.model_path)

    risk_ctrl = RiskController(settings)
    orchestrator = CoreStrategy(
        config={"risk": {"max_positions": 10}},
    )

    # ── Build Backtrader cerebro ─────────────
    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        ETFStrategy,
        orchestrator=orchestrator,
        features_df=pd.DataFrame(),
        sentiment_df=pd.DataFrame(),
        rebalance_freq=args.freq,
    )
    cerebro.broker.setcash(args.capital)

    # Add data feeds
    price_pivot = prices.pivot_table(index="trade_date", columns="ticker", values="close")
    for ticker in tickers:
        df = prices[prices["ticker"] == ticker].set_index("trade_date")
        if df.empty:
            continue
        feed = bt.feeds.PandasData(
            dataname=df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}),
            name=ticker,
        )
        cerebro.adddata(feed)

    # Commission & slippage
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)
    comm_info = CorrectBilateralCommission(commission=args.commission)
    cerebro.broker.addcommissioninfo(comm_info)
    cerebro.broker.set_slippage_perc(perc=0.0001)  # 1 bp default slippage

    # Analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.03)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name="annreturn")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    # ── Run ──────────────────────────────────
    logger.info("Starting backtest: %s → %s, capital=%.0f", start_date, end_date, args.capital)
    init_value = cerebro.broker.getvalue()
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    strategy = results[0]

    total_ret = (final_value / init_value) - 1
    total_ann = (1 + total_ret) ** (252 / max((end_date - start_date).days, 1)) - 1

    # ── Print results ────────────────────────
    print("\n" + "=" * 60)
    print("  ETF Strategy Backtest Results")
    print("=" * 60)
    print(f"  Period:          {start_date} → {end_date}")
    print(f"  Initial Capital: ${init_value:,.0f}")
    print(f"  Final Value:     ${final_value:,.0f}")
    print(f"  Total Return:    {total_ret:.2%}")
    print(f"  Annual Return:   {total_ann:.2%}")

    sharpe = strategy.analyzers.sharpe.get_analysis()
    print(f"  Sharpe Ratio:    {sharpe.get('sharperatio', 'N/A')}")

    dd = strategy.analyzers.drawdown.get_analysis()
    print(f"  Max Drawdown:    {dd.get('max', {}).get('drawdown', 0):.2%}")

    ann = strategy.analyzers.annreturn.get_analysis()
    print(f"  Annual Return:   {ann.get('R', 'N/A')}")

    trades = strategy.analyzers.trades.get_analysis()
    print(f"  Total Trades:    {trades.get('total', {}).get('total', 0)}")
    won = trades.get("won", {}).get("total", 0)
    lost = trades.get("lost", {}).get("total", 0)
    if won + lost > 0:
        print(f"  Win Rate:        {won / (won + lost):.1%}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
