#!/usr/bin/env python3
"""Fetch China macro indicators via AKShare and populate the macro_indicator table.

Covers the core indicators used in professional brokerage research:
GDP, CPI, PMI (mfg/non-mfg), M1/M2, Social Financing, PPI,
LPR (1Y/5Y), Industrial Production, Trade Balance, Unemployment.

Usage:
    python scripts/fetch_macro_data.py              # fetch all, insert to DB
    python scripts/fetch_macro_data.py --dry-run    # print without inserting
"""

from __future__ import annotations

import logging
import re
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import akshare as ak

from data_pipeline.db_manager import DatabaseManager
from config.settings import Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_macro")

DRY_RUN = "--dry-run" in sys.argv

# ── Indicator definitions ──────────────────────────────────────────
# Each entry: (indicator_name, akshare_func, kwargs, date_col, value_col, date_parser)
INDICATORS = [
    # ── Growth ──
    {
        "name": "GDP年度同比",
        "func": ak.macro_china_gdp_yearly,
        "value_col": "今值",
        "date_col": "日期",
        "date_parser": "release_to_quarter",  # release date → quarter start
    },
    # ── Inflation ──
    {
        "name": "CPI月率",
        "func": ak.macro_china_cpi_monthly,
        "value_col": "今值",
        "date_col": "日期",
        "date_parser": "release_to_prev_month",
    },
    {
        "name": "PPI年率",
        "func": ak.macro_china_ppi_yearly,
        "value_col": "今值",
        "date_col": "日期",
        "date_parser": "release_to_prev_month",
    },
    # ── PMI ──
    {
        "name": "制造业PMI",
        "func": ak.macro_china_pmi,
        "value_col": "制造业-指数",
        "date_col": "月份",
        "date_parser": "chinese_month",
    },
    {
        "name": "非制造业PMI",
        "func": ak.macro_china_non_man_pmi,
        "value_col": "今值",
        "date_col": "日期",
        "date_parser": "release_to_prev_month",
    },
    # ── Money & Credit ──
    {
        "name": "M2同比",
        "func": ak.macro_china_money_supply,
        "value_col": "货币和准货币(M2)-同比增长",
        "date_col": "月份",
        "date_parser": "chinese_month",
    },
    {
        "name": "M1同比",
        "func": ak.macro_china_money_supply,
        "value_col": "货币(M1)-同比增长",
        "date_col": "月份",
        "date_parser": "chinese_month",
    },
    {
        "name": "社会融资规模增量",
        "func": ak.macro_china_shrzgm,
        "value_col": "社会融资规模增量",
        "date_col": "月份",
        "date_parser": "yyyymm",
    },
    # ── Interest Rates ──
    {
        "name": "LPR1Y",
        "func": ak.macro_china_lpr,
        "value_col": "LPR1Y",
        "date_col": "TRADE_DATE",
        "date_parser": "datetime",
        "cache_key": "lpr",
    },
    {
        "name": "LPR5Y",
        "func": ak.macro_china_lpr,
        "value_col": "LPR5Y",
        "date_col": "TRADE_DATE",
        "date_parser": "datetime",
        "cache_key": "lpr",
    },
    # ── Real Economy ──
    {
        "name": "工业增加值年率",
        "func": ak.macro_china_industrial_production_yoy,
        "value_col": "今值",
        "date_col": "日期",
        "date_parser": "release_to_prev_month",
    },
    {
        "name": "贸易差额_亿美元",
        "func": ak.macro_china_trade_balance,
        "value_col": "今值",
        "date_col": "日期",
        "date_parser": "release_to_prev_month",
    },
    {
        "name": "城镇调查失业率",
        "func": ak.macro_china_urban_unemployment,
        "value_col": "value",
        "date_col": "date",
        "date_parser": "yyyymm_filter_urban",
    },
]


# ── Date parsers ───────────────────────────────────────────────────

def _parse_chinese_month(raw: str) -> date | None:
    """'2008年03月份' → date(2008, 3, 1)"""
    m = re.search(r"(\d{4})年(\d{2})月", str(raw))
    if m:
        return date(int(m.group(1)), int(m.group(2)), 1)
    return None


def _parse_yyyymm(raw: str) -> date | None:
    """'202510' → date(2025, 10, 1)"""
    raw = str(raw).strip()
    if len(raw) == 6 and raw.isdigit():
        return date(int(raw[:4]), int(raw[4:6]), 1)
    return None


def _parse_release_to_prev_month(raw: str) -> date | None:
    """Release date '2025-08-09' → observation date for prev month (1st)."""
    try:
        dt = datetime.strptime(str(raw)[:10], "%Y-%m-%d").date()
        # go back 2 months then forward to 1st to handle Jan edge case
        y, m = dt.year, dt.month
        if m == 1:
            return date(y - 1, 12, 1)
        return date(y, m - 1, 1)
    except (ValueError, IndexError):
        return None


def _parse_release_to_quarter(raw: str) -> date | None:
    """Release date → start of that quarter."""
    try:
        dt = datetime.strptime(str(raw)[:10], "%Y-%m-%d").date()
        qtr_month = ((dt.month - 1) // 3) * 3 + 1
        return date(dt.year, qtr_month, 1)
    except (ValueError, IndexError):
        return None


def _parse_datetime(raw) -> date | None:
    """Direct date parse."""
    try:
        if hasattr(raw, "date"):
            return raw.date()
        return datetime.strptime(str(raw)[:10], "%Y-%m-%d").date()
    except (ValueError, IndexError):
        return None


def _parse_yyyymm_urban(raw, item: str = "") -> date | None:
    """'202604' → date(2026, 4, 1), only for urban survey unemployment."""
    raw = str(raw).strip()
    if len(raw) == 6 and raw.isdigit():
        return date(int(raw[:4]), int(raw[4:6]), 1)
    return None


PARSERS = {
    "chinese_month": _parse_chinese_month,
    "yyyymm": _parse_yyyymm,
    "yyyymm_filter_urban": _parse_yyyymm_urban,
    "release_to_prev_month": _parse_release_to_prev_month,
    "release_to_quarter": _parse_release_to_quarter,
    "datetime": _parse_datetime,
}


# ── Main fetch logic ───────────────────────────────────────────────

def fetch_all_indicators() -> list[dict]:
    """Fetch all defined macro indicators. Returns list of {name, date, value} dicts."""
    records: list[dict] = []
    fetch_cache: dict = {}  # deduplicate shared fetches (by cache_key)

    for spec in INDICATORS:
        name = spec["name"]
        cache_key = spec.get("cache_key", name)
        if cache_key in fetch_cache:
            df = fetch_cache[cache_key]
        else:
            logger.info("Fetching %s ...", name)
            try:
                df = spec["func"]()
            except Exception as e:
                logger.warning("  Failed to fetch %s: %s", name, e)
                continue
            if cache_key != name:
                fetch_cache[cache_key] = df

        if df is None or df.empty:
            logger.warning("  %s returned empty DataFrame", name)
            continue

        date_col = spec["date_col"]
        value_col = spec["value_col"]
        parser = PARSERS[spec["date_parser"]]

        for _, row in df.iterrows():
            raw_date = row[date_col]
            raw_val = row[value_col]

            # Skip NaN values
            try:
                if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
                    continue
            except (TypeError, ValueError):
                pass

            obs_date = parser(raw_date)
            if obs_date is None:
                continue

            # Special handling: urban unemployment has multiple items, filter
            if name == "城镇调查失业率":
                item = str(row.get("item", ""))
                if "全国城镇调查失业率" not in item:
                    continue

            try:
                value = float(raw_val)
            except (ValueError, TypeError):
                continue

            records.append({
                "indicator_name": name,
                "obs_date": obs_date,
                "value": value,
            })

        logger.info("  %s: %d records", name,
                     sum(1 for r in records if r["indicator_name"] == name))

    return records


def main() -> None:
    logger.info("Fetching macro indicators from AKShare ...")
    records = fetch_all_indicators()
    logger.info("Total: %d records across %d indicators",
                len(records), len(set(r["indicator_name"] for r in records)))

    if DRY_RUN:
        logger.info("--dry-run: printing sample records")
        for name in sorted(set(r["indicator_name"] for r in records)):
            subset = [r for r in records if r["indicator_name"] == name]
            latest = sorted(subset, key=lambda r: r["obs_date"])[-3:]
            for r in latest:
                print(f"  {r['indicator_name']:20s}  {r['obs_date']}  {r['value']:>12.4f}")
        return

    db = DatabaseManager(Settings())
    with db._session_factory() as sess:
        from sqlalchemy.dialects.mysql import insert as mysql_insert

        from data_pipeline.models import MacroIndicator

        for rec in records:
            stmt = mysql_insert(MacroIndicator).values(
                indicator_name=rec["indicator_name"],
                obs_date=rec["obs_date"],
                value=rec["value"],
            ).on_duplicate_key_update(value=rec["value"])
            sess.execute(stmt)

        sess.commit()
        logger.info("Upserted %d records into macro_indicator", len(records))


if __name__ == "__main__":
    main()
