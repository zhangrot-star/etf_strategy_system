#!/usr/bin/env python3
"""Seed ETFIssuer / IndexMeta / ETFProfile tables with realistic mock data."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.db_manager import DatabaseManager
from data_pipeline.models import ETFIssuer, IndexMeta, ETFProfile
from config.settings import Settings
from sqlalchemy import delete

# ── Fund company data ──────────────────────────────────────────────

ISSUERS = [
    {"issuer_id": "chinaamc",      "name": "华夏基金", "aum_rank": 1,  "roe": 0.22, "industry_median_roe": 0.15},
    {"issuer_id": "efund",          "name": "易方达基金", "aum_rank": 2,  "roe": 0.25, "industry_median_roe": 0.15},
    {"issuer_id": "southern",       "name": "南方基金", "aum_rank": 3,  "roe": 0.18, "industry_median_roe": 0.15},
    {"issuer_id": "guotai",         "name": "国泰基金", "aum_rank": 4,  "roe": 0.20, "industry_median_roe": 0.15},
    {"issuer_id": "htpinebridge",   "name": "华泰柏瑞基金", "aum_rank": 8,  "roe": 0.16, "industry_median_roe": 0.15},
    {"issuer_id": "fullgoal",       "name": "富国基金", "aum_rank": 9,  "roe": 0.19, "industry_median_roe": 0.15},
    {"issuer_id": "huaan",          "name": "华安基金", "aum_rank": 10, "roe": 0.17, "industry_median_roe": 0.15},
    {"issuer_id": "hwabao",         "name": "华宝基金", "aum_rank": 15, "roe": 0.14, "industry_median_roe": 0.15},
]

# ── ETF profile + index meta per ticker ────────────────────────────

# Format: ticker → {profile dict, index_meta dict, issuer_id}
ETF_DATA = {
    "510050": {
        "profile": {"name": "华夏上证50ETF", "issuer_id": "chinaamc", "inception_date": date(2004, 12, 30), "expense_ratio": 0.0050, "aum": 1.2e11, "avg_daily_volume": 2.5e9},
        "index_meta": {"index_code": "000016", "tracking_error": 0.008, "is_public": True, "n_constituents": 50, "has_transparent_rebal": True, "rebal_quarterly": True, "dividend_yield": 0.028, "category_div_yield_median": 0.025, "premium_discount_std": 0.003},
    },
    "510300": {
        "profile": {"name": "华泰柏瑞沪深300ETF", "issuer_id": "htpinebridge", "inception_date": date(2012, 5, 4), "expense_ratio": 0.0050, "aum": 1.8e11, "avg_daily_volume": 3.2e9},
        "index_meta": {"index_code": "000300", "tracking_error": 0.006, "is_public": True, "n_constituents": 300, "has_transparent_rebal": True, "rebal_quarterly": True, "dividend_yield": 0.022, "category_div_yield_median": 0.025, "premium_discount_std": 0.002},
    },
    "510500": {
        "profile": {"name": "南方中证500ETF", "issuer_id": "southern", "inception_date": date(2013, 2, 6), "expense_ratio": 0.0050, "aum": 8.0e10, "avg_daily_volume": 1.5e9},
        "index_meta": {"index_code": "000905", "tracking_error": 0.010, "is_public": True, "n_constituents": 500, "has_transparent_rebal": True, "rebal_quarterly": True, "dividend_yield": 0.018, "category_div_yield_median": 0.018, "premium_discount_std": 0.004},
    },
    "159915": {
        "profile": {"name": "易方达创业板ETF", "issuer_id": "efund", "inception_date": date(2011, 9, 20), "expense_ratio": 0.0050, "aum": 5.0e10, "avg_daily_volume": 1.8e9},
        "index_meta": {"index_code": "399006", "tracking_error": 0.012, "is_public": True, "n_constituents": 100, "has_transparent_rebal": True, "rebal_quarterly": True, "dividend_yield": 0.010, "category_div_yield_median": 0.012, "premium_discount_std": 0.005},
    },
    "588000": {
        "profile": {"name": "华夏科创50ETF", "issuer_id": "chinaamc", "inception_date": date(2020, 9, 28), "expense_ratio": 0.0050, "aum": 6.0e10, "avg_daily_volume": 1.2e9},
        "index_meta": {"index_code": "000688", "tracking_error": 0.015, "is_public": True, "n_constituents": 50, "has_transparent_rebal": True, "rebal_quarterly": True, "dividend_yield": 0.008, "category_div_yield_median": 0.012, "premium_discount_std": 0.006},
    },
    "159845": {
        "profile": {"name": "华夏中证1000ETF", "issuer_id": "chinaamc", "inception_date": date(2021, 3, 18), "expense_ratio": 0.0050, "aum": 3.0e10, "avg_daily_volume": 6.0e8},
        "index_meta": {"index_code": "000852", "tracking_error": 0.014, "is_public": True, "n_constituents": 1000, "has_transparent_rebal": True, "rebal_quarterly": True, "dividend_yield": 0.012, "category_div_yield_median": 0.012, "premium_discount_std": 0.006},
    },
    "515050": {
        "profile": {"name": "华夏中证5G通信ETF", "issuer_id": "chinaamc", "inception_date": date(2019, 9, 30), "expense_ratio": 0.0050, "aum": 4.0e10, "avg_daily_volume": 9.0e8},
        "index_meta": {"index_code": "931079", "tracking_error": 0.016, "is_public": True, "n_constituents": 50, "has_transparent_rebal": True, "rebal_quarterly": True, "dividend_yield": 0.006, "category_div_yield_median": 0.012, "premium_discount_std": 0.007},
    },
    "159995": {
        "profile": {"name": "华夏国证半导体芯片ETF", "issuer_id": "chinaamc", "inception_date": date(2020, 1, 20), "expense_ratio": 0.0050, "aum": 5.5e10, "avg_daily_volume": 1.5e9},
        "index_meta": {"index_code": "990001", "tracking_error": 0.018, "is_public": True, "n_constituents": 30, "has_transparent_rebal": True, "rebal_quarterly": True, "dividend_yield": 0.004, "category_div_yield_median": 0.012, "premium_discount_std": 0.008},
    },
    "159819": {
        "profile": {"name": "易方达中证人工智能ETF", "issuer_id": "efund", "inception_date": date(2020, 7, 2), "expense_ratio": 0.0050, "aum": 3.5e10, "avg_daily_volume": 7.0e8},
        "index_meta": {"index_code": "990002", "tracking_error": 0.017, "is_public": True, "n_constituents": 50, "has_transparent_rebal": True, "rebal_quarterly": True, "dividend_yield": 0.003, "category_div_yield_median": 0.012, "premium_discount_std": 0.008},
    },
    "512720": {
        "profile": {"name": "富国中证计算机ETF", "issuer_id": "fullgoal", "inception_date": date(2019, 7, 11), "expense_ratio": 0.0050, "aum": 2.0e10, "avg_daily_volume": 4.0e8},
        "index_meta": {"index_code": "930651", "tracking_error": 0.016, "is_public": True, "n_constituents": 50, "has_transparent_rebal": True, "rebal_quarterly": True, "dividend_yield": 0.005, "category_div_yield_median": 0.012, "premium_discount_std": 0.007},
    },
    "516510": {
        "profile": {"name": "易方达中证云计算ETF", "issuer_id": "efund", "inception_date": date(2021, 4, 14), "expense_ratio": 0.0050, "aum": 1.5e10, "avg_daily_volume": 3.0e8},
        "index_meta": {"index_code": "930712", "tracking_error": 0.017, "is_public": True, "n_constituents": 40, "has_transparent_rebal": True, "rebal_quarterly": True, "dividend_yield": 0.002, "category_div_yield_median": 0.012, "premium_discount_std": 0.008},
    },
    "512880": {
        "profile": {"name": "国泰中证证券ETF", "issuer_id": "guotai", "inception_date": date(2016, 7, 26), "expense_ratio": 0.0050, "aum": 4.5e10, "avg_daily_volume": 1.0e9},
        "index_meta": {"index_code": "399975", "tracking_error": 0.009, "is_public": True, "n_constituents": 50, "has_transparent_rebal": True, "rebal_quarterly": True, "dividend_yield": 0.020, "category_div_yield_median": 0.025, "premium_discount_std": 0.003},
    },
    "512690": {
        "profile": {"name": "富国中证酒ETF", "issuer_id": "fullgoal", "inception_date": date(2019, 4, 24), "expense_ratio": 0.0050, "aum": 2.5e10, "avg_daily_volume": 5.0e8},
        "index_meta": {"index_code": "399987", "tracking_error": 0.011, "is_public": True, "n_constituents": 30, "has_transparent_rebal": True, "rebal_quarterly": True, "dividend_yield": 0.025, "category_div_yield_median": 0.025, "premium_discount_std": 0.005},
    },
    "512010": {
        "profile": {"name": "华夏中证医药ETF", "issuer_id": "chinaamc", "inception_date": date(2013, 9, 13), "expense_ratio": 0.0050, "aum": 3.0e10, "avg_daily_volume": 6.0e8},
        "index_meta": {"index_code": "000933", "tracking_error": 0.010, "is_public": True, "n_constituents": 80, "has_transparent_rebal": True, "rebal_quarterly": True, "dividend_yield": 0.018, "category_div_yield_median": 0.025, "premium_discount_std": 0.004},
    },
    "516970": {
        "profile": {"name": "易方达中证军工ETF", "issuer_id": "efund", "inception_date": date(2021, 7, 8), "expense_ratio": 0.0050, "aum": 1.2e10, "avg_daily_volume": 2.5e8},
        "index_meta": {"index_code": "399967", "tracking_error": 0.013, "is_public": True, "n_constituents": 50, "has_transparent_rebal": True, "rebal_quarterly": True, "dividend_yield": 0.008, "category_div_yield_median": 0.012, "premium_discount_std": 0.006},
    },
    "512660": {
        "profile": {"name": "国泰中证军工ETF", "issuer_id": "guotai", "inception_date": date(2016, 7, 26), "expense_ratio": 0.0050, "aum": 2.0e10, "avg_daily_volume": 4.5e8},
        "index_meta": {"index_code": "399967", "tracking_error": 0.012, "is_public": True, "n_constituents": 50, "has_transparent_rebal": True, "rebal_quarterly": True, "dividend_yield": 0.009, "category_div_yield_median": 0.012, "premium_discount_std": 0.005},
    },
    "515790": {
        "profile": {"name": "华夏中证光伏ETF", "issuer_id": "chinaamc", "inception_date": date(2020, 12, 7), "expense_ratio": 0.0050, "aum": 3.5e10, "avg_daily_volume": 8.0e8},
        "index_meta": {"index_code": "931151", "tracking_error": 0.019, "is_public": True, "n_constituents": 50, "has_transparent_rebal": True, "rebal_quarterly": True, "dividend_yield": 0.012, "category_div_yield_median": 0.012, "premium_discount_std": 0.009},
    },
    "512800": {
        "profile": {"name": "华宝中证银行ETF", "issuer_id": "hwabao", "inception_date": date(2017, 6, 28), "expense_ratio": 0.0050, "aum": 3.0e10, "avg_daily_volume": 6.0e8},
        "index_meta": {"index_code": "399986", "tracking_error": 0.008, "is_public": True, "n_constituents": 40, "has_transparent_rebal": True, "rebal_quarterly": True, "dividend_yield": 0.038, "category_div_yield_median": 0.025, "premium_discount_std": 0.003},
    },
    "515220": {
        "profile": {"name": "华夏中证煤炭ETF", "issuer_id": "chinaamc", "inception_date": date(2020, 12, 18), "expense_ratio": 0.0050, "aum": 2.0e10, "avg_daily_volume": 4.0e8},
        "index_meta": {"index_code": "399998", "tracking_error": 0.014, "is_public": True, "n_constituents": 30, "has_transparent_rebal": True, "rebal_quarterly": True, "dividend_yield": 0.045, "category_div_yield_median": 0.025, "premium_discount_std": 0.006},
    },
    "511010": {
        "profile": {"name": "国泰上证国债ETF", "issuer_id": "guotai", "inception_date": date(2013, 3, 5), "expense_ratio": 0.0030, "aum": 1.0e10, "avg_daily_volume": 2.0e8},
        "index_meta": {"index_code": "000012", "tracking_error": 0.005, "is_public": True, "n_constituents": 100, "has_transparent_rebal": True, "rebal_quarterly": False, "dividend_yield": 0.032, "category_div_yield_median": 0.032, "premium_discount_std": 0.002},
    },
    "511260": {
        "profile": {"name": "华泰柏瑞上证10年国债ETF", "issuer_id": "htpinebridge", "inception_date": date(2017, 8, 4), "expense_ratio": 0.0030, "aum": 8.0e9, "avg_daily_volume": 1.5e8},
        "index_meta": {"index_code": "000013", "tracking_error": 0.006, "is_public": True, "n_constituents": 50, "has_transparent_rebal": True, "rebal_quarterly": False, "dividend_yield": 0.035, "category_div_yield_median": 0.032, "premium_discount_std": 0.003},
    },
    "518880": {
        "profile": {"name": "华安黄金ETF", "issuer_id": "huaan", "inception_date": date(2013, 7, 29), "expense_ratio": 0.0050, "aum": 2.0e11, "avg_daily_volume": 2.0e9},
        "index_meta": {"index_code": "AU9999", "tracking_error": 0.004, "is_public": True, "n_constituents": 1, "has_transparent_rebal": False, "rebal_quarterly": False, "dividend_yield": 0.0, "category_div_yield_median": 0.0, "premium_discount_std": 0.008},
    },
}

# ── Main ───────────────────────────────────────────────────────────

def main():
    db = DatabaseManager(Settings())

    with db.session() as sess:
        # Clear existing data
        sess.execute(delete(ETFProfile))
        sess.execute(delete(IndexMeta))
        sess.execute(delete(ETFIssuer))
        print("Cleared existing metadata.")

        # Insert issuers
        for iss in ISSUERS:
            sess.add(ETFIssuer(**iss))
        print(f"Inserted {len(ISSUERS)} issuers.")

        # Insert profiles and index metadata
        for ticker, data in ETF_DATA.items():
            prof = ETFProfile(ticker=ticker, **data["profile"])
            sess.add(prof)
            meta = IndexMeta(ticker=ticker, **data["index_meta"])
            sess.add(meta)
        print(f"Inserted {len(ETF_DATA)} profiles + index_meta records.")

    print("Done — metadata seeded successfully.")


if __name__ == "__main__":
    main()
