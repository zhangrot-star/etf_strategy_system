#!/usr/bin/env python3
"""Validate ETF scoring framework: does it predict forward returns?

At each month-end, computes scores from trailing 252d price history, then
tracks forward 1m returns. Reports hit rate and rank correlation.
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.db_manager import DatabaseManager
from config.settings import Settings
from scoring.etf_scorer import ETFScorer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("validate_scoring")

MIN_LOOKBACK = 252  # trading days before first scoring
MIN_SAMPLES = 126   # minimum price data needed per ticker

# ── Load data ────────────────────────────────────────────────────

db = DatabaseManager(Settings())

# Discover all available tickers from the database
from data_pipeline.models import ETFPrice
from sqlalchemy import select
with db._engine.connect() as conn:
    result = conn.execute(select(ETFPrice.ticker).distinct())
    tickers = sorted([r[0] for r in result.fetchall()])

all_prices = db.load_prices(
    tickers, pd.Timestamp("2024-01-01"), pd.Timestamp("2026-06-01")
)

if all_prices.empty:
    logger.error("No price data — run data_pipeline first.")
    sys.exit(1)

all_prices["trade_date"] = pd.to_datetime(all_prices["trade_date"])
logger.info("Loaded %d rows, %d tickers", len(all_prices), all_prices["ticker"].nunique())

# ── Monthly scoring walk-forward ──────────────────────────────────

scorer = ETFScorer()
all_dates = sorted(all_prices["trade_date"].unique())
trade_dates = pd.DatetimeIndex(all_dates)

# Find month-end dates that have enough history
month_ends: list[pd.Timestamp] = []
for dt in trade_dates:
    if len(trade_dates[trade_dates <= dt]) >= MIN_LOOKBACK:
        # Check if this is the last trading day of its month
        month = dt.month
        future_dates = [d for d in trade_dates if d > dt and d.month == month]
        if len(future_dates) == 0:
            month_ends.append(dt)

logger.info("Scoring at %d month-end points: %s → %s", len(month_ends), month_ends[0].date(), month_ends[-1].date())

# ── Walk forward ────────────────────────────────────────────────

records: list[dict] = []

for i, score_dt in enumerate(month_ends):
    # Use all data up to this date
    hist_prices = all_prices[all_prices["trade_date"] <= score_dt].copy()

    # Each ticker needs enough data
    ticker_dates = hist_prices.groupby("ticker").size()
    valid_tickers = ticker_dates[ticker_dates >= MIN_SAMPLES].index.tolist()
    if len(valid_tickers) < 2:
        continue

    hist_prices = hist_prices[hist_prices["ticker"].isin(valid_tickers)]

    # Compute scores
    scores_df = scorer.score_all(hist_prices, score_date=score_dt.date())
    if scores_df.empty:
        continue

    # Find forward 1-month return for each ticker
    # Pick end date: score_dt + 1 month (or next available month-end)
    if i + 1 < len(month_ends):
        fwd_end = month_ends[i + 1]
    else:
        fwd_end = trade_dates[-1]

    for _, row in scores_df.iterrows():
        ticker = row["ticker"]
        # Price at scoring date
        start_price = hist_prices[
            (hist_prices["ticker"] == ticker) & (hist_prices["trade_date"] == score_dt)
        ]["close"].values
        if len(start_price) == 0:
            continue

        # Forward price
        fwd_data = all_prices[
            (all_prices["ticker"] == ticker)
            & (all_prices["trade_date"] > score_dt)
            & (all_prices["trade_date"] <= fwd_end)
        ]["close"]
        if len(fwd_data) < 5:
            continue

        fwd_ret = float(fwd_data.iloc[-1] / start_price[0] - 1)

        records.append({
            "score_date": score_dt,
            "ticker": ticker,
            "raw_total": row["raw_total"],
            "fund_module": row.get("fund_module_total", 0),
            "fwd_return": fwd_ret,
        })

if not records:
    logger.error("No scoring records produced — check data.")
    sys.exit(1)

results = pd.DataFrame(records)
logger.info("Generated %d score-return pairs across %d periods",
            len(results), results["score_date"].nunique())

# ── Analysis ──────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  Scoring Framework Validation")
print("=" * 65)

# 1. Hit rate: percentage of months where top-score ETF > median forward return
hit_months = 0
total_months = 0
for dt, grp in results.groupby("score_date"):
    if len(grp) < 2:
        continue
    grp_sorted = grp.sort_values("raw_total", ascending=False)
    top_score = grp_sorted.iloc[0]
    median_ret = grp_sorted["fwd_return"].median()
    total_months += 1
    if top_score["fwd_return"] > median_ret:
        hit_months += 1

hit_rate = hit_months / total_months * 100 if total_months else 0
print(f"\n  Top-score > median return:   {hit_months}/{total_months} ({hit_rate:.0f}%)")

# 2. Long/short spread: top half vs bottom half
long_rets: list[float] = []
short_rets: list[float] = []
for dt, grp in results.groupby("score_date"):
    if len(grp) < 3:
        continue
    grp_sorted = grp.sort_values("raw_total", ascending=False)
    n = len(grp_sorted)
    top_half = grp_sorted.iloc[: n // 2]["fwd_return"].mean()
    bot_half = grp_sorted.iloc[-(n // 2):]["fwd_return"].mean()
    long_rets.append(top_half)
    short_rets.append(bot_half)

long_arr = np.array(long_rets)
short_arr = np.array(short_rets)
spread_mean = (long_arr - short_arr).mean() * 100
spread_std = (long_arr - short_arr).std() * 100
spread_hit = (long_arr > short_arr).mean() * 100

print(f"  Long (top-half) mean return: {long_arr.mean() * 100:+.2f}%")
print(f"  Short (bot-half) mean return: {short_arr.mean() * 100:+.2f}%")
print(f"  Spread mean:                  {spread_mean:+.2f}% (std {spread_std:.2f}%)")
print(f"  Spread > 0 months:            {spread_hit:.0f}%")

# 3. Rank correlation (Spearman per period)
spearman_vals: list[float] = []
for dt, grp in results.groupby("score_date"):
    if len(grp) < 3:
        continue
    rho = grp["raw_total"].corr(grp["fwd_return"], method="spearman")
    if not np.isnan(rho):
        spearman_vals.append(rho)

s_arr = np.array(spearman_vals)
s_pos = (s_arr > 0).mean() * 100
print(f"  Mean Spearman ρ:              {s_arr.mean():+.3f}")
print(f"  Positive ρ months:            {s_pos:.0f}%")

# 4. Per-ticker grid: average score vs average forward return
print("\n  ── Per-Ticker Summary ──")
ticker_summary = results.groupby("ticker").agg(
    avg_score=("raw_total", "mean"),
    avg_fwd_ret=("fwd_return", "mean"),
    months=("fwd_return", "count"),
).sort_values("avg_score", ascending=False)
for ticker, r in ticker_summary.iterrows():
    print(f"  {ticker}: score={r['avg_score']:.1f}  fwd_ret={r['avg_fwd_ret']:+.2%}  n={int(r['months'])}")

# 5. Monthly detail
print("\n  ── Monthly Detail ──")
for dt, grp in results.groupby("score_date"):
    grp_sorted = grp.sort_values("raw_total", ascending=False)
    top_t = grp_sorted.iloc[0]
    bot_t = grp_sorted.iloc[-1]
    spread = (grp_sorted.iloc[:len(grp_sorted)//2]["fwd_return"].mean()
              - grp_sorted.iloc[-(len(grp_sorted)//2):]["fwd_return"].mean())
    print(f"  {dt.date()}  top={top_t['ticker']}({top_t['raw_total']:.0f},{top_t['fwd_return']:+.2%})  "
          f"bot={bot_t['ticker']}({bot_t['raw_total']:.0f},{bot_t['fwd_return']:+.2%})  "
          f"spread={spread:+.2%}")

print("\n" + "=" * 65)
