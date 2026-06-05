#!/usr/bin/env python3
"""End-to-end pipeline test: scoring + ML modulation + sentiment risk control."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from data_pipeline.db_manager import DatabaseManager
from data_pipeline.models import ETFPrice, SentimentRecord
from config.settings import Settings
from recommendation.pipeline import DailyRecommendationPipeline
from sqlalchemy import select

db = DatabaseManager(Settings())

with db._engine.connect() as conn:
    result = conn.execute(select(ETFPrice.ticker).distinct())
    tickers = sorted([r[0] for r in result.fetchall()])
    srows = conn.execute(select(SentimentRecord).order_by(SentimentRecord.event_date.desc()).limit(1000))
    sentiment = pd.DataFrame([dict(r._mapping) for r in srows.fetchall()])

prices = db.load_prices(tickers, pd.Timestamp("2024-06-01"), pd.Timestamp("2026-06-01"))

# Run pipeline + also check internals
from scoring.etf_scorer import ETFScorer
from scoring.modulation import MLScoreModulator
from core.feature_utils import build_features_from_prices
from core.strategy import CoreStrategy
from datetime import date

scorer = ETFScorer()
raw_scores = scorer.score_all(prices, score_date=date.today())

# Get ML signals
features = build_features_from_prices(prices)
strategy = CoreStrategy()
strategy.ensemble.load("models/xgboost_etf")
latest_date = features.index.get_level_values(1).max()
latest_features = features.xs(latest_date, level=1)
result = strategy.allocate(latest_features, pd.DataFrame(), date.today())
predictions = {s.ticker: (s.signal, max(s.prob_buy, s.prob_hold, s.prob_sell)) for s in result.signals}

# Modulate with sentiment
modulator = MLScoreModulator()
modulated = modulator.modulate_dataframe(raw_scores.copy(), predictions, sentiment)

# Full pipeline for comparison
pipeline = DailyRecommendationPipeline()
pipeline_result = pipeline.run(prices, sentiment)

print(f"\nPrices: {len(prices)} rows x {prices['ticker'].nunique()} tickers")
print(f"Sentiment records: {len(sentiment)}")
if not sentiment.empty:
    print(f"Mean polarity: {sentiment['polarity'].mean():.3f}")

print(f"\n{'='*90}")
print(f"  ETF Recommendation — {pipeline_result.date}")
print(f"  Risk Status: {pipeline_result.risk_status}  |  Cash: {pipeline_result.cash_weight:.1%}")
print(f"{'='*90}")
print(f"  {'Ticker':<8} {'Raw':>6} {'Adj':>6} {'Factor':>7} {'ML':>6} {'Rec':>14} {'Alloc':>6} {'Rating':>6}")
print(f"  {'-'*78}")

for _, r in modulated.head(15).iterrows():
    weight = pipeline_result.ranked_etfs[0:15]
    wt = next((e.allocation_weight for e in pipeline_result.ranked_etfs if e.ticker == r["ticker"]), 0)
    print(f"  {r['ticker']:<8} {r['raw_total']:>6.1f} {r['adjusted_total']:>6.1f} "
          f"{r['modulation_factor']:>7.3f} {r['ml_signal']:>6} {r['recommendation']:>14} "
          f"{wt:>6.1%} {r.get('rating','-'):>6}")

# Risk details
risk_status = pipeline_result.risk_status
if risk_status == "WARNING":
    print(f"\n  WARNING: Single position cap = 15%")
elif risk_status == "BREACHED":
    print(f"\n  BREACHED: Equity forced to 50%, position cap = 10%")
else:
    print(f"\n  NORMAL: Full allocation, no caps")

# Count modulated rows
ml_affected = sum(modulated["modulation_factor"] != 1.0)
sent_affected = sum(modulated["modulation_factor"] < 0.99)
print(f"  ML modulation affected: {ml_affected}/22  |  Sentiment penalty: {sent_affected}/22")
print(f"{'='*90}")
