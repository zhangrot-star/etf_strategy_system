#!/usr/bin/env python3
"""Generate direct sentiment data from price momentum (no LLM needed).

For each ticker, computes trailing 21d return and maps to polarity via a
sigmoid function.  This is a practical stand-in for LLM sentiment; real
deployment would replace this with Claude/DeepSeek API calls via the
sentiment/ package.
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
from data_pipeline.models import ETFPrice
from config.settings import Settings
from sqlalchemy import select

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("gen_sentiment")

# ── Load price data ──────────────────────────────────────────────

db = DatabaseManager(Settings())

with db._engine.connect() as conn:
    result = conn.execute(select(ETFPrice.ticker).distinct())
    tickers = sorted([r[0] for r in result.fetchall()])

all_prices = db.load_prices(
    tickers, pd.Timestamp("2024-01-01"), pd.Timestamp("2026-06-01")
)
all_prices["trade_date"] = pd.to_datetime(all_prices["trade_date"])

logger.info("Loaded %d rows for %d tickers", len(all_prices), len(tickers))

# ── Generate sentiment from momentum ─────────────────────────────

MOMENTUM_WINDOW = 21
records: list[dict] = []

# Category templates for variety
CATEGORIES = {
    "strong_positive": ("market_sentiment", "大幅上涨，资金持续流入"),
    "positive": ("technical_signal", "技术指标向好，短期动能增强"),
    "weak_positive": ("sector_rotation", "板块轮动中受益，温和上涨"),
    "neutral": ("other", "市场横盘整理，多空胶着"),
    "weak_negative": ("macro_data", "宏观不确定性增加，小幅承压"),
    "negative": ("technical_signal", "技术指标转弱，短期下行风险"),
    "strong_negative": ("market_sentiment", "恐慌情绪蔓延，资金大幅流出"),
}

for ticker in tickers:
    t_data = all_prices[all_prices["ticker"] == ticker].sort_values("trade_date")
    if len(t_data) < MOMENTUM_WINDOW + 5:
        continue

    close = t_data["close"].values
    dates = t_data["trade_date"].values

    # Generate monthly sentiment records
    for i in range(MOMENTUM_WINDOW, len(close), 21):  # every ~month
        # Trailing 21d return
        mom_ret = close[i] / close[i - MOMENTUM_WINDOW] - 1

        # Recent volatility (for confidence)
        recent_rets = np.diff(close[i - MOMENTUM_WINDOW : i + 1]) / close[i - MOMENTUM_WINDOW : i]
        vol = np.std(recent_rets) if len(recent_rets) > 0 else 0.02

        # Map momentum to polarity using sigmoid
        # mom_ret=+5% → ~+0.7, mom_ret=-5% → ~-0.7
        polarity = float(np.tanh(mom_ret * 15))
        polarity = round(np.clip(polarity, -1.0, 1.0), 3)

        # Confidence from inverse volatility (low vol = high confidence)
        confidence = round(float(np.clip(1.0 - vol * 50, 0.3, 0.95)), 3)

        # Category based on momentum strength
        if polarity > 0.5:
            cat, summary = CATEGORIES["strong_positive"]
        elif polarity > 0.2:
            cat, summary = CATEGORIES["positive"]
        elif polarity > 0.05:
            cat, summary = CATEGORIES["weak_positive"]
        elif polarity > -0.05:
            cat, summary = CATEGORIES["neutral"]
        elif polarity > -0.2:
            cat, summary = CATEGORIES["weak_negative"]
        elif polarity > -0.5:
            cat, summary = CATEGORIES["negative"]
        else:
            cat, summary = CATEGORIES["strong_negative"]

        record_date = pd.Timestamp(dates[i]).date()

        records.append({
            "ticker": ticker,
            "event_date": record_date,
            "polarity": polarity,
            "confidence": confidence,
            "event_category": cat,
            "summary": f"{summary}。21日动量={mom_ret:+.2%}，波动率={vol:.2%}",
            "raw_response": f'{{"polarity": {polarity}, "confidence": {confidence}, "event_category": "{cat}"}}',
        })

sentiment_df = pd.DataFrame(records)
logger.info("Generated %d sentiment records", len(sentiment_df))

# ── Store to MySQL ───────────────────────────────────────────────

db.upsert_sentiment(sentiment_df)
logger.info("Stored sentiment to MySQL.")

# ── Quick stats ──────────────────────────────────────────────────

print(f"\nSentiment distribution:")
print(f"  Records: {len(sentiment_df)}")
print(f"  Polarity range: [{sentiment_df['polarity'].min():.2f}, {sentiment_df['polarity'].max():.2f}]")
print(f"  Mean polarity: {sentiment_df['polarity'].mean():.3f}")
print(f"  Positive: {(sentiment_df['polarity'] > 0.05).sum()}")
print(f"  Neutral:  {((sentiment_df['polarity'] >= -0.05) & (sentiment_df['polarity'] <= 0.05)).sum()}")
print(f"  Negative: {(sentiment_df['polarity'] < -0.05).sum()}")
print(f"  WARNING level (<-0.3): {(sentiment_df['polarity'] < -0.3).sum()}")
print(f"  BREACH level (<-0.5): {(sentiment_df['polarity'] < -0.5).sum()}")
