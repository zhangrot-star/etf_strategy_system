"""Shared test fixtures."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_prices_df() -> pd.DataFrame:
    """Generate a realistic multi-ticker OHLCV panel."""
    tickers = ["SPY", "QQQ", "IWM"]
    dates = pd.bdate_range("2023-01-01", "2023-06-30")
    np.random.seed(42)

    rows = []
    for t in tickers:
        base_price = {"SPY": 380, "QQQ": 280, "IWM": 175}[t]
        price = base_price
        for d in dates:
            ret = np.random.normal(0.0003, 0.012)
            price *= 1 + ret
            rows.append({
                "ticker": t,
                "trade_date": d.date(),
                "open": round(price * (1 - np.random.uniform(0, 0.003)), 2),
                "high": round(price * (1 + np.random.uniform(0, 0.008)), 2),
                "low": round(price * (1 - np.random.uniform(0, 0.008)), 2),
                "close": round(price, 2),
                "volume": np.random.randint(500_000, 10_000_000),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def sample_sentiment_responses() -> list[dict]:
    """Sample Claude API JSON responses."""
    return [
        {"polarity": 0.7, "confidence": 0.85, "event_category": "monetary_policy", "key_entities": ["SPY", "QQQ"], "summary": "Dovish tone."},
        {"polarity": -0.9, "confidence": 0.92, "event_category": "geopolitical", "key_entities": ["IWM"], "summary": "Escalation fears."},
        {"polarity": 0.1, "confidence": 0.30, "event_category": "other", "key_entities": [], "summary": "Unclear impact."},
    ]


@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    """Synthetic feature matrix for XGBoost training."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "roc_5d": np.random.randn(n) * 0.02,
        "rsi_14d": np.random.uniform(20, 80, n),
        "atr_21d": np.random.uniform(0.5, 5, n),
        "volume_ma_ratio": np.random.uniform(0.5, 2.0, n),
        "polarity": np.random.uniform(-1, 1, n),
        "confidence": np.random.uniform(0, 1, n),
        "cat_monetary_policy": np.random.choice([0, 1], n),
        "cat_geopolitical": np.random.choice([0, 1], n),
    })


@pytest.fixture
def sample_labels() -> pd.Series:
    """Synthetic 3-class labels."""
    np.random.seed(42)
    return pd.Series(np.random.choice([0, 1, 2], 200))
