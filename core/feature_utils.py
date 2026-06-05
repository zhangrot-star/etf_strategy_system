"""Shared feature and label construction utilities.

Single source of truth — replaces duplicated logic previously spread across:
- engine/backtest.py:_build_features_and_labels
- strategy/core_strategy.py:_build_features_and_labels
- strategy/optimizer.py:_build_optimization_dataset
- scripts/run_demo.py:compute_simple_factors
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from core.ensemble import XGBoostEnsemble
from factors.technical import TechnicalFactorBuilder

logger = logging.getLogger(__name__)


def build_features_from_prices(
    prices: pd.DataFrame,
    sentiment: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute technical factors from OHLCV prices and optionally merge sentiment.

    Returns wide-format DataFrame indexed by (ticker, trade_date) with factors
    as columns.
    """
    if prices.empty:
        return pd.DataFrame()

    builder = TechnicalFactorBuilder()
    factors_long = builder.build_all_factors(prices)

    if factors_long.empty:
        return pd.DataFrame()

    features = factors_long.pivot_table(
        index=["ticker", "trade_date"],
        columns="factor_name",
        values="value",
    ).dropna(how="all")

    if sentiment is not None and not sentiment.empty:
        features = _merge_sentiment_features(features, sentiment)

    return features


def build_labels_from_prices(
    prices: pd.DataFrame,
    forward_window: int = 21,
    sell_quantile: float = 0.33,
    buy_quantile: float = 0.67,
) -> pd.Series:
    """Compute forward-return labels from OHLCV prices.

    Returns Series indexed by (ticker, trade_date) with values 0=SELL, 1=HOLD, 2=BUY.
    """
    if prices.empty:
        return pd.Series(dtype=int)

    labels_list: list[dict[str, Any]] = []
    for _ticker, group in prices.groupby("ticker"):
        group = group.sort_values("trade_date")
        group["fwd_ret"] = group["close"].pct_change(forward_window).shift(-forward_window)
        for _, row in group.iterrows():
            labels_list.append({
                "ticker": row["ticker"],
                "trade_date": row["trade_date"],
                "label": row["fwd_ret"],
            })

    labels_df = pd.DataFrame(labels_list).dropna(subset=["label"])
    labels_df = labels_df.set_index(["ticker", "trade_date"])
    return XGBoostEnsemble.labels_from_forward_returns(labels_df["label"])


def build_features_and_labels(
    prices: pd.DataFrame,
    sentiment: pd.DataFrame | None = None,
    forward_window: int = 21,
    sell_quantile: float = 0.33,
    buy_quantile: float = 0.67,
) -> tuple[pd.DataFrame, pd.Series]:
    """Combined feature and label construction, aligned on shared index."""
    features = build_features_from_prices(prices, sentiment)
    labels = build_labels_from_prices(prices, forward_window, sell_quantile, buy_quantile)

    if features.empty or labels.empty:
        return pd.DataFrame(), pd.Series(dtype=int)

    common = features.index.intersection(labels.index)
    return features.loc[common], labels.loc[common]


def _merge_sentiment_features(
    features: pd.DataFrame,
    sentiment: pd.DataFrame,
) -> pd.DataFrame:
    """Merge sentiment data into the feature matrix via backward as-of join."""
    if "event_date" not in sentiment.columns:
        return features

    features = features.reset_index()
    sentiment = sentiment.rename(columns={"event_date": "trade_date"})
    sentiment["trade_date"] = pd.to_datetime(sentiment["trade_date"])

    if "trade_date" in features.columns:
        features["trade_date"] = pd.to_datetime(features["trade_date"])

    merged = pd.merge_asof(
        features.sort_values("trade_date"),
        sentiment[["ticker", "trade_date", "polarity", "confidence"]].sort_values("trade_date"),
        on="trade_date",
        by="ticker",
        direction="backward",
        tolerance=pd.Timedelta("7D"),
    )

    merged["polarity"] = merged["polarity"].fillna(0.0)
    merged["confidence"] = merged["confidence"].fillna(0.0)

    return merged.set_index(["ticker", "trade_date"])
