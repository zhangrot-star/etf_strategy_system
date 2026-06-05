"""Data cleaning: forward-fill, winsorization, and timestamp alignment."""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """Enforces data quality guarantees before storage or model consumption.

    Pipeline order:
    1. Sort and drop duplicates
    2. Forward-fill missing values within each ticker group
    3. Winsorize at specified quantile bounds
    4. Timestamp alignment (as-of merge) between structured & unstructured sources
    """

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> None:
        self.lower = lower_quantile
        self.upper = upper_quantile

    # ── Main pipeline ───────────────────────────────────────

    def clean_etf_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run full cleaning pipeline on raw ETF price data."""
        if df.empty:
            return df
        df = self._sort_and_dedup(df, group_col="ticker", date_col="trade_date")
        df = self._forward_fill(df, group_col="ticker", date_col="trade_date")
        df = self._winsorize(df, value_cols=["open", "high", "low", "close", "volume"])
        return df

    def clean_macro(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run cleaning on macro indicator data."""
        if df.empty:
            return df
        df = self._sort_and_dedup(df, group_col="indicator_name", date_col="obs_date")
        df = self._forward_fill(df, group_col="indicator_name", date_col="obs_date")
        df = self._winsorize(df, value_cols=["value"])
        return df

    # ── Core methods ────────────────────────────────────────

    def _sort_and_dedup(
        self, df: pd.DataFrame, group_col: str, date_col: str
    ) -> pd.DataFrame:
        df = df.sort_values([group_col, date_col]).reset_index(drop=True)
        before = len(df)
        df = df.drop_duplicates(subset=[group_col, date_col], keep="last")
        if len(df) < before:
            logger.debug("Dropped %d duplicate rows.", before - len(df))
        return df

    def _forward_fill(
        self, df: pd.DataFrame, group_col: str, date_col: str
    ) -> pd.DataFrame:
        """Forward-fill missing values within each group, respecting date order.

        A full date range is reindexed per group to catch gaps that would leak
        future information.
        """
        filled_frames: list[pd.DataFrame] = []
        for _, group in df.groupby(group_col):
            group = group.set_index(date_col).sort_index()
            full_range = pd.date_range(start=group.index.min(), end=group.index.max(), freq="B")
            group = group.reindex(full_range)
            group.index.name = date_col
            group = group.ffill()
            filled_frames.append(group.reset_index())

        result = pd.concat(filled_frames, ignore_index=True)
        logger.debug("Forward-filled data: %d rows after reindex.", len(result))
        return result

    def _winsorize(self, df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
        """Clip extreme values at the lower/upper quantile bounds per column."""
        for col in value_cols:
            if col not in df.columns:
                continue
            lo = df[col].quantile(self.lower)
            hi = df[col].quantile(self.upper)
            before = ((df[col] < lo) | (df[col] > hi)).sum()
            df[col] = df[col].clip(lower=lo, upper=hi)
            if before:
                logger.debug("Winsorized %d extreme values in column='%s'.", before, col)
        return df

    # ── Timestamp alignment ─────────────────────────────────

    @staticmethod
    def align_sentiment_to_prices(
        prices: pd.DataFrame,
        sentiment: pd.DataFrame,
        price_date_col: str = "trade_date",
        sent_date_col: str = "event_date",
        tolerance: str = "3D",
    ) -> pd.DataFrame:
        """Merge sentiment records onto price panel via as-of merge.

        sentiment records are mapped to the nearest prior (or same-day)
        price date within `tolerance`.  Forward-direction is not allowed
        to prevent look-ahead bias.
        """
        if prices.empty or sentiment.empty:
            return prices

        prices = prices.copy()
        sentiment = sentiment.copy()

        prices["_date_parsed"] = pd.to_datetime(prices[price_date_col])
        sentiment["_date_parsed"] = pd.to_datetime(sentiment[sent_date_col])

        prices = prices.sort_values("_date_parsed")
        sentiment = sentiment.sort_values("_date_parsed")

        merged = pd.merge_asof(
            prices,
            sentiment,
            left_on="_date_parsed",
            right_on="_date_parsed",
            by="ticker",
            direction="backward",
            tolerance=pd.Timedelta(tolerance),
        )

        merged = merged.drop(columns=["_date_parsed"])
        return merged
