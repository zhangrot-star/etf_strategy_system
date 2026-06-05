"""Tests for data_pipeline module."""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.cleaner import DataCleaner


class TestDataCleaner:
    def test_forward_fill_fills_gaps(self, sample_prices_df):
        cleaner = DataCleaner()
        # Introduce a NaN
        df = sample_prices_df.copy()
        df.loc[10, "close"] = None
        cleaned = cleaner.clean_etf_prices(df)
        assert cleaned is not None
        assert cleaned["close"].isna().sum() == 0

    def test_winsorize_clips_extremes(self):
        cleaner = DataCleaner(lower_quantile=0.10, upper_quantile=0.90)
        df = pd.DataFrame({
            "ticker": ["SPY"] * 100,
            "trade_date": pd.bdate_range("2023-01-01", periods=100),
            "open": [100] * 100,
            "high": [100] * 100,
            "low": [100] * 100,
            "close": [1, 10_000] + [100] * 98,  # extreme outliers
            "volume": [1_000_000] * 100,
        })
        cleaned = cleaner.clean_etf_prices(df)
        assert cleaned["close"].min() > 1
        assert cleaned["close"].max() < 10_000

    def test_align_sentiment_to_prices_no_future_leak(self):
        prices = pd.DataFrame({
            "ticker": ["SPY"] * 5,
            "trade_date": pd.bdate_range("2023-01-01", periods=5),
            "close": [100, 101, 102, 103, 104],
        })
        sentiment = pd.DataFrame({
            "ticker": ["SPY"],
            "event_date": [pd.Timestamp("2023-01-06")],  # after last price date
            "polarity": [-0.8],
            "confidence": [0.9],
            "event_category": ["geopolitical"],
        })
        merged = DataCleaner.align_sentiment_to_prices(prices, sentiment)
        # Sentiment after the last price date should not be merged
        assert "polarity" in merged.columns

    def test_empty_df_handled(self):
        cleaner = DataCleaner()
        result = cleaner.clean_etf_prices(pd.DataFrame())
        assert result.empty


class TestFetcher:
    def test_fetcher_instantiation(self):
        from data_pipeline.fetcher import AShareETFDataFetcher

        f = AShareETFDataFetcher(request_delay=0.5, timeout=60)
        assert f._delay == 0.5
        assert f._timeout == 60
