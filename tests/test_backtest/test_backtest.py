"""Tests for backtesting module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.attribution import PerformanceAttribution, AttributionResult


class TestPerformanceAttribution:
    @pytest.fixture
    def sample_returns(self) -> tuple[pd.Series, pd.Series]:
        np.random.seed(42)
        dates = pd.bdate_range("2023-01-01", "2023-12-31")
        n = len(dates)
        market_rets = pd.Series(np.random.normal(0.0003, 0.01, n), index=dates)
        # Portfolio: beta 1.2 market + alpha + noise
        portfolio_rets = 1.2 * market_rets + 0.0002 + np.random.normal(0, 0.005, n)
        return portfolio_rets, market_rets

    def test_decompose_produces_metrics(self, sample_returns):
        portfolio, market = sample_returns
        attr = PerformanceAttribution()
        result = attr.decompose(portfolio, market)
        assert isinstance(result, AttributionResult)
        assert result.market_beta is not None
        assert result.sharpe_ratio is not None
        assert "Sharpe" in result.summary

    def test_decompose_with_ff_factors(self, sample_returns):
        portfolio, market = sample_returns
        dates = pd.bdate_range("2023-01-01", "2023-12-31")
        ff = pd.DataFrame({
            "Mkt-RF": market.values - 0.03 / 252,
            "SMB": np.random.normal(0.0001, 0.005, len(dates)),
            "HML": np.random.normal(0.0001, 0.004, len(dates)),
        }, index=dates)
        attr = PerformanceAttribution()
        result = attr.decompose(portfolio, market, ff_factors=ff)
        assert not np.isnan(result.ff3_alpha)
        assert not np.isnan(result.ff3_hml_beta)
        assert not np.isnan(result.ff3_smb_beta)

    def test_decompose_with_sentiment_events(self, sample_returns):
        portfolio, market = sample_returns
        dates = pd.bdate_range("2023-01-01", "2023-12-31")
        sentiment_events = pd.DataFrame({
            "date": dates[:30],
            "is_breached": np.random.choice([True, False], 30, p=[0.1, 0.9]),
        })
        attr = PerformanceAttribution()
        result = attr.decompose(portfolio, market, sentiment_events=sentiment_events)
        assert result.sentiment_breach_count is not None
        assert "Sentiment" in result.summary

    def test_empty_returns(self):
        attr = PerformanceAttribution()
        result = attr.decompose(pd.Series(), pd.Series())
        assert result.total_return == 0.0
        assert result.market_beta == 0.0
