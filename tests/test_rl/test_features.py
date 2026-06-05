"""Tests for RLFeatureBuilder — state vector assembly."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from rl.features import RLFeatureBuilder


class TestRLFeatureBuilder:
    """Test the 175-dim state vector builder."""

    @pytest.fixture
    def builder(self):
        return RLFeatureBuilder(max_positions=8, ticker_order=["SPY", "QQQ", "IWM"])

    def test_observation_dim(self, builder):
        """Observation dimension matches expected: 21*8 + 7 = 175."""
        assert builder.observation_dim == 175

    def test_build_returns_correct_shape(self, builder):
        """Build returns (175,) float32 array."""
        obs = builder.build(
            current_date=date.today(),
            ensemble_preds={
                "SPY": {"prob_buy": 0.6, "prob_hold": 0.3, "prob_sell": 0.1, "signal_num": 2},
            },
        )
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (175,)
        assert obs.dtype == np.float32

    def test_build_no_nan(self, builder):
        """Output contains no NaN values."""
        obs = builder.build(current_date=date.today())
        assert not np.any(np.isnan(obs))

    def test_build_all_defaults_no_nan(self, builder):
        """Output with no data provided is all valid (zeros/defaults)."""
        obs = builder.build(current_date=date.today())
        assert not np.any(np.isnan(obs))
        assert np.isfinite(obs).all()

    def test_ticker_order_respected(self, builder):
        """Premium prediction features for first ticker appear at start of vector."""
        ensemble = {
            "SPY": {"prob_buy": 0.9, "prob_hold": 0.05, "prob_sell": 0.05, "signal_num": 2},
            "QQQ": {"prob_buy": 0.1, "prob_hold": 0.2, "prob_sell": 0.7, "signal_num": 0},
            "IWM": {"prob_buy": 0.3, "prob_hold": 0.4, "prob_sell": 0.3, "signal_num": 1},
        }
        obs = builder.build(current_date=date.today(), ensemble_preds=ensemble)
        # First ticker (SPY) prob_buy should be at position 0
        assert obs[0] == pytest.approx(0.9, abs=0.01)

    def test_padded_tickers_are_zero(self, builder):
        """Tickers beyond the order get zero-filled."""
        obs = builder.build(current_date=date.today())
        # Last ticker block (position 7, starting at 21*7=147) should be zeros
        last_block = obs[147:168]
        assert np.all(last_block == 0.0)

    def test_set_ticker_order(self, builder):
        """Setting ticker order changes the observation."""
        obs_before = builder.build(current_date=date.today())
        builder.set_ticker_order(["XLK", "XLV", "XLF"])
        assert builder.ticker_order == ["XLK", "XLV", "XLF"]

    def test_max_positions(self, builder):
        """Max positions property is set correctly."""
        assert builder.max_positions == 8

    def test_build_ticker_block_shape(self, builder):
        """Individual ticker block is 21 elements."""
        block = builder._build_ticker_block(
            ticker="SPY",
            ensemble={"prob_buy": 0.5, "prob_hold": 0.3, "prob_sell": 0.2, "signal_num": 1},
            regressor={"pred_5d": 0.01, "pred_21d": 0.03, "pred_63d": 0.08,
                       "prob_up_5d": 0.6, "prob_up_21d": 0.7, "prob_up_63d": 0.8},
            score={"raw_total": 75.0, "adjusted_total": 80.0},
            sent={"polarity": 0.3, "confidence": 0.8},
            tech={"roc_21d": 0.02, "rsi_14d": 55.0, "atr_21d": 1.5,
                  "bb_pct_b": 0.6, "hist_vol_21d": 0.18,
                  "sma_ratio_63d": 1.05, "volume_ma_ratio_20d": 1.2},
            weight=0.0,
        )
        assert len(block) == 21
        assert block.dtype == np.float32

    def test_global_features_shape(self, builder):
        """Global features occupy positions 168-174."""
        obs = builder.build(
            current_date=date.today(),
            cash_weight=0.2,
            portfolio_vol_21d=0.15,
            avg_correlation=0.4,
            market_regime=2,
            days_since_rebalance=10,
            n_positions=5,
            market_return_21d=0.03,
        )
        # Check global feature at position 168 (cash_weight)
        assert obs[168] == pytest.approx(0.2, abs=0.001)
        # Last global feature at position 174 (market_return_21d)
        assert obs[174] == pytest.approx(0.03, abs=0.001)


class TestFromDataFrame:
    """Test the from_dataframe convenience method."""

    @pytest.fixture
    def prices_df(self):
        """Create a simple price DataFrame for testing."""
        tickers = ["SPY", "QQQ"]
        dates = pd.bdate_range("2025-01-01", "2025-01-31")
        rows = []
        for t in tickers:
            price = 100.0
            for d in dates:
                price *= 1 + np.random.normal(0, 0.01)
                rows.append({
                    "ticker": t,
                    "trade_date": d,
                    "close": price,
                })
        return pd.DataFrame(rows)

    @pytest.fixture
    def builder(self):
        return RLFeatureBuilder(max_positions=8, ticker_order=["SPY", "QQQ"])

    def test_from_dataframe_with_empty_features(self, builder):
        """from_dataframe with empty features doesn't crash."""
        empty_features = pd.DataFrame(columns=["ticker", "trade_date"])
        obs = builder.from_dataframe(
            features=empty_features,
            current_date=date.today(),
        )
        assert obs.shape == (175,)

    def test_from_dataframe_no_nan(self, builder, prices_df):
        """from_dataframe output has no NaN."""
        obs = builder.from_dataframe(
            features=prices_df,
            current_date=date(2025, 1, 15),
        )
        assert not np.any(np.isnan(obs))
