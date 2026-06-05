"""Tests for XGBoostReturnRegressor, MultiHorizonRegressor, and _compute_forward_returns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tempfile, os

from prediction.regressor import (
    XGBoostReturnRegressor, MultiHorizonRegressor,
    HorizonPrediction, MultiHorizonPrediction,
    _compute_forward_returns,
)
from config.settings import Settings


class TestForwardReturns:
    def test_returns_horizon_math(self, sample_prices_df):
        rets = _compute_forward_returns(sample_prices_df, 5)
        assert isinstance(rets, pd.Series)
        assert rets.index.nlevels == 2  # (ticker, trade_date)
        assert len(rets) > 0

    def test_returns_are_finite(self, sample_prices_df):
        rets = _compute_forward_returns(sample_prices_df, 10)
        assert np.isfinite(rets).all()


class TestXGBoostReturnRegressor:
    @pytest.fixture
    def sample_Xy(self):
        np.random.seed(42)
        n = 200
        X = pd.DataFrame({
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "f3": np.random.randn(n),
        })
        y = pd.Series(np.random.randn(n) * 0.02, name="ret")
        return X, y

    def test_fit_and_predict(self, sample_Xy):
        X, y = sample_Xy
        reg = XGBoostReturnRegressor(horizon_days=21)
        reg.fit(X, y)
        assert reg.is_fitted
        assert reg.error_std > 0

        preds = reg.predict(X, tickers=pd.Series(["A"] * len(X)), dates=pd.Series([None] * len(X)))
        assert len(preds) == len(X)
        for p in preds:
            assert isinstance(p, HorizonPrediction)
            assert p.horizon_days == 21
            assert -1.0 <= p.predicted_return <= 1.0
            assert 0.0 <= p.prob_up <= 1.0

    def test_predict_raises_if_not_fitted(self, sample_Xy):
        X, _ = sample_Xy
        reg = XGBoostReturnRegressor(horizon_days=5)
        with pytest.raises(RuntimeError):
            reg.predict(X)

    def test_save_load_roundtrip(self, sample_Xy):
        X, y = sample_Xy
        reg = XGBoostReturnRegressor(horizon_days=21)
        reg.fit(X, y)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test_reg")
            reg.save(path)
            assert os.path.exists(f"{path}.xgb")
            assert os.path.exists(f"{path}.pkl")

            reg2 = XGBoostReturnRegressor(horizon_days=21)
            reg2.load(path)
            assert reg2.is_fitted
            assert reg2.error_std == pytest.approx(reg.error_std)
            assert reg2.horizon_days == 21

            preds1 = reg.predict(X, tickers=pd.Series(["A"] * len(X)), dates=pd.Series([None] * len(X)))
            preds2 = reg2.predict(X, tickers=pd.Series(["A"] * len(X)), dates=pd.Series([None] * len(X)))
            for p1, p2 in zip(preds1, preds2):
                assert p1.predicted_return == pytest.approx(p2.predicted_return)
                assert p1.prob_up == pytest.approx(p2.prob_up)

    def test_error_std_nonzero(self, sample_Xy):
        X, y = sample_Xy
        reg = XGBoostReturnRegressor(horizon_days=21)
        reg.fit(X, y)
        assert reg.error_std >= 0.001

    def test_save_raises_if_not_fitted(self):
        reg = XGBoostReturnRegressor(horizon_days=21)
        with pytest.raises(RuntimeError):
            reg.save("/tmp/test")


class TestMultiHorizonRegressor:
    @pytest.fixture
    def sample_reg_data(self, sample_prices_df):
        np.random.seed(42)
        X = build_features_for_test(sample_prices_df)
        return X, sample_prices_df

    def test_init_creates_regressors(self):
        mhr = MultiHorizonRegressor(horizons=[5, 21])
        assert not mhr.is_fitted
        assert len(mhr._regressors) == 2

    def test_fit_all_and_predict(self, sample_reg_data):
        X, prices = sample_reg_data
        mhr = MultiHorizonRegressor(horizons=[5, 21])
        mhr.fit_all(X, prices)

        preds = mhr.predict_all(
            X.iloc[:10],
            tickers=pd.Series(X.iloc[:10].index.get_level_values(0)),
            dates=pd.Series([None] * 10),
        )
        assert len(preds) > 0
        for mp in preds:
            assert isinstance(mp, MultiHorizonPrediction)
            assert len(mp.horizons) > 0
            for hp in mp.horizons.values():
                assert isinstance(hp, HorizonPrediction)
                assert 0.0 <= hp.prob_up <= 1.0


def build_features_for_test(prices: pd.DataFrame) -> pd.DataFrame:
    """Build a small feature matrix from synthetic prices for testing."""
    features_list = []
    for ticker, group in prices.groupby("ticker"):
        g = group.sort_values("trade_date").set_index("trade_date")
        rets = g["close"].pct_change()
        data = pd.DataFrame(index=g.index)
        data["roc_5d"] = g["close"].pct_change(5)
        data["roc_10d"] = g["close"].pct_change(10)
        data["roc_21d"] = g["close"].pct_change(21)
        data["roc_63d"] = g["close"].pct_change(63)
        data["rsi_5d"] = 50.0
        data["rsi_10d"] = 50.0
        data["rsi_21d"] = 50.0
        data["rsi_63d"] = 50.0
        data["mom_5d"] = g["close"].diff(5)
        data["mom_10d"] = g["close"].diff(10)
        data["mom_21d"] = g["close"].diff(21)
        data["mom_63d"] = g["close"].diff(63)
        data["macd_macd"] = 0.0
        data["macd_signal"] = 0.0
        data["macd_histogram"] = 0.0
        data["atr_21d"] = 0.5
        data["bb_upper_21d"] = g["close"] * 1.05
        data["bb_lower_21d"] = g["close"] * 0.95
        data["bb_pct_b_21d"] = 0.5
        data["hist_vol_21d"] = rets.rolling(21).std() * np.sqrt(252)
        data["volume_ma_ratio_20d"] = 1.0
        data["turnover_proxy"] = g["volume"] * g["close"]
        data["sma_ratio_63d"] = 1.0
        data["max_dd_63d"] = 0.0
        data = data.dropna()
        data.index = pd.MultiIndex.from_tuples(
            [(ticker, dt) for dt in data.index], names=["ticker", "trade_date"]
        )
        features_list.append(data)
    return pd.concat(features_list) if features_list else pd.DataFrame()
