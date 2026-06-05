"""Multi-horizon XGBoost return regressor for ETF forward return prediction.

Mirrors the persistence and feature patterns of XGBoostEnsemble (core/ensemble.py).
Uses scipy.stats.norm.cdf to calibrate prob_up from prediction / error_std.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config.settings import Settings

logger = logging.getLogger(__name__)


# ── Dataclasses ──────────────────────────────────────────────────

@dataclass
class HorizonPrediction:
    ticker: str
    pred_date: object
    horizon_days: int
    predicted_return: float       # e.g. 0.023 = +2.3%
    prob_up: float                # P(return > 0) via norm.cdf
    model_version: str = ""


@dataclass
class MultiHorizonPrediction:
    ticker: str
    pred_date: object
    horizons: dict[int, HorizonPrediction] = field(default_factory=dict)


# ── Single-horizon regressor ─────────────────────────────────────

class XGBoostReturnRegressor:
    """XGBoost regressor for a single fixed-horizon forward return."""

    def __init__(self, horizon_days: int, settings: Settings | None = None) -> None:
        self._horizon = horizon_days
        self._settings = settings or Settings()
        self._model: XGBRegressor | None = None
        self._scaler: StandardScaler = StandardScaler()
        self._feature_names: list[str] = []
        self._error_std: float = 0.05
        self._is_fitted: bool = False

    # ── Properties ──────────────────────────────────────────

    @property
    def horizon_days(self) -> int:
        return self._horizon

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def error_std(self) -> float:
        return self._error_std

    # ── Fit / Predict ───────────────────────────────────────

    def fit(
        self, X: pd.DataFrame, y: pd.Series,
        feature_names: list[str] | None = None,
        val_split: float = 0.2,
    ) -> None:
        self._feature_names = feature_names or list(X.columns)
        X_sel = X[self._feature_names].copy()
        X_scaled = self._scaler.fit_transform(X_sel)

        n_val = max(int(len(X_scaled) * val_split), 5)
        X_train, X_val = X_scaled[:-n_val], X_scaled[-n_val:]
        y_train, y_val = y.iloc[:-n_val], y.iloc[-n_val:]

        s = self._settings
        self._model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            max_depth=s.xgb_max_depth,
            learning_rate=s.xgb_learning_rate,
            n_estimators=s.xgb_n_estimators,
            subsample=s.xgb_subsample,
            colsample_bytree=s.xgb_colsample_bytree,
            reg_alpha=s.xgb_reg_alpha,
            reg_lambda=s.xgb_reg_lambda,
            min_child_weight=s.xgb_min_child_weight,
            random_state=42,
            verbosity=0,
        )
        self._model.fit(X_train, y_train)

        val_preds = self._model.predict(X_val)
        residuals = y_val.values - val_preds
        self._error_std = max(float(np.std(residuals)), 0.001)
        self._is_fitted = True
        logger.info("Regressor %dd trained — RMSE=%.4f, error_std=%.4f, samples=%d",
                     self._horizon, float(np.sqrt(np.mean(residuals**2))),
                     self._error_std, len(X_train))

    def predict(
        self, X: pd.DataFrame,
        tickers: pd.Series | None = None,
        dates: pd.Series | None = None,
    ) -> list[HorizonPrediction]:
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_sel = X[self._feature_names].copy()
        X_scaled = self._scaler.transform(X_sel)
        preds = self._model.predict(X_scaled)

        results: list[HorizonPrediction] = []
        for i in range(len(preds)):
            pred_return = float(preds[i])
            prob_up = float(norm.cdf(pred_return / self._error_std))
            results.append(HorizonPrediction(
                ticker=str(tickers.iloc[i]) if tickers is not None else "",
                pred_date=dates.iloc[i] if dates is not None else None,
                horizon_days=self._horizon,
                predicted_return=round(pred_return, 6),
                prob_up=round(prob_up, 4),
            ))
        return results

    # ── Persistence ─────────────────────────────────────────

    def save(self, path: str) -> None:
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        self._model.save_model(f"{path}.xgb")
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump({
                "feature_names": self._feature_names,
                "scaler": self._scaler,
                "error_std": self._error_std,
                "horizon_days": self._horizon,
            }, f)
        logger.info("Regressor %dd saved to %s.{xgb,pkl}", self._horizon, path)

    def load(self, path: str) -> None:
        self._model = XGBRegressor()
        self._model.load_model(f"{path}.xgb")
        with open(f"{path}.pkl", "rb") as f:
            meta = pickle.load(f)
        self._feature_names = meta["feature_names"]
        self._scaler = meta["scaler"]
        self._error_std = meta["error_std"]
        self._horizon = meta.get("horizon_days", self._horizon)
        self._is_fitted = True
        logger.info("Regressor %dd loaded from %s", self._horizon, path)


# ── Multi-horizon aggregator ─────────────────────────────────────

class MultiHorizonRegressor:
    """Manages multiple XGBoostReturnRegressor instances for different horizons."""

    DEFAULT_HORIZONS = [5, 21, 63]

    def __init__(
        self, horizons: list[int] | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._horizons = horizons or self.DEFAULT_HORIZONS
        self._settings = settings or Settings()
        self._regressors: dict[int, XGBoostReturnRegressor] = {
            h: XGBoostReturnRegressor(h, self._settings) for h in self._horizons
        }

    @property
    def is_fitted(self) -> bool:
        return all(r.is_fitted for r in self._regressors.values())

    def fit_all(
        self, X: pd.DataFrame, prices: pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> None:
        for horizon in self._horizons:
            labels = _compute_forward_returns(prices, horizon)
            common_idx = X.index.intersection(labels.index)
            if len(common_idx) < 20:
                logger.warning("Only %d common samples for horizon %dd — skipping", len(common_idx), horizon)
                continue
            X_aligned = X.loc[common_idx]
            y_aligned = labels.loc[common_idx]
            self._regressors[horizon].fit(X_aligned, y_aligned, feature_names)

    def predict_all(
        self, X: pd.DataFrame,
        tickers: pd.Series | None = None,
        dates: pd.Series | None = None,
    ) -> list[MultiHorizonPrediction]:
        per_horizon: dict[int, list[HorizonPrediction]] = {}
        for horizon, reg in self._regressors.items():
            if reg.is_fitted:
                per_horizon[horizon] = reg.predict(X, tickers, dates)

        grouped: dict[tuple, dict[int, HorizonPrediction]] = {}
        for horizon, preds in per_horizon.items():
            for p in preds:
                key = (p.ticker, p.pred_date)
                grouped.setdefault(key, {})[horizon] = p

        return [
            MultiHorizonPrediction(ticker=k[0], pred_date=k[1], horizons=h)
            for k, h in grouped.items()
        ]

    def save_all(self, base_path: str) -> None:
        for horizon, reg in self._regressors.items():
            if reg.is_fitted:
                reg.save(f"{base_path}_{horizon}d")

    def load_all(self, base_path: str) -> None:
        import os
        for horizon in self._horizons:
            p = f"{base_path}_{horizon}d"
            if os.path.exists(f"{p}.xgb") and os.path.exists(f"{p}.pkl"):
                self._regressors[horizon].load(p)


# ── Helper ───────────────────────────────────────────────────────

def _compute_forward_returns(prices: pd.DataFrame, horizon: int) -> pd.Series:
    """Compute forward returns at a fixed horizon.

    Args:
        prices: OHLCV DataFrame with [ticker, trade_date, close].
        horizon: Number of trading days to look forward.

    Returns:
        Series indexed by (ticker, trade_date) of forward returns.
    """
    results: list[pd.Series] = []
    for ticker, group in prices.groupby("ticker"):
        g = group.sort_values("trade_date").set_index("trade_date")
        fwd_ret = g["close"].pct_change(horizon).shift(-horizon).dropna()
        fwd_ret = pd.Series(fwd_ret.values, index=pd.MultiIndex.from_tuples(
            [(ticker, dt) for dt in fwd_ret.index], names=["ticker", "trade_date"]
        ))
        results.append(fwd_ret)

    if not results:
        return pd.Series(dtype=float)
    return pd.concat(results).sort_index()
