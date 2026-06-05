"""XGBoost ensemble model for nonlinear fusion of quantitative and sentiment features."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from config.settings import Settings

logger = logging.getLogger(__name__)

SIGNAL_LABELS: dict[int, str] = {0: "SELL", 1: "HOLD", 2: "BUY"}


@dataclass
class EnsemblePrediction:
    """Output of the XGBoost ensemble for a single observation."""

    ticker: str
    date: object
    signal: str  # BUY | HOLD | SELL
    prob_buy: float
    prob_hold: float
    prob_sell: float
    feature_importance: Optional[dict[str, float]] = None


class XGBoostEnsemble:
    """XGBoost classifier for ETF signal generation.

    Trains on a panel of technical factors + sentiment features, outputs
    a 3-class probability distribution that drives portfolio allocation.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._model: xgb.XGBClassifier | None = None
        self._scaler: StandardScaler = StandardScaler()
        self._feature_names: list[str] = []
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    # ── Training ─────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list[str] | None = None,
    ) -> None:
        """Train the XGBoost classifier.

        Args:
            X: Feature matrix (n_samples × n_features).
            y: Labels — 0=SELL, 1=HOLD, 2=BUY.  Will be generated from forward returns
               if not already labelled.
            feature_names: Column names to persist for prediction.
        """
        if X.empty:
            raise ValueError("Training data is empty.")

        self._feature_names = feature_names or list(X.columns)
        X_scaled = self._scaler.fit_transform(X)

        self._model = xgb.XGBClassifier(
            max_depth=self._settings.xgb_max_depth,
            learning_rate=self._settings.xgb_learning_rate,
            n_estimators=self._settings.xgb_n_estimators,
            subsample=self._settings.xgb_subsample,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            use_label_encoder=False,
        )
        self._model.fit(X_scaled, y)
        self._is_fitted = True
        logger.info(
            "XGBoost trained on %d samples, %d features.",
            len(X),
            len(self._feature_names),
        )

    # ── Prediction ───────────────────────────────────────────

    def predict(
        self, X: pd.DataFrame, tickers: pd.Series | None = None, dates: pd.Series | None = None
    ) -> list[EnsemblePrediction]:
        """Generate ensemble predictions with probabilities.

        Args:
            X: Feature matrix.
            tickers: Optional ticker labels for each row.
            dates: Optional date labels for each row.

        Returns:
            List of EnsemblePrediction objects.
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = X[self._feature_names] if self._feature_names else X
        X_scaled = self._scaler.transform(X)
        probas = self._model.predict_proba(X_scaled)
        # probas columns: [SELL, HOLD, BUY] (0, 1, 2)

        predictions: list[EnsemblePrediction] = []
        for i in range(len(X)):
            signal_idx = int(np.argmax(probas[i]))
            predictions.append(
                EnsemblePrediction(
                    ticker=tickers.iloc[i] if tickers is not None else "",
                    date=dates.iloc[i] if dates is not None else None,
                    signal=SIGNAL_LABELS[signal_idx],
                    prob_sell=float(probas[i][0]),
                    prob_hold=float(probas[i][1]),
                    prob_buy=float(probas[i][2]),
                )
            )
        return predictions

    def predict_single(self, features: dict[str, float]) -> EnsemblePrediction:
        """Predict for a single feature vector (for real-time use)."""
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model not fitted.")
        row = pd.DataFrame([features])[self._feature_names]
        X_scaled = self._scaler.transform(row)
        probas = self._model.predict_proba(X_scaled)[0]
        signal_idx = int(np.argmax(probas))
        return EnsemblePrediction(
            ticker=features.get("ticker", ""),
            date=None,
            signal=SIGNAL_LABELS[signal_idx],
            prob_sell=float(probas[0]),
            prob_hold=float(probas[1]),
            prob_buy=float(probas[2]),
        )

    # ── Feature importance ───────────────────────────────────

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance scores as a dict."""
        if not self._is_fitted or self._model is None:
            return {}
        importance = self._model.feature_importances_
        return dict(zip(self._feature_names, importance.tolist()))

    # ── Label construction ───────────────────────────────────

    @staticmethod
    def labels_from_forward_returns(
        returns: pd.Series,
        sell_quantile: float = 0.33,
        buy_quantile: float = 0.67,
    ) -> pd.Series:
        """Generate 3-class labels from forward returns.

        Bottom 33% → SELL (0), middle → HOLD (1), top 33% → BUY (2).
        """
        lo = returns.quantile(sell_quantile)
        hi = returns.quantile(buy_quantile)
        labels = pd.Series(1, index=returns.index)  # default HOLD
        labels[returns <= lo] = 0  # SELL
        labels[returns >= hi] = 2  # BUY
        return labels

    # ── Persistence ──────────────────────────────────────────

    def save(self, path: str) -> None:
        import pickle

        if self._model is None:
            raise RuntimeError("No model to save.")
        self._model.save_model(f"{path}.xgb")
        meta = {
            "feature_names": self._feature_names,
            "scaler": self._scaler,
        }
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(meta, f)
        logger.info("Model saved to %s.xgb / %s.pkl", path, path)

    def load(self, path: str) -> None:
        import pickle

        self._model = xgb.XGBClassifier()
        self._model.load_model(f"{path}.xgb")
        with open(f"{path}.pkl", "rb") as f:
            meta = pickle.load(f)
        self._feature_names = meta["feature_names"]
        self._scaler = meta["scaler"]
        self._is_fitted = True
        logger.info("Model loaded from %s.xgb / %s.pkl", path, path)
