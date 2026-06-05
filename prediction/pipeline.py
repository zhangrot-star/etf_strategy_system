"""Prediction pipeline — feature construction + multi-horizon inference."""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from core.feature_utils import build_features_from_prices
from prediction.regressor import MultiHorizonRegressor

logger = logging.getLogger(__name__)


class PredictionPipeline:
    """Orchestrates multi-horizon ETF return prediction.

    Usage:
        pipeline = PredictionPipeline(config)
        rows = pipeline.run(prices)  # list of dict ready for DB upsert
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        pred_cfg = self._config.get("prediction", {})
        horizons = pred_cfg.get("horizons", [5, 21, 63])
        model_path = pred_cfg.get("model_path", "models/xgboost_reg")
        self._model_path = model_path
        self._regressor = MultiHorizonRegressor(horizons=horizons)
        self._load_models()

    def _load_models(self) -> None:
        import os
        first_path = f"{self._model_path}_{self._regressor._horizons[0]}d"
        if os.path.exists(f"{first_path}.xgb") and os.path.exists(f"{first_path}.pkl"):
            try:
                self._regressor.load_all(self._model_path)
                logger.info("Loaded %d regressors from %s_*d", len(self._regressor._horizons), self._model_path)
            except Exception:
                logger.warning("Failed to load regressors", exc_info=True)

    @property
    def is_fitted(self) -> bool:
        return self._regressor.is_fitted

    def run(
        self, prices: pd.DataFrame, run_date: date | None = None,
    ) -> list[dict[str, Any]]:
        """Run multi-horizon prediction and return DB-ready rows.

        Args:
            prices: OHLCV DataFrame with [ticker, trade_date, close, ...].
            run_date: Date for predictions (defaults to today).

        Returns:
            List of dicts with keys matching ETFPrediction columns.
        """
        if run_date is None:
            run_date = date.today()

        if prices.empty or not self.is_fitted:
            return []

        try:
            features = build_features_from_prices(prices)
        except Exception:
            logger.warning("Feature construction failed", exc_info=True)
            return []

        if features.empty:
            return []

        latest_date = features.index.get_level_values(1).max()
        latest_features = features.xs(latest_date, level=1)

        tickers = pd.Series(latest_features.index, name="ticker")
        dates = pd.Series([pd.Timestamp(run_date)] * len(tickers), name="pred_date")

        try:
            results = self._regressor.predict_all(latest_features, tickers, dates)
        except Exception:
            logger.warning("Prediction inference failed", exc_info=True)
            return []

        rows: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc)
        for mp in results:
            for horizon, hp in mp.horizons.items():
                rows.append({
                    "ticker": mp.ticker,
                    "pred_date": run_date,
                    "horizon_days": horizon,
                    "predicted_return": hp.predicted_return,
                    "prob_up": hp.prob_up,
                    "realized": False,
                    "model_version": "1.0",
                    "created_at": now,
                })

        logger.info("Generated %d prediction rows for %d ETFs", len(rows), len(results))
        return rows
