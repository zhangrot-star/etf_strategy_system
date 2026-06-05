"""Prediction accuracy evaluation and model performance tracking.

Walk-forward historical backtesting, per-horizon metrics, and prob_up calibration.
Designed to answer: "How accurate is the model, and is it getting better?"

Key metrics:
- RMSE / MAE: prediction error magnitude
- R²: fraction of return variance explained
- Direction accuracy: % of times predicted sign matches actual sign
- Prob calibration: does prob_up=0.7 actually mean 70% chance of up?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from core.feature_utils import build_features_from_prices
from prediction.regressor import (
    HorizonPrediction,
    MultiHorizonRegressor,
    _compute_forward_returns,
)

logger = logging.getLogger(__name__)


# ── Result dataclasses ─────────────────────────────────────────────

@dataclass
class HorizonMetrics:
    horizon_days: int
    n_predictions: int = 0
    rmse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0
    direction_accuracy: float = 0.0
    mean_prediction: float = 0.0
    mean_actual: float = 0.0
    pred_std: float = 0.0
    actual_std: float = 0.0
    # Prob calibration binned
    prob_bins: list[dict] = field(default_factory=list)
    calibration_error: float = 0.0  # ECE: expected calibration error

    def to_dict(self) -> dict:
        return {
            "horizon_days": self.horizon_days,
            "n_predictions": self.n_predictions,
            "rmse": round(self.rmse, 6),
            "mae": round(self.mae, 6),
            "r2": round(self.r2, 4),
            "direction_accuracy": round(self.direction_accuracy, 4),
            "mean_prediction": round(self.mean_prediction, 6),
            "mean_actual": round(self.mean_actual, 6),
            "pred_std": round(self.pred_std, 6),
            "actual_std": round(self.actual_std, 6),
            "calibration_error": round(self.calibration_error, 4),
        }


@dataclass
class EvaluationReport:
    evaluated_at: str
    model_path: str
    n_samples: int
    per_horizon: dict[int, HorizonMetrics] = field(default_factory=dict)
    overall_rating: str = ""

    def summary(self) -> str:
        lines = [
            f"=== Prediction Evaluation Report ===",
            f"Date: {self.evaluated_at}",
            f"Model: {self.model_path}",
            f"Samples: {self.n_samples}",
            f"",
            f"{'Horizon':<12} {'RMSE':>8} {'MAE':>8} {'R²':>7} {'DirAcc':>8} {'CalErr':>7}",
            f"{'-'*55}",
        ]
        for h in sorted(self.per_horizon.keys()):
            m = self.per_horizon[h]
            lines.append(
                f"{f'{h}d':<12} {m.rmse:>8.4f} {m.mae:>8.4f} {m.r2:>7.3f} "
                f"{m.direction_accuracy:>7.1%} {m.calibration_error:>6.1%}"
            )
        lines.append(f"{'-'*55}")
        lines.append(f"Rating: {self.overall_rating}")
        return "\n".join(lines)


# ── Evaluator ──────────────────────────────────────────────────────

class PredictionEvaluator:
    """Walk-forward evaluator for multi-horizon return predictions.

    Loads trained regressors, builds features on historical windows,
    generates out-of-sample predictions, and compares against realized returns.
    """

    def __init__(
        self,
        model_path: str = "models/xgboost_reg",
    ) -> None:
        self._model_path = model_path
        self._regressor = MultiHorizonRegressor()
        try:
            self._regressor.load_all(model_path)
            fitted = [h for h, r in self._regressor._regressors.items() if r.is_fitted]
            logger.info("Loaded %d fitted regressors from %s", len(fitted), model_path)
        except Exception as e:
            logger.warning("Could not load models from %s: %s", model_path, e)

    @property
    def is_ready(self) -> bool:
        return self._regressor.is_fitted

    @property
    def available_horizons(self) -> list[int]:
        return [h for h, r in self._regressor._regressors.items() if r.is_fitted]

    def evaluate(
        self,
        prices: pd.DataFrame,
        eval_start: str = "2025-01-01",
        step_days: int = 21,
    ) -> EvaluationReport:
        """Walk-forward historical evaluation.

        For each evaluation date (every `step_days` from eval_start), builds
        features from prices up to that date, predicts forward returns, and
        compares against realized returns.

        Args:
            prices: OHLCV DataFrame with [ticker, trade_date, ...].
            eval_start: First date to evaluate predictions.
            step_days: Days between evaluation points.

        Returns:
            EvaluationReport with per-horizon metrics.
        """
        if not self.is_ready:
            raise RuntimeError("No fitted models loaded. Cannot evaluate.")

        prices = prices.copy()
        prices["trade_date"] = pd.to_datetime(prices["trade_date"])
        tickers = sorted(prices["ticker"].unique())

        all_comparisons: dict[int, list[dict]] = {}
        horizons = self.available_horizons

        # Generate evaluation dates
        trade_dates = sorted(prices["trade_date"].unique())
        eval_cutoff = pd.Timestamp(eval_start)
        eval_dates = [
            d for d in trade_dates
            if d >= eval_cutoff and (d - trade_dates[0]).days % step_days == 0
        ]

        if not eval_dates:
            # Fallback: evaluate on last few dates
            eval_dates = trade_dates[-10:]

        logger.info("Walk-forward evaluation on %d dates (every %dd) ...",
                     len(eval_dates), step_days)

        for i, eval_date in enumerate(eval_dates):
            if (i + 1) % 10 == 0:
                logger.info("  Evaluating %s (%d/%d) ...", eval_date.date(), i + 1, len(eval_dates))

            # Slice prices up to eval_date for feature building
            hist_prices = prices[prices["trade_date"] <= eval_date]

            try:
                features = build_features_from_prices(hist_prices)
            except Exception:
                continue

            if features.empty:
                continue

            # Get latest feature row per ticker for prediction day
            features = features.reset_index()
            features["trade_date"] = pd.to_datetime(features["trade_date"])
            latest_features = (
                features.sort_values("trade_date")
                .groupby("ticker")
                .tail(1)
                .set_index(["ticker", "trade_date"])
            )

            if latest_features.empty:
                continue

            # Build prediction input: latest features × all tickers
            X_input = latest_features.reset_index(level="trade_date", drop=True)
            X_input = X_input.reindex(tickers)
            X_input = X_input.dropna(how="all").fillna(0)

            valid_tickers = X_input.index.tolist()
            if len(valid_tickers) < 3:
                continue

            # Predict for all horizons
            try:
                multi_preds = self._regressor.predict_all(
                    X_input,
                    tickers=pd.Series(valid_tickers),
                    dates=pd.Series([eval_date] * len(valid_tickers)),
                )
            except Exception:
                continue

            # Compute actual forward returns for each horizon
            for horizon in horizons:
                # Find actual forward return: close at eval_date → close at eval_date + horizon
                actual_returns: dict[str, float] = {}
                for ticker in valid_tickers:
                    ticker_prices = prices[
                        (prices["ticker"] == ticker) &
                        (prices["trade_date"] >= eval_date)
                    ].sort_values("trade_date")

                    if len(ticker_prices) < horizon + 1:
                        continue

                    start_close = ticker_prices.iloc[0]["close"]
                    # Find the trade date closest to eval_date + horizon trading days
                    end_idx = min(horizon, len(ticker_prices) - 1)
                    end_close = ticker_prices.iloc[end_idx]["close"]

                    if start_close > 0:
                        actual_returns[ticker] = float(end_close / start_close - 1)

                # Match predictions against actuals
                horizon_preds = [
                    mp.horizons[horizon]
                    for mp in multi_preds
                    if horizon in mp.horizons
                ]

                for pred in horizon_preds:
                    if pred.ticker in actual_returns:
                        if horizon not in all_comparisons:
                            all_comparisons[horizon] = []
                        all_comparisons[horizon].append({
                            "ticker": pred.ticker,
                            "eval_date": eval_date,
                            "predicted": pred.predicted_return,
                            "actual": actual_returns[pred.ticker],
                            "prob_up": pred.prob_up,
                        })

        # Compute metrics per horizon
        per_horizon: dict[int, HorizonMetrics] = {}
        for horizon in horizons:
            comps = all_comparisons.get(horizon, [])
            if len(comps) < 10:
                continue
            metrics = _compute_horizon_metrics(horizon, comps)
            per_horizon[horizon] = metrics

            logger.info(
                "  %dd: RMSE=%.4f MAE=%.4f R²=%.3f DirAcc=%.1f%% ECE=%.1f%% (n=%d)",
                horizon, metrics.rmse, metrics.mae, metrics.r2,
                metrics.direction_accuracy * 100, metrics.calibration_error * 100,
                metrics.n_predictions,
            )

        overall = _rate_overall(per_horizon)
        return EvaluationReport(
            evaluated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            model_path=self._model_path,
            n_samples=sum(m.n_predictions for m in per_horizon.values()),
            per_horizon=per_horizon,
            overall_rating=overall,
        )


# ── Metric computation ─────────────────────────────────────────────

def _compute_horizon_metrics(horizon: int, comparisons: list[dict]) -> HorizonMetrics:
    preds = np.array([c["predicted"] for c in comparisons])
    actuals = np.array([c["actual"] for c in comparisons])
    probs = np.array([c["prob_up"] for c in comparisons])

    n = len(preds)
    rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
    mae = float(mean_absolute_error(actuals, preds))

    # R²
    try:
        r2 = float(r2_score(actuals, preds))
    except (ValueError, TypeError):
        r2 = float("nan") if n == 0 else 0.0

    # Direction accuracy
    dir_correct = np.sum((preds > 0) == (actuals > 0))
    dir_acc = float(dir_correct / n) if n > 0 else 0.0

    # Prob calibration (ECE)
    prob_bins, ece = _compute_calibration(probs, actuals, n_bins=5)

    return HorizonMetrics(
        horizon_days=horizon,
        n_predictions=n,
        rmse=rmse,
        mae=mae,
        r2=r2 if not np.isnan(r2) else 0.0,
        direction_accuracy=dir_acc,
        mean_prediction=float(np.mean(preds)),
        mean_actual=float(np.mean(actuals)),
        pred_std=float(np.std(preds)),
        actual_std=float(np.std(actuals)),
        prob_bins=prob_bins,
        calibration_error=ece,
    )


def _compute_calibration(
    probs: np.ndarray, actuals: np.ndarray, n_bins: int = 5,
) -> tuple[list[dict], float]:
    """Compute expected calibration error (ECE) via binned prob_up vs actual up rate."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_records = []

    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            bin_records.append({
                "bin_low": round(bins[i], 2),
                "bin_high": round(bins[i + 1], 2),
                "n": 0, "avg_prob": 0, "actual_up_rate": 0,
            })
            continue

        bin_probs = probs[mask]
        bin_actuals = actuals[mask]
        avg_prob = float(np.mean(bin_probs))
        actual_up_rate = float(np.mean(bin_actuals > 0))
        bin_weight = mask.sum() / len(probs)

        ece += bin_weight * abs(avg_prob - actual_up_rate)

        bin_records.append({
            "bin_low": round(bins[i], 2),
            "bin_high": round(bins[i + 1], 2),
            "n": int(mask.sum()),
            "avg_prob": round(avg_prob, 3),
            "actual_up_rate": round(actual_up_rate, 3),
        })

    return bin_records, float(ece)


def _rate_overall(per_horizon: dict[int, HorizonMetrics]) -> str:
    """Assign an overall rating based on combined metrics."""
    if not per_horizon:
        return "No data"

    # Average R² and direction accuracy across horizons
    avg_r2 = np.mean([m.r2 for m in per_horizon.values()])
    avg_dir = np.mean([m.direction_accuracy for m in per_horizon.values()])

    if avg_r2 > 0.3 and avg_dir > 0.65:
        return "A — Excellent (strong predictive power)"
    elif avg_r2 > 0.1 and avg_dir > 0.55:
        return "B — Good (usable signal)"
    elif avg_dir > 0.52:
        return "C — Marginal (slightly better than random)"
    else:
        return "D — Poor (needs retraining or feature improvement)"
