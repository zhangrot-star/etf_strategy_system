"""Optuna hyperparameter optimization for the XGBoost ensemble.

Runs a multi-objective search over XGBoost hyperparameters using
walk-forward cross-validation, then returns the best parameter set
for use in production backtesting.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def optimize_xgboost_params(
    prices: pd.DataFrame,
    tickers: list[str],
    n_trials: int = 100,
    cv_folds: int = 5,
    oos_ratio: float = 0.2,
    study_name: str = "xgboost_etf_optimization",
    storage: str = "sqlite:///optuna.db",
    random_seed: int = 42,
) -> dict[str, Any]:
    """Run Optuna hyperparameter search for XGBoost ETF signal generation.

    Args:
        prices: OHLCV panel (ticker, trade_date, open, high, low, close, volume).
        tickers: List of ETF ticker codes.
        n_trials: Number of Optuna trials.
        cv_folds: Number of time-series cross-validation folds.
        oos_ratio: Fraction of data reserved for out-of-sample validation.
        study_name: Optuna study name for persistent storage.
        storage: Optuna storage URL (SQLite by default).
        random_seed: Random seed for reproducibility.

    Returns:
        Dict of best hyperparameters: max_depth, learning_rate, n_estimators,
        subsample, colsample_bytree, reg_alpha, reg_lambda, min_child_weight.
    """
    # Build feature matrix and labels
    from core.feature_utils import build_features_and_labels
    features, labels = build_features_and_labels(prices[prices["ticker"].isin(tickers)])
    if features.empty or labels.empty:
        logger.warning("Could not build optimization dataset — returning defaults.")
        return _default_params()

    # Split into train/OOS
    split_idx = int(len(features) * (1 - oos_ratio))
    X_train, X_oos = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_oos = labels.iloc[:split_idx], labels.iloc[split_idx:]

    if len(X_train) < 200:
        logger.warning("Insufficient training data (%d samples) — returning defaults.", len(X_train))
        return _default_params()

    import optuna

    # Create or load study
    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
        )
    except Exception:
        logger.warning("Could not connect to Optuna storage — using in-memory study.")
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
        )

    objective = _Objective(X_train, y_train, X_oos, y_oos, cv_folds, random_seed)

    # Prune existing trials to avoid re-running
    n_existing = len(study.trials)
    remaining = max(0, n_trials - n_existing)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)
    else:
        logger.info("Study already has %d trials — using existing best params.", n_existing)

    best = study.best_params
    logger.info("Best trial #%d — Sharpe: %.4f, params: %s",
                 study.best_trial.number, study.best_value, best)

    return {
        "max_depth": best.get("max_depth", 6),
        "learning_rate": best.get("learning_rate", 0.05),
        "n_estimators": best.get("n_estimators", 200),
        "subsample": best.get("subsample", 0.8),
        "colsample_bytree": best.get("colsample_bytree", 0.8),
        "reg_alpha": best.get("reg_alpha", 0.1),
        "reg_lambda": best.get("reg_lambda", 1.0),
        "min_child_weight": best.get("min_child_weight", 3),
    }


class _Objective:
    """Optuna objective: maximize Sharpe ratio of XGBoost ETF strategy."""

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_oos: pd.DataFrame,
        y_oos: pd.Series,
        cv_folds: int,
        random_seed: int,
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_oos = X_oos
        self.y_oos = y_oos
        self.cv_folds = cv_folds
        self.random_seed = random_seed

    def __call__(self, trial: Any) -> float:
        import xgboost as xgb
        from sklearn.model_selection import TimeSeriesSplit

        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "random_state": self.random_seed,
            "use_label_encoder": False,
            "verbosity": 0,
        }

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        cv_scores: list[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(self.X_train)):
            X_tr = self.X_train.iloc[train_idx]
            X_val = self.X_train.iloc[val_idx]
            y_tr = self.y_train.iloc[train_idx]
            y_val = self.y_train.iloc[val_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, verbose=False)

            # Evaluate via Sharpe of a simple long-only strategy on validation
            probas = model.predict_proba(X_val)
            # Strategy: go long tickers with highest prob_buy, short lowest
            # Simplified: use BUY prob as position signal
            buy_probs = probas[:, 2]  # class 2 = BUY
            # Daily signal-based returns (simplified approximation)
            signal_sharpe = _estimate_signal_sharpe(buy_probs, y_val)
            cv_scores.append(signal_sharpe)

        # CV score = mean Sharpe across folds
        cv_score = float(np.mean(cv_scores)) if cv_scores else -1.0

        # OOS validation as tiebreaker
        if not self.X_oos.empty:
            model = xgb.XGBClassifier(**params)
            model.fit(self.X_train, self.y_train, verbose=False)
            oos_probas = model.predict_proba(self.X_oos)
            oos_buy = oos_probas[:, 2]
            oos_sharpe = _estimate_signal_sharpe(oos_buy, self.y_oos)
            # Weighted: 70% CV + 30% OOS
            score = 0.7 * cv_score + 0.3 * oos_sharpe
        else:
            score = cv_score

        return score


def _estimate_signal_sharpe(buy_probs: np.ndarray, y_true: pd.Series) -> float:
    """Estimate Sharpe ratio from signal probabilities.

    Long tickers with prob_buy > 0.6, short tickers with prob_buy < 0.3.
    """
    y_vals = y_true.values if hasattr(y_true, "values") else np.array(y_true)
    if len(buy_probs) != len(y_vals):
        return 0.0

    # Simulate daily returns: long top-quartile signals, short bottom-quartile
    long_mask = buy_probs > 0.6
    short_mask = buy_probs < 0.3

    daily_returns = np.zeros(len(buy_probs))
    # Approximation: y_true is the class label (0=SELL, 1=HOLD, 2=BUY)
    # Map to returns: BUY→+1%, HOLD→0%, SELL→-1%
    return_map = {0: -0.005, 1: 0.0, 2: 0.005}
    mapped_returns = np.array([return_map.get(int(y), 0.0) for y in y_vals])

    daily_returns[long_mask] = mapped_returns[long_mask]
    daily_returns[short_mask] = -mapped_returns[short_mask]

    mean_ret = float(np.mean(daily_returns))
    std_ret = float(np.std(daily_returns))
    if std_ret <= 0:
        return 0.0

    # Annualized Sharpe (assuming 252 trading days)
    sharpe = (mean_ret / std_ret) * np.sqrt(252)
    return float(np.clip(sharpe, -5.0, 5.0))


def _default_params() -> dict[str, Any]:
    return {
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 3,
    }
