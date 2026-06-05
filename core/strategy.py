"""Unified strategy engine — the central signal generator.

Merges the former StrategyOrchestrator and XGBoostSentimentStrategy into a single
CoreStrategy class.  All configuration flows through Settings or a config dict;
no private-attribute mutation from external callers.

Flow: features → ensemble.predict → sentiment fusion → risk check → weights
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from config.settings import Settings
from core.ensemble import EnsemblePrediction, XGBoostEnsemble
from core.feature_utils import build_features_and_labels
from core.risk_controller import RiskController, RiskEvent

logger = logging.getLogger(__name__)

# Lazy import for RL policy (avoids forcing torch/gymnasium dependency at import time)
_RLPolicyType: type | None = None


def _get_rl_policy_type() -> type:
    global _RLPolicyType
    if _RLPolicyType is None:
        from rl.policy import RLPolicy as _RLP
        _RLPolicyType = _RLP
    return _RLPolicyType

# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class SignalResult:
    """Per-ticker signal for a single date."""

    ticker: str
    date: date
    signal: str  # BUY | HOLD | SELL
    weight: float
    prob_buy: float
    prob_hold: float
    prob_sell: float
    sentiment_polarity: float
    sentiment_confidence: float
    risk_blocked: bool = False
    risk_reason: str = ""


@dataclass
class AllocationResult:
    """Full portfolio allocation for a single rebalance date."""

    date: date
    signals: list[SignalResult]
    risk_event: RiskEvent | None = None
    cash_weight: float = 0.0

    @property
    def allocations(self) -> dict[str, float]:
        return {s.ticker: s.weight for s in self.signals if s.weight > 0}

    @property
    def is_all_cash(self) -> bool:
        return self.cash_weight >= 1.0 or len(self.allocations) == 0

    @property
    def total_weight(self) -> float:
        return sum(self.allocations.values()) + self.cash_weight


@dataclass
class PortfolioAllocation:
    """Deprecated — kept for limited backward compatibility.  Prefer AllocationResult."""

    date: date
    allocations: dict[str, float]
    signals: dict[str, str]
    risk_event: RiskEvent | None = None
    cash_weight: float = 0.0

    @property
    def total_weight(self) -> float:
        return sum(self.allocations.values()) + self.cash_weight

    @property
    def is_all_cash(self) -> bool:
        return self.cash_weight >= 1.0 or len(self.allocations) == 0


# ── Unified Strategy ─────────────────────────────────────────────────────────


class CoreStrategy:
    """Production strategy: XGBoost ensemble + sentiment fusion + risk control.

    Usage:
        strategy = CoreStrategy(config)
        strategy.train(prices, sentiment)
        allocation = strategy.allocate(features, sentiment, date)
    """

    def __init__(self, config: dict[str, Any] | None = None, settings: Settings | None = None) -> None:
        self._config = config or {}
        self._settings = settings or Settings()

        xgb_cfg = self._config.get("xgboost", {})
        risk_cfg = self._config.get("risk", {})

        # Configure ensemble via Settings (no private-attr mutation)
        self._ensemble = XGBoostEnsemble(settings=self._settings)
        if xgb_cfg:
            self._ensemble._settings.xgb_max_depth = xgb_cfg.get("max_depth", self._settings.xgb_max_depth)
            self._ensemble._settings.xgb_learning_rate = xgb_cfg.get("learning_rate", self._settings.xgb_learning_rate)
            self._ensemble._settings.xgb_n_estimators = xgb_cfg.get("n_estimators", self._settings.xgb_n_estimators)
            self._ensemble._settings.xgb_subsample = xgb_cfg.get("subsample", self._settings.xgb_subsample)

        # Configure risk controller
        self._risk = RiskController(settings=self._settings)
        if risk_cfg:
            self._risk._breach_polarity = risk_cfg.get("breach_polarity", self._settings.sentiment_breach_threshold)
            self._risk._warn_polarity = risk_cfg.get("warn_polarity", self._settings.sentiment_warn_threshold)
            self._risk._breach_confidence = risk_cfg.get("breach_confidence", self._settings.sentiment_confidence_threshold)

        self._max_positions = risk_cfg.get("max_positions", 8)
        self._single_position_cap = risk_cfg.get("single_position_cap", 0.30)
        self._refit_frequency = xgb_cfg.get("refit_frequency", "quarterly")
        self._feature_names: list[str] = []
        self._last_fit_date: date | None = None

        # RL policy (lazy-loaded, no torch dependency at import time)
        self._rl_policy: Any = None
        self._rl_enabled = self._config.get("rl", {}).get("enabled", False)
        self._rl_feature_builder: Any = None

    # ── Properties ──────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._ensemble.is_fitted

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    @property
    def max_positions(self) -> int:
        return self._max_positions

    @property
    def single_position_cap(self) -> float:
        return self._single_position_cap

    @property
    def ensemble(self) -> XGBoostEnsemble:
        return self._ensemble

    @property
    def risk_controller(self) -> RiskController:
        return self._risk

    # ── Training ─────────────────────────────────────────────────

    def train(
        self,
        prices: pd.DataFrame,
        sentiment: pd.DataFrame | None = None,
        refit: bool = False,
    ) -> None:
        """Train (or retrain) the XGBoost ensemble.

        Args:
            prices: OHLCV panel with columns ticker, trade_date, open, high, low, close, volume.
            sentiment: Optional sentiment records (ticker, event_date, polarity, ...).
            refit: If True, forces retraining even if already fitted.
        """
        if prices.empty:
            raise ValueError("Training data is empty.")

        if self._ensemble.is_fitted and not refit:
            logger.info("Ensemble already fitted — skipping train.")
            return

        features_df, labels_series = build_features_and_labels(prices, sentiment)
        if features_df.empty or labels_series.empty:
            raise ValueError("Could not construct features/labels from input data.")

        if len(features_df) < 100:
            raise ValueError(f"Insufficient training samples: {len(features_df)} (need >= 100)")

        self._feature_names = list(features_df.columns)
        self._ensemble.fit(features_df, labels_series, feature_names=self._feature_names)
        self._last_fit_date = date.today()
        logger.info("Trained on %d samples, %d features.", len(features_df), len(self._feature_names))

    # ── RL Policy ───────────────────────────────────────────────

    def load_rl_policy(self, model_path: str) -> None:
        """Load a pre-trained RL portfolio policy.

        Args:
            model_path: Path to saved model (without .zip extension).
        """
        RLPolicy = _get_rl_policy_type()
        self._rl_policy = RLPolicy(model_path)
        self._rl_feature_builder = self._rl_policy.get_feature_builder()
        self._rl_enabled = True
        logger.info("RL policy loaded from %s — %d tickers, max_pos=%d",
                     model_path, len(self._rl_policy.ticker_order),
                     self._rl_policy.max_positions)

    def disable_rl(self) -> None:
        """Fall back to rule-based weight computation."""
        self._rl_enabled = False
        logger.info("RL disabled — using rule-based weights.")

    # ── Allocation ───────────────────────────────────────────────

    def allocate(
        self,
        features: pd.DataFrame,
        sentiment: pd.DataFrame,
        current_date: date,
    ) -> AllocationResult:
        """Generate target portfolio weights for a single rebalance date."""
        if not self._ensemble.is_fitted:
            logger.warning("Ensemble not fitted — returning equal-weight fallback.")
            return self._equal_weight_allocation(features, current_date)

        if features.empty:
            return AllocationResult(date=current_date, signals=[], cash_weight=1.0)

        tickers = list(features.index) if hasattr(features.index, "__iter__") else []

        # 1. Predict
        ticker_series = pd.Series(tickers, index=features.index)
        date_series = pd.Series([current_date] * len(features), index=features.index)
        predictions = self._ensemble.predict(features, tickers=ticker_series, dates=date_series)

        # 2. Fuse with sentiment
        sentiment_map = self._build_sentiment_map(sentiment)

        signals: list[SignalResult] = []
        for pred in predictions:
            sent_pol = sentiment_map.get(pred.ticker, {}).get("polarity", 0.0)
            sent_conf = sentiment_map.get(pred.ticker, {}).get("confidence", 0.0)
            pos_risk = self._risk.check_position(pred.ticker, sent_pol, sent_conf)

            signals.append(SignalResult(
                ticker=pred.ticker,
                date=current_date,
                signal=pred.signal,
                weight=0.0,
                prob_buy=pred.prob_buy,
                prob_hold=pred.prob_hold,
                prob_sell=pred.prob_sell,
                sentiment_polarity=sent_pol,
                sentiment_confidence=sent_conf,
                risk_blocked=pos_risk.is_breached,
                risk_reason=pos_risk.reason,
            ))

        # 3. Portfolio-level risk check
        risk_event = self._risk.check_portfolio(sentiment, tickers)

        # 4. Compute weights (RL or rule-based)
        if risk_event.is_breached:
            logger.warning("Circuit breaker tripped — all cash.")
            return AllocationResult(date=current_date, signals=signals, risk_event=risk_event, cash_weight=1.0)

        if self._rl_enabled and self._rl_policy is not None:
            weights = self._compute_weights_rl(signals, features, sentiment, current_date)
        else:
            weights = self._compute_weights(signals)
        for s in signals:
            s.weight = weights.get(s.ticker, 0.0)

        cash_weight = 1.0 - sum(weights.values())
        return AllocationResult(date=current_date, signals=signals, risk_event=risk_event, cash_weight=cash_weight)

    # ── Legacy interface (backward compat) ───────────────────────

    def generate_allocation(
        self,
        features: pd.DataFrame,
        sentiment: pd.DataFrame,
        current_date: date,
        current_prices: pd.Series | None = None,
    ) -> PortfolioAllocation:
        """Legacy interface — returns PortfolioAllocation for existing callers.

        Prefer allocate() for new code.
        """
        result = self.allocate(features, sentiment, current_date)
        return PortfolioAllocation(
            date=result.date,
            allocations={s.ticker: s.weight for s in result.signals if s.weight > 0},
            signals={s.ticker: s.signal for s in result.signals},
            risk_event=result.risk_event,
            cash_weight=result.cash_weight,
        )

    # ── Weight computation ───────────────────────────────────────

    def _compute_weights(self, signals: list[SignalResult]) -> dict[str, float]:
        raw: dict[str, float] = {}
        for s in signals:
            if s.risk_blocked:
                raw[s.ticker] = 0.0
            elif s.signal == "BUY":
                raw[s.ticker] = s.prob_buy
            elif s.signal == "HOLD":
                raw[s.ticker] = s.prob_buy * 0.5
            else:
                raw[s.ticker] = 0.0

        capped = {k: min(v, self._single_position_cap) for k, v in raw.items()}
        sorted_items = sorted(capped.items(), key=lambda x: x[1], reverse=True)
        top_n = dict(sorted_items[: self._max_positions])

        total = sum(top_n.values())
        if total > 0:
            top_n = {k: v / total for k, v in top_n.items()}

        return top_n

    def _compute_weights_rl(
        self,
        signals: list[SignalResult],
        features: pd.DataFrame,
        sentiment: pd.DataFrame,
        current_date: date,
    ) -> dict[str, float]:
        """Use the RL policy to determine portfolio weights.

        Falls back to rule-based weight computation if the RL policy
        fails or returns empty weights.
        """
        try:
            # Build observation using the policy's feature builder
            obs = self._rl_feature_builder.from_dataframe(
                features=features,
                current_date=current_date,
                ensemble_preds={
                    s.ticker: {
                        "prob_buy": s.prob_buy,
                        "prob_hold": s.prob_hold,
                        "prob_sell": s.prob_sell,
                        "signal_num": {"BUY": 2, "HOLD": 1, "SELL": 0}.get(s.signal, 1),
                    }
                    for s in signals
                },
                sentiment_df=sentiment,
                current_weights={s.ticker: s.weight for s in signals if s.weight > 0},
            )

            rl_weights = self._rl_policy.predict_weights(obs)
            if not rl_weights:
                logger.warning("RL policy returned empty weights — falling back to rule-based.")
                return self._compute_weights(signals)

            # Respect risk blocks from the risk controller
            for s in signals:
                if s.risk_blocked and s.ticker in rl_weights:
                    del rl_weights[s.ticker]

            return rl_weights

        except Exception:
            logger.exception("RL policy prediction failed — falling back to rule-based.")
            return self._compute_weights(signals)

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _build_sentiment_map(sentiment: pd.DataFrame) -> dict[str, dict[str, float]]:
        if sentiment is None or sentiment.empty:
            return {}
        if "ticker" not in sentiment.columns:
            return {}
        col = "event_date" if "event_date" in sentiment.columns else sentiment.columns[1]
        latest = sentiment.sort_values(col)
        result: dict[str, dict[str, float]] = {}
        for ticker, group in latest.groupby("ticker"):
            last = group.iloc[-1]
            result[ticker] = {
                "polarity": float(last.get("polarity", 0.0)),
                "confidence": float(last.get("confidence", 0.0)),
            }
        return result

    def _equal_weight_allocation(self, features: pd.DataFrame, current_date: date) -> AllocationResult:
        tickers = list(features.index) if hasattr(features.index, "__iter__") else []
        n = len(tickers)
        if n == 0:
            return AllocationResult(date=current_date, signals=[], cash_weight=1.0)
        w = 1.0 / n
        signals = [
            SignalResult(
                ticker=t, date=current_date, signal="HOLD", weight=w,
                prob_buy=0.33, prob_hold=0.34, prob_sell=0.33,
                sentiment_polarity=0.0, sentiment_confidence=0.0,
            )
            for t in tickers
        ]
        return AllocationResult(date=current_date, signals=signals, cash_weight=0.0)
