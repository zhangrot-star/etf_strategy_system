"""LLM sentiment-based risk controller with circuit breaker logic.

When sentiment polarity breaches hardcoded thresholds, the controller forces
a full liquidation override, regardless of the ensemble's signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum

import pandas as pd

from config.settings import Settings

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    BREACHED = "BREACHED"


@dataclass
class RiskEvent:
    """Output of a risk controller check."""

    timestamp: datetime
    risk_level: RiskLevel
    is_breached: bool
    reason: str
    # If breached, override_weights forces all positions to 0
    override_weights: dict[str, float] = field(default_factory=dict)
    current_polarity: float = 0.0
    current_confidence: float = 0.0

    @property
    def should_liquidate(self) -> bool:
        return self.risk_level == RiskLevel.BREACHED


class RiskController:
    """Hardcoded sentiment-driven circuit breaker.

    Thresholds (configurable via Settings):
    - BREACH: polarity < -0.7  OR  (confidence > 0.85 AND polarity < -0.5)
    - WARNING: polarity < -0.3  (but not yet breached)

    When breached, outputs a full liquidation override.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._breach_polarity = self._settings.sentiment_breach_threshold
        self._warn_polarity = self._settings.sentiment_warn_threshold
        self._breach_confidence = self._settings.sentiment_confidence_threshold

    # ── Single check ─────────────────────────────────────────

    def check(
        self, polarity: float, confidence: float, context: str = ""
    ) -> RiskEvent:
        """Evaluate risk for a single polarity/confidence observation.

        Args:
            polarity: Sentiment polarity in [-1, +1].
            confidence: LLM confidence in [0, 1].
            context: Optional identifier or description.

        Returns:
            RiskEvent with level and override instructions.
        """
        now = datetime.now(tz=timezone.utc).replace(tzinfo=None)

        # Breach conditions
        if polarity < self._breach_polarity:
            return RiskEvent(
                timestamp=now,
                risk_level=RiskLevel.BREACHED,
                is_breached=True,
                reason=f"Polarity {polarity:.2f} below hard breach threshold {self._breach_polarity}. {context}",
                current_polarity=polarity,
                current_confidence=confidence,
            )

        if confidence > self._breach_confidence and polarity < self._warn_polarity:
            return RiskEvent(
                timestamp=now,
                risk_level=RiskLevel.BREACHED,
                is_breached=True,
                reason=(
                    f"High-confidence bearish signal: polarity={polarity:.2f}, "
                    f"confidence={confidence:.2f} > {self._breach_confidence}. {context}"
                ),
                current_polarity=polarity,
                current_confidence=confidence,
            )

        # Warning condition
        if polarity < self._warn_polarity:
            return RiskEvent(
                timestamp=now,
                risk_level=RiskLevel.WARNING,
                is_breached=False,
                reason=f"Polarity {polarity:.2f} below warning threshold {self._warn_polarity}. {context}",
                current_polarity=polarity,
                current_confidence=confidence,
            )

        # All clear
        return RiskEvent(
            timestamp=now,
            risk_level=RiskLevel.NORMAL,
            is_breached=False,
            reason="Sentiment within normal range.",
            current_polarity=polarity,
            current_confidence=confidence,
        )

    # ── Portfolio-level check ────────────────────────────────

    def check_portfolio(
        self,
        sentiment_df: pd.DataFrame,
        holdings: list[str],
    ) -> RiskEvent:
        """Aggregate sentiment across current holdings.

        sentiment_df must have columns: ticker, polarity, confidence.
        The worst (most negative) sentiment across all holdings determines
        the portfolio risk level.

        Returns a RiskEvent; if breached, override_weights will zero out
        all holdings.
        """
        now = datetime.now(tz=timezone.utc).replace(tzinfo=None)

        if sentiment_df.empty:
            return RiskEvent(
                timestamp=now,
                risk_level=RiskLevel.NORMAL,
                is_breached=False,
                reason="No sentiment data available.",
            )

        relevant = sentiment_df[sentiment_df["ticker"].isin(holdings)]
        if relevant.empty:
            return RiskEvent(
                timestamp=now,
                risk_level=RiskLevel.NORMAL,
                is_breached=False,
                reason="No sentiment data for current holdings.",
            )

        # Find worst sentiment
        worst_idx = relevant["polarity"].idxmin()
        worst_polarity = relevant.loc[worst_idx, "polarity"]
        worst_confidence = relevant.loc[worst_idx, "confidence"]
        worst_ticker = relevant.loc[worst_idx, "ticker"]

        event = self.check(
            polarity=worst_polarity,
            confidence=worst_confidence,
            context=f"Worst ticker={worst_ticker}",
        )

        if event.is_breached:
            event.override_weights = {t: 0.0 for t in holdings}
            logger.warning(
                "CIRCUIT BREAKER TRIPPED: %s. Liquidating all positions.",
                event.reason,
            )

        return event

    # ── Position-level check ─────────────────────────────────

    def check_position(
        self, ticker: str, polarity: float, confidence: float
    ) -> RiskEvent:
        """Check if a single position should be closed."""
        event = self.check(polarity, confidence, context=f"ticker={ticker}")
        if event.is_breached:
            event.override_weights = {ticker: 0.0}
        return event
