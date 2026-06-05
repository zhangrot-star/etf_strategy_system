"""Performance attribution: decomposing total return into beta, alpha, and sentiment-alpha."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """Decomposed performance attribution."""

    total_return: float
    total_annualized_return: float
    total_volatility: float
    sharpe_ratio: float

    # Beta decomposition
    market_beta: float
    systematic_beta_return: float  # portion of return explained by market beta

    # Alpha decomposition
    statistical_alpha: float       # residual from factor regression (annualized)
    ff3_alpha: float               # Fama-French 3-factor alpha
    ff3_hml_beta: float            # value factor loading
    ff3_smb_beta: float            # size factor loading

    # Sentiment overlay
    sentiment_alpha: float         # excess return attributable to sentiment strategy overlay
    sentiment_breach_count: int    # number of times circuit breaker triggered

    # Summary
    summary: str = ""


class PerformanceAttribution:
    """Decompose total portfolio return into systematic and idiosyncratic components.

    Three-layer decomposition:
    1. CAPM beta → systematic market contribution
    2. Fama-French 3-factor → statistical alpha net of style factors
    3. Sentiment overlay → incremental alpha from LLM-driven risk control
    """

    def __init__(self, rf_rate: float = 0.03) -> None:
        """rf_rate: annualized risk-free rate (default 3%)."""
        self._rf_rate = rf_rate
        self._rf_daily = (1 + rf_rate) ** (1 / 252) - 1

    # ── Main attribution ────────────────────────────────────

    def decompose(
        self,
        portfolio_returns: pd.Series,
        market_returns: pd.Series,
        ff_factors: pd.DataFrame | None = None,
        sentiment_events: pd.DataFrame | None = None,
    ) -> AttributionResult:
        """Run full performance attribution.

        Args:
            portfolio_returns: Daily portfolio return series.
            market_returns: Daily market benchmark return series (e.g., SPY).
            ff_factors: Fama-French factors — columns: Mkt-RF, SMB, HML.
            sentiment_events: DataFrame with columns date, is_breached for
                              each circuit breaker trigger.

        Returns:
            AttributionResult with decomposed metrics.
        """
        aligned = pd.DataFrame(
            {"portfolio": portfolio_returns, "market": market_returns}
        ).dropna()

        if aligned.empty:
            return self._empty_result()

        excess_ret = aligned["portfolio"] - self._rf_daily
        excess_mkt = aligned["market"] - self._rf_daily

        # ── Layer 1: CAPM beta ───────────────────────────────
        beta_model = sm.OLS(excess_ret, sm.add_constant(excess_mkt)).fit()
        market_beta = beta_model.params.iloc[1] if len(beta_model.params) > 1 else 1.0
        systematic_return = market_beta * excess_mkt.mean() * 252
        capm_alpha = beta_model.params.iloc[0] * 252  # annualized

        # ── Layer 2: Fama-French 3-factor ────────────────────
        ff3_alpha = np.nan
        ff3_smb = np.nan
        ff3_hml = np.nan
        if ff_factors is not None and not ff_factors.empty:
            ff_merged = aligned.join(ff_factors, how="inner")
            if len(ff_merged) > 60:
                ff_exog = ff_merged[["Mkt-RF", "SMB", "HML"]]
                ff_model = sm.OLS(
                    ff_merged["portfolio"] - self._rf_daily,
                    sm.add_constant(ff_exog),
                ).fit()
                ff3_alpha = ff_model.params.iloc[0] * 252
                ff3_smb = ff_model.params.get("SMB", np.nan)
                ff3_hml = ff_model.params.get("HML", np.nan)

        # ── Layer 3: Sentiment alpha ──────────────────────────
        sentiment_alpha = 0.0
        breach_count = 0
        if sentiment_events is not None and not sentiment_events.empty:
            sent_merged = aligned.join(
                sentiment_events.set_index("date")["is_breached"].fillna(False),
                how="left",
            )
            sent_merged["is_breached"] = sent_merged["is_breached"].fillna(False)
            breach_count = int(sent_merged["is_breached"].sum())

            # Sentiment alpha = excess return on days *with signal* minus
            # expected return from CAPM+FF on those same days
            # (simplified: compare mean return on non-breached days vs all days)
            normal_days = sent_merged[sent_merged["is_breached"] == False]
            breached_days = sent_merged[sent_merged["is_breached"] == True]

            if len(normal_days) > 0 and len(breached_days) > 0:
                avg_normal = normal_days["portfolio"].mean()
                avg_breached = breached_days["portfolio"].mean()
                # Alpha = avoided loss (if breaching correctly avoided drawdowns)
                sentiment_alpha = (avg_normal - avg_breached) * breach_count

        # ── Aggregate metrics ─────────────────────────────────
        total_ret = aligned["portfolio"].sum()
        total_ann = (1 + aligned["portfolio"].mean()) ** 252 - 1
        total_vol = aligned["portfolio"].std() * np.sqrt(252)
        sharpe = (total_ann - self._rf_rate) / total_vol if total_vol > 0 else 0.0

        return AttributionResult(
            total_return=total_ret,
            total_annualized_return=total_ann,
            total_volatility=total_vol,
            sharpe_ratio=sharpe,
            market_beta=market_beta,
            systematic_beta_return=systematic_return,
            statistical_alpha=capm_alpha,
            ff3_alpha=ff3_alpha,
            ff3_hml_beta=ff3_hml,
            ff3_smb_beta=ff3_smb,
            sentiment_alpha=sentiment_alpha,
            sentiment_breach_count=breach_count,
            summary=(
                f"Total Return: {total_ann:.2%}, Sharpe: {sharpe:.2f}\n"
                f"CAPM β: {market_beta:.2f}, Systematic Return: {systematic_return:.2%}\n"
                f"CAPM α: {capm_alpha:.2%}, FF3 α: {ff3_alpha:.2%}\n"
                f"Sentiment α: {sentiment_alpha:.4%}, Breaches: {breach_count}"
            ),
        )

    @staticmethod
    def _empty_result() -> AttributionResult:
        return AttributionResult(
            total_return=0.0,
            total_annualized_return=0.0,
            total_volatility=0.0,
            sharpe_ratio=0.0,
            market_beta=0.0,
            systematic_beta_return=0.0,
            statistical_alpha=0.0,
            ff3_alpha=0.0,
            ff3_hml_beta=0.0,
            ff3_smb_beta=0.0,
            sentiment_alpha=0.0,
            sentiment_breach_count=0,
            summary="Insufficient data for attribution.",
        )
