"""Module 2: Index / Strategy Quality evaluation (40% weight).

Sub-factors:
  - Tracking error (8%)
  - Index methodology quality (8%)
  - Liquidity (8%)
  - Fund age (4%)
  - Expense ratio (4%)
  - Dividend yield (4%)
  - Premium/discount stability (4%)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class IndexQualityFactorComputer:
    """Compute Module 2 factors: index quality, tracking, liquidity, etc."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        iq = cfg.get("index_quality", {})
        self._tracking_error_target: float = iq.get("tracking_error_target", 0.02)
        self._liquidity_target: float = iq.get("liquidity_target", 0.05)
        self._min_fund_age_years: float = iq.get("min_fund_age_years", 3.0)
        self._expense_ratio_max_a: float = iq.get("expense_ratio_max_a", 0.005)
        self._expense_ratio_max_us: float = iq.get("expense_ratio_max_us", 0.0015)
        self._premium_discount_stable: float = iq.get("premium_discount_stable", 0.005)

    # ── Individual factor scorers ─────────────────────────────

    def tracking_error_score(self, te: float | None) -> float:
        """Score tracking error (lower is better).

        te: Annualized std of (ETF return - benchmark return).
        """
        if te is None or np.isnan(te):
            return 5.0
        if te < 0.01:
            return 10.0
        if te < 0.02:
            return 8.0
        if te < 0.03:
            return 6.0
        if te < 0.05:
            return 4.0
        return 2.0

    def methodology_score(self, is_public: bool, n_constituents: int,
                          has_transparent_rebal: bool, rebal_quarterly: bool) -> float:
        """Score index methodology quality (composite)."""
        score = 0.0
        if is_public:
            score += 3.0
        if n_constituents >= 50:
            score += 3.0
        if has_transparent_rebal:
            score += 2.0
        if rebal_quarterly:
            score += 2.0
        return score

    def liquidity_score(self, daily_turnover: float | None) -> float:
        """Score liquidity by average daily volume / AUM ratio.

        daily_turnover: fraction of AUM traded daily.
        """
        if daily_turnover is None or np.isnan(daily_turnover):
            return 5.0
        if daily_turnover > 0.05:
            return 10.0
        if daily_turnover > 0.02:
            return 8.0
        if daily_turnover > 0.01:
            return 6.0
        if daily_turnover > 0.005:
            return 4.0
        return 2.0

    def fund_age_score(self, years: float | None) -> float:
        """Score fund age (older = more established)."""
        if years is None:
            return 5.0
        if years > 5:
            return 10.0
        if years > 3:
            return 8.0
        if years > 1:
            return 6.0
        if years > 0.5:
            return 4.0
        return 2.0

    def expense_ratio_score(self, ter: float | None, market: str = "A") -> float:
        """Score expense ratio (lower is better)."""
        if ter is None or np.isnan(ter):
            return 5.0
        limit = self._expense_ratio_max_a if market == "A" else self._expense_ratio_max_us
        if ter < limit * 0.5:
            return 10.0
        if ter < limit:
            return 8.0
        if ter < limit * 2:
            return 6.0
        if ter < limit * 4:
            return 4.0
        return 2.0

    def dividend_yield_score(self, div_yield: float | None, category_median: float | None) -> float:
        """Score dividend yield as z-score within peer category."""
        if div_yield is None or category_median is None:
            return 5.0
        if category_median == 0:
            return 5.0
        z = (div_yield - category_median) / abs(category_median) if abs(category_median) > 0 else 0.0
        return float(np.clip(5.0 + z * 2.5, 0.0, 10.0))

    def premium_discount_score(self, pd_std: float | None) -> float:
        """Score stability of premium/discount (lower std = better)."""
        if pd_std is None or np.isnan(pd_std):
            return 5.0
        if pd_std < 0.005:
            return 10.0
        if pd_std < 0.01:
            return 8.0
        if pd_std < 0.02:
            return 6.0
        if pd_std < 0.03:
            return 4.0
        return 2.0

    # ── Module aggregator ────────────────────────────────────

    def compute_module(
        self, profiles: pd.DataFrame, index_meta: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute Module 2 scores for all ETFs.

        Args:
            profiles: DataFrame with columns [ticker, inception_date, expense_ratio, market, avg_daily_volume, aum].
            index_meta: DataFrame with columns [ticker, tracking_error, is_public, n_constituents,
                        has_transparent_rebal, rebal_quarterly, dividend_yield, category_div_yield_median, premium_discount_std].

        Returns:
            DataFrame with all Module 2 sub-scores and total.
        """
        if profiles.empty:
            return pd.DataFrame()

        result = profiles[["ticker"]].copy()

        # Merge meta
        meta = index_meta.copy() if not index_meta.empty else pd.DataFrame(columns=["ticker"])
        merged = result.merge(meta, on="ticker", how="left") if not meta.empty else result

        # Tracking error
        result["tracking_error_score"] = merged.get("tracking_error", pd.Series()).apply(self.tracking_error_score) if "tracking_error" in merged.columns else 5.0

        # Methodology
        if all(c in merged.columns for c in ["is_public", "n_constituents", "has_transparent_rebal", "rebal_quarterly"]):
            result["methodology_score"] = merged.apply(
                lambda r: self.methodology_score(r["is_public"], r["n_constituents"], r["has_transparent_rebal"], r["rebal_quarterly"]),
                axis=1,
            )
        else:
            result["methodology_score"] = 5.0

        # Liquidity
        if "avg_daily_volume" in profiles.columns and "aum" in profiles.columns:
            turnover = profiles["avg_daily_volume"] / profiles["aum"].replace(0, np.nan)
            result["liquidity_score"] = turnover.apply(self.liquidity_score)
        else:
            result["liquidity_score"] = 5.0

        # Fund age
        if "inception_date" in profiles.columns:
            today = pd.Timestamp.now()
            age_years = (today - pd.to_datetime(profiles["inception_date"])).dt.days / 365.25
            result["fund_age_score"] = age_years.apply(self.fund_age_score)
        else:
            result["fund_age_score"] = 5.0

        # Expense ratio
        if "expense_ratio" in profiles.columns:
            market = profiles.get("market", pd.Series(["A"] * len(profiles)))
            result["expense_ratio_score"] = profiles.apply(
                lambda r: self.expense_ratio_score(r["expense_ratio"], market.loc[r.name] if hasattr(market, "loc") else "A"),
                axis=1,
            )
        else:
            result["expense_ratio_score"] = 5.0

        # Dividend yield
        result["dividend_yield_score"] = 5.0
        if "dividend_yield" in merged.columns:
            cat_med = merged.get("category_div_yield_median", pd.Series())
            result["dividend_yield_score"] = merged.apply(
                lambda r: self.dividend_yield_score(
                    r.get("dividend_yield"), r.get("category_div_yield_median"),
                ), axis=1,
            )

        # Premium/discount stability
        result["premium_discount_score"] = merged.get("premium_discount_std", pd.Series()).apply(self.premium_discount_score) if "premium_discount_std" in merged.columns else 5.0

        # Weighted total
        result["index_module_total"] = (
            result["tracking_error_score"] * 0.8
            + result["methodology_score"] * 0.8
            + result["liquidity_score"] * 0.8
            + result["fund_age_score"] * 0.4
            + result["expense_ratio_score"] * 0.4
            + result["dividend_yield_score"] * 0.4
            + result["premium_discount_score"] * 0.4
        )

        return result
