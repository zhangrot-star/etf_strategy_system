"""Module 1: Fund Issuer / Company evaluation (10% weight).

Sub-factors:
  - Issuer size ranking (5%)
  - Issuer profitability vs industry average (5%)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class IssuerFactorComputer:
    """Compute Module 1 factors: issuer size and profitability."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        issuer_cfg = cfg.get("issuer", {})
        self._top_tier_limit: int = issuer_cfg.get("top_tier_limit", 5)
        self._second_tier_limit: int = issuer_cfg.get("second_tier_limit", 10)
        self._third_tier_limit: int = issuer_cfg.get("third_tier_limit", 20)

    def issuer_size_score(self, aum_rank: int | None) -> float:
        """Score issuer by AUM ranking.

        Args:
            aum_rank: 1-based rank (1 = largest). None if unknown.

        Returns:
            Score 0-10.
        """
        if aum_rank is None:
            return 4.0
        if aum_rank <= self._top_tier_limit:
            return 10.0
        if aum_rank <= self._second_tier_limit:
            return 8.0
        if aum_rank <= self._third_tier_limit:
            return 6.0
        return 4.0

    def issuer_profitability_score(self, roe: float | None, industry_median_roe: float | None) -> float:
        """Score issuer by ROE relative to industry median.

        Args:
            roe: Issuer return on equity.
            industry_median_roe: Industry median ROE for the same period.

        Returns:
            Score 0-10.
        """
        if roe is None or industry_median_roe is None:
            return 5.0
        return 10.0 if roe >= industry_median_roe else 5.0

    def compute_module(
        self,
        issuer_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute Module 1 scores for all issuers.

        Args:
            issuer_df: DataFrame with columns [issuer_id, aum_rank, roe, industry_median_roe].

        Returns:
            DataFrame with columns [issuer_id, issuer_size_score, issuer_profitability_score, issuer_module_total].
        """
        if issuer_df.empty:
            return pd.DataFrame(columns=["issuer_id", "issuer_size_score", "issuer_profitability_score", "issuer_module_total"])

        result = issuer_df[["issuer_id"]].copy()

        result["issuer_size_score"] = issuer_df["aum_rank"].apply(self.issuer_size_score) if "aum_rank" in issuer_df.columns else 5.0
        result["issuer_profitability_score"] = issuer_df.apply(
            lambda r: self.issuer_profitability_score(r.get("roe"), r.get("industry_median_roe")), axis=1,
        ) if "roe" in issuer_df.columns else 5.0

        result["issuer_module_total"] = (
            result["issuer_size_score"] * 0.5
            + result["issuer_profitability_score"] * 0.5
        )

        return result
