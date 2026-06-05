"""ETF ranking and filtering engine."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCORE_TO_RATING: list[tuple[float, str]] = [
    (90, "S"), (80, "A"), (70, "B"), (55, "C"), (40, "D"), (0, "F"),
]


class ETFRanker:
    """Filters, ranks, and computes allocation weights for scored ETFs."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        r = cfg.get("recommendation", {})
        self._top_n: int = r.get("top_n", 10)
        self._min_aum: float = r.get("min_aum", 1e8)
        self._min_age_days: int = r.get("min_age_days", 180)
        self._max_per_sector: int = r.get("max_per_sector", 3)
        self._default_weight_method: str = r.get("default_weight_method", "score_weighted")

    def rank(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Sort by adjusted_total descending and assign ranks."""
        col = "adjusted_total" if "adjusted_total" in scores_df.columns else "raw_total"
        result = scores_df.sort_values(col, ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)
        return result

    def compute_weights(
        self,
        scores_df: pd.DataFrame,
        top_n: int | None = None,
        method: str | None = None,
        max_per_position: float = 0.30,
    ) -> dict[str, float]:
        """Convert scores to allocation weights.

        Methods:
          - score_weighted: weight proportional to adjusted_total
          - equal_weight: 1/N each

        Args:
            max_per_position: Cap on any single position weight (0.0-1.0).
        """
        top_n = top_n or self._top_n
        method = method or self._default_weight_method

        col = "adjusted_total" if "adjusted_total" in scores_df.columns else "raw_total"
        top = scores_df.nlargest(top_n, col)

        if method == "equal_weight":
            raw = {r["ticker"]: 1.0 / len(top) for _, r in top.iterrows()}
        else:
            total = top[col].sum()
            if total <= 0:
                w = 1.0 / len(top)
                raw = {r["ticker"]: w for _, r in top.iterrows()}
            else:
                raw = {r["ticker"]: r[col] / total for _, r in top.iterrows()}

        # Apply position cap and redistribute excess
        if not raw:
            return {}

        capped: dict[str, float] = {}
        excess = 0.0
        for ticker, weight in raw.items():
            if weight > max_per_position:
                capped[ticker] = max_per_position
                excess += weight - max_per_position
            else:
                capped[ticker] = weight

        if excess > 0:
            eligible = [t for t, w in capped.items() if w < max_per_position]
            if eligible:
                add = excess / len(eligible)
                for t in eligible:
                    capped[t] = min(capped[t] + add, max_per_position)

        return capped

    def assign_ratings(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Map adjusted_total to S/A/B/C/D/F ratings."""
        col = "adjusted_total" if "adjusted_total" in scores_df.columns else "raw_total"

        def _rating(score: float) -> str:
            for threshold, grade in SCORE_TO_RATING:
                if score >= threshold:
                    return grade
            return "F"

        scores_df["rating"] = scores_df[col].apply(_rating)
        return scores_df

    def risk_level(self, score: float) -> str:
        if score >= 80:
            return "LOW"
        if score >= 60:
            return "MEDIUM"
        return "HIGH"
