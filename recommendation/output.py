"""Output formatters for recommendation results."""

from __future__ import annotations

import json
from typing import Any

from recommendation.schemas import DailyRecommendation, RecommendedETF


def to_json(recommendation: DailyRecommendation, indent: int = 2) -> str:
    """Serialize to JSON string."""
    return json.dumps(recommendation.model_dump(mode="json"), indent=indent, ensure_ascii=False)


def to_markdown(recommendation: DailyRecommendation) -> str:
    """Format as Markdown table."""
    lines = [
        f"# ETF Recommendation — {recommendation.date}",
        f"",
        f"**Risk Status:** {recommendation.risk_status} | **Cash Weight:** {recommendation.cash_weight:.1%}",
        f"",
        f"| Rank | Ticker | Score | Rating | Signal | Recommendation | Weight | Risk |",
        f"|------|--------|-------|--------|--------|----------------|--------|------|",
    ]
    for etf in recommendation.ranked_etfs[:20]:
        lines.append(
            f"| {etf.ticker} | {etf.total_score:.1f} | {etf.rating} | "
            f"{etf.ml_signal} | {etf.recommendation} | {etf.allocation_weight:.1%} | {etf.risk_level} |"
        )
    return "\n".join(lines)


def to_dict(recommendation: DailyRecommendation) -> dict[str, Any]:
    """Convert to plain dict (JSON-serializable)."""
    return recommendation.model_dump(mode="json")
