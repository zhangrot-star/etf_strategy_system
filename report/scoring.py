"""100-point composite strategy scoring algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScoreBreakdown:
    """Detailed component scores for the 100-point scoring system."""

    annual_return_score: float = 0.0    # max 25
    sharpe_score: float = 0.0           # max 25
    max_drawdown_score: float = 0.0     # max 20
    win_rate_score: float = 0.0         # max 15
    calmar_score: float = 0.0           # max 15

    total_score: float = 0.0

    # Raw metrics for display
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    calmar_ratio: float = 0.0

    rating: str = "C"
    rating_label: str = "一般"


class CompositeScorer:
    """100-point multi-dimensional scoring system for ETF strategies.

    Scoring rules:
    - Annualized Return (25 pts): linear 0→25 from return 0% → 30%+
    - Sharpe Ratio (25 pts): linear 0→25 from Sharpe 0 → 3.0+
    - Max Drawdown (20 pts): linear inverse 20→0 from DD 0% → 30%+
    - Win Rate (15 pts): linear 0→15 from win rate 0% → 100%
    - Calmar Ratio (15 pts): linear 0→15 from Calmar 0 → 3.0+
    """

    def compute(
        self,
        annual_return: float,
        sharpe_ratio: float,
        max_drawdown: float,
        win_rate: float,
        calmar_ratio: float,
    ) -> ScoreBreakdown:
        breakdown = ScoreBreakdown(
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            calmar_ratio=calmar_ratio,
        )

        # Return: 0% → 0 pts, 30%+ → 25 pts
        breakdown.annual_return_score = _scale_score(annual_return, 0.0, 0.30, 25.0)

        # Sharpe: 0 → 0 pts, 3.0+ → 25 pts
        breakdown.sharpe_score = _scale_score(sharpe_ratio, 0.0, 3.0, 25.0)

        # Drawdown: 0% → 20 pts, 30%+ → 0 pts (inverse linear)
        breakdown.max_drawdown_score = _scale_score_inverse(max_drawdown, 0.0, 0.30, 20.0)

        # Win rate: 0% → 0 pts, 100% → 15 pts
        breakdown.win_rate_score = _scale_score(win_rate, 0.0, 1.0, 15.0)

        # Calmar: 0 → 0 pts, 3.0+ → 15 pts
        breakdown.calmar_score = _scale_score(calmar_ratio, 0.0, 3.0, 15.0)

        breakdown.total_score = round(
            breakdown.annual_return_score
            + breakdown.sharpe_score
            + breakdown.max_drawdown_score
            + breakdown.win_rate_score
            + breakdown.calmar_score,
            1,
        )

        breakdown.rating, breakdown.rating_label = _rating(total=breakdown.total_score)
        return breakdown


def _scale_score(value: float, lo: float, hi: float, max_points: float) -> float:
    """Linear scale: value in [lo, hi] → points in [0, max_points]."""
    if hi <= lo:
        return 0.0
    scaled = (value - lo) / (hi - lo)
    return round(max(0.0, min(max_points, scaled * max_points)), 1)


def _scale_score_inverse(value: float, lo: float, hi: float, max_points: float) -> float:
    """Inverse linear: value in [lo, hi] → points in [max_points, 0]."""
    if hi <= lo:
        return max_points
    scaled = 1.0 - (value - lo) / (hi - lo)
    return round(max(0.0, min(max_points, scaled * max_points)), 1)


def _rating(total: float) -> tuple[str, str]:
    if total >= 90:
        return "S", "卓越"
    elif total >= 80:
        return "A", "优秀"
    elif total >= 65:
        return "B", "良好"
    elif total >= 50:
        return "C", "一般"
    elif total >= 35:
        return "D", "较差"
    else:
        return "F", "高风险"
