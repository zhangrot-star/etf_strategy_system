"""Tests for report module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from report.scoring import CompositeScorer, ScoreBreakdown
from report.renderer import ReportRenderer


class TestCompositeScorer:
    def test_perfect_score(self):
        scorer = CompositeScorer()
        score = scorer.compute(
            annual_return=0.35,
            sharpe_ratio=3.5,
            max_drawdown=0.0,
            win_rate=1.0,
            calmar_ratio=4.0,
        )
        assert score.total_score > 90
        assert score.rating in ("S", "A")

    def test_poor_score(self):
        scorer = CompositeScorer()
        score = scorer.compute(
            annual_return=-0.10,
            sharpe_ratio=-0.5,
            max_drawdown=0.45,
            win_rate=0.30,
            calmar_ratio=-0.2,
        )
        assert score.total_score < 35

    def test_score_is_between_0_and_100(self):
        scorer = CompositeScorer()
        score = scorer.compute(0.05, 0.5, 0.15, 0.5, 0.5)
        assert 0 <= score.total_score <= 100

    def test_score_breakdown_fields(self):
        scorer = CompositeScorer()
        score = scorer.compute(0.15, 1.5, 0.10, 0.55, 1.0)
        assert 0 <= score.annual_return_score <= 25
        assert 0 <= score.sharpe_score <= 25
        assert 0 <= score.max_drawdown_score <= 20
        assert 0 <= score.win_rate_score <= 15
        assert 0 <= score.calmar_score <= 15
        assert score.rating_label in ("卓越", "优秀", "良好", "一般", "较差", "高风险")


class TestReportRenderer:
    def test_render_produces_html(self):
        renderer = ReportRenderer()
        metrics = {"annual_return": 0.15, "sharpe_ratio": 1.2, "max_drawdown": 0.10, "win_rate": 0.55, "calmar_ratio": 0.8}
        scorer = CompositeScorer()
        score = scorer.compute(**{k: metrics[k] for k in ("annual_return", "sharpe_ratio", "max_drawdown", "win_rate", "calmar_ratio")})

        chart_data = {
            "equity_curve": [{"date": "2023-01-01", "portfolio": 1.0, "benchmark": 1.0}],
            "drawdown": [{"date": "2023-01-01", "value": 0}],
            "monthly_returns": [],
            "factor_exposure": [{"factor": "momentum", "value": 0.5}],
        }
        allocation = [{"ticker": "SPY", "weight": 0.5, "signal": "BUY", "polarity": 0.5}]

        html = renderer.render(
            metrics=metrics,
            score=score,
            chart_data=chart_data,
            allocation_table=allocation,
            ai_commentary="Test analysis.",
        )
        assert "<html" in html
        assert "echarts" in html
        assert "SPY" in html
        assert "Test analysis." in html

    def test_render_to_file(self):
        renderer = ReportRenderer()
        scorer = CompositeScorer()
        metrics = {"annual_return": 0.10, "sharpe_ratio": 0.8, "max_drawdown": 0.15, "win_rate": 0.5, "calmar_ratio": 0.5}
        score = scorer.compute(**{k: metrics[k] for k in ("annual_return", "sharpe_ratio", "max_drawdown", "win_rate", "calmar_ratio")})
        chart_data = {"equity_curve": [], "drawdown": [], "monthly_returns": [], "factor_exposure": []}
        allocation = []

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        renderer.render_to_file(
            output_path=output_path,
            metrics=metrics,
            score=score,
            chart_data=chart_data,
            allocation_table=allocation,
        )
        content = Path(output_path).read_text()
        assert "<html" in content
        Path(output_path).unlink()
