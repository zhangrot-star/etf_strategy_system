"""Jinja2 + ECharts HTML report renderer — professional brokerage research format."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from report.scoring import ScoreBreakdown

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


class ReportRenderer:
    """Renders institutional-grade interactive research reports.

    Produces a self-contained HTML file styled after Chinese brokerage
    (中信证券 / 国泰海通) research report conventions: light professional
    theme, numbered sections, key-point summaries, risk disclosures, and
    embedded ECharts visualizations.
    """

    def __init__(self, template_dir: str | None = None) -> None:
        template_path = template_dir or str(_TEMPLATE_DIR)
        self._env = Environment(
            loader=FileSystemLoader(template_path),
            autoescape=True,
        )
        self._template = self._env.get_template("report.html")

    def render(
        self,
        metrics: dict[str, Any],
        score: ScoreBreakdown,
        chart_data: dict[str, Any],
        allocation_table: list[dict[str, Any]],
        ai_commentary: str = "",
        strategy_name: str = "ETF Multi-Factor Strategy",
        benchmark_name: str = "沪深300",
        start_date: date | None = None,
        end_date: date | None = None,
        *,
        key_points: list[dict[str, str]] | None = None,
        macro_commentary: str = "",
        sentiment_summary: str = "",
        risk_summary: str = "",
        risk_factors: list[dict[str, str]] | None = None,
        sector_commentary: str = "",
        prediction_commentary: str = "",
        allocation_commentary: str = "",
        technical_commentary: str = "",
    ) -> str:
        """Render the complete report HTML.

        Args:
            metrics: Dict of backtest KPIs (annual_return, sharpe, max_dd, win_rate, calmar).
            score: ScoreBreakdown from CompositeScorer.
            chart_data: JSON-serializable dict with chart series data.
            allocation_table: List of per-ETF allocation rows.
            ai_commentary: Full AI-generated commentary text.
            strategy_name: Display name for the strategy.
            benchmark_name: Display name for the benchmark.
            start_date: Backtest start date.
            end_date: Backtest end date.
            key_points: Investment thesis bullet points (label + content).
            macro_commentary: Market environment analysis text.
            sentiment_summary: Market sentiment overview.
            risk_summary: One-line risk summary for key points.
            risk_factors: Structured risk items (title + desc).
            sector_commentary: Sector allocation analysis.
            prediction_commentary: Multi-horizon prediction analysis.
            allocation_commentary: Portfolio allocation commentary.
            technical_commentary: Technical analysis commentary.

        Returns:
            Complete HTML string.
        """
        context = {
            "strategy_name": strategy_name,
            "benchmark_name": benchmark_name,
            "start_date": str(start_date) if start_date else "N/A",
            "end_date": str(end_date) if end_date else "N/A",
            "report_date": date.today().isoformat(),
            "metrics": metrics,
            "score": score,
            "chart_data_json": json.dumps(chart_data, ensure_ascii=False, default=str),
            "allocation_table": allocation_table,
            "ai_commentary": ai_commentary,
            # Brokerage-style sections
            "key_points": key_points or [],
            "macro_commentary": macro_commentary,
            "sentiment_summary": sentiment_summary,
            "risk_summary": risk_summary,
            "risk_factors": risk_factors or [],
            "sector_commentary": sector_commentary,
            "prediction_commentary": prediction_commentary,
            "allocation_commentary": allocation_commentary,
            "technical_commentary": technical_commentary,
        }
        return self._template.render(**context)

    def render_to_file(
        self,
        output_path: str,
        metrics: dict[str, Any],
        score: ScoreBreakdown,
        chart_data: dict[str, Any],
        allocation_table: list[dict[str, Any]],
        ai_commentary: str = "",
        strategy_name: str = "ETF Multi-Factor Strategy",
        benchmark_name: str = "沪深300",
        start_date: date | None = None,
        end_date: date | None = None,
        **kwargs: Any,
    ) -> None:
        """Render the report and write to an HTML file.

        Accepts all keyword arguments from ``render()``.
        """
        html = self.render(
            metrics=metrics,
            score=score,
            chart_data=chart_data,
            allocation_table=allocation_table,
            ai_commentary=ai_commentary,
            strategy_name=strategy_name,
            benchmark_name=benchmark_name,
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html, encoding="utf-8")
        logger.info("Report written to %s", output_path)
