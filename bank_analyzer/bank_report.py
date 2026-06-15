"""银行合作潜力分析报告渲染器 — Jinja2 + ECharts 交互式报告.

输出符合机构研究标准（中信证券 / 国泰海通风格）的 HTML 分析报告。
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


class BankReportRenderer:
    """银行商务拓展合作潜力分析报告渲染器."""

    def __init__(self, template_dir: str | None = None) -> None:
        template_path = template_dir or str(_TEMPLATE_DIR)
        self._env = Environment(
            loader=FileSystemLoader(template_path),
            autoescape=True,
        )
        self._template = self._env.get_template("bank_report.html")

    def render(
        self,
        scores: list[dict[str, Any]],
        pain_points: list[dict[str, Any]],
        chart_data: dict[str, Any],
        summary_text: str = "",
        report_date: date | None = None,
    ) -> str:
        """渲染完整的银行分析报告 HTML.

        Args:
            scores: 排序后的银行评分列表（BankScore.__dict__）
            pain_points: 银行痛点分析列表（BankPainPointAnalysis.__dict__）
            chart_data: ECharts 图表数据
            summary_text: AI 生成的综合分析摘要
            report_date: 报告日期

        Returns:
            完整 HTML 字符串.
        """
        context = {
            "report_date": (report_date or date.today()).isoformat(),
            "scores": scores,
            "pain_points": pain_points,
            "chart_data_json": json.dumps(chart_data, ensure_ascii=False, default=str),
            "summary_text": summary_text,
            "total_banks": len(scores),
            "top_3": [s for s in scores[:3]],
            "recommend_count": sum(1 for s in scores if s.get("cooperation_potential", 0) >= 65),
        }
        return self._template.render(**context)

    def render_to_file(
        self,
        output_path: str,
        scores: list[dict[str, Any]],
        pain_points: list[dict[str, Any]],
        chart_data: dict[str, Any],
        summary_text: str = "",
    ) -> None:
        """渲染报告并写入文件."""
        html = self.render(scores, pain_points, chart_data, summary_text)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html, encoding="utf-8")
        logger.info("银行分析报告已输出至 %s", output_path)
