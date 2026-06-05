#!/usr/bin/env python3
"""CLI entry point: generate an HTML research report from backtest results."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from report.renderer import ReportRenderer
from report.scoring import CompositeScorer
from config.settings import Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("generate_report")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate ETF strategy research report")
    parser.add_argument("--output", default="./reports/etf_report.html", help="Output HTML path")
    parser.add_argument("--results", default=None, help="JSON file with backtest results (optional)")
    parser.add_argument("--strategy-name", default="ETF 多因子轮动策略", help="Strategy display name")
    parser.add_argument("--benchmark", default="沪深300", help="Benchmark display name")
    parser.add_argument("--commentary", default="", help="AI commentary text")
    args = parser.parse_args(argv)

    # ── Load or generate sample data ─────────
    if args.results and Path(args.results).exists():
        with open(args.results) as f:
            results = json.load(f)
        metrics = results.get("metrics", {})
        chart_data = results.get("chart_data", {})
        allocation = results.get("allocation", [])
    else:
        logger.info("No results file provided — generating sample report with mock data.")
        metrics, chart_data, allocation = _generate_sample_data()

    # ── Compute score ────────────────────────
    scorer = CompositeScorer()
    score = scorer.compute(
        annual_return=metrics.get("annual_return", 0.0),
        sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
        max_drawdown=metrics.get("max_drawdown", 0.0),
        win_rate=metrics.get("win_rate", 0.0),
        calmar_ratio=metrics.get("calmar_ratio", 0.0),
    )

    # ── Render ───────────────────────────────
    renderer = ReportRenderer()
    renderer.render_to_file(
        output_path=args.output,
        metrics=metrics,
        score=score,
        chart_data=chart_data,
        allocation_table=allocation,
        ai_commentary=args.commentary,
        strategy_name=args.strategy_name,
        benchmark_name=args.benchmark,
        start_date=date(2023, 1, 1),
        end_date=date.today(),
    )

    print(f"Report generated: {args.output}")


def _generate_sample_data():
    """Generate plausible sample data for a demo report."""
    dates = pd.bdate_range("2023-01-01", "2025-12-31")
    n = len(dates)
    np.random.seed(42)

    # Simulated equity curve (cumsum of random returns + drift)
    rets = np.random.normal(0.0006, 0.012, n)
    benchmark_rets = np.random.normal(0.0003, 0.010, n)

    portfolio = 1.0 + np.cumsum(rets)
    benchmark = 1.0 + np.cumsum(benchmark_rets)

    # Drawdown
    peak = np.maximum.accumulate(portfolio)
    dd = (peak - portfolio) / peak

    chart_data = {
        "equity_curve": [
            {"date": d.strftime("%Y-%m-%d"), "portfolio": round(v, 4), "benchmark": round(b, 4)}
            for d, v, b in zip(dates[::5], portfolio[::5], benchmark[::5])
        ],
        "drawdown": [
            {"date": d.strftime("%Y-%m-%d"), "value": round(dv * -100, 2)}
            for d, dv in zip(dates[::5], dd[::5])
        ],
        "monthly_returns": [],
        "factor_exposure": [
            {"factor": "动量", "value": 0.45},
            {"factor": "波动率", "value": -0.30},
            {"factor": "流动性", "value": 0.20},
            {"factor": "情绪", "value": 0.35},
            {"factor": "价值", "value": -0.15},
            {"factor": "质量", "value": 0.38},
        ],
    }

    metrics = {
        "annual_return": 0.18,
        "sharpe_ratio": 1.45,
        "max_drawdown": 0.14,
        "win_rate": 0.56,
        "calmar_ratio": 1.28,
    }

    allocation = [
        {"ticker": "SPY", "weight": 0.28, "signal": "BUY", "polarity": 0.62},
        {"ticker": "QQQ", "weight": 0.25, "signal": "BUY", "polarity": 0.55},
        {"ticker": "XLK", "weight": 0.18, "signal": "BUY", "polarity": 0.48},
        {"ticker": "XLV", "weight": 0.14, "signal": "HOLD", "polarity": 0.10},
        {"ticker": "XLF", "weight": 0.10, "signal": "HOLD", "polarity": -0.05},
        {"ticker": "XLE", "weight": 0.05, "signal": "SELL", "polarity": -0.38},
    ]

    return metrics, chart_data, allocation


if __name__ == "__main__":
    main()
