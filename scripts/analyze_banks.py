#!/usr/bin/env python3
"""银行信用卡业务合作潜力分析 — 完整管线执行脚本.

用法:
  python scripts/analyze_banks.py                    # 全量分析 18 家银行
  python scripts/analyze_banks.py --top 5            # 只输出 Top 5
  python scripts/analyze_banks.py --bank CMB,ICBC    # 指定银行分析
  python scripts/analyze_banks.py --train            # 训练评分模型
  python scripts/analyze_banks.py --report           # 生成 HTML 报告
  python scripts/analyze_banks.py --output json      # JSON 格式输出
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bank_analyzer.bank_data import BankDataPipeline, BANK_UNIVERSE
from bank_analyzer.bank_scorer import BankScorer
from bank_analyzer.bank_pain_points import BankPainPointAnalyzer
from bank_analyzer.bank_report import BankReportRenderer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("analyze_banks")


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_chart_data(scores: list[Any]) -> dict[str, Any]:
    """从评分结果构建 ECharts 图表数据."""
    bank_names = [s.bank_name for s in scores]
    cooperation_scores = [round(s.cooperation_potential, 1) for s in scores]

    # 雷达图数据（Top 6）
    top6 = scores[:6]
    radar_banks = [s.bank_name for s in top6]
    radar_data = [
        [s.module_a_total, s.module_b_total, s.module_c_total, s.module_d_total]
        for s in top6
    ]

    # 热力图数据（全部）
    heatmap_banks = bank_names
    heatmap_data = [
        [s.module_a_total, s.module_b_total, s.module_c_total, s.module_d_total]
        for s in scores
    ]

    return {
        "bank_names": bank_names,
        "cooperation_scores": cooperation_scores,
        "radar_banks": radar_banks,
        "radar_data": radar_data,
        "heatmap_banks": heatmap_banks,
        "heatmap_data": heatmap_data,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="银行信用卡业务合作潜力分析")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--top", type=int, default=0, help="只输出 Top N 结果")
    parser.add_argument("--bank", type=str, default="", help="指定银行ID，逗号分隔 (如 CMB,ICBC)")
    parser.add_argument("--train", action="store_true", help="训练 XGBoost 评分模型")
    parser.add_argument("--report", action="store_true", help="生成 HTML 报告")
    parser.add_argument("--output", choices=["table", "json"], default="table", help="输出格式")
    parser.add_argument("--no-pain-points", action="store_true", help="跳过 AI 痛点分析")
    args = parser.parse_args()

    config = load_config(args.config)
    bank_cfg = config.get("bank_analyzer", {})

    # 确定目标银行
    if args.bank:
        bank_ids = [b.strip() for b in args.bank.split(",")]
    else:
        bank_ids = bank_cfg.get("monitored_banks", list(BANK_UNIVERSE.keys()))

    logger.info("目标银行: %d 家", len(bank_ids))

    # ── 1. 数据管线 ──
    logger.info("=" * 50)
    logger.info("Step 1/4: 银行数据采集与特征工程")
    pipeline = BankDataPipeline(config)
    df = pipeline.run(bank_ids)
    logger.info("数据采集完成 — %d 行, %d 特征列", len(df), len(df.columns) - 2)

    # ── 2. 评分模型 ──
    logger.info("=" * 50)
    logger.info("Step 2/4: 合作潜力评分")
    scorer = BankScorer(config)

    if args.train:
        logger.info("训练 XGBoost 评分模型...")
        scorer.fit(df)
        model_path = bank_cfg.get("model_path", "models/bank_scorer")
        scorer.save(model_path)
        logger.info("模型已保存至 %s", model_path)
    else:
        # 尝试加载已有模型
        model_path = bank_cfg.get("model_path", "models/bank_scorer")
        try:
            scorer.load(model_path)
            logger.info("已加载已有评分模型: %s", model_path)
        except Exception:
            logger.info("未找到已有模型，使用启发式评分 + 自动训练")
            scorer.fit(df)

    scores = scorer.score_all(df)
    logger.info("评分完成 — %d 家银行", len(scores))

    # ── 3. 痛点分析 ──
    pain_points = []
    if not args.no_pain_points:
        logger.info("=" * 50)
        logger.info("Step 3/4: AI 银行痛点分析")
        analyzer = BankPainPointAnalyzer(config)

        # 收集 BankProfile 用于 AI 分析
        profiles = pipeline._collector.collect_all_profiles(bank_ids)
        pain_points = analyzer.analyze_all(profiles)
        logger.info("痛点分析完成 — %d 家银行", len(pain_points))
    else:
        logger.info("Step 3/4: 跳过 AI 痛点分析 (--no-pain-points)")

    # ── 4. 输出 ──
    logger.info("=" * 50)
    logger.info("Step 4/4: 结果输出")

    if args.top > 0:
        scores = scores[:args.top]

    if args.output == "json":
        output = {
            "report_date": date.today().isoformat(),
            "total_banks": len(scores),
            "scores": [
                {
                    "rank": s.rank, "bank_name": s.bank_name, "bank_type": s.bank_type,
                    "module_a_total": round(s.module_a_total, 1),
                    "module_b_total": round(s.module_b_total, 1),
                    "module_c_total": round(s.module_c_total, 1),
                    "module_d_total": round(s.module_d_total, 1),
                    "cooperation_potential": round(s.cooperation_potential, 1),
                    "ml_signal": s.ml_signal,
                    "recommendation": s.recommendation,
                    "risk_warning": s.risk_warning,
                }
                for s in scores
            ],
            "pain_points": [
                {
                    "bank_name": pp.bank_name,
                    "strategic_focus": pp.strategic_focus,
                    "business_pain_points": pp.business_pain_points,
                    "cooperation_opportunities": pp.cooperation_opportunities,
                    "risk_assessment": pp.risk_assessment,
                    "summary": pp.summary,
                }
                for pp in pain_points
            ],
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        # 表格输出
        header = f"{'排名':<4} {'银行':<8} {'类型':<6} {'规模A':<7} {'卡B':<7} {'数字C':<7} {'稳定D':<7} {'潜力分':<8} {'信号':<12} {'推荐':<8}"
        print(f"\n{'=' * len(header)}")
        print("  银行信用卡业务合作潜力分析报告")
        print(f"  报告日期: {date.today().isoformat()}")
        print(f"{'=' * len(header)}")
        print(header)
        print("-" * len(header))

        for s in scores:
            print(
                f"{s.rank:<4} "
                f"{s.bank_name:<8} "
                f"{s.bank_type:<6} "
                f"{s.module_a_total:<7.1f} "
                f"{s.module_b_total:<7.1f} "
                f"{s.module_c_total:<7.1f} "
                f"{s.module_d_total:<7.1f} "
                f"{s.cooperation_potential:<8.1f} "
                f"{s.ml_signal:<12} "
                f"{s.recommendation:<8}"
            )

        print("-" * len(header))
        print(f"  总计: {len(scores)} 家银行 | 推荐合作: {sum(1 for s in scores if s.cooperation_potential >= 65)} 家")

        # 风险提示
        risky = [s for s in scores if s.risk_warning != "无显著风险"]
        if risky:
            print(f"\n  ⚠️ 风险提示: {len(risky)} 家银行存在风险因素")

    # ── 5. HTML 报告 ──
    if args.report:
        logger.info("=" * 50)
        logger.info("生成 HTML 报告...")
        chart_data = build_chart_data(scores)

        # 构建 AI 摘要
        summary_parts = []
        if pain_points:
            for pp in pain_points[:3]:
                summary_parts.append(f"【{pp.bank_name}】{pp.summary}")

        renderer = BankReportRenderer()
        output_dir = bank_cfg.get("report_output_dir", "./reports/bank")
        output_path = f"{output_dir}/bank_analysis_{date.today().isoformat()}.html"
        renderer.render_to_file(
            output_path=output_path,
            scores=[s.__dict__ for s in scores],
            pain_points=[pp.__dict__ for pp in pain_points],
            chart_data=chart_data,
            summary_text="\n\n".join(summary_parts) if summary_parts else "基于多维度量化评分和AI痛点分析的银行合作潜力评估报告。",
        )
        logger.info("✅ 报告已生成: %s", output_path)

    logger.info("=" * 50)
    logger.info("分析完成！")


if __name__ == "__main__":
    main()
