#!/usr/bin/env python3
"""Prediction accuracy evaluation — walk-forward backtesting and live prediction tracking.

Two modes:
  eval  — Walk-forward historical evaluation (how well would the model have predicted?)
  track — Check realized returns for expired live predictions, update metrics

Usage:
  python scripts/evaluate_predictions.py eval
  python scripts/evaluate_predictions.py track
  python scripts/evaluate_predictions.py full     # eval + track
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.db_manager import DatabaseManager
from data_pipeline.models import ETFPrice
from config.settings import Settings
from prediction.evaluator import PredictionEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_pred")

db = DatabaseManager(Settings())


def load_prices() -> pd.DataFrame:
    with db._session_factory() as sess:
        tickers = sorted([r[0] for r in sess.query(ETFPrice.ticker).distinct().all()])
    return db.load_prices(tickers, pd.Timestamp("2024-01-01"), pd.Timestamp.today())


def run_eval() -> None:
    """Walk-forward historical evaluation."""
    prices = load_prices()
    evaluator = PredictionEvaluator(model_path="models/xgboost_reg")

    if not evaluator.is_ready:
        logger.error("No fitted models found. Run retrain_model.py first.")
        return

    report = evaluator.evaluate(prices, eval_start="2025-06-01", step_days=21)

    print()
    print(report.summary())
    print()

    # Per-horizon detail
    for h in sorted(report.per_horizon.keys()):
        m = report.per_horizon[h]
        print(f"━━━ {h}天预测详细指标 ━━━")
        print(f"  预测数: {m.n_predictions}")
        print(f"  RMSE:   {m.rmse:.4f}")
        print(f"  MAE:    {m.mae:.4f}")
        print(f"  R²:     {m.r2:.3f}")
        print(f"  方向准确率: {m.direction_accuracy:.1%}")
        print(f"  预测均值:   {m.mean_prediction:+.4f}  (实际均值: {m.mean_actual:+.4f})")
        print(f"  预测波动:   {m.pred_std:.4f}  (实际波动: {m.actual_std:.4f})")
        print(f"  校准误差(ECE): {m.calibration_error:.2%}")
        print(f"  概率校准分布:")
        print(f"    {'区间':<14} {'样本数':>6} {'预测上涨概率':>12} {'实际上涨比例':>12}")
        for b in m.prob_bins:
            print(f"    [{b['bin_low']:.2f}-{b['bin_high']:.2f})  "
                  f"{b['n']:>6d}  {b['avg_prob']:>12.3f}  {b['actual_up_rate']:>12.3f}")
        print()

    # Recommendations
    print("━━━ 改进建议 ━━━")
    worst_horizon = max(report.per_horizon.items(), key=lambda x: x[1].calibration_error)
    print(f"  校准误差最大的周期: {worst_horizon[0]}d (ECE={worst_horizon[1].calibration_error:.2%})")
    print(f"  建议: {'重新训练该周期的模型' if worst_horizon[1].calibration_error > 0.15 else '当前校准可接受'}")
    print(f"  总体评级: {report.overall_rating}")


def run_track() -> None:
    """Track realized returns for expired predictions."""
    n_updated = db.update_realized_returns()
    logger.info("Updated %d predictions with realized returns.", n_updated)

    if n_updated == 0:
        logger.info("No predictions have expired yet. "
                     "The earliest predictions will mature after 5 trading days.")
        return

    # Show updated predictions summary
    with db._session_factory() as sess:
        from data_pipeline.models import ETFPrediction
        realized = sess.query(ETFPrediction).filter(
            ETFPrediction.realized == True
        ).all()

        if not realized:
            return

        df = pd.DataFrame([{
            "ticker": r.ticker,
            "horizon_days": r.horizon_days,
            "predicted": r.predicted_return,
            "actual": r.target_return,
            "error": abs(r.predicted_return - r.target_return) if r.target_return else None,
            "correct_dir": (r.predicted_return > 0) == (r.target_return > 0) if r.target_return else None,
        } for r in realized])

        print("\n━━━ 实盘预测追踪 ━━━")
        print(f"已到期预测数: {len(df)}")
        print(f"\n按周期汇总:")
        for h in sorted(df["horizon_days"].unique()):
            sub = df[df["horizon_days"] == h]
            dir_acc = sub["correct_dir"].mean() if not sub.empty else 0
            mae = sub["error"].mean() if not sub.empty else 0
            print(f"  {h}d: 样本数={len(sub)}  方向准确率={dir_acc:.1%}  MAE={mae:.4f}")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "eval"

    if mode == "eval":
        run_eval()
    elif mode == "track":
        run_track()
    elif mode == "full":
        run_eval()
        print("\n" + "=" * 60 + "\n")
        run_track()
    else:
        print(f"Unknown mode: {mode}. Use: eval | track | full")
        sys.exit(1)


if __name__ == "__main__":
    main()
