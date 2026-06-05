#!/usr/bin/env python3
"""Evaluate a trained RL agent against the rule-based baseline.

Compares the RL policy and the existing rule-based weight computation
on out-of-sample historical data. Reports Sharpe ratio, max drawdown,
turnover, win rate, and other key metrics.

Usage:
    python scripts/evaluate_rl_agent.py --model models/rl/ppo_portfolio
    python scripts/evaluate_rl_agent.py --model models/rl/ppo_portfolio --tickers SPY,QQQ,IWM
    python scripts/evaluate_rl_agent.py --model models/rl/ppo_portfolio --bootstrap 5000
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.db_manager import DatabaseManager
from config.settings import Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("eval_rl")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate RL agent vs rule-based baseline")
    p.add_argument("--model", required=True,
                   help="Path to trained RL model (without extension)")
    p.add_argument("--tickers", default="SPY,QQQ,IWM,XLK,XLF,XLV,XLE,XLC",
                   help="Comma-separated ETF tickers")
    p.add_argument("--start", default="2024-01-01", help="Evaluation start date")
    p.add_argument("--end", default="2026-01-01", help="Evaluation end date")
    p.add_argument("--initial-capital", type=float, default=1_000_000.0)
    p.add_argument("--rebalance-freq", default="monthly",
                   choices=["daily", "weekly", "monthly"])
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="Bootstrap samples for confidence intervals")
    p.add_argument("--output", default=None,
                   help="Save results to CSV")
    return p.parse_args()


def load_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    db = DatabaseManager(Settings())
    return db.load_prices(tickers, pd.Timestamp(start), pd.Timestamp(end))


def compute_metrics(
    equity_curve: list[float],
    trades: list[dict] | None = None,
    rf_rate: float = 0.03,
) -> dict[str, float]:
    """Compute standard performance metrics from an equity curve."""
    if len(equity_curve) < 2:
        return {}

    eq = pd.Series(equity_curve)
    daily_rets = eq.pct_change().dropna()

    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1)
    n_days = len(eq)
    ann_return = float((1 + total_return) ** (252 / n_days) - 1) if n_days > 0 else 0.0
    ann_vol = float(daily_rets.std() * np.sqrt(252))
    sharpe = (ann_return - rf_rate) / ann_vol if ann_vol > 0 else 0.0

    # Max drawdown
    peak = eq.iloc[0]
    max_dd = 0.0
    for v in eq:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    # Calmar ratio
    calmar = ann_return / max_dd if max_dd > 0 else 0.0

    return {
        "total_return": round(total_return, 4),
        "annual_return": round(ann_return, 4),
        "annual_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "calmar_ratio": round(calmar, 4),
        "n_days": n_days,
    }


def run_simulation(
    prices: pd.DataFrame,
    tickers: list[str],
    config: dict,
    rl_model_path: str | None = None,
) -> dict[str, float]:
    """Run a single backtest simulation with RL or rule-based policy."""
    from core.strategy import CoreStrategy
    from core.feature_utils import build_features_from_prices

    features = build_features_from_prices(prices)
    if features.empty:
        return {}

    # Dates for rebalance
    all_dates = sorted(prices["trade_date"].unique())
    lookback = 63
    rebalance_dates = all_dates[lookback::21]  # ~monthly

    portfolio_value = config.get("backtest", {}).get("initial_capital", 1_000_000.0)
    initial_value = portfolio_value
    equity_curve = [portfolio_value]
    current_weights: dict[str, float] = {}
    turnover_total = 0.0

    strategy = CoreStrategy(config=config)

    # Train ensemble on pre-evaluation data
    train_prices = prices[prices["trade_date"] < all_dates[lookback]]
    if not train_prices.empty:
        try:
            strategy.train(train_prices.copy())
        except Exception:
            logger.warning("Could not train ensemble — using EQ weights")

    if rl_model_path:
        try:
            strategy.load_rl_policy(rl_model_path)
        except Exception as e:
            logger.warning("Could not load RL policy: %s — using rule-based", e)

    sentiment_empty = pd.DataFrame()

    for i, rebal_date in enumerate(rebalance_dates):
        date_mask = prices["trade_date"] <= rebal_date
        hist_prices = prices[date_mask]

        # Build features for this date
        try:
            feats = build_features_from_prices(hist_prices)
            if feats.empty:
                continue
            # Get latest features per ticker
            if hasattr(feats.index, "names") and feats.index.names == ["ticker", "trade_date"]:
                latest = feats.groupby("ticker").tail(1).reset_index(level="trade_date", drop=True)
            else:
                continue

            allocation = strategy.allocate(
                features=latest,
                sentiment=sentiment_empty,
                current_date=rebal_date,
            )
            new_weights = allocation.allocations

            # Compute turnover
            all_tickers = set(new_weights) | set(current_weights)
            turnover = sum(
                abs(new_weights.get(t, 0.0) - current_weights.get(t, 0.0))
                for t in all_tickers
            ) / 2.0
            turnover_total += turnover
            current_weights = new_weights

        except Exception:
            continue

        # Simulate forward performance until next rebalance
        next_idx = min(i + 1, len(rebalance_dates) - 1)
        next_date = rebalance_dates[next_idx]
        if next_date <= rebal_date:
            continue

        fwd_prices = prices[
            (prices["trade_date"] > rebal_date)
            & (prices["trade_date"] <= next_date)
        ]

        for _, day_data in fwd_prices.groupby("trade_date"):
            daily_ret = 0.0
            for ticker, w in current_weights.items():
                ticker_day = day_data[day_data["ticker"] == ticker]
                if not ticker_day.empty:
                    close = float(ticker_day.iloc[0]["close"])
                    # Approximate daily return from price changes
                    tp = prices[
                        (prices["ticker"] == ticker)
                        & (prices["trade_date"] <= day_data.iloc[0]["trade_date"])
                    ].sort_values("trade_date")
                    if len(tp) >= 2:
                        ret = float(tp.iloc[-1]["close"] / tp.iloc[-2]["close"] - 1)
                        daily_ret += w * ret
            portfolio_value *= (1.0 + daily_ret)
            equity_curve.append(portfolio_value)

    return {
        "equity_curve": equity_curve,
        "metrics": compute_metrics(equity_curve),
        "turnover_total": turnover_total,
    }


def bootstrap_sharpe_diff(
    rl_equity: list[float],
    baseline_equity: list[float],
    n_bootstrap: int = 1000,
) -> dict[str, float]:
    """Bootstrap 95% CI for the difference in Sharpe ratios."""
    rl_eq = pd.Series(rl_equity)
    bl_eq = pd.Series(baseline_equity)

    # Align lengths
    min_len = min(len(rl_eq), len(bl_eq))
    rl_eq = rl_eq.iloc[-min_len:].reset_index(drop=True)
    bl_eq = bl_eq.iloc[-min_len:].reset_index(drop=True)

    rl_rets = rl_eq.pct_change().dropna()
    bl_rets = bl_eq.pct_change().dropna()
    min_ret_len = min(len(rl_rets), len(bl_rets))
    rl_rets = rl_rets.iloc[-min_ret_len:]
    bl_rets = bl_rets.iloc[-min_ret_len:]

    diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(rl_rets), len(rl_rets), replace=True)
        rl_s = float(np.mean(rl_rets.iloc[idx]) / max(rl_rets.iloc[idx].std(), 1e-10) * np.sqrt(252))
        bl_s = float(np.mean(bl_rets.iloc[idx]) / max(bl_rets.iloc[idx].std(), 1e-10) * np.sqrt(252))
        diffs.append(rl_s - bl_s)

    diffs = np.array(diffs)
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_lower": float(np.percentile(diffs, 2.5)),
        "ci_upper": float(np.percentile(diffs, 97.5)),
        "p_value": float(np.mean(diffs <= 0)),
    }


def main() -> None:
    args = parse_args()
    tickers = [t.strip() for t in args.tickers.split(",")]

    import yaml
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # ── Load data ────────────────────────────────────────────
    logger.info("Loading prices for %d tickers ...", len(tickers))
    prices = load_data(tickers, args.start, args.end)
    if prices.empty:
        logger.error("No data found. Exiting.")
        sys.exit(1)
    logger.info("Loaded %d rows: %s → %s",
                len(prices), prices["trade_date"].min(), prices["trade_date"].max())

    # ── Run baseline ─────────────────────────────────────────
    logger.info("Running rule-based baseline ...")
    baseline_config = dict(config)
    baseline_config["rl"] = {"enabled": False}
    baseline_result = run_simulation(
        prices.copy(), tickers, baseline_config, rl_model_path=None
    )

    # ── Run RL agent ────────────────────────────────────────
    logger.info("Running RL agent ...")
    rl_result = run_simulation(
        prices.copy(), tickers, config, rl_model_path=args.model
    )

    # ── Print comparison ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RL Agent vs Rule-Based Baseline — Performance Comparison")
    print("=" * 70)
    print(f"  Period: {args.start} → {args.end}")
    print(f"  Tickers: {tickers}")
    print(f"{'─'*70}")
    print(f"  {'Metric':<25} {'RL Agent':>12} {'Baseline':>12} {'Δ':>12}")
    print(f"{'─'*70}")

    rl_metrics = rl_result.get("metrics", {})
    bl_metrics = baseline_result.get("metrics", {})

    metric_labels = [
        ("total_return", "Total Return", ".2%"),
        ("annual_return", "Annual Return", ".2%"),
        ("annual_volatility", "Ann. Volatility", ".2%"),
        ("sharpe_ratio", "Sharpe Ratio", ".3f"),
        ("max_drawdown", "Max Drawdown", ".2%"),
        ("calmar_ratio", "Calmar Ratio", ".3f"),
    ]

    for key, label, fmt in metric_labels:
        rl_v = rl_metrics.get(key, 0.0)
        bl_v = bl_metrics.get(key, 0.0)
        delta = rl_v - bl_v
        rl_str = f"{rl_v:{fmt}}" if isinstance(rl_v, (int, float)) else str(rl_v)
        bl_str = f"{bl_v:{fmt}}" if isinstance(bl_v, (int, float)) else str(bl_v)
        delta_str = f"{delta:+{fmt}}" if isinstance(delta, (int, float)) else str(delta)
        print(f"  {label:<25} {rl_str:>12} {bl_str:>12} {delta_str:>12}")

    print(f"{'─'*70}")

    # ── Bootstrap confidence intervals ───────────────────────
    if rl_result.get("equity_curve") and baseline_result.get("equity_curve"):
        bs = bootstrap_sharpe_diff(
            rl_result["equity_curve"],
            baseline_result["equity_curve"],
            n_bootstrap=args.bootstrap,
        )
        print(f"\n  Sharpe Difference Bootstrap (n={args.bootstrap}):")
        print(f"    Mean ΔSharpe:        {bs['mean_diff']:.4f}")
        print(f"    95% CI:              [{bs['ci_lower']:.4f}, {bs['ci_upper']:.4f}]")
        print(f"    P(RL ≤ Baseline):    {bs['p_value']:.4f}")
        print(f"    {'Significant ✓' if bs['p_value'] < 0.05 else 'Not significant'} "
              f"(α=0.05, one-sided)")

    print(f"\n{'='*70}")

    # ── Optional CSV output ──────────────────────────────────
    if args.output:
        rows = []
        for key, label, _ in metric_labels:
            rows.append({
                "metric": label,
                "rl_agent": rl_metrics.get(key),
                "baseline": bl_metrics.get(key),
                "delta": rl_metrics.get(key, 0.0) - bl_metrics.get(key, 0.0),
            })
        pd.DataFrame(rows).to_csv(args.output, index=False)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
