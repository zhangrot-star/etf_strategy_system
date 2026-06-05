#!/usr/bin/env python3
"""Train an RL portfolio optimization agent using walk-forward validation.

Loads historical price data, builds features, trains a PPO agent through
expanding walk-forward folds, and saves the best model for production use.

Usage:
    python scripts/train_rl_agent.py                                    # default: US tickers
    python scripts/train_rl_agent.py --tickers SPY,QQQ,IWM,XLK,XLF      # custom tickers
    python scripts/train_rl_agent.py --start 2022-01-01 --end 2026-01-01
    python scripts/train_rl_agent.py --total-timesteps 500000
    python scripts/train_rl_agent.py --skip-walkforward                  # single-pass training
"""

from __future__ import annotations

import argparse
import logging
import sys

# Avoid segfault on macOS ARM64 / Python 3.14+ with torch multi-threading
import torch

try:
    torch.set_num_threads(1)
except RuntimeError:
    pass
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.db_manager import DatabaseManager
from config.settings import Settings
from core.feature_utils import build_features_from_prices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("train_rl")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RL portfolio optimization agent")
    p.add_argument("--tickers", default="SPY,QQQ,IWM,XLK,XLF,XLV,XLE,XLC",
                   help="Comma-separated ETF tickers")
    p.add_argument("--start", default="2020-01-01", help="Training start date")
    p.add_argument("--end", default="2026-01-01", help="Training end date")
    p.add_argument("--total-timesteps", type=int, default=200_000,
                   help="Total PPO training steps")
    p.add_argument("--walk-forward-folds", type=int, default=5,
                   help="Number of walk-forward folds")
    p.add_argument("--val-months", type=int, default=12,
                   help="Validation months per fold")
    p.add_argument("--output", default="models/rl/ppo_portfolio",
                   help="Output model path (without extension)")
    p.add_argument("--rebalance-freq", default="monthly",
                   choices=["daily", "weekly", "monthly"])
    p.add_argument("--initial-capital", type=float, default=1_000_000.0)
    p.add_argument("--device", default="auto",
                   help="Torch device (cpu, cuda, auto)")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--skip-walkforward", action="store_true",
                   help="Single-pass training (no walk-forward)")
    return p.parse_args()


def load_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Load price data from MySQL."""
    db = DatabaseManager(Settings())
    return db.load_prices(
        tickers, pd.Timestamp(start), pd.Timestamp(end)
    )


def make_folds(
    start: str,
    end: str,
    n_folds: int,
    val_months: int,
) -> list[tuple[str, str, str, str]]:
    """Generate walk-forward fold date ranges.

    Returns list of (train_start, train_end, val_start, val_end).
    """
    train_start = pd.Timestamp(start)
    total_end = pd.Timestamp(end)

    folds = []
    for i in range(n_folds):
        val_start = train_start + pd.DateOffset(months=(i + 1) * (
            ((total_end.year - train_start.year) * 12 + total_end.month - train_start.month)
            // (n_folds + 1)
        ))
        val_end = val_start + pd.DateOffset(months=val_months)
        if val_end > total_end:
            val_end = total_end

        folds.append((
            str(train_start.date()),
            str(val_start.date()),
            str(val_start.date()),
            str(val_end.date()),
        ))

        # Expanding window: train_end moves forward
        train_start = train_start  # keeps expanding

    # Ensure minimum data for training
    folds = [f for f in folds if f[1] > f[0] and f[3] > f[2]]
    return folds


def train_single_fold(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    tickers: list[str],
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    config: dict,
    fold_idx: int,
) -> dict:
    """Train and evaluate a single walk-forward fold."""
    from rl.env import PortfolioOptEnv
    from rl.agent import RLAgent
    from rl.callbacks import PortfolioEvalCallback

    rl_cfg = config.get("rl", {})

    # ── Split data ──────────────────────────────────────────
    train_prices = prices[
        (prices["trade_date"] >= pd.Timestamp(train_start))
        & (prices["trade_date"] < pd.Timestamp(train_end))
    ].copy()
    val_prices = prices[
        (prices["trade_date"] >= pd.Timestamp(val_start))
        & (prices["trade_date"] <= pd.Timestamp(val_end))
    ].copy()

    if train_prices.empty or val_prices.empty:
        logger.warning("Fold %d: empty train/val split — skipping.", fold_idx)
        return {"fold": fold_idx, "skipped": True}

    # Ensure trade_date is Timestamp
    train_prices["trade_date"] = pd.to_datetime(train_prices["trade_date"])
    val_prices["trade_date"] = pd.to_datetime(val_prices["trade_date"])

    logger.info(
        "Fold %d: train=%s→%s (%d bars)  val=%s→%s (%d bars)",
        fold_idx, train_start, train_end, len(train_prices),
        val_start, val_end, len(val_prices),
    )

    # ── Build features ───────────────────────────────────────
    train_features = build_features_from_prices(train_prices)
    val_features = build_features_from_prices(val_prices)

    # ── Create environments ─────────────────────────────────
    train_dates = train_prices["trade_date"].nunique()
    train_lookback = max(5, min(train_dates // 3, 252))
    val_dates = val_prices["trade_date"].nunique()
    val_lookback = max(5, min(val_dates // 3, 60))

    train_env = PortfolioOptEnv(
        prices=train_prices,
        features=train_features,
        tickers=tickers,
        initial_capital=config.get("backtest", {}).get("initial_capital", 1_000_000),
        rebalance_freq=rl_cfg.get("rebalance_freq", "monthly"),
        lookback_days=train_lookback,
        max_positions=config.get("risk", {}).get("max_positions", 8),
        single_position_cap=config.get("risk", {}).get("single_position_cap", 0.30),
        reward_weights=rl_cfg.get("reward", {}),
    )

    val_env = PortfolioOptEnv(
        prices=val_prices,
        features=val_features,
        tickers=tickers,
        initial_capital=config.get("backtest", {}).get("initial_capital", 1_000_000),
        rebalance_freq=rl_cfg.get("rebalance_freq", "monthly"),
        lookback_days=val_lookback,
        max_positions=config.get("risk", {}).get("max_positions", 8),
        single_position_cap=config.get("risk", {}).get("single_position_cap", 0.30),
        reward_weights=rl_cfg.get("reward", {}),
    )

    # ── Train agent ──────────────────────────────────────────
    ppo_cfg = rl_cfg.get("ppo", {})
    total_ts = rl_cfg.get("total_timesteps", 200_000)

    agent = RLAgent(
        env=train_env,
        ppo_kwargs=ppo_cfg,
        device=config.get("rl_device", "auto"),
    )

    eval_callback = PortfolioEvalCallback(
        val_env,
        eval_freq=max(1000, total_ts // 20),
        best_model_save_path=f"{config.get('rl', {}).get('model_path', 'models/rl/ppo_portfolio')}_fold{fold_idx}_best",
    )

    train_metrics = agent.train(
        total_timesteps=total_ts,
        callback=eval_callback,
        progress_bar=True,
    )

    # ── Final evaluation on val ────────────────────────────
    obs, _ = val_env.reset()
    done = False
    val_rewards = []
    val_info = {}
    while not done:
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = val_env.step(action)
        done = terminated or truncated
        val_rewards.append(reward)
        if done:
            val_info = info

    result = {
        "fold": fold_idx,
        "train_start": train_start,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "total_reward": float(np.sum(val_rewards)),
        "cum_return": val_info.get("cum_return", 0.0),
        "max_drawdown": val_info.get("drawdown", 0.0),
        "eval_history": eval_callback.eval_history,
        "best_sharpe": float(eval_callback.best_sharpe),
    }

    logger.info(
        "Fold %d result: cum_return=%.4f  dd=%.4f  total_reward=%.2f  best_sharpe=%.4f",
        fold_idx, result["cum_return"], result["max_drawdown"],
        result["total_reward"], result["best_sharpe"],
    )

    return result


def main() -> None:
    args = parse_args()
    tickers = [t.strip() for t in args.tickers.split(",")]

    # Load config
    import yaml
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Propagate CLI overrides
    config.setdefault("rl", {})
    config["rl"]["total_timesteps"] = args.total_timesteps
    config["rl"]["rebalance_freq"] = args.rebalance_freq
    config["rl"]["model_path"] = args.output
    if "ppo" not in config["rl"]:
        config["rl"]["ppo"] = {}
    config["rl"]["ppo"]["learning_rate"] = args.learning_rate
    if "reward" not in config["rl"]:
        config["rl"]["reward"] = {
            "sharpe_weight": 1.0,
            "turnover_weight": 0.5,
            "drawdown_weight": 1.0,
            "diversification_weight": 0.2,
        }

    # ── Load data ────────────────────────────────────────────
    logger.info("Loading prices for %d tickers: %s", len(tickers), tickers)
    prices = load_data(tickers, args.start, args.end)

    if prices.empty:
        logger.error("No price data found for tickers=%s", tickers)
        sys.exit(1)

    logger.info("Loaded %d rows from %s to %s",
                len(prices),
                prices["trade_date"].min(),
                prices["trade_date"].max())

    # Ensure trade_date is Timestamp for consistent comparison
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])

    # ── Build features ───────────────────────────────────────
    logger.info("Building features ...")
    features = build_features_from_prices(prices)
    logger.info("Features: %d rows, %d columns", len(features), features.shape[1])

    # ── Train ────────────────────────────────────────────────
    if args.skip_walkforward:
        logger.info("Single-pass training (no walk-forward) ...")
        from rl.env import PortfolioOptEnv
        from rl.agent import RLAgent

        env = PortfolioOptEnv(
            prices=prices,
            features=features,
            tickers=tickers,
            initial_capital=args.initial_capital,
            rebalance_freq=args.rebalance_freq,
            max_positions=config.get("risk", {}).get("max_positions", 8),
            single_position_cap=config.get("risk", {}).get("single_position_cap", 0.30),
            reward_weights=config["rl"].get("reward", {}),
        )

        agent = RLAgent(
            env=env,
            ppo_kwargs=config["rl"].get("ppo", {}),
            device=args.device,
        )
        agent.train(total_timesteps=args.total_timesteps)
        agent.save(args.output)
        logger.info("Model saved to %s", args.output)

    else:
        logger.info("Walk-forward training: %d folds ...", args.walk_forward_folds)
        folds = make_folds(args.start, args.end, args.walk_forward_folds, args.val_months)

        if not folds:
            logger.error("Could not generate walk-forward folds — date range too short?")
            sys.exit(1)

        for i, (tr_s, tr_e, v_s, v_e) in enumerate(folds):
            logger.info("=" * 50)
            logger.info("Fold %d/%d", i + 1, len(folds))

            result = train_single_fold(
                prices=prices,
                features=features,
                tickers=tickers,
                train_start=tr_s,
                train_end=tr_e,
                val_start=v_s,
                val_end=v_e,
                config=config,
                fold_idx=i,
            )

            if result.get("skipped"):
                continue

            print(f"\n{'─'*60}")
            print(f"Fold {i}: cum_return={result['cum_return']:.4f}  "
                  f"md={result['max_drawdown']:.4f}  "
                  f"best_sharpe={result['best_sharpe']:.4f}")
            print(f"{'─'*60}\n")

        # ── Final model: train on full dataset ──────────────
        logger.info("Training final model on full dataset ...")
        from rl.env import PortfolioOptEnv
        from rl.agent import RLAgent

        final_env = PortfolioOptEnv(
            prices=prices,
            features=features,
            tickers=tickers,
            initial_capital=args.initial_capital,
            rebalance_freq=args.rebalance_freq,
            max_positions=config.get("risk", {}).get("max_positions", 8),
            single_position_cap=config.get("risk", {}).get("single_position_cap", 0.30),
            reward_weights=config["rl"].get("reward", {}),
        )

        final_agent = RLAgent(
            env=final_env,
            ppo_kwargs=config["rl"].get("ppo", {}),
            device=args.device,
        )
        final_agent.train(total_timesteps=args.total_timesteps)
        final_agent.save(args.output)

        logger.info("Final model saved to %s", args.output)

    print(f"\n=== Training complete ===")
    print(f"Model: {args.output}.zip + {args.output}_meta.json")
    print(f"To evaluate: python scripts/evaluate_rl_agent.py --model {args.output}")


if __name__ == "__main__":
    main()
