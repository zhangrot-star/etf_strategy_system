"""Custom SB3 callbacks for RL portfolio optimization.

- PortfolioEvalCallback: evaluate on validation env, track best Sharpe
- RewardLogCallback: log per-episode reward stats
- TensorboardCallback: log custom metrics (drawdown, turnover, n_positions)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class PortfolioEvalCallback(BaseCallback):
    """Evaluate the RL policy on a held-out validation environment periodically.

    Tracks the best validation Sharpe ratio and saves the best model.
    """

    def __init__(
        self,
        eval_env: Any,
        eval_freq: int = 10000,
        n_eval_episodes: int = 1,
        log_path: str | None = None,
        best_model_save_path: str | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._eval_env = eval_env
        self._eval_freq = eval_freq
        self._n_eval_episodes = n_eval_episodes
        self._best_sharpe = -np.inf
        self._best_model_save_path = best_model_save_path
        self._eval_history: list[dict[str, float]] = []

    def _on_step(self) -> bool:
        if self.n_calls % self._eval_freq != 0:
            return True

        all_rewards = []
        all_returns = []
        all_drawdowns = []
        all_turnovers = []

        for _ in range(self._n_eval_episodes):
            obs, _ = self._eval_env.reset()
            done = False
            episode_rewards = []
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self._eval_env.step(action)
                done = terminated or truncated
                episode_rewards.append(reward)

                if done:
                    all_returns.append(info.get("cum_return", 0.0))
                    all_drawdowns.append(info.get("drawdown", 0.0))
                    all_turnovers.append(info.get("turnover", 0.0))

            all_rewards.append(np.sum(episode_rewards))

        mean_reward = float(np.mean(all_rewards))
        mean_return = float(np.mean(all_returns)) if all_returns else 0.0
        mean_dd = float(np.mean(all_drawdowns)) if all_drawdowns else 0.0
        mean_turnover = float(np.mean(all_turnovers)) if all_turnovers else 0.0

        # Estimate Sharpe from cumulative return and drawdown
        # (exact Sharpe would need full equity curve; use approximation)
        approx_sharpe = mean_return / max(mean_dd + 0.01, 0.01)

        record = {
            "step": self.n_calls,
            "mean_reward": mean_reward,
            "mean_return": mean_return,
            "approx_sharpe": approx_sharpe,
            "mean_drawdown": mean_dd,
            "mean_turnover": mean_turnover,
        }
        self._eval_history.append(record)

        # Save best model
        if approx_sharpe > self._best_sharpe and self._best_model_save_path:
            self._best_sharpe = approx_sharpe
            self.model.save(self._best_model_save_path)
            logger.info(
                "New best model: approx_sharpe=%.4f (step %d)", approx_sharpe, self.n_calls
            )

        if self.verbose > 0:
            logger.info(
                "Eval @ step %d: reward=%.4f  ret=%.4f  sharpe≈%.4f  dd=%.4f",
                self.n_calls, mean_reward, mean_return, approx_sharpe, mean_dd,
            )

        # Log to TensorBoard if available
        if self.logger is not None:
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_return", mean_return)
            self.logger.record("eval/approx_sharpe", approx_sharpe)
            self.logger.record("eval/mean_drawdown", mean_dd)
            self.logger.record("eval/mean_turnover", mean_turnover)

        return True

    @property
    def best_sharpe(self) -> float:
        return self._best_sharpe

    @property
    def eval_history(self) -> list[dict[str, float]]:
        return self._eval_history


class RewardLogCallback(BaseCallback):
    """Log per-episode reward statistics."""

    def __init__(
        self, log_interval: int = 10, verbose: int = 0
    ) -> None:
        super().__init__(verbose)
        self._log_interval = log_interval
        self._episode_rewards: list[float] = []
        self._episode_count: int = 0

    def _on_step(self) -> bool:
        # SB3 environments wrapped in Monitor provide "episode" info
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict) and "episode" in info:
                ep_info = info["episode"]
                self._episode_rewards.append(float(ep_info["r"]))
                self._episode_count += 1

                if self._episode_count % self._log_interval == 0:
                    recent = self._episode_rewards[-self._log_interval :]
                    mean_r = np.mean(recent)
                    logger.info(
                        "Episode %d: mean reward (last %d) = %.4f",
                        self._episode_count, self._log_interval, mean_r,
                    )

                if self.logger is not None:
                    self.logger.record("rollout/ep_rew_mean", float(ep_info["r"]))

        return True

    @property
    def episode_rewards(self) -> list[float]:
        return self._episode_rewards


class MetricsCallback(BaseCallback):
    """Log custom portfolio metrics to TensorBoard during training."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.metrics_history: list[dict[str, float]] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict):
                metrics = {
                    "metrics/drawdown": info.get("drawdown", 0.0),
                    "metrics/turnover": info.get("turnover", 0.0),
                    "metrics/n_positions": info.get("n_positions", 0),
                    "metrics/cum_return": info.get("cum_return", 0.0),
                }
                self.metrics_history.append(metrics)

                if self.logger is not None:
                    for k, v in metrics.items():
                        self.logger.record(k, v)

        return True
