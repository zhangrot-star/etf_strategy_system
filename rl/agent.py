"""RL Agent — stable-baselines3 PPO wrapper for portfolio optimization.

Handles environment creation, training, model persistence, and inference.
Designed as a drop-in component that can be used from training scripts or
loaded in production via RLPolicy.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

# Avoid segfault on macOS ARM64 / Python 3.14+ with torch threading
import torch

try:
    torch.set_num_threads(1)
except RuntimeError:
    pass
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from rl.env import PortfolioOptEnv

logger = logging.getLogger(__name__)

# Default PPO hyperparameters tuned for financial portfolio optimization
DEFAULT_PPO_KWARGS: dict[str, Any] = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 0,
}

DEFAULT_POLICY_KWARGS: dict[str, Any] = {
    "net_arch": {"pi": [256, 256], "vf": [256, 256]},
    "ortho_init": False,  # Avoids segfault on macOS ARM64 / Py3.14
    "activation_fn": torch.nn.ReLU,
}


class RLAgent:
    """PPO agent that learns portfolio allocation from historical price data.

    Usage:
        # Training
        env = PortfolioOptEnv(prices, tickers=["SPY","QQQ","IWM"])
        agent = RLAgent(env=env, ppo_kwargs={"learning_rate": 3e-4})
        agent.train(total_timesteps=200_000)

        # Save
        agent.save("models/rl/ppo_portfolio")

        # Inference
        agent = RLAgent(model_path="models/rl/ppo_portfolio")
        weights = agent.predict(observation)
    """

    def __init__(
        self,
        env: PortfolioOptEnv | None = None,
        model_path: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        ppo_kwargs: dict[str, Any] | None = None,
        device: str = "auto",
        seed: int | None = 42,
    ) -> None:
        self._device = device
        self._model_path = model_path
        self._seed = seed
        self._ticker_order: list[str] = []
        self._max_positions: int = 8
        self._single_position_cap: float = 0.30

        effective_ppo = {**DEFAULT_PPO_KWARGS, **(ppo_kwargs or {})}
        effective_policy = {**DEFAULT_POLICY_KWARGS, **(policy_kwargs or {})}

        if model_path and Path(f"{model_path}.zip").exists():
            self._model = PPO.load(f"{model_path}.zip", device=device)
            self._load_meta(model_path)
            logger.info("Loaded RL agent from %s", model_path)
        elif env is not None:
            import torch

            self._model = PPO(
                policy="MlpPolicy",
                env=env,
                policy_kwargs=effective_policy,
                learning_rate=effective_ppo["learning_rate"],
                n_steps=effective_ppo["n_steps"],
                batch_size=effective_ppo["batch_size"],
                n_epochs=effective_ppo["n_epochs"],
                gamma=effective_ppo["gamma"],
                gae_lambda=effective_ppo["gae_lambda"],
                clip_range=effective_ppo["clip_range"],
                ent_coef=effective_ppo["ent_coef"],
                vf_coef=effective_ppo["vf_coef"],
                max_grad_norm=effective_ppo["max_grad_norm"],
                verbose=effective_ppo.get("verbose", 1),
                device=device,
                seed=seed,
            )
            self._ticker_order = list(env._tickers) if hasattr(env, "_tickers") else []
            self._max_positions = env._max_positions if hasattr(env, "_max_positions") else 8
        else:
            raise ValueError("Provide either env (for training) or model_path (for inference).")

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def ticker_order(self) -> list[str]:
        return self._ticker_order

    @property
    def max_positions(self) -> int:
        return self._max_positions

    # ── Training ────────────────────────────────────────────────────────

    def train(
        self,
        total_timesteps: int = 200_000,
        callback: BaseCallback | None = None,
        progress_bar: bool = True,
    ) -> dict[str, Any]:
        """Train the PPO agent.

        Args:
            total_timesteps: Total environment steps to train.
            callback: Optional SB3 callback (e.g., EvalCallback).
            progress_bar: Show tqdm progress bar.

        Returns:
            Dict with training metrics.
        """
        logger.info("Training PPO agent for %d timesteps ...", total_timesteps)
        self._model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=progress_bar,
        )
        logger.info("Training complete.")
        return {
            "total_timesteps": total_timesteps,
            "model_class": "PPO",
        }

    def train_with_walk_forward(
        self,
        train_env: PortfolioOptEnv,
        val_env: PortfolioOptEnv,
        total_timesteps: int = 200_000,
        eval_freq: int = 10000,
    ) -> dict[str, Any]:
        """Train with periodic evaluation on validation environment.

        This is the recommended training approach — saves the best model
        (highest validation Sharpe) during training.

        Args:
            train_env: Training environment.
            val_env: Validation environment.
            total_timesteps: Total training steps.
            eval_freq: Evaluate every N steps.

        Returns:
            Dict with training and evaluation metrics.
        """
        from stable_baselines3.common.callbacks import EvalCallback

        eval_callback = EvalCallback(
            val_env,
            best_model_save_path=str(Path(self._model_path or "models/rl").parent / "tmp_best"),
            log_path=str(Path(self._model_path or "models/rl").parent / "tmp_logs"),
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
        )

        self._model.set_env(train_env)
        self.train(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)

        return {
            "total_timesteps": total_timesteps,
            "best_mean_reward": float(eval_callback.best_mean_reward),
        }

    # ── Prediction ──────────────────────────────────────────────────────

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Return action (raw weight vector) for a single observation.

        Args:
            observation: (175,) float32 state vector.
            deterministic: If True, use greedy policy (no exploration noise).

        Returns:
            (max_positions,) float32 weight vector.
        """
        action, _states = self._model.predict(observation, deterministic=deterministic)
        return action

    # ── Persistence ─────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save model and metadata.

        Produces:
            {path}.zip  — SB3 PPO model
            {path}_meta.json — ticker order, constraints, training info
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._model.save(f"{path}.zip")

        meta = {
            "ticker_order": self._ticker_order,
            "max_positions": self._max_positions,
            "single_position_cap": self._single_position_cap,
            "observation_dim": self._model.observation_space.shape[0]
            if hasattr(self._model.observation_space, "shape")
            else 0,
            "action_dim": self._model.action_space.shape[0]
            if hasattr(self._model.action_space, "shape")
            else 0,
            "model_class": "PPO",
            "seed": self._seed,
        }
        with open(f"{path}_meta.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info("Saved model to %s.zip + _meta.json", path)
        self._model_path = path

    def load(self, path: str) -> None:
        """Load model and metadata."""
        self._model = PPO.load(f"{path}.zip", device=self._device)
        self._load_meta(path)
        self._model_path = path
        logger.info("Loaded model from %s", path)

    def _load_meta(self, path: str) -> None:
        """Load metadata JSON."""
        meta_path = f"{path}_meta.json"
        if Path(meta_path).exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self._ticker_order = meta.get("ticker_order", [])
            self._max_positions = meta.get("max_positions", 8)
            self._single_position_cap = meta.get("single_position_cap", 0.30)
