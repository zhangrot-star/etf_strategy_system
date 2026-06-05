"""RLPolicy — inference-only wrapper for production use.

Loads a trained PPO model + metadata and provides a clean predict() interface
that replaces the rule-based weight computation in CoreStrategy._compute_weights().

Usage:
    policy = RLPolicy("models/rl/ppo_portfolio")
    weights_array = policy.predict(observation)      # (8,) float32 array
    ticker_weights = policy.predict_as_dict(obs)      # {"SPY": 0.25, ...}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from rl.features import RLFeatureBuilder

logger = logging.getLogger(__name__)


class RLPolicy:
    """Production inference wrapper for trained PPO portfolio policy.

    Loads a saved RL agent (model + metadata) and exposes a simple
    predict() method that maps state → constrained portfolio weights.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
    ) -> None:
        zip_path = f"{model_path}.zip"
        meta_path = f"{model_path}_meta.json"

        if not Path(zip_path).exists():
            raise FileNotFoundError(f"Model file not found: {zip_path}")

        self._model = PPO.load(zip_path, device=device)
        self._model_path = model_path
        self._device = device

        # Load metadata
        self._meta: dict = {}
        if Path(meta_path).exists():
            with open(meta_path) as f:
                self._meta = json.load(f)

        self._ticker_order: list[str] = self._meta.get("ticker_order", [])
        self._max_positions: int = self._meta.get("max_positions", 8)
        self._single_position_cap: float = self._meta.get("single_position_cap", 0.30)
        self._obs_dim: int = self._meta.get("observation_dim", 175)
        self._feature_builder = RLFeatureBuilder(
            max_positions=self._max_positions,
            ticker_order=self._ticker_order,
        )

        logger.info(
            "RLPolicy loaded: %d tickers, max_pos=%d, cap=%.0f%%, obs_dim=%d",
            len(self._ticker_order),
            self._max_positions,
            self._single_position_cap * 100,
            self._obs_dim,
        )

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def ticker_order(self) -> list[str]:
        return self._ticker_order

    @property
    def max_positions(self) -> int:
        return self._max_positions

    @property
    def single_position_cap(self) -> float:
        return self._single_position_cap

    @property
    def observation_dim(self) -> int:
        return self._obs_dim

    # ── Prediction ──────────────────────────────────────────────────────

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Return raw action vector for a single observation.

        Args:
            observation: (obs_dim,) float32 state vector.
            deterministic: If True, use greedy policy.

        Returns:
            (max_positions,) float32 action vector in [0, 1].
        """
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        action, _ = self._model.predict(
            observation.astype(np.float32), deterministic=deterministic
        )
        return action.flatten()

    def predict_weights(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> dict[str, float]:
        """Return constrained portfolio weights as a ticker→weight dict.

        Applies the same constraint projection used during training.
        """
        raw_action = self.predict(observation, deterministic=deterministic)
        return self._project_to_constraints(raw_action)

    # ── Constraint projection ───────────────────────────────────────────

    def _project_to_constraints(self, raw_action: np.ndarray) -> dict[str, float]:
        """Project raw action to feasible weights matching training constraints.

        Steps:
        1. Map action to ticker weights
        2. Clip to [0, single_position_cap]
        3. Keep top max_positions, zero rest
        4. Cap total sum at 1.0 (remaining is cash — weights can sum < 1.0)
        """
        action = np.asarray(raw_action, dtype=np.float64).flatten()
        n_tickers = min(len(self._ticker_order), self._max_positions)

        weights: dict[str, float] = {}
        for i in range(n_tickers):
            w = float(action[i]) if i < len(action) else 0.0
            weights[self._ticker_order[i]] = max(0.0, min(w, self._single_position_cap))

        for i in range(n_tickers, len(self._ticker_order)):
            weights[self._ticker_order[i]] = 0.0

        # Top-N
        sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top_keys = {k for k, _ in sorted_items[: self._max_positions]}
        for k in list(weights.keys()):
            if k not in top_keys:
                weights[k] = 0.0

        # Cap total at 1.0
        total = sum(weights.values())
        if total > 1.0:
            for k in weights:
                weights[k] /= total

        # Re-clip
        for k in weights:
            weights[k] = max(0.0, min(weights[k], self._single_position_cap))

        # Re-check total
        total = sum(weights.values())
        if total > 1.0:
            for k in weights:
                weights[k] /= total

        return {k: v for k, v in weights.items() if v > 1e-6}

    # ── Convenience ─────────────────────────────────────────────────────

    def get_feature_builder(self) -> RLFeatureBuilder:
        """Return a RLFeatureBuilder configured with this policy's ticker order."""
        return RLFeatureBuilder(
            max_positions=self._max_positions,
            ticker_order=self._ticker_order,
        )

    def to_config_dict(self) -> dict:
        """Export policy settings for config propagation."""
        return {
            "ticker_order": self._ticker_order,
            "max_positions": self._max_positions,
            "single_position_cap": self._single_position_cap,
            "model_path": self._model_path,
        }
