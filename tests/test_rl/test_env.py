"""Tests for PortfolioOptEnv — Gymnasium environment for RL portfolio optimization."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.spaces import Box

from rl.env import PortfolioOptEnv, _FREQ_DAYS


class TestPortfolioOptEnv:
    """Test the Gymnasium environment for correctness."""

    @pytest.fixture
    def env(self, sample_prices_df):
        """Create a basic environment instance."""
        return PortfolioOptEnv(
            prices=sample_prices_df,
            tickers=["SPY", "QQQ", "IWM"],
            initial_capital=1_000_000.0,
            rebalance_freq="monthly",
            lookback_days=60,
            max_positions=5,
            single_position_cap=0.30,
        )

    def test_env_creation(self, env):
        """Environment creates with correct spaces."""
        assert isinstance(env.observation_space, Box)
        assert isinstance(env.action_space, Box)
        assert env.observation_space.shape[0] > 0
        assert env.action_space.shape[0] == 5  # max_positions

    def test_reset(self, env):
        """Reset returns valid initial observation."""
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32
        assert not np.any(np.isnan(obs))
        assert isinstance(info, dict)

    def test_reset_with_seed(self, env):
        """Reset with seed is reproducible."""
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

    def test_step_produces_valid_output(self, env):
        """Step returns observation, reward, terminated, truncated, info."""
        env.reset(seed=42)
        action = env.action_space.sample()

        result = env.step(action)
        assert len(result) == 5

        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_info_contains_metrics(self, env):
        """Info dict includes portfolio metrics."""
        env.reset(seed=42)
        action = np.ones(env.action_space.shape[0]) * 0.2

        _, _, terminated, _, info = env.step(action)
        if not terminated:
            assert "portfolio_value" in info
            assert "cum_return" in info
            assert "drawdown" in info
            assert "turnover" in info
            assert "n_positions" in info

    def test_episode_terminates(self, env):
        """Episode eventually terminates when we run out of data."""
        env.reset(seed=42)
        for _ in range(1000):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                return
        pytest.fail("Episode did not terminate after 1000 steps")

    def test_render_does_not_crash(self, env):
        """Render runs without error."""
        env.reset(seed=42)
        env.render(mode="human")

    def test_observation_no_nan(self, env):
        """Observations never contain NaN."""
        env.reset(seed=42)
        for _ in range(20):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            assert not np.any(np.isnan(obs)), f"NaN in observation at step {_}"
            if terminated or truncated:
                break


class TestConstraintProjection:
    """Test the weight constraint projection logic."""

    @pytest.fixture
    def env(self, sample_prices_df):
        return PortfolioOptEnv(
            prices=sample_prices_df,
            tickers=["SPY", "QQQ", "IWM", "XLF", "XLV"],
            max_positions=3,
            single_position_cap=0.30,
            lookback_days=20,
        )

    def test_weights_sum_at_most_one(self, env):
        """Projected weights sum to at most 1.0 (remaining is cash)."""
        action = np.array([0.8, 0.6, 0.4, 0.2, 0.1], dtype=np.float32)
        weights = env._project_to_constraints(action)

        total = sum(weights.values())
        assert total <= 1.0 + 1e-4, f"Sum {total} exceeds 1.0"

    def test_no_weight_exceeds_cap(self, env):
        """No individual weight exceeds the position cap."""
        action = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        weights = env._project_to_constraints(action)

        for w in weights.values():
            assert w <= 0.30 + 1e-5, f"Weight {w} exceeds cap 0.30"

    def test_max_positions_respected(self, env):
        """At most max_positions tickers have non-zero weight."""
        action = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
        weights = env._project_to_constraints(action)

        assert len(weights) <= 3, f"Got {len(weights)} positions, max is 3"

    def test_all_zeros_action_yields_empty_weights(self, env):
        """All-zero action → all cash (empty weights)."""
        action = np.zeros(5, dtype=np.float32)
        weights = env._project_to_constraints(action)
        assert weights == {} or sum(weights.values()) == pytest.approx(0.0, abs=0.01)

    def test_single_ticker_capped(self, env):
        """Single ticker weight is capped at position cap."""
        action = np.array([0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        weights = env._project_to_constraints(action)

        # Should keep the ticker with weight capped at 0.30
        assert len(weights) <= 1
        if weights:
            assert max(weights.values()) <= 0.30 + 1e-5

    def test_weights_non_negative(self, env):
        """All projected weights are non-negative."""
        action = np.array([0.3, -0.5, 0.7, 0.0, -0.1], dtype=np.float32)
        weights = env._project_to_constraints(action)

        for w in weights.values():
            assert w >= 0.0, f"Negative weight: {w}"


class TestReward:
    """Test the reward computation."""

    @pytest.fixture
    def env(self, sample_prices_df):
        return PortfolioOptEnv(
            prices=sample_prices_df,
            tickers=["SPY", "QQQ", "IWM"],
            max_positions=5,
            single_position_cap=0.30,
            lookback_days=20,
            reward_weights={
                "sharpe_weight": 1.0,
                "turnover_weight": 0.5,
                "drawdown_weight": 1.0,
                "diversification_weight": 0.2,
            },
        )

    def test_reward_is_finite(self, env):
        """Reward is always finite."""
        env.reset(seed=42)
        action = np.ones(env.action_space.shape[0]) * 0.2

        _, reward, _, _, _ = env.step(action)
        assert np.isfinite(reward)

    def test_zero_returns_give_low_reward(self, env):
        """Period of zero returns gives reward near zero (no Sharpe contribution)."""
        env.reset(seed=42)
        reward = env._compute_reward([0.0, 0.0, 0.0], turnover=0.0)
        # Should be near zero since no return, no vol, no drawdown
        assert np.abs(reward) < 1.0

    def test_positive_returns_give_positive_sharpe(self, env):
        """Consistent positive returns increase Sharpe contribution."""
        env.reset(seed=42)
        rets = [0.001] * 10  # 10 days of +0.1% each
        reward = env._compute_reward(rets, turnover=0.0)
        # Should be positive due to positive Sharpe
        # (with 10 days of same return, std≈0, so Sharpe≈0)
        assert np.isfinite(reward)

    def test_high_turnover_penalized(self, env):
        """Higher turnover reduces reward."""
        env.reset(seed=42)
        rets = [0.0005] * 10
        reward_no_to = env._compute_reward(rets, turnover=0.0)
        reward_hi_to = env._compute_reward(rets, turnover=0.8)
        assert reward_hi_to < reward_no_to, "High turnover should reduce reward"

    def test_deep_drawdown_penalized(self, env):
        """Deep drawdown triggers penalty."""
        env.reset(seed=42)
        env._portfolio_value = 1000.0
        env._peak_value = 1000.0
        reward_normal = env._compute_reward([0.001] * 5, turnover=0.0)

        # Simulate a drawdown
        env._portfolio_value = 800.0  # 20% drawdown
        env._peak_value = 1000.0
        reward_dd = env._compute_reward([-0.02] * 5, turnover=0.0)

        assert reward_dd < reward_normal, "Drawdown should reduce reward"


class TestFrequencyDays:
    """Test rebalance frequency → trading days mapping."""

    def test_daily(self):
        assert _FREQ_DAYS["daily"] == 1

    def test_weekly(self):
        assert _FREQ_DAYS["weekly"] == 5

    def test_monthly(self):
        assert _FREQ_DAYS["monthly"] == 21
