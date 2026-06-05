"""Tests for RLPolicy — inference-only wrapper and constraint projection."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from rl.features import RLFeatureBuilder


class TestRLPolicyImport:
    """Test that RLPolicy can be imported and its dependencies are available."""

    def test_import_rl_policy(self):
        """RLPolicy imports successfully."""
        from rl.policy import RLPolicy
        assert RLPolicy is not None

    def test_import_rl_agent(self):
        """RLAgent imports successfully."""
        from rl.agent import RLAgent
        assert RLAgent is not None

    def test_policy_file_not_found_raises(self):
        """Policy raises FileNotFoundError for non-existent model."""
        from rl.policy import RLPolicy
        with pytest.raises(FileNotFoundError):
            RLPolicy("nonexistent/model/path")


class TestRLPolicyConstraintProjection:
    """Test constraint projection logic (can test without trained model)."""

    @pytest.fixture
    def policy_class(self):
        from rl.policy import RLPolicy
        return RLPolicy

    def test_projection_class_accessible(self, policy_class):
        """Constraint projection method exists."""
        assert hasattr(policy_class, "_project_to_constraints")

    def test_projection_is_instance_method(self, policy_class):
        """_project_to_constraints is an instance method."""
        import inspect
        assert inspect.isfunction(policy_class._project_to_constraints)


class TestRLFeatureBuilderInPolicy:
    """Test that RLFeatureBuilder is compatible with policy integration."""

    def test_get_feature_builder(self):
        """RLFeatureBuilder.get_feature_builder() works for policy config."""
        from rl.features import RLFeatureBuilder
        builder = RLFeatureBuilder(
            max_positions=8,
            ticker_order=["SPY", "QQQ", "IWM", "XLK", "XLF", "XLV", "XLE", "XLC"],
        )
        assert builder.observation_dim == 175
        assert builder.max_positions == 8
        assert len(builder.ticker_order) == 8

    def test_build_policy_compatible_observation(self):
        """Observation from builder matches what the policy expects."""
        builder = RLFeatureBuilder(
            max_positions=8,
            ticker_order=["SPY", "QQQ", "IWM"],
        )
        obs = builder.build(
            current_date=date.today(),
            ensemble_preds={
                "SPY": {"prob_buy": 0.6, "prob_hold": 0.2, "prob_sell": 0.2, "signal_num": 2},
                "QQQ": {"prob_buy": 0.3, "prob_hold": 0.5, "prob_sell": 0.2, "signal_num": 1},
                "IWM": {"prob_buy": 0.1, "prob_hold": 0.3, "prob_sell": 0.6, "signal_num": 0},
            },
        )
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (175,)
        assert obs.dtype == np.float32
        assert np.all(np.isfinite(obs))

    def test_observation_variance(self):
        """Different inputs produce different observations."""
        builder = RLFeatureBuilder(max_positions=8, ticker_order=["SPY", "QQQ"])

        obs1 = builder.build(
            current_date=date.today(),
            ensemble_preds={
                "SPY": {"prob_buy": 1.0, "prob_hold": 0.0, "prob_sell": 0.0, "signal_num": 2},
            },
        )

        obs2 = builder.build(
            current_date=date.today(),
            ensemble_preds={
                "SPY": {"prob_buy": 0.0, "prob_hold": 0.0, "prob_sell": 1.0, "signal_num": 0},
            },
        )

        assert not np.array_equal(obs1, obs2), "Different inputs should produce different obs"


class TestStrategyIntegration:
    """Test CoreStrategy RL integration (without requiring trained model)."""

    def test_load_rl_policy_method_exists(self):
        """CoreStrategy has load_rl_policy method."""
        from core.strategy import CoreStrategy
        strategy = CoreStrategy()
        assert hasattr(strategy, "load_rl_policy")

    def test_disable_rl_method_exists(self):
        """CoreStrategy has disable_rl method."""
        from core.strategy import CoreStrategy
        strategy = CoreStrategy()
        assert hasattr(strategy, "disable_rl")

    def test_compute_weights_rl_method_exists(self):
        """CoreStrategy has _compute_weights_rl method."""
        from core.strategy import CoreStrategy
        strategy = CoreStrategy()
        assert hasattr(strategy, "_compute_weights_rl")

    def test_rl_disabled_by_default(self):
        """RL is disabled by default."""
        from core.strategy import CoreStrategy
        strategy = CoreStrategy()
        assert strategy._rl_enabled is False
        assert strategy._rl_policy is None

    def test_rl_enabled_with_config(self):
        """RL can be enabled via config."""
        from core.strategy import CoreStrategy
        strategy = CoreStrategy(config={"rl": {"enabled": True}})
        assert strategy._rl_enabled is True

    def test_compute_weights_still_works(self, sample_feature_df):
        """Rule-based weight computation still works (backward compat)."""
        from core.strategy import CoreStrategy, SignalResult
        from datetime import date as dt_date

        strategy = CoreStrategy()
        signals = [
            SignalResult(
                ticker="SPY", date=dt_date.today(), signal="BUY", weight=0.0,
                prob_buy=0.7, prob_hold=0.2, prob_sell=0.1,
                sentiment_polarity=0.0, sentiment_confidence=0.0,
            ),
            SignalResult(
                ticker="QQQ", date=dt_date.today(), signal="HOLD", weight=0.0,
                prob_buy=0.3, prob_hold=0.5, prob_sell=0.2,
                sentiment_polarity=0.0, sentiment_confidence=0.0,
            ),
            SignalResult(
                ticker="IWM", date=dt_date.today(), signal="SELL", weight=0.0,
                prob_buy=0.1, prob_hold=0.2, prob_sell=0.7,
                sentiment_polarity=0.0, sentiment_confidence=0.0,
            ),
        ]

        weights = strategy._compute_weights(signals)
        assert isinstance(weights, dict)
        assert "SPY" in weights
        assert weights.get("IWM", 0.0) == pytest.approx(0.0, abs=0.01)  # SELL gets zeroed
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=0.01)
