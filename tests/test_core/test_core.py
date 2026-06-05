"""Tests for core decision engine module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import date, datetime

from core.ensemble import XGBoostEnsemble, EnsemblePrediction
from core.risk_controller import RiskController, RiskLevel, RiskEvent
from core.strategy import CoreStrategy, AllocationResult, SignalResult


class TestXGBoostEnsemble:
    def test_fit_and_predict(self, sample_feature_df, sample_labels):
        ensemble = XGBoostEnsemble()
        ensemble.fit(sample_feature_df, sample_labels)
        assert ensemble.is_fitted
        assert len(ensemble.feature_names) > 0

        predictions = ensemble.predict(sample_feature_df.head(10))
        assert len(predictions) == 10
        for p in predictions:
            assert p.signal in ("BUY", "HOLD", "SELL")
            assert 0 <= p.prob_buy <= 1
            assert abs(p.prob_buy + p.prob_hold + p.prob_sell - 1.0) < 0.01

    def test_feature_importance(self, sample_feature_df, sample_labels):
        ensemble = XGBoostEnsemble()
        ensemble.fit(sample_feature_df, sample_labels)
        importance = ensemble.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) > 0

    def test_labels_from_forward_returns(self):
        returns = pd.Series(np.random.randn(100) / 100)
        labels = XGBoostEnsemble.labels_from_forward_returns(returns)
        assert set(labels.unique()).issubset({0, 1, 2})

    def test_predict_raises_if_not_fitted(self):
        ensemble = XGBoostEnsemble()
        with pytest.raises(RuntimeError):
            ensemble.predict(pd.DataFrame({"a": [1]}))


class TestRiskController:
    def test_normal_sentiment(self):
        rc = RiskController()
        event = rc.check(polarity=0.5, confidence=0.5)
        assert event.risk_level == RiskLevel.NORMAL
        assert not event.is_breached
        assert not event.should_liquidate

    def test_breach_by_polarity(self):
        rc = RiskController()
        event = rc.check(polarity=-0.9, confidence=0.5)
        assert event.risk_level == RiskLevel.BREACHED
        assert event.is_breached

    def test_breach_by_high_confidence_bearish(self):
        rc = RiskController()
        event = rc.check(polarity=-0.6, confidence=0.90)
        assert event.risk_level == RiskLevel.BREACHED
        assert event.is_breached

    def test_warning_zone(self):
        rc = RiskController()
        # Default warn threshold is -0.5; -0.6 is below warn but above breach (-0.7)
        event = rc.check(polarity=-0.6, confidence=0.5)
        assert event.risk_level == RiskLevel.WARNING
        assert not event.is_breached

    def test_check_portfolio_liquidates_on_breach(self):
        rc = RiskController()
        sentiment = pd.DataFrame([
            {"ticker": "SPY", "polarity": 0.5, "confidence": 0.6},
            {"ticker": "QQQ", "polarity": -0.9, "confidence": 0.7},
            {"ticker": "IWM", "polarity": 0.1, "confidence": 0.3},
        ])
        event = rc.check_portfolio(sentiment, ["SPY", "QQQ", "IWM"])
        assert event.is_breached
        assert event.override_weights == {"SPY": 0.0, "QQQ": 0.0, "IWM": 0.0}


class TestCoreStrategy:
    def test_equal_weight_fallback(self):
        strategy = CoreStrategy(config={"risk": {"max_positions": 8}})
        features = pd.DataFrame(index=["SPY", "QQQ"])
        allocation = strategy._equal_weight_allocation(features, date.today())
        assert len(allocation.signals) == 2
        assert {s.ticker for s in allocation.signals} == {"SPY", "QQQ"}

    def test_all_cash_when_empty_features(self):
        strategy = CoreStrategy()
        allocation = strategy.generate_allocation(
            features=pd.DataFrame(),
            sentiment=pd.DataFrame(),
            current_date=date.today(),
        )
        assert allocation.is_all_cash

    def test_constraints_applied(self):
        strategy = CoreStrategy(
            config={"risk": {"max_positions": 3, "single_position_cap": 0.25}}
        )
        signals = [
            SignalResult(ticker=k, date=date.today(), signal="BUY", weight=0.0,
                         prob_buy=v, prob_hold=0.0, prob_sell=0.0,
                         sentiment_polarity=0.0, sentiment_confidence=0.0)
            for k, v in [("A", 0.4), ("B", 0.35), ("C", 0.2), ("D", 0.3), ("E", 0.1)]
        ]
        result = strategy._compute_weights(signals)
        assert len(result) == 3
        assert abs(sum(result.values()) - 1.0) < 0.01
