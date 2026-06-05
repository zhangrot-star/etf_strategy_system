"""Tests for factors module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    from factors.technical import TechnicalFactorBuilder
    _HAS_PANDAS_TA = True
except ImportError:
    _HAS_PANDAS_TA = False
    TechnicalFactorBuilder = None

from factors.causal import CausalInferenceEngine
from factors.factor_registry import FactorRegistry


@pytest.mark.skipif(not _HAS_PANDAS_TA, reason="pandas_ta not available")
class TestTechnicalFactorBuilder:
    def test_build_all_factors_returns_long_format(self, sample_prices_df):
        builder = TechnicalFactorBuilder()
        result = builder.build_all_factors(sample_prices_df)
        assert "ticker" in result.columns
        assert "trade_date" in result.columns
        assert "factor_name" in result.columns
        assert "value" in result.columns
        assert len(result) > 0
        # Check some expected factor names exist
        names = set(result["factor_name"])
        assert any("roc_" in n for n in names)
        assert any("rsi_" in n for n in names)
        assert any("atr_" in n for n in names)

    def test_empty_input(self):
        builder = TechnicalFactorBuilder()
        result = builder.build_all_factors(pd.DataFrame())
        assert result.empty


class TestCausalInferenceEngine:
    @pytest.fixture
    def panel_data(self) -> pd.DataFrame:
        """Synthetic panel with clear treatment effect."""
        np.random.seed(42)
        tickers = ["A", "B", "C"]
        dates = pd.bdate_range("2020-01-01", "2021-12-31")
        rows = []
        for t in tickers:
            for i, d in enumerate(dates):
                treated = 1 if t in ["A", "B"] else 0
                post = 1 if d >= pd.Timestamp("2020-07-01") else 0
                # Outcome: beta * market + treatment effect + noise
                market_ret = np.random.normal(0.0002, 0.01)
                treat_effect = 0.002 if (treated and post) else 0.0
                ret = 0.8 * market_ret + treat_effect + np.random.normal(0, 0.005)
                rows.append({
                    "ticker": t,
                    "trade_date": d.date(),
                    "return": ret,
                    "market_return": market_ret,
                    "treated": treated,
                    "post": post,
                })
        return pd.DataFrame(rows)

    def test_run_did_estimates_effect(self, panel_data):
        engine = CausalInferenceEngine()
        result = engine.run_did(
            panel_data,
            outcome_col="return",
            treatment_col="treated",
            post_col="post",
            entity_col="ticker",
            time_col="trade_date",
            control_vars=["market_return"],
        )
        assert result.method == "DID"
        assert result.n_observations > 0

    def test_run_panel_beta(self, panel_data):
        engine = CausalInferenceEngine()
        betas = engine.run_panel_beta(
            panel_data,
            return_col="return",
            market_col="market_return",
            entity_col="ticker",
            time_col="trade_date",
        )
        assert len(betas) == 3
        assert "beta" in betas.columns
        assert "alpha" in betas.columns


class TestFactorRegistry:
    def test_register_and_compute(self):
        registry = FactorRegistry()
        registry.register("test_factor", "momentum", lambda df: df["close"] * 2)
        assert "test_factor" in registry.factor_names
        assert registry.by_category("momentum") == ["test_factor"]

    def test_summary_dataframe(self):
        registry = FactorRegistry()
        registry.register("f1", "momentum", lambda df: df["close"])
        registry.register("f2", "volatility", lambda df: df["close"])
        summary = registry.summary()
        assert len(summary) == 2
        assert set(summary["category"]) == {"momentum", "volatility"}
