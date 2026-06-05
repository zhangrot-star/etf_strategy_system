"""Tests for the ETF scoring framework."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import date

from scoring.etf_scorer import ETFScorer, FundScore
from scoring.factors_issuer import IssuerFactorComputer
from scoring.factors_index import IndexQualityFactorComputer
from scoring.factors_fund import FundFactorComputer
from scoring.modulation import MLScoreModulator


class TestIssuerFactorComputer:
    def test_issuer_size_score_tiers(self):
        comp = IssuerFactorComputer()
        assert comp.issuer_size_score(1) == 10.0
        assert comp.issuer_size_score(6) == 8.0
        assert comp.issuer_size_score(15) == 6.0
        assert comp.issuer_size_score(25) == 4.0
        assert comp.issuer_size_score(None) == 4.0

    def test_issuer_profitability_score(self):
        comp = IssuerFactorComputer()
        assert comp.issuer_profitability_score(0.20, 0.15) == 10.0
        assert comp.issuer_profitability_score(0.10, 0.15) == 5.0
        assert comp.issuer_profitability_score(None, 0.15) == 5.0

    def test_compute_module_empty(self):
        comp = IssuerFactorComputer()
        result = comp.compute_module(pd.DataFrame())
        assert result.empty


class TestIndexQualityFactorComputer:
    def test_tracking_error_score(self):
        comp = IndexQualityFactorComputer()
        assert comp.tracking_error_score(0.005) == 10.0
        assert comp.tracking_error_score(0.025) == 6.0
        assert comp.tracking_error_score(0.06) == 2.0
        assert comp.tracking_error_score(None) == 5.0

    def test_liquidity_score(self):
        comp = IndexQualityFactorComputer()
        assert comp.liquidity_score(0.10) == 10.0
        assert comp.liquidity_score(0.001) == 2.0
        assert comp.liquidity_score(None) == 5.0

    def test_expense_ratio_score(self):
        comp = IndexQualityFactorComputer()
        assert comp.expense_ratio_score(0.001, "A") == 10.0  # < 0.25%
        assert comp.expense_ratio_score(0.005, "A") == 6.0  # < 1%
        assert comp.expense_ratio_score(0.0005, "US") == 10.0


class TestFundFactorComputer:
    def test_sharpe_score(self):
        comp = FundFactorComputer()
        assert comp.sharpe_score(2.0) == 10.0   # >= 1.5 = excellent
        assert comp.sharpe_score(0.2) == 2.0    # < 0.3 = poor
        assert comp.sharpe_score(None) == 5.0

    def test_drawdown_score(self):
        comp = FundFactorComputer()
        assert comp.drawdown_score(-0.03) == 10.0  # < 10% = excellent
        assert comp.drawdown_score(-0.35) == 2.0   # >= 30% = poor

    def test_volatility_score(self):
        comp = FundFactorComputer()
        assert comp.volatility_score(0.10) == 10.0   # < 20% = excellent
        assert comp.volatility_score(0.50) == 2.0    # >= 45% = poor

    def test_compute_from_prices(self, sample_prices_df):
        comp = FundFactorComputer()
        result = comp.compute_from_prices(sample_prices_df)
        assert not result.empty
        assert "ticker" in result.columns
        assert "sharpe_1y" in result.columns

    def test_compute_module(self, sample_prices_df):
        comp = FundFactorComputer()
        result = comp.compute_module(sample_prices_df)
        assert not result.empty
        assert "fund_module_total" in result.columns
        assert (result["fund_module_total"] >= 0).all()
        assert (result["fund_module_total"] <= 50).all()


class TestETFScorer:
    def test_score_all(self, sample_prices_df):
        scorer = ETFScorer()
        result = scorer.score_all(sample_prices_df)
        assert not result.empty
        assert "raw_total" in result.columns
        assert "ticker" in result.columns

    def test_score_all_empty(self):
        scorer = ETFScorer()
        result = scorer.score_all(pd.DataFrame())
        assert result.empty


class TestMLScoreModulator:
    def test_buy_modulation(self):
        mod = MLScoreModulator()
        fs = FundScore(ticker="TEST", score_date=date.today(), raw_total=80.0)
        mod.modulate(fs, "BUY", 0.8, 0.0, 0.0)
        assert fs.modulation_factor == 1.15
        assert fs.adjusted_total == 92.0
        assert fs.recommendation == "STRONG_BUY"

    def test_sell_modulation(self):
        mod = MLScoreModulator()
        fs = FundScore(ticker="TEST", score_date=date.today(), raw_total=80.0)
        mod.modulate(fs, "SELL", 0.8, 0.0, 0.0)
        assert fs.modulation_factor == 0.80
        assert fs.adjusted_total == 64.0

    def test_hold_no_change(self):
        mod = MLScoreModulator()
        fs = FundScore(ticker="TEST", score_date=date.today(), raw_total=70.0)
        mod.modulate(fs, "HOLD", 0.5, 0.0, 0.0)
        assert fs.modulation_factor == 1.0
        assert fs.adjusted_total == 70.0

    def test_sentiment_risk_warning(self):
        mod = MLScoreModulator()
        fs = FundScore(ticker="TEST", score_date=date.today(), raw_total=70.0)
        mod.modulate(fs, "HOLD", 0.5, sentiment_polarity=-0.8, sentiment_confidence=0.9)
        assert "HIGH_RISK" in fs.risk_warning

    def test_modulate_dataframe(self):
        mod = MLScoreModulator()
        df = pd.DataFrame([
            {"ticker": "A", "raw_total": 85.0},
            {"ticker": "B", "raw_total": 65.0},
            {"ticker": "C", "raw_total": 45.0},
        ])
        preds = {"A": ("BUY", 0.8), "B": ("HOLD", 0.5), "C": ("SELL", 0.8)}
        result = mod.modulate_dataframe(df, preds)
        assert result.loc[0, "ticker"] == "A"
        assert result.loc[0, "adjusted_total"] == pytest.approx(97.75)
        assert result.loc[2, "adjusted_total"] == pytest.approx(36.0)
