"""Core ETF scoring orchestrator — adapted from '工匠之选' methodology.

3-module structure (100-point scale):
  Module 1 (10%): Issuer quality
  Module 2 (40%): Index / strategy quality
  Module 3 (50%): Individual fund evaluation

After raw scoring, ML predictions and sentiment modulate the total.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from scoring.factors_fund import FundFactorComputer
from scoring.factors_index import IndexQualityFactorComputer
from scoring.factors_issuer import IssuerFactorComputer

logger = logging.getLogger(__name__)


@dataclass
class FundScore:
    """Full scoring breakdown for a single ETF on a single date."""

    ticker: str
    score_date: date

    # Module 1
    issuer_size_score: float = 0.0
    issuer_profitability_score: float = 0.0
    issuer_module_total: float = 0.0

    # Module 2
    tracking_error_score: float = 0.0
    methodology_score: float = 0.0
    liquidity_score: float = 0.0
    fund_age_score: float = 0.0
    expense_ratio_score: float = 0.0
    dividend_yield_score: float = 0.0
    premium_discount_score: float = 0.0
    index_module_total: float = 0.0

    # Module 3
    scale_score: float = 0.0
    return_score: float = 0.0
    ranking_1y_score: float = 0.0
    ranking_3y_score: float = 0.0
    ranking_5y_score: float = 0.0
    sharpe_1y_score: float = 0.0
    sharpe_3y_score: float = 0.0
    drawdown_1y_score: float = 0.0
    drawdown_3y_score: float = 0.0
    recovery_time_score: float = 0.0
    volatility_1y_score: float = 0.0
    volatility_3y_score: float = 0.0
    win_prob_3m_score: float = 0.0
    win_prob_6m_score: float = 0.0
    fund_module_total: float = 0.0

    # Final
    raw_total: float = 0.0
    ml_signal: str = ""
    ml_confidence: float = 0.0
    modulation_factor: float = 1.0
    adjusted_total: float = 0.0
    rank: int = -1
    recommendation: str = ""
    risk_warning: str = ""


class ETFScorer:
    """Three-module scoring engine for ETFs.

    Usage:
        scorer = ETFScorer(config)
        scores = scorer.score_all(prices)
        # returns ranked DataFrame of FundScores
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._issuer = IssuerFactorComputer(config)
        self._index = IndexQualityFactorComputer(config)
        self._fund = FundFactorComputer(config)

    def score_all(
        self,
        prices: pd.DataFrame,
        issuer_df: pd.DataFrame | None = None,
        profiles: pd.DataFrame | None = None,
        index_meta: pd.DataFrame | None = None,
        score_date: date | None = None,
    ) -> pd.DataFrame:
        """Score the full ETF universe and return a ranked DataFrame.

        Args:
            prices: OHLCV with [ticker, trade_date, open, high, low, close, volume].
            issuer_df: Optional issuer data for Module 1.
            profiles: Optional ETF profile data for Module 2 (needs inception_date, expense_ratio, etc.).
            index_meta: Optional index metadata for Module 2.
            score_date: Scoring date (defaults to today).

        Returns:
            DataFrame of FundScores sorted by adjusted_total descending.
        """
        if score_date is None:
            score_date = date.today()

        if prices.empty:
            return pd.DataFrame()

        tickers = sorted(prices["ticker"].unique())

        # Module 3: Fund factors (always computable from price data)
        fund_scores = self._fund.compute_module(prices)

        # Module 1: Issuer factors
        if issuer_df is not None and not issuer_df.empty:
            issuer_scores = self._issuer.compute_module(issuer_df)
        else:
            issuer_scores = pd.DataFrame()

        # Module 2: Index quality factors
        if profiles is not None and not profiles.empty and index_meta is not None and not index_meta.empty:
            index_scores = self._index.compute_module(profiles, index_meta)
        else:
            index_scores = pd.DataFrame()

        # Assemble FundScores
        scores: list[FundScore] = []
        for ticker in tickers:
            fs = FundScore(ticker=ticker, score_date=score_date)

            # Module 3
            fr = fund_scores[fund_scores["ticker"] == ticker]
            if not fr.empty:
                r = fr.iloc[0]
                fs.return_score = r.get("return_score", 5.0)
                fs.ranking_1y_score = r.get("ranking_1y_score", 5.0)
                fs.ranking_3y_score = r.get("ranking_3y_score", 5.0)
                fs.ranking_5y_score = r.get("ranking_5y_score", 5.0)
                fs.sharpe_1y_score = r.get("sharpe_1y_score", 5.0)
                fs.sharpe_3y_score = r.get("sharpe_3y_score", 5.0)
                fs.drawdown_1y_score = r.get("drawdown_1y_score", 5.0)
                fs.drawdown_3y_score = r.get("drawdown_3y_score", 5.0)
                fs.recovery_time_score = r.get("recovery_time_score", 5.0)
                fs.volatility_1y_score = r.get("volatility_1y_score", 5.0)
                fs.volatility_3y_score = r.get("volatility_3y_score", 5.0)
                fs.win_prob_3m_score = r.get("win_prob_3m_score", 5.0)
                fs.win_prob_6m_score = r.get("win_prob_6m_score", 5.0)
                fs.fund_module_total = r.get("fund_module_total", 25.0)

            # Module 1 (defaults to neutral 5/10 if no data)
            if not issuer_scores.empty:
                # Map ticker → issuer_id via profiles, then look up issuer scores
                issuer_id = None
                if profiles is not None and not profiles.empty:
                    prof = profiles[profiles["ticker"] == ticker]
                    if not prof.empty:
                        issuer_id = prof.iloc[0].get("issuer_id")
                if issuer_id:
                    ir = issuer_scores[issuer_scores["issuer_id"] == issuer_id]
                else:
                    ir = issuer_scores[issuer_scores["issuer_id"] == ticker]  # fallback
                if not ir.empty:
                    fs.issuer_size_score = ir.iloc[0].get("issuer_size_score", 5.0)
                    fs.issuer_profitability_score = ir.iloc[0].get("issuer_profitability_score", 5.0)
                    fs.issuer_module_total = ir.iloc[0].get("issuer_module_total", 0.5)
            if fs.issuer_module_total == 0.0:
                fs.issuer_size_score = 5.0
                fs.issuer_profitability_score = 5.0
                fs.issuer_module_total = 5.0  # neutral = 5/10 each * 0.5 weight * 2 factors

            # Module 2 (defaults to neutral 5/10 per sub-factor if no data)
            if not index_scores.empty:
                ix = index_scores[index_scores["ticker"] == ticker]
                if not ix.empty:
                    r = ix.iloc[0]
                    fs.tracking_error_score = r.get("tracking_error_score", 5.0)
                    fs.methodology_score = r.get("methodology_score", 5.0)
                    fs.liquidity_score = r.get("liquidity_score", 5.0)
                    fs.fund_age_score = r.get("fund_age_score", 5.0)
                    fs.expense_ratio_score = r.get("expense_ratio_score", 5.0)
                    fs.dividend_yield_score = r.get("dividend_yield_score", 5.0)
                    fs.premium_discount_score = r.get("premium_discount_score", 5.0)
                    fs.index_module_total = r.get("index_module_total", 2.0)
            if fs.index_module_total == 0.0:
                fs.tracking_error_score = 5.0
                fs.methodology_score = 5.0
                fs.liquidity_score = 5.0
                fs.fund_age_score = 5.0
                fs.expense_ratio_score = 5.0
                fs.dividend_yield_score = 5.0
                fs.premium_discount_score = 5.0
                fs.index_module_total = 20.0  # neutral = 5/10 each, weights sum to 4.0

            # Raw total (pre-modulation)
            fs.raw_total = fs.issuer_module_total + fs.index_module_total + fs.fund_module_total
            if fs.raw_total < 0.01:
                fs.raw_total = 50.0  # neutral default

            scores.append(fs)

        return pd.DataFrame([s.__dict__ for s in scores]).sort_values("raw_total", ascending=False)
