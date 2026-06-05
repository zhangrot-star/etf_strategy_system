"""Daily ETF recommendation pipeline — full end-to-end flow."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from core.feature_utils import build_features_from_prices
from core.strategy import CoreStrategy
from recommendation.ranker import ETFRanker
from recommendation.schemas import DailyRecommendation, RecommendedETF
from scoring.etf_scorer import ETFScorer
from scoring.modulation import MLScoreModulator

logger = logging.getLogger(__name__)


class DailyRecommendationPipeline:
    """Orchestrates the full daily recommendation flow.

    Flow:
      1. Score all ETFs through ETFScorer
      2. Run XGBoost predictions via CoreStrategy
      3. Modulate scores with ML signals
      4. Rank, filter, assign weights
      5. Generate output
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._scorer = ETFScorer(config)
        self._strategy = CoreStrategy(config)
        self._modulator = MLScoreModulator(config)
        self._ranker = ETFRanker(config)
        self._load_model()

    def _load_model(self) -> None:
        """Load trained XGBoost model if available."""
        import os
        from pathlib import Path

        model_path = self._config.get("recommendation", {}).get("model_path", "models/xgboost_etf")
        if os.path.exists(f"{model_path}.xgb") and os.path.exists(f"{model_path}.pkl"):
            try:
                self._strategy.ensemble.load(model_path)
                logger.info("Loaded ML model from %s", model_path)
            except Exception:
                logger.warning("Failed to load model from %s", model_path, exc_info=True)

    def run(
        self,
        prices: pd.DataFrame,
        sentiment: pd.DataFrame | None = None,
        issuer_df: pd.DataFrame | None = None,
        profiles: pd.DataFrame | None = None,
        index_meta: pd.DataFrame | None = None,
        run_date: date | None = None,
    ) -> DailyRecommendation:
        """Execute full daily pipeline.

        Args:
            prices: OHLCV DataFrame with [ticker, trade_date, close, ...].
            sentiment: Optional sentiment data.
            issuer_df: Optional issuer data for Module 1.
            profiles: Optional ETF profile data for Module 2.
            index_meta: Optional index metadata for Module 2.
            run_date: Date for this run (defaults to today).

        Returns:
            DailyRecommendation with ranked ETF list.
        """
        if run_date is None:
            run_date = date.today()

        if prices.empty:
            return DailyRecommendation(date=run_date, total_universe=0, generated_at=datetime.now(timezone.utc).isoformat())

        # 1. Score all ETFs
        scores_df = self._scorer.score_all(prices, issuer_df=issuer_df, profiles=profiles, index_meta=index_meta, score_date=run_date)

        if scores_df.empty:
            return DailyRecommendation(date=run_date, total_universe=0, generated_at=datetime.now(timezone.utc).isoformat())

        # 2. Build features and try to run ML predictions
        try:
            features = build_features_from_prices(prices)
            if not features.empty and self._strategy.is_fitted:
                ticker_list = list(features.index.get_level_values(0).unique())
                # Take features for latest date
                latest_date = features.index.get_level_values(1).max()
                latest_features = features.xs(latest_date, level=1)

                result = self._strategy.allocate(
                    latest_features,
                    sentiment if sentiment is not None and not sentiment.empty else pd.DataFrame(),
                    run_date,
                )

                # Build prediction map: ticker → (signal, confidence)
                predictions: dict[str, tuple[str, float]] = {}
                for s in result.signals:
                    predictions[s.ticker] = (s.signal, max(s.prob_buy, s.prob_hold, s.prob_sell))
            else:
                predictions = {}
        except Exception:
            logger.warning("ML prediction failed — using raw scores only.", exc_info=True)
            predictions = {}

        # 3. Modulate scores with ML predictions
        if predictions:
            scores_df = self._modulator.modulate_dataframe(scores_df, predictions, sentiment)
        else:
            scores_df["adjusted_total"] = scores_df["raw_total"]
            scores_df["ml_signal"] = "HOLD"
            scores_df["ml_confidence"] = 0.0
            scores_df["modulation_factor"] = 1.0
            scores_df["recommendation"] = "HOLD"

        # 4. Determine risk status from sentiment (before position sizing)
        risk_status = "NORMAL"
        if sentiment is not None and not sentiment.empty:
            avg_pol = float(sentiment["polarity"].mean()) if "polarity" in sentiment.columns else 0.0
            if avg_pol < -0.7:
                risk_status = "BREACHED"
            elif avg_pol < -0.5:
                risk_status = "WARNING"

        # 5. Rank and assign ratings
        scores_df = scores_df.sort_values("adjusted_total", ascending=False).reset_index(drop=True)
        scores_df["rank"] = range(1, len(scores_df) + 1)
        scores_df = self._ranker.assign_ratings(scores_df)

        # 6. Compute allocation weights with sentiment-driven position caps
        if risk_status == "BREACHED":
            max_position = 0.10   # heavily constrained
        elif risk_status == "WARNING":
            max_position = 0.15   # reduced from 30% to 15%
        else:
            max_position = 0.30

        weights = self._ranker.compute_weights(scores_df, max_per_position=max_position)

        # 7. For BREACHED status, force high cash weight
        if risk_status == "BREACHED":
            cash_weight = 0.50
            total_weight = 1.0 - cash_weight
            # Scale all weights proportionally
            current_total = sum(weights.values())
            if current_total > 0:
                for t in weights:
                    weights[t] = weights[t] * total_weight / current_total
        else:
            cash_weight = 1.0 - sum(weights.values())

        # 8. Build output
        ranked_etfs: list[RecommendedETF] = []
        for _, r in scores_df.iterrows():
            ticker = r["ticker"]
            ranked_etfs.append(RecommendedETF(
                ticker=ticker,
                total_score=r.get("adjusted_total", r["raw_total"]),
                rating=r.get("rating", "C"),
                ml_signal=r.get("ml_signal", "HOLD"),
                recommendation=r.get("recommendation", "HOLD"),
                allocation_weight=weights.get(ticker, 0.0),
                risk_level=self._ranker.risk_level(r.get("adjusted_total", r["raw_total"])),
                module_scores={
                    "issuer": r.get("issuer_module_total", 0),
                    "index_quality": r.get("index_module_total", 0),
                    "individual_fund": r.get("fund_module_total", 0),
                },
            ))

        return DailyRecommendation(
            date=run_date,
            total_universe=len(scores_df),
            ranked_etfs=ranked_etfs,
            risk_status=risk_status,
            cash_weight=cash_weight,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
