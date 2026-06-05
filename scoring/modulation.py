"""ML prediction and sentiment modulation for ETF scores.

Bridges XGBoost signals (BUY/HOLD/SELL) and sentiment polarity into score
multipliers and risk caps.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from scoring.etf_scorer import FundScore

logger = logging.getLogger(__name__)


class MLScoreModulator:
    """Modulates FundScore total based on XGBoost predictions and sentiment."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        m = cfg.get("modulation", {})
        self._buy_high_boost: float = m.get("buy_high_boost", 1.15)
        self._buy_low_boost: float = m.get("buy_low_boost", 1.05)
        self._sell_high_penalty: float = m.get("sell_high_penalty", 0.80)
        self._sell_low_penalty: float = m.get("sell_low_penalty", 0.90)
        self._sell_very_low_penalty: float = m.get("sell_very_low_penalty", 0.95)

    def modulate(
        self,
        score: FundScore,
        ml_signal: str,
        ml_confidence: float,
        sentiment_polarity: float = 0.0,
        sentiment_confidence: float = 0.0,
    ) -> FundScore:
        """Apply ML and sentiment modulation to a FundScore.

        Returns the modified FundScore (mutates input and returns it).
        """
        score.ml_signal = ml_signal
        score.ml_confidence = ml_confidence

        # ML modulation
        if ml_signal == "BUY":
            if ml_confidence > 0.7:
                score.modulation_factor = self._buy_high_boost
            else:
                score.modulation_factor = self._buy_low_boost
        elif ml_signal == "SELL":
            if ml_confidence > 0.7:
                score.modulation_factor = self._sell_high_penalty
            elif ml_confidence > 0.4:
                score.modulation_factor = self._sell_low_penalty
            else:
                score.modulation_factor = self._sell_very_low_penalty
        else:
            score.modulation_factor = 1.0

        score.adjusted_total = round(score.raw_total * score.modulation_factor, 2)

        # Recommendation
        score.recommendation = self._to_recommendation(score.adjusted_total, ml_signal)

        # Sentiment-based risk warning
        if sentiment_polarity < -0.7 and sentiment_confidence > 0.85:
            score.risk_warning = "HIGH_RISK: extreme bearish sentiment"
        elif sentiment_polarity < -0.5:
            score.risk_warning = "CAUTION: elevated bearish sentiment"
        else:
            score.risk_warning = ""

        return score

    def modulate_dataframe(
        self,
        scores_df: pd.DataFrame,
        predictions: dict[str, tuple[str, float]],  # ticker → (signal, confidence)
        sentiment: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Modulate a DataFrame of FundScores in bulk.

        Args:
            scores_df: From ETFScorer.score_all().
            predictions: Dict mapping ticker → (BUY/HOLD/SELL, confidence).
            sentiment: Optional sentiment DataFrame with ticker, polarity, confidence.

        Returns:
            DataFrame with modulation columns added, sorted by adjusted_total.
        """
        for idx, row in scores_df.iterrows():
            ticker = row["ticker"]
            signal, conf = predictions.get(ticker, ("HOLD", 0.5))

            sent_pol, sent_conf = 0.0, 0.0
            if sentiment is not None and not sentiment.empty:
                s = sentiment[sentiment["ticker"] == ticker]
                if not s.empty:
                    sent_pol = float(s.iloc[-1].get("polarity", 0.0))
                    sent_conf = float(s.iloc[-1].get("confidence", 0.0))

            if signal == "BUY":
                if conf > 0.7:
                    factor = self._buy_high_boost
                else:
                    factor = self._buy_low_boost
            elif signal == "SELL":
                if conf > 0.7:
                    factor = self._sell_high_penalty
                elif conf > 0.4:
                    factor = self._sell_low_penalty
                else:
                    factor = self._sell_very_low_penalty
            else:
                factor = 1.0

            # Sentiment overlay: additional penalty for bearish signals
            sent_penalty = 1.0
            if sent_pol < -0.7 and sent_conf > 0.85:
                sent_penalty = 0.90
            elif sent_pol < -0.5:
                sent_penalty = 0.95

            scores_df.at[idx, "ml_signal"] = signal
            scores_df.at[idx, "ml_confidence"] = conf
            scores_df.at[idx, "modulation_factor"] = round(factor * sent_penalty, 3)
            scores_df.at[idx, "adjusted_total"] = round(row["raw_total"] * factor * sent_penalty, 2)
            scores_df.at[idx, "recommendation"] = self._to_recommendation(
                row["raw_total"] * factor * sent_penalty, signal,
            )

        result = scores_df.sort_values("adjusted_total", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)
        return result

    @staticmethod
    def _to_recommendation(adjusted_total: float, ml_signal: str) -> str:
        if ml_signal == "BUY" and adjusted_total >= 80:
            return "STRONG_BUY"
        if ml_signal == "BUY":
            return "BUY"
        if adjusted_total >= 75:
            return "BUY"
        if ml_signal == "SELL" and adjusted_total < 50:
            return "STRONG_SELL"
        if ml_signal == "SELL":
            return "SELL"
        if adjusted_total < 40:
            return "SELL"
        return "HOLD"
