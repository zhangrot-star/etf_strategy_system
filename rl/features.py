"""RLFeatureBuilder — assemble 175-dim fixed-length observation vector.

Flattens per-ticker features (21 features × 8 max positions = 168) plus
7 global features into a single numpy array for the PPO policy network.

Missing or unavailable features are filled with zeros.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

# Per-ticker feature count
_PER_TICKER_FEATURES = 21
_GLOBAL_FEATURES = 7


class RLFeatureBuilder:
    """Build fixed-size observation vector from heterogeneous data sources."""

    def __init__(self, max_positions: int = 8, ticker_order: list[str] | None = None) -> None:
        self._max_positions = max_positions
        self._ticker_order = ticker_order or []
        self._per_ticker = _PER_TICKER_FEATURES
        self._obs_dim = self._per_ticker * max_positions + _GLOBAL_FEATURES

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def observation_dim(self) -> int:
        return self._obs_dim

    @property
    def max_positions(self) -> int:
        return self._max_positions

    @property
    def ticker_order(self) -> list[str]:
        return self._ticker_order

    def set_ticker_order(self, tickers: list[str]) -> None:
        self._ticker_order = list(tickers)

    def build(
        self,
        current_date: date,
        ensemble_preds: dict[str, dict[str, float]] | None = None,
        regressor_preds: dict[str, dict[str, float]] | None = None,
        scores: dict[str, dict[str, float]] | None = None,
        sentiment: dict[str, dict[str, float]] | None = None,
        technicals: dict[str, dict[str, float]] | None = None,
        current_weights: dict[str, float] | None = None,
        cash_weight: float = 0.0,
        portfolio_vol_21d: float = 0.0,
        avg_correlation: float = 0.0,
        market_regime: int = 1,
        days_since_rebalance: int = 0,
        n_positions: int = 0,
        market_return_21d: float = 0.0,
    ) -> np.ndarray:
        """Assemble the full 175-dim observation vector.

        Args:
            current_date: The rebalance/evaluation date.
            ensemble_preds: Map ticker → {prob_buy, prob_hold, prob_sell, signal_num}.
            regressor_preds: Map ticker → {pred_5d, pred_21d, pred_63d, prob_up_5d, prob_up_21d, prob_up_63d}.
            scores: Map ticker → {raw_total, adjusted_total}.
            sentiment: Map ticker → {polarity, confidence}.
            technicals: Map ticker → {roc_21d, rsi_14d, atr_21d, bb_pct_b, hist_vol_21d,
                                      sma_ratio_63d, volume_ma_ratio_20d}.
            current_weights: Map ticker → current portfolio weight.
            cash_weight: Current portfolio cash fraction.
            portfolio_vol_21d: Rolling 21-day portfolio volatility (annualized).
            avg_correlation: Mean pairwise correlation among holdings.
            market_regime: 0=BEAR, 1=NEUTRAL, 2=BULL.
            days_since_rebalance: Days since last rebalance.
            n_positions: Count of active positions.
            market_return_21d: Broad market return over past 21 trading days.

        Returns:
            (175,) float32 numpy array.
        """
        ensemble_preds = ensemble_preds or {}
        regressor_preds = regressor_preds or {}
        scores = scores or {}
        sentiment_data = sentiment or {}
        technicals = technicals or {}
        current_weights = current_weights or {}
        weights_map = current_weights

        # Use provided ticker order; fall back to union of all tickers seen in data
        tickers = self._ticker_order or self._infer_ticker_order(
            ensemble_preds, regressor_preds, scores, sentiment_data, technicals, weights_map
        )

        per_ticker_vecs: list[np.ndarray] = []

        for i in range(self._max_positions):
            if i < len(tickers):
                t = tickers[i]
                vec = self._build_ticker_block(
                    ticker=t,
                    ensemble=ensemble_preds.get(t, {}),
                    regressor=regressor_preds.get(t, {}),
                    score=scores.get(t, {}),
                    sent=sentiment_data.get(t, {}),
                    tech=technicals.get(t, {}),
                    weight=weights_map.get(t, 0.0),
                )
            else:
                vec = np.zeros(self._per_ticker, dtype=np.float32)
            per_ticker_vecs.append(vec)

        global_vec = np.array(
            [
                float(cash_weight),
                float(portfolio_vol_21d),
                float(avg_correlation),
                float(market_regime),
                float(days_since_rebalance) / 252.0,
                float(n_positions) / max(self._max_positions, 1),
                float(market_return_21d),
            ],
            dtype=np.float32,
        )

        obs = np.concatenate([np.concatenate(per_ticker_vecs), global_vec])
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return obs.astype(np.float32)

    def from_dataframe(
        self,
        features: pd.DataFrame,
        current_date: date,
        ensemble_preds: dict[str, dict[str, float]] | None = None,
        regressor_preds: dict[str, dict[str, float]] | None = None,
        sentiment_df: pd.DataFrame | None = None,
        scores_df: pd.DataFrame | None = None,
        current_weights: dict[str, float] | None = None,
        **global_kwargs: Any,
    ) -> np.ndarray:
        """Convenience method: extract features from DataFrames at a given date.

        Args:
            features: Feature panel indexed by (ticker, trade_date) or with
                      columns [ticker, trade_date, ...features...].
            current_date: The date to extract features for.
            ensemble_preds, regressor_preds, sentiment_df, scores_df:
                Pre-computed prediction/sentiment/score data.
            current_weights: Current portfolio weights per ticker.
            **global_kwargs: Passed through to build().
        """
        # Extract technical features for current_date
        technicals: dict[str, dict[str, float]] = {}

        if hasattr(features.index, "names") and features.index.names == ["ticker", "trade_date"]:
            # MultiIndex: (ticker, trade_date)
            try:
                slice_df = features.xs(pd.Timestamp(current_date), level="trade_date", drop_level=False)
                for ticker in slice_df.index.get_level_values("ticker"):
                    row = slice_df.loc[ticker]
                    # Try to extract known technical feature columns
                    tech = {}
                    for col in [
                        "roc_21d", "rsi_14d", "atr_21d", "bb_pct_b",
                        "hist_vol_21d", "sma_ratio_63d", "volume_ma_ratio_20d",
                    ]:
                        if col in row.index:
                            val = row[col]
                            tech[col] = float(val.iloc[0]) if hasattr(val, "iloc") else float(val)
                        else:
                            tech[col] = 0.0
                    technicals[ticker] = tech
            except (KeyError, IndexError):
                pass
        elif "trade_date" in features.columns and "ticker" in features.columns:
            date_mask = pd.to_datetime(features["trade_date"]) == pd.Timestamp(current_date)
            day_df = features[date_mask]
            for _, row in day_df.iterrows():
                t = row["ticker"]
                tech = {}
                for col in [
                    "roc_21d", "rsi_14d", "atr_21d", "bb_pct_b",
                    "hist_vol_21d", "sma_ratio_63d", "volume_ma_ratio_20d",
                ]:
                    tech[col] = float(row.get(col, 0.0))
                technicals[t] = tech

        # Extract sentiment for current_date
        sentiment_map: dict[str, dict[str, float]] = {}
        if sentiment_df is not None and not sentiment_df.empty:
            date_col = "event_date" if "event_date" in sentiment_df.columns else sentiment_df.columns[1]
            sent_filtered = sentiment_df[
                pd.to_datetime(sentiment_df[date_col]) <= pd.Timestamp(current_date)
            ]
            for ticker, group in sent_filtered.groupby("ticker"):
                last = group.sort_values(date_col).iloc[-1]
                sentiment_map[ticker] = {
                    "polarity": float(last.get("polarity", 0.0)),
                    "confidence": float(last.get("confidence", 0.0)),
                }

        return self.build(
            current_date=current_date,
            ensemble_preds=ensemble_preds,
            regressor_preds=regressor_preds,
            scores=None,
            sentiment=sentiment_map,
            technicals=technicals,
            current_weights=current_weights,
            **global_kwargs,
        )

    # ── Private helpers ─────────────────────────────────────────────────

    def _build_ticker_block(
        self,
        ticker: str,
        ensemble: dict[str, float],
        regressor: dict[str, float],
        score: dict[str, float],
        sent: dict[str, float],
        tech: dict[str, float],
        weight: float,
    ) -> np.ndarray:
        """Build the 21-element feature vector for a single ticker."""
        vec = np.array(
            [
                # Ensemble signals (4)
                ensemble.get("prob_buy", 0.0),
                ensemble.get("prob_hold", 0.0),
                ensemble.get("prob_sell", 0.0),
                ensemble.get("signal_num", 1.0),  # HOLD=1 default
                # Regressor predictions (6)
                regressor.get("pred_5d", 0.0),
                regressor.get("pred_21d", 0.0),
                regressor.get("pred_63d", 0.0),
                regressor.get("prob_up_5d", 0.5),
                regressor.get("prob_up_21d", 0.5),
                regressor.get("prob_up_63d", 0.5),
                # Multi-factor scores (2)
                score.get("raw_total", 50.0),
                score.get("adjusted_total", 50.0),
                # Sentiment (2)
                sent.get("polarity", 0.0),
                sent.get("confidence", 0.0),
                # Technical factors (7)
                tech.get("roc_21d", 0.0),
                tech.get("rsi_14d", 50.0),
                tech.get("atr_21d", 0.0),
                tech.get("bb_pct_b", 0.5),
                tech.get("hist_vol_21d", 0.0),
                tech.get("sma_ratio_63d", 1.0),
                tech.get("volume_ma_ratio_20d", 1.0),
            ],
            dtype=np.float32,
        )
        return vec

    @staticmethod
    def _infer_ticker_order(*dicts: dict) -> list[str]:
        seen: set[str] = set()
        order: list[str] = []
        for d in dicts:
            if d:
                for k in d:
                    if k not in seen:
                        seen.add(k)
                        order.append(k)
        return sorted(order)
