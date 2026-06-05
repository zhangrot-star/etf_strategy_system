"""Gymnasium environment for RL-based ETF portfolio optimization.

Walks through historical price data step-by-step. On each rebalance step,
receives an action (raw weight vector) → projects to feasible weights →
steps through daily bars → accumulates portfolio returns → computes reward.

State: 175-dim vector (RLFeatureBuilder)
Action: Box(8,) ∈ [0,1]
Reward: Sharpe contribution - turnover penalty - drawdown penalty + diversity bonus
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from gymnasium import Env, spaces

from rl.features import RLFeatureBuilder

logger = logging.getLogger(__name__)

# Rebalance frequency → approximate trading days per step
_FREQ_DAYS: dict[str, int] = {
    "daily": 1,
    "weekly": 5,
    "monthly": 21,
}


def _maybe_date(d: object) -> date:
    """Coerce to date."""
    if isinstance(d, date):
        return d
    if isinstance(d, pd.Timestamp):
        return d.date()
    return pd.Timestamp(d).date()


class PortfolioOptEnv(Env):
    """Gymnasium environment wrapping historical ETF price data for portfolio optimization.

    The agent observes market/portfolio state, outputs target portfolio weights, and
    receives a reward based on risk-adjusted returns between rebalance dates.

    Episode = full backtest period. Step = one rebalance decision.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame | None = None,
        tickers: list[str] | None = None,
        initial_capital: float = 1_000_000.0,
        rebalance_freq: str = "monthly",
        lookback_days: int = 252,
        max_positions: int = 8,
        single_position_cap: float = 0.30,
        commission_rate: float = 0.0003,
        reward_weights: dict[str, float] | None = None,
        feature_builder: RLFeatureBuilder | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self._prices = prices.copy()
        self._prices["trade_date"] = pd.to_datetime(self._prices["trade_date"])
        self._features = features
        self._feature_builder = feature_builder or RLFeatureBuilder(max_positions=max_positions)
        self._initial_capital = initial_capital
        self._rebalance_freq = rebalance_freq
        self._bars_per_step = _FREQ_DAYS.get(rebalance_freq, 21)
        self._lookback_days = lookback_days
        self._max_positions = max_positions
        self._single_position_cap = single_position_cap
        self._commission_rate = commission_rate

        # Reward weights
        rw = reward_weights or {}
        self._w_sharpe = rw.get("sharpe_weight", 1.0)
        self._w_turnover = rw.get("turnover_weight", 0.5)
        self._w_drawdown = rw.get("drawdown_weight", 1.0)
        self._w_diversity = rw.get("diversification_weight", 0.2)

        # Tickers
        if tickers:
            self._tickers = list(tickers)
        else:
            self._tickers = sorted(self._prices["ticker"].unique())
        self._feature_builder.set_ticker_order(self._tickers)
        self._n_tickers = len(self._tickers)

        # Build date index
        self._all_dates = sorted(self._prices["trade_date"].unique())
        self._lookback_days = min(self._lookback_days, len(self._all_dates) - 1)
        self._date_index = self._all_dates[self._lookback_days:]
        self._n_steps = max(1, len(self._date_index) // self._bars_per_step)

        # Spaces
        obs_dim = self._feature_builder.observation_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self._max_positions,), dtype=np.float32
        )

        # Internal state (set in reset)
        self._step_idx: int = 0
        self._current_weights: np.ndarray = np.zeros(self._max_positions, dtype=np.float32)
        self._prev_weights: np.ndarray = np.zeros(self._max_positions, dtype=np.float32)
        self._portfolio_value: float = initial_capital
        self._peak_value: float = initial_capital
        self._equity_curve: list[float] = []
        self._daily_returns: list[float] = []
        self._current_date: date | None = None
        self._days_since_rebalance: int = 0
        self._n_positions: int = 0

    # ── Gymnasium API ───────────────────────────────────────────────────

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)

        self._step_idx = 0
        self._current_weights = np.zeros(self._max_positions, dtype=np.float32)
        self._current_weights[-1] = 1.0  # All cash represented by last weight slot
        self._prev_weights = self._current_weights.copy()
        self._portfolio_value = self._initial_capital
        self._peak_value = self._initial_capital
        self._equity_curve = [self._initial_capital]
        self._daily_returns = []
        self._days_since_rebalance = 0
        self._n_positions = 0

        obs = self._build_observation()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one rebalance decision.

        Args:
            action: (max_positions,) float32 array in [0, 1].

        Returns:
            observation, reward, terminated, truncated, info
        """
        # 1. Project to feasible weights (per-ticker)
        ticker_weights = self._project_to_constraints(action)

        # 2. Compute turnover from previous weights
        prev_ticker_weights = self._weights_array_to_dict(self._prev_weights)
        turnover = self._compute_turnover(
            ticker_weights, prev_ticker_weights
        )
        self._prev_weights = self._weights_dict_to_array(ticker_weights)

        # 3. Step through daily bars
        start_date_idx = self._lookback_days + self._step_idx * self._bars_per_step
        end_date_idx = min(start_date_idx + self._bars_per_step, len(self._all_dates))

        period_returns: list[float] = []
        for date_idx in range(start_date_idx, end_date_idx):
            if date_idx >= len(self._all_dates):
                break
            current_d = _maybe_date(self._all_dates[date_idx])
            daily_ret = self._compute_daily_return(ticker_weights, current_d)
            self._portfolio_value *= (1.0 + daily_ret)
            self._daily_returns.append(daily_ret)
            self._equity_curve.append(self._portfolio_value)
            if self._portfolio_value > self._peak_value:
                self._peak_value = self._portfolio_value
            period_returns.append(daily_ret)

        self._step_idx += 1
        self._days_since_rebalance += (end_date_idx - start_date_idx)
        self._current_weights = self._weights_dict_to_array(ticker_weights)
        self._current_date = _maybe_date(self._all_dates[end_date_idx - 1]) if end_date_idx > start_date_idx else None
        self._n_positions = sum(1 for w in ticker_weights.values() if w > 0.001)

        # 4. Compute reward
        reward = self._compute_reward(period_returns, turnover)
        # add small per-step reward from cumulative performance
        cum_ret = (self._portfolio_value / self._initial_capital) - 1.0
        reward += 0.01 * cum_ret  # directional signal

        # 5. Build next observation
        obs = self._build_observation()

        # 6. Check termination
        terminated = end_date_idx >= len(self._all_dates)
        truncated = False

        info = {
            "weights": ticker_weights,
            "portfolio_value": float(self._portfolio_value),
            "cum_return": float(cum_ret),
            "drawdown": float(1.0 - self._portfolio_value / max(self._peak_value, 1.0)),
            "turnover": float(turnover),
            "n_positions": self._n_positions,
            "step": self._step_idx,
        }

        return obs, float(reward), terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        if mode == "human":
            cum_ret = (self._portfolio_value / self._initial_capital - 1.0) * 100
            dd = (1.0 - self._portfolio_value / max(self._peak_value, 1.0)) * 100
            print(
                f"Step {self._step_idx:4d} | "
                f"Value: {self._portfolio_value:>12,.0f} | "
                f"Return: {cum_ret:>+6.2f}% | "
                f"DD: {dd:.1f}% | "
                f"Positions: {self._n_positions}"
            )

    # ── Constraint projection ───────────────────────────────────────────

    def _project_to_constraints(self, raw_action: np.ndarray) -> dict[str, float]:
        """Project raw network output to feasible portfolio weights.

        Steps:
        1. Map action elements to tickers
        2. Clip to [0, single_position_cap]
        3. Keep top-max_positions, zero the rest
        4. Cap total sum at 1.0 (remaining is cash — weights can sum < 1.0)
        """
        action = np.asarray(raw_action, dtype=np.float64).flatten()
        n_tickers = min(len(self._tickers), self._max_positions)

        # Map action to tickers, clip to cap
        weights: dict[str, float] = {}
        for i in range(n_tickers):
            w = float(action[i]) if i < len(action) else 0.0
            weights[self._tickers[i]] = max(0.0, min(w, self._single_position_cap))

        # Zero out tickers beyond action dimension
        for i in range(n_tickers, len(self._tickers)):
            weights[self._tickers[i]] = 0.0

        # Top-N: keep only largest max_positions
        sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top_keys = {k for k, _ in sorted_items[: self._max_positions]}
        for k in list(weights.keys()):
            if k not in top_keys:
                weights[k] = 0.0

        # Cap total sum at 1.0 — if sum exceeds 1.0, scale down proportionally
        total = sum(weights.values())
        if total > 1.0:
            for k in weights:
                weights[k] /= total

        # Re-clip after scaling (in case scaling pushed some over cap)
        for k in weights:
            weights[k] = max(0.0, min(weights[k], self._single_position_cap))

        # Re-normalize after re-clip if still over 1.0
        total = sum(weights.values())
        if total > 1.0:
            for k in weights:
                weights[k] /= total

        # Remove zero-weight entries
        final = {k: v for k, v in weights.items() if v > 1e-6}
        return final

    # ── Reward computation ──────────────────────────────────────────────

    def _compute_reward(
        self,
        period_returns: list[float],
        turnover: float,
    ) -> float:
        """Compute composite reward for a rebalance period."""
        rets = np.array(period_returns)
        n = len(rets)

        # Sharpe contribution
        if n > 1 and rets.std() > 1e-10:
            period_vol = float(rets.std() * np.sqrt(252))
            period_ret = float(np.mean(rets)) * 252
            sharpe_contrib = period_ret / period_vol
        else:
            sharpe_contrib = 0.0

        # Drawdown penalty
        current_dd = 1.0 - self._portfolio_value / max(self._peak_value, 1.0)
        dd_penalty = max(0.0, current_dd - 0.10)

        # Diversification: penalize high concentration (HHI > 0.30)
        weights_arr = self._current_weights
        nonzero = weights_arr[weights_arr > 1e-6]
        if len(nonzero) > 0:
            hhi = float(np.sum((nonzero / nonzero.sum()) ** 2))
        else:
            hhi = 1.0
        div_bonus = -max(0.0, hhi - 0.30)

        reward = (
            self._w_sharpe * sharpe_contrib
            - self._w_turnover * turnover
            - self._w_drawdown * dd_penalty
            + self._w_diversity * div_bonus
        )

        return float(reward)

    # ── Daily return computation ────────────────────────────────────────

    def _compute_daily_return(
        self, weights: dict[str, float], current_date: date
    ) -> float:
        """Compute daily portfolio return given weights and price changes."""
        if not weights:
            return 0.0

        daily_ret = 0.0
        for ticker, w in weights.items():
            if w <= 1e-6:
                continue
            ticker_prices = self._prices[
                (self._prices["ticker"] == ticker)
                & (pd.to_datetime(self._prices["trade_date"]) <= pd.Timestamp(current_date))
            ].sort_values("trade_date")

            if len(ticker_prices) < 2:
                continue

            prev_close = float(ticker_prices.iloc[-2]["close"])
            curr_close = float(ticker_prices.iloc[-1]["close"])
            if prev_close > 0:
                daily_ret += w * (curr_close / prev_close - 1.0)

        # Subtract commission on trades (approximate: proportional to turnover × rate)
        return daily_ret

    @staticmethod
    def _compute_turnover(
        new_weights: dict[str, float], old_weights: dict[str, float]
    ) -> float:
        """Compute one-way turnover fraction."""
        all_tickers = set(new_weights) | set(old_weights)
        turnover = 0.0
        for t in all_tickers:
            turnover += abs(new_weights.get(t, 0.0) - old_weights.get(t, 0.0))
        return turnover / 2.0

    # ── Observation building ────────────────────────────────────────────

    def _build_observation(self) -> np.ndarray:
        """Build the full observation vector for the current state."""
        current_date = self._current_date or (
            _maybe_date(self._all_dates[self._lookback_days]) if self._all_dates else date.today()
        )

        # Market regime from recent returns
        market_regime = self._detect_market_regime()

        # Portfolio volatility (from daily returns)
        if len(self._daily_returns) >= 21:
            recent_rets = self._daily_returns[-21:]
            port_vol = float(np.std(recent_rets) * np.sqrt(252))
        else:
            port_vol = 0.0

        # Market return (21d) — use average return across all tickers
        market_ret_21d = self._compute_market_return_21d()

        # Average correlation among current positions
        avg_corr = self._compute_avg_correlation()

        # Current weights as dict
        current_weights = self._weights_array_to_dict(self._current_weights)

        # Build ticker features
        ensemble_map: dict[str, dict[str, float]] = {}
        regressor_map: dict[str, dict[str, float]] = {}
        technicals: dict[str, dict[str, float]] = {}

        for ticker in self._tickers:
            # Get technical features from feature panel if available
            tech = self._get_technical_features(ticker, current_date)
            technicals[ticker] = tech

            # Default ensemble/regressor predictions (can be overridden externally)
            ensemble_map[ticker] = {
                "prob_buy": 0.33, "prob_hold": 0.34, "prob_sell": 0.33, "signal_num": 1.0,
            }
            regressor_map[ticker] = {
                "pred_5d": 0.0, "pred_21d": 0.0, "pred_63d": 0.0,
                "prob_up_5d": 0.5, "prob_up_21d": 0.5, "prob_up_63d": 0.5,
            }

        return self._feature_builder.build(
            current_date=current_date,
            ensemble_preds=ensemble_map,
            regressor_preds=regressor_map,
            technicals=technicals,
            current_weights=current_weights,
            cash_weight=float(1.0 - sum(current_weights.values())),
            portfolio_vol_21d=port_vol,
            avg_correlation=avg_corr,
            market_regime=market_regime,
            days_since_rebalance=self._days_since_rebalance,
            n_positions=self._n_positions,
            market_return_21d=market_ret_21d,
        )

    def _get_technical_features(self, ticker: str, current_date: date) -> dict[str, float]:
        """Extract technical features for a ticker at a given date from the feature panel."""
        tech = {
            "roc_21d": 0.0, "rsi_14d": 50.0, "atr_21d": 0.0,
            "bb_pct_b": 0.5, "hist_vol_21d": 0.0,
            "sma_ratio_63d": 1.0, "volume_ma_ratio_20d": 1.0,
        }

        if self._features is None or self._features.empty:
            # Compute simple features from prices
            return self._compute_simple_technicals(ticker, current_date)

        try:
            if hasattr(self._features.index, "names") and self._features.index.names == ["ticker", "trade_date"]:
                date_idx = pd.Timestamp(current_date)
                if (ticker, date_idx) in self._features.index:
                    row = self._features.loc[(ticker, date_idx)]
                    for k in tech:
                        if k in row.index:
                            tech[k] = float(row[k])
        except (KeyError, IndexError):
            pass

        return tech

    def _compute_simple_technicals(self, ticker: str, current_date: date) -> dict[str, float]:
        """Compute simple technical features from price data."""
        tech = {
            "roc_21d": 0.0, "rsi_14d": 50.0, "atr_21d": 0.0,
            "bb_pct_b": 0.5, "hist_vol_21d": 0.0,
            "sma_ratio_63d": 1.0, "volume_ma_ratio_20d": 1.0,
        }

        tp = self._prices[
            (self._prices["ticker"] == ticker)
            & (pd.to_datetime(self._prices["trade_date"]) <= pd.Timestamp(current_date))
        ].sort_values("trade_date")

        if len(tp) < 63:
            return tech

        closes = tp["close"].values
        if len(closes) >= 22:
            tech["roc_21d"] = float((closes[-1] / closes[-22] - 1.0) if closes[-22] > 0 else 0.0)
        if len(closes) >= 64:
            sma_63 = float(np.mean(closes[-63:]))
            tech["sma_ratio_63d"] = float(closes[-1] / sma_63) if sma_63 > 0 else 1.0

        if len(closes) >= 22:
            daily_rets = np.diff(closes[-22:]) / (closes[-22:-1] + 1e-10)
            tech["hist_vol_21d"] = float(np.std(daily_rets) * np.sqrt(252))

        if len(closes) >= 15:
            delta = np.diff(closes[-15:])
            gain = np.mean(delta[delta > 0]) if np.any(delta > 0) else 0.0
            loss = -np.mean(delta[delta < 0]) if np.any(delta < 0) else 1e-10
            rs = gain / (loss + 1e-10)
            tech["rsi_14d"] = float(100.0 - 100.0 / (1.0 + rs))

        return tech

    def _detect_market_regime(self) -> int:
        """Classify current market regime: 0=BEAR, 1=NEUTRAL, 2=BULL."""
        if not self._daily_returns:
            return 1
        if len(self._daily_returns) < 21:
            return 1
        recent = np.array(self._daily_returns[-21:])
        cum = np.prod(1.0 + recent) - 1.0
        vol = float(np.std(recent))
        if vol < 1e-6:
            return 1
        sharpe = float(cum / (vol * np.sqrt(21 / 252)))
        if sharpe > 0.5:
            return 2
        if sharpe < -0.5:
            return 0
        return 1

    def _compute_market_return_21d(self) -> float:
        """Compute equal-weighted market return over past 21 days."""
        if not self._tickers:
            return 0.0
        ticker_rets = []
        for t in self._tickers:
            tp = self._prices[self._prices["ticker"] == t].sort_values("trade_date")
            if len(tp) >= 22:
                ret = float(tp.iloc[-1]["close"] / tp.iloc[-22]["close"] - 1)
                ticker_rets.append(ret)
        return float(np.mean(ticker_rets)) if ticker_rets else 0.0

    def _compute_avg_correlation(self) -> float:
        """Compute mean pairwise return correlation among current positions."""
        active_tickers = [
            t for t, w in self._weights_array_to_dict(self._current_weights).items() if w > 1e-6
        ]
        if len(active_tickers) < 2:
            return 0.0

        returns_by_ticker = {}
        for t in active_tickers:
            tp = self._prices[self._prices["ticker"] == t].sort_values("trade_date")
            if len(tp) >= 22:
                closes = tp["close"].values
                rets = np.diff(closes[-22:]) / (closes[-22:-1] + 1e-10)
                returns_by_ticker[t] = rets

        if len(returns_by_ticker) < 2:
            return 0.0

        n = min(21, min(len(r) for r in returns_by_ticker.values()))
        mat = np.column_stack([list(r[-n:]) for r in returns_by_ticker.values()])
        corr = np.corrcoef(mat, rowvar=False)
        n_assets = corr.shape[0]
        # Mean of upper triangle
        if n_assets > 1:
            upper = corr[np.triu_indices(n_assets, k=1)]
            return float(np.nanmean(upper))
        return 0.0

    # ── Weight helpers ──────────────────────────────────────────────────

    def _weights_array_to_dict(self, arr: np.ndarray) -> dict[str, float]:
        """Convert (max_positions,) array to ticker→weight dict."""
        result: dict[str, float] = {}
        for i in range(min(len(self._tickers), self._max_positions)):
            w = float(arr[i]) if i < len(arr) else 0.0
            if w > 0:
                result[self._tickers[i]] = w
        return result

    def _weights_dict_to_array(self, weights: dict[str, float]) -> np.ndarray:
        """Convert ticker→weight dict to (max_positions,) array."""
        arr = np.zeros(self._max_positions, dtype=np.float32)
        for i, t in enumerate(self._tickers[: self._max_positions]):
            arr[i] = float(weights.get(t, 0.0))
        return arr

    # ── Public helpers ──────────────────────────────────────────────────

    def set_predictions(
        self,
        ensemble_preds: dict[str, dict[str, float]] | None = None,
        regressor_preds: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Pre-load ensemble/regressor predictions (called before training)."""
        self._cached_ensemble = ensemble_preds or {}
        self._cached_regressor = regressor_preds or {}

    def get_equity_curve(self) -> list[float]:
        return list(self._equity_curve)
