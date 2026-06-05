"""Module 3: Individual Fund Evaluation (50% weight).

Sub-factors:
  - Fund scale / AUM (10%)
  - Profitability / annualized return vs benchmark (10%)
  - Peer ranking 1y/3y/5y (3%+4%+3% = 10%)
  - Sharpe ratio 1y/3y (2%+2% = 4%)
  - Max drawdown 1y/3y + recovery time (2.5%+2.5%+3% = 8%)
  - Volatility 1y/3y + 3m/6m win probability (2%+2%+2%+2% = 8%)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FundFactorComputer:
    """Compute Module 3 factors: individual fund performance, risk, and holding experience."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        f = cfg.get("scoring", {}).get("individual_fund", {})
        self._optimal_aum_min: float = f.get("optimal_aum_min", 1e9)
        self._optimal_aum_max: float = f.get("optimal_aum_max", 10e9)
        self._return_outperform: float = f.get("return_outperform_threshold", 0.15)
        self._top_rank_pct: float = f.get("top_rank_pct", 0.10)
        # A-share calibrated thresholds
        self._sharpe_excellent: float = f.get("sharpe_excellent", 1.5)
        self._sharpe_good: float = f.get("sharpe_good", 1.0)
        self._sharpe_ok: float = f.get("sharpe_ok", 0.5)
        self._vol_excellent: float = f.get("vol_excellent", 0.20)
        self._vol_good: float = f.get("vol_good", 0.25)
        self._vol_ok: float = f.get("vol_ok", 0.35)
        self._vol_high: float = f.get("vol_high", 0.45)
        self._dd_excellent: float = f.get("dd_excellent", 0.10)
        self._dd_good: float = f.get("dd_good", 0.15)
        self._dd_ok: float = f.get("dd_ok", 0.20)
        self._dd_high: float = f.get("dd_high", 0.30)
        self._win_excellent: float = f.get("win_excellent", 0.60)
        self._win_good: float = f.get("win_good", 0.55)
        self._win_ok: float = f.get("win_ok", 0.50)
        self._win_low: float = f.get("win_low", 0.45)
        self._return_excellent: float = f.get("return_excellent", 0.50)
        self._return_good: float = f.get("return_good", 0.30)
        self._return_ok: float = f.get("return_ok", 0.15)
        self._return_low: float = f.get("return_low", 0.0)

    # ── Individual factor scorers ─────────────────────────────

    def scale_score(self, aum: float | None) -> float:
        """Score fund AUM scale. Mid-size (1-10bn A-share) is optimal."""
        if aum is None or np.isnan(aum):
            return 5.0
        if self._optimal_aum_min <= aum <= self._optimal_aum_max:
            return 10.0
        if aum > self._optimal_aum_max * 1.5:
            return 2.0
        if aum > self._optimal_aum_max:
            return 8.0
        if aum < self._optimal_aum_min * 0.5:
            return 4.0
        return 8.0

    def return_score(self, ann_return: float | None, benchmark_return: float | None) -> float:
        """Score annualized return vs benchmark (A-share calibrated)."""
        if ann_return is None or np.isnan(ann_return):
            return 5.0
        bench = benchmark_return or 0.0
        excess = ann_return - bench
        if excess > self._return_excellent:
            return 10.0
        if excess > self._return_good:
            return 8.0
        if excess > self._return_ok:
            return 6.0
        if excess > self._return_low:
            return 4.0
        return 2.0

    def ranking_score(self, pct_rank: float | None) -> float:
        """Score peer percentile ranking (lower % = better)."""
        if pct_rank is None or np.isnan(pct_rank):
            return 5.0
        if pct_rank <= self._top_rank_pct:
            return 10.0
        if pct_rank <= 0.25:
            return 8.0
        if pct_rank <= 0.50:
            return 6.0
        if pct_rank <= 0.75:
            return 4.0
        return 2.0

    def sharpe_score(self, sharpe: float | None) -> float:
        """Score Sharpe ratio (A-share calibrated)."""
        if sharpe is None or np.isnan(sharpe):
            return 5.0
        if sharpe >= self._sharpe_excellent:
            return 10.0
        if sharpe >= self._sharpe_good:
            return 8.0
        if sharpe >= self._sharpe_ok:
            return 6.0
        if sharpe >= 0.3:
            return 4.0
        return 2.0

    def drawdown_score(self, max_dd: float | None) -> float:
        """Score max drawdown (A-share calibrated)."""
        if max_dd is None or np.isnan(max_dd):
            return 5.0
        dd_abs = abs(max_dd)
        if dd_abs < self._dd_excellent:
            return 10.0
        if dd_abs < self._dd_good:
            return 8.0
        if dd_abs < self._dd_ok:
            return 6.0
        if dd_abs < self._dd_high:
            return 4.0
        return 2.0

    def recovery_time_score(self, days: int | None) -> float:
        """Score max drawdown recovery time in calendar days."""
        if days is None:
            return 5.0
        if days <= 30:
            return 10.0
        if days <= 60:
            return 8.0
        if days <= 120:
            return 6.0
        if days <= 252:
            return 4.0
        return 2.0

    def volatility_score(self, vol: float | None) -> float:
        """Score annualized volatility (A-share calibrated)."""
        if vol is None or np.isnan(vol):
            return 5.0
        if vol < self._vol_excellent:
            return 10.0
        if vol < self._vol_good:
            return 8.0
        if vol < self._vol_ok:
            return 6.0
        if vol < self._vol_high:
            return 4.0
        return 2.0

    def win_prob_score(self, win_pct: float | None) -> float:
        """Score holding-period win probability (A-share calibrated)."""
        if win_pct is None or np.isnan(win_pct):
            return 5.0
        if win_pct >= self._win_excellent:
            return 10.0
        if win_pct >= self._win_good:
            return 8.0
        if win_pct >= self._win_ok:
            return 6.0
        if win_pct >= self._win_low:
            return 4.0
        return 2.0

    # ── Compute helpers from price data ───────────────────────

    def compute_from_prices(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute Module 3 factor values directly from price history.

        Args:
            prices: OHLCV DataFrame with ticker, trade_date, close columns.

        Returns:
            DataFrame with factor values per ticker: ann_return, ann_vol, sharpe_1y,
            max_drawdown_1y, recovery_days, win_prob_3m, win_prob_6m.
        """
        if prices.empty:
            return pd.DataFrame()

        results = []
        for ticker, grp in prices.groupby("ticker"):
            df = grp.sort_values("trade_date").set_index("trade_date")
            c = df["close"]

            if len(c) < 21:
                continue

            daily_ret = c.pct_change().dropna()
            ann_ret = float(daily_ret.mean() * 252) if len(daily_ret) > 0 else 0.0
            ann_vol = float(daily_ret.std() * np.sqrt(252)) if len(daily_ret) > 0 else 0.0
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

            # Max drawdown
            cummax = c.expanding().max()
            dd = c / cummax - 1
            max_dd = float(dd.min())

            # Recovery time (longest underwater period in last 252 days)
            recovery = self._compute_recovery_days(c, window=252)

            # Win probability
            win_3m = float((daily_ret.tail(63) > 0).mean()) if len(daily_ret) >= 63 else 0.0
            win_6m = float((daily_ret.tail(126) > 0).mean()) if len(daily_ret) >= 126 else 0.0

            results.append({
                "ticker": ticker,
                "ann_return": ann_ret,
                "ann_volatility": ann_vol,
                "sharpe_1y": sharpe,
                "max_drawdown_1y": max_dd,
                "recovery_days": recovery,
                "win_prob_3m": win_3m,
                "win_prob_6m": win_6m,
            })

        return pd.DataFrame(results)

    def compute_module(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute full Module 3 scores from price data.

        Returns DataFrame with all sub-scores and fund_module_total.
        """
        factors = self.compute_from_prices(prices)
        if factors.empty:
            return pd.DataFrame()

        result = factors[["ticker"]].copy()

        # Return score (no benchmark, use 0)
        result["return_score"] = factors["ann_return"].apply(lambda x: self.return_score(x, 0.0))

        # Ranking — use percentile within universe
        if len(factors) > 1:
            rank_pct = factors["ann_return"].rank(pct=True, ascending=False)
            result["ranking_1y_score"] = rank_pct.apply(self.ranking_score)
        else:
            result["ranking_1y_score"] = 5.0
        result["ranking_3y_score"] = result["ranking_1y_score"]  # proxy
        result["ranking_5y_score"] = result["ranking_1y_score"]  # proxy

        # Sharpe
        result["sharpe_1y_score"] = factors["sharpe_1y"].apply(self.sharpe_score)
        result["sharpe_3y_score"] = result["sharpe_1y_score"]  # proxy

        # Drawdown
        result["drawdown_1y_score"] = factors["max_drawdown_1y"].apply(self.drawdown_score)
        result["drawdown_3y_score"] = result["drawdown_1y_score"]  # proxy

        # Recovery
        result["recovery_time_score"] = factors["recovery_days"].apply(self.recovery_time_score)

        # Volatility
        result["volatility_1y_score"] = factors["ann_volatility"].apply(self.volatility_score)
        result["volatility_3y_score"] = result["volatility_1y_score"]  # proxy

        # Win probability
        result["win_prob_3m_score"] = factors["win_prob_3m"].apply(self.win_prob_score)
        result["win_prob_6m_score"] = factors["win_prob_6m"].apply(self.win_prob_score)

        # Scale score — AUM not available from prices, use neutral default
        result["scale_score"] = 5.0

        # Weighted total (weights sum to 5.0 = 50% of 100-point scale)
        result["fund_module_total"] = (
            result["scale_score"] * 1.0
            + result["return_score"] * 1.0
            + result["ranking_1y_score"] * 0.3
            + result["ranking_3y_score"] * 0.4
            + result["ranking_5y_score"] * 0.3
            + result["sharpe_1y_score"] * 0.2
            + result["sharpe_3y_score"] * 0.2
            + result["drawdown_1y_score"] * 0.25
            + result["drawdown_3y_score"] * 0.25
            + result["recovery_time_score"] * 0.3
            + result["volatility_1y_score"] * 0.2
            + result["volatility_3y_score"] * 0.2
            + result["win_prob_3m_score"] * 0.2
            + result["win_prob_6m_score"] * 0.2
        )

        return result

    @staticmethod
    def _compute_recovery_days(close: pd.Series, window: int = 252) -> int:
        """Estimate max drawdown recovery time in calendar days."""
        c = close.iloc[-window:] if len(close) > window else close
        if len(c) < 5:
            return 365
        cummax = c.expanding().max()
        dd = c / cummax - 1

        trough_idx = dd.idxmin()
        peak_level = cummax[trough_idx]

        after_trough = c.loc[trough_idx:]
        recovered = after_trough[after_trough >= peak_level * 0.99]
        if recovered.empty:
            return (c.index[-1] - trough_idx).days
        return (recovered.index[0] - trough_idx).days
