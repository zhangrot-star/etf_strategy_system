"""Fixed analyzers: drawdown, turnover, and trade statistics.

Fixes the original TurnoverAnalyzer which accumulated nothing in
_daily_turnovers and relied on a broken `self.strategy.orders` reference.
"""

from __future__ import annotations

import backtrader as bt
import numpy as np


class DrawdownAnalyzer(bt.Analyzer):
    """Tracks peak-to-trough drawdown and drawdown duration."""

    def __init__(self) -> None:
        super().__init__()
        self._peak: float = float("-inf")
        self._max_drawdown: float = 0.0
        self._max_dd_len: int = 0
        self._current_dd_len: int = 0
        self._dd_start: int = 0
        self._equity_curve: list[float] = []

    def next(self) -> None:
        cur_value: float = self.strategy.broker.getvalue()
        self._equity_curve.append(cur_value)

        if cur_value > self._peak:
            self._peak = cur_value
            self._current_dd_len = 0
        else:
            self._current_dd_len += 1

        if self._peak > 0:
            dd = (self._peak - cur_value) / self._peak
            if dd > self._max_drawdown:
                self._max_drawdown = dd
                self._max_dd_len = self._current_dd_len

    def get_analysis(self) -> dict:
        return {
            "max_drawdown": self._max_drawdown,
            "max_drawdown_len": self._max_dd_len,
            "equity_curve": self._equity_curve,
        }


class TurnoverAnalyzer(bt.Analyzer):
    """Tracks portfolio turnover correctly.

    Turnover = total traded value / (2 * average portfolio value).
    This is the standard definition: sum of |trades| / (2 * avg NAV).
    """

    def __init__(self) -> None:
        super().__init__()
        self._traded_values: list[float] = []
        self._daily_values: list[float] = []
        self._trade_count: int = 0

    def next(self) -> None:
        self._daily_values.append(self.strategy.broker.getvalue())

    def notify_trade(self, trade: bt.Trade) -> None:
        if trade.isclosed:
            self._traded_values.append(trade.value)
            self._trade_count += 1

    def get_analysis(self) -> dict:
        total_traded = sum(abs(v) for v in self._traded_values)
        avg_value = np.mean(self._daily_values) if self._daily_values else 1.0
        # Standard turnover formula
        turnover_ratio = total_traded / (2 * avg_value) if avg_value > 0 else 0.0
        return {
            "total_traded_value": total_traded,
            "turnover_ratio": turnover_ratio,
            "trade_count": self._trade_count,
            "n_bars": len(self._daily_values),
        }


class TradeStatsAnalyzer(bt.Analyzer):
    """Collects per-trade statistics: win rate, avg win/loss, profit factor."""

    def __init__(self) -> None:
        super().__init__()
        self._pnls: list[float] = []
        self._wins: int = 0
        self._losses: int = 0

    def notify_trade(self, trade: bt.Trade) -> None:
        if trade.isclosed:
            pnl = trade.pnlcomm  # PnL net of commission
            self._pnls.append(pnl)
            if pnl > 0:
                self._wins += 1
            elif pnl < 0:
                self._losses += 1

    def get_analysis(self) -> dict:
        n = len(self._pnls)
        if n == 0:
            return {
                "total_trades": 0, "win_rate": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0,
                "profit_factor": 0.0, "total_pnl": 0.0,
            }
        wins = [p for p in self._pnls if p > 0]
        losses = [p for p in self._pnls if p < 0]
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = abs(np.mean(losses)) if losses else 0.0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        return {
            "total_trades": n,
            "win_rate": self._wins / n if n > 0 else 0.0,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "total_pnl": sum(self._pnls),
        }
