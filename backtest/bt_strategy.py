"""Backtrader strategy implementation for the ETF strategy system."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import backtrader as bt
import pandas as pd

from core.strategy import CoreStrategy, PortfolioAllocation

logger = logging.getLogger(__name__)


class ETFStrategy(bt.Strategy):
    """Backtrader strategy driven by the CoreStrategy orchestrator.

    On each rebalance date, queries the orchestrator for target weights
    and adjusts positions accordingly.
    """

    params = (
        ("orchestrator", None),        # CoreStrategy instance
        ("features_df", None),         # Pre-computed feature panel
        ("sentiment_df", None),        # Pre-computed sentiment panel
        ("rebalance_freq", "monthly"), # daily | weekly | monthly
        ("lookback_days", 252),        # Min lookback before trading starts
    )

    def __init__(self) -> None:
        self._orchestrator: CoreStrategy = self.p.orchestrator
        self._features: pd.DataFrame = self.p.features_df
        self._sentiment: pd.DataFrame = self.p.sentiment_df
        self._bar_count: int = 0
        self._last_rebalance: pd.Timestamp | None = None

        # Track order status
        self._pending_orders: dict[str, Any] = {}

    def next(self) -> None:
        self._bar_count += 1
        current_dt: pd.Timestamp = self.datas[0].datetime.datetime()  # type: ignore[union-attr]

        # Wait for sufficient history
        if self._bar_count < self.p.lookback_days:
            return

        # Check if rebalance is due
        if not self._should_rebalance(current_dt):
            return

        self._last_rebalance = current_dt
        self._execute_rebalance(current_dt)

    # ── Rebalance logic ──────────────────────────────────────

    def _should_rebalance(self, current_dt: pd.Timestamp) -> bool:
        if self._last_rebalance is None:
            return True

        if self.p.rebalance_freq == "daily":
            return True
        elif self.p.rebalance_freq == "weekly":
            return (current_dt - self._last_rebalance).days >= 5
        elif self.p.rebalance_freq == "monthly":
            # Rebalance on first trading day of new month
            return current_dt.month != self._last_rebalance.month
        return False

    def _execute_rebalance(self, current_dt: pd.Timestamp) -> None:
        """Query orchestrator and adjust portfolio."""
        current_date = current_dt.date()

        # Build feature slice for current date
        feature_slice = self._get_feature_slice(current_date)
        sentiment_slice = self._get_sentiment_slice(current_date)

        if feature_slice.empty:
            return

        # Generate allocation
        allocation: PortfolioAllocation = self._orchestrator.generate_allocation(
            features=feature_slice,
            sentiment=sentiment_slice,
            current_date=current_date,
        )

        if allocation.is_all_cash:
            self._close_all_positions()
            return

        target_weights = allocation.allocations
        port_value = self.broker.getvalue()

        # Close positions not in target
        for data in self.datas:
            ticker = data._name
            position = self.getposition(data).size
            if position != 0 and ticker not in target_weights:
                self.close(data=data)

        # Adjust existing positions and open new ones
        for ticker, target_weight in target_weights.items():
            data = self.getdatabyname(ticker)
            if data is None:
                continue

            target_value = port_value * target_weight
            current_value = self.getposition(data).size * data.close[0]
            diff_value = target_value - current_value

            if abs(diff_value) / port_value < 0.005:  # tolerance 0.5%
                continue

            if diff_value > 0:
                self.buy(data=data, size=diff_value / data.close[0])
            else:
                self.sell(data=data, size=abs(diff_value) / data.close[0])

    def _close_all_positions(self) -> None:
        for data in self.datas:
            if self.getposition(data).size != 0:
                self.close(data=data)

    # ── Data access helpers ──────────────────────────────────

    def _get_feature_slice(self, current_date: date) -> pd.DataFrame:
        """Extract the feature row(s) for the current date."""
        if self._features is None or self._features.empty:
            return pd.DataFrame()

        features = self._features
        # Features indexed by (ticker, trade_date) multi-index
        if isinstance(features.index, pd.MultiIndex):
            try:
                return features.xs(current_date, level=1)
            except KeyError:
                # Try nearest prior date
                available_dates = sorted(
                    set(features.index.get_level_values(1))
                )
                prior = [d for d in available_dates if d <= current_date]
                if not prior:
                    return pd.DataFrame()
                return features.xs(prior[-1], level=1)

        # If features is indexed by ticker only (single-date slice)
        return features

    def _get_sentiment_slice(self, current_date: date) -> pd.DataFrame:
        """Extract sentiment data for the current date."""
        if self._sentiment is None or self._sentiment.empty:
            return pd.DataFrame()

        sentiment = self._sentiment
        if "event_date" in sentiment.columns:
            return sentiment[sentiment["event_date"] <= current_date]
        return sentiment

    # ── Lifecycle ────────────────────────────────────────────

    def notify_order(self, order: bt.Order) -> None:
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                logger.debug("BUY %s @ %.2f size=%d", order.data._name, order.executed.price, order.executed.size)
            else:
                logger.debug("SELL %s @ %.2f size=%d", order.data._name, order.executed.price, order.executed.size)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning("Order %s: %s", order.data._name, order.getstatusname())

    def stop(self) -> None:
        """Called at end of backtest — log final values."""
        final_value = self.broker.getvalue()
        initial_value = self.broker.startingcash
        total_return = (final_value / initial_value) - 1
        logger.info("Backtest complete. Final value: %.2f, Return: %.2f%%", final_value, total_return * 100)
