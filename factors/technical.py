"""Technical factor computation — pure pandas/numpy, no pandas_ta dependency.

Computes momentum, volatility, and liquidity factors from OHLCV data.
All computations are ticker-aware and avoid look-ahead bias.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalFactorBuilder:
    """Computes momentum, volatility, and liquidity factors from OHLCV data."""

    MOMENTUM_WINDOWS: tuple[int, ...] = (5, 10, 21, 63)
    VOLATILITY_WINDOW: int = 21
    REGIME_WINDOW: int = 63
    VOLUME_MA_WINDOW: int = 20

    def build_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute full suite of technical factors.

        Returns long-format DataFrame: ticker, trade_date, factor_name, value
        """
        if df.empty:
            return df

        results: list[pd.DataFrame] = []
        for ticker, group in df.groupby("ticker"):
            prices = group.sort_values("trade_date").set_index("trade_date")
            factors = self._compute_factors_for_ticker(prices)
            factors["ticker"] = ticker
            results.append(factors.reset_index())

        if not results:
            return pd.DataFrame()

        combined = pd.concat(results, ignore_index=True)
        id_vars = ["ticker", "trade_date"]
        factor_cols = [c for c in combined.columns if c not in id_vars]
        long = combined.melt(
            id_vars=id_vars, value_vars=factor_cols,
            var_name="factor_name", value_name="value",
        ).dropna(subset=["value"])
        return long

    def _compute_factors_for_ticker(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical factors for a single ticker's time series."""
        c, h, l, v = prices["close"], prices["high"], prices["low"], prices["volume"]
        factors = pd.DataFrame(index=prices.index)

        # ── Momentum factors ─────────────────────────────────
        for w in self.MOMENTUM_WINDOWS:
            factors[f"roc_{w}d"] = c.pct_change(w) * 100
            factors[f"rsi_{w}d"] = self._rsi(c, w)
            factors[f"mom_{w}d"] = c - c.shift(w)

        # MACD
        macd_line, signal_line, histogram = self._macd(c)
        factors["macd_macd"] = macd_line
        factors["macd_signal"] = signal_line
        factors["macd_histogram"] = histogram

        # ── Volatility factors ────────────────────────────────
        w = self.VOLATILITY_WINDOW
        factors[f"atr_{w}d"] = self._atr(h, l, c, w)
        factors[f"bb_upper_{w}d"], factors[f"bb_lower_{w}d"], factors[f"bb_pct_b_{w}d"] = self._bbands(c, w)
        factors[f"hist_vol_{w}d"] = c.pct_change().rolling(w).std() * np.sqrt(252)

        # ── Liquidity / volume factors ────────────────────────
        factors[f"volume_ma_ratio_{self.VOLUME_MA_WINDOW}d"] = v / v.rolling(self.VOLUME_MA_WINDOW).mean()
        factors["turnover_proxy"] = v * c

        # ── Trend / regime factors ────────────────────────────
        factors[f"sma_ratio_{self.REGIME_WINDOW}d"] = c / c.rolling(self.REGIME_WINDOW).mean()
        rolling_max = c.rolling(self.REGIME_WINDOW).max()
        factors[f"max_dd_{self.REGIME_WINDOW}d"] = (rolling_max - c) / rolling_max

        return factors.dropna(how="all")

    # ── Indicator implementations ─────────────────────────────

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = close.ewm(span=fast, min_periods=fast).mean()
        ema_slow = close.ewm(span=slow, min_periods=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, min_periods=period).mean()

    @staticmethod
    def _bbands(close: pd.Series, period: int = 21, std_dev: float = 2.0):
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        pct_b = (close - lower) / (upper - lower)
        return upper, lower, pct_b
