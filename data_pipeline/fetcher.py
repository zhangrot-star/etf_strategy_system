"""Multi-market data fetching via Factory pattern.

Supports A-share (akshare) and US (yfinance) ETF data sources.
The Factory selects the correct fetcher based on config.yaml `market` key.

A-share uses a Provider pattern with automatic fallback:
  Source 1 → Sina  (ak.fund_etf_hist_sina)  — direct HTTP, no proxy needed
  Source 2 → EastMoney (ak.fund_etf_hist_em) — proxy-dependent backup
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)


# ── Abstract base (fetcher) ──────────────────────────────────────────


class BaseFetcher(ABC):
    """Interface for market-specific data fetchers."""

    @abstractmethod
    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        """Fetch daily OHLCV for a list of tickers.

        Returns columns: ticker, trade_date, open, high, low, close, volume
        """
        ...

    @abstractmethod
    def fetch_macro(self, start: date, end: date) -> pd.DataFrame:
        """Fetch macro indicators for the market.

        Returns columns: indicator_name, obs_date, value
        """
        ...

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic cleaning: drop rows with missing OHLC, sort, dedup."""
        if df.empty:
            return df
        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        df = df.dropna(subset=required)
        df = df.sort_values(["ticker", "trade_date"]).reset_index(drop=True)
        df = df.drop_duplicates(subset=["ticker", "trade_date"], keep="last")
        return df


# ── Abstract base (provider) ─────────────────────────────────────────


class ETFDataProvider(ABC):
    """A single data source that can fetch ETF daily kline data."""

    @abstractmethod
    def fetch_one(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Fetch OHLCV for one ticker; return standardised columns."""
        ...

    @staticmethod
    def _standardize(df: pd.DataFrame, ticker: str,
                     col_map: dict[str, str]) -> pd.DataFrame:
        """Rename columns to canonical names and keep only needed columns."""
        df = df.rename(columns=col_map)
        df["ticker"] = ticker
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
        keep = ["ticker", "trade_date", "open", "high", "low", "close", "volume"]
        return df[[c for c in keep if c in df.columns]]


# ── Provider: Sina ───────────────────────────────────────────────────


class SinaETFProvider(ETFDataProvider):
    """Fetches ETF daily kline from Sina Finance via akshare.

    Uses ``ak.fund_etf_hist_sina()`` which fetches from Sina's HTTP API
    without requiring a proxy — it works with direct connections.
    """

    @staticmethod
    def _to_sina_symbol(ticker: str) -> str:
        """Map plain ticker to Sina prefixed symbol.

        Shanghai: 5xxxxx, 6xxxxx, 9xxxxx  → sh{ticker}
        Shenzhen: 0xxxxx, 1xxxxx, 3xxxxx  → sz{ticker}
        """
        if ticker.startswith(("5", "6", "9")):
            return f"sh{ticker}"
        return f"sz{ticker}"

    def fetch_one(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        import akshare as ak

        symbol = self._to_sina_symbol(ticker)
        raw = ak.fund_etf_hist_sina(symbol=symbol)
        if raw is None or raw.empty:
            return pd.DataFrame()

        col_map = {
            "date": "trade_date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
        df = self._standardize(raw, ticker, col_map)
        # Sina reports volume in shares (股); convert to lots (手 = 100 shares)
        df["volume"] = df["volume"] // 100
        # Filter to requested date range
        df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
        return df


# ── Provider: EastMoney ──────────────────────────────────────────────


class EastMoneyETFProvider(ETFDataProvider):
    """Fetches ETF daily kline from EastMoney via akshare.

    Uses ``ak.fund_etf_hist_em()``.  Requires a working proxy with
    China-routed exit node to reach push2his.eastmoney.com.
    """

    def fetch_one(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        import akshare as ak

        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        raw = ak.fund_etf_hist_em(
            symbol=ticker,
            period="daily",
            start_date=start_str,
            end_date=end_str,
            adjust="qfq",
        )
        if raw is None or raw.empty:
            return pd.DataFrame()

        col_map = {
            "日期": "trade_date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
        }
        return self._standardize(raw, ticker, col_map)


# ── A-share fetcher ──────────────────────────────────────────────────


class AShareETFDataFetcher(BaseFetcher):
    """Fetches A-share ETF daily OHLCV and macro indicators.

    Data providers are tried in order; the first successful provider
    for each ticker wins.  Sina is tried first (direct connection, no
    proxy needed), then EastMoney as backup.
    """

    _PROVIDERS: list[type[ETFDataProvider]] = [
        SinaETFProvider,
        EastMoneyETFProvider,
    ]

    def __init__(self, request_delay: float = 0.3, timeout: int = 30) -> None:
        self._delay = request_delay
        self._timeout = timeout
        # Instantiate providers once
        self._providers = [cls() for cls in self._PROVIDERS]

    def _fetch_one(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Try each provider in order; return first non-empty result."""
        for provider in self._providers:
            try:
                df = provider.fetch_one(ticker, start, end)
                if not df.empty:
                    return df
            except Exception:
                logger.debug(
                    "Provider %s failed for %s",
                    type(provider).__name__, ticker, exc_info=True,
                )
        return pd.DataFrame()

    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()

        results: list[pd.DataFrame] = []
        for i, t in enumerate(tickers):
            if i > 0:
                time.sleep(self._delay)
            try:
                df = self._fetch_one(t, start, end)
                if df.empty:
                    logger.warning("No data for %s (all providers exhausted)", t)
                else:
                    results.append(df)
            except Exception:
                logger.exception("Failed to fetch %s", t)

        if not results:
            return pd.DataFrame()
        combined = pd.concat(results, ignore_index=True)
        logger.info(
            "A-share: fetched %d rows across %d tickers.",
            len(combined), len(results),
        )
        return combined

    # ── Macro indicators ─────────────────────────────────────────────

    def fetch_macro(self, start: date, end: date) -> pd.DataFrame:
        import akshare as ak

        results: list[pd.DataFrame] = []

        macro_fetchers = {
            "PMI": lambda: self._fetch_pmi(ak),
            "CPI_YoY": lambda: self._fetch_cpi(ak),
            "M2": lambda: self._fetch_m2(ak),
            "SHIBOR_ON": lambda: self._fetch_shibor(ak),
        }

        for name, fetcher_fn in macro_fetchers.items():
            try:
                df = fetcher_fn()
                if df is not None and not df.empty:
                    results.append(df)
            except Exception:
                logger.warning("Failed to fetch macro: %s", name)

        if not results:
            return pd.DataFrame()
        combined = pd.concat(results, ignore_index=True)
        if start:
            combined = combined[combined["obs_date"] >= start]
        if end:
            combined = combined[combined["obs_date"] <= end]
        logger.info("A-share macro: fetched %d rows.", len(combined))
        return combined

    @staticmethod
    def _fetch_pmi(ak) -> pd.DataFrame | None:
        raw = ak.macro_china_pmi()
        if raw is None or raw.empty:
            return None
        raw = raw.rename(columns={"日期": "obs_date", "制造业": "value"})
        raw["indicator_name"] = "PMI"
        raw["obs_date"] = pd.to_datetime(raw["obs_date"]).dt.date
        raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
        return raw[["indicator_name", "obs_date", "value"]].dropna()

    @staticmethod
    def _fetch_cpi(ak) -> pd.DataFrame | None:
        raw = ak.macro_china_cpi_yearly()
        if raw is None or raw.empty:
            return None
        raw = raw.rename(columns={"日期": "obs_date", "居民消费价格指数(上年同月=100)": "value"})
        raw["indicator_name"] = "CPI_YoY"
        raw["obs_date"] = pd.to_datetime(raw["obs_date"]).dt.date
        raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
        return raw[["indicator_name", "obs_date", "value"]].dropna()

    @staticmethod
    def _fetch_m2(ak) -> pd.DataFrame | None:
        raw = ak.macro_china_money_supply()
        if raw is None or raw.empty:
            return None
        raw = raw.rename(columns={"月份": "obs_date", "货币和准货币(M2)数量(亿元)": "value"})
        raw["indicator_name"] = "M2"
        raw["obs_date"] = pd.to_datetime(raw["obs_date"].astype(str), errors="coerce").dt.date
        raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
        return raw[["indicator_name", "obs_date", "value"]].dropna()

    @staticmethod
    def _fetch_shibor(ak) -> pd.DataFrame | None:
        raw = ak.macro_china_shibor_all()
        if raw is None or raw.empty:
            return None
        raw = raw.rename(columns={"日期": "obs_date", "隔夜": "value"})
        raw["indicator_name"] = "SHIBOR_ON"
        raw["obs_date"] = pd.to_datetime(raw["obs_date"]).dt.date
        raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
        return raw[["indicator_name", "obs_date", "value"]].dropna()


# ── US fetcher (yfinance) ────────────────────────────────────────────


class USETFDataFetcher(BaseFetcher):
    """Fetches US ETF daily OHLCV and macro indicators via yfinance."""

    def __init__(self, max_workers: int = 8, request_timeout: int = 30) -> None:
        self._max_workers = max_workers
        self._timeout = request_timeout

    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()

        import yfinance as yf

        try:
            raw = yf.download(
                tickers=" ".join(tickers),
                start=start,
                end=end,
                group_by="ticker",
                threads=True,
                auto_adjust=True,
            )
        except Exception:
            logger.exception("yfinance download failed")
            return pd.DataFrame()

        if raw is None or raw.empty:
            logger.warning("yfinance returned no data for %s", tickers)
            return pd.DataFrame()

        results: list[pd.DataFrame] = []
        for t in tickers:
            try:
                if len(tickers) == 1:
                    t_data = raw.copy()
                elif t in raw.columns.levels[0] if hasattr(raw.columns, "levels") else False:
                    t_data = raw[t].copy()
                else:
                    continue

                t_data = t_data.reset_index()
                t_data.columns = [str(c).lower().replace(" ", "_") for c in t_data.columns]
                t_data["ticker"] = t

                col_map = {"date": "trade_date"}
                t_data = t_data.rename(columns=col_map)
                if "trade_date" in t_data.columns:
                    t_data["trade_date"] = pd.to_datetime(t_data["trade_date"]).dt.date

                keep = ["ticker", "trade_date", "open", "high", "low", "close", "volume"]
                results.append(t_data[[c for c in keep if c in t_data.columns]])
            except Exception:
                logger.exception("Failed to parse yfinance data for %s", t)

        if not results:
            return pd.DataFrame()
        combined = pd.concat(results, ignore_index=True)
        logger.info("US: fetched %d rows across %d tickers.", len(combined), len(results))
        return combined

    def fetch_macro(self, start: date, end: date) -> pd.DataFrame:
        import yfinance as yf

        macro_tickers = {
            "^VIX": "VIX",
            "^TNX": "US10Y_Yield",
            "DX-Y.NYB": "USD_Index",
        }
        results: list[pd.DataFrame] = []

        for symbol, name in macro_tickers.items():
            try:
                tkr = yf.Ticker(symbol)
                hist = tkr.history(start=start, end=end)
                if hist.empty:
                    continue
                hist = hist.reset_index()
                hist["obs_date"] = pd.to_datetime(hist["Date"]).dt.date
                hist["indicator_name"] = name
                hist["value"] = pd.to_numeric(hist["Close"], errors="coerce")
                results.append(hist[["indicator_name", "obs_date", "value"]].dropna())
            except Exception:
                logger.warning("Failed to fetch US macro: %s", symbol)

        if not results:
            return pd.DataFrame()
        combined = pd.concat(results, ignore_index=True)
        logger.info("US macro: fetched %d rows.", len(combined))
        return combined


# ── Factory ───────────────────────────────────────────────────────────


class DataFetcherFactory:
    """Factory that returns the correct BaseFetcher based on market code.

    Usage:
        fetcher = DataFetcherFactory.create("A")   # akshare
        fetcher = DataFetcherFactory.create("US")  # yfinance
        prices = fetcher.fetch(tickers, start, end)
        cleaned = fetcher.clean(prices)
    """

    _registry: dict[str, type[BaseFetcher]] = {
        "A": AShareETFDataFetcher,
        "US": USETFDataFetcher,
    }

    @classmethod
    def create(cls, market: str, **kwargs) -> BaseFetcher:
        """Create a fetcher for the given market.

        Args:
            market: "A" for A-share (akshare), "US" for US (yfinance).
            **kwargs: Passed to the fetcher constructor.
        """
        fetcher_cls = cls._registry.get(market.upper())
        if fetcher_cls is None:
            raise ValueError(
                f"Unknown market '{market}'. Supported: {list(cls._registry)}"
            )
        return fetcher_cls(**kwargs)

    @classmethod
    def register(cls, market: str, fetcher_cls: type[BaseFetcher]) -> None:
        """Register a custom fetcher for a market code."""
        cls._registry[market.upper()] = fetcher_cls
        logger.info("Registered fetcher for market '%s': %s", market, fetcher_cls.__name__)
