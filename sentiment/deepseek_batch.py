"""Offline LLM batch sentiment engine.

Decouples LLM calls from the hot backtest path.  Runs as a scheduled /
triggered job that:
1. Reads unprocessed news / research items from ChromaDB or a text queue
2. Batches calls through DeepSeek (via Anthropic SDK)
3. Stores structured sentiment vectors in MySQL + ChromaDB

This ensures the backtest loop never blocks on HTTP calls.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """A single news article or research snippet to be scored."""

    ticker: str
    text: str
    source: str = ""
    published_date: date | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Aggregate results from a batch sentiment run."""

    records: list[dict[str, Any]]
    total_processed: int
    total_failed: int
    elapsed_sec: float
    avg_polarity: float
    errors: list[str] = field(default_factory=list)


class DeepSeekBatchEngine:
    """Offline batch processor for DeepSeek-driven sentiment scoring.

    Pipeline:
      load_items → chunk → call_llm → parse → store
    """

    def __init__(
        self,
        client,                # ClaudeSentimentClient instance
        db_manager,            # DatabaseManager instance
        batch_size: int = 10,
        batch_delay_sec: float = 0.5,
        max_retries: int = 3,
    ) -> None:
        self._client = client
        self._db = db_manager
        self._batch_size = batch_size
        self._batch_delay = batch_delay_sec
        self._max_retries = max_retries

    # ── Main entry point ────────────────────────────────────────

    def run(self, items: list[NewsItem]) -> BatchResult:
        """Process a list of news items through the sentiment pipeline."""
        if not items:
            return BatchResult(
                records=[], total_processed=0, total_failed=0,
                elapsed_sec=0.0, avg_polarity=0.0,
            )

        start = time.perf_counter()
        all_records: list[dict[str, Any]] = []
        errors: list[str] = []
        failed = 0

        # Chunk into batches
        for i in range(0, len(items), self._batch_size):
            batch = items[i:i + self._batch_size]
            batch_inputs = [
                {
                    "text": item.text,
                    "ticker": item.ticker,
                    "context": f"source={item.source}",
                }
                for item in batch
            ]

            try:
                batch_results = self._client.analyze_batch(
                    batch_inputs,
                    concurrency_sleep=self._batch_delay,
                )
            except Exception:
                logger.exception("Batch %d-%d failed", i, i + len(batch))
                failed += len(batch)
                errors.append(f"Batch {i}-{i + len(batch)}: LLM call failed")
                continue

            for item, result in zip(batch, batch_results):
                try:
                    record = self._normalize(item, result)
                    all_records.append(record)
                except Exception:
                    logger.exception("Parse failed for %s", item.ticker)
                    failed += 1
                    # Store a neutral fallback
                    all_records.append(self._neutral_record(item))

            if self._batch_delay > 0 and i + self._batch_size < len(items):
                time.sleep(self._batch_delay)

        elapsed = time.perf_counter() - start
        avg_pol = float(np.mean([r["polarity"] for r in all_records])) if all_records else 0.0

        # Persist to DB
        if all_records:
            self._persist(all_records)

        logger.info(
            "Batch complete: %d processed, %d failed in %.1fs. Avg polarity=%.3f",
            len(all_records), failed, elapsed, avg_pol,
        )

        return BatchResult(
            records=all_records,
            total_processed=len(all_records),
            total_failed=failed,
            elapsed_sec=elapsed,
            avg_polarity=avg_pol,
            errors=errors,
        )

    # ── Item loading ────────────────────────────────────────────

    @staticmethod
    def load_from_dataframe(
        df: pd.DataFrame,
        text_col: str = "text",
        ticker_col: str = "ticker",
        source_col: str = "source",
        date_col: str | None = None,
    ) -> list[NewsItem]:
        """Build NewsItem list from a DataFrame of raw text."""
        items: list[NewsItem] = []
        for _, row in df.iterrows():
            pub_date = None
            if date_col and date_col in df.columns:
                try:
                    pub_date = pd.Timestamp(row[date_col]).date()
                except (ValueError, TypeError):
                    pass
            items.append(NewsItem(
                ticker=str(row[ticker_col]),
                text=str(row[text_col]),
                source=str(row.get(source_col, "")),
                published_date=pub_date,
            ))
        return items

    @staticmethod
    def load_from_jsonl(path: str | Path) -> list[NewsItem]:
        """Load news items from a JSONL file.

        Expected format per line:
        {"ticker": "588000", "text": "...", "source": "eastmoney", "date": "2026-01-15"}
        """
        items: list[NewsItem] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    pub_date = None
                    if "date" in obj:
                        try:
                            pub_date = date.fromisoformat(obj["date"])
                        except (ValueError, TypeError):
                            pass
                    items.append(NewsItem(
                        ticker=obj["ticker"],
                        text=obj["text"],
                        source=obj.get("source", ""),
                        published_date=pub_date,
                        metadata=obj.get("metadata", {}),
                    ))
                except (KeyError, json.JSONDecodeError):
                    logger.warning("Skipping malformed JSONL line: %s", line[:100])
        logger.info("Loaded %d items from %s", len(items), path)
        return items

    # ── Generation of synthetic news (for backtesting) ──────────

    @staticmethod
    def generate_synthetic_news(
        tickers: list[str],
        prices: pd.DataFrame,
        n_per_ticker: int = 12,
        seed: int = 42,
    ) -> list[NewsItem]:
        """Generate realistic synthetic news items from price momentum.

        Used for backtesting when no real news corpus is available.
        """
        rng = np.random.default_rng(seed)
        items: list[NewsItem] = []

        templates_positive = [
            "{ticker}获大额资金净流入，机构看好后市表现",
            "政策利好推动{ticker}相关板块走强，成交放量",
            "{ticker}估值处于历史低位，配置价值凸显",
            "北向资金持续加仓{ticker}，市场情绪回暖",
            "行业景气度回升，{ticker}龙头ETF受益明显",
        ]
        templates_negative = [
            "{ticker}遭遇大额净赎回，短期承压明显",
            "宏观不确定性增加，{ticker}避险情绪升温",
            "技术指标走弱，{ticker}面临调整压力",
            "板块轮动加速，{ticker}资金流出显著",
            "外围市场波动加剧，{ticker}联动下行风险上升",
        ]
        templates_neutral = [
            "{ticker}维持震荡格局，多空力量均衡",
            "市场观望情绪浓厚，{ticker}窄幅整理",
            "{ticker}成交量萎缩，等待方向选择",
        ]

        prices = prices.copy()
        if "trade_date" not in prices.columns:
            prices = prices.reset_index()

        for ticker in tickers:
            t_data = prices[prices["ticker"] == ticker].sort_values("trade_date")
            if t_data.empty:
                continue

            dates = t_data["trade_date"].unique()
            if len(dates) < n_per_ticker:
                sampled_dates = dates
            else:
                indices = np.linspace(0, len(dates) - 1, n_per_ticker, dtype=int)
                sampled_dates = dates[indices]

            for d in sampled_dates:
                # Determine sentiment direction from recent returns
                window = t_data[t_data["trade_date"] <= d].tail(10)
                if len(window) >= 5:
                    ret = (window["close"].iloc[-1] / window["close"].iloc[0]) - 1
                    if ret > 0.02:
                        tmpl = rng.choice(templates_positive)
                    elif ret < -0.02:
                        tmpl = rng.choice(templates_negative)
                    else:
                        tmpl = rng.choice(templates_neutral)
                else:
                    tmpl = rng.choice(templates_neutral)

                items.append(NewsItem(
                    ticker=ticker,
                    text=tmpl.format(ticker=ticker),
                    source="synthetic",
                    published_date=d if isinstance(d, date) else d.date() if hasattr(d, "date") else date.today(),
                ))

        logger.info("Generated %d synthetic news items for %d tickers.", len(items), len(tickers))
        return items

    # ── Internal ────────────────────────────────────────────────

    def _normalize(self, item: NewsItem, llm_result: dict[str, Any]) -> dict[str, Any]:
        """Normalize a single LLM result into a standard record."""
        return {
            "ticker": item.ticker,
            "event_date": item.published_date or date.today(),
            "polarity": float(np.clip(llm_result.get("polarity", 0.0), -1.0, 1.0)),
            "confidence": float(np.clip(llm_result.get("confidence", 0.5), 0.0, 1.0)),
            "event_category": str(llm_result.get("event_category", "other")),
            "summary": str(llm_result.get("summary", ""))[:500],
            "raw_response": json.dumps(llm_result, ensure_ascii=False),
            "source": item.source,
        }

    @staticmethod
    def _neutral_record(item: NewsItem) -> dict[str, Any]:
        return {
            "ticker": item.ticker,
            "event_date": item.published_date or date.today(),
            "polarity": 0.0,
            "confidence": 0.0,
            "event_category": "other",
            "summary": "LLM parse failed — neutral default",
            "raw_response": "{}",
            "source": item.source,
        }

    def _persist(self, records: list[dict[str, Any]]) -> None:
        """Store sentiment records in MySQL."""
        try:
            df = pd.DataFrame(records)
            keep_cols = [
                "ticker", "event_date", "polarity", "confidence",
                "event_category", "summary", "raw_response",
            ]
            df = df[[c for c in keep_cols if c in df.columns]]
            self._db.upsert_sentiment(df)
        except Exception:
            logger.exception("Failed to persist sentiment batch to DB")
