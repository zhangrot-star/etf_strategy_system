"""Sentiment response parser — validates, normalizes, and converts to feature vectors."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Valid event categories from the prompt taxonomy
VALID_CATEGORIES: frozenset = frozenset(
    {
        "monetary_policy",
        "fiscal_policy",
        "earnings",
        "macro_data",
        "geopolitical",
        "sector_rotation",
        "technical_signal",
        "market_sentiment",
        "other",
    }
)


@dataclass
class ParsedSentiment:
    """Normalized sentiment record after validation."""

    ticker: str
    event_date: date
    polarity: float
    confidence: float
    event_category: str
    key_entities: list[str] = field(default_factory=list)
    summary: str = ""
    raw_response: str = ""

    @property
    def is_valid(self) -> bool:
        return self.confidence > 0.0 and self.event_category in VALID_CATEGORIES


class SentimentParser:
    """Validates and transforms raw Claude JSON into structured feature vectors.

    Two output formats are supported:
    1. ParsedSentiment dataclass (for database storage)
    2. Feature vector DataFrame (one-hot event_category + polarity + confidence)
    """

    # ── Validation ───────────────────────────────────────────

    def parse(self, raw: dict[str, Any], ticker: str, event_date: date) -> ParsedSentiment:
        """Validate and normalize a single Claude response."""
        polarity = self._clamp_float(raw.get("polarity", 0.0), -1.0, 1.0)
        confidence = self._clamp_float(raw.get("confidence", 0.0), 0.0, 1.0)
        category = raw.get("event_category", "other")
        if category not in VALID_CATEGORIES:
            category = "other"

        return ParsedSentiment(
            ticker=ticker,
            event_date=event_date,
            polarity=polarity,
            confidence=confidence,
            event_category=category,
            key_entities=raw.get("key_entities", []),
            summary=str(raw.get("summary", ""))[:200],
            raw_response=str(raw),
        )

    # ── Batch processing ──────────────────────────────────────

    def parse_batch(self, responses: list[dict[str, Any]]) -> list[ParsedSentiment]:
        """Parse a batch of Claude responses into ParsedSentiment records."""
        parsed: list[ParsedSentiment] = []
        for item in responses:
            meta = item.get("_input", {})
            ticker = meta.get("ticker", "")
            event_date = meta.get("date", date.today())
            if isinstance(event_date, str):
                event_date = date.fromisoformat(event_date)
            parsed.append(self.parse(item, ticker, event_date))
        return parsed

    # ── Feature vector construction ──────────────────────────

    def to_dataframe(self, parsed_records: list[ParsedSentiment]) -> pd.DataFrame:
        """Convert parsed sentiment records to a feature DataFrame.

        Produces columns: ticker, event_date, polarity, confidence,
        cat_{monetary_policy}, cat_{fiscal_policy}, ... (one-hot encoded categories).
        """
        if not parsed_records:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for rec in parsed_records:
            row: dict[str, Any] = {
                "ticker": rec.ticker,
                "event_date": rec.event_date,
                "polarity": rec.polarity,
                "confidence": rec.confidence,
            }
            # One-hot encode event categories
            for cat in VALID_CATEGORIES:
                row[f"cat_{cat}"] = 1.0 if rec.event_category == cat else 0.0
            rows.append(row)

        return pd.DataFrame(rows)

    def to_sentiment_records(self, parsed_records: list[ParsedSentiment]) -> pd.DataFrame:
        """Produce a DataFrame suitable for bulk upsert into SentimentRecord table.

        Columns: ticker, event_date, polarity, confidence, event_category, summary, raw_response
        """
        if not parsed_records:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "ticker": r.ticker,
                    "event_date": r.event_date,
                    "polarity": r.polarity,
                    "confidence": r.confidence,
                    "event_category": r.event_category,
                    "summary": r.summary,
                    "raw_response": r.raw_response,
                }
                for r in parsed_records
            ]
        )

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _clamp_float(value: Any, lo: float, hi: float) -> float:
        try:
            v = float(value)
            return max(lo, min(hi, v))
        except (TypeError, ValueError):
            return 0.0
