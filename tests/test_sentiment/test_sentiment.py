"""Tests for sentiment analysis module."""

from __future__ import annotations

from datetime import date

import pytest

from sentiment.parser import SentimentParser, VALID_CATEGORIES


class TestSentimentParser:
    def test_parse_valid_response(self, sample_sentiment_responses):
        parser = SentimentParser()
        result = parser.parse(sample_sentiment_responses[0], "SPY", date(2023, 6, 1))
        assert result.polarity == 0.7
        assert result.confidence == 0.85
        assert result.event_category == "monetary_policy"
        assert result.is_valid

    def test_parse_invalid_category_defaults_to_other(self):
        parser = SentimentParser()
        result = parser.parse(
            {"polarity": 0.5, "confidence": 0.7, "event_category": "invalid_cat"},
            "SPY", date.today(),
        )
        assert result.event_category == "other"

    def test_parse_clamps_polarity(self):
        parser = SentimentParser()
        result = parser.parse({"polarity": 2.5, "confidence": 0.5}, "SPY", date.today())
        assert result.polarity == 1.0

        result = parser.parse({"polarity": -3.0, "confidence": 0.5}, "SPY", date.today())
        assert result.polarity == -1.0

    def test_to_dataframe_produces_onehot_columns(self, sample_sentiment_responses):
        parser = SentimentParser()
        parsed = parser.parse_batch(sample_sentiment_responses)
        df = parser.to_dataframe(parsed)
        assert "polarity" in df.columns
        assert "confidence" in df.columns
        for cat in VALID_CATEGORIES:
            assert f"cat_{cat}" in df.columns

    def test_to_sentiment_records(self, sample_sentiment_responses):
        parser = SentimentParser()
        parsed = parser.parse_batch(sample_sentiment_responses)
        df = parser.to_sentiment_records(parsed)
        assert len(df) == 3
        assert "polarity" in df.columns
        assert "raw_response" in df.columns


class TestPrompts:
    def test_system_prompt_is_string(self):
        from sentiment.prompts import SYSTEM_PROMPT_FINANCIAL_SENTIMENT
        assert isinstance(SYSTEM_PROMPT_FINANCIAL_SENTIMENT, str)
        assert "JSON" in SYSTEM_PROMPT_FINANCIAL_SENTIMENT
        assert "polarity" in SYSTEM_PROMPT_FINANCIAL_SENTIMENT
