"""Claude API client wrapper with retry, rate-limiting, and structured output."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import numpy as np
from anthropic import Anthropic, RateLimitError, APIError, APITimeoutError

from config.settings import Settings
from sentiment.prompts import SYSTEM_PROMPT_FINANCIAL_SENTIMENT

logger = logging.getLogger(__name__)


class ClaudeSentimentClient:
    """Wraps the Anthropic SDK for financial sentiment extraction.

    Features:
    - Exponential backoff (1s, 2s, 4s) for transient errors
    - Token budget control via max_tokens
    - Structured JSON output enforcement
    """

    def __init__(self, settings: Settings | None = None, model: str | None = None) -> None:
        self._settings = settings or Settings()
        api_key = self._settings.anthropic_api_key or self._settings.anthropic_auth_token
        kwargs: dict = {"api_key": api_key}
        if self._settings.anthropic_base_url:
            kwargs["base_url"] = self._settings.anthropic_base_url
        self._client = Anthropic(**kwargs)
        self._model = model or self._settings.anthropic_model
        self._max_retries = self._settings.llm_max_retries
        self._timeout = self._settings.llm_request_timeout

    # ── Public API ───────────────────────────────────────────

    def analyze_news(
        self,
        text: str,
        ticker: str,
        extra_context: str = "",
    ) -> dict[str, Any]:
        """Analyze a single news article for a given ETF ticker.

        Returns a dict with polarity, confidence, event_category, key_entities, summary.
        """
        user_message = f"Ticker: {ticker}\n\nText:\n{text}"
        if extra_context:
            user_message = f"{extra_context}\n\n{user_message}"

        raw = self._call_claude(SYSTEM_PROMPT_FINANCIAL_SENTIMENT, user_message)
        return self._parse_response(raw)

    def analyze_batch(
        self,
        items: list[dict[str, Any]],
        concurrency_sleep: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Analyze multiple news items sequentially with rate-limiting.

        Each item dict should have: 'text', 'ticker', and optionally 'context'.
        """
        results: list[dict[str, Any]] = []
        for item in items:
            result = self.analyze_news(
                text=item["text"],
                ticker=item["ticker"],
                extra_context=item.get("context", ""),
            )
            result["_input"] = item
            results.append(result)
            if concurrency_sleep > 0 and len(items) > 1:
                time.sleep(concurrency_sleep)
        return results

    def generate_commentary(self, context: str) -> str:
        """Generate free-form strategy commentary — no JSON parsing needed.

        Uses higher max_tokens (1024) for rich analysis output.
        """
        system_prompt = (
            "你是一位资深量化策略分析师。请基于提供的回测数据，撰写一段200字以内的"
            "专业策略评价，包括整体评价、主要亮点、风险点和优化建议。直接返回评论文本，"
            "不需要JSON格式。"
        )
        return self._call_claude(
            system_prompt=system_prompt,
            user_message=context,
            max_tokens=1024,
        )

    # ── Internal ─────────────────────────────────────────────

    def _call_claude(
        self, system_prompt: str, user_message: str, max_tokens: int = 2048
    ) -> str:
        """Send a message to Claude with retry logic."""
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    temperature=0.1,  # low temperature for consistent structured output
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )
                # Extract text from the last TextBlock (skip ThinkingBlock in DeepSeek)
                text_blocks = [
                    b.text for b in response.content
                    if hasattr(b, "text") and getattr(b, "text", "")
                ]
                content = text_blocks[-1] if text_blocks else ""
                if not content:
                    raise ValueError(f"No text block in response: {response.content}")
                logger.debug("Claude response received (attempt %d).", attempt + 1)
                return content

            except RateLimitError as e:
                wait = 2 ** attempt
                logger.warning("Rate limited — retrying in %ds (attempt %d/%d).", wait, attempt + 1, self._max_retries)
                last_error = e
                if attempt < self._max_retries:
                    time.sleep(wait)

            except APITimeoutError as e:
                logger.warning("Timeout — retrying (attempt %d/%d).", attempt + 1, self._max_retries)
                last_error = e
                if attempt < self._max_retries:
                    time.sleep(1)

            except APIError as e:
                logger.error("API error: %s", e)
                last_error = e
                if attempt < self._max_retries:
                    time.sleep(2 ** attempt)

        raise RuntimeError(
            f"Claude API call failed after {self._max_retries + 1} attempts"
        ) from last_error

    @staticmethod
    def _parse_response(raw: str) -> dict[str, Any]:
        """Extract JSON from response — handles markdown fences and non-numeric polarity."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse Claude JSON response: %s", raw[:200])
            return {
                "polarity": 0.0,
                "confidence": 0.0,
                "event_category": "other",
                "key_entities": [],
                "summary": "Parse error",
                "_raw": raw,
            }

        # Normalize polarity: DeepSeek sometimes outputs strings like "positive"/"negative"
        pol = data.get("polarity", 0.0)
        if isinstance(pol, str):
            pol_str = pol.lower().strip()
            if pol_str in ("positive", "bullish", "up"):
                pol = 0.6
            elif pol_str in ("negative", "bearish", "down"):
                pol = -0.6
            elif pol_str in ("neutral", "neutral/mixed", "neutral/slightly bullish"):
                pol = 0.1
            else:
                try:
                    pol = float(pol_str)
                except (ValueError, TypeError):
                    pol = 0.0
        data["polarity"] = float(np.clip(pol, -1.0, 1.0))

        # Normalize confidence
        conf = data.get("confidence", 0.5)
        if isinstance(conf, str):
            try:
                conf = float(conf)
            except (ValueError, TypeError):
                conf = 0.5
        data["confidence"] = float(np.clip(conf, 0.0, 1.0))

        # Normalize event_category
        valid_cats = {
            "monetary_policy", "fiscal_policy", "earnings", "macro_data",
            "geopolitical", "sector_rotation", "technical_signal",
            "market_sentiment", "other",
        }
        cat = data.get("event_category", "other")
        if isinstance(cat, str) and cat.lower() not in valid_cats:
            # Map common DeepSeek outputs
            cat_map = {
                "stock_movement": "technical_signal",
                "positive": "market_sentiment",
                "negative": "market_sentiment",
                "neutral": "other",
            }
            data["event_category"] = cat_map.get(cat.lower(), "other")
        elif not isinstance(cat, str):
            data["event_category"] = "other"

        return data
