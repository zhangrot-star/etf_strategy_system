"""Pydantic schemas for recommendation API responses."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class RecommendedETF(BaseModel):
    ticker: str
    name: str = ""
    total_score: float = Field(ge=0, le=115)
    rating: str = Field(default="C")
    ml_signal: str = Field(default="HOLD")
    recommendation: str = Field(default="HOLD")
    allocation_weight: float = Field(ge=0, le=1)
    risk_level: str = Field(default="MEDIUM")
    rationale: str = ""
    module_scores: dict[str, float] = Field(default_factory=dict)


class DailyRecommendation(BaseModel):
    date: date
    market: str = Field(default="A")
    total_universe: int = 0
    ranked_etfs: list[RecommendedETF] = Field(default_factory=list)
    risk_status: str = Field(default="NORMAL")
    cash_weight: float = Field(default=0.0)
    generated_at: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoreRequest(BaseModel):
    tickers: list[str] | None = None
    date: str | None = None
