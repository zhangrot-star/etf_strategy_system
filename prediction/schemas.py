"""Pydantic schemas for prediction API endpoints."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class ETFReturnPrediction(BaseModel):
    ticker: str
    pred_date: date
    horizon_days: int
    predicted_return: float
    prob_up: float
    target_return: float | None = None
    realized: bool = False
    model_version: str = ""


class PredictionSummary(BaseModel):
    ticker: str
    pred_date: date
    horizons: dict[str, ETFReturnPrediction] = Field(default_factory=dict)
    consensus_direction: str = "NEUTRAL"  # BULLISH / BEARISH / NEUTRAL


class PredictionRunResult(BaseModel):
    date: date
    total_etfs: int
    total_predictions: int
    generated_at: str = ""
