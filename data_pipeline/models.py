"""SQLAlchemy ORM models for structured financial data."""

from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import Date, DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ETFPrice(Base):
    __tablename__ = "etf_price"
    __table_args__ = (UniqueConstraint("ticker", "trade_date", name="uq_ticker_date"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    trade_date: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class MacroIndicator(Base):
    __tablename__ = "macro_indicator"
    __table_args__ = (UniqueConstraint("indicator_name", "obs_date", name="uq_macro_name_date"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    indicator_name: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    obs_date: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class FactorValue(Base):
    __tablename__ = "factor_value"
    __table_args__ = (UniqueConstraint("ticker", "factor_name", "calc_date", name="uq_factor_ticker_name_date"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    factor_name: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    calc_date: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class SentimentRecord(Base):
    __tablename__ = "sentiment_record"
    __table_args__ = (UniqueConstraint("ticker", "event_date", "event_category", name="uq_sentiment_ticker_date_cat"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    event_date: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    polarity: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    event_category: Mapped[str] = mapped_column(String(64), nullable=False)
    summary: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    raw_response: Mapped[str | None] = mapped_column(String(8192), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class ETFIssuer(Base):
    """ETF issuer / fund company data."""
    __tablename__ = "etf_issuer"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    issuer_id: Mapped[str] = mapped_column(String(20), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    aum_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    roe: Mapped[float | None] = mapped_column(Float, nullable=True)
    industry_median_roe: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class IndexMeta(Base):
    """Index methodology metadata."""
    __tablename__ = "index_meta"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), unique=True, index=True, nullable=False)
    index_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    tracking_error: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_public: Mapped[bool] = mapped_column(default=True)
    n_constituents: Mapped[int | None] = mapped_column(Integer, nullable=True)
    has_transparent_rebal: Mapped[bool] = mapped_column(default=True)
    rebal_quarterly: Mapped[bool] = mapped_column(default=True)
    dividend_yield: Mapped[float | None] = mapped_column(Float, nullable=True)
    category_div_yield_median: Mapped[float | None] = mapped_column(Float, nullable=True)
    premium_discount_std: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class ETFProfile(Base):
    """ETF profile / static data."""
    __tablename__ = "etf_profile"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), unique=True, index=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    issuer_id: Mapped[str | None] = mapped_column(String(20), nullable=True)
    market: Mapped[str] = mapped_column(String(4), default="A")
    inception_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    expense_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    aum: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_daily_volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class ETFPrediction(Base):
    """Multi-horizon return prediction record."""
    __tablename__ = "etf_prediction"
    __table_args__ = (UniqueConstraint("ticker", "pred_date", "horizon_days", name="uq_pred_ticker_date_horizon"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    pred_date: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_return: Mapped[float] = mapped_column(Float, nullable=False)
    prob_up: Mapped[float] = mapped_column(Float, nullable=False)
    target_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    realized: Mapped[bool] = mapped_column(default=False)
    model_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class DailyScore(Base):
    """Cached daily ETF scoring output."""
    __tablename__ = "daily_score"
    __table_args__ = (UniqueConstraint("ticker", "score_date", name="uq_score_ticker_date"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    score_date: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    total_score: Mapped[float] = mapped_column(Float, nullable=False)
    adjusted_total: Mapped[float] = mapped_column(Float, nullable=False)
    ml_signal: Mapped[str] = mapped_column(String(10), nullable=False)
    recommendation: Mapped[str] = mapped_column(String(20), nullable=False)
    module_scores_json: Mapped[str | None] = mapped_column(String(4096), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
