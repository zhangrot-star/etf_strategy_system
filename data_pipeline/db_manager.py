"""Database session management and CRUD operations via SQLAlchemy 2.0."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import date
from typing import Generator

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.orm import Session, sessionmaker

from config.settings import Settings
from data_pipeline.models import (
    Base, DailyScore, ETFIssuer, ETFPrice, ETFProfile, FactorValue,
    IndexMeta, MacroIndicator, SentimentRecord, ETFPrediction,
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._engine = create_engine(
            self.settings.mysql_url,
            pool_size=self.settings.mysql_pool_size,
            max_overflow=self.settings.mysql_pool_overflow,
            pool_pre_ping=True,
            echo=False,
        )
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)

    def create_all(self) -> None:
        Base.metadata.create_all(self._engine)
        logger.info("All tables created / verified.")

    def drop_all(self) -> None:
        Base.metadata.drop_all(self._engine)
        logger.warning("All tables dropped.")

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        sess = self._session_factory()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()

    # ── ETF Price ─────────────────────────────────────────────
    def upsert_prices(self, df: pd.DataFrame) -> int:
        with self.session() as sess:
            records = df.to_dict("records")
            stmt = mysql_insert(ETFPrice).values(records)
            stmt = stmt.on_duplicate_key_update(
                open=stmt.inserted.open, high=stmt.inserted.high,
                low=stmt.inserted.low, close=stmt.inserted.close,
                volume=stmt.inserted.volume,
            )
            result = sess.execute(stmt)
            logger.info("Upserted %d price rows.", len(records))
            return result.rowcount

    def load_prices(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        with self.session() as sess:
            stmt = (
                select(ETFPrice)
                .where(ETFPrice.ticker.in_(tickers), ETFPrice.trade_date >= start, ETFPrice.trade_date <= end)
                .order_by(ETFPrice.ticker, ETFPrice.trade_date)
            )
            rows = sess.execute(stmt).scalars().all()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([{
            "ticker": r.ticker, "trade_date": r.trade_date, "open": r.open,
            "high": r.high, "low": r.low, "close": r.close, "volume": r.volume,
        } for r in rows])

    # ── Macro Indicators ──────────────────────────────────────
    def upsert_macro(self, df: pd.DataFrame) -> int:
        with self.session() as sess:
            records = df.to_dict("records")
            stmt = mysql_insert(MacroIndicator).values(records)
            stmt = stmt.on_duplicate_key_update(value=stmt.inserted.value)
            result = sess.execute(stmt)
            return result.rowcount

    def load_macro(self, indicators: list[str], start: date, end: date) -> pd.DataFrame:
        with self.session() as sess:
            stmt = (
                select(MacroIndicator)
                .where(MacroIndicator.indicator_name.in_(indicators),
                       MacroIndicator.obs_date >= start, MacroIndicator.obs_date <= end)
                .order_by(MacroIndicator.indicator_name, MacroIndicator.obs_date)
            )
            rows = sess.execute(stmt).scalars().all()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([{
            "indicator_name": r.indicator_name, "obs_date": r.obs_date, "value": r.value,
        } for r in rows])

    # ── Factor Values ─────────────────────────────────────────
    def upsert_factors(self, df: pd.DataFrame) -> int:
        with self.session() as sess:
            records = df.to_dict("records")
            stmt = mysql_insert(FactorValue).values(records)
            stmt = stmt.on_duplicate_key_update(value=stmt.inserted.value)
            result = sess.execute(stmt)
            return result.rowcount

    def load_factors(self, tickers: list[str], factor_names: list[str], start: date, end: date) -> pd.DataFrame:
        with self.session() as sess:
            stmt = (
                select(FactorValue)
                .where(FactorValue.ticker.in_(tickers), FactorValue.factor_name.in_(factor_names),
                       FactorValue.calc_date >= start, FactorValue.calc_date <= end)
                .order_by(FactorValue.ticker, FactorValue.factor_name, FactorValue.calc_date)
            )
            rows = sess.execute(stmt).scalars().all()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([{
            "ticker": r.ticker, "factor_name": r.factor_name, "calc_date": r.calc_date, "value": r.value,
        } for r in rows])

    # ── Sentiment Records ─────────────────────────────────────
    def upsert_sentiment(self, df: pd.DataFrame) -> int:
        with self.session() as sess:
            records = df.to_dict("records")
            stmt = mysql_insert(SentimentRecord).values(records)
            stmt = stmt.on_duplicate_key_update(
                polarity=stmt.inserted.polarity, confidence=stmt.inserted.confidence,
                summary=stmt.inserted.summary, raw_response=stmt.inserted.raw_response,
            )
            result = sess.execute(stmt)
            return result.rowcount

    def load_sentiment(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        with self.session() as sess:
            stmt = (
                select(SentimentRecord)
                .where(SentimentRecord.ticker.in_(tickers),
                       SentimentRecord.event_date >= start, SentimentRecord.event_date <= end)
                .order_by(SentimentRecord.ticker, SentimentRecord.event_date)
            )
            rows = sess.execute(stmt).scalars().all()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([{
            "ticker": r.ticker, "event_date": r.event_date, "polarity": r.polarity,
            "confidence": r.confidence, "event_category": r.event_category, "summary": r.summary,
        } for r in rows])

    # ── ETF Metadata ───────────────────────────────────────────

    def load_issuers(self) -> pd.DataFrame:
        with self.session() as sess:
            rows = sess.execute(select(ETFIssuer)).scalars().all()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([{
            "issuer_id": r.issuer_id, "name": r.name, "aum_rank": r.aum_rank,
            "roe": r.roe, "industry_median_roe": r.industry_median_roe,
        } for r in rows])

    def load_profiles(self, tickers: list[str] | None = None) -> pd.DataFrame:
        with self.session() as sess:
            stmt = select(ETFProfile)
            if tickers:
                stmt = stmt.where(ETFProfile.ticker.in_(tickers))
            rows = sess.execute(stmt).scalars().all()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([{
            "ticker": r.ticker, "name": r.name, "issuer_id": r.issuer_id,
            "market": r.market, "inception_date": r.inception_date,
            "expense_ratio": r.expense_ratio, "aum": r.aum,
            "avg_daily_volume": r.avg_daily_volume,
        } for r in rows])

    def load_index_meta(self, tickers: list[str] | None = None) -> pd.DataFrame:
        with self.session() as sess:
            stmt = select(IndexMeta)
            if tickers:
                stmt = stmt.where(IndexMeta.ticker.in_(tickers))
            rows = sess.execute(stmt).scalars().all()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([{
            "ticker": r.ticker, "index_code": r.index_code,
            "tracking_error": r.tracking_error, "is_public": r.is_public,
            "n_constituents": r.n_constituents,
            "has_transparent_rebal": r.has_transparent_rebal,
            "rebal_quarterly": r.rebal_quarterly,
            "dividend_yield": r.dividend_yield,
            "category_div_yield_median": r.category_div_yield_median,
            "premium_discount_std": r.premium_discount_std,
        } for r in rows])

    # ── ETF Predictions ───────────────────────────────────────

    def upsert_predictions(self, records: list[dict]) -> int:
        with self.session() as sess:
            stmt = mysql_insert(ETFPrediction).values(records)
            stmt = stmt.on_duplicate_key_update(
                predicted_return=stmt.inserted.predicted_return,
                prob_up=stmt.inserted.prob_up,
                model_version=stmt.inserted.model_version,
            )
            result = sess.execute(stmt)
            return result.rowcount

    def load_predictions(
        self, tickers: list[str] | None = None,
        pred_date: date | None = None, limit: int = 500,
    ) -> pd.DataFrame:
        with self.session() as sess:
            stmt = select(ETFPrediction).order_by(ETFPrediction.pred_date.desc())
            if tickers:
                stmt = stmt.where(ETFPrediction.ticker.in_(tickers))
            if pred_date:
                stmt = stmt.where(ETFPrediction.pred_date == pred_date)
            rows = sess.execute(stmt.limit(limit)).scalars().all()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([{
            "ticker": r.ticker, "pred_date": r.pred_date,
            "horizon_days": r.horizon_days, "predicted_return": r.predicted_return,
            "prob_up": r.prob_up, "target_return": r.target_return,
            "realized": r.realized, "model_version": r.model_version,
        } for r in rows])

    def update_realized_returns(self) -> int:
        """Backfill target_return for predictions whose horizon has elapsed.

        For each unrealized prediction, looks up the closing price at
        pred_date and at pred_date + horizon_days trading days, computes
        the actual return, and marks the prediction as realized.

        Returns count of updated predictions.
        """
        from datetime import date as dt_date, timedelta

        with self.session() as sess:
            pending = sess.query(ETFPrediction).filter(
                ETFPrediction.realized == False
            ).all()

            if not pending:
                return 0

            # collect all (ticker, date) pairs we need
            price_queries: set[tuple[str, dt_date]] = set()
            pred_map: dict[tuple[str, dt_date, int], ETFPrediction] = {}
            for pred in pending:
                price_queries.add((pred.ticker, pred.pred_date))
                pred_map[(pred.ticker, pred.pred_date, pred.horizon_days)] = pred

            # batch-load needed prices
            tickers_needed = {t for t, _ in price_queries}
            all_dates = sorted({d for _, d in price_queries})
            if not all_dates:
                return 0

            all_prices = sess.query(ETFPrice).filter(
                ETFPrice.ticker.in_(tickers_needed),
                ETFPrice.trade_date >= all_dates[0],
            ).order_by(ETFPrice.ticker, ETFPrice.trade_date).all()

            # index: (ticker, trade_date) → close
            price_idx: dict[tuple[str, dt_date], float] = {}
            ticker_dates: dict[str, list[dt_date]] = {}
            for p in all_prices:
                price_idx[(p.ticker, p.trade_date)] = p.close
                ticker_dates.setdefault(p.ticker, []).append(p.trade_date)

            count = 0
            today = dt_date.today()
            for (ticker, pred_d, horizon), pred in pred_map.items():
                target_date = pred_d + timedelta(days=int(horizon * 1.5))
                if today < target_date:
                    continue

                avail_dates = ticker_dates.get(ticker, [])
                start_close = price_idx.get((ticker, pred_d))
                if start_close is None or start_close <= 0:
                    continue

                # find the date closest to pred_d + horizon trading days
                future_dates = [d for d in avail_dates if d >= pred_d]
                if len(future_dates) < horizon + 1:
                    continue
                end_d = future_dates[min(horizon, len(future_dates) - 1)]
                end_close = price_idx.get((ticker, end_d))
                if end_close is None:
                    continue

                pred.target_return = float(end_close / start_close - 1)
                pred.realized = True
                count += 1

            # session() context manager auto-commits
            return count
