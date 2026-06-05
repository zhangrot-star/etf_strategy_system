"""FastAPI entry point — REST API for ETF quant strategy system.

Provides:
  GET  /health                     — Liveness check
  GET  /recommendation/daily       — Latest daily ETF recommendation
  GET  /recommendation/etf/{ticker} — Detail for a single ETF
  POST /recommendation/score       — Score ETFs on-demand
  POST /recommendation/run         — Trigger full recommendation pipeline
  POST /backtest/run               — Trigger backtest with config
  POST /data/update                — Trigger data pipeline refresh
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime
from typing import Any

import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("quant_api")

app = FastAPI(title="ETF Quant Strategy API", version="2.1.0")

_CONFIG: dict[str, Any] = {}
_JOBS: dict[str, dict[str, Any]] = {}

from config.settings import Settings
from data_pipeline.db_manager import DatabaseManager

_DB = DatabaseManager(Settings())
_SETTINGS = Settings()


def load_config(path: str = "config.yaml") -> dict[str, Any]:
    global _CONFIG
    with open(path) as f:
        _CONFIG = yaml.safe_load(f)
    return _CONFIG


@app.on_event("startup")
async def startup() -> None:
    load_config()
    logger.info("Config loaded — market=%s", _CONFIG.get("market", "?"))


# ── Schemas ──────────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    tickers: list[str] | None = None
    date: str | None = None       # YYYY-MM-DD

class BacktestRequest(BaseModel):
    tickers: list[str] | None = None
    start_date: str | None = None
    end_date: str | None = None
    initial_capital: float | None = None
    optimize: bool = False

class DataUpdateRequest(BaseModel):
    tickers: list[str] | None = None
    start_date: str | None = None
    end_date: str | None = None

class JobStatus(BaseModel):
    job_id: str
    status: str
    created_at: str
    result: dict[str, Any] | None = None


# ── Helper ───────────────────────────────────────────────────────

def _resolve_tickers(req_tickers: list[str] | None) -> list[str]:
    if req_tickers:
        return req_tickers
    data_cfg = _CONFIG.get("data", {})
    tickers_cfg = data_cfg.get("tickers", {})
    market = _CONFIG.get("market", "A")
    return tickers_cfg.get(market, tickers_cfg.get("US", []))


# ── Health ───────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "market": _CONFIG.get("market", "?")}


# ── Recommendation routes ────────────────────────────────────────

@app.get("/recommendation/daily")
async def get_daily_recommendation() -> dict[str, Any]:
    """Return the latest daily recommendation (scores + allocations)."""
    try:
        from data_pipeline.models import DailyScore
        from sqlalchemy import select

        with _DB._session_factory() as session:
            latest = session.query(DailyScore.score_date).order_by(
                DailyScore.score_date.desc()
            ).first()

            if latest is None:
                return {"status": "no_data", "message": "Run POST /recommendation/run first"}

            latest_date = latest[0]

            rows = session.query(DailyScore).filter(
                DailyScore.score_date == latest_date
            ).order_by(DailyScore.adjusted_total.desc()).all()

        return {
            "date": str(latest_date),
            "total": len(rows),
            "etfs": [
                {
                    "ticker": r.ticker, "total_score": r.adjusted_total,
                    "raw_total": r.total_score, "ml_signal": r.ml_signal,
                    "recommendation": r.recommendation,
                    "module_scores": r.module_scores_json,
                }
                for r in rows
            ],
        }
    except Exception as e:
        logger.exception("Daily recommendation failed")
        return {"status": "error", "message": str(e)}


@app.get("/recommendation/etf/{ticker}")
async def get_etf_detail(ticker: str) -> dict[str, Any]:
    """Return full scoring detail and history for a single ETF ticker."""
    try:
        from data_pipeline.models import DailyScore, SentimentRecord

        with _DB._session_factory() as session:
            latest_score = session.query(DailyScore).filter(
                DailyScore.ticker == ticker
            ).order_by(DailyScore.score_date.desc()).first()

            if latest_score is None:
                return {"status": "not_found", "ticker": ticker, "message": "No scores yet"}

            history = session.query(DailyScore).filter(
                DailyScore.ticker == ticker
            ).order_by(DailyScore.score_date.desc()).limit(12).all()

            latest_sent = session.query(SentimentRecord).filter(
                SentimentRecord.ticker == ticker
            ).order_by(SentimentRecord.event_date.desc()).first()

        return {
            "ticker": ticker,
            "latest": {
                "date": str(latest_score.score_date),
                "total_score": latest_score.adjusted_total,
                "raw_total": latest_score.total_score,
                "ml_signal": latest_score.ml_signal,
                "recommendation": latest_score.recommendation,
                "module_scores": latest_score.module_scores_json,
            },
            "history": [
                {"date": str(h.score_date), "score": h.adjusted_total, "signal": h.ml_signal}
                for h in history
            ],
            "sentiment": {
                "polarity": latest_sent.polarity if latest_sent else None,
                "confidence": latest_sent.confidence if latest_sent else None,
                "event_date": str(latest_sent.event_date) if latest_sent else None,
            } if latest_sent else {},
        }
    except Exception as e:
        logger.exception("ETF detail failed for %s", ticker)
        return {"status": "error", "ticker": ticker, "message": str(e)}


@app.post("/recommendation/score")
async def score_etfs(req: ScoreRequest) -> dict[str, Any]:
    """Score ETFs on-demand using data from the database."""
    try:
        tickers = req.tickers or _resolve_tickers(None)
        run_date = date.fromisoformat(req.date) if req.date else date.today()

        import pandas as pd
        from recommendation.pipeline import DailyRecommendationPipeline

        start = pd.Timestamp(run_date.replace(year=run_date.year - 2))
        end = pd.Timestamp(run_date)
        prices = _DB.load_prices(tickers, start, end)

        # Load sentiment
        with _DB.session() as sess:
            srows = sess.query(SentimentRecord).filter(
                SentimentRecord.ticker.in_(tickers),
                SentimentRecord.event_date <= run_date,
            ).order_by(SentimentRecord.event_date.desc()).limit(500).all()
        sentiment = pd.DataFrame([{
            "ticker": r.ticker, "event_date": r.event_date,
            "polarity": r.polarity, "confidence": r.confidence,
        } for r in srows])

        issuer_df = _DB.load_issuers()
        profiles = _DB.load_profiles(tickers)
        index_meta = _DB.load_index_meta(tickers)

        pipeline = DailyRecommendationPipeline(config=_CONFIG)
        result = pipeline.run(prices, sentiment if not sentiment.empty else None,
                              issuer_df=issuer_df if not issuer_df.empty else None,
                              profiles=profiles if not profiles.empty else None,
                              index_meta=index_meta if not index_meta.empty else None,
                              run_date=run_date)

        return result.model_dump(mode="json")
    except Exception as e:
        logger.exception("Scoring failed")
        return {"status": "error", "message": str(e)}


@app.post("/recommendation/run")
async def run_recommendation_pipeline(
    req: ScoreRequest, bg: BackgroundTasks,
) -> dict[str, str]:
    """Trigger a full recommendation pipeline run and cache results to DB."""
    job_id = str(uuid.uuid4())[:8]
    _JOBS[job_id] = {"status": "pending", "created_at": datetime.now().isoformat(), "result": None}
    bg.add_task(_execute_recommendation, job_id, req)
    return {"job_id": job_id, "status": "pending"}


# ── Backtest routes ──────────────────────────────────────────────

@app.post("/backtest/run")
async def run_backtest(req: BacktestRequest, bg: BackgroundTasks) -> dict[str, str]:
    job_id = str(uuid.uuid4())[:8]
    _JOBS[job_id] = {"status": "pending", "created_at": datetime.now().isoformat(), "result": None}
    bg.add_task(_execute_backtest, job_id, req)
    return {"job_id": job_id, "status": "pending"}


@app.get("/backtest/status/{job_id}")
async def backtest_status(job_id: str) -> JobStatus:
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return JobStatus(job_id=job_id, **job)


@app.get("/report/live", response_class=HTMLResponse)
async def get_live_report() -> HTMLResponse:
    """Return the latest live strategy report."""
    from pathlib import Path
    report_path = Path("./reports/etf_live_report.html")
    if not report_path.exists():
        raise HTTPException(404, "Live report not found. Run generate_live_report.py first.")
    return HTMLResponse(report_path.read_text(encoding="utf-8"))


@app.get("/report/{job_id}")
async def get_report(job_id: str) -> HTMLResponse:
    from pathlib import Path
    report_path = Path(_CONFIG.get("report_output_dir", "./reports")) / f"report_{job_id}.html"
    if not report_path.exists():
        raise HTTPException(404, f"Report for job {job_id} not found")
    return HTMLResponse(report_path.read_text(encoding="utf-8"))


# ── Prediction routes ─────────────────────────────────────────────


@app.get("/prediction/all")
async def get_all_predictions() -> dict[str, Any]:
    """Return latest predictions for all ETFs grouped by ticker with 3 horizons."""
    try:
        from data_pipeline.models import ETFPrediction

        with _DB._session_factory() as sess:
            latest = sess.query(ETFPrediction.pred_date).order_by(
                ETFPrediction.pred_date.desc()
            ).first()
            if latest is None:
                return {"status": "no_data", "message": "Run POST /prediction/run first"}
            pred_date = latest[0]
            rows = sess.query(ETFPrediction).filter(
                ETFPrediction.pred_date == pred_date
            ).order_by(ETFPrediction.ticker, ETFPrediction.horizon_days).all()

        grouped: dict[str, dict] = {}
        for r in rows:
            if r.ticker not in grouped:
                grouped[r.ticker] = {"ticker": r.ticker, "pred_date": str(r.pred_date), "horizons": {}}
            grouped[r.ticker]["horizons"][f"{r.horizon_days}d"] = {
                "ticker": r.ticker, "pred_date": str(r.pred_date),
                "horizon_days": r.horizon_days, "predicted_return": r.predicted_return,
                "prob_up": r.prob_up, "target_return": r.target_return,
                "realized": r.realized, "model_version": r.model_version,
            }

        predictions = list(grouped.values())
        for p in predictions:
            probs = [h["prob_up"] for h in p["horizons"].values()]
            avg_prob = sum(probs) / len(probs) if probs else 0.5
            p["consensus_direction"] = "BULLISH" if avg_prob > 0.65 else ("BEARISH" if avg_prob < 0.40 else "NEUTRAL")

        predictions.sort(key=lambda p: sum(
            abs(h["predicted_return"]) for h in p["horizons"].values()
        ), reverse=True)

        return {"date": str(pred_date), "total": len(predictions), "predictions": predictions}
    except Exception as e:
        logger.exception("Prediction all failed")
        return {"status": "error", "message": str(e)}


@app.get("/prediction/etf/{ticker}")
async def get_etf_prediction_history(ticker: str) -> dict[str, Any]:
    """Return prediction history for a single ETF."""
    try:
        from data_pipeline.models import ETFPrediction

        with _DB._session_factory() as sess:
            rows = sess.query(ETFPrediction).filter(
                ETFPrediction.ticker == ticker
            ).order_by(ETFPrediction.pred_date.desc()).limit(90).all()

        if not rows:
            return {"status": "not_found", "ticker": ticker, "message": "No predictions yet"}

        # Latest as grouped by date
        latest_date = rows[0].pred_date
        latest_rows = [r for r in rows if r.pred_date == latest_date]
        latest_horizons = {}
        for r in latest_rows:
            latest_horizons[f"{r.horizon_days}d"] = {
                "ticker": r.ticker, "pred_date": str(r.pred_date),
                "horizon_days": r.horizon_days, "predicted_return": r.predicted_return,
                "prob_up": r.prob_up, "target_return": r.target_return,
                "realized": r.realized, "model_version": r.model_version,
            }

        probs = [h["prob_up"] for h in latest_horizons.values()]
        avg_prob = sum(probs) / len(probs) if probs else 0.5
        consensus = "BULLISH" if avg_prob > 0.65 else ("BEARISH" if avg_prob < 0.40 else "NEUTRAL")

        # History: one entry per date with all horizons
        hist_by_date: dict[str, dict] = {}
        for r in rows:
            ds = str(r.pred_date)
            if ds not in hist_by_date:
                hist_by_date[ds] = {"date": ds, "horizons": {}}
            hist_by_date[ds]["horizons"][f"{r.horizon_days}d"] = {
                "horizon_days": r.horizon_days, "predicted_return": r.predicted_return,
                "prob_up": r.prob_up,
            }

        return {
            "ticker": ticker,
            "latest": {"date": str(latest_date), "horizons": latest_horizons, "consensus": consensus},
            "history": list(hist_by_date.values()),
        }
    except Exception as e:
        logger.exception("Prediction detail failed for %s", ticker)
        return {"status": "error", "ticker": ticker, "message": str(e)}


@app.post("/prediction/run")
async def run_prediction_pipeline(req: ScoreRequest, bg: BackgroundTasks) -> dict[str, str]:
    """Trigger multi-horizon return prediction run."""
    job_id = str(uuid.uuid4())[:8]
    _JOBS[job_id] = {"status": "pending", "created_at": datetime.now().isoformat(), "result": None}
    bg.add_task(_execute_prediction, job_id, req)
    return {"job_id": job_id, "status": "pending"}


# ── Data routes ──────────────────────────────────────────────────

@app.post("/data/update")
async def update_data(req: DataUpdateRequest, bg: BackgroundTasks) -> dict[str, str]:
    job_id = str(uuid.uuid4())[:8]
    _JOBS[job_id] = {"status": "pending", "created_at": datetime.now().isoformat(), "result": None}
    bg.add_task(_execute_data_update, job_id, req)
    return {"job_id": job_id, "status": "pending"}


# ── Background tasks ─────────────────────────────────────────────

async def _execute_recommendation(job_id: str, req: ScoreRequest) -> None:
    import pandas as pd
    try:
        _JOBS[job_id]["status"] = "running"
        tickers = req.tickers or _resolve_tickers(None)
        run_date = date.fromisoformat(req.date) if req.date else date.today()

        from recommendation.pipeline import DailyRecommendationPipeline
        from data_pipeline.models import SentimentRecord
        start = pd.Timestamp(run_date.replace(year=run_date.year - 2))
        prices = _DB.load_prices(tickers, start, pd.Timestamp(run_date))

        with _DB.session() as sess:
            srows = sess.query(SentimentRecord).filter(
                SentimentRecord.ticker.in_(tickers),
                SentimentRecord.event_date <= run_date,
            ).order_by(SentimentRecord.event_date.desc()).limit(500).all()
        sentiment = pd.DataFrame([{
            "ticker": r.ticker, "event_date": r.event_date,
            "polarity": r.polarity, "confidence": r.confidence,
        } for r in srows])

        issuer_df = _DB.load_issuers()
        profiles = _DB.load_profiles(tickers)
        index_meta = _DB.load_index_meta(tickers)

        pipeline = DailyRecommendationPipeline(config=_CONFIG)
        result = pipeline.run(prices, sentiment if not sentiment.empty else None,
                              issuer_df=issuer_df if not issuer_df.empty else None,
                              profiles=profiles if not profiles.empty else None,
                              index_meta=index_meta if not index_meta.empty else None,
                              run_date=run_date)

        # Cache to DailyScore table
        from data_pipeline.models import DailyScore
        with _DB._engine.connect() as conn:
            from sqlalchemy import delete as sqla_delete
            conn.execute(sqla_delete(DailyScore).where(DailyScore.score_date == run_date))
            for etf in result.ranked_etfs:
                ms = etf.module_scores
                raw_total = ms.get("issuer", 0) + ms.get("index_quality", 0) + ms.get("individual_fund", 0)
                conn.execute(
                    DailyScore.__table__.insert().values(
                        ticker=etf.ticker,
                        score_date=run_date,
                        total_score=round(raw_total, 1),
                        adjusted_total=etf.total_score,
                        ml_signal=etf.ml_signal,
                        recommendation=etf.recommendation,
                        module_scores_json=str(etf.module_scores),
                    )
                )
            conn.commit()

        _JOBS[job_id].update({"status": "completed", "result": result.model_dump(mode="json")})
        logger.info("Recommendation %s completed — %d ETFs", job_id, result.total_universe)
    except Exception as e:
        _JOBS[job_id].update({"status": "failed", "result": {"error": str(e)}})
        logger.exception("Recommendation %s failed", job_id)


async def _execute_prediction(job_id: str, req: ScoreRequest) -> None:
    import pandas as pd
    try:
        _JOBS[job_id]["status"] = "running"
        tickers = req.tickers or _resolve_tickers(None)
        run_date = date.fromisoformat(req.date) if req.date else date.today()

        from prediction.pipeline import PredictionPipeline

        start = pd.Timestamp(run_date.replace(year=run_date.year - 2))
        prices = _DB.load_prices(tickers, start, pd.Timestamp(run_date))

        pipeline = PredictionPipeline(config=_CONFIG)
        rows = pipeline.run(prices, run_date)

        if rows:
            _DB.upsert_predictions(rows)
            logger.info("Prediction %s completed — %d rows", job_id, len(rows))
        else:
            logger.warning("Prediction %s produced no rows — model may not be trained", job_id)

        _JOBS[job_id].update({"status": "completed", "result": {"rows": len(rows), "date": str(run_date)}})
    except Exception as e:
        _JOBS[job_id].update({"status": "failed", "result": {"error": str(e)}})
        logger.exception("Prediction %s failed", job_id)


async def _execute_backtest(job_id: str, req: BacktestRequest) -> None:
    try:
        _JOBS[job_id]["status"] = "running"
        tickers = _resolve_tickers(req.tickers)
        start = date.fromisoformat(req.start_date) if req.start_date else date.today().replace(year=date.today().year - 2)
        end = date.fromisoformat(req.end_date) if req.end_date else date.today()

        from engine.backtest import run_full_backtest
        result = run_full_backtest(
            job_id=job_id, tickers=tickers, start=start, end=end,
            initial_capital=req.initial_capital or _CONFIG.get("backtest", {}).get("initial_capital", 1_000_000),
            optimize=req.optimize, config=_CONFIG,
        )
        _JOBS[job_id].update({"status": "completed", "result": result})
        logger.info("Backtest %s completed", job_id)
    except Exception as e:
        _JOBS[job_id].update({"status": "failed", "result": {"error": str(e)}})
        logger.exception("Backtest %s failed", job_id)


async def _execute_data_update(job_id: str, req: DataUpdateRequest) -> None:
    try:
        _JOBS[job_id]["status"] = "running"
        tickers = _resolve_tickers(req.tickers)
        start = date.fromisoformat(req.start_date) if req.start_date else date.today().replace(year=date.today().year - 2)
        end = date.fromisoformat(req.end_date) if req.end_date else date.today()

        from data_pipeline.fetcher import DataFetcherFactory
        from data_pipeline.cleaner import DataCleaner

        fetcher = DataFetcherFactory.create(_CONFIG.get("market", "A"))
        prices = fetcher.fetch(tickers, start, end)
        cleaner = DataCleaner(*_SETTINGS.winsorize_bounds)
        cleaned = cleaner.clean_etf_prices(prices)
        _DB.upsert_prices(cleaned)

        _JOBS[job_id].update({
            "status": "completed",
            "result": {"rows": len(cleaned), "tickers": int(cleaned["ticker"].nunique()),
                       "date_range": f"{cleaned['trade_date'].min()} ~ {cleaned['trade_date'].max()}"},
        })
        logger.info("Data update %s completed", job_id)
    except Exception as e:
        _JOBS[job_id].update({"status": "failed", "result": {"error": str(e)}})
        logger.exception("Data update %s failed", job_id)
