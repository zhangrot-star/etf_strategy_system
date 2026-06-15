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

# Execution system globals (initialised in startup)
_broker = None
_position_tracker = None
_auto_trader = None


def load_config(path: str = "config.yaml") -> dict[str, Any]:
    global _CONFIG
    with open(path) as f:
        _CONFIG = yaml.safe_load(f)
    return _CONFIG


@app.on_event("startup")
async def startup() -> None:
    global _broker, _position_tracker, _auto_trader
    load_config()

    # Init broker
    from execution.broker_interface import PaperBroker
    _broker = PaperBroker(initial_cash=1_000_000.0)
    _broker.connect()
    logger.info("Broker initialised — paper mode")

    # Init position tracker
    from portfolio.position_tracker import PositionTracker
    _position_tracker = PositionTracker()
    _position_tracker.sync_from_broker(_broker)

    # Init auto trader
    from portfolio.position_tracker import AutoTrader
    _auto_trader = AutoTrader(config=_CONFIG, db_manager=_DB)

    # Optionally start scheduler if enabled
    sched_cfg = _CONFIG.get("schedule", {})
    if sched_cfg.get("enabled", False):
        try:
            from scheduler.jobs import start_scheduler
            start_scheduler(
                _auto_trader,
                cron_expr=sched_cfg.get("cron", "0 15 * * 1-5"),
                timezone=sched_cfg.get("timezone", "Asia/Shanghai"),
            )
            logger.info("Scheduler started")
        except Exception:
            logger.warning("Failed to start scheduler", exc_info=True)

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


# ── Portfolio / Execution routes ──────────────────────────────────


@app.get("/portfolio/current")
async def get_current_portfolio() -> dict[str, Any]:
    """Return current portfolio positions from broker."""
    global _broker, _position_tracker
    try:
        if _broker is None:
            return {"status": "error", "message": "Broker not initialised"}

        _position_tracker.sync_from_broker(_broker)
        positions = _position_tracker.get_snapshot()
        account = _broker.get_account_info()

        return {
            "account": {
                "total_asset": account.total_asset,
                "available_cash": account.available_cash,
                "frozen_cash": account.frozen_cash,
                "total_return": account.total_return,
            },
            "positions": [
                {
                    "ticker": p.ticker, "name": p.name, "shares": p.shares,
                    "avg_cost": p.avg_cost, "current_price": p.current_price,
                    "market_value": p.market_value, "pnl": p.pnl,
                }
                for p in positions
            ],
            "position_count": len(positions),
        }
    except Exception as e:
        logger.exception("Portfolio current failed")
        return {"status": "error", "message": str(e)}


@app.get("/portfolio/target")
async def get_target_portfolio() -> dict[str, Any]:
    """Return the latest target allocation from recommendation pipeline."""
    try:
        from recommendation.pipeline import DailyRecommendationPipeline
        from datetime import date
        import pandas as pd

        run_date = date.today()
        tickers = _resolve_tickers(None)
        start = pd.Timestamp(run_date.replace(year=run_date.year - 2))
        prices = _DB.load_prices(tickers, start, pd.Timestamp(run_date))

        pipeline = DailyRecommendationPipeline(config=_CONFIG)
        result = pipeline.run(prices, run_date=run_date)

        return {
            "date": str(result.date),
            "risk_status": result.risk_status,
            "cash_weight": result.cash_weight,
            "targets": [
                {
                    "ticker": etf.ticker, "name": etf.name,
                    "weight": etf.allocation_weight, "rating": etf.rating,
                    "ml_signal": etf.ml_signal, "recommendation": etf.recommendation,
                    "total_score": etf.total_score,
                }
                for etf in result.ranked_etfs
            ],
        }
    except Exception as e:
        logger.exception("Portfolio target failed")
        return {"status": "error", "message": str(e)}


@app.get("/portfolio/rebalance")
async def get_rebalance_preview() -> dict[str, Any]:
    """Preview rebalance diff without executing."""
    global _broker, _position_tracker
    try:
        from datetime import date
        import pandas as pd
        from recommendation.pipeline import DailyRecommendationPipeline
        from portfolio.rebalance_engine import RebalanceEngine

        # Get current positions
        _position_tracker.sync_from_broker(_broker)
        positions = _position_tracker.get_snapshot()
        account = _broker.get_account_info()

        # Get target recommendation
        run_date = date.today()
        tickers = _resolve_tickers(None)
        start = pd.Timestamp(run_date.replace(year=run_date.year - 2))
        prices = _DB.load_prices(tickers, start, pd.Timestamp(run_date))
        pipeline = DailyRecommendationPipeline(config=_CONFIG)
        recommendation = pipeline.run(prices, run_date=run_date)

        # Compute diff
        pos_dicts = [{
            "ticker": p.ticker, "market_value": p.market_value,
            "shares": p.shares, "current_price": p.current_price,
        } for p in positions]

        engine = RebalanceEngine(config=_CONFIG)
        diff = engine.compute_diff(pos_dicts, recommendation, account.total_asset)

        return {
            "date": str(run_date),
            "total_asset": account.total_asset,
            "diff_summary": diff.summary(),
            "diffs": [
                {
                    "ticker": d.ticker, "current_weight": d.current_weight,
                    "target_weight": d.target_weight, "delta_w": d.delta_w,
                    "action": d.action, "trade_value": d.trade_value,
                }
                for d in diff.items if d.action != "HOLD"
            ],
        }
    except Exception as e:
        logger.exception("Rebalance preview failed")
        return {"status": "error", "message": str(e)}


@app.post("/execution/dry-run")
async def post_dry_run() -> dict[str, Any]:
    """Execute a dry-run rebalance (no real orders)."""
    global _auto_trader
    try:
        if _auto_trader is None:
            from portfolio.position_tracker import AutoTrader
            _auto_trader = AutoTrader(config=_CONFIG, db_manager=_DB)

        result = _auto_trader.run(dry_run=True)
        return result.summary()
    except Exception as e:
        logger.exception("Dry-run failed")
        return {"status": "error", "message": str(e)}


@app.post("/execution/rebalance")
async def post_live_rebalance() -> dict[str, Any]:
    """Execute a live rebalance (REAL orders — use with caution)."""
    global _auto_trader
    try:
        exec_cfg = _CONFIG.get("execution", {})
        if exec_cfg.get("dry_run", True):
            return {
                "status": "blocked",
                "message": "execution.dry_run is true in config — set to false to enable live trading",
            }

        if _auto_trader is None:
            from portfolio.position_tracker import AutoTrader
            _auto_trader = AutoTrader(config=_CONFIG, db_manager=_DB)

        result = _auto_trader.run(dry_run=False)
        summary = result.summary()
        summary["orders"] = [
            {
                "order_id": getattr(r, "order_id", ""),
                "ticker": getattr(r, "ticker", ""),
                "side": getattr(r, "side", ""),
                "quantity": getattr(r, "quantity", 0),
                "price": getattr(r, "price", 0),
                "status": getattr(r, "status", "unknown"),
                "filled_qty": getattr(r, "filled_qty", 0),
            }
            for r in result.order_results
        ]
        return summary
    except Exception as e:
        logger.exception("Live rebalance failed")
        return {"status": "error", "message": str(e)}


@app.get("/execution/orders")
async def get_orders(status: str | None = None) -> dict[str, Any]:
    """Query recent orders from broker."""
    global _broker
    try:
        if _broker is None:
            return {"status": "error", "message": "Broker not initialised"}

        orders = _broker.query_orders(status=status)
        return {
            "count": len(orders),
            "orders": [
                {
                    "order_id": o.order_id, "ticker": o.ticker,
                    "side": o.side, "quantity": o.quantity,
                    "price": o.price, "status": o.status,
                    "filled_qty": o.filled_qty, "timestamp": o.timestamp,
                }
                for o in orders
            ],
        }
    except Exception as e:
        logger.exception("Query orders failed")
        return {"status": "error", "message": str(e)}


@app.get("/execution/order/{order_id}")
async def get_order_detail(order_id: str) -> dict[str, Any]:
    """Get detail for a specific order."""
    global _broker
    try:
        if _broker is None:
            return {"status": "error", "message": "Broker not initialised"}

        orders = _broker.query_orders()
        for o in orders:
            if o.order_id == order_id:
                return {
                    "order_id": o.order_id, "ticker": o.ticker,
                    "side": o.side, "quantity": o.quantity,
                    "price": o.price, "status": o.status,
                    "filled_qty": o.filled_qty, "timestamp": o.timestamp,
                }
        return {"status": "not_found", "order_id": order_id}
    except Exception as e:
        logger.exception("Order detail failed")
        return {"status": "error", "message": str(e)}


@app.get("/execution/status")
async def get_execution_status() -> dict[str, Any]:
    """Get execution system status."""
    global _broker, _position_tracker, _auto_trader
    try:
        sched_cfg = _CONFIG.get("schedule", {})
        exec_cfg = _CONFIG.get("execution", {})

        next_run = None
        try:
            from scheduler.jobs import get_next_run_time
            next_run = get_next_run_time()
        except Exception:
            next_run = "scheduler not available"

        return {
            "broker": exec_cfg.get("broker", "paper"),
            "dry_run": exec_cfg.get("dry_run", True),
            "broker_connected": _broker is not None,
            "positions_tracked": len(_position_tracker.get_snapshot()) if _position_tracker else 0,
            "schedule": {
                "enabled": sched_cfg.get("enabled", False),
                "cron": sched_cfg.get("cron", ""),
                "next_run": next_run,
            },
            "circuit_breaker": exec_cfg.get("circuit_breaker", True),
            "min_trade_amount": exec_cfg.get("min_trade_amount", 5000),
        }
    except Exception as e:
        logger.exception("Execution status failed")
        return {"status": "error", "message": str(e)}


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


# ═══════════════════════════════════════════════════════════════════
#  Bank Analyzer Routes — 银行信用卡业务合作潜力分析
# ═══════════════════════════════════════════════════════════════════


@app.get("/bank/ranking")
async def get_bank_ranking(top: int | None = None) -> dict[str, Any]:
    """返回银行合作潜力排名.

    Query params:
        top: 返回前 N 名 (默认返回全部 18 家)
    """
    try:
        from bank_analyzer.bank_data import BankDataPipeline
        from bank_analyzer.bank_scorer import BankScorer

        pipeline = BankDataPipeline(_CONFIG)
        df = pipeline.run()

        scorer = BankScorer(_CONFIG)
        bank_cfg = _CONFIG.get("bank_analyzer", {})
        model_path = bank_cfg.get("model_path", "models/bank_scorer")
        try:
            scorer.load(model_path)
        except Exception:
            scorer.fit(df)

        all_scores = scorer.score_all(df)
        if top:
            all_scores = all_scores[:top]

        return {
            "status": "ok",
            "date": str(date.today()),
            "total": len(all_scores),
            "ranking": [
                {
                    "rank": s.rank,
                    "bank_id": s.bank_id,
                    "bank_name": s.bank_name,
                    "bank_type": s.bank_type,
                    "module_a_total": round(s.module_a_total, 1),
                    "module_b_total": round(s.module_b_total, 1),
                    "module_c_total": round(s.module_c_total, 1),
                    "module_d_total": round(s.module_d_total, 1),
                    "cooperation_potential": round(s.cooperation_potential, 1),
                    "ml_signal": s.ml_signal,
                    "recommendation": s.recommendation,
                    "risk_warning": s.risk_warning,
                }
                for s in all_scores
            ],
        }
    except Exception as e:
        logger.exception("Bank ranking failed")
        return {"status": "error", "message": str(e)}


@app.get("/bank/{bank_id}")
async def get_bank_detail(bank_id: str) -> dict[str, Any]:
    """获取单个银行的详细分析.

    Path params:
        bank_id: 银行 ID，如 CMB, ICBC, CCB
    """
    try:
        from bank_analyzer.bank_data import BankDataCollector, BANK_UNIVERSE
        from bank_analyzer.bank_scorer import BankScorer
        from bank_analyzer.bank_pain_points import BankPainPointAnalyzer

        if bank_id.upper() not in BANK_UNIVERSE:
            return {"status": "not_found", "bank_id": bank_id, "supported": list(BANK_UNIVERSE.keys())}

        # 数据采集
        collector = BankDataCollector(_CONFIG)
        profile = collector.collect_one(bank_id.upper())

        # 评分
        scorer = BankScorer(_CONFIG)
        pipeline = __import__("bank_analyzer.bank_data", fromlist=["BankDataPipeline"]).BankDataPipeline(_CONFIG)
        df = pipeline.run([bank_id.upper()])
        bank_cfg = _CONFIG.get("bank_analyzer", {})
        model_path = bank_cfg.get("model_path", "models/bank_scorer")
        try:
            scorer.load(model_path)
        except Exception:
            scorer.fit(df)
        scores = scorer.score_all(df)

        # 痛点分析
        analyzer = BankPainPointAnalyzer(_CONFIG)
        pain = analyzer.analyze_one(
            bank_name=profile.name,
            bank_type=profile.bank_type,
            profile_text=analyzer._profile_to_text(profile),
        )

        return {
            "status": "ok",
            "bank_id": bank_id.upper(),
            "profile": {
                "name": profile.name,
                "bank_type": profile.bank_type,
                "aum_rank": profile.aum_rank,
                "total_assets": profile.total_assets,
                "revenue": profile.revenue,
                "net_profit": profile.net_profit,
                "roe": profile.roe,
                "car": profile.car,
                "npl_ratio": profile.npl_ratio,
                "credit_card_volume": profile.credit_card_volume,
                "credit_card_active_users": profile.credit_card_active_users,
                "credit_card_transaction": profile.credit_card_transaction,
                "credit_card_revenue": profile.credit_card_revenue,
                "mobile_bank_users": profile.mobile_bank_users,
                "digital_transaction_ratio": profile.digital_transaction_ratio,
                "fintech_investment": profile.fintech_investment,
                "market_share_pct": profile.market_share_pct,
                "yoy_growth_pct": profile.yoy_growth_pct,
                "data_source": profile.data_source,
            },
            "score": {
                "cooperation_potential": round(scores[0].cooperation_potential, 1) if scores else 0,
                "recommendation": scores[0].recommendation if scores else "N/A",
            } if scores else {},
            "pain_point_analysis": {
                "strategic_focus": pain.strategic_focus,
                "business_pain_points": pain.business_pain_points,
                "cooperation_opportunities": pain.cooperation_opportunities,
                "risk_assessment": pain.risk_assessment,
                "summary": pain.summary,
            },
        }
    except Exception as e:
        logger.exception("Bank detail failed for %s", bank_id)
        return {"status": "error", "bank_id": bank_id, "message": str(e)}


@app.post("/bank/analyze")
async def analyze_banks(req: ScoreRequest, bg: BackgroundTasks) -> dict[str, str]:
    """触发完整的银行分析管线（异步执行）.

    Body:
        tickers: 指定银行 ID 列表，可选（默认全部）
        date: 分析日期，可选
    """
    job_id = str(uuid.uuid4())[:8]
    _JOBS[job_id] = {"status": "pending", "created_at": datetime.now().isoformat(), "result": None}
    bg.add_task(_execute_bank_analysis, job_id, req)
    return {"job_id": job_id, "status": "pending"}


@app.get("/bank/report", response_class=HTMLResponse)
async def get_bank_report() -> HTMLResponse:
    """返回最新的银行分析 HTML 报告."""
    from pathlib import Path
    bank_cfg = _CONFIG.get("bank_analyzer", {})
    report_dir = Path(bank_cfg.get("report_output_dir", "./reports/bank"))
    if not report_dir.exists():
        raise HTTPException(404, "No bank report found. Run POST /bank/analyze first.")
    reports = sorted(report_dir.glob("bank_analysis_*.html"), reverse=True)
    if not reports:
        raise HTTPException(404, "No bank report found. Run POST /bank/analyze first.")
    return HTMLResponse(reports[0].read_text(encoding="utf-8"))


async def _execute_bank_analysis(job_id: str, req: ScoreRequest) -> None:
    """后台执行完整银行分析管线."""
    try:
        _JOBS[job_id]["status"] = "running"

        from bank_analyzer.bank_data import BankDataPipeline, BANK_UNIVERSE
        from bank_analyzer.bank_scorer import BankScorer
        from bank_analyzer.bank_pain_points import BankPainPointAnalyzer
        from bank_analyzer.bank_report import BankReportRenderer

        bank_cfg = _CONFIG.get("bank_analyzer", {})
        bank_ids = req.tickers if req.tickers else bank_cfg.get("monitored_banks", list(BANK_UNIVERSE.keys()))

        # 1. 数据管线
        pipeline = BankDataPipeline(_CONFIG)
        df = pipeline.run(bank_ids)

        # 2. 评分
        scorer = BankScorer(_CONFIG)
        model_path = bank_cfg.get("model_path", "models/bank_scorer")
        try:
            scorer.load(model_path)
        except Exception:
            scorer.fit(df)
            scorer.save(model_path)

        scores = scorer.score_all(df)

        # 3. 痛点分析
        analyzer = BankPainPointAnalyzer(_CONFIG)
        profiles = pipeline._collector.collect_all_profiles(bank_ids)
        pain_points = analyzer.analyze_all(profiles)

        # 4. 生成报告
        from scripts.analyze_banks import build_chart_data
        chart_data = build_chart_data(scores)

        summary_parts = [f"【{pp.bank_name}】{pp.summary}" for pp in pain_points[:3]]
        summary_text = "\n\n".join(summary_parts)

        renderer = BankReportRenderer()
        output_dir = bank_cfg.get("report_output_dir", "./reports/bank")
        output_path = f"{output_dir}/bank_analysis_{date.today().isoformat()}.html"
        renderer.render_to_file(
            output_path=output_path,
            scores=[s.__dict__ for s in scores],
            pain_points=[pp.__dict__ for pp in pain_points],
            chart_data=chart_data,
            summary_text=summary_text,
        )

        _JOBS[job_id].update({
            "status": "completed",
            "result": {
                "total_banks": len(scores),
                "top_3": [s.bank_name for s in scores[:3]],
                "recommend_count": sum(1 for s in scores if s.cooperation_potential >= 65),
                "report_path": output_path,
            },
        })
        logger.info("Bank analysis %s completed — %d banks", job_id, len(scores))
    except Exception as e:
        _JOBS[job_id].update({"status": "failed", "result": {"error": str(e)}})
        logger.exception("Bank analysis %s failed", job_id)
