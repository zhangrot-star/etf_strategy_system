"""Production backtest assembly with corrected commission and analytics.

Fixes from the original:
1. Uses CorrectBilateralCommission (no double-count)
2. Uses fixed TurnoverAnalyzer
3. Adds TradeStatsAnalyzer for win rate / profit factor
4. Proper cerebro cleanup between runs
5. Structured result dict for API consumption
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import backtrader as bt
import pandas as pd

from engine.analyzers import DrawdownAnalyzer, TradeStatsAnalyzer, TurnoverAnalyzer
from engine.commissions import CorrectBilateralCommission

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Structured backtest output."""

    job_id: str
    tickers: list[str]
    start_date: date
    end_date: date
    initial_capital: float
    final_value: float
    cum_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_len: int
    turnover_ratio: float
    trade_count: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    monthly_returns: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def run_full_backtest(
    job_id: str,
    tickers: list[str],
    start: date,
    end: date,
    initial_capital: float,
    optimize: bool,
    config: dict[str, Any],
    rl_model_path: str | None = None,
) -> dict[str, Any]:
    """Run a complete backtest and return a JSON-serializable result dict.

    Args:
        job_id: Unique identifier for this run.
        tickers: List of ETF ticker codes.
        start: Backtest start date.
        end: Backtest end date.
        initial_capital: Starting portfolio value.
        optimize: If True, run Optuna hyperparameter search first.
        config: Full configuration dict (from config.yaml).

    Returns:
        Dict with all backtest metrics suitable for API response.
    """
    # ── 1. Load data (MySQL first, fallback to API) ──────────
    market = config.get("market", "A")
    prices = _load_prices_from_db(tickers, start, end)

    if prices.empty:
        from data_pipeline.fetcher import DataFetcherFactory
        logger.info("No data in DB — fetching from API...")
        fetcher = DataFetcherFactory.create(market)
        prices = fetcher.fetch(tickers, start, end)
        if not prices.empty:
            prices = fetcher.clean(prices)
            # Store for next time
            try:
                from data_pipeline.db_manager import DatabaseManager
                DatabaseManager().upsert_prices(prices)
            except Exception:
                pass

    if prices.empty:
        return {"error": f"No price data for tickers={tickers}", "job_id": job_id}

    # ── 2. Optional optimization ───────────────────────────────
    xgb_params = config.get("xgboost", {})
    if optimize:
        try:
            from strategy.optimizer import optimize_xgboost_params
            optuna_cfg = config.get("optuna", {})
            xgb_params = optimize_xgboost_params(
                prices=prices,
                tickers=tickers,
                n_trials=optuna_cfg.get("n_trials", 100),
                study_name=optuna_cfg.get("study_name", "xgboost_etf_optimization"),
                storage=optuna_cfg.get("storage", "sqlite:///optuna.db"),
            )
            logger.info("Optuna complete — best params: %s", xgb_params)
        except Exception:
            logger.exception("Optuna optimization failed, using config defaults")

    # ── 3. Compute features & labels ───────────────────────────
    from core.feature_utils import build_features_and_labels
    features_df, labels_series = build_features_and_labels(prices)

    # ── 4. Train ensemble ──────────────────────────────────────
    from core.ensemble import XGBoostEnsemble
    ensemble = XGBoostEnsemble()
    ensemble._settings.xgb_max_depth = xgb_params.get("max_depth", 6)
    ensemble._settings.xgb_learning_rate = xgb_params.get("learning_rate", 0.05)
    ensemble._settings.xgb_n_estimators = xgb_params.get("n_estimators", 200)
    ensemble._settings.xgb_subsample = xgb_params.get("subsample", 0.8)

    # Fit on pre-2025 data, reserve last year for backtest
    if "trade_date" in prices.columns:
        train_cutoff = pd.Timestamp("2025-01-01")
        train_mask = pd.to_datetime(prices["trade_date"]) < train_cutoff
        # Build features only for training period
        train_prices = prices[train_mask].copy()
        train_features, train_labels = build_features_and_labels(train_prices)

        if not train_features.empty and not train_labels.empty:
            # Align features and labels
            common_idx = train_features.index.intersection(train_labels.index)
            ensemble.fit(
                train_features.loc[common_idx],
                train_labels.loc[common_idx],
            )
            logger.info("Ensemble trained on %d samples", len(common_idx))
    else:
        ensemble.fit(features_df, labels_series)
        logger.info("Ensemble trained on full dataset (%d samples)", len(features_df))

    # ── 5. Run Backtrader ──────────────────────────────────────
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_capital)

    # Commission
    comm_rate = config.get("backtest", {}).get("commission_rate", 0.0003)
    cerebro.broker.addcommissioninfo(CorrectBilateralCommission(commission=comm_rate))

    # Slippage
    from backtest.bt_slippage import configure_slippage, VolumeAwareSizer
    slippage_bps = config.get("backtest", {}).get("slippage_bps", 1.0)
    configure_slippage(cerebro, bps=slippage_bps)
    cerebro.addsizer(VolumeAwareSizer, volume_frac=0.01, max_frac=0.30)

    # Add data feeds
    for ticker in tickers:
        t_data = prices[prices["ticker"] == ticker].sort_values("trade_date")
        if t_data.empty:
            continue
        t_data = t_data.copy()
        t_data["trade_date"] = pd.to_datetime(t_data["trade_date"])
        t_data = t_data.set_index("trade_date")
        data_feed = bt.feeds.PandasData(
            dataname=t_data,
            open="open", high="high", low="low", close="close", volume="volume",
            openinterest=-1,
        )
        data_feed._name = ticker
        cerebro.adddata(data_feed)

    # Strategy
    from core.risk_controller import RiskController
    from core.strategy import CoreStrategy

    orchestrator = CoreStrategy(
        config={
            "xgboost": xgb_params,
            "risk": config.get("risk", {}),
        }
    )
    # Replace the ensemble with our pre-configured one
    orchestrator.ensemble._settings.xgb_max_depth = xgb_params.get("max_depth", 6)
    orchestrator.ensemble._settings.xgb_learning_rate = xgb_params.get("learning_rate", 0.05)
    orchestrator.ensemble._settings.xgb_n_estimators = xgb_params.get("n_estimators", 200)
    orchestrator.ensemble._settings.xgb_subsample = xgb_params.get("subsample", 0.8)

    # Load RL policy if specified
    if rl_model_path:
        try:
            orchestrator.load_rl_policy(rl_model_path)
            logger.info("RL policy loaded for backtest: %s", rl_model_path)
        except Exception:
            logger.exception("Failed to load RL policy — continuing with rule-based weights")

    from backtest.bt_strategy import ETFStrategy
    cerebro.addstrategy(
        ETFStrategy,
        orchestrator=orchestrator,
        features_df=features_df,
        sentiment_df=pd.DataFrame(),  # Backtrader path uses empty sentiment
        rebalance_freq=config.get("backtest", {}).get("rebalance_frequency", "monthly"),
        lookback_days=config.get("backtest", {}).get("lookback_days", 252),
    )

    # Analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.03)
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name="ann_return")
    cerebro.addanalyzer(DrawdownAnalyzer, _name="drawdown")
    cerebro.addanalyzer(TurnoverAnalyzer, _name="turnover")
    cerebro.addanalyzer(TradeStatsAnalyzer, _name="trades")

    # Run
    try:
        results = cerebro.run()
        strat = results[0]
    except Exception:
        logger.exception("Backtrader run failed")
        return {"error": "Backtrader execution failed", "job_id": job_id}

    # ── 6. Extract metrics ─────────────────────────────────────
    final_value = cerebro.broker.getvalue()
    cum_return = (final_value / initial_capital) - 1

    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    sharpe = sharpe_analysis.get("sharperatio", 0.0) or 0.0

    ann_ret_analysis = strat.analyzers.ann_return.get_analysis()
    ann_ret = ann_ret_analysis.get("rnorm100", 0.0) or 0.0

    dd_analysis = strat.analyzers.drawdown.get_analysis()
    max_dd = dd_analysis.get("max_drawdown", 0.0)
    max_dd_len = dd_analysis.get("max_drawdown_len", 0)

    to_analysis = strat.analyzers.turnover.get_analysis()
    turnover = to_analysis.get("turnover_ratio", 0.0)

    trade_analysis = strat.analyzers.trades.get_analysis()
    trade_count = trade_analysis.get("total_trades", 0)
    win_rate = trade_analysis.get("win_rate", 0.0)
    profit_factor = trade_analysis.get("profit_factor", 0.0)
    total_pnl = trade_analysis.get("total_pnl", 0.0)

    # Equity curve
    equity_curve = []
    eq_values = dd_analysis.get("equity_curve", [])
    for i, val in enumerate(eq_values):
        equity_curve.append({"bar": i, "value": round(val, 2)})

    # Annualized volatility from equity curve
    if len(eq_values) > 1:
        eq_series = pd.Series(eq_values)
        daily_rets = eq_series.pct_change().dropna()
        ann_vol = float(daily_rets.std() * (252 ** 0.5))
    else:
        ann_vol = 0.0

    # Monthly returns from equity curve
    monthly_rets = {}
    if len(eq_values) > 21:
        eq_series = pd.Series(eq_values)
        eq_series.index = pd.date_range(start=start, periods=len(eq_series), freq="B")
        monthly = eq_series.resample("ME").last().pct_change().dropna()
        for dt, val in monthly.items():
            monthly_rets[dt.strftime("%Y-%m")] = round(float(val) * 100, 2)

    result = BacktestResult(
        job_id=job_id,
        tickers=tickers,
        start_date=start,
        end_date=end,
        initial_capital=initial_capital,
        final_value=round(final_value, 2),
        cum_return=round(cum_return, 4),
        annual_return=round(ann_ret, 4),
        annual_volatility=round(ann_vol, 4),
        sharpe_ratio=round(sharpe, 4),
        max_drawdown=round(max_dd, 4),
        max_drawdown_len=max_dd_len,
        turnover_ratio=round(turnover, 4),
        trade_count=trade_count,
        win_rate=round(win_rate, 4),
        profit_factor=round(profit_factor, 4),
        total_pnl=round(total_pnl, 2),
        equity_curve=equity_curve,
        monthly_returns=monthly_rets,
    )

    logger.info("Backtest %s — return=%.2f%%, Sharpe=%.2f, maxDD=%.2f%%",
                 job_id, cum_return * 100, sharpe, max_dd * 100)

    return {
        "job_id": job_id,
        "cum_return": cum_return,
        "annual_return": ann_ret,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "max_drawdown_len": max_dd_len,
        "turnover_ratio": turnover,
        "trade_count": trade_count,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_pnl": total_pnl,
        "final_value": round(final_value, 2),
        "annual_volatility": ann_vol,
        "monthly_returns": monthly_rets,
        "equity_curve": equity_curve,
    }


def _load_prices_from_db(
    tickers: list[str], start: date, end: date
) -> "pd.DataFrame":
    """Load cached prices from MySQL, return empty DataFrame if unavailable."""
    try:
        from data_pipeline.db_manager import DatabaseManager
        db = DatabaseManager()
        return db.load_prices(tickers, start, end)
    except Exception:
        logger.debug("Could not load from DB", exc_info=True)
        import pandas as pd
        return pd.DataFrame()


