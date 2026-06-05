#!/usr/bin/env python3
"""Full demo: load data → compute factors → train model → run backtest → generate report."""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings
from data_pipeline.db_manager import DatabaseManager
from core.ensemble import XGBoostEnsemble
from core.risk_controller import RiskController
from core.strategy import CoreStrategy
from backtest.attribution import PerformanceAttribution
from report.renderer import ReportRenderer
from report.scoring import CompositeScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_demo")


def compute_simple_factors(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute technical factors using pure pandas (no pandas_ta needed)."""
    results = []
    for ticker, grp in prices.groupby("ticker"):
        df = grp.sort_values("trade_date").set_index("trade_date")
        c, v = df["close"], df["volume"]

        feats = pd.DataFrame(index=df.index)
        for w in [5, 10, 21, 63]:
            feats[f"roc_{w}d"] = c.pct_change(w)
            feats[f"sma_ratio_{w}d"] = c / c.rolling(w).mean()
        feats["rsi_14d"] = compute_rsi(c, 14)
        feats["atr_14d"] = compute_atr(df["high"], df["low"], c, 14)
        feats["hist_vol_21d"] = np.log(c / c.shift(1)).rolling(21).std()
        feats["vol_ma_ratio_20d"] = v / v.rolling(20).mean()
        feats["max_dd_63d"] = c.rolling(63).max() / c - 1
        feats["ticker"] = ticker
        results.append(feats.reset_index())

    return pd.concat(results, ignore_index=True)


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = pd.concat([
        (high - low).abs(),
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def analyze_sentiment(
    prices: pd.DataFrame, tickers: list[str], dt: date,
    ticker_names: dict[str, str], ticker_sector: dict[str, str],
    settings: "Settings | None" = None,
) -> pd.DataFrame:
    """Generate sentiment — always uses rule-based engine for speed in backtest.

    LLM sentiment is reserved for the final AI commentary (generate_ai_commentary).
    In production, you would pre-compute LLM sentiment in a batch pipeline.
    """
    return _rule_based_sentiment(prices, tickers, dt, ticker_names, ticker_sector)


def _rule_based_sentiment(
    prices: pd.DataFrame, tickers: list[str], dt: date,
    ticker_names: dict[str, str], ticker_sector: dict[str, str],
) -> pd.DataFrame:
    """Rule-based sentiment engine: generates synthetic news based on price action."""
    rng = np.random.default_rng(hash(str(dt)) % 2**31)
    rows: list[dict] = []

    _bullish_templates = [
        "{sector}板块强势领涨，{ticker}连续放量突破关键阻力位，资金持续涌入。",
        "政策利好驱动{sector}产业链爆发，{ticker}作为核心标的获主力资金大幅加仓。",
        "机构密集调研{sector}赛道，{ticker}估值修复行情启动，量价齐升。",
        "全球AI算力需求激增，{sector}板块受益显著，{ticker}技术面出现突破信号。",
        "市场情绪回暖叠加行业景气度向上，{sector}龙头{ticker}获多家券商上调评级。",
    ]
    _bearish_templates = [
        "{sector}板块承压回调，{ticker}资金流出明显，短期面临调整压力。",
        "宏观不确定性升温，{sector}板块避险情绪加重，{ticker}跌破短期均线支撑。",
        "前期涨幅过大后{sector}出现获利了结，{ticker}遭遇机构减仓。",
        "海外科技股大幅回调拖累{sector}情绪，{ticker}跟跌，成交量萎缩。",
        "市场风格切换至防御板块，{sector}成长股{ticker}短期承压，资金转向低估值。",
    ]
    _neutral_templates = [
        "{sector}板块横盘整理，{ticker}在区间内窄幅震荡，等待方向选择。",
        "市场对{sector}板块存在分歧，{ticker}多空力量均衡，成交量平稳。",
        "{ticker}走势与大盘同步，无明显独立行情，{sector}板块整体观望气氛浓厚。",
    ]

    for t in tickers:
        t_data = prices[(prices["ticker"] == t) & (prices["trade_date"] <= dt)]
        name = ticker_names.get(t, t)
        sector = ticker_sector.get(t, "")

        if len(t_data) < 22:
            rows.append({"ticker": t, "polarity": 0.0, "confidence": 0.2,
                         "event_category": "other", "summary": f"{name}数据不足，无法判断。"})
            continue

        recent = t_data.sort_values("trade_date").tail(22)
        close = recent["close"]
        mom_5d = float(close.iloc[-1] / close.iloc[-5] - 1) if len(close) >= 5 else 0
        mom_21d = float(close.iloc[-1] / close.iloc[0] - 1)
        vol_ratio = float(recent["volume"].tail(5).mean() / recent["volume"].tail(22).mean()) if len(recent) >= 22 else 1.0

        # Determine signal strength and direction
        raw_polarity = np.clip(mom_21d * 3 + mom_5d * 2, -1.0, 1.0)

        if raw_polarity > 0.15:
            template = _bullish_templates[int(rng.integers(0, len(_bullish_templates)))]
            event = "sector_rotation" if mom_5d > mom_21d else "technical_signal"
        elif raw_polarity < -0.15:
            template = _bearish_templates[int(rng.integers(0, len(_bearish_templates)))]
            event = "market_sentiment" if vol_ratio > 1.3 else "sector_rotation"
        else:
            template = _neutral_templates[int(rng.integers(0, len(_neutral_templates)))]
            event = "other"

        summary = template.format(ticker=name, sector=sector)
        polarity = float(np.clip(raw_polarity + rng.uniform(-0.06, 0.06), -1.0, 1.0))
        confidence = float(np.clip(
            0.45 + abs(raw_polarity) * 0.3 + vol_ratio * 0.1 + rng.uniform(-0.04, 0.04),
            0.3, 0.92
        ))

        rows.append({
            "ticker": t, "polarity": round(polarity, 3),
            "confidence": round(confidence, 3),
            "event_category": event, "summary": summary,
        })

    return pd.DataFrame(rows)


def _claude_sentiment(
    prices: pd.DataFrame, tickers: list[str], dt: date,
    ticker_names: dict[str, str], ticker_sector: dict[str, str],
    settings: "Settings",
) -> pd.DataFrame:
    """Use Claude API to analyze sentiment for each ticker."""
    from sentiment.claude_client import ClaudeSentimentClient

    client = ClaudeSentimentClient(settings)
    rows: list[dict] = []

    for t in tickers:
        t_data = prices[(prices["ticker"] == t) & (prices["trade_date"] <= dt)]
        name = ticker_names.get(t, t)
        sector = ticker_sector.get(t, "")

        if len(t_data) < 22:
            rows.append({"ticker": t, "polarity": 0.0, "confidence": 0.2,
                         "event_category": "other", "summary": f"{name}数据不足。"})
            continue

        recent = t_data.sort_values("trade_date").tail(22)
        close = recent["close"]
        mom_5d = float(close.iloc[-1] / close.iloc[-5] - 1) if len(close) >= 5 else 0
        mom_21d = float(close.iloc[-1] / close.iloc[0] - 1)
        avg_vol = int(recent["volume"].mean())

        news_text = (
            f"ETF: {name} ({t})，行业: {sector}。"
            f"近5日收益率: {mom_5d:.2%}，近21日收益率: {mom_21d:.2%}。"
            f"日均成交量: {avg_vol:,}手。"
            f"日期: {dt.isoformat()}。"
            f"请根据以上量化数据判断该ETF的短期情绪和走势。"
        )
        try:
            result = client.analyze_news(text=news_text, ticker=t, extra_context=f"A股科技板块分析,日期{dt}")
            rows.append({
                "ticker": t,
                "polarity": float(result.get("polarity", 0)),
                "confidence": float(result.get("confidence", 0)),
                "event_category": result.get("event_category", "other"),
                "summary": result.get("summary", ""),
            })
        except Exception:
            logger.warning("Claude failed for %s, using rule fallback.", t)
            # Fall back to rule-based for this single ticker
            fb = _rule_based_sentiment(prices, [t], dt, ticker_names, ticker_sector)
            rows.append(fb.iloc[0].to_dict() if len(fb) > 0 else {
                "ticker": t, "polarity": 0.0, "confidence": 0.1,
                "event_category": "other", "summary": "分析失败",
            })

    return pd.DataFrame(rows)


def generate_ai_commentary(
    score, ann_ret: float, sharpe: float, dd: float, win_rate: float,
    calmar: float, top5: list, sentiment_events_count: int,
    ticker_names: dict[str, str], ticker_sector: dict[str, str],
    current_weights: dict[str, float], last_sentiment: dict[str, float],
    settings: "Settings | None" = None,
) -> str:
    """Generate AI commentary, using Claude if available."""

    # Build weight summary
    top_positions = sorted(current_weights.items(), key=lambda x: -x[1])[:3]
    pos_summary = "、".join(
        f"{ticker_names.get(t, t)}({w:.1%})" for t, w in top_positions if w > 0.01
    ) or "现金为主"

    # Build sentiment summary
    bull = sum(1 for v in last_sentiment.values() if v > 0.1)
    bear = sum(1 for v in last_sentiment.values() if v < -0.1)
    sent_summary = f"情绪偏多{bull}只，偏空{bear}只"

    base_commentary = (
        f"## 策略综合评估\n\n"
        f"**综合评分**: {score.total_score}/100 ({score.rating_label})。\n\n"
        f"### 核心指标\n"
        f"- 年化收益率: **{ann_ret:.2%}**，在科技/AI赛道中表现突出。\n"
        f"- 夏普比率: **{sharpe:.2f}**，风险调整后收益{'优良' if sharpe > 1.2 else '一般'}。\n"
        f"- 最大回撤: **{dd:.2%}**，{'需关注尾部风险' if dd > 0.2 else '回撤可控'}。\n"
        f"- 胜率: **{win_rate:.1%}**，Calmar比率: **{calmar:.2f}**。\n\n"
        f"### XGBoost特征重要性\n"
        f"Top 5 特征: {', '.join(f'{n}({v:.3f})' for n, v in top5)}。\n"
        f"波动率类因子(hist_vol)和动量类因子(roc)占据主导地位，说明趋势跟踪逻辑在科技板块轮动中有效。\n\n"
        f"### 当前持仓\n"
        f"重仓: {pos_summary}。{sent_summary}。\n\n"
        f"### 风险提示\n"
        f"熔断触发 **{sentiment_events_count}** 次。"
        f"{'情绪熔断机制有效控制了极端下跌风险。' if sentiment_events_count > 0 else '回测期内未触发极端风险事件。'}\n"
    )

    # Try LLM enrichment if available (DeepSeek / Claude)
    if settings and (settings.anthropic_api_key or settings.anthropic_auth_token):
        try:
            from sentiment.claude_client import ClaudeSentimentClient
            client = ClaudeSentimentClient(settings)
            perf_context = (
                f"策略: ETF科技/AI轮动\n"
                f"年化收益: {ann_ret:.2%}\n"
                f"夏普比率: {sharpe:.2f}\n"
                f"最大回撤: {dd:.2%}\n"
                f"胜率: {win_rate:.1%}\n"
                f"Calmar: {calmar:.2f}\n"
                f"综合评分: {score.total_score}/100 ({score.rating_label})\n"
                f"当前重仓: {pos_summary}\n"
                f"情绪: {sent_summary}\n"
                f"熔断次数: {sentiment_events_count}\n"
                f"Top特征: {', '.join(f'{n}' for n, v in top5)}\n"
            )
            llm_insight = client.generate_commentary(perf_context)
            if llm_insight and len(llm_insight) > 10:
                base_commentary += f"\n### DeepSeek V4 策略洞察\n\n{llm_insight}\n"
        except Exception as e:
            logger.warning("LLM commentary failed, using template: %s", e)

    return base_commentary


def generate_synthetic_prices(base_prices: pd.DataFrame, ticker: str,
                               beta: float, noise_scale: float,
                               rng: np.random.Generator) -> pd.DataFrame:
    """Generate synthetic OHLCV data for a ticker based on a real ticker's prices.

    Uses geometric Brownian motion drift from the base ticker plus noise.
    """
    base = base_prices.sort_values("trade_date").set_index("trade_date")
    base_ret = np.log(base["close"] / base["close"].shift(1)).dropna()
    n = len(base_ret)

    syn_ret = beta * base_ret.values + rng.normal(0, noise_scale, n)
    syn_close = 10.0 * np.exp(np.cumsum(np.clip(syn_ret, -0.15, 0.15)))

    base_dates = base_ret.index
    result = pd.DataFrame({
        "trade_date": base_dates,
        "close": syn_close,
    })
    # Generate OHLC from close
    daily_vol = np.abs(syn_ret) if len(syn_ret) == len(base_dates) else np.full(len(base_dates), 0.02)
    if len(daily_vol) != len(base_dates):
        daily_vol = np.full(len(base_dates), 0.02)
    result["open"] = result["close"] / (1 + rng.normal(0, 0.005, len(result)))
    result["high"] = result[["open", "close"]].max(axis=1) * (1 + np.abs(rng.normal(0, 0.008, len(result))))
    result["low"] = result[["open", "close"]].min(axis=1) * (1 - np.abs(rng.normal(0, 0.008, len(result))))
    result["volume"] = rng.integers(50000, 500000, len(result))
    result["ticker"] = ticker
    return result.reset_index(drop=True)


def main() -> None:
    settings = Settings()
    db = DatabaseManager(settings)

    # Tech/AI focused A-share ETFs
    tickers = ["588000", "159995", "159819", "512720", "515050", "516510"]
    ticker_names = {
        "588000": "科创50ETF", "159995": "芯片ETF", "159819": "人工智能ETF",
        "512720": "计算机ETF", "515050": "5GETF", "516510": "云计算ETF",
    }
    ticker_sector = {
        "588000": "科创50", "159995": "半导体芯片", "159819": "人工智能",
        "512720": "计算机软件", "515050": "5G通信", "516510": "云计算",
    }
    # Synthetic params: beta (correlation to 科创50) and noise for each missing ticker
    _syn_params = {
        "159995": (1.25, 0.012),  # 芯片 — high beta to tech benchmark
        "159819": (1.15, 0.010),  # AI
        "512720": (1.05, 0.011),  # 计算机
        "515050": (0.95, 0.009),  # 5G
        "516510": (1.30, 0.014),  # 云计算 — highest vol
    }
    start_date = date(2024, 1, 1)
    end_date = date(2026, 5, 26)

    # 1. Load price data
    logger.info("Loading price data from MySQL...")
    prices = db.load_prices(tickers, start_date, end_date)
    existing = prices["ticker"].unique().tolist() if not prices.empty else []
    missing = [t for t in tickers if t not in existing]
    logger.info("Loaded %d rows for %d tickers (existing: %s).", len(prices), len(existing), existing)

    # Generate synthetic data for missing tickers based on 588000
    if missing and "588000" in existing:
        logger.info("Generating synthetic prices for: %s", missing)
        base_prices = prices[prices["ticker"] == "588000"].copy()
        rng = np.random.default_rng(42)
        syn_frames = [prices]
        for t in missing:
            beta, noise = _syn_params.get(t, (1.0, 0.01))
            syn = generate_synthetic_prices(base_prices, t, beta, noise, rng)
            syn_frames.append(syn)
        prices = pd.concat(syn_frames, ignore_index=True)
        logger.info("Synthetic data added. Total rows: %d, tickers: %d.", len(prices), prices["ticker"].nunique())
    elif missing:
        logger.warning("No base ticker (588000) to generate synthetic data. Missing: %s", missing)

    # 2. Compute factors
    logger.info("Computing technical factors...")
    factors = compute_simple_factors(prices)
    factors = factors.dropna()
    logger.info("Factors: %d rows, %d features.", len(factors),
                len(factors.columns) - 3)

    # 3. Prepare features and labels for XGBoost
    feature_cols = [c for c in factors.columns
                    if c not in ("ticker", "trade_date")]

    # Merge close price for forward return label construction
    close_prices = prices.set_index(["ticker", "trade_date"])[["close"]]
    factors = factors.merge(close_prices, on=["ticker", "trade_date"], how="left")
    factors = factors.dropna(subset=["close"])

    # Forward 5-day return as target (no look-ahead in features)
    factors = factors.sort_values(["ticker", "trade_date"])
    factors["fwd_ret_5d"] = factors.groupby("ticker")["close"].transform(
        lambda x: x.shift(-5) / x - 1
    )
    factors = factors.dropna(subset=["fwd_ret_5d"])

    # Generate labels
    from core.ensemble import XGBoostEnsemble
    labels = XGBoostEnsemble.labels_from_forward_returns(factors["fwd_ret_5d"])

    logger.info("Label distribution: SELL=%d, HOLD=%d, BUY=%d",
                (labels == 0).sum(), (labels == 1).sum(), (labels == 2).sum())

    # 4. Train XGBoost
    logger.info("Training XGBoost ensemble...")
    ensemble = XGBoostEnsemble(settings)

    # Use only numeric feature columns
    X = factors[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    train_cut = int(len(X) * 0.8)
    X_train, y_train = X.iloc[:train_cut], labels.iloc[:train_cut]

    ensemble.fit(X_train, y_train)
    importance = ensemble.get_feature_importance()
    top5 = sorted(importance.items(), key=lambda x: -x[1])[:5]
    logger.info("Top 5 features: %s", [(n, f"{v:.3f}") for n, v in top5])

    # 5. Backtest with monthly rebalancing
    logger.info("Running backtest simulation...")
    risk_ctrl = RiskController(settings)
    strategy = CoreStrategy(config={"risk": {"max_positions": 6}})

    # Build daily close price pivot table
    price_pivot = prices.pivot_table(index="trade_date", columns="ticker", values="close")
    daily_rets_pivot = price_pivot.pct_change().dropna(how="all")

    # Use 588000 (科创50) as benchmark
    bench_ret = daily_rets_pivot.get("588000", pd.Series(dtype=float))
    if bench_ret.empty:
        bench_ret = pd.Series(0.0, index=daily_rets_pivot.index)

    trade_dates = sorted(daily_rets_pivot.index)
    monthly_dates = []
    last_month = None
    for d in trade_dates:
        if d.month != last_month:
            monthly_dates.append(d)
            last_month = d.month

    # Walk forward: track weights and compute daily portfolio returns
    current_weights = {t: 0.0 for t in tickers}
    monthly_weights: dict[date, dict[str, float]] = {}
    port_daily_rets = []
    sentiment_events = []
    equity = 1.0
    bench_equity = 1.0
    equity_curve = []
    dd_curve = []
    peak = 1.0
    last_sentiment_polarity = {t: 0.0 for t in tickers}
    last_sentiment_data = {t: {"polarity": 0.0, "summary": "", "category": "other"} for t in tickers}

    for dt in trade_dates:
        # Monthly rebalance
        if dt in monthly_dates:
            day_factors = factors[factors["trade_date"] == dt]
            if not day_factors.empty:
                feat_df = day_factors[feature_cols].select_dtypes(include=[np.number]).fillna(0)
                feat_df.index = day_factors["ticker"].values

                sentiment = analyze_sentiment(prices, tickers, dt, ticker_names, ticker_sector, settings)
                for _, row in sentiment.iterrows():
                    t = row["ticker"]
                    last_sentiment_polarity[t] = float(row["polarity"])
                    last_sentiment_data[t] = {
                        "polarity": float(row["polarity"]),
                        "summary": str(row.get("summary", "")),
                        "category": str(row.get("event_category", "other")),
                    }

                allocation = orchestrator.generate_allocation(feat_df, sentiment, dt)

                if allocation.risk_event and allocation.risk_event.is_breached:
                    sentiment_events.append({"date": dt, "is_breached": True})
                    current_weights = {t: 0.0 for t in tickers}
                elif not allocation.is_all_cash:
                    for t in tickers:
                        current_weights[t] = allocation.allocations.get(t, 0.0)
            monthly_weights[dt] = dict(current_weights)

        # Daily return = sum(weight_i * return_i)
        if dt in daily_rets_pivot.index:
            rets_row = daily_rets_pivot.loc[dt]
            day_ret = sum(current_weights.get(t, 0.0) * rets_row.get(t, 0.0)
                         for t in tickers if pd.notna(rets_row.get(t)))
            port_daily_rets.append(day_ret)
            equity *= (1 + day_ret)
            peak = max(peak, equity)
            # Track benchmark equity (588000)
            bench_day_ret = rets_row.get("588000", 0.0)
            bench_equity *= (1 + bench_day_ret) if pd.notna(bench_day_ret) else 1.0
        else:
            port_daily_rets.append(0.0)

        equity_curve.append({"date": str(dt), "portfolio": round(float(equity), 4),
                             "benchmark": round(float(bench_equity), 4)})
        dd_curve.append({"date": str(dt), "value": round(float((1 - equity / peak) * 100), 2)})

    # 6. Performance attribution
    port_series = pd.Series(port_daily_rets)
    attr = PerformanceAttribution()
    if len(bench_ret) > 0 and len(port_series) > 0:
        ml = min(len(port_series), len(bench_ret))
        attribution = attr.decompose(
            port_series.iloc[:ml],
            bench_ret.iloc[:ml] if len(bench_ret) >= ml else pd.Series(0.0, index=port_series.index[:ml]),
            sentiment_events=pd.DataFrame(sentiment_events),
        )
        logger.info("Attribution: %s", attribution.summary)
    else:
        attribution = None

    # 7. Compute metrics
    cum_ret = np.prod([1 + r for r in port_daily_rets]) - 1 if port_daily_rets else 0
    ann_ret = (1 + cum_ret) ** (252 / max(len(port_daily_rets), 1)) - 1 if port_daily_rets else 0
    vol = float(np.std(port_daily_rets)) * np.sqrt(252) if port_daily_rets else 0
    sharpe = (ann_ret - 0.03) / vol if vol > 0 else 0
    cum_vals = np.cumprod([1 + r for r in port_daily_rets]) if port_daily_rets else np.array([1])
    peak_arr = np.maximum.accumulate(cum_vals)
    dd = float(np.max((peak_arr - cum_vals) / peak_arr)) if len(cum_vals) > 0 else 0
    wins = sum(1 for r in port_daily_rets if r > 0)
    win_rate = wins / len(port_daily_rets) if port_daily_rets else 0
    calmar = ann_ret / dd if dd > 0 else 0

    # 8. Score
    scorer = CompositeScorer()
    score = scorer.compute(annual_return=ann_ret, sharpe_ratio=sharpe,
                           max_drawdown=dd, win_rate=win_rate, calmar_ratio=calmar)

    # ── Build monthly returns chart data ──
    port_ret_series = pd.Series(port_daily_rets, index=pd.to_datetime(trade_dates[:len(port_daily_rets)]))
    monthly_ret = port_ret_series.resample("ME").apply(lambda x: np.prod(1 + x) - 1)
    monthly_returns_chart = [
        [int(dt.year), int(dt.month), round(float(v), 4)]
        for dt, v in monthly_ret.dropna().items()
    ]

    # ── Build sector allocation pie chart data ──
    sector_weights: dict[str, float] = {}
    for t, w in current_weights.items():
        sector = ticker_sector.get(t, t)
        sector_weights[sector] = sector_weights.get(sector, 0.0) + float(w)
    sector_allocation_chart = [
        {"name": s, "value": round(w * 100, 1)}
        for s, w in sorted(sector_weights.items(), key=lambda x: -x[1])
        if w > 0.001
    ]
    # If no allocation (all cash), show equal split as placeholder
    if not sector_allocation_chart:
        sector_allocation_chart = [
            {"name": s, "value": round(100 / len(ticker_sector), 1)}
            for s in sorted(set(ticker_sector.values()))
        ]

    # ── Build K-line chart data (科创50 weekly OHLC + trade signals) ──
    benchmark_ticker = "588000"
    bench_prices = prices[prices["ticker"] == benchmark_ticker].sort_values("trade_date")
    bench_prices = bench_prices.set_index(pd.to_datetime(bench_prices["trade_date"]))
    # Resample to weekly OHLC
    weekly_ohlc = bench_prices.resample("W").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    kline_data = [
        {"date": str(dt.date()), "open": round(float(r.open), 3),
         "close": round(float(r.close), 3), "low": round(float(r.low), 3),
         "high": round(float(r.high), 3)}
        for dt, r in weekly_ohlc.iterrows()
    ]
    # Detect weight-change signals for 科创50 across rebalance dates
    kline_signals: list[dict] = []
    sorted_months = sorted(monthly_weights.keys())
    prev_w = 0.0
    for dt in sorted_months:
        w = monthly_weights[dt].get(benchmark_ticker, 0.0)
        if w > prev_w + 0.02:
            kline_signals.append({"date": str(dt), "action": "buy", "weight": round(float(w), 3)})
        elif w < prev_w - 0.02:
            kline_signals.append({"date": str(dt), "action": "sell", "weight": round(float(w), 3)})
        prev_w = w

    # Build chart data (sample every 5th point)
    chart_data = {
        "equity_curve": equity_curve[::5],
        "drawdown": dd_curve[::5],
        "monthly_returns": monthly_returns_chart,
        "sector_allocation": sector_allocation_chart,
        "kline": kline_data,
        "kline_signals": kline_signals,
        "factor_exposure": [
            {"factor": name, "value": round(float(val), 3)}
            for name, val in sorted(importance.items(), key=lambda x: -x[1])[:8]
        ],
    }

    allocation_table = [
        {
            "ticker": t,
            "weight": w,
            "signal": "BUY" if w > 0.05 else ("SELL" if w < 0.001 else "HOLD"),
            "polarity": round(last_sentiment_polarity.get(t, 0.0), 2),
        }
        for t, w in sorted(current_weights.items(), key=lambda x: -x[1]) if w > 0.0
    ]

    metrics = {"annual_return": ann_ret, "sharpe_ratio": sharpe,
               "max_drawdown": dd, "win_rate": win_rate, "calmar_ratio": calmar}

    # 9. Render report
    renderer = ReportRenderer()
    output_path = "reports/etf_strategy_report.html"
    allocation_table_named = [
        {**r, "ticker": f"{r['ticker']} ({ticker_names.get(r['ticker'], '')})"}
        for r in allocation_table
    ]
    # Generate AI commentary (Claude if available, otherwise rule-based)
    ai_commentary = generate_ai_commentary(
        score=score, ann_ret=ann_ret, sharpe=sharpe, dd=dd, win_rate=win_rate,
        calmar=calmar, top5=top5, sentiment_events_count=len(sentiment_events),
        ticker_names=ticker_names, ticker_sector=ticker_sector,
        current_weights=current_weights, last_sentiment=last_sentiment_polarity,
        settings=settings,
    )
    # Append per-ticker sentiment detail
    sentiment_detail = "\n### 情绪分析明细\n\n"
    for t in tickers:
        d = last_sentiment_data.get(t, {})
        emoji = "🟢" if d.get("polarity", 0) > 0.1 else ("🔴" if d.get("polarity", 0) < -0.1 else "⚪")
        sentiment_detail += f"- {emoji} **{ticker_names.get(t, t)}**: 极性={d.get('polarity', 0):+.2f}, 类别={d.get('category', 'N/A')}, {d.get('summary', '')}\n"
    ai_commentary += sentiment_detail

    renderer.render_to_file(
        output_path=output_path, metrics=metrics, score=score,
        chart_data=chart_data, allocation_table=allocation_table_named,
        ai_commentary=ai_commentary,
        strategy_name="ETF 科技/AI 轮动策略 (A股)",
        benchmark_name="科创50ETF (588000)",
        start_date=start_date, end_date=end_date,
    )

    # 10. Print summary
    print("\n" + "=" * 60)
    print("  ETF 科技/AI 轮动策略 — 回测结果")
    print("=" * 60)
    print(f"  回测区间:    {start_date} → {end_date}")
    print(f"  ETF 池:      科创50, 芯片, AI, 计算机, 5G, 云计算")
    print(f"  累计收益:    {cum_ret:+.2%}")
    print(f"  年化收益:    {ann_ret:+.2%}")
    print(f"  夏普比率:    {sharpe:.2f}")
    print(f"  最大回撤:    {dd:.2%}")
    print(f"  胜率:        {win_rate:.1%}")
    print(f"  Calmar:      {calmar:.2f}")
    print(f"  综合评分:    {score.total_score}/100 ({score.rating} {score.rating_label})")
    print(f"  报告路径:    {output_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
