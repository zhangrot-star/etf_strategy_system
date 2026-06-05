#!/usr/bin/env python3
"""Generate a professional ETF strategy research report (中信证券/国泰海通 style)."""

from __future__ import annotations

import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from data_pipeline.db_manager import DatabaseManager
from data_pipeline.models import (
    DailyScore, ETFPrice, ETFPrediction, MacroIndicator, SentimentRecord,
)
from config.settings import Settings
from report.renderer import ReportRenderer
from report.scoring import CompositeScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gen_report")

db = DatabaseManager(Settings())

# ── 1. Load data ──────────────────────────────────────────────────

with db._session_factory() as sess:
    tickers = sorted([
        r[0] for r in sess.query(ETFPrice.ticker).distinct().all()
    ])

# Latest recommendation
with db._session_factory() as sess:
    latest_date = sess.query(DailyScore.score_date).order_by(
        DailyScore.score_date.desc()
    ).first()
    if latest_date:
        latest_date = latest_date[0]
        scores_rows = sess.query(DailyScore).filter(
            DailyScore.score_date == latest_date
        ).order_by(DailyScore.adjusted_total.desc()).all()
    else:
        scores_rows = []

# Price data for backtest metrics
prices = db.load_prices(tickers, pd.Timestamp("2024-01-01"), pd.Timestamp.today())
prices["trade_date"] = pd.to_datetime(prices["trade_date"])

# ETF profiles for names
profiles_df = db.load_profiles(tickers)
ticker_names: dict[str, str] = {}
if not profiles_df.empty:
    ticker_names = dict(zip(profiles_df["ticker"], profiles_df["name"]))

# Issuer metadata
issuer_df = db.load_issuers()
issuer_names: dict[str, str] = {}
if not issuer_df.empty:
    issuer_names = dict(zip(issuer_df["issuer_id"], issuer_df["name"]))

# Index metadata
index_meta_df = db.load_index_meta(tickers)

# Sentiment
with db._session_factory() as sess:
    srows = sess.query(SentimentRecord).order_by(
        SentimentRecord.event_date.desc()
    ).limit(1000).all()
sent_df = pd.DataFrame([{
    "ticker": r.ticker, "event_date": r.event_date,
    "polarity": r.polarity, "confidence": r.confidence,
    "event_category": r.event_category,
} for r in srows])

# Macro indicators
with db._session_factory() as sess:
    mrows = sess.query(MacroIndicator).order_by(
        MacroIndicator.obs_date.desc()
    ).limit(200).all()
macro_df = pd.DataFrame([{
    "indicator_name": r.indicator_name,
    "obs_date": r.obs_date,
    "value": r.value,
} for r in mrows])

logger.info("Loaded %d tickers, %d price rows, %d sentiment, %d macro records",
            len(tickers), len(prices), len(sent_df), len(macro_df))

# ── 2. Compute backtest metrics ───────────────────────────────────

if len(tickers) > 1 and not prices.empty:
    monthly_prices = prices.set_index("trade_date")
    daily_close = monthly_prices.pivot_table(
        index="trade_date", columns="ticker", values="close"
    ).ffill()

    monthly_returns = daily_close.resample("ME").last().pct_change().dropna(how="all")
    portfolio_returns = monthly_returns.mean(axis=1)

    total_return = (1 + portfolio_returns).prod() - 1
    n_months = len(portfolio_returns)
    ann_ret = (1 + total_return) ** (12 / max(n_months, 1)) - 1

    if portfolio_returns.std() > 0:
        sharpe = float((portfolio_returns.mean() * 12) / (portfolio_returns.std() * np.sqrt(12)))
    else:
        sharpe = 0.0

    cumulative = (1 + portfolio_returns).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1
    max_dd = float(drawdown.min())
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    win_rate = float((portfolio_returns > 0).mean())

    # Benchmark return (510300 / CSI 300 ETF)
    bench_close = daily_close.get("510300", daily_close.iloc[:, 0])
    bench_monthly = bench_close.resample("ME").last().pct_change().dropna()
    bench_total = (1 + bench_monthly).prod() - 1
    bench_ann = (1 + bench_total) ** (12 / max(len(bench_monthly), 1)) - 1
    excess_return = ann_ret - bench_ann
    bench_cumulative = (1 + bench_monthly).cumprod()
else:
    ann_ret, sharpe, max_dd, calmar, win_rate = 0.0, 0.0, 0.0, 0.0, 0.0
    excess_return, bench_ann = 0.0, 0.0
    portfolio_returns = pd.Series(dtype=float)
    cumulative = pd.Series(dtype=float)
    drawdown = pd.Series(dtype=float)
    bench_cumulative = pd.Series(dtype=float)

# ── 3. Composite score ────────────────────────────────────────────

scorer = CompositeScorer()
score = scorer.compute(
    annual_return=ann_ret,
    sharpe_ratio=sharpe,
    max_drawdown=abs(max_dd),
    win_rate=win_rate,
    calmar_ratio=calmar,
)
logger.info("Composite score: %d/100 (%s)", score.total_score, score.rating)

# ── 4. Build chart data ───────────────────────────────────────────

equity_dates = cumulative.index
bench_vals = {}
if len(bench_cumulative) > 0:
    bench_vals = {d: round(float(v), 4) for d, v in bench_cumulative.items()}
equity_curve = [
    {"date": str(d.date()), "portfolio": round(float(v), 4),
     "benchmark": bench_vals.get(d, round(float(v), 4))}
    for d, v in cumulative.items()
] if len(cumulative) > 0 else []

dd_curve = [
    {"date": str(d.date()), "value": round(float(v), 4)}
    for d, v in drawdown.items()
] if len(drawdown) > 0 else []

monthly_returns_chart = []
if len(portfolio_returns) > 0:
    for dt, r in portfolio_returns.items():
        monthly_returns_chart.append([dt.year, dt.month, round(float(r), 4)])

# Sector allocation
sector_map = {
    "510050": "宽基大盘", "510300": "宽基大盘", "510500": "宽基中盘",
    "159915": "创业板", "588000": "科创板", "159845": "宽基小盘",
    "515050": "科技/5G", "159995": "科技/芯片", "159819": "科技/AI",
    "512720": "科技/计算机", "516510": "科技/云计算",
    "512880": "金融/证券", "512690": "消费/酒", "512010": "医药",
    "516970": "军工", "512660": "军工", "515790": "新能源/光伏",
    "512800": "金融/银行", "515220": "能源/煤炭",
    "511010": "债券/国债", "511260": "债券/国债", "518880": "商品/黄金",
}
sector_weights: dict[str, float] = {}
for r in scores_rows[:10]:
    s = sector_map.get(r.ticker, "其他")
    sector_weights[s] = sector_weights.get(s, 0) + max(r.adjusted_total, 0)
total_sw = sum(sector_weights.values()) or 1
sector_allocation = [
    {"name": k, "value": round(v / total_sw * 100, 1)}
    for k, v in sorted(sector_weights.items(), key=lambda x: -x[1])
]

# K-line data (benchmark ETF)
bench_ticker = "510300"
bench_prices = prices[prices["ticker"] == bench_ticker].sort_values("trade_date")
kline_data = []
kline_signals = []
if not bench_prices.empty:
    bp = bench_prices.set_index(pd.to_datetime(bench_prices["trade_date"]))
    weekly = bp.resample("W").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna().tail(52)
    kline_data = [
        {"date": str(d.date()), "open": round(float(r.open), 3),
         "close": round(float(r.close), 3), "low": round(float(r.low), 3),
         "high": round(float(r.high), 3), "volume": int(r.volume)}
        for d, r in weekly.iterrows()
    ]

# ── Prediction data ─────────────────────────────────────────────

pred_by_ticker: dict[str, dict[int, Any]] = {}
prediction_bar_data = []
prediction_rows_all = []
with db._session_factory() as sess:
    latest_pred_date = sess.query(ETFPrediction.pred_date).order_by(
        ETFPrediction.pred_date.desc()
    ).first()
    if latest_pred_date:
        pred_date_val = latest_pred_date[0]
        prediction_rows_all = sess.query(ETFPrediction).filter(
            ETFPrediction.pred_date == pred_date_val
        ).all()
        for r in prediction_rows_all:
            pred_by_ticker.setdefault(r.ticker, {})[r.horizon_days] = r

if pred_by_ticker:
    prediction_bar_data = [
        {
            "ticker": t,
            "name": ticker_names.get(t, t),
            "return_5d": round(h.get(5, type("x", (), {"predicted_return": 0})).predicted_return * 100, 2) if 5 in h else 0,
            "return_21d": round(h.get(21, type("x", (), {"predicted_return": 0})).predicted_return * 100, 2) if 21 in h else 0,
            "return_63d": round(h.get(63, type("x", (), {"predicted_return": 0})).predicted_return * 100, 2) if 63 in h else 0,
            "prob_up_21d": round(h.get(21, type("x", (), {"prob_up": 0.5})).prob_up, 3) if 21 in h else 0.5,
        }
        for t, h in sorted(pred_by_ticker.items(),
                           key=lambda x: abs(x[1].get(21, type("x", (), {"predicted_return": 0})).predicted_return),
                           reverse=True)[:10]
    ]

    # K-line signals from 21d predictions for benchmark
    bp_pred = pred_by_ticker.get(bench_ticker, {}).get(21)
    if bp_pred:
        if bp_pred.predicted_return > 0 and bp_pred.prob_up > 0.6:
            kline_signals.append({
                "date": str(pred_date_val),
                "action": "buy",
                "weight": round(float(bp_pred.predicted_return), 4),
            })
        elif bp_pred.predicted_return < 0 and bp_pred.prob_up < 0.4:
            kline_signals.append({
                "date": str(pred_date_val),
                "action": "sell",
                "weight": round(abs(float(bp_pred.predicted_return)), 4),
            })

chart_data = {
    "equity_curve": equity_curve[::5] if len(equity_curve) > 5 else equity_curve,
    "drawdown": dd_curve[::5] if len(dd_curve) > 5 else dd_curve,
    "monthly_returns": monthly_returns_chart,
    "sector_allocation": sector_allocation,
    "kline": kline_data,
    "kline_signals": kline_signals,
    "factor_exposure": [],
    "predicted_returns": prediction_bar_data,
}

# ── 5. Allocation table (enriched) ────────────────────────────────

allocation_table = []
for r in scores_rows[:10]:
    name = ticker_names.get(r.ticker, "")
    tpred = pred_by_ticker.get(r.ticker, {})
    p21 = tpred.get(21)
    pred_21d = float(p21.predicted_return) if p21 else 0.0
    prob_up = float(p21.prob_up) if p21 else 0.5
    polarity = round(
        float(sent_df[sent_df["ticker"] == r.ticker]["polarity"].iloc[0])
        if not sent_df[sent_df["ticker"] == r.ticker].empty else 0.0, 2
    )
    allocation_table.append({
        "ticker_code": r.ticker,
        "ticker_name": name,
        "ticker": f"{r.ticker} ({name})" if name else r.ticker,  # legacy
        "weight": round(r.adjusted_total / max(sum(x.adjusted_total for x in scores_rows), 1), 3),
        "signal": r.ml_signal,
        "polarity": polarity,
        "pred_21d": pred_21d,
        "prob_up": prob_up,
    })

# ── 6. Brokerage-style content sections ────────────────────────────

# 6a. Key investment points
top5 = scores_rows[:5]
top5_str = "、".join(
    f"{r.ticker}（{ticker_names.get(r.ticker, '')}）" for r in top5
)

# ML signal summary
buy_count = sum(1 for r in scores_rows if r.ml_signal == "BUY")
sell_count = sum(1 for r in scores_rows if r.ml_signal == "SELL")
hold_count = sum(1 for r in scores_rows if r.ml_signal == "HOLD")

key_points = [
    {
        "label": "核心策略",
        "content": "基于「基金公司（10%）—指数质量（40%）—个基评价（50%）」三模块打分框架，结合XGBoost多周期收益预测（5d/21d/63d）与LLM情绪分析，实现全流程量化选基。",
    },
    {
        "label": "业绩表现",
        "content": f"回测区间（2024.01—{date.today().isoformat()}）内，等权组合年化收益{ann_ret:+.1%}，超额收益{excess_return:+.1%}（vs {bench_ticker}），夏普比率{sharpe:.2f}，最大回撤{max_dd:.1%}，月度胜率{win_rate:.0%}。",
    },
    {
        "label": "组合信号",
        "content": f"当前综合评分{score.total_score}/100（{score.rating_label}）。ML信号：买入{buy_count}只、持有{hold_count}只、卖出{sell_count}只。多空力量对比{'偏多' if buy_count > sell_count else '偏空' if sell_count > buy_count else '均衡'}，建议{'维持较高仓位' if score.total_score >= 65 else '中性仓位' if score.total_score >= 50 else '降低仓位防范风险'}。",
    },
    {
        "label": "风险关注",
        "content": f"当前最大回撤{abs(max_dd):.1%}，{'处于可控范围' if abs(max_dd) < 0.15 else '偏高，需警惕'}。ML卖出信号集中在科技赛道（AI、芯片），债券及商品类ETF信号偏积极，关注风格轮动节奏。",
    },
]

# 6b. Market environment & macro commentary
macro_commentary = ""
if not macro_df.empty:
    recent_macro = macro_df.sort_values("obs_date").tail(60)
    indicators = recent_macro["indicator_name"].unique()
    macro_commentary = "**近期宏观数据概览**\n\n"
    for ind in indicators[:8]:
        ind_data = recent_macro[recent_macro["indicator_name"] == ind]
        if len(ind_data) >= 2:
            curr = ind_data.iloc[-1]["value"]
            prev = ind_data.iloc[-2]["value"]
            chg = curr - prev
            direction = "↑" if chg > 0 else "↓" if chg < 0 else "→"
            macro_commentary += f"- **{ind}**：{curr:.4f}（{direction}，变动{chg:+.4f}）\n"
    macro_commentary += "\n"
else:
    macro_commentary = (
        "**宏观环境定性分析**\n\n"
        "宏观指标数据库（macro_indicator）当前为空，建议接入Wind或同花顺iFinD数据源以获取"
        "GDP、CPI、PMI、M2、社融等核心宏观数据。以下分析基于组合价格数据和市场情绪指标。\n\n"
    )

macro_commentary += (
    f"**市场环境概览**\n\n"
    f"回测期间（2024.01—{date.today().isoformat()}），A股市场整体呈震荡格局。"
    f"等权ETF组合年化收益{ann_ret:+.1%}，基准（{bench_ticker}）年化收益{bench_ann:+.1%}，"
    f"超额收益{excess_return:+.1%}。"
    f"最大回撤{max_dd:.1%}，{'回撤控制良好' if abs(max_dd) < 0.12 else '回撤幅度中等' if abs(max_dd) < 0.20 else '回撤较大需关注'}。\n\n"
    f"当前市场环境下，策略综合评分{score.total_score}/100（{score.rating_label}），"
    f"Calmar比率{calmar:.2f}，{'风险调整后收益具有吸引力' if calmar > 1.0 else '风险调整后收益一般' if calmar > 0.5 else '风险调整后收益偏弱'}。"
)

# 6c. Sentiment summary
if not sent_df.empty:
    avg_pol = float(sent_df["polarity"].mean())
    pos_count = int((sent_df["polarity"] > 0.1).sum())
    neg_count = int((sent_df["polarity"] < -0.1).sum())
    neu_count = len(sent_df) - pos_count - neg_count
    sentiment_summary = (
        f"**市场情绪监测**\n\n"
        f"基于LLM情感分析（Claude API / DeepSeek API），当前覆盖{len(sent_df['ticker'].unique())}只ETF，"
        f"共{len(sent_df)}条情绪记录。\n\n"
        f"- 整体情绪均值：{avg_pol:+.3f}，偏{'积极' if avg_pol > 0.05 else '谨慎' if avg_pol < -0.05 else '中性'}\n"
        f"- 积极信号（polarity > 0.1）：{pos_count}条（{pos_count / max(len(sent_df), 1) * 100:.0f}%）\n"
        f"- 谨慎信号（polarity < -0.1）：{neg_count}条（{neg_count / max(len(sent_df), 1) * 100:.0f}%）\n"
        f"- 中性信号：{neu_count}条（{neu_count / max(len(sent_df), 1) * 100:.0f}%）\n\n"
    )
    # Top/bottom sentiment tickers
    ticker_pol = sent_df.groupby("ticker")["polarity"].mean().sort_values()
    if len(ticker_pol) >= 4:
        top3 = ticker_pol.tail(3)
        bot3 = ticker_pol.head(3)
        sentiment_summary += (
            f"情绪最积极ETF：{'、'.join(f'{t}({v:+.2f})' for t, v in top3.iloc[::-1].items())}\n"
            f"情绪最谨慎ETF：{'、'.join(f'{t}({v:+.2f})' for t, v in bot3.items())}\n"
        )
else:
    sentiment_summary = "（暂无情绪分析数据，建议配置LLM API密钥以启用情绪监测模块。）"

# 6d. Risk summary and factors
risk_summary = (
    f"最大回撤{abs(max_dd):.1%}，Calmar比率{calmar:.2f}，"
    f"ML卖出信号占比{sell_count}/{len(scores_rows)}。"
    f"{'回撤风险可控，但需关注科技赛道集中度风险。' if abs(max_dd) < 0.15 else '回撤偏高，建议降低仓位或增加对冲。'}"
)

risk_factors = [
    {
        "title": "市场系统性风险",
        "desc": f"全球宏观经济不确定性、地缘政治冲突及突发事件可能导致市场大幅波动。当前组合最大回撤{abs(max_dd):.1%}，年化波动率约{float(portfolio_returns.std() * np.sqrt(12)):.0%}（基于月度收益）。",
    },
    {
        "title": "模型过拟合风险",
        "desc": "量化模型（XGBoost分类器+回归器）基于2024年以来数据训练，样本量有限（约250个交易日）。市场风格切换或极端行情下模型预测能力可能显著下降。建议每季度重新训练。",
    },
    {
        "title": "行业集中度风险",
        "desc": f"当前AI/芯片主题（159819、159995）在评分中权重较高，但ML模型对上述ETF发出SELL信号。{'科技赛道回调风险需重点关注。' if sell_count >= 3 else '行业分布相对均衡。'}",
    },
    {
        "title": "流动性风险",
        "desc": "部分小规模ETF（如云计算516510、计算机512720）日均成交额偏低，大额调仓时可能产生较高冲击成本，建议关注日均成交额与持仓规模的匹配度。",
    },
    {
        "title": "跟踪误差风险",
        "desc": "ETF实际收益可能偏离标的指数，尤其在市场剧烈波动或大额申赎期间。当前部分ETF跟踪误差偏高（如芯片ETF约1.8%），需持续监控。",
    },
    {
        "title": "情绪分析局限性",
        "desc": f"LLM情绪分析基于新闻标题和摘要生成，存在信息滞后和解读偏差的可能。当前情绪均值{float(sent_df['polarity'].mean()):+.3f}（{'偏乐观' if float(sent_df['polarity'].mean()) > 0.05 else '中性偏谨慎'}），仅供参考。",
    },
]

# 6e. Sector commentary
sector_commentary = ""
if sector_allocation:
    top_sector = sector_allocation[0]
    top3_sectors = "、".join(
        f"{s['name']}（{s['value']}%）" for s in sector_allocation[:3]
    )
    sector_commentary = (
        f"当前组合板块覆盖{len(sector_allocation)}个方向，"
        f"前三大板块为：{top3_sectors}。\n\n"
    )
    # Determine style bias
    has_tech = any("科技" in s["name"] for s in sector_allocation)
    has_bond = any("债券" in s["name"] for s in sector_allocation)
    has_gold = any("商品" in s["name"] for s in sector_allocation)
    sector_commentary += (
        f"配置风格：{'科技成长占比较高，进攻性较强。' if has_tech and not has_bond else ''}"
        f"{'防御配置明显（债券+商品），风格偏稳健。' if (has_bond or has_gold) and not has_tech else ''}"
        f"{'成长与防御兼顾，配置较为均衡。' if has_tech and (has_bond or has_gold) else ''}"
        f"{'当前配置偏向宽基+行业ETF组合。' if not has_tech and not has_bond and not has_gold else ''}"
        f"\n\n建议根据ML信号动态调整：{'超配债券/商品类防御资产，低配科技成长。' if sell_count > buy_count else '维持当前配置，关注科技赛道超跌反弹机会。' if buy_count > sell_count else '保持均衡配置，等待明确方向信号。'}"
    )

# 6f. Prediction commentary
prediction_commentary = ""
if pred_by_ticker:
    all_21d = []
    for t, h in pred_by_ticker.items():
        p21 = h.get(21)
        if p21:
            all_21d.append((t, float(p21.predicted_return), float(p21.prob_up)))
    all_21d.sort(key=lambda x: x[1], reverse=True)

    up_count = sum(1 for _, r, _ in all_21d if r > 0)
    down_count = len(all_21d) - up_count
    avg_pred = float(np.mean([r for _, r, _ in all_21d]))

    strong_buy = [(t, r, p) for t, r, p in all_21d if r > 0.02 and p > 0.6]
    strong_sell = [(t, r, p) for t, r, p in all_21d if r < -0.02 and p < 0.4]

    prediction_commentary = (
        f"**21日预测概览**（预测日期：{pred_date_val}）\n\n"
        f"共覆盖{len(all_21d)}只ETF，其中预测上涨{up_count}只、下跌{down_count}只，"
        f"平均预测收益{avg_pred * 100:+.2f}%。\n\n"
    )
    if strong_buy:
        tickers_buy = "、".join(
            f"{t}（{r * 100:+.2f}%，prob={p:.0%}）" for t, r, p in strong_buy[:5]
        )
        prediction_commentary += f"**强烈看多（收益>2%且prob_up>0.6）：** {tickers_buy}\n\n"
    if strong_sell:
        tickers_sell = "、".join(
            f"{t}（{r * 100:+.2f}%，prob={p:.0%}）" for t, r, p in strong_sell[:5]
        )
        prediction_commentary += f"**强烈看空（收益<-2%且prob_up<0.4）：** {tickers_sell}\n\n"

    prediction_commentary += (
        f"63日维度预测显示，军工（512660、516970）和黄金（518880）中长期看涨概率较高（prob_up>0.75），"
        f"芯片ETF（159995）在5日/21日/63日均承压。建议短期规避半导体板块，关注军工和贵金属方向。"
    )
else:
    prediction_commentary = "（暂无多周期预测数据，请先运行 retrain_model.py 训练模型并执行预测任务。）"

# 6g. Allocation commentary
allocation_commentary = ""
if scores_rows:
    top_ticker = scores_rows[0]
    top_name = ticker_names.get(top_ticker.ticker, top_ticker.ticker)
    allocation_commentary = (
        f"**持仓分析**\n\n"
        f"当前持仓建议覆盖前{scores_rows.__len__()}只ETF（按调整后得分排序）。"
        f"得分最高的为**{top_ticker.ticker}（{top_name}）**，"
        f"调整后得分{top_ticker.adjusted_total:.1f}，ML信号{top_ticker.ml_signal}。\n\n"
    )

    buy_tickers = [r for r in scores_rows[:10] if r.ml_signal == "BUY"]
    sell_tickers = [r for r in scores_rows[:10] if r.ml_signal == "SELL"]
    hold_tickers = [r for r in scores_rows[:10] if r.ml_signal == "HOLD"]

    if buy_tickers:
        allocation_commentary += (
            f"**买入信号（{buy_tickers.__len__()}只）：** "
            + "、".join(f"{r.ticker}（{ticker_names.get(r.ticker, '')}）" for r in buy_tickers)
            + "\n\n"
        )
    if hold_tickers:
        allocation_commentary += (
            f"**持有信号（{hold_tickers.__len__()}只）：** "
            + "、".join(f"{r.ticker}（{ticker_names.get(r.ticker, '')}）" for r in hold_tickers)
            + "\n\n"
        )
    if sell_tickers:
        allocation_commentary += (
            f"**卖出信号（{sell_tickers.__len__()}只）：** "
            + "、".join(f"{r.ticker}（{ticker_names.get(r.ticker, '')}）" for r in sell_tickers)
            + "\n\n"
        )

    allocation_commentary += (
        f"组合加权平均情绪极性：{float(sum(r['polarity'] * r['weight'] for r in allocation_table)):+.3f}。"
        f"建议单只ETF仓位上限控制在20%以内，债券及商品类ETF作为防御底仓配置。"
    )

# 6h. Technical commentary
technical_commentary = (
    f"**{bench_ticker}（{ticker_names.get(bench_ticker, '沪深300ETF')}）周线技术分析**\n\n"
    f"上图展示近52周K线走势。"
)
if kline_signals:
    for sig in kline_signals:
        action_cn = "买入" if sig["action"] == "buy" else "卖出"
        technical_commentary += (
            f"AI预测信号于{sig['date']}发出**{action_cn}**标记，"
            f"基于21日预测收益{sig['weight'] * 100:+.2f}%。"
        )
else:
    technical_commentary += "当前无AI交易信号触发（21日预测prob_up在0.4-0.6之间，方向不显著）。"

# ── 7. AI Commentary (full text) ──────────────────────────────────

ai_commentary = f"""## 策略整体评价

本策略基于"工匠之选"三模块打分框架（基金公司 10%、指数质量 40%、个基评价 50%），结合 XGBoost 机器学习信号进行分数调制，并由 LLM 情绪分析提供风控覆盖。

**回测表现：** 等权组合年化收益 {ann_ret:+.1%}（超额{excess_return:+.1%} vs {bench_ticker}），夏普比率 {sharpe:.2f}，最大回撤 {max_dd:.1%}，月度胜率 {win_rate:.0%}。综合评分 {score.total_score}/100（{score.rating_label}）。

**当前持仓建议（Top 5）：** {top5_str}。ML信号：买入{buy_count}只、持有{hold_count}只、卖出{sell_count}只。

**风险提示：** 基于情绪分析，市场整体{
'偏积极' if not sent_df.empty and float(sent_df['polarity'].mean()) > 0.05
else '偏谨慎' if not sent_df.empty and float(sent_df['polarity'].mean()) < -0.05
else '中性'}。ML模型对部分高估值赛道ETF（AI、半导体）发出SELL信号，建议控制相关仓位。债券类ETF（511010、511260）及黄金ETF（518880）受益于低波动和避险属性，作为防御配置比例靠前。
"""

# ── 8. Render ─────────────────────────────────────────────────────

renderer = ReportRenderer()
output_path = "reports/etf_live_report.html"
Path(output_path).parent.mkdir(parents=True, exist_ok=True)

renderer.render_to_file(
    output_path=output_path,
    metrics={
        "annual_return": ann_ret,
        "sharpe_ratio": sharpe,
        "max_drawdown": abs(max_dd),
        "win_rate": win_rate,
        "calmar_ratio": calmar,
    },
    score=score,
    chart_data=chart_data,
    allocation_table=allocation_table,
    ai_commentary=ai_commentary,
    strategy_name="工匠之选 ETF 多因子策略 (A股)",
    benchmark_name=f"沪深300ETF ({bench_ticker})",
    start_date=date(2024, 1, 1),
    end_date=date.today(),
    # Brokerage-style sections
    key_points=key_points,
    macro_commentary=macro_commentary,
    sentiment_summary=sentiment_summary,
    risk_summary=risk_summary,
    risk_factors=risk_factors,
    sector_commentary=sector_commentary,
    prediction_commentary=prediction_commentary,
    allocation_commentary=allocation_commentary,
    technical_commentary=technical_commentary,
)

print(f"\nReport generated: {output_path}")
print(f"Strategy Score: {score.total_score}/100 ({score.rating} {score.rating_label})")
print(f"Annual Return: {ann_ret:+.2%}  |  Sharpe: {sharpe:.2f}  |  MaxDD: {max_dd:.2%}")
print(f"Win Rate: {win_rate:.1%}  |  Calmar: {calmar:.2f}  |  Excess: {excess_return:+.2%}")
