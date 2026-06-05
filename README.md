# ETF Quant Strategy System

Production-grade ETF quantitative strategy system with six decoupled core modules for the A-share and US markets.

## Architecture

```
etf_strategy_system/
├── config/           # Settings & configuration (Pydantic-based)
├── data_pipeline/    # Data fetching (akshare/yfinance), cleaning, DB, vector store
├── factors/          # Technical & causal factor computation, factor registry
├── prediction/       # XGBoost return prediction (5d / 21d / 63d horizons)
├── rl/               # PPO-based portfolio optimization (Gymnasium + SB3)
├── scoring/          # Multi-factor ETF scoring (issuer → index → fund, 100-pt scale)
├── recommendation/   # Ranking, filtering, weight allocation pipeline
├── sentiment/        # LLM-driven sentiment (DeepSeek / Claude API)
├── backtest/         # Backtesting framework with slippage & attribution
├── engine/           # Backtrader engine, commission models, analyzers
├── strategy/         # Strategy optimizer (Optuna)
├── report/           # HTML report generation (Jinja2 templates)
├── models/           # Trained models (XGBoost + RL PPO agents)
├── scripts/          # CLI entry points for all operations
└── main.py           # FastAPI REST API server
```

## Pipeline

```
Data → Factors → Scoring → Prediction Modulation → Sentiment Modulation
                                                       ↓
                                            Recommendation (Top-N + Weights)
                                                       ↓
                                               RL Portfolio Optimizer
                                                       ↓
                                                   Backtest
                                                       ↓
                                                  HTML Report
```

## Quick Start

### 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys and database credentials
```

Edit `config.yaml` to set market (`A` for A-shares, `US` for US market) and adjust parameters.

### 3. Run

```bash
# Full pipeline: data → scoring → recommendation
python scripts/run_pipeline.py

# Generate strategy report
python scripts/generate_report.py

# Run backtest
python scripts/run_backtest.py

# Train RL agent
python scripts/train_rl_agent.py

# Start API server
python main.py
```

### Docker

```bash
docker-compose up -d
```

## Key Features

- **Multi-factor scoring**: 3-module methodology (issuer 10% + index 40% + fund 50%) on 100-point scale
- **ML return prediction**: XGBoost regressors for 5-day, 21-day, and 63-day horizons
- **LLM sentiment**: Structured financial sentiment extraction via DeepSeek/Claude API
- **RL portfolio optimization**: PPO agent with Sharpe/drawdown/diversification rewards
- **Full backtesting**: Backtrader-based with slippage modeling and performance attribution
- **REST API**: FastAPI server exposing all pipeline steps as endpoints

## Markets

| Market | Data Source | Example Tickers |
|--------|-------------|-----------------|
| A-share (A) | akshare | 510050, 510300, 159915, 588000 |
| US | yfinance | SPY, QQQ, IWM, XLK |

## Tech Stack

Python 3.10+ • XGBoost • Stable-Baselines3 • Gymnasium • Backtrader • FastAPI • SQLAlchemy • Optuna • Anthropic SDK • ChromaDB

## License

MIT
