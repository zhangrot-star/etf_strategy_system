from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "etf_strategy"
    mysql_pool_size: int = 10
    mysql_pool_overflow: int = 20

    @property
    def mysql_url(self) -> str:
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )

    # ── ChromaDB ──────────────────────────────────────────────
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection_name: str = "etf_research"

    # ── LLM / Claude ──────────────────────────────────────────
    anthropic_api_key: str = ""
    anthropic_auth_token: str = ""
    anthropic_base_url: str = ""
    anthropic_model: str = "claude-sonnet-4-6"
    llm_max_retries: int = 3
    llm_request_timeout: int = 60

    # ── Risk Control ──────────────────────────────────────────
    sentiment_breach_threshold: float = -0.7
    sentiment_warn_threshold: float = -0.5
    sentiment_confidence_threshold: float = 0.85

    # ── Backtest ──────────────────────────────────────────────
    initial_capital: float = 1_000_000.0
    commission_rate: float = 0.0003       # 3 bps per side
    slippage_bps: float = 1.0             # 1 bp proportional
    rebalance_frequency: str = "monthly"  # daily | weekly | monthly

    # ── Factor Engineering ────────────────────────────────────
    momentum_windows: list[int] = [5, 10, 21, 63]
    volatility_window: int = 21
    winsorize_bounds: tuple[float, float] = (0.01, 0.99)

    # ── XGBoost ───────────────────────────────────────────────
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_n_estimators: int = 200
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_reg_alpha: float = 0.1
    xgb_reg_lambda: float = 1.0
    xgb_min_child_weight: int = 3
    xgb_refit_frequency: str = "quarterly"

    # ── Prediction ────────────────────────────────────────────
    prediction_model_path: str = "models/xgboost_reg"
    prediction_horizons: list[int] = [5, 21, 63]
    prediction_retrain_frequency: str = "quarterly"

    # ── RL Portfolio Optimization ────────────────────────────
    rl_enabled: bool = False
    rl_model_path: str = "models/rl/ppo_portfolio"
    rl_total_timesteps: int = 200_000
    rl_rebalance_freq: str = "monthly"
    rl_ppo_learning_rate: float = 3e-4
    rl_ppo_n_steps: int = 2048
    rl_ppo_batch_size: int = 64
    rl_ppo_n_epochs: int = 10
    rl_ppo_gamma: float = 0.99
    rl_ppo_ent_coef: float = 0.01
    rl_reward_sharpe_weight: float = 1.0
    rl_reward_turnover_weight: float = 0.5
    rl_reward_drawdown_weight: float = 1.0
    rl_reward_diversification_weight: float = 0.2
    rl_walk_forward_folds: int = 5

    # ── Report ────────────────────────────────────────────────
    report_output_dir: str = "./reports"

    # ── Paths ─────────────────────────────────────────────────
    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent
