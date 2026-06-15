"""银行信用卡业务合作潜力评分模型 — XGBoost 多维度评分引擎.

评分维度（总分 100）:
  Module A (25%): 规模与市场地位 — 总资产、营收、市场份额
  Module B (30%): 信用卡业务质量 — 发卡量、活跃率、交易额、收入占比
  Module C (25%): 数字化能力 — 手机银行、线上交易、金融科技投入
  Module D (20%): 风险与稳定性 — 资本充足率、不良率、ROE、历史稳定性

辅助维度: 合作亲和度 — 对合作开放程度、现有合作基础、区域协同

模型: XGBoost 分类 + 排序 — 输出 0-100 合作潜力分
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from config.settings import Settings

logger = logging.getLogger(__name__)

# ── 评分权重配置 ─────────────────────────────────────────────────
SCORING_WEIGHTS = {
    "scale": 0.25,        # Module A: 规模与市场地位
    "card_quality": 0.30, # Module B: 信用卡业务质量
    "digital": 0.25,      # Module C: 数字化能力
    "stability": 0.20,    # Module D: 风险与稳定性
}


@dataclass
class BankScore:
    """单个银行的合作潜力评分."""

    bank_id: str
    bank_name: str
    bank_type: str
    score_date: date

    # Module A: 规模与市场地位 (0-25)
    scale_assets_score: float = 0.0
    scale_market_score: float = 0.0
    module_a_total: float = 0.0

    # Module B: 信用卡业务质量 (0-30)
    card_volume_score: float = 0.0
    card_active_score: float = 0.0
    card_revenue_score: float = 0.0
    card_growth_score: float = 0.0
    module_b_total: float = 0.0

    # Module C: 数字化能力 (0-25)
    digital_mobile_score: float = 0.0
    digital_fintech_score: float = 0.0
    digital_ratio_score: float = 0.0
    module_c_total: float = 0.0

    # Module D: 风险与稳定性 (0-20)
    stability_car_score: float = 0.0
    stability_npl_score: float = 0.0
    stability_roe_score: float = 0.0
    module_d_total: float = 0.0

    # Final
    raw_total: float = 0.0            # 基础分
    ml_signal: str = ""               # 模型信号: STRONG_BUY / BUY / NEUTRAL / WEAK
    ml_confidence: float = 0.0        # 模型置信度
    cooperation_potential: float = 0.0 # 合作潜力分 (0-100)
    rank: int = -1                    # 排名
    recommendation: str = ""          # 推荐等级: 优先合作 / 推荐 / 可考虑 / 观望
    risk_warning: str = ""            # 风险提示


class BankScorer:
    """银行合作潜力评分引擎.

    使用多维度量化和 XGBoost 模型对银行信用卡合作潜力进行打分排序。

    Usage:
        scorer = BankScorer(config)
        df = pipeline.run()  # 获取特征数据
        scores = scorer.score_all(df)
        # 返回排序后的 BankScore 列表
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._model: XGBRegressor | None = None
        self._is_fitted: bool = False
        self._feature_names: list[str] = []
        self._settings = Settings()

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, df: pd.DataFrame, labels: pd.Series | None = None) -> None:
        """训练合作潜力评分模型.

        Args:
            df: 特征 DataFrame（来自 BankDataPipeline.run()）
            labels: 可选的人工标注合作潜力分，None 则使用启发式标签.
        """
        if labels is None:
            labels = self._generate_heuristic_labels(df)

        features = self._extract_features(df)
        self._feature_names = list(features.columns)

        s = self._settings
        self._model = XGBRegressor(
            objective="reg:squarederror",
            max_depth=s.xgb_max_depth,
            learning_rate=s.xgb_learning_rate,
            n_estimators=min(s.xgb_n_estimators, 100),
            subsample=s.xgb_subsample,
            colsample_bytree=s.xgb_colsample_bytree,
            reg_alpha=s.xgb_reg_alpha,
            reg_lambda=s.xgb_reg_lambda,
            min_child_weight=s.xgb_min_child_weight,
            random_state=42,
            verbosity=0,
        )
        self._model.fit(features, labels)
        self._is_fitted = True
        logger.info("银行评分模型训练完成 — %d 样本, %d 特征", len(features), len(self._feature_names))

    def score_all(self, df: pd.DataFrame) -> list[BankScore]:
        """对 DataFrame 中所有银行进行评分.

        Args:
            df: 特征 DataFrame（来自 BankDataPipeline.run()）

        Returns:
            按合作潜力分降序排列的 BankScore 列表.
        """
        features = self._extract_features(df)

        # ML 预测
        if self._is_fitted and self._model is not None:
            ml_predictions = self._model.predict(features[self._feature_names] if self._feature_names else features)
            ml_predictions = np.clip(ml_predictions, 0, 100)
        else:
            # 未训练时使用启发式评分
            ml_predictions = self._heuristic_score(features)

        # 计算动态评分范围：使用实际数据的 P5 和 P95 作为上下限
        def _data_score(df_: pd.DataFrame, col: str, reverse: bool = False) -> pd.Series:
            """基于数据分布的 10 分制动态评分."""
            if col not in df_.columns:
                return pd.Series([5.0] * len(df_), index=df_.index)
            vals = df_[col].dropna()
            if len(vals) < 2:
                return pd.Series([5.0] * len(df_), index=df_.index)
            low = vals.quantile(0.05)
            high = vals.quantile(0.95)
            if high <= low:
                return pd.Series([5.0] * len(df_), index=df_.index)
            normalized = (df_[col] - low) / (high - low)
            if reverse:
                normalized = 1 - normalized
            return (2.0 + 8.0 * normalized.clip(0.0, 1.0)).fillna(5.0)

        scores: list[BankScore] = []
        for i, (_, row) in enumerate(df.iterrows()):
            bs = BankScore(
                bank_id=row.get("bank_id", ""),
                bank_name=row.get("name", ""),
                bank_type=row.get("bank_type", ""),
                score_date=date.today(),
            )

            # Module A: 规模与市场地位 (0-25)
            bs.scale_assets_score = _data_score(df, "total_assets").iloc[i]
            bs.scale_market_score = _data_score(df, "market_share_pct").iloc[i]
            bs.module_a_total = (bs.scale_assets_score * 0.6 + bs.scale_market_score * 0.4) * SCORING_WEIGHTS["scale"] * 100 / 10

            # Module B: 信用卡业务质量 (0-30)
            bs.card_volume_score = _data_score(df, "credit_card_volume").iloc[i]
            bs.card_active_score = _data_score(df, "card_active_rate").iloc[i]
            bs.card_revenue_score = _data_score(df, "card_revenue_ratio").iloc[i]
            bs.card_growth_score = _data_score(df, "yoy_growth_pct").iloc[i]
            bs.module_b_total = sum([
                bs.card_volume_score * 0.30,
                bs.card_active_score * 0.25,
                bs.card_revenue_score * 0.25,
                bs.card_growth_score * 0.20,
            ]) * SCORING_WEIGHTS["card_quality"] * 100 / 10

            # Module C: 数字化能力 (0-25)
            bs.digital_mobile_score = _data_score(df, "mobile_bank_users").iloc[i]
            bs.digital_fintech_score = _data_score(df, "fintech_investment").iloc[i]
            bs.digital_ratio_score = _data_score(df, "digital_transaction_ratio").iloc[i]
            bs.module_c_total = sum([
                bs.digital_mobile_score * 0.35,
                bs.digital_fintech_score * 0.35,
                bs.digital_ratio_score * 0.30,
            ]) * SCORING_WEIGHTS["digital"] * 100 / 10

            # Module D: 风险与稳定性 (0-20)
            bs.stability_car_score = _data_score(df, "car").iloc[i]
            bs.stability_npl_score = _data_score(df, "npl_ratio", reverse=True).iloc[i]
            bs.stability_roe_score = _data_score(df, "roe").iloc[i]
            bs.module_d_total = sum([
                bs.stability_car_score * 0.35,
                bs.stability_npl_score * 0.35,
                bs.stability_roe_score * 0.30,
            ]) * SCORING_WEIGHTS["stability"] * 100 / 10

            # 汇总
            bs.raw_total = bs.module_a_total + bs.module_b_total + bs.module_c_total + bs.module_d_total

            # ML 合成
            ml_score = float(ml_predictions[i])
            bs.ml_confidence = round(min(0.95, 0.5 + abs(ml_score - 50) / 100), 2)
            bs.cooperation_potential = round(0.4 * bs.raw_total + 0.6 * ml_score, 1)
            bs.cooperation_potential = max(0, min(100, bs.cooperation_potential))

            # 信号判断
            bs.ml_signal = self._classify_signal(bs.cooperation_potential)
            bs.recommendation = self._classify_recommendation(bs.cooperation_potential)
            bs.risk_warning = self._generate_risk_warning(row)

            scores.append(bs)

        # 排名
        scores.sort(key=lambda s: s.cooperation_potential, reverse=True)
        for i, s in enumerate(scores):
            s.rank = i + 1

        return scores

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        self._model.save_model(f"{path}.xgb")
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump({
                "feature_names": self._feature_names,
            }, f)
        logger.info("银行评分模型保存至 %s.{xgb,pkl}", path)

    def load(self, path: str) -> None:
        import os
        self._model = XGBRegressor()
        self._model.load_model(f"{path}.xgb")
        with open(f"{path}.pkl", "rb") as f:
            meta = pickle.load(f)
        self._feature_names = meta["feature_names"]
        self._is_fitted = True
        logger.info("银行评分模型加载自 %s", path)

    # ── 内部方法 ─────────────────────────────────────────────────

    @staticmethod
    def _extract_features(df: pd.DataFrame) -> pd.DataFrame:
        """从原始 DataFrame 提取模型特征."""
        feature_cols = [
            "total_assets", "revenue", "net_profit", "roe", "car", "npl_ratio",
            "credit_card_volume", "credit_card_active_users", "credit_card_transaction",
            "credit_card_revenue", "credit_card_loan_balance",
            "mobile_bank_users", "digital_transaction_ratio", "fintech_investment",
            "market_share_pct", "yoy_growth_pct",
            "card_active_rate", "avg_transaction_per_card", "card_revenue_ratio",
            "digital_score", "profit_efficiency", "risk_adjusted_margin",
        ]
        # 只取存在的列
        available = [c for c in feature_cols if c in df.columns]
        return df[available].copy()

    @staticmethod
    def _generate_heuristic_labels(df: pd.DataFrame) -> pd.Series:
        """生成启发式训练标签（基于简单规则）."""
        scores = pd.Series(50.0, index=df.index)

        # 量化每个维度
        for dim, cols in {
            "scale": ["total_assets", "market_share_pct"],
            "card": ["credit_card_volume", "card_active_rate", "yoy_growth_pct"],
            "digital": ["digital_score", "fintech_investment"],
            "stability": ["car", "roe"],
        }.items():
            for col in cols:
                if col in df.columns:
                    col_rank = df[col].rank(pct=True)
                    weight = {"scale": 0.25, "card": 0.35, "digital": 0.25, "stability": 0.15}[dim]
                    scores += col_rank * weight * 12.5

        # 不良率反向
        if "npl_ratio" in df.columns:
            npl_rank = (1 - df["npl_ratio"].rank(pct=True))
            scores += npl_rank * 3.0

        return scores.clip(10, 95)

    @staticmethod
    def _heuristic_score(features: pd.DataFrame) -> np.ndarray:
        """未训练时的纯启发式评分."""
        scores = np.full(len(features), 50.0)
        for col in features.columns:
            if features[col].std() > 0:
                scores += (features[col].rank(pct=True) - 0.5) * 5.0
        return np.clip(scores, 10, 95)

    @staticmethod
    def _percentile_score(value: float, low: float, high: float) -> float:
        """基于百分位的 10 分制评分.

        Args:
            value: 实际值
            low: 低阈值（得 2 分）
            high: 高阈值（得 10 分）

        Returns:
            2-10 之间的评分.
        """
        if high <= low:
            return 5.0
        normalized = (value - low) / (high - low)
        return round(2.0 + 8.0 * np.clip(normalized, 0.0, 1.0), 1)

    @staticmethod
    def _classify_signal(potential: float) -> str:
        if potential >= 80:
            return "STRONG_BUY"
        elif potential >= 65:
            return "BUY"
        elif potential >= 50:
            return "NEUTRAL"
        return "WEAK"

    @staticmethod
    def _classify_recommendation(potential: float) -> str:
        if potential >= 80:
            return "优先合作"
        elif potential >= 65:
            return "推荐"
        elif potential >= 50:
            return "可考虑"
        return "观望"

    @staticmethod
    def _generate_risk_warning(row: pd.Series) -> str:
        """生成风险提示."""
        warnings = []
        if row.get("npl_ratio", 1.5) > 2.0:
            warnings.append("不良率偏高")
        if row.get("car", 13) < 11.0:
            warnings.append("资本充足率偏低")
        if row.get("yoy_growth_pct", 10) < 0:
            warnings.append("信用卡业务负增长")
        if row.get("card_active_rate", 0.5) < 0.4:
            warnings.append("用户活跃度偏低")
        return "; ".join(warnings) if warnings else "无显著风险"
