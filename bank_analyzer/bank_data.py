"""银行数据采集模块 — 财报数据、信用卡业务营收、市场份额。

数据源：
  - akshare: 银行财务报表、A股银行列表
  - 公开数据: 信用卡发卡量、交易额、用户规模（通过公开报告/行业数据）
  - 模拟回退: 当实时数据不可用时使用行业标准回退值
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── 12 家全国性股份制银行 + 6 大国有银行 ──────────────────────────

BANK_UNIVERSE = {
    # 国有大型商业银行
    "ICBC":    {"name": "工商银行", "code": "601398", "type": "国有", "aum_rank": 1},
    "CCB":     {"name": "建设银行", "code": "601939", "type": "国有", "aum_rank": 2},
    "ABC":     {"name": "农业银行", "code": "601288", "type": "国有", "aum_rank": 3},
    "BOC":     {"name": "中国银行", "code": "601988", "type": "国有", "aum_rank": 4},
    "BOCOM":   {"name": "交通银行", "code": "601328", "type": "国有", "aum_rank": 5},
    "PSBC":    {"name": "邮储银行", "code": "601658", "type": "国有", "aum_rank": 6},
    # 股份制商业银行
    "CMB":     {"name": "招商银行", "code": "600036", "type": "股份制", "aum_rank": 7},
    "CITIC":   {"name": "中信银行", "code": "601998", "type": "股份制", "aum_rank": 8},
    "SPDB":    {"name": "浦发银行", "code": "600000", "type": "股份制", "aum_rank": 9},
    "CMBC":    {"name": "民生银行", "code": "600016", "type": "股份制", "aum_rank": 10},
    "CIB":     {"name": "兴业银行", "code": "601166", "type": "股份制", "aum_rank": 11},
    "CEB":     {"name": "光大银行", "code": "601818", "type": "股份制", "aum_rank": 12},
    "PAB":     {"name": "平安银行", "code": "000001", "type": "股份制", "aum_rank": 13},
    "HXB":     {"name": "华夏银行", "code": "600015", "type": "股份制", "aum_rank": 14},
    "GDB":     {"name": "广发银行", "code": "",        "type": "股份制", "aum_rank": 15},
    "BQD":     {"name": "浙商银行", "code": "601916", "type": "股份制", "aum_rank": 16},
    "BOB":     {"name": "渤海银行", "code": "601166", "type": "股份制", "aum_rank": 17},
    "HFB":     {"name": "恒丰银行", "code": "",        "type": "股份制", "aum_rank": 18},
}


@dataclass
class BankProfile:
    """单个银行的完整画像数据."""

    bank_id: str
    name: str
    bank_type: str          # 国有 / 股份制
    aum_rank: int

    # 规模指标
    total_assets: float          # 总资产（亿元）
    revenue: float               # 营收（亿元）
    net_profit: float            # 净利润（亿元）
    roe: float                   # ROE（%）
    car: float                   # 资本充足率（%）
    npl_ratio: float            # 不良贷款率（%）

    # 信用卡业务指标
    credit_card_volume: float     # 信用卡累计发卡量（万张）
    credit_card_active_users: float  # 信用卡活跃用户（万）
    credit_card_transaction: float   # 信用卡交易额（亿元）
    credit_card_revenue: float       # 信用卡业务收入（亿元）
    credit_card_loan_balance: float  # 信用卡贷款余额（亿元）

    # 数字化指标
    mobile_bank_users: float     # 手机银行用户数（万）
    digital_transaction_ratio: float  # 线上交易占比（%）
    fintech_investment: float    # 金融科技投入（亿元）

    # 市场数据
    market_share_pct: float      # 信用卡市场份额（%）
    yoy_growth_pct: float        # 信用卡业务同比增长率（%）

    # 数据采集元信息
    data_source: str = ""
    data_date: date | None = None


class BankDataCollector:
    """银行数据采集器 — 多渠道获取银行信用卡业务相关数据.

    数据获取优先级：
      1. akshare 实时财报数据
      2. 行业基准推算（基于公开的行业均值/中位数）
      3. 模拟回退值（标注来源）
    """

    # 信用卡行业基准数据（2024年行业均值，来源：中国银联/央行报告）
    _INDUSTRY_BENCHMARKS: dict[str, dict] = {
        "国有": {
            "avg_credit_card_volume": 12000.0,      # 万张
            "avg_credit_card_active": 6500.0,
            "avg_credit_card_transaction": 25000.0,  # 亿元
            "avg_credit_card_revenue": 180.0,
            "avg_mobile_users": 35000.0,
            "avg_digital_ratio": 92.0,
            "avg_fintech_invest": 85.0,              # 亿元
        },
        "股份制": {
            "avg_credit_card_volume": 4500.0,
            "avg_credit_card_active": 2200.0,
            "avg_credit_card_transaction": 8000.0,
            "avg_credit_card_revenue": 60.0,
            "avg_mobile_users": 8000.0,
            "avg_digital_ratio": 88.0,
            "avg_fintech_invest": 25.0,
        },
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        bank_cfg = self._config.get("bank_analyzer", {})
        self._use_synthetic: bool = bank_cfg.get("use_synthetic_data", True)
        self._akshare_available: bool | None = None  # lazy check

    # ── 公开 API ─────────────────────────────────────────────────

    def collect_all_profiles(
        self,
        bank_ids: list[str] | None = None,
    ) -> list[BankProfile]:
        """采集全部银行的完整画像数据.

        Args:
            bank_ids: 指定银行 ID 列表，None 表示采集全量 18 家.

        Returns:
            BankProfile 列表.
        """
        if bank_ids is None:
            bank_ids = list(BANK_UNIVERSE.keys())

        profiles: list[BankProfile] = []
        for bid in bank_ids:
            try:
                profile = self.collect_one(bid)
                profiles.append(profile)
            except Exception:
                logger.exception("采集 %s 银行数据失败", bid)

        logger.info("银行数据采集完成 — %d/%d 家成功", len(profiles), len(bank_ids))
        return profiles

    def collect_one(self, bank_id: str) -> BankProfile:
        """采集单个银行的完整画像.

        尝试顺序：
          1. akshare 实时数据
          2. 行业基准 + 排名推算
        """
        info = BANK_UNIVERSE.get(bank_id)
        if info is None:
            raise ValueError(f"未知银行ID: {bank_id}")

        # 仅当用户明确需要实时数据时才尝试 akshare（延迟导入避免 7s 开销）
        fundamental: dict[str, float] = {}
        if not self._use_synthetic:
            fundamental = self._fetch_fundamental_from_akshare(info["code"]) if info["code"] else {}

        # 根据银行类型和排名推算业务数据
        btype = info["type"]
        bench = self._INDUSTRY_BENCHMARKS.get(btype, self._INDUSTRY_BENCHMARKS["股份制"])
        rank = info["aum_rank"]

        # 规模指标：基于 AUM 排名衰减（排名越高 → 数值越大）
        rank_factor = max(0.3, 1.0 - (rank - 1) * 0.04)

        profile = BankProfile(
            bank_id=bank_id,
            name=info["name"],
            bank_type=btype,
            aum_rank=rank,

            # 规模指标（优先真实数据）
            total_assets=fundamental.get("total_assets", 80000.0 * rank_factor),
            revenue=fundamental.get("revenue", 3000.0 * rank_factor),
            net_profit=fundamental.get("net_profit", 800.0 * rank_factor),
            roe=fundamental.get("roe", 14.0 * rank_factor + np.random.uniform(-2, 2)),
            car=fundamental.get("car", 13.5 + np.random.uniform(-1.5, 1.5)),
            npl_ratio=fundamental.get("npl_ratio", 1.5 + np.random.uniform(-0.5, 1.0)),

            # 信用卡业务指标（按排名倍数推算）
            credit_card_volume=self._estimate_card_volume(bench, rank_factor),
            credit_card_active_users=round(bench["avg_credit_card_active"] * rank_factor * np.random.uniform(0.85, 1.15)),
            credit_card_transaction=round(bench["avg_credit_card_transaction"] * rank_factor * np.random.uniform(0.9, 1.1), 1),
            credit_card_revenue=round(bench["avg_credit_card_revenue"] * rank_factor * np.random.uniform(0.85, 1.15), 1),
            credit_card_loan_balance=round(bench["avg_credit_card_transaction"] * 0.3 * rank_factor, 1),

            # 数字化指标
            mobile_bank_users=round(bench["avg_mobile_users"] * rank_factor * np.random.uniform(0.9, 1.1)),
            digital_transaction_ratio=round(min(98.0, bench["avg_digital_ratio"] + np.random.uniform(-3, 5)), 1),
            fintech_investment=round(bench["avg_fintech_invest"] * rank_factor * np.random.uniform(0.7, 1.3), 1),

            # 市场数据
            market_share_pct=round(rank_factor * 12.0 / 18, 2),
            yoy_growth_pct=round(rank_factor * 8.0 + np.random.uniform(-3, 5), 1),

            data_source="akshare" if fundamental else "industry_benchmark+synthetic",
            data_date=date.today(),
        )
        return profile

    def to_dataframe(self, profiles: list[BankProfile]) -> pd.DataFrame:
        """将 BankProfile 列表转为 pandas DataFrame.

        Returns:
            DataFrame，行=银行，列=各维度指标.
        """
        rows = []
        for p in profiles:
            rows.append({
                "bank_id": p.bank_id,
                "name": p.name,
                "bank_type": p.bank_type,
                "aum_rank": p.aum_rank,
                "total_assets": p.total_assets,
                "revenue": p.revenue,
                "net_profit": p.net_profit,
                "roe": p.roe,
                "car": p.car,
                "npl_ratio": p.npl_ratio,
                "credit_card_volume": p.credit_card_volume,
                "credit_card_active_users": p.credit_card_active_users,
                "credit_card_transaction": p.credit_card_transaction,
                "credit_card_revenue": p.credit_card_revenue,
                "credit_card_loan_balance": p.credit_card_loan_balance,
                "mobile_bank_users": p.mobile_bank_users,
                "digital_transaction_ratio": p.digital_transaction_ratio,
                "fintech_investment": p.fintech_investment,
                "market_share_pct": p.market_share_pct,
                "yoy_growth_pct": p.yoy_growth_pct,
                "data_source": p.data_source,
            })
        return pd.DataFrame(rows)

    # ── 内部方法 ─────────────────────────────────────────────────

    @staticmethod
    def _estimate_card_volume(bench: dict, rank_factor: float) -> float:
        """估算信用卡发卡量 — 头部银行显著高于均值."""
        # 排名前 3 的发卡量远高于行业均值
        if rank_factor > 0.85:
            return round(bench["avg_credit_card_volume"] * rank_factor * np.random.uniform(1.0, 1.3))
        return round(bench["avg_credit_card_volume"] * rank_factor * np.random.uniform(0.8, 1.1))

    @staticmethod
    def _fetch_fundamental_from_akshare(stock_code: str) -> dict[str, float]:
        """从 akshare 获取银行财报基本面数据.

        Args:
            stock_code: A股代码（如 '600036'）.

        Returns:
            提取的关键指标 dict，失败时返回空 dict.
        """
        try:
            import akshare as ak  # lazy import — 仅在需要实时数据时加载

            # 获取主要财务指标
            raw = ak.stock_financial_abstract_ths(symbol=stock_code, indicator="按报告期")
            if raw is None or raw.empty:
                return {}

            # 取最新一期数据
            latest = raw.iloc[-1]

            return {
                "total_assets": float(latest.get("总资产", 0)) / 1e8 if latest.get("总资产") else 0.0,
                "revenue": float(latest.get("营业总收入", 0)) / 1e8 if latest.get("营业总收入") else 0.0,
                "net_profit": float(latest.get("净利润", 0)) / 1e8 if latest.get("净利润") else 0.0,
                "roe": float(latest.get("ROE", 0)) if latest.get("ROE") else 0.0,
                "car": float(latest.get("资本充足率", 0)) if latest.get("资本充足率") else 0.0,
                "npl_ratio": float(latest.get("不良贷款比率", 0)) if latest.get("不良贷款比率") else 0.0,
            }
        except Exception:
            logger.debug("akshare 财报获取失败: %s", stock_code)
            return {}


class BankDataPipeline:
    """银行数据 ETL 管线 — 采集 + 清洗 + 特征工程."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._collector = BankDataCollector(config)
        self._config = config or {}

    def run(
        self,
        bank_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """执行完整数据管线.

        Returns:
            特征完备的 DataFrame，可直接用于模型训练/推理.
        """
        # 1. 采集
        profiles = self._collector.collect_all_profiles(bank_ids)
        df = self._collector.to_dataframe(profiles)

        # 2. 特征工程：生成衍生特征
        df = self._engineer_features(df)

        # 3. 标准化（关键列填充缺失值）
        df = self._normalize(df)

        logger.info("银行数据管线完成 — %d 行, %d 列", len(df), len(df.columns))
        return df

    @staticmethod
    def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """衍生特征工程 — 生成模型所需的高级特征."""
        # 信用卡活跃率
        df["card_active_rate"] = np.where(
            df["credit_card_volume"] > 0,
            df["credit_card_active_users"] / df["credit_card_volume"],
            0.5,
        )
        # 卡均交易额（万元/张/年）
        df["avg_transaction_per_card"] = np.where(
            df["credit_card_active_users"] > 0,
            df["credit_card_transaction"] / df["credit_card_active_users"] * 10000 / 10000,
            1.0,
        )
        # 信用卡收入占比
        df["card_revenue_ratio"] = np.where(
            df["revenue"] > 0,
            df["credit_card_revenue"] / df["revenue"] * 100,
            5.0,
        )
        # 数字化综合评分
        df["digital_score"] = (
            0.4 * df["digital_transaction_ratio"] / 100
            + 0.3 * np.where(df["mobile_bank_users"] > 0,
                             np.log1p(df["mobile_bank_users"]) / np.log1p(50000), 0.3)
            + 0.3 * np.where(df["fintech_investment"] > 0,
                             np.log1p(df["fintech_investment"]) / np.log1p(150), 0.3)
        )
        # 盈利效率
        df["profit_efficiency"] = np.where(
            df["total_assets"] > 0,
            df["net_profit"] / df["total_assets"] * 100,
            0.5,
        )
        # 风险调整收益
        df["risk_adjusted_margin"] = np.where(
            df["car"] > 0,
            df["roe"] * df["car"] / 100,
            1.5,
        )

        return df

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        """标准化处理：填充缺失值、裁剪异常值."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0.0)
            q01, q99 = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = df[col].clip(q01, q99)
        return df
