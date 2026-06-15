"""银行痛点分析模块 — AI 驱动的战略重点与业务短板分析.

基于银行年报、新闻、行业报告，使用 LLM 自动提取:
  - 战略重点: 银行当前的战略方向、重点投入领域
  - 业务短板: 经营中的薄弱环节、面临的挑战
  - 合作切入点: 基于痛点的合作建议
  - 风险评估: 合作中需注意的风险因素
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from sentiment.claude_client import ClaudeSentimentClient

logger = logging.getLogger(__name__)

# ── 专用 System Prompt ────────────────────────────────────────────

SYSTEM_PROMPT_BANK_ANALYSIS = """\
你是一位资深银行商务拓展分析师，专注于为金融科技公司提供银行合作伙伴评估。

分析提供的银行信息，输出一个严格的 JSON 对象，包含以下字段：

- "strategic_focus": 数组，银行当前的 3-5 个战略重点方向（如 "零售数字化转型"、"信用卡业务扩张"、"财富管理"）
- "business_pain_points": 数组，银行面临的 3-5 个业务痛点或挑战（如 "信用卡获客成本高"、"线上渠道占比低"、"不良率上升压力"）
- "cooperation_opportunities": 数组，3-5 个具体的合作切入点建议（如 "联合发卡合作"、"数据风控赋能"、"积分商城运营"），每个建议包含以下子字段：
    - "angle": 合作角度
    - "value_proposition": 价值主张（你能为银行带来什么）
    - "priority": 优先级（high/medium/low）
    - "estimated_revenue": 预估年合作收入区间（万元）
- "risk_assessment": 对象，包含:
    - "overall_risk": 合作风险等级（low/medium/high）
    - "key_risks": 2-3 个关键风险因素
    - "mitigation": 风险缓解建议
- "summary": 200 字以内的综合分析摘要

返回纯 JSON，不要 markdown 代码块、前言或后记。"""


@dataclass
class BankPainPointAnalysis:
    """银行痛点分析结果."""

    bank_id: str
    bank_name: str

    # AI 分析结果
    strategic_focus: list[str] = field(default_factory=list)
    business_pain_points: list[str] = field(default_factory=list)
    cooperation_opportunities: list[dict[str, Any]] = field(default_factory=list)
    risk_assessment: dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    # 元信息
    ai_model: str = ""
    analysis_timestamp: str = ""


class BankPainPointAnalyzer:
    """银行痛点 AI 分析器.

    使用 LLM（Claude / DeepSeek）分析银行业务痛点和合作机会。

    Usage:
        analyzer = BankPainPointAnalyzer(config)
        results = analyzer.analyze_all(profiles)
        # 返回每个银行的痛点和合作建议
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        llm_cfg = self._config.get("llm", {})
        self._provider = llm_cfg.get("provider", "claude")
        self._model = llm_cfg.get("model", "claude-sonnet-4-6")
        self._client: ClaudeSentimentClient | None = None

    def _ensure_client(self) -> ClaudeSentimentClient:
        if self._client is None:
            self._client = ClaudeSentimentClient(model=self._model)
        return self._client

    def analyze_one(
        self,
        bank_name: str,
        bank_type: str,
        profile_text: str,
    ) -> BankPainPointAnalysis:
        """分析单个银行的痛点和合作机会.

        Args:
            bank_name: 银行名称
            bank_type: 银行类型（国有/股份制）
            profile_text: 银行画像文本描述

        Returns:
            包含战略重点、痛点、合作建议的分析结果.
        """
        user_msg = f"""银行名称：{bank_name}（{bank_type}）

银行画像数据：
{profile_text}

请基于以上信息分析该银行的战略重点、业务痛点和合作机会。"""

        try:
            client = self._ensure_client()
            raw = client._call_claude(
                system_prompt=SYSTEM_PROMPT_BANK_ANALYSIS,
                user_message=user_msg,
            )
            parsed = self._parse_analysis(raw)
        except Exception:
            logger.warning("AI 分析失败: %s，使用模板回退", bank_name)
            parsed = self._generate_template_analysis(bank_name, bank_type)

        from datetime import datetime
        return BankPainPointAnalysis(
            bank_id="",
            bank_name=bank_name,
            strategic_focus=parsed.get("strategic_focus", []),
            business_pain_points=parsed.get("business_pain_points", []),
            cooperation_opportunities=parsed.get("cooperation_opportunities", []),
            risk_assessment=parsed.get("risk_assessment", {}),
            summary=parsed.get("summary", ""),
            ai_model=self._model,
            analysis_timestamp=datetime.now().isoformat(),
        )

    def analyze_all(
        self,
        profiles: list[Any],  # list[BankProfile]
    ) -> list[BankPainPointAnalysis]:
        """批量分析所有银行的痛点和合作机会.

        Args:
            profiles: BankProfile 列表

        Returns:
            每个银行的痛点分析结果.
        """
        results: list[BankPainPointAnalysis] = []
        for i, profile in enumerate(profiles):
            profile_text = self._profile_to_text(profile)

            result = self.analyze_one(
                bank_name=profile.name,
                bank_type=profile.bank_type,
                profile_text=profile_text,
            )
            result.bank_id = profile.bank_id
            results.append(result)

            # 速率限制
            if i < len(profiles) - 1:
                import time
                time.sleep(0.5)

        logger.info("银行痛点分析完成 — %d 家", len(results))
        return results

    @staticmethod
    def _profile_to_text(profile: Any) -> str:
        """将 BankProfile 转为 LLM 可读的文本描述."""
        lines = [
            f"总资产: {getattr(profile, 'total_assets', '?')} 亿元",
            f"营收: {getattr(profile, 'revenue', '?')} 亿元",
            f"净利润: {getattr(profile, 'net_profit', '?')} 亿元",
            f"ROE: {getattr(profile, 'roe', '?')}%",
            f"资本充足率: {getattr(profile, 'car', '?')}%",
            f"不良贷款率: {getattr(profile, 'npl_ratio', '?')}%",
            f"信用卡发卡量: {getattr(profile, 'credit_card_volume', '?')} 万张",
            f"信用卡活跃用户: {getattr(profile, 'credit_card_active_users', '?')} 万",
            f"信用卡交易额: {getattr(profile, 'credit_card_transaction', '?')} 亿元",
            f"信用卡业务收入: {getattr(profile, 'credit_card_revenue', '?')} 亿元",
            f"手机银行用户: {getattr(profile, 'mobile_bank_users', '?')} 万",
            f"线上交易占比: {getattr(profile, 'digital_transaction_ratio', '?')}%",
            f"金融科技投入: {getattr(profile, 'fintech_investment', '?')} 亿元",
            f"信用卡市场份额: {getattr(profile, 'market_share_pct', '?')}%",
            f"同比增长率: {getattr(profile, 'yoy_growth_pct', '?')}%",
        ]
        return "\n".join(lines)

    @staticmethod
    def _parse_analysis(raw: str) -> dict[str, Any]:
        """解析 AI 响应为结构化数据."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("AI 返回格式异常，尝试提取 JSON")
            # 尝试提取 JSON 片段
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            return {}

    @staticmethod
    def _generate_template_analysis(bank_name: str, bank_type: str) -> dict[str, Any]:
        """LLM 不可用时的模板回退分析."""
        templates = {
            "国有": {
                "strategic_focus": [
                    "零售数字化转型",
                    "信用卡业务精细化运营",
                    "金融科技赋能网点转型",
                    "财富管理与私行服务",
                ],
                "business_pain_points": [
                    "网点客户流量持续下降",
                    "信用卡获客成本逐年上升",
                    "线上渠道活跃度待提升",
                ],
                "cooperation_opportunities": [
                    {"angle": "联合发卡与场景获客", "value_proposition": "互联网流量导入+精准营销", "priority": "high", "estimated_revenue": "500-2000"},
                    {"angle": "数据风控赋能", "value_proposition": "AI风控模型降低欺诈和信用风险", "priority": "high", "estimated_revenue": "300-1000"},
                    {"angle": "积分商城联合运营", "value_proposition": "提升积分兑换率和用户粘性", "priority": "medium", "estimated_revenue": "100-500"},
                ],
                "risk_assessment": {
                    "overall_risk": "low",
                    "key_risks": ["审批流程较长", "需要总行级决策"],
                    "mitigation": "提前半年布局，分阶段推进",
                },
                "summary": f"{bank_name}作为大型{bank_type}银行，信用卡业务规模大但数字化程度有提升空间，合作潜力高。建议以数据风控和场景获客为切入点。",
            },
            "股份制": {
                "strategic_focus": [
                    "信用卡业务差异化竞争",
                    "消费金融场景拓展",
                    "开放银行生态建设",
                    "AI 驱动的精准营销",
                ],
                "business_pain_points": [
                    "与大行直面竞争获客难",
                    "信用卡活卡率偏低",
                    "金融科技人才缺口",
                ],
                "cooperation_opportunities": [
                    {"angle": "场景化获客方案", "value_proposition": "电商/出行/本地生活场景嵌入", "priority": "high", "estimated_revenue": "300-1500"},
                    {"angle": "AI客服与智能营销", "value_proposition": "大模型驱动客服降本增效", "priority": "high", "estimated_revenue": "200-800"},
                    {"angle": "联合风控建模", "value_proposition": "联邦学习+多方数据源提升风控精度", "priority": "medium", "estimated_revenue": "200-600"},
                ],
                "risk_assessment": {
                    "overall_risk": "medium",
                    "key_risks": ["决策链较短但预算受限", "业务优先级可能调整"],
                    "mitigation": "快速验证POC，展示ROI后争取预算",
                },
                "summary": f"{bank_name}作为{bank_type}银行，决策灵活、创新意愿强，是优质的合作伙伴。建议以快速POC切入，展示ROI后扩大合作。",
            },
        }

        tmpl = templates.get(bank_type, templates["股份制"])
        # 个性化 summary
        tmpl["summary"] = tmpl["summary"].format(bank_name=bank_name, bank_type=bank_type)
        return tmpl
