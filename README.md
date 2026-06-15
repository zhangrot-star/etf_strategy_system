# 金融机构合作潜力分析系统

面向银行商务拓展场景的 AI 增强型量化分析平台。基于 ETF 多因子量化策略引擎，拓展了银行信用卡业务合作潜力评估能力，覆盖全国 18 家国有及股份制银行。

## 核心能力

| 模块 | 功能 | 说明 |
|------|------|------|
| 📊 银行数据采集 | 18 家银行画像自动采集 | 财报数据、信用卡业务指标、数字化程度、市场份额 |
| 🎯 合作潜力评分 | XGBoost 多维度量化评分 | 规模/信用卡质量/数字化/稳定性 4 模块 100 分制 |
| 🤖 AI 痛点分析 | LLM 驱动的银行战略与短板分析 | 自动提取战略重点、业务痛点、合作切入点、风险评估 |
| 📈 HTML 可视化报告 | Jinja2 + ECharts 机构级报告 | 排名柱状图、雷达图、热力图、AI 分析摘要 |
| 🔌 REST API | FastAPI 服务化交付 | `/bank/ranking` `/bank/{id}` `/bank/analyze` `/bank/report` |
| 📋 ETF 量化策略 | 多因子评分 + ML 预测 + RL 组合优化 | 原 ETF 系统核心能力，A 股+美股 22 只 ETF |

## 架构

```
bank_analyzer/          # 银行合作潜力分析（新增）
├── bank_data.py        #   数据采集（akshare + 行业基准双源）
├── bank_scorer.py      #   XGBoost 四维度评分模型
├── bank_pain_points.py #   LLM 银行痛点与合作机会分析
├── bank_report.py      #   Jinja2 + ECharts HTML 报告
└── templates/          #   报告模板

etf_strategy_system/    # ETF 量化策略（原有核心）
├── data_pipeline/      #   数据管道（akshare / yfinance）
├── factors/            #   因子工程（技术面 + 因果因子）
├── prediction/         #   XGBoost 多周期收益预测（5/21/63 日）
├── rl/                 #   PPO 强化学习组合优化
├── scoring/            #   多因子 ETF 评分（发行人→指数→基金）
├── sentiment/          #   LLM 市场情绪分析（DeepSeek / Claude）
├── backtest/           #   回测引擎
├── report/             #   机构级策略报告
└── main.py             #   FastAPI REST API（13 个端点）
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 银行合作潜力分析
python scripts/analyze_banks.py --report

# 只看 Top 5 银行
python scripts/analyze_banks.py --top 5 --no-pain-points

# ETF 策略系统
python scripts/run_pipeline.py
```

## API 示例

```bash
# 银行合作潜力排名
curl http://localhost:8000/bank/ranking

# 单家银行深度分析
curl http://localhost:8000/bank/CMB

# ETF 每日推荐
curl http://localhost:8000/recommendation/daily
```

## 技术栈

Python · XGBoost · FastAPI · akshare · DeepSeek/Claude API · Jinja2 · ECharts · Docker · GitHub Actions
