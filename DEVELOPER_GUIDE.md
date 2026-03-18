# AgentEconomy Developer Guide

> 完整的开发者文档：项目结构、模块详解、函数说明、配置指南、运行方法。
> 适用于接手开发、修改实验参数、扩展功能的研究人员和工程师。

---

## 目录

- [1. 项目概览](#1-项目概览)
- [2. 快速开始](#2-快速开始)
- [3. 项目结构](#3-项目结构)
- [4. 配置系统](#4-配置系统)
- [5. 核心模块详解](#5-核心模块详解)
  - [5.1 Agent 层](#51-agent-层)
  - [5.2 Center 层（市场与中枢）](#52-center-层市场与中枢)
  - [5.3 Market 层（抽象资源与中间品）](#53-market-层抽象资源与中间品)
  - [5.4 LLM 层](#54-llm-层)
  - [5.5 Simulation 层](#55-simulation-层)
  - [5.6 Utils 工具层](#56-utils-工具层)
- [6. 数据模型（Model.py）](#6-数据模型modelpy)
- [7. 仿真运行流程](#7-仿真运行流程)
- [8. Demo 展示系统](#8-demo-展示系统)
- [9. 独立实验与测试](#9-独立实验与测试)
- [10. 常见修改场景](#10-常见修改场景)
- [11. 故障排查](#11-故障排查)

---

## 1. 项目概览

AgentEconomy 是一个 **LLM 驱动的多智能体经济仿真平台**，支持三类实验：

| 实验类型 | 智能体规模 | LLM 角色 | 验证指标 |
|---|---|---|---|
| 宏观经济仿真 | 300 家庭 + 66 企业 | 消费决策、就业决策 | Phillips 曲线 (r = −0.56) |
| 股票市场仿真 | 15,000+ 交易者 | 机构投资决策 | 价格发现精度 |
| 供应链分析 | FactSet 实体图谱 | 供应商评估 | Precision / Recall |

核心技术栈：
- **LLM**: DeepSeek-V3 / GPT 系列，通过 LiteLLM Router 统一调用
- **Agent Method**: 6 种推理策略（Direct / CoT / Self-Refine / Reflexion / Debate / Discussion）
- **向量数据库**: Qdrant + MiniLM-L6-v2（消费管线的语义商品匹配）
- **分布式计算**: Ray（EconomicCenter / ProductMarket / LaborMarket 均为 Ray Actor）
- **异步并发**: asyncio，最高 400 并发 LLM 调用
- **数据源**: PSID（家庭）、O*NET（职业技能）、BLS（劳动统计）、BEA IO 表（产业关联）、FactSet（供应链）

---

## 2. 快速开始

### 2.1 环境准备

```bash
# Python >= 3.11
pip install poetry
poetry install

# 或直接 pip
pip install -r requirements.txt  # 如果有的话
```

### 2.2 配置 `.env`

```bash
cp .env.example .env  # 如果有模板的话
```

`.env` 关键字段：

```env
# LLM 配置
BASE_URL=https://api.deepseek.com       # LLM API 地址
OPENAI_API_KEY=sk-xxx                    # API Key
MODEL=deepseek-chat                      # 模型名称
SIMPLE_MODEL=deepseek-chat               # 简单推理模型
STRONG_MODEL=deepseek-chat               # 强推理模型
LLM_MAX_CONCURRENCY=200                  # LLM 最大并发数

# Embedding 模型
MODEL_PATH=agenteconomy/model/all-MiniLM-L6-v2  # 本地 embedding 模型路径

# Qdrant 向量数据库
QDRANT_MODE=cloud                        # cloud / local / docker
QDRANT_URL=https://xxx.gcp.cloud.qdrant.io  # Qdrant Cloud 地址
QDRANT_API_KEY=xxx                       # Qdrant API Key
```

### 2.3 运行仿真

```bash
# 完整宏观经济仿真（需要 Ray）
python -m agenteconomy.simulation.simulator

# 或使用配置文件
python run_simulation.py --config config/config_normal.yaml
```

### 2.4 运行 Demo

```bash
streamlit run demo_app.py
# 浏览器打开 http://localhost:8501
```

### 2.5 运行消费测试（不需要完整仿真环境）

```bash
# 测试 5 个家庭的 LLM 消费决策
python test_consumption.py --n 5 --mode llm

# 对比 LLM vs 规则
python test_consumption.py --n 10 --mode both

# 不使用 Qdrant（跳过 Step 2-3）
python test_consumption.py --n 5 --mode llm --no-qdrant
```

---

## 3. 项目结构

```
AgentEconomy/
├── .env                              # 环境变量（API Key、模型配置）
├── pyproject.toml                    # Poetry 项目定义
├── demo_app.py                       # Streamlit Demo 入口
├── test_consumption.py               # 独立消费测试脚本
│
├── config/
│   ├── config.py                     # SimulationConfig 数据类 + YAML 加载器
│   └── config_normal.yaml            # 默认仿真配置（30月、300家庭、税率等）
│
├── agenteconomy/                     # 核心仿真库
│   ├── agent/                        # 智能体
│   │   ├── household.py              # 家庭智能体（LLM 消费 + 就业决策）
│   │   ├── firm.py                   # 企业（制造/零售/服务，含生产管线）
│   │   ├── government.py             # 政府（税收、再分配、公共就业）
│   │   └── bank.py                   # 银行（存款、利息）
│   │
│   ├── center/                       # 中枢系统（Ray Actor）
│   │   ├── Ecocenter.py              # 经济中心（账本、交易、税收、GDP）
│   │   ├── LaborMarket.py            # 劳动力市场（岗位发布、技能匹配、工资）
│   │   ├── ProductMarket.py          # 商品市场（SKU 管理、库存、价格、向量搜索）
│   │   ├── Model.py                  # 所有 Pydantic 数据模型
│   │   ├── transaction.py            # 交易支持结构
│   │   └── transaction_processor.py  # （占位，暂未实现）
│   │
│   ├── market/                       # 市场扩展
│   │   ├── AbstractResourceMarket.py # 抽象资源市场（电力、运输、金融服务）
│   │   ├── IntermediateGoodsProcurement.py  # 中间品采购
│   │   └── initialize_resources.py   # 抽象资源初始化
│   │
│   ├── llm/                          # LLM 调用层
│   │   ├── llm.py                    # 统一 LLM 接口（LiteLLM Router）
│   │   ├── config.yaml               # Agent Method 配置
│   │   ├── prompt_template.py        # 所有 Prompt 模板
│   │   ├── structured_output.py      # （占位）
│   │   └── agent_method/             # 推理策略
│   │       ├── __init__.py           # 注册表
│   │       ├── base.py               # 共用工具函数
│   │       ├── cot.py                # Chain-of-Thought
│   │       ├── self_refine.py        # Self-Refine
│   │       ├── reflexion.py          # Reflexion
│   │       ├── debate.py             # Debate
│   │       └── discussion.py         # Discussion
│   │
│   ├── simulation/                   # 仿真引擎
│   │   ├── simulator.py              # 主仿真器（月度循环）
│   │   ├── initializer.py            # 独立初始化框架
│   │   ├── agent_loader.py           # 智能体批量创建
│   │   ├── checkpoint.py             # 检查点保存/恢复
│   │   └── data_loader.py            # 数据加载工具
│   │
│   ├── utils/                        # 工具函数
│   │   ├── logger.py                 # 日志
│   │   ├── embedding.py              # 文本向量化
│   │   ├── load_qdrant_client.py     # Qdrant 客户端
│   │   ├── load_io_table.py          # BEA IO 表加载
│   │   ├── price_calculator.py       # 价格计算（零售/批发/出厂）
│   │   ├── plot_figures.py           # 经济指标绘图
│   │   ├── metrics.py                # 系统监控
│   │   ├── product_attribute_loader.py  # 商品属性加载
│   │   └── product_type_design.py    # 商品类型设计文档
│   │
│   ├── data/                         # 静态数据
│   │   ├── industry_cate_map.py      # 产业分类映射
│   │   └── category_to_retailer.py   # 品类→零售商映射
│   │
│   └── model/                        # 本地模型
│       └── all-MiniLM-L6-v2/         # Sentence Transformer 模型
│
├── demo_pages/                       # Streamlit 页面
│   ├── data_utils.py                 # 数据加载工具
│   ├── page_overview.py              # 系统概览
│   ├── page_system.py                # 平台能力 + Agent Method 说明
│   ├── page_macro.py                 # 宏观经济动态
│   ├── page_stylized.py              # Stylized Facts 验证
│   ├── page_labor.py                 # 劳动力市场匹配
│   ├── page_consumption.py           # 消费 Live Demo + 对比实验
│   └── page_ablation.py              # LLM vs 规则消融实验
│
├── marketsim/                        # 股票市场仿真（独立模块）
├── supply_chain/                     # 供应链分析（独立模块）
├── output/                           # 运行输出目录
└── docs/                             # 设计文档
```

---

## 4. 配置系统

### 4.1 仿真配置 `config/config_normal.yaml`

```yaml
simulation:
  num_months: 30              # 正式仿真月数
  preheat_months: 3           # 预热月数（建立初始需求和价格）
  num_households: 300          # 家庭数量
  num_firms: 100               # 企业数量（实际由 IO 表决定，约 66 家）
  debug_logging: false         # 调试日志
  local_record_dir: "output/records"  # 月度记录输出目录

tax:
  enable_progressive_tax_system: true
  income_tax_rate:             # 美国累进所得税税率
    - cutoff: 11600
      rate: 0.10
    - cutoff: 47150
      rate: 0.12
    - cutoff: 100525
      rate: 0.22
    - cutoff: 191950
      rate: 0.24
    - cutoff: 243725
      rate: 0.32
    - cutoff: 609350
      rate: 0.35
    - cutoff: 1000000000
      rate: 0.37
  corporate_tax_rate: 0.21
  vat_rate: 0.08
  fica_tax_rate: 0.0765
  interest_rate: 0.005         # 月利率

concurrent_config:
  max_concurrent_tasks: 100
  max_llm_concurrent: 400      # LLM 最大并发

checkpoint:
  checkpoint_interval: 1       # 每月保存一次检查点
  checkpoint_output_dir: "output/checkpoints"
  checkpoint_compress: true
```

**修改指南**：
- 调整仿真规模：改 `num_months` 和 `num_households`
- 调整税率：改 `tax` 部分
- 调整 LLM 并发：改 `max_llm_concurrent`（受 API 限制）

### 4.2 LLM 配置 `agenteconomy/llm/config.yaml`

```yaml
agent_name: cot    # 推理策略：none / cot / self_refine / reflexion / debate / discussion
```

这个文件只有一个关键字段 `agent_name`，控制所有 LLM 调用使用的推理策略。

| 值 | LLM 调用次数 | 说明 |
|---|---|---|
| `none` | 1 | 直接调用，最快最便宜 |
| `cot` | 1 | Chain-of-Thought，提示 LLM 逐步思考 |
| `self_refine` | 2-3 | 生成→审计→修正 |
| `reflexion` | 2-3 | 生成→反思→修正 |
| `debate` | 3 | 提议者→批评者→裁判 |
| `discussion` | 4 | 3 专家→主持人综合 |

**修改指南**：直接改 `agent_name` 的值即可全局切换。Demo 页面也支持在 UI 上临时切换。

### 4.3 环境变量 `.env`

| 变量 | 说明 | 示例 |
|---|---|---|
| `BASE_URL` | LLM API 地址 | `https://api.deepseek.com` |
| `OPENAI_API_KEY` | API Key | `sk-xxx` |
| `MODEL` | 默认模型名 | `deepseek-chat` |
| `SIMPLE_MODEL` | 简单推理模型 | `deepseek-chat` |
| `STRONG_MODEL` | 强推理模型 | `deepseek-chat` |
| `LLM_MAX_CONCURRENCY` | 最大并发 | `200` |
| `MODEL_PATH` | Embedding 模型路径 | `agenteconomy/model/all-MiniLM-L6-v2` |
| `QDRANT_MODE` | Qdrant 模式 | `cloud` / `local` / `docker` |
| `QDRANT_URL` | Qdrant 地址 | `https://xxx.cloud.qdrant.io` |
| `QDRANT_API_KEY` | Qdrant Key | `xxx` |


---

## 5. 核心模块详解

### 5.1 Agent 层

#### 5.1.1 `household.py` — 家庭智能体

家庭是系统中最复杂的智能体，拥有 LLM 驱动的消费和就业决策能力。

**类**: `Household`

**关键实例变量**:

| 变量 | 类型 | 说明 |
|---|---|---|
| `household_id` | str | 唯一标识 |
| `csv_values` | Dict | PSID 数据集原始值（收入、资产、家庭规模等） |
| `csv_decoded` | Dict | 解码后的可读值 |
| `persona` | Dict | 人格画像（来自 personas_final.json） |
| `past_household_status` | Dict | 当前状态（就业、收入、资产等） |
| `past_household_status_text` | str | 状态的自然语言描述（用于 Prompt） |
| `labor_hours` | List[LaborHour] | 户主(RP)和配偶(SP)的劳动力 |
| `consumption_categories` | List[str] | 20 个消费品类 |
| `_last_month_consumption` | float | 上月消费额（消费惯性用） |
| `economic_center` / `product_market` / `labor_market` | Ray Actor | 依赖的市场组件 |

**核心方法（按功能分组）**:

**消费管线（4 步 LLM Pipeline）**:

| 方法 | 异步 | LLM | 说明 |
|---|---|---|---|
| `consume_v2()` | ✅ | ✅ | 端到端消费流程入口，依次调用 step0→step1→step2→step3 |
| `consumption_step0_major_budget_allocation()` | ✅ | ✅ | Step0: LLM 分配月度总预算到 6 大类（零售、住房、医疗、交通、水电、保险） |
| `consumption_step1_needs_by_category()` | ✅ | ✅ | Step1: LLM 为零售品类生成具体需求描述 |
| `consumption_step2_vector_match()` | ✅ | ❌ | Step2: 用 Qdrant 向量搜索匹配商品候选 |
| `consumption_step3_purchase_llm()` | ✅ | ✅ | Step3: LLM 从候选商品中选择并分配预算 |
| `consumption_step4_validate()` | ❌ | ❌ | Step4: 验证（占位，始终返回 True） |
| `generate_fallback_consumption_plan()` | ❌ | ❌ | 规则兜底：LLM 失败时按固定比例分配 |

**就业决策**:

| 方法 | 异步 | LLM | 说明 |
|---|---|---|---|
| `list_job_seekers()` | ❌ | ❌ | 筛选失业的 RP/SP |
| `match_jobs_topk_by_loss()` | ❌ | ❌ | 规则匹配：技能/能力的非对称 z-score 损失 |
| `decide_job_applications()` | ✅ | ✅ | LLM 决定申请哪些岗位（可降级为规则） |
| `decide_offer()` | ✅ | ✅ | LLM 决定接受哪个 offer（可降级为规则） |

**数据与状态**:

| 方法 | 说明 |
|---|---|
| `get_persona_prompt_view()` | 返回用于 Prompt 的人格画像子集 |
| `get_field_text(field_name)` | 获取字段的可读文本（自动解码） |
| `_build_past_household_status()` | 构建当前状态字典 |
| `update_last_month_consumption(amount)` | 更新上月消费（消费惯性） |
| `update_rp_income(wage)` / `update_sp_income(wage)` | 记录工资收入 |
| `apply_consumption(spending_by_bucket)` | 记录消费支出 |
| `build_labor_hours()` | 根据 PSID 职业代码构建 LaborHour |
| `refresh_past_household_status_text_and_persona()` | 异步，LLM 更新人格画像 |
| `initialize_in_system()` | 注册到 EconomicCenter + LaborMarket |

**内部工具**:

| 方法 | 说明 |
|---|---|
| `_llm_chat(system, user, temperature)` | 核心 LLM 调用封装 |
| `_json_loads_loose(text)` | 鲁棒 JSON 解析（处理 LLM 输出的各种格式问题） |
| `_call_economic_center(method, ...)` | Ray-aware 的 EconomicCenter 调用 |
| `_call_product_market(method, ...)` | Ray-aware 的 ProductMarket 调用 |
| `_call_labor_market(method, ...)` | Ray-aware 的 LaborMarket 调用 |
| `census2010_to_soc2010(code)` | 人口普查职业代码 → SOC2010 映射 |


#### 5.1.2 `firm.py` — 企业智能体

企业分三种子类，共享基类 `Firm`。

**类继承关系**:
```
Firm (基类)
├── ManufactureFirm   # 制造业，生产 SKU 商品
├── RetailFirm        # 零售业，从制造商进货
└── ServiceFirm       # 服务业，提供抽象资源
```

**`Firm` 基类关键方法**:

| 方法 | 异步 | LLM | 说明 |
|---|---|---|---|
| `post_jobs(period)` | ✅ | ✅(兜底) | 发布岗位到劳动力市场。优先数据驱动（NAICS→SOC），失败时 LLM 兜底 |
| `_compute_labor_budget(period)` | ❌ | ❌ | 计算劳动力预算（基于需求/收入/现金，含 Beveridge 曲线调整） |
| `_decide_job_postings_from_data(period)` | ❌ | ❌ | 数据驱动的岗位决策（NAICS→SOC 映射 + 就业分布） |
| `add_employee(employee)` / `remove_employee(employee)` | ❌ | ❌ | 员工管理 |

**`ManufactureFirm` 额外方法**:

| 方法 | 说明 |
|---|---|
| `produce(production_plan, sku_base_prices, period)` | 完整生产管线：计算产值→IO 供应商→采购中间品→采购抽象资源→计算成本→更新库存→调整价格 |
| `procure_intermediate_goods(production_value, suppliers, period)` | 从其他制造商购买 SKU 中间品 |
| `procure_abstract_resources(production_value, io_suppliers, period)` | 从 AbstractResourceMarket 购买电力/运输等 |
| `get_io_suppliers(threshold)` | 获取 IO 表中的供应商（分中间品 vs 抽象资源） |
| `calculate_total_cost(...)` | 汇总所有成本（中间品 + 抽象资源 + 劳动 + 税） |

**`RetailFirm`**: 仅初始化，加载供应链映射（`supply_chain_entry`），无额外方法。

**`ServiceFirm`**: 仅初始化，设置 `industry_type="service"`。

**模块级工具函数**（`firm.py` 顶部）:

| 函数 | 说明 |
|---|---|
| `_load_io_industry_names()` | 加载 IO 表产业代码→名称映射 |
| `_load_job_skill_data()` | 加载 SOC 职业技能/能力数据 |
| `_load_soc_distribution()` | 加载 SOC 就业分布 |
| `_load_naics_to_soc()` | 加载 NAICS→SOC 映射 |
| `_load_retail_supply_chain_map()` | 加载零售→制造商供应链映射 |
| `_match_naics_for_io(industry_code, industry_name)` | IO 产业代码匹配最佳 NAICS |

#### 5.1.3 `government.py` — 政府智能体

**完全规则驱动**，不使用 LLM。

| 方法 | 说明 |
|---|---|
| `initialize()` | 注册到 EconomicCenter |
| `post_jobs(period)` | 发布政府岗位 + 公共就业岗位（失业率高时自动创建） |
| `procure_goods_and_services(period, budget)` | 政府采购（按 IO 表权重分配到各产业） |
| `update_tax_policy(new_policy)` | 更新税收政策 |
| `get_balance()` | 查询政府账户余额 |
| `_compute_procurement_budget(period)` | 计算采购预算（需求注入或税收驱动） |
| `_create_public_employment_jobs(period)` | 创建公共就业安全网岗位 |

#### 5.1.4 `bank.py` — 银行智能体

**完全规则驱动**，不使用 LLM。

| 方法 | 说明 |
|---|---|
| `initialize()` | 注册到 EconomicCenter |
| `create_savings_account(household_id)` | 为家庭创建储蓄账户 |
| `deposit(household_id, amount, month)` | 存款 |
| `withdraw(household_id, amount, month)` | 取款 |
| `calculate_and_pay_monthly_interest(month)` | 计算并支付月度利息（年利率 0.5%） |
| `update_deposit(household_id, amount)` | 直接设置存款余额 |


### 5.2 Center 层（市场与中枢）

这三个类都是 **Ray Actor**，运行在独立进程中，通过 `.remote()` 调用。

#### 5.2.1 `Ecocenter.py` — EconomicCenter

**装饰器**: `@ray.remote(num_cpus=8)`

经济系统的核心中枢，管理所有账本、交易、税收和 GDP 计算。约 3200 行。

**核心数据结构**:
- `self.ledger: Dict[str, Ledger]` — 所有智能体的现金账户
- `self.tx_history: List[Transaction]` — 全部交易记录
- `self.tx_by_month` / `tx_by_type` / `tx_by_party` — 交易索引
- `self.firm_monthly_data` — 企业月度财务数据（收入/支出/工资/税/生产成本）
- `self.period_statistics: Dict[int, PeriodStatistics]` — 各期统计

**关键方法分组**:

**注册与账本**:

| 方法 | 说明 |
|---|---|
| `register_id(agent_id, agent_type)` | 注册智能体 |
| `init_agent_ledger(agent_id, initial_amount)` | 初始化账本 |
| `query_balance(agent_id)` | 查询余额 |
| `deposit_funds(agent_id, amount)` | 存入资金 |
| `get_all_balances()` | 获取所有余额（用于 checkpoint） |
| `restore_balances(data)` | 恢复余额（从 checkpoint） |

**交易处理**:

| 方法 | 说明 |
|---|---|
| `process_purchase(month, buyer_id, seller_id, product_id, ...)` | 商品购买（含 VAT） |
| `process_wholesale(month, buyer_id, seller_id, ...)` | 批发交易 |
| `process_batch_purchases(month, buyer_id, purchase_list)` | 批量购买 |
| `process_wage(month, wage_hour, household_id, firm_id, ...)` | 工资支付（含个税 + FICA） |
| `record_intermediate_goods_purchase(...)` | 中间品采购记录 |
| `record_resource_purchase(...)` | 抽象资源采购记录 |
| `add_government_procurement_transaction(...)` | 政府采购 |
| `add_interest_tx(...)` | 利息交易 |
| `add_redistribution_tx(...)` | 再分配交易 |

**税收系统**:

| 方法 | 说明 |
|---|---|
| `calculate_progressive_income_tax(gross_wage)` | 累进所得税计算 |
| `get_monthly_tax_collection(month)` | 月度税收汇总 |
| `settle_monthly_corporate_tax(month)` | 企业所得税结算 |
| `redistribute_monthly_taxes(month, strategy)` | 税收再分配（6 种策略：equal / income_proportional / poverty_focused / unemployment_focused / family_size / mixed） |
| `update_tax_rates(...)` | 动态调整税率 |

**GDP 与统计**:

| 方法 | 说明 |
|---|---|
| `calculate_nominal_gdp_and_health(month)` | 名义 GDP + 系统健康度 |
| `calculate_monthly_gdp(month)` | 月度 GDP（生产法/支出法/收入法） |
| `calculate_gdp_comprehensive(month)` | 综合 GDP 计算 |
| `collect_sales_statistics(month)` | 销售统计 |
| `summarize_households_monthly(month)` | 家庭月度汇总 |

#### 5.2.2 `LaborMarket.py` — 劳动力市场

**装饰器**: `@ray.remote`

管理岗位发布、技能匹配、工资计算。

**核心数据结构**:
- `self.job_openings: List[Job]` — 当前空缺岗位
- `self.matched_jobs: List[MatchedJob]` — 已匹配的工作
- `self.labor_hours: List[LaborHour]` — 所有劳动力
- `self.job_applications` — 求职申请
- `self.offers` — 录用通知

**关键方法**:

| 方法 | 说明 |
|---|---|
| `register_labor_hours(labor_hours)` | 注册劳动力 |
| `post_job(job)` | 发布岗位 |
| `match_jobs(labor_hour)` | 为劳动力匹配 top-3 岗位 |
| `rank_jobs_for_labor(labor_hour, loss_threshold)` | 技能匹配排序 |
| `_compute_matching_loss(worker, required)` | **核心匹配算法**：非对称 z-score 损失（过度胜任 0.1x，轻微不足 0.3x，严重不足 0.5x） |
| `submit_application(application, labor_hour)` | 提交申请 |
| `evaluate_applications(job_id)` | 评估候选人 |
| `make_offers(month)` | 发放 offer |
| `accept_offer(offer_id)` / `reject_offer(offer_id)` | 接受/拒绝 offer |
| `resolve_offers(month, policy)` | 多轮 offer 解决 |
| `calculate_monthly_wage(...)` | 计算月工资 |
| `process_monthly_wages(month)` | 批量发放工资 |
| `layoff_to_budget(firm_id, target_wage_cap, ...)` | 裁员到预算（策略：highest_wage / lowest_wage / lifo / fifo） |
| `terminate_employment(firm_id, household_id, ...)` | 终止雇佣 |
| `summary()` | 市场汇总（就业率、失业率、平均工资等） |

#### 5.2.3 `ProductMarket.py` — 商品市场

**装饰器**: `@ray.remote(num_cpus=8, max_concurrency=200)`

管理 ~30,000 个 SKU 商品，支持向量搜索。

**关键方法**:

| 方法 | 说明 |
|---|---|
| `initialize_products(csv_path)` | 从 CSV 加载商品 |
| `get_product(product_id)` / `get_price(product_id)` | 查询商品/价格 |
| `get_product_snapshot(product_id)` | 获取商品快照（用于消费决策） |
| `update_stock(product_id, quantity_change)` | 更新库存 |
| `reserve_stock(product_id, quantity)` | 预留库存（并发安全） |
| `search_by_vector(query, top_k)` | **Qdrant 向量搜索**（语义匹配商品） |
| `update_manufacturer_price(product_id, new_price)` | 更新出厂价 |
| `batch_update_prices_by_industry(code, avg_cost)` | 批量价格更新（平滑 + 上限） |
| `adjust_prices_by_supply_demand(code)` | 供需价格调整（含均值回归） |
| `record_demand(code, qty)` / `record_supply(code, qty)` | 记录供需 |
| `activate_skus(sku_ids)` / `deactivate_all_skus()` | SKU 激活管理 |


### 5.3 Market 层（抽象资源与中间品）

#### `AbstractResourceMarket.py`

**非 Ray Actor**，普通 Python 类。管理电力、运输、金融服务等同质化资源。

| 方法 | 说明 |
|---|---|
| `initialize_resource(industry_code, name, base_price, supply_capacity)` | 初始化资源 |
| `register_firm(industry_code, firm_id)` | 注册服务企业（收入路由） |
| `get_receiver_id(industry_code)` | 获取收款方（ServiceFirm 或政府） |
| `purchase(industry_code, buyer_id, quantity, period)` | 按数量购买（企业用） |
| `purchase_by_budget(industry_code, buyer_id, budget, period)` | 按预算购买（家庭用） |
| `adjust_prices(period)` | 供需价格调整（均值回归，上下限 30%） |
| `get_state_snapshot()` / `restore_state(data)` | Checkpoint 支持 |

#### `IntermediateGoodsProcurement.py`

制造商之间的中间品采购。将 IO 系数转换为"等效单位"，然后从 ProductMarket 购买 SKU。

| 方法 | 说明 |
|---|---|
| `procure_intermediate_goods(manufacturer_id, production_value, io_suppliers, period)` | 完整采购流程 |
| `purchase_by_equivalent_units(industry, units, buyer_id, period, strategy)` | 按等效单位购买（策略：random / cheapest / balanced） |
| `calculate_procurement_target(industry, production_value, io_coefficient)` | 计算采购目标 |

### 5.4 LLM 层

#### `llm.py` — 统一 LLM 接口

```python
# 核心调用链
call_llm(prompt, system_prompt, model_type, timeout)
  → _read_configured_agent_name()          # 读取 config.yaml 的 agent_name
  → if agent_method exists:
      _call_agent_method(prompt, system_prompt, model_type, agent_name)
        → get_agent_method(agent_name)      # 从注册表获取方法
        → agent_method(call_model, prompt, system_prompt, model_type)
    else:
      _call_direct_llm_raw(prompt, system_prompt, model_type)
        → router.acompletion(model, messages)  # LiteLLM Router
```

**关键函数**:

| 函数 | 说明 |
|---|---|
| `call_llm(prompt, system_prompt, model_type, timeout)` | 统一入口，自动路由到 agent method 或直接调用 |
| `call_llm_simple(prompt)` | 快捷方式，使用 simple 模型 |
| `call_llm_strong(prompt)` | 快捷方式，使用 strong 模型 |
| `configure_concurrency(max_concurrent)` | 动态调整并发限制 |
| `_read_configured_agent_name()` | 从 config.yaml 读取当前 agent method |
| `_call_direct_llm_raw(prompt, system_prompt, model_type)` | 直接 LiteLLM 调用 |
| `_call_agent_method(prompt, system_prompt, model_type, agent_name)` | 通过 agent method 调用 |

**模型路由**（在代码中配置，非 config.yaml）:
- `"simple"` → 环境变量 `SIMPLE_MODEL`（默认 `gpt-4o-mini`）
- `"strong"` → 环境变量 `STRONG_MODEL`（默认 `gpt-4o`）
- 两者共用 `OPENAI_API_KEY` 和 `BASE_URL`

#### `prompt_template.py` — Prompt 模板

| 模板常量 | 用途 | 调用位置 |
|---|---|---|
| `CONSUMPTION_MAJOR_BUDGET_PROMPT` | Step0: 月度预算分配到 6 大类 | `household.consumption_step0_major_budget_allocation()` |
| `CONSUMPTION_NEEDS_BY_CATEGORY_PROMPT` | Step1: 零售品类预算 + 需求描述 | `household.consumption_step1_needs_by_category()` |
| `PURCHASE_BY_CATEGORY_PROMPT` | Step3: 单品类商品选择 | `household.consumption_step3_purchase_llm()` |
| `PURCHASE_ALL_CATEGORIES_PROMPT` | Step3: 全品类商品选择（备选） | `household.consumption_step3_purchase_llm()` |
| `JOB_APPLICATION_DECISION_PROMPT` | 求职申请决策 | `household.decide_job_applications()` |
| `JOB_OFFER_DECISION_PROMPT` | Offer 接受决策 | `household.decide_offer()` |
| `PERSONA_UPDATE_PROMPT` | 人格画像更新 | `household.refresh_past_household_status_text_and_persona()` |

**修改 Prompt 指南**：所有 Prompt 都在 `prompt_template.py` 中集中管理。修改时注意保持 `{placeholder}` 格式变量与调用方一致。

#### `agent_method/` — 推理策略

每个策略文件导出一个 `async def run(call_model, prompt, system_prompt, model_type) -> str` 函数。

| 文件 | 策略 | 流程 | LLM 调用数 |
|---|---|---|---|
| `cot.py` | Chain-of-Thought | 增强 system prompt → 单次调用 | 1 |
| `self_refine.py` | Self-Refine | 初始回答 → 审计 → 修正（如需要） | 2-3 |
| `reflexion.py` | Reflexion | 初始回答 → 反思 → 修正（如需要） | 2-3 |
| `debate.py` | Debate | 提议者 → 批评者 → 裁判 | 3 |
| `discussion.py` | Discussion | 3 专家（正确性/实用性/风险） → 主持人 | 4 |

**`base.py` 共用工具**:
- `compose_system_prompt(base, *extras)` — 组合 system prompt
- `join_sections(*sections)` — 格式化多段内容
- `indicates_correctness(feedback)` — 判断反馈是否表示"正确"（用于跳过修正步骤）

**添加新策略**:
1. 在 `agent_method/` 下创建 `my_method.py`
2. 实现 `async def run(call_model, prompt, system_prompt, model_type) -> str`
3. 在 `__init__.py` 的 `AGENT_METHOD_REGISTRY` 中注册
4. 在 `config.yaml` 中设置 `agent_name: my_method`


### 5.5 Simulation 层

#### `simulator.py` — 主仿真器

约 3600 行，是整个系统的运行引擎。

**生命周期**:

```python
sim = Simulator(config)
await sim.setup_simulation_environment()  # 创建所有 Ray Actor 和智能体
await sim.run_simulation()                # 预热 + 正式月度循环
```

**月度循环 `_run_month(month)` 执行顺序**:

```
1. 劳动力市场
   ├── _process_layoffs()          # 裁员（根据上月收入调整工资帽）
   ├── _post_jobs()                # 企业 + 政府发布岗位
   ├── _match_jobs()               # 技能匹配 + 申请 + offer
   └── _pay_wages()                # 发放工资（含个税 + FICA）

2. 商品市场
   ├── _collect_consumption_plans()    # 所有家庭并行 LLM 消费决策
   ├── _build_orders()                 # 消费计划 → 具体订单
   ├── _record_demand_to_market()      # 记录需求
   ├── _ensure_production()            # 制造商生产补货
   ├── _record_supply_and_adjust_prices()  # 供给记录 + 价格调整
   ├── _retailer_procurement()         # 零售商进货
   ├── _execute_orders()               # 执行购买交易
   ├── _execute_service_consumption()  # 服务消费（抽象资源）
   └── _update_household_consumption_history()  # 更新消费历史

3. 政府采购
   └── _execute_government_procurement()  # 政府按 IO 权重采购

4. 月末结算
   ├── _settle_corporate_tax()         # 企业所得税
   ├── _pay_bank_interest()            # 银行利息
   ├── _redistribute_taxes()           # 税收再分配
   └── _distribute_dividends()         # 企业分红

5. 记录
   ├── _record_month_summary()         # 写入月度 JSON 记录
   └── checkpoint save                 # 保存检查点
```

**预热阶段 `_run_preheat(months)`**:
- Phase 0: 需求发现（搜索全部 30K SKU），激活有需求的 SKU，初始化企业资本
- Phase 1+: 正常月度循环（但不记录正式数据）
- 最后一个预热月：锁定 CPI 消费篮子

**关键辅助方法**:

| 方法 | 说明 |
|---|---|
| `_compute_macro_indicators()` | 计算通胀率、失业率、利率、税率 |
| `_record_month_summary(...)` | 写入综合月度记录（GDP、劳动、政府、家庭分布等） |
| `_calc_gini(values)` | 计算基尼系数 |
| `_set_fixed_consumption_basket(stats, cache)` | 锁定 CPI 篮子 |
| `resume_from_checkpoint(path)` | 从检查点恢复 |

#### `agent_loader.py` — 智能体批量创建

| 函数 | 说明 |
|---|---|
| `create_households(limit, economic_center, labor_market, product_market)` | 从 PSID CSV 批量创建家庭（过滤负资产，P90 封顶） |
| `create_firms(economic_center, labor_market, product_market, abstract_resource_market)` | 从 IO 表创建所有企业（制造/零售/服务） |
| `load_all_households(...)` | 核心加载器：读数据→过滤→实例化 |
| `build_preloaded_bundle(...)` | 预加载所有数据文件到内存（避免重复 I/O） |

#### `checkpoint.py` — 检查点管理

| 方法 | 说明 |
|---|---|
| `save_checkpoint(simulator, month, preheat)` | 序列化全部状态（家庭/企业/政府/银行/市场/账本） |
| `load_checkpoint(path)` | 读取检查点文件 |
| `restore_simulator(simulator, data)` | 反序列化恢复到运行中的 Simulator |
| `list_checkpoints()` | 列出所有可用检查点 |
| `get_latest_checkpoint()` | 获取最新检查点路径 |

### 5.6 Utils 工具层

| 文件 | 说明 |
|---|---|
| `logger.py` | `get_logger(name)` — 统一日志格式 `HH:MM:SS | PID | name | msg` |
| `embedding.py` | `embedding(text)` — MiniLM-L6-v2 文本向量化（均值池化 + L2 归一化） |
| `load_qdrant_client.py` | `load_client()` — Qdrant 客户端（支持 cloud/local/docker 三种模式） |
| `load_io_table.py` | IO 表工具：`load_io_table()`, `get_cost_structure(code)`, `get_suppliers_for_industry(code)`, `get_customers_for_industry(code)` |
| `price_calculator.py` | `PriceCalculator` — 零售/批发/出厂价格转换 |
| `plot_figures.py` | `EconomyPlotter` — 从月度 JSON 生成 8 种经济图表（GDP、通胀、Phillips 曲线等） |
| `metrics.py` | `SystemMetrics` — CPU/内存监控 |
| `product_attribute_loader.py` | 商品属性加载（营养、满意度、耐久性） |


---

## 6. 数据模型（Model.py）

`agenteconomy/center/Model.py` 定义了所有 Pydantic 数据模型。

| 模型 | 说明 | 关键字段 |
|---|---|---|
| `TaxBracket` | 税率档位 | `cutoff`, `rate` |
| `TaxPolicy` | 税收政策 | `income_tax_rate` (List[TaxBracket]), `corporate_tax_rate`, `vat_rate` |
| `Asset` | 资产基类 | `name`, `asset_type`, `price`, `amount` |
| `Ledger(Asset)` | 现金账户 | `agent_id`, `amount` |
| `SavingsAccount` | 储蓄账户 | `household_id`, `balance`, `annual_interest_rate` |
| `LaborHour` | 劳动力 | `agent_id`, `skill_profile`, `ability_profile`, `lh_type` (head/spouse), `firm_id`, `job_SOC` |
| `Product(Asset)` | 商品 | `manufacturer_price`, `retail_price`, `manufacturer_code`, `retailer_code`, `available_stock`, `category` |
| `InventoryReservation` | 库存预留 | `buyer_id`, `product_id`, `quantity`, `status` |
| `Job` | 岗位 | `SOC`, `title`, `wage_per_hour`, `required_skills`, `required_abilities`, `firm_id`, `positions_available` |
| `JobApplication` | 求职申请 | `job_id`, `household_id`, `lh_type`, `expected_wage` |
| `MatchedJob` | 匹配结果 | `job`, `household_id`, `firm_id`, `skill_match_score` |
| `Wage` | 工资记录 | `agent_id`, `amount`, `month` |
| `Transaction` | 交易 | `sender_id`, `receiver_id`, `amount`, `type`, `status`, `month` |
| `PurchaseRecord` | 购买记录 | `product_id`, `quantity`, `price_per_unit`, `total_spent` |
| `FirmInnovationConfig` | 创新配置 | `firm_id`, `innovation_strategy`, `labor_productivity_factor` |
| `FirmInnovationEvent` | 创新事件 | `firm_id`, `innovation_type`, `old_value`, `new_value` |

**Transaction 类型** (`type` 字段):
`purchase` / `interest` / `service` / `redistribution` / `consume_tax` / `labor_tax` / `fica_tax` / `corporate_tax` / `labor_payment` / `government_procurement` / `transfer` / `product_sale` / `resource_purchase` / `tax_collection` / `financial` / `wholesale`

---

## 7. 仿真运行流程

### 7.1 完整仿真

```
1. 初始化
   Simulator.__init__(config)
   └── setup_simulation_environment()
       ├── 创建 EconomicCenter (Ray Actor)
       ├── 创建 ProductMarket (Ray Actor, 加载 ~30K SKU)
       ├── 创建 LaborMarket (Ray Actor)
       ├── 创建 Government + Bank
       ├── 创建 66 Firms (ManufactureFirm + RetailFirm + ServiceFirm)
       ├── 创建 300 Households (从 PSID 数据)
       └── 注册所有智能体到 EconomicCenter

2. 预热 (3 个月)
   └── _run_preheat(3)
       ├── Phase 0: 需求发现 → 激活 SKU → 初始化企业资本
       ├── Phase 1-2: 正常月度循环（建立价格和消费基线）
       └── 锁定 CPI 消费篮子

3. 正式仿真 (30 个月)
   └── for month in 1..30:
       └── _run_month(month)  # 见 5.5 节的月度循环

4. 输出
   ├── output/records/{run_id}/month_{N}.json  # 月度记录
   └── output/checkpoints/{run_id}/            # 检查点
```

### 7.2 消费管线详细流程

```
Household.consume_v2()
│
├── Step 0: consumption_step0_major_budget_allocation()
│   ├── 输入: balance, income, macro_indicators, persona, past_status
│   ├── LLM: 分配总预算到 6 大类
│   └── 输出: MajorBudgetOutput {total_budget, budgets: {Retail: X, housing: Y, ...}}
│
├── Step 1: consumption_step1_needs_by_category()
│   ├── 输入: retail_budget, 20 个消费品类
│   ├── LLM: 为每个品类分配预算 + 生成需求描述
│   └── 输出: CategoryNeedsOutput {category_plans: [{category, budget, need_descriptions}]}
│
├── Step 2: consumption_step2_vector_match()
│   ├── 输入: category_plans (需求描述)
│   ├── Qdrant: 每个需求描述 → top-K 商品候选
│   └── 输出: Dict {category: {budget, candidates: [Product]}}
│
├── Step 3: consumption_step3_purchase_llm()
│   ├── 输入: category_bundles (候选商品)
│   ├── LLM: 从候选中选择商品 + 分配预算
│   └── 输出: BudgetedPurchasePlan {purchases: [{product_id, quantity, budget}]}
│
└── Step 4: consumption_step4_validate() → True (占位)
```


---

## 8. Demo 展示系统

### 8.1 入口 `demo_app.py`

```bash
streamlit run demo_app.py
```

侧边栏导航，7 个页面：

| 页面 | 文件 | 功能 |
|---|---|---|
| System Overview | `page_overview.py` | 系统架构图、关键指标 |
| Platform & Methods | `page_system.py` | 平台能力总览、Agent Method 说明、技术栈 |
| Macro Dynamics | `page_macro.py` | GDP、CPI、通胀率、劳动份额时序图 |
| Stylized Facts | `page_stylized.py` | Phillips 曲线、Beveridge 曲线、Okun 定律 |
| Labor Matching | `page_labor.py` | 技能匹配算法可视化 |
| Live Consumption | `page_consumption.py` | 实时消费 Demo + A/B 对比实验 |
| Ablation Study | `page_ablation.py` | LLM vs 规则消融对比 |

### 8.2 数据加载 `data_utils.py`

| 函数 | 说明 |
|---|---|
| `load_monthly_data()` | 从硬编码的 run 目录加载月度 JSON 记录 |
| `load_ablation_data()` | 加载消费测试结果 JSON |

**注意**: `load_monthly_data()` 中的路径是硬编码的，需要根据实际运行输出修改。

### 8.3 消费 Demo 页面 `page_consumption.py`

支持两种模式：

**Single Run**: 配置参数 → 运行 Step0 + Step1 → 展示预算饼图 + 需求描述

**Comparison**: A/B 两组独立参数 + Agent Method 选择 → 并排对比

预设模板：
- 高收入 vs 低收入
- 高存款 vs 低存款
- 高通胀 vs 低通胀
- 繁荣 vs 衰退
- 不同家庭（同参数）
- 有工作 vs 失业
- Direct vs CoT
- Direct vs Debate
- CoT vs Reflexion

**Agent Method 切换机制**: 运行前临时修改 `config.yaml` 的 `agent_name`，运行后自动恢复。见 `_temporary_agent_method()` 上下文管理器。

---

## 9. 独立实验与测试

### 9.1 消费测试 `test_consumption.py`

不需要 Ray 或完整仿真环境，可独立运行。

```bash
# LLM 模式，测试 5 个家庭
python test_consumption.py --n 5 --mode llm

# 规则模式
python test_consumption.py --n 5 --mode rule

# 对比模式（LLM vs 规则）
python test_consumption.py --n 10 --mode both

# 跳过 Qdrant（只运行 Step 0-1）
python test_consumption.py --n 5 --mode llm --no-qdrant
```

输出到 `output/consumption_test_results.json`。

### 9.2 Agent Method 测试

```bash
python tests/test_llm_agent_methods.py
```

### 9.3 股票市场仿真 `marketsim/`

独立模块，ABIDES 风格的金融市场仿真器。

```bash
cd marketsim
pip install -r requirements.txt
# 具体运行方式见 marketsim 内部文档
```

### 9.4 供应链分析 `supply_chain/`

独立模块，LLM 驱动的供应商评估。

```bash
cd supply_chain
# 具体运行方式见 supply_chain 内部文档
```

### 9.5 经济图表生成

```python
from agenteconomy.utils.plot_figures import EconomyPlotter

plotter = EconomyPlotter(record_dir="output/records/YOUR_RUN_ID")
plotter.plot_all(output_dir="output/figures")
# 生成: GDP、通胀、Phillips 曲线、Beveridge 曲线、Okun 定律、基尼系数、劳动市场、政府财政
```



---

## 10. 常见修改场景

### 10.1 修改消费决策逻辑

**改 Prompt（最常见）**:
- 文件: `agenteconomy/llm/prompt_template.py`
- 修改 `CONSUMPTION_MAJOR_BUDGET_PROMPT`（Step0 预算分配）
- 修改 `CONSUMPTION_NEEDS_BY_CATEGORY_PROMPT`（Step1 需求生成）
- 注意保持 `{placeholder}` 与调用方一致

**改消费管线流程**:
- 文件: `agenteconomy/agent/household.py`
- 方法: `consume_v2()` — 端到端流程
- 方法: `consumption_step0_major_budget_allocation()` — Step0 具体实现
- 方法: `consumption_step1_needs_by_category()` — Step1 具体实现

**改规则兜底**:
- 文件: `agenteconomy/agent/household.py`
- 方法: `generate_fallback_consumption_plan()` — 固定比例分配
- 方法: `_calculate_reasonable_monthly_budget()` — 预算计算（含消费惯性）

### 10.2 修改就业匹配逻辑

**改匹配算法**:
- 文件: `agenteconomy/center/LaborMarket.py`
- 方法: `_compute_matching_loss()` — 核心损失函数（非对称 z-score）
- 方法: `rank_jobs_for_labor()` — 排序逻辑

**改岗位发布**:
- 文件: `agenteconomy/agent/firm.py`
- 方法: `_decide_job_postings_from_data()` — 数据驱动的岗位决策
- 方法: `_compute_labor_budget()` — 劳动力预算

**改裁员策略**:
- 文件: `agenteconomy/center/LaborMarket.py`
- 方法: `layoff_to_budget()` — 策略参数: highest_wage / lowest_wage / lifo / fifo

### 10.3 修改税收政策

**改税率**:
- 文件: `config/config_normal.yaml` — 修改 tax 部分
- 或运行时: `EconomicCenter.update_tax_rates(...)` — 动态调整

**改再分配策略**:
- 文件: `agenteconomy/center/Ecocenter.py`
- 方法: `redistribute_monthly_taxes(month, strategy)` — 6 种策略可选

### 10.4 修改价格机制

**改供需价格调整**:
- 文件: `agenteconomy/center/ProductMarket.py`
- 方法: `adjust_prices_by_supply_demand()` — 供需 + 均值回归
- 文件: `agenteconomy/market/AbstractResourceMarket.py`
- 方法: `adjust_prices()` — 抽象资源价格调整（上下限 30%）

### 10.5 添加新的 Agent Method

1. 创建 `agenteconomy/llm/agent_method/my_method.py`:

```python
from .base import ModelCaller, ModelType, compose_system_prompt

async def run(call_model: ModelCaller, prompt: str, system_prompt: str, model_type: ModelType) -> str:
    enhanced_prompt = compose_system_prompt(system_prompt, "Your custom instruction")
    return await call_model(prompt, enhanced_prompt, model_type)
```

2. 在 `__init__.py` 注册:

```python
from .my_method import run as run_my_method

AGENT_METHOD_REGISTRY = {
    ...
    "my_method": run_my_method,
}
```

3. 使用: 在 `config.yaml` 设置 `agent_name: my_method`

### 10.6 修改仿真规模

- 文件: `config/config_normal.yaml`
- `num_households`: 家庭数量（影响运行时间和 LLM 成本）
- `num_months`: 仿真月数
- `preheat_months`: 预热月数
- `max_llm_concurrent`: LLM 并发（受 API 限制）

### 10.7 修改 Demo 页面

- 所有页面在 `demo_pages/` 目录
- 每个页面导出 `render()` 函数
- 在 `demo_app.py` 中注册页面路由
- 数据加载在 `data_utils.py`（注意硬编码路径）

### 10.8 切换 LLM 模型

修改 `.env`:

```env
# 切换到 GPT-4
BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-xxx
SIMPLE_MODEL=gpt-4o-mini
STRONG_MODEL=gpt-4o

# 切换到本地 Ollama
BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=dummy
SIMPLE_MODEL=ollama/llama3
STRONG_MODEL=ollama/llama3
```

注意: LiteLLM Router 的模型配置在 `agenteconomy/llm/llm.py` 代码中（非 config.yaml），如需修改路由逻辑需改代码。


---

## 11. 故障排查

### 11.1 LLM 调用失败

| 症状 | 原因 | 解决 |
|---|---|---|
| AuthenticationError | API Key 无效 | 检查 .env 的 OPENAI_API_KEY |
| RateLimitError | 并发过高 | 降低 LLM_MAX_CONCURRENCY |
| TimeoutError | API 响应慢 | 增加 timeout 参数（默认 180s） |
| JSON 解析失败 | LLM 输出格式不规范 | _json_loads_loose() 已有鲁棒处理，检查 Prompt 格式约束 |

### 11.2 Ray 相关

| 症状 | 原因 | 解决 |
|---|---|---|
| ray.init() 失败 | Ray 未安装或端口冲突 | pip install ray 或 ray stop && ray start |
| Actor 调用超时 | Actor 过载 | 检查 max_concurrency 设置 |
| 内存不足 | 30K SKU + 300 家庭 | 减少 num_households 或增加内存 |

### 11.3 Qdrant 相关

| 症状 | 原因 | 解决 |
|---|---|---|
| 连接失败 | Qdrant 未启动 | 检查 QDRANT_MODE 和对应配置 |
| 搜索无结果 | Collection 为空 | 需要先运行数据导入（products_process.ipynb） |
| 超时 | 并发过高 | Qdrant Cloud 有并发限制 |

### 11.4 Streamlit Demo

| 症状 | 原因 | 解决 |
|---|---|---|
| 页面空白 | 数据路径错误 | 检查 data_utils.py 中的硬编码路径 |
| Task was destroyed 警告 | asyncio 事件循环冲突 | 已通过 nest_asyncio 修复 |
| 消费结果不变 | LLM 锚定在收入上 | 尝试改变 income/balance 而非仅改宏观指标 |

---

## 附录: 关键文件速查表

| 你想改什么 | 改哪个文件 | 改哪个函数/配置 |
|---|---|---|
| 消费 Prompt | agenteconomy/llm/prompt_template.py | CONSUMPTION_MAJOR_BUDGET_PROMPT 等 |
| 推理策略 | agenteconomy/llm/config.yaml | agent_name 字段 |
| 新增推理策略 | agenteconomy/llm/agent_method/ | 新建 .py + 注册到 __init__.py |
| LLM 模型 | .env | SIMPLE_MODEL / STRONG_MODEL / BASE_URL |
| 税率 | config/config_normal.yaml | tax 部分 |
| 仿真规模 | config/config_normal.yaml | num_months / num_households |
| 技能匹配 | agenteconomy/center/LaborMarket.py | _compute_matching_loss() |
| 价格调整 | agenteconomy/center/ProductMarket.py | adjust_prices_by_supply_demand() |
| 月度流程 | agenteconomy/simulation/simulator.py | _run_month() |
| Demo 页面 | demo_pages/page_*.py | render() 函数 |
| 数据加载路径 | demo_pages/data_utils.py | load_monthly_data() |
| 企业生产 | agenteconomy/agent/firm.py | ManufactureFirm.produce() |
| 政府采购 | agenteconomy/agent/government.py | procure_goods_and_services() |
| 银行利息 | agenteconomy/agent/bank.py | calculate_and_pay_monthly_interest() |
| 向量搜索 | agenteconomy/center/ProductMarket.py | search_by_vector() |
| Embedding | agenteconomy/utils/embedding.py | embedding() |
