"""
Simulation Configuration Module

Loads and manages simulation configuration from YAML files.
"""

import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path

from agenteconomy.utils.logger import get_logger
from agenteconomy.center.Model import TaxBracket

logger = get_logger(name="simulation_config")


@dataclass
class SimulationConfig:
    """
    Simulation configuration class.
    
    Can be initialized with default values or loaded from a YAML file.
    """
    # Simulation parameters
    num_months: int = 12
    preheat_months: int = 0
    num_households: int = 1000
    num_firms: int = 100
    num_banks: int = 10
    num_government: int = 1
    num_laborhours: int = 100
    num_products: int = 100
    num_transactions: int = 100
    num_wages: int = 100
    num_innovations: int = 100
    num_research: int = 100
    num_investments: int = 100
    debug_logging: bool = True
    debug_max_households: int = 0
    debug_max_firms: int = 0
    local_record_dir: Optional[str] = None
    enable_accounting_invariant_checks: bool = True
    
    # Tax Policy
    enable_progressive_tax_system: bool = True
    income_tax_rate: List[TaxBracket] = field(default_factory=lambda: [
        TaxBracket(cutoff=0.0, rate=0.10),
        TaxBracket(cutoff=10000.0, rate=0.15),
        TaxBracket(cutoff=40000.0, rate=0.22),
        TaxBracket(cutoff=85000.0, rate=0.24),
        TaxBracket(cutoff=160000.0, rate=0.32),
        TaxBracket(cutoff=200000.0, rate=0.35)
    ])
    fallback_income_tax_rate: float = 0.2
    corporate_tax_rate: float = 0.21
    vat_rate: float = 0.08
    fica_tax_rate: float = 0.0765
    
    # Interest rate
    interest_rate: float = 0.005  # Annual rate
    
    # Demand logging configuration
    enable_month1_household_demand_logging_only: bool = False
    demand_logging_unlimited_stock_amount: float = 1e12
    demand_logging_output_dir: str = "output/month1_household_demand_3000sku"
    demand_logging_write_csv: bool = True
    demand_logging_run_employment_setup: bool = True
    demand_logging_pay_wages_before_consumption: bool = True
    
    # Concurrent configuration
    max_concurrent_tasks: int = 100
    max_llm_concurrent: int = 400

    # Checkpoint configuration
    checkpoint_interval: int = 1  # 每隔几个月保存一次 checkpoint（1=每月保存）
    checkpoint_output_dir: str = "output/checkpoints"  # checkpoint 保存目录
    checkpoint_compress: bool = True  # 是否压缩（gzip）

    # Firm initial capital configuration
    firm_initial_capital_multiplier: float = 1.5  # 企业初始现金 = 预期月成本 * 此系数
    firm_min_initial_cash: float = 10000.0  # 企业最低初始资金
    firm_initial_inventory_cover_months: float = 1.0
    firm_initial_inactive_sku_stock: float = 0.0
    firm_initial_min_active_sku_stock: float = 1.0
    firm_capital_output_ratio: float = 3.0
    firm_inventory_value_share: float = 0.5
    firm_credit_annual_interest_rate: float = 0.08
    firm_credit_repayment_cash_buffer: float = 1000.0
    firm_credit_default_distress_months: int = 3

    # Behavioral policy configuration
    consumption_use_llm: bool = True  # LLM-assisted consumption is constrained by empirical budget anchors
    consumption_llm_mode: str = "monthly"  # "profile" for long runs, "monthly" for full monthly LLM decisions, "off" for rules
    consumption_profile_refresh_months: int = 12
    unemployment_replacement_rate: float = 1.0  # 失业家庭消费收入锚=替代率×PSID基线收入(1.0=旧行为,全额PSID)
    demand_shock_std: float = 0.0   # 总需求 AR(1) 冲击标准差(0=关闭);ABM 标准做法,制造景气波动使宏观规律涌现
    demand_shock_rho: float = 0.7   # 需求冲击 AR(1) 持续性
    random_seed: int = 12345        # 随机种子(冲击可复现)
    endogenous_wages: bool = False  # 内生工资:按劳动市场松紧调整工资水平(紧→涨/松→跌)→自均衡失业+Phillips
    wage_adjustment_speed: float = 0.3   # 工资调整速度 kappa
    wage_target_unemployment: float = 0.08  # 工资调整目标失业率
    labor_demand_smoothing: float = 1.0  # 企业劳动需求信号 EMA(alpha<1 平滑,抑制周期-2 蛛网震荡;1=不平滑)
    # 就业部分调整惯性:企业投放劳动市场的有效需求向上月实际雇佣额收敛(招聘/解雇都有粘性),
    # 阻尼周期-2 蛛网震荡的频率但保留方向(区别于 labor_demand_smoothing 平滑需求信号会削弱
    # Okun/Phillips 传导)。0=当前行为;0<λ<1 平滑就业存量调整。
    employment_adjustment_inertia: float = 0.0
    # 贝弗里奇超额发布强度:_compute_labor_budget 在低失业时把劳动预算乘以一个 >1 的超发倍数
    # (U<5%→×2.5 等),意图是"低失业→企业抢人→多发岗位→高空缺"。但该倍数直接进预算→职位→
    # 被失业者填满→成真实工资帐单,而裁员工资帽不含此倍数 → 招聘按 ×2.5 过冲、裁员按 ×1.0 拉回,
    # 形成劳动市场的周期-2 蛛网(就业 53/46/53/46 交替,U volatility 爆炸)。strength 线性缩放
    # 超发幅度:adj=1+(adj-1)*strength。1.0=完全保留旧行为;0.0=无幻影超发(空缺由真实匹配摩擦+
    # 需求→发岗链内生,劳动市场不再过冲)。
    firm_beveridge_overposting_strength: float = 1.0
    household_dollar_scale: float = 1.0  # 家庭部门美元缩放(收入/财富/支出),与工资尺度一致(1=不缩放)
    # 家庭财富异质性:保留负财富(债务尾部)+ 削顶分位可调。默认复现旧行为(丢负财富、p90 削顶,
    # 财富分布被压缩)。keep_negative_wealth=True + wealth_cap_percentile=1.0 恢复债务/富尾,
    # 拉高 wealth Gini 向经验值 0.85 并恢复 wealth>income 排序。
    household_keep_negative_wealth: bool = False
    household_wealth_cap_percentile: float = 0.90
    # 家庭抽样方式:"head"=取前 N 行(旧行为);"representative"=按净财富排序后跨全分布
    # 等距抽样,使小样本(num_households<总数)保留债务+富尾,wealth Gini 不被截断压平。
    household_sampling: str = "head"
    wage_scale_init: float = 1.0  # 初始工资缩放(内生工资从此起调);=AGENTECO_WAGE_SCALE 的 config 化
    firm_job_posting_use_llm: bool = False
    labor_match_top_k: int = 8
    retailer_procurement_safety_factor: float = 1.0
    labor_match_offer_backups: int = 3
    labor_match_demand_priority_weight: float = 5000.0
    labor_offer_acceptance_policy: str = "best_loss"
    labor_offer_demand_wage_bonus: float = 0.0
    # 企业工资竞价(B):企业按自身空缺填补率内生调整出价工资,劳动紧张→加薪→真实
    # 劳动成本上升→价格上升→菲利普斯涌现(取代 reduced-form 内生工资公式+成本推动捷径)。
    firm_wage_bidding_enabled: bool = False
    firm_wage_bid_up: float = 0.04          # 有未填补空缺时,工资溢价上调步长
    firm_wage_bid_down: float = 0.02         # 满员时,溢价向 1.0 回落步长
    firm_wage_premium_min: float = 0.5
    firm_wage_premium_max: float = 2.5
    # 货币政策(Taylor 规则央行):政策利率内生响应通胀缺口与失业缺口,经
    # 信贷成本/储蓄/固定投资/消费传导到实体,使利率有真实宏观效应(支撑货币政策冲击 IRF)。
    # 默认关闭,保持既有结果与测试不变。
    taylor_rule_enabled: bool = False
    taylor_target_inflation: float = 0.02   # 年化通胀目标 π*
    taylor_phi_pi: float = 1.5              # 通胀缺口反应系数 φ_π
    taylor_phi_u: float = 0.5               # 失业缺口反应系数 φ_u(失业高→降息)
    taylor_target_unemployment: float = 0.05  # 自然失业率 u*
    taylor_natural_rate: float = 0.005      # 自然/中性利率 r0(年化)
    taylor_rate_min: float = 0.0            # 政策利率下限(年化)
    taylor_rate_max: float = 0.20           # 政策利率上限(年化)
    taylor_rate_inertia: float = 0.7        # 利率平滑 ρ(i_t=ρ·i_{t-1}+(1-ρ)·i_target)
    taylor_rate_shock: float = 0.0          # 外生政策利率冲击(年化,用于 IRF 实验)
    # 利率传导:企业信贷利率=政策利率+利差;家庭储蓄利率=政策利率×传递系数。
    firm_credit_spread: float = 0.075       # 信贷利差(政策利率之上),默认接近原 0.08
    savings_rate_passthrough: float = 1.0   # 储蓄利率对政策利率的传递系数
    # 固定资本投资:企业按产能压力+利率内生投资,购买资本品(实物 SKU)→进入支出法 I。
    # 真实资本品购买(非平衡表 capex),保证三方核算闭合。默认关闭。
    fixed_investment_enabled: bool = False
    investment_propensity: float = 0.05     # 投资倾向(占可投资基数比例)
    investment_rate_sensitivity: float = 2.0  # 投资对利率偏离的敏感度
    capital_depreciation_annual_rate: float = 0.08  # 资本年折旧率
    investment_capacity_trigger: float = 0.8  # 产能利用率超此值才扩张投资
    # 消费利率敏感性:实际利率上升→MPC 下降(储蓄增加)。默认 0 = 不改变现有行为。
    consumption_rate_sensitivity: float = 0.0
    # 当期收入消费渠道:>0 时在永久收入之上叠加当期收入偏离项并下调 habit 惯性,使消费随当期
    # 收入波动(提高 MPC、修正 cons_rel_volatility 方向)。默认 0 = 不改变现有行为。
    consumption_current_income_weight: float = 0.0
    firm_min_part_time_hours_per_month: float = 20.0
    firm_max_startup_part_time_hours_per_month: float = 160.0
    firm_min_job_budget_coverage: float = 1.0
    firm_labor_backlog_demand_share: float = 0.5
    firm_retail_labor_value_share: float = 0.25
    firm_allow_cash_based_startup_hiring: bool = False
    firm_layoff_min_wage_cap: float = 0.0
    firm_layoff_min_employees_to_keep: int = 0
    firm_layoff_wage_cap_tolerance: float = 0.25
    # 解雇速度(劳动力囤积):裁员目标帽 = current - speed*(current - new_cap)。1.0=立即裁到帽
    # (旧行为);speed<1=渐进裁员=labor hoarding(企业面对单月需求波动保留员工,只对持续超支
    # 逐步调整)。这阻尼 layoff→rehire 的劳动市场周期-2 蛛网(无平滑时企业每月把超预算员工裁到底、
    # 下月需求回来再招回 → 就业 53/47 交替)。仅作用于裁员侧,招聘侧仍完全响应当期需求,故不削弱
    # Okun 的需求→就业上行传导(区别于 employment_adjustment_inertia/labor_demand_smoothing 同时
    # 阻尼招聘 → 杀 Okun)。
    firm_layoff_speed: float = 1.0
    retail_channel_diversification_enabled: bool = True
    retail_channel_max_household_share: float = 0.70
    retail_channel_min_purchase_count: int = 3

    # Government fiscal and public employment policy configuration
    government_procurement_ratio: float = 0.35
    government_demand_injection_ratio: float = 0.30
    government_min_procurement_budget: float = 150000.0
    government_max_procurement_budget: float = 350000.0
    government_min_procurement_budget_per_household: float = 500.0
    government_max_procurement_budget_per_household: float = 1166.6667
    government_labor_budget_share_of_balance: float = 0.15
    government_initial_labor_budget: float = 20000.0
    government_min_labor_budget: float = 5000.0
    government_max_labor_budget: float = 120000.0
    government_max_labor_budget_per_household: float = 400.0
    public_employment_target_unemployment: float = 0.15
    public_employment_min_wage: float = 15.0
    public_employment_max_budget: float = 150000.0
    public_employment_max_budget_per_household: float = 500.0
    public_employment_start_period: int = 2
    public_employment_warmup_max_monthly_jobs: int = 20
    public_employment_max_monthly_jobs: int = 40
    public_employment_max_monthly_job_share: float = 0.10
    public_employment_max_new_job_share: float = 0.10
    public_employment_max_stock_share: float = 0.20
    public_employment_shrink_threshold_multiplier: float = 0.5
    public_employment_max_monthly_shrink_ratio: float = 0.20

    # Production planning configuration
    active_production_planning: bool = True
    production_target_inventory_months: float = 1.0
    production_ema_alpha: float = 0.5
    production_apply_capacity_constraints: bool = True
    production_labor_productivity: float = 2.5
    production_value_calibrated_labor_productivity: bool = True
    production_value_calibrated_productivity_cap: float = 20.0
    production_capital_productivity: float = 1.0
    production_unit_cash_cost: Optional[float] = None
    production_unit_cash_cost_share: float = 0.6
    production_cash_reserve: float = 0.0

    # Category profit margins (optional, can be loaded from config)
    category_profit_margins: Optional[Dict[str, float]] = None
    
    # Alias for compatibility with existing code
    @property
    def gov_tax_brackets(self) -> List[TaxBracket]:
        """Alias for income_tax_rate for backward compatibility."""
        return self.income_tax_rate
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'SimulationConfig':
        """
        Load configuration from a YAML file.
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            SimulationConfig instance with values loaded from YAML
            
        Example:
            config = SimulationConfig.from_yaml("config/config_normal.yaml")
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if not data:
            logger.warning(f"YAML file {yaml_path} is empty, using defaults")
            return cls()
        
        # Extract simulation parameters
        sim_data = data.get('simulation', {})
        
        # Extract tax parameters
        tax_data = data.get('tax', {})
        
        # Extract demand logging parameters
        demand_data = data.get('demand list test', {})
        
        # Extract concurrent config
        concurrent_data = data.get('concurrent config', {})

        # Extract checkpoint config
        checkpoint_data = data.get('checkpoint', {})

        # Extract interest rate
        interest_data = tax_data.get('interest', {})
        
        # Parse income tax brackets
        income_tax_brackets = []
        if 'income_tax_rate' in tax_data:
            for bracket_data in tax_data['income_tax_rate']:
                income_tax_brackets.append(
                    TaxBracket(
                        cutoff=float(bracket_data.get('cutoff', 0)),
                        rate=float(bracket_data.get('rate', 0))
                    )
                )
        
        # Create config instance
        config = cls(
            # Simulation parameters
            num_months=sim_data.get('num_months', 12),
            preheat_months=sim_data.get('preheat_months', sim_data.get('warmup_months', 0)),
            num_households=sim_data.get('num_households', 100),
            num_firms=sim_data.get('num_firms', 100),
            num_banks=sim_data.get('num_banks', 10),
            num_government=sim_data.get('num_government', 1),
            num_laborhours=sim_data.get('num_laborhours', 100),
            num_products=sim_data.get('num_products', 100),
            num_transactions=sim_data.get('num_transactions', 100),
            num_wages=sim_data.get('num_wages', 100),
            num_innovations=sim_data.get('num_innovations', 100),
            num_research=sim_data.get('num_research', 100),
            num_investments=sim_data.get('num_investments', 100),
            debug_logging=bool(sim_data.get('debug_logging', True)),
            debug_max_households=int(sim_data.get('debug_max_households', 0) or 0),
            debug_max_firms=int(sim_data.get('debug_max_firms', 0) or 0),
            local_record_dir=sim_data.get('local_record_dir'),
            enable_accounting_invariant_checks=bool(sim_data.get('enable_accounting_invariant_checks', True)),
            
            # Tax parameters
            enable_progressive_tax_system=True,  # Default to True if tax data exists
            income_tax_rate=income_tax_brackets if income_tax_brackets else [
                TaxBracket(cutoff=0.0, rate=0.10),
                TaxBracket(cutoff=10000.0, rate=0.15),
                TaxBracket(cutoff=40000.0, rate=0.22),
                TaxBracket(cutoff=85000.0, rate=0.24),
                TaxBracket(cutoff=160000.0, rate=0.32),
                TaxBracket(cutoff=200000.0, rate=0.35)
            ],
            corporate_tax_rate=float(tax_data.get('corporate_tax_rate', 0.21)),
            vat_rate=float(tax_data.get('vat_rate', 0.08)),
            fica_tax_rate=float(tax_data.get('fica_tax_rate', 0.0765)),
            
            # Interest rate
            interest_rate=float(interest_data.get('rate', 0.005)),
            
            # Demand logging
            enable_month1_household_demand_logging_only=demand_data.get('enable_month1_household_demand_logging_only', False),
            demand_logging_unlimited_stock_amount=float(demand_data.get('demand_logging_unlimited_stock_amount', 1e12)),
            demand_logging_output_dir=demand_data.get('demand_logging_output_dir', 'output/month1_household_demand_3000sku'),
            demand_logging_write_csv=demand_data.get('demand_logging_write_csv', True),
            demand_logging_run_employment_setup=demand_data.get('demand_logging_run_employment_setup', True),
            demand_logging_pay_wages_before_consumption=demand_data.get('demand_logging_pay_wages_before_consumption', True),
            
            # Concurrent configuration
            max_concurrent_tasks=concurrent_data.get('max_concurrent_tasks', 100),
            max_llm_concurrent=concurrent_data.get('max_llm_concurrent', 400),

            # Checkpoint configuration
            checkpoint_interval=int(checkpoint_data.get('checkpoint_interval', 1)),
            checkpoint_output_dir=checkpoint_data.get('checkpoint_output_dir', 'output/checkpoints'),
            checkpoint_compress=bool(checkpoint_data.get('checkpoint_compress', True)),

            # Behavioral policy configuration
            consumption_use_llm=bool(sim_data.get('consumption_use_llm', True)),
            consumption_llm_mode=str(sim_data.get('consumption_llm_mode', 'profile')),
            consumption_profile_refresh_months=int(sim_data.get('consumption_profile_refresh_months', 12)),
            unemployment_replacement_rate=float(sim_data.get('unemployment_replacement_rate', 1.0)),
            demand_shock_std=float(sim_data.get('demand_shock_std', 0.0)),
            demand_shock_rho=float(sim_data.get('demand_shock_rho', 0.7)),
            random_seed=int(sim_data.get('random_seed', 12345)),
            endogenous_wages=bool(sim_data.get('endogenous_wages', False)),
            wage_adjustment_speed=float(sim_data.get('wage_adjustment_speed', 0.3)),
            wage_target_unemployment=float(sim_data.get('wage_target_unemployment', 0.08)),
            labor_demand_smoothing=float(sim_data.get('labor_demand_smoothing', 1.0)),
            employment_adjustment_inertia=float(sim_data.get('employment_adjustment_inertia', 0.0)),
            firm_beveridge_overposting_strength=float(sim_data.get('firm_beveridge_overposting_strength', 1.0)),
            household_dollar_scale=float(sim_data.get('household_dollar_scale', 1.0)),
            household_keep_negative_wealth=bool(sim_data.get('household_keep_negative_wealth', False)),
            household_wealth_cap_percentile=float(sim_data.get('household_wealth_cap_percentile', 0.90)),
            household_sampling=str(sim_data.get('household_sampling', 'head')),
            wage_scale_init=float(sim_data.get('wage_scale_init', 1.0)),
            firm_job_posting_use_llm=bool(sim_data.get('firm_job_posting_use_llm', False)),
            labor_match_top_k=int(sim_data.get('labor_match_top_k', 8)),
            retailer_procurement_safety_factor=float(sim_data.get('retailer_procurement_safety_factor', 1.0)),
            labor_match_offer_backups=int(sim_data.get('labor_match_offer_backups', 3)),
            labor_match_demand_priority_weight=float(sim_data.get('labor_match_demand_priority_weight', 5000.0)),
            labor_offer_acceptance_policy=str(sim_data.get('labor_offer_acceptance_policy', 'best_loss')),
            labor_offer_demand_wage_bonus=float(sim_data.get('labor_offer_demand_wage_bonus', 0.0)),
            firm_wage_bidding_enabled=bool(sim_data.get('firm_wage_bidding_enabled', False)),
            firm_wage_bid_up=float(sim_data.get('firm_wage_bid_up', 0.04)),
            firm_wage_bid_down=float(sim_data.get('firm_wage_bid_down', 0.02)),
            firm_wage_premium_min=float(sim_data.get('firm_wage_premium_min', 0.5)),
            firm_wage_premium_max=float(sim_data.get('firm_wage_premium_max', 2.5)),
            taylor_rule_enabled=bool(sim_data.get('taylor_rule_enabled', False)),
            taylor_target_inflation=float(sim_data.get('taylor_target_inflation', 0.02)),
            taylor_phi_pi=float(sim_data.get('taylor_phi_pi', 1.5)),
            taylor_phi_u=float(sim_data.get('taylor_phi_u', 0.5)),
            taylor_target_unemployment=float(sim_data.get('taylor_target_unemployment', 0.05)),
            taylor_natural_rate=float(sim_data.get('taylor_natural_rate', 0.005)),
            taylor_rate_min=float(sim_data.get('taylor_rate_min', 0.0)),
            taylor_rate_max=float(sim_data.get('taylor_rate_max', 0.20)),
            taylor_rate_inertia=float(sim_data.get('taylor_rate_inertia', 0.7)),
            taylor_rate_shock=float(sim_data.get('taylor_rate_shock', 0.0)),
            firm_credit_spread=float(sim_data.get('firm_credit_spread', 0.075)),
            savings_rate_passthrough=float(sim_data.get('savings_rate_passthrough', 1.0)),
            fixed_investment_enabled=bool(sim_data.get('fixed_investment_enabled', False)),
            investment_propensity=float(sim_data.get('investment_propensity', 0.05)),
            investment_rate_sensitivity=float(sim_data.get('investment_rate_sensitivity', 2.0)),
            capital_depreciation_annual_rate=float(sim_data.get('capital_depreciation_annual_rate', 0.08)),
            investment_capacity_trigger=float(sim_data.get('investment_capacity_trigger', 0.8)),
            consumption_rate_sensitivity=float(sim_data.get('consumption_rate_sensitivity', 0.0)),
            consumption_current_income_weight=float(sim_data.get('consumption_current_income_weight', 0.0)),
            firm_min_part_time_hours_per_month=float(sim_data.get('firm_min_part_time_hours_per_month', 20.0)),
            firm_max_startup_part_time_hours_per_month=float(sim_data.get('firm_max_startup_part_time_hours_per_month', 160.0)),
            firm_min_job_budget_coverage=float(sim_data.get('firm_min_job_budget_coverage', 1.0)),
            firm_labor_backlog_demand_share=float(sim_data.get('firm_labor_backlog_demand_share', 0.5)),
            firm_retail_labor_value_share=float(sim_data.get('firm_retail_labor_value_share', 0.25)),
            firm_allow_cash_based_startup_hiring=bool(sim_data.get('firm_allow_cash_based_startup_hiring', False)),
            firm_layoff_min_wage_cap=float(sim_data.get('firm_layoff_min_wage_cap', 0.0)),
            firm_layoff_min_employees_to_keep=int(sim_data.get('firm_layoff_min_employees_to_keep', 0)),
            firm_layoff_wage_cap_tolerance=float(sim_data.get('firm_layoff_wage_cap_tolerance', 0.25)),
            firm_layoff_speed=float(sim_data.get('firm_layoff_speed', 1.0)),
            retail_channel_diversification_enabled=bool(
                sim_data.get('retail_channel_diversification_enabled', True)
            ),
            retail_channel_max_household_share=float(
                sim_data.get('retail_channel_max_household_share', 0.70)
            ),
            retail_channel_min_purchase_count=int(
                sim_data.get('retail_channel_min_purchase_count', 3)
            ),

            # Government fiscal and public employment policy configuration
            government_procurement_ratio=float(sim_data.get('government_procurement_ratio', 0.35)),
            government_demand_injection_ratio=float(sim_data.get('government_demand_injection_ratio', 0.30)),
            government_min_procurement_budget=float(sim_data.get('government_min_procurement_budget', 150000.0)),
            government_max_procurement_budget=float(sim_data.get('government_max_procurement_budget', 350000.0)),
            government_min_procurement_budget_per_household=float(
                sim_data.get('government_min_procurement_budget_per_household', 500.0)
            ),
            government_max_procurement_budget_per_household=float(
                sim_data.get('government_max_procurement_budget_per_household', 1166.6667)
            ),
            government_labor_budget_share_of_balance=float(
                sim_data.get('government_labor_budget_share_of_balance', 0.15)
            ),
            government_initial_labor_budget=float(sim_data.get('government_initial_labor_budget', 20000.0)),
            government_min_labor_budget=float(sim_data.get('government_min_labor_budget', 5000.0)),
            government_max_labor_budget=float(sim_data.get('government_max_labor_budget', 120000.0)),
            government_max_labor_budget_per_household=float(
                sim_data.get('government_max_labor_budget_per_household', 400.0)
            ),
            public_employment_target_unemployment=float(sim_data.get('public_employment_target_unemployment', 0.15)),
            public_employment_min_wage=float(sim_data.get('public_employment_min_wage', 15.0)),
            public_employment_max_budget=float(sim_data.get('public_employment_max_budget', 150000.0)),
            public_employment_max_budget_per_household=float(
                sim_data.get('public_employment_max_budget_per_household', 500.0)
            ),
            public_employment_start_period=int(sim_data.get('public_employment_start_period', 2)),
            public_employment_warmup_max_monthly_jobs=int(
                sim_data.get('public_employment_warmup_max_monthly_jobs', 20)
            ),
            public_employment_max_monthly_jobs=int(sim_data.get('public_employment_max_monthly_jobs', 40)),
            public_employment_max_monthly_job_share=float(
                sim_data.get('public_employment_max_monthly_job_share', 0.10)
            ),
            public_employment_max_new_job_share=float(
                sim_data.get('public_employment_max_new_job_share', sim_data.get('public_employment_max_monthly_job_share', 0.10))
            ),
            public_employment_max_stock_share=float(
                sim_data.get('public_employment_max_stock_share', 0.20)
            ),
            public_employment_shrink_threshold_multiplier=float(
                sim_data.get('public_employment_shrink_threshold_multiplier', 0.5)
            ),
            public_employment_max_monthly_shrink_ratio=float(
                sim_data.get('public_employment_max_monthly_shrink_ratio', 0.20)
            ),

            # Production planning configuration
            active_production_planning=bool(sim_data.get('active_production_planning', True)),
            production_target_inventory_months=float(sim_data.get('production_target_inventory_months', 1.0)),
            production_ema_alpha=float(sim_data.get('production_ema_alpha', 0.5)),
            production_apply_capacity_constraints=bool(sim_data.get('production_apply_capacity_constraints', True)),
            production_labor_productivity=float(sim_data.get('production_labor_productivity', 2.5)),
            production_value_calibrated_labor_productivity=bool(
                sim_data.get('production_value_calibrated_labor_productivity', True)
            ),
            production_capital_productivity=float(sim_data.get('production_capital_productivity', 1.0)),
            production_unit_cash_cost=(
                None
                if sim_data.get('production_unit_cash_cost') is None
                else float(sim_data.get('production_unit_cash_cost'))
            ),
            production_unit_cash_cost_share=float(sim_data.get('production_unit_cash_cost_share', 0.6)),
            production_cash_reserve=float(sim_data.get('production_cash_reserve', 0.0)),

            firm_initial_capital_multiplier=float(sim_data.get('firm_initial_capital_multiplier', 1.5)),
            firm_min_initial_cash=float(sim_data.get('firm_min_initial_cash', 10000.0)),
            firm_initial_inventory_cover_months=float(sim_data.get('firm_initial_inventory_cover_months', 1.0)),
            firm_initial_inactive_sku_stock=float(sim_data.get('firm_initial_inactive_sku_stock', 0.0)),
            firm_initial_min_active_sku_stock=float(sim_data.get('firm_initial_min_active_sku_stock', 1.0)),
            firm_capital_output_ratio=float(sim_data.get('firm_capital_output_ratio', 3.0)),
            firm_inventory_value_share=float(sim_data.get('firm_inventory_value_share', 0.5)),
            firm_credit_annual_interest_rate=float(sim_data.get('firm_credit_annual_interest_rate', 0.08)),
            firm_credit_repayment_cash_buffer=float(sim_data.get('firm_credit_repayment_cash_buffer', 1000.0)),
            firm_credit_default_distress_months=int(sim_data.get('firm_credit_default_distress_months', 3)),
        )

        logger.info(f"Configuration loaded from {yaml_path}")
        logger.info(f"  - Simulation: {config.num_months} months, {config.num_households} households, {config.num_firms} firms")
        logger.info(f"  - Tax: Corporate {config.corporate_tax_rate:.1%}, VAT {config.vat_rate:.1%}")
        logger.info(f"  - Interest rate: {config.interest_rate:.1%} (annual)")
        logger.info(f"  - Checkpoint: interval={config.checkpoint_interval}, dir={config.checkpoint_output_dir}")

        return config
