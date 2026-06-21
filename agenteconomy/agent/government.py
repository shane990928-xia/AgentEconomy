from typing import Optional, Dict, List, Any, Tuple
from functools import lru_cache
from collections import defaultdict
from math import ceil
import os
import json
import pandas as pd
from agenteconomy.center.Model import *
from agenteconomy.center.Ecocenter import EconomicCenter
from agenteconomy.utils.logger import get_logger
from agenteconomy.utils.load_io_table import get_cost_structure
import ray


# IO代码到行业名称的映射（用于政府采购）
@lru_cache(maxsize=1)
def _load_io_code_to_industry_name() -> Dict[str, str]:
    """加载 IO 代码到行业名称的映射"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    mapping_path = os.path.join(data_dir, "industry_map.json")
    
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# 政府服务行业代码（IO表中的政府部门）
GOVERNMENT_SERVICE_CODES = frozenset({"GFGD", "GFGN", "GFE", "GSLG", "GSLE"})

# 政府部门到 NAICS 代码的映射（来自 occupation.xlsx Table 1.12）
# 999100: Federal government, excluding postal service
# 999200: State government, excluding education and hospitals  
# 999300: Local government, excluding education and hospitals
GOVERNMENT_NAICS_MAPPING = {
    "GFGD": "999100",  # Federal defense -> Federal government
    "GFGN": "999100",  # Federal non-defense -> Federal government
    "GFE": "999100",   # Federal enterprises -> Federal government
    "GSLG": "999200",  # State/local government -> State government
    "GSLE": "999300",  # State/local enterprises -> Local government
}

# 政府采购相关常量
# IO表中不参与采购的部分（增值部分）
GOVERNMENT_NON_PROCUREMENT_ROWS = frozenset({"V001", "V002", "V003", "Other"})
# 最小采购系数阈值（小于此值不采购）
GOVERNMENT_PROCUREMENT_MIN_COEFFICIENT = 0.005
# 政府采购占总支出的比例（基于IO表分析，实际采购约35%，其余为补偿、折旧等）
GOVERNMENT_PROCUREMENT_RATIO = 0.35

# ========== 政府作为"无限资金"需求注入器的参数 ==========
# 政府支出占家庭消费总预算的比例（凯恩斯主义需求刺激）
# 设为0.30意味着政府额外注入相当于家庭消费30%的需求，帮助维持企业生存
GOVERNMENT_DEMAND_INJECTION_RATIO = 0.30
# 政府最低采购预算（保证政府始终能采购，维持企业最低运营）
GOVERNMENT_MIN_PROCUREMENT_BUDGET = 150000.0
# 政府最高采购预算上限（防止无限膨胀）
GOVERNMENT_MAX_PROCUREMENT_BUDGET = 350000.0

# ========== 政府公共就业计划（兜底就业）参数 ==========
# 目标失业率（当失业率高于此值时，政府启动公共就业计划）
PUBLIC_EMPLOYMENT_TARGET_UNEMPLOYMENT = 0.15  # 目标：失业率不超过15%
# 公共就业岗位的最低工资（使用较低工资，鼓励市场就业）
PUBLIC_EMPLOYMENT_MIN_WAGE = 15.0  # $15/小时
# 公共就业预算上限（政府每月用于公共就业的最大支出）
PUBLIC_EMPLOYMENT_MAX_BUDGET = 150000.0
PUBLIC_EMPLOYMENT_MATCHING_LOSS_FLOOR = 20000.0
# 公共就业岗位类型（低技能要求的通用岗位）
PUBLIC_EMPLOYMENT_SOC_CODES = [
    "43-9061",  # Office Clerks, General
    "37-2011",  # Janitors and Cleaners
    "53-7062",  # Laborers and Material Movers
    "39-9011",  # Childcare Workers
    "31-1120",  # Home Health and Personal Care Aides
]


class Government:
    """
    # Government Agent
    Distributed government entity managing fiscal policy and taxation.
    
    ## Features
    - Tax policy management
    - LLM-assisted policy updates
    - Budget tracking via EconomicCenter
    - 政府服务费通过 EconomicCenter 记录和收取（非税收入）
    
    ## 政府服务费说明
    政府服务费（规费、牌照费、公立企业服务费等）不在 Government 内部维护，
    而是通过 EconomicCenter 的交易记录系统统一管理。
    当企业采购政府服务时，AbstractResourceMarket 会直接调用 EconomicCenter
    的 record_resource_purchase()，将资金转入政府账户。
    """
    
    def __init__(self,
                 government_id: str,
                 initial_budget: float = 0.0,
                 tax_policy: TaxPolicy = None,
                 economic_center: Optional[EconomicCenter] = None,
                 household_count: Optional[int] = None,
                 procurement_ratio: float = GOVERNMENT_PROCUREMENT_RATIO,
                 demand_injection_ratio: float = GOVERNMENT_DEMAND_INJECTION_RATIO,
                 min_procurement_budget: float = GOVERNMENT_MIN_PROCUREMENT_BUDGET,
                 max_procurement_budget: float = GOVERNMENT_MAX_PROCUREMENT_BUDGET,
                 min_procurement_budget_per_household: float = 500.0,
                 max_procurement_budget_per_household: float = 1166.6667,
                 government_labor_budget_share_of_balance: float = 0.15,
                 government_initial_labor_budget: float = 20000.0,
                 government_min_labor_budget: float = 5000.0,
                 government_max_labor_budget: float = 120000.0,
                 government_max_labor_budget_per_household: float = 400.0,
                 public_employment_target_unemployment: float = PUBLIC_EMPLOYMENT_TARGET_UNEMPLOYMENT,
                 public_employment_min_wage: float = PUBLIC_EMPLOYMENT_MIN_WAGE,
                 public_employment_max_budget: float = PUBLIC_EMPLOYMENT_MAX_BUDGET,
                 public_employment_max_budget_per_household: float = 500.0,
                 public_employment_start_period: int = 2,
                 public_employment_warmup_max_monthly_jobs: int = 20,
                 public_employment_max_monthly_jobs: int = 40,
                 public_employment_max_monthly_job_share: float = 0.10,
                 public_employment_max_new_job_share: float = 0.10,
                 public_employment_max_stock_share: float = 0.20,
                 public_employment_shrink_threshold_multiplier: float = 0.5,
                 public_employment_max_monthly_shrink_ratio: float = 0.20):
        """
        ## Initialize Government Agent
        Creates a new government agent with full tax management capabilities.
        
        ### Parameters
        - `government_id` (str): Unique identifier for the government
        - `initial_budget` (float): Starting budget allocation
        - `tax_policy` (TaxPolicy): Initial tax policy configuration
        - `economic_center` (EconomicCenter): Economic state manager
        
        ### Raises
        - ValueError: If government_id is empty or invalid
        """
        # Validate government ID
        if not government_id or not isinstance(government_id, str):
            raise ValueError("government_id must be a non-empty string")
        
        # Use default policy if none provided
        if tax_policy is None:
            tax_policy = TaxPolicy()
        
        # Store core state directly
        self.government_id = government_id
        self.tax_policy = tax_policy.model_copy()
        self.initial_budget = initial_budget
        self.household_count = max(0, int(household_count or 0))
        self.procurement_ratio = max(0.0, float(procurement_ratio or 0.0))
        self.demand_injection_ratio = max(0.0, float(demand_injection_ratio or 0.0))
        self.min_procurement_budget = max(0.0, float(min_procurement_budget or 0.0))
        self.max_procurement_budget = max(0.0, float(max_procurement_budget or 0.0))
        self.min_procurement_budget_per_household = max(
            0.0, float(min_procurement_budget_per_household or 0.0)
        )
        self.max_procurement_budget_per_household = max(
            0.0, float(max_procurement_budget_per_household or 0.0)
        )
        self.government_labor_budget_share_of_balance = max(
            0.0, float(government_labor_budget_share_of_balance or 0.0)
        )
        self.government_initial_labor_budget = max(0.0, float(government_initial_labor_budget or 0.0))
        self.government_min_labor_budget = max(0.0, float(government_min_labor_budget or 0.0))
        self.government_max_labor_budget = max(0.0, float(government_max_labor_budget or 0.0))
        self.government_max_labor_budget_per_household = max(
            0.0, float(government_max_labor_budget_per_household or 0.0)
        )
        self.public_employment_target_unemployment = min(
            1.0, max(0.0, float(public_employment_target_unemployment or 0.0))
        )
        self.public_employment_min_wage = max(0.0, float(public_employment_min_wage or 0.0))
        self.public_employment_max_budget = max(0.0, float(public_employment_max_budget or 0.0))
        self.public_employment_max_budget_per_household = max(
            0.0, float(public_employment_max_budget_per_household or 0.0)
        )
        self.public_employment_start_period = max(0, int(public_employment_start_period or 0))
        self.public_employment_warmup_max_monthly_jobs = max(
            0, int(public_employment_warmup_max_monthly_jobs or 0)
        )
        self.public_employment_max_monthly_jobs = max(
            0, int(public_employment_max_monthly_jobs or 0)
        )
        self.public_employment_max_monthly_job_share = max(
            0.0, float(public_employment_max_monthly_job_share or 0.0)
        )
        self.public_employment_max_new_job_share = max(
            0.0, float(public_employment_max_new_job_share or 0.0)
        )
        self.public_employment_max_stock_share = min(
            1.0, max(0.0, float(public_employment_max_stock_share or 0.0))
        )
        self.public_employment_shrink_threshold_multiplier = max(
            0.0, float(public_employment_shrink_threshold_multiplier or 0.0)
        )
        self.public_employment_max_monthly_shrink_ratio = max(
            0.0, float(public_employment_max_monthly_shrink_ratio or 0.0)
        )
        # Store dependencies
        self.economic_center = economic_center
        self.logger = get_logger(name="government")

    def _scale_budget_by_households(self, base_budget: float, per_household: float) -> float:
        if self.household_count <= 0 or per_household <= 0:
            return max(0.0, float(base_budget or 0.0))
        return min(max(0.0, float(base_budget or 0.0)), per_household * self.household_count)

    def _effective_min_procurement_budget(self) -> float:
        return self._scale_budget_by_households(
            self.min_procurement_budget,
            self.min_procurement_budget_per_household,
        )

    def _effective_max_procurement_budget(self, min_budget: Optional[float] = None) -> float:
        effective_max = self._scale_budget_by_households(
            self.max_procurement_budget,
            self.max_procurement_budget_per_household,
        )
        floor = self._effective_min_procurement_budget() if min_budget is None else float(min_budget)
        return max(floor, effective_max)

    def _effective_public_employment_budget(self) -> float:
        return self._scale_budget_by_households(
            self.public_employment_max_budget,
            self.public_employment_max_budget_per_household,
        )

    def _effective_government_labor_budget_ceiling(self) -> float:
        return self._scale_budget_by_households(
            self.government_max_labor_budget,
            self.government_max_labor_budget_per_household,
        )

    def initialize(self):
        """
        ## Initialize Government Agent
        Asynchronously initializes the government agent.
        """
        if self.economic_center:
            try:
                ray.get([self.economic_center.init_agent_ledger.remote(self.government_id, self.initial_budget),
                                     self.economic_center.register_id.remote(self.government_id, 'government')]
                )
                self.logger.info(f"Government {self.government_id} registered in EconomicCenter")
            except Exception as e:
                self.logger.warning(f"[Government Init] Failed to register ledger for {self.government_id}: {e}")

    def get_service_fee_summary(self, period: Optional[int] = None) -> Dict:
        """
        从 EconomicCenter 查询政府服务费收入汇总
        
        Args:
            period: 指定期数，None 表示全部
            
        Returns:
            汇总信息 {
                "total": 总金额,
                "by_type": {行业代码: 金额},
                "count": 笔数
            }
        """
        if self.economic_center is None:
            return {"total": 0.0, "by_type": {}, "count": 0}
        
        try:
            # 从 EconomicCenter 查询交易记录
            # 政府服务费的 receiver_id 是 self.government_id，tx_type 是 "resource_purchase"
            transactions = ray.get(
                self.economic_center.get_transactions_by_receiver.remote(
                    receiver_id=self.government_id,
                    tx_type="resource_purchase",
                    month=period
                )
            )
            
            by_type: Dict[str, float] = {}
            total = 0.0
            
            for tx in transactions:
                industry_code = tx.get("metadata", {}).get("industry_code", "unknown")
                amount = tx.get("amount", 0.0)
                
                if industry_code in GOVERNMENT_SERVICE_CODES:
                    by_type[industry_code] = by_type.get(industry_code, 0.0) + amount
                    total += amount
            
            return {
                "total": total,
                "by_type": by_type,
                "count": len([t for t in transactions if t.get("metadata", {}).get("industry_code") in GOVERNMENT_SERVICE_CODES])
            }
        except Exception as e:
            self.logger.error(f"查询政府服务费失败: {e}")
            return {"total": 0.0, "by_type": {}, "count": 0}

    def get_balance(self) -> float:
        """
        从 EconomicCenter 查询政府当前余额
        
        Returns:
            政府账户余额
        """
        if self.economic_center is None:
            return self.initial_budget
        
        try:
            return ray.get(self.economic_center.query_balance.remote(self.government_id))
        except Exception as e:
            self.logger.error(f"查询政府余额失败: {e}")
            return 0.0

    async def update_tax_policy(self, new_policy: TaxPolicy) -> None: 
        """
        ## Update Tax Policy
        Applies new tax policy while ensuring validation and consistency.
        
        ### Parameters
        - `new_policy` (TaxPolicy): Validated tax policy to apply
        
        ### Validation
        - Ensures non-null input
        - Performs deep copy to prevent external state modification
        
        ### Raises
        - ValueError: If new_policy is None
        """
        # Validate new policy
        if not new_policy:
            raise ValueError("new_policy cannot be None")
        
        # Update internal state
        self.tax_policy = new_policy.model_copy()

    # ===========================
    # 政府招工功能
    # ===========================
    
    def set_labor_market(self, labor_market) -> None:
        """设置劳动力市场引用"""
        self.labor_market = labor_market
    
    def _call_labor_market(self, method_name: str, *args, **kwargs):
        """调用劳动力市场的方法"""
        if not hasattr(self, 'labor_market') or self.labor_market is None:
            return None
        method = getattr(self.labor_market, method_name, None)
        if method is None:
            return None
        if 'ActorHandle' in str(type(self.labor_market)):
            return ray.get(method.remote(*args, **kwargs))
        return method(*args, **kwargs)
    
    def _get_government_compensation_ratios(self) -> Dict[str, float]:
        """
        获取各政府部门的 compensation 系数
        
        Returns:
            {行业代码: compensation比率}
        """
        ratios = {}
        for gov_code in GOVERNMENT_SERVICE_CODES:
            try:
                cost_struct = get_cost_structure(gov_code)
                ratios[gov_code] = cost_struct.get("compensation", 0.4)
            except Exception as e:
                self.logger.warning(f"获取 {gov_code} compensation系数失败: {e}")
                ratios[gov_code] = 0.4  # 默认40%
        return ratios
    
    def _compute_labor_budget(self, period: Optional[int] = None) -> float:
        """
        计算政府劳动力预算
        
        基于政府账户余额（税收收入积累）：
        - 政府通过税收（个税+消费税+企业所得税）获得收入
        - 将一部分用于公务员工资（劳动力预算）
        - 现实中政府支出约占 GDP 的 15-20%，其中约 60% 是人员经费
        
        Returns:
            本期可用于雇佣的预算
        """
        # 查询政府账户余额
        gov_balance = 0.0
        if self.economic_center is not None:
            try:
                import ray
                method = getattr(self.economic_center, "query_balance")
                if 'ActorHandle' in str(type(self.economic_center)):
                    gov_balance = float(ray.get(method.remote(self.government_id)) or 0.0)
                else:
                    gov_balance = float(method(self.government_id) or 0.0)
            except Exception:
                pass
        
        if gov_balance <= 0:
            fallback_budget = min(
                self.government_initial_labor_budget,
                self._effective_government_labor_budget_ceiling(),
            )
            self.logger.info(f"政府余额不足({gov_balance:.0f})，使用初始预算: {fallback_budget:.2f}")
            return fallback_budget
        
        # 基于余额计算劳动预算
        labor_budget = gov_balance * self.government_labor_budget_share_of_balance
        budget_ceiling = self._effective_government_labor_budget_ceiling()
        budget_floor = min(self.government_min_labor_budget, budget_ceiling)
        final_budget = max(min(labor_budget, budget_ceiling), budget_floor)
        
        self.logger.info(
            f"政府劳动预算: {final_budget:.2f} "
            f"(余额={gov_balance:.0f}, 计算值={labor_budget:.0f}, "
            f"占比={self.government_labor_budget_share_of_balance:.0%})"
        )
        return final_budget
    
    def _load_government_naics_to_soc(self) -> Dict[str, List[str]]:
        """
        加载政府部门的 NAICS 到 SOC 职业映射
        
        Returns:
            {NAICS代码: [SOC代码列表]}
        """
        # 导入 firm.py 中的函数
        from agenteconomy.agent.firm import _load_naics_to_soc
        
        naics_to_soc, _ = _load_naics_to_soc()
        
        # 筛选政府相关的 NAICS 代码
        gov_naics_codes = ["999100", "999200", "999300"]
        gov_mapping = {}
        for code in gov_naics_codes:
            if code in naics_to_soc:
                gov_mapping[code] = naics_to_soc[code]
        
        return gov_mapping
    
    def _decide_job_postings(self, period: Optional[int] = None, max_job_types: int = 15) -> List[Job]:
        """
        决定政府招工计划
        
        参考企业的 _decide_job_postings_from_data() 逻辑：
        1. 根据 NAICS 到 SOC 的映射找到候选职业
        2. 根据 labor_budget 和工资决定各职位的招聘数量
        
        Args:
            period: 当前期数
            max_job_types: 最大职位类型数
            
        Returns:
            Job 列表
        """
        from agenteconomy.agent.firm import _load_job_skill_data, _load_soc_distribution
        
        job_data = _load_job_skill_data()
        if not job_data:
            self.logger.warning("无法加载职位数据")
            return []
        
        # 获取政府部门的职业列表
        gov_naics_to_soc = self._load_government_naics_to_soc()
        
        # 合并所有政府部门的候选职业
        candidate_socs = []
        for naics_code, socs in gov_naics_to_soc.items():
            for soc in socs:
                if soc not in candidate_socs:
                    candidate_socs.append(soc)
        
        # 如果没有政府特定职业，使用分布最广的职业
        if not candidate_socs:
            distribution = _load_soc_distribution()
            candidate_socs = [
                soc for soc, _ in sorted(
                    distribution.items(), key=lambda kv: kv[1], reverse=True
                )
            ][:max_job_types]
        
        # 过滤出有数据的职业
        candidate_socs = [soc for soc in candidate_socs if soc in job_data]
        if not candidate_socs:
            self.logger.warning("没有找到可用的政府职业")
            return []
        
        # 获取职业分布权重
        distribution = _load_soc_distribution()
        weights = {}
        total_weight = 0.0
        for soc in candidate_socs:
            w = float(distribution.get(soc, 1.0) or 1.0)
            weights[soc] = w
            total_weight += w
        if total_weight <= 0:
            total_weight = float(len(candidate_socs))
        
        # 计算劳动力预算
        labor_budget = self._compute_labor_budget(period)
        if labor_budget <= 0:
            self.logger.info("政府劳动预算不足，不招聘")
            return []
        
        hours_per_week = 40.0
        weeks_per_month = 4.0
        hours_per_period = hours_per_week * weeks_per_month
        
        jobs: List[Job] = []
        remaining_budget = labor_budget
        
        # 按权重排序职业
        ranked_socs = sorted(candidate_socs, key=lambda s: weights.get(s, 0.0), reverse=True)
        
        for soc in ranked_socs[:max_job_types]:
            info = job_data.get(soc)
            if not info:
                continue
            
            hourly_wage = float(info.get("wage", 0.0) or 0.0)
            if hourly_wage <= 0:
                continue
            
            monthly_wage = hourly_wage * hours_per_period
            if monthly_wage <= 0:
                continue
            
            # 根据权重分配预算
            budget_share = labor_budget * (weights.get(soc, 1.0) / total_weight)
            positions = int(budget_share // monthly_wage)
            
            if positions <= 0:
                continue
            
            max_affordable = int(remaining_budget // monthly_wage)
            if max_affordable <= 0:
                continue
            
            positions = min(positions, max_affordable)
            remaining_budget -= positions * monthly_wage
            
            # 创建 Job 对象
            job = Job.create(
                soc=soc,
                title=info.get("title") or soc,
                wage_per_hour=hourly_wage,
                firm_id=self.government_id,  # 使用政府 ID
                description=info.get("description"),
                hours_per_period=hours_per_period,
                required_skills=info.get("skills") or {},
                required_abilities=info.get("abilities") or {},
            )
            job.positions_available = positions
            jobs.append(job)
        
        if jobs:
            total_positions = sum(j.positions_available for j in jobs)
            self.logger.info(
                f"政府招聘计划: {len(jobs)} 个职位类型, {total_positions} 个岗位, "
                f"预算 {labor_budget:.2f}"
            )
        
        return jobs
    
    def _create_public_employment_jobs(self, period: Optional[int] = None) -> List[Job]:
        """
        创建公共就业计划岗位（政府兜底就业）
        
        当失业率超过目标值时，政府发布低技能公益岗位吸收失业人员。
        这些岗位没有技能要求，任何劳动力都可以匹配。
        
        设计原则：
        - 预热期不启动（让企业先招人，避免挤出私人就业）
        - 渐进式雇佣（每月最多创造 MAX_MONTHLY_NEW_JOBS 个岗位）
        - 预算上限约束
        
        Args:
            period: 当前期数
            
        Returns:
            公共就业 Job 列表
        """
        from agenteconomy.agent.firm import _load_job_skill_data
        
        # 早期先让私人部门招聘，再按配置启动兜底就业。
        if period is not None and period < self.public_employment_start_period:
            self.logger.info(f"公共就业计划: 预热期(M{period})，不启动，让企业优先招聘")
            return []
        
        # 获取劳动力市场统计
        labor_stats = self._call_labor_market("summary")
        if not isinstance(labor_stats, dict):
            self.logger.info("无法获取劳动力市场数据，跳过公共就业计划")
            return []
        
        total_labor = int(labor_stats.get("total_labor_hours", 0) or 0)
        total_matched = int(labor_stats.get("total_matched_jobs", 0) or 0)
        current_public_jobs = int(labor_stats.get("gov_matched_jobs", 0) or 0)
        
        if total_labor <= 0:
            return []
        
        unemployment_rate = (total_labor - total_matched) / total_labor
        
        target_unemployment = self.public_employment_target_unemployment
        if target_unemployment <= 0.0:
            return []

        # 检查是否需要启动公共就业计划
        if unemployment_rate <= target_unemployment:
            self.logger.info(
                f"公共就业计划: 失业率 {unemployment_rate:.1%} <= 目标 {target_unemployment:.1%}，无需启动"
            )
            return []
        
        # 计算需要创造的岗位数量（渐进式：每月最多 MAX_MONTHLY_NEW_JOBS 个）
        # 预热期上限更低（20人），正式期 40 人
        if period is not None and period <= 3:
            max_monthly_new_jobs = self.public_employment_warmup_max_monthly_jobs
        else:
            max_monthly_new_jobs = self.public_employment_max_monthly_jobs
        if self.public_employment_max_monthly_job_share > 0.0:
            labor_scaled_max = max(1, ceil(total_labor * self.public_employment_max_monthly_job_share))
            max_monthly_new_jobs = min(max_monthly_new_jobs, labor_scaled_max)
        if self.public_employment_max_new_job_share > 0.0:
            labor_scaled_max = max(1, ceil(total_labor * self.public_employment_max_new_job_share))
            max_monthly_new_jobs = min(max_monthly_new_jobs, labor_scaled_max)
        unemployed = total_labor - total_matched
        target_employed = int(total_labor * (1.0 - target_unemployment))
        jobs_needed = max(0, target_employed - total_matched)
        if self.public_employment_max_stock_share > 0.0:
            max_public_stock = max(1, int(total_labor * self.public_employment_max_stock_share))
            jobs_needed = min(jobs_needed, max(0, max_public_stock - current_public_jobs))
        
        # 渐进式限制 + 预算限制
        hours_per_period = 160.0  # 月工时
        monthly_wage = self.public_employment_min_wage * hours_per_period
        if monthly_wage <= 0.0:
            return []
        max_affordable = int(self._effective_public_employment_budget() / monthly_wage)
        jobs_to_create = min(jobs_needed, max_affordable, max_monthly_new_jobs)
        
        if jobs_to_create <= 0:
            return []
        
        self.logger.info(
            f"公共就业计划: 失业率 {unemployment_rate:.1%} > 目标 {target_unemployment:.1%}, "
            f"失业人数 {unemployed}, 计划创造 {jobs_to_create} 个公益岗位"
        )
        
        # 加载职业数据
        job_data = _load_job_skill_data()
        
        # 创建公益岗位（均分到各SOC类型）
        jobs: List[Job] = []
        soc_codes = PUBLIC_EMPLOYMENT_SOC_CODES.copy()
        positions_per_soc = max(1, jobs_to_create // len(soc_codes))
        remaining = jobs_to_create
        
        for soc in soc_codes:
            if remaining <= 0:
                break
            
            positions = min(positions_per_soc, remaining)
            remaining -= positions
            
            info = job_data.get(soc, {})
            title = str(info.get("title", f"Public Service Worker ({soc})"))
            
            # 公益岗位：无技能要求，任何人都可以匹配
            job = Job(
                job_id=f"pub_{self.government_id}_{period}_{soc}",
                SOC=soc,
                title=title,  # Job模型使用title，不是job_title
                firm_id=self.government_id,
                wage_per_hour=self.public_employment_min_wage,
                hours_per_period=hours_per_period,
                positions_available=positions,
                required_skills={},  # 无技能要求
                required_abilities={},  # 无能力要求
                matching_loss_floor=PUBLIC_EMPLOYMENT_MATCHING_LOSS_FLOOR,
                is_valid=True,
            )
            jobs.append(job)
        
        # 剩余岗位分配给第一个SOC
        if remaining > 0 and jobs:
            jobs[0].positions_available += remaining
        
        total_positions = sum(j.positions_available for j in jobs)
        total_cost = total_positions * monthly_wage
        self.logger.info(
            f"公共就业计划: 创建 {total_positions} 个公益岗位, "
            f"预计月支出 ${total_cost:,.2f}"
        )
        
        return jobs
    
    async def post_jobs(self, period: Optional[int] = None) -> List[Job]:
        """
        发布政府招聘岗位到劳动力市场
        
        包括两部分：
        1. 常规政府岗位（基于服务费收入）
        2. 公共就业计划岗位（兜底失业人员）
        
        Args:
            period: 当前期数
            
        Returns:
            发布的 Job 列表
        """
        # 常规政府岗位
        regular_jobs = self._decide_job_postings(period=period)
        
        # 公共就业计划岗位
        public_jobs = self._create_public_employment_jobs(period=period)
        
        # 合并所有岗位
        jobs = regular_jobs + public_jobs
        
        if jobs:
            # 查询现有岗位快照
            snapshot = self._call_labor_market("get_firm_job_snapshot", self.government_id)
            if not isinstance(snapshot, dict):
                snapshot = {}
            
            to_post: List[Job] = []
            for job in jobs:
                desired = int(job.positions_available or 0)
                existing = int(snapshot.get(job.SOC, 0) or 0)
                delta = desired - existing
                
                if delta <= 0:
                    continue
                
                job.positions_available = delta
                to_post.append(job)
            
            if to_post:
                self._call_labor_market("apply_job_plan", self.government_id, to_post)
                self.logger.info(f"政府发布 {len(to_post)} 个岗位到劳动力市场")
            
            return to_post
        
        return []
    
    def add_employee(self, employee: LaborHour) -> None:
        """添加政府雇员"""
        if not hasattr(self, 'employee_list'):
            self.employee_list = []
            self.employee_count = 0
        self.employee_list.append(employee)
        self.employee_count += 1
    
    def remove_employee(self, employee: LaborHour) -> None:
        """移除政府雇员"""
        if hasattr(self, 'employee_list') and employee in self.employee_list:
            self.employee_list.remove(employee)
            self.employee_count = max(0, self.employee_count - 1)
    
    def get_employees(self) -> List[LaborHour]:
        """获取政府雇员列表"""
        return getattr(self, 'employee_list', [])
    
    def get_employee_count(self) -> int:
        """获取政府雇员数量"""
        return getattr(self, 'employee_count', 0)
    
    # ===========================
    # 政府采购功能
    # ===========================
    
    def set_product_market(self, product_market) -> None:
        """设置产品市场引用"""
        self.product_market = product_market
    
    def _call_product_market(self, method_name: str, *args, **kwargs):
        """调用产品市场的方法"""
        if not hasattr(self, 'product_market') or self.product_market is None:
            return None
        method = getattr(self.product_market, method_name, None)
        if method is None:
            return None
        if 'ActorHandle' in str(type(self.product_market)):
            return ray.get(method.remote(*args, **kwargs))
        return method(*args, **kwargs)
    
    def _call_economic_center(self, method_name: str, *args, **kwargs):
        """调用经济中心的方法"""
        if self.economic_center is None:
            return None
        method = getattr(self.economic_center, method_name, None)
        if method is None:
            return None
        if 'ActorHandle' in str(type(self.economic_center)):
            return ray.get(method.remote(*args, **kwargs))
        return method(*args, **kwargs)
    
    @lru_cache(maxsize=1)
    def _get_government_procurement_weights(self) -> Dict[str, float]:
        """
        从IO表计算政府采购权重
        
        基于 Direct Total Requirements 表中政府部门列的系数
        排除 V001 (补偿), V002, V003 (盈余) 等增值部分
        
        Returns:
            {行业代码: 采购权重} （归一化为总和=1）
        """
        # 找到IO表文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(current_dir), "data")
        io_table_path = os.path.join(
            data_dir, 
            "Direct Total Requirements, After Redefinitions - Summary.csv"
        )
        
        if not os.path.exists(io_table_path):
            self.logger.warning(f"IO表文件不存在: {io_table_path}")
            return {}
        
        try:
            df = pd.read_csv(io_table_path, index_col=0, skiprows=[1])
            df = df.apply(pd.to_numeric, errors='coerce')
            
            # 合并所有政府部门的采购系数
            gov_cols = list(GOVERNMENT_SERVICE_CODES)
            combined = pd.Series(0.0, index=df.index)
            valid_cols = 0
            for col in gov_cols:
                if col in df.columns:
                    combined += df[col].fillna(0)
                    valid_cols += 1
            
            if valid_cols > 0:
                combined = combined / valid_cols  # 平均
            
            # 过滤无效行
            combined = combined[combined.index.notna()]
            for exclude_row in GOVERNMENT_NON_PROCUREMENT_ROWS:
                if exclude_row in combined.index:
                    combined = combined.drop(exclude_row)
            
            # 过滤小于阈值的行业
            combined = combined[combined >= GOVERNMENT_PROCUREMENT_MIN_COEFFICIENT]
            
            # 归一化
            total = combined.sum()
            if total > 0:
                combined = combined / total
            
            result = combined.to_dict()
            
            self.logger.info(
                f"政府采购权重加载完成: {len(result)} 个行业, "
                f"前5名: {list(sorted(result.items(), key=lambda x: -x[1]))[:5]}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"加载政府采购权重失败: {e}")
            return {}
    
    def _compute_procurement_budget(
        self, 
        period: Optional[int] = None,
        household_consumption_budget: Optional[float] = None
    ) -> float:
        """
        计算政府采购预算
        
        采用"无限资金"模式：政府作为需求注入器，不依赖税收
        预算 = 家庭消费总预算 × 需求注入比例
        
        这实现了凯恩斯主义的政府支出乘数效应：
        - 政府支出增加 → 企业收入增加 → 招聘增加 → 工资增加 → 消费增加 → ...
        
        Args:
            period: 当前期数
            household_consumption_budget: 家庭消费总预算（由 simulator 传入）
            
        Returns:
            可用于采购的预算
        """
        # 方法1：基于家庭消费预算（优先）
        if household_consumption_budget is not None and household_consumption_budget > 0:
            # 政府支出 = 家庭消费 × 注入比例
            demand_injection = household_consumption_budget * self.demand_injection_ratio
            
            # 应用上下限约束
            min_budget = self._effective_min_procurement_budget()
            max_budget = self._effective_max_procurement_budget(min_budget)
            procurement_budget = max(demand_injection, min_budget)
            procurement_budget = min(procurement_budget, max_budget)
            
            self.logger.info(
                f"政府采购预算: ${procurement_budget:,.2f} "
                f"(家庭消费=${household_consumption_budget:,.2f} × {self.demand_injection_ratio:.0%} "
                f"= ${demand_injection:,.2f}, 约束后=${procurement_budget:,.2f})"
            )
            return procurement_budget
        
        # 方法2：基于上月税收（兜底）
        min_budget = self._effective_min_procurement_budget()
        max_budget = self._effective_max_procurement_budget(min_budget)
        if self.economic_center is None:
            return min_budget
        
        current_period = period or 0
        prev_period = current_period - 1
        total_tax = 0.0
        
        if prev_period >= 0:
            try:
                tax_summary = self._call_economic_center(
                    "get_monthly_tax_collection", prev_period
                )
                if isinstance(tax_summary, dict):
                    total_tax = float(tax_summary.get("total_tax", 0.0) or 0.0)
            except Exception as e:
                self.logger.warning(f"获取上月税收汇总失败: {e}")
        
        if total_tax > 0:
            tax_based_budget = total_tax * self.procurement_ratio
            procurement_budget = max(tax_based_budget, min_budget)
            procurement_budget = min(procurement_budget, max_budget)
            self.logger.info(
                f"政府采购预算: ${procurement_budget:,.2f} "
                f"(税收=${total_tax:,.2f} × {self.procurement_ratio:.0%}, 最低=${min_budget:,.2f})"
            )
        else:
            # 使用最低预算保障
            procurement_budget = min_budget
            self.logger.info(
                f"政府采购预算: ${procurement_budget:,.2f} (使用最低保障预算)"
            )
        
        return procurement_budget
    
    def procure_goods_and_services(
        self, 
        period: int,
        budget_override: Optional[float] = None,
        household_consumption_budget: Optional[float] = None,
        planned_demand_by_product: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        执行政府采购
        
        从制造商购买商品 + 从服务商购买服务
        使用IO表系数决定各行业的采购比例
        不收取VAT（避免政府自我征税）
        
        政府作为"无限资金"的需求注入器：
        - 预算基于家庭消费总预算的一定比例
        - 即使税收为0，也会有最低采购预算
        - 这样可以启动经济循环
        
        Args:
            period: 当前期数
            budget_override: 可选的预算覆盖（用于测试）
            household_consumption_budget: 家庭消费总预算（用于计算需求注入）
            
        Returns:
            {
                "total_spent": 总支出,
                "by_industry": {行业代码: 支出金额},
                "items_count": 采购项数量,
                "success": 是否成功
            }
        """
        self.logger.info(f"[政府采购] 开始执行采购，期数={period}")
        
        if not hasattr(self, 'product_market') or self.product_market is None:
            self.logger.warning("[政府采购] 失败: 产品市场未设置")
            return {
                "total_spent": 0.0,
                "by_industry": {},
                "items_count": 0,
                "success": False,
                "error": "product_market not set"
            }
        
        # 计算采购预算（优先使用家庭消费预算计算）
        if budget_override is not None:
            budget = budget_override
        else:
            budget = self._compute_procurement_budget(
                period=period,
                household_consumption_budget=household_consumption_budget
            )
        self.logger.info(f"[政府采购] 计算采购预算=${budget:,.2f}")
        
        if budget <= 0:
            self.logger.info("[政府采购] 预算为0，跳过采购")
            return {
                "total_spent": 0.0,
                "by_industry": {},
                "items_count": 0,
                "success": True
            }
        
        # 获取采购权重
        weights = self._get_government_procurement_weights()
        self.logger.info(f"[政府采购] 采购权重加载完成，{len(weights)}个行业")
        
        if not weights:
            self.logger.warning("[政府采购] 采购权重为空，跳过采购")
            return {
                "total_spent": 0.0,
                "by_industry": {},
                "items_count": 0,
                "success": False,
                "error": "no procurement weights"
            }
        
        # First clear the ex-ante SKU demand signal. Production planning has
        # already used this demand, so execution should not strand it behind
        # IO weights for sectors that have no modeled SKU inventory.
        total_spent = 0.0
        spent_by_industry = {}
        items_count = 0
        planned_spent, planned_count, planned_by_industry = self._procure_planned_skus(
            budget=budget,
            period=period,
            weights=weights,
            planned_demand_by_product=planned_demand_by_product or {},
        )
        if planned_spent > 0.0:
            total_spent += planned_spent
            items_count += planned_count
            for industry_code, spent in planned_by_industry.items():
                spent_by_industry[str(industry_code)] = (
                    spent_by_industry.get(str(industry_code), 0.0) + spent
                )

        remaining_budget = max(0.0, budget - total_spent)
        available_industry_weights = self._available_procurement_industry_weights(
            weights,
            period=period,
        )
        weight_total = sum(weight for _, weight in available_industry_weights)
        industries_attempted = len(available_industry_weights)
        industries_succeeded = 0
        if planned_spent > 0.0:
            industries_succeeded += len(planned_by_industry)

        # Use remaining budget only where modeled inventory exists. Raw IO
        # weights contain many sectors absent from the reduced simulation;
        # without this reweighting most government demand silently vanishes.
        for industry_code, weight in available_industry_weights:
            if remaining_budget <= 0.0 or weight_total <= 0.0:
                break
            industry_budget = remaining_budget * (weight / weight_total)
            if industry_budget < 1.0:  # 最小采购金额
                continue
            
            # 尝试从该行业采购
            try:
                spent, count = self._procure_from_industry(
                    industry_code=industry_code,
                    budget=industry_budget,
                    period=period
                )
                if spent > 0:
                    spent_by_industry[industry_code] = spent_by_industry.get(industry_code, 0.0) + spent
                    total_spent += spent
                    items_count += count
                    industries_succeeded += 1
                    self.logger.debug(
                        f"[政府采购] 行业 {industry_code}: "
                        f"预算=${industry_budget:,.2f}, 实际支出=${spent:,.2f}, 采购项={count}"
                    )
            except Exception as e:
                self.logger.warning(f"[政府采购] 从行业 {industry_code} 采购失败: {e}")
                continue
        
        fulfillment_rate = (total_spent / budget * 100) if budget > 0 else 0
        self.logger.info(
            f"[政府采购] 完成: 总预算=${budget:,.2f}, 总支出=${total_spent:,.2f} ({fulfillment_rate:.1f}%), "
            f"尝试{industries_attempted}个行业, 成功{industries_succeeded}个, {items_count}个采购项"
        )
        
        return {
            "total_spent": total_spent,
            "by_industry": spent_by_industry,
            "items_count": items_count,
            "planned_items_count": sum(
                1
                for qty in (planned_demand_by_product or {}).values()
                if self._safe_positive_float(qty) > 0.0
            ),
            "success": True
        }

    def plan_procurement_demand(
        self,
        period: int,
        budget_override: Optional[float] = None,
        household_consumption_budget: Optional[float] = None,
        max_skus_per_industry: int = 3,
    ) -> Dict[str, Any]:
        """
        Build an ex-ante product demand signal for government procurement.

        This method is intentionally non-mutating: it does not reserve stock,
        purchase inventory, or write ledger transactions. Actual procurement is
        still performed by procure_goods_and_services after production.
        """
        if not hasattr(self, "product_market") or self.product_market is None:
            return {
                "budget": 0.0,
                "total_planned_value": 0.0,
                "total_planned_qty": 0.0,
                "demand_by_product": {},
                "by_industry": {},
                "items_count": 0,
                "success": False,
                "error": "product_market not set",
            }

        if budget_override is not None:
            budget = max(0.0, float(budget_override or 0.0))
        else:
            budget = self._compute_procurement_budget(
                period=period,
                household_consumption_budget=household_consumption_budget,
            )
        if budget <= 0.0:
            return {
                "budget": 0.0,
                "total_planned_value": 0.0,
                "total_planned_qty": 0.0,
                "demand_by_product": {},
                "by_industry": {},
                "items_count": 0,
                "success": True,
            }

        weights = self._get_government_procurement_weights()
        if not weights:
            return {
                "budget": budget,
                "total_planned_value": 0.0,
                "total_planned_qty": 0.0,
                "demand_by_product": {},
                "by_industry": {},
                "items_count": 0,
                "success": False,
                "error": "no procurement weights",
            }

        max_skus = max(1, int(max_skus_per_industry or 1))
        demand_by_product: Dict[str, float] = defaultdict(float)
        by_industry: Dict[str, Dict[str, Any]] = {}
        total_planned_value = 0.0
        total_planned_qty = 0.0
        items_count = 0
        industries_attempted = 0
        industries_planned = 0

        industry_candidates: List[Tuple[str, float, List[Any]]] = []
        candidate_weight_total = 0.0
        for industry_code, weight in self._iter_weighted_procurement_industries(weights):
            industries_attempted += 1

            candidates = self._get_procurement_candidate_skus(
                str(industry_code),
                period=period,
                available_only=False,
            )
            candidates = self._select_procurement_plan_candidates(candidates, max_skus)
            if not candidates:
                continue
            industry_candidates.append((str(industry_code), weight, candidates))
            candidate_weight_total += weight

        if candidate_weight_total <= 0.0:
            return {
                "budget": budget,
                "total_planned_value": 0.0,
                "total_planned_qty": 0.0,
                "demand_by_product": {},
                "by_industry": {},
                "items_count": 0,
                "industries_attempted": industries_attempted,
                "industries_planned": 0,
                "success": True,
            }

        for industry_code, weight, candidates in industry_candidates:
            industry_budget = budget * (weight / candidate_weight_total)
            if industry_budget < 1.0:
                continue

            per_sku_budget = industry_budget / len(candidates)
            industry_value = 0.0
            industry_qty = 0.0
            industry_items = 0
            for sku in candidates:
                product_id = getattr(sku, "product_id", None) or getattr(sku, "id", None)
                if not product_id:
                    continue
                unit_price = self._get_procurement_unit_price(sku)
                if unit_price <= 0.0:
                    continue

                quantity = int(per_sku_budget / unit_price)
                if quantity <= 0:
                    continue

                planned_value = float(quantity) * unit_price
                product_key = str(product_id)
                demand_by_product[product_key] += float(quantity)
                industry_value += planned_value
                industry_qty += float(quantity)
                industry_items += 1

            if industry_items <= 0:
                continue

            by_industry[str(industry_code)] = {
                "budget": industry_budget,
                "planned_value": industry_value,
                "planned_qty": industry_qty,
                "items_count": industry_items,
            }
            total_planned_value += industry_value
            total_planned_qty += industry_qty
            items_count += industry_items
            industries_planned += 1

        return {
            "budget": budget,
            "total_planned_value": total_planned_value,
            "total_planned_qty": total_planned_qty,
            "demand_by_product": dict(demand_by_product),
            "by_industry": by_industry,
            "items_count": items_count,
            "industries_attempted": industries_attempted,
            "industries_planned": industries_planned,
            "success": True,
        }

    def _get_procurement_candidate_skus(
        self,
        industry_code: str,
        period: int,
        available_only: bool,
    ) -> List[Any]:
        io_to_name = _load_io_code_to_industry_name()
        industry_name = io_to_name.get(str(industry_code), str(industry_code))

        candidates: List[Any] = []
        seen = set()
        for lookup_key in (industry_name, str(industry_code)):
            if not lookup_key:
                continue
            skus = None
            if not available_only:
                skus = self._call_product_market(
                    "get_skus_by_industry",
                    lookup_key,
                    available_only=False,
                )
            if skus is None:
                skus = self._call_product_market(
                    "get_available_skus",
                    industry=lookup_key,
                    period=period,
                )
            for sku in skus or []:
                product_id = getattr(sku, "product_id", None) or getattr(sku, "id", None)
                if not product_id:
                    continue
                product_key = str(product_id)
                if product_key in seen:
                    continue
                seen.add(product_key)
                candidates.append(sku)

        return candidates

    def _select_procurement_plan_candidates(
        self,
        candidates: List[Any],
        max_skus: int,
    ) -> List[Any]:
        valid = []
        for sku in candidates or []:
            product_id = getattr(sku, "product_id", None) or getattr(sku, "id", None)
            unit_price = self._get_procurement_unit_price(sku)
            if product_id and unit_price > 0.0:
                valid.append(sku)
        valid.sort(
            key=lambda sku: (
                -float(getattr(sku, "available_stock", 0.0) or getattr(sku, "quantity", 0.0) or 0.0),
                str(getattr(sku, "product_id", None) or getattr(sku, "id", "")),
            )
        )
        return valid[:max(1, int(max_skus or 1))]

    def _get_procurement_unit_price(self, sku: Any) -> float:
        for attr in ("manufacturer_price", "base_manufacturer_price", "price"):
            try:
                value = float(getattr(sku, attr, 0.0) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0.0:
                return value
        return 0.0

    @staticmethod
    def _safe_positive_float(value: Any) -> float:
        try:
            return max(0.0, float(value or 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _iter_weighted_procurement_industries(
        self,
        weights: Dict[str, float],
    ) -> List[Tuple[str, float]]:
        weighted_industries: List[Tuple[str, float]] = []
        for industry_code, weight in (weights or {}).items():
            try:
                normalized_weight = max(0.0, float(weight or 0.0))
            except (TypeError, ValueError):
                continue
            if normalized_weight <= 0.0:
                continue
            weighted_industries.append((str(industry_code), normalized_weight))
        return weighted_industries

    def _available_procurement_industry_weights(
        self,
        weights: Dict[str, float],
        period: int,
    ) -> List[Tuple[str, float]]:
        available: List[Tuple[str, float]] = []
        for industry_code, weight in self._iter_weighted_procurement_industries(weights):
            candidates = self._get_procurement_candidate_skus(
                industry_code,
                period=period,
                available_only=True,
            )
            if candidates:
                available.append((industry_code, weight))
                continue

            io_to_name = _load_io_code_to_industry_name()
            industry_name = io_to_name.get(str(industry_code), str(industry_code))
            services = self._call_product_market(
                "get_available_services",
                industry=industry_name,
                period=period,
            )
            if services:
                available.append((industry_code, weight))
        return available

    def _procure_planned_skus(
        self,
        budget: float,
        period: int,
        weights: Dict[str, float],
        planned_demand_by_product: Dict[str, float],
    ) -> Tuple[float, int, Dict[str, float]]:
        if not planned_demand_by_product or budget <= 0.0:
            return 0.0, 0, {}

        planned_ids = {
            str(product_id): self._safe_positive_float(qty)
            for product_id, qty in planned_demand_by_product.items()
            if product_id and self._safe_positive_float(qty) > 0.0
        }
        if not planned_ids:
            return 0.0, 0, {}

        planned_skus: List[Tuple[str, Any]] = []
        seen_products = set()
        for industry_code, _ in self._iter_weighted_procurement_industries(weights):
            candidate_skus = self._get_procurement_candidate_skus(
                industry_code,
                period=period,
                available_only=True,
            )
            for sku in candidate_skus:
                product_id = getattr(sku, "product_id", None) or getattr(sku, "id", None)
                if not product_id:
                    continue
                product_key = str(product_id)
                if product_key not in planned_ids or product_key in seen_products:
                    continue
                seen_products.add(product_key)
                planned_skus.append((industry_code, sku))

        planned_skus.sort(
            key=lambda item: (
                -planned_ids.get(str(getattr(item[1], "product_id", None) or getattr(item[1], "id", "")), 0.0)
                * self._get_procurement_unit_price(item[1]),
                str(getattr(item[1], "product_id", None) or getattr(item[1], "id", "")),
            )
        )

        total_spent = 0.0
        items_purchased = 0
        spent_by_industry: Dict[str, float] = defaultdict(float)
        remaining_budget = max(0.0, float(budget or 0.0))
        for industry_code, sku in planned_skus:
            if remaining_budget <= 0.0:
                break
            product_id = getattr(sku, "product_id", None) or getattr(sku, "id", None)
            planned_qty = planned_ids.get(str(product_id), 0.0)
            if planned_qty <= 0.0:
                continue
            spent, count = self._purchase_procurement_sku(
                sku=sku,
                industry_code=industry_code,
                period=period,
                remaining_budget=remaining_budget,
                desired_quantity=planned_qty,
            )
            if spent <= 0.0 or count <= 0:
                continue
            total_spent += spent
            remaining_budget -= spent
            items_purchased += count
            spent_by_industry[str(industry_code)] += spent

        return total_spent, items_purchased, dict(spent_by_industry)

    def _procure_planned_skus_from_industry(
        self,
        industry_code: str,
        budget: float,
        period: int,
        planned_demand_by_product: Dict[str, float],
    ) -> Tuple[float, int]:
        if not planned_demand_by_product or budget <= 0.0:
            return 0.0, 0

        candidate_skus = self._get_procurement_candidate_skus(
            industry_code,
            period=period,
            available_only=True,
        )
        planned_ids = {
            str(product_id): self._safe_positive_float(qty)
            for product_id, qty in planned_demand_by_product.items()
            if product_id and self._safe_positive_float(qty) > 0.0
        }
        if not planned_ids:
            return 0.0, 0

        planned_skus = []
        for sku in candidate_skus:
            product_id = getattr(sku, "product_id", None) or getattr(sku, "id", None)
            if product_id and str(product_id) in planned_ids:
                planned_skus.append(sku)
        planned_skus.sort(key=lambda sku: str(getattr(sku, "product_id", None) or getattr(sku, "id", "")))

        total_spent = 0.0
        items_purchased = 0
        remaining_budget = max(0.0, float(budget or 0.0))
        for sku in planned_skus:
            if remaining_budget <= 0.0:
                break
            product_id = getattr(sku, "product_id", None) or getattr(sku, "id", None)
            planned_qty = planned_ids.get(str(product_id), 0.0)
            if planned_qty <= 0.0:
                continue
            spent, count = self._purchase_procurement_sku(
                sku=sku,
                industry_code=industry_code,
                period=period,
                remaining_budget=remaining_budget,
                desired_quantity=planned_qty,
            )
            if spent <= 0.0 or count <= 0:
                continue
            total_spent += spent
            remaining_budget -= spent
            items_purchased += count

        return total_spent, items_purchased
    
    def _procure_from_industry(
        self,
        industry_code: str,
        budget: float,
        period: int
    ) -> Tuple[float, int]:
        """
        从指定行业采购
        
        Args:
            industry_code: 行业代码 (IO表代码，如 "325")
            budget: 分配给该行业的预算
            period: 当前期数
            
        Returns:
            (实际支出, 采购项数量)
        """
        # 将 IO 代码转换为行业名称（产品市场使用名称索引）
        io_to_name = _load_io_code_to_industry_name()
        industry_name = io_to_name.get(industry_code, industry_code)
        
        # 获取可用产品（使用行业名称查询）
        available_skus = self._call_product_market(
            'get_available_skus',
            industry=industry_name,
            period=period
        )
        
        if not available_skus:
            # 尝试用原始代码查询（兼容旧逻辑）
            available_skus = self._call_product_market(
                'get_available_skus',
                industry=industry_code,
                period=period
            )
        
        if not available_skus:
            # 尝试服务类行业
            available_skus = self._call_product_market(
                'get_available_services',
                industry=industry_name,
                period=period
            )
        
        if not available_skus:
            return 0.0, 0
        
        import random
        random.shuffle(available_skus)
        
        total_spent = 0.0
        items_purchased = 0
        remaining_budget = budget
        
        for sku in available_skus:
            if remaining_budget <= 0:
                break

            spent, count = self._purchase_procurement_sku(
                sku=sku,
                industry_code=industry_code,
                period=period,
                remaining_budget=remaining_budget,
            )
            if spent <= 0.0 or count <= 0:
                continue
            total_spent += spent
            remaining_budget -= spent
            items_purchased += count
        
        return total_spent, items_purchased

    def _purchase_procurement_sku(
        self,
        sku: Any,
        industry_code: str,
        period: int,
        remaining_budget: float,
        desired_quantity: Optional[float] = None,
    ) -> Tuple[float, int]:
        unit_price = self._get_procurement_unit_price(sku)
        available_stock = getattr(sku, "available_stock", 0) or getattr(sku, "quantity", 0)
        product_id = getattr(sku, "product_id", None) or getattr(sku, "id", "")
        product_name = getattr(sku, "product_name", "") or getattr(sku, "name", "Unknown")

        try:
            available_qty = int(float(available_stock or 0.0))
        except (TypeError, ValueError):
            available_qty = 0
        if unit_price <= 0.0 or available_qty <= 0 or not product_id:
            return 0.0, 0

        max_affordable = int(float(remaining_budget or 0.0) / unit_price)
        desired_cap = available_qty
        if desired_quantity is not None:
            desired_cap = min(desired_cap, int(max(0.0, float(desired_quantity or 0.0))))
        quantity_to_buy = min(max_affordable, desired_cap)
        if quantity_to_buy <= 0:
            return 0.0, 0

        stock_result = self._call_product_market(
            "purchase_manufacturer_stock",
            product_id,
            quantity_to_buy,
        ) or {}
        actual_quantity = int(float(stock_result.get("actual_quantity", 0.0) or 0.0))
        if actual_quantity <= 0:
            return 0.0, 0

        purchase_cost = actual_quantity * unit_price
        receiver_id = getattr(sku, "firm_id", None) or getattr(sku, "seller_id", None)
        if receiver_id is None:
            receiver_id = self._call_product_market("get_seller_id", product_id)
        if receiver_id is None:
            receiver_id = f"market_{industry_code}"

        try:
            self._call_economic_center(
                "add_government_procurement_transaction",
                month=period,
                sender_id=self.government_id,
                receiver_id=receiver_id,
                amount=purchase_cost,
                product_id=product_id,
                quantity=actual_quantity,
                product_name=product_name,
                unit_price=unit_price,
                product_classification=industry_code,
                consume_inventory=True,
            )
            return purchase_cost, 1
        except Exception as e:
            self._call_product_market("restore_manufacturer_stock", product_id, actual_quantity)
            self.logger.warning(f"政府采购交易失败 {product_id}: {e}")
            return 0.0, 0
