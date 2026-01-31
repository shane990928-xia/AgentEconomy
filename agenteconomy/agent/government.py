from typing import Optional, Dict, List, Any, Tuple
from functools import lru_cache
from collections import defaultdict
import os
import pandas as pd
from agenteconomy.center.Model import *
from agenteconomy.center.Ecocenter import EconomicCenter
from agenteconomy.utils.logger import get_logger
from agenteconomy.utils.load_io_table import get_cost_structure
import ray


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
                 economic_center: Optional[EconomicCenter] = None):
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
        # Store dependencies
        self.economic_center = economic_center
        self.logger = get_logger(name="government")

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
        
        基于政府服务费收入（企业购买政府服务支付的费用）：
        - 政府服务费 × compensation系数 = 劳动力预算
        
        逻辑：服务费收入 → 雇佣公务员 → 提供政府服务
        
        如果没有服务费收入（如预热第一个月），使用初始预算。
        
        Returns:
            本期可用于雇佣的预算
        """
        # 初始预算：保证政府在第一个月也能招聘公务员
        INITIAL_GOVERNMENT_LABOR_BUDGET = 50000.0  # 政府初始劳动预算
        
        # 获取政府服务费收入
        service_fee_summary = self.get_service_fee_summary(period=period)
        service_fee_income = service_fee_summary.get("total", 0.0)
        by_type = service_fee_summary.get("by_type", {})
        
        if service_fee_income <= 0:
            self.logger.info(f"政府服务费收入为0，使用初始预算: {INITIAL_GOVERNMENT_LABOR_BUDGET:.2f}")
            return INITIAL_GOVERNMENT_LABOR_BUDGET
        
        # 获取各政府部门的 compensation 比率
        ratios = self._get_government_compensation_ratios()
        
        # 按各部门服务费收入加权计算劳动力预算
        labor_budget = 0.0
        for gov_code, fee_amount in by_type.items():
            comp_ratio = ratios.get(gov_code, 0.4)
            # 服务费收入 × compensation系数 = 该部门的劳动力预算
            labor_budget += fee_amount * comp_ratio
        
        # 如果没有分类数据，使用平均比率
        if labor_budget <= 0 and service_fee_income > 0:
            avg_ratio = sum(ratios.values()) / len(ratios) if ratios else 0.4
            labor_budget = service_fee_income * avg_ratio
        
        self.logger.info(
            f"政府劳动预算: {labor_budget:.2f} "
            f"(服务费收入={service_fee_income:.2f}, 部门数={len(by_type)})"
        )
        return max(0.0, labor_budget)
    
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
    
    async def post_jobs(self, period: Optional[int] = None) -> List[Job]:
        """
        发布政府招聘岗位到劳动力市场
        
        Args:
            period: 当前期数
            
        Returns:
            发布的 Job 列表
        """
        jobs = self._decide_job_postings(period=period)
        
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
    
    def _compute_procurement_budget(self, period: Optional[int] = None) -> float:
        """
        计算政府采购预算
        
        基于税收收入的一部分作为采购预算
        
        Args:
            period: 当前期数
            
        Returns:
            可用于采购的预算
        """
        if self.economic_center is None:
            return 0.0
        
        # 获取税收收入
        try:
            tax_summary = self._call_economic_center(
                "get_monthly_tax_collection", period or 0
            )
            if not tax_summary:
                tax_summary = {}
        except Exception as e:
            self.logger.warning(f"获取税收汇总失败: {e}")
            tax_summary = {}
        
        # 计算总税收
        total_tax = tax_summary.get("total_tax", 0.0) if isinstance(tax_summary, dict) else 0.0
        
        if total_tax <= 0:
            # 尝试从余额推算
            balance = self.get_balance()
            # 使用余额的一部分作为预算（保守策略）
            procurement_budget = balance * GOVERNMENT_PROCUREMENT_RATIO * 0.5
        else:
            # 税收 × 采购比例 = 采购预算
            procurement_budget = total_tax * GOVERNMENT_PROCUREMENT_RATIO
        
        self.logger.info(
            f"政府采购预算: {procurement_budget:.2f} "
            f"(税收={total_tax:.2f}, 比例={GOVERNMENT_PROCUREMENT_RATIO})"
        )
        
        return max(0.0, procurement_budget)
    
    def procure_goods_and_services(
        self, 
        period: int,
        budget_override: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        执行政府采购
        
        从制造商购买商品 + 从服务商购买服务
        使用IO表系数决定各行业的采购比例
        不收取VAT（避免政府自我征税）
        
        Args:
            period: 当前期数
            budget_override: 可选的预算覆盖（用于测试）
            
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
        
        # 计算采购预算
        budget = budget_override if budget_override is not None else self._compute_procurement_budget(period)
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
        
        # 按行业分配预算并采购
        total_spent = 0.0
        spent_by_industry = {}
        items_count = 0
        industries_attempted = 0
        industries_succeeded = 0
        
        for industry_code, weight in weights.items():
            industry_budget = budget * weight
            if industry_budget < 1.0:  # 最小采购金额
                continue
            
            industries_attempted += 1
            
            # 尝试从该行业采购
            try:
                spent, count = self._procure_from_industry(
                    industry_code=industry_code,
                    budget=industry_budget,
                    period=period
                )
                if spent > 0:
                    spent_by_industry[industry_code] = spent
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
            "success": True
        }
    
    def _procure_from_industry(
        self,
        industry_code: str,
        budget: float,
        period: int
    ) -> Tuple[float, int]:
        """
        从指定行业采购
        
        Args:
            industry_code: 行业代码
            budget: 分配给该行业的预算
            period: 当前期数
            
        Returns:
            (实际支出, 采购项数量)
        """
        # 获取可用产品
        available_skus = self._call_product_market(
            'get_available_skus',
            industry=industry_code,
            period=period
        )
        
        if not available_skus:
            # 尝试服务类行业
            available_skus = self._call_product_market(
                'get_available_services',
                industry=industry_code,
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
            
            # 获取SKU信息
            unit_price = getattr(sku, 'manufacturer_price', None) or getattr(sku, 'price', 0)
            available_stock = getattr(sku, 'available_stock', 0) or getattr(sku, 'quantity', 0)
            product_id = getattr(sku, 'product_id', None) or getattr(sku, 'id', '')
            product_name = getattr(sku, 'product_name', '') or getattr(sku, 'name', 'Unknown')
            
            if unit_price <= 0 or available_stock <= 0:
                continue
            
            # 计算可购买数量
            max_affordable = int(remaining_budget / unit_price)
            quantity_to_buy = min(max_affordable, int(available_stock))
            
            if quantity_to_buy <= 0:
                continue
            
            purchase_cost = quantity_to_buy * unit_price
            
            # 确定卖方ID
            receiver_id = getattr(sku, 'firm_id', None) or getattr(sku, 'seller_id', None)
            if receiver_id is None:
                # 尝试从产品市场获取
                receiver_id = self._call_product_market(
                    'get_seller_id', product_id
                )
            if receiver_id is None:
                receiver_id = f"market_{industry_code}"
            
            # 执行采购交易（不含VAT）
            try:
                self._call_economic_center(
                    "add_government_procurement_transaction",
                    month=period,
                    sender_id=self.government_id,
                    receiver_id=receiver_id,
                    amount=purchase_cost,
                    product_id=product_id,
                    quantity=quantity_to_buy,
                    product_name=product_name,
                    unit_price=unit_price,
                    product_classification=industry_code,
                    consume_inventory=True
                )
                
                # 更新库存
                self._call_product_market('update_stock', product_id, -quantity_to_buy)
                
                total_spent += purchase_cost
                remaining_budget -= purchase_cost
                items_purchased += 1
                
            except Exception as e:
                self.logger.warning(
                    f"政府采购交易失败 {product_id}: {e}"
                )
                continue
        
        return total_spent, items_purchased