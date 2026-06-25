from dotenv import load_dotenv
load_dotenv()
from agenteconomy.utils.logger import get_logger
logger = get_logger(name="simulator")
from agenteconomy.center.Model import *
from config.config import SimulationConfig
from agenteconomy.center.Ecocenter import EconomicCenter
from agenteconomy.center.LaborMarket import LaborMarket
from agenteconomy.center.ProductMarket import ProductMarket
from agenteconomy.agent.firm import Firm, ManufactureFirm, RetailFirm
from agenteconomy.agent.household import Household, JobMatch, CategoryPlan, consumption_progress
from agenteconomy.agent.government import Government
from agenteconomy.agent.bank import Bank
from agenteconomy.simulation.agent_loader import create_firms, create_households
from agenteconomy.simulation.checkpoint import CheckpointManager
from agenteconomy.simulation.init_calibration import (
    FirmInitializationCalibrator,
    FirmInitCalibrationInput,
)
from agenteconomy.market.AbstractResourceMarket import AbstractResourceMarket
from agenteconomy.utils.accounting_invariants import check_accounting_invariants
from datetime import datetime
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
from contextlib import contextmanager
import ray
import hashlib

# 家庭服务消费类别 → 行业代码映射
# 用于将 step0 的预算分配类别映射到 AbstractResourceMarket 的行业代码
# 格式: {类别: [(行业代码, 权重), ...]}
# 权重基于 IO 表中各行业占该类别总消费的比例（近似值）
HOUSEHOLD_SERVICE_CATEGORY_TO_INDUSTRY = {
    # Housing: HS 是主要住房支出，ORE 是其他房产相关（如物业管理）
    "housing": [("HS", 0.85), ("ORE", 0.15)],
    # Healthcare: 门诊最多，医院次之
    "healthcare": [("621", 0.45), ("622", 0.35), ("623", 0.12), ("624", 0.08)],
    # Transportation: 卡车运输（网购配送）、航空、公交
    "transportation": [("484", 0.40), ("481", 0.25), ("485", 0.25), ("487OS", 0.10)],
    # Utilities: 只有电力/燃气/水
    "utilities": [("22", 1.0)],
    # Insurance: 只有保险
    "insurance": [("524", 1.0)],
}


class Simulator:
    def __init__(self, config:SimulationConfig):
        """
        Initialize Simulator with configuration
        
        Args:
            config: SimulationConfig instance loaded from YAML
        """
        self.config: SimulationConfig = config

        # 把家庭/工资缩放从环境变量提升为 config 驱动（可复现）；在任何 agent 创建前设置。
        _hds = float(getattr(config, "household_dollar_scale", 1.0) or 1.0)
        _wsi = float(getattr(config, "wage_scale_init", 1.0) or 1.0)
        if _hds != 1.0:
            os.environ["AGENTECO_HOUSEHOLD_SCALE"] = str(_hds)
        if _wsi != 1.0:
            os.environ["AGENTECO_WAGE_SCALE"] = str(_wsi)

        # Basic entities and markets
        self.economic_center: Optional[EconomicCenter] = None
        self.labor_market: Optional[LaborMarket] = None
        self.product_market: Optional[ProductMarket] = None
        self.firms: Optional[List[Firm]] = None
        self.households: Optional[List[Household]] = None
        self.government: Optional[Government] = None
        self.bank: Optional[Bank] = None
        self.manufacturers_by_industry: Dict[str, ManufactureFirm] = {}
        self.retailers_by_industry: Dict[str, RetailFirm] = {}
        self.abstract_resource_market: Optional[AbstractResourceMarket] = None
        self._firm_by_id: Dict[str, Firm] = {}
        self._household_by_id: Dict[str, Household] = {}
        
        self.current_month = 1
        self._record_dir: Optional[str] = None
        self._record_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._stage_event_seq = 0
        self._last_price_index: Optional[float] = None  # 基于 100 的价格指数
        self._last_inflation_rate: Optional[float] = None  # 上次计算的通胀率
        self._last_balance_by_household: Dict[str, float] = {}
        self._last_expected_income_by_household: Dict[str, float] = {}
        self._last_sales_by_product: Dict[str, float] = {}
        self._last_planned_demand_by_product: Dict[str, float] = {}
        self._last_unmet_demand_by_product: Dict[str, float] = {}
        self._last_demand_stats: Dict[str, Any] = {}
        self._last_production_gap_value_by_firm: Dict[str, float] = {}
        self._last_production_value_by_firm: Dict[str, float] = {}
        self._last_service_value_by_industry: Dict[str, float] = {}
        self._firm_labor_priority: Dict[str, float] = {}
        self._last_gdp_comprehensive: Optional[Dict[str, Any]] = None  # 缓存的GDP计算结果
        self._last_initial_inventory_calibration: Dict[str, Any] = {}
        
        # 固定消费篮子（用于价格指数计算）
        # 在预热最后一个月结束时设置，之后保持不变
        self._fixed_consumption_basket: Optional[Dict[str, Dict[str, float]]] = None  # {sku_id: {base_price, weight}}
        
        # 配置更大的线程池以支持更高的并发度
        # 默认线程池大小是 min(32, cpu_count+4)，对于大量household并发消费不够用
        # 这里设置为 household 数量的 2 倍 或最小 64
        self._thread_pool_size = min(512, max(self.config.num_households, 64))
        self._thread_pool: Optional[ThreadPoolExecutor] = None

        # Checkpoint 管理器
        checkpoint_dir = os.path.join(
            getattr(self.config, "checkpoint_output_dir", "output/checkpoints"),
            self._record_run_id
        )
        self._checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            compress=getattr(self.config, "checkpoint_compress", True)
        )
        self._save_checkpoint_interval: int = getattr(self.config, "checkpoint_interval", 1)
        
        # 配置 LLM 并发限制（使用配置文件中的值）
        llm_concurrency = int(getattr(self.config, "max_llm_concurrent", 400))
        try:
            from agenteconomy.llm.llm import configure_concurrency
            configure_concurrency(llm_concurrency)
        except ImportError:
            logger.warning("Could not import LLM module for concurrency configuration")
        
        # Metrics
        
        logger.info(f"Simulator initialized with {self.config.num_months} months and {self.config.num_households} households")
        logger.info(f"Checkpoint enabled: interval={self._save_checkpoint_interval}, dir={checkpoint_dir}")
        logger.info(f"LLM concurrency: {llm_concurrency}")

    async def setup_simulation_environment(self):
        """Setup simulation environment"""
        logger.info("Setting up simulation environment...")
        
        try:
            if self.config.enable_progressive_tax_system:
                tax_policy = TaxPolicy(
                    income_tax_rate=self.config.gov_tax_brackets,
                    corporate_tax_rate=self.config.corporate_tax_rate,
                    vat_rate=self.config.vat_rate,
                    fica_tax_rate=getattr(self.config, "fica_tax_rate", 0.0),
                )
            # Initialize core components (pass in tax policy)
            self.economic_center = EconomicCenter.remote(
                tax_policy=tax_policy
            )
            self.product_market = ProductMarket.remote()
            ray.get(self.product_market.initialize_products.remote())
            # LaborMarket needs economic_center for wage transfers
            self.labor_market = LaborMarket.remote(economic_center=self.economic_center)
            ray.get(self.economic_center.set_labor_market.remote(self.labor_market))
            
            self.government = Government(
                    government_id="gov_main_simulation",
                    initial_budget=10000000.0,
                    tax_policy=tax_policy,
                    economic_center=self.economic_center,
                    household_count=self.config.num_households,
                    procurement_ratio=self.config.government_procurement_ratio,
                    demand_injection_ratio=self.config.government_demand_injection_ratio,
                    min_procurement_budget=self.config.government_min_procurement_budget,
                    max_procurement_budget=self.config.government_max_procurement_budget,
                    min_procurement_budget_per_household=(
                        self.config.government_min_procurement_budget_per_household
                    ),
                    max_procurement_budget_per_household=(
                        self.config.government_max_procurement_budget_per_household
                    ),
                    government_labor_budget_share_of_balance=(
                        self.config.government_labor_budget_share_of_balance
                    ),
                    government_initial_labor_budget=self.config.government_initial_labor_budget,
                    government_min_labor_budget=self.config.government_min_labor_budget,
                    government_max_labor_budget=self.config.government_max_labor_budget,
                    government_max_labor_budget_per_household=(
                        self.config.government_max_labor_budget_per_household
                    ),
                    public_employment_target_unemployment=(
                        self.config.public_employment_target_unemployment
                    ),
                    public_employment_min_wage=self.config.public_employment_min_wage,
                    public_employment_max_budget=self.config.public_employment_max_budget,
                    public_employment_max_budget_per_household=(
                        self.config.public_employment_max_budget_per_household
                    ),
                    public_employment_start_period=self.config.public_employment_start_period,
                    public_employment_warmup_max_monthly_jobs=(
                        self.config.public_employment_warmup_max_monthly_jobs
                    ),
                    public_employment_max_monthly_jobs=(
                        self.config.public_employment_max_monthly_jobs
                    ),
                    public_employment_max_monthly_job_share=(
                        self.config.public_employment_max_monthly_job_share
                    ),
                    public_employment_max_new_job_share=(
                        self.config.public_employment_max_new_job_share
                    ),
                    public_employment_max_stock_share=(
                        self.config.public_employment_max_stock_share
                    ),
                    public_employment_shrink_threshold_multiplier=(
                        self.config.public_employment_shrink_threshold_multiplier
                    ),
                    public_employment_max_monthly_shrink_ratio=(
                        self.config.public_employment_max_monthly_shrink_ratio
                    ),
                )

            self.government.initialize()
            self.government.set_labor_market(self.labor_market)  # 设置劳动力市场引用
            self.government.set_product_market(self.product_market)  # 设置产品市场引用
            
            # Initialize bank
            self.bank = Bank(
                bank_id="bank",
                initial_capital=1000000.0,
                economic_center=self.economic_center
            )
            self.bank.initialize()
            logger.info("Bank system initialized")
            
            # Load simulation data
            logger.info("Loading simulation data...")
            
            # Create households
            self._create_households()
            
            # Create firms
            self._create_firms()
            self._initialize_firm_assets_without_preheat()
            
            # Verify creation results
            if len(self.households) == 0:
                logger.error("No households created")
                return False
            
            if len(self.firms) == 0:
                logger.error("No firms created")
                return False
            
            return True

        except Exception as e:
            logger.error(f"Simulation environment setup failed: {e}")
            return False

    def _create_households(self):
        """Create households"""
        self.households = create_households(
            limit=self.config.num_households,
            economic_center=self.economic_center,
            labor_market=self.labor_market,
            product_market=self.product_market,
        )
        self._household_by_id = {hh.household_id: hh for hh in (self.households or [])}

    def _create_firms(self):
        """Create firms"""
        # 初始化抽象资源市场
        # 传入 EconomicCenter 以记录所有交易
        # 传入政府 ID 以将政府服务费路由到政府账户
        self.abstract_resource_market = AbstractResourceMarket(
            economic_center=self.economic_center,
            government_id=self.government.government_id
        )
        
        # 初始化抽象资源的基准价格和供给能力
        from agenteconomy.market.initialize_resources import initialize_abstract_resource_market
        economy_scale = float(getattr(self.config, "economy_scale", 10_000_000) or 10_000_000)
        initialize_abstract_resource_market(self.abstract_resource_market, economy_scale)
        
        self.firms = create_firms(
            economic_center=self.economic_center, 
            labor_market=self.labor_market, 
            product_market=self.product_market, 
            abstract_resource_market=self.abstract_resource_market,
            limit=self.config.num_firms,
        )
        self._firm_by_id = {f.firm_id: f for f in (self.firms or [])}
        self._index_firms()
        if self.economic_center is not None and self.firms:
            # 批量注册企业 ID 和初始化账本（并行执行）
            register_args = [(firm.firm_id, "firm") for firm in self.firms]
            ledger_args = [(firm.firm_id, 0.0) for firm in self.firms]
            self._call_actor_batch(self.economic_center, "register_id", register_args)
            self._call_actor_batch(self.economic_center, "init_agent_ledger", ledger_args)
            # 注册企业行业映射，供 GDP/部门流核算按行业归集增加值与销售
            # （否则 _get_firm_industry 全部返回 'Unknown'，行业分桶塌缩为一类）。
            industry_map = {
                firm.firm_id: str(getattr(firm, "industry", "") or "")
                for firm in self.firms
            }
            self._call_actor(self.economic_center, "register_firm_industries", industry_map)

        # 预注册所有制造商到中间品市场价格注册表
        # 这样在生产时可以正确解析receiver_id
        self._register_manufacturers_for_intermediate_goods()

    def _call_actor(self, actor, method_name: str, *args, **kwargs):
        """单个 Actor 调用（同步阻塞）"""
        if actor is None:
            return None
        method = getattr(actor, method_name, None)
        if method is None:
            return None
        if hasattr(method, "remote"):
            return ray.get(method.remote(*args, **kwargs))
        return method(*args, **kwargs)
    
    def _call_actor_batch(self, actor, method_name: str, args_list: List[tuple]) -> List[Any]:
        """
        批量调用 Actor 方法（并行执行，一次性等待所有结果）
        
        Args:
            actor: Ray Actor 实例
            method_name: 方法名称
            args_list: 参数列表，每个元素是一个 tuple，如 [(arg1, arg2), (arg3, arg4), ...]
            
        Returns:
            结果列表，与 args_list 顺序对应
        """
        if actor is None or not args_list:
            return []
        method = getattr(actor, method_name, None)
        if method is None:
            return []
        
        if hasattr(method, "remote"):
            # 收集所有 futures
            futures = [method.remote(*args) for args in args_list]
            # 一次性等待所有结果
            return ray.get(futures)
        else:
            # 非 Ray Actor，直接顺序调用
            return [method(*args) for args in args_list]
    
    def _call_actor_batch_kwargs(self, actor, method_name: str, kwargs_list: List[dict]) -> List[Any]:
        """
        批量调用 Actor 方法（使用 kwargs，并行执行）
        
        Args:
            actor: Ray Actor 实例
            method_name: 方法名称
            kwargs_list: 关键字参数列表，如 [{"id": 1}, {"id": 2}, ...]
            
        Returns:
            结果列表
        """
        if actor is None or not kwargs_list:
            return []
        method = getattr(actor, method_name, None)
        if method is None:
            return []
        
        if hasattr(method, "remote"):
            futures = [method.remote(**kwargs) for kwargs in kwargs_list]
            return ray.get(futures)
        else:
            return [method(**kwargs) for kwargs in kwargs_list]

    def _index_firms(self):
        self.manufacturers_by_industry = {}
        self.retailers_by_industry = {}
        # 建立行业名称到代码的反向映射（用于匹配产品的 manufacturer_code）
        self._industry_name_to_code: Dict[str, str] = {}
        # 建立行业代码到名称的映射（用于供给记录时的键转换）
        self._industry_code_to_name: Dict[str, str] = {}

        for firm in self.firms or []:
            if not firm.industry:
                continue
            if isinstance(firm, ManufactureFirm):
                self.manufacturers_by_industry[firm.industry] = firm
                # 同时建立名称到代码的映射
                if hasattr(firm, 'industry_name') and firm.industry_name:
                    self._industry_name_to_code[firm.industry_name] = firm.industry
            elif isinstance(firm, RetailFirm):
                self.retailers_by_industry[firm.industry] = firm

        # 从 industry_cate_map 补充名称到代码的映射（及反向映射）
        from agenteconomy.data.industry_cate_map import industry_cate_map
        cat1 = industry_cate_map.get("category_1_manufacturers", {}).get("industries", {})
        for code, name in cat1.items():
            self._industry_name_to_code[name] = code
            self._industry_code_to_name[code] = name
            # 同时用名称作为 key 注册制造商（如果存在）
            if code in self.manufacturers_by_industry:
                self.manufacturers_by_industry[name] = self.manufacturers_by_industry[code]

        logger.info(f"[索引] 制造商: {len(self.manufacturers_by_industry)} 个, 零售商: {len(self.retailers_by_industry)} 个")
    
    async def resume_from_checkpoint(self, checkpoint_path: Optional[str] = None) -> bool:
        """
        从 Checkpoint 恢复模拟状态并继续执行
        
        Args:
            checkpoint_path: Checkpoint 文件路径。如果为 None，则使用最新的 checkpoint
            
        Returns:
            是否成功恢复
        """
        # 找到 checkpoint 文件
        if checkpoint_path is None:
            checkpoint_path = self._checkpoint_manager.get_latest_checkpoint()
        
        if checkpoint_path is None:
            logger.error("没有找到可用的 Checkpoint 文件")
            return False
        
        logger.info(f"从 Checkpoint 恢复: {checkpoint_path}")
        
        try:
            # 加载 checkpoint 数据
            checkpoint_data = self._checkpoint_manager.load_checkpoint(checkpoint_path)
            
            # 验证配置兼容性
            saved_config = checkpoint_data.get("config", {})
            if saved_config.get("num_households") != self.config.num_households:
                logger.warning(
                    f"Checkpoint 家庭数量 ({saved_config.get('num_households')}) "
                    f"与当前配置 ({self.config.num_households}) 不匹配，可能导致恢复不完整"
                )
            
            # 恢复状态
            self._checkpoint_manager.restore_simulator(self, checkpoint_data)
            
            # 更新 current_month 以从下一月开始
            resume_month = int(checkpoint_data.get("month", 1))
            self.current_month = resume_month + 1
            
            logger.info(f"✅ 状态恢复成功，将从月份 {self.current_month} 继续执行")
            return True
            
        except Exception as e:
            logger.error(f"从 Checkpoint 恢复失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_simulation_from_checkpoint(self, checkpoint_path: Optional[str] = None):
        """
        从 Checkpoint 恢复并继续运行模拟

        Args:
            checkpoint_path: Checkpoint 文件路径。如果为 None，则使用最新的 checkpoint
        """
        # 先恢复状态
        if not await self.resume_from_checkpoint(checkpoint_path):
            logger.error("无法从 Checkpoint 恢复，退出")
            return

        # 刷新家庭就业状态（从 LaborMarket 同步）
        self._refresh_household_employment_status()
        # 刷新企业员工数（从 LaborMarket 同步）
        self._refresh_firm_employee_count()

        # 配置线程池
        loop = asyncio.get_running_loop()
        self._thread_pool = ThreadPoolExecutor(
            max_workers=self._thread_pool_size,
            thread_name_prefix="household_consumption"
        )
        loop.set_default_executor(self._thread_pool)
        logger.info(f"Configured thread pool with {self._thread_pool_size} workers")
        
        try:
            # 从恢复的月份继续执行
            for month in range(self.current_month, self.config.num_months + 1):
                await self._run_month(month)
        finally:
            if self._thread_pool is not None:
                self._thread_pool.shutdown(wait=False)
                self._thread_pool = None

    def _register_manufacturers_for_intermediate_goods(self):
        """
        预注册所有制造商到中间品市场价格注册表
        这样在中间品采购时可以正确解析receiver_id
        """
        if self.economic_center is None:
            return
        
        registered_count = 0
        for industry_code, firm in self.manufacturers_by_industry.items():
            self._call_actor(
                self.economic_center, 
                "register_market_price", 
                "intermediate_goods", 
                industry_code, 
                firm.firm_id, 
                1.0
            )
            registered_count += 1
        
        logger.info(f"Pre-registered {registered_count} manufacturers for intermediate goods market")

    def _debug_enabled(self) -> bool:
        return bool(getattr(self.config, "debug_logging", True))

    def _limit_list(self, items: List[Any], limit: int) -> Tuple[List[Any], int]:
        if not items:
            return [], 0
        if limit and limit > 0 and len(items) > limit:
            return list(items)[:limit], len(items) - limit
        return list(items), 0

    def _format_firm(self, firm_id: str) -> str:
        firm = self._firm_by_id.get(firm_id)
        if firm is None:
            return str(firm_id)
        industry = firm.industry or "unknown"
        return f"{firm_id}({industry})"

    def _labor_priority_for_firm(self, firm_id: Optional[str]) -> float:
        if not firm_id:
            return 0.0
        return float(self._firm_labor_priority.get(str(firm_id), 0.0) or 0.0)

    def _prioritize_labor_jobs(self, jobs: List[Job]) -> List[Job]:
        if not jobs:
            return []
        demand_weight = max(0.0, float(getattr(self.config, "labor_match_demand_priority_weight", 0.0) or 0.0))
        demand_wage_bonus = max(0.0, float(getattr(self.config, "labor_offer_demand_wage_bonus", 0.0) or 0.0))

        for job in jobs:
            priority = self._labor_priority_for_firm(str(getattr(job, "firm_id", "") or ""))
            setattr(job, "demand_priority", priority)
            setattr(job, "demand_wage_bonus", demand_wage_bonus)

        def sort_key(job: Job) -> Tuple[int, float, float]:
            firm_id = str(getattr(job, "firm_id", "") or "")
            is_government = 1 if firm_id.startswith("gov_") else 0
            priority = self._labor_priority_for_firm(firm_id)
            wage = float(getattr(job, "wage_per_hour", 0.0) or 0.0)
            return (is_government, -priority * demand_weight, -wage)

        return sorted(list(jobs), key=sort_key)

    def _prioritize_labor_matches(self, matches: List[JobMatch]) -> List[JobMatch]:
        if not matches:
            return []
        demand_weight = max(0.0, float(getattr(self.config, "labor_match_demand_priority_weight", 0.0) or 0.0))

        def sort_key(match: JobMatch) -> Tuple[int, float, float, float]:
            job = match.job
            firm_id = str(getattr(job, "firm_id", "") or "")
            is_government = 1 if firm_id.startswith("gov_") else 0
            priority = self._labor_priority_for_firm(firm_id)
            wage = float(getattr(job, "wage_per_hour", 0.0) or 0.0)
            adjusted_loss = float(match.loss or 0.0) - priority * demand_weight
            return (is_government, adjusted_loss, -priority, -wage)

        return sorted(list(matches), key=sort_key)

    def _econ_month(self, month: int, preheat: bool) -> int:
        offset = int(getattr(self.config, "preheat_months", 0) or 0)
        if preheat or offset <= 0:
            return month
        return month + offset

    def _consumption_llm_mode_for_phase(self, *, phase: str = "formal", preheat: bool = False) -> str:
        del phase, preheat  # Reserved for phase-specific policy switches.
        raw_mode = str(getattr(self.config, "consumption_llm_mode", "monthly") or "monthly").strip().lower()
        aliases = {
            "": "monthly",
            "true": "monthly",
            "llm": "monthly",
            "full": "monthly",
            "step": "monthly",
            "steps": "monthly",
            "false": "off",
            "none": "off",
            "rule": "off",
            "rules": "off",
            "rule_based": "off",
            "cached": "profile",
            "cached_profile": "profile",
            "profile_rule": "profile",
            "hybrid": "profile",
        }
        mode = aliases.get(raw_mode, raw_mode)
        if mode not in {"monthly", "profile", "off"}:
            logger.warning(f"[消费策略] Unknown consumption_llm_mode={raw_mode!r}; falling back to monthly")
            return "monthly"
        return mode

    @contextmanager
    def _time_block(self, label: str, month: Optional[int] = None, preheat: Optional[bool] = None):
        parts = []
        if month is not None:
            parts.append(f"month={month}")
        if preheat is not None:
            parts.append(f"preheat={preheat}")
        suffix = f" {' '.join(parts)}" if parts else ""
        logger.info(f"[计时开始] {label}{suffix}")
        self._write_stage_event("start", label, month=month, preheat=preheat)
        start = time.perf_counter()
        status = "ok"
        error = None
        try:
            yield
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed = time.perf_counter() - start
            self._write_stage_event(
                "end",
                label,
                month=month,
                preheat=preheat,
                elapsed_seconds=elapsed,
                status=status,
                error=error,
            )
            logger.info(f"[计时] {label} elapsed={elapsed:.4f}s{suffix}")

    def _recording_enabled(self) -> bool:
        record_dir = getattr(self.config, "local_record_dir", None)
        if record_dir is None:
            return True
        if isinstance(record_dir, str) and record_dir.strip() == "":
            return False
        return True

    def _ensure_record_dir(self) -> Optional[str]:
        if not self._recording_enabled():
            return None
        if self._record_dir:
            return self._record_dir
        base_dir = getattr(self.config, "local_record_dir", None) or "output/monthly_records"
        run_dir = os.path.join(base_dir, f"run_{self._record_run_id}")
        os.makedirs(run_dir, exist_ok=True)
        self._record_dir = run_dir
        return self._record_dir

    def _write_stage_event(
        self,
        event: str,
        stage: str,
        month: Optional[int] = None,
        preheat: Optional[bool] = None,
        elapsed_seconds: Optional[float] = None,
        status: str = "ok",
        error: Optional[str] = None,
    ) -> None:
        try:
            record_dir = self._ensure_record_dir()
            if not record_dir:
                return
            econ_month = None
            if month is not None:
                try:
                    econ_month = self._econ_month(int(month), bool(preheat))
                except Exception:
                    econ_month = month
            self._stage_event_seq += 1
            payload = {
                "seq": self._stage_event_seq,
                "run_id": self._record_run_id,
                "timestamp": datetime.now().isoformat(),
                "event": event,
                "stage": stage,
                "month": month,
                "econ_month": econ_month,
                "preheat": preheat,
                "status": status,
            }
            if elapsed_seconds is not None:
                payload["elapsed_seconds"] = float(elapsed_seconds)
            if error:
                payload["error"] = error
            path = os.path.join(record_dir, "stage_events.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            logger.debug(f"[阶段事件] 写入失败: {exc}")

    def _calc_distribution_stats(self, values: List[float]) -> Dict[str, float]:
        vals = [float(v) for v in values if v is not None]
        if not vals:
            return {
                "count": 0,
                "total": 0.0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "p10": 0.0,
                "p90": 0.0,
            }
        vals.sort()
        count = len(vals)
        total = float(sum(vals))
        mean = total / count
        mid = count // 2
        if count % 2 == 1:
            median = vals[mid]
        else:
            median = (vals[mid - 1] + vals[mid]) / 2.0
        p10 = vals[int((count - 1) * 0.10)]
        p90 = vals[int((count - 1) * 0.90)]
        return {
            "count": count,
            "total": total,
            "min": vals[0],
            "max": vals[-1],
            "mean": mean,
            "median": median,
            "p10": p10,
            "p90": p90,
        }

    def _calc_gini(self, values: List[float]) -> float:
        vals = [float(v) for v in values if v is not None]
        if not vals:
            return 0.0
        vals.sort()
        min_val = vals[0]
        if min_val < 0:
            shift = -min_val
            vals = [v + shift for v in vals]
        total = float(sum(vals))
        if total <= 0:
            return 0.0
        n = len(vals)
        cum = 0.0
        for i, v in enumerate(vals, start=1):
            cum += float(i) * float(v)
        gini = (2.0 * cum) / (n * total) - (n + 1) / n
        if gini < 0:
            return 0.0
        if gini > 1:
            return 1.0
        return float(gini)
    
    def _set_fixed_consumption_basket(
        self,
        consumption_stats: Dict[str, Any],
        snapshot_cache: Dict[str, Optional[Dict[str, Any]]]
    ) -> None:
        """
        在预热最后一个月结束时，将当月消费结构设置为固定消费篮子。
        之后的价格指数计算将使用这个固定篮子的权重。
        
        Args:
            consumption_stats: 消费统计 {by_sku: {sku_id: {qty, value}}}
            snapshot_cache: 产品快照缓存
        """
        by_sku = consumption_stats.get("by_sku", {})
        if not by_sku:
            logger.warning("[消费篮子] 预热最后一月无消费数据，无法设置固定篮子")
            return
        
        basket: Dict[str, Dict[str, float]] = {}
        total_value = 0.0
        
        for sku_id, stats in by_sku.items():
            qty = float(stats.get("qty", 0) or 0)
            value = float(stats.get("value", 0) or 0)
            if qty <= 0 or value <= 0:
                continue
            
            # 获取基准价格
            snapshot = snapshot_cache.get(sku_id) or self._get_product_snapshot_cached(sku_id, snapshot_cache)
            if snapshot:
                base_price = float(snapshot.get("base_retail_price") or snapshot.get("retail_price") or 0)
            else:
                base_price = value / qty if qty > 0 else 0
            
            if base_price > 0:
                basket[sku_id] = {
                    "base_price": base_price,
                    "qty": qty,  # 固定数量权重
                    "base_value": base_price * qty,  # 基期价值
                }
                total_value += base_price * qty
        
        # 计算每个 SKU 的消费权重
        for sku_id in basket:
            basket[sku_id]["weight"] = basket[sku_id]["base_value"] / total_value if total_value > 0 else 0
        
        self._fixed_consumption_basket = basket
        logger.info(f"[消费篮子] 固定消费篮子已设置: {len(basket)} 个SKU, 基期总值=${total_value:,.2f}")

    def _calc_consumption_category_distribution(
        self,
        consumption_stats: Optional[Dict[str, Any]],
        service_consumption_stats: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        计算家庭消费的类别分布结构
        
        Returns:
            {
                "goods": {
                    "total": float,
                    "by_industry": {industry_code: value}  # 按制造商行业
                },
                "services": {
                    "total": float,
                    "by_category": {category: value}  # housing, healthcare, etc.
                },
                "total": float,
                "goods_share": float,  # 商品占比
                "services_share": float,  # 服务占比
            }
        """
        result = {
            "goods": {"total": 0.0, "by_industry": {}},
            "services": {"total": 0.0, "by_category": {}},
            "total": 0.0,
            "goods_share": 0.0,
            "services_share": 0.0,
        }
        
        # 商品消费（从 consumption_stats）
        goods_total = 0.0
        if consumption_stats:
            goods_total = float(consumption_stats.get("total_value", 0) or 0)
            result["goods"]["total"] = goods_total
            
            # 尝试按行业分类（从 by_firm 数据）
            by_firm = consumption_stats.get("by_firm", {})
            industry_totals: Dict[str, float] = {}
            for firm_id, stats in by_firm.items():
                value = float(stats.get("value", 0) or 0)
                # 从 firm_id 提取行业代码（格式: ret_XXX_hash 或 mfg_XXX_hash）
                parts = str(firm_id).split("_")
                if len(parts) >= 2:
                    industry_code = parts[1]
                    industry_totals[industry_code] = industry_totals.get(industry_code, 0) + value
            result["goods"]["by_industry"] = industry_totals
        
        # 服务消费（从 service_consumption_stats）
        services_total = 0.0
        if service_consumption_stats:
            services_total = float(service_consumption_stats.get("total_service_consumption", 0) or 0)
            result["services"]["total"] = services_total
            result["services"]["by_category"] = dict(service_consumption_stats.get("by_category", {}) or {})
        
        # 计算总额和占比
        total = goods_total + services_total
        result["total"] = total
        if total > 0:
            result["goods_share"] = goods_total / total
            result["services_share"] = services_total / total
        
        return result

    def _write_month_record(self, month: int, payload: Dict[str, Any], preheat: bool) -> None:
        record_dir = self._ensure_record_dir()
        if not record_dir:
            return
        prefix = "preheat" if preheat else "month"
        path = os.path.join(record_dir, f"{prefix}_{month:04d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    async def run_simulation(self):
        """Run simulation"""
        logger.info("Running simulation...")
        
        # 配置更大的线程池以支持更高的并发度
        # asyncio.to_thread() 默认使用的线程池太小，无法支持大量household并发
        loop = asyncio.get_running_loop()
        self._thread_pool = ThreadPoolExecutor(
            max_workers=self._thread_pool_size,
            thread_name_prefix="household_consumption"
        )
        loop.set_default_executor(self._thread_pool)
        logger.info(f"Configured thread pool with {self._thread_pool_size} workers for parallel consumption")
        
        try:
            if self.config.preheat_months > 0:
                logger.info(f"Running preheat for {self.config.preheat_months} months...")
                await self._run_preheat(self.config.preheat_months)
                if self.economic_center is not None:
                    self._call_actor(self.economic_center, "reset_transactions")
            for month in range(1, self.config.num_months + 1):
                await self._run_month(month)
        finally:
            # 清理线程池
            if self._thread_pool is not None:
                self._thread_pool.shutdown(wait=False)
                self._thread_pool = None
            
    async def _run_preheat(self, months: int):
        """
        两阶段预热：
        Phase 0: 需求发现 - 关闭活跃过滤，家庭搜索全部SKU，收集被需求的SKU，初始化企业资金
        Phase 1+: 正常预热 - 激活被需求SKU，开启过滤，只在活跃SKU中搜索
        """
        # ====== Phase 0: 需求发现 ======
        logger.info("Preheat Phase 0: Demand Discovery (no active filter)...")
        
        # 确保关闭活跃过滤
        if self.product_market is not None:
            self._call_actor(self.product_market, "set_active_filter_mode", False)
        
        # 收集家庭消费需求（搜索全部30K SKU）
        # 家庭有初始储蓄(CSV中的ER85692字段)，所以可以直接使用
        with self._time_block("Phase0-需求发现", month=0, preheat=True):
            consumption_results = await self._collect_consumption_plans(
                top_k=10,
                current_month=self._econ_month(0, preheat=True),
                phase="phase0",
                preheat=True,
            )
            demand_by_product, _, snapshot_cache, demand_stats = self._build_orders(consumption_results)
            self._last_demand_stats = dict(demand_stats or {})
        
        # 激活被需求的SKU
        demanded_sku_ids = list(demand_by_product.keys())
        if demanded_sku_ids and self.product_market is not None:
            activated_count = self._call_actor(self.product_market, "activate_skus", demanded_sku_ids)
            logger.info(f"Preheat Phase 0: Activated {activated_count} demanded SKUs")
        else:
            logger.warning("Preheat Phase 0: No demanded SKUs found! Check household savings.")

        with self._time_block("Phase0-初始化商品库存", month=0, preheat=True):
            self._apply_initial_inventory_from_demand(demand_by_product)
        
        # 根据需求初始化企业资金
        # 企业需要初始资金来支付第一个月的工资
        with self._time_block("Phase0-初始化企业资金", month=0, preheat=True):
            self._initialize_firm_capital_from_demand(demand_stats)

        # 保存Phase 0的数据
        econ_month_0 = self._econ_month(0, preheat=True)
        self._record_month_summary(
            month=0,
            econ_month=econ_month_0,
            preheat=True,
            demand_stats=demand_stats,
            production_stats=None,
            consumption_stats=None,
            wage_stats=None,
            procurement_stats=None,
        )

        # 开启活跃过滤
        if self.product_market is not None:
            self._call_actor(self.product_market, "set_active_filter_mode", True)
        
        logger.info(f"Preheat Phase 0 complete. Active SKUs: {len(demanded_sku_ids)}")
        
        # ====== Phase 1+: 正常预热月 ======
        for idx in range(1, int(months) + 1):
            logger.info(f"Preheat month {idx}...")
            await self._run_warmup_month(idx)

    def _apply_initial_inventory_from_demand(self, demand_by_product: Dict[str, float]) -> None:
        """Calibrate ProductMarket initial stocks from Phase 0 SKU demand."""
        if self.product_market is None:
            return
        if not demand_by_product:
            self._last_initial_inventory_calibration = {
                "active_sku_count": 0,
                "reason": "no_phase0_demand",
            }
            return

        inventory_cover_months = max(
            0.0,
            float(getattr(self.config, "firm_initial_inventory_cover_months", 1.0) or 0.0),
        )
        inactive_stock = max(
            0.0,
            float(getattr(self.config, "firm_initial_inactive_sku_stock", 0.0) or 0.0),
        )
        min_active_stock = max(
            0.0,
            float(getattr(self.config, "firm_initial_min_active_sku_stock", 1.0) or 0.0),
        )
        stock_targets = {
            str(product_id): max(float(qty or 0.0) * inventory_cover_months, min_active_stock)
            for product_id, qty in demand_by_product.items()
            if product_id
        }
        stats = self._call_actor(
            self.product_market,
            "apply_initial_stock_targets",
            stock_targets,
            inactive_stock=inactive_stock,
            min_active_stock=min_active_stock,
        ) or {}
        stats.update(
            {
                "inventory_cover_months": inventory_cover_months,
                "configured_inactive_stock": inactive_stock,
                "configured_min_active_stock": min_active_stock,
                "target_sku_count": len(stock_targets),
            }
        )
        self._last_initial_inventory_calibration = stats
        logger.info(
            "[库存初始化] active_skus=%s total_stock %.0f -> %.0f",
            stats.get("active_sku_count", 0),
            float(stats.get("total_stock_before", 0.0) or 0.0),
            float(stats.get("total_stock_after", 0.0) or 0.0),
        )
    
    def _initialize_firm_capital_from_demand(self, demand_stats: Dict[str, Any]):
        """
        根据Phase 0发现的需求，为企业分配初始资金
        
        逻辑：企业初始cash = 预期月收入 * 系数（用于支付首月工资）
        系数默认为 1.5（足够支付 1.5 个月的运营成本）
        """
        capital_multiplier = float(getattr(self.config, "firm_initial_capital_multiplier", 1.5) or 1.5)
        inventory_cover_months = float(getattr(self.config, "firm_initial_inventory_cover_months", 1.0) or 1.0)
        capital_output_ratio = float(getattr(self.config, "firm_capital_output_ratio", 3.0) or 3.0)
        inventory_value_share = float(getattr(self.config, "firm_inventory_value_share", 0.5) or 0.5)
        calibrator = FirmInitializationCalibrator()
        
        # 从demand_stats中获取各企业的预期收入
        demand_by_mfg = demand_stats.get("by_mfg_firm", {})
        demand_by_retail = demand_stats.get("by_retail_firm", {})
        
        # 获取最低资金配置
        min_cash = float(getattr(self.config, "firm_min_initial_cash", 10000.0) or 10000.0)
        total_initialized = 0
        asset_allocations: Dict[str, Dict[str, float]] = {}
        calibration_summary: Dict[str, Dict[str, Any]] = {}
        
        # 为制造商分配初始资金
        for firm_id, stats in demand_by_mfg.items():
            firm = self._firm_by_id.get(firm_id)
            if firm is not None:
                expected_revenue = float(stats.get("value", 0.0) or 0.0)
                expected_units = float(stats.get("qty", 0.0) or 0.0)
                expected_cost = expected_revenue * (1.0 - float(getattr(firm, "compensation_ratio", 0.2) or 0.2))
                result = calibrator.calibrate(
                    FirmInitCalibrationInput(
                        firm_id=firm_id,
                        industry_code=str(getattr(firm, "industry", "") or ""),
                        industry_type=str(getattr(firm, "industry_type", "") or ""),
                        expected_monthly_revenue=expected_revenue,
                        expected_monthly_cost=expected_cost,
                        expected_monthly_sales_units=expected_units,
                        inventory_cover_months=inventory_cover_months,
                        cash_multiplier=capital_multiplier,
                        min_cash=min_cash,
                        capital_output_ratio=capital_output_ratio,
                        inventory_value_share=inventory_value_share,
                    )
                )
                firm.cash = result.initial_cash
                firm.capital_stock = result.initial_capital_stock
                result_data = result.to_dict()
                setattr(firm, "initialization_calibration", result_data)
                asset_allocations[firm_id] = {
                    "cash": result.initial_cash,
                    "capital_stock": result.initial_capital_stock,
                }
                calibration_summary[firm_id] = result_data
                logger.info(
                    f"[资产初始化] 制造商 {firm_id}: revenue={expected_revenue:.2f}, "
                    f"cash={result.initial_cash:.2f}, capital={result.initial_capital_stock:.2f}"
                )
                total_initialized += 1
            elif firm is None:
                logger.warning(f"[资金初始化] 制造商 {firm_id} 在 _firm_by_id 中未找到")

        # 为零售商分配初始资金
        for firm_id, stats in demand_by_retail.items():
            firm = self._firm_by_id.get(firm_id)
            if firm is not None:
                expected_revenue = float(stats.get("value", 0.0) or 0.0)
                expected_units = float(stats.get("qty", 0.0) or 0.0)
                expected_cost = expected_revenue * (
                    1.0 - float(getattr(firm, "compensation_ratio", 0.2) or 0.2)
                )
                result = calibrator.calibrate(
                    FirmInitCalibrationInput(
                        firm_id=firm_id,
                        industry_code=str(getattr(firm, "industry", "") or ""),
                        industry_type=str(getattr(firm, "industry_type", "") or ""),
                        expected_monthly_revenue=expected_revenue,
                        expected_monthly_cost=expected_cost,
                        expected_monthly_sales_units=expected_units,
                        inventory_cover_months=inventory_cover_months,
                        cash_multiplier=capital_multiplier,
                        min_cash=min_cash,
                        capital_output_ratio=capital_output_ratio,
                        inventory_value_share=inventory_value_share,
                    )
                )
                firm.cash = result.initial_cash
                firm.capital_stock = result.initial_capital_stock
                result_data = result.to_dict()
                setattr(firm, "initialization_calibration", result_data)
                asset_allocations[firm_id] = {
                    "cash": result.initial_cash,
                    "capital_stock": result.initial_capital_stock,
                }
                calibration_summary[firm_id] = result_data
                total_initialized += 1
                logger.debug(
                    f"Initialized {firm_id} assets: cash={result.initial_cash:.2f}, "
                    f"capital={result.initial_capital_stock:.2f}"
                )

        # 为完全没有需求的企业设置最低资金（用于基本运营）
        for firm in (self.firms or []):
            if firm.cash <= 0:
                result = calibrator.calibrate(
                    FirmInitCalibrationInput(
                        firm_id=firm.firm_id,
                        industry_code=str(getattr(firm, "industry", "") or ""),
                        industry_type=str(getattr(firm, "industry_type", "") or ""),
                        expected_monthly_revenue=0.0,
                        expected_monthly_cost=0.0,
                        expected_monthly_sales_units=0.0,
                        inventory_cover_months=inventory_cover_months,
                        cash_multiplier=capital_multiplier,
                        min_cash=min_cash,
                        capital_output_ratio=capital_output_ratio,
                        inventory_value_share=inventory_value_share,
                    )
                )
                firm.cash = result.initial_cash
                firm.capital_stock = result.initial_capital_stock
                result_data = result.to_dict()
                setattr(firm, "initialization_calibration", result_data)
                asset_allocations[firm.firm_id] = {
                    "cash": result.initial_cash,
                    "capital_stock": result.initial_capital_stock,
                }
                calibration_summary[firm.firm_id] = result_data
                total_initialized += 1
        if self.economic_center is not None and asset_allocations:
            self._call_actor(self.economic_center, "register_firm_assets", asset_allocations)
        self._last_firm_initialization_calibration = calibration_summary
        
        logger.info(
            f"Phase 0: Initialized assets for {total_initialized} firms "
            f"(min_cash={min_cash:.2f}, inventory_cover={inventory_cover_months:.2f})"
        )

    def _initialize_firm_assets_without_preheat(self) -> None:
        """
        Give firms a calibrated baseline balance when the run has no Phase 0
        demand-discovery pass. Preheat later overwrites this with demand-based
        allocations.
        """
        if int(getattr(self.config, "preheat_months", 0) or 0) > 0:
            return
        if self.economic_center is None or not self.firms:
            return

        min_cash = float(getattr(self.config, "firm_min_initial_cash", 10000.0) or 10000.0)
        capital_output_ratio = float(getattr(self.config, "firm_capital_output_ratio", 3.0) or 3.0)
        calibrator = FirmInitializationCalibrator()
        asset_allocations: Dict[str, Dict[str, float]] = {}
        calibration_summary: Dict[str, Dict[str, Any]] = {}

        for firm in self.firms:
            result = calibrator.calibrate(
                FirmInitCalibrationInput(
                    firm_id=firm.firm_id,
                    industry_code=str(getattr(firm, "industry", "") or ""),
                    industry_type=str(getattr(firm, "industry_type", "") or ""),
                    expected_monthly_revenue=min_cash,
                    expected_monthly_cost=min_cash * 0.5,
                    expected_monthly_sales_units=0.0,
                    cash_multiplier=1.0,
                    min_cash=min_cash,
                    capital_output_ratio=capital_output_ratio,
                )
            )
            firm.cash = result.initial_cash
            firm.capital_stock = result.initial_capital_stock
            setattr(firm, "initialization_calibration", result.to_dict())
            asset_allocations[firm.firm_id] = {
                "cash": result.initial_cash,
                "capital_stock": result.initial_capital_stock,
            }
            calibration_summary[firm.firm_id] = result.to_dict()

        self._call_actor(self.economic_center, "register_firm_assets", asset_allocations)
        self._last_firm_initialization_calibration = calibration_summary
        logger.info(
            f"[资产初始化] 无预热运行: initialized {len(asset_allocations)} firms "
            f"with baseline cash={min_cash:.2f}"
        )

    async def _run_warmup_month(self, month: int):
        econ_month = self._econ_month(month, preheat=True)

        # 月初重置供需追踪
        self._call_actor(self.product_market, "reset_supply_demand_tracking")

        # ========== 劳动力市场（与正式月顺序一致）==========
        with self._time_block("裁员处理", month=month, preheat=True):
            layoff_stats = await self._process_layoffs(econ_month)
        layoff_firm_ids = set((layoff_stats.get("by_firm") or {}).keys())
        with self._time_block("发布岗位", month=month, preheat=True):
            await self._post_jobs(econ_month, skip_firm_ids=layoff_firm_ids)
        with self._time_block("招聘匹配", month=month, preheat=True):
            await self._match_jobs(econ_month, use_llm=False)
        with self._time_block("发放工资", month=month, preheat=True):
            wage_stats = self._pay_wages(econ_month, record_transactions=True)

        # ========== 商品市场 ==========
        with self._time_block("消费决策", month=month, preheat=True):
            consumption_results = await self._collect_consumption_plans(
                top_k=10,
                current_month=econ_month,
                phase="preheat",
                preheat=True,
            )
        with self._time_block("构建订单", month=month, preheat=True):
            demand_by_product, orders_by_household, snapshot_cache, demand_stats = self._build_orders(consumption_results)
            self._last_demand_stats = dict(demand_stats or {})

        # 计算家庭消费总预算，供政府采购计划和实际采购共用
        household_total_budget = self._get_household_consumption_budget(consumption_results)
        government_procurement_plan_stats = self._plan_government_procurement_demand(
            econ_month,
            household_consumption_budget=household_total_budget,
        )
        production_demand_by_product = self._merge_product_demands(
            demand_by_product,
            (government_procurement_plan_stats or {}).get("demand_by_product", {}),
        )

        # 记录需求到ProductMarket
        self._record_demand_to_market(production_demand_by_product, snapshot_cache)

        with self._time_block("生产补货", month=month, preheat=True):
            production_demand = self._build_production_demand_signal(production_demand_by_product)
            production_stats = self._ensure_production(
                production_demand,
                snapshot_cache,
                econ_month,
                record_transactions=True,
                sales_history_by_product=self._last_sales_by_product,
                unmet_demand_by_product=self._last_unmet_demand_by_product,
            )

        # 记录供给并根据供需调整价格
        self._record_supply_and_adjust_prices(production_stats, econ_month)

        # ========== 政府采购 ==========
        # 政府采购是生产前已声明的最终需求，应在零售商为家庭渠道进货前
        # 清算，否则家庭零售渠道会先耗尽同一批制造商库存，导致政府需求
        # 信号进了生产计划却无法成交。
        with self._time_block("政府采购", month=month, preheat=True):
            government_procurement_stats = await self._execute_government_procurement(
                econ_month,
                household_consumption_budget=household_total_budget,
                planned_demand_by_product=(government_procurement_plan_stats or {}).get("demand_by_product", {}),
            )
            snapshot_cache.clear()

        with self._time_block("零售商进货", month=month, preheat=True):
            procurement_stats = self._retailer_procurement(
                demand_by_product,
                snapshot_cache,
                econ_month,
                record_transactions=True,
                demand_by_retailer_product=(demand_stats or {}).get("by_retailer_product", {}),
            )
        with self._time_block("执行购买", month=month, preheat=True):
            consumption_stats = self._execute_orders(orders_by_household, snapshot_cache, econ_month, record_transactions=True)
        with self._time_block("服务消费", month=month, preheat=True):
            service_consumption_stats = self._execute_service_consumption(consumption_results, econ_month)
        
        # 更新家庭消费历史（用于下月消费惯性计算）
        self._update_household_consumption_history(
            consumption_stats, service_consumption_stats, consumption_results
        )
        
        # 更新销售记录
        self._update_last_sales(econ_month, consumption_stats)
        self._update_demand_memory(production_demand_by_product, econ_month)
        self._update_layoff_support_memory(production_stats, service_consumption_stats)
        
        # ========== 月末结算 ==========
        with self._time_block("企业所得税", month=month, preheat=True):
            self._settle_corporate_tax(econ_month)
        with self._time_block("银行利息", month=month, preheat=True):
            await self._pay_bank_interest(econ_month)
        with self._time_block("企业信贷结算", month=month, preheat=True):
            firm_credit_stats = self._settle_firm_credit(econ_month)
            self._apply_credit_default_labor_closure(econ_month, firm_credit_stats)
        with self._time_block("税收再分配", month=month, preheat=True):
            await self._redistribute_taxes(econ_month)
        with self._time_block("企业分红", month=month, preheat=True):
            self._distribute_dividends(econ_month)
        
        with self._time_block("月度汇总", month=month, preheat=True):
            self._record_month_summary(
                month=month,
                econ_month=econ_month,
                preheat=True,
                demand_stats=demand_stats,
                production_stats=production_stats,
                consumption_stats=consumption_stats,
                wage_stats=wage_stats,
                procurement_stats=procurement_stats,
                service_consumption_stats=service_consumption_stats,
                government_procurement_stats=government_procurement_stats,
                government_procurement_plan_stats=government_procurement_plan_stats,
                firm_credit_stats=firm_credit_stats,
            )
        
        # ========== 设置固定消费篮子（预热最后一月）==========
        preheat_months = int(getattr(self.config, "preheat_months", 0) or 0)
        if month == preheat_months and consumption_stats:
            self._set_fixed_consumption_basket(consumption_stats, snapshot_cache)
        
        # ========== 保存 Checkpoint (预热阶段) ==========
        # 预热阶段最后一个月保存 checkpoint
        if month == preheat_months and self._save_checkpoint_interval > 0:
            with self._time_block("保存检查点", month=month, preheat=True):
                try:
                    checkpoint_path = self._checkpoint_manager.save_checkpoint(
                        simulator=self,
                        month=month,
                        preheat=True
                    )
                    logger.info(f"💾 预热 Checkpoint 已保存: {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"保存预热 Checkpoint 失败: {e}")

    def _update_endogenous_wage(self, econ_month: int) -> None:
        """内生工资调整：按上月劳动市场松紧度调整全局工资水平（AGENTECO_WAGE_SCALE）。

        紧（失业<目标）→ 工资上调；松（失业>目标）→ 工资下调。自均衡失业率到目标附近，
        并经成本定价产生 Phillips 关系（低失业→工资涨→价格涨→通胀）。
        """
        if not bool(getattr(self.config, "endogenous_wages", False)):
            return
        import os as _os
        if not hasattr(self, "_wage_level"):
            try:
                self._wage_level = float(_os.getenv("AGENTECO_WAGE_SCALE", "1.0") or 1.0)
            except (TypeError, ValueError):
                self._wage_level = 1.0
        u = None
        try:
            summ = self._call_actor(self.labor_market, "summary")
            if isinstance(summ, dict):
                u = float(summ.get("unemployment_rate", 0.0) or 0.0)
        except Exception:
            u = None
        if u is None:
            return
        kappa = float(getattr(self.config, "wage_adjustment_speed", 0.3) or 0.3)
        target = float(getattr(self.config, "wage_target_unemployment", 0.08) or 0.08)
        self._wage_level *= (1.0 + kappa * (target - u))
        self._wage_level = max(0.02, min(3.0, self._wage_level))
        _os.environ["AGENTECO_WAGE_SCALE"] = str(self._wage_level)
        self._last_wage_level = self._wage_level

    async def _run_month(self, month: int):
        """Run a single month"""
        econ_month = self._econ_month(month, preheat=False)
        # 内生工资：按上月劳动市场松紧调整工资水平 → 自均衡失业 + Phillips（松紧→工资→价格→通胀）
        self._update_endogenous_wage(econ_month)
        title = f" 月份 {month} (经济周期 M{econ_month}) "
        padding = (60 - len(title)) // 2
        logger.info(f"\n{'━' * 60}")
        logger.info(f"{'━' * padding}{title}{'━' * (60 - padding - len(title))}")
        logger.info(f"{'━' * 60}")
        
        # 月初重置供需追踪
        self._call_actor(self.product_market, "reset_supply_demand_tracking")
        
        # ========== 劳动力市场 ==========
        logger.info(f"\n┌{'─' * 38}┐")
        logger.info(f"│ 📋 劳动力市场                        │")
        logger.info(f"└{'─' * 38}┘")

        # 先处理裁员（根据上月收入调整工资帽）
        with self._time_block("裁员处理", month=month, preheat=False):
            layoff_stats = await self._process_layoffs(econ_month)
        if layoff_stats.get("total_layoffs", 0) > 0:
            logger.info(
                f"[裁员] 本月裁员{layoff_stats['total_layoffs']}人, "
                f"节省工资${layoff_stats['total_saved']:.2f}"
            )

        # 再发布岗位（可能补缺或扩招）
        layoff_firm_ids = set((layoff_stats.get("by_firm") or {}).keys())
        with self._time_block("发布岗位", month=month, preheat=False):
            await self._post_jobs(econ_month, skip_firm_ids=layoff_firm_ids)  # 使用 econ_month 以便正确查询上月数据

        # 匹配
        with self._time_block("招聘匹配", month=month, preheat=False):
            await self._match_jobs(econ_month, use_llm=False)  # 使用 econ_month 保持数据一致性

        # 发工资
        with self._time_block("发放工资", month=month, preheat=False):
            wage_stats = self._pay_wages(econ_month, record_transactions=True)
        self._log_wage_stats(wage_stats)

        # ========== 商品市场 ==========
        logger.info(f"\n┌{'─' * 38}┐")
        logger.info(f"│ 🛒 商品市场                          │")
        logger.info(f"└{'─' * 38}┘")
        with self._time_block("消费决策", month=month, preheat=False):
            consumption_results = await self._collect_consumption_plans(
                top_k=10,
                current_month=econ_month,
                phase="formal",
                preheat=False,
            )
        self._log_consumption_plans(consumption_results)
        
        with self._time_block("构建订单", month=month, preheat=False):
            demand_by_product, orders_by_household, snapshot_cache, demand_stats = self._build_orders(consumption_results)
        self._last_demand_stats = dict(demand_stats or {})
        self._log_demand_stats(demand_stats)

        # 生产前形成政府采购计划需求；实际采购仍在生产和家庭购买后清算
        household_total_budget = self._get_household_consumption_budget(consumption_results)
        government_procurement_plan_stats = self._plan_government_procurement_demand(
            econ_month,
            household_consumption_budget=household_total_budget,
        )
        self._log_government_procurement_plan_stats(government_procurement_plan_stats)
        # 存本月政府采购需求（按行业），供下月 _post_jobs 计入企业劳动预算：
        # 政府采购计划在招聘之后形成，企业默认不为其招人→产不出→采购无货可买（G 截断）。
        # 用上月政府需求作为前瞻信号让企业提前为政府需求备产（稳态自洽）。
        self._last_gov_demand_by_industry = {
            str(ind): float((v or {}).get("planned_value", 0.0) or 0.0)
            for ind, v in ((government_procurement_plan_stats or {}).get("by_industry") or {}).items()
        }
        production_demand_by_product = self._merge_product_demands(
            demand_by_product,
            (government_procurement_plan_stats or {}).get("demand_by_product", {}),
        )
        
        # 记录需求到ProductMarket
        self._record_demand_to_market(production_demand_by_product, snapshot_cache)
        
        with self._time_block("生产补货", month=month, preheat=False):
            production_demand = self._build_production_demand_signal(production_demand_by_product)
            production_stats = self._ensure_production(
                production_demand,
                snapshot_cache,
                econ_month,
                record_transactions=True,
                sales_history_by_product=self._last_sales_by_product,
                unmet_demand_by_product=self._last_unmet_demand_by_product,
            )
        self._log_production_stats(production_stats)
        
        # 记录供给并根据供需调整价格
        self._record_supply_and_adjust_prices(production_stats, econ_month)

        # ========== 政府采购 ==========
        logger.info(f"\n┌{'─' * 38}┐")
        logger.info(f"│ 🏛️  政府采购                          │")
        logger.info(f"└{'─' * 38}┘")
        with self._time_block("政府采购", month=month, preheat=False):
            government_procurement_stats = await self._execute_government_procurement(
                econ_month,
                household_consumption_budget=household_total_budget,
                planned_demand_by_product=(government_procurement_plan_stats or {}).get("demand_by_product", {}),
            )
            snapshot_cache.clear()
        self._log_government_procurement_stats(government_procurement_stats)
        
        with self._time_block("零售商进货", month=month, preheat=False):
            procurement_stats = self._retailer_procurement(
                demand_by_product,
                snapshot_cache,
                econ_month,
                record_transactions=True,
                demand_by_retailer_product=(demand_stats or {}).get("by_retailer_product", {}),
            )
        self._log_procurement_stats(procurement_stats)
        
        with self._time_block("执行购买", month=month, preheat=False):
            consumption_stats = self._execute_orders(orders_by_household, snapshot_cache, econ_month, record_transactions=True)
        self._log_consumption_stats(consumption_stats)
        
        with self._time_block("服务消费", month=month, preheat=False):
            service_consumption_stats = self._execute_service_consumption(consumption_results, econ_month)
        self._log_service_consumption_stats(service_consumption_stats)
        
        # 更新家庭消费历史（用于下月消费惯性计算）
        self._update_household_consumption_history(
            consumption_stats, service_consumption_stats, consumption_results
        )
        
        # 更新销售记录（在政府采购之后，以便包含政府采购数据）
        self._update_last_sales(econ_month, consumption_stats)
        self._update_demand_memory(production_demand_by_product, econ_month)
        self._update_layoff_support_memory(production_stats, service_consumption_stats)
        
        # ========== 月末结算 ==========
        logger.info(f"\n┌{'─' * 38}┐")
        logger.info(f"│ 📊 月末结算                          │")
        logger.info(f"└{'─' * 38}┘")

        # 企业所得税（销售完成后结算，此时当月收入已记录）
        with self._time_block("企业所得税", month=month, preheat=False):
            corporate_tax_stats = self._settle_corporate_tax(econ_month)
        self._log_corporate_tax_stats(corporate_tax_stats)

        with self._time_block("银行利息", month=month, preheat=False):
            interest_stats = await self._pay_bank_interest(econ_month)
        with self._time_block("企业信贷结算", month=month, preheat=False):
            firm_credit_stats = self._settle_firm_credit(econ_month)
            self._apply_credit_default_labor_closure(econ_month, firm_credit_stats)
        
        with self._time_block("税收再分配", month=month, preheat=False):
            redistribution_stats = await self._redistribute_taxes(econ_month)
        self._log_redistribution_stats(redistribution_stats)

        with self._time_block("企业分红", month=month, preheat=False):
            dividend_stats = self._distribute_dividends(econ_month)

        if self._debug_enabled() and self.economic_center is not None:
            self._log_firm_financials(econ_month)
        with self._time_block("月度汇总", month=month, preheat=False):
            self._record_month_summary(
                month=month,
                econ_month=econ_month,
                preheat=False,
                demand_stats=demand_stats,
                production_stats=production_stats,
                consumption_stats=consumption_stats,
                wage_stats=wage_stats,
                procurement_stats=procurement_stats,
                service_consumption_stats=service_consumption_stats,
                government_procurement_stats=government_procurement_stats,
                government_procurement_plan_stats=government_procurement_plan_stats,
                firm_credit_stats=firm_credit_stats,
            )
        
        # 月末余额汇总
        self._log_month_end_summary(econ_month)
        
        # ========== 保存 Checkpoint ==========
        if self._save_checkpoint_interval > 0 and month % self._save_checkpoint_interval == 0:
            with self._time_block("保存检查点", month=month, preheat=False):
                try:
                    checkpoint_path = self._checkpoint_manager.save_checkpoint(
                        simulator=self,
                        month=month,
                        preheat=False
                    )
                    logger.info(f"💾 Checkpoint 已保存: {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"保存 Checkpoint 失败: {e}")

    def _compute_macro_indicators(self) -> Dict[str, Any]:
        """
        计算当前的宏观经济指标，用于影响家庭消费决策。

        Returns:
            Dict with keys:
                - inflation_rate: 月度通胀率（基于价格指数变化）
                - unemployment_rate: 失业率
                - interest_rate: 月度利率
                - tax_rate: 有效税率（VAT + 平均所得税）
                - price_index: 当前价格指数（基准=100）
        """
        # 1. 通胀率和价格指数（使用上次月末计算的值）
        inflation_rate = float(self._last_inflation_rate or 0.0)
        price_index = 100.0
        if self._last_price_index is not None:
            price_index = float(self._last_price_index)

        # 2. 失业率（使用LaborMarket中实际注册的劳动力数据）
        unemployment_rate = 0.0
        if self.labor_market is not None:
            try:
                labor_summary = self._call_actor(self.labor_market, "summary") or {}
                # 使用实际注册的劳动力数量，而不是假设的家庭数*2
                total_labor_force = int(labor_summary.get("total_labor_hours", 0) or 0)
                total_employed = int(labor_summary.get("total_matched_jobs", 0) or 0)
                
                # 数据校验：剔除负值和极端值
                total_labor_force = max(0, total_labor_force)
                total_employed = max(0, min(total_employed, total_labor_force))
                
                if total_labor_force > 0:
                    unemployment_rate = 1.0 - (total_employed / total_labor_force)
                    # 限制在合理范围 [0, 1]
                    unemployment_rate = max(0.0, min(1.0, unemployment_rate))
            except Exception as e:
                logger.debug(f"计算失业率失败: {e}")

        # 3. 利率（从配置获取，转换为月度）
        try:
            annual_interest_rate = float(getattr(self.config, "interest_rate", 0.005) or 0.005)
        except (TypeError, ValueError):
            annual_interest_rate = 0.005
        monthly_interest_rate = annual_interest_rate / 12.0

        # 4. 税率（VAT + 估算的平均所得税率）
        try:
            vat_rate = float(getattr(self.config, "vat_rate", 0.08) or 0.08)
        except (TypeError, ValueError):
            vat_rate = 0.08

        # 估算平均所得税率（取中间档）
        avg_income_tax_rate = 0.15  # 默认估算
        try:
            brackets = getattr(self.config, "income_tax_rate", None)
            if brackets and len(brackets) >= 3:
                mid_bracket = brackets[len(brackets) // 2]
                # TaxBracket 对象有 .rate 属性
                if hasattr(mid_bracket, "rate"):
                    avg_income_tax_rate = float(mid_bracket.rate)
                elif isinstance(mid_bracket, dict):
                    avg_income_tax_rate = float(mid_bracket.get("rate", 0.15))
        except (TypeError, ValueError, AttributeError) as e:
            logger.debug(f"获取所得税率失败，使用默认值: {e}")

        effective_tax_rate = vat_rate + avg_income_tax_rate

        macro_indicators = {
            "inflation_rate": round(inflation_rate, 4),
            "unemployment_rate": round(unemployment_rate, 4),
            "interest_rate": round(monthly_interest_rate, 6),
            "tax_rate": round(effective_tax_rate, 4),
            "price_index": round(price_index, 2),
        }

        logger.info(
            f"[宏观指标] 通胀率={macro_indicators['inflation_rate']:.2%}, "
            f"失业率={macro_indicators['unemployment_rate']:.2%}, "
            f"利率={macro_indicators['interest_rate']:.4%}/月, "
            f"税率={macro_indicators['tax_rate']:.2%}, "
            f"价格指数={macro_indicators['price_index']:.1f}"
        )

        return macro_indicators

    async def _prefetch_rule_consumption_candidates(
        self,
        top_k: int,
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Prefetch shared SKU candidates for household consumption.

        Each household still computes household-specific budgets, preferences,
        and product picks. The expensive vector lookup is category-level and can
        be shared across households for the same period in both rule and LLM
        consumption modes.
        """
        if not self.households or self.product_market is None:
            return None

        categories: List[str] = []
        seen = set()
        for household in self.households:
            for category in getattr(household, "consumption_categories", []) or []:
                category_name = str(category or "").strip()
                if not category_name or category_name in seen:
                    continue
                categories.append(category_name)
                seen.add(category_name)

        if not categories:
            return None

        category_plans = [
            CategoryPlan(
                category=category,
                budget_amount=0.0,
                need_descriptions=[
                    f"{category} basic recurring household needs",
                    f"{category} budget conscious replenishment",
                ],
            )
            for category in categories
        ]
        search_top_k = max(int(top_k or 0), int(os.getenv("RULE_CONSUMPTION_SHARED_TOP_K", "40")))
        logger.info(
            f"[消费候选预取] shared categories={len(category_plans)} top_k={search_top_k}"
        )
        return await self.households[0].consumption_step2_vector_match(
            category_plans=category_plans,
            top_k=search_top_k,
            product_market=self.product_market,
            merge_need_queries=True,
        )

    async def _collect_consumption_plans(
        self,
        top_k: int = 10,
        *,
        current_month: Optional[int] = None,
        phase: str = "formal",
        preheat: bool = False,
    ) -> List[Tuple[Household, Dict[str, Any]]]:
        results: List[Tuple[Household, Dict[str, Any]]] = []
        if not self.households:
            return results
        
        balance_by_household: Dict[str, float] = {}
        if self.economic_center is not None and self.households:
            # 批量查询所有家庭余额（并行执行）
            query_args = [(hh.household_id,) for hh in self.households]
            balances = self._call_actor_batch(self.economic_center, "query_balance", query_args)
            for hh, bal in zip(self.households, balances):
                balance_by_household[hh.household_id] = float(bal or 0.0)

        expected_income_by_household: Dict[str, float] = defaultdict(float)
        if self.labor_market is not None:
            matched = self._call_actor(self.labor_market, "get_matched_jobs") or []
            for rec in matched:
                household_id = rec.get("household_id")
                if not household_id:
                    continue
                wage_per_hour = float(rec.get("wage_per_hour") or 0.0)
                if wage_per_hour <= 0:
                    continue
                hours_per_period = rec.get("hours_per_period")
                if hours_per_period is None:
                    total_hours = 160.0
                else:
                    total_hours = float(hours_per_period or 0.0)
                expected_income_by_household[household_id] += wage_per_hour * total_hours

        # P4 失业收入耦合：失业家庭的消费收入锚 = 替代率 × PSID 基线月收入(ER85629)，
        # 而非全额 PSID 回退。替代率<1 → 就业带来净收入增量 → 总需求随就业上升，
        # 打破"工资降→就业升但需求不跟→通缩"的链条。替代率=1.0 时等价旧行为。
        replacement_rate = float(getattr(self.config, "unemployment_replacement_rate", 1.0) or 1.0)
        if replacement_rate < 1.0:
            for hh in self.households:
                if expected_income_by_household.get(hh.household_id, 0.0) <= 0.0:
                    try:
                        baseline = float(hh.csv_values.get("ER85629") or 0.0)
                    except (ValueError, TypeError, AttributeError):
                        baseline = 0.0
                    expected_income_by_household[hh.household_id] = replacement_rate * baseline

        # 总需求 AR(1) 随机冲击（ABM 标准：Lengnick 2013 / Dosi 等）。制造景气周期波动，
        # 使 Beveridge/Okun/Phillips 等规律从周期共动中涌现，而非仅来自一次性收敛过渡。
        # 冲击作用于家庭预期收入（情绪/需求冲击）→ 消费 → 产出 → 就业。
        shock_std = float(getattr(self.config, "demand_shock_std", 0.0) or 0.0)
        if shock_std > 0.0:
            import math as _math
            import random as _random
            if not hasattr(self, "_shock_rng") or self._shock_rng is None:
                self._shock_rng = _random.Random(int(getattr(self.config, "random_seed", 12345) or 12345))
                self._demand_shock = 0.0
            rho = float(getattr(self.config, "demand_shock_rho", 0.7) or 0.7)
            self._demand_shock = (
                rho * float(getattr(self, "_demand_shock", 0.0) or 0.0)
                + _math.sqrt(max(0.0, 1.0 - rho * rho)) * shock_std * self._shock_rng.gauss(0.0, 1.0)
            )
            shock_mult = _math.exp(self._demand_shock)
            self._last_demand_shock_mult = shock_mult
            for _hid in list(expected_income_by_household.keys()):
                expected_income_by_household[_hid] = expected_income_by_household[_hid] * shock_mult

        self._last_balance_by_household = dict(balance_by_household)
        self._last_expected_income_by_household = dict(expected_income_by_household)

        # 计算宏观经济指标
        macro_indicators = self._compute_macro_indicators()

        consumption_use_llm = bool(getattr(self.config, "consumption_use_llm", True))
        consumption_llm_mode = self._consumption_llm_mode_for_phase(phase=phase, preheat=preheat)
        raw_refresh_months = getattr(self.config, "consumption_profile_refresh_months", 12)
        profile_refresh_months = 12 if raw_refresh_months is None else int(raw_refresh_months)
        # 初始化消费进度追踪器
        # log_interval: 每完成 10% 的家庭打印一次进度
        log_interval = max(1, len(self.households) // 10)
        consumption_progress.reset(
            total=len(self.households),
            log_interval=log_interval,
            logger_instance=logger,
            mode=consumption_llm_mode,
        )
        logger.info(f"[消费进度] 开始收集 {len(self.households)} 个家庭的消费计划...")
        logger.info(
            f"[消费策略] use_llm={consumption_use_llm}, mode={consumption_llm_mode}, "
            f"profile_refresh_months={profile_refresh_months}, phase={phase}, month={current_month}"
        )
        shared_consumption_candidates = None
        try:
            shared_consumption_candidates = await self._prefetch_rule_consumption_candidates(top_k=top_k)
        except Exception as e:
            logger.warning(f"[消费候选预取] 失败，回退到逐户检索: {e}")
            shared_consumption_candidates = None

        # 分批处理家庭消费，避免ProductMarket actor过载
        # 默认 50（原 100），每个家庭约 12 个向量搜索请求，50 家庭 ≈ 600 个请求
        # 配合 ProductMarket._qdrant_semaphore 控制实际 Qdrant 并发
        batch_size = int(os.getenv("CONSUMPTION_BATCH_SIZE", "50"))
        all_outputs = []

        for batch_start in range(0, len(self.households), batch_size):
            batch_end = min(batch_start + batch_size, len(self.households))
            batch_households = self.households[batch_start:batch_end]
            logger.info(f"[消费进度] 处理批次 {batch_start//batch_size + 1}/{(len(self.households) + batch_size - 1)//batch_size}, 家庭 {batch_start+1}-{batch_end}")

            tasks = []
            for hh in batch_households:
                available_balance = balance_by_household.get(hh.household_id)
                expected_income = expected_income_by_household.get(hh.household_id, 0.0)
                # Hard liquidity cap: the simulator already records wages before
                # this step, so current ledger balance is the spendable budget.
                available_budget = float(available_balance) if available_balance is not None else None

                task = hh.consume_v2(
                    top_k=top_k,
                    product_market=self.product_market,
                    available_balance=available_balance,
                    expected_income=expected_income,
                    available_budget=available_budget,
                    macro_indicators=macro_indicators,
                    use_llm=consumption_use_llm,
                    llm_mode=consumption_llm_mode,
                    current_month=current_month,
                    profile_refresh_months=profile_refresh_months,
                    candidate_products_by_category=shared_consumption_candidates,
                )
                tasks.append(task)

            batch_outputs = await asyncio.gather(*tasks, return_exceptions=True)
            all_outputs.extend(zip(batch_households, batch_outputs))

        outputs = [out for _, out in all_outputs]

        # 统计异常数量并使用降级方案
        fallback_count = 0

        # 禁用进度追踪并打印最终状态
        status = consumption_progress.get_status()
        consumption_progress.disable()
        logger.info(
            f"[消费进度] 完成! "
            f"Step0:{status['step0_done']}/{status['total']} "
            f"Step1:{status['step1_done']}/{status['total']} "
            f"Step2:{status['step2_done']}/{status['total']} "
            f"Step3:{status['step3_done']}/{status['total']}"
        )

        for hh, out in all_outputs:
            if isinstance(out, Exception):
                logger.error(f"Household {hh.household_id} consumption failed: {out}")
                # 异常时使用降级方案，使用合理的月度预算上限
                available_balance = balance_by_household.get(hh.household_id, 0.0)
                expected_income = expected_income_by_household.get(hh.household_id, 0.0)
                
                # 计算合理的月度预算：基于收入或历史消费，而非全部储蓄
                # 优先使用历史月度支出，其次是月收入，最后是默认上限
                monthly_expenditure = 0.0
                try:
                    monthly_expenditure = float(hh.csv_values.get("ER85768") or 0.0)
                except (ValueError, TypeError, AttributeError):
                    pass
                
                monthly_income = expected_income
                if monthly_income <= 0:
                    try:
                        monthly_income = float(hh.csv_values.get("ER85629") or 0.0)
                    except (ValueError, TypeError, AttributeError):
                        pass
                
                # 使用较合理的预算：历史支出 > 月收入 > 默认上限($10,000)
                # 并且不超过可用余额
                default_max_monthly_budget = 10000.0
                reasonable_budget = monthly_expenditure if monthly_expenditure > 0 else (
                    monthly_income if monthly_income > 0 else default_max_monthly_budget
                )
                fallback_budget = min(reasonable_budget, available_balance)
                
                out = hh.generate_fallback_consumption_plan(
                    available_budget=float(fallback_budget),
                    product_market=self.product_market,
                )
                fallback_count += 1
                results.append((hh, out))
            elif isinstance(out, dict):
                results.append((hh, out))

        if fallback_count > 0:
            logger.warning(f"[消费进度] 降级:{fallback_count}")
            
        if self._debug_enabled():
            self._log_consumption_plans(results)
            self._log_consumption_budget_status(results, balance_by_household, expected_income_by_household)
        return results

    def _get_household_consumption_budget(
        self, 
        consumption_results: List[Tuple[Household, Dict[str, Any]]]
    ) -> float:
        """
        计算所有家庭的消费总预算
        
        用于政府需求注入计算：政府支出 = 家庭消费预算 × 注入比例
        
        Args:
            consumption_results: 消费计划结果列表 [(Household, plan_dict), ...]
            
        Returns:
            所有家庭的消费总预算
        """
        total_budget = 0.0
        for hh, out in consumption_results:
            if not isinstance(out, dict):
                continue
            step0 = out.get("step0", {})
            if isinstance(step0, dict):
                budget = float(step0.get("total_budget") or 0.0)
                total_budget += budget
        return total_budget
    
    def _log_consumption_plans(self, results: List[Tuple[Household, Dict[str, Any]]]) -> None:
        limit = int(getattr(self.config, "debug_max_households", 0) or 0)
        items, skipped = self._limit_list(results, limit)
        for hh, out in items:
            step0 = out.get("step0", {}) if isinstance(out, dict) else {}
            total_budget = float(step0.get("total_budget") or 0.0)
            budgets = step0.get("budgets", {}) if isinstance(step0, dict) else {}
            retail_budget = float(budgets.get("Retail merchandise") or 0.0)
            step3 = out.get("step3", {}) if isinstance(out, dict) else {}
            purchases = list(step3.get("purchases") or [])
            planned_spend = sum(float(p.get("allocated_budget") or 0.0) for p in purchases)
            logger.info(
                f"[消费计划] {hh.household_id} total_budget={total_budget:.2f} retail_budget={retail_budget:.2f} "
                f"planned_items={len(purchases)} planned_spend={planned_spend:.2f} budgets={budgets}"
            )
        if skipped > 0:
            logger.info(f"[消费计划] 其余家庭未打印数量={skipped}")

    def _log_consumption_budget_status(
        self,
        results: List[Tuple[Household, Dict[str, Any]]],
        balance_by_household: Dict[str, float],
        expected_income_by_household: Dict[str, float],
    ) -> None:
        limit = int(getattr(self.config, "debug_max_households", 0) or 0)
        items, skipped = self._limit_list(results, limit)
        for hh, out in items:
            balance = balance_by_household.get(hh.household_id)
            expected_income = expected_income_by_household.get(hh.household_id, 0.0)
            available_budget = None
            if balance is not None:
                available_budget = float(balance) + float(max(0.0, expected_income))
            step0 = out.get("step0", {}) if isinstance(out, dict) else {}
            total_budget = float(step0.get("total_budget") or 0.0)
            balance_text = f"{float(balance or 0.0):.2f}" if balance is not None else "None"
            avail_text = f"{float(available_budget or 0.0):.2f}" if available_budget is not None else "None"
            logger.info(
                f"[消费预算] {hh.household_id} available_balance={balance_text} "
                f"expected_income={float(expected_income or 0.0):.2f} "
                f"available_budget={avail_text} total_budget={total_budget:.2f}"
            )
        if skipped > 0:
            logger.info(f"[消费预算] 其余家庭未打印数量={skipped}")

    def _log_demand_summary(
        self,
        demand_by_retail_firm: Dict[str, Dict[str, float]],
        demand_by_mfg_firm: Dict[str, Dict[str, float]]
    ) -> None:
        limit = int(getattr(self.config, "debug_max_firms", 0) or 0)
        retail_items = sorted(
            demand_by_retail_firm.items(),
            key=lambda kv: kv[1].get("value", 0.0),
            reverse=True,
        )
        retail_items, retail_skipped = self._limit_list(retail_items, limit)
        for firm_id, stats in retail_items:
            logger.info(
                f"[需求-零售] {self._format_firm(firm_id)} qty={stats.get('qty', 0.0):.0f} "
                f"value={stats.get('value', 0.0):.2f} "
                f"gross={stats.get('gross_value', stats.get('value', 0.0)):.2f}"
            )
        if retail_skipped > 0:
            logger.info(f"[需求-零售] 其余企业未打印数量={retail_skipped}")

        mfg_items = sorted(
            demand_by_mfg_firm.items(),
            key=lambda kv: kv[1].get("value", 0.0),
            reverse=True,
        )
        mfg_items, mfg_skipped = self._limit_list(mfg_items, limit)
        for firm_id, stats in mfg_items:
            logger.info(
                f"[需求-制造] {self._format_firm(firm_id)} qty={stats.get('qty', 0.0):.0f} "
                f"value={stats.get('value', 0.0):.2f}"
            )
        if mfg_skipped > 0:
            logger.info(f"[需求-制造] 其余企业未打印数量={mfg_skipped}")

    def _log_consumption_summary(
        self,
        consumption_by_household: Dict[str, Dict[str, float]],
        revenue_by_firm: Dict[str, Dict[str, float]],
    ) -> None:
        h_limit = int(getattr(self.config, "debug_max_households", 0) or 0)
        h_items = sorted(
            consumption_by_household.items(),
            key=lambda kv: kv[1].get("value", 0.0),
            reverse=True,
        )
        h_items, h_skipped = self._limit_list(h_items, h_limit)
        for household_id, stats in h_items:
            logger.info(
                f"[消费结果] {household_id} qty={stats.get('qty', 0.0):.0f} "
                f"value={stats.get('value', 0.0):.2f}"
            )
        if h_skipped > 0:
            logger.info(f"[消费结果] 其余家庭未打印数量={h_skipped}")

        f_limit = int(getattr(self.config, "debug_max_firms", 0) or 0)
        f_items = sorted(
            revenue_by_firm.items(),
            key=lambda kv: kv[1].get("value", 0.0),
            reverse=True,
        )
        f_items, f_skipped = self._limit_list(f_items, f_limit)
        for firm_id, stats in f_items:
            logger.info(
                f"[销售收入] {self._format_firm(firm_id)} qty={stats.get('qty', 0.0):.0f} "
                f"value={stats.get('value', 0.0):.2f}"
            )
        if f_skipped > 0:
            logger.info(f"[销售收入] 其余企业未打印数量={f_skipped}")

    def _log_job_postings(self, postings_by_firm: Dict[str, List[Job]]) -> None:
        limit = int(getattr(self.config, "debug_max_firms", 0) or 0)
        items = list(postings_by_firm.items())
        items, skipped = self._limit_list(items, limit)
        for firm_id, jobs in items:
            total_positions = sum(int(getattr(j, "positions_available", 0) or 0) for j in jobs)
            planned_wage = 0.0
            for j in jobs:
                hours = float(getattr(j, "hours_per_period", 0.0) or 0.0)
                if hours <= 0:
                    hours = 160.0
                planned_wage += float(getattr(j, "wage_per_hour", 0.0) or 0.0) * hours * int(
                    getattr(j, "positions_available", 0) or 0
                )
            logger.info(
                f"[岗位发布] {self._format_firm(firm_id)} positions={total_positions} planned_wage={planned_wage:.2f}"
            )
            for j in jobs:
                logger.info(
                    f"[岗位发布-明细] {self._format_firm(firm_id)} soc={getattr(j, 'SOC', None)} "
                    f"title={getattr(j, 'title', None)} positions={int(getattr(j, 'positions_available', 0) or 0)} "
                    f"wage_per_hour={float(getattr(j, 'wage_per_hour', 0.0) or 0.0):.2f} "
                    f"hours_per_period={float(getattr(j, 'hours_per_period', 0.0) or 0.0):.1f}"
                )
        if skipped > 0:
            logger.info(f"[岗位发布] 其余企业未打印数量={skipped}")

    def _log_matching_summary(self, matched: List[Dict[str, Any]]) -> None:
        hires_by_firm: Dict[str, List[str]] = defaultdict(list)
        for rec in matched:
            firm_id = rec.get("firm_id")
            if not firm_id:
                continue
            household_id = rec.get("household_id")
            lh_type = rec.get("lh_type")
            wage = float(rec.get("wage_per_hour") or 0.0)
            soc = rec.get("soc")
            hires_by_firm[firm_id].append(f"{household_id}:{lh_type}:wage={wage:.2f}:soc={soc}")

        limit = int(getattr(self.config, "debug_max_firms", 0) or 0)
        items = list(hires_by_firm.items())
        items, skipped = self._limit_list(items, limit)
        for firm_id, hires in items:
            logger.info(f"[招聘结果] {self._format_firm(firm_id)} hired={len(hires)} {hires}")
        if skipped > 0:
            logger.info(f"[招聘结果] 其余企业未打印数量={skipped}")

        open_jobs = self._call_actor(self.labor_market, "query_opening_jobs") or []
        open_by_firm: Dict[str, int] = defaultdict(int)
        for job in open_jobs:
            open_by_firm[str(getattr(job, "firm_id", ""))] += int(getattr(job, "positions_available", 0) or 0)
        if open_by_firm:
            items = list(open_by_firm.items())
            items, skipped = self._limit_list(items, limit)
            for firm_id, positions in items:
                logger.info(f"[岗位空缺] {self._format_firm(firm_id)} positions={positions}")
            if skipped > 0:
                logger.info(f"[岗位空缺] 其余企业未打印数量={skipped}")

    def _log_wage_summary(
        self,
        wage_by_firm: Dict[str, float],
        wage_by_household: Dict[str, float],
    ) -> None:
        f_limit = int(getattr(self.config, "debug_max_firms", 0) or 0)
        f_items = sorted(wage_by_firm.items(), key=lambda kv: kv[1], reverse=True)
        f_items, f_skipped = self._limit_list(f_items, f_limit)
        for firm_id, amount in f_items:
            logger.info(f"[工资支出] {self._format_firm(firm_id)} total_wage={amount:.2f}")
        if f_skipped > 0:
            logger.info(f"[工资支出] 其余企业未打印数量={f_skipped}")

        h_limit = int(getattr(self.config, "debug_max_households", 0) or 0)
        h_items = sorted(wage_by_household.items(), key=lambda kv: kv[1], reverse=True)
        h_items, h_skipped = self._limit_list(h_items, h_limit)
        for household_id, amount in h_items:
            logger.info(f"[工资收入] {household_id} total_wage={amount:.2f}")
        if h_skipped > 0:
            logger.info(f"[工资收入] 其余家庭未打印数量={h_skipped}")

    def _log_firm_financials(self, month: int) -> None:
        if self.economic_center is None:
            return
        data = self._call_actor(self.economic_center, "query_all_firms_monthly_financials", month)
        if not isinstance(data, dict):
            return
        limit = int(getattr(self.config, "debug_max_firms", 0) or 0)
        items = list(data.items())
        items.sort(key=lambda kv: kv[1].get("monthly_income", 0.0), reverse=True)
        items, skipped = self._limit_list(items, limit)
        for firm_id, stats in items:
            logger.info(
                f"[企业财务] {self._format_firm(firm_id)} income={float(stats.get('monthly_income', 0.0)):.2f} "
                f"expenses={float(stats.get('monthly_expenses', 0.0)):.2f} "
                f"profit={float(stats.get('monthly_profit', 0.0)):.2f}"
            )
        if skipped > 0:
            logger.info(f"[企业财务] 其余企业未打印数量={skipped}")

    def _get_product_snapshot_cached(self, product_id: str, cache: Dict[str, Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        if product_id in cache:
            return cache[product_id]
        snapshot = self._call_actor(self.product_market, "get_product_snapshot", product_id)
        cache[product_id] = snapshot if isinstance(snapshot, dict) else None
        return cache[product_id]

    @staticmethod
    def _stable_index(key: str, modulo: int) -> int:
        if modulo <= 0:
            return 0
        digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
        return int(digest[:12], 16) % modulo

    def _should_diversify_retail_channel(
        self,
        *,
        household_purchase_count: int,
        goods_budget: float,
        current_retailer_value: float,
        next_order_value: float,
    ) -> bool:
        if not bool(getattr(self.config, "retail_channel_diversification_enabled", True)):
            return False
        min_count = max(1, int(getattr(self.config, "retail_channel_min_purchase_count", 3) or 3))
        if int(household_purchase_count or 0) < min_count:
            return False
        if goods_budget <= 0.0 or next_order_value <= 0.0:
            return False
        max_share = float(getattr(self.config, "retail_channel_max_household_share", 0.70) or 0.70)
        max_share = min(1.0, max(0.05, max_share))
        return (float(current_retailer_value or 0.0) + float(next_order_value or 0.0)) > goods_budget * max_share

    def _select_retail_channel_for_order(
        self,
        *,
        household_id: str,
        product_id: str,
        original_retailer_code: Optional[str],
        household_retail_value: Dict[str, float],
        goods_budget: float,
        order_value: float,
        household_purchase_count: int,
    ) -> Tuple[Optional[str], bool]:
        retailers = sorted(str(code) for code in (getattr(self, "retailers_by_industry", {}) or {}).keys() if code)
        original = str(original_retailer_code or "")
        if not original or original not in retailers or len(retailers) <= 1:
            return original_retailer_code, False
        if not self._should_diversify_retail_channel(
            household_purchase_count=household_purchase_count,
            goods_budget=goods_budget,
            current_retailer_value=float(household_retail_value.get(original, 0.0) or 0.0),
            next_order_value=order_value,
        ):
            return original, False

        alternatives = [code for code in retailers if code != original]
        if not alternatives:
            return original, False
        start = self._stable_index(f"{household_id}:{product_id}:{original}", len(alternatives))
        max_share = min(1.0, max(0.05, float(getattr(self.config, "retail_channel_max_household_share", 0.70) or 0.70)))
        for offset in range(len(alternatives)):
            candidate = alternatives[(start + offset) % len(alternatives)]
            if float(household_retail_value.get(candidate, 0.0) or 0.0) + order_value <= goods_budget * max_share:
                return candidate, True
        return alternatives[start], True

    def _build_orders(
        self,
        consumption_results: List[Tuple[Household, Dict[str, Any]]]
    ) -> Tuple[
        Dict[str, float],
        List[Tuple[Household, List[Dict[str, Any]]]],
        Dict[str, Optional[Dict[str, Any]]],
        Dict[str, Any],
    ]:
        demand_by_product: Dict[str, float] = defaultdict(float)
        # 元组包含: (Household, 订单列表, 商品预算)
        orders_by_household: List[Tuple[Household, List[Dict[str, Any]], float]] = []
        snapshot_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        demand_by_retail_firm: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"qty": 0.0, "value": 0.0, "gross_value": 0.0}
        )
        demand_by_mfg_firm: Dict[str, Dict[str, float]] = defaultdict(lambda: {"qty": 0.0, "value": 0.0})
        demand_by_retailer_product: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        total_demand_qty = 0.0
        total_demand_value = 0.0
        diversified_orders = 0

        for hh, result in consumption_results:
            step0 = result.get("step0", {}) if isinstance(result, dict) else {}
            budgets = step0.get("budgets", {}) if isinstance(step0, dict) else {}
            
            # 提取商品消费预算 (Retail merchandise)
            goods_budget = float(budgets.get("Retail merchandise", 0.0) or 0.0)

            purchases = []
            step3 = result.get("step3", {}) if isinstance(result, dict) else {}
            purchase_recs = list(step3.get("purchases", []) or [])
            household_retail_value: Dict[str, float] = defaultdict(float)
            for rec in purchase_recs:
                product_id = str(rec.get("product_id") or "").strip()
                if not product_id:
                    continue
                budget = float(rec.get("allocated_budget") or 0.0)
                if budget <= 0:
                    continue
                snapshot = self._get_product_snapshot_cached(product_id, snapshot_cache)
                if not snapshot:
                    continue
                unit_price = float(snapshot.get("retail_price") or 0.0)
                if unit_price <= 0:
                    continue
                desired_qty = int(budget // unit_price)
                if desired_qty <= 0:
                    desired_qty = 1
                order_value = float(desired_qty) * unit_price
                manufacturer_unit_price = float(
                    snapshot.get("base_manufacturer_price")
                    or snapshot.get("manufacturer_price")
                    or unit_price
                    or 0.0
                )
                manufacturer_value = float(desired_qty) * max(0.0, manufacturer_unit_price)
                retail_margin_value = max(0.0, order_value - manufacturer_value)
                retailer_code = snapshot.get("retailer_code")
                selected_retailer_code, diversified = self._select_retail_channel_for_order(
                    household_id=str(getattr(hh, "household_id", "")),
                    product_id=product_id,
                    original_retailer_code=retailer_code,
                    household_retail_value=household_retail_value,
                    goods_budget=goods_budget,
                    order_value=order_value,
                    household_purchase_count=len(purchase_recs),
                )
                if diversified:
                    diversified_orders += 1
                if selected_retailer_code:
                    household_retail_value[str(selected_retailer_code)] += order_value
                seller = self._resolve_seller_firm(
                    {
                        "retailer_code": selected_retailer_code,
                        "manufacturer_code": snapshot.get("manufacturer_code"),
                    }
                )
                purchases.append(
                    {
                        "product_id": product_id,
                        "desired_qty": desired_qty,
                        "unit_price": unit_price,
                        "product_name": snapshot.get("name"),
                        "manufacturer_code": snapshot.get("manufacturer_code"),
                        "retailer_code": selected_retailer_code,
                        "seller_id": getattr(seller, "firm_id", None) if seller is not None else None,
                        "origin_retailer_code": retailer_code,
                        "retail_channel_diversified": bool(diversified),
                        "base_retail_price": float(
                            snapshot.get("base_retail_price")
                            or snapshot.get("retail_price")
                            or unit_price
                            or 0.0
                        ),
                        "base_manufacturer_price": float(
                            manufacturer_unit_price
                        ),
                    }
                )
                demand_by_product[product_id] += desired_qty
                total_demand_qty += float(desired_qty)
                total_demand_value += order_value
                manufacturer_code = snapshot.get("manufacturer_code")
                ret_firm = self.retailers_by_industry.get(selected_retailer_code) if selected_retailer_code else None
                mfg_firm = self.manufacturers_by_industry.get(manufacturer_code) if manufacturer_code else None
                if ret_firm is not None:
                    stats = demand_by_retail_firm[ret_firm.firm_id]
                    stats["qty"] += float(desired_qty)
                    # Retail labor should be tied to retail distribution output
                    # (channel margin), not the full merchandise pass-through.
                    stats["value"] += retail_margin_value
                    stats["gross_value"] += order_value
                    demand_by_retailer_product[str(selected_retailer_code)][product_id] += float(desired_qty)
                if mfg_firm is not None:
                    stats = demand_by_mfg_firm[mfg_firm.firm_id]
                    stats["qty"] += float(desired_qty)
                    stats["value"] += manufacturer_value

            orders_by_household.append((hh, purchases, goods_budget))

        if self._debug_enabled():
            self._log_demand_summary(demand_by_retail_firm, demand_by_mfg_firm)

        demand_summary = {
            "total_qty": total_demand_qty,
            "total_value": total_demand_value,
            "by_retail_firm": {
                str(k): {
                    "qty": v["qty"],
                    "value": v["value"],
                    "gross_value": v.get("gross_value", v["value"]),
                }
                for k, v in demand_by_retail_firm.items()
            },
            "by_mfg_firm": {str(k): {"qty": v["qty"], "value": v["value"]} for k, v in demand_by_mfg_firm.items()},
            "by_retailer_product": {
                str(retailer_code): {str(product_id): float(qty) for product_id, qty in products.items()}
                for retailer_code, products in demand_by_retailer_product.items()
            },
            "retail_channel_diversified_orders": diversified_orders,
        }
        return demand_by_product, orders_by_household, snapshot_cache, demand_summary

    def _ensure_production(
        self,
        demand_by_product: Dict[str, float],
        snapshot_cache: Dict[str, Optional[Dict[str, Any]]],
        month: int,
        record_transactions: bool,
        sales_history_by_product: Optional[Dict[str, float]] = None,
        unmet_demand_by_product: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        sales_history_by_product = {
            str(product_id): max(0.0, float(qty or 0.0))
            for product_id, qty in (sales_history_by_product or {}).items()
            if product_id
        }
        unmet_demand_by_product = {
            str(product_id): max(0.0, float(qty or 0.0))
            for product_id, qty in (unmet_demand_by_product or {}).items()
            if product_id
        }
        firm_plans: Dict[ManufactureFirm, Dict[str, int]] = defaultdict(dict)
        unmet_products: List[str] = []
        production_stats = {
            "total_qty": 0.0,
            "total_value": 0.0,
            "by_firm": {},
            "planning": {
                "active": bool(getattr(self.config, "active_production_planning", True)),
                "target_inventory_months": float(getattr(self.config, "production_target_inventory_months", 1.0) or 0.0),
                "by_firm": {},
            },
            # GDP 计算需要的字段
            "total_output_value": 0.0,
            "total_production_cost": 0.0,
            "firm_production_value": {},
            "firm_production_cost": {},
            "firm_total_production_cost_with_labor": {},
            "firm_labor_cost_in_production": {},
        }
        active_planning = bool(getattr(self.config, "active_production_planning", True))
        target_inventory_months = max(
            0.0,
            float(getattr(self.config, "production_target_inventory_months", 1.0) or 0.0),
        )
        defaulted_firm_ids = self._get_defaulted_firm_ids()
        firm_targets: Dict[ManufactureFirm, Dict[str, Dict[str, float]]] = defaultdict(dict)
        production_gap_value_by_firm: Dict[str, float] = {}
        for product_id, demand_qty in (demand_by_product or {}).items():
            snapshot = self._get_product_snapshot_cached(product_id, snapshot_cache)
            if not snapshot:
                continue
            available = int(snapshot.get("available_stock") or 0)
            demand_qty_float = max(0.0, float(demand_qty or 0.0))
            if active_planning:
                target_stock = demand_qty_float * target_inventory_months
                desired_qty = max(0.0, demand_qty_float + target_stock - float(available))
            else:
                desired_qty = max(0.0, float(int(demand_qty_float) - available))
            planned_qty = int(desired_qty + 0.999999)
            if planned_qty <= 0:
                continue
            mfg_code = snapshot.get("manufacturer_code")
            firm = self.manufacturers_by_industry.get(mfg_code)
            if firm is None:
                unmet_products.append(product_id)
                continue
            if str(firm.firm_id) in defaulted_firm_ids:
                production_stats["planning"]["by_firm"].setdefault(
                    firm.firm_id,
                    {
                        "defaulted": True,
                        "desired_output": 0.0,
                        "feasible_output": 0.0,
                        "production_gap": float(planned_qty),
                        "aggregate_demand_signal": demand_qty_float,
                    },
                )
                continue
            if active_planning:
                firm_targets[firm][product_id] = {
                    "desired_qty": float(planned_qty),
                    "demand_qty": demand_qty_float,
                    "sales_qty": sales_history_by_product.get(str(product_id), 0.0),
                    "unmet_qty": unmet_demand_by_product.get(str(product_id), 0.0),
                    "available_stock": float(available),
                }
            else:
                firm_plans[firm][product_id] = firm_plans[firm].get(product_id, 0) + planned_qty

        if active_planning:
            apply_constraints = bool(getattr(self.config, "production_apply_capacity_constraints", False))
            firm_cash_balances: Dict[str, float] = {}
            firm_credit_limits: Dict[str, float] = {}
            firm_debt_balances: Dict[str, float] = {}
            if apply_constraints and self.economic_center is not None:
                try:
                    balances = self._call_actor(self.economic_center, "get_all_balances") or {}
                    firm_cash_balances = {
                        str(fid): float(value or 0.0)
                        for fid, value in balances.items()
                        if str(fid) in self._firm_by_id
                    }
                except Exception as exc:
                    logger.debug(f"[生产计划] 获取企业现金余额失败: {exc}")
                try:
                    firm_credit_limits = {
                        str(fid): float(value or 0.0)
                        for fid, value in (
                            self._call_actor(self.economic_center, "get_all_firm_credit_limits") or {}
                        ).items()
                    }
                except Exception as exc:
                    logger.debug(f"[生产计划] 获取企业信用额度失败: {exc}")
                try:
                    firm_debt_balances = {
                        str(fid): float(value or 0.0)
                        for fid, value in (
                            self._call_actor(self.economic_center, "get_all_firm_debt_balances") or {}
                        ).items()
                    }
                except Exception as exc:
                    logger.debug(f"[生产计划] 获取企业债务余额失败: {exc}")

            for firm, product_targets in firm_targets.items():
                aggregate_demand = sum(float(v.get("demand_qty", 0.0) or 0.0) for v in product_targets.values())
                aggregate_sales_history = sum(float(v.get("sales_qty", 0.0) or 0.0) for v in product_targets.values())
                aggregate_unmet_history = sum(float(v.get("unmet_qty", 0.0) or 0.0) for v in product_targets.values())
                aggregate_inventory = sum(float(v.get("available_stock", 0.0) or 0.0) for v in product_targets.values())
                aggregate_desired = sum(float(v.get("desired_qty", 0.0) or 0.0) for v in product_targets.values())
                if aggregate_desired <= 0.0:
                    continue
                product_snapshots = [
                    self._get_product_snapshot_cached(product_id, snapshot_cache)
                    for product_id in product_targets.keys()
                ]
                unit_cash_cost = self._estimate_unit_cash_cost_for_targets(product_snapshots)
                configured_unit_cash_cost = getattr(self.config, "production_unit_cash_cost", None)
                if configured_unit_cash_cost is not None:
                    unit_cash_cost = float(configured_unit_cash_cost)
                aggregate_desired_value = 0.0
                for product_id, target in product_targets.items():
                    snapshot = self._get_product_snapshot_cached(product_id, snapshot_cache)
                    if not snapshot:
                        continue
                    price = float(
                        snapshot.get("manufacturer_price")
                        or snapshot.get("base_manufacturer_price")
                        or snapshot.get("retail_price")
                        or 0.0
                    )
                    aggregate_desired_value += max(0.0, float(target.get("desired_qty", 0.0) or 0.0)) * max(0.0, price)
                avg_target_price = (
                    aggregate_desired_value / aggregate_desired
                    if aggregate_desired > 0.0 and aggregate_desired_value > 0.0
                    else max(0.0, float(unit_cash_cost or 0.0))
                )
                labor_productivity = self._estimate_labor_productivity_for_targets(
                    firm,
                    product_snapshots,
                )
                firm_id = str(firm.firm_id)
                cash_balance = firm_cash_balances.get(firm_id)
                credit_limit = firm_credit_limits.get(firm_id)
                credit_outstanding = firm_debt_balances.get(firm_id, 0.0)
                has_observed_history = aggregate_sales_history > 0.0 or aggregate_unmet_history > 0.0

                plan_result = firm.build_production_plan(
                    sales_history=[aggregate_sales_history] if has_observed_history else [],
                    unmet_demand_history=[aggregate_unmet_history] if has_observed_history else [],
                    current_inventory=aggregate_inventory,
                    target_inventory_months=target_inventory_months,
                    ema_alpha=float(getattr(self.config, "production_ema_alpha", 0.5) or 0.5),
                    fallback_expected_demand=aggregate_demand,
                    labor_productivity=labor_productivity,
                    capital_productivity=float(getattr(self.config, "production_capital_productivity", 1.0) or 1.0),
                    cash=cash_balance,
                    unit_cash_cost=unit_cash_cost,
                    cash_reserve=float(getattr(self.config, "production_cash_reserve", 0.0) or 0.0),
                    credit_limit=credit_limit,
                    credit_outstanding=credit_outstanding,
                    use_current_state_constraints=apply_constraints,
                )
                feasible_output = max(0.0, float(plan_result.feasible_output or 0.0))
                scale = min(1.0, feasible_output / aggregate_desired) if aggregate_desired > 0.0 else 0.0
                production_gap_value = max(0.0, float(plan_result.production_gap or 0.0)) * avg_target_price
                if production_gap_value > 0.0:
                    production_gap_value_by_firm[firm_id] = production_gap_value
                for product_id, target in product_targets.items():
                    qty = int(float(target.get("desired_qty", 0.0) or 0.0) * scale)
                    if qty <= 0 and scale > 0.0:
                        qty = 1
                    if qty > 0:
                        firm_plans[firm][product_id] = firm_plans[firm].get(product_id, 0) + qty
                production_stats["planning"]["by_firm"][firm.firm_id] = {
                    "expected_demand": plan_result.expected_demand,
                    "target_inventory": plan_result.target_inventory,
                    "current_inventory": plan_result.current_inventory,
                    "aggregate_demand_signal": aggregate_demand,
                    "sales_history_demand": aggregate_sales_history,
                    "unmet_demand": aggregate_unmet_history,
                    "desired_output": plan_result.desired_output,
                    "feasible_output": plan_result.feasible_output,
                    "production_gap": plan_result.production_gap,
                    "desired_output_value": aggregate_desired_value,
                    "feasible_output_value": feasible_output * avg_target_price,
                    "production_gap_value": production_gap_value,
                    "limiting_constraints": list(plan_result.diagnostics.get("limiting_constraints", [])),
                    "binding_constraints": list(plan_result.diagnostics.get("binding_constraints", [])),
                    "labor_productivity": labor_productivity,
                    "unit_cash_cost": unit_cash_cost,
                    "cash": cash_balance,
                    "credit_limit": credit_limit,
                    "credit_outstanding": credit_outstanding,
                }

        if self._debug_enabled():
            limit = int(getattr(self.config, "debug_max_firms", 0) or 0)
            items = [
                (firm.firm_id, sum(plan.values()), len(plan))
                for firm, plan in firm_plans.items()
            ]
            items.sort(key=lambda x: x[1], reverse=True)
            items, skipped = self._limit_list(items, limit)
            for firm_id, qty, sku_count in items:
                logger.info(
                    f"[生产计划] {self._format_firm(firm_id)} planned_qty={qty} skus={sku_count}"
                )
            if skipped > 0:
                logger.info(f"[生产计划] 其余企业未打印数量={skipped}")
            if unmet_products:
                logger.info(f"[生产计划] 无法匹配制造商的SKU数量={len(unmet_products)}")

        self._last_production_gap_value_by_firm = {
            str(firm_id): float(value)
            for firm_id, value in production_gap_value_by_firm.items()
            if float(value or 0.0) > 0.0
        }

        for firm, plan in firm_plans.items():
            sku_base_prices = {}
            sku_current_prices = {}
            for sku_id in plan.keys():
                snapshot = self._get_product_snapshot_cached(sku_id, snapshot_cache)
                if not snapshot:
                    continue
                price = float(
                    snapshot.get("base_manufacturer_price")
                    or snapshot.get("manufacturer_price")
                    or snapshot.get("retail_price")
                    or 1.0
                )
                sku_base_prices[sku_id] = price
                # 生产价值使用当前市场价（动态），而非静态基准价
                current_price = float(
                    snapshot.get("manufacturer_price")
                    or snapshot.get("base_manufacturer_price")
                    or snapshot.get("retail_price")
                    or 1.0
                )
                sku_current_prices[sku_id] = current_price
            # 初始化生产成本（后面会被实际值覆盖）
            firm_production_cost = 0.0
            firm_total_cost_with_labor = 0.0
            firm_labor_cost = 0.0
            actual_plan = dict(plan or {})
            
            if record_transactions:
                # 注意：制造商已在初始化时预注册到中间品市场
                # 不再需要每次生产时重复注册
                produce_result = firm.produce(
                    production_plan=plan,
                    sku_base_prices=sku_base_prices,
                    period=month,
                    update_inventory=True,
                )
                actual_plan = {
                    str(sku_id): int(float(qty or 0.0))
                    for sku_id, qty in (produce_result.get("production_plan") or plan or {}).items()
                    if float(qty or 0.0) > 0.0
                }
                # 获取实际生产成本（中间消耗）
                cost_breakdown = produce_result.get("cost_breakdown", {}) or {}
                firm_labor_cost = float(cost_breakdown.get("labor", 0.0) or 0.0)
                firm_total_cost_with_labor = float(produce_result.get("total_cost", 0.0) or 0.0)
                firm_production_cost = max(0.0, firm_total_cost_with_labor - firm_labor_cost)
                
                # 记录原材料需求（用于原材料价格调整）
                intermediate_by_industry = produce_result.get("intermediate_by_industry", {})
                for industry_code, cost in intermediate_by_industry.items():
                    if industry_code and cost > 0:
                        if "raw_material_demand" not in production_stats:
                            production_stats["raw_material_demand"] = {}
                        production_stats["raw_material_demand"][industry_code] = (
                            production_stats["raw_material_demand"].get(industry_code, 0.0) + cost
                        )
            else:
                self._manual_produce(firm, plan, sku_base_prices, month)

            firm_qty = float(sum(float(qty or 0.0) for qty in actual_plan.values()))
            firm_value = 0.0
            for sku_id, qty in (actual_plan or {}).items():
                firm_value += float(qty or 0.0) * float(sku_current_prices.get(sku_id, 0.0) or 0.0)
            if record_transactions and isinstance(produce_result, dict):
                produced_value = produce_result.get("production_value")
                if produced_value is not None:
                    try:
                        firm_value = float(produced_value or 0.0)
                    except (TypeError, ValueError):
                        pass
            if firm_qty > 0.0 or firm_value > 0.0:
                production_stats["by_firm"][firm.firm_id] = {"qty": firm_qty, "value": firm_value}
                production_stats["total_qty"] += firm_qty
                production_stats["total_value"] += firm_value
            
            # 更新 GDP 计算需要的统计
            production_stats["firm_production_value"][firm.firm_id] = firm_value
            production_stats["firm_production_cost"][firm.firm_id] = firm_production_cost
            production_stats["firm_total_production_cost_with_labor"][firm.firm_id] = (
                firm_total_cost_with_labor if record_transactions else firm_production_cost
            )
            production_stats["firm_labor_cost_in_production"][firm.firm_id] = (
                firm_labor_cost if record_transactions else 0.0
            )
            production_stats["total_output_value"] += firm_value
            production_stats["total_production_cost"] += firm_production_cost

            # 生产完成后，清空已生产产品的缓存，以便后续步骤获取最新库存
            for sku_id in set(plan.keys()) | set(actual_plan.keys()):
                if sku_id in snapshot_cache:
                    del snapshot_cache[sku_id]

        # 记录原材料需求到ProductMarket（用于价格调整）
        raw_material_demand = production_stats.get("raw_material_demand", {})
        if raw_material_demand and self.product_market is not None:
            for industry_code, demand_value in raw_material_demand.items():
                # 记录原材料需求（以成本值作为需求指标）
                self._call_actor(self.product_market, "record_raw_material_demand", industry_code, demand_value)

        return production_stats

    def _estimate_labor_productivity_for_targets(
        self,
        firm: ManufactureFirm,
        product_snapshots: List[Optional[Dict[str, Any]]],
    ) -> float:
        """
        Convert the labor constraint from a fixed units/hour number into a
        value-consistent units/hour number for the targeted SKU mix.

        Job posting budgets are based on expected revenue times the firm's
        compensation ratio. If production capacity is constrained by a fixed
        unit count, low-price SKU industries can pay a whole worker while the
        labor constraint only allows a few hundred dollars of output. This
        calibration keeps the default configured productivity as a floor while
        raising units/hour when the SKU price level requires it.
        """
        base_productivity = max(
            0.0,
            float(getattr(self.config, "production_labor_productivity", 1.0) or 1.0),
        )
        if not bool(getattr(self.config, "production_value_calibrated_labor_productivity", True)):
            return base_productivity

        prices: List[float] = []
        for snapshot in product_snapshots or []:
            if not snapshot:
                continue
            # 用基准批发价(base_manufacturer_price)做价格锚，而非当前价格。
            # 否则通缩期当前价格下跌→校准生产率(=目标产值/(价格×工时))暴涨→产量暴增→
            # 供过于求→价格更低→生产率更高，形成产量失控棘轮(实测生产率从2.5飙到1900+，
            # 月产为真实需求的10倍，存货虚增，GDP 支出法 I 失真)。基准价稳定，打破该反馈。
            price = float(
                snapshot.get("base_manufacturer_price")
                or snapshot.get("manufacturer_price")
                or snapshot.get("retail_price")
                or 0.0
            )
            if price > 0.0:
                prices.append(price)
        if not prices:
            return base_productivity

        total_wage = 0.0
        total_hours = 0.0
        for employee in getattr(firm, "employee_list", []) or []:
            hours = getattr(employee, "total_hours", None)
            if hours is None:
                hours = getattr(employee, "hours_per_period", None)
            try:
                hours_value = max(0.0, float(hours or 0.0))
            except (TypeError, ValueError):
                hours_value = 0.0
            wage_per_hour = getattr(employee, "wage_per_hour", None)
            if wage_per_hour is None:
                wage_per_hour = getattr(employee, "average_wage", None)
            try:
                wage_value = max(0.0, float(wage_per_hour or 0.0))
            except (TypeError, ValueError):
                wage_value = 0.0
            total_hours += hours_value
            total_wage += hours_value * wage_value

        if total_hours <= 0.0 or total_wage <= 0.0:
            return base_productivity

        avg_price = sum(prices) / len(prices)
        if avg_price <= 0.0:
            return base_productivity
        compensation_ratio = max(0.01, float(getattr(firm, "compensation_ratio", 0.2) or 0.2))
        target_output_value = total_wage / compensation_ratio
        calibrated_units_per_hour = target_output_value / (avg_price * total_hours)
        # 生产率上限：校准值不超过基准生产率的固定倍数，防止极端价格/工资比导致产量失控。
        productivity_cap = base_productivity * float(
            getattr(self.config, "production_value_calibrated_productivity_cap", 20.0) or 20.0
        )
        return max(base_productivity, min(calibrated_units_per_hour, productivity_cap))

    def _estimate_unit_cash_cost_for_targets(
        self,
        product_snapshots: List[Optional[Dict[str, Any]]],
    ) -> Optional[float]:
        """
        Conservative cash-cost proxy for production planning.

        Full COGS is still computed in `produce()`. The planner needs a
        pre-production cash gate, so when no explicit config value is provided
        it uses a fraction of current manufacturer prices for the targeted SKUs.
        """
        values: List[float] = []
        for snapshot in product_snapshots or []:
            if not snapshot:
                continue
            price = float(
                snapshot.get("manufacturer_price")
                or snapshot.get("base_manufacturer_price")
                or snapshot.get("retail_price")
                or 0.0
            )
            if price > 0.0:
                values.append(price)
        if not values:
            return None
        cash_share = float(getattr(self.config, "production_unit_cash_cost_share", 0.6) or 0.6)
        cash_share = max(0.0, min(1.0, cash_share))
        return (sum(values) / len(values)) * cash_share

    def _manual_produce(
        self,
        firm: ManufactureFirm,
        production_plan: Dict[str, int],
        sku_base_prices: Dict[str, float],
        month: int
    ) -> None:
        for sku_id, qty in (production_plan or {}).items():
            if qty <= 0:
                continue
            self._call_actor(self.product_market, "update_stock", sku_id, qty)
        production_value = 0.0
        for sku_id, qty in (production_plan or {}).items():
            production_value += float(sku_base_prices.get(sku_id, 0.0) or 0.0) * float(qty or 0.0)
        firm.production_history.append(
            {
                "period": month,
                "production_plan": dict(production_plan or {}),
                "production_value": production_value,
                "total_cost": 0.0,
            }
        )

    def _retailer_procurement(
        self,
        demand_by_product: Dict[str, float],
        snapshot_cache: Dict[str, Optional[Dict[str, Any]]],
        month: int,
        record_transactions: bool,
        demand_by_retailer_product: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """
        零售商进货阶段：根据预测需求，零售商从制造商批发采购商品
        
        流程：
        1. 按零售商汇总需要进货的产品
        2. 零售商支付批发价（manufacturer_price）给制造商
        3. 更新零售商库存记录
        
        Args:
            demand_by_product: 各产品的需求量
            snapshot_cache: 产品快照缓存
            month: 当前月份
            record_transactions: 是否记录交易
            
        Returns:
            进货统计 {total_qty, total_value, by_retailer, by_manufacturer}
        """
        # 按零售商汇总进货需求: {retailer_code: {product_id: qty}}
        # 零售商进货保守系数：家庭计划需求(desired_qty)>实际成交量(受预算/价格约束)，
        # 满额按计划进货会让卖不掉的库存累积成亏损→零售商破产→商品市场崩溃。
        # 按保守系数(<1)进货，留滞销缓冲，使进货贴近真实动销。
        procurement_safety = max(
            0.1, min(1.0, float(getattr(self.config, "retailer_procurement_safety_factor", 1.0) or 1.0))
        )
        procurement_by_retailer: Dict[str, Dict[str, int]] = defaultdict(dict)
        if demand_by_retailer_product:
            for retailer_code, products in (demand_by_retailer_product or {}).items():
                if not retailer_code or retailer_code not in self.retailers_by_industry:
                    continue
                if not isinstance(products, dict):
                    continue
                for product_id, demand_qty in products.items():
                    qty = int(float(demand_qty or 0.0) * procurement_safety)
                    if qty <= 0:
                        continue
                    snapshot = self._get_product_snapshot_cached(str(product_id), snapshot_cache)
                    if not snapshot:
                        continue
                    procurement_by_retailer[str(retailer_code)][str(product_id)] = (
                        procurement_by_retailer[str(retailer_code)].get(str(product_id), 0) + qty
                    )
        else:
            for product_id, demand_qty in (demand_by_product or {}).items():
                if demand_qty <= 0:
                    continue
                snapshot = self._get_product_snapshot_cached(product_id, snapshot_cache)
                if not snapshot:
                    continue
                
                retailer_code = snapshot.get("retailer_code")
                if not retailer_code or retailer_code not in self.retailers_by_industry:
                    continue
                
                procurement_by_retailer[str(retailer_code)][str(product_id)] = int(float(demand_qty or 0.0) * procurement_safety)

        procurement_by_retailer = self._allocate_retailer_procurement_by_stock(
            procurement_by_retailer,
            snapshot_cache,
        )
        
        # 统计数据
        procurement_stats = {
            "total_qty": 0.0,
            "total_value": 0.0,
            "by_retailer": {},
            "by_manufacturer": defaultdict(lambda: {"qty": 0.0, "value": 0.0}),
        }
        
        # 执行进货
        for retailer_code, products in procurement_by_retailer.items():
            retailer = self.retailers_by_industry.get(retailer_code)
            if retailer is None:
                continue
            
            retailer_qty = 0.0
            retailer_value = 0.0
            
            for product_id, qty in products.items():
                snapshot = self._get_product_snapshot_cached(product_id, snapshot_cache)
                if not snapshot:
                    continue
                
                # 检查库存是否充足（制造商已生产）。渠道需求可让多个零售商
                # 竞争同一SKU，ProductMarket 会按实际制造商库存裁剪。
                available = int(snapshot.get("available_stock") or 0)
                qty_to_procure = min(int(qty), available)
                if qty_to_procure <= 0:
                    continue

                # 批发价 = manufacturer_price
                wholesale_price = float(
                    snapshot.get("base_manufacturer_price")
                    or snapshot.get("manufacturer_price")
                    or 0.0
                )
                if wholesale_price <= 0:
                    continue
                
                manufacturer_code = snapshot.get("manufacturer_code")
                manufacturer = self.manufacturers_by_industry.get(manufacturer_code)
                if manufacturer is None:
                    continue
                
                stock_result = self._call_actor(
                    self.product_market,
                    "purchase_manufacturer_stock",
                    product_id,
                    qty_to_procure,
                ) or {}
                actual_qty = float(stock_result.get("actual_quantity", 0.0) or 0.0)
                if actual_qty <= 0.0:
                    if product_id in snapshot_cache and snapshot_cache[product_id]:
                        snapshot_cache[product_id]["available_stock"] = float(
                            stock_result.get("available_after", 0.0) or 0.0
                        )
                    continue

                amount = wholesale_price * actual_qty
                
                if record_transactions and self.economic_center is not None:
                    # 零售商支付给制造商
                    try:
                        tx_id = self._call_actor(
                            self.economic_center,
                            "process_wholesale",
                            month,
                            retailer.firm_id,
                            manufacturer.firm_id,
                            amount,
                            actual_qty,
                            product_id,
                            snapshot.get("name"),
                            wholesale_price,
                        )
                    except Exception as exc:
                        logger.warning(
                            f"[零售商进货] 结算异常，回滚库存: retailer={retailer.firm_id} "
                            f"product={product_id} qty={actual_qty} error={exc}"
                        )
                        tx_id = None
                    if tx_id:
                        self._call_actor(
                            self.product_market,
                            "receive_retailer_inventory",
                            retailer.firm_id,
                            product_id,
                            actual_qty,
                            wholesale_price,
                        )
                        retailer_qty += actual_qty
                        retailer_value += amount
                        procurement_stats["by_manufacturer"][manufacturer.firm_id]["qty"] += actual_qty
                        procurement_stats["by_manufacturer"][manufacturer.firm_id]["value"] += amount
                    else:
                        restore_result = self._restore_reserved_stock(
                            stock_result=stock_result,
                            product_id=product_id,
                            quantity=actual_qty,
                            snapshot_cache=snapshot_cache,
                        )
                        if restore_result.get("success"):
                            stock_result["available_after"] = restore_result.get("available_after")
                else:
                    # 不记录交易模式：直接转移资金
                    if retailer.cash >= amount:
                        retailer.cash -= amount
                        manufacturer.cash += amount
                        self._call_actor(
                            self.product_market,
                            "receive_retailer_inventory",
                            retailer.firm_id,
                            product_id,
                            actual_qty,
                            wholesale_price,
                        )
                        retailer_qty += actual_qty
                        retailer_value += amount
                        procurement_stats["by_manufacturer"][manufacturer.firm_id]["qty"] += actual_qty
                        procurement_stats["by_manufacturer"][manufacturer.firm_id]["value"] += amount
                    else:
                        restore_result = self._restore_reserved_stock(
                            stock_result=stock_result,
                            product_id=product_id,
                            quantity=actual_qty,
                            snapshot_cache=snapshot_cache,
                        )
                        if restore_result.get("success"):
                            stock_result["available_after"] = restore_result.get("available_after")

                if product_id in snapshot_cache and snapshot_cache[product_id]:
                    snapshot_cache[product_id]["available_stock"] = float(
                        stock_result.get("available_after", 0.0) or 0.0
                    )
            
            if retailer_qty > 0:
                procurement_stats["by_retailer"][retailer.firm_id] = {
                    "qty": retailer_qty,
                    "value": retailer_value,
                }
                procurement_stats["total_qty"] += retailer_qty
                procurement_stats["total_value"] += retailer_value
        
        if self._debug_enabled():
            logger.info(
                f"[零售商进货] 总数量={procurement_stats['total_qty']:.0f}, "
                f"总金额={procurement_stats['total_value']:.2f}, "
                f"零售商数={len(procurement_stats['by_retailer'])}"
            )
        
        return procurement_stats

    def _allocate_retailer_procurement_by_stock(
        self,
        procurement_by_retailer: Dict[str, Dict[str, int]],
        snapshot_cache: Dict[str, Optional[Dict[str, Any]]],
    ) -> Dict[str, Dict[str, int]]:
        """
        Allocate scarce manufacturer stock across retailer channels by SKU.

        This avoids fixed retailer iteration order becoming a hidden market
        power mechanism when several retailers carry demand for the same SKU.
        """
        by_product: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        for retailer_code, products in (procurement_by_retailer or {}).items():
            for product_id, qty in (products or {}).items():
                qty_int = int(qty or 0)
                if qty_int > 0:
                    by_product[str(product_id)].append((str(retailer_code), qty_int))

        allocated: Dict[str, Dict[str, int]] = defaultdict(dict)
        for product_id, requests in by_product.items():
            total_requested = sum(qty for _, qty in requests)
            if total_requested <= 0:
                continue
            snapshot = self._get_product_snapshot_cached(product_id, snapshot_cache)
            if not snapshot:
                continue
            available = max(0, int(float(snapshot.get("available_stock") or 0.0)))
            total_to_allocate = min(total_requested, available)
            if total_to_allocate <= 0:
                continue
            if total_to_allocate >= total_requested:
                for retailer_code, qty in requests:
                    allocated[retailer_code][product_id] = qty
                continue

            floor_allocations: List[Tuple[str, int, float]] = []
            used = 0
            for retailer_code, qty in requests:
                exact = (float(qty) / float(total_requested)) * float(total_to_allocate)
                floor_qty = int(exact)
                floor_allocations.append((retailer_code, floor_qty, exact - floor_qty))
                used += floor_qty
            remainder = total_to_allocate - used
            floor_allocations.sort(key=lambda item: (-item[2], item[0]))
            for idx, (retailer_code, qty, _) in enumerate(floor_allocations):
                allocated_qty = qty + (1 if idx < remainder else 0)
                if allocated_qty > 0:
                    allocated[retailer_code][product_id] = allocated_qty

        return {retailer_code: dict(products) for retailer_code, products in allocated.items()}

    def _restore_reserved_stock(
        self,
        *,
        stock_result: Optional[Dict[str, Any]],
        product_id: str,
        quantity: float,
        seller_id: Optional[str] = None,
        snapshot_cache: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        if quantity <= 0.0 or self.product_market is None:
            return {}
        source = (stock_result or {}).get("source")
        try:
            if source == "retailer" and seller_id:
                restore_result = self._call_actor(
                    self.product_market,
                    "restore_retailer_inventory",
                    seller_id,
                    product_id,
                    quantity,
                ) or {}
            else:
                restore_result = self._call_actor(
                    self.product_market,
                    "restore_manufacturer_stock",
                    product_id,
                    quantity,
                ) or {}
        except Exception as exc:
            logger.warning(f"[库存回滚] {product_id} qty={quantity} failed: {exc}")
            return {}

        if snapshot_cache is not None and product_id in snapshot_cache and snapshot_cache[product_id]:
            if restore_result.get("success") and "available_after" in restore_result:
                snapshot_cache[product_id]["available_stock"] = float(
                    restore_result.get("available_after", 0.0) or 0.0
                )
        return restore_result

    def _resolve_seller_firm(self, order: Dict[str, Any]) -> Optional[Firm]:
        """
        解析订单的卖家企业
        
        在新的流程中，零售商是主要销售渠道：
        1. 优先返回零售商（如果有）
        2. 只有在没有零售商信息时才回退到制造商（用于直销场景）
        
        注意：制造商在零售商进货阶段已经获得收入，
        所以家庭购买时应该付款给零售商
        """
        retailer_code = order.get("retailer_code")
        retailers_by_industry = getattr(self, "retailers_by_industry", {}) or {}
        if retailer_code and retailer_code in retailers_by_industry:
            return retailers_by_industry[retailer_code]
        
        # 只有在没有 retailer_code 的情况下，才回退到制造商（直销）
        # 如果有 retailer_code 但找不到对应零售商，不应回退到制造商
        if not retailer_code:
            manufacturer_code = order.get("manufacturer_code")
            manufacturers_by_industry = getattr(self, "manufacturers_by_industry", {}) or {}
            if manufacturer_code and manufacturer_code in manufacturers_by_industry:
                return manufacturers_by_industry[manufacturer_code]
        
        return None

    def _execute_service_consumption(
        self,
        consumption_results: List[Tuple[Household, Dict[str, Any]]],
        month: int,
    ) -> Dict[str, Any]:
        """
        执行家庭服务类消费（housing, healthcare, utilities 等）
        
        将 step0 中分配的服务类预算通过 AbstractResourceMarket 转化为实际消费，
        资金流向对应的 ServiceFirm 或 Government。
        
        Args:
            consumption_results: [(Household, consume_v2 输出), ...]
            month: 当前经济月

        Returns:
            {
                "total_service_consumption": float,
                "by_category": {category: total_amount},
                "by_industry": {industry_code: total_amount},
                "by_household": {household_id: {value, by_category, by_industry}},
                "household_count": int
            }
        """
        if self.abstract_resource_market is None:
            logger.warning("AbstractResourceMarket 未初始化，跳过服务消费")
            return {
                "total_service_consumption": 0.0,
                "by_category": {},
                "by_industry": {},
                "by_household": {},
                "household_count": 0,
                "skipped_insufficient_balance": 0,
            }

        total_consumption = 0.0
        by_category: Dict[str, float] = defaultdict(float)
        by_industry: Dict[str, float] = defaultdict(float)
        by_household: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "value": 0.0,
                "by_category": defaultdict(float),
                "by_industry": defaultdict(float),
            }
        )
        household_count = 0
        skipped_insufficient_balance = 0

        for hh, out in consumption_results:
            step0 = out.get("step0", {}) if isinstance(out, dict) else {}
            budgets = step0.get("budgets", {}) if isinstance(step0, dict) else {}

            has_service_consumption = False

            # 获取家庭当前余额（商品消费后的实际余额）
            current_balance = 0.0
            if self.economic_center is not None:
                try:
                    current_balance = float(self._call_actor(
                        self.economic_center, "query_balance", hh.household_id
                    ) or 0.0)
                except Exception:
                    current_balance = 0.0

            # 计算服务消费总预算
            total_service_budget = sum(
                float(budgets.get(cat) or 0.0)
                for cat in HOUSEHOLD_SERVICE_CATEGORY_TO_INDUSTRY.keys()
            )

            # 如果余额不足以支付全部服务预算，按比例缩减
            budget_scale = 1.0
            if total_service_budget > 0 and current_balance < total_service_budget:
                if current_balance <= 0:
                    # 余额为0或负数，跳过服务消费
                    skipped_insufficient_balance += 1
                    continue
                budget_scale = current_balance / total_service_budget

            # 遍历服务类别
            for category, industry_weights in HOUSEHOLD_SERVICE_CATEGORY_TO_INDUSTRY.items():
                category_budget = float(budgets.get(category) or 0.0) * budget_scale
                if category_budget <= 0:
                    continue

                # 按权重分配预算到各行业
                for industry_code, weight in industry_weights:
                    industry_budget = category_budget * weight
                    if industry_budget <= 0:
                        continue

                    # 检查该行业是否在 AbstractResourceMarket 中注册
                    if industry_code not in self.abstract_resource_market.resources:
                        logger.debug(f"行业 {industry_code} 未在 AbstractResourceMarket 注册，跳过")
                        continue

                    try:
                        # 使用 purchase_by_budget 执行消费
                        transaction = self.abstract_resource_market.purchase_by_budget(
                            industry_code=industry_code,
                            buyer_id=hh.household_id,
                            budget=industry_budget,
                            period=month,
                        )

                        if transaction:
                            actual_cost = float(transaction.get("total_cost") or 0.0)
                            total_consumption += actual_cost
                            by_category[category] += actual_cost
                            by_industry[industry_code] += actual_cost
                            hh_stats = by_household[hh.household_id]
                            hh_stats["value"] += actual_cost
                            hh_stats["by_category"][category] += actual_cost
                            hh_stats["by_industry"][industry_code] += actual_cost
                            has_service_consumption = True

                    except Exception as e:
                        logger.error(f"家庭 {hh.household_id} 消费 {industry_code} 失败: {e}")

            if has_service_consumption:
                household_count += 1

        if total_consumption > 0:
            logger.info(
                f"[服务消费] 总额=${total_consumption:.2f}, "
                f"家庭数={household_count}, "
                f"分类={dict(by_category)}"
            )
        if skipped_insufficient_balance > 0:
            logger.info(f"[服务消费] 因余额不足跳过: {skipped_insufficient_balance} 个家庭")

        return {
            "total_service_consumption": total_consumption,
            "by_category": dict(by_category),
            "by_industry": dict(by_industry),
            "by_household": {
                str(hh_id): {
                    "value": float(stats.get("value", 0.0) or 0.0),
                    "by_category": dict(stats.get("by_category", {}) or {}),
                    "by_industry": dict(stats.get("by_industry", {}) or {}),
                }
                for hh_id, stats in by_household.items()
            },
            "household_count": household_count,
            "skipped_insufficient_balance": skipped_insufficient_balance,
        }

    def _sync_household_actual_consumption_state(
        self,
        hh: "Household",
        goods_spent: float,
        service_by_category: Dict[str, float],
        total_spent: float,
    ) -> None:
        csv_values = getattr(hh, "csv_values", None)
        if not isinstance(csv_values, dict):
            return

        def set_float_field(field: str, value: float) -> None:
            new_value = float(value or 0.0)
            csv_values[field] = new_value
            setattr(hh, field, new_value)

        set_float_field("expenditure_retail_merchandise", goods_spent)
        set_float_field("ER85701", service_by_category.get("housing", 0.0))
        set_float_field("ER85747", service_by_category.get("healthcare", 0.0))
        set_float_field("expenditure_transportation", service_by_category.get("transportation", 0.0))
        set_float_field("expenditure_utilities", service_by_category.get("utilities", 0.0))
        set_float_field("expenditure_insurance", service_by_category.get("insurance", 0.0))
        set_float_field("ER85768", total_spent)

        try:
            current_wealth = float(csv_values.get("ER85692") or 0.0)
        except (TypeError, ValueError):
            current_wealth = 0.0
        set_float_field("ER85692", current_wealth - float(total_spent or 0.0))
    
    def _update_household_consumption_history(
        self,
        consumption_stats: Optional[Dict[str, Any]],
        service_consumption_stats: Optional[Dict[str, Any]],
        consumption_results: List[Tuple["Household", Dict[str, Any]]],
    ) -> None:
        """
        更新每个家庭的上月实际消费，用于下月消费惯性计算。
        
        Args:
            consumption_stats: 商品消费统计 {by_household: {hh_id: {qty, value}}}
            service_consumption_stats: 服务消费统计
            consumption_results: [(Household, consume_v2 输出), ...]
        """
        # 获取商品消费金额（按家庭）
        goods_by_hh: Dict[str, float] = {}
        if consumption_stats:
            by_household = consumption_stats.get("by_household", {})
            for hh_id, stats in by_household.items():
                goods_by_hh[hh_id] = float(stats.get("value", 0.0) or 0.0)
        
        # 服务消费优先使用 _execute_service_consumption 记录的每户实际成交额。
        service_by_hh: Dict[str, float] = {}
        service_category_by_hh: Dict[str, Dict[str, float]] = {}
        if service_consumption_stats:
            service_households = service_consumption_stats.get("by_household", {})
            if isinstance(service_households, dict):
                for hh_id, stats in service_households.items():
                    if isinstance(stats, dict):
                        service_by_hh[str(hh_id)] = float(stats.get("value", 0.0) or 0.0)
                        by_category = stats.get("by_category", {}) or {}
                        if isinstance(by_category, dict):
                            service_category_by_hh[str(hh_id)] = {
                                str(category): float(amount or 0.0)
                                for category, amount in by_category.items()
                            }
                    else:
                        service_by_hh[str(hh_id)] = float(stats or 0.0)
        
        # 更新每个家庭的上月消费
        updated_count = 0
        for hh, _ in consumption_results:
            goods_spent = float(goods_by_hh.get(hh.household_id, 0.0) or 0.0)
            service_spent = float(service_by_hh.get(hh.household_id, 0.0) or 0.0)
            total_spent = goods_spent + service_spent
            self._sync_household_actual_consumption_state(
                hh,
                goods_spent,
                service_category_by_hh.get(hh.household_id, {}),
                total_spent,
            )
            
            # 调用家庭的更新方法
            if hasattr(hh, "update_last_month_consumption"):
                hh.update_last_month_consumption(total_spent)
                updated_count += 1
        
        if updated_count > 0:
            total_goods = sum(goods_by_hh.values())
            total_service = sum(service_by_hh.values())
            logger.debug(
                f"[消费惯性] 更新 {updated_count} 个家庭的上月消费: "
                f"商品=${total_goods:.2f}, 服务=${total_service:.2f}"
            )
    
    async def _execute_government_procurement(
        self,
        month: int,
        household_consumption_budget: Optional[float] = None,
        planned_demand_by_product: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        政府采购：通过 Government 对象执行采购，向经济体注入需求

        预算基于家庭消费总预算的一定比例（凯恩斯主义需求刺激）
        """
        if self.government is None:
            logger.warning("[政府采购] Government 未初始化")
            return {"total_spent": 0.0, "by_industry": {}, "items_count": 0, "success": False}

        try:
            result = self.government.procure_goods_and_services(
                period=month,
                household_consumption_budget=household_consumption_budget,
                planned_demand_by_product=planned_demand_by_product,
            )
            return result or {"total_spent": 0.0, "by_industry": {}, "items_count": 0, "success": True}
        except Exception as e:
            logger.error(f"[政府采购] 执行失败: {e}")
            import traceback
            traceback.print_exc()
            return {"total_spent": 0.0, "by_industry": {}, "items_count": 0, "success": False, "error": str(e)}

    def _plan_government_procurement_demand(
        self,
        month: int,
        household_consumption_budget: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Plan government procurement as an ex-ante demand signal for production.

        This does not execute purchases or mutate inventory. Hard settlement
        still happens later in _execute_government_procurement.
        """
        empty = {
            "budget": 0.0,
            "total_planned_value": 0.0,
            "total_planned_qty": 0.0,
            "demand_by_product": {},
            "by_industry": {},
            "items_count": 0,
            "success": False,
        }
        if self.government is None:
            return {**empty, "error": "government not set"}

        try:
            result = self._call_actor(
                self.government,
                "plan_procurement_demand",
                period=month,
                household_consumption_budget=household_consumption_budget,
            )
        except Exception as e:
            logger.warning(f"[政府采购计划] 生成失败: {e}")
            return {**empty, "error": str(e)}

        if not isinstance(result, dict):
            return {**empty, "success": True}

        demand_by_product = self._merge_product_demands(result.get("demand_by_product", {}))
        normalized = dict(result)
        normalized["demand_by_product"] = demand_by_product
        normalized["budget"] = float(normalized.get("budget", 0.0) or 0.0)
        normalized["total_planned_value"] = float(normalized.get("total_planned_value", 0.0) or 0.0)
        normalized["total_planned_qty"] = float(normalized.get("total_planned_qty", 0.0) or 0.0)
        normalized["items_count"] = int(normalized.get("items_count", 0) or 0)
        normalized.setdefault("by_industry", {})
        normalized.setdefault("success", True)
        return normalized

    def _execute_orders(
        self,
        orders_by_household: List[Tuple[Household, List[Dict[str, Any]], float]],
        snapshot_cache: Dict[str, Optional[Dict[str, Any]]],
        month: int,
        record_transactions: bool
    ) -> Dict[str, Any]:
        consumption_by_household: Dict[str, Dict[str, float]] = defaultdict(lambda: {"qty": 0.0, "value": 0.0})
        consumption_by_sku: Dict[str, Dict[str, float]] = defaultdict(lambda: {"qty": 0.0, "value": 0.0})
        revenue_by_firm: Dict[str, Dict[str, float]] = defaultdict(lambda: {"qty": 0.0, "value": 0.0})
        price_index_stats = {"base_value": 0.0, "current_value": 0.0, "index": None}
        # 商品消费剩余预算（限制商品消费不超过预算，为服务消费预留资金）
        remaining_goods_budget: Dict[str, float] = {}
        # 实际账户余额（用于检查是否真的有钱）
        actual_balance_by_household: Dict[str, float] = {}
        tax_multiplier = 1.0
        if record_transactions and self.economic_center is not None:
            tax_multiplier = 1.0 + float(getattr(self.config, "vat_rate", 0.0) or 0.0)
            # 初始化商品预算和实际余额
            for hh, _, goods_budget in orders_by_household:
                bal = self._call_actor(self.economic_center, "query_balance", hh.household_id)
                actual_balance = float(bal or 0.0)
                actual_balance_by_household[hh.household_id] = actual_balance
                # 商品消费上限 = min(商品预算, 实际余额)，确保为服务消费预留
                remaining_goods_budget[hh.household_id] = min(goods_budget, actual_balance) if goods_budget > 0 else actual_balance
        for hh, orders, goods_budget in orders_by_household:
            for order in orders:
                product_id = order.get("product_id")
                if not product_id:
                    continue
                snapshot = self._get_product_snapshot_cached(product_id, snapshot_cache)
                if not snapshot:
                    continue
                desired_qty = int(order.get("desired_qty") or 0)
                if desired_qty <= 0:
                    continue
                unit_price = float(order.get("unit_price") or snapshot.get("retail_price") or 0.0)
                if unit_price <= 0:
                    continue
                seller = self._resolve_seller_firm(order)
                if seller is None:
                    continue
                available = float(
                    self._call_actor(
                        self.product_market,
                        "get_seller_stock",
                        product_id,
                        seller.firm_id,
                        bool(order.get("retailer_code")),
                    ) or 0.0
                )
                qty = min(desired_qty, int(available))
                if qty <= 0:
                    if record_transactions and self.economic_center is not None:
                        try:
                            self._call_actor(
                                self.economic_center,
                                "record_unmet_demand",
                                month,
                                hh.household_id,
                                seller.firm_id,
                                product_id,
                                order.get("product_name") or "",
                                desired_qty,
                                available,
                            )
                        except Exception:
                            pass
                    continue
                # 使用商品预算限制消费，而不是总余额
                if remaining_goods_budget:
                    remaining = float(remaining_goods_budget.get(hh.household_id, 0.0) or 0.0)
                    # 同时检查实际余额
                    actual_bal = float(actual_balance_by_household.get(hh.household_id, 0.0) or 0.0)
                    effective_limit = min(remaining, actual_bal)
                    max_affordable = int(effective_limit // (unit_price * tax_multiplier)) if tax_multiplier > 0 else 0
                    if max_affordable <= 0:
                        continue
                    qty = min(qty, max_affordable)
                    if qty <= 0:
                        continue
                stock_result = self._call_actor(
                    self.product_market,
                    "purchase_from_seller_stock",
                    product_id,
                    seller.firm_id,
                    qty,
                    bool(order.get("retailer_code")),
                ) or {}
                actual_qty = int(float(stock_result.get("actual_quantity", 0.0) or 0.0))
                if actual_qty <= 0:
                    continue
                qty = min(qty, actual_qty)
                amount = float(qty) * unit_price
                if record_transactions and self.economic_center is not None and desired_qty > available:
                    try:
                        self._call_actor(
                            self.economic_center,
                            "record_unmet_demand",
                            month,
                            hh.household_id,
                            seller.firm_id,
                            product_id,
                            order.get("product_name") or "",
                            desired_qty,
                            available,
                        )
                    except Exception:
                        pass
                if record_transactions and self.economic_center is not None:
                    base_unit_price = float(
                        order.get("base_manufacturer_price")
                        or snapshot.get("base_manufacturer_price")
                        or snapshot.get("manufacturer_price")
                        or unit_price
                        or 0.0
                    )
                    try:
                        tx_id = self._call_actor(
                            self.economic_center,
                            "process_purchase",
                            month,
                            hh.household_id,
                            seller.firm_id,
                            amount,
                            qty,
                            product_id,
                            order.get("product_name"),
                            unit_price,
                            base_unit_price,
                        )
                    except Exception as exc:
                        logger.warning(
                            f"[执行购买] 结算异常，回滚库存: buyer={hh.household_id} "
                            f"seller={seller.firm_id} product={product_id} qty={qty} error={exc}"
                        )
                        tx_id = None
                    if tx_id:
                        # 更新 snapshot_cache 中的库存，防止后续家庭超卖
                        if product_id in snapshot_cache and snapshot_cache[product_id]:
                            old_stock = int(snapshot_cache[product_id].get("available_stock") or 0)
                            if stock_result.get("source") == "manufacturer":
                                snapshot_cache[product_id]["available_stock"] = max(0, old_stock - qty)
                        stats = consumption_by_household[hh.household_id]
                        stats["qty"] += float(qty)
                        stats["value"] += amount
                        stats = consumption_by_sku[product_id]
                        stats["qty"] += float(qty)
                        stats["value"] += amount
                        stats = revenue_by_firm[seller.firm_id]
                        stats["qty"] += float(qty)
                        stats["value"] += amount
                        base_price = float(snapshot.get("base_retail_price") or unit_price or 0.0)
                        price_index_stats["base_value"] += base_price * float(qty)
                        price_index_stats["current_value"] += amount
                        # 更新商品预算剩余和实际余额
                        spent = amount * tax_multiplier
                        if remaining_goods_budget:
                            remaining_goods_budget[hh.household_id] = max(0.0, float(
                                remaining_goods_budget.get(hh.household_id, 0.0) - spent
                            ))
                        if actual_balance_by_household:
                            actual_balance_by_household[hh.household_id] = max(0.0, float(
                                actual_balance_by_household.get(hh.household_id, 0.0) - spent
                            ))
                    else:
                        self._restore_reserved_stock(
                            stock_result=stock_result,
                            product_id=product_id,
                            quantity=qty,
                            seller_id=seller.firm_id,
                            snapshot_cache=snapshot_cache,
                        )
                else:
                    # 更新 snapshot_cache 中的库存，防止后续家庭超卖
                    if product_id in snapshot_cache and snapshot_cache[product_id]:
                        old_stock = int(snapshot_cache[product_id].get("available_stock") or 0)
                        if stock_result.get("source") == "manufacturer":
                            snapshot_cache[product_id]["available_stock"] = max(0, old_stock - qty)
                    seller.cash += amount
                    stats = consumption_by_household[hh.household_id]
                    stats["qty"] += float(qty)
                    stats["value"] += amount
                    stats = consumption_by_sku[product_id]
                    stats["qty"] += float(qty)
                    stats["value"] += amount
                    stats = revenue_by_firm[seller.firm_id]
                    stats["qty"] += float(qty)
                    stats["value"] += amount
                    base_price = float(snapshot.get("base_retail_price") or unit_price or 0.0)
                    price_index_stats["base_value"] += base_price * float(qty)
                    price_index_stats["current_value"] += amount
        if self._debug_enabled():
            self._log_consumption_summary(consumption_by_household, revenue_by_firm)
        
        # 计算价格指数
        # 如果有固定消费篮子，使用固定篮子的权重计算 Laspeyres 价格指数
        if self._fixed_consumption_basket:
            basket_base_value = 0.0
            basket_current_value = 0.0
            for sku_id, basket_info in self._fixed_consumption_basket.items():
                base_price = float(basket_info.get("base_price", 0) or 0)
                fixed_qty = float(basket_info.get("qty", 0) or 0)
                
                # 获取当前价格
                current_price = base_price  # 默认使用基准价格
                if sku_id in snapshot_cache and snapshot_cache[sku_id]:
                    current_price = float(
                        snapshot_cache[sku_id].get("retail_price") or 
                        snapshot_cache[sku_id].get("base_retail_price") or 
                        base_price
                    )
                else:
                    # 尝试获取当前产品价格
                    try:
                        snapshot = self._get_product_snapshot_cached(sku_id, snapshot_cache)
                        if snapshot:
                            current_price = float(snapshot.get("retail_price") or base_price)
                    except Exception:
                        pass
                
                basket_base_value += base_price * fixed_qty
                basket_current_value += current_price * fixed_qty
            
            if basket_base_value > 0:
                price_index_stats["base_value"] = basket_base_value
                price_index_stats["current_value"] = basket_current_value
                price_index_stats["index"] = basket_current_value / basket_base_value
                price_index_stats["method"] = "laspeyres_fixed_basket"
            else:
                # 固定篮子无效，使用当期消费加权
                base_value = float(price_index_stats.get("base_value") or 0.0)
                current_value = float(price_index_stats.get("current_value") or 0.0)
                if base_value > 0:
                    price_index_stats["index"] = current_value / base_value
                price_index_stats["method"] = "current_weighted"
        else:
            # 预热阶段：使用当期消费加权
            base_value = float(price_index_stats.get("base_value") or 0.0)
            current_value = float(price_index_stats.get("current_value") or 0.0)
            if base_value > 0:
                price_index_stats["index"] = current_value / base_value
            price_index_stats["method"] = "current_weighted"
        
        total_qty = float(sum(stats.get("qty", 0.0) for stats in consumption_by_household.values()))
        total_value = float(sum(stats.get("value", 0.0) for stats in consumption_by_household.values()))
        household_values = [stats.get("value", 0.0) for stats in consumption_by_household.values()]
        return {
            "total_qty": total_qty,
            "total_value": total_value,
            "by_firm": {str(k): {"qty": v["qty"], "value": v["value"]} for k, v in revenue_by_firm.items()},
            "by_household": {str(k): {"qty": v["qty"], "value": v["value"]} for k, v in consumption_by_household.items()},
            "by_sku": {str(k): {"qty": v["qty"], "value": v["value"]} for k, v in consumption_by_sku.items()},
            "household_stats": self._calc_distribution_stats(household_values),
            "price_index": price_index_stats,
        }

    async def _process_layoffs(
        self,
        month: int,
        production_stats: Optional[Dict[str, Any]] = None,
        service_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        处理企业裁员（在发工资前执行）

        逻辑：
        1. 根据当月预计产量（来自上月销售或传入的 production_stats）计算工资帽
        2. 获取当前工资支出
        3. 如果当前工资支出 > 工资帽，则裁员至工资帽附近（但不低于工资帽）

        工资帽是产能下限，裁员后工资支出不能低于工资帽，以保证产能。

        Args:
            month: 当前经济月份
            production_stats: 本月生产统计 {by_firm: {firm_id: {qty, value}}}（预热阶段传入）
            service_stats: 本月服务消费统计 {by_industry: {industry_code: amount}}（预热阶段传入）

        Returns:
            裁员统计
        """
        if self.labor_market is None:
            return {"total_layoffs": 0, "total_saved": 0.0, "by_firm": {}}

        production_by_firm = (production_stats or {}).get("by_firm", {})
        service_by_industry = (service_stats or {}).get("by_industry", {})
        demand_by_mfg = (self._last_demand_stats or {}).get("by_mfg_firm", {})
        demand_by_retail = (self._last_demand_stats or {}).get("by_retail_firm", {})
        last_production_by_firm = getattr(self, "_last_production_value_by_firm", {}) or {}
        last_service_by_industry = getattr(self, "_last_service_value_by_industry", {}) or {}
        backlog_by_firm = getattr(self, "_last_production_gap_value_by_firm", {}) or {}
        wage_cap_tolerance = max(
            0.0,
            float(getattr(self.config, "firm_layoff_wage_cap_tolerance", 0.25) or 0.0),
        )

        total_layoffs = 0
        total_saved = 0.0
        layoffs_by_firm = {}

        for firm in (self.firms or []):
            expected_revenue = 0.0

            # 优先使用传入的当月生产/服务数据（预热阶段）
            if production_by_firm or service_by_industry:
                firm_production = production_by_firm.get(firm.firm_id, {})
                production_value = float(firm_production.get("value", 0.0) or 0.0)
                service_income = float(service_by_industry.get(firm.industry, 0.0) or 0.0)
                expected_revenue = production_value + service_income
            else:
                # 正式模拟：查询上月收入作为当月预计产量
                if month > 1 and self.economic_center is not None:
                    stats = self._call_actor(
                        self.economic_center,
                        "query_firm_monthly_financials",
                        firm_id=firm.firm_id,
                        month=month - 1,
                    )
                    if isinstance(stats, dict):
                        expected_revenue = float(stats.get("monthly_income", 0.0) or 0.0)

            # Do not force layoffs solely because realized sales lagged demand.
            # Production firms can have valid orders but low same-month sales when
            # inventory, labor, or procurement constraints delay fulfillment.
            for demand_source in (demand_by_mfg, demand_by_retail):
                firm_demand = demand_source.get(firm.firm_id, {}) if isinstance(demand_source, dict) else {}
                if not isinstance(firm_demand, dict):
                    continue
                try:
                    demand_value = max(0.0, float(firm_demand.get("value", 0.0) or 0.0))
                except (TypeError, ValueError):
                    demand_value = 0.0
                expected_revenue = max(expected_revenue, demand_value)
            try:
                last_production_value = max(0.0, float(last_production_by_firm.get(str(firm.firm_id), 0.0) or 0.0))
            except (TypeError, ValueError):
                last_production_value = 0.0
            expected_revenue = max(expected_revenue, last_production_value)
            try:
                last_service_value = max(0.0, float(last_service_by_industry.get(str(getattr(firm, "industry", "") or ""), 0.0) or 0.0))
            except (TypeError, ValueError):
                last_service_value = 0.0
            expected_revenue = max(expected_revenue, last_service_value)
            try:
                backlog_value = max(0.0, float(backlog_by_firm.get(str(firm.firm_id), 0.0) or 0.0))
            except (TypeError, ValueError):
                backlog_value = 0.0
            expected_revenue = max(expected_revenue, backlog_value)

            # 计算新工资帽（产能下限）
            compensation_ratio = float(getattr(firm, "compensation_ratio", 0.2) or 0.2)
            new_wage_cap = max(0.0, expected_revenue) * compensation_ratio

            min_wage_cap = max(0.0, float(getattr(self.config, "firm_layoff_min_wage_cap", 0.0) or 0.0))
            if min_wage_cap > 0.0:
                new_wage_cap = max(new_wage_cap, min_wage_cap)

            min_employees_to_keep = max(
                0,
                int(getattr(self.config, "firm_layoff_min_employees_to_keep", 0) or 0),
            )
            current_employees = int(getattr(firm, "employee_count", 0) or 0)
            if min_employees_to_keep > 0 and current_employees <= min_employees_to_keep:
                continue  # 跳过裁员，保持最低员工数
            wage_info = self._call_actor(self.labor_market, "get_firm_wage_bill", firm.firm_id) or {}
            current_wage_bill = max(0.0, float(wage_info.get("total_wage", 0.0) or 0.0))
            if current_wage_bill <= 0.0:
                continue
            tolerated_wage_cap = new_wage_cap * (1.0 + wage_cap_tolerance)
            if current_wage_bill <= tolerated_wage_cap:
                continue

            # 执行裁员
            result = self._call_actor(
                self.labor_market,
                "layoff_to_budget",
                firm_id=firm.firm_id,
                target_wage_cap=new_wage_cap,
                reason="budget_reduction",
                month=month,
                strategy="highest_wage",
            )

            if result and result.get("layoffs"):
                layoff_count = len(result["layoffs"])
                saved = result.get("saved_wage", 0.0)

                # 更新企业员工数
                firm.employee_count = max(0, firm.employee_count - layoff_count)

                layoffs_by_firm[firm.firm_id] = {
                    "count": layoff_count,
                    "saved": saved,
                    "new_wage_bill": result.get("new_wage_bill", 0.0),
                    "wage_cap": new_wage_cap,
                    "expected_revenue": expected_revenue,
                }
                total_layoffs += layoff_count
                total_saved += saved

        # 政府公共就业：当失业率低于目标时，逐步缩减公共就业岗位
        # 让劳动力回流到私人部门，避免政府工资永久膨胀
        if self.government is not None and self.labor_market is not None:
            gov_id = self.government.government_id
            labor_stats = self._call_actor(self.labor_market, "get_labor_stats") or {}
            total_labor = float(labor_stats.get("total_labor_hours", 0) or 0)
            total_matched = float(labor_stats.get("total_matched_jobs", 0) or 0)
            if total_labor > 0:
                unemployment_rate = (total_labor - total_matched) / total_labor
                # 当失业率低于目标的一半时，开始缩减公共就业
                target_unemployment = float(
                    getattr(self.government, "public_employment_target_unemployment", 0.15) or 0.0
                )
                shrink_threshold_multiplier = float(
                    getattr(self.government, "public_employment_shrink_threshold_multiplier", 0.5) or 0.0
                )
                max_shrink_ratio = float(
                    getattr(self.government, "public_employment_max_monthly_shrink_ratio", 0.20) or 0.0
                )
                shrink_threshold = target_unemployment * shrink_threshold_multiplier
                if unemployment_rate < shrink_threshold:
                    # 获取当前政府工资支出
                    gov_wage_info = self._call_actor(
                        self.labor_market, "get_firm_wage_bill", gov_id
                    ) or {}
                    current_gov_wage = float(gov_wage_info.get("total_wage", 0.0) or 0.0)
                    if current_gov_wage > 0:
                        # 缩减比例：失业率越低，缩减越多（最多缩减20%的工资支出）
                        shrink_ratio = 0.0
                        if shrink_threshold > 0.0 and max_shrink_ratio > 0.0:
                            shrink_ratio = min(
                                max_shrink_ratio,
                                (shrink_threshold - unemployment_rate) / shrink_threshold * max_shrink_ratio,
                            )
                        target_wage_cap = current_gov_wage * (1.0 - shrink_ratio)
                        result = self._call_actor(
                            self.labor_market,
                            "layoff_to_budget",
                            firm_id=gov_id,
                            target_wage_cap=target_wage_cap,
                            reason="public_employment_shrink",
                            month=month,
                            strategy="lowest_wage",
                        )
                        if result and result.get("layoffs"):
                            gov_layoffs = len(result["layoffs"])
                            gov_saved = result.get("saved_wage", 0.0)
                            self.government.employee_count = max(
                                0, int(getattr(self.government, "employee_count", 0) or 0) - gov_layoffs
                            )
                            total_layoffs += gov_layoffs
                            total_saved += gov_saved
                            logger.info(
                                f"[政府] 公共就业缩减: 裁减{gov_layoffs}人, "
                                f"节省工资${gov_saved:,.2f} (失业率={unemployment_rate:.1%})"
                            )
                else:
                    logger.debug(f"[政府] 公共就业岗位维持 (失业率={unemployment_rate:.1%})")

        if total_layoffs > 0:
            logger.info(
                f"[裁员汇总] 月份{month}: 共裁员{total_layoffs}人, "
                f"节省工资${total_saved:.2f}, 涉及{len(layoffs_by_firm)}家企业"
            )

        return {
            "total_layoffs": total_layoffs,
            "total_saved": total_saved,
            "by_firm": layoffs_by_firm,
        }

    async def _post_jobs(
        self,
        month: int,
        production_stats: Optional[Dict[str, Any]] = None,
        service_stats: Optional[Dict[str, Any]] = None,
        demand_stats: Optional[Dict[str, Any]] = None,
        skip_firm_ids: Optional[Set[str]] = None,
    ) -> None:
        """
        企业发布岗位
        
        Args:
            month: 当前月份
            production_stats: 本月生产统计 {by_firm: {firm_id: {qty, value}}}
            service_stats: 本月服务消费统计 {by_industry: {industry_code: amount}}
            demand_stats: 本月需求统计 {by_mfg_firm: {firm_id: {qty, value}}, by_retail_firm: ...}
            skip_firm_ids: 本月刚裁员的私人企业，跳过同月新增招聘
        """
        # 从 demand_stats 提取各制造商的需求价值作为劳动预算基础
        # 需求价值更能反映企业的真实经营状况，而不仅仅是实际生产量
        if demand_stats is None and self._last_demand_stats:
            demand_stats = self._last_demand_stats
        demand_by_mfg = (demand_stats or {}).get("by_mfg_firm", {}) if demand_stats else {}
        demand_by_retail = (demand_stats or {}).get("by_retail_firm", {}) if demand_stats else {}
        production_by_firm = (production_stats or {}).get("by_firm", {}) if production_stats else {}
        service_by_industry = (service_stats or {}).get("by_industry", {}) if service_stats else {}
        backlog_share = max(0.0, float(getattr(self.config, "firm_labor_backlog_demand_share", 0.5) or 0.0))
        backlog_demand_by_firm: Dict[str, float] = {}
        for firm_id, value in (getattr(self, "_last_production_gap_value_by_firm", {}) or {}).items():
            try:
                gap_value = max(0.0, float(value or 0.0))
            except (TypeError, ValueError):
                continue
            if gap_value > 0.0:
                backlog_demand_by_firm[str(firm_id)] = gap_value * backlog_share
        
        # 打印需求数据统计
        if demand_by_mfg or demand_by_retail:
            total_demand = sum(float(v.get("value", 0.0) or 0.0) for v in demand_by_mfg.values())
            total_retail_demand = sum(float(v.get("value", 0.0) or 0.0) for v in demand_by_retail.values())
            logger.info(
                f"[岗位发布] 使用需求数据: 制造商数={len(demand_by_mfg)}, "
                f"零售商数={len(demand_by_retail)}, "
                f"总需求=${total_demand + total_retail_demand:,.2f}"
            )
        priority_values: Dict[str, float] = {}
        for demand_source in (demand_by_mfg, demand_by_retail):
            for firm_id, stats in demand_source.items():
                if not isinstance(stats, dict):
                    continue
                value = max(0.0, float(stats.get("value", 0.0) or 0.0))
                priority_values[str(firm_id)] = max(priority_values.get(str(firm_id), 0.0), value)
        for firm_id, backlog_value in backlog_demand_by_firm.items():
            priority_values[firm_id] = max(priority_values.get(firm_id, 0.0), backlog_value)
        max_priority_value = max(priority_values.values(), default=0.0)
        self._firm_labor_priority = (
            {firm_id: value / max_priority_value for firm_id, value in priority_values.items()}
            if max_priority_value > 0.0
            else {}
        )
        
        # 企业发布岗位，传入本月需求价值
        # 注意：只传制造商/零售商的需求数据，不传服务企业的资源流水
        # 服务企业的"收入"是中间消耗流水（远大于GDP），不应作为劳动预算基础
        # 服务企业会回退到 _compute_labor_budget 内部的上月实际收入逻辑
        tasks = []
        task_firms: List[Firm] = []
        defaulted_firm_ids = self._get_defaulted_firm_ids()
        skipped_firm_ids = {str(fid) for fid in (skip_firm_ids or set())}
        service_firms_with_income = []  # 记录有服务收入的企业（仅用于日志）
        for firm in (self.firms or []):
            if str(firm.firm_id) in defaulted_firm_ids:
                if self.labor_market is not None:
                    self._call_actor(self.labor_market, "close_firm_positions", firm.firm_id, "credit_default")
                continue
            # 获取该企业本月的需求价值（优先使用需求数据，其次使用生产数据）
            demand_value = 0.0
            
            # 对于制造商/零售商，优先使用对应需求数据
            firm_demand = demand_by_mfg.get(firm.firm_id, {})
            if not firm_demand:
                firm_demand = demand_by_retail.get(firm.firm_id, {})
            demand_value = float(firm_demand.get("value", 0.0) or 0.0)
            
            # 如果没有需求数据，回退到生产数据
            if demand_value <= 0:
                firm_production = production_by_firm.get(firm.firm_id, {})
                demand_value = float(firm_production.get("value", 0.0) or 0.0)

            demand_value = max(demand_value, backlog_demand_by_firm.get(str(firm.firm_id), 0.0))
            
            # 服务企业用当期家庭服务消费作为招聘信号。服务交易已经在
            # AbstractResourceMarket 中转化为企业收入，这里只用于岗位预算，
            # 避免服务就业总是滞后一月。
            service_income = float(service_by_industry.get(firm.industry, 0.0) or 0.0)
            if service_income > 0:
                service_firms_with_income.append((firm.firm_id, firm.industry, service_income))
                demand_value = max(demand_value, service_income)
            # 政府采购需求（上月计划，按行业）叠加进劳动预算，使企业为政府需求招人备产。
            # 政府需求是家庭需求之外的额外需求，故加总而非取 max。
            gov_demand = float(getattr(self, "_last_gov_demand_by_industry", {}).get(str(firm.industry), 0.0) or 0.0)
            if gov_demand > 0:
                demand_value = demand_value + gov_demand
            # 劳动需求平滑(EMA)：企业按平滑后的需求招人（劳动调整有粘性），抑制
            # demand→hiring→production→demand 的周期-2 蛛网震荡，得到持续性周期。
            _ls_alpha = float(getattr(self.config, "labor_demand_smoothing", 1.0) or 1.0)
            if _ls_alpha < 1.0:
                if not hasattr(self, "_firm_labor_demand_ema"):
                    self._firm_labor_demand_ema = {}
                _fid = str(firm.firm_id)
                _prev = self._firm_labor_demand_ema.get(_fid)
                if _prev is not None:
                    demand_value = _ls_alpha * demand_value + (1.0 - _ls_alpha) * _prev
                self._firm_labor_demand_ema[_fid] = demand_value
            if str(firm.firm_id) in skipped_firm_ids and demand_value <= 0.0:
                if self.labor_market is not None:
                    self._call_actor(self.labor_market, "close_firm_positions", firm.firm_id, "layoff_cooldown")
                continue
            tasks.append(
                firm.post_jobs(
                    period=month,
                    current_demand_value=demand_value,
                    min_part_time_hours_per_month=float(
                        getattr(self.config, "firm_min_part_time_hours_per_month", 20.0) or 20.0
                    ),
                    max_startup_part_time_hours_per_month=float(
                        getattr(self.config, "firm_max_startup_part_time_hours_per_month", 160.0) or 160.0
                    ),
                    min_job_budget_coverage=float(
                        getattr(self.config, "firm_min_job_budget_coverage", 1.0) or 0.0
                    ),
                    allow_cash_based_startup_hiring=bool(
                        getattr(self.config, "firm_allow_cash_based_startup_hiring", False)
                    ),
                    use_llm=bool(getattr(self.config, "firm_job_posting_use_llm", False)),
                    wage_bidding_enabled=bool(
                        getattr(self.config, "firm_wage_bidding_enabled", False)
                    ),
                    wage_bid_up=float(getattr(self.config, "firm_wage_bid_up", 0.04) or 0.0),
                    wage_bid_down=float(getattr(self.config, "firm_wage_bid_down", 0.02) or 0.0),
                    wage_premium_min=float(getattr(self.config, "firm_wage_premium_min", 0.5) or 0.0),
                    wage_premium_max=float(getattr(self.config, "firm_wage_premium_max", 2.5) or 1.0),
                )
            )
            task_firms.append(firm)
        
        # 打印服务收入统计
        if service_firms_with_income:
            total_service_income = sum(inc for _, _, inc in service_firms_with_income)
            logger.info(f"[岗位发布] 服务企业收入统计: 企业数={len(service_firms_with_income)}, 总收入=${total_service_income:.2f}")
            for firm_id, industry, income in service_firms_with_income[:5]:
                logger.info(f"  - {firm_id} ({industry}): ${income:.2f}")
            if len(service_firms_with_income) > 5:
                logger.info(f"  ... 还有 {len(service_firms_with_income) - 5} 个服务企业")
        
        postings_by_firm: Dict[str, List[Job]] = {}
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for firm, res in zip(task_firms, results):
                if isinstance(res, Exception):
                    logger.error(f"[岗位发布] {firm.firm_id} 发布失败: {res}")
                    continue
                jobs: List[Job] = []
                if isinstance(res, list):
                    jobs = [j for j in res if isinstance(j, Job)]
                postings_by_firm[firm.firm_id] = jobs
        
        # 政府发布岗位
        if self.government:
            try:
                gov_jobs = await self.government.post_jobs(period=month)
                if gov_jobs:
                    postings_by_firm[self.government.government_id] = gov_jobs
                    logger.info(f"[岗位发布] 政府发布 {len(gov_jobs)} 个职位类型")
            except Exception as e:
                logger.error(f"[岗位发布] 政府发布失败: {e}")
        
        if self._debug_enabled():
            self._log_job_postings(postings_by_firm)

    async def _match_jobs(self, month: int, use_llm: bool = False) -> None:
        if self.labor_market is None:
            return
        self._call_actor(self.labor_market, "reset_matching_state")
        jobs = self._prioritize_labor_jobs(self._call_actor(self.labor_market, "query_opening_jobs") or [])
        labor_match_top_k = max(1, int(getattr(self.config, "labor_match_top_k", 8) or 8))
        raw_match_limit = max(labor_match_top_k, len(jobs))
        for hh in self.households or []:
            seekers = hh.list_job_seekers(hh.labor_hours)
            for labor_hour in seekers:
                matches = hh.match_jobs_topk_by_loss(labor_hour=labor_hour, jobs=jobs, top_k=raw_match_limit)
                matches = self._prioritize_labor_matches(matches)[:labor_match_top_k]
                applications = await hh.decide_job_applications(
                    month=month,
                    labor_hour=labor_hour,
                    top_matches=matches,
                    use_llm=use_llm,
                )
                for app in applications:
                    self._call_actor(self.labor_market, "submit_application", app, labor_hour)

        offer_backups = max(0, int(getattr(self.config, "labor_match_offer_backups", 3) or 0))
        self._call_actor(self.labor_market, "make_offers", month, max_backups=offer_backups, reset_existing=True)
        acceptance_policy = str(
            getattr(self.config, "labor_offer_acceptance_policy", "best_loss") or "best_loss"
        )
        self._call_actor(self.labor_market, "resolve_offers", month, acceptance_policy=acceptance_policy)
        self._refresh_household_employment_status()
        self._refresh_firm_employee_count()
        if self._debug_enabled():
            matched = self._call_actor(self.labor_market, "get_matched_jobs") or []
            self._log_matching_summary(matched)

    def _refresh_firm_employee_count(self) -> None:
        """
        从 LaborMarket 同步企业员工数
        """
        if self.labor_market is None:
            return

        matched = self._call_actor(self.labor_market, "get_matched_jobs") or []

        # 统计每个企业的员工数
        employee_count_by_firm: Dict[str, int] = defaultdict(int)
        employee_list_by_firm: Dict[str, List[LaborHour]] = defaultdict(list)
        for rec in matched:
            firm_id = rec.get("firm_id")
            if firm_id:
                employee_count_by_firm[firm_id] += 1
                labor_hour = LaborHour.create(
                    agent_id=str(rec.get("household_id") or ""),
                    total_hours=max(1.0, float(rec.get("hours_per_period") or 160.0)),
                    template=f"matched_{rec.get('lh_type') or 'head'}",
                    skill_profile={},
                    ability_profile={},
                    lh_type=rec.get("lh_type") if rec.get("lh_type") in {"head", "spouse"} else "head",
                )
                labor_hour.is_valid = False
                labor_hour.firm_id = firm_id
                labor_hour.job_SOC = rec.get("soc")
                labor_hour.job_title = rec.get("title")
                labor_hour.wage_per_hour = float(rec.get("wage_per_hour") or 0.0)
                employee_list_by_firm[firm_id].append(labor_hour)

        # 更新企业员工数
        for firm in (self.firms or []):
            firm.employee_count = employee_count_by_firm.get(firm.firm_id, 0)
            firm.employee_list = employee_list_by_firm.get(firm.firm_id, [])

        # 更新政府员工数
        if self.government:
            self.government.employee_count = employee_count_by_firm.get(self.government.government_id, 0)
            self.government.employee_list = employee_list_by_firm.get(self.government.government_id, [])

    def _refresh_household_employment_status(self) -> None:
        if self.labor_market is None:
            return
        snapshot = self._call_actor(self.labor_market, "get_labor_status_snapshot")
        if not isinstance(snapshot, dict):
            return
        for hh in self.households or []:
            entry = snapshot.get(hh.household_id, {})
            for lh in hh.labor_hours or []:
                status = entry.get(getattr(lh, "lh_type", ""))
                if status is None:
                    continue
                if status.get("public_employment"):
                    lh.is_valid = True
                    lh.firm_id = None
                    lh.job_SOC = None
                    lh.job_title = None
                elif status.get("employed"):
                    lh.is_valid = False
                    lh.firm_id = status.get("firm_id")
                    lh.job_SOC = status.get("job_SOC")
                    lh.job_title = status.get("job_title")
                else:
                    lh.is_valid = True
                    lh.firm_id = None
                    lh.job_SOC = None
                    lh.job_title = None

            head = entry.get("head")
            spouse = entry.get("spouse")
            if head is not None:
                code = (
                    hh._NOT_EMPLOYED_CODE
                    if head.get("public_employment")
                    else hh._EMPLOYED_CODE if head.get("employed") else hh._NOT_EMPLOYED_CODE
                )
                hh.ER82433 = code
                hh.csv_values["ER82433"] = code
            if spouse is not None:
                code = (
                    hh._NOT_EMPLOYED_CODE
                    if spouse.get("public_employment")
                    else hh._EMPLOYED_CODE if spouse.get("employed") else hh._NOT_EMPLOYED_CODE
                )
                hh.SP_employment_status = code
                hh.csv_values["SP_employment_status"] = code

    def _get_defaulted_firm_ids(self) -> set[str]:
        if self.economic_center is None:
            return set()
        try:
            snapshot = self._call_actor(self.economic_center, "get_all_firm_credit_defaulted") or {}
        except Exception:
            return set()
        if not isinstance(snapshot, dict):
            return set()
        return {str(firm_id) for firm_id, defaulted in snapshot.items() if bool(defaulted)}

    def _pay_wages(self, month: int, record_transactions: bool) -> Dict[str, Any]:
        if self.labor_market is None:
            return {}
        matched = self._call_actor(self.labor_market, "get_matched_jobs") or []
        wage_by_firm: Dict[str, float] = defaultdict(float)
        wage_by_household: Dict[str, float] = defaultdict(float)
        for rec in matched:
            household_id = rec.get("household_id")
            hh = self._household_by_id.get(household_id)
            if hh is None:
                continue
            firm_id = rec.get("firm_id")
            if not firm_id:
                continue
            wage_per_hour = float(rec.get("wage_per_hour") or 0.0)
            if wage_per_hour <= 0:
                continue
            hours_per_period = rec.get("hours_per_period")
            if hours_per_period is None:
                hours = 40.0
                ppm = 4.0
            else:
                hours = float(hours_per_period or 0.0)
                ppm = 1.0
            gross = wage_per_hour * hours * ppm
            paid = True
            if record_transactions and self.economic_center is not None:
                tx_id = self._call_actor(
                    self.economic_center,
                    "process_wage",
                    month,
                    wage_per_hour,
                    household_id,
                    firm_id,
                    hours,
                    ppm,
                )
                paid = bool(tx_id)
            else:
                # 企业或政府支付工资（非交易记录模式）
                firm = self._firm_by_id.get(firm_id)
                if firm is not None:
                    firm.cash -= gross
                elif self.government and firm_id == self.government.government_id:
                    # 政府支付工资时，通过 EconomicCenter 扣减余额
                    if self.economic_center:
                        self._call_actor(
                            self.economic_center,
                            "transfer",
                            self.government.government_id,
                            household_id,
                            gross,
                        )
            if not paid:
                continue

            if rec.get("lh_type") == "spouse":
                hh.update_sp_income(gross)
            else:
                hh.update_rp_income(gross)
            wage_by_firm[firm_id] += float(gross)
            wage_by_household[household_id] += float(gross)
        if self._debug_enabled():
            self._log_wage_summary(wage_by_firm, wage_by_household)
        household_wages = list(wage_by_household.values())
        return {
            "total": float(sum(wage_by_firm.values())),
            "by_firm": {str(k): float(v) for k, v in wage_by_firm.items()},
            "household_stats": self._calc_distribution_stats(household_wages),
        }

    def _record_month_summary(
        self,
        month: int,
        econ_month: int,
        preheat: bool,
        demand_stats: Optional[Dict[str, Any]],
        production_stats: Optional[Dict[str, Any]],
        consumption_stats: Optional[Dict[str, Any]],
        wage_stats: Optional[Dict[str, Any]],
        procurement_stats: Optional[Dict[str, Any]] = None,
        service_consumption_stats: Optional[Dict[str, Any]] = None,
        government_procurement_stats: Optional[Dict[str, Any]] = None,
        government_procurement_plan_stats: Optional[Dict[str, Any]] = None,
        firm_credit_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._recording_enabled():
            return
        labor_summary = self._call_actor(self.labor_market, "summary") or {}
        firm_financials = {}
        gdp_stats = {}
        gdp_comprehensive = {}
        tax_stats = {}
        household_summary = {}
        redistribution_per_person = 0.0
        if self.economic_center is not None:
            firm_financials = self._call_actor(self.economic_center, "query_all_firms_monthly_financials", econ_month)
            gdp_stats = self._call_actor(self.economic_center, "calculate_monthly_gdp", econ_month, production_stats)
            # 使用新的综合 GDP 计算，传入 Simulator 的价格指数以正确计算 real_gdp
            gdp_comprehensive = self._call_actor(
                self.economic_center, "calculate_gdp_comprehensive", econ_month, production_stats, 0,
                external_price_index_100=self._last_price_index
            )
            # 缓存 GDP 结果用于增长率计算和月末报告打印
            if gdp_comprehensive:
                self._call_actor(self.economic_center, "cache_gdp_result", econ_month, gdp_comprehensive)
                # 缓存完整结果用于打印
                self._last_gdp_comprehensive = gdp_comprehensive
            tax_stats = self._call_actor(self.economic_center, "get_monthly_tax_collection", econ_month)
            household_summary = self._call_actor(self.economic_center, "summarize_households_monthly", econ_month)
            redistribution_per_person = self._call_actor(
                self.economic_center, "query_redistribution_record_per_person", econ_month
            )

        price_index = {}
        if consumption_stats:
            price_index = dict(consumption_stats.get("price_index") or {})
        
        # price_index["index"] 是 current_value / base_value 的比率
        # 转换为基于 100 的价格指数（CPI 风格）
        current_ratio = price_index.get("index")
        current_index_100 = float(current_ratio) * 100.0 if current_ratio is not None else None
        
        # 如果没有消费数据（如 Phase 0），初始化价格指数为 100.0
        if current_index_100 is None and self._last_price_index is None:
            self._last_price_index = 100.0  # 基准价格指数
            current_index_100 = 100.0
        
        inflation_rate = None
        if current_index_100 is not None:
            prev = self._last_price_index
            if prev is not None and prev > 0:
                # 通胀率 = (当前指数 - 上期指数) / 上期指数
                inflation_rate = (current_index_100 - prev) / prev
            # 存储基于 100 的价格指数
            self._last_price_index = current_index_100
        
        # 存储通胀率供下次使用
        if inflation_rate is not None:
            self._last_inflation_rate = inflation_rate
        price_index["inflation_rate"] = inflation_rate
        price_index["index_100"] = current_index_100  # 添加基于 100 的指数

        household_agg = (household_summary or {}).get("aggregate", {}) or {}
        household_by = (household_summary or {}).get("by_household", {}) or {}
        balances = [rec.get("balance") for rec in household_by.values()]
        assets_stats = self._calc_distribution_stats([float(v or 0.0) for v in balances])
        assets_stats["gini"] = self._calc_gini([float(v or 0.0) for v in balances])
        # 收入基尼系数（比财富基尼更能反映经济周期的分配效应）
        incomes = [float((rec.get("income", {}) or {}).get("total", 0.0) or 0.0) for rec in household_by.values()]
        assets_stats["income_gini"] = self._calc_gini(incomes)

        # 使用实际注册的劳动力数量，数据校验：剔除负值和极端值
        total_labor = max(0.0, float(labor_summary.get("total_labor_hours", 0.0) or 0.0))
        employed_labor = max(0.0, float(labor_summary.get("total_matched_jobs", 0.0) or 0.0))
        employed_labor = min(employed_labor, total_labor)  # 就业人数不能超过总劳动力
        employment_rate = employed_labor / total_labor if total_labor > 0 else 0.0
        employment_rate = max(0.0, min(1.0, employment_rate))  # 限制在 [0, 1]
        net_wage_total = float((household_agg.get("income", {}) or {}).get("wage", 0.0) or 0.0)
        labor_tax_total = float((tax_stats or {}).get("labor_tax", 0.0) or 0.0)
        gross_wage_total = net_wage_total + labor_tax_total
        average_wage = gross_wage_total / employed_labor if employed_labor > 0 else 0.0

        household_consumption = (household_agg.get("consumption", {}) or {})
        household_income = (household_agg.get("income", {}) or {})

        production_stats = production_stats or {}
        firm_output = production_stats.get("by_firm", {}) if isinstance(production_stats, dict) else {}
        firm_profit_pressure = self._summarize_firm_profit_pressure(
            production_stats=production_stats,
            wage_stats=wage_stats,
            firm_financials=firm_financials,
        )
        accounting_invariants = (
            self._run_accounting_invariants(econ_month)
            if bool(getattr(self.config, "enable_accounting_invariant_checks", True))
            else {}
        )

        payload = {
            "preheat": preheat,
            "month": month,
            "econ_month": econ_month,
            "timestamp": datetime.utcnow().isoformat(),
            "population": {
                "households": len(self.households or []),
                "firms": len(self.firms or []),
            },
            "household": {
                "aggregate": {
                    "income": {
                        "wage": float(household_income.get("wage", 0.0) or 0.0),
                        "interest": float(household_income.get("interest", 0.0) or 0.0),
                        "redistribution": float(household_income.get("redistribution", 0.0) or 0.0),
                        "total": float(household_income.get("total", 0.0) or 0.0),
                    },
                    "consumption": {
                        "purchase": float(household_consumption.get("purchase", 0.0) or 0.0),
                        "tax": float(household_consumption.get("tax", 0.0) or 0.0),
                        "total": float(household_consumption.get("total", 0.0) or 0.0),
                    },
                    "assets_distribution": {
                        "max": float(assets_stats.get("max", 0.0) or 0.0),
                        "min": float(assets_stats.get("min", 0.0) or 0.0),
                        "mean": float(assets_stats.get("mean", 0.0) or 0.0),
                        "median": float(assets_stats.get("median", 0.0) or 0.0),
                        "gini": float(assets_stats.get("gini", 0.0) or 0.0),
                        "income_gini": float(assets_stats.get("income_gini", 0.0) or 0.0),
                        "count": int(assets_stats.get("count", 0) or 0),
                    },
                    # 消费类别分布
                    "consumption_by_category": self._calc_consumption_category_distribution(
                        consumption_stats, service_consumption_stats
                    ),
                },
                "by_household": household_by,
            },
            "labor_market": {
                "total_labor": total_labor,
                "employed_labor": employed_labor,
                "employment_rate": employment_rate,
                "unemployment_rate": float(labor_summary.get("unemployment_rate", 0.0) or 0.0),
                "total_job_positions": int(labor_summary.get("total_job_positions", 0) or 0),
                "total_matched_jobs": int(labor_summary.get("total_matched_jobs", 0) or 0),
                "job_fill_rate": float(labor_summary.get("job_fill_rate", 0.0) or 0.0),
                "vacancy_rate": (lambda pos, mat: (pos - mat) / pos if pos > 0 else 0.0)(
                    int(labor_summary.get("total_job_positions", 0) or 0),
                    int(labor_summary.get("total_matched_jobs", 0) or 0),
                ),
                "total_wage_gross": gross_wage_total,
                "total_wage_net": net_wage_total,
                "average_wage": average_wage,
            },
            "product_market": {
                "household_purchase_total": float(household_consumption.get("purchase", 0.0) or 0.0),
                "government_procurement_total": float(
                    (household_summary or {}).get("government_procurement_total", 0.0) or 0.0
                ),
                "total_output": float(production_stats.get("total_value", 0.0) or 0.0),
                "firm_output": firm_output or {},
            },
            "government": {
                "total_tax": float((tax_stats or {}).get("total_tax", 0.0) or 0.0),
                "personal_income_tax": float((tax_stats or {}).get("labor_tax", 0.0) or 0.0),
                "consume_tax": float((tax_stats or {}).get("consume_tax", 0.0) or 0.0),
                "corporate_tax": float((tax_stats or {}).get("corporate_tax", 0.0) or 0.0),
                "redistribution_total": float(household_income.get("redistribution", 0.0) or 0.0),
                "redistribution_per_person": float(redistribution_per_person or 0.0),
                # 政府采购支出
                "procurement_spending": float(
                    (government_procurement_stats or {}).get("total_spent", 0.0) or 0.0
                ),
                "procurement_items": int(
                    (government_procurement_stats or {}).get("items_count", 0) or 0
                ),
                "procurement_planned_budget": float(
                    (government_procurement_plan_stats or {}).get("budget", 0.0) or 0.0
                ),
                "procurement_planned_value": float(
                    (government_procurement_plan_stats or {}).get("total_planned_value", 0.0) or 0.0
                ),
                "procurement_planned_qty": float(
                    (government_procurement_plan_stats or {}).get("total_planned_qty", 0.0) or 0.0
                ),
                "procurement_planned_items": int(
                    (government_procurement_plan_stats or {}).get("items_count", 0) or 0
                ),
            },
            "macro": {
                # 旧版 GDP 统计（保持兼容）
                "gdp": gdp_stats or {},
                "gdp_value": float(((gdp_stats or {}).get("gdp", {}) or {}).get("production_approach", 0.0) or 0.0),
                # 新版综合 GDP 统计
                "gdp_comprehensive": gdp_comprehensive or {},
                "nominal_gdp": float((gdp_comprehensive or {}).get("nominal_gdp", 0.0) or 0.0),
                "real_gdp": float((gdp_comprehensive or {}).get("real_gdp", 0.0) or 0.0),
                "gdp_growth_rate": (gdp_comprehensive or {}).get("growth_rates", {}).get("nominal_gdp_growth"),
                "real_gdp_growth_rate": (gdp_comprehensive or {}).get("growth_rates", {}).get("real_gdp_growth"),
                # 价格指标
                "price_index": price_index,
                "gdp_deflator": float((gdp_comprehensive or {}).get("deflator", 1.0) or 1.0),
                "inflation_rate": inflation_rate,
                # 关键比率
                "consumption_rate": float((gdp_comprehensive or {}).get("ratios", {}).get("consumption_rate", 0.0) or 0.0),
                "investment_rate": float((gdp_comprehensive or {}).get("ratios", {}).get("investment_rate", 0.0) or 0.0),
                "government_rate": float((gdp_comprehensive or {}).get("ratios", {}).get("government_rate", 0.0) or 0.0),
                "labor_share": float((gdp_comprehensive or {}).get("ratios", {}).get("labor_share", 0.0) or 0.0),
                "wage_scale": float(getattr(self, "_wage_level", 1.0) or 1.0),
            },
            "details": {
                "labor_market_raw": labor_summary,
                "demand": demand_stats or {},
                "production": production_stats or {},
                "procurement": procurement_stats or {},
                "consumption": consumption_stats or {},
                "service_consumption": service_consumption_stats or {},
                "government_procurement_plan": government_procurement_plan_stats or {},
                "wages": wage_stats or {},
                "firm_credit": firm_credit_stats or {},
                "firm_initialization_calibration": getattr(self, "_last_firm_initialization_calibration", {}) or {},
                "initial_inventory_calibration": getattr(self, "_last_initial_inventory_calibration", {}) or {},
                "firm_financials": firm_financials or {},
                "firm_profit_pressure": firm_profit_pressure,
                "tax": tax_stats or {},
                "accounting_invariants": accounting_invariants,
            },
        }
        self._write_month_record(month, payload, preheat=preheat)

    def _summarize_firm_profit_pressure(
        self,
        production_stats: Optional[Dict[str, Any]],
        wage_stats: Optional[Dict[str, Any]],
        firm_financials: Optional[Dict[str, Dict[str, float]]],
    ) -> Dict[str, Any]:
        """
        Diagnostic-only decomposition of why firm cashflow is under pressure.

        This deliberately does not alter accounting. It compares realized
        income, wage expense, produced output value, and production cost so
        smoke runs can distinguish macro demand leakage from production/payroll
        implementation issues.
        """
        production_stats = production_stats or {}
        wage_stats = wage_stats or {}
        firm_financials = firm_financials or {}

        output_by_firm = production_stats.get("firm_production_value") or {}
        if not output_by_firm:
            output_by_firm = {
                str(fid): float((stats or {}).get("value", 0.0) or 0.0)
                for fid, stats in (production_stats.get("by_firm", {}) or {}).items()
            }
        production_cost_by_firm = production_stats.get("firm_production_cost") or {}
        wage_by_firm = wage_stats.get("by_firm") or {}

        firm_ids = set(str(fid) for fid in firm_financials.keys())
        firm_ids.update(str(fid) for fid in output_by_firm.keys())
        firm_ids.update(str(fid) for fid in production_cost_by_firm.keys())
        firm_ids.update(
            str(fid)
            for fid in wage_by_firm.keys()
            if str(fid) in firm_financials or str(fid) in self._firm_by_id
        )

        by_firm: Dict[str, Dict[str, Any]] = {}
        totals = {
            "realized_income": 0.0,
            "realized_expenses": 0.0,
            "realized_profit": 0.0,
            "wage_expense": 0.0,
            "production_output_value": 0.0,
            "production_cost": 0.0,
            "sales_gap_to_wages": 0.0,
            "sales_gap_to_wages_and_production_cost": 0.0,
            "inventory_adjusted_gap_to_wages": 0.0,
            "inventory_adjusted_gap_to_wages_and_production_cost": 0.0,
            "output_gap_to_production_cost": 0.0,
        }

        for firm_id in sorted(firm_ids):
            financial = firm_financials.get(firm_id, {}) or {}
            realized_income = float(financial.get("monthly_income", 0.0) or 0.0)
            realized_expenses = float(financial.get("monthly_expenses", 0.0) or 0.0)
            realized_profit = float(
                financial.get("monthly_profit", realized_income - realized_expenses) or 0.0
            )
            wage_expense = float(wage_by_firm.get(firm_id, 0.0) or 0.0)
            output_value = float(output_by_firm.get(firm_id, 0.0) or 0.0)
            production_cost = float(production_cost_by_firm.get(firm_id, 0.0) or 0.0)

            sales_gap_to_wages = max(0.0, wage_expense - realized_income)
            sales_gap_to_wages_and_cost = max(0.0, wage_expense + production_cost - realized_income)
            inventory_adjusted_income = realized_income + output_value
            inventory_adjusted_gap_to_wages = max(0.0, wage_expense - inventory_adjusted_income)
            inventory_adjusted_gap_to_wages_and_cost = max(
                0.0,
                wage_expense + production_cost - inventory_adjusted_income,
            )
            output_gap_to_cost = max(0.0, production_cost - output_value)

            row = {
                "realized_income": realized_income,
                "realized_expenses": realized_expenses,
                "realized_profit": realized_profit,
                "wage_expense": wage_expense,
                "production_output_value": output_value,
                "production_cost": production_cost,
                "inventory_adjusted_income": inventory_adjusted_income,
                "income_to_wage_ratio": self._safe_ratio(realized_income, wage_expense),
                "inventory_adjusted_income_to_wage_ratio": self._safe_ratio(
                    inventory_adjusted_income,
                    wage_expense,
                ),
                "wage_to_income_ratio": self._safe_ratio(wage_expense, realized_income),
                "production_cost_to_output_ratio": self._safe_ratio(production_cost, output_value),
                "sales_gap_to_wages": sales_gap_to_wages,
                "sales_gap_to_wages_and_production_cost": sales_gap_to_wages_and_cost,
                "inventory_adjusted_gap_to_wages": inventory_adjusted_gap_to_wages,
                "inventory_adjusted_gap_to_wages_and_production_cost": inventory_adjusted_gap_to_wages_and_cost,
                "output_gap_to_production_cost": output_gap_to_cost,
            }
            by_firm[firm_id] = row

            totals["realized_income"] += realized_income
            totals["realized_expenses"] += realized_expenses
            totals["realized_profit"] += realized_profit
            totals["wage_expense"] += wage_expense
            totals["production_output_value"] += output_value
            totals["production_cost"] += production_cost
            totals["sales_gap_to_wages"] += sales_gap_to_wages
            totals["sales_gap_to_wages_and_production_cost"] += sales_gap_to_wages_and_cost
            totals["inventory_adjusted_gap_to_wages"] += inventory_adjusted_gap_to_wages
            totals["inventory_adjusted_gap_to_wages_and_production_cost"] += (
                inventory_adjusted_gap_to_wages_and_cost
            )
            totals["output_gap_to_production_cost"] += output_gap_to_cost

        aggregate = dict(totals)
        aggregate.update(
            {
                "firm_count": len(by_firm),
                "firms_income_below_wages": sum(
                    1 for row in by_firm.values() if row["sales_gap_to_wages"] > 0.0
                ),
                "firms_income_below_wages_and_production_cost": sum(
                    1
                    for row in by_firm.values()
                    if row["sales_gap_to_wages_and_production_cost"] > 0.0
                ),
                "firms_inventory_adjusted_income_below_wages": sum(
                    1 for row in by_firm.values() if row["inventory_adjusted_gap_to_wages"] > 0.0
                ),
                "firms_inventory_adjusted_income_below_wages_and_production_cost": sum(
                    1
                    for row in by_firm.values()
                    if row["inventory_adjusted_gap_to_wages_and_production_cost"] > 0.0
                ),
                "firms_production_cost_above_output": sum(
                    1 for row in by_firm.values() if row["output_gap_to_production_cost"] > 0.0
                ),
                "income_to_wage_ratio": self._safe_ratio(
                    totals["realized_income"],
                    totals["wage_expense"],
                ),
                "inventory_adjusted_income_to_wage_ratio": self._safe_ratio(
                    totals["realized_income"] + totals["production_output_value"],
                    totals["wage_expense"],
                ),
                "wage_to_income_ratio": self._safe_ratio(
                    totals["wage_expense"],
                    totals["realized_income"],
                ),
                "production_cost_to_output_ratio": self._safe_ratio(
                    totals["production_cost"],
                    totals["production_output_value"],
                ),
            }
        )

        return {
            "aggregate": aggregate,
            "by_firm": by_firm,
            "top_sales_gap_to_wages": self._top_pressure_rows(
                by_firm,
                "sales_gap_to_wages",
            ),
            "top_inventory_adjusted_gap_to_wages": self._top_pressure_rows(
                by_firm,
                "inventory_adjusted_gap_to_wages",
            ),
            "top_output_gap_to_production_cost": self._top_pressure_rows(
                by_firm,
                "output_gap_to_production_cost",
            ),
        }

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
        denominator = float(denominator or 0.0)
        if denominator <= 0.0:
            return None
        return float(numerator or 0.0) / denominator

    @staticmethod
    def _top_pressure_rows(
        by_firm: Dict[str, Dict[str, Any]],
        field_name: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        rows = [
            {"firm_id": firm_id, field_name: float(row.get(field_name, 0.0) or 0.0)}
            for firm_id, row in by_firm.items()
            if float(row.get(field_name, 0.0) or 0.0) > 0.0
        ]
        rows.sort(key=lambda row: row[field_name], reverse=True)
        return rows[:limit]

    def _run_accounting_invariants(self, econ_month: int) -> Dict[str, Any]:
        """Collect lightweight actor snapshots and run accounting invariant checks."""
        try:
            balances = {}
            transactions = []
            firm_debt_balances = {}
            if self.economic_center is not None:
                balances = self._call_actor(self.economic_center, "get_all_balances") or {}
                transactions = self._call_actor(self.economic_center, "get_transactions", econ_month) or []
                firm_debt_balances = self._call_actor(self.economic_center, "get_all_firm_debt_balances") or {}

            products = []
            if self.product_market is not None:
                products = self._call_actor(self.product_market, "get_all_products_snapshot") or []
            product_market_snapshot = {
                "products_by_id": {
                    str(product.get("product_id")): product
                    for product in products
                    if isinstance(product, dict) and product.get("product_id")
                }
            }

            result = check_accounting_invariants(
                ledger={str(agent_id): {"amount": amount} for agent_id, amount in (balances or {}).items()},
                transactions=transactions,
                firm_ids=[firm.firm_id for firm in (self.firms or [])],
                household_ids=[household.household_id for household in (self.households or [])],
                government_ids=(
                    [self.government.government_id]
                    if self.government is not None and getattr(self.government, "government_id", None)
                    else []
                ),
                bank_ids=(
                    [self.bank.bank_id]
                    if self.bank is not None and getattr(self.bank, "bank_id", None)
                    else []
                ),
                loan_balances=firm_debt_balances,
                product_market=product_market_snapshot,
                month=econ_month,
            )
            data = result.to_dict()

            # 货币守恒诊断（SFC 闭合）：补齐银行双分录后，所有账户余额之和应逐月守恒。
            # 首次检查月自动设为基线，后续月份偏移 > 容差即提示可能存在单边记账泄漏。
            try:
                total_cash = float(sum(float(v or 0.0) for v in (balances or {}).values()))
                if getattr(self, "_conservation_baseline_cash", None) is None:
                    self._conservation_baseline_cash = total_cash
                    self._conservation_baseline_month = econ_month
                baseline = float(self._conservation_baseline_cash or 0.0)
                drift = total_cash - baseline
                tol = max(abs(baseline) * 1e-6, 1e-3)
                data["cash_conservation"] = {
                    "baseline_month": getattr(self, "_conservation_baseline_month", econ_month),
                    "baseline_total_cash": baseline,
                    "total_cash": total_cash,
                    "drift": drift,
                    "ok": bool(abs(drift) <= tol),
                }
                if abs(drift) > tol:
                    logger.warning(
                        "[货币守恒] econ_month=%s 总现金=%.4f 基线=%.4f 偏移=%.4f（疑似单边记账泄漏）",
                        econ_month, total_cash, baseline, drift,
                    )
            except Exception as _cons_exc:
                logger.debug(f"[货币守恒] 诊断跳过: {_cons_exc}")

            if not result.ok:
                logger.warning(
                    "[会计检查] errors=%s warnings=%s metrics=%s",
                    len(result.errors),
                    len(result.warnings),
                    result.metrics,
                )
            elif result.warnings:
                logger.info(
                    "[会计检查] warnings=%s metrics=%s",
                    len(result.warnings),
                    result.metrics,
                )
            return data
        except Exception as exc:
            logger.warning(f"[会计检查] 运行失败: {exc}")
            return {
                "ok": False,
                "errors": [
                    {
                        "code": "accounting_invariant_check_failed",
                        "message": str(exc),
                        "subject": None,
                        "severity": "error",
                        "details": {},
                    }
                ],
                "warnings": [],
                "metrics": {},
            }

    def _update_last_sales(self, econ_month: int, consumption_stats: Optional[Dict[str, Any]] = None) -> None:
        sales_by_product: Dict[str, float] = {}
        if self.economic_center is not None:
            try:
                stats = self._call_actor(self.economic_center, "collect_sales_statistics", econ_month)
                if isinstance(stats, dict):
                    for _, rec in stats.items():
                        if not isinstance(rec, dict):
                            continue
                        product_id = rec.get("product_id")
                        if not product_id:
                            continue
                        qty = float(rec.get("quantity_sold", 0.0) or 0.0)
                        sales_by_product[str(product_id)] = sales_by_product.get(str(product_id), 0.0) + qty
            except Exception:
                sales_by_product = {}

        if not sales_by_product and consumption_stats:
            by_sku = consumption_stats.get("by_sku", {}) if isinstance(consumption_stats, dict) else {}
            for pid, rec in (by_sku or {}).items():
                if not isinstance(rec, dict):
                    continue
                qty = float(rec.get("qty", 0.0) or 0.0)
                sales_by_product[str(pid)] = sales_by_product.get(str(pid), 0.0) + qty

        self._last_sales_by_product = sales_by_product

    def _update_layoff_support_memory(
        self,
        production_stats: Optional[Dict[str, Any]] = None,
        service_consumption_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        production_by_firm = (production_stats or {}).get("by_firm", {}) if production_stats else {}
        self._last_production_value_by_firm = {}
        for firm_id, stats in (production_by_firm or {}).items():
            if not isinstance(stats, dict):
                continue
            try:
                value = max(0.0, float(stats.get("value", 0.0) or 0.0))
            except (TypeError, ValueError):
                continue
            if value > 0.0:
                self._last_production_value_by_firm[str(firm_id)] = value

        service_by_industry = (
            (service_consumption_stats or {}).get("by_industry", {})
            if service_consumption_stats
            else {}
        )
        self._last_service_value_by_industry = {}
        for industry, value in (service_by_industry or {}).items():
            try:
                numeric = max(0.0, float(value or 0.0))
            except (TypeError, ValueError):
                continue
            if numeric > 0.0:
                self._last_service_value_by_industry[str(industry)] = numeric

    def _build_production_demand_signal(self, current_demand_by_product: Dict[str, float]) -> Dict[str, float]:
        demand_signal: Dict[str, float] = {}
        # 需求信号只取"真实"来源：本月家庭需求、历史实际销售、未满足缺口。
        # 不再纳入 _last_planned_demand_by_product(上月计划需求)——计划需求含上月的
        # target-inventory 加成，若用 max 累积会形成棘轮：计划高→产出高→记为计划→下月
        # 仍高，产出脱离真实需求(实测制造商月产为真实需求的4倍，存货虚增、GDP 支出法 I 虚高)。
        sources = (
            current_demand_by_product or {},
            self._last_sales_by_product or {},
            self._last_unmet_demand_by_product or {},
        )
        for source in sources:
            for product_id, qty in source.items():
                if not product_id:
                    continue
                try:
                    value = max(0.0, float(qty or 0.0))
                except (TypeError, ValueError):
                    continue
                if value > 0.0:
                    key = str(product_id)
                    demand_signal[key] = max(demand_signal.get(key, 0.0), value)
        return demand_signal

    def _merge_product_demands(self, *sources: Optional[Dict[str, Any]]) -> Dict[str, float]:
        merged: Dict[str, float] = defaultdict(float)
        for source in sources:
            if not isinstance(source, dict):
                continue
            for product_id, qty in source.items():
                if not product_id:
                    continue
                try:
                    value = max(0.0, float(qty or 0.0))
                except (TypeError, ValueError):
                    continue
                if value > 0.0:
                    merged[str(product_id)] += value
        return dict(merged)

    def _update_demand_memory(
        self,
        demand_by_product: Dict[str, float],
        econ_month: int,
    ) -> None:
        planned: Dict[str, float] = {}
        for product_id, qty in (demand_by_product or {}).items():
            if not product_id:
                continue
            try:
                value = max(0.0, float(qty or 0.0))
            except (TypeError, ValueError):
                continue
            if value > 0.0:
                planned[str(product_id)] = value
        self._last_planned_demand_by_product = planned

        unmet_by_product: Dict[str, float] = defaultdict(float)
        if self.economic_center is not None:
            try:
                unmet_records = self._call_actor(self.economic_center, "query_unmet_demand", econ_month)
            except Exception:
                unmet_records = {}
            if isinstance(unmet_records, dict):
                for key, rec in unmet_records.items():
                    if not isinstance(rec, dict):
                        continue
                    product_id = rec.get("product_id")
                    if not product_id and key:
                        product_id = str(key).split("@", 1)[0]
                    if not product_id:
                        continue
                    qty_short = max(0.0, float(rec.get("qty_short", 0.0) or 0.0))
                    if qty_short > 0.0:
                        unmet_by_product[str(product_id)] += qty_short
        self._last_unmet_demand_by_product = dict(unmet_by_product)

    def _record_demand_to_market(
        self,
        demand_by_product: Dict[str, float],
        snapshot_cache: Dict[str, Optional[Dict[str, Any]]]
    ) -> None:
        """
        将需求数据记录到商品市场，用于供需比计算
        """
        if not self.product_market:
            return
        
        # 按制造商行业汇总需求
        demand_by_industry: Dict[str, float] = defaultdict(float)
        for product_id, qty in (demand_by_product or {}).items():
            snapshot = self._get_product_snapshot_cached(product_id, snapshot_cache)
            if not snapshot:
                continue
            mfg_code = snapshot.get("manufacturer_code")
            if mfg_code:
                demand_by_industry[mfg_code] += float(qty)
        
        # 记录到商品市场
        for mfg_code, total_demand in demand_by_industry.items():
            self._call_actor(self.product_market, "record_demand", mfg_code, total_demand)
        
        if self._debug_enabled():
            logger.info(f"[供需追踪] 记录需求: {len(demand_by_industry)} 个行业")

    def _record_supply_and_adjust_prices(
        self,
        production_stats: Dict[str, Any],
        econ_month: int = 0
    ) -> None:
        """
        记录供给数据并根据供需比调整价格
        包括：
        1. 消费品行业（根据家庭需求和生产供给，使用供需比）
        2. 原材料行业（根据企业采购需求变化，使用需求变化率）
        """
        if not self.product_market:
            return

        by_firm = production_stats.get("by_firm", {}) if isinstance(production_stats, dict) else {}
        
        # 调试：检查supply数据
        if self._debug_enabled() and len(by_firm) > 0:
            logger.info(f"[供需调试] 生产企业数={len(by_firm)}, 总生产量={production_stats.get('total_qty', 0):.0f}")

        # 记录供给量（消费品）。ProductMarket 的 products_by_industry 和
        # 供需追踪都使用产品的 manufacturer_code，因此需求、供给、调价必须
        # 统一使用行业代码，不能混用行业名称。
        supply_count = 0
        for firm_id, stats in by_firm.items():
            firm = self._firm_by_id.get(firm_id)
            if firm is None:
                continue
            mfg_code = getattr(firm, "industry", None)
            if not mfg_code:
                continue
            supply_qty = float(stats.get("qty", 0.0) or 0.0)
            if supply_qty > 0:
                self._call_actor(self.product_market, "record_supply", str(mfg_code), supply_qty)
                supply_count += 1

        if self._debug_enabled() and supply_count > 0:
            logger.info(f"[供需调试] 记录供给企业数={supply_count}")
        
        # 收集消费品行业（有生产的行业），使用 manufacturer_code 作为键
        consumer_goods_industries = set()
        for firm_id in by_firm.keys():
            firm = self._firm_by_id.get(firm_id)
            if firm is None:
                continue
            mfg_code = getattr(firm, "industry", None)
            if mfg_code:
                consumer_goods_industries.add(str(mfg_code))

        # 收集原材料行业（有采购需求的行业，但不是消费品生产行业）
        raw_material_industries = set()
        raw_material_demand = production_stats.get("raw_material_demand", {})
        for industry_code in raw_material_demand.keys():
            if industry_code and industry_code not in consumer_goods_industries:
                raw_material_industries.add(industry_code)

        # 1. 消费品行业：根据供需比调整价格
        # 收集所有有需求的行业（不仅是有生产的行业）
        all_industries_with_demand = set()
        supply_demand_data = self._call_actor(self.product_market, "get_all_supply_demand_stats")
        if supply_demand_data and isinstance(supply_demand_data, dict):
            all_industries_with_demand = set(supply_demand_data.keys())
            # 调试：显示供需数据
            if self._debug_enabled():
                sample_data = list(supply_demand_data.items())[:3]
                for code, stats in sample_data:
                    logger.info(f"[供需调试] {code}: demand={stats.get('demand',0):.1f}, supply={stats.get('supply',0):.1f}")
        
        # 合并有生产和有需求的行业
        industries_to_adjust = consumer_goods_industries | all_industries_with_demand
        
        adjusted_count = 0
        for mfg_code in industries_to_adjust:
            count = self._call_actor(
                self.product_market,
                "adjust_prices_by_supply_demand",
                mfg_code,
                0.05,  # base_adjustment
                0.15   # max_adjustment
            )
            if count and count > 0:
                adjusted_count += count
        
        if adjusted_count > 0:
            logger.info(f"[价格调整] 调整了 {len(industries_to_adjust)} 个行业的 {adjusted_count} 个产品价格")

        # 2. 原材料行业：根据需求变化率调整价格
        for industry_code in raw_material_industries:
            self._call_actor(
                self.product_market,
                "adjust_raw_material_prices",
                industry_code,
                0.03,  # base_adjustment（更保守）
                0.10   # max_adjustment
            )

        # 3. 结束当期原材料需求记录，为下一期做准备
        self._call_actor(self.product_market, "finalize_raw_material_demand")

        # 调整抽象资源（服务等）的价格
        if self.abstract_resource_market is not None:
            self.abstract_resource_market.adjust_prices(period=econ_month)

        if self._debug_enabled():
            logger.info(
                f"[供需追踪] 调整价格: 消费品={len(consumer_goods_industries)}个行业, "
                f"原材料={len(raw_material_industries)}个行业 + 抽象资源"
            )

    def _settle_corporate_tax(self, month: int) -> Dict[str, Any]:
        """
        征收企业所得税
        
        时机：工资发放后、生产前
        税基：当月收入 - 当月支出（不含企业税）
        """
        if self.economic_center is None:
            return {}
        
        try:
            result = self._call_actor(self.economic_center, "settle_monthly_corporate_tax", month)
            total_tax = sum(result.values()) if isinstance(result, dict) else 0.0
            
            if self._debug_enabled():
                logger.info(f"[企业所得税] 月份={month}, 总额={total_tax:.2f}, 企业数={len(result or {})}")
            
            return {
                "total_tax": total_tax,
                "by_firm": result or {},
                "firm_count": len(result or {})
            }
        except Exception as e:
            logger.error(f"企业所得税征收失败: {e}")
            return {}

    async def _pay_bank_interest(self, month: int) -> Dict[str, Any]:
        """
        银行发放存款利息
        
        直接根据家庭在经济中心的余额发放利息，年利率0.5%，按月计算
        """
        if self.bank is None:
            return {}
        
        try:
            # 获取所有家庭ID
            household_ids = [hh.household_id for hh in (self.households or [])]
            
            total_interest = await self.bank.calculate_and_pay_monthly_interest(
                month=month,
                household_ids=household_ids
            )
            
            return {
                "total_interest": total_interest,
                "month": month,
                "households_count": len(household_ids)
            }
        except Exception as e:
            logger.error(f"银行利息发放失败: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _settle_firm_credit(self, month: int) -> Dict[str, Any]:
        """
        企业信用月末结算：计息、自动还款、违约状态标记。
        """
        if self.economic_center is None:
            return {}
        try:
            result = self._call_actor(
                self.economic_center,
                "settle_firm_credit_month",
                month,
                float(getattr(self.config, "firm_credit_annual_interest_rate", 0.08) or 0.08),
                float(getattr(self.config, "firm_credit_repayment_cash_buffer", 1000.0) or 0.0),
                int(getattr(self.config, "firm_credit_default_distress_months", 3) or 3),
            )
            if self._debug_enabled() and result:
                logger.info(
                    "[企业信贷] 月份=%s, 利息=%.2f, 还款=%.2f, 违约企业=%s",
                    month,
                    float(result.get("interest_total", 0.0) or 0.0),
                    float(result.get("repayment_total", 0.0) or 0.0),
                    int(result.get("defaulted_count", 0) or 0),
                )
            return result or {}
        except Exception as e:
            logger.error(f"企业信贷结算失败: {e}")
            return {}

    def _apply_credit_default_labor_closure(
        self,
        month: int,
        firm_credit_stats: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Propagate hard credit default into the labor market.
        """
        if self.labor_market is None:
            return {"defaulted_firms": [], "total_layoffs": 0, "closed_positions": 0, "by_firm": {}}

        defaulted_firms = set()
        firms = (firm_credit_stats or {}).get("firms", {}) if isinstance(firm_credit_stats, dict) else {}
        if isinstance(firms, dict):
            for firm_id, stats in firms.items():
                if isinstance(stats, dict) and bool(stats.get("defaulted", False)):
                    defaulted_firms.add(str(firm_id))

        defaulted_firms.update(self._get_defaulted_firm_ids())

        if not defaulted_firms:
            return {"defaulted_firms": [], "total_layoffs": 0, "closed_positions": 0, "by_firm": {}}

        total_layoffs = 0
        total_saved = 0.0
        closed_positions = 0
        by_firm: Dict[str, Dict[str, Any]] = {}
        for firm_id in sorted(defaulted_firms):
            closed = self._call_actor(self.labor_market, "close_firm_positions", firm_id, "credit_default") or 0
            result = self._call_actor(
                self.labor_market,
                "layoff_to_budget",
                firm_id=firm_id,
                target_wage_cap=0.0,
                reason="credit_default",
                month=month,
                strategy="highest_wage",
            ) or {}
            layoffs = list(result.get("layoffs", []) or [])
            layoff_count = len(layoffs)
            saved = float(result.get("saved_wage", 0.0) or 0.0)
            total_layoffs += layoff_count
            total_saved += saved
            closed_positions += int(closed or 0)
            firm = self._firm_by_id.get(firm_id)
            if firm is not None:
                firm.employee_count = 0
                firm.employee_list = []
            by_firm[firm_id] = {
                "layoffs": layoff_count,
                "saved_wage": saved,
                "closed_positions": int(closed or 0),
            }

        self._refresh_household_employment_status()
        self._refresh_firm_employee_count()
        if total_layoffs > 0 or closed_positions > 0:
            logger.info(
                "[企业违约] labor closure month=%s firms=%s layoffs=%s closed_positions=%s saved_wage=%.2f",
                month,
                len(defaulted_firms),
                total_layoffs,
                closed_positions,
                total_saved,
            )

        return {
            "defaulted_firms": sorted(defaulted_firms),
            "total_layoffs": total_layoffs,
            "total_saved": total_saved,
            "closed_positions": closed_positions,
            "by_firm": by_firm,
        }

    async def _redistribute_taxes(self, month: int) -> Dict[str, Any]:
        """
        税收再分配
        
        将当月税收收入按人均分配给所有家庭
        """
        if self.economic_center is None:
            return {}
        
        try:
            # 获取所有家庭ID
            household_ids = [hh.household_id for hh in (self.households or [])]
            if not household_ids:
                return {}
            
            # 调用再分配方法
            result = await self.economic_center.redistribute_monthly_taxes.remote(
                month=month,
                strategy="equal",  # 人均平等分配
            )
            result = ray.get(result) if hasattr(result, '__ray_terminate__') else result
            
            if self._debug_enabled():
                total = float((result or {}).get("total_redistributed", 0.0))
                recipients = int((result or {}).get("recipients", 0))
                per_person = float((result or {}).get("per_person", 0.0))
                logger.info(f"[税收再分配] 月份={month}, 总额={total:.2f}, 人数={recipients}, 人均={per_person:.2f}")
            
            return result or {}
        except Exception as e:
            logger.error(f"税收再分配失败: {e}")
            return {}
    
    def _distribute_dividends(self, month: int, dividend_rate: float = 0.80) -> Dict[str, Any]:
        """
        企业分红：将税后利润的一部分分配给家庭
        
        在真实经济中，企业利润通过股息、租金收入等方式回流给家庭。
        特别是住房行业(HS)的毛利(V003=79.4%)本质是租金收入，应回流给家庭。
        
        分配权重按家庭财富(ER85692)占比，财富越多持有的"股份"越多。
        
        Args:
            month: 当前经济月份
            dividend_rate: 分红比率（税后利润的百分比），默认80%
            
        Returns:
            分红统计 {total_dividends, by_firm, recipients, per_household_avg}
        """
        if self.economic_center is None or not self.firms or not self.households:
            return {"total_dividends": 0.0, "by_firm": {}, "recipients": 0}
        
        # 1. 计算每个企业的可分红利润
        dividends_by_firm: Dict[str, float] = {}
        total_dividends = 0.0
        
        for firm in self.firms:
            # 查询企业本月财务（企业所得税已扣除后）
            stats = self._call_actor(
                self.economic_center,
                "query_firm_monthly_financials",
                firm_id=firm.firm_id,
                month=month,
            )
            if not isinstance(stats, dict):
                continue
            
            income = float(stats.get("monthly_income", 0.0) or 0.0)
            expenses = float(stats.get("monthly_expenses", 0.0) or 0.0)
            profit = income - expenses
            
            # 只有正利润才分红
            if profit <= 0:
                continue
            
            # 企业所得税已在 _settle_corporate_tax 中通过 record_firm_monthly_expense
            # 计入 expenses，因此此处 profit = income - expenses 已是税后净利润。
            # 不可再乘 (1 - corporate_tax_rate)，否则对同一笔利润二次扣税。
            after_tax_profit = profit
            
            dividend = after_tax_profit * dividend_rate
            if dividend <= 0.01:
                continue
            
            # 检查企业余额是否足够支付分红
            firm_balance = float(self._call_actor(
                self.economic_center, "query_balance", firm.firm_id
            ) or 0.0)
            
            # 预留运营资金：至少保留上月支出的 1.2 倍，确保下月能正常发工资和进货
            reserved = expenses * 1.2
            distributable = firm_balance - reserved
            if distributable <= 0:
                continue
            dividend = min(dividend, distributable)
            
            if dividend > 0.01:
                dividends_by_firm[firm.firm_id] = dividend
                total_dividends += dividend
        
        if total_dividends <= 0:
            return {"total_dividends": 0.0, "by_firm": {}, "recipients": 0}
        
        # 2. 按家庭财富权重分配
        # 使用 PSID ER85692 (Constructed Wealth Including Equity) 作为权重
        household_weights: Dict[str, float] = {}
        total_weight = 0.0
        for hh in self.households:
            wealth = 0.0
            try:
                wealth = float(hh.csv_values.get("ER85692") or 0.0)
            except (ValueError, TypeError, AttributeError):
                pass
            # 确保非负权重，最低给1.0（保证所有人都能分到一点）
            weight = max(wealth, 1.0)
            household_weights[hh.household_id] = weight
            total_weight += weight
        
        if total_weight <= 0:
            return {"total_dividends": 0.0, "by_firm": dividends_by_firm, "recipients": 0}
        
        # 3. 执行分红转账（通过 update_balance 批量操作）
        recipients = 0
        
        # 从企业账户扣除
        for firm_id, div_amount in dividends_by_firm.items():
            self._call_actor(self.economic_center, "update_balance", firm_id, -div_amount)
        
        # 按权重分配给家庭
        for hh in self.households:
            weight = household_weights.get(hh.household_id, 1.0)
            share = total_dividends * (weight / total_weight)
            if share > 0.01:
                self._call_actor(self.economic_center, "update_balance", hh.household_id, share)
                recipients += 1
        
        logger.info(
            f"[企业分红] 月份={month}, 总额=${total_dividends:,.2f}, "
            f"企业数={len(dividends_by_firm)}, 受益家庭={recipients}, "
            f"人均=${total_dividends/max(recipients,1):,.2f}"
        )
        
        return {
            "total_dividends": total_dividends,
            "by_firm": dividends_by_firm,
            "recipients": recipients,
            "per_household_avg": total_dividends / max(recipients, 1),
        }

    # =========================================================================
    # 调试日志辅助方法
    # =========================================================================
    
    def _log_wage_stats(self, wage_stats: Optional[Dict[str, Any]]) -> None:
        """打印工资发放统计"""
        if not wage_stats:
            logger.info("  ⚠️  工资统计为空")
            return
        total = wage_stats.get("total", 0.0)
        by_firm = wage_stats.get("by_firm", {})
        count = len(by_firm)
        household_stats = wage_stats.get("household_stats", {})
        avg = household_stats.get("mean", 0) if household_stats else (total / count if count > 0 else 0)
        
        logger.info(f"  💵 工资发放: 总额=${total:,.2f}, 雇主数={count}, 人均=${avg:,.2f}")
        
        # 按企业/政府分类
        if by_firm:
            top_employers = sorted(by_firm.items(), key=lambda x: x[1], reverse=True)[:5]
            for emp_id, amount in top_employers:
                logger.info(f"    - {emp_id}: ${amount:,.2f}")
    
    def _log_corporate_tax_stats(self, tax_stats: Optional[Dict[str, Any]]) -> None:
        """打印企业所得税统计"""
        if not tax_stats:
            logger.info("  ⚠️  企业所得税统计为空")
            return
        total = tax_stats.get("total_tax", 0.0)
        count = tax_stats.get("firm_count", 0)
        by_firm = tax_stats.get("by_firm", {})
        logger.info(f"  🏢 企业所得税: 总额=${total:,.2f}, 企业数={count}")
        
        # 打印缴税最多的前5个企业
        if by_firm:
            top_firms = sorted(by_firm.items(), key=lambda x: x[1], reverse=True)[:5]
            for firm_id, amount in top_firms:
                if amount > 0:
                    logger.info(f"    - {firm_id}: ${amount:,.2f}")
    
    def _log_consumption_plans(self, results: List[Tuple[Any, Dict[str, Any]]]) -> None:
        """打印消费计划统计（含宏观劳动力市场指标）"""
        if not results:
            logger.info("  ⚠️  消费计划为空")
            return
        
        # 获取劳动力市场统计
        labor_stats = {}
        if self.labor_market is not None:
            labor_stats = self._call_actor(self.labor_market, "summary") or {}
        
        # 数据校验：剔除负值和极端值
        total_labor = max(0, int(labor_stats.get("total_labor_hours", 0) or 0))
        total_matched = max(0, int(labor_stats.get("total_matched_jobs", 0) or 0))
        total_positions = max(0, int(labor_stats.get("total_job_positions", 0) or 0))
        total_matched = min(total_matched, total_labor)  # 就业人数不能超过总劳动力
        
        # 重新计算就业率/失业率（使用校验后的数据）
        employment_rate = total_matched / total_labor if total_labor > 0 else 0.0
        employment_rate = max(0.0, min(1.0, employment_rate))
        unemployment_rate = 1.0 - employment_rate
        
        # 岗位空缺率 = (总岗位 - 已匹配) / 总岗位
        vacancy_rate = (total_positions - total_matched) / total_positions if total_positions > 0 else 0.0
        vacancy_rate = max(0.0, min(1.0, vacancy_rate))
        
        # 打印宏观劳动力市场状况
        logger.info(f"  📊 宏观市场状况:")
        logger.info(f"      - 总劳动力: {total_labor} 人")
        logger.info(f"      - 就业率: {employment_rate*100:.1f}% ({total_matched}/{total_labor})")
        logger.info(f"      - 失业率: {unemployment_rate*100:.1f}%")
        logger.info(f"      - 岗位空缺率: {vacancy_rate*100:.1f}% ({total_positions - total_matched}/{total_positions})")
        
        total_budget = 0.0
        total_service = 0.0
        total_goods = 0.0

        for hh, plan in results:
            step0 = plan.get("step0", {})
            # total_budget 在 step0 内部
            budget = float(step0.get("total_budget") or 0.0) if isinstance(step0, dict) else 0.0
            total_budget += budget

            budgets = step0.get("budgets", {}) if isinstance(step0, dict) else {}
            for cat, alloc in budgets.items():
                # alloc 可能是 dict 或 float
                if isinstance(alloc, dict):
                    amount = alloc.get("budget", 0.0)
                elif isinstance(alloc, (int, float)):
                    amount = float(alloc) if alloc else 0.0
                else:
                    # 跳过非数值类型（如字符串）
                    continue

                if cat in HOUSEHOLD_SERVICE_CATEGORY_TO_INDUSTRY:
                    total_service += amount
                else:
                    total_goods += amount
        
        logger.info(f"  🛍️  消费计划: 家庭数={len(results)}, 总预算=${total_budget:,.2f}")
        logger.info(f"      - 商品消费预算: ${total_goods:,.2f}")
        logger.info(f"      - 服务消费预算: ${total_service:,.2f}")
    
    def _log_demand_stats(self, demand_stats: Optional[Dict[str, Any]]) -> None:
        """打印需求统计"""
        if not demand_stats:
            logger.info("  ⚠️  需求统计为空")
            return
        total_qty = demand_stats.get("total_qty", 0)
        total_value = demand_stats.get("total_value", 0.0)
        by_retail = demand_stats.get("by_retail_firm", {})
        by_mfg = demand_stats.get("by_mfg_firm", {})
        sku_count = len(by_retail) + len(by_mfg)
        logger.info(f"  📦 商品需求: 零售商数={len(by_retail)}, 制造商数={len(by_mfg)}, 总数量={total_qty:,.0f}, 总价值=${total_value:,.2f}")
    
    def _log_production_stats(self, production_stats: Optional[Dict[str, Any]]) -> None:
        """打印生产统计"""
        if not production_stats:
            logger.info("  ⚠️  生产统计为空")
            return
        total_qty = production_stats.get("total_qty", 0)
        total_value = production_stats.get("total_value", 0.0)
        by_firm = production_stats.get("by_firm", {})
        logger.info(f"  🏭 生产补货: 企业数={len(by_firm)}, 总数量={total_qty:,.0f}, 总价值=${total_value:,.2f}")
        
        # 打印前5个生产企业
        if by_firm:
            top_firms = sorted(by_firm.items(), key=lambda x: x[1].get("value", 0), reverse=True)[:5]
            for firm_id, stats in top_firms:
                logger.info(f"    - {firm_id}: 数量={stats.get('qty', 0):,.0f}, 价值=${stats.get('value', 0):,.2f}")
    
    def _log_procurement_stats(self, procurement_stats: Optional[Dict[str, Any]]) -> None:
        """打印零售商进货统计"""
        if not procurement_stats:
            logger.info("  ⚠️  进货统计为空")
            return
        total = procurement_stats.get("total_value", 0.0)
        count = len(procurement_stats.get("by_retailer", {}))
        logger.info(f"  📥 零售商进货: 零售商数={count}, 总成本=${total:,.2f}")
    
    def _log_consumption_stats(self, consumption_stats: Optional[Dict[str, Any]]) -> None:
        """打印家庭消费统计"""
        if not consumption_stats:
            logger.info("  ⚠️  消费统计为空")
            return
        total_qty = consumption_stats.get("total_qty", 0)
        total_value = consumption_stats.get("total_value", 0.0)
        household_count = len(consumption_stats.get("by_household", {}))
        logger.info(f"  🛒 家庭商品消费: 家庭数={household_count}, 总数量={total_qty:,.0f}, 总金额=${total_value:,.2f}")
    
    def _log_service_consumption_stats(self, stats: Optional[Dict[str, Any]]) -> None:
        """打印服务消费统计"""
        if not stats:
            logger.info("  ⚠️  服务消费统计为空")
            return
        total = stats.get("total_service_consumption", 0.0)
        household_count = stats.get("household_count", 0)
        by_category = stats.get("by_category", {})
        by_industry = stats.get("by_industry", {})
        
        logger.info(f"  🏠 家庭服务消费: 家庭数={household_count}, 总金额=${total:,.2f}")
        if by_category:
            logger.info(f"      按类别:")
            for cat, amount in sorted(by_category.items(), key=lambda x: -x[1]):
                logger.info(f"        - {cat}: ${amount:,.2f}")
        if by_industry:
            logger.info(f"      按行业:")
            for ind, amount in sorted(by_industry.items(), key=lambda x: -x[1])[:5]:
                logger.info(f"        - {ind}: ${amount:,.2f}")
    
    def _log_government_procurement_stats(self, stats: Optional[Dict[str, Any]]) -> None:
        """打印政府采购统计"""
        if not stats:
            logger.info("  ⚠️  政府采购统计为空")
            return
        
        success = stats.get("success", False)
        total_spent = stats.get("total_spent", 0.0)
        items_count = stats.get("items_count", 0)
        by_industry = stats.get("by_industry", {})
        error = stats.get("error")
        
        if error:
            logger.info(f"  ❌ 政府采购失败: {error}")
            return
        
        if not success or total_spent <= 0:
            logger.info(f"  ℹ️  政府采购: 本月无采购 (可能税收不足或无可采购商品)")
            return
        
        logger.info(f"  🏛️  政府采购: 总支出=${total_spent:,.2f}, 采购项数={items_count}, 涉及行业={len(by_industry)}")
        if by_industry:
            logger.info(f"      按行业:")
            for ind, amount in sorted(by_industry.items(), key=lambda x: -x[1])[:10]:
                pct = amount / total_spent * 100 if total_spent > 0 else 0
                logger.info(f"        - {ind}: ${amount:,.2f} ({pct:.1f}%)")

    def _log_government_procurement_plan_stats(self, stats: Optional[Dict[str, Any]]) -> None:
        """打印政府采购计划需求统计"""
        if not stats:
            logger.info("  ⚠️  政府采购计划为空")
            return

        error = stats.get("error")
        if error:
            logger.info(f"  ℹ️  政府采购计划不可用: {error}")
            return

        planned_value = float(stats.get("total_planned_value", 0.0) or 0.0)
        planned_qty = float(stats.get("total_planned_qty", 0.0) or 0.0)
        items_count = int(stats.get("items_count", 0) or 0)
        budget = float(stats.get("budget", 0.0) or 0.0)
        if planned_value <= 0.0 or items_count <= 0:
            logger.info(f"  ℹ️  政府采购计划: 无可映射商品需求 (预算=${budget:,.2f})")
            return

        logger.info(
            f"  🏛️  政府采购计划: 预算=${budget:,.2f}, "
            f"计划需求=${planned_value:,.2f}, 数量={planned_qty:,.0f}, SKU={items_count}"
        )
    
    def _log_redistribution_stats(self, stats: Optional[Dict[str, Any]]) -> None:
        """打印税收再分配统计"""
        if not stats:
            logger.info("  ⚠️  税收再分配统计为空")
            return
        total = stats.get("total_redistributed", 0.0)
        recipients = stats.get("recipients", 0)
        per_person = stats.get("per_person", 0.0)
        logger.info(f"  💸 税收再分配: 总额=${total:,.2f}, 受益人数={recipients}, 人均=${per_person:,.2f}")
    
    def _log_month_end_summary(self, month: int) -> None:
        """打印美化的月末汇总报告"""
        
        # ═══════════════════════════════════════════════════════════════════════
        # 月度经济报告
        # ═══════════════════════════════════════════════════════════════════════
        
        box_width = 70
        
        def box_line(char: str = "═") -> str:
            return char * box_width
        
        def box_title(title: str) -> str:
            padding = (box_width - len(title) - 4) // 2
            return f"║{'─' * padding} {title} {'─' * (box_width - padding - len(title) - 4)}║"
        
        def box_row(label: str, value: str, indent: int = 2) -> str:
            content = f"{' ' * indent}{label}: {value}"
            return f"║ {content:<{box_width - 4}} ║"
        
        def box_row_pair(label1: str, val1: str, label2: str, val2: str) -> str:
            half = (box_width - 6) // 2
            left = f"  {label1}: {val1}"
            right = f"  {label2}: {val2}"
            return f"║ {left:<{half}}{right:<{half}} ║"
        
        def box_separator() -> str:
            return f"╟{'─' * (box_width - 2)}╢"
        
        def format_money(v: float) -> str:
            if abs(v) >= 1_000_000:
                return f"${v / 1_000_000:,.2f}M"
            elif abs(v) >= 1_000:
                return f"${v / 1_000:,.1f}K"
            else:
                return f"${v:,.2f}"
        
        def format_pct(v: float) -> str:
            if v is None:
                return "N/A"
            return f"{v * 100:.1f}%"
        
        # 获取数据
        if self.economic_center is None:
            logger.info("  ⚠️  经济中心未初始化")
            return
        
        try:
            # 使用缓存的 GDP 数据（在 _record_month_summary 中已计算）
            # 避免重复计算，且缓存的数据包含正确的 production_stats
            gdp_data = getattr(self, "_last_gdp_comprehensive", None) or {}
            if not gdp_data:
                # 兜底：如果没有缓存，重新计算（但这种情况下 production_stats 为 None）
                gdp_data = self._call_actor(self.economic_center, "calculate_gdp_comprehensive", month, None, 0) or {}
            
            # 获取税收汇总
            tax_summary = self._call_actor(self.economic_center, "get_monthly_tax_collection", month) or {}
            
            # 获取账户余额
            gov_balance = self._call_actor(self.economic_center, "query_balance", "gov_main_simulation") or 0.0
            
            # 家庭和企业余额（批量查询）
            total_hh_balance = 0.0
            if self.households:
                hh_query_args = [(hh.household_id,) for hh in self.households]
                hh_balances = self._call_actor_batch(self.economic_center, "query_balance", hh_query_args)
                total_hh_balance = sum(float(bal or 0) for bal in hh_balances)
            avg_hh_balance = total_hh_balance / len(self.households) if self.households else 0
            
            total_firm_balance = 0.0
            if self.firms:
                firm_query_args = [(firm.firm_id,) for firm in self.firms]
                firm_balances = self._call_actor_batch(self.economic_center, "query_balance", firm_query_args)
                total_firm_balance = sum(float(bal or 0) for bal in firm_balances)
            avg_firm_balance = total_firm_balance / len(self.firms) if self.firms else 0
            
            # 劳动力市场数据（使用实际注册的劳动力，数据校验）
            labor_summary = self._call_actor(self.labor_market, "summary") or {}
            total_labor = max(0.0, float(labor_summary.get("total_labor_hours", 0.0) or 0.0))
            employed = max(0.0, float(labor_summary.get("total_matched_jobs", 0.0) or 0.0))
            employed = min(employed, total_labor)  # 就业人数不能超过总劳动力
            employment_rate = employed / total_labor if total_labor > 0 else 0.0
            employment_rate = max(0.0, min(1.0, employment_rate))  # 限制在 [0, 1]
            
            # 提取 GDP 分项
            nominal_gdp = float(gdp_data.get("nominal_gdp", 0.0) or 0.0)
            real_gdp = float(gdp_data.get("real_gdp", 0.0) or 0.0)
            gdp_growth = gdp_data.get("growth_rates", {}).get("nominal_gdp_growth")
            
            exp_comp = gdp_data.get("expenditure_components", {})
            consumption = float(exp_comp.get("consumption", {}).get("total", 0.0) or 0.0)
            gov_spending = float(exp_comp.get("government", {}).get("total", 0.0) or 0.0)
            investment = float(exp_comp.get("investment", {}).get("inventory_investment", 0.0) or 0.0)
            
            income_comp = gdp_data.get("income_components", {})
            total_wages = float(income_comp.get("compensation_of_employees", {}).get("total", 0.0) or 0.0)
            operating_surplus = float(income_comp.get("operating_surplus", 0.0) or 0.0)
            
            ratios = gdp_data.get("ratios", {})
            consumption_rate = ratios.get("consumption_rate", 0)
            labor_share = ratios.get("labor_share", 0)
            
            # 税收数据
            total_tax = float(tax_summary.get("total_tax", 0.0) or 0.0)
            vat = float(tax_summary.get("consume_tax", 0.0) or 0.0)
            labor_tax = float(tax_summary.get("labor_tax", 0.0) or 0.0)
            corp_tax = float(tax_summary.get("corporate_tax", 0.0) or 0.0)
            
            # 构建报告
            lines = []
            lines.append("")
            lines.append(f"╔{box_line()}╗")
            lines.append(f"║{' ' * ((box_width - 20) // 2)}📊 月度经济报告 (M{month}){' ' * ((box_width - 21) // 2)}║")
            lines.append(f"╠{box_line()}╣")
            
            # GDP 概览
            lines.append(box_title("GDP 概览"))
            lines.append(box_row("名义 GDP", format_money(nominal_gdp)))
            lines.append(box_row("实际 GDP", format_money(real_gdp)))
            lines.append(box_row("GDP 增长率", format_pct(gdp_growth) if gdp_growth else "首月"))
            lines.append(box_separator())
            
            # 支出分解
            lines.append(box_title("支出法分解 (C + G + I)"))
            lines.append(box_row("消费 (C)", f"{format_money(consumption)} ({format_pct(consumption_rate)})"))
            lines.append(box_row("政府 (G)", format_money(gov_spending)))
            lines.append(box_row("投资 (I)", format_money(investment)))
            lines.append(box_separator())
            
            # 收入分配
            lines.append(box_title("收入分配"))
            lines.append(box_row("劳动报酬", f"{format_money(total_wages)} ({format_pct(labor_share)})"))
            lines.append(box_row("营业盈余", format_money(operating_surplus)))
            lines.append(box_separator())
            
            # 税收
            lines.append(box_title("税收"))
            lines.append(box_row_pair("总税收", format_money(total_tax), "VAT", format_money(vat)))
            lines.append(box_row_pair("个人所得税", format_money(labor_tax), "企业所得税", format_money(corp_tax)))
            lines.append(box_separator())
            
            # 就业
            lines.append(box_title("劳动力市场"))
            lines.append(box_row_pair("就业率", format_pct(employment_rate), "就业人数", f"{int(employed):,}"))
            lines.append(box_separator())
            
            # 账户余额
            lines.append(box_title("账户余额"))
            lines.append(box_row("政府", format_money(gov_balance)))
            lines.append(box_row_pair("家庭总计", format_money(total_hh_balance), "家庭平均", format_money(avg_hh_balance)))
            lines.append(box_row_pair("企业总计", format_money(total_firm_balance), "企业平均", format_money(avg_firm_balance)))
            
            lines.append(f"╚{box_line()}╝")
            lines.append("")
            
            # 输出报告
            for line in lines:
                logger.info(line)
                
        except Exception as e:
            logger.error(f"  ❌ 生成月度报告失败: {e}")
            import traceback
            traceback.print_exc()
