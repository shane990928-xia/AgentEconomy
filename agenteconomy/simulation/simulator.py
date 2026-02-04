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
from agenteconomy.agent.household import Household, consumption_progress
from agenteconomy.agent.government import Government
from agenteconomy.agent.bank import Bank
from agenteconomy.simulation.agent_loader import create_firms, create_households
from agenteconomy.simulation.checkpoint import CheckpointManager
from agenteconomy.market.AbstractResourceMarket import AbstractResourceMarket
from datetime import datetime
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
from contextlib import contextmanager
import ray

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
        self._last_price_index: Optional[float] = None  # 基于 100 的价格指数
        self._last_inflation_rate: Optional[float] = None  # 上次计算的通胀率
        self._last_balance_by_household: Dict[str, float] = {}
        self._last_expected_income_by_household: Dict[str, float] = {}
        self._last_sales_by_product: Dict[str, float] = {}
        self._last_gdp_comprehensive: Optional[Dict[str, Any]] = None  # 缓存的GDP计算结果
        
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
                    vat_rate=self.config.vat_rate
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
                    economic_center=self.economic_center
                )

            self.government.initialize()
            self.government.set_labor_market(self.labor_market)  # 设置劳动力市场引用
            
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
            abstract_resource_market=self.abstract_resource_market
        )
        self._firm_by_id = {f.firm_id: f for f in (self.firms or [])}
        self._index_firms()
        if self.economic_center is not None and self.firms:
            # 批量注册企业 ID 和初始化账本（并行执行）
            register_args = [(firm.firm_id, "firm") for firm in self.firms]
            ledger_args = [(firm.firm_id, 0.0) for firm in self.firms]
            self._call_actor_batch(self.economic_center, "register_id", register_args)
            self._call_actor_batch(self.economic_center, "init_agent_ledger", ledger_args)
        
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

    def _econ_month(self, month: int, preheat: bool) -> int:
        offset = int(getattr(self.config, "preheat_months", 0) or 0)
        if preheat or offset <= 0:
            return month
        return month + offset

    @contextmanager
    def _time_block(self, label: str, month: Optional[int] = None, preheat: Optional[bool] = None):
        parts = []
        if month is not None:
            parts.append(f"month={month}")
        if preheat is not None:
            parts.append(f"preheat={preheat}")
        suffix = f" {' '.join(parts)}" if parts else ""
        logger.info(f"[计时开始] {label}{suffix}")
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
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
            consumption_results = await self._collect_consumption_plans(top_k=10)
            demand_by_product, _, snapshot_cache, demand_stats = self._build_orders(consumption_results)
        
        # 激活被需求的SKU
        demanded_sku_ids = list(demand_by_product.keys())
        if demanded_sku_ids and self.product_market is not None:
            activated_count = self._call_actor(self.product_market, "activate_skus", demanded_sku_ids)
            logger.info(f"Preheat Phase 0: Activated {activated_count} demanded SKUs")
        else:
            logger.warning("Preheat Phase 0: No demanded SKUs found! Check household savings.")
        
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
    
    def _initialize_firm_capital_from_demand(self, demand_stats: Dict[str, Any]):
        """
        根据Phase 0发现的需求，为企业分配初始资金
        
        逻辑：企业初始cash = 预期月收入 * 系数（用于支付首月工资）
        系数默认为 1.5（足够支付 1.5 个月的运营成本）
        """
        capital_multiplier = float(getattr(self.config, "firm_initial_capital_multiplier", 1.5) or 1.5)
        
        # 从demand_stats中获取各企业的预期收入
        demand_by_mfg = demand_stats.get("by_mfg_firm", {})
        demand_by_retail = demand_stats.get("by_retail_firm", {})
        
        # 获取最低资金配置
        min_cash = float(getattr(self.config, "firm_min_initial_cash", 10000.0) or 10000.0)
        total_initialized = 0
        
        # 为制造商分配初始资金
        for firm_id, stats in demand_by_mfg.items():
            expected_revenue = float(stats.get("value", 0.0))
            calculated_cash = expected_revenue * capital_multiplier
            # 确保不低于最低资金要求
            initial_cash = max(calculated_cash, min_cash)

            firm = self._firm_by_id.get(firm_id)
            if firm is not None:
                firm.cash = initial_cash
                # 同步到 EconomicCenter 的 ledger
                if self.economic_center is not None:
                    result = self._call_actor(self.economic_center, "set_agent_balance", firm_id, initial_cash)
                    logger.info(f"[资金初始化] 制造商 {firm_id}: expected_revenue={expected_revenue:.2f}, "
                               f"calculated={calculated_cash:.2f}, min={min_cash:.2f}, initial_cash={initial_cash:.2f}")
                total_initialized += 1
            elif firm is None:
                logger.warning(f"[资金初始化] 制造商 {firm_id} 在 _firm_by_id 中未找到")

        # 为零售商分配初始资金
        for firm_id, stats in demand_by_retail.items():
            expected_revenue = float(stats.get("value", 0.0))
            calculated_cash = expected_revenue * capital_multiplier
            # 确保不低于最低资金要求
            initial_cash = max(calculated_cash, min_cash)

            firm = self._firm_by_id.get(firm_id)
            if firm is not None:
                firm.cash = initial_cash
                # 同步到 EconomicCenter 的 ledger
                if self.economic_center is not None:
                    self._call_actor(self.economic_center, "set_agent_balance", firm_id, initial_cash)
                total_initialized += 1
                logger.debug(f"Initialized {firm_id} cash: {initial_cash:.2f} (from demand {expected_revenue:.2f})")

        # 为完全没有需求的企业设置最低资金（用于基本运营）
        for firm in (self.firms or []):
            if firm.cash <= 0:
                firm.cash = min_cash
                # 同步到 EconomicCenter 的 ledger
                if self.economic_center is not None:
                    self._call_actor(self.economic_center, "set_agent_balance", firm.firm_id, min_cash)
                total_initialized += 1
        
        logger.info(f"Phase 0: Initialized capital for {total_initialized} firms (min_cash={min_cash:.2f})")

    async def _run_warmup_month(self, month: int):
        econ_month = self._econ_month(month, preheat=True)

        # 月初重置供需追踪
        self._call_actor(self.product_market, "reset_supply_demand_tracking")

        # ========== 商品市场 ==========
        with self._time_block("消费决策", month=month, preheat=True):
            consumption_results = await self._collect_consumption_plans(top_k=10)
        with self._time_block("构建订单", month=month, preheat=True):
            demand_by_product, orders_by_household, snapshot_cache, demand_stats = self._build_orders(consumption_results)

        # 记录需求到ProductMarket
        self._record_demand_to_market(demand_by_product, snapshot_cache)

        with self._time_block("生产补货", month=month, preheat=True):
            production_demand = self._last_sales_by_product or demand_by_product
            production_stats = self._ensure_production(production_demand, snapshot_cache, econ_month, record_transactions=True)

        # 记录供给并根据供需调整价格
        self._record_supply_and_adjust_prices(production_stats, econ_month)

        with self._time_block("零售商进货", month=month, preheat=True):
            procurement_stats = self._retailer_procurement(demand_by_product, snapshot_cache, econ_month, record_transactions=True)
        with self._time_block("执行购买", month=month, preheat=True):
            consumption_stats = self._execute_orders(orders_by_household, snapshot_cache, econ_month, record_transactions=True)
        with self._time_block("服务消费", month=month, preheat=True):
            service_consumption_stats = self._execute_service_consumption(consumption_results, econ_month)
        
        # 更新家庭消费历史（用于下月消费惯性计算）
        self._update_household_consumption_history(
            consumption_stats, service_consumption_stats, consumption_results
        )
        
        # 计算家庭消费总预算（用于政府需求注入）
        household_total_budget = self._get_household_consumption_budget(consumption_results)
        
        # ========== 政府采购 ==========
        with self._time_block("政府采购", month=month, preheat=True):
            government_procurement_stats = await self._execute_government_procurement(
                econ_month, 
                household_consumption_budget=household_total_budget
            )
        
        # 更新销售记录（在政府采购之后，以便包含政府采购数据）
        self._update_last_sales(econ_month, consumption_stats)
        
        # ========== 劳动力市场 ==========
        with self._time_block("发布岗位", month=month, preheat=True):
            await self._post_jobs(econ_month, production_stats=production_stats, service_stats=service_consumption_stats, demand_stats=demand_stats)
        with self._time_block("招聘匹配", month=month, preheat=True):
            await self._match_jobs(econ_month, use_llm=False)
        with self._time_block("发放工资", month=month, preheat=True):
            wage_stats = self._pay_wages(econ_month, record_transactions=True)
        
        # ========== 月末结算 ==========
        with self._time_block("企业所得税", month=month, preheat=True):
            self._settle_corporate_tax(econ_month)
        with self._time_block("银行利息", month=month, preheat=True):
            await self._pay_bank_interest(econ_month)
        with self._time_block("税收再分配", month=month, preheat=True):
            await self._redistribute_taxes(econ_month)
        
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

    async def _run_month(self, month: int):
        """Run a single month"""
        econ_month = self._econ_month(month, preheat=False)
        
        # 月份标题
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
        with self._time_block("发布岗位", month=month, preheat=False):
            await self._post_jobs(econ_month)  # 使用 econ_month 以便正确查询上月数据

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
            consumption_results = await self._collect_consumption_plans(top_k=10)
        self._log_consumption_plans(consumption_results)
        
        with self._time_block("构建订单", month=month, preheat=False):
            demand_by_product, orders_by_household, snapshot_cache, demand_stats = self._build_orders(consumption_results)
        self._log_demand_stats(demand_stats)
        
        # 记录需求到ProductMarket
        self._record_demand_to_market(demand_by_product, snapshot_cache)
        
        with self._time_block("生产补货", month=month, preheat=False):
            production_demand = self._last_sales_by_product or demand_by_product
            production_stats = self._ensure_production(production_demand, snapshot_cache, econ_month, record_transactions=True)
        self._log_production_stats(production_stats)
        
        # 记录供给并根据供需调整价格
        self._record_supply_and_adjust_prices(production_stats, econ_month)
        
        with self._time_block("零售商进货", month=month, preheat=False):
            procurement_stats = self._retailer_procurement(demand_by_product, snapshot_cache, econ_month, record_transactions=True)
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
        
        # 计算家庭消费总预算（用于政府需求注入）
        household_total_budget = self._get_household_consumption_budget(consumption_results)

        # ========== 政府采购 ==========
        logger.info(f"\n┌{'─' * 38}┐")
        logger.info(f"│ 🏛️  政府采购                          │")
        logger.info(f"└{'─' * 38}┘")
        with self._time_block("政府采购", month=month, preheat=False):
            government_procurement_stats = await self._execute_government_procurement(
                econ_month, 
                household_consumption_budget=household_total_budget
            )
        self._log_government_procurement_stats(government_procurement_stats)
        
        # 更新销售记录（在政府采购之后，以便包含政府采购数据）
        self._update_last_sales(econ_month, consumption_stats)

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
        
        with self._time_block("税收再分配", month=month, preheat=False):
            redistribution_stats = await self._redistribute_taxes(econ_month)
        self._log_redistribution_stats(redistribution_stats)

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

    async def _collect_consumption_plans(self, top_k: int = 10) -> List[Tuple[Household, Dict[str, Any]]]:
        results: List[Tuple[Household, Dict[str, Any]]] = []
        if not self.households:
            return results
        
        # 初始化消费进度追踪器
        # log_interval: 每完成 10% 的家庭打印一次进度
        log_interval = max(1, len(self.households) // 10)
        consumption_progress.reset(
            total=len(self.households),
            log_interval=log_interval,
            logger_instance=logger
        )
        logger.info(f"[消费进度] 开始收集 {len(self.households)} 个家庭的消费计划...")
        
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

        self._last_balance_by_household = dict(balance_by_household)
        self._last_expected_income_by_household = dict(expected_income_by_household)

        # 计算宏观经济指标
        macro_indicators = self._compute_macro_indicators()

        # 分批处理家庭消费，避免ProductMarket actor过载
        batch_size = int(os.getenv("CONSUMPTION_BATCH_SIZE", "100"))
        all_outputs = []

        for batch_start in range(0, len(self.households), batch_size):
            batch_end = min(batch_start + batch_size, len(self.households))
            batch_households = self.households[batch_start:batch_end]
            logger.info(f"[消费进度] 处理批次 {batch_start//batch_size + 1}/{(len(self.households) + batch_size - 1)//batch_size}, 家庭 {batch_start+1}-{batch_end}")

            tasks = []
            for hh in batch_households:
                available_balance = balance_by_household.get(hh.household_id)
                expected_income = expected_income_by_household.get(hh.household_id, 0.0)
                # 消费预算只基于当前余额，不包含预期收入
                # 因为工资是在消费之后才发放的
                available_budget = float(available_balance) if available_balance is not None else None

                # 不设置整体超时，依赖单个LLM调用的超时控制
                task = hh.consume_v2(
                    top_k=top_k,
                    product_market=self.product_market,
                    available_balance=available_balance,
                    expected_income=expected_income,
                    available_budget=available_budget,
                    macro_indicators=macro_indicators,
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
                f"value={stats.get('value', 0.0):.2f}"
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
        demand_by_retail_firm: Dict[str, Dict[str, float]] = defaultdict(lambda: {"qty": 0.0, "value": 0.0})
        demand_by_mfg_firm: Dict[str, Dict[str, float]] = defaultdict(lambda: {"qty": 0.0, "value": 0.0})
        total_demand_qty = 0.0
        total_demand_value = 0.0

        for hh, result in consumption_results:
            step0 = result.get("step0", {}) if isinstance(result, dict) else {}
            budgets = step0.get("budgets", {}) if isinstance(step0, dict) else {}
            if budgets:
                hh.apply_consumption(budgets)
            
            # 提取商品消费预算 (Retail merchandise)
            goods_budget = float(budgets.get("Retail merchandise", 0.0) or 0.0)

            purchases = []
            step3 = result.get("step3", {}) if isinstance(result, dict) else {}
            for rec in step3.get("purchases", []) or []:
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
                purchases.append(
                    {
                        "product_id": product_id,
                        "desired_qty": desired_qty,
                        "unit_price": unit_price,
                        "product_name": snapshot.get("name"),
                        "manufacturer_code": snapshot.get("manufacturer_code"),
                        "retailer_code": snapshot.get("retailer_code"),
                        "base_retail_price": float(
                            snapshot.get("base_retail_price")
                            or snapshot.get("retail_price")
                            or unit_price
                            or 0.0
                        ),
                        "base_manufacturer_price": float(
                            snapshot.get("base_manufacturer_price")
                            or snapshot.get("manufacturer_price")
                            or unit_price
                            or 0.0
                        ),
                    }
                )
                demand_by_product[product_id] += desired_qty
                order_value = float(desired_qty) * unit_price
                total_demand_qty += float(desired_qty)
                total_demand_value += order_value
                retailer_code = snapshot.get("retailer_code")
                manufacturer_code = snapshot.get("manufacturer_code")
                ret_firm = self.retailers_by_industry.get(retailer_code) if retailer_code else None
                mfg_firm = self.manufacturers_by_industry.get(manufacturer_code) if manufacturer_code else None
                if ret_firm is not None:
                    stats = demand_by_retail_firm[ret_firm.firm_id]
                    stats["qty"] += float(desired_qty)
                    stats["value"] += order_value
                if mfg_firm is not None:
                    stats = demand_by_mfg_firm[mfg_firm.firm_id]
                    stats["qty"] += float(desired_qty)
                    stats["value"] += order_value

            orders_by_household.append((hh, purchases, goods_budget))

        if self._debug_enabled():
            self._log_demand_summary(demand_by_retail_firm, demand_by_mfg_firm)

        demand_summary = {
            "total_qty": total_demand_qty,
            "total_value": total_demand_value,
            "by_retail_firm": {str(k): {"qty": v["qty"], "value": v["value"]} for k, v in demand_by_retail_firm.items()},
            "by_mfg_firm": {str(k): {"qty": v["qty"], "value": v["value"]} for k, v in demand_by_mfg_firm.items()},
        }
        return demand_by_product, orders_by_household, snapshot_cache, demand_summary

    def _ensure_production(
        self,
        demand_by_product: Dict[str, float],
        snapshot_cache: Dict[str, Optional[Dict[str, Any]]],
        month: int,
        record_transactions: bool
    ) -> Dict[str, Any]:
        firm_plans: Dict[ManufactureFirm, Dict[str, int]] = defaultdict(dict)
        unmet_products: List[str] = []
        production_stats = {
            "total_qty": 0.0,
            "total_value": 0.0,
            "by_firm": {},
            # GDP 计算需要的字段
            "total_output_value": 0.0,
            "total_production_cost": 0.0,
            "firm_production_value": {},
            "firm_production_cost": {},
        }
        for product_id, demand_qty in (demand_by_product or {}).items():
            snapshot = self._get_product_snapshot_cached(product_id, snapshot_cache)
            if not snapshot:
                continue
            available = int(snapshot.get("available_stock") or 0)
            shortage = int(demand_qty) - available
            if shortage <= 0:
                continue
            mfg_code = snapshot.get("manufacturer_code")
            firm = self.manufacturers_by_industry.get(mfg_code)
            if firm is None:
                unmet_products.append(product_id)
                continue
            firm_plans[firm][product_id] = firm_plans[firm].get(product_id, 0) + shortage

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

        for firm, plan in firm_plans.items():
            sku_base_prices = {}
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
            firm_qty = float(sum(plan.values()))
            firm_value = 0.0
            for sku_id, qty in (plan or {}).items():
                firm_value += float(qty or 0.0) * float(sku_base_prices.get(sku_id, 0.0) or 0.0)
            production_stats["by_firm"][firm.firm_id] = {"qty": firm_qty, "value": firm_value}
            production_stats["total_qty"] += firm_qty
            production_stats["total_value"] += firm_value
            
            # 初始化生产成本（后面会被实际值覆盖）
            firm_production_cost = 0.0
            
            if record_transactions:
                # 注意：制造商已在初始化时预注册到中间品市场
                # 不再需要每次生产时重复注册
                produce_result = firm.produce(
                    production_plan=plan,
                    sku_base_prices=sku_base_prices,
                    period=month,
                    update_inventory=True,
                )
                # 获取实际生产成本（中间消耗）
                firm_production_cost = float(produce_result.get("total_cost", 0.0) or 0.0)
                
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
            
            # 更新 GDP 计算需要的统计
            production_stats["firm_production_value"][firm.firm_id] = firm_value
            production_stats["firm_production_cost"][firm.firm_id] = firm_production_cost
            production_stats["total_output_value"] += firm_value
            production_stats["total_production_cost"] += firm_production_cost

            # 生产完成后，清空已生产产品的缓存，以便后续步骤获取最新库存
            for sku_id in plan.keys():
                if sku_id in snapshot_cache:
                    del snapshot_cache[sku_id]

        # 记录原材料需求到ProductMarket（用于价格调整）
        raw_material_demand = production_stats.get("raw_material_demand", {})
        if raw_material_demand and self.product_market is not None:
            for industry_code, demand_value in raw_material_demand.items():
                # 记录原材料需求（以成本值作为需求指标）
                self._call_actor(self.product_market, "record_raw_material_demand", industry_code, demand_value)

        return production_stats

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
        record_transactions: bool
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
        procurement_by_retailer: Dict[str, Dict[str, int]] = defaultdict(dict)
        
        for product_id, demand_qty in (demand_by_product or {}).items():
            if demand_qty <= 0:
                continue
            snapshot = self._get_product_snapshot_cached(product_id, snapshot_cache)
            if not snapshot:
                continue
            
            retailer_code = snapshot.get("retailer_code")
            if not retailer_code or retailer_code not in self.retailers_by_industry:
                continue
            
            # 检查库存是否充足（制造商已生产）
            available = int(snapshot.get("available_stock") or 0)
            qty_to_procure = min(int(demand_qty), available)
            if qty_to_procure <= 0:
                continue
            
            procurement_by_retailer[retailer_code][product_id] = qty_to_procure
        
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
                
                amount = wholesale_price * qty
                
                if record_transactions and self.economic_center is not None:
                    # 零售商支付给制造商
                    tx_id = self._call_actor(
                        self.economic_center,
                        "process_wholesale",
                        month,
                        retailer.firm_id,
                        manufacturer.firm_id,
                        amount,
                        qty,
                        product_id,
                        snapshot.get("name"),
                        wholesale_price,
                    )
                    if tx_id:
                        retailer_qty += qty
                        retailer_value += amount
                        procurement_stats["by_manufacturer"][manufacturer.firm_id]["qty"] += qty
                        procurement_stats["by_manufacturer"][manufacturer.firm_id]["value"] += amount
                else:
                    # 不记录交易模式：直接转移资金
                    if retailer.cash >= amount:
                        retailer.cash -= amount
                        manufacturer.cash += amount
                        retailer_qty += qty
                        retailer_value += amount
                        procurement_stats["by_manufacturer"][manufacturer.firm_id]["qty"] += qty
                        procurement_stats["by_manufacturer"][manufacturer.firm_id]["value"] += amount
            
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
        if retailer_code and retailer_code in self.retailers_by_industry:
            return self.retailers_by_industry[retailer_code]
        
        # 只有在没有 retailer_code 的情况下，才回退到制造商（直销）
        # 如果有 retailer_code 但找不到对应零售商，不应回退到制造商
        if not retailer_code:
            manufacturer_code = order.get("manufacturer_code")
            if manufacturer_code and manufacturer_code in self.manufacturers_by_industry:
                return self.manufacturers_by_industry[manufacturer_code]
        
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
                "household_count": int
            }
        """
        if self.abstract_resource_market is None:
            logger.warning("AbstractResourceMarket 未初始化，跳过服务消费")
            return {"total_service_consumption": 0.0, "by_category": {}, "by_industry": {}, "household_count": 0}

        total_consumption = 0.0
        by_category: Dict[str, float] = defaultdict(float)
        by_industry: Dict[str, float] = defaultdict(float)
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
            "household_count": household_count,
            "skipped_insufficient_balance": skipped_insufficient_balance,
        }
    
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
        
        # 服务消费通过 step0 budgets 和实际执行比例估算
        # 由于服务消费按家庭汇总较复杂，这里使用 step0 的服务预算作为近似
        service_by_hh: Dict[str, float] = {}
        
        
        for hh, out in consumption_results:
            step0 = out.get("step0", {}) if isinstance(out, dict) else {}
            budgets = step0.get("budgets", {}) if isinstance(step0, dict) else {}
            
            # 计算服务消费总预算
            total_service_budget = sum(
                float(budgets.get(cat) or 0.0)
                for cat in HOUSEHOLD_SERVICE_CATEGORY_TO_INDUSTRY.keys()
            )
            
            # 服务消费的实际执行率（简化假设：如果总体有消费，则按比例执行）
            # 更精确的方式是在 _execute_service_consumption 中追踪每个家庭的实际消费
            if service_consumption_stats:
                total_service = float(service_consumption_stats.get("total_service_consumption", 0.0) or 0.0)
                # 假设服务预算大于0的家庭按比例消费
                # 这里简化为：如果家庭有服务预算且全局服务消费 > 0，则认为其预算被消费了
                if total_service > 0 and total_service_budget > 0:
                    service_by_hh[hh.household_id] = total_service_budget
            else:
                service_by_hh[hh.household_id] = 0.0
        
        # 更新每个家庭的上月消费
        updated_count = 0
        for hh, _ in consumption_results:
            goods_spent = float(goods_by_hh.get(hh.household_id, 0.0) or 0.0)
            service_spent = float(service_by_hh.get(hh.household_id, 0.0) or 0.0)
            total_spent = goods_spent + service_spent
            
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
        household_consumption_budget: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        执行政府采购
        
        政府作为"无限资金"的需求注入器：
        - 预算基于家庭消费总预算的一定比例（默认30%）
        - 即使税收为0，也会有最低采购预算（$50,000）
        - 使用IO表系数决定各行业的采购比例
        - 不收取VAT（避免政府自我征税）
        
        凯恩斯主义需求刺激：
        政府支出↑ → 企业收入↑ → 招聘↑ → 工资↑ → 消费↑ → 良性循环
        
        Args:
            month: 当前经济月份
            household_consumption_budget: 家庭消费总预算（用于计算政府需求注入）
            
        Returns:
            采购统计信息
        """
        if self.government is None:
            return {
                "total_spent": 0.0,
                "by_industry": {},
                "items_count": 0,
                "success": False,
                "error": "no government"
            }
        
        # 确保政府有产品市场引用
        if self.product_market is not None and not hasattr(self.government, 'product_market'):
            self.government.set_product_market(self.product_market)
        
        # 执行采购（传入家庭消费预算用于计算政府需求注入）
        try:
            result = self.government.procure_goods_and_services(
                period=month,
                household_consumption_budget=household_consumption_budget
            )
            
            if result.get("success") and result.get("total_spent", 0) > 0:
                logger.info(
                    f"[政府采购] 总支出=${result['total_spent']:,.2f}, "
                    f"涉及{len(result.get('by_industry', {}))}个行业, "
                    f"{result.get('items_count', 0)}个采购项"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"[政府采购] 执行失败: {e}")
            return {
                "total_spent": 0.0,
                "by_industry": {},
                "items_count": 0,
                "success": False,
                "error": str(e)
            }

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
                available = int(snapshot.get("available_stock") or 0)
                desired_qty = int(order.get("desired_qty") or 0)
                qty = min(desired_qty, available)
                if qty <= 0:
                    continue
                unit_price = float(order.get("unit_price") or snapshot.get("retail_price") or 0.0)
                if unit_price <= 0:
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
                amount = float(qty) * unit_price
                seller = self._resolve_seller_firm(order)
                if seller is None:
                    continue
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
                    )
                    if tx_id:
                        self._call_actor(self.product_market, "update_stock", product_id, -qty)
                        # 更新 snapshot_cache 中的库存，防止后续家庭超卖
                        if product_id in snapshot_cache and snapshot_cache[product_id]:
                            old_stock = int(snapshot_cache[product_id].get("available_stock") or 0)
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
                    self._call_actor(self.product_market, "update_stock", product_id, -qty)
                    # 更新 snapshot_cache 中的库存，防止后续家庭超卖
                    if product_id in snapshot_cache and snapshot_cache[product_id]:
                        old_stock = int(snapshot_cache[product_id].get("available_stock") or 0)
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

            # 如果没有收入数据，跳过（避免误裁）
            if expected_revenue <= 0:
                continue

            # 计算新工资帽（产能下限）
            compensation_ratio = float(getattr(firm, "compensation_ratio", 0.2) or 0.2)
            new_wage_cap = expected_revenue * compensation_ratio
            
            # 🛡️ 最低工资帽保护：防止过度裁员导致的死亡螺旋
            # 至少保留2-3名员工的工资（约 $6000/月），维持基本运营
            MIN_WAGE_CAP = 6000.0
            new_wage_cap = max(new_wage_cap, MIN_WAGE_CAP)
            
            # 📊 额外保护：如果当前员工很少，不裁员
            # 防止企业从少量员工再裁减到0
            MIN_EMPLOYEES_TO_KEEP = 2
            current_employees = int(getattr(firm, "employee_count", 0) or 0)
            if current_employees <= MIN_EMPLOYEES_TO_KEEP:
                continue  # 跳过裁员，保持最低员工数

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

        # 政府公共就业：不参与常规裁员逻辑
        # 公共就业岗位是政策工具，作为"最后雇主"存在，不应该因预算削减而裁员
        # 政府通过无限资金保证公共就业岗位的稳定性
        if self.government is not None:
            gov_id = self.government.government_id
            # 不执行政府裁员，保持公共就业的稳定性
            logger.debug(f"[政府] 公共就业岗位免于裁员（政策工具）")

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
    ) -> None:
        """
        企业发布岗位
        
        Args:
            month: 当前月份
            production_stats: 本月生产统计 {by_firm: {firm_id: {qty, value}}}
            service_stats: 本月服务消费统计 {by_industry: {industry_code: amount}}
            demand_stats: 本月需求统计 {by_mfg_firm: {firm_id: {qty, value}}, by_retail_firm: ...}
        """
        # 从 demand_stats 提取各制造商的需求价值作为劳动预算基础
        # 需求价值更能反映企业的真实经营状况，而不仅仅是实际生产量
        demand_by_mfg = (demand_stats or {}).get("by_mfg_firm", {}) if demand_stats else {}
        production_by_firm = (production_stats or {}).get("by_firm", {}) if production_stats else {}
        service_by_industry = (service_stats or {}).get("by_industry", {}) if service_stats else {}
        
        # 打印需求数据统计
        if demand_by_mfg:
            total_demand = sum(float(v.get("value", 0.0) or 0.0) for v in demand_by_mfg.values())
            logger.info(f"[岗位发布] 使用需求数据: 制造商数={len(demand_by_mfg)}, 总需求=${total_demand:,.2f}")
        
        # 企业发布岗位，传入本月需求价值
        tasks = []
        service_firms_with_income = []  # 记录有服务收入的企业
        for firm in (self.firms or []):
            # 获取该企业本月的需求价值（优先使用需求数据，其次使用生产数据）
            demand_value = 0.0
            
            # 对于制造商，优先使用需求数据
            firm_demand = demand_by_mfg.get(firm.firm_id, {})
            demand_value = float(firm_demand.get("value", 0.0) or 0.0)
            
            # 如果没有需求数据，回退到生产数据
            if demand_value <= 0:
                firm_production = production_by_firm.get(firm.firm_id, {})
                demand_value = float(firm_production.get("value", 0.0) or 0.0)
            
            # 服务企业使用服务消费数据
            service_income = float(service_by_industry.get(firm.industry, 0.0) or 0.0)
            # 记录有服务收入的企业
            if service_income > 0:
                service_firms_with_income.append((firm.firm_id, firm.industry, service_income))
            # 传给企业作为本月需求基础
            tasks.append(firm.post_jobs(period=month, current_demand_value=demand_value + service_income))
        
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
            for firm, res in zip(self.firms or [], results):
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
        jobs = self._call_actor(self.labor_market, "query_opening_jobs") or []
        for hh in self.households or []:
            seekers = hh.list_job_seekers(hh.labor_hours)
            for labor_hour in seekers:
                matches = hh.match_jobs_topk_by_loss(labor_hour=labor_hour, jobs=jobs, top_k=3)
                applications = await hh.decide_job_applications(
                    month=month,
                    labor_hour=labor_hour,
                    top_matches=matches,
                    use_llm=use_llm,
                )
                for app in applications:
                    self._call_actor(self.labor_market, "submit_application", app, labor_hour)

        self._call_actor(self.labor_market, "make_offers", month, max_backups=3, reset_existing=True)
        self._call_actor(self.labor_market, "resolve_offers", month, acceptance_policy="best_loss")
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
        for rec in matched:
            firm_id = rec.get("firm_id")
            if firm_id:
                employee_count_by_firm[firm_id] += 1

        # 更新企业员工数
        for firm in (self.firms or []):
            firm.employee_count = employee_count_by_firm.get(firm.firm_id, 0)

        # 更新政府员工数
        if self.government:
            self.government.employee_count = employee_count_by_firm.get(self.government.government_id, 0)

    def _refresh_household_employment_status(self) -> None:
        if self.labor_market is None:
            return
        snapshot = self._call_actor(self.labor_market, "get_labor_status_snapshot")
        if not isinstance(snapshot, dict):
            return
        for hh in self.households or []:
            entry = snapshot.get(hh.household_id, {})
            head = entry.get("head")
            spouse = entry.get("spouse")
            if head is not None:
                code = hh._EMPLOYED_CODE if head.get("employed") else hh._NOT_EMPLOYED_CODE
                hh.ER82433 = code
                hh.csv_values["ER82433"] = code
            if spouse is not None:
                code = hh._EMPLOYED_CODE if spouse.get("employed") else hh._NOT_EMPLOYED_CODE
                hh.SP_employment_status = code
                hh.csv_values["SP_employment_status"] = code

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
            if rec.get("lh_type") == "spouse":
                hh.update_sp_income(gross)
            else:
                hh.update_rp_income(gross)
            wage_by_firm[firm_id] += float(gross)
            wage_by_household[household_id] += float(gross)

            if record_transactions and self.economic_center is not None:
                self._call_actor(
                    self.economic_center,
                    "process_wage",
                    month,
                    wage_per_hour,
                    household_id,
                    firm_id,
                    hours,
                    ppm,
                )
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
            # 使用新的综合 GDP 计算
            gdp_comprehensive = self._call_actor(
                self.economic_center, "calculate_gdp_comprehensive", econ_month, production_stats, 0
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
            },
            "details": {
                "labor_market_raw": labor_summary,
                "demand": demand_stats or {},
                "production": production_stats or {},
                "procurement": procurement_stats or {},
                "consumption": consumption_stats or {},
                "service_consumption": service_consumption_stats or {},
                "wages": wage_stats or {},
                "firm_financials": firm_financials or {},
                "tax": tax_stats or {},
            },
        }
        self._write_month_record(month, payload, preheat=preheat)

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

        # 记录供给量（消费品）
        # 注意：需求使用行业名称作为键（来自产品的 manufacturer_code），
        #       所以供给也需要转换为行业名称以匹配
        supply_count = 0
        for firm_id, stats in by_firm.items():
            firm = self._firm_by_id.get(firm_id)
            if firm is None:
                continue
            mfg_code = getattr(firm, "industry", None)
            if not mfg_code:
                continue
            # 将行业代码转换为行业名称（与需求记录的键保持一致）
            industry_name = self._industry_code_to_name.get(mfg_code, mfg_code)
            supply_qty = float(stats.get("qty", 0.0) or 0.0)
            if supply_qty > 0:
                self._call_actor(self.product_market, "record_supply", industry_name, supply_qty)
                supply_count += 1

        if self._debug_enabled() and supply_count > 0:
            logger.info(f"[供需调试] 记录供给企业数={supply_count}")
        
        # 收集消费品行业（有生产的行业）- 使用行业名称作为键
        consumer_goods_industries = set()
        for firm_id in by_firm.keys():
            firm = self._firm_by_id.get(firm_id)
            if firm is None:
                continue
            mfg_code = getattr(firm, "industry", None)
            if mfg_code:
                # 将行业代码转换为名称
                industry_name = self._industry_code_to_name.get(mfg_code, mfg_code)
                consumer_goods_industries.add(industry_name)

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
        total = procurement_stats.get("total_cost", 0.0)
        count = procurement_stats.get("retailers_count", 0)
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
