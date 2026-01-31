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
from agenteconomy.agent.household import Household
from agenteconomy.agent.government import Government
from agenteconomy.agent.bank import Bank
from agenteconomy.simulation.agent_loader import create_firms, create_households
from agenteconomy.market.AbstractResourceMarket import AbstractResourceMarket
from datetime import datetime
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import json
import os
import time
from contextlib import contextmanager
import ray

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
        self._firm_by_id: Dict[str, Firm] = {}
        self._household_by_id: Dict[str, Household] = {}
        
        self.current_month = 1
        self._record_dir: Optional[str] = None
        self._record_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._last_price_index: Optional[float] = None
        self._last_balance_by_household: Dict[str, float] = {}
        self._last_expected_income_by_household: Dict[str, float] = {}
        self._last_sales_by_product: Dict[str, float] = {}

        # Metrics
        
        logger.info(f"Simulator initialized with {self.config.num_months} months and {self.config.num_households} households")

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
        abstract_resource_market = AbstractResourceMarket(
            economic_center=self.economic_center,
            government_id=self.government.government_id
        )
        self.firms = create_firms(
            economic_center=self.economic_center, 
            labor_market=self.labor_market, 
            product_market=self.product_market, 
            abstract_resource_market=abstract_resource_market
        )
        self._firm_by_id = {f.firm_id: f for f in (self.firms or [])}
        self._index_firms()
        if self.economic_center is not None:
            for firm in self.firms or []:
                self._call_actor(self.economic_center, "register_id", firm.firm_id, "firm")
                self._call_actor(self.economic_center, "init_agent_ledger", firm.firm_id, 0.0)

    def _call_actor(self, actor, method_name: str, *args, **kwargs):
        if actor is None:
            return None
        method = getattr(actor, method_name, None)
        if method is None:
            return None
        if hasattr(method, "remote"):
            return ray.get(method.remote(*args, **kwargs))
        return method(*args, **kwargs)

    def _index_firms(self):
        self.manufacturers_by_industry = {}
        self.retailers_by_industry = {}
        for firm in self.firms or []:
            if not firm.industry:
                continue
            if isinstance(firm, ManufactureFirm):
                self.manufacturers_by_industry[firm.industry] = firm
            elif isinstance(firm, RetailFirm):
                self.retailers_by_industry[firm.industry] = firm

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

    def _write_month_record(self, month: int, payload: Dict[str, Any], preheat: bool) -> None:
        record_dir = self._ensure_record_dir()
        if not record_dir:
            return
        prefix = "preheat" if preheat else "month"
        path = os.path.join(record_dir, f"{prefix}_{month:04d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True)

    async def run_simulation(self):
        """Run simulation"""
        logger.info("Running simulation...")
        if self.config.preheat_months > 0:
            logger.info(f"Running preheat for {self.config.preheat_months} months...")
            await self._run_preheat(self.config.preheat_months)
            if self.economic_center is not None:
                self._call_actor(self.economic_center, "reset_transactions")
        for month in range(1, self.config.num_months + 1):
            await self._run_month(month)
            
    async def _run_preheat(self, months: int):
        for idx in range(1, int(months) + 1):
            logger.info(f"Preheat month {idx}...")
            await self._run_warmup_month(idx)

    async def _run_warmup_month(self, month: int):
        econ_month = self._econ_month(month, preheat=True)
        with self._time_block("消费决策", month=month, preheat=True):
            consumption_results = await self._collect_consumption_plans(top_k=10)
        with self._time_block("构建订单", month=month, preheat=True):
            demand_by_product, orders_by_household, snapshot_cache, demand_stats = self._build_orders(consumption_results)
        with self._time_block("生产补货", month=month, preheat=True):
            production_demand = self._last_sales_by_product or demand_by_product
            production_stats = self._ensure_production(production_demand, snapshot_cache, econ_month, record_transactions=True)
        with self._time_block("执行购买", month=month, preheat=True):
            consumption_stats = self._execute_orders(orders_by_household, snapshot_cache, econ_month, record_transactions=True)
        self._update_last_sales(econ_month, consumption_stats)
        with self._time_block("发布岗位", month=month, preheat=True):
            await self._post_jobs(month)
        with self._time_block("招聘匹配", month=month, preheat=True):
            await self._match_jobs(month, use_llm=False)
        with self._time_block("发放工资", month=month, preheat=True):
            wage_stats = self._pay_wages(econ_month, record_transactions=True)
        with self._time_block("月度汇总", month=month, preheat=True):
            self._record_month_summary(
                month=month,
                econ_month=econ_month,
                preheat=True,
                demand_stats=demand_stats,
                production_stats=production_stats,
                consumption_stats=consumption_stats,
                wage_stats=wage_stats,
            )

    async def _run_month(self, month: int):
        """Run a single month"""
        econ_month = self._econ_month(month, preheat=False)
        logger.info(f"Running month {month}...")
        with self._time_block("发布岗位", month=month, preheat=False):
            await self._post_jobs(month)
        with self._time_block("招聘匹配", month=month, preheat=False):
            await self._match_jobs(month, use_llm=False)
        with self._time_block("发放工资", month=month, preheat=False):
            wage_stats = self._pay_wages(econ_month, record_transactions=True)

        with self._time_block("消费决策", month=month, preheat=False):
            consumption_results = await self._collect_consumption_plans(top_k=10)
        with self._time_block("构建订单", month=month, preheat=False):
            demand_by_product, orders_by_household, snapshot_cache, demand_stats = self._build_orders(consumption_results)
        with self._time_block("生产补货", month=month, preheat=False):
            production_demand = self._last_sales_by_product or demand_by_product
            production_stats = self._ensure_production(production_demand, snapshot_cache, econ_month, record_transactions=True)
        with self._time_block("执行购买", month=month, preheat=False):
            consumption_stats = self._execute_orders(orders_by_household, snapshot_cache, econ_month, record_transactions=True)
        self._update_last_sales(econ_month, consumption_stats)
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
            )

    async def _collect_consumption_plans(self, top_k: int = 10) -> List[Tuple[Household, Dict[str, Any]]]:
        results: List[Tuple[Household, Dict[str, Any]]] = []
        if not self.households:
            return results
        balance_by_household: Dict[str, float] = {}
        if self.economic_center is not None:
            for hh in self.households:
                bal = self._call_actor(self.economic_center, "query_balance", hh.household_id)
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

        tasks = []
        for hh in self.households:
            available_balance = balance_by_household.get(hh.household_id)
            expected_income = expected_income_by_household.get(hh.household_id, 0.0)
            available_budget = None
            if available_balance is not None:
                available_budget = float(available_balance) + float(max(0.0, expected_income))
            tasks.append(
                hh.consume_v2(
                    top_k=top_k,
                    product_market=self.product_market,
                    available_balance=available_balance,
                    expected_income=expected_income,
                    available_budget=available_budget,
                )
            )
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        for hh, out in zip(self.households, outputs):
            if isinstance(out, Exception):
                logger.error(f"Household {hh.household_id} consumption failed: {out}")
                continue
            if not isinstance(out, dict):
                continue
            results.append((hh, out))
        if self._debug_enabled():
            self._log_consumption_plans(results)
            self._log_consumption_budget_status(results, balance_by_household, expected_income_by_household)
        return results

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
        orders_by_household: List[Tuple[Household, List[Dict[str, Any]]]] = []
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

            orders_by_household.append((hh, purchases))

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
        production_stats = {"total_qty": 0.0, "total_value": 0.0, "by_firm": {}}
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
            if record_transactions:
                if self.economic_center is not None:
                    self._call_actor(self.economic_center, "register_market_price", "intermediate_goods", firm.industry, firm.firm_id, 1.0)
                firm.produce(
                    production_plan=plan,
                    sku_base_prices=sku_base_prices,
                    period=month,
                    update_inventory=True,
                )
            else:
                self._manual_produce(firm, plan, sku_base_prices, month)
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

    def _resolve_seller_firm(self, order: Dict[str, Any]) -> Optional[Firm]:
        retailer_code = order.get("retailer_code")
        if retailer_code and retailer_code in self.retailers_by_industry:
            return self.retailers_by_industry[retailer_code]
        manufacturer_code = order.get("manufacturer_code")
        if manufacturer_code and manufacturer_code in self.manufacturers_by_industry:
            return self.manufacturers_by_industry[manufacturer_code]
        return None

    def _execute_orders(
        self,
        orders_by_household: List[Tuple[Household, List[Dict[str, Any]]]],
        snapshot_cache: Dict[str, Optional[Dict[str, Any]]],
        month: int,
        record_transactions: bool
    ) -> Dict[str, Any]:
        consumption_by_household: Dict[str, Dict[str, float]] = defaultdict(lambda: {"qty": 0.0, "value": 0.0})
        consumption_by_sku: Dict[str, Dict[str, float]] = defaultdict(lambda: {"qty": 0.0, "value": 0.0})
        revenue_by_firm: Dict[str, Dict[str, float]] = defaultdict(lambda: {"qty": 0.0, "value": 0.0})
        price_index_stats = {"base_value": 0.0, "current_value": 0.0, "index": None}
        remaining_balance_by_household: Dict[str, float] = {}
        tax_multiplier = 1.0
        if record_transactions and self.economic_center is not None:
            tax_multiplier = 1.0 + float(getattr(self.config, "vat_rate", 0.0) or 0.0)
            if self._last_balance_by_household:
                remaining_balance_by_household = dict(self._last_balance_by_household)
            else:
                for hh, _ in orders_by_household:
                    bal = self._call_actor(self.economic_center, "query_balance", hh.household_id)
                    remaining_balance_by_household[hh.household_id] = float(bal or 0.0)
        for hh, orders in orders_by_household:
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
                if remaining_balance_by_household:
                    remaining = float(remaining_balance_by_household.get(hh.household_id, 0.0) or 0.0)
                    max_affordable = int(remaining // (unit_price * tax_multiplier)) if tax_multiplier > 0 else 0
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
                        if remaining_balance_by_household:
                            remaining_balance_by_household[hh.household_id] = float(
                                remaining_balance_by_household.get(hh.household_id, 0.0) - amount * tax_multiplier
                            )
                else:
                    self._call_actor(self.product_market, "update_stock", product_id, -qty)
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
        base_value = float(price_index_stats.get("base_value") or 0.0)
        current_value = float(price_index_stats.get("current_value") or 0.0)
        if base_value > 0:
            price_index_stats["index"] = current_value / base_value
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

    async def _post_jobs(self, month: int) -> None:
        tasks = [firm.post_jobs(period=month) for firm in (self.firms or [])]
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
        if self._debug_enabled():
            matched = self._call_actor(self.labor_market, "get_matched_jobs") or []
            self._log_matching_summary(matched)

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
                firm = self._firm_by_id.get(firm_id)
                if firm is not None:
                    firm.cash -= gross
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
    ) -> None:
        if not self._recording_enabled():
            return
        labor_summary = self._call_actor(self.labor_market, "summary") or {}
        firm_financials = {}
        gdp_stats = {}
        tax_stats = {}
        household_summary = {}
        redistribution_per_person = 0.0
        if self.economic_center is not None:
            firm_financials = self._call_actor(self.economic_center, "query_all_firms_monthly_financials", econ_month)
            gdp_stats = self._call_actor(self.economic_center, "calculate_monthly_gdp", econ_month, production_stats)
            tax_stats = self._call_actor(self.economic_center, "get_monthly_tax_collection", econ_month)
            household_summary = self._call_actor(self.economic_center, "summarize_households_monthly", econ_month)
            redistribution_per_person = self._call_actor(
                self.economic_center, "query_redistribution_record_per_person", econ_month
            )

        price_index = {}
        if consumption_stats:
            price_index = dict(consumption_stats.get("price_index") or {})
        current_index = price_index.get("index")
        inflation_rate = None
        if current_index is not None:
            prev = self._last_price_index
            if prev is not None:
                inflation_rate = (float(current_index) / float(prev)) - 1.0
            self._last_price_index = float(current_index)
        price_index["inflation_rate"] = inflation_rate

        household_agg = (household_summary or {}).get("aggregate", {}) or {}
        household_by = (household_summary or {}).get("by_household", {}) or {}
        balances = [rec.get("balance") for rec in household_by.values()]
        assets_stats = self._calc_distribution_stats([float(v or 0.0) for v in balances])
        assets_stats["gini"] = self._calc_gini([float(v or 0.0) for v in balances])

        total_labor = float(labor_summary.get("total_labor_hours", 0.0) or 0.0)
        employed_labor = float(labor_summary.get("total_matched_jobs", 0.0) or 0.0)
        employment_rate = employed_labor / total_labor if total_labor > 0 else 0.0
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
                },
                "by_household": household_by,
            },
            "labor_market": {
                "total_labor": total_labor,
                "employed_labor": employed_labor,
                "employment_rate": employment_rate,
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
            },
            "macro": {
                "gdp": gdp_stats or {},
                "gdp_value": float(((gdp_stats or {}).get("gdp", {}) or {}).get("production_approach", 0.0) or 0.0),
                "price_index": price_index,
                "inflation_rate": inflation_rate,
            },
            "details": {
                "labor_market_raw": labor_summary,
                "demand": demand_stats or {},
                "production": production_stats or {},
                "consumption": consumption_stats or {},
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
        
