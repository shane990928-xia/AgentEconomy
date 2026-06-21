"""
Checkpoint System for Simulation State Persistence

支持断点续模拟：
1. save_checkpoint() - 保存完整状态快照
2. load_checkpoint() - 从快照恢复状态
3. resume_simulation() - 从指定月份继续模拟
"""

import json
import os
import gzip
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import ray

from agenteconomy.utils.logger import get_logger

if TYPE_CHECKING:
    from agenteconomy.simulation.simulator import Simulator
    from agenteconomy.agent.household import Household
    from agenteconomy.agent.firm import Firm, ManufactureFirm, RetailFirm
    from agenteconomy.agent.government import Government
    from agenteconomy.agent.bank import Bank

logger = get_logger(name="checkpoint")


class CheckpointManager:
    """管理模拟器状态的保存和恢复"""
    
    VERSION = "1.0"  # Checkpoint 格式版本
    
    def __init__(self, checkpoint_dir: str, compress: bool = True):
        """
        Args:
            checkpoint_dir: Checkpoint 保存目录
            compress: 是否压缩（使用 gzip）
        """
        self.checkpoint_dir = checkpoint_dir
        self.compress = compress
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def _get_checkpoint_path(self, month: int, preheat: bool = False) -> str:
        """获取 checkpoint 文件路径"""
        prefix = "preheat" if preheat else "month"
        ext = ".json.gz" if self.compress else ".json"
        return os.path.join(self.checkpoint_dir, f"checkpoint_{prefix}_{month:04d}{ext}")
    
    def _serialize_household(self, hh: 'Household') -> Dict[str, Any]:
        """序列化 Household 状态"""
        return {
            "household_id": hh.household_id,
            "name": hh.name,
            "description": hh.description,
            "owner": hh.owner,
            "csv_values": dict(hh.csv_values or {}),
            "csv_raw": dict(hh.csv_raw or {}),
            "persona": hh.persona,
            "past_household_status": hh.past_household_status,
            "past_household_status_text": hh.past_household_status_text,
            "consumption_categories": list(hh.consumption_categories or []),
            "household_info": dict(hh.household_info or {}),
            # 就业状态相关
            "ER82433": getattr(hh, "ER82433", None),
            "SP_employment_status": getattr(hh, "SP_employment_status", None),
            "ER82181": getattr(hh, "ER82181", None),
            "ER82500": getattr(hh, "ER82500", None),
            "RP_income": getattr(hh, "RP_income", 0.0),
            "SP_income": getattr(hh, "SP_income", 0.0),
            # 消费惯性相关
            "_last_month_consumption": getattr(hh, "_last_month_consumption", 0.0),
        }
    
    def _deserialize_household(self, hh: 'Household', data: Dict[str, Any]) -> None:
        """反序列化 Household 状态"""
        hh.csv_values = dict(data.get("csv_values") or {})
        hh.csv_raw = dict(data.get("csv_raw") or {})
        hh.persona = data.get("persona")
        hh.past_household_status = data.get("past_household_status") or {}
        hh.past_household_status_text = data.get("past_household_status_text") or ""
        hh.consumption_categories = list(data.get("consumption_categories") or hh.consumption_categories)
        hh.household_info = dict(data.get("household_info") or {})
        
        # 就业状态
        if data.get("ER82433") is not None:
            hh.ER82433 = data["ER82433"]
            hh.csv_values["ER82433"] = data["ER82433"]
        if data.get("SP_employment_status") is not None:
            hh.SP_employment_status = data["SP_employment_status"]
            hh.csv_values["SP_employment_status"] = data["SP_employment_status"]
        if data.get("ER82181") is not None:
            hh.ER82181 = data["ER82181"]
            hh.csv_values["ER82181"] = data["ER82181"]
        if data.get("ER82500") is not None:
            hh.ER82500 = data["ER82500"]
            hh.csv_values["ER82500"] = data["ER82500"]
        hh.RP_income = float(data.get("RP_income") or 0.0)
        hh.SP_income = float(data.get("SP_income") or 0.0)
        
        # 消费惯性
        hh._last_month_consumption = float(data.get("_last_month_consumption") or 0.0)
    
    def _serialize_firm(self, firm: 'Firm') -> Dict[str, Any]:
        """序列化 Firm 状态"""
        data = {
            "firm_id": firm.firm_id,
            "name": firm.name,
            "description": firm.description,
            "industry": firm.industry,
            "industry_type": firm.industry_type,
            "employee_count": firm.employee_count,
            "capital_stock": firm.capital_stock,
            "cash": firm.cash,
            "cost_structure": firm.cost_structure,
            "compensation_ratio": firm.compensation_ratio,
            "current_period": firm.current_period,
            "production_history": list(firm.production_history or []),
            "is_agent": firm.is_agent,
        }
        
        # ManufactureFirm 特有字段
        if hasattr(firm, "inventory"):
            data["inventory"] = dict(getattr(firm, "inventory", {}) or {})
        if hasattr(firm, "monthly_production_target"):
            data["monthly_production_target"] = getattr(firm, "monthly_production_target", 0.0)
        if hasattr(firm, "unit_cost"):
            data["unit_cost"] = getattr(firm, "unit_cost", 0.0)
        if hasattr(firm, "suppliers"):
            data["suppliers"] = getattr(firm, "suppliers", None)
        
        # RetailFirm 特有字段
        if hasattr(firm, "retail_inventory"):
            data["retail_inventory"] = dict(getattr(firm, "retail_inventory", {}) or {})
        if hasattr(firm, "supply_chain"):
            data["supply_chain"] = getattr(firm, "supply_chain", None)
            
        return data
    
    def _deserialize_firm(self, firm: 'Firm', data: Dict[str, Any]) -> None:
        """反序列化 Firm 状态"""
        firm.employee_count = int(data.get("employee_count") or 0)
        firm.capital_stock = float(data.get("capital_stock") or 0.0)
        firm.cash = float(data.get("cash") or 0.0)
        firm.cost_structure = data.get("cost_structure")
        firm.compensation_ratio = float(data.get("compensation_ratio") or 0.2)
        firm.current_period = int(data.get("current_period") or 0)
        firm.production_history = list(data.get("production_history") or [])
        
        # ManufactureFirm 特有字段
        if hasattr(firm, "inventory") and "inventory" in data:
            firm.inventory = dict(data.get("inventory") or {})
        if hasattr(firm, "monthly_production_target") and "monthly_production_target" in data:
            firm.monthly_production_target = float(data.get("monthly_production_target") or 0.0)
        if hasattr(firm, "unit_cost") and "unit_cost" in data:
            firm.unit_cost = float(data.get("unit_cost") or 0.0)
        if hasattr(firm, "suppliers") and "suppliers" in data:
            firm.suppliers = data.get("suppliers")
        
        # RetailFirm 特有字段
        if hasattr(firm, "retail_inventory") and "retail_inventory" in data:
            firm.retail_inventory = dict(data.get("retail_inventory") or {})
        if hasattr(firm, "supply_chain") and "supply_chain" in data:
            firm.supply_chain = data.get("supply_chain")
    
    def _serialize_government(self, gov: 'Government') -> Dict[str, Any]:
        """序列化 Government 状态"""
        return {
            "government_id": gov.government_id,
            "budget": float(getattr(gov, "budget", 0.0) or 0.0),
            "total_tax_collected": float(getattr(gov, "total_tax_collected", 0.0) or 0.0),
            "total_redistributed": float(getattr(gov, "total_redistributed", 0.0) or 0.0),
            "tax_history": list(getattr(gov, "tax_history", []) or []),
            "redistribution_history": list(getattr(gov, "redistribution_history", []) or []),
            "procurement_history": list(getattr(gov, "procurement_history", []) or []),
        }
    
    def _deserialize_government(self, gov: 'Government', data: Dict[str, Any]) -> None:
        """反序列化 Government 状态"""
        gov.budget = float(data.get("budget") or 0.0)
        if hasattr(gov, "total_tax_collected"):
            gov.total_tax_collected = float(data.get("total_tax_collected") or 0.0)
        if hasattr(gov, "total_redistributed"):
            gov.total_redistributed = float(data.get("total_redistributed") or 0.0)
        if hasattr(gov, "tax_history"):
            gov.tax_history = list(data.get("tax_history") or [])
        if hasattr(gov, "redistribution_history"):
            gov.redistribution_history = list(data.get("redistribution_history") or [])
        if hasattr(gov, "procurement_history"):
            gov.procurement_history = list(data.get("procurement_history") or [])
    
    def _serialize_bank(self, bank: 'Bank') -> Dict[str, Any]:
        """序列化 Bank 状态"""
        return {
            "bank_id": bank.bank_id,
            "capital": float(getattr(bank, "capital", 0.0) or 0.0),
            "total_deposits": float(getattr(bank, "total_deposits", 0.0) or 0.0),
            "total_loans": float(getattr(bank, "total_loans", 0.0) or 0.0),
            "interest_rate": float(getattr(bank, "interest_rate", 0.0) or 0.0),
            "loan_rate": float(getattr(bank, "loan_rate", 0.0) or 0.0),
        }
    
    def _deserialize_bank(self, bank: 'Bank', data: Dict[str, Any]) -> None:
        """反序列化 Bank 状态"""
        bank.capital = float(data.get("capital") or 0.0)
        if hasattr(bank, "total_deposits"):
            bank.total_deposits = float(data.get("total_deposits") or 0.0)
        if hasattr(bank, "total_loans"):
            bank.total_loans = float(data.get("total_loans") or 0.0)
        if hasattr(bank, "interest_rate"):
            bank.interest_rate = float(data.get("interest_rate") or 0.0)
        if hasattr(bank, "loan_rate"):
            bank.loan_rate = float(data.get("loan_rate") or 0.0)
    
    def _is_ray_actor(self, obj) -> bool:
        """检查对象是否为 Ray Actor"""
        return obj is not None and "ActorHandle" in str(type(obj))
    
    def _get_ledger_snapshot(self, economic_center) -> Dict[str, float]:
        """获取 EconomicCenter 账本快照"""
        if economic_center is None:
            return {}
        try:
            if self._is_ray_actor(economic_center):
                # Ray Actor
                ledger = ray.get(economic_center.get_all_balances.remote())
            else:
                ledger = economic_center.get_all_balances()
            return {k: float(v) for k, v in (ledger or {}).items()}
        except Exception as e:
            logger.warning(f"Failed to get ledger snapshot: {e}")
            return {}

    def _get_firm_monthly_data_snapshot(self, economic_center) -> Dict[str, Any]:
        """获取企业月度数据快照"""
        if economic_center is None:
            return {}
        try:
            if self._is_ray_actor(economic_center):
                data = ray.get(economic_center.get_firm_monthly_data_snapshot.remote())
            else:
                data = economic_center.get_firm_monthly_data_snapshot()
            return data or {}
        except Exception as e:
            logger.warning(f"Failed to get firm monthly data snapshot: {e}")
            return {}

    def _get_firm_credit_state_snapshot(self, economic_center) -> Dict[str, Any]:
        """获取企业信用状态快照"""
        if economic_center is None:
            return {}
        try:
            if self._is_ray_actor(economic_center):
                data = ray.get(economic_center.get_firm_credit_state_snapshot.remote())
            else:
                data = economic_center.get_firm_credit_state_snapshot()
            return data or {}
        except Exception as e:
            logger.warning(f"Failed to get firm credit state snapshot: {e}")
            return {}
    
    def _get_product_market_snapshot(self, product_market) -> Dict[str, Any]:
        """获取 ProductMarket 快照"""
        if product_market is None:
            return {}
        try:
            if self._is_ray_actor(product_market):
                # Ray Actor - 并行获取所有数据
                stats_future = product_market.get_market_stats.remote()
                products_future = product_market.get_all_products_snapshot.remote()
                market_state_future = product_market.get_market_state_snapshot.remote()
                # 一次性等待所有结果
                stats, products_snapshot, market_state = ray.get([
                    stats_future, products_future, market_state_future
                ])
            else:
                stats = product_market.get_market_stats()
                products_snapshot = product_market.get_all_products_snapshot()
                market_state = product_market.get_market_state_snapshot()
            return {
                "stats": stats or {},
                "products": products_snapshot or [],
                "market_state": market_state or {},
            }
        except Exception as e:
            logger.warning(f"Failed to get product market snapshot: {e}")
            return {}
    
    def _get_labor_market_snapshot(self, labor_market) -> Dict[str, Any]:
        """获取 LaborMarket 快照"""
        if labor_market is None:
            return {}
        try:
            if self._is_ray_actor(labor_market):
                # 并行获取
                summary_future = labor_market.summary.remote()
                matched_jobs_future = labor_market.get_matched_jobs_snapshot.remote()
                summary, matched_jobs = ray.get([summary_future, matched_jobs_future])
            else:
                summary = labor_market.summary()
                matched_jobs = labor_market.get_matched_jobs_snapshot()
            return {
                "summary": summary or {},
                "matched_jobs": matched_jobs or [],
            }
        except Exception as e:
            logger.warning(f"Failed to get labor market snapshot: {e}")
            return {}
    
    def _get_abstract_resource_market_snapshot(self, abstract_resource_market) -> Dict[str, Any]:
        """获取 AbstractResourceMarket 快照"""
        if abstract_resource_market is None:
            return {}
        try:
            # AbstractResourceMarket 不是 Ray Actor，直接调用
            return abstract_resource_market.get_state_snapshot()
        except Exception as e:
            logger.warning(f"Failed to get abstract resource market snapshot: {e}")
            return {}
    
    def save_checkpoint(
        self,
        simulator: 'Simulator',
        month: int,
        preheat: bool = False,
    ) -> str:
        """
        保存完整的模拟状态快照
        
        Args:
            simulator: 模拟器实例
            month: 当前月份
            preheat: 是否是预热阶段
            
        Returns:
            checkpoint 文件路径
        """
        logger.info(f"Saving checkpoint for month {month} (preheat={preheat})...")
        
        checkpoint = {
            "version": self.VERSION,
            "timestamp": datetime.utcnow().isoformat(),
            "month": month,
            "preheat": preheat,
            "config": {
                "num_months": simulator.config.num_months,
                "num_households": simulator.config.num_households,
                "preheat_months": simulator.config.preheat_months,
            },
            
            # Households
            "households": [
                self._serialize_household(hh) for hh in (simulator.households or [])
            ],
            
            # Firms
            "firms": [
                self._serialize_firm(firm) for firm in (simulator.firms or [])
            ],
            
            # Government
            "government": self._serialize_government(simulator.government) if simulator.government else None,
            
            # Bank
            "bank": self._serialize_bank(simulator.bank) if simulator.bank else None,
            
            # EconomicCenter ledger (账户余额)
            "ledger": self._get_ledger_snapshot(simulator.economic_center),

            # EconomicCenter 企业月度数据（用于企业所得税计算）
            "firm_monthly_data": self._get_firm_monthly_data_snapshot(simulator.economic_center),

            # EconomicCenter 企业信用状态（用于恢复融资约束和违约状态）
            "firm_credit_state": self._get_firm_credit_state_snapshot(simulator.economic_center),

            # ProductMarket (产品库存和价格)
            "product_market": self._get_product_market_snapshot(simulator.product_market),
            
            # LaborMarket (雇佣关系)
            "labor_market": self._get_labor_market_snapshot(simulator.labor_market),
            
            # AbstractResourceMarket (服务/资源价格)
            "abstract_resource_market": self._get_abstract_resource_market_snapshot(simulator.abstract_resource_market),
            
            # Simulator 内部状态
            "simulator_state": {
                "current_month": simulator.current_month,
                "_last_price_index": simulator._last_price_index,
                "_last_inflation_rate": getattr(simulator, "_last_inflation_rate", None),
                "_last_balance_by_household": dict(simulator._last_balance_by_household or {}),
                "_last_expected_income_by_household": dict(simulator._last_expected_income_by_household or {}),
                "_last_sales_by_product": dict(simulator._last_sales_by_product or {}),
                "_last_planned_demand_by_product": dict(getattr(simulator, "_last_planned_demand_by_product", {}) or {}),
                "_last_unmet_demand_by_product": dict(getattr(simulator, "_last_unmet_demand_by_product", {}) or {}),
                "_last_production_gap_value_by_firm": dict(
                    getattr(simulator, "_last_production_gap_value_by_firm", {}) or {}
                ),
                "_last_production_value_by_firm": dict(
                    getattr(simulator, "_last_production_value_by_firm", {}) or {}
                ),
                "_last_service_value_by_industry": dict(
                    getattr(simulator, "_last_service_value_by_industry", {}) or {}
                ),
                "_fixed_consumption_basket": dict(getattr(simulator, "_fixed_consumption_basket", None) or {}),
            },
        }
        
        # 保存文件
        path = self._get_checkpoint_path(month, preheat)
        if self.compress:
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(checkpoint, f, ensure_ascii=False, default=str)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Checkpoint saved to {path}")
        return path
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        加载 checkpoint 文件
        
        Args:
            path: checkpoint 文件路径
            
        Returns:
            checkpoint 数据
        """
        logger.info(f"Loading checkpoint from {path}...")
        
        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        
        logger.info(f"Checkpoint loaded: month={data.get('month')}, version={data.get('version')}")
        return data
    
    def restore_simulator(
        self,
        simulator: 'Simulator',
        checkpoint_data: Dict[str, Any],
    ) -> None:
        """
        从 checkpoint 恢复模拟器状态
        
        Args:
            simulator: 模拟器实例
            checkpoint_data: checkpoint 数据
        """
        logger.info("Restoring simulator state from checkpoint...")
        
        # 恢复 Households
        households_data = {d["household_id"]: d for d in checkpoint_data.get("households", [])}
        hh_restored = 0
        for hh in (simulator.households or []):
            if hh.household_id in households_data:
                self._deserialize_household(hh, households_data[hh.household_id])
                hh_restored += 1
        logger.info(f"Restored {hh_restored}/{len(households_data)} households (matched/in checkpoint)")
        if hh_restored < len(households_data):
            logger.warning(f"⚠️ {len(households_data) - hh_restored} households in checkpoint not matched!")
        
        # 恢复 Firms
        firms_data = {d["firm_id"]: d for d in checkpoint_data.get("firms", [])}
        firms_restored = 0
        for firm in (simulator.firms or []):
            if firm.firm_id in firms_data:
                self._deserialize_firm(firm, firms_data[firm.firm_id])
                firms_restored += 1
        logger.info(f"Restored {firms_restored}/{len(firms_data)} firms (matched/in checkpoint)")
        if firms_restored < len(firms_data):
            logger.warning(
                f"⚠️ {len(firms_data) - firms_restored} firms in checkpoint not matched! "
                f"Checkpoint IDs sample: {list(firms_data.keys())[:3]}, "
                f"Current IDs sample: {[f.firm_id for f in (simulator.firms or [])[:3]]}"
            )
        
        # 恢复 Government
        if simulator.government and checkpoint_data.get("government"):
            self._deserialize_government(simulator.government, checkpoint_data["government"])
            logger.info("Restored government")
        
        # 恢复 Bank
        if simulator.bank and checkpoint_data.get("bank"):
            self._deserialize_bank(simulator.bank, checkpoint_data["bank"])
            logger.info("Restored bank")
        
        # 并行恢复各个组件状态（使用批量 ray.get）
        ledger_data = checkpoint_data.get("ledger", {})
        firm_monthly_data = checkpoint_data.get("firm_monthly_data", {})
        firm_credit_state = checkpoint_data.get("firm_credit_state", {})
        product_market_data = checkpoint_data.get("product_market", {})
        labor_market_data = checkpoint_data.get("labor_market", {})
        
        # 收集所有需要执行的恢复操作（Ray Actor 远程调用）
        restore_futures = []
        restore_names = []
        
        # EconomicCenter 恢复
        if simulator.economic_center and self._is_ray_actor(simulator.economic_center):
            if ledger_data:
                restore_futures.append(simulator.economic_center.restore_balances.remote(ledger_data))
                restore_names.append(f"ledger ({len(ledger_data)} accounts)")
            if firm_monthly_data:
                restore_futures.append(simulator.economic_center.restore_firm_monthly_data.remote(firm_monthly_data))
                restore_names.append(f"firm monthly data ({len(firm_monthly_data)} firms)")
            if firm_credit_state:
                restore_futures.append(simulator.economic_center.restore_firm_credit_state.remote(firm_credit_state))
                restore_names.append("firm credit state")
        
        # ProductMarket 恢复
        if simulator.product_market and self._is_ray_actor(simulator.product_market):
            if product_market_data.get("products"):
                restore_futures.append(simulator.product_market.restore_products_snapshot.remote(
                    product_market_data["products"]
                ))
                restore_names.append("product market products")
            if product_market_data.get("market_state"):
                restore_futures.append(simulator.product_market.restore_market_state.remote(
                    product_market_data["market_state"]
                ))
                restore_names.append("product market state")
        
        # LaborMarket 恢复
        if simulator.labor_market and self._is_ray_actor(simulator.labor_market):
            if labor_market_data.get("matched_jobs"):
                restore_futures.append(simulator.labor_market.restore_matched_jobs.remote(
                    labor_market_data["matched_jobs"]
                ))
                restore_names.append("labor market")
        
        # 批量等待所有恢复操作完成
        if restore_futures:
            try:
                ray.get(restore_futures)
                for name in restore_names:
                    logger.info(f"Restored {name}")
            except Exception as e:
                logger.warning(f"Some restore operations failed: {e}")
        
        # 非 Ray Actor 的恢复（顺序执行）
        if simulator.economic_center and not self._is_ray_actor(simulator.economic_center):
            if ledger_data:
                simulator.economic_center.restore_balances(ledger_data)
            if firm_monthly_data:
                simulator.economic_center.restore_firm_monthly_data(firm_monthly_data)
            if firm_credit_state:
                simulator.economic_center.restore_firm_credit_state(firm_credit_state)
        
        if simulator.product_market and not self._is_ray_actor(simulator.product_market):
            if product_market_data.get("products"):
                simulator.product_market.restore_products_snapshot(product_market_data["products"])
            if product_market_data.get("market_state"):
                simulator.product_market.restore_market_state(product_market_data["market_state"])
        
        if simulator.labor_market and not self._is_ray_actor(simulator.labor_market):
            if labor_market_data.get("matched_jobs"):
                simulator.labor_market.restore_matched_jobs(labor_market_data["matched_jobs"])
        
        # 恢复 AbstractResourceMarket (服务/资源价格)
        abstract_resource_market_data = checkpoint_data.get("abstract_resource_market", {})
        if abstract_resource_market_data and simulator.abstract_resource_market:
            try:
                # AbstractResourceMarket 不是 Ray Actor，直接调用
                simulator.abstract_resource_market.restore_state(abstract_resource_market_data)
                logger.info(f"Restored abstract resource market state")
            except Exception as e:
                logger.warning(f"Failed to restore abstract resource market: {e}")
        
        # 恢复 Simulator 内部状态
        sim_state = checkpoint_data.get("simulator_state", {})
        simulator.current_month = int(sim_state.get("current_month") or checkpoint_data.get("month", 1))
        simulator._last_price_index = sim_state.get("_last_price_index")
        simulator._last_inflation_rate = sim_state.get("_last_inflation_rate")
        simulator._last_balance_by_household = dict(sim_state.get("_last_balance_by_household") or {})
        simulator._last_expected_income_by_household = dict(sim_state.get("_last_expected_income_by_household") or {})
        simulator._last_sales_by_product = dict(sim_state.get("_last_sales_by_product") or {})
        simulator._last_planned_demand_by_product = dict(sim_state.get("_last_planned_demand_by_product") or {})
        simulator._last_unmet_demand_by_product = dict(sim_state.get("_last_unmet_demand_by_product") or {})
        simulator._last_production_gap_value_by_firm = dict(
            sim_state.get("_last_production_gap_value_by_firm") or {}
        )
        simulator._last_production_value_by_firm = dict(
            sim_state.get("_last_production_value_by_firm") or {}
        )
        simulator._last_service_value_by_industry = dict(
            sim_state.get("_last_service_value_by_industry") or {}
        )
        # 恢复固定消费篮子
        basket = sim_state.get("_fixed_consumption_basket")
        if basket:
            simulator._fixed_consumption_basket = dict(basket)
        else:
            simulator._fixed_consumption_basket = None
        
        logger.info(f"Simulator state restored. Ready to resume from month {simulator.current_month}")
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        列出所有可用的 checkpoint
        
        Returns:
            checkpoint 信息列表
        """
        checkpoints = []
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith("checkpoint_") and (filename.endswith(".json") or filename.endswith(".json.gz")):
                path = os.path.join(self.checkpoint_dir, filename)
                try:
                    data = self.load_checkpoint(path)
                    checkpoints.append({
                        "path": path,
                        "filename": filename,
                        "month": data.get("month"),
                        "preheat": data.get("preheat"),
                        "timestamp": data.get("timestamp"),
                        "version": data.get("version"),
                    })
                except Exception as e:
                    logger.warning(f"Failed to read checkpoint {path}: {e}")
        
        # 按月份排序
        checkpoints.sort(key=lambda x: (x.get("preheat", False), x.get("month", 0)))
        return checkpoints
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        获取最新的 checkpoint 路径
        
        Returns:
            最新 checkpoint 的路径，如果没有则返回 None
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        return checkpoints[-1]["path"]
