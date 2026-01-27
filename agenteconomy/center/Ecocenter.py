"""
Economic Center Module

This module implements the central economic system that manages all economic activities
in the agent-based economic simulation, including:

- Asset Management: Ledgers, products, labor hours, capital stocks
- Transaction Processing: Purchases, labor payments, taxes
- Tax System: Progressive income tax, corporate tax, VAT
- Firm Finances: Revenue, expenses, depreciation, innovation
- Job Market: Job postings, applications, matching
- Inventory Management: Reservations, stock tracking
- GDP Calculation: Production-based and expenditure-based GDP
- Government Operations: Tax collection and redistribution

Key Components:
    - EconomicCenter: Main class coordinating all economic activities
"""

import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TYPE_CHECKING
from uuid import uuid4

import numpy as np
import ray
from dotenv import load_dotenv

from agenteconomy.center.Model import *
from agenteconomy.center.transaction import PeriodStatistics
from agenteconomy.utils.logger import get_logger
from agenteconomy.utils.product_attribute_loader import inject_product_attributes

# Avoid circular import by using TYPE_CHECKING
if TYPE_CHECKING:
    from agenteconomy.agent.firm import Firm
# Initialize environment and logger
load_dotenv()


# =============================================================================
# Economic Center Class
# =============================================================================

@ray.remote(num_cpus=8)
class EconomicCenter:

    # =========================================================================
    # Initialization
    # =========================================================================
    def __init__(self, tax_policy: TaxPolicy = None, category_profit_margins: Dict[str, float] = None):
        """
        Initialize EconomicCenter with tax rates
        
        Args:
            tax_policy: 税收政策配置（包含累进税阶梯）
            category_profit_margins: 各行业毛利率配置
        """
        self.logger = get_logger(name="economic_center")
        
        # =========================================================================
        # 1️⃣ 税收配置 (Tax Configuration)
        # =========================================================================
        # Tax rate configuration - if not provided, use default value
        if tax_policy is None:
            tax_policy = TaxPolicy()  # Use default configuration
        
        self.income_tax_rate = tax_policy.income_tax_rate       # List[TaxBracket] - 累进税阶梯
        self.vat_rate = tax_policy.vat_rate                     # float - 消费税率
        self.corporate_tax_rate = tax_policy.corporate_tax_rate # float - 企业所得税率

        # =========================================================================
        # 2️⃣ 商品毛利率配置 (Category Profit Margins)
        # =========================================================================
        if category_profit_margins is None:
            self.category_profit_margins = {
                "Beverages": 25.0,                              # Beverages
                "Confectionery and Snacks": 32.0,               # Confectionery and Snacks
                "Dairy Products": 15.0,                         # Dairy Products
                "Furniture and Home Furnishing": 30.0,          # Furniture and Home Furnishing
                "Garden and Outdoor": 28.0,                     # Garden and Outdoor
                "Grains and Bakery": 18.0,                      # Grains and Bakery
                "Household Appliances and Equipment": 30.0,     # Household Appliances and Equipment
                "Meat and Seafood": 16.0,                       # Meat and Seafood
                "Personal Care and Cleaning": 40.0,            # Personal Care and Cleaning
                "Pharmaceuticals and Health": 45.0,            # Pharmaceuticals and Health
                "Retail and Stores": 25.0,                      # Retail and Stores
                "Sugars, Oils, and Seasonings": 20.0,           # Sugars, Oils, and Seasonings
            }
        else:
            self.category_profit_margins = category_profit_margins

        # =========================================================================
        # 3️⃣ 资产存储 (Asset Storage)
        # =========================================================================
        self.ledger: Dict[str, Ledger] = defaultdict(Ledger)            # 现金账本
        self.products: Dict[str, List[Product]] = defaultdict(list)     # 商品库存
        self.laborhour: Dict[str, List[LaborHour]] = defaultdict(list)  # 劳动力

        # =========================================================================
        # 4️⃣ Agent ID 注册表 (Agent ID Registry)
        # =========================================================================
        self.firm_id: List[str] = []        # 企业 ID 列表
        self.government_id: List[str] = []  # 政府 ID 列表
        self.household_id: List[str] = []   # 家庭 ID 列表
        self.bank_id: List[str] = []        # 银行 ID 列表
        
        # =========================================================================
        # 5️⃣ 交易记录 (Transaction History)
        # =========================================================================
        self.tx_history: List[Transaction] = []  # Store transaction history
        self.tx_by_month: Dict[int, List[Transaction]] = defaultdict(list)
        self.tx_by_type: Dict[str, List[Transaction]] = defaultdict(list)
        self.tx_by_party: Dict[str, List[Transaction]] = defaultdict(list)
        self.period_statistics: Dict[int, PeriodStatistics] = {}
        self.wage_history: List[Wage] = []
        self.redistribution_record_per_person: Dict[int, float] = defaultdict(float)
        
        # =========================================================================
        # 6️⃣ 企业财务追踪 (Firm Financial Tracking)
        # =========================================================================
        self.firm_financials: Dict[str, Dict[str, float]] = defaultdict(lambda: {"total_income": 0.0, "total_expenses": 0.0})
        self.firm_monthly_financials: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: {"income": 0.0, "expenses": 0.0}))
        self.firm_production_stats: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: {"base_production": 0.0, "labor_production": 0.0}))
        self.firm_monthly_wage_expenses: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.firm_monthly_corporate_tax: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.firm_monthly_production_cost: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.firm_monthly_labor_production_value: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self._corporate_tax_settled_months: set[int] = set()

        # =========================================================================
        # 7️⃣ 创新系统 (Innovation System)
        # =========================================================================
        self.enable_innovation_module: bool = False
        self.firm_innovation_strategy: Dict[str, str] = {}
        self.firm_research_share: List[Dict[str, Tuple[float, int]]] = []
        self.firm_innovation_config: Dict[str, FirmInnovationConfig] = {}
        self.firm_innovation_events: List[FirmInnovationEvent] = []

        # =========================================================================
        # 8️⃣ 库存预留系统 (Inventory Reservation System)
        # =========================================================================
        self.inventory_reservations: Dict[str, InventoryReservation] = {}
        self.reservation_timeout: float = 300.0

        # =========================================================================
        # 9️⃣ 未满足需求追踪 (Unmet Demand Tracking)
        # =========================================================================
        self.unmet_demand_by_month: Dict[int, Dict[str, Dict[str, float]]] = defaultdict(dict)

        # =========================================================================
        # 🔟 企业资本与折旧 (Firm Capital & Depreciation)
        # =========================================================================
        self.firm_capital_stock: Dict[str, float] = defaultdict(float)
        self.firm_monthly_depreciation: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.firm_monthly_capital_investment: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.firm_capital_stock_history: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

        # =========================================================================
        # 1️⃣1️⃣ 企业实例列表 (Firm Instance List)
        # =========================================================================
        self.firm: List['Firm'] = []

        # =========================================================================
        # 📋 初始化日志
        # =========================================================================
        # Initialize log
        print(f"EconomicCenter initialized with tax policy:")
        print(f"  📊 个人所得税: 累进税制 ({len(self.income_tax_rate)} 档)")
        for i, bracket in enumerate(self.income_tax_rate):
            if i + 1 < len(self.income_tax_rate):
                print(f"     档位{i+1}: ${bracket.cutoff:>8,.0f} - ${self.income_tax_rate[i+1].cutoff:>8,.0f} → {bracket.rate:>5.1%}")
            else:
                print(f"     档位{i+1}: ${bracket.cutoff:>8,.0f}+          → {bracket.rate:>5.1%}")
        print(f"  💼 企业所得税: {self.corporate_tax_rate:.1%} (固定税率)")
        print(f"  🛒 消费税(VAT): {self.vat_rate:.1%}")

        # =========================================================================
        # 1️⃣2️⃣ CD 生产函数校准 (CD Production Function Calibration)
        # =========================================================================
        self._cd_calibration: Dict[str, Any] = {}
        self._cd_industry_A: Dict[str, float] = {}
        self._cd_industry_K_tot: Dict[str, float] = {}
        self._cd_firm_K: Dict[str, float] = {}
        self._cd_firm_A: Dict[str, float] = {}

    def register_firm(self, firm: 'Firm'):
        """
        Register a firm in the economic center.
        
        Args:
            firm: Firm instance to register
        """
        # Import here to avoid circular import
        from agenteconomy.agent.firm import Firm as FirmType
        if not isinstance(firm, FirmType):
            raise TypeError(f"Expected Firm instance, got {type(firm)}")
        self.firm.append(firm)
        
    @staticmethod
    def _monthly_rate_from_annual(annual_rate: float) -> float:
        """
        Geometric conversion:
            r_m = 1 - (1 - r_a)^(1/12)
        """
        try:
            r = float(annual_rate or 0.0)
        except Exception:
            r = 0.0
        r = max(0.0, min(0.99, r))
        return float(1.0 - ((1.0 - r) ** (1.0 / 12.0)))


    # =========================================================================
    # Firm Asset Management (Capital & Inventory)
    # =========================================================================
    def register_firm_assets(self, allocations: Dict[str, Dict[str, float]], overwrite_cash: bool = True, overwrite_capital: bool = True) -> Dict[str, float]:
        """
        批量注册企业“资本存量(K) + 现金(Cash)”。

        allocations:
            {firm_id: {"capital_stock": float, "cash": float}}
        """
        if not isinstance(allocations, dict):
            return {"firms_updated": 0, "capital_total": 0.0, "cash_total": 0.0}

        firms_updated = 0
        cap_total = 0.0
        cash_total = 0.0

        for cid, rec in allocations.items():
            firm_id = str(cid)
            if not firm_id:
                continue
            if firm_id not in self.ledger:
                self.ledger[firm_id] = Ledger.create(firm_id, 0.0)
            if firm_id not in self.firm_id:
                # 确保是企业ID（避免因为初始化顺序导致漏注册）
                self.firm_id.append(firm_id)

            try:
                cap = float((rec or {}).get("capital_stock", 0.0) or 0.0)
            except Exception:
                cap = 0.0
            try:
                cash = float((rec or {}).get("cash", 0.0) or 0.0)
            except Exception:
                cash = 0.0

            if overwrite_capital:
                self.firm_capital_stock[firm_id] = max(0.0, cap)
            if overwrite_cash:
                self.ledger[firm_id].amount = float(cash)

            firms_updated += 1
            cap_total += max(0.0, cap)
            cash_total += cash

        return {"firms_updated": int(firms_updated), "capital_total": float(cap_total), "cash_total": float(cash_total)}

    def overwrite_product_amounts(
        self,
        inventory_by_firm: Dict[str, Dict[str, float]],
        set_unmentioned_to_zero: bool = False,
    ) -> Dict[str, Any]:
        """
        Overwrite inventory amounts for existing products.

        inventory_by_firm:
            {firm_id: {product_id: amount}}
        """
        if not isinstance(inventory_by_firm, dict):
            return {"firms_updated": 0, "products_updated": 0, "products_missing": 0}

        firms_updated = 0
        products_updated = 0
        products_missing = 0

        for cid, prod_map in (inventory_by_firm or {}).items():
            firm_id = str(cid or "")
            if not firm_id or not isinstance(prod_map, dict):
                continue

            if firm_id not in self.products:
                # Inventory overwrite assumes products are already registered for the firm
                self.products[firm_id] = []

            existing = {str(getattr(p, "product_id", "") or ""): p for p in (self.products.get(firm_id) or [])}

            touched = set()
            for pid, qty in (prod_map or {}).items():
                product_id = str(pid or "")
                if not product_id:
                    continue
                touched.add(product_id)
                try:
                    amount = float(qty or 0.0)
                except Exception:
                    amount = 0.0
                if amount < 0:
                    amount = 0.0

                if product_id in existing:
                    try:
                        existing[product_id].amount = amount
                        products_updated += 1
                    except Exception:
                        products_missing += 1
                else:
                    products_missing += 1

            if set_unmentioned_to_zero:
                for pid, p in existing.items():
                    if pid and pid not in touched:
                        try:
                            p.amount = 0.0
                            products_updated += 1
                        except Exception:
                            continue

            firms_updated += 1

        return {
            "firms_updated": int(firms_updated),
            "products_updated": int(products_updated),
            "products_missing": int(products_missing),
        }

    def query_firm_assets(self, firm_id: str) -> Dict[str, float]:
        cid = str(firm_id or "")
        if not cid:
            return {"capital_stock": 0.0, "cash_balance": 0.0, "net_assets": 0.0}
        if cid not in self.ledger:
            self.ledger[cid] = Ledger.create(cid, 0.0)
        capital = float(self.firm_capital_stock.get(cid, 0.0) or 0.0)
        cash = float(self.ledger[cid].amount or 0.0)
        return {"capital_stock": capital, "cash_balance": cash, "net_assets": float(capital + cash)}

    def query_all_firm_assets(self) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for cid in list(self.firm_id or []):
            result[str(cid)] = self.query_firm_assets(str(cid))
        return result

    def invest_in_capital(self, firm_id: str, amount: float, month: int, allow_negative_cash: bool = True) -> Dict[str, float]:
        """
        Capital investment (capex):
        - Decrease firm cash balance (ledger) by `amount`
        - Increase firm capital stock K by `amount`
        - Record monthly capex for reporting

        Note: does NOT count as current-period expense (depreciation will amortize).
        """
        cid = str(firm_id or "")
        try:
            m = int(month or 0)
        except Exception:
            m = 0
        if not cid or m <= 0:
            return {"invested": 0.0, "capital_stock": 0.0, "cash_balance": 0.0}

        try:
            amt = float(amount or 0.0)
        except Exception:
            amt = 0.0
        if amt <= 0:
            assets = self.query_firm_assets(cid)
            assets["invested"] = 0.0
            return assets

        if cid not in self.ledger:
            self.ledger[cid] = Ledger.create(cid, 0.0)
        cash = float(self.ledger[cid].amount or 0.0)
        if (not allow_negative_cash) and cash < amt:
            amt = max(0.0, cash)
        if amt <= 0:
            assets = self.query_firm_assets(cid)
            assets["invested"] = 0.0
            return assets

        # Cash outflow
        self.ledger[cid].amount -= amt
        # Capital stock inflow
        k0 = float(self.firm_capital_stock.get(cid, 0.0) or 0.0)
        k1 = max(0.0, k0 + amt)
        self.firm_capital_stock[cid] = k1
        self.firm_capital_stock_history[cid][m] = k1
        self.firm_monthly_capital_investment[cid][m] += amt

        return {"invested": float(amt), "capital_stock": float(k1), "cash_balance": float(self.ledger[cid].amount or 0.0)}

    def apply_monthly_depreciation(self, month: int, annual_depreciation_rate: float = 0.08, reduce_capital_stock: bool = True) -> Dict[str, float]:
        """
        对所有企业计提月度折旧：
        - 折旧费用计入 firm_monthly_financials[month]["expenses"]（用于企业税基/利润口径）
        - 默认同时减少 firm_capital_stock（K_{t+1} = (1-δ_m)K_t）
        """
        try:
            m = int(month or 0)
        except Exception:
            m = 0
        if m <= 0:
            return {"depreciation_total": 0.0, "firms": 0}

        r_m = self._monthly_rate_from_annual(annual_depreciation_rate)
        if r_m <= 0:
            return {"depreciation_total": 0.0, "firms": 0}

        total_dep = 0.0
        firms = 0
        for cid in list(self.firm_id or []):
            firm_id = str(cid)
            k0 = float(self.firm_capital_stock.get(firm_id, 0.0) or 0.0)
            if k0 <= 0:
                continue
            dep = float(k0 * r_m)
            if dep <= 1e-12:
                continue

            self.firm_monthly_depreciation[firm_id][m] += dep
            total_dep += dep
            firms += 1

            # 费用发生制：折旧计入支出（不扣现金）
            self.record_firm_expense(firm_id, dep)
            self.record_firm_monthly_expense(firm_id, m, dep)

            if reduce_capital_stock:
                k1 = max(0.0, k0 - dep)
                self.firm_capital_stock[firm_id] = k1
                self.firm_capital_stock_history[firm_id][m] = k1
            else:
                self.firm_capital_stock_history[firm_id][m] = k0

        return {"depreciation_total": float(total_dep), "firms": int(firms), "monthly_rate": float(r_m)}

    def query_firm_monthly_depreciation(self, firm_id: str, month: int) -> float:
        try:
            m = int(month or 0)
        except Exception:
            m = 0
        if m <= 0:
            return 0.0
        return float(self.firm_monthly_depreciation.get(str(firm_id), {}).get(m, 0.0) or 0.0)

    def query_all_firms_monthly_depreciation(self, month: int) -> Dict[str, float]:
        try:
            m = int(month or 0)
        except Exception:
            m = 0
        if m <= 0:
            return {}
        return {str(cid): float(self.firm_monthly_depreciation.get(str(cid), {}).get(m, 0.0) or 0.0) for cid in list(self.firm_id or [])}

    @staticmethod
    def _unmet_key(product_id: str, seller_id: str) -> str:
        return f"{str(product_id)}@{str(seller_id)}"

    def record_unmet_demand(
        self,
        month: int,
        buyer_id: str,
        seller_id: str,
        product_id: str,
        product_name: str,
        quantity_requested: float,
        available_stock: float,
        reason: str = "reserve_failed",
    ) -> None:
        """
        记录“未满足需求”（预留失败/库存不足）。

        qty_short = max(0, requested - available_stock)
        """
        try:
            m = int(month or 0)
            if m <= 0:
                return
            qty_req = max(0.0, float(quantity_requested or 0.0))
            avail = max(0.0, float(available_stock or 0.0))
            qty_short = max(0.0, qty_req - avail)
            if qty_req <= 0:
                return

            key = self._unmet_key(product_id, seller_id)
        # ===== Unmet Demand Tracking =====
            rec = (self.unmet_demand_by_month.get(m, {}) or {}).get(key)
            if rec is None:
                rec = {"attempts": 0.0, "qty_requested": 0.0, "qty_short": 0.0}
        # ===== Unmet Demand Tracking =====
                self.unmet_demand_by_month[m][key] = rec
            rec["attempts"] = float(rec.get("attempts", 0.0) or 0.0) + 1.0
            rec["qty_requested"] = float(rec.get("qty_requested", 0.0) or 0.0) + qty_req
            rec["qty_short"] = float(rec.get("qty_short", 0.0) or 0.0) + qty_short
        except Exception:
            return

    def query_unmet_demand(self, month: int) -> Dict[str, Dict[str, float]]:
        """查询指定月份的未满足需求统计（可序列化）。"""
        try:
            m = int(month or 0)
        except Exception:
            m = 0
        if m <= 0:
            return {}
        # ===== Unmet Demand Tracking =====
        return dict(self.unmet_demand_by_month.get(m, {}) or {})

    def set_cd_calibration(self, calibration: Dict[str, Any]) -> bool:
        """
        接收并固化 month=1 的 CD 校准结果。

        calibration 结构（建议）：
          - industry_A: {industry: A_s}
          - industry_K_tot: {industry: K_s_tot}
          - firm_K: {firm_id: K_i}
          - firm_w: {firm_id: w_i}
          - firm_A: {firm_id: A_i}  (可选；若无则回退 industry_A)
          - meta: {...}
        """
        try:
            if not isinstance(calibration, dict):
                return False
            self._cd_calibration = calibration
            self._cd_industry_A = dict(calibration.get("industry_A", {}) or {})
            self._cd_industry_K_tot = dict(calibration.get("industry_K_tot", {}) or {})
            self._cd_firm_K = dict(calibration.get("firm_K", {}) or {})
            self._cd_firm_A = dict(calibration.get("firm_A", {}) or {})
            self.logger.info(
                f"✅ CD校准结果已写入EconomicCenter: industries={len(self._cd_industry_A)}, firms(K)={len(self._cd_firm_K)}, firms(A)={len(self._cd_firm_A)}"
            )
            return True
        except Exception as e:
            self.logger.error(f"写入CD校准结果失败: {e}")
            return False


    # =========================================================================
    # Agent Initialization
    # =========================================================================
    def init_agent_ledger(self, agent_id: str, initial_amount: float = 0.0):
        """
        Initialize a ledger for an agent with a given initial amount.
        If the agent already exists, it will not overwrite the existing ledger.
        """
        if agent_id not in self.ledger:
            ledger = Ledger.create(agent_id, amount=initial_amount)
            self.ledger[agent_id] = ledger
            # self.logger.info(f"Initialized ledger for agent {agent_id} with amount {initial_amount}")
    
    def init_agent_product(self, agent_id: str, product: Optional[Product]=None):
        """
        Initialize a product for an agent. If the product already exists, it will merge the amounts.
        """
        if agent_id not in self.products:
            # print(f"Initialized product for agent {agent_id}")
            self.products[agent_id] = []
        
        if product:
            self._add_or_merge_product(agent_id, product)
            # self.logger.info(f"Initialized product {product.name} for agent {agent_id} with amount {product.amount}")

    def init_agent_labor(self, agent_id:str, labor:[LaborHour]=[]):
        """
        Initialize the labor hour for an agent.
        """
        if agent_id not in self.laborhour:
            self.laborhour[agent_id] = []
        if labor:
            self.laborhour[agent_id] = labor

    def register_id(self, agent_id: str, agent_type: Literal['government', 'household', 'firm', 'bank']):
        """
        Register an agent ID based on its type.
        """
        if agent_type == 'government':
            self.government_id.append(agent_id)
        elif agent_type == 'household':
            self.household_id.append(agent_id)
        elif agent_type == 'firm':
            self.firm_id.append(agent_id)
        elif agent_type == 'bank':
            self.bank_id.append(agent_id)


    # =========================================================================
    # Query Methods
    # =========================================================================
    def query_all_products(self):
        return self.products

    def query_all_tx(self):
        return self.tx_history

    def set_all_firm_products_amount(self, amount: float) -> Dict[str, float]:
        """
        将所有企业名下商品库存 amount 设为统一值（用于需求采样/压力测试）。

        Returns:
            {"products_updated": int, "amount": float}
        """
        try:
            amt = float(amount)
        except Exception:
            amt = 0.0
        if amt < 0:
            amt = 0.0
        updated = 0
        for owner_id, products in (self.products or {}).items():
            if owner_id not in (self.firm_id or []):
                continue
            for p in (products or []):
                try:
                    p.amount = amt
                    updated += 1
                except Exception:
                    continue
        return {"products_updated": int(updated), "amount": float(amt)}
    
    def query_exsiting_agents(self, agent_type: Literal['government', 'household', 'firm']) -> List[str]:
        """
        Query existing agents based on their type.
        """
        if agent_type == 'government':
            return self.government_id
        elif agent_type == 'household':
            return self.household_id
        elif agent_type == 'firm':
            return self.firm_id
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
    # query interface
    def query_balance(self, agent_id: str) -> float:
        """
        Query the cash balance of an agent.
        
        Args:
            agent_id: Unique identifier of the agent
            
        Returns:
            Current cash balance
        """
        if agent_id in self.ledger:
            return self.ledger[agent_id].amount
        else:
            return 0.0

    def query_redistribution_record_per_person(self, month: int) -> float:
        return self.redistribution_record_per_person[month]
    
    def query_products(self, agent_id: str) -> List[Product]:
        """
        Query all products owned by an agent.
        
        Args:
            agent_id: Unique identifier of the agent
            
        Returns:
            List of products owned by the agent
        """
        return self.products[agent_id]
    
    def query_price(self, agent_id: str, product_id: str) -> float:
        for product in self.products[agent_id]:
            if product.product_id == product_id:
                return product.price
        return 0.0
    
    def query_financial_summary(self, agent_id: str) -> Dict[str, float]:
        """查询代理的财务摘要：余额、总收入、总支出（企业适用）"""
        result = {}
        
        if agent_id in self.ledger:
            result["balance"] = self.ledger[agent_id].amount
        else:
            result["balance"] = 0.0
        
        # 如果是企业，添加收支记录
        if agent_id in self.firm_financials:
            result.update(self.firm_financials[agent_id])
            result["net_profit"] = result.get("total_income", 0.0) - result.get("total_expenses", 0.0)
        
        result['total_income'] = self.firm_financials[agent_id].get("total_income", 0.0)
        result['total_expenses'] = self.firm_financials[agent_id].get("total_expenses", 0.0)
        return result

    def get_transactions(
        self,
        month: Optional[int] = None,
        tx_type: Optional[str] = None,
        party_id: Optional[str] = None,
    ) -> List[Transaction]:
        if month is None and tx_type is None and party_id is None:
            return list(self.tx_history)

        base: List[Transaction]
        if month is not None:
            month = int(month)
            if month in self.tx_by_month:
                base = self.tx_by_month.get(month, []) or []
            else:
                base = self.tx_history
        elif party_id is not None:
            party_id = str(party_id)
            if party_id in self.tx_by_party:
                base = self.tx_by_party.get(party_id, []) or []
            else:
                base = self.tx_history
        elif tx_type is not None:
            tx_type = str(tx_type)
            if tx_type in self.tx_by_type:
                base = self.tx_by_type.get(tx_type, []) or []
            else:
                base = self.tx_history
        else:
            base = self.tx_history

        filtered = base
        if party_id is not None:
            party_id = str(party_id)
            filtered = [
                tx for tx in filtered
                if getattr(tx, "sender_id", None) == party_id or getattr(tx, "receiver_id", None) == party_id
            ]
        if month is not None:
            month = int(month)
            filtered = [tx for tx in filtered if int(getattr(tx, "month", 0) or 0) == month]
        if tx_type is not None:
            tx_type = str(tx_type)
            filtered = [tx for tx in filtered if str(getattr(tx, "type", "")) == tx_type]
        return filtered

    def get_transactions_by_receiver(
        self,
        receiver_id: str,
        tx_type: Optional[str] = None,
        month: Optional[int] = None,
    ) -> List[Dict]:
        """
        按接收方查询交易记录
        
        Args:
            receiver_id: 接收方 ID
            tx_type: 交易类型（可选）
            month: 月份（可选，None 表示全部）
            
        Returns:
            交易记录列表，每条记录为字典格式
        """
        filtered = []
        
        for tx in self.tx_history:
            # 检查接收方
            if getattr(tx, "receiver_id", None) != receiver_id:
                continue
            
            # 检查交易类型
            if tx_type is not None and str(getattr(tx, "type", "")) != tx_type:
                continue
            
            # 检查月份
            if month is not None and int(getattr(tx, "month", 0) or 0) != int(month):
                continue
            
            # 转换为字典格式
            filtered.append({
                "id": getattr(tx, "id", ""),
                "sender_id": getattr(tx, "sender_id", ""),
                "receiver_id": getattr(tx, "receiver_id", ""),
                "amount": float(getattr(tx, "amount", 0.0) or 0.0),
                "type": getattr(tx, "type", ""),
                "month": int(getattr(tx, "month", 0) or 0),
                "metadata": getattr(tx, "metadata", {}) or {},
            })
        
        return filtered

    def get_period_statistics(self, month: int) -> PeriodStatistics:
        month = int(month)
        if month in self.period_statistics:
            return self.period_statistics[month]

        self._get_period_stats(month)
        transactions = self.tx_by_month.get(month)
        if transactions is None:
            transactions = [tx for tx in self.tx_history if int(getattr(tx, "month", 0) or 0) == month]
        for tx in transactions:
            self._update_period_statistics(tx)
        return self.period_statistics[month]
    
    def record_firm_income(self, firm_id: str, amount: float):
        """记录企业收入"""
        self.firm_financials[firm_id]["total_income"] += amount
        
    def record_firm_expense(self, firm_id: str, amount: float):
        """记录企业支出"""
        self.firm_financials[firm_id]["total_expenses"] += amount
    
    def record_firm_monthly_income(self, firm_id: str, month: int, amount: float):
        """记录企业月度收入"""
        self.firm_monthly_financials[firm_id][month]["income"] += amount
        
    def record_firm_monthly_expense(self, firm_id: str, month: int, amount: float):
        """记录企业月度支出"""
        self.firm_monthly_financials[firm_id][month]["expenses"] += amount
    
    def query_firm_monthly_financials(self, firm_id: str, month: int) -> Dict[str, float]:
        """查询企业指定月份的财务数据"""
        if firm_id in self.firm_monthly_financials and month in self.firm_monthly_financials[firm_id]:
            monthly_data = self.firm_monthly_financials[firm_id][month]
            depreciation = float(self.firm_monthly_depreciation.get(firm_id, {}).get(month, 0.0) or 0.0)
            return {
                "monthly_income": monthly_data["income"],
                "monthly_expenses": monthly_data["expenses"],
                "monthly_profit": monthly_data["income"] - monthly_data["expenses"],
                "monthly_depreciation": depreciation,
            }
        depreciation = float(self.firm_monthly_depreciation.get(firm_id, {}).get(month, 0.0) or 0.0)
        return {
            "monthly_income": 0.0,
            "monthly_expenses": 0.0,
            "monthly_profit": 0.0,
            "monthly_depreciation": depreciation,
        }

    def query_all_firms_monthly_financials(self, month: int) -> Dict[str, Dict[str, float]]:
        """
        批量查询“所有企业”在指定月份的财务数据（减少Ray远程调用次数）。

        Returns:
            {firm_id: {"monthly_income":..., "monthly_expenses":..., "monthly_profit":...}}
        """
        result: Dict[str, Dict[str, float]] = {}
        try:
            for cid in list(self.firm_id or []):
                data = self.firm_monthly_financials.get(cid, {}).get(month, None)
                if data:
                    inc = float(data.get("income", 0.0) or 0.0)
                    exp = float(data.get("expenses", 0.0) or 0.0)
                else:
                    inc = 0.0
                    exp = 0.0
                dep = float(self.firm_monthly_depreciation.get(str(cid), {}).get(month, 0.0) or 0.0)
                result[str(cid)] = {
                    "monthly_income": inc,
                    "monthly_expenses": exp,
                    "monthly_profit": inc - exp,
                    "monthly_depreciation": dep,
                }
        except Exception:
            # 兜底：返回已收集到的部分结果
            return result
        return result

    def query_firm_monthly_wage_expenses(self, firm_id: str, month: int) -> float:
        """
        查询企业指定月份的工资总支出（税前 gross_wage）。

        注意：工资在 process_labor 中以 gross_wage 计入 firm_monthly_wage_expenses，
        与 tx_history 的 labor_payment（税后）不同。
        """
        try:
            return float(self.firm_monthly_wage_expenses.get(firm_id, {}).get(month, 0.0) or 0.0)
        except Exception:
            return 0.0
    
    def query_firm_production_stats(self, firm_id: str, month: int) -> Dict[str, float]:
        """查询企业指定月份的生产统计数据"""
        if firm_id in self.firm_production_stats and month in self.firm_production_stats[firm_id]:
            production_data = self.firm_production_stats[firm_id][month]
            return {
                "base_production": production_data["base_production"],
                "labor_production": production_data["labor_production"],
                "total_production": production_data["base_production"] + production_data["labor_production"]
            }
        return {"base_production": 0.0, "labor_production": 0.0, "total_production": 0.0}
    
    def query_firm_all_monthly_financials(self, firm_id: str) -> Dict[int, Dict[str, float]]:
        """查询企业所有月份的财务数据"""
        result = {}
        if firm_id in self.firm_monthly_financials:
            for month, data in self.firm_monthly_financials[firm_id].items():
                result[month] = {
                    "monthly_income": data["income"],
                    "monthly_expenses": data["expenses"],
                    "monthly_profit": data["income"] - data["expenses"]
                }
        return result

    def query_income(self, agent_id: str, month: int) -> float:
        total_wage = 0.0
        for wage in self.wage_history:
            if wage.agent_id == agent_id and wage.month == month:
                total_wage += wage.amount
        return total_wage

    def query_net_wage(self, household_id: str, month: int) -> float:
        """
        查询家庭指定月份的“税后工资”（来自 labor_payment 交易）。

        说明：
        - wage_history 记录的是税前工资（gross），用于企业成本/宏观核算；
        - 家庭可支配收入应以 labor_payment（net）为准，避免把个税/FICA也当作可消费收入。
        """
        total = 0.0
        try:
            transactions = self.tx_by_month.get(int(month))
            if transactions is None:
                transactions = self.tx_history
            for tx in transactions:
                if int(getattr(tx, "month", 0) or 0) != int(month):
                    continue
                if getattr(tx, "type", None) != "labor_payment":
                    continue
                if str(getattr(tx, "receiver_id", "") or "") != str(household_id):
                    continue
                total += float(getattr(tx, "amount", 0.0) or 0.0)
        except Exception:
            return 0.0
        return float(total)


    def query_labor(self, agent_id: str) -> List[LaborHour]:
        return self.laborhour[agent_id]

    def deposit_funds(self, agent_id: str, amount: float):
        """
        Deposit funds into an agent's ledger.
        
        Args:
            agent_id: Unique identifier of the agent
            amount: Amount to deposit
        """
        self.ledger[agent_id].amount += amount
    
    def set_agent_balance(self, agent_id: str, amount: float) -> float:
        """
        设置代理余额为指定值（覆盖式）。
        用于“企业初始资金 = 初始库存总价值”等初始化场景。
        """
        if agent_id not in self.ledger:
            self.ledger[agent_id] = Ledger()
        self.ledger[agent_id].amount = float(amount)
        return self.ledger[agent_id].amount
    
    def update_balance(self, agent_id: str, amount: float):
        """
        更新代理的余额（可以是正数或负数）
        
        Args:
            agent_id: 代理ID
            amount: 变动金额（正数增加，负数减少）
        """
        if agent_id not in self.ledger:
            self.ledger[agent_id] = Ledger()
        self.ledger[agent_id].amount += amount
    
    def consume_product_inventory(self, firm_id: str, product_id: str, quantity: float) -> bool:
        """
        减少企业商品库存
        
        Args:
            firm_id: 企业ID
            product_id: 商品ID
            quantity: 消耗数量
            
        Returns:
            bool: 是否成功消耗
        """
        if firm_id not in self.products:
            self.logger.warning(f"企业 {firm_id} 没有产品库存")
            return False
        
        for product in self.products[firm_id]:
            if product.product_id == product_id:
                if product.amount >= quantity:
                    product.amount -= quantity
                    # self.logger.info(f"企业 {firm_id} 商品 {product_id} 消耗 {quantity} 单位，剩余 {product.amount}")
                    return True
                else:
                    self.logger.warning(f"企业 {firm_id} 商品 {product_id} 库存不足: {product.amount} < {quantity}")
                    return False
        
        self.logger.warning(f"企业 {firm_id} 没有找到商品 {product_id}")
        return False
    

    # =========================================================================
    # Product Management
    # =========================================================================
    def register_product(self, agent_id: str, product: Product):
        """
        Register a product for an agent. If the product already exists, it will merge the amounts.
        """
        if agent_id not in self.products:
            # print(f"Initialized product for agent {agent_id}")
            self.products[agent_id] = []
        
        self._add_or_merge_product(agent_id, product, product.amount)
        # self.logger.info(f"Registered product {product.name} for agent {agent_id} with amount {product.amount}")

    def _add_or_merge_product(self, agent_id:str, product: Product, quantity: float = 1.0):

        product.owner_id = agent_id
        product.amount = quantity
        for existing_product in self.products[agent_id]:
            if existing_product.product_id == product.product_id:
                existing_product.amount += quantity
                return
        self.products[agent_id].append(product)

    def _check_and_reserve_inventory(self, seller_id: str, product: Product, quantity: float) -> bool:
        """
        检查并预留库存，确保原子性购买操作
        返回True表示库存充足且已预留，False表示库存不足
        """
        if seller_id not in self.products:
            return False

        # 🔒 兼容预留系统：无 reservation_id 的购买也应考虑“已被其他人预留”的数量
        try:
            available_stock = self._get_available_stock(seller_id, product.product_id)
            return available_stock >= quantity
        except Exception:
            # 回退旧逻辑
            for existing_product in self.products[seller_id]:
                if existing_product.product_id == product.product_id:
                    return existing_product.amount >= quantity
            return False
    
    def _get_profit_margin(self, category: str) -> float:
        """
        根据商品大类获取毛利率（用于利润计算）
        
        Args:
            category: 商品大类名称（daily_cate）
            
        Returns:
            毛利率（百分比，如25.0表示25%）
        """
        # 如果配置中有该大类，返回配置的毛利率
        if category in self.category_profit_margins:
            return self.category_profit_margins[category]
        
        # 如果找不到该大类，返回默认毛利率25%
        self.logger.warning(f"未找到大类 '{category}' 的毛利率配置，使用默认值25%")
        return 25.0

    def _ensure_product_cost_fields(self, product: Product, default_category: Optional[str] = None) -> None:
        """
        Ensure product has stable base_price and unit_cost.

        - base_price: original (initial) price used for cost derivation
        - unit_cost: derived from base_price and category gross margin (kept stable even if price changes)
        """
        try:
            current_price = float(getattr(product, "price", 0.0) or 0.0)
        except Exception:
            current_price = 0.0

        try:
            base_price = getattr(product, "base_price", None)
            base_price = float(base_price) if base_price is not None else 0.0
        except Exception:
            base_price = 0.0

        if base_price <= 0 and current_price > 0:
            product.base_price = current_price
            base_price = current_price

        try:
            unit_cost = getattr(product, "unit_cost", None)
            unit_cost = float(unit_cost) if unit_cost is not None else 0.0
        except Exception:
            unit_cost = 0.0

        if unit_cost <= 0 and base_price > 0:
            category = getattr(product, "classification", None) or default_category or "Unknown"
            try:
                margin_pct = float(self.category_profit_margins.get(category, 25.0) or 25.0)
            except Exception:
                margin_pct = 25.0
            margin_pct = max(0.0, min(80.0, margin_pct))
            unit_cost = base_price * (1.0 - margin_pct / 100.0)
            if unit_cost <= 1e-6:
                unit_cost = max(0.01, base_price * 0.2)
            product.unit_cost = float(unit_cost)
    
    def _reduce_or_remove_product(self, agent_id: str, product: Product, quantity: float = 1.0):
        """
        减少商品库存（在确认库存充足后调用）
        """
        for existing_product in self.products[agent_id]:
            if existing_product.product_id == product.product_id:
                # 再次检查库存（双重保险）
                if existing_product.amount < quantity:
                    raise ValueError(f"库存不足: 需要 {quantity}，但只有 {existing_product.amount}")
                
                existing_product.amount -= quantity
                return
        raise ValueError("Asset not found or insufficient amount to reduce.")
    
    # register_middleware
    def register_middleware(self, tx_type: str, middleware_fn: Callable[[Transaction, Dict[str, float]], None], tag: Optional[str] = None):
        """
        Register a middleware function for transaction processing.
        
        Args:
            tx_type: Type of transaction to apply middleware to
            middleware_fn: Middleware function to execute
            tag: Optional tag for identifying/replacing middleware
        """
        if tag:
            self.middleware.register(tx_type, middleware_fn, tag)
        else:
            self.middleware.register(tx_type, middleware_fn)
    
    # ============================================================================
    # 🔒 库存预留系统（解决并发竞争问题）
    # ============================================================================
    

    # =========================================================================
    # Inventory Reservation System
    # =========================================================================
    def reserve_inventory(self, buyer_id: str, seller_id: str, product_id: str,
                         product_name: str, quantity: float,
                         timeout_seconds: float = None,
                         month: Optional[int] = None) -> Optional[str]:
        """
        预留库存
        
        Args:
            buyer_id: 买家ID
            seller_id: 卖家ID
            product_id: 商品ID
            product_name: 商品名称
            quantity: 预留数量
            timeout_seconds: 超时时间（秒），默认使用系统配置
        
        Returns:
            预留ID（成功）或 None（失败）
        """
        # 清理过期预留
        self._cleanup_expired_reservations()
        
        # 检查库存是否充足（考虑已预留的数量）
        available_stock = self._get_available_stock(seller_id, product_id)
        
        if available_stock < quantity:
            self.logger.warning(f"🔒 库存预留失败: {product_name} 可用库存 {available_stock:.2f} < 需求 {quantity:.2f}")
            try:
                if month is not None:
                    self.record_unmet_demand(
                        month=int(month),
                        buyer_id=str(buyer_id),
                        seller_id=str(seller_id),
                        product_id=str(product_id),
                        product_name=str(product_name),
                        quantity_requested=float(quantity or 0.0),
                        available_stock=float(available_stock or 0.0),
                        reason="reserve_failed",
                    )
            except Exception:
                pass
            return None
        
        # 创建预留记录
        timeout = timeout_seconds if timeout_seconds is not None else self.reservation_timeout
        reservation = InventoryReservation.create(
            buyer_id=buyer_id,
            seller_id=seller_id,
            product_id=product_id,
            product_name=product_name,
            quantity=quantity,
            timeout_seconds=timeout
        )
        
        # 保存预留记录
        # ===== Inventory Reservation System =====
        self.inventory_reservations[reservation.reservation_id] = reservation
        
        self.logger.info(f"✅ 库存预留成功: {product_name} × {quantity:.2f} (预留ID: {reservation.reservation_id[:8]}...)")
        return reservation.reservation_id
    
    def confirm_reservation(self, reservation_id: str) -> bool:
        """
        确认预留（购买成功后调用）
        
        Args:
            reservation_id: 预留ID
        
        Returns:
            是否成功确认
        """
        # ===== Inventory Reservation System =====
        if reservation_id not in self.inventory_reservations:
            self.logger.warning(f"⚠️ 预留ID不存在: {reservation_id[:8]}...")
            return False
        
        # ===== Inventory Reservation System =====
        reservation = self.inventory_reservations[reservation_id]

        # 只允许确认“活跃”的预留，避免重复确认/错误确认
        if reservation.status != 'active':
            self.logger.warning(
                f"⚠️ 预留状态不可确认: {reservation.product_name} status={reservation.status} "
                f"(预留ID: {reservation_id[:8]}...)"
            )
            return False
        
        # 检查预留是否已过期
        if time.time() > reservation.expires_at:
            self.logger.warning(f"⚠️ 预留已过期: {reservation.product_name} (预留ID: {reservation_id[:8]}...)")
            reservation.status = 'expired'
            return False
        
        # 标记为已确认
        reservation.status = 'confirmed'
        self.logger.info(f"✅ 预留已确认: {reservation.product_name} × {reservation.quantity:.2f}")
        
        return True

    def validate_reservation(
        self,
        reservation_id: str,
        buyer_id: Optional[str] = None,
        seller_id: Optional[str] = None,
        product_id: Optional[str] = None,
        quantity: Optional[float] = None,
    ) -> bool:
        """
        校验预留是否可用于本次购买（不改变预留状态）。

        说明：预留在“商品转移完成后”才会被 confirm；在此之前保持 active，
        使得 _get_available_stock 能正确扣除已预留数量，避免并发超卖。
        """
        self._cleanup_expired_reservations()

        # ===== Inventory Reservation System =====
        if reservation_id not in self.inventory_reservations:
            self.logger.warning(f"⚠️ 预留ID不存在: {reservation_id[:8]}...")
            return False

        # ===== Inventory Reservation System =====
        reservation = self.inventory_reservations[reservation_id]
        if reservation.status != 'active':
            self.logger.warning(
                f"⚠️ 预留不可用: {reservation.product_name} status={reservation.status} "
                f"(预留ID: {reservation_id[:8]}...)"
            )
            return False

        if time.time() > reservation.expires_at:
            reservation.status = 'expired'
            self.logger.warning(f"⚠️ 预留已过期: {reservation.product_name} (预留ID: {reservation_id[:8]}...)")
            return False

        if buyer_id is not None and reservation.buyer_id != buyer_id:
            self.logger.warning(f"⚠️ 预留buyer不匹配: expected={buyer_id} got={reservation.buyer_id}")
            return False
        if seller_id is not None and reservation.seller_id != seller_id:
            self.logger.warning(f"⚠️ 预留seller不匹配: expected={seller_id} got={reservation.seller_id}")
            return False
        if product_id is not None and reservation.product_id != product_id:
            self.logger.warning(f"⚠️ 预留product不匹配: expected={product_id} got={reservation.product_id}")
            return False
        if quantity is not None and abs(float(reservation.quantity) - float(quantity)) > 1e-6:
            self.logger.warning(f"⚠️ 预留quantity不匹配: expected={quantity} got={reservation.quantity}")
            return False

        return True
    
    def release_reservation(self, reservation_id: str, reason: str = "cancelled") -> bool:
        """
        释放预留（购买失败或取消时调用）
        
        Args:
            reservation_id: 预留ID
            reason: 释放原因
        
        Returns:
            是否成功释放
        """
        # ===== Inventory Reservation System =====
        if reservation_id not in self.inventory_reservations:
            return False
        
        # ===== Inventory Reservation System =====
        reservation = self.inventory_reservations[reservation_id]
        reservation.status = 'released'
        
        self.logger.info(f"🔓 预留已释放: {reservation.product_name} × {reservation.quantity:.2f} (原因: {reason})")
        return True
    
    def _get_available_stock(self, seller_id: str, product_id: str) -> float:
        """
        获取可用库存（实际库存 - 已预留数量）
        
        Args:
            seller_id: 卖家ID
            product_id: 商品ID
        
        Returns:
            可用库存数量
        """
        # 获取实际库存
        actual_stock = 0.0
        for product in self.products.get(seller_id, []):
            if product.product_id == product_id:
                actual_stock = product.amount
                break
        
        # 计算已预留数量（只统计活跃状态的预留）
        reserved_quantity = 0.0
        # ===== Inventory Reservation System =====
        for reservation in self.inventory_reservations.values():
            if (reservation.seller_id == seller_id and
                reservation.product_id == product_id and
                reservation.status == 'active' and
                time.time() <= reservation.expires_at):
                reserved_quantity += reservation.quantity
        
        available = actual_stock - reserved_quantity
        return max(0.0, available)  # 确保不返回负数
    
    def _cleanup_expired_reservations(self):
        """清理过期的预留记录"""
        current_time = time.time()
        expired_ids = []
        
        # ===== Inventory Reservation System =====
        for reservation_id, reservation in self.inventory_reservations.items():
            if reservation.status == 'active' and current_time > reservation.expires_at:
                reservation.status = 'expired'
                expired_ids.append(reservation_id)
        
        if expired_ids:
            self.logger.info(f"🧹 清理了 {len(expired_ids)} 个过期预留")
    
    def get_reservation_stats(self) -> Dict[str, int]:
        """获取预留统计信息（用于监控）"""
        stats = {
        # ===== Inventory Reservation System =====
            'total': len(self.inventory_reservations),
            'active': 0,
            'confirmed': 0,
            'released': 0,
            'expired': 0
        }
        
        # ===== Inventory Reservation System =====
        for reservation in self.inventory_reservations.values():
            stats[reservation.status] += 1
        
        return stats
    

    # =========================================================================
    # Transaction Processing
    # =========================================================================
    def _ensure_ledger_entry(self, agent_id: str, initial_balance: float = 0.0) -> None:
        if agent_id not in self.ledger:
            self.ledger[agent_id] = Ledger.create(agent_id, float(initial_balance or 0.0))

    def _get_period_stats(self, month: int) -> PeriodStatistics:
        if month not in self.period_statistics:
            self.period_statistics[month] = PeriodStatistics(period=month)
        return self.period_statistics[month]

    def _index_transaction(self, tx: Transaction) -> None:
        try:
            month = int(getattr(tx, "month", 0) or 0)
        except Exception:
            month = 0

        self.tx_by_month[month].append(tx)
        tx_type = str(getattr(tx, "type", "unknown") or "unknown")
        self.tx_by_type[tx_type].append(tx)

        sender_id = getattr(tx, "sender_id", None)
        if sender_id:
            self.tx_by_party[str(sender_id)].append(tx)
        receiver_id = getattr(tx, "receiver_id", None)
        if receiver_id:
            self.tx_by_party[str(receiver_id)].append(tx)

    def _update_period_statistics(self, tx: Transaction) -> None:
        try:
            month = int(getattr(tx, "month", 0) or 0)
        except Exception:
            month = 0

        stats = self._get_period_stats(month)
        amount = float(getattr(tx, "amount", 0.0) or 0.0)
        tx_type = str(getattr(tx, "type", "") or "")

        stats.total_transactions += 1
        stats.total_volume += amount

        if tx_type in ("purchase", "product_sale", "inherent_market", "government_procurement"):
            stats.product_volume += amount
            if tx_type == "purchase" and getattr(tx, "sender_id", None) in self.household_id:
                stats.total_consumption += amount
        elif tx_type == "labor_payment":
            stats.wage_volume += amount
        elif tx_type == "resource_purchase":
            stats.resource_volume += amount
        elif tx_type in ("consume_tax", "labor_tax", "corporate_tax", "tax_collection"):
            stats.tax_volume += amount

        if tx_type == "government_procurement":
            stats.total_government_spending += amount

    def _record_transaction(
        self,
        sender_id: str,
        receiver_id: str,
        amount: float,
        tx_type: str,
        month: int,
        assets: Optional[List[Any]] = None,
        labor_hours: Optional[List[LaborHour]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        related_transaction_id: Optional[str] = None,
        status: str = TransactionStatus.COMPLETED,
    ) -> Transaction:
        tx_kwargs = {
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "amount": float(amount or 0.0),
            "type": tx_type,
            "month": month,
            "status": status,
            "metadata": metadata or {},
        }
        if assets is not None:
            tx_kwargs["assets"] = assets
        if labor_hours is not None:
            tx_kwargs["labor_hours"] = labor_hours
        if related_transaction_id:
            tx_kwargs["related_transaction_id"] = related_transaction_id
        tx = Transaction(**tx_kwargs)
        self.tx_history.append(tx)
        self._index_transaction(tx)
        self._update_period_statistics(tx)
        return tx

    def record_intermediate_goods_purchase(
        self,
        month: int,
        buyer_id: str,
        total_cost: float,
        costs_by_industry: Optional[Dict[str, float]] = None,
        items: Optional[List[Dict[str, Any]]] = None,
        receiver_id: Optional[str] = None,
    ) -> Optional[str]:
        total_cost = float(total_cost or 0.0)
        if total_cost <= 0:
            return None

        if receiver_id is None:
            receiver_id = "market_intermediate_goods"

        self._ensure_ledger_entry(buyer_id)
        self._ensure_ledger_entry(receiver_id)

        is_company = buyer_id in self.firm_id
        if not is_company and self.ledger[buyer_id].amount < total_cost:
            raise ValueError(
                f"Insufficient balance for {buyer_id}: ${self.ledger[buyer_id].amount:.2f} < ${total_cost:.2f}"
            )
        elif is_company and self.ledger[buyer_id].amount < total_cost:
            self.self.logger.info(
                f"💳 Company {buyer_id} intermediate goods purchase with negative balance: "
                f"${self.ledger[buyer_id].amount:.2f} → ${self.ledger[buyer_id].amount - total_cost:.2f}"
            )

        self.ledger[buyer_id].amount -= total_cost
        self.ledger[receiver_id].amount += total_cost

        if is_company:
            self.record_firm_expense(buyer_id, total_cost)
            self.record_firm_monthly_expense(buyer_id, month, total_cost)
            self.firm_monthly_production_cost[buyer_id][month] += total_cost

        tx = self._record_transaction(
            sender_id=buyer_id,
            receiver_id=receiver_id,
            amount=total_cost,
            tx_type="product_sale",
            month=month,
            metadata={
                "purchase_category": "intermediate_goods",
                "costs_by_industry": costs_by_industry or {},
                "items": items or [],
            },
        )
        return tx.id

    def record_resource_purchase(
        self,
        month: int,
        buyer_id: str,
        industry_code: str,
        quantity: float,
        unit_price: float,
        total_cost: float,
        unit: Optional[str] = None,
        base_price: Optional[float] = None,
        receiver_id: Optional[str] = None,
    ) -> Optional[str]:
        total_cost = float(total_cost or 0.0)
        if total_cost <= 0:
            return None

        if receiver_id is None:
            receiver_id = f"market_resource_{industry_code}"

        self._ensure_ledger_entry(buyer_id)
        self._ensure_ledger_entry(receiver_id)

        is_company = buyer_id in self.firm_id
        if not is_company and self.ledger[buyer_id].amount < total_cost:
            raise ValueError(
                f"Insufficient balance for {buyer_id}: ${self.ledger[buyer_id].amount:.2f} < ${total_cost:.2f}"
            )
        elif is_company and self.ledger[buyer_id].amount < total_cost:
            self.logger.info(
                f"💳 Company {buyer_id} resource purchase with negative balance: "
                f"${self.ledger[buyer_id].amount:.2f} → ${self.ledger[buyer_id].amount - total_cost:.2f}"
            )

        self.ledger[buyer_id].amount -= total_cost
        self.ledger[receiver_id].amount += total_cost

        if is_company:
            self.record_firm_expense(buyer_id, total_cost)
            self.record_firm_monthly_expense(buyer_id, month, total_cost)
            self.firm_monthly_production_cost[buyer_id][month] += total_cost

        tx = self._record_transaction(
            sender_id=buyer_id,
            receiver_id=receiver_id,
            amount=total_cost,
            tx_type="resource_purchase",
            month=month,
            metadata={
                "industry_code": industry_code,
                "quantity": float(quantity or 0.0),
                "unit_price": float(unit_price or 0.0),
                "unit": unit,
                "base_price": base_price,
            },
        )
        return tx.id

    def process_batch_purchases(self, month: int, buyer_id: str, purchase_list: List[Dict]) -> List[Optional[str]]:
        """
        批量处理购买，减少Ray远程调用次数
        
        Args:
            month: 当前月份
            buyer_id: 购买者ID
            purchase_list: 购买列表，每项包含 {'seller_id', 'product', 'quantity', 'reservation_id'(可选)}
        
        Returns:
            交易ID列表（成功返回tx_id，失败返回None）
        """
        results = []
        for purchase in purchase_list:
            seller_id = purchase['seller_id']
            product = purchase['product']
            quantity = purchase.get('quantity', 1.0)
            reservation_id = purchase.get('reservation_id')  # 🔒 新增：预留ID
            
            tx_result = self.process_purchase(month, buyer_id, seller_id, product, quantity, reservation_id)
            
            # 🔧 处理返回值：Transaction对象或False
            if tx_result and hasattr(tx_result, 'id'):
                results.append(tx_result.id)  # 返回交易ID
            else:
                results.append(None)  # 购买失败
        return results
    
    def process_purchase(self, month: int, buyer_id: str, seller_id: str, product: Product,
                         quantity: float = 1.0, reservation_id: Optional[str] = None) -> Optional[str]:
        """
        处理购买交易
        
        Args:
            month: 当前月份
            buyer_id: 买家ID
            seller_id: 卖家ID
            product: 商品对象
            quantity: 购买数量
            reservation_id: 预留ID（如果有）
        
        Returns:
            Transaction对象（成功）或 False（失败）
        """
        # 计算总费用：标价 + 消费税
        base_price = product.price * quantity
        total_cost_with_tax = base_price * (1 + self.vat_rate)  # 家庭支付标价+消费税
        
        # 检查家庭余额是否足够支付含税价格
        if self.ledger[buyer_id].amount < total_cost_with_tax:
            # 如果有预留，释放它
            if reservation_id:
                self.release_reservation(reservation_id, reason="insufficient_funds")
            return False

        # 🔒 新版库存检查：优先使用预留机制（先校验，不改变状态；成功转移后再 confirm）
        if reservation_id:
            if not self.validate_reservation(
                reservation_id,
                buyer_id=buyer_id,
                seller_id=seller_id,
                product_id=getattr(product, "product_id", None),
                quantity=quantity,
            ):
                # 尽量释放无效预留，避免“卡死库存”
                self.release_reservation(reservation_id, reason="invalid_reservation")
                self.logger.warning(f"预留无效，购买失败: {product.name} (预留ID: {reservation_id[:8]}...)")
                return False
        else:
            # 无预留ID：使用旧的检查方式（向后兼容）
            if not self._check_and_reserve_inventory(seller_id, product, quantity):
                # 获取当前库存用于调试
                current_stock = 0
                for pro in self.products.get(seller_id, []):
                    if pro.product_id == product.product_id:
                        current_stock = pro.amount
                        break
                self.logger.warning(f"库存不足，购买失败: {product.name} 需要 {quantity}，但库存不足, 剩余库存: {current_stock}")
                try:
                    self.record_unmet_demand(
                        month=int(month),
                        buyer_id=str(buyer_id),
                        seller_id=str(seller_id),
                        product_id=str(getattr(product, "product_id", "")),
                        product_name=str(getattr(product, "name", "")),
                        quantity_requested=float(quantity or 0.0),
                        available_stock=float(current_stock or 0.0),
                        reason="purchase_no_reservation_insufficient_stock",
                    )
                except Exception:
                    pass
                return False

        # 家庭支付含税价格
        self.ledger[buyer_id].amount -= total_cost_with_tax

        # 创建消费税交易记录（税收部分）
        tax_amount = base_price * self.vat_rate
        tax_tx = self._record_transaction(
            sender_id=buyer_id,
            receiver_id="gov_main_simulation",  # 固定政府ID
            amount=tax_amount,
            tx_type='consume_tax',
            month=month,
            metadata={
                "tax_base": base_price,
                "tax_rate": self.vat_rate,
            },
        )
        
        # 政府收取消费税
        self.ledger["gov_main_simulation"].amount += tax_amount

        # 创建购买交易记录（企业收入部分）
        # 🔧 交易资产必须携带“本次成交数量”，否则销售统计会误读为“当时库存量”
        try:
            tx_product_id = str(getattr(product, "product_id", "") or "")
            tx_name = str(getattr(product, "name", "Unknown") or "Unknown")
            tx_classification = getattr(product, "classification", None) or "Unknown"
            tx_price = float(getattr(product, "price", 0.0) or 0.0)
            tx_base_price = float(getattr(product, "base_price", 0.0) or 0.0)
            tx_unit_cost = float(getattr(product, "unit_cost", 0.0) or 0.0)

            if seller_id in self.products and tx_product_id:
                for inv_p in (self.products.get(seller_id) or []):
                    if str(getattr(inv_p, "product_id", "") or "") == tx_product_id:
                        self._ensure_product_cost_fields(inv_p, default_category=tx_classification)
                        tx_name = str(getattr(inv_p, "name", tx_name) or tx_name)
                        tx_classification = getattr(inv_p, "classification", tx_classification) or tx_classification
                        tx_base_price = float(getattr(inv_p, "base_price", tx_base_price) or tx_base_price)
                        tx_unit_cost = float(getattr(inv_p, "unit_cost", tx_unit_cost) or tx_unit_cost)
                        break

            if tx_base_price <= 0 and tx_price > 0:
                tx_base_price = tx_price
            if tx_unit_cost <= 0 and tx_base_price > 0:
                margin_pct = float(self.category_profit_margins.get(tx_classification, 25.0) or 25.0)
                margin_pct = max(0.0, min(80.0, margin_pct))
                tx_unit_cost = tx_base_price * (1.0 - margin_pct / 100.0)
                if tx_unit_cost <= 1e-6:
                    tx_unit_cost = max(0.01, tx_base_price * 0.2)

            product_kwargs = dict(
                asset_type="products",
                product_id=tx_product_id,
                name=tx_name,
                owner_id=seller_id,
                amount=float(quantity or 0.0),
                price=tx_price,
                classification=tx_classification,
                base_price=float(tx_base_price),
                unit_cost=float(tx_unit_cost),
            )
            product_kwargs = inject_product_attributes(product_kwargs, tx_product_id)
            product_asset = Product(**product_kwargs)
        except Exception:
            # 兜底：至少保证 amount=quantity，避免销量统计爆炸
            product_asset = Product.create(
                name=str(getattr(product, "name", "Unknown") or "Unknown"),
                price=float(getattr(product, "price", 0.01) or 0.01),
                owner_id=seller_id,
                amount=float(quantity or 0.0),
                classification=getattr(product, "classification", None),
                product_id=getattr(product, "product_id", None),
                base_price=getattr(product, "base_price", None),
                unit_cost=getattr(product, "unit_cost", None),
            )

        purchase_tx = self._record_transaction(
            sender_id=buyer_id,
            receiver_id=seller_id,
            amount=base_price,
            assets=[product_asset],
            tx_type='purchase',
            month=month,
            metadata={
                "product_id": getattr(product_asset, "product_id", None),
                "quantity": float(quantity or 0.0),
                "unit_price": float(getattr(product_asset, "price", 0.0) or 0.0),
                "reservation_id": reservation_id,
            },
        )

        # 💰 企业收入（现金流口径）：只记录真实收款额
        # 说明：生产成本应在“生产补货阶段”作为当月支出记录，而不是在销售发生时扣除。
        revenue = base_price
        self.ledger[seller_id].amount += revenue
        self.record_firm_income(seller_id, revenue)
        self.record_firm_monthly_income(seller_id, month, revenue)
        
        # 企业所得税改为“月度结算”（按净利润计税），避免与生产预算形成循环依赖。
        
        # 商品转移
        try:
            self._add_or_merge_product(buyer_id, product, quantity)
            self._reduce_or_remove_product(seller_id, product, quantity)
        except Exception as e:
            if reservation_id:
                self.release_reservation(reservation_id, reason="transfer_failed")
            print(f"Warning: Failed to process purchase: {e}")
            return False

        # 🔒 成功完成商品转移后，确认预留
        if reservation_id:
            self.confirm_reservation(reservation_id)
        
        return purchase_tx

    def process_wage(
        self,
        month: int,
        wage_hour: float,
        household_id: str,
        firm_id: str,
        hours_per_period: float = 40.0,
        periods_per_month: float = 4.0,
    ) -> str:
        """
        发放工资（含税收拆分）

        口径：
        - 税前工资 w = wage_hour × hours_per_period × periods_per_month
          其中 hours_per_period 默认为每周40小时，periods_per_month 默认为4（按月折算）
        - 个人所得税：沿用既有的累进税计算（calculate_progressive_income_tax）
        """
        # 计算税前工资（w）
        try:
            hours = float(hours_per_period or 0.0)
        except Exception:
            hours = 0.0
        try:
            ppm = float(periods_per_month or 0.0)
        except Exception:
            ppm = 0.0
        hours = max(0.0, hours)
        ppm = max(0.0, ppm)
        gross_wage = float(wage_hour or 0.0) * hours * ppm
        
        # 计算个人所得税
        income_tax = self.calculate_progressive_income_tax(gross_wage)

        net_wage = gross_wage - income_tax  # 税后工资（仅扣个税）
        if net_wage < 0:
            net_wage = 0.0
        
        # 创建工资支付交易记录
        wage_tx = self._record_transaction(
            sender_id=firm_id,
            receiver_id=household_id,
            amount=net_wage,  # 家庭收到税后工资
            tx_type='labor_payment',
            month=month,
            metadata={
                "gross_wage": gross_wage,
                "net_wage": net_wage,
                "income_tax": income_tax,
                "wage_hour": wage_hour,
                "hours_per_period": hours,
                "periods_per_month": ppm,
            },
        )
        
        # 创建个人所得税交易记录
        tax_tx = self._record_transaction(
            sender_id=household_id,
            receiver_id="gov_main_simulation",
            amount=income_tax,
            tx_type='labor_tax',
            month=month,
            metadata={
                "gross_wage": gross_wage,
                "tax_rate": (income_tax / gross_wage) if gross_wage > 0 else 0.0,
            },
        )

        # 更新账本
        self.ledger[household_id].amount += net_wage  # 家庭收到税后工资
        self.ledger["gov_main_simulation"].amount += income_tax  # 政府收到个人所得税
        
        # 企业支出工资
        if firm_id:
            self.ledger[firm_id].amount -= gross_wage
            # 记录企业支出（经济中心层面）
            self.record_firm_expense(firm_id, gross_wage)
            # 记录企业月度支出
            self.record_firm_monthly_expense(firm_id, month, gross_wage)
            # 细分统计：月度工资支出（税前工资）
            self.firm_monthly_wage_expenses[firm_id][month] += gross_wage

        # 记录工资历史（记录税前工资）
        self.wage_history.append(Wage.create(household_id, gross_wage, month))
        # print(f"Month {month} Processed labor payment: ${gross_wage:.2f} gross (${net_wage:.2f} net, ${income_tax:.2f} tax) from {firm_id} to {household_id}")
        return wage_tx.id


    # =========================================================================
    # Tax Calculations
    # =========================================================================
    def calculate_progressive_income_tax(self, gross_wage: float) -> float:
        """
        Calculate the income tax for a given gross wage
        """
        total_tax = 0
        for i, bracket in enumerate(self.income_tax_rate):
            if gross_wage > bracket.cutoff:
                if i + 1 < len(self.income_tax_rate):
                    upper_bracket = self.income_tax_rate[i + 1].cutoff
                else:
                    upper_bracket = float('inf')
                taxable_in_bracket = min(gross_wage, upper_bracket) - bracket.cutoff
                total_tax += taxable_in_bracket * bracket.rate
            else:
                break
        return total_tax

    def compute_household_settlement(self, household_id: str):
        """
        Process household settlement, including asset and labor hour settlement.
        计算家庭累积收入和支出
        """
        household_key = str(household_id)

        total_income = 0
        total_expense = 0
        transactions = self.tx_by_party.get(household_key)
        if transactions is None:
            transactions = self.tx_history
        for tx in transactions:
            if tx.type == 'purchase' and tx.sender_id == household_key:
                total_expense += tx.amount

            elif tx.type == 'service' and tx.sender_id == household_key:
                total_expense += tx.amount  # 服务费用直接计入支出，不需要税收调整

            elif tx.type == 'labor_payment' and tx.receiver_id == household_key:
                total_income += tx.amount

            elif tx.type == 'redistribution' and tx.receiver_id == household_key:
                total_income += tx.amount

            elif tx.type == 'interest' and tx.receiver_id == household_key:
                total_income += tx.amount

        return total_income, total_expense

    def compute_household_monthly_stats(self, household_id: str, target_month: int = None):
        """
        计算家庭月度收入和支出统计(收入不统计再分配)
        如果不指定target_month，返回所有月份的统计
        """
        household_key = str(household_id)
        monthly_income = 0
        monthly_expense = 0
        
        month = target_month


        transactions = self.tx_by_party.get(household_key)
        if transactions is None:
            transactions = self.tx_history
        for tx in transactions:
            if tx.type == 'purchase' and tx.sender_id == household_key and tx.month == month:
                monthly_expense += tx.amount
            # 消费税属于“含税购物支出”的一部分（家庭真实现金流支出）
            elif tx.type == 'consume_tax' and tx.sender_id == household_key and tx.month == month:
                monthly_expense += tx.amount

            elif tx.type == 'service' and tx.sender_id == household_key and tx.month == month:
                monthly_expense += tx.amount

            elif tx.type == 'labor_payment' and tx.receiver_id == household_key and tx.month == month:
                monthly_income += tx.amount

            elif tx.type == 'interest' and tx.receiver_id == household_key and tx.month == month:
                monthly_income += tx.amount

            # elif tx.type == 'redistribution' and tx.receiver_id == household_id and tx.month == month:
            #     monthly_income += tx.amount

        return monthly_income, monthly_expense, self.ledger[household_id].amount
    

    # =========================================================================
    # Tax Collection & Redistribution
    # =========================================================================
    def get_monthly_tax_collection(self, month: int) -> Dict[str, float]:
        """
        获取指定月份的税收收入统计
        
        Args:
            month: 目标月份
            
        Returns:
            Dict: 各类税收收入统计
        """
        tax_summary = {
            "consume_tax": 0.0,
            "labor_tax": 0.0,
            "fica_tax": 0.0,
            "corporate_tax": 0.0,
            "total_tax": 0.0
        }
        
        transactions = self.tx_by_month.get(month)
        if transactions is None:
            transactions = self.tx_history
        for tx in transactions:
            if tx.month == month and tx.receiver_id == "gov_main_simulation":
                if tx.type == 'consume_tax':
                    tax_summary["consume_tax"] += tx.amount
                elif tx.type == 'labor_tax':
                    tax_summary["labor_tax"] += tx.amount
                elif tx.type == 'fica_tax':
                    tax_summary["fica_tax"] += tx.amount
                elif tx.type == 'corporate_tax':
                    tax_summary["corporate_tax"] += tx.amount
        
        tax_summary["total_tax"] = (tax_summary["consume_tax"] +
                                   tax_summary["labor_tax"] +
                                   tax_summary["fica_tax"] +
                                   tax_summary["corporate_tax"])
        
        return tax_summary
    

    async def redistribute_monthly_taxes(self, month: int, strategy: str = "equal",
                                       poverty_weight: float = 0.3,
                                       unemployment_weight: float = 0.2,
                                       family_size_weight: float = 0.1) -> Dict[str, float]:
        """
        税收再分配：支持多种分配策略
        
        Args:
            month: 当前月份
            strategy: 分配策略 ("none", "equal", "income_proportional", "poverty_focused", "unemployment_focused", "family_size", "mixed")
            poverty_weight: 贫困权重 (0-1)
            unemployment_weight: 失业权重 (0-1)
            family_size_weight: 家庭规模权重 (0-1)
            
        Returns:
            Dict: 再分配结果统计
        """
        # 如果策略为 "none"，不进行再分配
        if strategy == "none":
            tax_summary = self.get_monthly_tax_collection(month)
            return {
                "total_redistributed": 0.0,
                "recipients": 0,
                "per_person": 0.0,
                "total_tax_collected": tax_summary["total_tax"],
                "tax_breakdown": tax_summary
            }
        
        # 获取当月税收总额
        tax_summary = self.get_monthly_tax_collection(month)
        total_tax = tax_summary["total_tax"]
        
        if total_tax <= 0:
            print(f"Month {month}: No tax revenue to redistribute")
            return {"total_redistributed": 0.0, "recipients": 0, "per_person": 0.0}
        
        # 获取所有有劳动力的家庭ID（基于现有的laborhour字典）
        all_workers = [household_id for household_id, labor_hours in self.laborhour.items()
                      if labor_hours]  # 只包括有劳动力的家庭
        if not all_workers:
            print(f"Month {month}: No households with labor hours found for tax redistribution")
            return {"total_redistributed": 0.0, "recipients": 0, "per_person": 0.0}
        
        # 根据策略计算分配金额
        household_allocations = self._calculate_redistribution_allocations(
            all_workers, total_tax, strategy, poverty_weight, unemployment_weight, family_size_weight, month
        )
        
        total_redistributed = 0.0
        successful_redistributions = 0
        
        # 执行再分配
        for household_id, allocation_amount in household_allocations.items():
            try:
                if allocation_amount > 0:
                    # 政府向家庭转账
                    tx_id = self.add_redistribution_tx(
                        month=month,
                        sender_id="gov_main_simulation",
                        receiver_id=household_id,
                        amount=allocation_amount,
                    )
                    
                    total_redistributed += allocation_amount
                    successful_redistributions += 1
        
            except Exception as e:
                print(f"Failed to redistribute to household {household_id}: {e}")

        # 计算平均分配金额（用于记录）
        avg_allocation = total_redistributed / successful_redistributions if successful_redistributions > 0 else 0
        
        result = {
            "total_tax_collected": total_tax,
            "total_redistributed": total_redistributed,
            "recipients": successful_redistributions,
            "per_person": avg_allocation,
            "strategy": strategy,
            "tax_breakdown": tax_summary
        }
        self.redistribution_record_per_person[month] = avg_allocation

        print(f"Month {month} Tax Redistribution ({strategy}):")
        print(f"  Total tax collected: ${total_tax:.2f}")
        print(f"  Redistributed to {successful_redistributions} households: ${total_redistributed:.2f}")
        print(f"  Average per household: ${avg_allocation:.2f}")
        
        return result

    def _calculate_redistribution_allocations(self, all_workers: List[str], total_tax: float,
                                           strategy: str, poverty_weight: float,
                                           unemployment_weight: float, family_size_weight: float,
                                           month: int) -> Dict[str, float]:
        """
        根据策略计算每个家庭的分配金额
        
        Args:
            all_workers: 所有有劳动力的家庭ID列表
            total_tax: 税收总额
            strategy: 分配策略
            poverty_weight: 贫困权重
            unemployment_weight: 失业权重
            family_size_weight: 家庭规模权重
            month: 当前月份
            
        Returns:
            Dict[str, float]: 家庭ID到分配金额的映射
        """
        if strategy == "equal":
            return self._equal_allocation(all_workers, total_tax)
        elif strategy == "income_proportional":
            return self._income_proportional_allocation(all_workers, total_tax, month)
        elif strategy == "poverty_focused":
            return self._poverty_focused_allocation(all_workers, total_tax, month)
        elif strategy == "unemployment_focused":
            return self._unemployment_focused_allocation(all_workers, total_tax, month)
        elif strategy == "family_size":
            return self._family_size_allocation(all_workers, total_tax)
        elif strategy == "mixed":
            return self._mixed_allocation(all_workers, total_tax, poverty_weight,
                                        unemployment_weight, family_size_weight, month)
        else:
            print(f"Unknown redistribution strategy: {strategy}, using equal allocation")
            return self._equal_allocation(all_workers, total_tax)

    def _equal_allocation(self, all_workers: List[str], total_tax: float) -> Dict[str, float]:
        """平均分配策略"""
        amount_per_household = total_tax / len(all_workers)
        return {household_id: amount_per_household for household_id in all_workers}

    def _income_proportional_allocation(self, all_workers: List[str], total_tax: float, month: int) -> Dict[str, float]:
        """按收入比例分配策略"""
        household_incomes = {}
        total_income = 0.0
        
        for household_id in all_workers:
            monthly_income, _, _ = self.compute_household_monthly_stats(household_id, month)
            household_incomes[household_id] = monthly_income
            total_income += monthly_income
        
        if total_income <= 0:
            return self._equal_allocation(all_workers, total_tax)
        
        allocations = {}
        for household_id in all_workers:
            proportion = household_incomes[household_id] / total_income
            allocations[household_id] = total_tax * proportion
        
        return allocations

    def _poverty_focused_allocation(self, all_workers: List[str], total_tax: float, month: int) -> Dict[str, float]:
        """贫困导向分配策略（收入越低分配越多）"""
        household_incomes = {}
        household_balances = {}
        
        for household_id in all_workers:
            monthly_income, _, balance = self.compute_household_monthly_stats(household_id, month)
            household_incomes[household_id] = monthly_income
            household_balances[household_id] = balance
        
        if not household_incomes:
            return self._equal_allocation(all_workers, total_tax)
        
        max_income = max(household_incomes.values())
        min_income = min(household_incomes.values())
        max_balance = max(household_balances.values()) if household_balances else 0.0
        min_balance = min(household_balances.values()) if household_balances else 0.0
        
        # 若收入与存款都无差异，则退化为均分
        if max_income == min_income and max_balance == min_balance:
            return self._equal_allocation(all_workers, total_tax)
        
        # 计算贫困权重（收入越低、存款越低权重越高）
        # 组合权重：alpha 用于控制收入与存款的权重占比
        alpha = 0.5  # 可按需调整/暴露为超参数
        poverty_weights = {}
        total_weight = 0.0
        
        for household_id, income in household_incomes.items():
            # 收入成分（越低越高）
            income_component = 0.0
            if max_income != min_income:
                income_component = (max_income - income) / (max_income - min_income)
            
            # 存款成分（越低越高）
            balance = household_balances.get(household_id, 0.0)
            balance_component = 0.0
            if max_balance != min_balance:
                balance_component = (max_balance - balance) / (max_balance - min_balance)
            
            # 综合权重
            weight = alpha * income_component + (1 - alpha) * balance_component
            poverty_weights[household_id] = weight
            total_weight += weight
        
        allocations = {}
        for household_id in all_workers:
            proportion = poverty_weights[household_id] / total_weight
            allocations[household_id] = total_tax * proportion
        
        return allocations

    def _unemployment_focused_allocation(self, all_workers: List[str], total_tax: float, month: int) -> Dict[str, float]:
        """失业导向分配策略（失业者获得更多）"""
        unemployment_weights = {}
        total_weight = 0.0
        
        for household_id in all_workers:
            labor_hours = self.laborhour.get(household_id, [])
            employed_count = sum(1 for lh in labor_hours if not lh.is_valid and lh.firm_id is not None)
            unemployed_count = len(labor_hours) - employed_count
            
            # 失业者权重更高
            weight = unemployed_count * 2.0 + employed_count * 1.0
            unemployment_weights[household_id] = weight
            total_weight += weight
        
        if total_weight <= 0:
            return self._equal_allocation(all_workers, total_tax)
        
        allocations = {}
        for household_id in all_workers:
            proportion = unemployment_weights[household_id] / total_weight
            allocations[household_id] = total_tax * proportion
        
        return allocations

    def _family_size_allocation(self, all_workers: List[str], total_tax: float) -> Dict[str, float]:
        """按家庭规模分配策略"""
        family_weights = {}
        total_weight = 0.0
        
        for household_id in all_workers:
            labor_hours = self.laborhour.get(household_id, [])
            family_size = len(labor_hours)
            family_weights[household_id] = family_size
            total_weight += family_size
        
        if total_weight <= 0:
            return self._equal_allocation(all_workers, total_tax)
        
        allocations = {}
        for household_id in all_workers:
            proportion = family_weights[household_id] / total_weight
            allocations[household_id] = total_tax * proportion
        
        return allocations

    def _mixed_allocation(self, all_workers: List[str], total_tax: float,
                         poverty_weight: float, unemployment_weight: float,
                         family_size_weight: float, month: int) -> Dict[str, float]:
        """混合分配策略"""
        # 获取各种权重
        poverty_allocations = self._poverty_focused_allocation(all_workers, total_tax, month)
        unemployment_allocations = self._unemployment_focused_allocation(all_workers, total_tax, month)
        family_size_allocations = self._family_size_allocation(all_workers, total_tax)
        equal_allocations = self._equal_allocation(all_workers, total_tax)
        
        # 计算剩余权重
        remaining_weight = 1.0 - poverty_weight - unemployment_weight - family_size_weight
        if remaining_weight < 0:
            remaining_weight = 0.0
        
        # 混合分配
        allocations = {}
        for household_id in all_workers:
            mixed_amount = (
                poverty_allocations[household_id] * poverty_weight +
                unemployment_allocations[household_id] * unemployment_weight +
                family_size_allocations[household_id] * family_size_weight +
                equal_allocations[household_id] * remaining_weight
            )
            allocations[household_id] = mixed_amount
        
        return allocations


    # =========================================================================
    # Transaction Creation Methods
    # =========================================================================
    def add_interest_tx(self, month: int, sender_id: str, receiver_id: str, amount: float) -> str:
        """
        添加利息交易记录
        """
        tx = self._record_transaction(
            sender_id=sender_id,
            receiver_id=receiver_id,
            amount=amount,
            tx_type='interest',
            month=month,
        )
        return tx.id
    def add_redistribution_tx(self, month: int, sender_id: str, receiver_id: str, amount: float) -> str:
        """
        添加再分配交易记录
        """
        tx = self._record_transaction(
            sender_id=sender_id,
            receiver_id=receiver_id,
            amount=amount,
            tx_type='redistribution',
            month=month,
        )
        return tx.id

    def add_tx_service(self, month: int, sender_id: str, receiver_id: str, amount: float) -> str:
        """
        添加服务类型交易记录，直接更新账本并记录到交易历史
        用于政府服务、基础服务等不需要商品库存的交易
        
        Args:
            month: 交易月份
            sender_id: 付款方ID
            receiver_id: 收款方ID
            amount: 交易金额
            
        Returns:
            str: 交易ID
        """
        # 🔧 修改：只检查家庭的余额，企业允许负债
        # 判断是否是企业：firm_id 在 self.firm_id 列表中
        is_company = sender_id in self.firm_id
        
        if not is_company and self.ledger[sender_id].amount < amount:
            # 家庭余额不足，不允许交易
            raise ValueError(f"Insufficient balance for household {sender_id}: ${self.ledger[sender_id].amount:.2f} < ${amount:.2f}")
        elif is_company and self.ledger[sender_id].amount < amount:
            # 企业余额不足，允许负债交易，记录日志
            self.logger.info(f"💳 Company {sender_id} transaction with negative balance: "
                      f"${self.ledger[sender_id].amount:.2f} → ${self.ledger[sender_id].amount - amount:.2f}")
        
        # 直接更新账本
        self.ledger[sender_id].amount -= amount
        self.ledger[receiver_id].amount += amount
        
        # 创建服务交易记录
        tx = self._record_transaction(
            sender_id=sender_id,
            receiver_id=receiver_id,
            amount=amount,
            assets=[],  # 服务交易没有具体商品
            tx_type='service',  # 使用service类型
            month=month,
        )
       
        return tx.id
    
    def add_inherent_market_transaction(
        self,
        month: int,
        sender_id: str,
        receiver_id: str,
        amount: float,
        product_id: str,
        quantity: float,
        product_name: str = "Unknown",
        product_price: float = 0.0,
        product_classification: str = "Unknown",
        consume_inventory: bool = False,
    ) -> str:
        """
        添加固有市场交易记录（包含毛利率计算）
        用于记录政府通过固有市场购买企业商品的交易
        
        Args:
            month: 交易月份
            sender_id: 付款方ID (通常是政府)
            receiver_id: 收款方ID (企业)
            amount: 交易金额
            product_id: 商品ID
            quantity: 购买数量
            product_name: 商品名称
            product_price: 商品单价
            product_classification: 商品分类（daily_cate）
            
        Returns:
            str: 交易ID
        """
        # 🔧 修改：只检查家庭和政府的余额，企业允许负债
        is_company = sender_id in self.firm_id
        
        if not is_company and self.ledger[sender_id].amount < amount:
            # 家庭/政府余额不足，不允许交易
            raise ValueError(f"Insufficient balance for {sender_id}: ${self.ledger[sender_id].amount:.2f} < ${amount:.2f}")
        elif is_company and self.ledger[sender_id].amount < amount:
            # 企业余额不足，允许负债交易
            self.logger.info(f"💳 Company {sender_id} inherent market transaction with negative balance: "
                      f"${self.ledger[sender_id].amount:.2f} → ${self.ledger[sender_id].amount - amount:.2f}")

        # 🔒 注意：固有市场可选择在此处原子扣库存（consume_inventory=True），避免“先扣库存后记账”失败导致不一致。
        # 验证商品是否存在并记录当前库存状态
        product_found = False
        current_inventory = 0.0
        if receiver_id in self.products:
            for product in self.products[receiver_id]:
                if product.product_id == product_id:
                    product_found = True
                    current_inventory = product.amount
                    if consume_inventory:
                        eps = 1e-9
                        if current_inventory + eps < quantity:
                            raise ValueError(
                                f"Insufficient inventory for {receiver_id}:{product_id}: "
                                f"{current_inventory} < {quantity}"
                            )
                        product.amount = max(0.0, float(product.amount) - float(quantity))
                        current_inventory = product.amount
                        self.logger.info(
                            f"固有市场购买: 企业 {receiver_id} 商品 {product_name} 消耗 {quantity}件，剩余 {current_inventory}件"
                        )
                    else:
                        # 旧行为：库存已在调用方扣减，这里仅记录扣减后的库存
                        self.logger.info(
                            f"固有市场购买: 企业 {receiver_id} 商品 {product_name} 已消耗 {quantity}件，剩余 {current_inventory}件"
                        )
                    break

        if not product_found:
            self.logger.warning(f"固有市场购买: 未找到企业 {receiver_id} 的商品 {product_id}")
            if consume_inventory:
                raise ValueError(f"Product not found for inherent market: {receiver_id}:{product_id}")

        # 政府/买方支付企业（不含税销售额）
        self.ledger[sender_id].amount -= amount
        self.ledger[receiver_id].amount += amount

        # 🧾 固有市场同样计入 VAT（消费税）
        # 逻辑与家庭购买一致：税基为不含税销售额 amount，税额=amount*vat_rate。
        # 若 sender 本身就是政府（gov_main_simulation），该税款在账面上“转给自己”不会改变余额，
        # 但仍会生成 consume_tax 交易记录，供统计与GDP核算使用。
        tax_amount = float(amount or 0.0) * float(self.vat_rate or 0.0)
        if tax_amount > 0:
            gov_id = "gov_main_simulation"
            # 确保政府账本存在
            if gov_id in self.ledger:
                self.ledger[sender_id].amount -= tax_amount
                self.ledger[gov_id].amount += tax_amount
            tax_tx = self._record_transaction(
                sender_id=sender_id,
                receiver_id=gov_id,
                amount=tax_amount,
                tx_type='consume_tax',
                month=month,
                metadata={
                    "tax_base": amount,
                    "tax_rate": self.vat_rate,
                    "product_id": product_id,
                    "quantity": float(quantity or 0.0),
                },
            )
        
        # 💰 企业收入（现金流口径）：只记录真实收款额；生产成本在生产阶段记支出
        revenue = amount
        self.record_firm_income(receiver_id, revenue)
        self.record_firm_monthly_income(receiver_id, month, revenue)
        
        # 创建固有市场交易记录
        unit_price = product_price if product_price > 0 else (amount / quantity if quantity > 0 else 0)
        if unit_price <= 0:
            unit_price = 0.01
            
        product_kwargs = dict(
            asset_type='products',
            product_id=product_id,
            name=product_name,
            owner_id=receiver_id,
            amount=quantity,
            price=unit_price,
            classification=product_classification
        )
        product_kwargs = inject_product_attributes(product_kwargs, product_id)
        product_asset = Product(**product_kwargs)
        
        tx = self._record_transaction(
            sender_id=sender_id,
            receiver_id=receiver_id,
            amount=amount,
            assets=[product_asset],
            tx_type='inherent_market',
            month=month,
            metadata={
                "product_id": product_id,
                "product_name": product_name,
                "quantity": float(quantity or 0.0),
                "unit_price": float(unit_price or 0.0),
                "product_classification": product_classification,
                "consume_inventory": consume_inventory,
            },
        )
        
        # 企业所得税改为“月度结算”（按净利润计税），避免与生产预算形成循环依赖。
        
        # self.logger.info(f"固有市场交易: 政府购买商品 {product_name}(ID:{product_id}, {product_classification}) "
        #            f"数量 {quantity} 金额 ${amount:.2f}, 成本 ${cost:.2f}, 毛利润 ${gross_profit:.2f} (毛利率{profit_margin}%), "
        #            f"企业所得税 ${corporate_tax:.2f}")
        
        return tx.id

    def add_government_procurement_transaction(
        self,
        month: int,
        sender_id: str,
        receiver_id: str,
        amount: float,
        product_id: str,
        quantity: float,
        product_name: str = "Unknown",
        unit_price: float = 0.0,
        product_classification: str = "Unknown",
        consume_inventory: bool = True,
    ) -> str:
        """
        Government procurement transaction:
        - No VAT/consume_tax is generated (avoid government self-tax artifacts).
        - Books firm revenue (cashflow) equal to `amount` (ex-tax).
        - Optionally consumes inventory atomically.
        """
        # Balance check (government is not a company)
        is_company = sender_id in self.firm_id
        if not is_company and self.ledger[sender_id].amount < amount:
            raise ValueError(f"Insufficient balance for {sender_id}: ${self.ledger[sender_id].amount:.2f} < ${amount:.2f}")

        # Inventory consume
        if consume_inventory:
            product_found = False
            current_inventory = 0.0
            if receiver_id in self.products:
                for p in (self.products.get(receiver_id) or []):
                    if str(getattr(p, "product_id", "") or "") == str(product_id):
                        product_found = True
                        current_inventory = float(getattr(p, "amount", 0.0) or 0.0)
                        eps = 1e-9
                        if current_inventory + eps < float(quantity or 0.0):
                            raise ValueError(
                                f"Insufficient inventory for {receiver_id}:{product_id}: "
                                f"{current_inventory} < {quantity}"
                            )
                        p.amount = max(0.0, float(p.amount) - float(quantity))
                        current_inventory = float(p.amount)
                        # enrich fields from inventory product
                        try:
                            self._ensure_product_cost_fields(p, default_category=getattr(p, "classification", product_classification))
                            product_name = str(getattr(p, "name", product_name) or product_name)
                            product_classification = getattr(p, "classification", product_classification) or product_classification
                            if unit_price <= 0:
                                unit_price = float(getattr(p, "price", 0.0) or 0.0)
                        except Exception:
                            pass
                        break
            if not product_found:
                raise ValueError(f"Product not found for government procurement: {receiver_id}:{product_id}")

        # Ledger transfer
        self.ledger[sender_id].amount -= amount
        self.ledger[receiver_id].amount += amount

        # Firm revenue bookkeeping (cashflow)
        self.record_firm_income(receiver_id, amount)
        self.record_firm_monthly_income(receiver_id, month, amount)

        # Transaction asset payload (quantity = purchased quantity)
        if unit_price <= 0 and quantity and float(quantity) > 0:
            unit_price = float(amount) / float(quantity)
        if unit_price <= 0:
            unit_price = 0.01

        product_kwargs = dict(
            asset_type="products",
            product_id=str(product_id),
            name=str(product_name),
            owner_id=str(receiver_id),
            amount=float(quantity or 0.0),
            price=float(unit_price),
            classification=str(product_classification or "Unknown"),
        )
        product_kwargs = inject_product_attributes(product_kwargs, str(product_id))
        product_asset = Product(**product_kwargs)

        tx = self._record_transaction(
            sender_id=sender_id,
            receiver_id=receiver_id,
            amount=float(amount or 0.0),
            assets=[product_asset],
            tx_type="government_procurement",
            month=month,
            metadata={
                "product_id": product_id,
                "product_name": product_name,
                "quantity": float(quantity or 0.0),
                "unit_price": float(unit_price or 0.0),
                "product_classification": product_classification,
                "consume_inventory": consume_inventory,
            },
        )
        return tx.id
    

    # =========================================================================
    # Inventory & Pricing Management
    # =========================================================================
    def get_product_inventory(self, owner_id: str, product_id: str) -> float:
        """
        获取指定商品的当前库存数量
        """
        if owner_id not in self.products:
            return 0.0
        
        for product in self.products[owner_id]:
            if product.product_id == product_id:
                return product.amount
        return 0.0
    
    def get_all_product_inventory(self) -> Dict[tuple, float]:
        """
        批量获取所有商品的库存信息
        
        Returns:
            Dict[tuple, float]: {(product_id, owner_id): amount} 字典
        """
        inventory_dict = {}
        for owner_id, products in self.products.items():
            for product in products:
                key = (product.product_id, owner_id)
                inventory_dict[key] = product.amount
        return inventory_dict
    
    async def sync_product_inventory_to_market(self, product_market):
        """
        将EconomicCenter的库存信息同步到ProductMarket
        这个方法可以定期调用以保持两边数据一致
        """
        try:
            # 收集所有有库存的商品
            all_products = []
            for owner_id, products in self.products.items():
                if owner_id in self.firm_id:
                    for product in products:
                        if product.amount > 0:  # 只同步有库存的商品
                            all_products.append(product)
            
            # 更新ProductMarket的商品列表
            await product_market.update_products_from_economic_center.remote(all_products)
            self.logger.info(f"已同步 {len(all_products)} 个商品到ProductMarket")
            return True
        except Exception as e:
            self.logger.error(f"同步库存到ProductMarket失败: {e}")
            return False
    
    # Sales Statistics & Market Analysis
    # =========================================================================
    def collect_sales_statistics(self, month: int) -> Dict[tuple, Dict]:
        """
        收集指定月份的销售统计数据
        返回: {(product_id, seller_id): {
            "product_id": str,
            "seller_id": str,
            "quantity_sold": float,
            "revenue": float,
            "demand_level": str,
            "household_quantity": float,  # 家庭购买数量
            "household_revenue": float,  # 家庭购买收入
            "inherent_market_quantity": float,  # 固定市场消耗数量
            "inherent_market_revenue": float,  # 固有市场收入
            "government_procurement_quantity": float,  # 政府采购数量（不含税）
            "government_procurement_revenue": float,  # 政府采购收入（不含税）
        }}
        
        注意：使用 (product_id, seller_id) 作为key，支持竞争市场模式下同一商品由多个企业销售
        """
        sales_stats = {}
        
        # 从交易历史中收集销售数据
        transactions = self.tx_by_month.get(month)
        if transactions is None:
            transactions = self.tx_history
        for tx in transactions:
            if tx.month == month:
                seller_id = tx.receiver_id
                
                # 处理家庭购买（purchase类型）
                if tx.type == 'purchase':
                    for asset in tx.assets:
                        if hasattr(asset, 'product_id') and asset.product_id:
                            product_id = asset.product_id
                            key = (product_id, seller_id)
                            
                            if key not in sales_stats:
                                sales_stats[key] = {
                                    "product_id": product_id,
                                    "seller_id": seller_id,
                                    "quantity_sold": 0.0,
                                    "revenue": 0.0,
                                    "demand_level": "normal",
                                    "household_quantity": 0.0,
                                    "household_revenue": 0.0,  # 新增：家庭购买收入
                                    "inherent_market_quantity": 0.0,
                                    "inherent_market_revenue": 0.0,  # 新增：固有市场收入
                                    "government_procurement_quantity": 0.0,
                                    "government_procurement_revenue": 0.0,
                                }
                            
                            # 累计家庭销量和收入
                            household_revenue = asset.price * asset.amount
                            sales_stats[key]["quantity_sold"] += asset.amount
                            sales_stats[key]["household_quantity"] += asset.amount
                            sales_stats[key]["revenue"] += household_revenue
                            sales_stats[key]["household_revenue"] += household_revenue

                
                # 处理固定市场消耗（inherent_market类型）
                elif tx.type == 'inherent_market':
                    for asset in tx.assets:
                        if hasattr(asset, 'product_id') and asset.product_id:
                            product_id = asset.product_id
                            key = (product_id, seller_id)
                            
                            if key not in sales_stats:
                                sales_stats[key] = {
                                    "product_id": product_id,
                                    "seller_id": seller_id,
                                    "quantity_sold": 0.0,
                                    "revenue": 0.0,
                                    "demand_level": "normal",
                                    "household_quantity": 0.0,
                                    "household_revenue": 0.0,  # 新增：家庭购买收入
                                    "inherent_market_quantity": 0.0,
                                    "inherent_market_revenue": 0.0,  # 新增：固有市场收入
                                    "government_procurement_quantity": 0.0,
                                    "government_procurement_revenue": 0.0,
                                }
                            
                            # 累计固定市场销量和收入
                            inherent_revenue = tx.amount  # 固定市场交易的总金额
                            sales_stats[key]["quantity_sold"] += asset.amount
                            sales_stats[key]["inherent_market_quantity"] += asset.amount
                            sales_stats[key]["revenue"] += inherent_revenue
                            sales_stats[key]["inherent_market_revenue"] += inherent_revenue

                # 处理政府采购（government_procurement类型，不含税）
                elif tx.type == 'government_procurement':
                    for asset in tx.assets:
                        if hasattr(asset, 'product_id') and asset.product_id:
                            product_id = asset.product_id
                            key = (product_id, seller_id)

                            if key not in sales_stats:
                                sales_stats[key] = {
                                    "product_id": product_id,
                                    "seller_id": seller_id,
                                    "quantity_sold": 0.0,
                                    "revenue": 0.0,
                                    "demand_level": "normal",
                                    "household_quantity": 0.0,
                                    "household_revenue": 0.0,
                                    "inherent_market_quantity": 0.0,
                                    "inherent_market_revenue": 0.0,
                                    "government_procurement_quantity": 0.0,
                                    "government_procurement_revenue": 0.0,
                                }

                            gp_revenue = asset.price * asset.amount
                            sales_stats[key]["quantity_sold"] += asset.amount
                            sales_stats[key]["government_procurement_quantity"] += asset.amount
                            sales_stats[key]["revenue"] += gp_revenue
                            sales_stats[key]["government_procurement_revenue"] += gp_revenue
        
        # 根据销量确定需求水平
        # ===== Unmet Demand Tracking =====
        unmet_month = dict(self.unmet_demand_by_month.get(month, {}) or {})
        for key, stats in sales_stats.items():
            try:
                unmet_key = self._unmet_key(stats.get("product_id"), stats.get("seller_id"))
                rec = unmet_month.get(unmet_key, {}) if unmet_month else {}
                stats["unmet_attempts"] = float((rec or {}).get("attempts", 0.0) or 0.0)
                stats["unmet_qty_short"] = float((rec or {}).get("qty_short", 0.0) or 0.0)
            except Exception:
                stats["unmet_attempts"] = 0.0
                stats["unmet_qty_short"] = 0.0

            quantity = stats["quantity_sold"]
            if quantity > 100:
                stats["demand_level"] = "high"
            elif quantity < 10:
                stats["demand_level"] = "low"
            else:
                stats["demand_level"] = "normal"
        
        print(f"📊 销售数据收集: 月份{month}, 交易记录{len(self.tx_history)}条, 销售商品-企业组合{len(sales_stats)}种")
        
        # 计算总收入统计
        total_revenue = sum(s['revenue'] for s in sales_stats.values())
        total_household_revenue = sum(s.get('household_revenue', 0) for s in sales_stats.values())
        total_inherent_revenue = sum(s.get('inherent_market_revenue', 0) for s in sales_stats.values())
        total_gp_revenue = sum(s.get('government_procurement_revenue', 0) for s in sales_stats.values())
        
        if total_revenue > 0:
            household_ratio = (total_household_revenue / total_revenue) * 100
            inherent_ratio = (total_inherent_revenue / total_revenue) * 100
            gp_ratio = (total_gp_revenue / total_revenue) * 100
            print(f"💰 收入统计: 总收入${total_revenue:.2f} | "
                  f"家庭购买${total_household_revenue:.2f} ({household_ratio:.1f}%) | "
                  f"政府采购${total_gp_revenue:.2f} ({gp_ratio:.1f}%) | "
                  f"固有市场${total_inherent_revenue:.2f} ({inherent_ratio:.1f}%)")
        
        if sales_stats:
            # 显示销量最高的3个商品-企业组合，并区分家庭和固定市场
            top_sales = sorted(sales_stats.items(), key=lambda x: x[1]['quantity_sold'], reverse=True)[:3]
            for (product_id, seller_id), stats in top_sales:
                household_rev = stats.get('household_revenue', 0)
                inherent_rev = stats.get('inherent_market_revenue', 0)
                gp_rev = stats.get('government_procurement_revenue', 0)
                total_rev = stats['revenue']
                hh_ratio = (household_rev / total_rev * 100) if total_rev > 0 else 0
                in_ratio = (inherent_rev / total_rev * 100) if total_rev > 0 else 0
                gp_ratio = (gp_rev / total_rev * 100) if total_rev > 0 else 0
                
                print(f"   - {product_id}@{seller_id}: 总销量{stats['quantity_sold']:.1f} "
                      f"(家庭:{stats['household_quantity']:.1f} | 政府采购:{stats.get('government_procurement_quantity', 0.0):.1f} | 固有市场:{stats['inherent_market_quantity']:.1f}), "
                      f"总收入${total_rev:.2f} (家庭:${household_rev:.2f} {hh_ratio:.1f}% | "
                      f"政府:${gp_rev:.2f} {gp_ratio:.1f}% | 固有:${inherent_rev:.2f} {in_ratio:.1f}%)")
        return sales_stats

    def settle_monthly_corporate_tax(self, month: int) -> Dict[str, float]:
        """
        月度企业所得税结算（按净利润计税）。

        结算时点：应在“工资发放完成后、生产补货开始前”执行，以便生产预算上限为：
        income - corporate_tax - wages。

        税基口径（现金流/费用发生制，与你当前“当月成本K在生产阶段记支出”的记账一致）：
        - 税前利润 = 当月总收入 − 当月总支出（工资/生产成本/其它费用，不含企业税）
        - 税额 = max(0, 税前利润) × corporate_tax_rate

        注意：如果你希望把“未售出库存的生产成本”资本化（用 COGS/库存变动来核算利润），
        这里的税基也应同步切换为“收入−销货成本−工资…”的口径。
        """
        if month in self._corporate_tax_settled_months:
            return {}

        results: Dict[str, float] = {}
        gov_id = "gov_main_simulation"
        if gov_id not in self.ledger:
            # 若政府账本未初始化，直接跳过（避免崩溃）
            self._corporate_tax_settled_months.add(month)
            return results

        for firm_id in list(self.firm_id):
            if firm_id not in self.ledger:
                self.ledger[firm_id] = Ledger.create(firm_id, 0.0)

            income = float(self.firm_monthly_financials.get(firm_id, {}).get(month, {}).get("income", 0.0) or 0.0)
            expenses_pre_tax = float(self.firm_monthly_financials.get(firm_id, {}).get(month, {}).get("expenses", 0.0) or 0.0)
            taxable_profit = max(0.0, income - expenses_pre_tax)
            corporate_tax = taxable_profit * float(self.corporate_tax_rate or 0.0)

            if corporate_tax <= 1e-9:
                results[firm_id] = 0.0
                continue

            # 🔧 修改：允许企业负债缴税，即使余额为负也要扣税
            # 这样可以模拟企业即使亏损也需要缴纳企业所得税的情况
            if self.ledger[firm_id].amount < corporate_tax:
                self.logger.info(f"💳 Company {firm_id} paying tax with insufficient balance: "
                          f"${self.ledger[firm_id].amount:.2f} → ${self.ledger[firm_id].amount - corporate_tax:.2f}")
            
            # 直接扣税，允许余额变为负数
            # 如果企业原本就是负债，会进一步增加负债
            self.ledger[firm_id].amount -= corporate_tax
            self.ledger[gov_id].amount += corporate_tax

            # 账务记录
            self.record_firm_expense(firm_id, corporate_tax)
            self.record_firm_monthly_expense(firm_id, month, corporate_tax)
            self.firm_monthly_corporate_tax[firm_id][month] += corporate_tax

            corp_tax_tx = self._record_transaction(
                sender_id=firm_id,
                receiver_id=gov_id,
                amount=corporate_tax,
                tx_type='corporate_tax',
                month=month,
                metadata={
                    "taxable_profit": taxable_profit,
                    "tax_rate": self.corporate_tax_rate,
                    "income": income,
                    "expenses_pre_tax": expenses_pre_tax,
                },
            )
            results[firm_id] = corporate_tax

        self._corporate_tax_settled_months.add(month)
        return results

    # Production Statistics & GDP Calculation
    # =========================================================================
    async def update_tax_rates(self, income_tax_rate: float = None, vat_rate: float = None, corporate_tax_rate: float = None):
        """
        更新税率
        """
        if income_tax_rate is not None:
            self.income_tax_rate = income_tax_rate
        if vat_rate is not None:
            self.vat_rate = vat_rate
        if corporate_tax_rate is not None:
            self.corporate_tax_rate = corporate_tax_rate

        self.logger.info(f"税率已更新: income_tax_rate={self.income_tax_rate:.1%}, vat_rate={self.vat_rate:.1%}, corporate_tax_rate={self.corporate_tax_rate:.1%}")

# ======================== 创新系统相关方法 ========================


    # =========================================================================
    # Innovation System
    # =========================================================================
        return self.firm_innovation_events


    def query_production_stats_by_month(self, month: int) -> Dict[str, Any]:
        """查询并返回某个月份的生产统计（包含劳动与创新细节）。若无则返回空字典。"""
        return self.production_stats_by_month.get(month, {})

    # ======================== GDP 核算（生产法/支出法/收入法） ========================
    def _infer_firm_category(self, firm_id: str) -> Optional[str]:
        """
        尝试从企业库存中推断企业所属大类（用于毛利率）。
        规则：取该企业库存中第一个带 classification 的商品。
        """
        try:
            for p in (self.products.get(firm_id, []) or []):
                cate = getattr(p, "classification", None)
                if cate:
                    return cate
        except Exception:
            pass
        return None

    def _get_firm_margin_rate(self, firm_id: str) -> float:
        """
        获取企业毛利率（rate），默认 25%。
        注意：毛利率定义为 (售价-成本)/售价，因此 售价 = 成本 / (1-毛利率)。
        """
        try:
            cate = self._infer_firm_category(firm_id) or "Unknown"
            margin_pct = float(self.category_profit_margins.get(cate, 25.0) or 25.0)
            margin_pct = max(0.0, min(80.0, margin_pct))
            return margin_pct / 100.0
        except Exception:
            return 0.25

    def calculate_nominal_gdp_and_health(self, month: int) -> Dict[str, Any]:
        """
        计算"名义GDP"及系统健康度指标
        
        名义GDP定义：家庭消费 + 固有市场销售（含税，反映实际交易规模）
        同时输出生产总值作为对比指标
        
        这不是严格的国民核算GDP，而是系统活跃度/规模的代理指标。
        同时提供多个维度的健康度指标用于诊断系统运行状态。
        """
        # 1) 主指标：名义GDP（交易总额法）
        sales_stats = self.collect_sales_statistics(month)
        household_sales_ex_tax = float(sum((s.get("household_revenue", 0.0) or 0.0) for s in (sales_stats or {}).values()) or 0.0)
        inherent_sales_ex_tax = float(sum((s.get("inherent_market_revenue", 0.0) or 0.0) for s in (sales_stats or {}).values()) or 0.0)
        gov_sales_ex_tax = float(sum((s.get("government_procurement_revenue", 0.0) or 0.0) for s in (sales_stats or {}).values()) or 0.0)
        total_sales_ex_tax = household_sales_ex_tax + inherent_sales_ex_tax + gov_sales_ex_tax

        transactions = self.tx_by_month.get(month)
        if transactions is None:
            transactions = self.tx_history
        
        # VAT
        vat_collected = 0.0
        for tx in transactions:
            if tx.month == month and tx.type == "consume_tax":
                vat_collected += float(tx.amount or 0.0)
        
        # 名义GDP（主指标）= 总消费（含税）
        nominal_gdp_transaction = total_sales_ex_tax + vat_collected
        
        # 2) 对比指标：名义GDP（生产总值法/生产侧口径）
        # ✅ 统一CD生产后，生产统计会直接给出 total_output_value（按售价估值的产出总价值）。
        ps = self.production_stats_by_month.get(month, {}) if hasattr(self, "production_stats_by_month") else {}
        total_production_cost = float(ps.get("total_production_cost", 0.0) or 0.0)

        total_output_value = float(ps.get("total_output_value", 0.0) or 0.0)
        if total_output_value > 0:
            nominal_gdp_production = total_output_value
            total_cost_based_production_value = total_output_value  # 保持字段含义：生产侧总价值
            total_labor_production_value = 0.0
        else:
            # 兼容旧统计：若没有 total_output_value，则回退到“成本推算 + 劳动力价值”
            total_labor_production_value = float(ps.get("total_labor_production_value", 0.0) or 0.0)
            total_cost_based_production_value = total_production_cost / (1 - 0.2) if total_production_cost > 0 else 0.0  # 旧：假设平均毛利率20%
            nominal_gdp_production = total_cost_based_production_value + total_labor_production_value
        
        # 供需匹配度：交易额 / 生产总值（理想值接近1.0）
        supply_demand_ratio = (nominal_gdp_transaction / nominal_gdp_production) if nominal_gdp_production > 0 else 0.0
        
        # 3) 收入分配
        total_wages = 0.0
        for tx in transactions:
            if tx.month == month and tx.type == "labor_payment":
                total_wages += float(tx.amount or 0.0)
        
        total_firm_revenue = total_sales_ex_tax
        total_firm_profit = total_firm_revenue - total_production_cost - total_wages  # 简化估算
        
        # 4) 库存健康
        total_inventory_value = 0.0
        for owner_id, products in self.products.items():
            for p in products:
                total_inventory_value += float(getattr(p, "amount", 0.0) or 0.0) * float(getattr(p, "price", 0.0) or 0.0)
        inventory_to_gdp_ratio = (total_inventory_value / nominal_gdp_transaction) if nominal_gdp_transaction > 0 else 0.0
        
        # 5) 财政健康
        labor_tax_collected = 0.0
        fica_tax_collected = 0.0
        corporate_tax_collected = 0.0
        for tx in transactions:
            if tx.month == month:
                if tx.type == "labor_tax":
                    labor_tax_collected += float(tx.amount or 0.0)
                elif tx.type == "fica_tax":
                    fica_tax_collected += float(tx.amount or 0.0)
                elif tx.type == "corporate_tax":
                    corporate_tax_collected += float(tx.amount or 0.0)
        
        total_tax_revenue = vat_collected + labor_tax_collected + fica_tax_collected + corporate_tax_collected
        gov_balance = self.ledger.get("gov_main_simulation", type('obj', (), {'amount': 0.0})()).amount
        
        # 6) 就业市场
        # ✅ 不依赖 self.households.employment_status（并行消费/轻量对象场景会缺失），改用交易与 laborhour 存量推断
        employed_count = 0
        for tx in transactions:
            if tx.month == month and tx.type == "labor_payment":
                employed_count += 1  # 每笔 labor_payment 近似对应一个劳动力单元（head/spouse）

        total_labor_force_units = 0
        try:
            for _hid, lhs in (self.laborhour or {}).items():
                total_labor_force_units += len(lhs or [])
        except Exception:
            total_labor_force_units = 0

        unemployed_count = max(0, int(total_labor_force_units) - int(employed_count))
        employment_rate = (float(employed_count) / float(total_labor_force_units)) if total_labor_force_units > 0 else 0.0
        average_wage = (total_wages / employed_count) if employed_count > 0 else 0.0
        
        # 7) 价格水平（简化：所有产品的加权平均价格）
        total_price_weighted = 0.0
        total_quantity = 0.0
        for owner_id, products in self.products.items():
            for p in products:
                qty = float(getattr(p, "amount", 0.0) or 0.0)
                price = float(getattr(p, "price", 0.0) or 0.0)
                total_price_weighted += price * qty
                total_quantity += qty
        average_price_level = (total_price_weighted / total_quantity) if total_quantity > 0 else 0.0
        
        return {
            "month": month,
            "nominal_gdp": nominal_gdp_transaction,  # 主指标：交易总额法
            "nominal_gdp_alternative": nominal_gdp_production,  # 对比指标：生产总值法
            "supply_demand_ratio": supply_demand_ratio,  # 供需匹配度（理想值~1.0）
            "gdp_components": {
                "household_consumption": household_sales_ex_tax + (household_sales_ex_tax * self.vat_rate),
                "government_procurement": gov_sales_ex_tax,  # 不含税（政府采购不缴VAT）
                "inherent_market_sales": inherent_sales_ex_tax + (inherent_sales_ex_tax * self.vat_rate),
                "vat_collected": vat_collected
            },
            "production_metrics": {
                "total_production_value": nominal_gdp_production,  # 生产总值法的GDP
                "cost_based_production_value": total_cost_based_production_value,  # 成本生产部分
                "total_production_cost": total_production_cost,
                "total_labor_production": total_labor_production_value
            },
            "income_distribution": {
                "total_wages": total_wages,
                "total_firm_profit": total_firm_profit,
                "wage_share": (total_wages / nominal_gdp_transaction) if nominal_gdp_transaction > 0 else 0.0,
                "profit_share": (total_firm_profit / nominal_gdp_transaction) if nominal_gdp_transaction > 0 else 0.0
            },
            "inventory_health": {
                "total_inventory_value": total_inventory_value,
                "inventory_to_gdp_ratio": inventory_to_gdp_ratio
            },
            "fiscal_health": {
                "total_tax_revenue": total_tax_revenue,
                "vat_revenue": vat_collected,
                "labor_tax_revenue": labor_tax_collected,
                "fica_tax_revenue": fica_tax_collected,
                "corporate_tax_revenue": corporate_tax_collected,
                "government_balance": gov_balance
            },
            "labor_market": {
                "employment_rate": employment_rate,
                "employed": employed_count,
                "unemployed": unemployed_count,
                "average_monthly_wage": average_wage
            },
            "price_level": {
                "average_price": average_price_level
            }
        }

    def calculate_monthly_gdp(self, month: int, production_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        计算月度 GDP（生产法 / 支出法 / 收入法），并给出分项。
        
        ⚠️ 注意：此方法计算的三种GDP方法在数学上是恒等的（由于采用同一套数据源），
        不代表真实的国民核算。如需系统健康度指标，建议使用 calculate_nominal_gdp_and_health()

        - 产品税：按"总消费（家庭+固有市场/政府等购买，均为不含税金额）× VAT税率"估算；
                 同时也会尝试从 tx_history 的 consume_tax 取"实际VAT"，若存在则优先使用。
        - 生产总价值（output）：优先使用生产阶段直接统计的“产出总价值”（例如统一CD生产的 total_output_value / firm_production_value）；
          若缺失才回退用"投入成本/ (1-毛利率)"粗略估算（仅用于兼容旧统计）。
        - 中间消耗：基础生产投入成本（production_cost）。
        - 库存投资：output - sales（sales 为不含税销售额，含家庭+固有市场等）。
        - 收入法：税金 + 工资 + 营业盈余，其中营业盈余 = (output - 中间消耗) - 工资。
        """
        ps = production_stats
        if ps is None:
            ps = self.production_stats_by_month.get(month, {}) if hasattr(self, "production_stats_by_month") else {}
        ps = ps or {}

        # 1) 销售/消费（不含税）
        sales_stats = self.collect_sales_statistics(month)
        total_sales_ex_tax = float(sum((s.get("revenue", 0.0) or 0.0) for s in (sales_stats or {}).values()) or 0.0)
        household_sales_ex_tax = float(sum((s.get("household_revenue", 0.0) or 0.0) for s in (sales_stats or {}).values()) or 0.0)
        inherent_sales_ex_tax = float(sum((s.get("inherent_market_revenue", 0.0) or 0.0) for s in (sales_stats or {}).values()) or 0.0)

        transactions = self.tx_by_month.get(month)
        if transactions is None:
            transactions = self.tx_history

        # 2) VAT（产品税）
        vat_rate = float(self.vat_rate or 0.0)
        vat_estimated = total_sales_ex_tax * vat_rate
        vat_actual = 0.0
        try:
            for tx in transactions:
                if tx.month == month and tx.type == "consume_tax":
                    vat_actual += float(tx.amount or 0.0)
        except Exception:
            vat_actual = 0.0
        product_taxes_vat = vat_actual if vat_actual > 0 else vat_estimated

        # 3) 生产：基础生产成本、基础生产总价值
        firm_cost = (ps.get("firm_production_cost", {}) or {})
        firm_value_reported = (ps.get("firm_production_value", {}) or {})
        total_base_cost = float(ps.get("total_production_cost", 0.0) or sum((float(v or 0.0) for v in firm_cost.values())) or 0.0)

        # 优先：统一CD生产会提供 total_output_value；否则用 firm_production_value 聚合
        total_output_value_reported = float(ps.get("total_output_value", 0.0) or 0.0)
        total_base_value_reported = (
            total_output_value_reported
            if total_output_value_reported > 0
            else float(sum((float(v or 0.0) for v in firm_value_reported.values())) or 0.0)
        )

        # 兼容旧统计：若缺失产出价值，再用“成本/(1-margin)”估算（最后兜底）
        base_value_inferred_from_cost_margin = {}
        total_base_value_inferred_from_cost_margin = 0.0
        if total_base_value_reported <= 1e-12:
            try:
                for cid, c in firm_cost.items():
                    cost = float(c or 0.0)
                    m = self._get_firm_margin_rate(str(cid))
                    denom = 1.0 - float(m)
                    value = (cost / denom) if denom > 1e-9 else 0.0
                    base_value_inferred_from_cost_margin[str(cid)] = value
                    total_base_value_inferred_from_cost_margin += value
            except Exception:
                total_base_value_inferred_from_cost_margin = 0.0

        total_base_value_used = total_base_value_reported if total_base_value_reported > 0 else total_base_value_inferred_from_cost_margin

        # 4) 劳动力生产总价值（保持现有逻辑，不改）
        total_labor_value = float(ps.get("total_labor_production_value", 0.0) or 0.0)
        if total_labor_value <= 0:
            # 兜底：按 firm 维度累加
            try:
                total_labor_value = float(sum((float(v or 0.0) for v in (ps.get("firm_labor_production_value", {}) or {}).values())) or 0.0)
            except Exception:
                total_labor_value = 0.0

        # 5) Output / 中间消耗 / 增加值
        output_value_total = float(total_base_value_used + total_labor_value)
        intermediate_consumption = float(total_base_cost)  # 你的设定：中间消耗=生产投入成本
        gross_value_added = float(output_value_total - intermediate_consumption)

        # 6) 库存投资（你的设定：产出总价值 - 销售额）
        inventory_investment = float(output_value_total - total_sales_ex_tax)

        # 7) 工资（优先用 tx_history 的 labor_payment；否则用生产统计里的 total_wage_expenses）
        wages_from_stats = float(ps.get("total_wage_expenses", 0.0) or 0.0) # 税前
        wages_from_tx = 0.0
        try:
            for tx in transactions:
                if tx.month == month and tx.type == "labor_payment":
                    wages_from_tx += float(tx.amount or 0.0) # 税后
        except Exception:
            wages_from_tx = 0.0
        wages_used = wages_from_tx if wages_from_tx > 0 else wages_from_stats

        # 8) 营业盈余（Operating surplus）
        operating_surplus = float(gross_value_added - wages_used)

        # 9) 三种 GDP（按同一套分项构造，理论上应一致，仅有浮点/口径差）
        gdp_production = float(gross_value_added + product_taxes_vat)
        gdp_expenditure = float((total_sales_ex_tax + product_taxes_vat) + inventory_investment - intermediate_consumption)
        gdp_income = float(product_taxes_vat + wages_used + operating_surplus)

        # 10) 统计误差
        max_gdp = max(gdp_production, gdp_expenditure, gdp_income)
        min_gdp = min(gdp_production, gdp_expenditure, gdp_income)

        return {
            "month": month,
            "rates": {
                "vat_rate": vat_rate,
            },
            "consumption": {
                "total_sales_ex_tax": total_sales_ex_tax,
                "household_sales_ex_tax": household_sales_ex_tax,
                "inherent_market_sales_ex_tax": inherent_sales_ex_tax,
            },
            "taxes": {
                "vat_estimated_from_sales": vat_estimated,
                "vat_actual_from_tx": vat_actual,
                "vat_used": product_taxes_vat,
            },
            "production": {
                "base_production_cost": total_base_cost,
                "base_production_value_by_margin": total_base_value_used,
                "base_production_value_reported": total_base_value_reported,
                "base_production_value_inferred_from_cost_margin": total_base_value_inferred_from_cost_margin,
                "base_production_value_source": (
                    "reported" if total_base_value_reported > 0 else "inferred_from_cost_margin"
                ),
                "labor_production_value": total_labor_value,
                "output_value_total": output_value_total,
            },
            "accounts": {
                "intermediate_consumption": intermediate_consumption,
                "gross_value_added": gross_value_added,
                "inventory_investment": inventory_investment,
                "wages_from_stats": wages_from_stats,
                "wages_from_tx": wages_from_tx,
                "wages_used": wages_used,
                "operating_surplus": operating_surplus,
            },
            "gdp": {
                "production_approach": gdp_production,
                "expenditure_approach": gdp_expenditure,
                "income_approach": gdp_income,
                "statistical_discrepancy": {
                    "max_minus_min": float(max_gdp - min_gdp),
                    "production_minus_expenditure": float(gdp_production - gdp_expenditure),
                    "production_minus_income": float(gdp_production - gdp_income),
                    "expenditure_minus_income": float(gdp_expenditure - gdp_income),
                },
            },
        }
