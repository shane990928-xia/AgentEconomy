"""
Economic Center Module

This module implements the central economic system that manages all economic activities
in the agent-based economic simulation, including:

- Asset Management: Ledgers, products, capital stocks
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

import ray
from dotenv import load_dotenv

from agenteconomy.center.Model import *
from agenteconomy.center.transaction import PeriodStatistics
from agenteconomy.utils.load_io_table import load_value_added_components
from agenteconomy.utils.logger import get_logger

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
    def __init__(self, tax_policy: TaxPolicy = None):
        """
        Initialize EconomicCenter with tax rates
        
        Args:
            tax_policy: 税收政策配置（包含累进税阶梯）
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
        # 2️⃣ 行业利润率配置 (从IO表V003加载)
        # =========================================================================
        # V003 = Gross Operating Surplus (毛营业盈余率)
        # 直接作为各行业的目标利润率
        self._io_gross_surplus: Optional[Dict[str, float]] = None  # 延迟加载

        # =========================================================================
        # 3️⃣ 资产存储 (Asset Storage)
        # =========================================================================
        # 使用 lambda 确保 Ledger 有默认 amount=0.0
        self.ledger: Dict[str, Ledger] = defaultdict(lambda: Ledger(amount=0.0))  # 现金账本
        self.labor_market = None

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
        self.market_price_registry: Dict[str, Dict[str, str]] = defaultdict(dict)
        
        # =========================================================================
        # 6️⃣ 企业财务追踪 (Firm Financial Tracking)
        # 统一数据结构: firm_monthly_data[firm_id][month] = {income, expenses, wage, tax, production_cost, depreciation}
        # =========================================================================
        def _default_firm_month() -> Dict[str, float]:
            return {"income": 0.0, "expenses": 0.0, "wage": 0.0, "tax": 0.0, "production_cost": 0.0}
        self.firm_monthly_data: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(lambda: defaultdict(_default_firm_month))
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
        # 8️⃣ 未满足需求追踪 (Unmet Demand Tracking)
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
        print(f"  📊 个人所得税: 累进税制 ({len(self.income_tax_rate)} 档，年阈值自动转换为月)")
        for i, bracket in enumerate(self.income_tax_rate):
            monthly_cutoff = bracket.cutoff / 12.0
            if i + 1 < len(self.income_tax_rate):
                next_monthly = self.income_tax_rate[i+1].cutoff / 12.0
                print(f"     档位{i+1}: 月收入 ${monthly_cutoff:>7,.0f} - ${next_monthly:>7,.0f} → {bracket.rate:>5.1%}")
            else:
                print(f"     档位{i+1}: 月收入 ${monthly_cutoff:>7,.0f}+          → {bracket.rate:>5.1%}")
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

    def register(self, agent_type: Literal['government', 'household', 'firm', 'bank']):
        """
        Register the EconomicCenter in the specified agent type list.
        """
        if agent_type == 'government':
            self.government_id.append("economic_center")
        elif agent_type == 'household':
            self.household_id.append("economic_center")
        elif agent_type == 'firm':
            self.firm_id.append("economic_center")
        elif agent_type == 'bank':
            self.bank_id.append("economic_center")
             
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
        - 折旧费用计入 firm_monthly_data[firm_id][month]["expenses"]
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

    def register_id(self, agent_id: str, agent_type: Literal['government', 'household', 'firm', 'bank']):
        """
        Register an agent ID based on its type.
        """
        if agent_type == 'government':
            if agent_id not in self.government_id:
                self.government_id.append(agent_id)
        elif agent_type == 'household':
            if agent_id not in self.household_id:
                self.household_id.append(agent_id)
        elif agent_type == 'firm':
            if agent_id not in self.firm_id:
                self.firm_id.append(agent_id)
        elif agent_type == 'bank':
            if agent_id not in self.bank_id:
                self.bank_id.append(agent_id)

    def get_all_agent_ids(self) -> Dict[str, List[str]]:
        """Get all registered agent IDs by type."""
        return {
            "government": list(self.government_id),
            "household": list(self.household_id),
            "firm": list(self.firm_id),
            "bank": list(self.bank_id),
        }

    def set_labor_market(self, labor_market):
        self.labor_market = labor_market

    def register_market_price(self, market_type: str, industry_code: str, firm_id: str, price: float = 0.0) -> None:
        if not market_type or not industry_code or not firm_id:
            return
        self.market_price_registry[str(market_type)][str(industry_code)] = str(firm_id)

    def resolve_market_price_id(self, market_type: str, industry_code: str) -> Optional[str]:
        if not market_type or not industry_code:
            return None
        return self.market_price_registry.get(str(market_type), {}).get(str(industry_code))

    def _call_labor_market_sync(self, method_name: str, *args, **kwargs):
        """同步调用 labor_market 方法（仅用于非异步上下文）"""
        if self.labor_market is None:
            return None
        method = getattr(self.labor_market, method_name, None)
        if method is None:
            return None
        if 'ActorHandle' in str(type(self.labor_market)):
            # 注意：这会在 async actor 中产生警告，但某些同步方法需要它
            return ray.get(method.remote(*args, **kwargs))
        return method(*args, **kwargs)
    
    async def _call_labor_market_async(self, method_name: str, *args, **kwargs):
        """异步调用 labor_market 方法（在 async 方法中使用）"""
        if self.labor_market is None:
            return None
        method = getattr(self.labor_market, method_name, None)
        if method is None:
            return None
        if 'ActorHandle' in str(type(self.labor_market)):
            return await method.remote(*args, **kwargs)
        return method(*args, **kwargs)

    def _get_labor_snapshot(self) -> Dict[str, Dict[str, int]]:
        snapshot = self._call_labor_market_sync("get_labor_force_snapshot")
        if isinstance(snapshot, dict):
            return snapshot
        return {}


    # =========================================================================
    # Query Methods
    # =========================================================================
    def query_all_tx(self):
        return self.tx_history
    
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
    
    def query_financial_summary(self, agent_id: str) -> Dict[str, float]:
        """查询代理的财务摘要：余额、总收入、总支出（企业适用）"""
        balance = self.ledger[agent_id].amount if agent_id in self.ledger else 0.0
        total_income = sum(d.get("income", 0.0) for d in self.firm_monthly_data.get(agent_id, {}).values())
        total_expenses = sum(d.get("expenses", 0.0) for d in self.firm_monthly_data.get(agent_id, {}).values())
        return {
            "balance": balance,
            "total_income": total_income,
            "total_expenses": total_expenses,
            "net_profit": total_income - total_expenses
        }

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
    
    def record_firm_income(self, firm_id: str, amount: float, month: int = 0):
        """记录企业收入"""
        self.firm_monthly_data[firm_id][month]["income"] += amount
        
    def record_firm_expense(self, firm_id: str, amount: float, month: int = 0):
        """记录企业支出"""
        self.firm_monthly_data[firm_id][month]["expenses"] += amount
    
    def record_firm_monthly_income(self, firm_id: str, month: int, amount: float):
        """记录企业月度收入"""
        self.firm_monthly_data[firm_id][month]["income"] += amount
        
    def record_firm_monthly_expense(self, firm_id: str, month: int, amount: float):
        """记录企业月度支出"""
        self.firm_monthly_data[firm_id][month]["expenses"] += amount
    
    def query_firm_monthly_financials(self, firm_id: str, month: int) -> Dict[str, float]:
        """查询企业指定月份的财务数据"""
        data = self.firm_monthly_data.get(firm_id, {}).get(month, {})
        depreciation = float(self.firm_monthly_depreciation.get(firm_id, {}).get(month, 0.0) or 0.0)
        inc = float(data.get("income", 0.0) or 0.0)
        exp = float(data.get("expenses", 0.0) or 0.0)
        return {
            "monthly_income": inc,
            "monthly_expenses": exp,
            "monthly_profit": inc - exp,
            "monthly_depreciation": depreciation,
        }

    def query_all_firms_monthly_financials(self, month: int) -> Dict[str, Dict[str, float]]:
        """
        批量查询“所有企业”在指定月份的财务数据（减少Ray远程调用次数）。

        Returns:
            {firm_id: {"monthly_income":..., "monthly_expenses":..., "monthly_profit":...}}
        """
        result: Dict[str, Dict[str, float]] = {}
        for cid in list(self.firm_id or []):
            data = self.firm_monthly_data.get(cid, {}).get(month, {})
            inc = float(data.get("income", 0.0) or 0.0)
            exp = float(data.get("expenses", 0.0) or 0.0)
            dep = float(self.firm_monthly_depreciation.get(str(cid), {}).get(month, 0.0) or 0.0)
            result[str(cid)] = {
                "monthly_income": inc,
                "monthly_expenses": exp,
                "monthly_profit": inc - exp,
                "monthly_depreciation": dep,
            }
        return result

    def query_firm_monthly_wage_expenses(self, firm_id: str, month: int) -> float:
        """查询企业指定月份的工资总支出（税前 gross_wage）。"""
        return float(self.firm_monthly_data.get(firm_id, {}).get(month, {}).get("wage", 0.0) or 0.0)
    
    def query_firm_all_monthly_financials(self, firm_id: str) -> Dict[int, Dict[str, float]]:
        """查询企业所有月份的财务数据"""
        result = {}
        for month, data in self.firm_monthly_data.get(firm_id, {}).items():
            result[month] = {
                "monthly_income": data.get("income", 0.0),
                "monthly_expenses": data.get("expenses", 0.0),
                "monthly_profit": data.get("income", 0.0) - data.get("expenses", 0.0)
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
        用于"企业初始资金 = 初始库存总价值"等初始化场景。
        """
        if agent_id not in self.ledger:
            self.ledger[agent_id] = Ledger.create(agent_id, float(amount))
        else:
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
            self.ledger[agent_id] = Ledger.create(agent_id, float(amount))
        else:
            self.ledger[agent_id].amount += amount
    
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

        if tx_type in ("purchase", "product_sale", "government_procurement"):
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

    def reset_transactions(self) -> Dict[str, int]:
        """
        清空交易与月度统计，但保留账本与注册的主体/就业状态。
        """
        counts = {
            "tx_history": len(self.tx_history),
            "tx_by_month": len(self.tx_by_month),
            "tx_by_type": len(self.tx_by_type),
            "tx_by_party": len(self.tx_by_party),
            "period_statistics": len(self.period_statistics),
            "wage_history": len(self.wage_history),
        }
        self.tx_history.clear()
        self.tx_by_month = defaultdict(list)
        self.tx_by_type = defaultdict(list)
        self.tx_by_party = defaultdict(list)
        self.period_statistics = {}
        self.wage_history = []
        self.redistribution_record_per_person = defaultdict(float)

        # 保留 firm_monthly_data：预热期积累的企业收入/支出数据
        # 正式期 FM1 需要查询预热期 PM3 的收入来计算工资帽和裁员决策
        # 如果清空，所有企业在 FM1 都会被判定为 zero_revenue → 大规模裁员
        # self.firm_monthly_data 不清空
        self._corporate_tax_settled_months = set()
        # 保留 firm_monthly_depreciation 和 capital_investment（与 firm_monthly_data 一致）
        # self.firm_monthly_depreciation 不清空
        # self.firm_monthly_capital_investment 不清空
        self.unmet_demand_by_month = defaultdict(dict)

        if hasattr(self, "production_stats_by_month"):
            self.production_stats_by_month = {}

        return counts

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

        # 使用小的容差来避免浮点数精度问题
        EPSILON = 0.01  # 1 分钱的容差
        balance = float(self.ledger[buyer_id].amount or 0.0)
        cost = float(total_cost or 0.0)
        
        is_company = buyer_id in self.firm_id
        if not is_company and balance + EPSILON < cost:
            raise ValueError(
                f"Insufficient balance for {buyer_id}: ${balance:.2f} < ${cost:.2f}"
            )
        elif is_company and balance + EPSILON < cost:
            self.logger.info(
                f"💳 Company {buyer_id} intermediate goods purchase with negative balance: "
                f"${balance:.2f} → ${balance - cost:.2f}"
            )

        self.ledger[buyer_id].amount -= total_cost
        self.ledger[receiver_id].amount += total_cost

        if is_company:
            self.record_firm_expense(buyer_id, total_cost)
            self.record_firm_monthly_expense(buyer_id, month, total_cost)
            self.firm_monthly_data[buyer_id][month]["production_cost"] += total_cost

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

        # 使用小的容差来避免浮点数精度问题（如 $141.08 < $141.08 因精度问题判断为 True）
        EPSILON = 0.01  # 1 分钱的容差
        balance = float(self.ledger[buyer_id].amount or 0.0)
        cost = float(total_cost or 0.0)
        
        is_company = buyer_id in self.firm_id
        if not is_company and balance + EPSILON < cost:
            raise ValueError(
                f"Insufficient balance for {buyer_id}: ${balance:.2f} < ${cost:.2f}"
            )
        elif is_company and balance + EPSILON < cost:
            self.logger.info(
                f"💳 Company {buyer_id} resource purchase with negative balance: "
                f"${balance:.2f} → ${balance - cost:.2f}"
            )

        self.ledger[buyer_id].amount -= total_cost
        self.ledger[receiver_id].amount += total_cost

        # 记录买方（企业）的支出
        if is_company:
            self.record_firm_expense(buyer_id, total_cost)
            self.record_firm_monthly_expense(buyer_id, month, total_cost)
            self.firm_monthly_data[buyer_id][month]["production_cost"] += total_cost

        # 记录接收方的收入
        is_receiver_company = receiver_id in self.firm_id
        is_buyer_company = buyer_id in self.firm_id
        if is_receiver_company:
            self.record_firm_income(receiver_id, total_cost)
            self.record_firm_monthly_income(receiver_id, month, total_cost)
            if not is_buyer_company:
                # 家庭服务消费 → ServiceFirm：自动记录隐含运营成本
                # 服务业成本结构：~55% 中间投入/运营成本（租金、设备、耗材、折旧等）
                # 剩余 45% = 工资(~20-30%) + 利润(~15-25%)
                # 这里只记账面成本，不实际扣钱，影响的是企业所得税税基
                SERVICE_IMPLICIT_COST_RATIO = 0.55
                implied_cost = total_cost * SERVICE_IMPLICIT_COST_RATIO
                self.record_firm_expense(receiver_id, implied_cost)
                self.record_firm_monthly_expense(receiver_id, month, implied_cost)

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
    
    def process_purchase(
        self, 
        month: int, 
        buyer_id: str, 
        seller_id: str, 
        amount: float,
        quantity: float = 1.0,
        product_id: Optional[str] = None,
        product_name: Optional[str] = None,
        unit_price: Optional[float] = None,
    ) -> Optional[str]:
        """
        处理购买交易（纯转账，商品管理由商品市场负责）
        
        Args:
            month: 当前月份
            buyer_id: 买家ID
            seller_id: 卖家ID
            amount: 交易金额（不含税）
            quantity: 购买数量（用于记录）
            product_id: 商品ID（可选，用于记录）
            product_name: 商品名称（可选，用于记录）
            unit_price: 单价（可选，用于记录）
        
        Returns:
            交易ID（成功）或 None（失败）
        """
        # 计算总费用：标价 + 消费税
        base_price = float(amount)
        total_cost_with_tax = base_price * (1 + self.vat_rate)
        
        # 检查买家余额（使用小容差避免浮点数精度问题）
        EPSILON = 0.01
        if buyer_id not in self.ledger:
            self.ledger[buyer_id] = Ledger.create(buyer_id, 0.0)
        if self.ledger[buyer_id].amount + EPSILON < total_cost_with_tax:
            self.logger.warning(f"购买失败: 买家 {buyer_id} 余额不足 (需要 {total_cost_with_tax:.2f})")
            return None

        # 买家支付含税价格
        self.ledger[buyer_id].amount -= total_cost_with_tax

        # 创建消费税交易记录
        tax_amount = base_price * self.vat_rate
        self._record_transaction(
            sender_id=buyer_id,
            receiver_id="gov_main_simulation",
            amount=tax_amount,
            tx_type='consume_tax',
            month=month,
            metadata={
                "tax_base": base_price,
                "tax_rate": self.vat_rate,
            },
        )
        
        # 政府收取消费税
        if "gov_main_simulation" not in self.ledger:
            self.ledger["gov_main_simulation"] = Ledger.create("gov_main_simulation", 0.0)
        self.ledger["gov_main_simulation"].amount += tax_amount

        # 创建购买交易记录
        purchase_tx = self._record_transaction(
            sender_id=buyer_id,
            receiver_id=seller_id,
            amount=base_price,
            tx_type='purchase',
            month=month,
            metadata={
                "product_id": product_id,
                "product_name": product_name,
                "quantity": float(quantity or 0.0),
                "unit_price": float(unit_price or 0.0) if unit_price else base_price / max(quantity, 1),
            },
        )

        # 企业收入
        if seller_id not in self.ledger:
            self.ledger[seller_id] = Ledger.create(seller_id, 0.0)
        self.ledger[seller_id].amount += base_price
        self.record_firm_income(seller_id, base_price)
        self.record_firm_monthly_income(seller_id, month, base_price)
        
        return purchase_tx.id

    def process_wholesale(
        self,
        month: int,
        retailer_id: str,
        manufacturer_id: str,
        amount: float,
        quantity: float = 1.0,
        product_id: Optional[str] = None,
        product_name: Optional[str] = None,
        unit_price: Optional[float] = None,
    ) -> Optional[str]:
        """
        处理零售商从制造商进货的批发交易
        
        Args:
            month: 当前月份
            retailer_id: 零售商ID
            manufacturer_id: 制造商ID
            amount: 交易金额（批发价，即 manufacturer_price * quantity）
            quantity: 进货数量
            product_id: 商品ID（可选）
            product_name: 商品名称（可选）
            unit_price: 批发单价（可选）
        
        Returns:
            交易ID（成功）或 None（失败）
        """
        wholesale_amount = float(amount)
        
        # 检查零售商余额
        if retailer_id not in self.ledger:
            self.ledger[retailer_id] = Ledger.create(retailer_id, 0.0)
        if self.ledger[retailer_id].amount < wholesale_amount:
            self.logger.warning(
                f"进货失败: 零售商 {retailer_id} 余额不足 "
                f"(需要 {wholesale_amount:.2f}, 余额 {self.ledger[retailer_id].amount:.2f})"
            )
            return None

        # 零售商支付批发价
        self.ledger[retailer_id].amount -= wholesale_amount

        # 记录零售商的进货成本（影响企业所得税税基）
        # 没有这一步，taxable_profit = 销售收入 - 工资，进货成本被遗漏
        # 导致零售商利润虚高 → 企业所得税虚高 → 现金枯竭 → 进货失败
        self.record_firm_expense(retailer_id, wholesale_amount)
        self.record_firm_monthly_expense(retailer_id, month, wholesale_amount)

        # 创建批发交易记录
        wholesale_tx = self._record_transaction(
            sender_id=retailer_id,
            receiver_id=manufacturer_id,
            amount=wholesale_amount,
            tx_type='wholesale',
            month=month,
            metadata={
                "product_id": product_id,
                "product_name": product_name,
                "quantity": float(quantity or 0.0),
                "unit_price": float(unit_price or 0.0) if unit_price else wholesale_amount / max(quantity, 1),
            },
        )

        # 制造商收入
        if manufacturer_id not in self.ledger:
            self.ledger[manufacturer_id] = Ledger.create(manufacturer_id, 0.0)
        self.ledger[manufacturer_id].amount += wholesale_amount
        self.record_firm_income(manufacturer_id, wholesale_amount)
        self.record_firm_monthly_income(manufacturer_id, month, wholesale_amount)
        
        self.logger.debug(
            f"批发交易完成: {retailer_id} -> {manufacturer_id}, "
            f"金额={wholesale_amount:.2f}, 数量={quantity}"
        )
        
        return wholesale_tx.id

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
            if firm_id not in self.ledger:
                self.ledger[firm_id] = Ledger.create(firm_id, 0.0)
            # 检查余额，防止过度透支
            if self.ledger[firm_id].amount < gross_wage:
                self.logger.warning(
                    f"工资透支: 企业 {firm_id} 余额不足 "
                    f"(需要 {gross_wage:.2f}, 余额 {self.ledger[firm_id].amount:.2f})"
                )
            self.ledger[firm_id].amount -= gross_wage
            # 记录企业支出（经济中心层面）
            self.record_firm_expense(firm_id, gross_wage)
            # 记录企业月度支出
            self.record_firm_monthly_expense(firm_id, month, gross_wage)
            # 细分统计：月度工资支出（税前工资）
            self.firm_monthly_data[firm_id][month]["wage"] += gross_wage

        # 记录工资历史（记录税前工资）
        self.wage_history.append(Wage.create(household_id, gross_wage, month))
        # print(f"Month {month} Processed labor payment: ${gross_wage:.2f} gross (${net_wage:.2f} net, ${income_tax:.2f} tax) from {firm_id} to {household_id}")
        return wage_tx.id


    # =========================================================================
    # Tax Calculations
    # =========================================================================
    def calculate_progressive_income_tax(self, gross_wage: float) -> float:
        """
        Calculate the income tax for a given gross wage (MONTHLY).
        
        注意：配置文件中的税率阈值是按年定义的（# by year），
        这里自动转换为月阈值（÷12）进行计算。
        
        Args:
            gross_wage: 月工资（税前）
        
        Returns:
            月应缴个人所得税
        """
        total_tax = 0.0
        
        for i, bracket in enumerate(self.income_tax_rate):
            # 将年阈值转换为月阈值
            monthly_cutoff = bracket.cutoff / 12.0
            
            if gross_wage > monthly_cutoff:
                if i + 1 < len(self.income_tax_rate):
                    # 下一档的月阈值
                    upper_monthly_cutoff = self.income_tax_rate[i + 1].cutoff / 12.0
                else:
                    upper_monthly_cutoff = float('inf')
                
                taxable_in_bracket = min(gross_wage, upper_monthly_cutoff) - monthly_cutoff
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

    def summarize_households_monthly(self, month: int) -> Dict[str, Any]:
        """
        汇总指定月份的家庭收入/消费/余额（按交易统计）。
        - 收入：工资(税后)、利息、再分配
        - 消费：购物金额(不含税) + 消费税
        """
        households = [hid for hid in (self.household_id or []) if hid and hid != "economic_center"]
        by_household: Dict[str, Dict[str, Any]] = {}
        for hid in households:
            bal = float(self.ledger.get(hid).amount) if hid in self.ledger else 0.0
            by_household[hid] = {
                "income": {"wage": 0.0, "interest": 0.0, "redistribution": 0.0, "total": 0.0},
                "consumption": {"purchase": 0.0, "tax": 0.0, "total": 0.0},
                "balance": bal,
            }

        totals = {
            "wage": 0.0,
            "interest": 0.0,
            "redistribution": 0.0,
            "purchase": 0.0,
            "tax": 0.0,
        }
        government_procurement_total = 0.0

        transactions = self.tx_by_month.get(month)
        if transactions is None:
            transactions = self.tx_history
        for tx in transactions:
            if tx.month != month:
                continue
            ttype = getattr(tx, "type", None)
            sender_id = getattr(tx, "sender_id", None)
            receiver_id = getattr(tx, "receiver_id", None)
            amount = float(getattr(tx, "amount", 0.0) or 0.0)

            if ttype == "labor_payment" and receiver_id in by_household:
                by_household[receiver_id]["income"]["wage"] += amount
                totals["wage"] += amount
            elif ttype == "interest" and receiver_id in by_household:
                by_household[receiver_id]["income"]["interest"] += amount
                totals["interest"] += amount
            elif ttype == "redistribution" and receiver_id in by_household:
                by_household[receiver_id]["income"]["redistribution"] += amount
                totals["redistribution"] += amount
            elif ttype == "purchase" and sender_id in by_household:
                by_household[sender_id]["consumption"]["purchase"] += amount
                totals["purchase"] += amount
            elif ttype == "consume_tax" and sender_id in by_household:
                by_household[sender_id]["consumption"]["tax"] += amount
                totals["tax"] += amount
            elif ttype == "government_procurement":
                government_procurement_total += amount

        for rec in by_household.values():
            income_total = (
                rec["income"]["wage"]
                + rec["income"]["interest"]
                + rec["income"]["redistribution"]
            )
            rec["income"]["total"] = income_total
            cons_total = rec["consumption"]["purchase"] + rec["consumption"]["tax"]
            rec["consumption"]["total"] = cons_total

        aggregate_income_total = totals["wage"] + totals["interest"] + totals["redistribution"]
        aggregate_consumption_total = totals["purchase"] + totals["tax"]
        aggregate = {
            "income": {
                "wage": totals["wage"],
                "interest": totals["interest"],
                "redistribution": totals["redistribution"],
                "total": aggregate_income_total,
            },
            "consumption": {
                "purchase": totals["purchase"],
                "tax": totals["tax"],
                "total": aggregate_consumption_total,
            },
            "household_count": len(by_household),
        }

        return {
            "aggregate": aggregate,
            "by_household": by_household,
            "government_procurement_total": government_procurement_total,
        }
    

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
        
        labor_snapshot = self._get_labor_snapshot()
        all_workers = list(labor_snapshot.keys())
        if not all_workers:
            print(f"Month {month}: No households with labor hours found for tax redistribution")
            return {"total_redistributed": 0.0, "recipients": 0, "per_person": 0.0}
        
        # 根据策略计算分配金额
        household_allocations = self._calculate_redistribution_allocations(
            all_workers, total_tax, strategy, poverty_weight, unemployment_weight, family_size_weight, month, labor_snapshot
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
                                           month: int, labor_snapshot: Optional[Dict[str, Dict[str, int]]] = None) -> Dict[str, float]:
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
            return self._unemployment_focused_allocation(all_workers, total_tax, month, labor_snapshot)
        elif strategy == "family_size":
            return self._family_size_allocation(all_workers, total_tax, labor_snapshot)
        elif strategy == "mixed":
            return self._mixed_allocation(all_workers, total_tax, poverty_weight,
                                        unemployment_weight, family_size_weight, month, labor_snapshot)
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

    def _unemployment_focused_allocation(
        self,
        all_workers: List[str],
        total_tax: float,
        month: int,
        labor_snapshot: Optional[Dict[str, Dict[str, int]]] = None
    ) -> Dict[str, float]:
        """失业导向分配策略（失业者获得更多）"""
        unemployment_weights = {}
        total_weight = 0.0
        snapshot = labor_snapshot if labor_snapshot is not None else self._get_labor_snapshot()
        
        for household_id in all_workers:
            entry = snapshot.get(household_id, {})
            employed_count = int(entry.get("employed", 0) or 0)
            unemployed_count = int(entry.get("unemployed", 0) or 0)
            
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

    def _family_size_allocation(
        self,
        all_workers: List[str],
        total_tax: float,
        labor_snapshot: Optional[Dict[str, Dict[str, int]]] = None
    ) -> Dict[str, float]:
        """按家庭规模分配策略"""
        family_weights = {}
        total_weight = 0.0
        snapshot = labor_snapshot if labor_snapshot is not None else self._get_labor_snapshot()
        
        for household_id in all_workers:
            entry = snapshot.get(household_id, {})
            family_size = int(entry.get("total", 0) or 0)
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
                         family_size_weight: float, month: int,
                         labor_snapshot: Optional[Dict[str, Dict[str, int]]] = None) -> Dict[str, float]:
        """混合分配策略"""
        # 获取各种权重
        poverty_allocations = self._poverty_focused_allocation(all_workers, total_tax, month)
        unemployment_allocations = self._unemployment_focused_allocation(all_workers, total_tax, month, labor_snapshot)
        family_size_allocations = self._family_size_allocation(all_workers, total_tax, labor_snapshot)
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
        # 使用小容差避免浮点数精度问题
        EPSILON = 0.01
        is_company = sender_id in self.firm_id
        balance = float(self.ledger[sender_id].amount or 0.0)
        amt = float(amount or 0.0)
        
        if not is_company and balance + EPSILON < amt:
            # 家庭余额不足，不允许交易
            raise ValueError(f"Insufficient balance for household {sender_id}: ${balance:.2f} < ${amt:.2f}")
        elif is_company and balance + EPSILON < amt:
            # 企业余额不足，允许负债交易，记录日志
            self.logger.info(f"💳 Company {sender_id} transaction with negative balance: "
                      f"${balance:.2f} → ${balance - amt:.2f}")
        
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
        政府采购交易（纯转账，商品库存由商品市场管理）
        - 不产生VAT/消费税（避免政府自我征税）
        - 记录企业收入
        """
        # 确保发送方有账本条目
        if sender_id not in self.ledger:
            self.ledger[sender_id] = Ledger(amount=0.0)
        
        # 确保接收方有账本条目（制造商可能还未注册）
        if receiver_id not in self.ledger:
            self.ledger[receiver_id] = Ledger(amount=0.0)
        
        # 余额检查（使用小容差避免浮点数精度问题）
        EPSILON = 0.01
        is_company = sender_id in self.firm_id
        balance = float(self.ledger[sender_id].amount or 0.0)
        amt = float(amount or 0.0)
        if not is_company and balance + EPSILON < amt:
            raise ValueError(f"Insufficient balance for {sender_id}: ${balance:.2f} < ${amt:.2f}")

        # 转账
        self.ledger[sender_id].amount -= amount
        self.ledger[receiver_id].amount += amount

        # 企业收入记账
        self.record_firm_income(receiver_id, amount)
        self.record_firm_monthly_income(receiver_id, month, amount)

        # 计算单价
        if unit_price <= 0 and quantity and float(quantity) > 0:
            unit_price = float(amount) / float(quantity)
        if unit_price <= 0:
            unit_price = 0.01

        tx = self._record_transaction(
            sender_id=sender_id,
            receiver_id=receiver_id,
            amount=float(amount or 0.0),
            tx_type="government_procurement",
            month=month,
            metadata={
                "product_id": product_id,
                "product_name": product_name,
                "quantity": float(quantity or 0.0),
                "unit_price": float(unit_price or 0.0),
                "product_classification": product_classification,
            },
        )
        return tx.id


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
            "government_procurement_quantity": float,  # 政府采购数量（不含税）
            "government_procurement_revenue": float,  # 政府采购收入（不含税）
        }}
        
        注意：使用 (product_id, seller_id) 作为key，支持竞争市场模式下同一商品由多个企业销售
        """
        sales_stats: Dict[tuple, Dict[str, float]] = {}

        def _ensure_key(product_id: str, seller_id: str) -> Dict[str, float]:
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
                    "government_procurement_quantity": 0.0,
                    "government_procurement_revenue": 0.0,
                }
            return sales_stats[key]

        def _accumulate(
            *,
            kind: str,
            product_id: Optional[str],
            seller_id: Optional[str],
            quantity: Optional[float],
            revenue: Optional[float],
        ) -> None:
            if not product_id or not seller_id:
                return
            qty = float(quantity or 0.0)
            rev = float(revenue or 0.0)
            stats = _ensure_key(product_id, seller_id)
            stats["quantity_sold"] += qty
            stats["revenue"] += rev
            if kind == "household":
                stats["household_quantity"] += qty
                stats["household_revenue"] += rev
            elif kind == "government":
                stats["government_procurement_quantity"] += qty
                stats["government_procurement_revenue"] += rev

        def _extract_from_metadata(tx) -> Tuple[Optional[str], Optional[float], Optional[float]]:
            meta = getattr(tx, "metadata", {}) or {}
            product_id = meta.get("product_id")
            quantity = meta.get("quantity")
            unit_price = meta.get("unit_price")
            revenue = float(getattr(tx, "amount", 0.0) or 0.0)
            if (quantity is None or float(quantity or 0.0) <= 0.0) and unit_price:
                try:
                    quantity = float(revenue) / float(unit_price) if float(unit_price) > 0 else None
                except Exception:
                    quantity = quantity
            if revenue <= 0.0 and quantity is not None and unit_price is not None:
                try:
                    revenue = float(quantity) * float(unit_price)
                except Exception:
                    revenue = float(revenue or 0.0)
            return product_id, quantity, revenue

        def _accumulate_from_assets(tx, kind: str) -> None:
            seller_id = getattr(tx, "receiver_id", None)
            for asset in getattr(tx, "assets", []) or []:
                product_id = getattr(asset, "product_id", None)
                if not product_id:
                    continue
                qty = getattr(asset, "amount", None)
                price = getattr(asset, "price", None)
                revenue = None
                if qty is not None and price is not None:
                    revenue = float(qty) * float(price)
                else:
                    revenue = float(getattr(tx, "amount", 0.0) or 0.0)
                _accumulate(
                    kind=kind,
                    product_id=product_id,
                    seller_id=seller_id,
                    quantity=qty,
                    revenue=revenue,
                )

        # 从交易历史中收集销售数据（当前流程以 metadata 为主）
        transactions = self.tx_by_month.get(month)
        if transactions is None:
            transactions = self.tx_history
        tx_count = 0
        for tx in transactions:
            if tx.month != month:
                continue
            tx_count += 1
            seller_id = getattr(tx, "receiver_id", None)

            if tx.type == 'purchase':
                product_id, quantity, revenue = _extract_from_metadata(tx)
                if product_id:
                    _accumulate(
                        kind="household",
                        product_id=product_id,
                        seller_id=seller_id,
                        quantity=quantity,
                        revenue=revenue,
                    )
                else:
                    _accumulate_from_assets(tx, "household")

            elif tx.type == 'government_procurement':
                product_id, quantity, revenue = _extract_from_metadata(tx)
                if product_id:
                    _accumulate(
                        kind="government",
                        product_id=product_id,
                        seller_id=seller_id,
                        quantity=quantity,
                        revenue=revenue,
                    )
                else:
                    _accumulate_from_assets(tx, "government")
        
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
        
        print(f"📊 销售数据收集: 月份{month}, 交易记录{tx_count}条, 销售商品-企业组合{len(sales_stats)}种")
        
        # 计算总收入统计
        total_revenue = sum(s['revenue'] for s in sales_stats.values())
        total_household_revenue = sum(s.get('household_revenue', 0) for s in sales_stats.values())
        total_gp_revenue = sum(s.get('government_procurement_revenue', 0) for s in sales_stats.values())
        
        if total_revenue > 0:
            household_ratio = (total_household_revenue / total_revenue) * 100
            gp_ratio = (total_gp_revenue / total_revenue) * 100
            print(f"💰 收入统计: 总收入${total_revenue:.2f} | "
                  f"家庭购买${total_household_revenue:.2f} ({household_ratio:.1f}%) | "
                  f"政府采购${total_gp_revenue:.2f} ({gp_ratio:.1f}%)")
        
        if sales_stats:
            # 显示销量最高的3个商品-企业组合
            top_sales = sorted(sales_stats.items(), key=lambda x: x[1]['quantity_sold'], reverse=True)[:3]
            for (product_id, seller_id), stats in top_sales:
                household_rev = stats.get('household_revenue', 0)
                gp_rev = stats.get('government_procurement_revenue', 0)
                total_rev = stats['revenue']
                hh_ratio = (household_rev / total_rev * 100) if total_rev > 0 else 0
                gp_ratio = (gp_rev / total_rev * 100) if total_rev > 0 else 0
                
                print(f"   - {product_id}@{seller_id}: 总销量{stats['quantity_sold']:.1f} "
                      f"(家庭:{stats['household_quantity']:.1f} | 政府采购:{stats.get('government_procurement_quantity', 0.0):.1f}), "
                      f"总收入${total_rev:.2f} (家庭:${household_rev:.2f} {hh_ratio:.1f}% | "
                      f"政府:${gp_rev:.2f} {gp_ratio:.1f}%)")
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

            income = float(self.firm_monthly_data.get(firm_id, {}).get(month, {}).get("income", 0.0) or 0.0)
            expenses_pre_tax = float(self.firm_monthly_data.get(firm_id, {}).get(month, {}).get("expenses", 0.0) or 0.0)
            
            # ServiceFirm（svc_前缀）的服务消费收入中，大部分是隐含成本
            # （地租/折旧/中间投入/工资），不是纯利润。
            # 按 BEA IO 表，服务业平均利润率约 10-15%。
            # 这里限制 ServiceFirm 的应税利润不超过收入的 15%。
            if firm_id.startswith("svc_"):
                max_profit = income * 0.15  # 服务业利润率上限 15%
                taxable_profit = min(max(0.0, income - expenses_pre_tax), max_profit)
            else:
                taxable_profit = max(0.0, income - expenses_pre_tax)
            # ServiceFirm（svc_前缀）的利润率上限为 15%
            # 防止服务消费收入被全额当作利润征税
            if firm_id.startswith("svc_") and income > 0:
                max_profit = income * 0.15
                taxable_profit = min(taxable_profit, max_profit)
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
            self.firm_monthly_data[firm_id][month]["tax"] += corporate_tax

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


    def _get_firm_margin_rate(self, firm_id: str, industry_code: Optional[str] = None) -> float:
        """
        获取企业毛利率（rate），默认 15%。
        注意：毛利率定义为 (售价-成本)/售价，因此 售价 = 成本 / (1-毛利率)。
        基于IO表V003获取。
        """
        try:
            if not industry_code:
                industry_code = "Unknown"
            return 0.15  # 默认15%，行业利润率由商品市场管理
        except Exception:
            return 0.15  # 默认15%

    def calculate_nominal_gdp_and_health(self, month: int) -> Dict[str, Any]:
        """
        计算"名义GDP"及系统健康度指标
        
        名义GDP定义：家庭消费 + 政府采购（含税，反映实际交易规模）
        同时输出生产总值作为对比指标
        
        这不是严格的国民核算GDP，而是系统活跃度/规模的代理指标。
        同时提供多个维度的健康度指标用于诊断系统运行状态。
        """
        # 1) 主指标：名义GDP（交易总额法）
        sales_stats = self.collect_sales_statistics(month)
        household_sales_ex_tax = float(sum((s.get("household_revenue", 0.0) or 0.0) for s in (sales_stats or {}).values()) or 0.0)
        gov_sales_ex_tax = float(sum((s.get("government_procurement_revenue", 0.0) or 0.0) for s in (sales_stats or {}).values()) or 0.0)
        total_sales_ex_tax = household_sales_ex_tax + gov_sales_ex_tax

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
        
        # 4) 库存健康 - 由商品市场管理，这里返回0
        total_inventory_value = 0.0
        inventory_to_gdp_ratio = 0.0
        
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
        # ✅ 不依赖 self.households.employment_status（并行消费/轻量对象场景会缺失），改用交易与 labor market 存量推断
        employed_count = 0
        for tx in transactions:
            if tx.month == month and tx.type == "labor_payment":
                employed_count += 1  # 每笔 labor_payment 近似对应一个劳动力单元（head/spouse）

        labor_snapshot = self._get_labor_snapshot()
        total_labor_force_units = 0
        for entry in labor_snapshot.values():
            total_labor_force_units += int(entry.get("total", 0) or 0)

        unemployed_count = max(0, int(total_labor_force_units) - int(employed_count))
        employment_rate = (float(employed_count) / float(total_labor_force_units)) if total_labor_force_units > 0 else 0.0
        average_wage = (total_wages / employed_count) if employed_count > 0 else 0.0
        
        # 7) 价格水平 - 由商品市场管理，这里返回0
        average_price_level = 0.0
        
        return {
            "month": month,
            "nominal_gdp": nominal_gdp_transaction,  # 主指标：交易总额法
            "nominal_gdp_alternative": nominal_gdp_production,  # 对比指标：生产总值法
            "supply_demand_ratio": supply_demand_ratio,  # 供需匹配度（理想值~1.0）
            "gdp_components": {
                "household_consumption": household_sales_ex_tax + (household_sales_ex_tax * self.vat_rate),
                "government_procurement": gov_sales_ex_tax,  # 不含税（政府采购不缴VAT）
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

        - 产品税：按"总消费（家庭+政府采购，均为不含税金额）× VAT税率"估算；
                 同时也会尝试从 tx_history 的 consume_tax 取"实际VAT"，若存在则优先使用。
        - 生产总价值（output）：优先使用生产阶段直接统计的“产出总价值”（例如统一CD生产的 total_output_value / firm_production_value）；
          若缺失才回退用"投入成本/ (1-毛利率)"粗略估算（仅用于兼容旧统计）。
        - 中间消耗：基础生产投入成本（production_cost）。
        - 库存投资：output - sales（sales 为不含税销售额，含家庭+政府采购）。
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
        gov_sales_ex_tax = float(sum((s.get("government_procurement_revenue", 0.0) or 0.0) for s in (sales_stats or {}).values()) or 0.0)

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
                "government_sales_ex_tax": gov_sales_ex_tax,
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
    # ======================== 改进版 GDP 计算（交易流法 + 行业分解 + 实际GDP） ========================

    def calculate_gdp_comprehensive(
        self, 
        month: int, 
        production_stats: Optional[Dict[str, Any]] = None,
        base_month: int = 0,
        external_price_index_100: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        综合 GDP 计算：基于实际交易流 + 行业分解 + 实际/名义 GDP
        
        计算方法：
        1. 支出法 GDP = C + G + I + (X - M)
           - C (家庭消费): 家庭购买商品/服务的交易金额
           - G (政府支出): 政府采购 + 政府工资支出
           - I (投资): 存货投资 = 产出 - 销售
           - X - M (净出口): 封闭经济，为 0
        
        2. 生产法 GDP = Σ(各行业增加值) + 产品税
           - 增加值 = 产出 - 中间投入
        
        3. 收入法 GDP = 劳动者报酬 + 生产税净额 + 营业盈余
        
        4. 实际 GDP = 名义 GDP / 价格指数
        
        Args:
            month: 当前月份
            production_stats: 生产统计（可选）
            base_month: 价格指数基期（默认0）
        
        Returns:
            包含多种 GDP 口径及分解的完整统计
        """
        transactions = self.tx_by_month.get(month, [])
        if not transactions:
            transactions = [tx for tx in self.tx_history if tx.month == month]
        
        ps = production_stats
        if ps is None:
            ps = self.production_stats_by_month.get(month, {}) if hasattr(self, "production_stats_by_month") else {}
        ps = ps or {}
        
        # =====================================================================
        # 1️⃣ 从交易流统计各支出组件
        # =====================================================================
        
        # C: 家庭消费（含税）
        household_consumption = 0.0
        household_consumption_ex_tax = 0.0
        
        # G: 政府支出
        government_procurement = 0.0  # 政府采购商品
        government_wages = 0.0         # 政府工资支出
        
        # 税收
        vat_collected = 0.0
        labor_tax_collected = 0.0
        fica_tax_collected = 0.0
        corporate_tax_collected = 0.0
        
        # 工资总额
        total_wages = 0.0
        private_wages = 0.0
        
        # 行业分解
        industry_sales: Dict[str, float] = defaultdict(float)
        industry_production: Dict[str, float] = defaultdict(float)
        industry_intermediate: Dict[str, float] = defaultdict(float)
        
        for tx in transactions:
            tx_type = str(getattr(tx, "type", "") or "")
            amount = float(getattr(tx, "amount", 0.0) or 0.0)
            sender_id = str(getattr(tx, "sender_id", "") or "")
            receiver_id = str(getattr(tx, "receiver_id", "") or "")
            metadata = getattr(tx, "metadata", {}) or {}
            
            # 家庭消费
            if tx_type == "purchase" and sender_id in self.household_id:
                household_consumption += amount
                # 从 metadata 获取不含税金额
                ex_tax = float(metadata.get("amount_ex_tax", amount / (1 + self.vat_rate)) or 0.0)
                household_consumption_ex_tax += ex_tax
                # 按行业分解
                industry = str(metadata.get("industry", "Unknown") or "Unknown")
                industry_sales[industry] += ex_tax
            
            # 政府采购
            elif tx_type == "government_procurement":
                government_procurement += amount
                industry = str(metadata.get("industry", "Unknown") or "Unknown")
                industry_sales[industry] += amount
            
            # 工资支付
            elif tx_type == "labor_payment":
                total_wages += amount
                # 区分政府工资和私人工资
                if sender_id in self.government_id or sender_id.startswith("gov"):
                    government_wages += amount
                else:
                    private_wages += amount
            
            # 税收
            elif tx_type == "consume_tax":
                vat_collected += amount
            elif tx_type == "labor_tax":
                labor_tax_collected += amount
            elif tx_type == "fica_tax":
                fica_tax_collected += amount
            elif tx_type == "corporate_tax":
                corporate_tax_collected += amount
            
            # 资源购买：区分家庭服务消费（最终消费 C）和企业原材料采购（中间投入）
            elif tx_type == "resource_purchase":
                industry = str(metadata.get("industry_code", metadata.get("industry", "Unknown")) or "Unknown")
                if sender_id in self.household_id:
                    # 家庭购买服务（住房、医疗、交通、水电、保险等）→ 最终消费 C
                    household_consumption += amount
                    household_consumption_ex_tax += amount  # 服务消费不含 VAT
                    industry_sales[industry] += amount
                else:
                    # 企业购买原材料 → 中间投入
                    industry_intermediate[industry] += amount
        
        # =====================================================================
        # 2️⃣ 计算支出法 GDP: C + G + I
        # =====================================================================
        
        # 总消费 C（含税，反映实际支付）
        consumption_total = household_consumption
        
        # 政府支出 G = 政府最终消费支出
        # SNA 口径：G = 政府采购商品/服务 + 政府部门增加值（雇员报酬）
        # SNA 国民账户标准：G = 政府最终消费 = 政府采购 + 公务员薪酬
        # 政府工资是政府"生产"公共服务的成本（增加值），属于 G 的一部分。
        # 家庭用政府工资收入进行的消费计入 C，两者不重复：
        #   G 衡量的是政府提供公共服务的价值（用成本法计量）
        #   C 衡量的是家庭的私人消费支出
        government_expenditure = government_procurement + government_wages
        
        # 投资 I（存货投资）
        # 注意：本模型中企业按需生产（shortage-driven），大量商品从库存卖出，
        # 导致 output << sales。用 output - sales 会严重低估 GDP。
        # 正确做法：存货投资 = 期末库存价值 - 期初库存价值
        # 但模型中没有跟踪期初/期末库存价值，因此：
        # 方案：在封闭经济、无固定资本投资的模型中，
        #       GDP ≈ C + G（最终消费支出），I 仅作为参考指标。
        total_output = float(ps.get("total_output_value", 0.0) or 0.0)
        total_sales_ex_tax = household_consumption_ex_tax + government_procurement
        inventory_investment_memo = total_output - total_sales_ex_tax  # 仅供参考
        
        # 支出法 GDP = C + G（封闭经济，无固定资本投资，存货变动不可靠时）
        gdp_expenditure = consumption_total + government_expenditure
        
        # =====================================================================
        # 3️⃣ 计算生产法 GDP: Σ(增加值) + 产品税
        # =====================================================================
        
        # 从生产统计获取行业产出
        firm_production_value = ps.get("firm_production_value", {}) or {}
        firm_production_cost = ps.get("firm_production_cost", {}) or {}
        
        # 按行业汇总增加值
        industry_value_added: Dict[str, float] = defaultdict(float)
        total_value_added = 0.0
        total_intermediate = 0.0
        
        # 使用生产统计中的数据
        for firm_id, output_value in firm_production_value.items():
            output = float(output_value or 0.0)
            cost = float(firm_production_cost.get(firm_id, 0.0) or 0.0)
            value_added = output - cost
            
            # 获取企业行业（从注册信息）
            industry = self._get_firm_industry(firm_id)
            industry_value_added[industry] += value_added
            industry_production[industry] += output
            
            total_value_added += value_added
            total_intermediate += cost
        
        # 如果没有按企业的统计，使用汇总数据
        if total_value_added <= 0:
            total_output = float(ps.get("total_output_value", 0.0) or 0.0)
            total_intermediate = float(ps.get("total_production_cost", 0.0) or 0.0)
            total_value_added = total_output - total_intermediate
        
        # 生产法 GDP = 增加值 + 产品税（VAT）
        gdp_production = total_value_added + vat_collected
        
        # =====================================================================
        # 4️⃣ 计算收入法 GDP: 劳动者报酬 + 生产税 + 营业盈余
        # =====================================================================
        
        # 劳动者报酬 = 私人部门工资（净额）
        # 注意：labor_tax 和 fica_tax 是从工资中扣除的（非雇主额外缴纳），
        # 且包含政府雇员的税，无法区分。因此 compensation 只用私人部门净工资。
        # 政府工资不计入（避免与 C 双重计算）。
        compensation_of_employees = private_wages
        
        # 生产税净额 = VAT + 企业所得税（简化，不考虑补贴）
        taxes_on_production = vat_collected + corporate_tax_collected
        
        # 营业盈余 = 增加值 - 劳动者报酬
        operating_surplus = total_value_added - compensation_of_employees
        
        # 收入法 GDP
        gdp_income = compensation_of_employees + taxes_on_production + operating_surplus
        
        # =====================================================================
        # 5️⃣ 计算价格指数与实际 GDP
        # =====================================================================
        
        # 获取价格指数
        # 优先使用外部传入的价格指数（由 Simulator 从消费篮子计算，更准确）
        if external_price_index_100 is not None and external_price_index_100 > 0:
            price_index = float(external_price_index_100)
        else:
            # 回退到内部计算
            current_price_index = self._calculate_price_index(month)
            base_price_index = self._calculate_price_index(base_month) if base_month != month else 1.0
            if base_price_index > 0:
                price_index = (current_price_index / base_price_index) * 100
            else:
                price_index = 100.0
        
        # 实际 GDP（去除价格因素）
        deflator = price_index / 100.0 if price_index > 0 else 1.0
        real_gdp = gdp_expenditure / deflator if deflator > 0 else gdp_expenditure
        
        # =====================================================================
        # 6️⃣ 计算增长率（与上月比较）
        # =====================================================================
        
        prev_gdp = self._get_previous_month_gdp(month - 1)
        gdp_growth_rate = None
        real_gdp_growth_rate = None
        if prev_gdp and prev_gdp.get("nominal_gdp", 0) > 0:
            gdp_growth_rate = (gdp_expenditure / prev_gdp["nominal_gdp"]) - 1.0
        if prev_gdp and prev_gdp.get("real_gdp", 0) > 0:
            real_gdp_growth_rate = (real_gdp / prev_gdp["real_gdp"]) - 1.0
        
        # =====================================================================
        # 7️⃣ 计算关键比率
        # =====================================================================
        
        # 消费率
        consumption_rate = consumption_total / gdp_expenditure if gdp_expenditure > 0 else 0.0
        
        # 投资率（参考值，不计入 GDP）
        investment_rate = inventory_investment_memo / gdp_expenditure if gdp_expenditure > 0 else 0.0
        
        # 政府支出占比
        government_rate = government_expenditure / gdp_expenditure if gdp_expenditure > 0 else 0.0
        
        # 劳动报酬占比（收入分配）— 分母用支出法 GDP（主指标）
        labor_share = min(1.0, compensation_of_employees / gdp_expenditure) if gdp_expenditure > 0 else 0.0
        
        # 资本回报率（营业盈余/资本存量）
        total_capital = sum(self.firm_capital_stock.values()) if hasattr(self, "firm_capital_stock") else 0.0
        capital_return_rate = operating_surplus / total_capital if total_capital > 0 else 0.0
        
        # =====================================================================
        # 8️⃣ 构建返回结果
        # =====================================================================
        
        return {
            "month": month,
            
            # 核心 GDP 指标
            "nominal_gdp": gdp_expenditure,  # 名义 GDP（支出法，主指标）
            "real_gdp": real_gdp,            # 实际 GDP（去除价格因素）
            "price_index": price_index,       # 价格指数（基期=100）
            "deflator": deflator,             # GDP 平减指数
            
            # 三种核算方法
            "gdp_by_method": {
                "expenditure": gdp_expenditure,   # 支出法
                "production": gdp_production,     # 生产法
                "income": gdp_income,             # 收入法
                "discrepancy": {
                    "exp_vs_prod": gdp_expenditure - gdp_production,
                    "exp_vs_income": gdp_expenditure - gdp_income,
                    "prod_vs_income": gdp_production - gdp_income,
                },
            },
            
            # 支出法分解: GDP = C + G + I
            "expenditure_components": {
                "consumption": {
                    "total": consumption_total,
                    "household_consumption_with_tax": household_consumption,
                    "household_consumption_ex_tax": household_consumption_ex_tax,
                    "vat_paid_by_household": household_consumption - household_consumption_ex_tax,
                },
                "government": {
                    "total": government_expenditure,
                    "procurement": government_procurement,
                    "wages_memo": government_wages,  # 备忘：政府工资（不计入 G，已通过 C 回流）
                },
                "investment": {
                    "inventory_investment_memo": inventory_investment_memo,  # 参考值（output - sales，不计入 GDP）
                    "total_output": total_output,
                    "total_sales_ex_tax": total_sales_ex_tax,
                },
                "net_exports": 0.0,  # 封闭经济
            },
            
            # 生产法分解: GDP = Σ(VA) + 税
            "production_components": {
                "total_output": total_output,
                "intermediate_consumption": total_intermediate,
                "gross_value_added": total_value_added,
                "taxes_on_products": vat_collected,
                "by_industry": {
                    industry: {
                        "output": industry_production.get(industry, 0.0),
                        "intermediate": industry_intermediate.get(industry, 0.0),
                        "value_added": va,
                        "share": va / total_value_added if total_value_added > 0 else 0.0,
                    }
                    for industry, va in industry_value_added.items()
                },
            },
            
            # 收入法分解: GDP = W + T + π
            "income_components": {
                "compensation_of_employees": {
                    "total": compensation_of_employees,
                    "wages_net": total_wages,
                    "labor_tax": labor_tax_collected,
                    "fica_tax": fica_tax_collected,
                    "private_wages": private_wages,
                    "government_wages": government_wages,
                },
                "taxes_on_production": {
                    "total": taxes_on_production,
                    "vat": vat_collected,
                    "corporate_tax": corporate_tax_collected,
                },
                "operating_surplus": operating_surplus,
            },
            
            # 税收统计
            "tax_revenue": {
                "total": vat_collected + labor_tax_collected + fica_tax_collected + corporate_tax_collected,
                "vat": vat_collected,
                "labor_tax": labor_tax_collected,
                "fica_tax": fica_tax_collected,
                "corporate_tax": corporate_tax_collected,
            },
            
            # 增长率
            "growth_rates": {
                "nominal_gdp_growth": gdp_growth_rate,
                "real_gdp_growth": real_gdp_growth_rate,
            },
            
            # 关键比率
            "ratios": {
                "consumption_rate": consumption_rate,
                "investment_rate": investment_rate,
                "government_rate": government_rate,
                "labor_share": labor_share,
                "capital_return_rate": capital_return_rate,
            },
            
            # 行业销售分布
            "industry_sales": dict(industry_sales),
        }
    
    def _get_firm_industry(self, firm_id: str) -> str:
        """获取企业所属行业"""
        # 尝试从企业实例获取
        if hasattr(self, "firm") and self.firm:
            for firm in self.firm:
                fid = getattr(firm, "firm_id", None) or getattr(firm, "id", None)
                if str(fid) == str(firm_id):
                    return str(getattr(firm, "industry", "Unknown") or "Unknown")
        # 从 firm_monthly_data 的元数据获取（如果有存储）
        return "Unknown"
    
    def _calculate_price_index(self, month: int) -> float:
        """
        计算价格指数（Laspeyres 拉氏指数）
        
        公式: P = Σ(p_t × q_0) / Σ(p_0 × q_0) × 100
        其中:
          - p_t: 当期价格
          - p_0: 基期价格  
          - q_0: 基期数量（固定权重）
        
        返回基于 100 的价格指数（基期=100）
        """
        # 尝试使用产品价格注册表计算
        if not hasattr(self, "market_price_registry") or not self.market_price_registry:
            return 100.0  # 默认返回基期值 100
        
        # 获取本月交易数据，用于计算当期价格
        transactions = self.tx_by_month.get(month, [])
        if not transactions:
            transactions = [tx for tx in self.tx_history if tx.month == month]
        
        # 按 SKU 统计当期平均价格
        sku_current_price: Dict[str, List[float]] = defaultdict(list)
        sku_quantity: Dict[str, float] = defaultdict(float)
        
        for tx in transactions:
            if str(getattr(tx, "type", "") or "") in ("purchase", "government_procurement"):
                metadata = getattr(tx, "metadata", {}) or {}
                sku_id = str(metadata.get("sku_id", metadata.get("product_id", "")) or "")
                quantity = float(metadata.get("quantity", 0.0) or 0.0)
                amount = float(getattr(tx, "amount", 0.0) or 0.0)
                if sku_id and quantity > 0:
                    unit_price = amount / quantity
                    sku_current_price[sku_id].append(unit_price)
                    sku_quantity[sku_id] += quantity
        
        if not sku_current_price:
            return 100.0  # 无交易数据，返回基期值
        
        # 计算拉氏指数
        # 使用基期数量作为权重，比较当期价格与基期价格
        base_value = 0.0      # Σ(p_0 × q_0)
        current_value = 0.0   # Σ(p_t × q_0)
        
        for sku_id, prices in sku_current_price.items():
            # 获取基期价格（从注册表）
            base_price = 0.0
            if sku_id in self.market_price_registry:
                entry = self.market_price_registry[sku_id]
                base_price = float(entry.get("base_retail_price", entry.get("retail_price", 0)) or 0)
            
            if base_price <= 0:
                # 无基期价格，使用当期价格作为基期
                base_price = sum(prices) / len(prices) if prices else 0
            
            # 当期平均价格
            current_price = sum(prices) / len(prices) if prices else base_price
            
            # 使用当期交易数量作为权重（简化版，实际应用固定篮子）
            weight_qty = sku_quantity[sku_id]
            
            base_value += base_price * weight_qty
            current_value += current_price * weight_qty
        
        # 计算指数（基期=100）
        if base_value > 0:
            return (current_value / base_value) * 100.0
        return 100.0
    
    def _get_previous_month_gdp(self, month: int) -> Optional[Dict[str, float]]:
        """获取上月 GDP 数据（用于计算增长率）"""
        if month < 0:
            return None
        
        # 尝试从缓存获取
        if hasattr(self, "_gdp_cache") and month in self._gdp_cache:
            return self._gdp_cache[month]
        
        # 简单计算上月数据
        try:
            prev_stats = self.calculate_gdp_comprehensive(month)
            return {
                "nominal_gdp": prev_stats.get("nominal_gdp", 0.0),
                "real_gdp": prev_stats.get("real_gdp", 0.0),
            }
        except Exception:
            return None
    
    def cache_gdp_result(self, month: int, result: Dict[str, Any]) -> None:
        """缓存 GDP 计算结果（用于增长率计算）"""
        if not hasattr(self, "_gdp_cache"):
            self._gdp_cache: Dict[int, Dict[str, float]] = {}
        self._gdp_cache[month] = {
            "nominal_gdp": result.get("nominal_gdp", 0.0),
            "real_gdp": result.get("real_gdp", 0.0),
        }

    # =========================================================================
    # Checkpoint Support (用于断点续跑)
    # =========================================================================
    def get_all_balances(self) -> Dict[str, float]:
        """
        获取所有代理的账户余额（用于 checkpoint）
        
        Returns:
            Dict[agent_id, balance]
        """
        return {
            agent_id: float(ledger.amount)
            for agent_id, ledger in self.ledger.items()
        }
    
    def get_all_household_ids(self) -> List[str]:
        """
        获取所有家庭 ID 列表
        
        Returns:
            List[household_id]
        """
        return list(self.household_id)
    
    def restore_balances(self, ledger_data: Dict[str, float]) -> int:
        """
        从 checkpoint 恢复所有代理的账户余额

        Args:
            ledger_data: Dict[agent_id, balance]

        Returns:
            恢复的账户数量
        """
        restored = 0
        for agent_id, balance in ledger_data.items():
            if agent_id not in self.ledger:
                self.ledger[agent_id] = Ledger.create(agent_id, float(balance))
            else:
                self.ledger[agent_id].amount = float(balance)
            restored += 1
        self.logger.info(f"Restored {restored} account balances from checkpoint")
        return restored

    def get_firm_monthly_data_snapshot(self) -> Dict[str, Dict[int, Dict[str, float]]]:
        """
        获取企业月度数据快照（用于 checkpoint）

        Returns:
            Dict[firm_id, Dict[month, Dict[field, value]]]
        """
        snapshot = {}
        for firm_id, months_data in self.firm_monthly_data.items():
            snapshot[firm_id] = {}
            for month, data in months_data.items():
                snapshot[firm_id][month] = dict(data)
        return snapshot

    def restore_firm_monthly_data(self, data: Dict[str, Dict[str, Dict[str, float]]]) -> int:
        """
        从 checkpoint 恢复企业月度数据

        Args:
            data: Dict[firm_id, Dict[month_str, Dict[field, value]]]
                  注意：JSON 序列化后 month 会变成字符串

        Returns:
            恢复的企业数量
        """
        restored = 0
        for firm_id, months_data in data.items():
            for month_str, month_data in months_data.items():
                month = int(month_str)
                for field, value in month_data.items():
                    self.firm_monthly_data[firm_id][month][field] = float(value or 0.0)
            restored += 1
        self.logger.info(f"Restored firm monthly data for {restored} firms from checkpoint")
        return restored