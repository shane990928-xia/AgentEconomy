"""
制造业中间品采购策略

解决Category 1（制造业）之间的中间品交易问题
"""

from typing import Dict, List, Any, Tuple, Mapping, Optional, Sequence
import random
from dataclasses import asdict, dataclass

try:
    import ray
except ImportError:  # pragma: no cover - only used when callers pass Ray actors
    ray = None

from agenteconomy.data.industry_cate_map import industry_cate_map


CATEGORY_1_CODES = frozenset(
    industry_cate_map["category_1_manufacturers"]["industries"].keys()
)
_CATEGORY_1_NUMERIC_PREFIXES = frozenset(
    "".join(ch for ch in code if ch.isdigit())
    for code in CATEGORY_1_CODES
    if code and code[0].isdigit()
)

@dataclass
class PurchaseItem:
    """采购项"""
    sku_id: str
    quantity: int
    unit_price: float
    total_cost: float


@dataclass(frozen=True)
class IntermediateGoodsReservation:
    """Pure planning reservation; it does not mutate market stock."""

    supplier_industry: str
    sku_id: str
    unit_price: float
    available_quantity: float
    reserved_quantity: float
    reserved_cost: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IntermediateGoodsProcurementPlan:
    """Quote/scale result for intermediate goods procurement."""

    target_output: float
    planned_cost: float
    feasible_scale: float
    scaled_cost: float
    by_industry: Dict[str, Dict[str, Any]]
    reservations: List[IntermediateGoodsReservation]
    shortages: List[Dict[str, Any]]
    shortage: bool
    reason: str
    diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["reservations"] = [reservation.to_dict() for reservation in self.reservations]
        return data


class IntermediateGoodsProcurement:
    """中间品采购策略"""
    
    def __init__(self, product_market, receiver_id_resolver=None):
        self.product_market = product_market
        self.industry_avg_prices = {}  # 缓存行业平均价格
        # 检测是否为Ray Actor（检查类型名称）
        self.is_ray_actor = 'ActorHandle' in str(type(product_market))
        self.receiver_id_resolver = receiver_id_resolver
    
    def _call_market_method(self, method_name: str, *args, **kwargs):
        """调用market方法，自动处理Ray Actor对象"""
        method = getattr(self.product_market, method_name)
        if self.is_ray_actor:
            if ray is None:
                raise RuntimeError("ray is required when product_market is a Ray Actor")
            # Ray Actor，需要调用.remote()并ray.get()
            result = method.remote(*args, **kwargs)
            return ray.get(result)
        else:
            # 普通对象，直接调用
            return method(*args, **kwargs)
    
    def get_industry_average_price(self, industry_code: str) -> float:
        """
        获取行业平均价格（基准价格）
        
        这个价格用于将IO表的"价值需求"转换为"等价单位"
        """
        if industry_code not in self.industry_avg_prices:
            # 从market获取行业平均价格
            avg_price = self._call_market_method('get_industry_avg_price', industry_code, 'manufacturer')
            self.industry_avg_prices[industry_code] = avg_price
        
        return self.industry_avg_prices[industry_code]
    
    def calculate_procurement_target(
        self,
        supplier_industry: str,
        production_value: float,
        io_coefficient: float
    ) -> Tuple[float, float]:
        """
        计算采购目标
        
        Returns:
            (value_needed, equivalent_units)
        """
        # 1. 从IO表计算需要的价值
        value_needed = production_value * io_coefficient
        
        # 2. 转换为等价单位
        industry_avg_price = self.get_industry_average_price(supplier_industry)
        equivalent_units = value_needed / industry_avg_price
        
        return value_needed, equivalent_units

    def quote_intermediate_goods_plan(
        self,
        target_output: float,
        io_suppliers: Sequence[Mapping[str, Any]],
        supplier_quotes: Any,
        budget: Optional[float] = None,
    ) -> IntermediateGoodsProcurementPlan:
        """
        Build a pure quote/scale plan for intermediate goods.

        This method does not purchase goods, reserve ProductMarket stock, or write
        ledger entries. It only computes the global scale that can be supported by
        supplier quotes and an optional budget.
        """
        return self.plan_intermediate_goods_procurement(
            target_output=target_output,
            io_suppliers=io_suppliers,
            supplier_quotes=supplier_quotes,
            budget=budget,
        )

    def plan_intermediate_goods_procurement(
        self,
        target_output: float,
        io_suppliers: Sequence[Mapping[str, Any]],
        supplier_quotes: Any,
        budget: Optional[float] = None,
    ) -> IntermediateGoodsProcurementPlan:
        """
        Plan intermediate goods procurement without executing transactions.

        Args:
            target_output: Desired production output/value for this planning step.
            io_suppliers: IO rows with supplier/industry and coefficient fields.
            supplier_quotes: Quote data keyed by industry or a flat list of quote
                dicts/objects. Quotes may expose price fields
                (unit_price/manufacturer_price/price) and stock fields
                (available_quantity/available_stock/quantity/stock), or an
                explicit available_value.
            budget: Optional cash budget for all intermediate inputs.

        Returns:
            IntermediateGoodsProcurementPlan with planned_cost, feasible_scale,
            scaled_cost, shortages, and industry-level cost scaling.
        """
        target = self._coerce_nonnegative_float(target_output)
        normalized_suppliers = self._normalize_io_suppliers(io_suppliers)
        by_industry: Dict[str, Dict[str, Any]] = {}
        shortages: List[Dict[str, Any]] = []
        input_scales: List[float] = []

        for supplier in normalized_suppliers:
            industry = supplier["supplier"]
            coefficient = supplier["coefficient"]
            planned_cost = target * coefficient
            quotes = self._quotes_for_industry(supplier_quotes, industry)
            available_cost = self._available_cost_from_quotes(quotes)

            if planned_cost > 0.0:
                available_scale = min(1.0, available_cost / planned_cost)
            else:
                available_scale = 1.0

            input_scales.append(available_scale)
            shortage_reason = None
            if planned_cost > 0.0 and available_scale < 1.0:
                shortage_reason = "no_supplier_quotes" if not quotes else "input_shortage"
                shortages.append(
                    {
                        "type": "input",
                        "reason": shortage_reason,
                        "supplier": industry,
                        "planned_cost": planned_cost,
                        "available_cost": available_cost,
                        "available_scale": available_scale,
                    }
                )

            by_industry[industry] = {
                "planned_cost": planned_cost,
                "available_cost": available_cost,
                "available_scale": available_scale,
                "feasible_scale": None,
                "scaled_cost": None,
                "shortage": shortage_reason,
                "quote_count": len(quotes),
            }

        planned_cost = sum(row["planned_cost"] for row in by_industry.values())
        budget_value = None if budget is None else self._coerce_nonnegative_float(budget)

        if target <= 0.0:
            feasible_scale = 0.0
            reason = "zero_target_output"
            budget_scale = 1.0
        elif planned_cost <= 0.0:
            feasible_scale = 1.0
            reason = "no_planned_intermediate_inputs"
            budget_scale = 1.0
        else:
            input_scale = min(input_scales) if input_scales else 1.0
            budget_scale = 1.0
            if budget_value is not None:
                budget_scale = min(1.0, budget_value / planned_cost)
                if budget_scale < 1.0:
                    shortages.append(
                        {
                            "type": "budget",
                            "reason": "budget_shortage",
                            "planned_cost": planned_cost,
                            "available_budget": budget_value,
                            "budget_scale": budget_scale,
                        }
                    )

            feasible_scale = max(0.0, min(1.0, input_scale, budget_scale))
            reason_types = {
                shortage["reason"]
                for shortage in shortages
                if shortage.get("reason") in {"input_shortage", "no_supplier_quotes", "budget_shortage"}
            }
            if not reason_types:
                reason = "fully_feasible"
            elif len(reason_types) > 1:
                reason = "multiple_constraints"
            elif "budget_shortage" in reason_types:
                reason = "budget_shortage"
            elif "no_supplier_quotes" in reason_types:
                reason = "no_supplier_quotes"
            else:
                reason = "input_shortage"

        for row in by_industry.values():
            row["feasible_scale"] = feasible_scale
            row["scaled_cost"] = row["planned_cost"] * feasible_scale

        scaled_cost = planned_cost * feasible_scale
        reservations = self._build_scaled_reservations(
            supplier_quotes=supplier_quotes,
            by_industry=by_industry,
        )

        diagnostics = {
            "budget": budget_value,
            "budget_scale": budget_scale,
            "included_supplier_count": len(normalized_suppliers),
            "ignored_supplier_count": max(0, len(io_suppliers or []) - len(normalized_suppliers)),
        }

        return IntermediateGoodsProcurementPlan(
            target_output=target,
            planned_cost=planned_cost,
            feasible_scale=feasible_scale,
            scaled_cost=scaled_cost,
            by_industry=by_industry,
            reservations=reservations,
            shortages=shortages,
            shortage=bool(shortages),
            reason=reason,
            diagnostics=diagnostics,
        )

    def _normalize_io_suppliers(
        self,
        io_suppliers: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        normalized = []
        for supplier in io_suppliers or []:
            supplier_code = (
                supplier.get("supplier")
                or supplier.get("industry")
                or supplier.get("supplier_industry")
                or supplier.get("industry_code")
            )
            if not supplier_code:
                continue
            supplier_code = str(supplier_code)
            if not self._is_category_1(supplier_code):
                continue

            coefficient = self._coerce_nonnegative_float(supplier.get("coefficient", 0.0))
            normalized.append({"supplier": supplier_code, "coefficient": coefficient})
        return normalized

    @classmethod
    def _coerce_nonnegative_float(cls, value: Any) -> float:
        try:
            result = float(value or 0.0)
        except (TypeError, ValueError):
            result = 0.0
        return max(0.0, result)

    def _quotes_for_industry(self, supplier_quotes: Any, industry: str) -> List[Any]:
        if supplier_quotes is None:
            return []

        if isinstance(supplier_quotes, Mapping):
            if industry in supplier_quotes:
                return self._as_quote_list(supplier_quotes[industry])
            matches = []
            for quote in supplier_quotes.values():
                if self._quote_industry(quote) == industry:
                    matches.extend(self._as_quote_list(quote))
            return matches

        matches = []
        for quote in self._as_quote_list(supplier_quotes):
            if self._quote_industry(quote) == industry:
                matches.append(quote)
        return matches

    @classmethod
    def _as_quote_list(cls, raw_quotes: Any) -> List[Any]:
        if raw_quotes is None:
            return []
        if isinstance(raw_quotes, (str, bytes)):
            return []
        if isinstance(raw_quotes, Mapping):
            return [raw_quotes]
        try:
            return list(raw_quotes)
        except TypeError:
            return [raw_quotes]

    def _quote_industry(self, quote: Any) -> Optional[str]:
        value = self._read_quote_field(
            quote,
            "supplier",
            "supplier_industry",
            "industry",
            "industry_code",
            "manufacturer_code",
        )
        if value in (None, ""):
            return None
        return str(value)

    def _available_cost_from_quotes(self, quotes: Sequence[Any]) -> float:
        return sum(self._quote_available_cost(quote) for quote in quotes)

    def _quote_available_cost(self, quote: Any) -> float:
        explicit_value = self._read_quote_field(
            quote,
            "available_value",
            "available_cost",
            "available_budget",
        )
        if explicit_value is not None:
            return self._coerce_nonnegative_float(explicit_value)

        price = self._quote_unit_price(quote)
        quantity = self._quote_available_quantity(quote)
        if price <= 0.0 or quantity <= 0.0:
            return 0.0
        return price * quantity

    def _quote_unit_price(self, quote: Any) -> float:
        value = self._read_quote_field(
            quote,
            "unit_price",
            "manufacturer_price",
            "price",
            "current_price",
        )
        return self._coerce_nonnegative_float(value)

    def _quote_available_quantity(self, quote: Any) -> float:
        value = self._read_quote_field(
            quote,
            "available_quantity",
            "available_stock",
            "quantity",
            "stock",
            "amount",
        )
        return self._coerce_nonnegative_float(value)

    @classmethod
    def _quote_sku_id(cls, quote: Any) -> str:
        value = cls._read_quote_field(quote, "sku_id", "product_id", "id")
        return "" if value in (None, "") else str(value)

    @classmethod
    def _read_quote_field(cls, quote: Any, *field_names: str) -> Any:
        if isinstance(quote, Mapping):
            for field_name in field_names:
                if field_name in quote:
                    return quote[field_name]
            return None
        for field_name in field_names:
            if hasattr(quote, field_name):
                return getattr(quote, field_name)
        return None

    def _build_scaled_reservations(
        self,
        supplier_quotes: Any,
        by_industry: Mapping[str, Dict[str, Any]],
    ) -> List[IntermediateGoodsReservation]:
        reservations: List[IntermediateGoodsReservation] = []
        for industry, row in by_industry.items():
            remaining_cost = float(row.get("scaled_cost") or 0.0)
            if remaining_cost <= 0.0:
                continue

            for quote in self._quotes_for_industry(supplier_quotes, industry):
                price = self._quote_unit_price(quote)
                available_quantity = self._quote_available_quantity(quote)
                available_cost = self._quote_available_cost(quote)
                if price <= 0.0 or available_cost <= 0.0:
                    continue

                reserved_cost = min(remaining_cost, available_cost)
                reserved_quantity = reserved_cost / price
                if available_quantity > 0.0:
                    reserved_quantity = min(reserved_quantity, available_quantity)
                    reserved_cost = reserved_quantity * price

                if reserved_cost <= 0.0:
                    continue

                reservations.append(
                    IntermediateGoodsReservation(
                        supplier_industry=industry,
                        sku_id=self._quote_sku_id(quote),
                        unit_price=price,
                        available_quantity=available_quantity,
                        reserved_quantity=reserved_quantity,
                        reserved_cost=reserved_cost,
                    )
                )
                remaining_cost -= reserved_cost
                if remaining_cost <= 1e-9:
                    break

        return reservations

    def purchase_by_equivalent_units(
        self,
        supplier_industry: str,
        equivalent_units: float,
        buyer_id: str,
        period: int,
        strategy: str = "random"
    ) -> List[PurchaseItem]:
        """
        按等价单位采购
        
        Args:
            supplier_industry: 供应商行业代码
            equivalent_units: 等价单位数（基于行业平均价格）
            buyer_id: 购买者ID
            period: 当前期数
            strategy: 采购策略
                - "random": 随机选择SKU
                - "cheapest": 优先选择便宜的
                - "balanced": 平衡选择
        
        Returns:
            采购项列表
        """
        # 获取可用SKU
        available_skus = self._call_market_method(
            'get_available_skus',
            industry=supplier_industry,
            period=period
        )
        
        if not available_skus:
            return []
        
        # 根据策略排序
        if strategy == "cheapest":
            available_skus.sort(key=lambda sku: sku.manufacturer_price)
        elif strategy == "random":
            random.shuffle(available_skus)
        
        # 采购逻辑
        purchased_items = []
        units_purchased = 0
        target_units = equivalent_units
        
        while units_purchased < target_units and available_skus:
            sku = available_skus[0]
            
            # 计算需要购买多少个
            # 将"等价单位"转换为"实际个数"
            units_remaining = target_units - units_purchased
            sku_equivalent_value = sku.base_manufacturer_price / self.get_industry_average_price(supplier_industry)
            quantity_to_buy = max(1, int(units_remaining / sku_equivalent_value))
            
            # 检查并扣减制造商库存；ProductMarket 负责按可用量截断，避免负库存
            product_id = getattr(sku, "product_id", None)
            stock_result = {}
            if product_id:
                try:
                    stock_result = self._call_market_method(
                        'purchase_manufacturer_stock',
                        product_id,
                        quantity_to_buy,
                    ) or {}
                except AttributeError:
                    stock_result = {}
            if stock_result:
                actual_quantity = stock_result.get("actual_quantity", 0.0)
            else:
                available_quantity = max(0.0, float(getattr(sku, "available_stock", 0.0) or 0.0))
                actual_quantity = min(quantity_to_buy, available_quantity)
                if actual_quantity > 0:
                    sku.available_stock = max(0.0, available_quantity - actual_quantity)
            
            if actual_quantity > 0:
                # 执行采购
                unit_price = sku.manufacturer_price
                total_cost = unit_price * actual_quantity
                receiver_id = None
                if self.receiver_id_resolver is not None:
                    try:
                        receiver_id = self.receiver_id_resolver(
                            getattr(sku, "product_id", None) or "",
                            supplier_industry,
                            sku,
                        )
                    except Exception:
                        receiver_id = None
                
                item = PurchaseItem(
                    sku_id=sku.product_id,
                    quantity=actual_quantity,
                    unit_price=unit_price,
                    total_cost=total_cost
                )
                if receiver_id:
                    setattr(item, "receiver_id", receiver_id)
                setattr(item, "supplier_industry", supplier_industry)
                purchased_items.append(item)
                
                # 更新已购买单位（使用等价单位）
                purchased_value = total_cost
                units_purchased += purchased_value / self.get_industry_average_price(supplier_industry)
                
                if stock_result and hasattr(sku, "available_stock"):
                    try:
                        sku.available_stock = max(
                            0.0,
                            float(stock_result.get("available_after", 0.0) or 0.0),
                        )
                    except Exception:
                        pass
            
            # 移除已尝试的SKU
            available_skus.pop(0)
        
        return purchased_items
    
    def procure_intermediate_goods(
        self,
        manufacturer_id: str,
        production_value: float,
        io_suppliers: List[Dict[str, Any]],
        period: int
    ) -> Dict[str, Any]:
        """
        完整的中间品采购流程
        
        Args:
            manufacturer_id: 制造商ID
            production_value: 生产价值
            io_suppliers: IO表供应商列表
            period: 当前期数
        
        Returns:
            采购结果 {
                "total_cost": 总成本,
                "by_industry": {行业: 成本},
                "items": 采购项列表
            }
        """
        all_items = []
        costs_by_industry = {}
        target_value_by_industry = {}
        fulfillment_ratio_by_industry = {}
        
        for supplier in io_suppliers:
            supplier_code = supplier['supplier']
            
            # 检查是否为制造业（Category 1）
            if not self._is_category_1(supplier_code):
                continue
            
            # 计算采购目标
            value_needed, equivalent_units = self.calculate_procurement_target(
                supplier_industry=supplier_code,
                production_value=production_value,
                io_coefficient=supplier['coefficient']
            )
            target_value_by_industry[supplier_code] = float(value_needed or 0.0)
            
            # 采购
            items = self.purchase_by_equivalent_units(
                supplier_industry=supplier_code,
                equivalent_units=equivalent_units,
                buyer_id=manufacturer_id,
                period=period,
                strategy="random"  # 可配置
            )
            
            # 汇总
            industry_cost = sum(item.total_cost for item in items)
            costs_by_industry[supplier_code] = industry_cost
            all_items.extend(items)
            if value_needed and value_needed > 0:
                fulfillment_ratio_by_industry[supplier_code] = float(industry_cost) / float(value_needed)
            else:
                fulfillment_ratio_by_industry[supplier_code] = 1.0

        if fulfillment_ratio_by_industry:
            bottleneck_ratio = min(fulfillment_ratio_by_industry.values())
        else:
            bottleneck_ratio = 1.0
        
        return {
            "total_cost": sum(costs_by_industry.values()),
            "by_industry": costs_by_industry,
            "items": all_items,
            "target_value_by_industry": target_value_by_industry,
            "fulfillment_ratio_by_industry": fulfillment_ratio_by_industry,
            "bottleneck_ratio": bottleneck_ratio,
        }
    
    def _is_category_1(self, industry_code: str) -> bool:
        """检查是否为制造业"""
        if not industry_code:
            return False
        code = str(industry_code).strip()
        if code in CATEGORY_1_CODES:
            return True
        if code and code[0].isdigit():
            i = 0
            for ch in code:
                if ch.isdigit():
                    i += 1
                else:
                    break
            if i:
                return code[:i] in _CATEGORY_1_NUMERIC_PREFIXES
        return False


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """使用示例"""
    import ray
    from agenteconomy.center.ProductMarket import ProductMarket
    
    # 初始化Ray和ProductMarket
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    product_market = ProductMarket.remote()
    ray.get(product_market.initialize_products.remote())
    
    # 初始化
    procurement = IntermediateGoodsProcurement(product_market)
    
    # 场景：服装厂生产$10,000的衣服
    manufacturer_id = "mfg_315AL_001"
    production_value = 10000
    
    # IO表供应商
    io_suppliers = [
        {"supplier": "313TT", "coefficient": 0.0912},  # 纺织品
        {"supplier": "325", "coefficient": 0.0050},     # 化学品
        # ... 其他制造业供应商
    ]
    
    # 执行采购
    result = procurement.procure_intermediate_goods(
        manufacturer_id=manufacturer_id,
        production_value=production_value,
        io_suppliers=io_suppliers,
        period=1
    )
    
    print(f"总成本: ${result['total_cost']:.2f}")
    print(f"\n按行业:")
    for industry, cost in result['by_industry'].items():
        print(f"  {industry}: ${cost:.2f}")
    
    print(f"\n具体采购项 (共{len(result['items'])}项):")
    for item in result['items'][:5]:  # 显示前5项
        print(f"  SKU {item.sku_id}: {item.quantity}件 × ${item.unit_price:.2f} = ${item.total_cost:.2f}")


# ============================================================================
# 关键设计说明
# ============================================================================
"""
1. 等价单位（Equivalent Units）
   - 定义：基于行业平均价格的标准化度量
   - 公式：等价单位 = 价值需求 ÷ 行业平均价格
   - 作用：将抽象的"价值需求"转换为具体的"采购目标"

2. 采购策略
   - Random: 随机选择，模拟市场多样性
   - Cheapest: 最便宜优先，模拟成本优化
   - Balanced: 平衡选择，模拟稳定供应链

3. 价格处理
   - base_price: 用于计算等价单位（技术层面）
   - current_price: 用于实际交易（价值层面）

4. 为什么这样设计？
   - ✅ 保持IO表的抽象性（行业级别）
   - ✅ 允许SKU级别的交易（具体）
   - ✅ 简单易实现（不需要预定义配方）
   - ✅ 灵活（可以动态调整策略）
   - ✅ 可扩展（未来可加入具体配方）

5. 与抽象资源的区别
   - 抽象资源：基准价=1.0，钱=量
   - 中间品：基准价=行业均价，需要转换
"""
