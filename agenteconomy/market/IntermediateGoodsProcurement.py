"""
制造业中间品采购策略

解决Category 1（制造业）之间的中间品交易问题
"""

from typing import Dict, List, Any, Tuple
import random
from dataclasses import dataclass
import ray

from agenteconomy.data.industry_cate_map import industry_cate_map


CATEGORY_1_CODES = frozenset(
    industry_cate_map["category_1_manufacturers"]["industries"].keys()
)

@dataclass
class PurchaseItem:
    """采购项"""
    sku_id: str
    quantity: int
    unit_price: float
    total_cost: float


class IntermediateGoodsProcurement:
    """中间品采购策略"""
    
    def __init__(self, product_market):
        self.product_market = product_market
        self.industry_avg_prices = {}  # 缓存行业平均价格
        # 检测是否为Ray Actor（检查类型名称）
        self.is_ray_actor = 'ActorHandle' in str(type(product_market))
    
    def _call_market_method(self, method_name: str, *args, **kwargs):
        """调用market方法，自动处理Ray Actor对象"""
        method = getattr(self.product_market, method_name)
        if self.is_ray_actor:
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
            
            # 检查库存
            available_quantity = sku.available_stock
            actual_quantity = min(quantity_to_buy, available_quantity)
            
            if actual_quantity > 0:
                # 执行采购
                unit_price = sku.manufacturer_price
                total_cost = unit_price * actual_quantity
                
                item = PurchaseItem(
                    sku_id=sku.product_id,
                    quantity=actual_quantity,
                    unit_price=unit_price,
                    total_cost=total_cost
                )
                purchased_items.append(item)
                
                # 更新已购买单位（使用等价单位）
                purchased_value = total_cost
                units_purchased += purchased_value / self.get_industry_average_price(supplier_industry)
                
                # 更新库存
                sku.available_stock -= actual_quantity
            
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
        
        return {
            "total_cost": sum(costs_by_industry.values()),
            "by_industry": costs_by_industry,
            "items": all_items
        }
    
    def _is_category_1(self, industry_code: str) -> bool:
        """检查是否为制造业"""
        return industry_code in CATEGORY_1_CODES


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
