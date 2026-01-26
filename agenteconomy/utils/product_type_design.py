"""
经济模型中的产品类型和交易机制设计

三种产品类型的处理方案
"""

from enum import Enum
from typing import Dict, List, Optional, Union
from dataclasses import dataclass


class ProductType(Enum):
    """产品类型枚举"""
    ABSTRACT_RESOURCE = "abstract"  # 抽象资源（服务、原材料）
    CONCRETE_SKU = "concrete"       # 具体SKU（制造业产品）
    RETAIL_GOOD = "retail"          # 零售商品（从制造商进货）


@dataclass
class AbstractResource:
    """
    抽象资源（第三类：服务和原材料）
    
    特点：
    - 无具体SKU，按"单位"交易
    - 同质化，只有价格区别
    - 例：电力、运输、金融服务
    """
    industry_code: str          # 行业代码（如 "22" - Utilities）
    resource_name: str          # 资源名称（如 "Electricity"）
    unit_price: float           # 单位价格（元/单位）
    available_units: float      # 可用单位数
    base_price: float = 1.0     # 基准价格
    
    def adjust_price(self, demand_ratio: float):
        """根据供需比例调整价格"""
        # demand_ratio = 需求量 / 供给量
        self.unit_price = self.base_price * (1 + 0.5 * (demand_ratio - 1))


@dataclass
class ConcreteSKU:
    """
    具体SKU（第一类：制造业产品）
    
    特点：
    - 有具体的产品ID和属性
    - 可以被零售商转售
    - 可以被其他制造商作为原材料
    """
    sku_id: str                 # SKU编号
    product_name: str           # 产品名称
    manufacturer_id: str        # 制造商ID
    industry_code: str          # 所属行业
    unit_price: float           # 单价
    inventory: int              # 库存数量
    attributes: Dict            # 产品属性（质量、规格等）


# ==================== 解决方案设计 ====================

class ProcurementStrategy:
    """
    采购策略：处理不同类型产品的采购需求
    """
    
    @staticmethod
    def get_procurement_plan(firm_industry: str, production_value: float) -> Dict:
        """
        根据投入产出表生成采购计划
        
        Args:
            firm_industry: 企业所属行业
            production_value: 计划生产价值
        
        Returns:
            采购计划，区分抽象资源和具体SKU
        """
        from agenteconomy.utils.load_io_table import get_suppliers_for_industry
        from agenteconomy.data.industry_cate_map import industry_cate_map
        
        suppliers = get_suppliers_for_industry(firm_industry, threshold=0.001)
        
        procurement_plan = {
            "abstract_resources": [],   # 抽象资源（按单位购买）
            "concrete_skus": [],        # 具体SKU（按件购买）
        }
        
        for supplier in suppliers:
            supplier_code = supplier['supplier']
            purchase_value = production_value * supplier['coefficient']
            
            # 判断供应商属于哪一类
            supplier_category = _get_industry_category(supplier_code)
            
            if supplier_category == "category_3_cost_drivers":
                # 第三类：抽象资源
                procurement_plan["abstract_resources"].append({
                    "industry": supplier_code,
                    "resource_type": _get_resource_type(supplier_code),
                    "required_value": purchase_value,
                    "required_units": None,  # 根据单价计算
                })
            
            elif supplier_category in ["category_1_manufacturers", "category_2_retailers"]:
                # 第一类或第二类：具体SKU
                procurement_plan["concrete_skus"].append({
                    "industry": supplier_code,
                    "required_value": purchase_value,
                    "sku_criteria": _get_sku_criteria(firm_industry, supplier_code),
                })
        
        return procurement_plan


def _get_industry_category(industry_code: str) -> str:
    """判断行业属于哪个类别"""
    from agenteconomy.data.industry_cate_map import industry_cate_map
    
    # 检查 category_1
    if industry_code in industry_cate_map["category_1_manufacturers"]["industries"]:
        return "category_1_manufacturers"
    
    # 检查 category_2
    if industry_code in industry_cate_map["category_2_retailers"]["industries"]:
        return "category_2_retailers"
    
    # 检查 category_3（有subgroups）
    for subgroup in industry_cate_map["category_3_cost_drivers"]["subgroups"].values():
        if industry_code in subgroup["industries"]:
            return "category_3_cost_drivers"
    
    return "unknown"


def _get_resource_type(industry_code: str) -> str:
    """获取抽象资源类型"""
    resource_map = {
        "22": "Utilities",
        "42": "Wholesale",
        "484": "Trucking",
        "521CI": "FinancialServices",
        "532RL": "Rental",
        # ... 更多映射
    }
    return resource_map.get(industry_code, "GenericService")


def _get_sku_criteria(buyer_industry: str, supplier_industry: str) -> Dict:
    """
    根据买方和卖方行业，确定SKU采购标准
    
    例如：食品制造(311FT)从农业(111CA)采购时，需要"原材料"级别的农产品
    """
    return {
        "category": "raw_material" if supplier_industry in ["111CA", "113FF"] else "intermediate_good",
        "quality_requirement": "standard",
    }


# ==================== 具体场景示例 ====================

"""
场景1: 食品制造企业的采购

投入产出表显示食品制造(311FT)需要：
- 111CA (农业): 23.85%        → 具体SKU（小麦、玉米等）
- 311FT (食品自身): 16.86%    → 具体SKU（半成品食品）
- 42 (批发): 8.91%             → 抽象服务（批发服务费）
- 484 (卡车运输): 3.22%        → 抽象资源（运输服务）
- 22 (电力): 0.92%             → 抽象资源（电力）
- 325 (化学品): 8.16%          → 具体SKU（添加剂、包装材料）

处理方式：
1. 抽象资源（批发、运输、电力）：
   - 在对应市场请求指定单位数
   - 支付费用即可获得
   - 不需要匹配具体产品

2. 具体SKU（农产品、化学品）：
   - 在ProductMarket搜索符合条件的SKU
   - 可能需要考虑质量、价格、供应商信誉
   - 实际购买具体商品实例
"""

"""
场景2: 零售商的采购

投入产出表显示零售商(452 - General merchandise)需要：
- 42 (批发): 2.8%              → 抽象服务（批发服务）
- ORE (房地产): 11.5%          → 抽象资源（租金）
- 484 (运输): 2.15%            → 抽象资源（物流）

零售商的主要"采购"不在投入产出表中，而是：
- 从制造商采购待售SKU（这是库存采购，不是生产投入）
- 投入产出表中的成本是"运营成本"

处理方式：
1. 运营成本（投入产出表）：
   - 批发、租金、运输 → 抽象资源
   
2. 库存采购（额外逻辑）：
   - 根据零售商类别，从对应制造业采购SKU
   - 使用 retailer_mfg.json 中的供应链关系
"""

"""
场景3: 制造业之间的采购

投入产出表显示计算机制造(334)需要：
- 334 (计算机自身): 10.37%    → 具体SKU（芯片、电路板等半成品）
- 335 (电气设备): 2.35%        → 具体SKU（电气元件）
- 326 (塑料橡胶): 0.65%        → 具体SKU（外壳、线缆）

处理方式：
- 这些都是具体SKU
- 在ProductMarket中搜索符合行业和规格的产品
- 可能需要建立长期供应关系
"""


# ==================== 市场交易实现 ====================

class AbstractResourceMarket:
    """抽象资源市场（第三类服务）"""
    
    def __init__(self):
        self.resources: Dict[str, AbstractResource] = {}
    
    def purchase_resource(self, industry_code: str, required_units: float, buyer_id: str) -> float:
        """
        购买抽象资源
        
        Returns:
            total_cost: 总成本
        """
        resource = self.resources.get(industry_code)
        if not resource or resource.available_units < required_units:
            raise ValueError(f"Resource {industry_code} not available")
        
        total_cost = resource.unit_price * required_units
        resource.available_units -= required_units
        
        return total_cost


class ProductMarket:
    """具体SKU市场（第一类制造业）"""
    
    def __init__(self):
        self.skus: List[ConcreteSKU] = []
    
    def search_skus(self, industry_code: str, criteria: Dict, max_price: float) -> List[ConcreteSKU]:
        """
        搜索符合条件的SKU
        
        Args:
            industry_code: 生产行业
            criteria: 采购标准
            max_price: 最高价格
        """
        matched_skus = [
            sku for sku in self.skus
            if sku.industry_code == industry_code
            and sku.unit_price <= max_price
            and sku.inventory > 0
        ]
        return matched_skus
    
    def purchase_sku(self, sku_id: str, quantity: int, buyer_id: str) -> float:
        """
        购买具体SKU
        
        Returns:
            total_cost: 总成本
        """
        sku = next((s for s in self.skus if s.sku_id == sku_id), None)
        if not sku or sku.inventory < quantity:
            raise ValueError(f"SKU {sku_id} not available")
        
        total_cost = sku.unit_price * quantity
        sku.inventory -= quantity
        
        return total_cost


# ==================== 总结 ====================

"""
核心设计原则：

1. 两种市场机制并存：
   - AbstractResourceMarket: 处理第三类抽象资源（按单位）
   - ProductMarket: 处理第一类具体SKU（按件）

2. 投入产出表的使用：
   - 确定需要购买的"价值量"
   - 根据供应商类别选择市场
   - 抽象资源：价值 ÷ 单价 = 单位数
   - 具体SKU：价值 ÷ 平均价 ≈ 件数

3. 零售业的特殊处理：
   - 运营成本：从投入产出表获取（抽象资源为主）
   - 库存采购：从 retailer_mfg.json 获取（具体SKU）

4. 制造业之间的交易：
   - 全部是具体SKU
   - 需要在ProductMarket匹配和购买
   - 可以建立长期供应商关系优化搜索

5. 价格机制：
   - 抽象资源：供需动态调价
   - 具体SKU：制造商定价 + 市场竞争
"""


