"""
从零售价倒推制造业出厂价的方法

基于投入产出表和供应链加价结构
"""

from typing import Dict, Optional
from functools import lru_cache


class PriceCalculator:
    """价格计算器：在供应链各环节之间转换价格"""
    
    # 基于投入产出表的零售业成本结构
    RETAIL_MARKUP_RATES = {
        "441": {  # 汽车经销商
            "name": "Motor vehicle and parts dealers",
            "operating_cost_ratio": 0.3817,  # 中间投入（不含进货成本）
            "labor_ratio": 0.2847,
            "profit_margin": 0.1912,
            "goods_cost_ratio": 0.1424,  # 实际进货成本占比（估算）
        },
        "445": {  # 食品店
            "name": "Food and beverage stores",
            "operating_cost_ratio": 0.3310,
            "labor_ratio": 0.4154,
            "profit_margin": 0.1414,
            "goods_cost_ratio": 0.1122,
        },
        "452": {  # 百货店
            "name": "General merchandise stores",
            "operating_cost_ratio": 0.2677,
            "labor_ratio": 0.4317,
            "profit_margin": 0.1084,
            "goods_cost_ratio": 0.1922,
        },
        "4A0": {  # 其他零售
            "name": "Other retail",
            "operating_cost_ratio": 0.3666,
            "labor_ratio": 0.2595,
            "profit_margin": 0.2610,
            "goods_cost_ratio": 0.1129,
        },
    }
    
    # 批发加价率（基于投入产出表）
    WHOLESALE_MARKUP_RATE = 0.15  # 批发商一般加价15%
    
    @classmethod
    def retail_to_manufacturer(cls, retail_price: float, retail_industry: str = "452") -> Dict[str, float]:
        """
        从零售价倒推出厂价
        
        价格传导链：
        出厂价 → (+批发加价) → 批发价 → (+零售加价) → 零售价
        
        Args:
            retail_price: 零售价格
            retail_industry: 零售商类型（默认百货店）
        
        Returns:
            价格分解
        """
        if retail_industry not in cls.RETAIL_MARKUP_RATES:
            retail_industry = "452"  # 默认使用百货店
        
        markup_info = cls.RETAIL_MARKUP_RATES[retail_industry]
        
        # 方法1: 基于成本结构倒推
        # 零售价 = 进货成本 + 运营成本 + 人工 + 利润
        # 进货成本 = 零售价 × (1 - 运营成本率 - 人工率 - 利润率)
        
        operating_cost = retail_price * markup_info['operating_cost_ratio']
        labor_cost = retail_price * markup_info['labor_ratio']
        profit = retail_price * markup_info['profit_margin']
        
        # 倒推进货成本（含批发加价）
        wholesale_price = retail_price - operating_cost - labor_cost - profit
        
        # 去掉批发加价，得到出厂价
        manufacturer_price = wholesale_price / (1 + cls.WHOLESALE_MARKUP_RATE)
        
        return {
            "retail_price": retail_price,
            "retail_operating_cost": operating_cost,
            "retail_labor_cost": labor_cost,
            "retail_profit": profit,
            "wholesale_price": wholesale_price,
            "wholesale_markup": wholesale_price - manufacturer_price,
            "manufacturer_price": manufacturer_price,
            "total_markup": retail_price - manufacturer_price,
            "markup_rate": (retail_price - manufacturer_price) / manufacturer_price,
        }
    
    @classmethod
    def manufacturer_to_retail(cls, manufacturer_price: float, retail_industry: str = "452") -> Dict[str, float]:
        """
        从出厂价计算零售价
        
        Args:
            manufacturer_price: 出厂价
            retail_industry: 零售商类型
        
        Returns:
            价格分解
        """
        if retail_industry not in cls.RETAIL_MARKUP_RATES:
            retail_industry = "452"
        
        markup_info = cls.RETAIL_MARKUP_RATES[retail_industry]
        
        # 加批发加价
        wholesale_price = manufacturer_price * (1 + cls.WHOLESALE_MARKUP_RATE)
        
        # 计算零售价
        # 进货成本 = wholesale_price
        # 零售价 = 进货成本 / (1 - 运营成本率 - 人工率 - 利润率)
        cost_ratio = (markup_info['operating_cost_ratio'] + 
                     markup_info['labor_ratio'] + 
                     markup_info['profit_margin'])
        
        retail_price = wholesale_price / (1 - cost_ratio)
        
        return {
            "manufacturer_price": manufacturer_price,
            "wholesale_price": wholesale_price,
            "retail_price": retail_price,
            "total_markup": retail_price - manufacturer_price,
            "markup_rate": (retail_price - manufacturer_price) / manufacturer_price,
        }
    
    @classmethod
    def estimate_manufacturer_price_simple(cls, retail_price: float, category: str = "general") -> float:
        """
        简化版本：快速估算出厂价
        
        使用经验规则：
        - 食品类：零售价的 50-60%
        - 服装类：零售价的 40-50%
        - 电子类：零售价的 60-70%
        - 一般商品：零售价的 50-60%
        
        Args:
            retail_price: 零售价
            category: 商品类别
        
        Returns:
            估算的出厂价
        """
        ratios = {
            "food": 0.55,       # 食品
            "apparel": 0.45,    # 服装
            "electronics": 0.65, # 电子
            "general": 0.55,    # 一般商品
            "auto": 0.75,       # 汽车（加价率低）
        }
        
        ratio = ratios.get(category, 0.55)
        return retail_price * ratio


def get_retailer_for_product_category(category: str) -> str:
    """
    根据商品类别确定对应的零售商类型
    
    Args:
        category: 商品类别（如 "Food", "Apparel"）
    
    Returns:
        零售商行业代码
    """
    mapping = {
        # 食品类
        "Food": "445",
        "Beverage": "445",
        "Grocery": "445",
        
        # 汽车类
        "Automotive": "441",
        "Auto Parts": "441",
        
        # 百货类（默认）
        "Apparel": "452",
        "Electronics": "452",
        "Home & Garden": "452",
        "Toys": "452",
        
        # 其他
        "Other": "4A0",
    }
    
    return mapping.get(category, "452")  # 默认百货店


# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("=== 从零售价倒推出厂价 ===\n")
    
    # 示例1: 一件T恤在百货店卖99元
    print("示例1: T恤零售价 99元")
    result = PriceCalculator.retail_to_manufacturer(99, "452")
    print(f"  零售价: {result['retail_price']:.2f}元")
    print(f"  零售商运营成本: {result['retail_operating_cost']:.2f}元")
    print(f"  零售商人工: {result['retail_labor_cost']:.2f}元")
    print(f"  零售商利润: {result['retail_profit']:.2f}元")
    print(f"  批发价: {result['wholesale_price']:.2f}元")
    print(f"  出厂价: {result['manufacturer_price']:.2f}元")
    print(f"  总加价率: {result['markup_rate']:.1%}\n")
    
    # 示例2: 一瓶牛奶在超市卖5元
    print("示例2: 牛奶零售价 5元")
    result = PriceCalculator.retail_to_manufacturer(5, "445")
    print(f"  零售价: {result['retail_price']:.2f}元")
    print(f"  出厂价: {result['manufacturer_price']:.2f}元")
    print(f"  加价率: {result['markup_rate']:.1%}\n")
    
    # 示例3: 快速估算
    print("示例3: 快速估算")
    retail_price = 199
    mfr_price = PriceCalculator.estimate_manufacturer_price_simple(retail_price, "electronics")
    print(f"  电子产品零售价 {retail_price}元")
    print(f"  估算出厂价: {mfr_price:.2f}元")
    print(f"  加价率: {(retail_price/mfr_price - 1):.1%}\n")
    
    # 示例4: 反向计算 - 从出厂价到零售价
    print("示例4: 从出厂价计算零售价")
    mfr_price = 50
    result = PriceCalculator.manufacturer_to_retail(mfr_price, "452")
    print(f"  出厂价: {result['manufacturer_price']:.2f}元")
    print(f"  批发价: {result['wholesale_price']:.2f}元")
    print(f"  零售价: {result['retail_price']:.2f}元")
    print(f"  加价率: {result['markup_rate']:.1%}")


