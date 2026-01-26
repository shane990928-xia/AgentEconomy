"""
初始化抽象资源市场的工具函数

基于IO表数据为每个第三类行业设置：
1. 基准价格（从成本结构推算）
2. 初始供给能力
3. 单位类型
"""

from typing import Dict
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from agenteconomy.utils.load_io_table import (
    get_cost_structure,
    get_industry_name_mapping
)
from agenteconomy.data.industry_cate_map import industry_cate_map
from agenteconomy.market.AbstractResourceMarket import (
    AbstractResourceMarket,
    ResourceUnit
)


def calculate_base_price_from_cost_structure(
    industry_code: str,
    normalization_factor: float = 1.0
) -> float:
    """
    从IO表的成本结构计算基准价格
    
    逻辑：
    1. 获取该行业的成本结构（人工、税费、利润等）
    2. 这些成本占比 × normalization_factor = 单位服务的基准价
    
    Args:
        industry_code: IO表行业代码
        normalization_factor: 归一化因子，用于将价值转换为"单位"
    
    Returns:
        基准价格（美元/单位）
    """
    try:
        cost_structure = get_cost_structure(industry_code)
        
        # 计算单位成本（考虑所有成本要素）
        # 这些系数加起来接近1.0
        total_cost_ratio = (
            cost_structure.get("compensation", 0) +  # 人工
            cost_structure.get("taxes", 0) +  # 税费
            cost_structure.get("gross_operating_surplus", 0) +  # 利润
            cost_structure.get("intermediate_inputs", 0)  # 中间投入
        )
        
        # 单位价格 = 总成本率 × 归一化因子
        base_price = total_cost_ratio * normalization_factor
        
        # 确保价格合理（不为0或过大）
        if base_price < 0.01:
            base_price = 0.10  # 最低$0.10
        elif base_price > 10.0:
            base_price = 1.0  # 最高$10
        
        return base_price
        
    except Exception as e:
        print(f"警告: 无法计算 {industry_code} 的成本结构: {e}")
        return 1.0  # 默认$1/单位


def initialize_abstract_resource_market(
    market: AbstractResourceMarket,
    total_economy_scale: float = 10_000_000  # 经济总规模$1000万
) -> AbstractResourceMarket:
    """
    初始化抽象资源市场
    
    为所有第三类行业（成本驱动型）创建抽象资源
    
    Args:
        market: AbstractResourceMarket实例
        total_economy_scale: 经济总规模，用于估算各行业供给能力
    
    Returns:
        初始化后的market
    """
    
    print("=== 初始化抽象资源市场 ===\n")
    
    # 获取行业名称映射
    industry_names = get_industry_name_mapping()
    
    # 遍历第三类行业
    category_3 = industry_cate_map["category_3_cost_drivers"]
    
    # 合并所有子组
    all_category_3_industries = {}
    for subgroup_name, subgroup_info in category_3["subgroups"].items():
        all_category_3_industries.update(subgroup_info["industries"])
    
    print(f"找到 {len(all_category_3_industries)} 个第三类行业\n")
    
    # 行业到归一化因子的映射
    # 
    # 关键设计决策：所有抽象资源的基准价格统一设为 1.0
    # 
    # 原因：
    # 1. 数学等价性：基准价格只是转换器，设为任何值最终成本相同
    #    例如：100美元需求 ÷ 1.0基准价 × 1.2当期价 = 120美元
    #         100美元需求 ÷ 0.5基准价 × 0.6当期价 = 120美元
    # 
    # 2. 代码简洁：物理需求量 = IO表金额（不需要除法）
    #    例如：IO表写0.01，就是需要0.01单位
    # 
    # 3. 价格即指数：当期价格直接表示通胀/通缩
    #    1.0 = 基准年  |  1.2 = 涨价20%  |  0.9 = 降价10%
    # 
    # "单位"的定义：1单位 = 基准年的1美元购买力
    # 
    normalization_factors = {
        # 所有抽象资源统一设为 1.0
        "default": 1.00
    }
    
    # 注：如果未来需要给特定资源设置不同基准价（例如为了调试），
    # 可以在这里覆盖个别行业，但默认建议保持1.0
    
    # 供给能力占总经济的比例（基于IO表的行业重要性）
    # 这里简化处理，实际应该基于IO表分析
    supply_ratios = {
        "22": 0.15,      # 电力 15%
        "484": 0.10,     # 运输 10%
        "521CI": 0.08,   # 金融 8%
        "513": 0.06,     # 通信 6%
        "default": 0.05  # 其他 5%
    }
    
    # 初始化每个资源
    for industry_code, industry_name in all_category_3_industries.items():
        # 获取归一化因子
        norm_factor = normalization_factors.get(
            industry_code,
            normalization_factors["default"]
        )
        
        # 计算基准价格
        base_price = calculate_base_price_from_cost_structure(
            industry_code,
            norm_factor
        )
        
        # 估算供给能力
        supply_ratio = supply_ratios.get(
            industry_code,
            supply_ratios["default"]
        )
        supply_capacity = total_economy_scale * supply_ratio / base_price
        
        # 初始化资源
        market.initialize_resource(
            industry_code=industry_code,
            name=industry_name,
            base_price=base_price,
            initial_supply_capacity=supply_capacity
        )
        
        print(f"  ✓ {industry_name} ({industry_code})")
        print(f"    基准价: ${base_price:.4f}/单位")
        print(f"    供给能力: {supply_capacity:,.0f} 单位/期")
        print()
    
    print(f"=== 初始化完成: {len(all_category_3_industries)} 个资源 ===\n")
    
    return market


def get_default_abstract_resource_market(
    economy_scale: float = 10_000_000
) -> AbstractResourceMarket:
    """
    快速获取一个已初始化的抽象资源市场
    
    Args:
        economy_scale: 经济规模（美元）
    
    Returns:
        已初始化的AbstractResourceMarket
    """
    market = AbstractResourceMarket()
    return initialize_abstract_resource_market(market, economy_scale)


if __name__ == "__main__":
    # 测试
    print("\n" + "=" * 60)
    print("测试抽象资源市场初始化")
    print("=" * 60 + "\n")
    
    # 创建市场
    market = get_default_abstract_resource_market(economy_scale=10_000_000)
    
    # 显示一些关键资源
    print("\n" + "=" * 60)
    print("关键资源示例")
    print("=" * 60 + "\n")
    
    key_resources = ["22", "484", "521CI", "513"]
    
    for code in key_resources:
        try:
            info = market.get_resource_info(code)
            print(f"{info.name} ({code}):")
            print(f"  单位: {info.unit.value}")
            print(f"  基准价: ${info.base_price:.4f}")
            print(f"  当期价: ${info.current_price:.4f}")
            print(f"  供给: {info.supply_capacity:,.0f} {info.unit.value}/期")
            print()
        except ValueError:
            print(f"  (未找到 {code})")
            print()
    
    # 测试物理需求计算
    print("=" * 60)
    print("物理需求计算测试")
    print("=" * 60 + "\n")
    
    production_value = 10000  # $10,000生产价值
    io_coeff = 0.0100  # 1%的电力系数
    
    qty, unit = market.calculate_physical_demand("22", production_value, io_coeff)
    base_price = market.get_base_price("22")
    current_price = market.get_current_price("22")
    
    print(f"生产价值: ${production_value:,.2f}")
    print(f"IO系数: {io_coeff:.4f}")
    print(f"电力基准价: ${base_price:.4f}/{unit}")
    print(f"电力当期价: ${current_price:.4f}/{unit}")
    print()
    print(f"物理需求: {qty:.2f} {unit}")
    print(f"基准成本: ${production_value * io_coeff:.2f}")
    print(f"实际成本: ${qty * current_price:.2f}")

