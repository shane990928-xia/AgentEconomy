from uuid import uuid4
from typing import List, Optional, TYPE_CHECKING

from agenteconomy.data.industry_cate_map import industry_cate_map
from agenteconomy.agent.firm import ManufactureFirm, RetailFirm, ServiceFirm, Firm

if TYPE_CHECKING:
    from agenteconomy.center.Ecocenter import EconomicCenter
    from agenteconomy.center.LaborMarket import LaborMarket
    from agenteconomy.center.ProductMarket import ProductMarket
    from agenteconomy.market.AbstractResourceMarket import AbstractResourceMarket

def create_households():
    pass

def create_firms(
    economic_center: Optional['EconomicCenter'] = None,
    labor_market: Optional['LaborMarket'] = None,
    product_market: Optional['ProductMarket'] = None,
    abstract_resource_market: Optional['AbstractResourceMarket'] = None,
) -> List[Firm]:
    """
    根据 industry_cate_map 创建所有行业的企业
    
    Args:
        economic_center: 经济中心
        labor_market: 劳动力市场
        product_market: 商品市场
        abstract_resource_market: 抽象资源市场（制造商生产时需要）
    
    Returns:
        List[Firm]: 包含制造商、零售商、服务商的企业列表
    """
    firms = []
    
    # Category 1: 制造商
    cat1 = industry_cate_map.get("category_1_manufacturers", {})
    for industry_code, industry_name in cat1.get("industries", {}).items():
        firm = ManufactureFirm(
            firm_id=f"mfg_{industry_code}_{uuid4().hex[:8]}",
            name=industry_name,
            industry=industry_code,
            industry_type="category_1_manufacturers",
            economic_center=economic_center,
            labor_market=labor_market,
            product_market=product_market,
            abstract_resource_market=abstract_resource_market,
        )
        firms.append(firm)
    
    # Category 2: 零售商
    cat2 = industry_cate_map.get("category_2_retailers", {})
    for industry_code, industry_name in cat2.get("industries", {}).items():
        firm = RetailFirm(
            firm_id=f"ret_{industry_code}_{uuid4().hex[:8]}",
            name=industry_name,
            industry=industry_code,
            industry_type="category_2_retailers",
            economic_center=economic_center,
            labor_market=labor_market,
            product_market=product_market,
            abstract_resource_market=abstract_resource_market,
        )
        firms.append(firm)
    
    # Category 3: 服务商（有 subgroups）
    # 注意：government_sectors 不创建为独立企业，其费用由 Government Agent 收取
    cat3 = industry_cate_map.get("category_3_cost_drivers", {})
    GOVERNMENT_SUBGROUP = "government_sectors"  # 政府行业子组，跳过不创建企业
    
    for subgroup_name, subgroup_info in cat3.get("subgroups", {}).items():
        # 跳过政府行业，这些由 Government Agent 处理
        if subgroup_name == GOVERNMENT_SUBGROUP:
            continue
            
        for industry_code, industry_name in subgroup_info.get("industries", {}).items():
            firm_id = f"svc_{industry_code}_{uuid4().hex[:8]}"
            firm = ServiceFirm(
                firm_id=firm_id,
                name=industry_name,
                industry=industry_code,
                industry_type=f"category_3_{subgroup_name}",
                economic_center=economic_center,
                labor_market=labor_market,
                product_market=product_market,
                abstract_resource_market=abstract_resource_market,
            )
            firms.append(firm)
            
            # 注册到 AbstractResourceMarket，建立 industry_code → firm_id 映射
            # 这样当有人采购该行业资源时，资金会流向真实的 ServiceFirm
            if abstract_resource_market is not None:
                abstract_resource_market.register_firm(industry_code, firm_id)
    
    return firms


# 政府行业代码集合（用于其他模块判断）
GOVERNMENT_INDUSTRY_CODES = frozenset(
    industry_cate_map.get("category_3_cost_drivers", {})
    .get("subgroups", {})
    .get("government_sectors", {})
    .get("industries", {})
    .keys()
)


def is_government_industry(industry_code: str) -> bool:
    """
    判断行业代码是否为政府行业
    
    Args:
        industry_code: 行业代码
        
    Returns:
        True if government industry, False otherwise
    """
    return industry_code in GOVERNMENT_INDUSTRY_CODES


def create_firms_by_category(
    category: str,
    economic_center: Optional['EconomicCenter'] = None,
    labor_market: Optional['LaborMarket'] = None,
    product_market: Optional['ProductMarket'] = None,
    abstract_resource_market: Optional['AbstractResourceMarket'] = None,
) -> List[Firm]:
    """
    创建指定类别的企业
    
    Args:
        category: "manufacturers", "retailers", or "services"
        economic_center: 经济中心
        labor_market: 劳动力市场
        product_market: 商品市场
        abstract_resource_market: 抽象资源市场
    
    Returns:
        List[Firm]: 指定类别的企业列表
    """
    firms = []
    
    if category == "manufacturers":
        cat = industry_cate_map.get("category_1_manufacturers", {})
        for industry_code, industry_name in cat.get("industries", {}).items():
            firm = ManufactureFirm(
                firm_id=f"mfg_{industry_code}_{uuid4().hex[:8]}",
                name=industry_name,
                industry=industry_code,
                industry_type="category_1_manufacturers",
                economic_center=economic_center,
                labor_market=labor_market,
                product_market=product_market,
                abstract_resource_market=abstract_resource_market,
            )
            firms.append(firm)
            
    elif category == "retailers":
        cat = industry_cate_map.get("category_2_retailers", {})
        for industry_code, industry_name in cat.get("industries", {}).items():
            firm = RetailFirm(
                firm_id=f"ret_{industry_code}_{uuid4().hex[:8]}",
                name=industry_name,
                industry=industry_code,
                industry_type="category_2_retailers",
                economic_center=economic_center,
                labor_market=labor_market,
                product_market=product_market,
                abstract_resource_market=abstract_resource_market,
            )
            firms.append(firm)
            
    elif category == "services":
        cat = industry_cate_map.get("category_3_cost_drivers", {})
        for subgroup_name, subgroup_info in cat.get("subgroups", {}).items():
            for industry_code, industry_name in subgroup_info.get("industries", {}).items():
                firm = ServiceFirm(
                    firm_id=f"svc_{industry_code}_{uuid4().hex[:8]}",
                    name=industry_name,
                    industry=industry_code,
                    industry_type=f"category_3_{subgroup_name}",
                    economic_center=economic_center,
                    labor_market=labor_market,
                    product_market=product_market,
                    abstract_resource_market=abstract_resource_market,
                )
                firms.append(firm)
    
    return firms


if __name__ == "__main__":
    # 测试
    all_firms = create_firms()
    print(f"Total firms: {len(all_firms)}")
    
    # 按类型统计
    mfg = [f for f in all_firms if isinstance(f, ManufactureFirm)]
    ret = [f for f in all_firms if isinstance(f, RetailFirm)]
    svc = [f for f in all_firms if isinstance(f, ServiceFirm)]
    
    print(f"Manufacturers: {len(mfg)}")
    print(f"Retailers: {len(ret)}")
    print(f"Services: {len(svc)}")
    
    # 打印几个示例
    print("\nSample firms:")
    for f in all_firms[:3]:
        print(f"  {f.firm_id}: {f.name} ({f.industry_type})")
