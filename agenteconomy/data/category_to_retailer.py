"""
产品分类到零售商的映射

根据产品的顶级分类（Category字段第一段）将SKU分配给合适的零售商

零售商分类：
- 441: Motor vehicle and parts dealers (汽车及配件)
- 445: Food and beverage stores (食品饮料)
- 452: General merchandise stores (综合百货 - Walmart/Costco)
- 4A0: Other retail (其他零售)
- 722: Food services and drinking places (餐饮服务)
"""

# 顶级分类 -> 零售商代码
CATEGORY_TO_RETAILER = {
    # 食品相关 -> 445 Food and beverage stores
    "Food": "445",
    "Pets": "445",  # 宠物食品也在食品店
    
    # 汽车相关 -> 441 Motor vehicle and parts dealers
    "Auto & Tires": "441",
    
    # 综合百货 -> 452 General merchandise stores (Walmart/Costco类)
    "Sports & Outdoors": "452",
    "Health": "452",
    "Household Essentials": "452",
    "Baby": "452",
    "Personal Care": "452",
    "Toys": "452",
    "Beauty": "452",
    "Home": "452",
    "Clothing": "452",
    "Premium Beauty": "452",
    "Patio & Garden": "452",
    "Home Improvement": "452",
    "Electronics": "452",
    "Office Supplies": "452",
    "Arts Crafts & Sewing": "452",
    "Arts, Crafts & Sewing": "452",
    "Jewelry": "452",
    "Video Games": "452",
    "Musical Instruments": "452",
    "Cell Phones": "452",
    
    # 其他零售 -> 4A0 Other retail
    "Industrial & Scientific": "4A0",
    "Books": "4A0",
    "Music": "4A0",
    "Collectibles": "4A0",
    "Seasonal": "4A0",
    "Party & Occasions": "4A0",
    "Shop by Brand": "4A0",
    "Shop by Movie": "4A0",
    "Shop by Video Game": "4A0",
    "Character Shop": "4A0",
    "Feature": "4A0",
    "Walmart for Business": "4A0",
}

# 默认零售商（未匹配时）
DEFAULT_RETAILER = "452"  # 综合百货最通用


def get_retailer_code(category: str) -> str:
    """
    根据产品分类获取零售商代码
    
    Args:
        category: 产品分类字符串（可能是多级分类，用 | 分隔）
        
    Returns:
        零售商代码
    """
    if not category or not isinstance(category, str):
        return DEFAULT_RETAILER
    
    # 提取顶级分类
    top_category = category.split("|")[0].strip()
    
    return CATEGORY_TO_RETAILER.get(top_category, DEFAULT_RETAILER)


def get_retailer_name(retailer_code: str) -> str:
    """获取零售商名称"""
    names = {
        "441": "Motor vehicle and parts dealers",
        "445": "Food and beverage stores",
        "452": "General merchandise stores",
        "4A0": "Other retail",
        "722": "Food services and drinking places",
    }
    return names.get(retailer_code, "Unknown")


if __name__ == "__main__":
    # 测试
    test_categories = [
        "Sports & Outdoors | Outdoor Sports | Hunting",
        "Food | Beverages | Coffee",
        "Auto & Tires | Automotive Tools",
        "Health | Medicine Cabinet",
        "Industrial & Scientific",
    ]
    
    for cat in test_categories:
        code = get_retailer_code(cat)
        print(f"{cat[:40]:40} -> {code} ({get_retailer_name(code)})")
