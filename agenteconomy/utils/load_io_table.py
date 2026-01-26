"""
投入产出表(Input-Output Table)加载和处理工具

该表展示了各行业之间的直接总需求系数，用于：
1. 确定企业的上游供应商
2. 计算生产成本结构
3. 模拟供应链关系
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from functools import lru_cache


@lru_cache(maxsize=1)
def load_io_table() -> pd.DataFrame:
    """
    加载投入产出表
    
    Returns:
        DataFrame: 行业代码为索引和列的投入产出系数矩阵
    """
    data_path = Path(__file__).resolve().parents[1] / "data" / "Direct Total Requirements, After Redefinitions - Summary.csv"
    
    # 读取CSV，第一列是行业代码
    df = pd.read_csv(data_path, index_col=0)
    
    # 列名就是行业代码（已经在header里）
    # 第一行（index 0）是行业描述，跳过
    # 从第二行开始到V001之前是投入产出数据
    
    # 找到 V001 行的位置
    try:
        v001_idx = df.index.get_loc('V001')
    except KeyError:
        # 如果找不到V001，使用倒数第5行
        v001_idx = len(df) - 5
    
    # 提取投入产出数据（跳过第一行描述，到V001之前）
    io_df = df.iloc[1:v001_idx].copy()
    
    # 移除第一列（Commodities/Industries描述列）
    if 'Commodities/Industries' in io_df.columns:
        io_df = io_df.drop('Commodities/Industries', axis=1)
    
    # 转换为浮点数
    io_df = io_df.astype(float)
    
    return io_df


@lru_cache(maxsize=1)
def load_value_added_components() -> Dict[str, pd.Series]:
    """
    加载增值部分（V001-V003）
    
    Returns:
        Dict: 包含员工薪酬、税收、毛利的字典
    """
    data_path = Path(__file__).resolve().parents[1] / "data" / "Direct Total Requirements, After Redefinitions - Summary.csv"
    df = pd.read_csv(data_path, index_col=0)
    
    # 移除描述列
    if 'Commodities/Industries' in df.columns:
        df = df.drop('Commodities/Industries', axis=1)
    
    # 提取V001-V003行并转换为浮点数
    compensation = df.loc['V001'].astype(float)
    taxes = df.loc['V002'].astype(float)
    gross_surplus = df.loc['V003'].astype(float)
    
    return {
        'compensation': compensation,
        'taxes': taxes,
        'gross_operating_surplus': gross_surplus,
    }


def get_suppliers_for_industry(industry_code: str, threshold: float = 0.01) -> List[Dict[str, float]]:
    """
    获取某个行业的主要上游供应商
    
    Args:
        industry_code: 行业代码（如 "311FT"）
        threshold: 系数阈值，只返回系数大于此值的供应商
    
    Returns:
        List[Dict]: 供应商列表，格式为 [{"supplier": "111CA", "coefficient": 0.075}, ...]
    """
    io_df = load_io_table()
    
    if industry_code not in io_df.columns:
        return []
    
    # 获取该行业的列（即它需要从其他行业购买的投入）
    suppliers = io_df[industry_code]
    
    # 筛选大于阈值的供应商
    major_suppliers = suppliers[suppliers > threshold].sort_values(ascending=False)
    
    return [
        {"supplier": supplier_code, "coefficient": float(coef)}
        for supplier_code, coef in major_suppliers.items()
    ]


def get_customers_for_industry(industry_code: str, threshold: float = 0.01) -> List[Dict[str, float]]:
    """
    获取某个行业的主要下游客户
    
    Args:
        industry_code: 行业代码（如 "111CA"）
        threshold: 系数阈值，只返回系数大于此值的客户
    
    Returns:
        List[Dict]: 客户列表，格式为 [{"customer": "311FT", "coefficient": 0.238}, ...]
    """
    io_df = load_io_table()
    
    if industry_code not in io_df.index:
        return []
    
    # 获取该行业的行（即它的产品被其他行业购买）
    customers = io_df.loc[industry_code]
    
    # 筛选大于阈值的客户
    major_customers = customers[customers > threshold].sort_values(ascending=False)
    
    return [
        {"customer": customer_code, "coefficient": float(coef)}
        for customer_code, coef in major_customers.items()
    ]


def get_cost_structure(industry_code: str) -> Optional[Dict[str, float]]:
    """
    获取某个行业的成本结构
    
    Args:
        industry_code: 行业代码
    
    Returns:
        Dict: 成本结构，包含中间投入、员工薪酬、税收、毛利等
    """
    io_df = load_io_table()
    value_added = load_value_added_components()
    
    if industry_code not in io_df.columns:
        return None
    
    # 中间投入（所有行业的总和）
    intermediate_inputs = float(io_df[industry_code].sum())
    
    return {
        "intermediate_inputs": intermediate_inputs,
        "compensation": float(value_added['compensation'][industry_code]),
        "taxes": float(value_added['taxes'][industry_code]),
        "gross_operating_surplus": float(value_added['gross_operating_surplus'][industry_code]),
        "total": intermediate_inputs + 
                 float(value_added['compensation'][industry_code]) + 
                 float(value_added['taxes'][industry_code]) + 
                 float(value_added['gross_operating_surplus'][industry_code])
    }


def get_industry_name_mapping() -> Dict[str, str]:
    """
    获取行业代码到名称的映射
    
    Returns:
        Dict: {行业代码: 行业名称}
    """
    data_path = Path(__file__).resolve().parents[1] / "data" / "Direct Total Requirements, After Redefinitions - Summary.csv"
    df = pd.read_csv(data_path)
    
    # 第一行（index 0）包含行业名称
    # 列从第3列开始（index 2）才是行业代码列
    industry_codes = df.columns[2:]  # 从第3列开始的列名就是行业代码
    industry_names = df.iloc[0, 2:].values  # 第一行从第3列开始的值是行业名称
    
    return dict(zip(industry_codes, industry_names))


if __name__ == "__main__":
    # 测试
    print("=== 投入产出表加载测试 ===\n")
    
    # 1. 加载表
    io_df = load_io_table()
    print(f"表格大小: {io_df.shape}")
    print(f"行业数量: {len(io_df.columns)}\n")
    
    # 2. 测试：食品制造业的供应商
    test_industry = "311FT"
    print(f"=== {test_industry} (食品制造) 的主要供应商 ===")
    suppliers = get_suppliers_for_industry(test_industry, threshold=0.01)
    for s in suppliers[:5]:
        print(f"  {s['supplier']}: {s['coefficient']:.4f}")
    
    # 3. 测试：农业的客户
    print(f"\n=== 111CA (农业) 的主要客户 ===")
    customers = get_customers_for_industry("111CA", threshold=0.01)
    for c in customers[:5]:
        print(f"  {c['customer']}: {c['coefficient']:.4f}")
    
    # 4. 测试：成本结构
    print(f"\n=== {test_industry} 的成本结构 ===")
    cost_structure = get_cost_structure(test_industry)
    for key, value in cost_structure.items():
        print(f"  {key}: {value:.4f}")
    
    # 5. 行业名称映射
    print("\n=== 行业代码示例 ===")
    name_mapping = get_industry_name_mapping()
    for code in list(name_mapping.keys())[:5]:
        print(f"  {code}: {name_mapping[code]}")

