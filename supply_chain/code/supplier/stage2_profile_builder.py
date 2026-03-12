#!/usr/bin/env python3
"""
Stage 2: 画像构建
================

功能：
- 构建公司自身画像(Self Profile)
- 构建供应商候选画像(Supplier Profile)
- 并行处理多个供应商画像

用法：
    from stage2_profile_builder import (
        build_self_profile,
        build_supplier_profile,
        parallel_prepare_supplier_profiles
    )
"""

import os
import pandas as pd
import numpy as np
import multiprocessing
import concurrent.futures
from typing import Dict, List, Optional, Any, Tuple, Set
from tqdm import tqdm

from stage0_config import DATA_BASE_PATH, _convert_for_json
from stage1_data_loader import (
    get_company_info_by_id, 
    get_industry_info, 
    get_segment_info,
    extract_financial_data,
    _get_all_related_fsym_ids,
    _get_fsym_ids_improved
)

# ============================================================================
# 全局变量 - 进程池数据共享
# ============================================================================

_process_dataframes: Optional[Dict[str, pd.DataFrame]] = None


def _initialize_supplier_process_pool(dataframes_dict: Dict[str, pd.DataFrame]):
    """初始化进程池，将数据放入全局变量"""
    global _process_dataframes
    _process_dataframes = dataframes_dict


# ============================================================================
# Self Profile 构建
# ============================================================================

def build_self_profile(
    target_entity_id: str, 
    decision_year: int, 
    all_dfs: Dict[str, pd.DataFrame],
    num_financial_reports: int = 1
) -> Dict[str, Any]:
    """
    构建目标公司的完整画像
    
    Args:
        target_entity_id: 目标公司的factset_entity_id
        decision_year: 决策年份
        all_dfs: 所有数据集
        num_financial_reports: 财务报告数量
    
    Returns:
        公司画像字典
    """
    cutoff_date = pd.Timestamp(f'{decision_year}-01-01')
    
    # 获取基本公司信息
    company_info = get_company_info_by_id(target_entity_id, all_dfs)
    if company_info is None:
        return {'error': f'Company not found: {target_entity_id}'}
    
    # 获取行业信息
    industry_info = get_industry_info(target_entity_id, all_dfs)
    
    # 获取业务段信息
    segment_info = get_segment_info(target_entity_id, all_dfs)
    
    # 获取财务数据
    financial_data = extract_financial_data(
        target_entity_id, cutoff_date, all_dfs, 
        num_reports=num_financial_reports
    )
    
    # 计算财务汇总指标
    financial_summary = _calculate_financial_summary(financial_data)
    
    # 获取历史供应商信息
    historical_suppliers = _get_historical_suppliers(target_entity_id, decision_year, all_dfs)
    
    # 构建画像
    profile = {
        'factset_entity_id': target_entity_id,
        'entity_name': company_info.get('entity_proper_name'),
        'company_id': company_info.get('company_id'),
        'industry': industry_info,
        'sic_code': company_info.get('primary_sic_code'),
        'country': company_info.get('iso_country'),
        'entity_type': company_info.get('entity_type'),
        'decision_year': decision_year,
        'cutoff_date': cutoff_date.strftime('%Y-%m-%d'),
        'segments': segment_info.get('segments') if segment_info else [],
        'financial_reports': financial_data,
        'financial_summary': financial_summary,
        'historical_supplier_count': len(historical_suppliers),
        'historical_supplier_ids': list(historical_suppliers)[:10],  # 只保留前10个
    }
    
    return profile


def _calculate_financial_summary(financial_data: List[Dict]) -> Dict:
    """计算财务汇总指标"""
    if not financial_data or 'financial_data_error' in financial_data[0]:
        return {'status': 'no_data'}
    
    try:
        # 收集各项指标
        sales = [d.get('ff_sales') for d in financial_data if d.get('ff_sales') is not None]
        margins = [d.get('ff_net_mgn') for d in financial_data if d.get('ff_net_mgn') is not None]
        debt_eq = [d.get('ff_debt_eq') for d in financial_data if d.get('ff_debt_eq') is not None]
        roic = [d.get('ff_roic') for d in financial_data if d.get('ff_roic') is not None]
        
        summary = {
            'avg_sales': np.mean(sales) if sales else None,
            'avg_margin': np.mean(margins) if margins else None,
            'avg_debt_equity': np.mean(debt_eq) if debt_eq else None,
            'avg_roic': np.mean(roic) if roic else None,
            'num_reports': len(financial_data),
        }
        
        # 转换为JSON可序列化格式
        return {k: _convert_for_json(v) for k, v in summary.items()}
        
    except Exception:
        return {'status': 'calculation_error'}


def _get_historical_suppliers(
    target_entity_id: str, 
    decision_year: int, 
    all_dfs: Dict[str, pd.DataFrame]
) -> Set[str]:
    """获取历史供应商ID集合"""
    historical_suppliers = set()
    
    df_rev = all_dfs.get('factset_revere_relationship')
    if df_rev is None or df_rev.empty:
        return historical_suppliers
    
    cutoff_date = pd.Timestamp(f'{decision_year}-01-01')
    
    # factset_revere_relationship 使用数值型 company_id（int64）。
    # company_factset.company_id 可能是字符串，用 pd.to_numeric 统一转换。
    df_cf = all_dfs.get('company_factset')
    numeric_id = None
    cf_numeric = None

    if df_cf is not None and 'fs_entity_id' in df_cf.columns and 'company_id' in df_cf.columns:
        cf_numeric = df_cf.copy()
        cf_numeric['_cid'] = pd.to_numeric(cf_numeric['company_id'], errors='coerce')
        row = cf_numeric[cf_numeric['fs_entity_id'] == target_entity_id]
        if not row.empty:
            numeric_id = row.iloc[0]['_cid']

    if numeric_id is None or pd.isna(numeric_id):
        return historical_suppliers

    # FactSet REVERE 语义: source=供应商(卖方), target=买方/客户
    # Apple 作为买方在 target 列，Apple 的供应商在 source 列
    supplier_relations = df_rev[
        (df_rev['rel_type'] == 'SUPPLIER') &
        (df_rev['target_company_id'] == numeric_id)
    ].copy()

    if supplier_relations.empty:
        return historical_suppliers

    # 确保start_列是datetime类型
    if 'start_' in supplier_relations.columns:
        supplier_relations['start_'] = pd.to_datetime(supplier_relations['start_'], errors='coerce')
        supplier_relations = supplier_relations[supplier_relations['start_'] < cutoff_date]

    # 供应商 ID 在 source_company_id 列，映射回 fs_entity_id
    if 'source_company_id' in supplier_relations.columns and cf_numeric is not None:
        supplier_numeric_ids = set(supplier_relations['source_company_id'].dropna().unique())
        mapped = cf_numeric[cf_numeric['_cid'].isin(supplier_numeric_ids)]['fs_entity_id']
        historical_suppliers = set(mapped.dropna().unique())
    
    return historical_suppliers


# ============================================================================
# Supplier Profile 构建
# ============================================================================

def build_supplier_profile(
    supplier_entity_id: str,
    self_profile: Dict,
    cutoff_date: pd.Timestamp,
    all_dfs: Dict[str, pd.DataFrame],
    num_financial_reports: int = 1
) -> Optional[Dict[str, Any]]:
    """
    构建供应商画像
    
    Args:
        supplier_entity_id: 供应商的factset_entity_id
        self_profile: 目标公司画像(用于相对比较)
        cutoff_date: 截止日期
        all_dfs: 所有数据集
        num_financial_reports: 财务报告数量
    
    Returns:
        供应商画像字典
    """
    # 获取基本公司信息
    company_info = get_company_info_by_id(supplier_entity_id, all_dfs)
    if company_info is None:
        return None
    
    # 获取行业信息
    industry_info = get_industry_info(supplier_entity_id, all_dfs)
    
    # 获取业务段信息
    segment_info = get_segment_info(supplier_entity_id, all_dfs)
    
    # 获取财务数据
    financial_data = extract_financial_data(
        supplier_entity_id, cutoff_date, all_dfs,
        num_reports=num_financial_reports
    )
    
    # 计算财务汇总指标
    financial_summary = _calculate_financial_summary(financial_data)
    
    # 计算与目标公司的相对指标
    relative_metrics = _calculate_relative_metrics(financial_summary, self_profile.get('financial_summary', {}))
    
    # 构建画像
    profile = {
        'factset_entity_id': supplier_entity_id,
        'entity_name': company_info.get('entity_proper_name'),
        'company_id': company_info.get('company_id'),
        'industry': industry_info,
        'sic_code': company_info.get('primary_sic_code'),
        'country': company_info.get('iso_country'),
        'entity_type': company_info.get('entity_type'),
        'segments': segment_info.get('segments') if segment_info else [],
        'financial_reports': financial_data,
        'financial_summary': financial_summary,
        'relative_metrics': relative_metrics,
    }
    
    return profile


def _calculate_relative_metrics(supplier_summary: Dict, self_summary: Dict) -> Dict:
    """计算供应商相对于目标公司的指标"""
    if not supplier_summary or not self_summary:
        return {'status': 'insufficient_data'}
    
    relative = {}
    
    # 规模比较
    supplier_sales = supplier_summary.get('avg_sales')
    self_sales = self_summary.get('avg_sales')
    if supplier_sales and self_sales and self_sales != 0:
        relative['sales_ratio'] = _convert_for_json(supplier_sales / self_sales)
    
    # 利润率比较
    supplier_margin = supplier_summary.get('avg_margin')
    self_margin = self_summary.get('avg_margin')
    if supplier_margin is not None and self_margin is not None:
        relative['margin_difference'] = _convert_for_json(supplier_margin - self_margin)
    
    # 杠杆比较
    supplier_debt = supplier_summary.get('avg_debt_equity')
    self_debt = self_summary.get('avg_debt_equity')
    if supplier_debt is not None and self_debt is not None:
        relative['debt_equity_difference'] = _convert_for_json(supplier_debt - self_debt)
    
    return relative


# ============================================================================
# 并行画像构建
# ============================================================================

def _prepare_single_supplier_profile(args: Tuple) -> Tuple[str, Optional[Dict]]:
    """单个供应商画像准备 - 用于并行处理"""
    global _process_dataframes
    
    supplier_entity_id, self_profile, cutoff_date, num_financial_reports = args
    
    try:
        profile = build_supplier_profile(
            supplier_entity_id,
            self_profile,
            cutoff_date,
            _process_dataframes,
            num_financial_reports
        )
        return supplier_entity_id, profile
    except Exception as e:
        return supplier_entity_id, {'error': str(e)}


def parallel_prepare_supplier_profiles(
    candidate_df: pd.DataFrame,
    self_profile: Dict,
    cutoff_date: pd.Timestamp,
    all_dfs: Dict[str, pd.DataFrame],
    num_financial_reports: int = 1,
    max_workers: int = 32
) -> Dict[str, Dict]:
    """
    并行准备多个供应商的画像
    
    Args:
        candidate_df: 候选供应商DataFrame
        self_profile: 目标公司画像
        cutoff_date: 截止日期
        all_dfs: 所有数据集
        num_financial_reports: 财务报告数量
        max_workers: 最大并行数
    
    Returns:
        {supplier_entity_id: profile} 字典
    """
    # 获取候选供应商ID列表
    candidate_ids = candidate_df['factset_entity_id'].unique().tolist()
    
    if not candidate_ids:
        return {}
    
    print(f"  -> 开始并行构建 {len(candidate_ids)} 个供应商画像...")
    
    # 准备任务
    tasks = [
        (sid, self_profile, cutoff_date, num_financial_reports)
        for sid in candidate_ids
    ]
    
    supplier_profiles = {}
    
    # 确定有效的worker数量
    effective_workers = min(max_workers, multiprocessing.cpu_count(), 16)
    
    # 使用进程池并行处理
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=effective_workers,
        initializer=_initialize_supplier_process_pool,
        initargs=(all_dfs,)
    ) as executor:
        futures = [executor.submit(_prepare_single_supplier_profile, task) for task in tasks]
        
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(futures), 
                          desc="Building Profiles"):
            try:
                supplier_id, profile = future.result()
                if profile is not None:
                    supplier_profiles[supplier_id] = profile
            except Exception as e:
                pass  # 静默处理异常
    
    print(f"  -> 完成构建 {len(supplier_profiles)} 个供应商画像")
    
    return supplier_profiles


# ============================================================================
# 主函数 - 测试用
# ============================================================================

if __name__ == "__main__":
    from stage1_data_loader import load_all_dataframes_parallel
    
    print("=== Stage 2: 画像构建测试 ===\n")
    
    # 加载数据
    all_dfs = load_all_dataframes_parallel()
    
    # 测试构建自身画像
    test_entity_id = "000C7F-E"  # Apple
    decision_year = 2020
    
    print(f"\n测试实体: {test_entity_id}")
    print(f"决策年份: {decision_year}")
    
    self_profile = build_self_profile(test_entity_id, decision_year, all_dfs)
    print(f"\n自身画像:")
    print(f"  - 公司名称: {self_profile.get('entity_name')}")
    print(f"  - 行业: {self_profile.get('industry')}")
    print(f"  - 历史供应商数量: {self_profile.get('historical_supplier_count')}")
