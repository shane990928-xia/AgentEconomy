#!/usr/bin/env python3
"""
Stage 1: 数据加载
================

功能：
- 并行加载所有必需的数据集
- 基础数据查询函数
- FSYM_ID映射函数
- 财务数据提取函数

用法：
    from stage1_data_loader import (
        load_all_dataframes_parallel,
        get_company_info_by_id,
        get_industry_info,
        get_segment_info
    )
"""

import os
import pandas as pd
import numpy as np
import multiprocessing
from typing import Dict, List, Optional, Any, Set
from tqdm import tqdm

from stage0_config import DATA_BASE_PATH

# ============================================================================
# 全局变量 - 进程池数据共享
# ============================================================================

_process_dataframes: Optional[Dict[str, pd.DataFrame]] = None


def _initialize_process_pool(dataframes_dict: Dict[str, pd.DataFrame]):
    """初始化进程池，将数据放入全局变量"""
    global _process_dataframes
    _process_dataframes = dataframes_dict


# ============================================================================
# 数据加载
# ============================================================================

def _load_single_df_wrapper(args):
    """单个DataFrame加载包装器"""
    name, path = args
    try:
        df = pd.read_parquet(path)
        
        # 数据类型处理
        if 'company_id' in df.columns:
            df['company_id'] = pd.to_numeric(df['company_id'], errors='coerce')
            df.dropna(subset=['company_id'], inplace=True)
            df['company_id'] = df['company_id'].astype('int64')
        
        if 'source_company_id' in df.columns:
            df['source_company_id'] = pd.to_numeric(df['source_company_id'], errors='coerce')
            df.dropna(subset=['source_company_id'], inplace=True)
            df['source_company_id'] = df['source_company_id'].astype('int64')
        
        if 'target_company_id' in df.columns:
            df['target_company_id'] = pd.to_numeric(df['target_company_id'], errors='coerce')
            df.dropna(subset=['target_company_id'], inplace=True)
            df['target_company_id'] = df['target_company_id'].astype('int64')
        
        # 日期处理
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce').dt.normalize()
        if 'period_end_date' in df.columns:
            df['period_end_date'] = pd.to_datetime(df['period_end_date'], errors='coerce').dt.normalize()
        if 'start_' in df.columns:
            df['start_'] = pd.to_datetime(df['start_'], errors='coerce').dt.normalize()
        
        in_memory_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        return name, df, in_memory_size_mb
    
    except Exception as e:
        print(f"\n[!!] 错误：加载 {name} 失败: {e}\n", flush=True)
        return name, pd.DataFrame(), 0.0


def load_all_dataframes_parallel() -> Dict[str, pd.DataFrame]:
    """并行加载所有必需的数据集"""
    print("--- [Stage 1] 开始并行加载所有必需的数据集 ---")
    
    paths = {
        'standard_entity': os.path.join(DATA_BASE_PATH, 'data', 'standard_entity.parquet'),
        'company_factset': os.path.join(DATA_BASE_PATH, 'data', 'company_factset.parquet'),
        'security_map': os.path.join(DATA_BASE_PATH, 'data', 'security_map.parquet'),
        'security_entity_map': os.path.join(DATA_BASE_PATH, 'data', 'security_entity_map.parquet'),
        'own_basic': os.path.join(DATA_BASE_PATH, 'factset_own', 'own_basic.parquet'),
        'ff_int_qf': os.path.join(DATA_BASE_PATH, 'data', 'ff_int_qf.parquet'),
        'ff_usc_qf': os.path.join(DATA_BASE_PATH, 'data', 'ff_usc_qf.parquet'),
        'factset_revere_relationship': os.path.join(DATA_BASE_PATH, 'data', 'factset_revere_relationship.parquet'),
        'segment_summary': os.path.join(DATA_BASE_PATH, 'data', 'Business Segment Exposure-Summary.parquet'),
        'sic_map': os.path.join(DATA_BASE_PATH, 'data', 'sic_map.parquet'),
    }
    
    dataframes = {}
    total_in_memory_size_mb = 0
    
    # 筛选存在的文件
    tasks = [(name, path) for name, path in paths.items() if os.path.exists(path)]
    if len(tasks) != len(paths):
        missing_files = set(paths.keys()) - {t[0] for t in tasks}
        print(f"[WARNING] 部分数据文件未找到: {', '.join(missing_files)}")
    
    # 并行加载（至少1个进程；tasks 为空时跳过）
    if tasks:
        num_procs = max(1, min(os.cpu_count() or 1, len(tasks)))
        with multiprocessing.Pool(processes=num_procs) as pool:
            with tqdm(total=len(tasks), desc="Loading Datasets", unit="file") as pbar:
                for name, df, size_mb in pool.imap_unordered(_load_single_df_wrapper, tasks):
                    dataframes[name] = df
                    total_in_memory_size_mb += size_mb
                    pbar.set_description(f"Loaded {name:<28}")
                    pbar.update(1)
    
    print(f"\n--- [Stage 1] 数据加载完成 ---")
    print(f"  -> 总计加载了 {len(dataframes)} 个数据集")
    print(f"  -> 总内存占用: {total_in_memory_size_mb:.2f} MB\n")
    
    return dataframes


# ============================================================================
# FSYM_ID 映射函数
# ============================================================================

def _get_all_related_fsym_ids(factset_entity_id: str, all_dfs: Dict[str, pd.DataFrame]) -> Set[str]:
    """
    获取实体相关的所有FSYM_ID
    实现多层次fallback机制
    """
    candidate_ids: Set[str] = set()

    # 方法1: 通过 security_entity_map 直接映射
    sec_ent_map = all_dfs.get('security_entity_map')
    if sec_ent_map is not None and not sec_ent_map.empty:
        direct_mappings = sec_ent_map[sec_ent_map['FACTSET_ENTITY_ID'] == factset_entity_id]['FSYM_ID'].unique()
        candidate_ids.update(direct_mappings)
        
        # 格式转换：-S 后缀转换为 -R 后缀
        for fsym_id in direct_mappings:
            if '-S' in str(fsym_id):
                base_id = str(fsym_id).replace('-S', '')
                candidate_ids.add(f"{base_id}-R")
        
        if candidate_ids:
            return candidate_ids

    # 方法2: 通过 own_basic 映射
    own_basic = all_dfs.get('own_basic')
    if own_basic is not None and not own_basic.empty:
        entity_securities = own_basic[own_basic['factset_entity_id'] == factset_entity_id]
        if not entity_securities.empty:
            primary_sec = entity_securities.sort_values('mkt_val', ascending=False).head(1)
            if not primary_sec.empty:
                fsym_id = primary_sec.iloc[0].get('fsym_id')
                if fsym_id:
                    candidate_ids.add(fsym_id)
                    # 格式转换
                    if '-S' in str(fsym_id):
                        base_id = str(fsym_id).replace('-S', '').split('-')[0]
                        candidate_ids.add(f"{base_id}-R")
            
            if candidate_ids:
                return candidate_ids

    # 方法3: 通过 company_factset + 财务数据表直接查找
    company_factset = all_dfs.get('company_factset')
    if company_factset is not None and not company_factset.empty:
        company_mapping = company_factset[company_factset['fs_entity_id'] == factset_entity_id]
        if not company_mapping.empty:
            for ff_dataset in ['ff_usc_qf', 'ff_int_qf']:
                df = all_dfs.get(ff_dataset)
                if df is not None and not df.empty:
                    if 'FACTSET_ENTITY_ID' in df.columns:
                        direct_financial = df[df['FACTSET_ENTITY_ID'] == factset_entity_id]['FSYM_ID'].unique()
                        candidate_ids.update(direct_financial)
                    
                    # 基于security_entity_map的间接查找
                    sec_ent_map = all_dfs.get('security_entity_map')
                    if sec_ent_map is not None and not sec_ent_map.empty:
                        entity_securities = sec_ent_map[sec_ent_map['FACTSET_ENTITY_ID'] == factset_entity_id]
                        if not entity_securities.empty:
                            mapped_fsym_ids = entity_securities['FSYM_ID'].dropna().unique()
                            financial_matches = df[df['FSYM_ID'].isin(mapped_fsym_ids)]['FSYM_ID'].unique()
                            if len(financial_matches) > 0:
                                candidate_ids.update(financial_matches)

    return candidate_ids


def _get_fsym_ids_improved(factset_entity_id: str, all_dfs: Dict[str, pd.DataFrame]) -> List[str]:
    """改进的FSYM_ID获取方法"""
    fsym_ids = list(_get_all_related_fsym_ids(factset_entity_id, all_dfs))
    return fsym_ids if fsym_ids else []


# ============================================================================
# 公司信息查询
# ============================================================================

def get_company_info_by_id(factset_entity_id: str, all_dfs: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
    """通过factset_entity_id获取公司完整信息"""
    df_se = all_dfs.get('standard_entity')
    df_cf = all_dfs.get('company_factset')
    
    if df_se is None or df_se.empty:
        return None
    
    entity_row = df_se[df_se['factset_entity_id'] == factset_entity_id]
    if entity_row.empty:
        return None
    
    entity_info = entity_row.iloc[0].to_dict()
    
    # 获取company_id
    company_id = None
    if df_cf is not None and not df_cf.empty:
        cf_match = df_cf[df_cf['fs_entity_id'] == factset_entity_id]
        if not cf_match.empty:
            company_id = cf_match.iloc[0].get('company_id')
    
    return {
        'factset_entity_id': factset_entity_id,
        'entity_proper_name': entity_info.get('entity_proper_name'),
        'primary_sic_code': entity_info.get('primary_sic_code'),
        'iso_country': entity_info.get('iso_country'),
        'entity_type': entity_info.get('entity_type'),
        'company_id': company_id,
    }


def get_industry_info(factset_entity_id: str, all_dfs: Dict[str, pd.DataFrame]) -> str:
    """
    获取公司的行业信息，实现fallback逻辑：
    1. 首先尝试从 Business Segment Exposure-Summary 获取
    2. 如果获取不到，则从 standard_entity 中的 sector_code 获取
    3. 通过 sic_map 查询对应的 sic_desc
    """
    try:
        # 方法1: 从 Business Segment Exposure-Summary 获取
        df_seg = all_dfs.get('segment_summary')
        df_cf = all_dfs.get('company_factset')
        
        if df_seg is not None and df_cf is not None:
            cf_match = df_cf[df_cf['fs_entity_id'] == factset_entity_id]
            if not cf_match.empty:
                company_id = cf_match.iloc[0].get('company_id')
                if company_id is not None:
                    company_segs = df_seg[df_seg['company_id'] == company_id]
                    if not company_segs.empty:
                        latest_seg = company_segs.sort_values('period_end_date', ascending=False).iloc[0]
                        if 'name' in latest_seg and pd.notna(latest_seg['name']):
                            return str(latest_seg['name'])
        
        # 方法2: 从 standard_entity 中的 sector_code 获取
        df_se = all_dfs.get('standard_entity')
        df_sic = all_dfs.get('sic_map')
        
        if df_se is not None:
            entity_row = df_se[df_se['factset_entity_id'] == factset_entity_id]
            if not entity_row.empty:
                sic_code = entity_row.iloc[0].get('primary_sic_code')
                if sic_code and df_sic is not None:
                    sic_match = df_sic[df_sic['sic_code'] == sic_code]
                    if not sic_match.empty:
                        return str(sic_match.iloc[0].get('sic_desc', 'Unknown'))
                elif sic_code:
                    return f"SIC: {sic_code}"
        
        return "Unknown"
        
    except Exception as e:
        return f"Error: {str(e)}"


def get_segment_info(factset_entity_id: str, all_dfs: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
    """获取公司的业务段信息"""
    df_seg = all_dfs.get('segment_summary')
    df_cf = all_dfs.get('company_factset')
    
    if df_seg is None or df_cf is None:
        return None
    
    # 获取company_id
    cf_match = df_cf[df_cf['fs_entity_id'] == factset_entity_id]
    if cf_match.empty:
        return None
    
    company_id = cf_match.iloc[0].get('company_id')
    if company_id is None:
        return None
    
    # 获取最新的业务段数据
    company_segs = df_seg[df_seg['company_id'] == company_id]
    if company_segs.empty:
        return None
    
    latest_date = company_segs['period_end_date'].max()
    latest_segs = company_segs[company_segs['period_end_date'] == latest_date]
    
    # 按收入占比排序
    if 'segment_percent' in latest_segs.columns:
        latest_segs = latest_segs.sort_values('segment_percent', ascending=False)
    
    segments = []
    for _, row in latest_segs.head(5).iterrows():
        segments.append({
            'path': row.get('path'),
            'name': row.get('name'),
            'segment_percent': row.get('segment_percent'),
        })
    
    return {
        'company_id': company_id,
        'period_end_date': latest_date,
        'segments': segments,
    }


# ============================================================================
# 财务数据提取
# ============================================================================

def extract_financial_data(
    factset_entity_id: str, 
    cutoff_date: pd.Timestamp, 
    all_dfs: Dict[str, pd.DataFrame], 
    fields: List[str] = None,
    num_reports: int = 1
) -> List[Dict]:
    """
    提取财务数据
    
    Args:
        factset_entity_id: 公司实体ID
        cutoff_date: 截止日期
        all_dfs: 所有数据集
        fields: 需要的字段列表
        num_reports: 返回的报告数量
    
    Returns:
        财务数据列表
    """
    if fields is None:
        fields = ['FF_SALES', 'FF_NET_MGN', 'FF_DEBT_EQ', 'FF_ROIC', 'FF_CURR_RATIO']
    
    try:
        # 获取FSYM_ID映射
        fsym_ids = _get_fsym_ids_improved(factset_entity_id, all_dfs)
        if not fsym_ids:
            return [{'financial_data_error': f'No FSYM_ID mapping found for {factset_entity_id}'}]
        
        financial_data = []
        
        # 从两个财务数据表获取数据
        for df_name in ['ff_usc_qf', 'ff_int_qf']:
            df = all_dfs.get(df_name)
            if df is None or df.empty:
                continue
            
            for fsym_id in fsym_ids:
                data = df[df['FSYM_ID'] == fsym_id].copy()
                if data.empty:
                    continue
                
                # 确保DATE列是datetime类型
                data['DATE'] = pd.to_datetime(data['DATE'], errors='coerce')
                
                # 过滤截止日期之前的数据
                data_filtered = data[data['DATE'] <= cutoff_date]
                if data_filtered.empty:
                    continue
                
                for _, row in data_filtered.iterrows():
                    record = {
                        'date': row.get('DATE', ''),
                        'currency': row.get('CURRENCY', 'USD'),
                        'source': df_name,
                    }
                    for field in fields:
                        record[field.lower()] = row.get(field, None)
                    
                    financial_data.append(record)
        
        if not financial_data:
            return [{'financial_data_error': f'No financial data found for {factset_entity_id}'}]
        
        # 按日期排序并返回最新的记录
        financial_data.sort(key=lambda x: x['date'] if x['date'] else '1900-01-01', reverse=True)
        return financial_data[:num_reports]
        
    except Exception as e:
        return [{'financial_data_error': f'Error: {str(e)}'}]


# ============================================================================
# 主函数 - 测试用
# ============================================================================

if __name__ == "__main__":
    print("=== Stage 1: 数据加载测试 ===\n")
    
    # 加载数据
    all_dfs = load_all_dataframes_parallel()
    
    # 测试查询
    test_entity_id = "000C7F-E"  # Apple
    print(f"\n测试实体: {test_entity_id}")
    
    company_info = get_company_info_by_id(test_entity_id, all_dfs)
    print(f"公司信息: {company_info}")
    
    industry = get_industry_info(test_entity_id, all_dfs)
    print(f"行业信息: {industry}")
    
    segments = get_segment_info(test_entity_id, all_dfs)
    print(f"业务段信息: {segments}")
