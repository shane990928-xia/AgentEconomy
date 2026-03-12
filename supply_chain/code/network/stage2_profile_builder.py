#!/usr/bin/env python3
"""
Stage 2: Company Profile Builder (批量查询优化版)
==================================

功能：为所有年份的公司节点生成完整的数据画像

优化：使用批量查询替代多进程，大幅提升性能
"""

import pandas as pd
import numpy as np
import os
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 全局配置
# ============================================================================

STAGE1_RESULT_DIR = "../result"
RESULT_DIR = "../result"
DATA_BASE_PATH = "/root/tmp/supplier/experiment/data/factset"

# ============================================================================
# 1. 数据加载
# ============================================================================

def load_all_dataframes() -> Dict[str, pd.DataFrame]:
    """加载所有必需的数据集"""
    print("--- [Step 1] 开始加载所有必需的数据集 ---")
    datasets = {
        'standard_entity': os.path.join(DATA_BASE_PATH, 'data', 'standard_entity.parquet'),
        'company_factset': os.path.join(DATA_BASE_PATH, 'data', 'company_factset.parquet'),
        'company': os.path.join(DATA_BASE_PATH, 'data', 'company.parquet'),
        'security_map': os.path.join(DATA_BASE_PATH, 'data', 'security_map.parquet'),
        'security_entity_map': os.path.join(DATA_BASE_PATH, 'data', 'security_entity_map.parquet'),
        'own_basic': os.path.join(DATA_BASE_PATH, 'factset_own', 'own_basic.parquet'),
        'ff_int_qf': os.path.join(DATA_BASE_PATH, 'data', 'ff_int_qf.parquet'),
        'ff_usc_qf': os.path.join(DATA_BASE_PATH, 'data', 'ff_usc_qf.parquet'),
        'factset_revere_relationship': os.path.join(DATA_BASE_PATH, 'data', 'factset_revere_relationship.parquet'),
        'segment_summary': os.path.join(DATA_BASE_PATH, 'data', 'Business Segment Exposure-Summary.parquet'),
        'sic_map': os.path.join(DATA_BASE_PATH, 'data', 'sic_map.parquet'),
    }

    all_dataframes = {}
    for name, path in tqdm(datasets.items(), desc="Loading datasets"):
        try:
            df = pd.read_parquet(path)
            all_dataframes[name] = df
            print(f"  ✓ {name}: {len(df):,} 行")
        except Exception as e:
            print(f"  ✗ {name}: 加载失败 - {e}")

    print(f"--- 成功加载 {len(all_dataframes)} / {len(datasets)} 个数据集 ---")
    return all_dataframes


# ============================================================================
# 2. 批量查询辅助函数
# ============================================================================

def _convert_for_json(value):
    """JSON转换辅助函数"""
    if pd.isna(value): return None
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, pd.Timestamp): return value.strftime('%Y-%m-%d')
    if isinstance(value, (pd.Timestamp, datetime)): return value.isoformat()
    if isinstance(value, (np.integer, int)): return int(value)
    if isinstance(value, (np.floating, float)): return float(value)
    return value


@np.vectorize
def _fast_split(s: str) -> str:
    """快速分割FSYM ID"""
    if isinstance(s, str) and '-' in s:
        return '-'.join(s.split('-')[:2])
    return s


# ============================================================================
# 3. 批量查询：基本信息
# ============================================================================

def batch_get_basic_info(
    entity_ids: List[str],
    all_dfs: Dict[str, pd.DataFrame]
) -> Dict[str, Dict]:
    """批量获取公司基本信息"""
    print("  [批量查询] 获取基本信息...")

    basic_info_dict = {}

    # 批量查询standard_entity
    df_se = all_dfs.get('standard_entity')
    if df_se is not None:
        se_batch = df_se[df_se['factset_entity_id'].isin(entity_ids)].copy()
        for _, row in se_batch.iterrows():
            entity_id = row['factset_entity_id']
            basic_info_dict[entity_id] = row.to_dict()

    # 批量查询company_factset获取company_id
    df_cf = all_dfs.get('company_factset')
    if df_cf is not None:
        cf_batch = df_cf[df_cf['fs_entity_id'].isin(entity_ids)].copy()

        # 优先使用pri='Y'的记录
        cf_primary = cf_batch[cf_batch['pri'] == 'Y'].drop_duplicates('fs_entity_id', keep='first')
        cf_fallback = cf_batch.drop_duplicates('fs_entity_id', keep='first')

        for entity_id in entity_ids:
            if entity_id in cf_primary['fs_entity_id'].values:
                company_id = cf_primary[cf_primary['fs_entity_id'] == entity_id].iloc[0]['company_id']
            elif entity_id in cf_fallback['fs_entity_id'].values:
                company_id = cf_fallback[cf_fallback['fs_entity_id'] == entity_id].iloc[0]['company_id']
            else:
                continue

            if entity_id in basic_info_dict:
                basic_info_dict[entity_id]['company_id'] = int(company_id)

    # 批量查询company获取ticker等信息
    company_df = all_dfs.get('company')
    if company_df is not None:
        company_ids = [info.get('company_id') for info in basic_info_dict.values() if 'company_id' in info]
        if company_ids:
            company_batch = company_df[company_df['id'].isin(company_ids)].copy()
            company_batch = company_batch.sort_values('start_').drop_duplicates('id', keep='last')

            for entity_id, info in basic_info_dict.items():
                if 'company_id' in info:
                    company_record = company_batch[company_batch['id'] == info['company_id']]
                    if not company_record.empty:
                        record = company_record.iloc[0]
                        info.update({
                            'company_name': record.get('name', info.get('entity_proper_name', '')),
                            'ticker': record.get('ticker', ''),
                            'company_start_date': record.get('start_', ''),
                            'company_end_date': record.get('end_', '')
                        })

    # 填充缺失的entity
    for entity_id in entity_ids:
        if entity_id not in basic_info_dict:
            basic_info_dict[entity_id] = {
                'factset_entity_id': entity_id,
                'entity_proper_name': f'Unknown Entity {entity_id}',
                'primary_sic_name': 'Unknown',
                'iso_country': 'Unknown',
                'company_id': None
            }

    print(f"    ✓ 获取了 {len(basic_info_dict)} 个公司的基本信息")
    return basic_info_dict


# ============================================================================
# 3.5 批量查询：基于年份的Company ID映射
# ============================================================================

def batch_get_company_id_for_year(
    entity_ids: List[str],
    year: int,
    all_dfs: Dict[str, pd.DataFrame]
) -> Dict[str, int]:
    """
    批量获取指定年份有效的entity_id -> company_id映射

    选项2实现：先用当前年份的entity_id获取对应的company_id（基于时间有效性）

    Args:
        entity_ids: entity ID列表
        year: 年份（如2016）
        all_dfs: 所有数据框

    Returns:
        字典 {entity_id: company_id}
    """
    print(f"  [批量查询] 获取{year}年有效的Company ID映射...")

    entity_company_map = {}

    df_cf = all_dfs.get('company_factset')
    if df_cf is None:
        print("    ⚠ company_factset表不存在")
        return entity_company_map

    # 确保时间列是datetime类型
    df_cf = df_cf.copy()
    df_cf['start_'] = pd.to_datetime(df_cf['start_'], errors='coerce')
    df_cf['end_'] = pd.to_datetime(df_cf['end_'], errors='coerce')

    # 定义年份的时间范围
    year_start = pd.Timestamp(f"{year}-01-01")
    year_end = pd.Timestamp(f"{year}-12-31")

    # 批量查询：找到在该年份有效的映射记录
    # 条件：start_ <= year_end AND end_ >= year_start
    cf_batch = df_cf[
        df_cf['fs_entity_id'].isin(entity_ids) &
        (df_cf['start_'] <= year_end) &
        (df_cf['end_'] >= year_start)
    ].copy()

    print(f"    → 找到 {len(cf_batch)} 条映射记录")

    # 对每个entity_id，优先选择pri='Y'的记录，然后按start_最新
    for entity_id in entity_ids:
        entity_records = cf_batch[cf_batch['fs_entity_id'] == entity_id]

        if entity_records.empty:
            continue

        # 优先pri='Y'
        primary_records = entity_records[entity_records['pri'] == 'Y']

        if not primary_records.empty:
            # 如果有多条pri='Y'，选择start_最新的
            selected = primary_records.sort_values('start_', ascending=False).iloc[0]
        else:
            # 否则选择start_最新的
            selected = entity_records.sort_values('start_', ascending=False).iloc[0]

        try:
            company_id = int(selected['company_id'])
            entity_company_map[entity_id] = company_id
        except (ValueError, TypeError):
            continue

    print(f"    ✓ 成功映射 {len(entity_company_map)}/{len(entity_ids)} 个entity")

    return entity_company_map


# ============================================================================
# 4. 批量查询：FSYM_ID映射
# ============================================================================

def batch_get_fsym_ids(
    entity_ids: List[str],
    all_dfs: Dict[str, pd.DataFrame]
) -> Dict[str, Set[str]]:
    """批量获取所有公司的FSYM_ID映射"""
    print("  [批量查询] 获取FSYM_ID映射...")

    entity_fsym_map = {entity_id: set() for entity_id in entity_ids}

    # 方法1: security_entity_map
    sec_ent_map = all_dfs.get('security_entity_map')
    if sec_ent_map is not None:
        sem_batch = sec_ent_map[sec_ent_map['FACTSET_ENTITY_ID'].isin(entity_ids)].copy()
        for _, row in sem_batch.iterrows():
            entity_id = row['FACTSET_ENTITY_ID']
            fsym_id = row['FSYM_ID']
            if pd.notna(fsym_id):
                entity_fsym_map[entity_id].add(fsym_id)
                # 生成-R变体
                if '-S' in str(fsym_id):
                    base_id = str(fsym_id).replace('-S', '')
                    entity_fsym_map[entity_id].add(f"{base_id}-R")

    # 方法2: own_basic
    own_basic = all_dfs.get('own_basic')
    if own_basic is not None:
        own_batch = own_basic[own_basic['factset_entity_id'].isin(entity_ids)].copy()

        for entity_id in entity_ids:
            entity_securities = own_batch[own_batch['factset_entity_id'] == entity_id]
            if entity_securities.empty:
                continue

            active_equity = entity_securities[
                (entity_securities['issue_type'] == 'EQ') &
                (entity_securities['active'] == 1)
            ]
            base_securities = active_equity if not active_equity.empty else entity_securities

            perm_ids = base_securities['fs_perm_sec_id'].dropna().unique()
            for perm_id in perm_ids:
                entity_fsym_map[entity_id].add(perm_id)

                perm_str = str(perm_id)
                if '-S-' in perm_str:
                    base_id = perm_str.split('-S-')[0]
                    entity_fsym_map[entity_id].add(f"{base_id}-R")
                elif '-S' in perm_str:
                    base_id = perm_str.replace('-S', '')
                    entity_fsym_map[entity_id].add(f"{base_id}-R")

            # 通过security_map扩展
            if len(perm_ids) > 0:
                base_ids = set(_fast_split(perm_ids))
                sec_map = all_dfs.get('security_map')
                if sec_map is not None:
                    mapped = sec_map[sec_map['FSYM_COMPANY_ID'].isin(base_ids)]
                    for fsym_id in mapped['FSYM_ID'].dropna().unique():
                        entity_fsym_map[entity_id].add(fsym_id)
                        if '-L' in str(fsym_id):
                            base_id = str(fsym_id).replace('-L', '').split('-')[0]
                            entity_fsym_map[entity_id].add(f"{base_id}-R")

    # 方法3: 直接从财务表查找
    for ff_dataset in ['ff_usc_qf', 'ff_int_qf']:
        df = all_dfs.get(ff_dataset)
        if df is None:
            continue

        # 3a: 通过FACTSET_ENTITY_ID直接查找
        if 'FACTSET_ENTITY_ID' in df.columns:
            direct_matches = df[df['FACTSET_ENTITY_ID'].isin(entity_ids)][
                ['FACTSET_ENTITY_ID', 'FSYM_ID']
            ].drop_duplicates()
            for _, row in direct_matches.iterrows():
                entity_id = row['FACTSET_ENTITY_ID']
                fsym_id = row['FSYM_ID']
                if pd.notna(fsym_id):
                    entity_fsym_map[entity_id].add(fsym_id)

        # 3b: 通过entity前缀模糊匹配（优化：批量向量化）
        # 收集所有>=6位的前缀
        entity_prefixes = {}
        for entity_id in entity_ids:
            prefix = entity_id.split('-')[0]
            if len(prefix) >= 6:
                if prefix not in entity_prefixes:
                    entity_prefixes[prefix] = []
                entity_prefixes[prefix].append(entity_id)

        if entity_prefixes and 'FSYM_ID' in df.columns:
            # 批量向量化：提取FSYM_ID的前缀列
            df_fsym_prefix = df['FSYM_ID'].str.split('-').str[0]

            # 批量过滤：FSYM_ID前缀在我们的前缀集合中
            prefix_set = set(entity_prefixes.keys())
            matched_df = df[df_fsym_prefix.isin(prefix_set)][['FSYM_ID']].drop_duplicates()

            # 为每个entity分配匹配的FSYM_IDs
            for fsym_id in matched_df['FSYM_ID']:
                fsym_prefix = str(fsym_id).split('-')[0]
                if fsym_prefix in entity_prefixes:
                    for entity_id in entity_prefixes[fsym_prefix]:
                        entity_fsym_map[entity_id].add(fsym_id)

    total_fsym = sum(len(fsyms) for fsyms in entity_fsym_map.values())
    print(f"    ✓ 获取了 {total_fsym} 个FSYM_ID (平均 {total_fsym/len(entity_ids):.1f} 个/公司)")
    return entity_fsym_map


# ============================================================================
# 5. 批量查询：财务数据
# ============================================================================

def batch_get_financial_data(
    entity_ids: List[str],
    entity_fsym_map: Dict[str, Set[str]],
    cutoff_date: pd.Timestamp,
    all_dfs: Dict[str, pd.DataFrame],
    fields: List[str]
) -> Dict[str, Dict]:
    """批量获取财务数据"""
    print("  [批量查询] 获取财务数据...")

    # 收集所有需要查询的FSYM_IDs
    all_fsym_ids = set()
    for fsyms in entity_fsym_map.values():
        all_fsym_ids.update(fsyms)

    if not all_fsym_ids:
        print("    ⚠ 没有找到任何FSYM_ID")
        return {entity_id: {'financial_data_error': 'No FSYM_ID found'}
                for entity_id in entity_ids}

    # 批量查询财务表
    financials_list = []
    for df_name in ['ff_usc_qf', 'ff_int_qf']:
        df = all_dfs.get(df_name)
        if df is not None:
            batch = df[df['FSYM_ID'].isin(all_fsym_ids)].copy()
            if not batch.empty:
                financials_list.append(batch)

    if not financials_list:
        print("    ⚠ 没有找到任何财务数据")
        return {entity_id: {'financial_data_error': 'No financial data found'}
                for entity_id in entity_ids}

    # 合并并预处理
    all_financials = pd.concat(financials_list, ignore_index=True)
    all_financials['DATE'] = pd.to_datetime(all_financials['DATE'], errors='coerce')
    all_financials = all_financials.dropna(subset=['DATE'])
    all_financials = all_financials[all_financials['DATE'] <= cutoff_date]

    # 按FF_MKT_VAL排序，每个日期保留最大市值的记录
    all_financials = all_financials.sort_values('FF_MKT_VAL', ascending=False)
    all_financials = all_financials.drop_duplicates(subset=['FSYM_ID', 'DATE'], keep='first')

    # 为每个entity提取最新财务数据
    financial_data_dict = {}
    for entity_id in entity_ids:
        fsym_ids = entity_fsym_map.get(entity_id, set())
        if not fsym_ids:
            financial_data_dict[entity_id] = {
                'financial_data_error': f'No FSYM_ID for {entity_id}'
            }
            continue

        entity_financials = all_financials[all_financials['FSYM_ID'].isin(fsym_ids)]
        if entity_financials.empty:
            financial_data_dict[entity_id] = {
                'financial_data_error': f'No financial records for {entity_id}'
            }
            continue

        # 取最新日期的记录
        latest_report = entity_financials.sort_values('DATE', ascending=False).iloc[0]

        report_dict = {
            field: _convert_for_json(latest_report.get(field))
            for field in fields
        }
        report_dict['report_date'] = _convert_for_json(latest_report.get('DATE'))

        financial_data_dict[entity_id] = report_dict

    success_count = sum(1 for v in financial_data_dict.values()
                       if 'financial_data_error' not in v)
    print(f"    ✓ 成功获取 {success_count}/{len(entity_ids)} 个公司的财务数据")
    return financial_data_dict


# ============================================================================
# 6. 批量查询：关系数据
# ============================================================================

def batch_get_relationship_data(
    entity_ids: List[str],
    basic_info_dict: Dict[str, Dict],
    cutoff_date: pd.Timestamp,
    all_dfs: Dict[str, pd.DataFrame]
) -> Dict[str, Dict]:
    """批量获取供应商关系数据"""
    print("  [批量查询] 获取关系数据...")

    relationship_dict = {entity_id: {} for entity_id in entity_ids}

    # 收集所有company_ids
    company_ids = []
    entity_to_company = {}
    for entity_id, info in basic_info_dict.items():
        company_id = info.get('company_id')
        if company_id is not None:
            try:
                company_id = int(company_id)
                company_ids.append(company_id)
                entity_to_company[entity_id] = company_id
            except (ValueError, TypeError):
                pass

    if not company_ids:
        print("    ⚠ 没有找到任何company_id")
        return relationship_dict

    # 批量查询关系表
    df_rels = all_dfs.get('factset_revere_relationship')
    if df_rels is None:
        return relationship_dict

    df_rels = df_rels.copy()
    df_rels['start_'] = pd.to_datetime(df_rels['start_'], errors='coerce')

    # 批量过滤
    rels_batch = df_rels[
        (df_rels['source_company_id'].isin(company_ids)) &
        (df_rels['start_'] <= cutoff_date)
    ].copy()

    # 为每个entity提取关系
    for entity_id, company_id in entity_to_company.items():
        company_rels = rels_batch[rels_batch['source_company_id'] == company_id]

        if company_rels.empty:
            continue

        # 优先提取CUSTOMER关系
        customer_rels = company_rels[company_rels['rel_type'] == 'CUSTOMER']

        if not customer_rels.empty:
            total_customers = len(customer_rels)
            total_revenue_percent = customer_rels['revenue_percent'].sum() if 'revenue_percent' in customer_rels.columns else 0
            avg_revenue_percent = customer_rels['revenue_percent'].mean() if 'revenue_percent' in customer_rels.columns else 0

            if 'revenue_percent' in customer_rels.columns and customer_rels['revenue_percent'].notna().any():
                main_rel = customer_rels.loc[customer_rels['revenue_percent'].idxmax()]
            else:
                main_rel = customer_rels.sort_values('start_', ascending=False).iloc[0]

            relationship_dict[entity_id] = {
                'revenue_percent': _convert_for_json(main_rel.get('revenue_percent')),
                'start_': _convert_for_json(main_rel.get('start_')),
                'total_customers': total_customers,
                'total_revenue_percent': total_revenue_percent,
                'avg_revenue_percent': avg_revenue_percent,
                'main_customer_id': _convert_for_json(main_rel.get('target_company_id')),
                'relationship_source': 'network_analysis_comprehensive'
            }
        else:
            # Fallback到其他关系类型
            if not company_rels.empty:
                latest_rel = company_rels.sort_values('start_', ascending=False).iloc[0]
                relationship_dict[entity_id] = {
                    'revenue_percent': _convert_for_json(latest_rel.get('revenue_percent')),
                    'start_': _convert_for_json(latest_rel.get('start_')),
                    'rel_type': latest_rel.get('rel_type'),
                    'relationship_source': 'network_analysis_fallback'
                }

    success_count = sum(1 for v in relationship_dict.values() if v)
    print(f"    ✓ 成功获取 {success_count}/{len(entity_ids)} 个公司的关系数据")
    return relationship_dict


# ============================================================================
# 6.5 单个查询：与指定Hub的关系数据（Hub相关Profile）
# ============================================================================

def get_relationship_with_hub(
    supplier_company_id: int,
    hub_entity_id: str,
    hub_company_id: int,
    cutoff_date: pd.Timestamp,
    all_dfs: Dict[str, pd.DataFrame]
) -> Dict:
    """
    获取供应商与指定Hub的关系数据（参考supplier_test.py的实现）

    这是Hub相关的Profile，会随着Hub的改变而改变

    Args:
        supplier_company_id: 供应商company_id
        hub_entity_id: Hub的entity_id
        hub_company_id: Hub的company_id
        cutoff_date: 截止日期
        all_dfs: 所有数据框

    Returns:
        关系数据字典，包含 revenue_percent, start_, relationship_source
    """
    relationship_data = {}

    df_rels = all_dfs.get('factset_revere_relationship')
    if df_rels is None or supplier_company_id is None or hub_company_id is None:
        return relationship_data

    # 确保start_列是日期类型
    if 'start_' in df_rels.columns:
        df_rels = df_rels.copy()
        df_rels['start_'] = pd.to_datetime(df_rels['start_'], errors='coerce')

    # 查询供应商 -> Hub的CUSTOMER关系（supplier是source，hub是target）
    rel_rows = df_rels[
        (df_rels['source_company_id'] == supplier_company_id) &
        (df_rels['target_company_id'] == hub_company_id) &
        (df_rels['rel_type'] == 'CUSTOMER') &
        (df_rels['start_'] <= cutoff_date)
    ]

    if not rel_rows.empty:
        # 取最新的关系记录
        latest_rel = rel_rows.sort_values(by='start_', ascending=False).iloc[0]
        relationship_data = {
            'revenue_percent': _convert_for_json(latest_rel.get('revenue_percent')),
            'start_': _convert_for_json(latest_rel.get('start_')),
            'relationship_source': f'hub_specific_{hub_entity_id}'
        }
    else:
        # Fallback: 尝试其他关系类型
        other_rel_rows = df_rels[
            (df_rels['source_company_id'] == supplier_company_id) &
            (df_rels['target_company_id'] == hub_company_id) &
            (df_rels['start_'] <= cutoff_date)
        ]
        if not other_rel_rows.empty:
            latest_rel = other_rel_rows.sort_values(by='start_', ascending=False).iloc[0]
            relationship_data = {
                'revenue_percent': _convert_for_json(latest_rel.get('revenue_percent')),
                'start_': _convert_for_json(latest_rel.get('start_')),
                'rel_type': latest_rel.get('rel_type'),
                'relationship_source': f'hub_specific_other_{hub_entity_id}'
            }

    return relationship_data


# ============================================================================
# 6.6 批量查询：Hub特定关系数据（批量优化版）
# ============================================================================

def batch_get_hub_specific_relationships(
    hub_candidate_info: List[Dict],
    cutoff_date: pd.Timestamp,
    all_dfs: Dict[str, pd.DataFrame]
) -> Dict[tuple, Dict]:
    """
    批量查询所有Hub-Supplier对的关系数据

    Args:
        hub_candidate_info: 列表，每个元素包含：
            {
                'hub_id': str,
                'supplier_id': str,
                'hub_company_id': int,
                'supplier_company_id': int
            }
        cutoff_date: 截止日期
        all_dfs: 所有数据框

    Returns:
        字典 {(hub_id, supplier_id): relationship_data}
    """
    print(f"\n  [批量查询] Hub特定关系数据 (共 {len(hub_candidate_info):,} 对)...")

    df_rels = all_dfs.get('factset_revere_relationship')
    if df_rels is None or len(hub_candidate_info) == 0:
        return {}

    # 确保start_列是日期类型
    if 'start_' in df_rels.columns:
        df_rels = df_rels.copy()
        df_rels['start_'] = pd.to_datetime(df_rels['start_'], errors='coerce')

    # 收集所有需要查询的company_id对
    valid_pairs = []
    pair_to_entities = {}  # {(supplier_cid, hub_cid): [(hub_id, supplier_id), ...]}

    for info in hub_candidate_info:
        supplier_cid = info['supplier_company_id']
        hub_cid = info['hub_company_id']
        hub_id = info['hub_id']
        supplier_id = info['supplier_id']

        if supplier_cid is not None and hub_cid is not None:
            pair = (supplier_cid, hub_cid)
            valid_pairs.append(pair)
            if pair not in pair_to_entities:
                pair_to_entities[pair] = []
            pair_to_entities[pair].append((hub_id, supplier_id))

    if len(valid_pairs) == 0:
        print(f"    ⚠ 没有有效的company_id对，跳过")
        return {}

    print(f"    → 有效查询对: {len(valid_pairs):,}")

    # 批量过滤：只保留截止日期前的关系
    rels_batch = df_rels[df_rels['start_'] <= cutoff_date].copy()
    print(f"    → 截止日期前的关系记录: {len(rels_batch):,}")

    # 批量查询：提取所有supplier_company_id和hub_company_id
    all_supplier_cids = list(set(p[0] for p in valid_pairs))
    all_hub_cids = list(set(p[1] for p in valid_pairs))

    # 一次性过滤出所有相关的关系记录
    relevant_rels = rels_batch[
        rels_batch['source_company_id'].isin(all_supplier_cids) &
        rels_batch['target_company_id'].isin(all_hub_cids)
    ]

    print(f"    → 匹配到的关系记录: {len(relevant_rels):,}")

    # 在内存中构建索引：{(source_cid, target_cid): [records]}
    relationship_index = {}
    for _, row in relevant_rels.iterrows():
        source_cid = row['source_company_id']
        target_cid = row['target_company_id']
        key = (source_cid, target_cid)

        if key not in relationship_index:
            relationship_index[key] = []
        relationship_index[key].append(row)

    # 为每个(hub_id, supplier_id)对生成relationship_data
    results = {}
    matched_count = 0

    for pair, entity_pairs in pair_to_entities.items():
        supplier_cid, hub_cid = pair

        # 查找该对的关系记录
        records = relationship_index.get(pair, [])

        for hub_id, supplier_id in entity_pairs:
            key = (hub_id, supplier_id)

            if len(records) == 0:
                # 没有关系数据
                results[key] = {}
                continue

            # 优先查找CUSTOMER类型
            customer_rels = [r for r in records if r.get('rel_type') == 'CUSTOMER']

            if len(customer_rels) > 0:
                # 取最新的CUSTOMER关系
                latest_rel = max(customer_rels, key=lambda r: r['start_'])
                results[key] = {
                    'revenue_percent': _convert_for_json(latest_rel.get('revenue_percent')),
                    'start_': _convert_for_json(latest_rel.get('start_')),
                    'relationship_source': f'hub_specific_{hub_id}'
                }
                matched_count += 1
            else:
                # Fallback: 取任意类型的最新关系
                latest_rel = max(records, key=lambda r: r['start_'])
                results[key] = {
                    'revenue_percent': _convert_for_json(latest_rel.get('revenue_percent')),
                    'start_': _convert_for_json(latest_rel.get('start_')),
                    'rel_type': latest_rel.get('rel_type'),
                    'relationship_source': f'hub_specific_other_{hub_id}'
                }
                matched_count += 1

    print(f"    ✓ 成功匹配 {matched_count}/{len(hub_candidate_info)} 个关系")

    return results


# ============================================================================
# 7. 批量查询：业务段数据
# ============================================================================

def batch_get_segment_data(
    entity_ids: List[str],
    basic_info_dict: Dict[str, Dict],
    cutoff_date: pd.Timestamp,
    all_dfs: Dict[str, pd.DataFrame]
) -> Dict[str, List[str]]:
    """批量获取业务段数据"""
    print("  [批量查询] 获取业务段数据...")

    segment_dict = {entity_id: [] for entity_id in entity_ids}

    df_seg = all_dfs.get('segment_summary')
    df_cf = all_dfs.get('company_factset')

    if df_seg is None or df_cf is None:
        return segment_dict

    # 创建entity_id到company_id的映射
    id_map = df_cf[['fs_entity_id', 'company_id']].copy()
    id_map['company_id'] = pd.to_numeric(id_map['company_id'], errors='coerce').astype('Int64')
    id_map = id_map.dropna(subset=['company_id']).drop_duplicates('fs_entity_id', keep='first')

    # 过滤出我们需要的entities
    id_map_filtered = id_map[id_map['fs_entity_id'].isin(entity_ids)]

    if id_map_filtered.empty:
        return segment_dict

    # 预处理segment表
    df_seg = df_seg.copy()
    df_seg['company_id'] = pd.to_numeric(df_seg['company_id'], errors='coerce').astype('Int64')
    df_seg = df_seg.dropna(subset=['company_id'])
    df_seg['period_end_date'] = pd.to_datetime(df_seg['period_end_date'], errors='coerce')

    # 批量join
    merged = pd.merge(df_seg, id_map_filtered, on='company_id', how='inner')
    merged = merged[merged['period_end_date'] <= cutoff_date]

    # 为每个entity提取top segments
    for entity_id in entity_ids:
        entity_segments = merged[merged['fs_entity_id'] == entity_id]
        if not entity_segments.empty:
            latest_segments = entity_segments.sort_values('period_end_date', ascending=False)
            top_level = latest_segments[latest_segments['depth'] == 1]
            segment_dict[entity_id] = list(top_level['path'].dropna().unique())

    success_count = sum(1 for v in segment_dict.values() if v)
    print(f"    ✓ 成功获取 {success_count}/{len(entity_ids)} 个公司的业务段数据")
    return segment_dict


# ============================================================================
# 8. 批量查询：行业信息
# ============================================================================

def batch_get_industry_info(
    entity_ids: List[str],
    basic_info_dict: Dict[str, Dict],
    all_dfs: Dict[str, pd.DataFrame]
) -> Dict[str, str]:
    """批量获取行业信息"""
    print("  [批量查询] 获取行业信息...")

    industry_dict = {}

    df_seg = all_dfs.get('segment_summary')
    df_cf = all_dfs.get('company_factset')
    df_se = all_dfs.get('standard_entity')
    df_sic = all_dfs.get('sic_map')

    # 方法1: 从segment_summary获取
    if df_seg is not None and df_cf is not None:
        seg_batch = df_seg[df_seg['company_id'].isin([
            info.get('company_id') for info in basic_info_dict.values()
            if info.get('company_id')
        ])].copy()

        if not seg_batch.empty:
            seg_batch = seg_batch.sort_values('report_end_date', ascending=False)
            seg_batch = seg_batch.drop_duplicates('company_id', keep='first')

            cf_map = df_cf[df_cf['fs_entity_id'].isin(entity_ids)][
                ['fs_entity_id', 'company_id']
            ].drop_duplicates()

            seg_with_entity = pd.merge(seg_batch, cf_map, on='company_id', how='inner')

            for _, row in seg_with_entity.iterrows():
                entity_id = row['fs_entity_id']
                name = row.get('name')
                if pd.notna(name) and name != '':
                    industry_dict[entity_id] = str(name)

    # 方法2: 从standard_entity的primary_sic_name获取
    if df_se is not None:
        se_batch = df_se[df_se['factset_entity_id'].isin(entity_ids)].copy()
        for _, row in se_batch.iterrows():
            entity_id = row['factset_entity_id']
            if entity_id in industry_dict:
                continue

            primary_sic_name = row.get('primary_sic_name')
            if pd.notna(primary_sic_name) and primary_sic_name != '' and primary_sic_name != 'Unknown':
                industry_dict[entity_id] = str(primary_sic_name)
                continue

            # 方法3: 从sector_code映射
            if df_sic is not None:
                sector_code = row.get('sector_code')
                if pd.notna(sector_code) and sector_code != '':
                    sector_code_str = str(int(sector_code))
                    sic_mapping = df_sic[df_sic['sic_code'].astype(str) == sector_code_str]
                    if not sic_mapping.empty:
                        industry_dict[entity_id] = str(sic_mapping.iloc[0]['sic_desc'])

    # 填充缺失
    for entity_id in entity_ids:
        if entity_id not in industry_dict:
            industry_dict[entity_id] = 'Unknown'

    print(f"    ✓ 获取了 {len(industry_dict)} 个公司的行业信息")
    return industry_dict


# ============================================================================
# 9. 批量构建Hub无关的Profiles
# ============================================================================

def batch_get_hub_independent_profiles(
    entity_ids: List[str],
    all_dfs: Dict[str, pd.DataFrame],
    cutoff_date: pd.Timestamp
) -> Dict[str, Dict]:
    """
    批量准备所有公司的Hub无关数据画像

    Hub无关的Profile包括：
    - basic_info（基本信息）
    - financial_data（财务数据）
    - segment_data（业务段数据）
    - industry（行业信息）

    不包括relationship_data（需要按hub单独查询）

    Args:
        entity_ids: 实体ID列表
        all_dfs: 所有数据框
        cutoff_date: 截止日期

    Returns:
        字典 {entity_id: hub_independent_profile}
    """
    print(f"\n--- [批量处理] 开始准备 {len(entity_ids)} 个公司的Hub无关数据画像 ---")
    print(f"  截止日期: {cutoff_date.strftime('%Y-%m-%d')}")

    # 1. 批量获取基本信息
    basic_info_dict = batch_get_basic_info(entity_ids, all_dfs)

    # 2. 批量获取FSYM_ID映射
    entity_fsym_map = batch_get_fsym_ids(entity_ids, all_dfs)

    # 3. 批量获取财务数据
    financial_fields = ['FF_SALES', 'FF_NET_MGN', 'FF_DEBT_EQ', 'FF_ROIC', 'FF_CURR_RATIO']
    financial_data_dict = batch_get_financial_data(
        entity_ids, entity_fsym_map, cutoff_date, all_dfs, financial_fields
    )

    # 4. 批量获取业务段数据
    segment_dict = batch_get_segment_data(
        entity_ids, basic_info_dict, cutoff_date, all_dfs
    )

    # 5. 批量获取行业信息
    industry_dict = batch_get_industry_info(
        entity_ids, basic_info_dict, all_dfs
    )

    # 6. 组装Hub无关的profiles
    print("\n  [组装] 构建Hub无关画像...")
    hub_independent_profiles = {}

    for entity_id in tqdm(entity_ids, desc="Assembling hub-independent profiles"):
        basic_info = basic_info_dict.get(entity_id, {})
        financial_data = financial_data_dict.get(entity_id, {})
        top_segments = segment_dict.get(entity_id, [])
        industry = industry_dict.get(entity_id, 'Unknown')

        profile = {
            **basic_info,
            "entity_id": entity_id,
            "factset_entity_id": entity_id,
            "company_name": basic_info.get('entity_proper_name', f'Unknown Entity {entity_id}'),
            "company_id": basic_info.get('company_id'),
            "industry": industry,
            "country": basic_info.get('iso_country', 'Unknown'),
            "cutoff_date": cutoff_date.strftime('%Y-%m-%d'),
            "financial_data": financial_data,
            "top_segments": top_segments
        }

        hub_independent_profiles[entity_id] = profile

    # 统计
    complete_profiles = sum(
        1 for p in hub_independent_profiles.values()
        if 'financial_data_error' not in p.get('financial_data', {})
    )
    partial_profiles = sum(
        1 for p in hub_independent_profiles.values()
        if 'financial_data_error' in p.get('financial_data', {})
    )

    print(f"\n--- Hub无关画像批量处理完成 ---")
    print(f"  总画像数: {len(hub_independent_profiles):,}")
    print(f"  完整画像: {complete_profiles:,}")
    print(f"  部分画像: {partial_profiles:,}")

    return hub_independent_profiles


# ============================================================================
# 10. 主函数：批量构建所有profiles（已弃用，保留兼容）
# ============================================================================

def batch_prepare_company_profiles(
    entity_ids: List[str],
    all_dfs: Dict[str, pd.DataFrame],
    cutoff_date: pd.Timestamp
) -> Dict[str, Dict]:
    """批量准备所有公司的数据画像"""
    print(f"\n--- [批量处理] 开始准备 {len(entity_ids)} 个公司的数据画像 ---")
    print(f"  截止日期: {cutoff_date.strftime('%Y-%m-%d')}")

    # 1. 批量获取基本信息
    basic_info_dict = batch_get_basic_info(entity_ids, all_dfs)

    # 2. 批量获取FSYM_ID映射
    entity_fsym_map = batch_get_fsym_ids(entity_ids, all_dfs)

    # 3. 批量获取财务数据
    financial_fields = ['FF_SALES', 'FF_NET_MGN', 'FF_DEBT_EQ', 'FF_ROIC', 'FF_CURR_RATIO']
    financial_data_dict = batch_get_financial_data(
        entity_ids, entity_fsym_map, cutoff_date, all_dfs, financial_fields
    )

    # 4. 批量获取关系数据
    relationship_dict = batch_get_relationship_data(
        entity_ids, basic_info_dict, cutoff_date, all_dfs
    )

    # 5. 批量获取业务段数据
    segment_dict = batch_get_segment_data(
        entity_ids, basic_info_dict, cutoff_date, all_dfs
    )

    # 6. 批量获取行业信息
    industry_dict = batch_get_industry_info(
        entity_ids, basic_info_dict, all_dfs
    )

    # 7. 组装所有profiles
    print("\n  [组装] 构建完整画像...")
    company_profiles = {}

    for entity_id in tqdm(entity_ids, desc="Assembling profiles"):
        basic_info = basic_info_dict.get(entity_id, {})
        financial_data = financial_data_dict.get(entity_id, {})
        relationship_data = relationship_dict.get(entity_id, {})
        top_segments = segment_dict.get(entity_id, [])
        industry = industry_dict.get(entity_id, 'Unknown')

        profile = {
            **basic_info,
            "entity_id": entity_id,
            "factset_entity_id": entity_id,
            "company_name": basic_info.get('entity_proper_name', f'Unknown Entity {entity_id}'),
            "industry": industry,
            "country": basic_info.get('iso_country', 'Unknown'),
            "cutoff_date": cutoff_date.strftime('%Y-%m-%d'),
            "financial_data": financial_data,
            **relationship_data,
            "top_segments": top_segments
        }

        company_profiles[entity_id] = profile

    # 统计
    complete_profiles = sum(
        1 for p in company_profiles.values()
        if 'financial_data_error' not in p.get('financial_data', {})
    )
    partial_profiles = sum(
        1 for p in company_profiles.values()
        if 'financial_data_error' in p.get('financial_data', {})
    )

    print(f"\n--- 批量处理完成 ---")
    print(f"  总画像数: {len(company_profiles):,}")
    print(f"  完整画像: {complete_profiles:,}")
    print(f"  部分画像: {partial_profiles:,}")

    return company_profiles


# ============================================================================
# 10. 主流程
# ============================================================================

def process_single_year(
    year: int,
    stage1_year_data: Dict,
    all_dfs: Dict[str, pd.DataFrame]
) -> Dict:
    """
    处理单个年份（新版：支持按Hub分组）

    Args:
        year: 年份
        stage1_year_data: Stage1的输出数据，包含：
            - hub_nodes: Hub节点列表
            - hub_candidate_pools: {hub_id: candidate_df}
            - all_nodes_need_profile: 所有需要画像的节点
        all_dfs: 所有数据框

    Returns:
        {
            'year': year,
            'cutoff_date': str,
            'hub_profiles': {hub_id: {supplier_id: profile}},
            'hub_independent_profiles': {entity_id: profile},
            'statistics': {...}
        }
    """
    print(f"\n{'='*80}")
    print(f"处理 {year} 年")
    print(f"{'='*80}")

    cutoff_date = pd.Timestamp(f"{year - 1}-12-31")

    # 获取Stage1数据
    hub_nodes = stage1_year_data['hub_nodes']
    hub_candidate_pools = stage1_year_data['hub_candidate_pools']
    all_nodes_need_profile = stage1_year_data['all_nodes_need_profile']

    print(f"\n[阶段1/2] 批量生成Hub无关画像")
    print(f"  节点总数: {len(all_nodes_need_profile):,}")
    print(f"  Hub数量: {len(hub_nodes):,}")

    # Step 1: 批量获取所有节点的Hub无关profile
    hub_independent_profiles = batch_get_hub_independent_profiles(
        all_nodes_need_profile,
        all_dfs,
        cutoff_date
    )

    print(f"\n[阶段2/2] 按Hub补充关系数据（批量查询优化）")

    # Step 2.1: 获取当年有效的company_id映射（选项2实现）
    print(f"\n  → 获取{year}年有效的company_id映射...")
    year_company_id_map = batch_get_company_id_for_year(
        all_nodes_need_profile,
        year,
        all_dfs
    )

    # Step 2.2: 收集所有需要查询的(hub, supplier)对信息
    hub_candidate_info_list = []
    hub_supplier_mapping = {}  # {hub_id: [(supplier_id, source_label), ...]}

    for hub_id in hub_nodes:
        if hub_id not in hub_candidate_pools:
            continue

        candidate_df = hub_candidate_pools[hub_id]
        hub_supplier_mapping[hub_id] = []

        # 使用当年有效的company_id映射
        hub_company_id = year_company_id_map.get(hub_id)

        if hub_company_id is None:
            print(f"  ⚠ Hub {hub_id} 在{year}年没有有效的company_id映射，跳过")
            continue

        # 收集该hub的所有候选者信息
        for _, row in candidate_df.iterrows():
            supplier_id = row['factset_entity_id']
            source_label = row['source_label']

            if supplier_id not in hub_independent_profiles:
                continue

            # 使用当年有效的company_id映射
            supplier_company_id = year_company_id_map.get(supplier_id)

            if supplier_company_id is None:
                continue

            # 添加到查询列表
            hub_candidate_info_list.append({
                'hub_id': hub_id,
                'supplier_id': supplier_id,
                'hub_company_id': hub_company_id,
                'supplier_company_id': supplier_company_id
            })

            # 保存映射关系
            hub_supplier_mapping[hub_id].append((supplier_id, source_label))

    print(f"  → 总共需要查询 {len(hub_candidate_info_list):,} 个Hub-Supplier对")

    # Step 2.3: 批量查询所有关系数据（使用cutoff_date过滤关系时间）
    relationship_results = batch_get_hub_specific_relationships(
        hub_candidate_info_list,
        cutoff_date,
        all_dfs
    )

    # Step 2.4: 组装完整的hub_profiles
    hub_profiles = {}
    total_hub_specific_profiles = 0

    for hub_id, supplier_list in hub_supplier_mapping.items():
        hub_profiles[hub_id] = {}

        for supplier_id, source_label in supplier_list:
            # 复制hub无关的基础profile
            profile = hub_independent_profiles[supplier_id].copy()

            # 添加hub特定的关系数据
            relationship_key = (hub_id, supplier_id)
            relationship_data = relationship_results.get(relationship_key, {})
            profile.update(relationship_data)

            # 添加source_label和hub_id
            profile['source_label'] = source_label
            profile['hub_id'] = hub_id

            hub_profiles[hub_id][supplier_id] = profile
            total_hub_specific_profiles += 1

    # 统计
    complete_profiles = sum(
        1 for p in hub_independent_profiles.values()
        if 'financial_data_error' not in p.get('financial_data', {})
    )
    partial_profiles = sum(
        1 for p in hub_independent_profiles.values()
        if 'financial_data_error' in p.get('financial_data', {})
    )

    print(f"\n{'='*80}")
    print(f"{year} 年处理完成")
    print(f"  Hub无关画像: {len(hub_independent_profiles):,}")
    print(f"  Hub数量: {len(hub_profiles):,}")
    print(f"  Hub特定画像总数: {total_hub_specific_profiles:,}")
    print(f"{'='*80}")

    return {
        'year': year,
        'cutoff_date': cutoff_date.strftime('%Y-%m-%d'),
        'hub_profiles': hub_profiles,  # 新增：按hub分组的profiles
        'hub_independent_profiles': hub_independent_profiles,  # 新增：hub无关的profiles
        'statistics': {
            'total_nodes': len(all_nodes_need_profile),
            'hub_count': len(hub_nodes),
            'hub_independent_profiles': len(hub_independent_profiles),
            'complete_profiles': complete_profiles,
            'partial_profiles': partial_profiles,
            'hub_specific_profiles': total_hub_specific_profiles
        }
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Stage 2: 生成公司数据画像（批量查询优化版）')
    parser.add_argument('--stage1_file', type=str, default=None)
    parser.add_argument('--result_dir', type=str, default=RESULT_DIR)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_year', type=int, default=2020)

    args = parser.parse_args()

    print("=" * 80)
    print("Stage 2: 公司画像生成 (批量查询优化版)")
    if args.debug:
        print("【DEBUG模式】")
    print("=" * 80)

    # Debug模式
    if args.debug:
        print(f"\n🔍 Debug模式：提取苹果公司 (000C7F-E) 画像")
        print(f"   年份: {args.debug_year}, 截止: {args.debug_year-1}-12-31")

        all_dfs = load_all_dataframes()
        apple_id = "000C7F-E"
        cutoff_date = pd.Timestamp(f"{args.debug_year - 1}-12-31")

        profiles = batch_prepare_company_profiles([apple_id], all_dfs, cutoff_date)
        profile = profiles[apple_id]

        print(f"\n{'='*80}")
        print("苹果公司画像")
        print(f"{'='*80}")

        import json
        print(json.dumps(profile, indent=2, default=str, ensure_ascii=False))

        output_file = os.path.join(args.result_dir, 'debug_apple_profile_batch.json')
        os.makedirs(args.result_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2, default=str, ensure_ascii=False)

        print(f"\n{'='*80}")
        print(f"✓ Debug完成！画像已保存到: {output_file}")
        print(f"{'='*80}")
        return

    # 正常模式
    if args.stage1_file:
        stage1_file = args.stage1_file
    else:
        stage1_files = list(Path(STAGE1_RESULT_DIR).glob("stage1_candidate_pools_*.pkl"))
        if not stage1_files:
            print("❌ 未找到Stage1输出文件")
            return
        stage1_file = str(sorted(stage1_files)[-1])

    print(f"Stage1文件: {stage1_file}")

    with open(stage1_file, 'rb') as f:
        stage1_results = pickle.load(f)

    years = list(stage1_results.keys())
    print(f"✓ Stage1包含 {len(years)} 个年份: {years}")

    all_dfs = load_all_dataframes()

    all_year_results = {}
    for year in years:
        try:
            stage1_year_data = stage1_results[year]

            # 调用新版process_single_year，传递完整的stage1_year_data
            year_result = process_single_year(year, stage1_year_data, all_dfs)
            all_year_results[year] = year_result

        except Exception as e:
            print(f"\n❌ {year} 年处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    stage1_basename = os.path.basename(stage1_file)
    hub_pct_match = stage1_basename.split('_hub')[-1].split('pct')[0]

    output_file = os.path.join(args.result_dir, f'stage2_company_profiles_hub{hub_pct_match}pct.pkl')

    print(f"\n{'='*80}")
    print(f"保存结果到: {output_file}")

    with open(output_file, 'wb') as f:
        pickle.dump(all_year_results, f)

    print(f"✓ Stage 2 完成！")

    print(f"\n{'='*80}")
    print("汇总统计")
    print(f"{'='*80}")
    print(f"成功处理年份: {len(all_year_results)}/{len(years)}\n")

    print("年份 | Hub无关 | Hub数 | Hub特定 | 完整 | 部分")
    print("-" * 80)
    for year, result in sorted(all_year_results.items()):
        stats = result['statistics']
        print(f"{year} | {stats['hub_independent_profiles']:>8,} | {stats['hub_count']:>6,} | "
              f"{stats['hub_specific_profiles']:>8,} | {stats['complete_profiles']:>4,} | "
              f"{stats['partial_profiles']:>4,}")


if __name__ == "__main__":
    main()