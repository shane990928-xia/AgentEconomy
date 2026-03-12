#!/usr/bin/env python3
"""
Stage 2: Company Profile Builder (Evolution Mode)
==================================================

功能：为指定年份构建所有公司的数据画像（Hub无关）

输入：
- stage1_candidate_pools_{year}/ (候选池文件夹)
- stage1_metadata_{year}.json (元数据)
- FactSet全局数据

输出：
- stage2_profiles_{year}.pkl
  格式：{
      'year': year,
      'cutoff_date': str,
      'hub_profiles': {hub_id: {supplier_id: profile}},
      'hub_independent_profiles': {entity_id: profile},
      'statistics': {...}
  }

Profile包含：
- Hub无关部分：基本信息、财务数据、业务段数据、行业信息
- Hub相关部分（仅在hub_profiles中）：relationship_data（与hub的关系）

依赖关系：
- ⚠️ 需要Stage 1的候选池输出

参考：
- network/code/stage2_profile_builder.py: 批量查询优化和hub特定关系
- network_continuous_simulation.py: parallel_prepare_company_profiles()
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
import argparse
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 全局配置
# ============================================================================

RESULT_DIR = "../result"
FACTSET_DATA_PATH = "/root/tmp/supplier/experiment/data/factset"

# ============================================================================
# 辅助函数：JSON转换
# ============================================================================

def _convert_for_json(value):
    """JSON转换辅助函数"""
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.strftime('%Y-%m-%d')
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


# ============================================================================
# 1. FactSet数据加载
# ============================================================================

def load_all_dataframes() -> Dict[str, pd.DataFrame]:
    """加载所有必需的FactSet数据集"""
    print("  → 加载FactSet全局数据...")

    datasets = {
        'standard_entity': os.path.join(FACTSET_DATA_PATH, 'data', 'standard_entity.parquet'),
        'company_factset': os.path.join(FACTSET_DATA_PATH, 'data', 'company_factset.parquet'),
        'company': os.path.join(FACTSET_DATA_PATH, 'data', 'company.parquet'),
        'security_entity_map': os.path.join(FACTSET_DATA_PATH, 'data', 'security_entity_map.parquet'),
        'own_basic': os.path.join(FACTSET_DATA_PATH, 'factset_own', 'own_basic.parquet'),
        'ff_int_qf': os.path.join(FACTSET_DATA_PATH, 'data', 'ff_int_qf.parquet'),
        'ff_usc_qf': os.path.join(FACTSET_DATA_PATH, 'data', 'ff_usc_qf.parquet'),
        'segment_summary': os.path.join(FACTSET_DATA_PATH, 'data', 'Business Segment Exposure-Summary.parquet'),
        'sic_map': os.path.join(FACTSET_DATA_PATH, 'data', 'sic_map.parquet'),
    }

    all_dataframes = {}

    for name, path in datasets.items():
        try:
            if os.path.exists(path):
                df = pd.read_parquet(path)
                all_dataframes[name] = df
                print(f"    ✓ {name}: {len(df):,} 行")
            else:
                print(f"    ⚠ {name}: 文件不存在，跳过")
                all_dataframes[name] = pd.DataFrame()
        except Exception as e:
            print(f"    ✗ {name}: 加载失败 - {e}")
            all_dataframes[name] = pd.DataFrame()

    return all_dataframes


# ============================================================================
# 2. 批量查询：基本信息
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
# 3. 批量查询：FSYM_ID映射
# ============================================================================

@np.vectorize
def _fast_split(s: str) -> str:
    """快速分割FSYM ID"""
    if isinstance(s, str) and '-' in s:
        return '-'.join(s.split('-')[:2])
    return s


def batch_get_fsym_ids(
    entity_ids: List[str],
    all_dfs: Dict[str, pd.DataFrame]
) -> Dict[str, set]:
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

    # 方法3: 从财务表查找
    for ff_dataset in ['ff_usc_qf', 'ff_int_qf']:
        df = all_dfs.get(ff_dataset)
        if df is None:
            continue

        # 通过FACTSET_ENTITY_ID直接查找
        if 'FACTSET_ENTITY_ID' in df.columns:
            direct_matches = df[df['FACTSET_ENTITY_ID'].isin(entity_ids)][
                ['FACTSET_ENTITY_ID', 'FSYM_ID']
            ].drop_duplicates()
            for _, row in direct_matches.iterrows():
                entity_id = row['FACTSET_ENTITY_ID']
                fsym_id = row['FSYM_ID']
                if pd.notna(fsym_id):
                    entity_fsym_map[entity_id].add(fsym_id)

    total_fsym = sum(len(fsyms) for fsyms in entity_fsym_map.values())
    print(f"    ✓ 获取了 {total_fsym} 个FSYM_ID (平均 {total_fsym/len(entity_ids):.1f} 个/公司)")
    return entity_fsym_map


# ============================================================================
# 4. 批量查询：财务数据
# ============================================================================

def batch_get_financial_data(
    entity_ids: List[str],
    entity_fsym_map: Dict[str, set],
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
# 5. 批量查询：业务段数据
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
# 6. 批量查询：行业信息
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
# 7. 批量查询：基于年份的Company ID映射
# ============================================================================

def batch_get_company_id_for_year(
    entity_ids: List[str],
    year: int,
    all_dfs: Dict[str, pd.DataFrame]
) -> Dict[str, int]:
    """
    批量获取指定年份有效的entity_id -> company_id映射

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
# 8. 批量查询：Hub特定关系数据
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
    if df_rels is None:
        print("    ⚠ factset_revere_relationship表不存在")
        # 加载relationship数据
        rel_path = os.path.join(FACTSET_DATA_PATH, 'data', 'factset_revere_relationship.parquet')
        if os.path.exists(rel_path):
            df_rels = pd.read_parquet(rel_path)
            all_dfs['factset_revere_relationship'] = df_rels
            print(f"    ✓ 加载了 factset_revere_relationship: {len(df_rels):,} 行")
        else:
            print("    ⚠ 找不到factset_revere_relationship文件，跳过关系数据查询")
            return {}

    if len(hub_candidate_info) == 0:
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
# 9. 批量构建Hub无关的Profiles
# ============================================================================

def batch_build_profiles(
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

    Args:
        entity_ids: 实体ID列表
        all_dfs: 所有数据框
        cutoff_date: 截止日期

    Returns:
        字典 {entity_id: profile}
    """
    print(f"\n  → 开始批量构建 {len(entity_ids)} 个公司的数据画像")
    print(f"  → 截止日期: {cutoff_date.strftime('%Y-%m-%d')}")

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

    # 6. 组装profiles
    print("\n  [组装] 构建完整画像...")
    profiles = {}

    for entity_id in tqdm(entity_ids, desc="  组装画像"):
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

        profiles[entity_id] = profile

    # 统计
    complete_profiles = sum(
        1 for p in profiles.values()
        if 'financial_data_error' not in p.get('financial_data', {})
    )
    partial_profiles = sum(
        1 for p in profiles.values()
        if 'financial_data_error' in p.get('financial_data', {})
    )

    print(f"\n  ✓ 画像批量处理完成")
    print(f"    总画像数: {len(profiles):,}")
    print(f"    完整画像: {complete_profiles:,}")
    print(f"    部分画像: {partial_profiles:,}")

    return profiles


# ============================================================================
# 10. 主处理流程
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Stage 2: 公司画像准备（演化模式）'
    )
    parser.add_argument(
        '--year',
        type=int,
        required=True,
        help='处理年份（必需）'
    )
    parser.add_argument(
        '--result_dir',
        type=str,
        default=RESULT_DIR,
        help='结果保存目录'
    )

    args = parser.parse_args()

    # 打印配置
    print("=" * 80)
    print("Stage 2: 公司画像准备（演化模式）")
    print("=" * 80)
    print(f"处理年份: {args.year}")
    print(f"结果目录: {args.result_dir}")
    print("=" * 80)

    # 步骤1: 检查Stage 1输出
    print(f"\n[1/5] 检查Stage 1输出")

    candidate_pools_dir = os.path.join(
        args.result_dir,
        f"stage1_candidate_pools_{args.year}"
    )

    metadata_file = os.path.join(
        args.result_dir,
        f"stage1_metadata_{args.year}.json"
    )

    if not os.path.exists(candidate_pools_dir):
        raise FileNotFoundError(
            f"找不到Stage 1候选池目录: {candidate_pools_dir}\n"
            f"请先运行: python stage1_candidate_pool.py --year {args.year}"
        )

    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"找不到Stage 1元数据: {metadata_file}")

    # 加载元数据
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    hub_nodes = metadata['decision_makers']  # Hub节点即核心决策者
    print(f"  ✓ 检测到 {len(hub_nodes)} 个Hub节点（核心决策者）")

    # 步骤2: 加载所有候选池并收集实体
    print(f"\n[2/5] 加载候选池并收集所有实体")

    all_entity_ids = set(hub_nodes)
    hub_candidate_pools = {}  # {hub_id: candidate_df}

    # 遍历所有候选池文件
    pool_files = list(Path(candidate_pools_dir).glob("*.pkl"))
    print(f"  → 加载 {len(pool_files)} 个候选池文件...")

    for pool_file in pool_files:
        decision_maker_id = pool_file.stem  # 文件名即decision_maker_id

        with open(pool_file, 'rb') as f:
            candidate_df = pickle.load(f)

        # 保存候选池
        hub_candidate_pools[decision_maker_id] = candidate_df

        # 收集候选者ID
        if 'factset_entity_id' in candidate_df.columns:
            candidates = candidate_df['factset_entity_id'].unique()
            all_entity_ids.update(candidates)

    all_entity_ids = sorted(list(all_entity_ids))
    print(f"  ✓ 总共需要构建 {len(all_entity_ids)} 个实体的画像")
    print(f"    - Hub节点: {len(hub_nodes)}")
    print(f"    - 候选供应商: {len(all_entity_ids) - len(hub_nodes)}")

    # 步骤3: 构建Hub无关画像
    print(f"\n[3/5] 批量构建Hub无关画像")

    all_dfs = load_all_dataframes()

    # 计算截止日期（前一年12月31日）
    cutoff_date = pd.Timestamp(f"{args.year - 1}-12-31")

    # 批量构建Hub无关画像
    hub_independent_profiles = batch_build_profiles(all_entity_ids, all_dfs, cutoff_date)

    # 步骤4: 补充Hub特定关系数据
    print(f"\n[4/5] 批量补充Hub特定关系数据")

    # 获取当年有效的company_id映射
    print(f"\n  → 获取{args.year}年有效的company_id映射...")
    year_company_id_map = batch_get_company_id_for_year(
        all_entity_ids,
        args.year,
        all_dfs
    )

    # 收集所有需要查询的(hub, supplier)对信息
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
            print(f"  ⚠ Hub {hub_id} 在{args.year}年没有有效的company_id映射，跳过")
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

    # 批量查询所有关系数据
    relationship_results = batch_get_hub_specific_relationships(
        hub_candidate_info_list,
        cutoff_date,
        all_dfs
    )

    # 组装完整的hub_profiles
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

    # 步骤5: 保存结果
    print(f"\n[5/5] 保存结果")

    output_file = os.path.join(
        args.result_dir,
        f"stage2_profiles_{args.year}.pkl"
    )

    os.makedirs(args.result_dir, exist_ok=True)

    result = {
        'year': args.year,
        'cutoff_date': cutoff_date.strftime('%Y-%m-%d'),
        'hub_profiles': hub_profiles,
        'hub_independent_profiles': hub_independent_profiles,
        'statistics': {
            'total_nodes': len(all_entity_ids),
            'hub_count': len(hub_nodes),
            'hub_independent_profiles': len(hub_independent_profiles),
            'complete_profiles': complete_profiles,
            'partial_profiles': partial_profiles,
            'hub_specific_profiles': total_hub_specific_profiles
        }
    }

    with open(output_file, 'wb') as f:
        pickle.dump(result, f)

    print(f"  ✓ 结果已保存到: {output_file}")

    # 打印完成信息
    print(f"\n{'='*80}")
    print(f"✓ Stage 2 完成！")
    print(f"{'='*80}")
    print(f"处理年份: {args.year}")
    print(f"截止日期: {cutoff_date.strftime('%Y-%m-%d')}")
    print(f"Hub无关画像: {len(hub_independent_profiles):,}")
    print(f"  - 完整画像: {complete_profiles:,}")
    print(f"  - 部分画像: {partial_profiles:,}")
    print(f"Hub数量: {len(hub_profiles):,}")
    print(f"Hub特定画像总数: {total_hub_specific_profiles:,}")
    print(f"\n{'='*80}")
    print(f"下一步: 运行 stage3a_llm_personality.py --year {args.year}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
