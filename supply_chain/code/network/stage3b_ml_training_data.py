#!/usr/bin/env python3
"""
Stage 3b: ML Training Data Extraction (重构版 - 直接查询FactSet)
=======================================

功能：从FactSet数据集直接提取机器学习训练数据

输入：
- Stage1: hub_nodes
- FactSet原始数据: optimal_network_{year-1}.parquet, 财务数据等

输出：
- stage3b_ml_training_data_*.pkl

数据逻辑（以year=2018为例）：
- 正样本：从2017年网络中查询hub的供应商，使用2016年底财务数据
- 负样本：从2017年所有公司中随机抽取等量噪声，使用2016年底财务数据
- 标签：是否是2017年网络中的供应商
- 用途：训练模型用t-1年数据预测t年供应关系

特征维度与supplier_baseline2.py对齐
"""

import pandas as pd
import numpy as np
import os
import pickle
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime
from tqdm import tqdm

# 导入stage2的数据加载和特征提取函数
sys.path.insert(0, os.path.dirname(__file__))
from stage2_profile_builder import (
    load_all_dataframes,
    batch_get_fsym_ids,
    batch_get_financial_data,
    batch_get_relationship_data,
    _convert_for_json
)

# ============================================================================
# 全局配置
# ============================================================================

STAGE1_RESULT_DIR = "../result"
RESULT_DIR = "../result"

# ============================================================================
# 辅助函数
# ============================================================================

def safe_float_conversion(value):
    """安全的浮点数转换"""
    if pd.isna(value):
        return np.nan
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def batch_query_suppliers_from_factset(
    hub_info_list: List[Dict],
    cutoff_date: pd.Timestamp,
    all_dfs: Dict[str, pd.DataFrame]
) -> Dict[str, Set[str]]:
    """
    批量从FactSet查询所有Hub的供应商（并行优化版）

    Args:
        hub_info_list: Hub信息列表，每个元素包含：
            {
                'hub_entity_id': str,
                'hub_company_id': int
            }
        cutoff_date: 截止日期（如2017-12-31）
        all_dfs: 所有数据框

    Returns:
        字典 {hub_entity_id: set(supplier_entity_ids)}
    """
    print(f"  [批量查询] 从FactSet查询 {len(hub_info_list)} 个Hub的供应商关系...")

    df_rels = all_dfs.get('factset_revere_relationship')
    df_cf = all_dfs.get('company_factset')

    if df_rels is None or df_cf is None:
        print(f"    ⚠ 关系表或映射表不存在，跳过")
        return {info['hub_entity_id']: set() for info in hub_info_list}

    # 确保时间列是日期类型
    df_rels = df_rels.copy()
    df_rels['start_'] = pd.to_datetime(df_rels['start_'], errors='coerce')

    # 收集所有Hub的company_id
    hub_company_ids = [info['hub_company_id'] for info in hub_info_list
                       if info['hub_company_id'] is not None]

    if len(hub_company_ids) == 0:
        print(f"    ⚠ 没有有效的Hub company_id")
        return {info['hub_entity_id']: set() for info in hub_info_list}

    print(f"    → 有效Hub数: {len(hub_company_ids)}")

    # 批量过滤：一次性获取所有Hub的供应商关系
    relevant_rels = df_rels[
        (df_rels['target_company_id'].isin(hub_company_ids)) &
        (df_rels['rel_type'] == 'CUSTOMER') &
        (df_rels['start_'] <= cutoff_date)
    ]

    print(f"    → 找到供应商关系记录: {len(relevant_rels):,}")

    if relevant_rels.empty:
        print(f"    ⚠ 未找到任何供应商关系")
        return {info['hub_entity_id']: set() for info in hub_info_list}

    # 构建索引：{hub_company_id: [supplier_company_ids]}
    hub_supplier_cid_map = {}
    for _, row in relevant_rels.iterrows():
        hub_cid = row['target_company_id']
        supplier_cid = row['source_company_id']

        if hub_cid not in hub_supplier_cid_map:
            hub_supplier_cid_map[hub_cid] = set()
        hub_supplier_cid_map[hub_cid].add(supplier_cid)

    # 收集所有supplier_company_ids
    all_supplier_cids = set()
    for supplier_cids in hub_supplier_cid_map.values():
        all_supplier_cids.update(supplier_cids)

    print(f"    → 唯一供应商company_id数: {len(all_supplier_cids):,}")

    # 批量映射：company_id -> entity_id
    df_cf = df_cf.copy()
    df_cf['start_'] = pd.to_datetime(df_cf['start_'], errors='coerce')
    df_cf['end_'] = pd.to_datetime(df_cf['end_'], errors='coerce')

    # 类型转换：company_factset表的company_id是object，需要转为int以匹配关系表
    df_cf['company_id'] = pd.to_numeric(df_cf['company_id'], errors='coerce').astype('Int64')

    # 筛选在cutoff_date对应年份有效的映射（与stage2逻辑一致）
    # 例如：cutoff_date = 2017-12-31，则检查映射是否在2017年有效
    cutoff_year = cutoff_date.year
    year_start = pd.Timestamp(f"{cutoff_year}-01-01")
    year_end = pd.Timestamp(f"{cutoff_year}-12-31")

    valid_mappings = df_cf[
        (df_cf['company_id'].isin(all_supplier_cids)) &
        (df_cf['start_'] <= year_end) &
        (df_cf['end_'] >= year_start)
    ]

    print(f"    → 有效映射记录: {len(valid_mappings):,}")

    # 构建company_id -> entity_id的映射（优先pri='Y'）
    cid_to_eid_map = {}

    # 先处理pri='Y'的记录
    primary_mappings = valid_mappings[valid_mappings['pri'] == 'Y']
    for _, row in primary_mappings.iterrows():
        cid = row['company_id']
        eid = row['fs_entity_id']
        if pd.notna(eid):
            cid_to_eid_map[cid] = eid

    # 填充没有pri='Y'的记录
    for _, row in valid_mappings.iterrows():
        cid = row['company_id']
        eid = row['fs_entity_id']
        if pd.notna(eid) and cid not in cid_to_eid_map:
            cid_to_eid_map[cid] = eid

    print(f"    → company_id -> entity_id 映射数: {len(cid_to_eid_map):,}")

    # 为每个Hub分配supplier_entity_ids
    results = {}
    hub_cid_to_eid = {info['hub_company_id']: info['hub_entity_id']
                      for info in hub_info_list}

    for hub_cid, hub_eid in hub_cid_to_eid.items():
        supplier_cids = hub_supplier_cid_map.get(hub_cid, set())

        # 将supplier_company_ids转换为entity_ids
        supplier_eids = set()
        for supplier_cid in supplier_cids:
            supplier_eid = cid_to_eid_map.get(supplier_cid)
            if supplier_eid:
                supplier_eids.add(supplier_eid)

        results[hub_eid] = supplier_eids

    # 填充没有找到供应商的Hub
    for info in hub_info_list:
        hub_eid = info['hub_entity_id']
        if hub_eid not in results:
            results[hub_eid] = set()

    matched_hubs = sum(1 for v in results.values() if len(v) > 0)
    print(f"    ✓ 成功查询 {matched_hubs}/{len(hub_info_list)} 个Hub的供应商")

    return results


def get_all_factset_entities(
    all_dfs: Dict[str, pd.DataFrame],
    exclude_entity_ids: Set[str] = None
) -> Set[str]:
    """
    获取FactSet中所有公司的entity_id（用于噪声抽样）

    Args:
        all_dfs: 所有数据框
        exclude_entity_ids: 需要排除的entity_id集合

    Returns:
        所有entity_id集合
    """
    df_se = all_dfs.get('standard_entity')

    if df_se is None:
        return set()

    all_entities = set(df_se['factset_entity_id'].dropna().unique())

    if exclude_entity_ids:
        all_entities = all_entities - exclude_entity_ids

    return all_entities


def sample_noise_from_factset(
    num_samples: int,
    all_entities: Set[str],
    exclude_entities: Set[str]
) -> Set[str]:
    """
    从FactSet所有公司中随机抽取噪声样本

    Args:
        num_samples: 需要抽取的样本数量
        all_entities: 所有可用entity（从FactSet）
        exclude_entities: 需要排除的entity（正样本+hub自己）

    Returns:
        噪声entity_id集合
    """
    candidate_pool = all_entities - exclude_entities

    if len(candidate_pool) == 0:
        return set()

    actual_num = min(num_samples, len(candidate_pool))
    noise_samples = set(np.random.choice(
        list(candidate_pool),
        size=actual_num,
        replace=False
    ))

    return noise_samples


# ============================================================================
# 特征提取
# ============================================================================

def extract_features_for_entity(
    entity_id: str,
    hub_id: str,
    is_actual_supplier: bool,
    entity_fsym_map: Dict[str, Set[str]],
    financial_data_dict: Dict[str, Dict],
    relationship_data_dict: Dict[str, Dict],
    year: int
) -> Dict:
    """
    为单个候选者提取特征（对齐supplier_baseline2.py）

    Args:
        entity_id: 候选者entity_id
        hub_id: Hub节点ID
        is_actual_supplier: 是否是真实供应商
        entity_fsym_map: FSYM_ID映射
        financial_data_dict: 财务数据字典
        relationship_data_dict: 关系数据字典
        year: 决策年份（例如2018，表示用2017年数据预测2018年关系）

    Returns:
        特征字典
    """
    # 标签
    label = 1 if is_actual_supplier else 0

    # 获取财务数据
    financial_data = financial_data_dict.get(entity_id, {})
    relationship_data = relationship_data_dict.get(entity_id, {})

    # 构建特征（对齐supplier_baseline2.py）
    features = {
        'supplier_entity_id': entity_id,
        'hub_entity_id': hub_id,
        'decision_year': year - 1,  # 使用t-1年数据
        'label': label,

        # 关系特征
        'revenue_percent': safe_float_conversion(relationship_data.get('revenue_percent')),

        # 财务特征
        'FF_SALES': safe_float_conversion(financial_data.get('FF_SALES')),
        'FF_NET_MGN': safe_float_conversion(financial_data.get('FF_NET_MGN')),
        'FF_DEBT_EQ': safe_float_conversion(financial_data.get('FF_DEBT_EQ')),
        'FF_CURR_RATIO': safe_float_conversion(financial_data.get('FF_CURR_RATIO')),
        'FF_ROIC': safe_float_conversion(financial_data.get('FF_ROIC')),

        # 元数据
        'has_fsym_id': len(entity_fsym_map.get(entity_id, set())) > 0,
        'has_financial_data': 'financial_data_error' not in financial_data,
        'has_relationship_data': bool(relationship_data)
    }

    return features


# ============================================================================
# 主流程
# ============================================================================

def process_single_year(
    year: int,
    stage1_data: Dict,
    all_dfs: Dict[str, pd.DataFrame]
) -> Dict:
    """
    处理单个年份（新逻辑：直接从FactSet查询）

    数据逻辑（以year=2018为例）：
    - Hub节点：从Stage1的2018年结果
    - 正样本：从FactSet查询hub在2017年底的供应商（factset_revere_relationship）
    - 负样本：从FactSet所有公司随机抽样，使用2016年底财务数据
    - 标签：是否是FactSet中2017年底的供应商

    Args:
        year: 年份（如2018）
        stage1_data: Stage1输出数据
        all_dfs: 所有FactSet数据框

    Returns:
        训练数据字典
    """
    print(f"\n{'='*80}")
    print(f"处理 {year} 年")
    print(f"{'='*80}")

    # 数据年份说明
    network_year = year - 1  # 2017
    relationship_cutoff = pd.Timestamp(f"{year - 1}-12-31")  # 2017-12-31（查询关系）
    financial_cutoff = pd.Timestamp(f"{year - 2}-12-31")  # 2016-12-31（财务数据）

    print(f"  数据逻辑:")
    print(f"    - Hub节点: {year}年 Stage1 结果")
    print(f"    - 供应商关系查询截止: {relationship_cutoff.strftime('%Y-%m-%d')} (FactSet)")
    print(f"    - 财务数据截止: {financial_cutoff.strftime('%Y-%m-%d')}")

    # 1. 获取Hub节点（从Stage1）
    hub_nodes = stage1_data['hub_nodes']
    print(f"\n[1/7] Hub节点数: {len(hub_nodes)}")

    # 2. 获取Hub的company_id映射（使用year-1年有效的映射）
    print(f"\n[2/7] 获取Hub的company_id映射 (基于{network_year}年)")
    from stage2_profile_builder import batch_get_company_id_for_year
    hub_company_id_map = batch_get_company_id_for_year(
        hub_nodes,
        network_year,
        all_dfs
    )
    print(f"  ✓ 成功映射 {len(hub_company_id_map)}/{len(hub_nodes)} 个Hub")

    # 3. 获取FactSet所有公司（用于噪声抽样）
    print(f"\n[3/7] 获取FactSet所有公司")
    all_factset_entities = get_all_factset_entities(all_dfs)
    print(f"  ✓ FactSet总公司数: {len(all_factset_entities):,}")

    # 4. 批量查询各Hub的供应商（并行优化）
    print(f"\n[4/7] 批量查询各Hub的供应商关系（从FactSet）")

    # 准备批量查询的输入
    hub_info_list = []
    for hub_id in hub_nodes:
        hub_company_id = hub_company_id_map.get(hub_id)
        if hub_company_id is not None:
            hub_info_list.append({
                'hub_entity_id': hub_id,
                'hub_company_id': hub_company_id
            })

    # 批量查询
    hub_suppliers_map = batch_query_suppliers_from_factset(
        hub_info_list,
        relationship_cutoff,
        all_dfs
    )

    # 5. 全局抽样负样本池（优化：减少查询节点总数）
    print(f"\n[5/7] 全局抽样负样本池")

    # 5.1 收集所有正样本（用于排除）
    all_positive_samples = set()
    for suppliers in hub_suppliers_map.values():
        all_positive_samples.update(suppliers)

    print(f"  → 总正样本数: {len(all_positive_samples):,}")

    # 5.2 找到最大供应商数量
    max_supplier_count = max(
        (len(suppliers) for suppliers in hub_suppliers_map.values()),
        default=0
    )
    print(f"  → 最大供应商数: {max_supplier_count}")

    if max_supplier_count == 0:
        print(f"  ⚠ 没有找到任何供应商，跳过该年份")
        return {
            'year': year,
            'training_data': pd.DataFrame(),
            'hub_stats': {},
            'statistics': {'error': 'No suppliers found'}
        }

    # 5.3 全局抽样负样本池
    exclude_entities = all_positive_samples | set(hub_nodes)
    global_noise_pool = sample_noise_from_factset(
        max_supplier_count,
        all_factset_entities,
        exclude_entities
    )
    global_noise_pool_list = list(global_noise_pool)

    print(f"  ✓ 全局负样本池大小: {len(global_noise_pool_list)}")

    # 5.4 为每个Hub从池中选取负样本
    print(f"\n  为各Hub分配负样本...")
    hub_candidate_info = {}  # {hub_id: {'suppliers': set, 'noise': set}}
    all_entities_to_query = set()

    for hub_id in hub_nodes:
        suppliers = hub_suppliers_map.get(hub_id, set())

        if len(suppliers) == 0:
            continue

        # 从全局池中随机选取等量负样本
        num_needed = min(len(suppliers), len(global_noise_pool_list))
        if num_needed > 0:
            noise = set(np.random.choice(
                global_noise_pool_list,
                size=num_needed,
                replace=False
            ))
        else:
            noise = set()

        hub_candidate_info[hub_id] = {
            'suppliers': suppliers,
            'noise': noise
        }

        # 收集所有需要查询的entity
        all_entities_to_query.update(suppliers)
        all_entities_to_query.update(noise)

    print(f"  ✓ 有效Hub数: {len(hub_candidate_info)}")
    print(f"  ✓ 需要查询画像的节点总数: {len(all_entities_to_query):,}")

    if len(hub_candidate_info) == 0:
        print(f"  ⚠ 没有有效的Hub-Supplier对，跳过该年份")
        return {
            'year': year,
            'training_data': pd.DataFrame(),
            'hub_stats': {},
            'statistics': {'error': 'No valid hub-supplier pairs'}
        }

    # 6. 批量查询FactSet数据（使用financial_cutoff）
    print(f"\n[6/7] 批量查询FactSet数据 (截止: {financial_cutoff.strftime('%Y-%m-%d')})")
    entity_list = list(all_entities_to_query)

    # 6.1 获取FSYM_ID映射
    entity_fsym_map = batch_get_fsym_ids(entity_list, all_dfs)

    # 6.2 获取财务数据
    financial_fields = ['FF_SALES', 'FF_NET_MGN', 'FF_DEBT_EQ', 'FF_ROIC', 'FF_CURR_RATIO']
    financial_data_dict = batch_get_financial_data(
        entity_list,
        entity_fsym_map,
        financial_cutoff,
        all_dfs,
        financial_fields
    )

    # 6.3 获取关系数据（注意：这里是Hub无关的关系数据）
    from stage2_profile_builder import batch_get_basic_info
    basic_info_dict = batch_get_basic_info(entity_list, all_dfs)
    relationship_data_dict = batch_get_relationship_data(
        entity_list,
        basic_info_dict,
        financial_cutoff,
        all_dfs
    )

    # 7. 为每个Hub提取训练特征
    print(f"\n[7/7] 提取训练特征")
    all_training_data = []
    hub_stats = {}

    for hub_id, candidates in tqdm(hub_candidate_info.items(), desc="Extracting features"):
        suppliers = candidates['suppliers']
        noise = candidates['noise']

        features_list = []

        # 正样本
        for supplier_id in suppliers:
            if supplier_id in all_entities_to_query:
                features = extract_features_for_entity(
                    supplier_id,
                    hub_id,
                    is_actual_supplier=True,
                    entity_fsym_map=entity_fsym_map,
                    financial_data_dict=financial_data_dict,
                    relationship_data_dict=relationship_data_dict,
                    year=year
                )
                features_list.append(features)

        # 负样本
        for noise_id in noise:
            if noise_id in all_entities_to_query:
                features = extract_features_for_entity(
                    noise_id,
                    hub_id,
                    is_actual_supplier=False,
                    entity_fsym_map=entity_fsym_map,
                    financial_data_dict=financial_data_dict,
                    relationship_data_dict=relationship_data_dict,
                    year=year
                )
                features_list.append(features)

        all_training_data.extend(features_list)

        # 统计
        num_positive = sum(1 for f in features_list if f['label'] == 1)
        num_negative = sum(1 for f in features_list if f['label'] == 0)

        hub_stats[hub_id] = {
            'total_samples': len(features_list),
            'positive_samples': num_positive,
            'negative_samples': num_negative
        }

    # 7. 转换为DataFrame
    print(f"\n[7/7] 构建训练数据DataFrame")
    training_df = pd.DataFrame(all_training_data)

    # 统计信息
    total_samples = len(training_df)
    positive_samples = (training_df['label'] == 1).sum() if total_samples > 0 else 0
    negative_samples = (training_df['label'] == 0).sum() if total_samples > 0 else 0

    print(f"\n训练数据统计:")
    print(f"  总样本数: {total_samples:,}")
    if total_samples > 0:
        print(f"  正样本数: {positive_samples:,} ({positive_samples/total_samples*100:.2f}%)")
        print(f"  负样本数: {negative_samples:,} ({negative_samples/total_samples*100:.2f}%)")

        # 数据质量统计
        has_financial = training_df['has_financial_data'].sum()
        has_relationship = training_df['has_relationship_data'].sum()
        print(f"  有财务数据: {has_financial:,} ({has_financial/total_samples*100:.2f}%)")
        print(f"  有关系数据: {has_relationship:,} ({has_relationship/total_samples*100:.2f}%)")

    return {
        'year': year,
        'training_data': training_df,
        'hub_stats': hub_stats,
        'statistics': {
            'total_samples': total_samples,
            'positive_samples': int(positive_samples),
            'negative_samples': int(negative_samples),
            'num_hubs': len(hub_stats)
        }
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Stage 3b: ML训练数据抽取（直接查询FactSet）')
    parser.add_argument('--stage1_file', type=str, default=None,
                       help='Stage1输出文件路径')
    parser.add_argument('--result_dir', type=str, default=RESULT_DIR)

    args = parser.parse_args()

    print("=" * 80)
    print("Stage 3b: ML训练数据抽取（直接查询FactSet）")
    print("=" * 80)

    # 查找Stage1输出文件
    if args.stage1_file:
        stage1_file = args.stage1_file
    else:
        stage1_files = list(Path(STAGE1_RESULT_DIR).glob("stage1_candidate_pools_*.pkl"))
        if not stage1_files:
            print("❌ 未找到Stage1输出文件")
            return
        stage1_file = str(sorted(stage1_files)[-1])

    print(f"Stage1文件: {stage1_file}")

    # 加载Stage1数据
    with open(stage1_file, 'rb') as f:
        stage1_results = pickle.load(f)

    years = sorted(stage1_results.keys())
    print(f"✓ Stage1包含 {len(years)} 个年份: {years}")

    # 加载FactSet数据集（一次性加载，所有年份复用）
    print(f"\n加载FactSet数据集...")
    all_dfs = load_all_dataframes()

    # 处理所有年份
    all_year_results = {}
    for year in years:
        try:
            year_result = process_single_year(
                year,
                stage1_results[year],
                all_dfs
            )
            all_year_results[year] = year_result

        except Exception as e:
            print(f"\n❌ {year} 年处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 保存结果
    stage1_basename = os.path.basename(stage1_file)
    hub_pct_match = stage1_basename.split('_hub')[-1].split('pct')[0] if '_hub' in stage1_basename else '20'
    noise_pct_match = stage1_basename.split('_noise')[-1].split('pct')[0] if '_noise' in stage1_basename else '100'

    output_file = os.path.join(
        args.result_dir,
        f'stage3b_ml_training_data_hub{hub_pct_match}pct_noise{noise_pct_match}pct.pkl'
    )

    print(f"\n{'='*80}")
    print(f"保存结果到: {output_file}")

    with open(output_file, 'wb') as f:
        pickle.dump(all_year_results, f)

    print(f"✓ Stage 3b 完成！")

    # 统计
    print(f"\n{'='*80}")
    print("汇总统计")
    print(f"{'='*80}")
    print(f"成功处理年份: {len(all_year_results)}/{len(years)}\n")

    print("年份 | 样本总数 | 正样本 | 负样本 | Hub数")
    print("-" * 80)
    for year, result in sorted(all_year_results.items()):
        stats = result['statistics']
        if 'error' not in stats:
            print(f"{year} | {stats['total_samples']:>8,} | {stats['positive_samples']:>6,} | "
                  f"{stats['negative_samples']:>7,} | {stats['num_hubs']:>5}")
        else:
            print(f"{year} | 跳过（{stats['error']}）")


if __name__ == "__main__":
    main()