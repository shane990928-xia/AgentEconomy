#!/usr/bin/env python3
"""
Stage 5c: Random Supplier Selection (Baseline)
================================================

功能：随机选择供应商（作为baseline对比）

输入：
- Stage1: hub_candidate_pools

输出：
- stage5c_random_selections_*.pkl
  {
    year: {
      'hub_selections': {
        'hub_id': {
          'selected_suppliers': [...],
          'statistics': {...}
        }
      }
    }
  }

随机策略：
- 对每个候选供应商，以50%概率随机选择
"""

import pandas as pd
import numpy as np
import os
import pickle
import argparse
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

# ============================================================================
# 全局配置
# ============================================================================

STAGE1_RESULT_DIR = "../result"
RESULT_DIR = "../result"

RANDOM_SELECTION_PROBABILITY = 0.5  # 50%概率选择每个供应商

# ============================================================================
# 随机选择
# ============================================================================

def random_select_suppliers(
    hub_id: str,
    candidate_pool: pd.DataFrame,
    random_seed: int = None
) -> Dict:
    """
    随机选择供应商

    Args:
        hub_id: Hub节点ID
        candidate_pool: 候选池DataFrame
        random_seed: 随机种子（可选）

    Returns:
        选择结果字典
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    selected_suppliers = []
    selection_details = {}

    for _, row in candidate_pool.iterrows():
        supplier_id = row['factset_entity_id']
        source_label = row['source_label']

        # 随机决策（50%概率，对齐network_test.py）
        prob = np.random.random()
        is_selected = prob < RANDOM_SELECTION_PROBABILITY  # < 0.5 则选择

        selection_details[supplier_id] = {
            'random_probability': float(prob),
            'is_selected': is_selected,
            'source_label': source_label
        }

        if is_selected:
            selected_suppliers.append(supplier_id)

    statistics = {
        'total_candidates': len(candidate_pool),
        'selected_count': len(selected_suppliers),
        'selection_rate': len(selected_suppliers) / len(candidate_pool) if len(candidate_pool) > 0 else 0.0
    }

    return {
        'selected_suppliers': selected_suppliers,
        'selection_details': selection_details,
        'statistics': statistics
    }


# ============================================================================
# 主流程
# ============================================================================

def process_single_year(
    year: int,
    stage1_data: Dict,
    random_seed: int = None
) -> Dict:
    """处理单个年份"""
    print(f"\n{'='*80}")
    print(f"处理 {year} 年")
    print(f"{'='*80}")

    # 获取数据
    hub_nodes = stage1_data['hub_nodes']
    hub_candidate_pools = stage1_data['hub_candidate_pools']

    print(f"  Hub节点数: {len(hub_nodes)}")

    # 为每个Hub执行随机选择
    all_hub_selections = {}

    for hub_id in tqdm(hub_nodes, desc="Random selecting"):
        try:
            # 检查候选池
            if hub_id not in hub_candidate_pools:
                continue

            candidate_pool = hub_candidate_pools[hub_id]

            # 执行随机选择
            selection_result = random_select_suppliers(
                hub_id,
                candidate_pool,
                random_seed=random_seed
            )

            all_hub_selections[hub_id] = selection_result

        except Exception as e:
            print(f"  [ERROR] {hub_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 统计
    total_hubs = len(hub_nodes)
    processed_hubs = len(all_hub_selections)
    total_selected = sum(len(s['selected_suppliers']) for s in all_hub_selections.values())

    print(f"\n{year} 年统计:")
    print(f"  处理Hub数: {processed_hubs}/{total_hubs}")
    print(f"  总选择供应商数: {total_selected}")

    return {
        'year': year,
        'hub_selections': all_hub_selections,
        'statistics': {
            'total_hubs': total_hubs,
            'processed_hubs': processed_hubs,
            'total_selected': total_selected
        }
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Stage 5c: 随机供应商选择（Baseline）')
    parser.add_argument('--stage1_file', type=str, default=None)
    parser.add_argument('--result_dir', type=str, default=RESULT_DIR)
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子（默认: 42，设为None则每次不同）')

    args = parser.parse_args()

    print("=" * 80)
    print("Stage 5c: 随机供应商选择（Baseline）")
    print(f"选择概率: {RANDOM_SELECTION_PROBABILITY * 100:.0f}%")
    print(f"随机种子: {args.random_seed}")
    print("=" * 80)

    # 查找输入文件
    if args.stage1_file:
        stage1_file = args.stage1_file
    else:
        stage1_files = list(Path(STAGE1_RESULT_DIR).glob("stage1_candidate_pools_*.pkl"))
        if not stage1_files:
            print("❌ 未找到Stage1输出文件")
            return
        stage1_file = str(sorted(stage1_files)[-1])

    print(f"Stage1文件: {stage1_file}")

    # 加载数据
    with open(stage1_file, 'rb') as f:
        stage1_results = pickle.load(f)

    years = sorted(stage1_results.keys())
    print(f"✓ Stage1包含 {len(years)} 个年份: {years}")

    # 处理所有年份
    all_year_results = {}
    for year in years:
        try:
            year_result = process_single_year(
                year,
                stage1_results[year],
                random_seed=args.random_seed
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

    output_file = os.path.join(
        args.result_dir,
        f'stage5c_random_selections_hub{hub_pct_match}pct.pkl'
    )

    print(f"\n{'='*80}")
    print(f"保存结果到: {output_file}")

    with open(output_file, 'wb') as f:
        pickle.dump(all_year_results, f)

    print(f"✓ Stage 5c 完成！")

    # 统计
    print(f"\n{'='*80}")
    print("汇总统计")
    print(f"{'='*80}")
    print(f"成功处理年份: {len(all_year_results)}/{len(years)}\n")

    print("年份 | Hub总数 | 处理Hub | 总选择供应商")
    print("-" * 80)
    for year, result in sorted(all_year_results.items()):
        stats = result['statistics']
        print(f"{year} | {stats['total_hubs']:>7} | {stats['processed_hubs']:>7} | "
              f"{stats['total_selected']:>12}")


if __name__ == "__main__":
    main()
