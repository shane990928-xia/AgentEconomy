#!/usr/bin/env python3
"""
Stage 3b: ML Training Data Extraction (Evolution Mode)
=======================================================

功能：从Stage 1和Stage 2的已有数据中提取机器学习训练数据

输入：
- stage1_candidate_pools_{year}/ (候选池，包含正负样本)
- stage2_profiles_{year}.pkl (画像，包含特征)

输出：
- stage3b_ml_training_data_{year}.pkl
  格式：DataFrame with columns:
    - supplier_entity_id
    - hub_entity_id
    - label (1=真实供应商, 0=噪声)
    - FF_SALES, FF_NET_MGN, FF_DEBT_EQ, FF_ROIC, FF_CURR_RATIO
    - revenue_percent (hub特定)
    - has_fsym_id, has_financial_data, has_relationship_data

数据逻辑（以year=2018为例）：
- 正样本：Stage 1候选池中的historical_suppliers + future_opportunities（label=1）
- 负样本：Stage 1候选池中的global_noise（label=0）
- 特征：Stage 2的画像（使用year-2年底财务数据）
- 用途：训练模型用t-1年数据预测t年供应关系

依赖关系：
- ⚠️ 需要Stage 1的候选池输出
- ⚠️ 需要Stage 2的画像输出

优势：
- ✅ 无需重复查询FactSet
- ✅ 直接复用已准备的数据
- ✅ 更快、更简洁
"""

import pandas as pd
import numpy as np
import os
import pickle
import argparse
import json
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 全局配置
# ============================================================================

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




# ============================================================================
# 特征提取
# ============================================================================

def extract_features_from_profile(
    supplier_id: str,
    hub_id: str,
    label: int,
    supplier_profile: Dict,
    year: int
) -> Dict:
    """
    从profile中提取特征（对齐supplier_baseline2.py）

    Args:
        supplier_id: 供应商entity_id
        hub_id: Hub节点ID
        label: 标签（1=真实供应商, 0=噪声）
        supplier_profile: 供应商的画像（来自hub_profiles[hub_id][supplier_id]）
        year: 决策年份

    Returns:
        特征字典
    """
    # 获取财务数据
    financial_data = supplier_profile.get('financial_data', {})

    # 获取hub特定的关系数据（revenue_percent等）
    revenue_percent = supplier_profile.get('revenue_percent')

    # 构建特征（对齐supplier_baseline2.py）
    features = {
        'supplier_entity_id': supplier_id,
        'hub_entity_id': hub_id,
        'decision_year': year - 1,  # 使用t-1年数据
        'label': label,

        # 关系特征（hub特定）
        'revenue_percent': safe_float_conversion(revenue_percent),

        # 财务特征
        'FF_SALES': safe_float_conversion(financial_data.get('FF_SALES')),
        'FF_NET_MGN': safe_float_conversion(financial_data.get('FF_NET_MGN')),
        'FF_DEBT_EQ': safe_float_conversion(financial_data.get('FF_DEBT_EQ')),
        'FF_CURR_RATIO': safe_float_conversion(financial_data.get('FF_CURR_RATIO')),
        'FF_ROIC': safe_float_conversion(financial_data.get('FF_ROIC')),

        # 元数据
        'has_financial_data': 'financial_data_error' not in financial_data,
        'has_relationship_data': revenue_percent is not None
    }

    return features


# ============================================================================
# 主处理流程
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Stage 3b: ML训练数据提取（演化模式）'
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
    print("Stage 3b: ML训练数据提取（演化模式）")
    print("=" * 80)
    print(f"处理年份: {args.year}")
    print(f"结果目录: {args.result_dir}")
    print("=" * 80)

    # 步骤1: 检查Stage 1和Stage 2输出
    print(f"\n[1/3] 检查Stage 1和Stage 2输出")

    candidate_pools_dir = os.path.join(
        args.result_dir,
        f"stage1_candidate_pools_{args.year}"
    )

    profiles_file = os.path.join(
        args.result_dir,
        f"stage2_profiles_{args.year}.pkl"
    )

    if not os.path.exists(candidate_pools_dir):
        raise FileNotFoundError(
            f"找不到Stage 1候选池目录: {candidate_pools_dir}\n"
            f"请先运行: python stage1_candidate_pool.py --year {args.year}"
        )

    if not os.path.exists(profiles_file):
        raise FileNotFoundError(
            f"找不到Stage 2画像: {profiles_file}\n"
            f"请先运行: python stage2_profile_builder.py --year {args.year}"
        )

    # 加载Stage 2画像
    print(f"  → 加载Stage 2画像...")
    with open(profiles_file, 'rb') as f:
        stage2_data = pickle.load(f)

    hub_profiles = stage2_data['hub_profiles']
    print(f"  ✓ 加载了 {len(hub_profiles)} 个Hub的画像")

    # 步骤2: 从候选池和画像中提取训练数据
    print(f"\n[2/3] 从候选池和画像中提取训练数据")
    print(f"  数据逻辑:")
    print(f"    - 正样本: historical_suppliers + future_opportunities (label=1)")
    print(f"    - 负样本: global_noise (label=0)")
    print(f"    - 特征: Stage 2的hub特定画像（使用{args.year-2}年底财务数据）")

    # 加载所有候选池文件
    pool_files = list(Path(candidate_pools_dir).glob("*.pkl"))
    print(f"\n  → 加载 {len(pool_files)} 个候选池文件...")

    all_training_data = []
    hub_stats = {}

    for pool_file in tqdm(pool_files, desc="  提取特征"):
        hub_id = pool_file.stem

        # 加载候选池
        with open(pool_file, 'rb') as f:
            candidate_df = pickle.load(f)

        if hub_id not in hub_profiles:
            continue

        hub_supplier_profiles = hub_profiles[hub_id]

        # 分类候选者
        positive_samples = candidate_df[
            candidate_df['source_label'].isin(['historical_suppliers', 'future_opportunities'])
        ]['factset_entity_id'].tolist()

        negative_samples = candidate_df[
            candidate_df['source_label'] == 'global_noise'
        ]['factset_entity_id'].tolist()

        features_list = []

        # 提取正样本特征
        for supplier_id in positive_samples:
            if supplier_id in hub_supplier_profiles:
                profile = hub_supplier_profiles[supplier_id]
                features = extract_features_from_profile(
                    supplier_id,
                    hub_id,
                    label=1,
                    supplier_profile=profile,
                    year=args.year
                )
                features_list.append(features)

        # 提取负样本特征
        for noise_id in negative_samples:
            if noise_id in hub_supplier_profiles:
                profile = hub_supplier_profiles[noise_id]
                features = extract_features_from_profile(
                    noise_id,
                    hub_id,
                    label=0,
                    supplier_profile=profile,
                    year=args.year
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

    # 转换为DataFrame
    print(f"\n  → 构建训练数据DataFrame...")
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

    # 步骤3: 保存结果
    print(f"\n[3/3] 保存结果")

    output_file = os.path.join(
        args.result_dir,
        f"stage3b_ml_training_data_{args.year}.pkl"
    )

    os.makedirs(args.result_dir, exist_ok=True)

    result = {
        'year': args.year,
        'training_data': training_df,
        'hub_stats': hub_stats,
        'statistics': {
            'total_samples': total_samples,
            'positive_samples': int(positive_samples),
            'negative_samples': int(negative_samples),
            'num_hubs': len(hub_stats)
        }
    }

    with open(output_file, 'wb') as f:
        pickle.dump(result, f)

    print(f"  ✓ 结果已保存到: {output_file}")

    # 打印完成信息
    print(f"\n{'='*80}")
    print(f"✓ Stage 3b 完成！")
    print(f"{'='*80}")
    print(f"处理年份: {args.year}")
    print(f"Hub数量: {len(hub_stats)}")
    print(f"训练样本总数: {total_samples:,}")
    print(f"  - 正样本: {positive_samples:,}")
    print(f"  - 负样本: {negative_samples:,}")
    print(f"\n{'='*80}")
    print(f"下一步: 运行 stage5b_ml_selection.py --year {args.year}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
