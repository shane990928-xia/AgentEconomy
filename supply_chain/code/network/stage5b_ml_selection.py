#!/usr/bin/env python3
"""
Stage 5b: ML-Based Supplier Selection
======================================

功能：使用机器学习模型进行供应商选择（三种模型）

输入：
- Stage1: hub_candidate_pools
- Stage2: hub_independent_profiles
- Stage3b: ml_training_data

输出：
- stage5b_ml_selections_*.pkl
  {
    year: {
      'model_selections': {
        'logistic_regression': {...},
        'random_forest': {...},
        'xgboost': {...}
      }
    }
  }

三种模型：
1. Logistic Regression
2. Random Forest
3. XGBoost

选择阈值：prediction_probability >= 0.5
"""

import pandas as pd
import numpy as np
import os
import pickle
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from tqdm import tqdm

# 机器学习依赖
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings('ignore')

# ============================================================================
# 全局配置
# ============================================================================

STAGE1_RESULT_DIR = "../result"
STAGE2_RESULT_DIR = "../result"
STAGE3B_RESULT_DIR = "../result"
RESULT_DIR = "../result"

SELECTION_PROBABILITY_THRESHOLD = 0.5  # 固定阈值

# ============================================================================
# 特征提取
# ============================================================================

def extract_features_for_candidates(
    hub_id: str,
    candidate_pool: pd.DataFrame,
    supplier_profiles: Dict[str, Dict],
    year: int
) -> pd.DataFrame:
    """
    为候选池中的供应商提取特征（完全对齐Stage3b的特征维度）

    Args:
        hub_id: Hub节点ID
        candidate_pool: 候选池DataFrame
        supplier_profiles: 供应商profiles字典
        year: 年份

    Returns:
        特征DataFrame
    """
    features_list = []

    for _, row in candidate_pool.iterrows():
        supplier_id = row['factset_entity_id']
        source_label = row['source_label']

        # 获取supplier profile
        supplier_profile = supplier_profiles.get(supplier_id)
        if not supplier_profile:
            continue

        # 提取财务数据
        financial_data = supplier_profile.get('financial_data', {})

        # 构建特征（完全对齐Stage3b）
        features = {
            'supplier_entity_id': supplier_id,
            'hub_entity_id': hub_id,
            'source_label': source_label,

            # 关系特征（对齐Stage3b）
            'revenue_percent': supplier_profile.get('revenue_percent'),

            # 财务特征
            'FF_SALES': financial_data.get('FF_SALES'),
            'FF_NET_MGN': financial_data.get('FF_NET_MGN'),
            'FF_DEBT_EQ': financial_data.get('FF_DEBT_EQ'),
            'FF_CURR_RATIO': financial_data.get('FF_CURR_RATIO'),
            'FF_ROIC': financial_data.get('FF_ROIC'),

            # 元数据（对齐Stage3b）
            'has_fsym_id': bool(supplier_profile.get('fsym_ids')),  # 假设profile中有fsym_ids
            'has_financial_data': 'financial_data_error' not in financial_data,
            'has_relationship_data': bool(supplier_profile.get('revenue_percent')) or bool(supplier_profile.get('start_'))
        }

        features_list.append(features)

    return pd.DataFrame(features_list)


# ============================================================================
# 模型训练
# ============================================================================

def train_model(
    training_data: pd.DataFrame,
    model_type: str,
    n_jobs: int = -1
) -> tuple:
    """
    训练单个模型

    Args:
        training_data: 训练数据DataFrame
        model_type: 模型类型 ('logistic_regression', 'random_forest', 'xgboost')
        n_jobs: 并行线程数 (-1表示使用所有CPU核心)

    Returns:
        (model, scaler, feature_columns)
    """
    # 确定特征列（排除非特征列）
    # 对齐supplier_baseline2.py的逻辑：只排除ID列、时间列、目标变量
    exclude_cols = ['supplier_entity_id', 'hub_entity_id', 'decision_year', 'label', 'source_label']
    feature_cols = [col for col in training_data.columns if col not in exclude_cols]

    X = training_data[feature_cols]
    y = training_data['label']

    # 检查是否有足够的类别
    if y.nunique() < 2:
        return None, None, feature_cols

    # 填充缺失值
    X = X.fillna(0)

    # 归一化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 训练模型
    if model_type == 'logistic_regression':
        model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000, n_jobs=n_jobs)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=n_jobs)
    elif model_type == 'xgboost':
        scale_pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1
        model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            n_jobs=n_jobs  # 并行训练
        )
    else:
        raise ValueError(f"未知的模型类型: {model_type}")

    model.fit(X_scaled, y)

    return model, scaler, feature_cols


# ============================================================================
# 供应商选择
# ============================================================================

def select_suppliers_for_hub_ml(
    hub_id: str,
    candidate_pool: pd.DataFrame,
    supplier_profiles: Dict[str, Dict],
    model: Any,
    scaler: Any,
    feature_cols: List[str],
    year: int
) -> Dict:
    """
    使用训练好的ML模型为单个Hub选择供应商

    Args:
        hub_id: Hub节点ID
        candidate_pool: 候选池DataFrame
        supplier_profiles: 供应商profiles字典
        model: 训练好的模型
        scaler: 训练好的scaler
        feature_cols: 特征列名列表
        year: 年份

    Returns:
        选择结果字典
    """
    # 1. 提取候选者特征
    candidate_features = extract_features_for_candidates(
        hub_id,
        candidate_pool,
        supplier_profiles,
        year
    )

    if candidate_features.empty:
        return {
            'selected_suppliers': [],
            'prediction_details': {},
            'statistics': {
                'total_candidates': len(candidate_pool),
                'selected_count': 0,
                'error': 'No valid candidate features'
            }
        }

    # 2. 预测
    X_test = candidate_features[feature_cols].fillna(0)
    X_test_scaled = scaler.transform(X_test)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # 3. 基于阈值选择
    selected_suppliers = []
    prediction_details = {}

    for idx, (_, row) in enumerate(candidate_features.iterrows()):
        supplier_id = row['supplier_entity_id']
        prob = y_pred_proba[idx]

        prediction_details[supplier_id] = {
            'prediction_probability': float(prob),
            'predicted_label': 1 if prob >= SELECTION_PROBABILITY_THRESHOLD else 0,
            'source_label': row['source_label']
        }

        if prob >= SELECTION_PROBABILITY_THRESHOLD:
            selected_suppliers.append(supplier_id)

    # 4. 统计
    statistics = {
        'total_candidates': len(candidate_pool),
        'total_features_extracted': len(candidate_features),
        'selected_count': len(selected_suppliers),
        'selection_rate': len(selected_suppliers) / len(candidate_features) if len(candidate_features) > 0 else 0.0,
        'num_features': len(feature_cols)
    }

    return {
        'selected_suppliers': selected_suppliers,
        'prediction_details': prediction_details,
        'statistics': statistics
    }


# ============================================================================
# 主流程
# ============================================================================

def process_single_year(
    year: int,
    stage1_data: Dict,
    stage2_data: Dict,
    stage3b_data: Dict,
    n_jobs: int = -1
) -> Dict:
    """处理单个年份

    Args:
        year: 年份
        stage1_data: Stage1数据
        stage2_data: Stage2数据
        stage3b_data: Stage3b数据
        n_jobs: 模型训练并行线程数 (-1表示使用所有CPU核心)
    """
    print(f"\n{'='*80}")
    print(f"处理 {year} 年")
    print(f"{'='*80}")

    # 获取数据
    hub_nodes = stage1_data['hub_nodes']
    hub_candidate_pools = stage1_data['hub_candidate_pools']
    hub_independent_profiles = stage2_data['hub_independent_profiles']

    # Stage3b的训练数据（包含所有Hub的训练样本）
    training_data_df = stage3b_data['training_data']
    hub_stats = stage3b_data.get('hub_stats', {})

    print(f"  Hub节点数: {len(hub_nodes)}")
    print(f"  训练数据样本数: {len(training_data_df)}")

    if training_data_df.empty:
        print(f"  ⚠ 训练数据为空，跳过该年份")
        return {
            'year': year,
            'model_selections': {
                'logistic_regression': {},
                'random_forest': {},
                'xgboost': {}
            },
            'statistics': {
                'total_hubs': len(hub_nodes),
                'stats_by_model': {
                    'logistic_regression': {'processed_hubs': 0, 'total_selected': 0},
                    'random_forest': {'processed_hubs': 0, 'total_selected': 0},
                    'xgboost': {'processed_hubs': 0, 'total_selected': 0}
                }
            }
        }

    # 三种模型的选择结果
    model_selections = {
        'logistic_regression': {},
        'random_forest': {},
        'xgboost': {}
    }

    model_types = ['logistic_regression', 'random_forest', 'xgboost']

    for model_type in model_types:
        print(f"\n--- 模型: {model_type.upper()} ---")

        # 为每个Hub训练独立的模型
        for hub_id in tqdm(hub_nodes, desc=f"  {model_type}"):
            try:
                # 检查候选池
                if hub_id not in hub_candidate_pools:
                    continue

                candidate_pool = hub_candidate_pools[hub_id]

                # 获取该Hub的训练数据
                hub_training_data = training_data_df[training_data_df['hub_entity_id'] == hub_id]

                if hub_training_data.empty:
                    continue

                # 1. 为该Hub训练模型
                model, scaler, feature_cols = train_model(hub_training_data, model_type, n_jobs)

                if model is None:
                    continue

                # 2. 使用训练好的模型进行选择
                selection_result = select_suppliers_for_hub_ml(
                    hub_id,
                    candidate_pool,
                    hub_independent_profiles,
                    model,
                    scaler,
                    feature_cols,
                    year
                )

                # 添加训练样本数信息
                selection_result['statistics']['training_samples'] = len(hub_training_data)

                model_selections[model_type][hub_id] = selection_result

            except Exception as e:
                print(f"\n  [ERROR] {hub_id} ({model_type}): {e}")
                import traceback
                traceback.print_exc()
                continue

    # 统计
    total_hubs = len(hub_nodes)
    stats_by_model = {}

    for model_type in model_types:
        processed_hubs = len(model_selections[model_type])
        total_selected = sum(
            len(s['selected_suppliers']) for s in model_selections[model_type].values()
        )

        stats_by_model[model_type] = {
            'processed_hubs': processed_hubs,
            'total_selected': total_selected
        }

        print(f"\n{model_type.upper()} 统计:")
        print(f"  处理Hub数: {processed_hubs}/{total_hubs}")
        print(f"  总选择供应商数: {total_selected}")

    return {
        'year': year,
        'model_selections': model_selections,
        'statistics': {
            'total_hubs': total_hubs,
            'stats_by_model': stats_by_model
        }
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Stage 5b: ML供应商选择（三种模型）')
    parser.add_argument('--stage1_file', type=str, default=None)
    parser.add_argument('--stage2_file', type=str, default=None)
    parser.add_argument('--stage3b_file', type=str, default=None)
    parser.add_argument('--result_dir', type=str, default=RESULT_DIR)
    parser.add_argument('--n_jobs', type=int, default=16,
                       help='模型训练并行线程数 (-1表示使用所有CPU核心, 默认: 64)')

    args = parser.parse_args()

    print("=" * 80)
    print("Stage 5b: ML供应商选择")
    print("模型: Logistic Regression, Random Forest, XGBoost")
    print("阈值: 0.5")
    print(f"模型训练并行线程数: {args.n_jobs if args.n_jobs > 0 else '所有CPU核心'}")
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

    if args.stage2_file:
        stage2_file = args.stage2_file
    else:
        stage2_files = list(Path(STAGE2_RESULT_DIR).glob("stage2_company_profiles_*.pkl"))
        if not stage2_files:
            print("❌ 未找到Stage2输出文件")
            return
        stage2_file = str(sorted(stage2_files)[-1])

    if args.stage3b_file:
        stage3b_file = args.stage3b_file
    else:
        stage3b_files = list(Path(STAGE3B_RESULT_DIR).glob("stage3b_ml_training_data_*.pkl"))
        if not stage3b_files:
            print("❌ 未找到Stage3b输出文件")
            return
        stage3b_file = str(sorted(stage3b_files)[-1])

    print(f"Stage1文件: {stage1_file}")
    print(f"Stage2文件: {stage2_file}")
    print(f"Stage3b文件: {stage3b_file}")

    # 加载数据
    with open(stage1_file, 'rb') as f:
        stage1_results = pickle.load(f)

    with open(stage2_file, 'rb') as f:
        stage2_results = pickle.load(f)

    with open(stage3b_file, 'rb') as f:
        stage3b_results = pickle.load(f)

    # 确保年份一致
    years = sorted(
        set(stage1_results.keys()) &
        set(stage2_results.keys()) &
        set(stage3b_results.keys())
    )
    print(f"✓ 共同年份: {len(years)} 个 - {years}")

    # 处理所有年份
    all_year_results = {}
    for year in years:
        try:
            year_result = process_single_year(
                year,
                stage1_results[year],
                stage2_results[year],
                stage3b_results[year],
                n_jobs=args.n_jobs
            )
            all_year_results[year] = year_result

        except Exception as e:
            print(f"\n❌ {year} 年处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 保存结果 - 为每种模型保存独立文件
    stage1_basename = os.path.basename(stage1_file)
    hub_pct_match = stage1_basename.split('_hub')[-1].split('pct')[0] if '_hub' in stage1_basename else '20'

    print(f"\n{'='*80}")
    print("保存结果（三个独立文件）")
    print(f"{'='*80}")

    model_types = ['logistic_regression', 'random_forest', 'xgboost']

    for model_type in model_types:
        # 为每种模型构建独立的结果结构（对齐Stage5a格式）
        model_results = {}

        for year, year_result in all_year_results.items():
            model_results[year] = {
                'year': year,
                'hub_selections': year_result['model_selections'][model_type],
                'statistics': {
                    'total_hubs': year_result['statistics']['total_hubs'],
                    'processed_hubs': year_result['statistics']['stats_by_model'][model_type]['processed_hubs'],
                    'total_selected': year_result['statistics']['stats_by_model'][model_type]['total_selected']
                }
            }

        # 保存到独立文件
        output_file = os.path.join(
            args.result_dir,
            f'stage5b_{model_type}_selections_hub{hub_pct_match}pct.pkl'
        )

        with open(output_file, 'wb') as f:
            pickle.dump(model_results, f)

        print(f"  ✓ {model_type}: {output_file}")

    print(f"\n✓ Stage 5b 完成！三种模型结果已分别保存")

    # 统计
    print(f"\n{'='*80}")
    print("汇总统计")
    print(f"{'='*80}")
    print(f"成功处理年份: {len(all_year_results)}/{len(years)}\n")

    for model_type in ['logistic_regression', 'random_forest', 'xgboost']:
        print(f"\n=== {model_type.upper()} ===")
        print("年份 | Hub总数 | 处理Hub | 总选择供应商")
        print("-" * 80)
        for year, result in sorted(all_year_results.items()):
            stats = result['statistics']['stats_by_model'][model_type]
            total_hubs = result['statistics']['total_hubs']
            print(f"{year} | {total_hubs:>7} | {stats['processed_hubs']:>7} | "
                  f"{stats['total_selected']:>12}")


if __name__ == "__main__":
    main()
