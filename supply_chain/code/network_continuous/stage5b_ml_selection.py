#!/usr/bin/env python3
"""
Stage 5b: ML Supplier Selection (Evolution Mode)
=================================================

功能：使用机器学习模型进行供应商选择并构建演化网络（演化模式）

输入：
- stage1_candidate_pools_{year}/ (候选池，用于获取ground truth数量)
- stage2_profiles_{year}.pkl (画像，包含hub_profiles)
- stage3b_ml_training_data_{year}.pkl (训练数据)

输出（为每种模型分别保存）：
- stage5b_{model}_selection_{year}.pkl
  格式：{
    'year': year,
    'model_type': model_type,
    'hub_selections': {
      hub_id: {
        'selected_suppliers': [...],
        'prediction_details': {...},
        'statistics': {...}
      }
    },
    'statistics': {...}
  }

- stage5b_{model}_evolving_graph_{year}.pkl
  格式：NetworkX有向图（供应商 -> 客户）

三种模型：
1. logistic_regression (Logistic Regression)
2. random_forest (Random Forest)
3. xgboost (XGBoost)

选择阈值：Equal_adjustment（动态阈值，基于ground truth数量）

参考：
- network/code/stage5b_ml_selection.py
- supplier_test.py: Equal_adjustment逻辑
"""

import pandas as pd
import numpy as np
import os
import pickle
import argparse
import networkx as nx
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any
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

RESULT_DIR = "../result"


# ============================================================================
# 辅助函数：加载真实网络
# ============================================================================

def load_sandbox_network(sandbox_folder: str, year: int) -> nx.DiGraph:
    """
    加载指定年份的沙盒网络图（真实网络）

    Args:
        sandbox_folder: 沙盒数据文件夹路径
        year: 年份

    Returns:
        NetworkX有向图（供应商 -> 客户）
    """
    nodes_file = os.path.join(sandbox_folder, f"{year}_nodes.csv")
    edges_file = os.path.join(sandbox_folder, f"{year}_edges.csv")

    if not os.path.exists(nodes_file) or not os.path.exists(edges_file):
        raise FileNotFoundError(f"{year}年的沙盒网络数据不存在")

    # 加载节点和边
    nodes_df = pd.read_csv(nodes_file)
    edges_df = pd.read_csv(edges_file)

    # 创建有向图
    G = nx.DiGraph()

    # 添加节点（适配真实数据字段名'Id'）
    for _, row in nodes_df.iterrows():
        node_id = str(row['Id'])  # 确保节点ID为字符串
        G.add_node(node_id)

    # 添加边（适配真实数据字段名'Source', 'Target'）
    for _, row in edges_df.iterrows():
        source = str(row['Source'])
        target = str(row['Target'])
        if source in G.nodes and target in G.nodes:
            G.add_edge(source, target)

    print(f"  ✓ 加载{year}年真实网络: {G.number_of_nodes()}个节点, {G.number_of_edges()}条边")

    return G

# ============================================================================
# Ground Truth 辅助函数
# ============================================================================

def get_ground_truth_supplier_count(candidate_pool_df: pd.DataFrame) -> int:
    """
    从候选池获取 ground truth 供应商数量

    Ground truth 数量 = historical_suppliers（继续存在的） + future_opportunities（新增的）

    Args:
        candidate_pool_df: 候选池DataFrame，包含 'source_label' 列

    Returns:
        Ground truth 供应商数量
    """
    if candidate_pool_df is None or candidate_pool_df.empty:
        return 0

    # 统计 historical_suppliers 和 future_opportunities
    ground_truth_suppliers = candidate_pool_df[
        candidate_pool_df['source_label'].isin(['historical_suppliers', 'future_opportunities'])
    ]

    return len(ground_truth_suppliers)

# ============================================================================
# 特征提取
# ============================================================================

def extract_features_for_candidates(
    hub_id: str,
    hub_supplier_profiles: Dict[str, Dict],
    year: int
) -> pd.DataFrame:
    """
    为候选池中的供应商提取特征（完全对齐Stage3b的特征维度）

    Args:
        hub_id: Hub节点ID
        hub_supplier_profiles: Hub特定的supplier profiles字典（包含hub自己）
        year: 年份

    Returns:
        特征DataFrame
    """
    features_list = []

    for supplier_id in hub_supplier_profiles.keys():
        if supplier_id == hub_id:
            continue  # 跳过hub自己

        supplier_profile = hub_supplier_profiles.get(supplier_id)
        if not supplier_profile:
            continue

        # 提取财务数据
        financial_data = supplier_profile.get('financial_data', {})

        # 获取hub特定的关系数据
        revenue_percent = supplier_profile.get('revenue_percent')

        # 构建特征（完全对齐Stage3b）
        features = {
            'supplier_entity_id': supplier_id,
            'hub_entity_id': hub_id,
            'source_label': supplier_profile.get('source_label', 'unknown'),

            # 关系特征（hub特定）
            'revenue_percent': revenue_percent,

            # 财务特征
            'FF_SALES': financial_data.get('FF_SALES'),
            'FF_NET_MGN': financial_data.get('FF_NET_MGN'),
            'FF_DEBT_EQ': financial_data.get('FF_DEBT_EQ'),
            'FF_CURR_RATIO': financial_data.get('FF_CURR_RATIO'),
            'FF_ROIC': financial_data.get('FF_ROIC'),

            # 元数据（对齐Stage3b）
            'has_financial_data': 'financial_data_error' not in financial_data,
            'has_relationship_data': revenue_percent is not None
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
            n_jobs=n_jobs
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
    hub_supplier_profiles: Dict[str, Dict],
    model: Any,
    scaler: Any,
    feature_cols: List[str],
    year: int,
    candidate_pool_df: Optional[pd.DataFrame] = None
) -> Dict:
    """
    使用训练好的ML模型为单个Hub选择供应商（Equal_adjustment阈值调整）

    Args:
        hub_id: Hub节点ID
        hub_supplier_profiles: Hub特定的supplier profiles字典（包含hub自己）
        model: 训练好的模型
        scaler: 训练好的scaler
        feature_cols: 特征列名列表
        year: 年份
        candidate_pool_df: 候选池DataFrame（用于获取ground truth数量）

    Returns:
        选择结果字典
    """
    # 1. 提取候选者特征
    candidate_features = extract_features_for_candidates(
        hub_id,
        hub_supplier_profiles,
        year
    )

    if candidate_features.empty:
        return {
            'selected_suppliers': [],
            'prediction_details': {},
            'statistics': {
                'total_candidates': 0,
                'selected_count': 0,
                'error': 'No valid candidate features'
            }
        }

    # 2. 预测
    X_test = candidate_features[feature_cols].fillna(0)
    X_test_scaled = scaler.transform(X_test)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # 3. Equal_adjustment 阈值调整（对齐supplier_test.py）
    selected_suppliers = []
    prediction_details = {}

    # 先收集所有预测概率和详情
    prediction_probabilities = {}
    for idx, (_, row) in enumerate(candidate_features.iterrows()):
        supplier_id = row['supplier_entity_id']
        prob = y_pred_proba[idx]

        prediction_probabilities[supplier_id] = float(prob)
        prediction_details[supplier_id] = {
            'prediction_probability': float(prob),
            'source_label': row['source_label']
        }

    # Equal_adjustment 阈值调整
    if len(prediction_probabilities) > 0:
        # 获取 ground truth 供应商数量
        n = get_ground_truth_supplier_count(candidate_pool_df)

        if n > 0:
            # 按概率从高到低排序
            sorted_probs = sorted(prediction_probabilities.values(), reverse=True)
            n = min(n, len(sorted_probs))

            # 找到唯一的候选阈值（避免重复概率导致的问题）
            unique_sorted_probs = sorted(set(sorted_probs), reverse=True)

            # 基于目标位置n，找到附近的唯一阈值进行测试
            target_threshold = sorted_probs[n - 1]

            # 收集候选阈值：目标阈值及其附近的唯一值
            candidate_thresholds = set()
            for idx in range(max(0, n - 3), min(len(sorted_probs), n + 3)):
                candidate_thresholds.add(sorted_probs[idx])

            # 如果候选阈值太少，添加更多唯一阈值
            if len(candidate_thresholds) < 5 and len(unique_sorted_probs) > 0:
                # 找到目标阈值在唯一列表中的位置
                try:
                    target_idx = unique_sorted_probs.index(target_threshold)
                    # 添加前后的唯一阈值
                    for offset in range(-2, 3):
                        idx = target_idx + offset
                        if 0 <= idx < len(unique_sorted_probs):
                            candidate_thresholds.add(unique_sorted_probs[idx])
                except ValueError:
                    # 如果找不到，使用前5个唯一阈值
                    candidate_thresholds.update(unique_sorted_probs[:5])

            best_threshold = None
            best_count_diff = float('inf')
            best_selected = []

            for threshold in candidate_thresholds:
                # 计算使用此阈值会选中多少个
                selected = [
                    supplier_id for supplier_id, prob in prediction_probabilities.items()
                    if prob >= threshold
                ]
                count_diff = abs(len(selected) - n)

                # 如果这个阈值更接近目标数量，则更新
                if count_diff < best_count_diff:
                    best_count_diff = count_diff
                    best_threshold = threshold
                    best_selected = selected

            # 使用最优阈值的选择结果
            selected_suppliers = best_selected

            # 更新 prediction_details 中的 predicted_label
            for supplier_id in prediction_details:
                prob = prediction_probabilities[supplier_id]
                prediction_details[supplier_id]['predicted_label'] = 1 if prob >= best_threshold else 0

            print(f"  → [Equal_adjustment] n={n}, threshold={best_threshold:.6f}, "
                  f"selected={len(selected_suppliers)}/{len(prediction_probabilities)} "
                  f"(diff={abs(len(selected_suppliers) - n)})")
        else:
            # 如果没有 ground truth，跳过阈值调整
            print(f"  → [Equal_adjustment] ground truth 数量为0，跳过阈值调整")

    # 4. 统计
    statistics = {
        'total_candidates': len(candidate_features),
        'selected_count': len(selected_suppliers),
        'selection_rate': len(selected_suppliers) / len(candidate_features) if len(candidate_features) > 0 else 0.0,
        'num_features': len(feature_cols),
        'ground_truth_count': get_ground_truth_supplier_count(candidate_pool_df)
    }

    return {
        'selected_suppliers': selected_suppliers,
        'prediction_details': prediction_details,
        'statistics': statistics
    }


# ============================================================================
# 演化网络构建
# ============================================================================

def build_evolving_graph(
    previous_graph: nx.DiGraph,
    hub_nodes: List[str],
    hub_selections: Dict[str, Dict],
    year: int
) -> nx.DiGraph:
    """
    基于前一年的网络和选择结果构建演化网络图

    Args:
        previous_graph: 前一年的演化网络图
        hub_nodes: Hub节点列表
        hub_selections: 所有Hub的选择结果
        year: 年份

    Returns:
        NetworkX有向图（供应商 -> 客户）

    流程：
    1. 复制前一年的网络
    2. 移除所有Hub的入边（要重新选择供应商）
    3. 根据选择结果添加新边和新节点
    4. 移除孤立节点（节点回收机制）
    """
    print(f"\n  → 构建{year}年演化网络图...")
    print(f"    前一年网络: {previous_graph.number_of_nodes()}个节点, {previous_graph.number_of_edges()}条边")

    # 1. 复制前一年的网络
    G = previous_graph.copy()

    # 2. 移除所有Hub的入边（准备重新选择供应商）
    edges_to_remove = []
    for hub_id in hub_nodes:
        if hub_id in G:
            in_edges = list(G.in_edges(hub_id))
            edges_to_remove.extend(in_edges)

    G.remove_edges_from(edges_to_remove)
    print(f"    移除 {len(edges_to_remove)} 条Hub入边")

    # 3. 根据选择结果添加新边和新节点
    new_nodes = set()
    new_edges = []

    for hub_id, selection_result in hub_selections.items():
        selected_suppliers = selection_result['selected_suppliers']

        for supplier_id in selected_suppliers:
            # 节点增加机制：如果供应商不在图中，添加它
            if supplier_id not in G:
                G.add_node(supplier_id)
                new_nodes.add(supplier_id)

            # 添加边（供应商 -> Hub）
            G.add_edge(supplier_id, hub_id)
            new_edges.append((supplier_id, hub_id))

    print(f"    新增 {len(new_nodes)} 个节点")
    print(f"    新增 {len(new_edges)} 条边")

    # 4. 节点回收机制：移除孤立节点（没有任何边的节点）
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    if len(isolated_nodes) > 0:
        print(f"    回收 {len(isolated_nodes)} 个孤立节点")

    print(f"  ✓ 演化网络: {G.number_of_nodes()}个节点, {G.number_of_edges()}条边")

    return G


# ============================================================================
# 主处理流程
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Stage 5b: ML供应商选择（演化模式，三种模型）'
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
    parser.add_argument(
        '--sandbox_folder',
        type=str,
        required=True,
        help='沙盒网络数据文件夹路径（用于加载真实网络）'
    )
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=16,
        help='模型训练并行线程数（-1表示使用所有CPU核心，默认16）'
    )

    args = parser.parse_args()

    # 打印配置
    print("=" * 80)
    print("Stage 5b: ML供应商选择（演化模式）")
    print("=" * 80)
    print(f"处理年份: {args.year}")
    print(f"结果目录: {args.result_dir}")
    print(f"模型训练并行线程数: {args.n_jobs if args.n_jobs > 0 else '所有CPU核心'}")
    print(f"阈值策略: Equal_adjustment（动态阈值）")
    print("=" * 80)

    # 步骤1: 检查Stage 1、2、3b输出
    print(f"\n[1/5] 检查Stage 1、2、3b输出")

    profiles_file = os.path.join(
        args.result_dir,
        f"stage2_profiles_{args.year}.pkl"
    )

    training_data_file = os.path.join(
        args.result_dir,
        f"stage3b_ml_training_data_{args.year}.pkl"
    )

    if not os.path.exists(profiles_file):
        raise FileNotFoundError(
            f"找不到Stage 2画像: {profiles_file}\n"
            f"请先运行: python stage2_profile_builder.py --year {args.year}"
        )

    if not os.path.exists(training_data_file):
        raise FileNotFoundError(
            f"找不到Stage 3b训练数据: {training_data_file}\n"
            f"请先运行: python stage3b_ml_training_data.py --year {args.year}"
        )

    # 加载数据
    print(f"  → 加载Stage 2画像...")
    with open(profiles_file, 'rb') as f:
        stage2_data = pickle.load(f)

    hub_profiles = stage2_data['hub_profiles']
    print(f"  ✓ 加载了 {len(hub_profiles)} 个Hub的画像")

    print(f"  → 加载Stage 3b训练数据...")
    with open(training_data_file, 'rb') as f:
        stage3b_data = pickle.load(f)

    training_data_df = stage3b_data['training_data']
    print(f"  ✓ 加载了 {len(training_data_df)} 个训练样本")

    if training_data_df.empty:
        raise ValueError("训练数据为空，无法训练模型")

    # 加载Stage 1候选池（用于获取ground truth数量）
    candidate_pool_dir = os.path.join(args.result_dir, f"stage1_candidate_pools_{args.year}")

    candidate_pools = {}
    if os.path.exists(candidate_pool_dir):
        print(f"  → 加载Stage 1候选池...")
        pool_files = [f for f in os.listdir(candidate_pool_dir) if f.endswith('.pkl')]
        for pool_file in pool_files:
            hub_id = pool_file.replace('.pkl', '')
            pool_path = os.path.join(candidate_pool_dir, pool_file)
            with open(pool_path, 'rb') as f:
                candidate_pools[hub_id] = pickle.load(f)
        print(f"  ✓ 加载了 {len(candidate_pools)} 个Hub的候选池")
    else:
        print(f"  ⚠ 警告：找不到候选池目录 {candidate_pool_dir}")
        print(f"  → 将使用默认阈值调整（可能影响准确性）")

    # 步骤2: 为三种模型分别训练和选择
    print(f"\n[2/5] 为三种模型分别训练和选择")

    model_types = ['logistic_regression', 'random_forest', 'xgboost']
    all_model_results = {}

    for model_type in model_types:
        print(f"\n{'='*80}")
        print(f"模型: {model_type.upper()}")
        print(f"{'='*80}")

        model_selections = {}
        success_count = 0

        # 获取所有Hub节点
        hub_nodes = list(hub_profiles.keys())

        for hub_id in tqdm(hub_nodes, desc=f"  {model_type}"):
            try:
                # 检查必需数据
                if hub_id not in hub_profiles:
                    continue

                hub_supplier_profiles = hub_profiles[hub_id]

                # 获取该Hub的训练数据
                hub_training_data = training_data_df[training_data_df['hub_entity_id'] == hub_id]

                if hub_training_data.empty:
                    continue

                # 1. 为该Hub训练模型
                model, scaler, feature_cols = train_model(hub_training_data, model_type, args.n_jobs)

                if model is None:
                    continue

                # 获取候选池
                candidate_pool_df = candidate_pools.get(hub_id, None)

                # 2. 使用训练好的模型进行选择
                selection_result = select_suppliers_for_hub_ml(
                    hub_id,
                    hub_supplier_profiles,
                    model,
                    scaler,
                    feature_cols,
                    args.year,
                    candidate_pool_df=candidate_pool_df
                )

                # 添加训练样本数信息
                selection_result['statistics']['training_samples'] = len(hub_training_data)

                model_selections[hub_id] = selection_result
                success_count += 1

            except Exception as e:
                print(f"\n  [ERROR] {hub_id} ({model_type}): {e}")
                import traceback
                traceback.print_exc()
                continue

        # 统计
        total_selected = sum(len(s['selected_suppliers']) for s in model_selections.values())

        print(f"\n{model_type.upper()} 统计:")
        print(f"  成功处理Hub数: {success_count}/{len(hub_nodes)}")
        print(f"  总选择供应商数: {total_selected}")

        # 保存该模型的结果
        all_model_results[model_type] = {
            'hub_selections': model_selections,
            'statistics': {
                'total_hubs': len(hub_nodes),
                'processed_hubs': success_count,
                'total_selected': total_selected
            }
        }

    # 步骤3: 为每种模型加载前一年的演化网络，构建演化网络并保存
    print(f"\n[3/5] 为每种模型加载前一年的演化网络，构建演化网络并保存")

    all_evolving_graphs = {}

    for model_type in model_types:
        print(f"\n{'='*80}")
        print(f"模型: {model_type.upper()} - 构建演化网络")
        print(f"{'='*80}")

        hub_selections = all_model_results[model_type]['hub_selections']

        # 加载前一年的evolving_graph作为基础
        prev_year = args.year - 1
        prev_graph_file = os.path.join(
            args.result_dir,
            f"stage5b_{model_type}_evolving_graph_{prev_year}.pkl"
        )

        if os.path.exists(prev_graph_file):
            print(f"  → 加载{prev_year}年的演化网络...")
            with open(prev_graph_file, 'rb') as f:
                previous_graph = pickle.load(f)
            print(f"  ✓ 前一年网络: {previous_graph.number_of_nodes()}个节点, {previous_graph.number_of_edges()}条边")
        else:
            # 第一年：使用真实网络初始化
            print(f"  → {prev_year}年的演化网络不存在，尝试加载{prev_year}年真实网络...")
            try:
                previous_graph = load_sandbox_network(args.sandbox_folder, prev_year)
                print(f"  ✓ 使用{prev_year}年真实网络作为初始演化网络")
            except FileNotFoundError as e:
                print(f"  ⚠ 警告：{e}")
                print(f"  → 创建空图初始化")
                previous_graph = nx.DiGraph()
                for hub_id in hub_nodes:
                    previous_graph.add_node(hub_id)

        # 构建演化网络
        evolving_graph = build_evolving_graph(
            previous_graph,
            hub_nodes,
            hub_selections,
            args.year
        )

        all_evolving_graphs[model_type] = evolving_graph

    # 步骤4: 保存结果（每种模型分别保存）
    print(f"\n[4/5] 保存结果（每种模型分别保存）")

    os.makedirs(args.result_dir, exist_ok=True)

    for model_type in model_types:
        # 保存选择结果
        selection_output_file = os.path.join(
            args.result_dir,
            f"stage5b_{model_type}_selection_{args.year}.pkl"
        )

        result = {
            'year': args.year,
            'model_type': model_type,
            'hub_selections': all_model_results[model_type]['hub_selections'],
            'statistics': all_model_results[model_type]['statistics']
        }

        with open(selection_output_file, 'wb') as f:
            pickle.dump(result, f)

        print(f"  ✓ {model_type} 选择结果: {selection_output_file}")

        # 保存演化网络图
        graph_output_file = os.path.join(
            args.result_dir,
            f"stage5b_{model_type}_evolving_graph_{args.year}.pkl"
        )

        with open(graph_output_file, 'wb') as f:
            pickle.dump(all_evolving_graphs[model_type], f)

        print(f"  ✓ {model_type} 演化网络: {graph_output_file}")

    # 打印完成信息
    print(f"\n{'='*80}")
    print(f"✓ Stage 5b 完成！")
    print(f"{'='*80}")
    print(f"处理年份: {args.year}")

    for model_type in model_types:
        stats = all_model_results[model_type]['statistics']
        graph = all_evolving_graphs[model_type]
        print(f"\n{model_type.upper()}:")
        print(f"  Hub总数: {stats['total_hubs']}")
        print(f"  成功处理Hub数: {stats['processed_hubs']}")
        print(f"  总选择供应商数: {stats['total_selected']}")
        print(f"  演化网络: {graph.number_of_nodes()}个节点, {graph.number_of_edges()}条边")

    print(f"\n{'='*80}")
    print(f"下一步: 运行 stage6_evaluation_reporting.py --year {args.year}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
