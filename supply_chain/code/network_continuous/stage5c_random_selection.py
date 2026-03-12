#!/usr/bin/env python3
"""
Stage 5c: Random Supplier Selection (Evolution Mode, Baseline)
===============================================================

功能：随机选择供应商并构建演化网络（演化模式，作为baseline对比）

输入：
- stage2_profiles_{year}.pkl (画像，包含hub_profiles)

输出：
- stage5c_random_selection_{year}.pkl
  格式：{
    'year': year,
    'hub_selections': {
      hub_id: {
        'selected_suppliers': [...],
        'selection_details': {...},
        'statistics': {...}
      }
    },
    'statistics': {...}
  }

- stage5c_random_evolving_graph_{year}.pkl
  格式：NetworkX有向图（供应商 -> 客户）

随机策略：
- 对每个候选供应商，以50%概率随机选择
- 使用随机种子保证可复现性

参考：
- network/code/stage5c_random_selection.py
"""

import pandas as pd
import numpy as np
import os
import pickle
import argparse
import networkx as nx
from typing import Dict, List
from tqdm import tqdm

# ============================================================================
# 全局配置
# ============================================================================

RESULT_DIR = "../result"

RANDOM_SELECTION_PROBABILITY = 0.5  # 50%概率选择每个供应商


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
# 随机选择
# ============================================================================

def random_select_suppliers(
    hub_id: str,
    hub_supplier_profiles: Dict[str, Dict],
    random_seed: int = None
) -> Dict:
    """
    随机选择供应商

    Args:
        hub_id: Hub节点ID
        hub_supplier_profiles: Hub特定的supplier profiles字典（包含hub自己）
        random_seed: 随机种子（可选）

    Returns:
        选择结果字典
    """
    if random_seed is not None:
        np.random.seed(random_seed + hash(hub_id) % 1000)  # 为每个hub使用不同的种子

    selected_suppliers = []
    selection_details = {}

    for supplier_id in hub_supplier_profiles.keys():
        if supplier_id == hub_id:
            continue  # 跳过hub自己

        supplier_profile = hub_supplier_profiles.get(supplier_id)
        if not supplier_profile:
            continue

        source_label = supplier_profile.get('source_label', 'unknown')

        # 随机决策（50%概率）
        prob = np.random.random()
        is_selected = prob < RANDOM_SELECTION_PROBABILITY  # < 0.5 则选择

        selection_details[supplier_id] = {
            'random_probability': float(prob),
            'is_selected': is_selected,
            'source_label': source_label
        }

        if is_selected:
            selected_suppliers.append(supplier_id)

    # 统计
    total_candidates = len(hub_supplier_profiles) - 1  # 减去hub自己
    statistics = {
        'total_candidates': total_candidates,
        'selected_count': len(selected_suppliers),
        'selection_rate': len(selected_suppliers) / total_candidates if total_candidates > 0 else 0.0
    }

    return {
        'selected_suppliers': selected_suppliers,
        'selection_details': selection_details,
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
        description='Stage 5c: 随机供应商选择（演化模式，Baseline）'
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
        '--random_seed',
        type=int,
        default=42,
        help='随机种子（默认: 42，保证可复现性）'
    )

    args = parser.parse_args()

    # 打印配置
    print("=" * 80)
    print("Stage 5c: 随机供应商选择（演化模式，Baseline）")
    print("=" * 80)
    print(f"处理年份: {args.year}")
    print(f"结果目录: {args.result_dir}")
    print(f"选择概率: {RANDOM_SELECTION_PROBABILITY * 100:.0f}%")
    print(f"随机种子: {args.random_seed}")
    print("=" * 80)

    # 步骤1: 检查Stage 2输出
    print(f"\n[1/3] 检查Stage 2输出")

    profiles_file = os.path.join(
        args.result_dir,
        f"stage2_profiles_{args.year}.pkl"
    )

    if not os.path.exists(profiles_file):
        raise FileNotFoundError(
            f"找不到Stage 2画像: {profiles_file}\n"
            f"请先运行: python stage2_profile_builder.py --year {args.year}"
        )

    # 加载数据
    print(f"  → 加载Stage 2画像...")
    with open(profiles_file, 'rb') as f:
        stage2_data = pickle.load(f)

    hub_profiles = stage2_data['hub_profiles']
    print(f"  ✓ 加载了 {len(hub_profiles)} 个Hub的画像")

    # 步骤2: 为每个Hub执行随机选择
    print(f"\n[2/3] 为每个Hub执行随机选择")

    all_hub_selections = {}
    success_count = 0

    # 获取所有Hub节点
    hub_nodes = list(hub_profiles.keys())
    print(f"  → 共有 {len(hub_nodes)} 个Hub需要处理")

    for hub_id in tqdm(hub_nodes, desc="  随机选择"):
        try:
            # 检查必需数据
            if hub_id not in hub_profiles:
                continue

            hub_supplier_profiles = hub_profiles[hub_id]

            # 执行随机选择
            selection_result = random_select_suppliers(
                hub_id,
                hub_supplier_profiles,
                random_seed=args.random_seed
            )

            all_hub_selections[hub_id] = selection_result
            success_count += 1

        except Exception as e:
            print(f"  [ERROR] {hub_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 统计
    total_selected = sum(len(s['selected_suppliers']) for s in all_hub_selections.values())

    print(f"\n{args.year} 年统计:")
    print(f"  成功处理Hub数: {success_count}/{len(hub_nodes)}")
    print(f"  总选择供应商数: {total_selected}")

    # 步骤3: 加载前一年的演化网络，构建本年度演化网络并保存结果
    print(f"\n[3/4] 加载前一年的演化网络")

    # 加载前一年的evolving_graph作为基础
    prev_year = args.year - 1
    prev_graph_file = os.path.join(args.result_dir, f"stage5c_random_evolving_graph_{prev_year}.pkl")

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

    # 步骤4: 构建演化网络并保存结果
    print(f"\n[4/4] 构建演化网络并保存结果")

    # 构建演化网络图
    evolving_graph = build_evolving_graph(
        previous_graph,
        hub_nodes,
        all_hub_selections,
        args.year
    )

    # 保存选择结果
    selection_output_file = os.path.join(
        args.result_dir,
        f"stage5c_random_selection_{args.year}.pkl"
    )

    os.makedirs(args.result_dir, exist_ok=True)

    result = {
        'year': args.year,
        'hub_selections': all_hub_selections,
        'statistics': {
            'total_hubs': len(hub_nodes),
            'processed_hubs': success_count,
            'total_selected': total_selected
        }
    }

    with open(selection_output_file, 'wb') as f:
        pickle.dump(result, f)

    print(f"  ✓ 选择结果已保存到: {selection_output_file}")

    # 保存演化网络图
    graph_output_file = os.path.join(
        args.result_dir,
        f"stage5c_random_evolving_graph_{args.year}.pkl"
    )

    with open(graph_output_file, 'wb') as f:
        pickle.dump(evolving_graph, f)

    print(f"  ✓ 演化网络已保存到: {graph_output_file}")

    # 打印完成信息
    print(f"\n{'='*80}")
    print(f"✓ Stage 5c 完成！")
    print(f"{'='*80}")
    print(f"处理年份: {args.year}")
    print(f"Hub总数: {len(hub_nodes)}")
    print(f"成功处理Hub数: {success_count}")
    print(f"总选择供应商数: {total_selected}")
    print(f"演化网络: {evolving_graph.number_of_nodes()}个节点, {evolving_graph.number_of_edges()}条边")
    print(f"\n{'='*80}")
    print(f"下一步: 运行 stage6_evaluation_reporting.py --year {args.year}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
