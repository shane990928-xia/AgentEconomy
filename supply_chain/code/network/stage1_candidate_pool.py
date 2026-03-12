#!/usr/bin/env python3
"""
Stage 1: Candidate Pool Construction
=====================================

功能：为所有年份构建Hub节点的候选池

输入：
- 网络数据（network/data/optimal_network_YYYY.parquet）
- 配置参数（simulation_years, hub_percentage）

输出：
- stage1_candidate_pools.pkl
  {
    2018: {
      'original_graph': nx.DiGraph,
      'hub_nodes': [hub_id1, hub_id2, ...],
      'hub_candidate_pools': {
        hub_id1: DataFrame(候选供应商),
        hub_id2: DataFrame(候选供应商),
      },
      'all_nodes_need_profile': [node1, node2, ...]
    },
    2019: {...},
    ...
  }

候选池构成（新逻辑）：
- 真实供应商：当年网络中该Hub的实际供应商
- 噪声：从（所有节点 - 真实供应商 - Hub自己）中随机抽样 noise_ratio × 总节点数 个节点
"""

import pandas as pd
import numpy as np
import networkx as nx
import os
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import Counter
from datetime import datetime


# ============================================================================
# 全局配置
# ============================================================================

# 数据路径
NETWORK_DATA_DIR = "../data"
RESULT_DIR = "../result"

# 默认参数
DEFAULT_SIMULATION_YEARS = [2018, 2019, 2020]
DEFAULT_HUB_PERCENTAGE = 0.2
DEFAULT_NOISE_RATIO = 0.1  # 默认使用所有噪声节点


# ============================================================================
# 1. 数据加载
# ============================================================================

def get_available_years(data_dir: str = NETWORK_DATA_DIR) -> List[int]:
    """
    获取所有可用的年份

    Args:
        data_dir: 网络数据目录

    Returns:
        可用年份列表
    """
    import glob
    import re

    years = set()

    # 查找所有网络数据文件
    parquet_files = glob.glob(os.path.join(data_dir, "optimal_network_*.parquet"))
    csv_files = glob.glob(os.path.join(data_dir, "optimal_network_*.csv"))

    all_files = parquet_files + csv_files

    for file_path in all_files:
        filename = os.path.basename(file_path)
        match = re.search(r'optimal_network_(\d{4})\.', filename)
        if match:
            years.add(int(match.group(1)))

    return sorted(list(years))


def load_network_data(year: int, data_dir: str = NETWORK_DATA_DIR) -> pd.DataFrame:
    """
    加载指定年份的网络数据

    Args:
        year: 年份
        data_dir: 网络数据目录

    Returns:
        包含source_id, target_id的DataFrame
    """
    parquet_file = os.path.join(data_dir, f"optimal_network_{year}.parquet")
    csv_file = os.path.join(data_dir, f"optimal_network_{year}.csv")

    if os.path.exists(parquet_file):
        df = pd.read_parquet(parquet_file)
        print(f"  ✓ 从parquet加载 {year} 年网络数据: {len(df)} 条边")
    elif os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        print(f"  ✓ 从CSV加载 {year} 年网络数据: {len(df)} 条边")
    else:
        raise FileNotFoundError(f"未找到 {year} 年的网络数据文件")

    # 确保必需列存在
    if 'source_id' not in df.columns or 'target_id' not in df.columns:
        raise ValueError(f"{year} 年网络数据缺少必需的列: source_id, target_id")

    return df


def extract_nodes_from_network(network_df: pd.DataFrame) -> List[str]:
    """
    从网络数据中提取所有节点

    Args:
        network_df: 网络边数据

    Returns:
        所有节点ID列表
    """
    sources = set(network_df['source_id'].unique())
    targets = set(network_df['target_id'].unique())
    all_nodes = sources | targets
    return sorted(list(all_nodes))


# ============================================================================
# 2. 网络图构建
# ============================================================================

def build_networkx_graph(network_df: pd.DataFrame, nodes: List[str]) -> nx.DiGraph:
    """
    从网络数据构建NetworkX有向图

    Args:
        network_df: 网络边数据（source_id, target_id）
        nodes: 节点列表（用于确保所有节点都在图中）

    Returns:
        NetworkX有向图
    """
    print(f"  → 构建NetworkX图...")

    # 创建有向图
    G = nx.DiGraph()

    # 添加所有节点
    G.add_nodes_from(nodes)

    # 添加所有边 (source -> target，即供应商 -> 客户)
    edges = [(row['source_id'], row['target_id'])
             for _, row in network_df.iterrows()]
    G.add_edges_from(edges)

    print(f"  ✓ 图构建完成: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")

    return G


# ============================================================================
# 3. Hub节点识别
# ============================================================================

def identify_hub_nodes(graph: nx.DiGraph, hub_percentage: float) -> List[str]:
    """
    识别Hub节点（基于入度排序，取top hub_percentage%）

    Args:
        graph: NetworkX有向图
        hub_percentage: Hub节点百分比（0.05 = top 5%）

    Returns:
        Hub节点ID列表
    """
    print(f"  → 识别Hub节点 (top {hub_percentage*100:.1f}% by in-degree)...")

    # 计算所有节点的入度
    in_degrees = dict(graph.in_degree())

    # 按入度排序
    sorted_nodes = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)

    # 取top hub_percentage%
    num_hubs = max(1, int(len(sorted_nodes) * hub_percentage))
    hub_nodes = [node for node, degree in sorted_nodes[:num_hubs]]

    # 统计信息
    hub_degrees = [in_degrees[node] for node in hub_nodes]
    avg_hub_degree = np.mean(hub_degrees) if hub_degrees else 0
    min_hub_degree = min(hub_degrees) if hub_degrees else 0
    max_hub_degree = max(hub_degrees) if hub_degrees else 0

    print(f"  ✓ 识别出 {len(hub_nodes)} 个Hub节点")
    print(f"    入度范围: {min_hub_degree:.0f} - {max_hub_degree:.0f}, 平均: {avg_hub_degree:.1f}")

    return hub_nodes


# ============================================================================
# 4. 候选池构建（新逻辑）
# ============================================================================

def build_candidate_pool_new_logic(
    hub_node_id: str,
    year: int,
    network_df: pd.DataFrame,
    all_network_nodes: List[str],
    noise_ratio: float = 1.0
) -> pd.DataFrame:
    """
    为单个Hub节点构建候选池（新逻辑）

    候选池构成：
    - 真实供应商：当年网络中该Hub的实际供应商
    - 噪声：从当年网络节点中随机抽样

    Args:
        hub_node_id: Hub节点ID
        year: 年份
        network_df: 当年网络数据
        all_network_nodes: 当年网络中的所有节点
        noise_ratio: 噪声节点比例（相对于总网络节点数）

    Returns:
        候选池DataFrame，包含列：factset_entity_id, source_label
    """
    # 1. 提取真实供应商
    actual_suppliers = set(
        network_df[network_df['target_id'] == hub_node_id]['source_id'].unique()
    )

    # 2. 计算噪声节点池（使用当年网络节点）
    all_nodes_set = set(all_network_nodes)
    noise_pool = all_nodes_set - actual_suppliers - {hub_node_id}

    # 3. 根据noise_ratio抽样噪声节点
    num_noise_to_sample = int(len(all_network_nodes) * noise_ratio)

    if noise_ratio >= 1.0 or num_noise_to_sample >= len(noise_pool):
        # 使用所有噪声节点
        noise_nodes = noise_pool
    else:
        # 随机抽样
        noise_nodes = set(np.random.choice(
            list(noise_pool),
            size=num_noise_to_sample,
            replace=False
        ))

    # 4. 构建DataFrame
    data = []

    # 添加真实供应商
    for supplier_id in actual_suppliers:
        data.append({
            'factset_entity_id': supplier_id,
            'source_label': 'actual_supplier'
        })

    # 添加噪声节点
    for noise_id in noise_nodes:
        data.append({
            'factset_entity_id': noise_id,
            'source_label': 'noise'
        })

    candidate_df = pd.DataFrame(data)

    return candidate_df


def build_all_candidate_pools(
    hub_nodes: List[str],
    year: int,
    network_df: pd.DataFrame,
    all_network_nodes: List[str],
    noise_ratio: float = 1.0
) -> Dict[str, pd.DataFrame]:
    """
    为所有Hub节点构建候选池

    Args:
        hub_nodes: Hub节点列表
        year: 年份
        network_df: 当年网络数据
        all_network_nodes: 所有网络节点
        noise_ratio: 噪声节点比例

    Returns:
        字典 {hub_id: candidate_pool_df}
    """
    print(f"  → 为 {len(hub_nodes)} 个Hub节点构建候选池...")
    print(f"    噪声池来源: 当年网络节点")
    print(f"    噪声比例: {noise_ratio*100:.1f}% × 总节点数 = {int(len(all_network_nodes)*noise_ratio)} 个噪声节点（最大）")

    hub_candidate_pools = {}

    for hub_id in hub_nodes:
        candidate_df = build_candidate_pool_new_logic(
            hub_id, year, network_df, all_network_nodes, noise_ratio
        )

        if not candidate_df.empty:
            hub_candidate_pools[hub_id] = candidate_df

            # 统计
            composition = candidate_df['source_label'].value_counts().to_dict()
            actual_count = composition.get('actual_supplier', 0)
            noise_count = composition.get('noise', 0)

            print(f"    • {hub_id}: {len(candidate_df)} 个候选者 "
                  f"(真实:{actual_count}, 噪声:{noise_count})")
        else:
            print(f"    ⚠ {hub_id}: 候选池为空（跳过）")

    print(f"  ✓ 成功为 {len(hub_candidate_pools)}/{len(hub_nodes)} 个Hub节点构建了候选池")

    return hub_candidate_pools


# ============================================================================
# 5. 收集需要画像的节点
# ============================================================================

def collect_nodes_need_profile(
    hub_nodes: List[str],
    hub_candidate_pools: Dict[str, pd.DataFrame]
) -> List[str]:
    """
    收集所有需要生成画像的节点（去重）

    包括：
    - 所有Hub节点（需要生成性格）
    - 所有候选池中的供应商节点

    Args:
        hub_nodes: Hub节点列表
        hub_candidate_pools: Hub候选池字典

    Returns:
        需要画像的节点ID列表（去重）
    """
    all_nodes = set(hub_nodes)

    # 收集所有候选池中的节点
    for hub_id, candidate_df in hub_candidate_pools.items():
        suppliers = set(candidate_df['factset_entity_id'].unique())
        all_nodes.update(suppliers)

    # 移除空值
    all_nodes = {node for node in all_nodes if node and isinstance(node, str)}

    return sorted(list(all_nodes))


# ============================================================================
# 6. 主处理流程
# ============================================================================

def process_single_year(
    year: int,
    hub_percentage: float,
    noise_ratio: float,
    data_dir: str
) -> Dict:
    """
    处理单个年份

    Args:
        year: 年份
        hub_percentage: Hub节点百分比
        noise_ratio: 噪声节点比例
        data_dir: 数据目录

    Returns:
        该年份的结果字典
    """
    print(f"\n{'='*80}")
    print(f"处理 {year} 年")
    print(f"{'='*80}")

    # 1. 加载网络数据
    print(f"\n[1/6] 加载 {year} 年网络数据")
    network_df = load_network_data(year, data_dir)

    # 2. 提取节点
    print(f"\n[2/6] 提取网络节点")
    all_nodes = extract_nodes_from_network(network_df)
    print(f"  ✓ 提取到 {len(all_nodes)} 个节点")

    # 3. 构建NetworkX图
    print(f"\n[3/6] 构建NetworkX图")
    original_graph = build_networkx_graph(network_df, all_nodes)

    # 4. 识别Hub节点
    print(f"\n[4/6] 识别Hub节点")
    hub_nodes = identify_hub_nodes(original_graph, hub_percentage)

    # 5. 构建候选池（使用当年网络节点）
    print(f"\n[5/6] 构建候选池（噪声比例: {noise_ratio*100:.1f}%）")
    hub_candidate_pools = build_all_candidate_pools(
        hub_nodes, year, network_df, all_nodes, noise_ratio
    )

    # 6. 收集需要画像的节点
    print(f"\n[6/6] 收集需要画像的节点")
    all_nodes_need_profile = collect_nodes_need_profile(hub_nodes, hub_candidate_pools)
    print(f"  ✓ 需要画像的节点总数: {len(all_nodes_need_profile)}")

    # 统计信息
    total_candidates = sum(len(df) for df in hub_candidate_pools.values())
    print(f"\n{year} 年统计:")
    print(f"  - 网络节点总数: {len(all_nodes)}")
    print(f"  - Hub节点数: {len(hub_nodes)}")
    print(f"  - 候选池总规模: {total_candidates}")
    print(f"  - 需要画像节点数: {len(all_nodes_need_profile)}")

    # 返回结果
    return {
        'year': year,
        'original_graph': original_graph,
        'hub_nodes': hub_nodes,
        'hub_candidate_pools': hub_candidate_pools,
        'all_nodes_need_profile': all_nodes_need_profile,
        'statistics': {
            'total_nodes': len(all_nodes),
            'total_edges': original_graph.number_of_edges(),
            'hub_count': len(hub_nodes),
            'total_candidates': total_candidates,
            'nodes_need_profile': len(all_nodes_need_profile)
        }
    }


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Stage 1: 构建Hub节点候选池（所有年份）'
    )
    parser.add_argument(
        '--years',
        type=str,
        default='all',
        help='模拟年份，逗号分隔（默认: 2018,2019,2020）'
    )
    parser.add_argument(
        '--hub_percentage',
        type=float,
        default=0.20,
        help='Hub节点百分比（默认: 0.20 = 20%%）'
    )
    parser.add_argument(
        '--noise_ratio',
        type=float,
        default=0.01,
        help='噪声节点比例（相对于总网络节点数，默认: 1.0 = 100%%）'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=NETWORK_DATA_DIR,
        help='网络数据目录'
    )
    parser.add_argument(
        '--result_dir',
        type=str,
        default=RESULT_DIR,
        help='结果保存目录'
    )

    args = parser.parse_args()

    # 解析年份
    if args.years.lower() == 'all':
        simulation_years = get_available_years(args.data_dir)
        print(f"检测到所有可用年份: {simulation_years}")
    else:
        simulation_years = [int(y.strip()) for y in args.years.split(',')]

    # 打印配置
    print("=" * 80)
    print("Stage 1: 候选池构建")
    print("=" * 80)
    print(f"模拟年份: {simulation_years}")
    print(f"Hub比例: {args.hub_percentage * 100:.1f}%")
    print(f"噪声比例: {args.noise_ratio * 100:.1f}%")
    print(f"数据目录: {args.data_dir}")
    print(f"结果目录: {args.result_dir}")
    print("=" * 80)

    # 创建结果目录
    os.makedirs(args.result_dir, exist_ok=True)

    # 处理所有年份
    all_year_results = {}

    for year in simulation_years:
        try:
            year_result = process_single_year(
                year, args.hub_percentage, args.noise_ratio, args.data_dir
            )
            all_year_results[year] = year_result
        except Exception as e:
            print(f"\n❌ {year} 年处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 保存结果
    output_file = os.path.join(
        args.result_dir,
        f'stage1_candidate_pools_hub{int(args.hub_percentage*100)}pct_noise{int(args.noise_ratio*100)}pct.pkl'
    )

    print(f"\n{'='*80}")
    print(f"保存结果到: {output_file}")

    with open(output_file, 'wb') as f:
        pickle.dump(all_year_results, f)

    print(f"✓ Stage 1 完成！")

    # 打印汇总统计
    print(f"\n{'='*80}")
    print("汇总统计")
    print(f"{'='*80}")
    print(f"成功处理年份: {len(all_year_results)}/{len(simulation_years)}")

    summary_data = []
    for year, result in sorted(all_year_results.items()):
        stats = result['statistics']
        summary_data.append([
            year,
            f"{stats['total_nodes']:,}",
            f"{stats['total_edges']:,}",
            f"{stats['hub_count']:,}",
            f"{stats['total_candidates']:,}",
            f"{stats['nodes_need_profile']:,}"
        ])

    # 打印表格
    print("\n年份 | 节点数 | 边数 | Hub数 | 候选池总规模 | 需要画像")
    print("-" * 80)
    for row in summary_data:
        print(f"{row[0]} | {row[1]:>8} | {row[2]:>8} | {row[3]:>6} | {row[4]:>12} | {row[5]:>10}")

    print(f"\n{'='*80}")
    print(f"下一步: 运行 stage2_profile_builder.py")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()