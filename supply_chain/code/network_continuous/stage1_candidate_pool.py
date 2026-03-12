#!/usr/bin/env python3
"""
Stage 1: Candidate Pool Construction (Evolution Mode)
======================================================

功能：为指定年份构建所有核心决策者的候选供应商池（演化模式）

输入：
- 沙盒网络数据（seed_networks_*/YYYY_nodes.csv, YYYY_edges.csv）
- 前一年的演化网络状态（stage5*_evolving_graph_{year-1}.pkl）
- 全局FactSet数据（可选，用于全局噪声采样）
- 配置参数（year, start_year, selection_method等）

输出：
- stage1_candidate_pools_{year}/
  ├── {decision_maker_id}.pkl  # 每个决策者的候选池DataFrame
  ├── ...
- stage1_metadata_{year}.json  # 元数据统计

候选池构成（演化模式）：
1. 历史供应商：
   - 如果year == start_year+1: 来自real_graphs[start_year]
   - 否则: 来自evolving_graph_{year-1}
2. 未来机会：real_graphs[year]中的新供应商（不在历史中）
3. 全局噪声：从沙盒节点池或FactSet全局数据中随机采样
   - 噪声数量 = 历史供应商数量（100%比例）
   - 例如：2015年有50个供应商 → 2016年采样50个噪声节点

依赖关系：
- ⚠️ 需要前一年的演化网络状态（stage5*_evolving_graph_{year-1}.pkl）
- ⚠️ 第一年（start_year+1）时，使用真实网络初始化

参考：
- network_continuous_simulation.py: build_candidate_pool_evolution()
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
import argparse
import networkx as nx
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 全局配置
# ============================================================================

RESULT_DIR = "../result"
DATA_DIR = "../data"
FACTSET_DATA_PATH = "/root/tmp/supplier/experiment/data/factset"
CLEANED_RELATIONSHIP_FILE = "../../factset_revere_relationship_cleaned.parquet"  # 清洗后的关系数据

# ============================================================================
# 辅助函数：加载清洗后的节点池
# ============================================================================

def load_cleaned_relationship_nodes():
    """
    从清洗后的关系数据中提取所有entity_id节点（用于噪声抽样）

    Returns:
        所有节点entity_id的集合
    """
    print(f"\n加载清洗后的关系数据用于噪声抽样...")

    if not os.path.exists(CLEANED_RELATIONSHIP_FILE):
        print(f"  ⚠ 清洗后的关系数据不存在: {CLEANED_RELATIONSHIP_FILE}")
        print(f"  → 将使用沙盒网络节点作为噪声池")
        return None

    try:
        # 加载清洗后的关系数据
        df_cleaned = pd.read_parquet(CLEANED_RELATIONSHIP_FILE)
        print(f"  → 加载了 {len(df_cleaned):,} 条清洗后的关系")

        # 加载company_factset映射表
        factset_path = os.path.join(FACTSET_DATA_PATH, 'data', 'company_factset.parquet')
        if not os.path.exists(factset_path):
            print(f"  ⚠ company_factset映射表不存在，无法映射到entity_id")
            return None

        df_cf = pd.read_parquet(factset_path)
        df_cf['company_id'] = pd.to_numeric(df_cf['company_id'], errors='coerce').astype('Int64')
        df_cf = df_cf.dropna(subset=['company_id', 'fs_entity_id'])

        # 创建company_id到entity_id的映射（优先pri='Y'）
        primary_map = df_cf[df_cf['pri'] == 'Y'][['company_id', 'fs_entity_id']].drop_duplicates('company_id', keep='first')
        fallback_map = df_cf[['company_id', 'fs_entity_id']].drop_duplicates('company_id', keep='first')

        company_to_entity = pd.concat([
            primary_map,
            fallback_map[~fallback_map['company_id'].isin(primary_map['company_id'])]
        ]).drop_duplicates('company_id', keep='first')

        print(f"  → 加载了 {len(company_to_entity):,} 个company_id映射")

        # 提取所有company_id
        sources = set(df_cleaned['source_company_id'].dropna().astype('int64'))
        targets = set(df_cleaned['target_company_id'].dropna().astype('int64'))
        all_company_ids = sources | targets

        # 映射到entity_id
        company_to_entity_dict = dict(zip(company_to_entity['company_id'], company_to_entity['fs_entity_id']))
        all_entity_ids = set()

        for cid in all_company_ids:
            eid = company_to_entity_dict.get(cid)
            if eid:
                all_entity_ids.add(eid)

        print(f"  ✓ 从清洗后的关系数据提取了 {len(all_entity_ids):,} 个entity_id节点")
        print(f"    （原始company_id数: {len(all_company_ids):,}，映射成功: {len(all_entity_ids):,}）")

        return all_entity_ids

    except Exception as e:
        print(f"  ✗ 加载清洗后的关系数据失败: {e}")
        import traceback
        traceback.print_exc()
        print(f"  → 将使用沙盒网络节点作为噪声池")
        return None


# ============================================================================
# 辅助函数：自动检测年份范围
# ============================================================================

def detect_available_years(sandbox_folder: str) -> List[int]:
    """
    自动检测沙盒数据文件夹中可用的年份

    Args:
        sandbox_folder: 沙盒数据文件夹路径

    Returns:
        可用年份列表（排序）
    """
    import glob
    import re

    nodes_files = glob.glob(os.path.join(sandbox_folder, "*_nodes.csv"))
    years = []

    for file_path in nodes_files:
        filename = os.path.basename(file_path)
        match = re.search(r'(\d{4})_nodes\.csv', filename)
        if match:
            years.append(int(match.group(1)))

    return sorted(years)

# ============================================================================
# 1. 沙盒网络数据加载
# ============================================================================

def load_sandbox_network(sandbox_folder: str, year: int) -> nx.DiGraph:
    """
    加载指定年份的沙盒网络图

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

    print(f"  ✓ 加载{year}年网络: {G.number_of_nodes()}个节点, {G.number_of_edges()}条边")

    return G


def load_core_decision_makers(
    sandbox_folder: str,
    start_year: int
) -> List[str]:
    """
    从第一年的nodes.csv文件中加载核心决策者（Is_Core=True的节点）

    Args:
        sandbox_folder: 沙盒数据文件夹路径
        start_year: 开始年份

    Returns:
        核心决策者ID列表
    """
    print(f"  → 从{start_year}年节点数据中加载核心决策者...")

    nodes_file = os.path.join(sandbox_folder, f"{start_year}_nodes.csv")

    if not os.path.exists(nodes_file):
        raise FileNotFoundError(f"找不到{start_year}年的节点文件: {nodes_file}")

    # 读取节点数据
    nodes_df = pd.read_csv(nodes_file)

    # 筛选Is_Core=True的节点
    core_nodes = nodes_df[nodes_df['Is_Core'] == True]

    # 提取节点ID（使用'Id'字段，与原版对齐）
    core_decision_makers = core_nodes['Id'].astype(str).tolist()

    print(f"  ✓ 加载了{len(core_decision_makers)}个核心决策者")

    return core_decision_makers


# ============================================================================
# 2. 候选池构建（演化模式）
# ============================================================================

def build_candidate_pool_for_decision_maker(
    decision_maker: str,
    year: int,
    real_graph_current: nx.DiGraph,
    evolving_graph_prev: nx.DiGraph,
    global_dataframes: Dict[str, pd.DataFrame],
    cleaned_nodes_pool: set = None
) -> pd.DataFrame:
    """
    为单个决策者构建候选池（演化模式）

    Args:
        decision_maker: 决策者ID
        year: 当前年份
        real_graph_current: 当年的真实网络图
        evolving_graph_prev: 前一年的演化网络图
        global_dataframes: 全局FactSet数据字典（用于噪声采样）
        cleaned_nodes_pool: 从清洗后关系数据提取的节点池（用于噪声抽样）

    Returns:
        候选池DataFrame，包含列：factset_entity_id, source_label
    """
    # 1. 历史供应商（来自前一年的演化网络）
    historical_suppliers = set()
    if decision_maker in evolving_graph_prev:
        historical_suppliers = set(evolving_graph_prev.predecessors(decision_maker))

    # 2. 未来机会（当年真实网络中的新供应商）
    current_suppliers = set()
    if decision_maker in real_graph_current:
        current_suppliers = set(real_graph_current.predecessors(decision_maker))

    future_opportunities = current_suppliers - historical_suppliers

    # 3. 全局噪声（优先从清洗后的关系数据采样）
    global_noise = set()
    noise_sample_size = len(historical_suppliers)  # 噪声数量 = 历史供应商数量

    # 优先使用cleaned_nodes_pool，否则使用FactSet全局数据
    if cleaned_nodes_pool is not None:
        # 使用清洗后的节点池
        excluded_entities = historical_suppliers | future_opportunities | {decision_maker}
        available_for_noise = cleaned_nodes_pool - excluded_entities

        if len(available_for_noise) > 0 and noise_sample_size > 0:
            np.random.seed(42 + hash(decision_maker + str(year)) % 1000)
            actual_sample_size = min(noise_sample_size, len(available_for_noise))
            global_noise = set(np.random.choice(
                list(available_for_noise),
                size=actual_sample_size,
                replace=False
            ))
    elif 'standard_entity' in global_dataframes and not global_dataframes['standard_entity'].empty:
        # Fallback: 从完整的FactSet数据集中随机采样全局噪声
        all_factset_entities = set(global_dataframes['standard_entity']['factset_entity_id'].dropna().unique())

        # 排除已知的候选者（历史供应商、未来机会、决策者本身）
        excluded_entities = historical_suppliers | future_opportunities | {decision_maker}
        available_for_noise = all_factset_entities - excluded_entities

        if len(available_for_noise) > 0 and noise_sample_size > 0:
            # 使用固定种子确保可复现
            np.random.seed(42 + hash(decision_maker + str(year)) % 1000)
            actual_sample_size = min(noise_sample_size, len(available_for_noise))
            global_noise = set(np.random.choice(
                list(available_for_noise),
                size=actual_sample_size,
                replace=False
            ))

    # 4. 构建DataFrame
    data = []

    # 添加历史供应商（使用复数标签与原版对齐）
    for supplier_id in historical_suppliers:
        data.append({
            'factset_entity_id': supplier_id,
            'source_label': 'historical_suppliers'
        })

    # 添加未来机会（使用复数标签与原版对齐）
    for supplier_id in future_opportunities:
        data.append({
            'factset_entity_id': supplier_id,
            'source_label': 'future_opportunities'
        })

    # 添加全局噪声
    for supplier_id in global_noise:
        data.append({
            'factset_entity_id': supplier_id,
            'source_label': 'global_noise'
        })

    candidate_df = pd.DataFrame(data)

    return candidate_df


def build_all_candidate_pools(
    core_decision_makers: List[str],
    year: int,
    sandbox_folder: str,
    evolving_graph_prev: nx.DiGraph,
    global_dataframes: Dict[str, pd.DataFrame],
    cleaned_nodes_pool: set = None
) -> Dict[str, pd.DataFrame]:
    """
    为所有核心决策者构建候选池

    Args:
        core_decision_makers: 核心决策者列表
        year: 当前年份
        sandbox_folder: 沙盒数据文件夹路径
        evolving_graph_prev: 前一年的演化网络图
        global_dataframes: 全局FactSet数据字典
        cleaned_nodes_pool: 从清洗后关系数据提取的节点池（用于噪声抽样）

    Returns:
        字典 {decision_maker_id: candidate_pool_df}
    """
    print(f"  → 为{len(core_decision_makers)}个决策者构建候选池...")

    # 加载当年真实网络
    real_graph_current = load_sandbox_network(sandbox_folder, year)

    candidate_pools = {}

    for decision_maker in core_decision_makers:
        candidate_df = build_candidate_pool_for_decision_maker(
            decision_maker,
            year,
            real_graph_current,
            evolving_graph_prev,
            global_dataframes,
            cleaned_nodes_pool
        )

        if not candidate_df.empty:
            candidate_pools[decision_maker] = candidate_df

            # 统计
            composition = candidate_df['source_label'].value_counts().to_dict()
            hist_count = composition.get('historical_suppliers', 0)
            future_count = composition.get('future_opportunities', 0)
            noise_count = composition.get('global_noise', 0)

            print(f"    • {decision_maker}: {len(candidate_df)}个候选者 "
                  f"(历史:{hist_count}, 未来:{future_count}, 噪声:{noise_count})")

    print(f"  ✓ 成功为{len(candidate_pools)}/{len(core_decision_makers)}个决策者构建了候选池")

    return candidate_pools


# ============================================================================
# 3. 前一年网络状态加载
# ============================================================================

def load_previous_evolving_graph(
    year: int,
    start_year: int,
    selection_method: str,
    sandbox_folder: str,
    result_dir: str
) -> nx.DiGraph:
    """
    加载前一年的演化网络状态

    Args:
        year: 当前年份
        start_year: 开始年份
        selection_method: 选择方法（llm/ml/random）
        sandbox_folder: 沙盒数据文件夹路径
        result_dir: 结果目录

    Returns:
        前一年的演化网络图
    """
    if year == start_year + 1:
        # 第一年：使用真实网络初始化
        print(f"  → 第一年（{year}），使用{start_year}年真实网络初始化")
        return load_sandbox_network(sandbox_folder, start_year)

    # 非第一年：加载前一年的演化网络
    prev_year = year - 1

    # 根据选择方法确定文件名
    if selection_method == 'llm':
        evolving_graph_file = os.path.join(
            result_dir,
            f"stage5a_evolving_graph_{prev_year}.pkl"
        )
    elif selection_method in ['logistic_regression', 'random_forest', 'xgboost']:
        # Stage5b为每个模型分别保存演化网络
        evolving_graph_file = os.path.join(
            result_dir,
            f"stage5b_{selection_method}_evolving_graph_{prev_year}.pkl"
        )
    elif selection_method == 'random':
        evolving_graph_file = os.path.join(
            result_dir,
            f"stage5c_random_evolving_graph_{prev_year}.pkl"
        )
    else:
        raise ValueError(f"未知的选择方法: {selection_method}")

    if not os.path.exists(evolving_graph_file):
        raise FileNotFoundError(
            f"找不到前一年的演化网络状态: {evolving_graph_file}\n"
            f"请先运行{prev_year}年的Stage 5"
        )

    print(f"  → 加载{prev_year}年的演化网络状态: {evolving_graph_file}")

    with open(evolving_graph_file, 'rb') as f:
        evolving_graph_prev = pickle.load(f)

    print(f"  ✓ 加载完成: {evolving_graph_prev.number_of_nodes()}个节点, "
          f"{evolving_graph_prev.number_of_edges()}条边")

    return evolving_graph_prev


# ============================================================================
# 4. FactSet全局数据加载
# ============================================================================

def load_all_dataframes() -> Dict[str, pd.DataFrame]:
    """
    加载所有必需的FactSet数据集（用于全局噪声采样）

    Returns:
        包含所有数据表的字典
    """
    print("  → 加载FactSet全局数据...")

    datasets = {
        'standard_entity': os.path.join(FACTSET_DATA_PATH, 'data', 'standard_entity.parquet'),
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
# 5. 主处理流程
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Stage 1: 候选池构建（演化模式）'
    )
    parser.add_argument(
        '--year',
        type=int,
        required=True,
        help='处理年份（必需）'
    )
    parser.add_argument(
        '--sandbox_folder',
        type=str,
        required=True,
        help='沙盒网络数据文件夹路径'
    )
    parser.add_argument(
        '--selection_method',
        type=str,
        default='llm',
        choices=['llm', 'logistic_regression', 'random_forest', 'xgboost', 'random'],
        help='选择方法（用于确定加载哪个演化网络，默认: llm）'
    )
    parser.add_argument(
        '--result_dir',
        type=str,
        default=RESULT_DIR,
        help='结果保存目录'
    )

    args = parser.parse_args()

    # 自动检测年份范围
    print("=" * 80)
    print("Stage 1: 候选池构建（演化模式）")
    print("=" * 80)
    print(f"沙盒文件夹: {args.sandbox_folder}")
    print("检测可用年份...")

    available_years = detect_available_years(args.sandbox_folder)

    if len(available_years) == 0:
        raise FileNotFoundError(f"在{args.sandbox_folder}中找不到任何年份的数据文件")

    print(f"  ✓ 检测到可用年份: {available_years}")

    # 自动设置 start_year（数据中最早的年份）
    start_year = min(available_years)
    print(f"  → 自动设置开始年份: {start_year} (数据最早)")

    # 自动设置 end_year（数据中最晚的年份）
    end_year = max(available_years)
    print(f"  → 自动设置结束年份: {end_year} (数据最晚)")

    # 打印配置
    print("=" * 80)
    print(f"处理年份: {args.year}")
    print(f"开始年份: {start_year} (用于初始化)")
    print(f"结束年份: {end_year} (数据范围)")
    print(f"选择方法: {args.selection_method}")
    print(f"结果目录: {args.result_dir}")
    print("=" * 80)

    # 创建结果目录
    os.makedirs(args.result_dir, exist_ok=True)

    # 检查年份有效性
    if args.year <= start_year:
        raise ValueError(
            f"处理年份（{args.year}）必须大于开始年份（{start_year}）\n"
            f"提示：演化评估从 start_year+1 开始"
        )

    if args.year > end_year:
        raise ValueError(
            f"处理年份（{args.year}）超出数据范围（最大：{end_year}）"
        )

    # 步骤1: 加载FactSet全局数据
    print(f"\n[1/4] 加载FactSet全局数据")
    global_dataframes = load_all_dataframes()

    # 加载清洗后的关系节点池（用于噪声抽样）
    cleaned_nodes_pool = load_cleaned_relationship_nodes()

    # 步骤2: 加载核心决策者
    print(f"\n[2/4] 加载核心决策者")
    core_decision_makers = load_core_decision_makers(
        args.sandbox_folder,
        start_year
    )

    if len(core_decision_makers) == 0:
        raise ValueError("未加载到任何核心决策者，请检查数据文件中的Is_Core字段")

    # 步骤3: 加载前一年的演化网络并构建候选池
    print(f"\n[3/4] 加载前一年的演化网络并构建{args.year}年的候选池")
    evolving_graph_prev = load_previous_evolving_graph(
        args.year,
        start_year,
        args.selection_method,
        args.sandbox_folder,
        args.result_dir
    )

    candidate_pools = build_all_candidate_pools(
        core_decision_makers,
        args.year,
        args.sandbox_folder,
        evolving_graph_prev,
        global_dataframes,
        cleaned_nodes_pool
    )

    # 保存结果
    output_dir = os.path.join(
        args.result_dir,
        f"stage1_candidate_pools_{args.year}"
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"保存结果到: {output_dir}")

    for decision_maker, candidate_df in candidate_pools.items():
        output_file = os.path.join(output_dir, f"{decision_maker}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(candidate_df, f)

    # 保存元数据
    metadata = {
        'year': args.year,
        'start_year': start_year,
        'end_year': end_year,
        'available_years': available_years,
        'selection_method': args.selection_method,
        'num_decision_makers': len(core_decision_makers),
        'decision_makers': core_decision_makers,
        'total_candidates': sum(len(df) for df in candidate_pools.values()),
        'timestamp': datetime.now().isoformat()
    }

    metadata_file = os.path.join(
        args.result_dir,
        f"stage1_metadata_{args.year}.json"
    )

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"✓ Stage 1 完成！")
    print(f"\n{'='*80}")
    print("统计信息")
    print(f"{'='*80}")
    print(f"处理年份: {args.year}")
    print(f"核心决策者数: {len(core_decision_makers)}")
    print(f"候选池总规模: {sum(len(df) for df in candidate_pools.values())}")
    print(f"平均候选者数: {sum(len(df) for df in candidate_pools.values()) / len(candidate_pools):.1f}")

    # 打印候选者来源统计
    all_labels = []
    for df in candidate_pools.values():
        all_labels.extend(df['source_label'].tolist())

    label_counts = pd.Series(all_labels).value_counts()
    print(f"\n候选者来源分布:")
    for label, count in label_counts.items():
        print(f"  - {label}: {count} ({count/len(all_labels)*100:.1f}%)")

    print(f"\n{'='*80}")
    print(f"下一步: 运行 stage2_profile_builder.py --year {args.year}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
