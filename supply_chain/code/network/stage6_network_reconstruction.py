#!/usr/bin/env python3
"""
Stage 6: Network Reconstruction and Analysis
=============================================

功能：基于Stage5的选择结果重构网络，并进行可视化和指标分析

输入：
- Stage1: original_graph, hub_nodes
- Stage5a: llm_selections
- Stage5b: ml_selections (3个模型)
- Stage5c: random_selections

输出：
- stage6_reconstructed_networks_*.pkl (重构的网络图)
- stage6_degree_distributions_*.pdf (度分布图)
- stage6_network_metrics_*.csv (网络指标 - 原始&重构)

网络重构逻辑：
1. 复制原始网络
2. 断开所有Hub的入边（供应商 -> Hub）
3. 根据Stage5选择结果重新连接

可视化：
- 入度/出度分布（log-log scale）
- Power-law拟合曲线

网络指标：
- 密度、平均路径长度、直径
- 聚集系数、PageRank、Hub Score、Coreness
"""

import pandas as pd
import numpy as np
import os
import pickle
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter
from scipy import stats
from scipy.stats import ks_2samp
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 全局配置
# ============================================================================

STAGE1_RESULT_DIR = "../result/20"
STAGE5_RESULT_DIR = "../result/20"
RESULT_DIR = "../result/20"
PLOT_DIR = "../result/20/stage6_plots"

# ============================================================================
# 网络重构
# ============================================================================

def reconstruct_network_for_method(
    original_graph: nx.DiGraph,
    hub_nodes: List[str],
    hub_selections: Dict[str, Dict],
    year: int
) -> Tuple[nx.DiGraph, Dict]:
    """
    根据选择结果重构网络

    Args:
        original_graph: 原始网络图
        hub_nodes: Hub节点列表
        hub_selections: Hub选择结果 {hub_id: {'selected_suppliers': [...]}}
        year: 年份

    Returns:
        (重构的网络图, 统计信息)
    """
    # 复制原始网络
    reconstructed_graph = original_graph.copy()

    # 统计信息
    stats = {
        'total_hubs': len(hub_nodes),
        'total_edges_removed': 0,
        'total_edges_added': 0,
        'hubs_with_selections': 0
    }

    # 1. 断开所有Hub的入边
    edges_to_remove = []
    for hub_id in hub_nodes:
        if hub_id in reconstructed_graph:
            # 获取所有指向hub的边
            in_edges = list(reconstructed_graph.in_edges(hub_id))
            edges_to_remove.extend(in_edges)

    reconstructed_graph.remove_edges_from(edges_to_remove)
    stats['total_edges_removed'] = len(edges_to_remove)

    # 2. 根据Stage5选择结果重新连接
    edges_to_add = []
    for hub_id in hub_nodes:
        if hub_id not in hub_selections:
            continue

        selection_result = hub_selections[hub_id]
        selected_suppliers = selection_result.get('selected_suppliers', [])

        if len(selected_suppliers) == 0:
            continue

        stats['hubs_with_selections'] += 1

        # 添加新边：supplier -> hub
        for supplier_id in selected_suppliers:
            # 确保supplier节点存在于图中
            if supplier_id in reconstructed_graph:
                edges_to_add.append((supplier_id, hub_id))

    reconstructed_graph.add_edges_from(edges_to_add)
    stats['total_edges_added'] = len(edges_to_add)

    return reconstructed_graph, stats


# ============================================================================
# Power-law拟合（参考network_test.py）
# ============================================================================

def calculate_power_law_parameters(degrees: List[int]) -> Dict:
    """
    计算度分布的power-law参数（对齐network_test.py）

    Args:
        degrees: 度序列

    Returns:
        {'alpha': ..., 'xmin': ..., 'pvalue': ...}
    """
    # 降低最小数据点要求，确保总是尝试拟合
    if len(degrees) < 2:
        return {'alpha': None, 'xmin': None, 'pvalue': None}

    degrees = [d for d in degrees if d > 0]
    if len(degrees) < 2:
        return {'alpha': None, 'xmin': None, 'pvalue': None}

    # 使用最小值作为xmin（简化版）
    xmin = min(degrees)

    # 计算度的计数
    degree_counts = Counter(degrees)
    valid_degrees = sorted([d for d in degree_counts.keys() if d >= xmin])

    if len(valid_degrees) < 2:
        return {'alpha': None, 'xmin': None, 'pvalue': None}

    log_x = np.log10(valid_degrees)
    log_y = np.log10([degree_counts[d] for d in valid_degrees])

    # 线性回归
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
        alpha = -slope
        
        return {
            'alpha': alpha,
            'xmin': xmin,
            'pvalue': p_value
        }
    except Exception as e:
        # 如果回归失败，返回None
        return {'alpha': None, 'xmin': None, 'pvalue': None}


# ============================================================================
# 度分布可视化（分离为入度和出度两个独立函数）
# ============================================================================

def plot_in_degree_distribution(
    original_graph: nx.DiGraph,
    reconstructed_graph: nx.DiGraph,
    method_name: str,
    year: int,
    output_path: str
) -> Dict:
    """
    绘制入度分布对比图（log-log scale + power-law拟合）

    Args:
        original_graph: 原始网络
        reconstructed_graph: 重构网络
        method_name: 方法名称
        year: 年份
        output_path: 输出路径

    Returns:
        包含度分布数据和拟合参数的字典
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # 原始网络入度
    original_in_degrees = dict(original_graph.in_degree())
    original_in_values = list(original_in_degrees.values())
    original_in_counts = Counter(original_in_values)
    original_in_degrees_sorted = sorted(original_in_counts.keys())
    original_in_counts_sorted = [original_in_counts[d] for d in original_in_degrees_sorted]

    # 重构网络入度
    reconstructed_in_degrees = dict(reconstructed_graph.in_degree())
    reconstructed_in_values = list(reconstructed_in_degrees.values())
    reconstructed_in_counts = Counter(reconstructed_in_values)
    reconstructed_in_degrees_sorted = sorted(reconstructed_in_counts.keys())
    reconstructed_in_counts_sorted = [reconstructed_in_counts[d] for d in reconstructed_in_degrees_sorted]

    # 绘制原始网络
    ax.loglog(original_in_degrees_sorted, original_in_counts_sorted,
              'bo-', label='Original Network', markersize=8, linewidth=2.5, alpha=0.7)

    # 绘制重构网络
    ax.loglog(reconstructed_in_degrees_sorted, reconstructed_in_counts_sorted,
              'ro-', label='Reconstructed', markersize=8, linewidth=2.5, alpha=0.7)

    # Power-law拟合 - 原始网络
    original_in_powerlaw = calculate_power_law_parameters(original_in_values)
    if original_in_powerlaw['alpha'] is not None:
        alpha = original_in_powerlaw['alpha']
        xmin = original_in_powerlaw['xmin']
        total_nodes = len(original_in_values)

        x_fit = np.logspace(np.log10(xmin), np.log10(max(original_in_values)), 100)
        C = total_nodes * (alpha - 1) * (xmin ** (alpha - 1))
        y_fit = C * (x_fit ** (-alpha))

        ax.loglog(x_fit, y_fit, 'b--', linewidth=2.5, label='Original Fit')

    # Power-law拟合 - 重构网络
    reconstructed_in_powerlaw = calculate_power_law_parameters(reconstructed_in_values)
    if reconstructed_in_powerlaw['alpha'] is not None:
        alpha = reconstructed_in_powerlaw['alpha']
        xmin = reconstructed_in_powerlaw['xmin']
        total_nodes = len(reconstructed_in_values)

        x_fit = np.logspace(np.log10(xmin), np.log10(max(reconstructed_in_values)), 100)
        C = total_nodes * (alpha - 1) * (xmin ** (alpha - 1))
        y_fit = C * (x_fit ** (-alpha))

        ax.loglog(x_fit, y_fit, 'r--', linewidth=2.5, label='Reconstructed Fit')

    # 设置坐标轴标签和图例（字体加大）
    ax.set_xlabel('In-Degree', fontsize=32)
    ax.set_ylabel('Count', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.legend(fontsize=28, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()

    print(f"  ✓ 入度分布图已保存: {output_path}")

    # 计算KS散度（比较原始和重构网络的入度分布）
    ks_statistic, ks_pvalue = ks_2samp(original_in_values, reconstructed_in_values)
    print(f"      → KS散度: {ks_statistic:.6f}, p-value: {ks_pvalue:.6f}")

    # 返回度分布数据
    return {
        'original': {
            'degrees': original_in_degrees_sorted,
            'counts': original_in_counts_sorted,
            'powerlaw': original_in_powerlaw
        },
        'reconstructed': {
            'degrees': reconstructed_in_degrees_sorted,
            'counts': reconstructed_in_counts_sorted,
            'powerlaw': reconstructed_in_powerlaw
        },
        'ks_statistic': ks_statistic,
        'ks_pvalue': ks_pvalue
    }


def plot_out_degree_distribution(
    original_graph: nx.DiGraph,
    reconstructed_graph: nx.DiGraph,
    method_name: str,
    year: int,
    output_path: str
) -> Dict:
    """
    绘制出度分布对比图（log-log scale + power-law拟合）

    Args:
        original_graph: 原始网络
        reconstructed_graph: 重构网络
        method_name: 方法名称
        year: 年份
        output_path: 输出路径

    Returns:
        包含度分布数据和拟合参数的字典
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # 原始网络出度
    original_out_degrees = dict(original_graph.out_degree())
    original_out_values = list(original_out_degrees.values())
    original_out_counts = Counter(original_out_values)
    original_out_degrees_sorted = sorted(original_out_counts.keys())
    original_out_counts_sorted = [original_out_counts[d] for d in original_out_degrees_sorted]

    # 重构网络出度
    reconstructed_out_degrees = dict(reconstructed_graph.out_degree())
    reconstructed_out_values = list(reconstructed_out_degrees.values())
    reconstructed_out_counts = Counter(reconstructed_out_values)
    reconstructed_out_degrees_sorted = sorted(reconstructed_out_counts.keys())
    reconstructed_out_counts_sorted = [reconstructed_out_counts[d] for d in reconstructed_out_degrees_sorted]

    # 绘制原始网络
    ax.loglog(original_out_degrees_sorted, original_out_counts_sorted,
              'bo-', label='Original Network', markersize=8, linewidth=2.5, alpha=0.7)

    # 绘制重构网络
    ax.loglog(reconstructed_out_degrees_sorted, reconstructed_out_counts_sorted,
              'ro-', label='Reconstructed', markersize=8, linewidth=2.5, alpha=0.7)

    # Power-law拟合 - 原始网络
    original_out_powerlaw = calculate_power_law_parameters(original_out_values)
    if original_out_powerlaw['alpha'] is not None:
        alpha = original_out_powerlaw['alpha']
        xmin = original_out_powerlaw['xmin']
        total_nodes = len(original_out_values)

        x_fit = np.logspace(np.log10(xmin), np.log10(max(original_out_values)), 100)
        C = total_nodes * (alpha - 1) * (xmin ** (alpha - 1))
        y_fit = C * (x_fit ** (-alpha))

        ax.loglog(x_fit, y_fit, 'b--', linewidth=2.5, label='Original Fit')

    # Power-law拟合 - 重构网络
    reconstructed_out_powerlaw = calculate_power_law_parameters(reconstructed_out_values)
    if reconstructed_out_powerlaw['alpha'] is not None:
        alpha = reconstructed_out_powerlaw['alpha']
        xmin = reconstructed_out_powerlaw['xmin']
        total_nodes = len(reconstructed_out_values)

        x_fit = np.logspace(np.log10(xmin), np.log10(max(reconstructed_out_values)), 100)
        C = total_nodes * (alpha - 1) * (xmin ** (alpha - 1))
        y_fit = C * (x_fit ** (-alpha))

        ax.loglog(x_fit, y_fit, 'r--', linewidth=2.5, label='Reconstructed Fit')

    # 设置坐标轴标签和图例（字体加大）
    ax.set_xlabel('Out-Degree', fontsize=32)
    ax.set_ylabel('Count', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.legend(fontsize=28, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()

    print(f"  ✓ 出度分布图已保存: {output_path}")

    # 计算KS散度（比较原始和重构网络的出度分布）
    ks_statistic, ks_pvalue = ks_2samp(original_out_values, reconstructed_out_values)
    print(f"      → KS散度: {ks_statistic:.6f}, p-value: {ks_pvalue:.6f}")

    # 返回度分布数据
    return {
        'original': {
            'degrees': original_out_degrees_sorted,
            'counts': original_out_counts_sorted,
            'powerlaw': original_out_powerlaw
        },
        'reconstructed': {
            'degrees': reconstructed_out_degrees_sorted,
            'counts': reconstructed_out_counts_sorted,
            'powerlaw': reconstructed_out_powerlaw
        },
        'ks_statistic': ks_statistic,
        'ks_pvalue': ks_pvalue
    }


# ============================================================================
# 网络指标计算
# ============================================================================

def calculate_network_metrics(graph: nx.DiGraph) -> Dict:
    """
    计算网络指标

    Args:
        graph: NetworkX图

    Returns:
        指标字典
    """
    metrics = {}

    # 基本指标
    metrics['num_nodes'] = graph.number_of_nodes()
    metrics['num_edges'] = graph.number_of_edges()
    metrics['density'] = nx.density(graph)

    # 聚集系数（无向图）
    undirected_graph = graph.to_undirected()
    metrics['average_clustering'] = nx.average_clustering(undirected_graph)

    # 连通性指标（针对最大弱连通分量）
    if nx.is_weakly_connected(graph):
        largest_wcc = graph
    else:
        largest_wcc = graph.subgraph(max(nx.weakly_connected_components(graph), key=len))

    metrics['largest_wcc_nodes'] = largest_wcc.number_of_nodes()
    metrics['largest_wcc_edges'] = largest_wcc.number_of_edges()

    # 路径长度和直径（仅对最大弱连通分量）
    try:
        if largest_wcc.number_of_nodes() > 1:
            # 转为无向图计算
            largest_wcc_undirected = largest_wcc.to_undirected()

            if nx.is_connected(largest_wcc_undirected):
                metrics['average_shortest_path_length'] = nx.average_shortest_path_length(largest_wcc_undirected)
                metrics['diameter'] = nx.diameter(largest_wcc_undirected)
            else:
                metrics['average_shortest_path_length'] = None
                metrics['diameter'] = None
        else:
            metrics['average_shortest_path_length'] = None
            metrics['diameter'] = None
    except Exception as e:
        print(f"    [WARNING] 路径长度/直径计算失败: {e}")
        metrics['average_shortest_path_length'] = None
        metrics['diameter'] = None

    # 中心性指标（采样计算，避免过慢）
    try:
        # PageRank
        pagerank = nx.pagerank(graph, max_iter=100)
        metrics['pagerank_mean'] = np.mean(list(pagerank.values()))
        metrics['pagerank_max'] = max(pagerank.values())

        # Hub Score (HITS算法)
        hubs, authorities = nx.hits(graph, max_iter=100)
        metrics['hub_score_mean'] = np.mean(list(hubs.values()))
        metrics['hub_score_max'] = max(hubs.values())

    except Exception as e:
        print(f"    [WARNING] 中心性指标计算失败: {e}")
        metrics['pagerank_mean'] = None
        metrics['pagerank_max'] = None
        metrics['hub_score_mean'] = None
        metrics['hub_score_max'] = None

    # Coreness
    try:
        core_numbers = nx.core_number(undirected_graph)
        metrics['coreness_mean'] = np.mean(list(core_numbers.values()))
        metrics['coreness_max'] = max(core_numbers.values())
    except Exception as e:
        print(f"    [WARNING] Coreness计算失败: {e}")
        metrics['coreness_mean'] = None
        metrics['coreness_max'] = None

    return metrics


# ============================================================================
# 主流程
# ============================================================================

def process_single_year(
    year: int,
    stage1_data: Dict,
    stage5_data: Dict[str, Dict]
) -> Dict:
    """
    处理单个年份（所有方法）

    Args:
        year: 年份
        stage1_data: Stage1数据
        stage5_data: Stage5数据（5个方法）

    Returns:
        结果字典
    """
    print(f"\n{'='*80}")
    print(f"处理 {year} 年")
    print(f"{'='*80}")

    # 获取原始网络和Hub节点
    original_graph = stage1_data['original_graph']
    hub_nodes = stage1_data['hub_nodes']

    print(f"  原始网络: {original_graph.number_of_nodes():,} 节点, {original_graph.number_of_edges():,} 边")
    print(f"  Hub节点数: {len(hub_nodes)}")

    # 处理每种方法
    method_results = {}

    for method_name, method_stage5_data in stage5_data.items():
        print(f"\n  --- 方法: {method_name.upper()} ---")

        # 获取该方法的选择结果
        year_data = method_stage5_data.get(year)
        if year_data is None:
            print(f"    ⚠ 未找到 {year} 年的数据，跳过")
            continue

        hub_selections = year_data.get('hub_selections', {})

        # 1. 重构网络
        print(f"    [1/3] 重构网络...")
        reconstructed_graph, recon_stats = reconstruct_network_for_method(
            original_graph,
            hub_nodes,
            hub_selections,
            year
        )

        print(f"      → 断开边数: {recon_stats['total_edges_removed']:,}")
        print(f"      → 新增边数: {recon_stats['total_edges_added']:,}")
        print(f"      → 有选择的Hub数: {recon_stats['hubs_with_selections']}/{recon_stats['total_hubs']}")

        # 2. 绘制度分布图（分离为入度和出度）
        print(f"    [2/3] 绘制度分布图...")
        
        # 入度分布图
        in_degree_plot_path = os.path.join(
            PLOT_DIR,
            f"in_degree_distribution_{method_name}_{year}.pdf"
        )
        in_degree_data = plot_in_degree_distribution(
            original_graph,
            reconstructed_graph,
            method_name,
            year,
            in_degree_plot_path
        )
        
        # 出度分布图
        out_degree_plot_path = os.path.join(
            PLOT_DIR,
            f"out_degree_distribution_{method_name}_{year}.pdf"
        )
        out_degree_data = plot_out_degree_distribution(
            original_graph,
            reconstructed_graph,
            method_name,
            year,
            out_degree_plot_path
        )

        # 3. 计算网络指标
        print(f"    [3/3] 计算网络指标...")
        original_metrics = calculate_network_metrics(original_graph)
        reconstructed_metrics = calculate_network_metrics(reconstructed_graph)

        print(f"      → 原始网络密度: {original_metrics['density']:.6f}")
        print(f"      → 重构网络密度: {reconstructed_metrics['density']:.6f}")

        # 保存结果
        method_results[method_name] = {
            'reconstructed_graph': reconstructed_graph,
            'reconstruction_stats': recon_stats,
            'original_metrics': original_metrics,
            'reconstructed_metrics': reconstructed_metrics,
            'in_degree_plot_path': in_degree_plot_path,
            'out_degree_plot_path': out_degree_plot_path,
            'in_degree_data': in_degree_data,
            'out_degree_data': out_degree_data
        }

    return {
        'year': year,
        'method_results': method_results
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Stage 6: 网络重构与分析')
    parser.add_argument('--stage1_file', type=str, default=None)
    parser.add_argument('--stage5a_file', type=str, default=None)
    parser.add_argument('--stage5b_lr_file', type=str, default=None)
    parser.add_argument('--stage5b_rf_file', type=str, default=None)
    parser.add_argument('--stage5b_xgb_file', type=str, default=None)
    parser.add_argument('--stage5c_file', type=str, default=None)
    parser.add_argument('--result_dir', type=str, default=RESULT_DIR)

    args = parser.parse_args()

    print("=" * 80)
    print("Stage 6: 网络重构与分析")
    print("=" * 80)

    # 创建输出目录
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ========================================
    # 1. 查找并加载输入文件
    # ========================================

    # Stage1
    if args.stage1_file:
        stage1_file = args.stage1_file
    else:
        stage1_files = list(Path(STAGE1_RESULT_DIR).glob("stage1_candidate_pools_*.pkl"))
        if not stage1_files:
            print("❌ 未找到Stage1输出文件")
            return
        stage1_file = str(sorted(stage1_files)[-1])

    # Stage5a (LLM)
    if args.stage5a_file:
        stage5a_file = args.stage5a_file
    else:
        stage5a_files = list(Path(STAGE5_RESULT_DIR).glob("stage5a_llm_selections_*.pkl"))
        if not stage5a_files:
            print("❌ 未找到Stage5a输出文件")
            return
        stage5a_file = str(sorted(stage5a_files)[-1])

    # Stage5b - Logistic Regression
    if args.stage5b_lr_file:
        stage5b_lr_file = args.stage5b_lr_file
    else:
        stage5b_lr_files = list(Path(STAGE5_RESULT_DIR).glob("stage5b_logistic_regression_selections_*.pkl"))
        if not stage5b_lr_files:
            print("❌ 未找到Stage5b LR输出文件")
            return
        stage5b_lr_file = str(sorted(stage5b_lr_files)[-1])

    # Stage5b - Random Forest
    if args.stage5b_rf_file:
        stage5b_rf_file = args.stage5b_rf_file
    else:
        stage5b_rf_files = list(Path(STAGE5_RESULT_DIR).glob("stage5b_random_forest_selections_*.pkl"))
        if not stage5b_rf_files:
            print("❌ 未找到Stage5b RF输出文件")
            return
        stage5b_rf_file = str(sorted(stage5b_rf_files)[-1])

    # Stage5b - XGBoost
    if args.stage5b_xgb_file:
        stage5b_xgb_file = args.stage5b_xgb_file
    else:
        stage5b_xgb_files = list(Path(STAGE5_RESULT_DIR).glob("stage5b_xgboost_selections_*.pkl"))
        if not stage5b_xgb_files:
            print("❌ 未找到Stage5b XGBoost输出文件")
            return
        stage5b_xgb_file = str(sorted(stage5b_xgb_files)[-1])

    # Stage5c (Random)
    if args.stage5c_file:
        stage5c_file = args.stage5c_file
    else:
        stage5c_files = list(Path(STAGE5_RESULT_DIR).glob("stage5c_random_selections_*.pkl"))
        if not stage5c_files:
            print("❌ 未找到Stage5c输出文件")
            return
        stage5c_file = str(sorted(stage5c_files)[-1])

    print(f"\n输入文件:")
    print(f"  Stage1: {stage1_file}")
    print(f"  Stage5a (LLM): {stage5a_file}")
    print(f"  Stage5b (LR): {stage5b_lr_file}")
    print(f"  Stage5b (RF): {stage5b_rf_file}")
    print(f"  Stage5b (XGB): {stage5b_xgb_file}")
    print(f"  Stage5c (Random): {stage5c_file}")

    # ========================================
    # 2. 加载所有数据
    # ========================================

    print(f"\n加载数据...")

    with open(stage1_file, 'rb') as f:
        stage1_results = pickle.load(f)

    with open(stage5a_file, 'rb') as f:
        stage5a_results = pickle.load(f)

    with open(stage5b_lr_file, 'rb') as f:
        stage5b_lr_results = pickle.load(f)

    with open(stage5b_rf_file, 'rb') as f:
        stage5b_rf_results = pickle.load(f)

    with open(stage5b_xgb_file, 'rb') as f:
        stage5b_xgb_results = pickle.load(f)

    with open(stage5c_file, 'rb') as f:
        stage5c_results = pickle.load(f)

    # 整合Stage5数据
    stage5_all_methods = {
        'llm': stage5a_results,
        'logistic_regression': stage5b_lr_results,
        'random_forest': stage5b_rf_results,
        'xgboost': stage5b_xgb_results,
        'random': stage5c_results
    }

    # 确定年份
    years = sorted(stage1_results.keys())
    print(f"✓ 共有 {len(years)} 个年份: {years}")

    # ========================================
    # 3. 处理所有年份
    # ========================================

    all_year_results = {}

    for year in years:
        try:
            year_result = process_single_year(
                year,
                stage1_results[year],
                stage5_all_methods
            )
            all_year_results[year] = year_result

        except Exception as e:
            print(f"\n❌ {year} 年处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ========================================
    # 4. 保存结果
    # ========================================

    stage1_basename = os.path.basename(stage1_file)
    hub_pct_match = stage1_basename.split('_hub')[-1].split('pct')[0] if '_hub' in stage1_basename else '20'

    # 保存重构网络
    output_file = os.path.join(
        args.result_dir,
        f'stage6_reconstructed_networks_hub{hub_pct_match}pct.pkl'
    )

    print(f"\n{'='*80}")
    print(f"保存重构网络到: {output_file}")

    with open(output_file, 'wb') as f:
        pickle.dump(all_year_results, f)

    # 保存网络指标到CSV
    metrics_records = []

    for year, year_result in all_year_results.items():
        for method_name, method_result in year_result['method_results'].items():
            # 原始网络指标
            original_metrics = method_result['original_metrics']
            record_original = {
                'year': year,
                'method': method_name,
                'network_type': 'original',
                **original_metrics
            }
            metrics_records.append(record_original)

            # 重构网络指标
            reconstructed_metrics = method_result['reconstructed_metrics']
            record_reconstructed = {
                'year': year,
                'method': method_name,
                'network_type': 'reconstructed',
                **reconstructed_metrics
            }
            metrics_records.append(record_reconstructed)

    metrics_df = pd.DataFrame(metrics_records)
    metrics_csv_path = os.path.join(
        args.result_dir,
        f'stage6_network_metrics_hub{hub_pct_match}pct.csv'
    )

    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"✓ 网络指标已保存: {metrics_csv_path}")

    # 保存度分布数据到CSV
    degree_distribution_records = []

    for year, year_result in all_year_results.items():
        for method_name, method_result in year_result['method_results'].items():
            # 入度分布 - 原始网络
            in_degree_data = method_result['in_degree_data']
            original_in_powerlaw = in_degree_data['original']['powerlaw']
            degree_distribution_records.append({
                'year': year,
                'method': method_name,
                'network_type': 'original',
                'degree_type': 'in_degree',
                'alpha': original_in_powerlaw['alpha'],
                'pvalue': original_in_powerlaw['pvalue'],
                'xmin': original_in_powerlaw['xmin'],
                'ks_statistic': None,  # 原始网络不需要KS散度
                'ks_pvalue': None
            })

            # 入度分布 - 重构网络
            reconstructed_in_powerlaw = in_degree_data['reconstructed']['powerlaw']
            degree_distribution_records.append({
                'year': year,
                'method': method_name,
                'network_type': 'reconstructed',
                'degree_type': 'in_degree',
                'alpha': reconstructed_in_powerlaw['alpha'],
                'pvalue': reconstructed_in_powerlaw['pvalue'],
                'xmin': reconstructed_in_powerlaw['xmin'],
                'ks_statistic': in_degree_data['ks_statistic'],
                'ks_pvalue': in_degree_data['ks_pvalue']
            })

            # 出度分布 - 原始网络
            out_degree_data = method_result['out_degree_data']
            original_out_powerlaw = out_degree_data['original']['powerlaw']
            degree_distribution_records.append({
                'year': year,
                'method': method_name,
                'network_type': 'original',
                'degree_type': 'out_degree',
                'alpha': original_out_powerlaw['alpha'],
                'pvalue': original_out_powerlaw['pvalue'],
                'xmin': original_out_powerlaw['xmin'],
                'ks_statistic': None,  # 原始网络不需要KS散度
                'ks_pvalue': None
            })

            # 出度分布 - 重构网络
            reconstructed_out_powerlaw = out_degree_data['reconstructed']['powerlaw']
            degree_distribution_records.append({
                'year': year,
                'method': method_name,
                'network_type': 'reconstructed',
                'degree_type': 'out_degree',
                'alpha': reconstructed_out_powerlaw['alpha'],
                'pvalue': reconstructed_out_powerlaw['pvalue'],
                'xmin': reconstructed_out_powerlaw['xmin'],
                'ks_statistic': out_degree_data['ks_statistic'],
                'ks_pvalue': out_degree_data['ks_pvalue']
            })

    degree_distribution_df = pd.DataFrame(degree_distribution_records)
    degree_distribution_csv_path = os.path.join(
        args.result_dir,
        f'stage6_degree_distributions_hub{hub_pct_match}pct.csv'
    )
    degree_distribution_df.to_csv(degree_distribution_csv_path, index=False)
    print(f"✓ 度分布数据已保存: {degree_distribution_csv_path}")

    print(f"\n✓ Stage 6 完成！")

    # ========================================
    # 5. 统计汇总
    # ========================================

    print(f"\n{'='*80}")
    print("汇总统计")
    print(f"{'='*80}")
    print(f"成功处理年份: {len(all_year_results)}/{len(years)}\n")

    print("年份 | 方法 | 原始边数 | 重构边数 | 密度(原始) | 密度(重构)")
    print("-" * 80)

    for year in sorted(all_year_results.keys()):
        year_result = all_year_results[year]
        for method_name, method_result in sorted(year_result['method_results'].items()):
            original_edges = method_result['original_metrics']['num_edges']
            reconstructed_edges = method_result['reconstructed_metrics']['num_edges']
            original_density = method_result['original_metrics']['density']
            reconstructed_density = method_result['reconstructed_metrics']['density']

            print(f"{year} | {method_name:20s} | {original_edges:>8,} | {reconstructed_edges:>8,} | "
                  f"{original_density:.6f} | {reconstructed_density:.6f}")


if __name__ == "__main__":
    main()
