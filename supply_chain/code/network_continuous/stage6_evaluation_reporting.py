#!/usr/bin/env python3
"""
Stage 6: Evaluation and Reporting (Evolution Mode)
===================================================

功能：评估不同选择方法的效果并生成报告（演化模式）

输入：
- 真实网络（sandbox_folder/{year}_nodes.csv, {year}_edges.csv）
- stage5a_evolving_graph_{year}.pkl（LLM方法的演化网络）
- stage5b_{model}_evolving_graph_{year}.pkl（ML方法的演化网络，3个模型）
- stage5c_random_evolving_graph_{year}.pkl（Random方法的演化网络）

输出：
- stage6_evaluation_report_{year}.json
  格式：{
    'year': year,
    'methods': {
      method_name: {
        'edge_metrics': {...},  # 边级别指标（准确率、召回率、F1等）
        'network_metrics': {...}  # 网络级别指标（密度、度分布等）
      }
    }
  }

- stage6_evaluation_plots_{year}/
  ├── edge_comparison_{year}.pdf  # 边对比图
  ├── degree_distribution_{method}_{year}.pdf  # 度分布对比图
  └── ...

评估指标：
1. 边级别：
   - Precision（精确率）
   - Recall（召回率）
   - F1 Score
   - Jaccard相似度

2. 网络级别：
   - 密度差异
   - 度分布差异（KS检验）
   - 聚集系数差异
   - 核心节点重合度

参考：
- network/code/stage6_network_reconstruction.py
"""

import pandas as pd
import numpy as np
import os
import pickle
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
from scipy import stats
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 全局配置
# ============================================================================

RESULT_DIR = "../result"
PLOT_DIR = "../result/stage6_plots"

# ============================================================================
# 真实网络加载
# ============================================================================

def load_real_network(sandbox_folder: str, year: int) -> nx.DiGraph:
    """
    加载指定年份的真实网络图

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

    # 添加节点
    for _, row in nodes_df.iterrows():
        node_id = str(row['Id'])
        G.add_node(node_id)

    # 添加边
    for _, row in edges_df.iterrows():
        source = str(row['Source'])
        target = str(row['Target'])
        if source in G.nodes and target in G.nodes:
            G.add_edge(source, target)

    print(f"  ✓ 加载真实网络: {G.number_of_nodes()}个节点, {G.number_of_edges()}条边")

    return G


# ============================================================================
# 边级别评估指标
# ============================================================================

def calculate_edge_metrics(
    real_graph: nx.DiGraph,
    predicted_graph: nx.DiGraph
) -> Dict:
    """
    计算边级别的评估指标

    Args:
        real_graph: 真实网络图
        predicted_graph: 预测网络图

    Returns:
        指标字典
    """
    real_edges = set(real_graph.edges())
    predicted_edges = set(predicted_graph.edges())

    # True Positive: 预测为边且实际为边
    tp = len(real_edges & predicted_edges)

    # False Positive: 预测为边但实际不是边
    fp = len(predicted_edges - real_edges)

    # False Negative: 预测不是边但实际是边
    fn = len(real_edges - predicted_edges)

    # True Negative: 预测不是边且实际不是边（难以计算，通常不使用）
    # tn = 不计算

    # 精确率（Precision）
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # 召回率（Recall）
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1分数
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Jaccard相似度
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        'true_positive': tp,
        'false_positive': fp,
        'false_negative': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'jaccard_similarity': jaccard,
        'real_edges_count': len(real_edges),
        'predicted_edges_count': len(predicted_edges)
    }


def calculate_auc_metrics(
    real_graph: nx.DiGraph,
    hub_selections: Dict[str, Dict],
    candidate_pools_dir: str,
    year: int
) -> Dict:
    """
    计算 AUC 指标（排序质量评估）

    Args:
        real_graph: 真实网络图
        hub_selections: Hub 选择结果（包含 selection_details）
        candidate_pools_dir: 候选池目录路径
        year: 年份

    Returns:
        AUC 指标字典
    """
    auc_scores = []
    valid_hubs = 0
    skipped_hubs = 0
    skipped_reasons = {
        'no_candidates': 0,
        'no_positive_samples': 0,
        'no_negative_samples': 0,
        'hub_not_in_graph': 0,
        'no_selection_details': 0
    }

    for hub_id, hub_data in hub_selections.items():
        # 1. 检查 selection_details 或 prediction_details 是否存在
        # LLM 方法使用 selection_details，ML 方法使用 prediction_details
        selection_details = hub_data.get('selection_details') or hub_data.get('prediction_details')
        if not selection_details:
            skipped_hubs += 1
            skipped_reasons['no_selection_details'] += 1
            continue

        # 2. 加载候选池
        candidate_pool_file = os.path.join(candidate_pools_dir, f"{hub_id}.pkl")
        if not os.path.exists(candidate_pool_file):
            skipped_hubs += 1
            skipped_reasons['no_candidates'] += 1
            continue

        with open(candidate_pool_file, 'rb') as f:
            candidate_pool_df = pickle.load(f)

        if len(candidate_pool_df) == 0:
            skipped_hubs += 1
            skipped_reasons['no_candidates'] += 1
            continue

        # 3. 获取真实供应商（供应商 -> hub）
        if hub_id not in real_graph.nodes:
            skipped_hubs += 1
            skipped_reasons['hub_not_in_graph'] += 1
            continue

        real_suppliers = set(real_graph.predecessors(hub_id))

        # 4. 构建标签和预测分数
        y_true = []
        y_scores = []

        for supplier_id in candidate_pool_df['factset_entity_id']:
            if supplier_id in selection_details:
                # Ground truth: 1 if supplier in real network, 0 otherwise
                y_true.append(1 if supplier_id in real_suppliers else 0)
                # Prediction score (兼容 LLM 和 ML 方法)
                prob = selection_details[supplier_id].get('selection_probability') or \
                       selection_details[supplier_id].get('prediction_probability', 0.0)
                y_scores.append(prob)

        # 5. 检查是否至少有一个正样本和一个负样本
        if len(y_true) == 0:
            skipped_hubs += 1
            skipped_reasons['no_candidates'] += 1
            continue

        unique_labels = set(y_true)
        if len(unique_labels) < 2:
            skipped_hubs += 1
            if 1 not in unique_labels:
                skipped_reasons['no_positive_samples'] += 1
            else:
                skipped_reasons['no_negative_samples'] += 1
            continue

        # 6. 计算 AUC
        try:
            auc = roc_auc_score(y_true, y_scores)
            auc_scores.append(auc)
            valid_hubs += 1
        except Exception as e:
            print(f"    ⚠ Hub {hub_id}: AUC 计算失败 - {e}")
            skipped_hubs += 1
            continue

    # 7. 汇总结果
    if len(auc_scores) > 0:
        mean_auc = np.mean(auc_scores)
        median_auc = np.median(auc_scores)
        std_auc = np.std(auc_scores)
        min_auc = np.min(auc_scores)
        max_auc = np.max(auc_scores)
    else:
        mean_auc = median_auc = std_auc = min_auc = max_auc = None

    return {
        'mean_auc': mean_auc,
        'median_auc': median_auc,
        'std_auc': std_auc,
        'min_auc': min_auc,
        'max_auc': max_auc,
        'valid_hubs': valid_hubs,
        'skipped_hubs': skipped_hubs,
        'skipped_reasons': skipped_reasons,
        'individual_auc_scores': auc_scores
    }


# ============================================================================
# 网络级别评估指标
# ============================================================================

def calculate_network_metrics(graph: nx.DiGraph) -> Dict:
    """
    计算网络级别的指标

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

    # 度统计
    in_degrees = dict(graph.in_degree())
    out_degrees = dict(graph.out_degree())

    metrics['avg_in_degree'] = np.mean(list(in_degrees.values()))
    metrics['avg_out_degree'] = np.mean(list(out_degrees.values()))
    metrics['max_in_degree'] = max(in_degrees.values()) if in_degrees else 0
    metrics['max_out_degree'] = max(out_degrees.values()) if out_degrees else 0

    # 聚集系数
    try:
        undirected_graph = graph.to_undirected()
        metrics['average_clustering'] = nx.average_clustering(undirected_graph)
    except:
        metrics['average_clustering'] = None

    return metrics


def calculate_degree_distribution_similarity(
    real_graph: nx.DiGraph,
    predicted_graph: nx.DiGraph
) -> Dict:
    """
    计算度分布的相似度（KS检验）

    Args:
        real_graph: 真实网络图
        predicted_graph: 预测网络图

    Returns:
        相似度指标字典
    """
    # 入度分布
    real_in_degrees = list(dict(real_graph.in_degree()).values())
    pred_in_degrees = list(dict(predicted_graph.in_degree()).values())

    # 出度分布
    real_out_degrees = list(dict(real_graph.out_degree()).values())
    pred_out_degrees = list(dict(predicted_graph.out_degree()).values())

    # KS检验
    ks_in_stat, ks_in_pvalue = stats.ks_2samp(real_in_degrees, pred_in_degrees)
    ks_out_stat, ks_out_pvalue = stats.ks_2samp(real_out_degrees, pred_out_degrees)

    return {
        'ks_in_degree_statistic': ks_in_stat,
        'ks_in_degree_pvalue': ks_in_pvalue,
        'ks_out_degree_statistic': ks_out_stat,
        'ks_out_degree_pvalue': ks_out_pvalue
    }


# ============================================================================
# 可视化
# ============================================================================

def plot_edge_comparison(
    method_metrics: Dict[str, Dict],
    year: int,
    output_path: str
):
    """
    绘制边级别指标对比图

    Args:
        method_metrics: 各方法的边级别指标
        year: 年份
        output_path: 输出路径
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    methods = list(method_metrics.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))

    # 1. Precision, Recall, F1对比
    ax1 = axes[0, 0]
    x = np.arange(len(methods))
    width = 0.25

    precisions = [method_metrics[m]['edge_metrics']['precision'] for m in methods]
    recalls = [method_metrics[m]['edge_metrics']['recall'] for m in methods]
    f1_scores = [method_metrics[m]['edge_metrics']['f1_score'] for m in methods]

    ax1.bar(x - width, precisions, width, label='Precision', color='steelblue', alpha=0.8)
    ax1.bar(x, recalls, width, label='Recall', color='darkorange', alpha=0.8)
    ax1.bar(x + width, f1_scores, width, label='F1 Score', color='forestgreen', alpha=0.8)

    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title(f'Precision, Recall, F1 Score ({year})', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Jaccard相似度
    ax2 = axes[0, 1]
    jaccards = [method_metrics[m]['edge_metrics']['jaccard_similarity'] for m in methods]
    ax2.bar(methods, jaccards, color=colors, alpha=0.8)
    ax2.set_ylabel('Jaccard Similarity', fontsize=12)
    ax2.set_title(f'Jaccard Similarity ({year})', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. TP, FP, FN对比
    ax3 = axes[1, 0]
    tps = [method_metrics[m]['edge_metrics']['true_positive'] for m in methods]
    fps = [method_metrics[m]['edge_metrics']['false_positive'] for m in methods]
    fns = [method_metrics[m]['edge_metrics']['false_negative'] for m in methods]

    x = np.arange(len(methods))
    width = 0.25

    ax3.bar(x - width, tps, width, label='True Positive', color='limegreen', alpha=0.8)
    ax3.bar(x, fps, width, label='False Positive', color='crimson', alpha=0.8)
    ax3.bar(x + width, fns, width, label='False Negative', color='gold', alpha=0.8)

    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title(f'TP, FP, FN Count ({year})', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. 边数对比
    ax4 = axes[1, 1]
    real_edges = [method_metrics[m]['edge_metrics']['real_edges_count'] for m in methods]
    pred_edges = [method_metrics[m]['edge_metrics']['predicted_edges_count'] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    ax4.bar(x - width/2, real_edges, width, label='Real Edges', color='navy', alpha=0.8)
    ax4.bar(x + width/2, pred_edges, width, label='Predicted Edges', color='teal', alpha=0.8)

    ax4.set_ylabel('Edge Count', fontsize=12)
    ax4.set_title(f'Edge Count Comparison ({year})', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(methods, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. AUC 对比
    ax5 = axes[2, 0]
    auc_values = []
    auc_methods = []
    for m in methods:
        auc_metrics = method_metrics[m].get('auc_metrics')
        if auc_metrics and auc_metrics.get('mean_auc') is not None:
            auc_values.append(auc_metrics['mean_auc'])
            auc_methods.append(m)

    if len(auc_values) > 0:
        method_colors = [colors[methods.index(m)] for m in auc_methods]
        ax5.bar(auc_methods, auc_values, color=method_colors, alpha=0.8)
        ax5.set_ylabel('Mean AUC', fontsize=12)
        ax5.set_title(f'Mean AUC Score ({year})', fontsize=14, fontweight='bold')
        ax5.set_xticklabels(auc_methods, rotation=45, ha='right')
        ax5.set_ylim([0, 1])
        ax5.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random Baseline')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    else:
        ax5.text(0.5, 0.5, 'No AUC data available',
                ha='center', va='center', fontsize=14, transform=ax5.transAxes)
        ax5.set_title(f'Mean AUC Score ({year})', fontsize=14, fontweight='bold')

    # 6. 隐藏最后一个子图（或用于其他用途）
    ax6 = axes[2, 1]
    ax6.axis('off')

    # 整体标题
    fig.suptitle(f'Edge-Level Metrics Comparison ({year})',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()

    print(f"  ✓ 边对比图已保存: {output_path}")


# ============================================================================
# 主处理流程
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Stage 6: 评估和报告（演化模式）'
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
        '--result_dir',
        type=str,
        default=RESULT_DIR,
        help='结果保存目录'
    )

    args = parser.parse_args()

    # 打印配置
    print("=" * 80)
    print("Stage 6: 评估和报告（演化模式）")
    print("=" * 80)
    print(f"处理年份: {args.year}")
    print(f"沙盒文件夹: {args.sandbox_folder}")
    print(f"结果目录: {args.result_dir}")
    print("=" * 80)

    # 创建输出目录
    plot_dir = os.path.join(PLOT_DIR, str(args.year))
    os.makedirs(plot_dir, exist_ok=True)

    # 步骤1: 加载真实网络
    print(f"\n[1/3] 加载{args.year}年真实网络")
    real_graph = load_real_network(args.sandbox_folder, args.year)

    # 步骤2: 加载各种方法的演化网络并评估
    print(f"\n[2/3] 加载演化网络并评估")

    # 定义方法列表
    methods = {
        'llm': f"stage5a_evolving_graph_{args.year}.pkl",
        'logistic_regression': f"stage5b_logistic_regression_evolving_graph_{args.year}.pkl",
        'random_forest': f"stage5b_random_forest_evolving_graph_{args.year}.pkl",
        'xgboost': f"stage5b_xgboost_evolving_graph_{args.year}.pkl",
        'random': f"stage5c_random_evolving_graph_{args.year}.pkl"
    }

    all_method_results = {}

    for method_name, graph_filename in methods.items():
        graph_file = os.path.join(args.result_dir, graph_filename)

        if not os.path.exists(graph_file):
            print(f"  ⚠ {method_name}: 文件不存在，跳过 ({graph_filename})")
            continue

        print(f"\n  --- 方法: {method_name.upper()} ---")

        # 加载演化网络
        with open(graph_file, 'rb') as f:
            predicted_graph = pickle.load(f)

        print(f"    → 演化网络: {predicted_graph.number_of_nodes()}个节点, {predicted_graph.number_of_edges()}条边")

        # 计算边级别指标
        print(f"    → 计算边级别指标...")
        edge_metrics = calculate_edge_metrics(real_graph, predicted_graph)

        print(f"      Precision: {edge_metrics['precision']:.4f}")
        print(f"      Recall: {edge_metrics['recall']:.4f}")
        print(f"      F1 Score: {edge_metrics['f1_score']:.4f}")
        print(f"      Jaccard Similarity: {edge_metrics['jaccard_similarity']:.4f}")

        # 计算网络级别指标
        print(f"    → 计算网络级别指标...")
        network_metrics = calculate_network_metrics(predicted_graph)
        real_network_metrics = calculate_network_metrics(real_graph)

        # 计算度分布相似度
        degree_dist_similarity = calculate_degree_distribution_similarity(real_graph, predicted_graph)

        # 计算 AUC 指标（仅针对有 selection_probability 的方法）
        auc_metrics = None
        if method_name != 'random':  # Random 方法没有 selection_probability
            print(f"    → 计算 AUC 指标...")

            # 加载对应的选择结果文件
            if method_name == 'llm':
                selection_file = os.path.join(args.result_dir, f"stage5a_llm_selection_{args.year}.pkl")
            elif method_name in ['logistic_regression', 'random_forest', 'xgboost']:
                selection_file = os.path.join(args.result_dir, f"stage5b_{method_name}_selection_{args.year}.pkl")
            else:
                selection_file = None

            if selection_file and os.path.exists(selection_file):
                try:
                    with open(selection_file, 'rb') as f:
                        selection_data = pickle.load(f)

                    hub_selections = selection_data.get('hub_selections', {})
                    candidate_pools_dir = os.path.join(args.result_dir, f"stage1_candidate_pools_{args.year}")

                    auc_metrics = calculate_auc_metrics(
                        real_graph,
                        hub_selections,
                        candidate_pools_dir,
                        args.year
                    )

                    if auc_metrics['mean_auc'] is not None:
                        print(f"      Mean AUC: {auc_metrics['mean_auc']:.4f}")
                        print(f"      Valid Hubs: {auc_metrics['valid_hubs']}/{auc_metrics['valid_hubs'] + auc_metrics['skipped_hubs']}")
                    else:
                        print(f"      ⚠ AUC 计算失败（没有有效的 Hub）")

                except Exception as e:
                    print(f"      ⚠ AUC 计算失败: {e}")
                    auc_metrics = None
            else:
                print(f"      ⚠ 选择结果文件不存在，跳过 AUC 计算")

        # 保存结果
        all_method_results[method_name] = {
            'edge_metrics': edge_metrics,
            'network_metrics': network_metrics,
            'real_network_metrics': real_network_metrics,
            'degree_distribution_similarity': degree_dist_similarity,
            'auc_metrics': auc_metrics
        }

    # 步骤3: 生成报告和可视化
    print(f"\n[3/3] 生成报告、保存重构网络和可视化")

    # 保存评估报告（JSON）
    report = {
        'year': args.year,
        'real_network': {
            'nodes': real_graph.number_of_nodes(),
            'edges': real_graph.number_of_edges(),
            'density': nx.density(real_graph)
        },
        'methods': all_method_results
    }

    report_file = os.path.join(args.result_dir, f"stage6_evaluation_report_{args.year}.json")

    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"  ✓ 评估报告已保存: {report_file}")

    # 保存重构网络（供下一年使用）
    # 注意：虽然Stage 5已经保存了evolving_graph，但Stage 6保存的包含完整的评估信息
    reconstructed_networks = {}
    for method_name in all_method_results.keys():
        # 重新加载evolving_graph（它就是重构后的网络）
        if method_name == 'llm':
            graph_file = os.path.join(args.result_dir, f"stage5a_evolving_graph_{args.year}.pkl")
        elif method_name in ['logistic_regression', 'random_forest', 'xgboost']:
            graph_file = os.path.join(args.result_dir, f"stage5b_{method_name}_evolving_graph_{args.year}.pkl")
        elif method_name == 'random':
            graph_file = os.path.join(args.result_dir, f"stage5c_random_evolving_graph_{args.year}.pkl")
        else:
            continue

        if os.path.exists(graph_file):
            with open(graph_file, 'rb') as f:
                reconstructed_graph = pickle.load(f)
                reconstructed_networks[method_name] = {
                    'graph': reconstructed_graph,
                    'edge_metrics': all_method_results[method_name]['edge_metrics'],
                    'network_metrics': all_method_results[method_name]['network_metrics']
                }

    # 保存重构网络集合
    reconstructed_file = os.path.join(args.result_dir, f"stage6_reconstructed_networks_{args.year}.pkl")
    with open(reconstructed_file, 'wb') as f:
        pickle.dump({
            'year': args.year,
            'real_graph': real_graph,
            'reconstructed_networks': reconstructed_networks
        }, f)

    print(f"  ✓ 重构网络已保存: {reconstructed_file}")

    # 绘制边对比图
    if len(all_method_results) > 0:
        plot_output_path = os.path.join(plot_dir, f"edge_comparison_{args.year}.pdf")
        plot_edge_comparison(all_method_results, args.year, plot_output_path)

    # 打印完成信息
    print(f"\n{'='*80}")
    print(f"✓ Stage 6 完成！")
    print(f"{'='*80}")
    print(f"处理年份: {args.year}")
    print(f"评估方法数: {len(all_method_results)}")

    # 打印评估结果汇总
    print(f"\n{'='*80}")
    print("评估结果汇总")
    print(f"{'='*80}")
    print(f"{'方法':<20} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Jaccard':<12} {'Mean AUC':<12}")
    print("-" * 80)

    for method_name in sorted(all_method_results.keys()):
        metrics = all_method_results[method_name]['edge_metrics']
        auc_metrics = all_method_results[method_name].get('auc_metrics')

        mean_auc_str = f"{auc_metrics['mean_auc']:.4f}" if auc_metrics and auc_metrics['mean_auc'] is not None else "N/A"

        print(f"{method_name:<20} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1_score']:<12.4f} {metrics['jaccard_similarity']:<12.4f} {mean_auc_str:<12}")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
