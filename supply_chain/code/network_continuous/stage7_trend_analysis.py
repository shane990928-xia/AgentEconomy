#!/usr/bin/env python3
"""
Stage 7: Trend Analysis (Evolution Mode)
=========================================

功能：跨年份趋势分析和可视化

输入：
- 多个年份的 stage6_evaluation_report_{year}.json

输出：
- stage7_trend_report.json（汇总报告）
- stage7_trend_plots/
  ├── performance_trends.pdf  # 性能指标趋势（Precision, Recall, F1, AUC）
  ├── network_size_trends.pdf  # 网络规模趋势（节点数、边数）
  ├── edge_metrics_trends.pdf  # 边指标趋势（TP, FP, FN）
  ├── jaccard_trends.pdf  # Jaccard相似度趋势
  └── method_comparison_heatmap.pdf  # 方法对比热力图

参考：
- stage6_evaluation_reporting.py
"""

import pandas as pd
import numpy as np
import os
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 全局配置
# ============================================================================

RESULT_DIR = "../result"

# 方法显示名称映射
METHOD_DISPLAY_NAMES = {
    'llm': 'Ours',
    'logistic_regression': 'Logistic Regression',
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost',
    'random': 'Random Baseline'
}

# 方法颜色映射
METHOD_COLORS = {
    'llm': '#FF6B6B',
    'logistic_regression': '#4ECDC4',
    'random_forest': '#45B7D1',
    'xgboost': '#FFA07A',
    'random': '#95A5A6'
}

# ============================================================================
# 数据加载
# ============================================================================

def load_evaluation_reports(result_dir: str, years: List[int]) -> Dict:
    """
    加载多个年份的评估报告
    
    Args:
        result_dir: 结果目录
        years: 年份列表
        
    Returns:
        年份 -> 报告数据的字典
    """
    reports = {}
    
    for year in years:
        report_file = os.path.join(result_dir, f"stage6_evaluation_report_{year}.json")
        
        if not os.path.exists(report_file):
            print(f"  ⚠ {year}年的评估报告不存在，跳过")
            continue
            
        with open(report_file, 'r') as f:
            reports[year] = json.load(f)
            
        print(f"  ✓ 加载 {year} 年评估报告")
    
    return reports


# ============================================================================
# 数据提取
# ============================================================================

def extract_trend_data(reports: Dict) -> pd.DataFrame:
    """
    从评估报告中提取趋势数据
    
    Args:
        reports: 年份 -> 报告数据的字典
        
    Returns:
        DataFrame with columns: year, method, metric_name, value
    """
    records = []
    
    for year, report in reports.items():
        methods_data = report.get('methods', {})
        
        for method_name, method_data in methods_data.items():
            # 边级别指标
            edge_metrics = method_data.get('edge_metrics', {})
            for metric_name, value in edge_metrics.items():
                if isinstance(value, (int, float)):
                    records.append({
                        'year': year,
                        'method': method_name,
                        'metric': f'edge_{metric_name}',
                        'value': value
                    })
            
            # 网络级别指标
            network_metrics = method_data.get('network_metrics', {})
            for metric_name, value in network_metrics.items():
                if isinstance(value, (int, float)):
                    records.append({
                        'year': year,
                        'method': method_name,
                        'metric': f'network_{metric_name}',
                        'value': value
                    })
            
            # AUC 指标
            auc_metrics = method_data.get('auc_metrics')
            if auc_metrics and auc_metrics.get('mean_auc') is not None:
                records.append({
                    'year': year,
                    'method': method_name,
                    'metric': 'auc_mean_auc',
                    'value': auc_metrics['mean_auc']
                })
                records.append({
                    'year': year,
                    'method': method_name,
                    'metric': 'auc_valid_hubs',
                    'value': auc_metrics['valid_hubs']
                })
        
        # 真实网络指标
        real_network = report.get('real_network', {})
        for metric_name, value in real_network.items():
            if isinstance(value, (int, float)):
                records.append({
                    'year': year,
                    'method': 'real',
                    'metric': f'real_{metric_name}',
                    'value': value
                })
    
    df = pd.DataFrame(records)
    return df


# ============================================================================
# 可视化：性能指标趋势
# ============================================================================

def plot_performance_trends(df: pd.DataFrame, output_dir: str):
    """
    绘制性能指标趋势图（Precision, Recall, F1, AUC）- 每个指标单独保存
    """
    metrics = [
        ('edge_precision', 'Precision'),
        ('edge_recall', 'Recall'),
        ('edge_f1_score', 'F1 Score'),
        ('auc_mean_auc', 'Mean AUC')
    ]
    
    for metric_name, display_name in metrics:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        metric_df = df[df['metric'] == metric_name]
        
        for method in sorted(metric_df['method'].unique()):
            method_df = metric_df[metric_df['method'] == method]
            method_df = method_df.sort_values('year')
            
            ax.plot(
                method_df['year'],
                method_df['value'],
                marker='o',
                linewidth=2.5,
                markersize=8,
                label=METHOD_DISPLAY_NAMES.get(method, method),
                color=METHOD_COLORS.get(method, None),
                alpha=0.85
            )
        
        ax.set_xlabel('Year', fontsize=24, fontweight='bold')
        ax.set_ylabel(display_name, fontsize=24, fontweight='bold')
        ax.set_title(f'{display_name}', fontsize=28, fontweight='bold')
        ax.legend(loc='best', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=25)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # y 轴范围自动调整，根据数据变化
        
        # 设置 x 轴为整数年份
        years = sorted(metric_df['year'].unique())
        ax.set_xticks(years)
        ax.set_xticklabels([int(y) for y in years])
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{metric_name}.pdf")
        plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  ✓ {display_name} 趋势图已保存: {output_path}")


# ============================================================================
# 可视化：网络规模趋势
# ============================================================================

def plot_network_size_trends(df: pd.DataFrame, output_dir: str):
    """
    绘制网络规模趋势图（节点数、边数、密度）- 每个指标单独保存
    """
    # 1. 真实网络节点数和边数
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    real_nodes_df = df[(df['metric'] == 'real_nodes') & (df['method'] == 'real')].sort_values('year')
    real_edges_df = df[(df['metric'] == 'real_edges') & (df['method'] == 'real')].sort_values('year')
    
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(real_nodes_df['year'], real_nodes_df['value'], 
                     marker='o', linewidth=2.5, markersize=8, 
                     color='steelblue', label='Nodes', alpha=0.85)
    line2 = ax1_twin.plot(real_edges_df['year'], real_edges_df['value'], 
                          marker='s', linewidth=2.5, markersize=8, 
                          color='darkorange', label='Edges', alpha=0.85)
    
    ax1.set_xlabel('Year', fontsize=24, fontweight='bold')
    ax1.set_ylabel('Number of Nodes', fontsize=24, fontweight='bold', color='steelblue')
    ax1_twin.set_ylabel('Number of Edges', fontsize=24, fontweight='bold', color='darkorange')
    ax1.set_title('Real Network Size', fontsize=28, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='steelblue', labelsize=25)
    ax1_twin.tick_params(axis='y', labelcolor='darkorange', labelsize=25)
    
    # 设置 x 轴为整数年份
    years = sorted(real_nodes_df['year'].unique())
    ax1.set_xticks(years)
    ax1.set_xticklabels([int(y) for y in years], fontsize=25)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', fontsize=18)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "real_network_size.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ 真实网络规模图已保存: {output_path}")
    
    # 2. 预测的边数（各方法对比）
    fig, ax2 = plt.subplots(1, 1, figsize=(10, 10))
    edge_count_df = df[df['metric'] == 'edge_predicted_edges_count']
    
    for method in sorted(edge_count_df['method'].unique()):
        method_df = edge_count_df[edge_count_df['method'] == method].sort_values('year')
        ax2.plot(
            method_df['year'],
            method_df['value'],
            marker='o',
            linewidth=2.5,
            markersize=8,
            label=METHOD_DISPLAY_NAMES.get(method, method),
            color=METHOD_COLORS.get(method, None),
            alpha=0.85
        )
    
    # 添加真实边数作为参考线
    ax2.plot(real_edges_df['year'], real_edges_df['value'],
            marker='*', linewidth=2.5, markersize=12,
            label='Real Network', color='black', linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Year', fontsize=24, fontweight='bold')
    ax2.set_ylabel('Number of Predicted Edges', fontsize=24, fontweight='bold')
    ax2.set_title('Predicted Edge Count', fontsize=28, fontweight='bold')
    ax2.legend(loc='best', fontsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=25)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 设置 x 轴为整数年份
    years = sorted(edge_count_df['year'].unique())
    ax2.set_xticks(years)
    ax2.set_xticklabels([int(y) for y in years])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "predicted_edge_count.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ 预测边数图已保存: {output_path}")
    
    # 3. 网络密度（真实网络）
    fig, ax3 = plt.subplots(1, 1, figsize=(10, 10))
    real_density_df = df[(df['metric'] == 'real_density') & (df['method'] == 'real')].sort_values('year')
    
    ax3.plot(real_density_df['year'], real_density_df['value'],
            marker='o', linewidth=2.5, markersize=8,
            color='forestgreen', alpha=0.85)
    
    ax3.set_xlabel('Year', fontsize=24, fontweight='bold')
    ax3.set_ylabel('Network Density', fontsize=24, fontweight='bold')
    ax3.set_title('Real Network Density', fontsize=28, fontweight='bold')
    ax3.tick_params(axis='both', which='major', labelsize=25)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # 设置 x 轴为整数年份
    years = sorted(real_density_df['year'].unique())
    ax3.set_xticks(years)
    ax3.set_xticklabels([int(y) for y in years])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "real_network_density.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ 真实网络密度图已保存: {output_path}")
    
    # 4. 平均入度（各方法对比）
    fig, ax4 = plt.subplots(1, 1, figsize=(10, 10))
    real_avg_degree_df = df[df['metric'] == 'network_avg_in_degree']
    
    for method in sorted(real_avg_degree_df['method'].unique()):
        method_df = real_avg_degree_df[real_avg_degree_df['method'] == method].sort_values('year')
        ax4.plot(
            method_df['year'],
            method_df['value'],
            marker='o',
            linewidth=2.5,
            markersize=8,
            label=METHOD_DISPLAY_NAMES.get(method, method),
            color=METHOD_COLORS.get(method, None),
            alpha=0.85
        )
    
    ax4.set_xlabel('Year', fontsize=24, fontweight='bold')
    ax4.set_ylabel('Average In-Degree', fontsize=24, fontweight='bold')
    ax4.set_title('Average In-Degree', fontsize=28, fontweight='bold')
    ax4.legend(loc='best', fontsize=18)
    ax4.tick_params(axis='both', which='major', labelsize=25)
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    # 设置 x 轴为整数年份
    years = sorted(real_avg_degree_df['year'].unique())
    ax4.set_xticks(years)
    ax4.set_xticklabels([int(y) for y in years])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "average_in_degree.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ 平均入度图已保存: {output_path}")


# ============================================================================
# 可视化：边指标趋势
# ============================================================================

def plot_edge_metrics_trends(df: pd.DataFrame, output_dir: str):
    """
    绘制边指标趋势图（TP, FP, FN）- 每个指标单独保存
    """
    metrics = [
        ('edge_true_positive', 'True Positive'),
        ('edge_false_positive', 'False Positive'),
        ('edge_false_negative', 'False Negative'),
        ('edge_jaccard_similarity', 'Jaccard Similarity')
    ]
    
    for metric_name, display_name in metrics:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        metric_df = df[df['metric'] == metric_name]
        
        for method in sorted(metric_df['method'].unique()):
            method_df = metric_df[metric_df['method'] == method]
            method_df = method_df.sort_values('year')
            
            ax.plot(
                method_df['year'],
                method_df['value'],
                marker='o',
                linewidth=2.5,
                markersize=8,
                label=METHOD_DISPLAY_NAMES.get(method, method),
                color=METHOD_COLORS.get(method, None),
                alpha=0.85
            )
        
        ax.set_xlabel('Year', fontsize=24, fontweight='bold')
        ax.set_ylabel(display_name, fontsize=24, fontweight='bold')
        ax.set_title(f'{display_name}', fontsize=28, fontweight='bold')
        ax.legend(loc='best', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=25)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # y 轴范围自动调整，根据数据变化
        
        # 设置 x 轴为整数年份
        years = sorted(metric_df['year'].unique())
        ax.set_xticks(years)
        ax.set_xticklabels([int(y) for y in years])
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{metric_name}.pdf")
        plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  ✓ {display_name} 趋势图已保存: {output_path}")


# ============================================================================
# 可视化：方法对比热力图
# ============================================================================

def plot_method_comparison_heatmap(df: pd.DataFrame, output_dir: str):
    """
    绘制方法对比热力图（方法 × 年份 × 指标）- 每个指标单独保存
    """
    key_metrics = [
        ('edge_precision', 'Precision'),
        ('edge_recall', 'Recall'),
        ('edge_f1_score', 'F1 Score'),
        ('auc_mean_auc', 'Mean AUC')
    ]
    
    for metric_name, display_name in key_metrics:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        metric_df = df[df['metric'] == metric_name]
        
        # 构建数据透视表
        pivot_table = metric_df.pivot(index='method', columns='year', values='value')
        
        # 重新排序行（方法）
        method_order = ['Ours', 'logistic_regression', 'random_forest', 'xgboost', 'random']
        pivot_table = pivot_table.reindex([m for m in method_order if m in pivot_table.index])
        
        # 重命名行标签
        pivot_table.index = [METHOD_DISPLAY_NAMES.get(m, m) for m in pivot_table.index]
        
        # 重命名列标签（年份为整数）
        pivot_table.columns = [int(y) for y in pivot_table.columns]
        
        # 绘制热力图
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=0.5 if metric_name != 'auc_mean_auc' else None,
            vmin=0,
            vmax=1,
            cbar_kws={'label': display_name},
            ax=ax,
            linewidths=0.5,
            linecolor='gray',
            annot_kws={'fontsize': 20}
        )
        
        ax.set_xlabel('Year', fontsize=24, fontweight='bold')
        ax.set_ylabel('Method', fontsize=24, fontweight='bold')
        ax.set_title(f'{display_name} Heatmap', fontsize=28, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=25)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"heatmap_{metric_name}.pdf")
        plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  ✓ {display_name} 热力图已保存: {output_path}")


# ============================================================================
# 可视化：Jaccard 相似度趋势
# ============================================================================

def plot_jaccard_trends(df: pd.DataFrame, output_dir: str):
    """
    绘制 Jaccard 相似度趋势（单独的详细图）- 每个图单独保存
    """
    metric_df = df[df['metric'] == 'edge_jaccard_similarity']
    
    # 1. Jaccard 相似度趋势线图
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    
    for method in sorted(metric_df['method'].unique()):
        method_df = metric_df[metric_df['method'] == method].sort_values('year')
        ax1.plot(
            method_df['year'],
            method_df['value'],
            marker='o',
            linewidth=2.5,
            markersize=8,
            label=METHOD_DISPLAY_NAMES.get(method, method),
            color=METHOD_COLORS.get(method, None),
            alpha=0.85
        )
    
    ax1.set_xlabel('Year', fontsize=24, fontweight='bold')
    ax1.set_ylabel('Jaccard Similarity', fontsize=24, fontweight='bold')
    ax1.set_title('Jaccard Similarity', fontsize=28, fontweight='bold')
    ax1.legend(loc='best', fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=25)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 设置 x 轴为整数年份
    years = sorted(metric_df['year'].unique())
    ax1.set_xticks(years)
    ax1.set_xticklabels([int(y) for y in years])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "jaccard_similarity_trend.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Jaccard 相似度趋势图已保存: {output_path}")
    
    # 2. Jaccard 相似度箱线图（按方法分组）
    fig, ax2 = plt.subplots(1, 1, figsize=(10, 10))
    
    # 准备数据
    boxplot_data = []
    boxplot_labels = []
    method_order = ['Ours', 'logistic_regression', 'random_forest', 'xgboost', 'random']
    
    for method in method_order:
        if method in metric_df['method'].values:
            method_values = metric_df[metric_df['method'] == method]['value'].values
            boxplot_data.append(method_values)
            boxplot_labels.append(METHOD_DISPLAY_NAMES.get(method, method))
    
    bp = ax2.boxplot(
        boxplot_data,
        labels=boxplot_labels,
        patch_artist=True,
        showmeans=True,
        meanline=True
    )
    
    # 设置颜色
    for patch, method in zip(bp['boxes'], method_order[:len(boxplot_data)]):
        patch.set_facecolor(METHOD_COLORS.get(method, 'lightgray'))
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Jaccard Similarity', fontsize=24, fontweight='bold')
    ax2.set_title('Jaccard Similarity Distribution by Method', fontsize=28, fontweight='bold')
    ax2.set_xticklabels(boxplot_labels, rotation=45, ha='right', fontsize=25)
    ax2.tick_params(axis='y', which='major', labelsize=25)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "jaccard_similarity_boxplot.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Jaccard 相似度箱线图已保存: {output_path}")


# ============================================================================
# 报告生成
# ============================================================================

def generate_trend_report(df: pd.DataFrame, reports: Dict, output_path: str):
    """
    生成趋势分析汇总报告
    """
    report = {
        'years': sorted(list(reports.keys())),
        'methods': list(METHOD_DISPLAY_NAMES.keys()),
        'summary': {}
    }
    
    # 计算每个方法的年均表现
    for method in report['methods']:
        method_data = {}
        
        # 关键指标的年均值
        key_metrics = [
            'edge_precision', 'edge_recall', 'edge_f1_score',
            'edge_jaccard_similarity', 'auc_mean_auc'
        ]
        
        for metric in key_metrics:
            metric_df = df[(df['method'] == method) & (df['metric'] == metric)]
            if len(metric_df) > 0:
                method_data[f'{metric}_mean'] = float(metric_df['value'].mean())
                method_data[f'{metric}_std'] = float(metric_df['value'].std())
                method_data[f'{metric}_min'] = float(metric_df['value'].min())
                method_data[f'{metric}_max'] = float(metric_df['value'].max())
        
        report['summary'][method] = method_data
    
    # 保存为 JSON
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"  ✓ 趋势报告已保存: {output_path}")


# ============================================================================
# 主处理流程
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Stage 7: 趋势分析（演化模式）'
    )
    parser.add_argument(
        '--result_dir',
        type=str,
        required=True,
        help='结果目录路径（包含所有年份的 stage6 评估报告）'
    )
    parser.add_argument(
        '--start_year',
        type=int,
        required=True,
        help='起始年份'
    )
    parser.add_argument(
        '--end_year',
        type=int,
        required=True,
        help='结束年份'
    )
    
    args = parser.parse_args()
    
    # 打印配置
    print("=" * 80)
    print("Stage 7: 趋势分析（演化模式）")
    print("=" * 80)
    print(f"结果目录: {args.result_dir}")
    print(f"年份范围: {args.start_year} - {args.end_year}")
    print("=" * 80)
    
    # 创建输出目录
    plot_dir = os.path.join(args.result_dir, "stage7_trend_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # 步骤1: 加载评估报告
    print(f"\n[1/4] 加载评估报告")
    years = list(range(args.start_year, args.end_year + 1))
    reports = load_evaluation_reports(args.result_dir, years)
    
    if len(reports) == 0:
        print("❌ 没有找到任何评估报告，退出")
        return
    
    print(f"  ✓ 共加载 {len(reports)} 个年份的评估报告")
    
    # 步骤2: 提取趋势数据
    print(f"\n[2/4] 提取趋势数据")
    df = extract_trend_data(reports)
    print(f"  ✓ 提取了 {len(df)} 条数据记录")
    
    # 步骤3: 生成可视化
    print(f"\n[3/4] 生成趋势可视化")
    
    # 3.1 性能指标趋势
    plot_performance_trends(df, plot_dir)
    
    # 3.2 网络规模趋势
    plot_network_size_trends(df, plot_dir)
    
    # 3.3 边指标趋势
    plot_edge_metrics_trends(df, plot_dir)
    
    # 3.4 方法对比热力图
    plot_method_comparison_heatmap(df, plot_dir)
    
    # 3.5 Jaccard 趋势
    plot_jaccard_trends(df, plot_dir)
    
    # 步骤4: 生成汇总报告
    print(f"\n[4/4] 生成汇总报告")
    generate_trend_report(
        df,
        reports,
        os.path.join(args.result_dir, "stage7_trend_report.json")
    )
    
    # 打印完成信息
    print(f"\n{'='*80}")
    print(f"✓ Stage 7 完成！")
    print(f"{'='*80}")
    print(f"年份范围: {args.start_year} - {args.end_year}")
    print(f"分析年份数: {len(reports)}")
    print(f"输出目录: {plot_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

