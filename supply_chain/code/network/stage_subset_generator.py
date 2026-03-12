#!/usr/bin/env python3
"""
从20% hub结果中提取5%和10%的hub子集，并重新生成网络重建结果

该脚本从已有的20%实验结果中提取数据，无需重新加载原始数据、训练模型或调用LLM API
"""

import os
import pickle
import argparse
import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


class SubsetGenerator:
    """从20%结果生成5%和10%子集的主类"""

    def __init__(self, base_result_dir: str = "network/result/20"):
        """
        初始化

        Args:
            base_result_dir: 20%结果的目录路径
        """
        self.base_result_dir = Path(base_result_dir)
        self.data_cache = {}

    def load_stage_data(self, stage_name: str) -> Any:
        """加载某个stage的数据"""
        if stage_name in self.data_cache:
            return self.data_cache[stage_name]

        file_path = self.base_result_dir / stage_name
        print(f"Loading {file_path}...")

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        self.data_cache[stage_name] = data
        return data

    def get_hub_sizes(self, year: int) -> Dict[str, int]:
        """
        获取每个hub的候选者数量

        Args:
            year: 年份

        Returns:
            字典，键为hub_id，值为候选者数量
        """
        pools = self.load_stage_data("stage1_candidate_pools_hub20pct_noise1pct.pkl")
        year_data = pools[year]

        hub_sizes = {}
        for hub_id, candidates in year_data['hub_candidate_pools'].items():
            hub_sizes[hub_id] = len(candidates)

        return hub_sizes

    def select_top_hubs(self, hub_sizes: Dict[str, int], percentage: float) -> List[str]:
        """
        按照候选者数量选择top N%的hub

        Args:
            hub_sizes: hub及其候选者数量
            percentage: 百分比 (如 0.05, 0.10)

        Returns:
            选中的hub列表
        """
        # 按候选者数量降序排序
        sorted_hubs = sorted(hub_sizes.items(), key=lambda x: x[1], reverse=True)

        # 计算要选择的hub数量
        total_hubs = len(sorted_hubs)
        num_to_select = max(1, int(total_hubs * percentage))

        # 选择top N个
        selected_hubs = [hub_id for hub_id, _ in sorted_hubs[:num_to_select]]

        print(f"Selected {num_to_select} hubs out of {total_hubs} ({percentage*100:.0f}%)")
        print(f"Candidate pool sizes: {sorted_hubs[0][1]} (max) to {sorted_hubs[num_to_select-1][1]} (min selected)")

        return selected_hubs

    def extract_subset_data(self, selected_hubs: List[str], year: int,
                          percentage: float) -> Dict[str, Any]:
        """
        提取选中hub的所有stage数据

        Args:
            selected_hubs: 选中的hub列表
            year: 年份
            percentage: 百分比标签

        Returns:
            包含所有stage数据的字典
        """
        subset_data = {}

        # Stage1: 候选池
        print(f"\nExtracting Stage1 data...")
        pools = self.load_stage_data("stage1_candidate_pools_hub20pct_noise1pct.pkl")
        year_data = pools[year]

        subset_data['stage1'] = {
            'year': year,
            'original_graph': year_data['original_graph'],
            'hub_nodes': selected_hubs,
            'hub_candidate_pools': {hub: year_data['hub_candidate_pools'][hub]
                                   for hub in selected_hubs},
            'all_nodes_need_profile': year_data['all_nodes_need_profile'],
            'statistics': self._compute_stage1_stats(selected_hubs, year_data)
        }

        # Stage2: 画像
        print(f"Extracting Stage2 data...")
        profiles = self.load_stage_data("stage2_company_profiles_hub20pct.pkl")
        subset_data['stage2'] = profiles[year]  # 画像是所有节点的，不需要筛选

        # Stage3a: LLM性格
        print(f"Extracting Stage3a data...")
        personalities = self.load_stage_data("stage3a_llm_personalities_hub20pct.pkl")
        year_personalities = personalities[year]
        subset_data['stage3a'] = {
            hub: year_personalities[hub]
            for hub in selected_hubs if hub in year_personalities
        }

        # Stage3b: ML训练数据
        print(f"Extracting Stage3b data...")
        ml_data = self.load_stage_data("stage3b_ml_training_data_hub20pct_noise1pct.pkl")
        subset_data['stage3b'] = ml_data[year]  # ML数据也是全局的

        # Stage4: LLM评估
        print(f"Extracting Stage4 data...")
        assessments = self.load_stage_data("stage4_llm_assessments_hub20pct.pkl")
        year_assessments = assessments[year]
        subset_data['stage4'] = {
            hub: year_assessments[hub]
            for hub in selected_hubs if hub in year_assessments
        }

        # Stage5a: LLM选择
        print(f"Extracting Stage5a data...")
        llm_selections = self.load_stage_data("stage5a_llm_selections_hub20pct.pkl")
        year_selections = llm_selections[year]
        subset_data['stage5a'] = {
            'year': year,
            'hub_selections': {hub: year_selections['hub_selections'][hub]
                             for hub in selected_hubs if hub in year_selections['hub_selections']},
            'statistics': self._compute_selection_stats(selected_hubs, year_selections['hub_selections'])
        }

        # Stage5b: ML选择
        print(f"Extracting Stage5b data...")
        for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
            ml_selections = self.load_stage_data(f"stage5b_{model_name}_selections_hub20pct.pkl")
            year_selections = ml_selections[year]
            subset_data[f'stage5b_{model_name}'] = {
                'year': year,
                'hub_selections': {hub: year_selections['hub_selections'][hub]
                                 for hub in selected_hubs if hub in year_selections['hub_selections']},
                'statistics': self._compute_selection_stats(selected_hubs, year_selections['hub_selections'])
            }

        # Stage5c: Random选择
        print(f"Extracting Stage5c data...")
        random_selections = self.load_stage_data("stage5c_random_selections_hub20pct.pkl")
        year_selections = random_selections[year]
        subset_data['stage5c'] = {
            'year': year,
            'hub_selections': {hub: year_selections['hub_selections'][hub]
                             for hub in selected_hubs if hub in year_selections['hub_selections']},
            'statistics': self._compute_selection_stats(selected_hubs, year_selections['hub_selections'])
        }

        return subset_data

    def _compute_stage1_stats(self, selected_hubs: List[str],
                            year_data: Dict) -> Dict:
        """计算stage1统计信息"""
        hub_pools = {hub: year_data['hub_candidate_pools'][hub]
                    for hub in selected_hubs}

        pool_sizes = [len(pool) for pool in hub_pools.values()]

        return {
            'num_hubs': len(selected_hubs),
            'total_candidates': sum(pool_sizes),
            'avg_pool_size': sum(pool_sizes) / len(pool_sizes),
            'max_pool_size': max(pool_sizes),
            'min_pool_size': min(pool_sizes)
        }

    def _compute_selection_stats(self, selected_hubs: List[str],
                                hub_selections: Dict) -> Dict:
        """计算选择统计信息"""
        total_selected = 0
        for hub in selected_hubs:
            if hub in hub_selections:
                total_selected += len(hub_selections[hub]['selected_suppliers'])

        return {
            'num_hubs': len(selected_hubs),
            'total_selected_suppliers': total_selected,
            'avg_suppliers_per_hub': total_selected / len(selected_hubs) if selected_hubs else 0
        }

    def reconstruct_networks(self, subset_data: Dict, year: int) -> Dict[str, Any]:
        """
        基于选择结果重建网络

        Args:
            subset_data: 提取的子集数据
            year: 年份

        Returns:
            重建的网络及指标
        """
        results = {}
        original_graph = subset_data['stage1']['original_graph']

        # 对每种方法重建网络
        methods = {
            'llm': subset_data['stage5a'],
            'logistic_regression': subset_data['stage5b_logistic_regression'],
            'random_forest': subset_data['stage5b_random_forest'],
            'xgboost': subset_data['stage5b_xgboost'],
            'random': subset_data['stage5c']
        }

        for method_name, selections in methods.items():
            print(f"\nReconstructing network for method: {method_name}")

            # 复制完整的原始网络（保留所有节点和边）
            reconstructed_graph = original_graph.copy()

            # 获取所有被选中的hub节点
            selected_hubs = list(selections['hub_selections'].keys())

            # 1. 移除所有被选中hub节点的入边
            edges_to_remove = []
            for hub_id in selected_hubs:
                if hub_id in reconstructed_graph:
                    in_edges = list(reconstructed_graph.in_edges(hub_id))
                    edges_to_remove.extend(in_edges)

            reconstructed_graph.remove_edges_from(edges_to_remove)

            # 2. 根据选择结果重新添加边
            for hub_id, hub_data in selections['hub_selections'].items():
                selected_suppliers = hub_data['selected_suppliers']

                # 添加边：supplier -> hub
                for supplier in selected_suppliers:
                    reconstructed_graph.add_edge(supplier, hub_id)

            # 计算网络指标
            metrics = self._compute_network_metrics(
                original_graph,
                reconstructed_graph,
                year,
                method_name
            )

            results[method_name] = {
                'graph': reconstructed_graph,
                'metrics': metrics
            }

        return results

    def _compute_network_metrics(self, original_graph: nx.DiGraph,
                                reconstructed_graph: nx.DiGraph,
                                year: int, method: str) -> Dict:
        """计算网络指标"""

        def safe_metrics(G: nx.DiGraph, network_type: str) -> Dict:
            """安全计算网络指标"""
            metrics = {
                'year': year,
                'method': method,
                'network_type': network_type,
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges(),
            }

            if G.number_of_nodes() == 0:
                return metrics

            # 密度
            metrics['density'] = nx.density(G)

            # 转换为无向图计算聚类系数
            G_undirected = G.to_undirected()
            metrics['average_clustering'] = nx.average_clustering(G_undirected)

            # 最大弱连通分量
            if G.number_of_nodes() > 0:
                largest_wcc = max(nx.weakly_connected_components(G), key=len)
                wcc_subgraph = G.subgraph(largest_wcc)
                metrics['largest_wcc_nodes'] = len(largest_wcc)
                metrics['largest_wcc_edges'] = wcc_subgraph.number_of_edges()

                # 在最大连通分量上计算路径长度
                wcc_undirected = wcc_subgraph.to_undirected()
                if nx.is_connected(wcc_undirected):
                    metrics['average_shortest_path_length'] = nx.average_shortest_path_length(wcc_undirected)
                    metrics['diameter'] = nx.diameter(wcc_undirected)
                else:
                    metrics['average_shortest_path_length'] = float('inf')
                    metrics['diameter'] = float('inf')

            # PageRank
            pagerank = nx.pagerank(G)
            metrics['pagerank_mean'] = sum(pagerank.values()) / len(pagerank)
            metrics['pagerank_max'] = max(pagerank.values())

            # Hub scores
            try:
                hub_scores, _ = nx.hits(G)
                metrics['hub_score_mean'] = sum(hub_scores.values()) / len(hub_scores)
                metrics['hub_score_max'] = max(hub_scores.values())
            except:
                metrics['hub_score_mean'] = 0
                metrics['hub_score_max'] = 0

            # Core number (remove self-loops first)
            G_undirected = G.to_undirected()
            G_undirected.remove_edges_from(nx.selfloop_edges(G_undirected))
            if G_undirected.number_of_nodes() > 0:
                core_numbers = nx.core_number(G_undirected)
                metrics['coreness_mean'] = sum(core_numbers.values()) / len(core_numbers)
                metrics['coreness_max'] = max(core_numbers.values())
            else:
                metrics['coreness_mean'] = 0
                metrics['coreness_max'] = 0

            return metrics

        # 计算原始网络和重建网络的指标
        original_metrics = safe_metrics(original_graph, 'original')
        reconstructed_metrics = safe_metrics(reconstructed_graph, 'reconstructed')

        return {
            'original': original_metrics,
            'reconstructed': reconstructed_metrics
        }

    def _calculate_power_law_parameters(self, degrees: List[int]) -> Dict:
        """
        计算度分布的power-law参数

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

        # 使用最小值作为xmin
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

    def _plot_in_degree_distribution(
        self,
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
        original_in_powerlaw = self._calculate_power_law_parameters(original_in_values)
        if original_in_powerlaw['alpha'] is not None:
            alpha = original_in_powerlaw['alpha']
            xmin = original_in_powerlaw['xmin']
            total_nodes = len(original_in_values)

            x_fit = np.logspace(np.log10(xmin), np.log10(max(original_in_values)), 100)
            C = total_nodes * (alpha - 1) * (xmin ** (alpha - 1))
            y_fit = C * (x_fit ** (-alpha))

            ax.loglog(x_fit, y_fit, 'b--', linewidth=2.5, label='Original Fit')

        # Power-law拟合 - 重构网络
        reconstructed_in_powerlaw = self._calculate_power_law_parameters(reconstructed_in_values)
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
            }
        }

    def _plot_out_degree_distribution(
        self,
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
        original_out_powerlaw = self._calculate_power_law_parameters(original_out_values)
        if original_out_powerlaw['alpha'] is not None:
            alpha = original_out_powerlaw['alpha']
            xmin = original_out_powerlaw['xmin']
            total_nodes = len(original_out_values)

            x_fit = np.logspace(np.log10(xmin), np.log10(max(original_out_values)), 100)
            C = total_nodes * (alpha - 1) * (xmin ** (alpha - 1))
            y_fit = C * (x_fit ** (-alpha))

            ax.loglog(x_fit, y_fit, 'b--', linewidth=2.5, label='Original Fit')

        # Power-law拟合 - 重构网络
        reconstructed_out_powerlaw = self._calculate_power_law_parameters(reconstructed_out_values)
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
            }
        }

    def save_results(self, subset_data: Dict, network_results: Dict,
                    year: int, percentage: float, output_dir: str):
        """
        保存结果

        Args:
            subset_data: 子集数据
            network_results: 网络重建结果
            year: 年份
            percentage: 百分比
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        pct_label = f"hub{int(percentage*100)}pct"

        # 保存stage数据
        print(f"\nSaving stage data to {output_dir}...")

        # Stage1
        with open(output_path / f"stage1_candidate_pools_{pct_label}_noise1pct.pkl", 'wb') as f:
            pickle.dump({year: subset_data['stage1']}, f)

        # Stage2
        with open(output_path / f"stage2_company_profiles_{pct_label}.pkl", 'wb') as f:
            pickle.dump({year: subset_data['stage2']}, f)

        # Stage3a
        with open(output_path / f"stage3a_llm_personalities_{pct_label}.pkl", 'wb') as f:
            pickle.dump({year: subset_data['stage3a']}, f)

        # Stage3b
        with open(output_path / f"stage3b_ml_training_data_{pct_label}_noise1pct.pkl", 'wb') as f:
            pickle.dump({year: subset_data['stage3b']}, f)

        # Stage4
        with open(output_path / f"stage4_llm_assessments_{pct_label}.pkl", 'wb') as f:
            pickle.dump({year: subset_data['stage4']}, f)

        # Stage5a
        with open(output_path / f"stage5a_llm_selections_{pct_label}.pkl", 'wb') as f:
            pickle.dump({year: subset_data['stage5a']}, f)

        # Stage5b
        for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
            with open(output_path / f"stage5b_{model_name}_selections_{pct_label}.pkl", 'wb') as f:
                pickle.dump({year: subset_data[f'stage5b_{model_name}']}, f)

        # Stage5c
        with open(output_path / f"stage5c_random_selections_{pct_label}.pkl", 'wb') as f:
            pickle.dump({year: subset_data['stage5c']}, f)

        # 保存网络重建结果
        print(f"Saving network reconstruction results...")
        reconstructed_networks = {}
        for method, result in network_results.items():
            reconstructed_networks[method] = result['graph']

        with open(output_path / f"stage6_reconstructed_networks_{pct_label}.pkl", 'wb') as f:
            pickle.dump({year: reconstructed_networks}, f)

        # 生成度分布图
        print(f"Generating degree distribution plots...")
        plot_dir = output_path / "stage6_plots"
        plot_dir.mkdir(exist_ok=True)

        original_graph = subset_data['stage1']['original_graph']

        # 收集度分布数据
        degree_distribution_data = {}

        for method, result in network_results.items():
            reconstructed_graph = result['graph']
            
            print(f"  Plotting {method} for {year}...")
            
            # 入度分布图
            in_degree_plot_path = plot_dir / f"in_degree_distribution_{method}_{year}.pdf"
            in_degree_data = self._plot_in_degree_distribution(
                original_graph,
                reconstructed_graph,
                method,
                year,
                str(in_degree_plot_path)
            )
            
            # 出度分布图
            out_degree_plot_path = plot_dir / f"out_degree_distribution_{method}_{year}.pdf"
            out_degree_data = self._plot_out_degree_distribution(
                original_graph,
                reconstructed_graph,
                method,
                year,
                str(out_degree_plot_path)
            )
            
            # 保存度分布数据
            degree_distribution_data[method] = {
                'in_degree': in_degree_data,
                'out_degree': out_degree_data
            }

        # 保存度分布数据到结果中（供后续CSV使用）
        for method in network_results:
            network_results[method]['degree_distribution_data'] = degree_distribution_data[method]

        print(f"\nResults saved to {output_dir}")
        print(f"- Stage data files (stage1-stage5)")
        print(f"- Network reconstruction: stage6_reconstructed_networks_{pct_label}.pkl")
        print(f"- Degree distribution plots: stage6_plots/ ({len(network_results)*2} plots)")

    def generate_comparison_report(self, percentages: List[float],
                                  years: List[int],
                                  base_dir: str = "network/result"):
        """
        生成5%, 10%, 20%的对比分析报告

        Args:
            percentages: 百分比列表
            years: 年份列表
            base_dir: 结果基础目录
        """
        print("\n" + "="*80)
        print("Generating Comparison Report")
        print("="*80)

        # 收集所有指标数据
        all_metrics = []

        for pct in percentages:
            pct_label = f"hub{int(pct*100)}pct"
            result_dir = Path(base_dir) / str(int(pct*100))
            metrics_file = result_dir / f"stage6_network_metrics_{pct_label}.csv"

            if metrics_file.exists():
                df = pd.read_csv(metrics_file)
                df['hub_percentage'] = pct * 100
                all_metrics.append(df)
            else:
                print(f"Warning: {metrics_file} not found")

        if not all_metrics:
            print("No metrics data found!")
            return

        # 合并所有数据
        combined_df = pd.concat(all_metrics, ignore_index=True)

        # 保存综合对比表
        output_path = Path(base_dir) / "comparison_report"
        output_path.mkdir(parents=True, exist_ok=True)

        combined_df.to_csv(output_path / "all_metrics_comparison.csv", index=False)

        # 生成可视化对比
        self._plot_comparisons(combined_df, output_path)

        # 生成文本报告
        self._generate_text_report(combined_df, output_path)

        print(f"\nComparison report saved to {output_path}")
        print(f"- all_metrics_comparison.csv")
        print(f"- comparison_plots/")
        print(f"- comparison_summary.txt")

    def _plot_comparisons(self, df: pd.DataFrame, output_path: Path):
        """生成对比可视化图表"""
        plot_dir = output_path / "comparison_plots"
        plot_dir.mkdir(exist_ok=True)

        # 只看重建网络的指标
        recon_df = df[df['network_type'] == 'reconstructed'].copy()

        metrics_to_plot = [
            ('num_edges', 'Number of Edges'),
            ('density', 'Network Density'),
            ('average_clustering', 'Average Clustering Coefficient'),
            ('average_shortest_path_length', 'Average Shortest Path Length'),
            ('pagerank_max', 'Max PageRank'),
            ('hub_score_max', 'Max Hub Score'),
            ('coreness_max', 'Max Coreness')
        ]

        methods = recon_df['method'].unique()

        for metric, title in metrics_to_plot:
            if metric not in recon_df.columns:
                continue

            plt.figure(figsize=(12, 6))

            for method in methods:
                method_data = recon_df[recon_df['method'] == method]
                plt.plot(method_data['hub_percentage'], method_data[metric],
                        marker='o', label=method, linewidth=2)

            plt.xlabel('Hub Percentage (%)', fontsize=12)
            plt.ylabel(title, fontsize=12)
            plt.title(f'{title} vs Hub Percentage', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plt.savefig(plot_dir / f"{metric}_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()

        print(f"Generated {len(metrics_to_plot)} comparison plots")

    def _generate_text_report(self, df: pd.DataFrame, output_path: Path):
        """生成文本统计报告"""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("Network Reconstruction Comparison Report")
        report_lines.append("Hub Percentage: 5%, 10%, 20%")
        report_lines.append("="*80)
        report_lines.append("")

        # 按百分比分组统计
        for pct in sorted(df['hub_percentage'].unique()):
            pct_data = df[df['hub_percentage'] == pct]
            recon_data = pct_data[pct_data['network_type'] == 'reconstructed']

            report_lines.append(f"\n{'='*80}")
            report_lines.append(f"Hub Percentage: {pct:.0f}%")
            report_lines.append(f"{'='*80}")

            for method in sorted(recon_data['method'].unique()):
                method_data = recon_data[recon_data['method'] == method].iloc[0]

                report_lines.append(f"\nMethod: {method}")
                report_lines.append(f"  Nodes: {method_data['num_nodes']:.0f}")
                report_lines.append(f"  Edges: {method_data['num_edges']:.0f}")
                report_lines.append(f"  Density: {method_data['density']:.6f}")
                report_lines.append(f"  Avg Clustering: {method_data['average_clustering']:.6f}")

                if 'average_shortest_path_length' in method_data:
                    apl = method_data['average_shortest_path_length']
                    report_lines.append(f"  Avg Path Length: {apl:.4f}" if apl != float('inf') else "  Avg Path Length: inf")

                report_lines.append(f"  Max PageRank: {method_data['pagerank_max']:.6f}")
                report_lines.append(f"  Max Hub Score: {method_data['hub_score_max']:.6f}")

        # 写入文件
        with open(output_path / "comparison_summary.txt", 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"Generated text report")


def main():
    parser = argparse.ArgumentParser(description='Generate 5% and 10% hub subsets from 20% results')
    parser.add_argument('--base-dir', type=str, default='../result/20',
                       help='Base directory containing 20%% results')
    parser.add_argument('--output-base', type=str, default='../result',
                       help='Base output directory')
    parser.add_argument('--years', type=int, nargs='+', default=[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
                       help='Years to process')
    parser.add_argument('--percentages', type=float, nargs='+', default=[0.05, 0.10],
                       help='Hub percentages to generate (e.g., 0.05 0.10)')

    args = parser.parse_args()

    generator = SubsetGenerator(args.base_dir)

    # 对每个百分比和年份生成结果
    for percentage in args.percentages:
        pct_label = int(percentage * 100)
        output_dir = Path(args.output_base) / str(pct_label)

        print(f"\n{'='*80}")
        print(f"Processing {pct_label}% hub subset")
        print(f"{'='*80}")

        # 收集所有年份的指标和度分布数据
        all_metrics = []
        all_degree_distributions = []

        for year in args.years:
            print(f"\n{'='*80}")
            print(f"Year: {year}")
            print(f"{'='*80}")

            # 1. 获取hub大小并选择top N%
            hub_sizes = generator.get_hub_sizes(year)
            selected_hubs = generator.select_top_hubs(hub_sizes, percentage)

            # 2. 提取子集数据
            subset_data = generator.extract_subset_data(selected_hubs, year, percentage)

            # 3. 重建网络
            network_results = generator.reconstruct_networks(subset_data, year)

            # 4. 保存结果（不保存CSV，累积指标）
            generator.save_results(subset_data, network_results, year, percentage, output_dir)

            # 收集指标
            for method, result in network_results.items():
                all_metrics.append(result['metrics']['original'])
                all_metrics.append(result['metrics']['reconstructed'])
                
                # 收集度分布数据
                if 'degree_distribution_data' in result:
                    degree_data = result['degree_distribution_data']
                    
                    # 入度分布 - 原始网络
                    all_degree_distributions.append({
                        'year': year,
                        'method': method,
                        'network_type': 'original',
                        'degree_type': 'in_degree',
                        'alpha': degree_data['in_degree']['original']['powerlaw']['alpha'],
                        'pvalue': degree_data['in_degree']['original']['powerlaw']['pvalue'],
                        'xmin': degree_data['in_degree']['original']['powerlaw']['xmin']
                    })
                    
                    # 入度分布 - 重构网络
                    all_degree_distributions.append({
                        'year': year,
                        'method': method,
                        'network_type': 'reconstructed',
                        'degree_type': 'in_degree',
                        'alpha': degree_data['in_degree']['reconstructed']['powerlaw']['alpha'],
                        'pvalue': degree_data['in_degree']['reconstructed']['powerlaw']['pvalue'],
                        'xmin': degree_data['in_degree']['reconstructed']['powerlaw']['xmin']
                    })
                    
                    # 出度分布 - 原始网络
                    all_degree_distributions.append({
                        'year': year,
                        'method': method,
                        'network_type': 'original',
                        'degree_type': 'out_degree',
                        'alpha': degree_data['out_degree']['original']['powerlaw']['alpha'],
                        'pvalue': degree_data['out_degree']['original']['powerlaw']['pvalue'],
                        'xmin': degree_data['out_degree']['original']['powerlaw']['xmin']
                    })
                    
                    # 出度分布 - 重构网络
                    all_degree_distributions.append({
                        'year': year,
                        'method': method,
                        'network_type': 'reconstructed',
                        'degree_type': 'out_degree',
                        'alpha': degree_data['out_degree']['reconstructed']['powerlaw']['alpha'],
                        'pvalue': degree_data['out_degree']['reconstructed']['powerlaw']['pvalue'],
                        'xmin': degree_data['out_degree']['reconstructed']['powerlaw']['xmin']
                    })

        # 保存所有年份的指标到CSV
        print(f"\nSaving all metrics to CSV...")
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(output_dir / f"stage6_network_metrics_hub{pct_label}pct.csv", index=False)
        
        # 保存度分布数据到CSV
        print(f"Saving degree distribution data to CSV...")
        degree_df = pd.DataFrame(all_degree_distributions)
        degree_df.to_csv(output_dir / f"stage6_degree_distributions_hub{pct_label}pct.csv", index=False)

    # 生成对比报告
    all_percentages = [0.05, 0.10, 0.20]
    generator.generate_comparison_report(all_percentages, args.years, args.output_base)

    print("\n" + "="*80)
    print("All tasks completed!")
    print("="*80)


if __name__ == "__main__":
    main()
