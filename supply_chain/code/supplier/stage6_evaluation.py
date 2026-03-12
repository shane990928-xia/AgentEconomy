#!/usr/bin/env python3
"""
Stage 6: 实验评估
=================

功能：
- 将LLM选择结果与真实供应商关系对比
- 计算准确率、召回率等指标
- 生成评估报告

用法：
    from stage6_evaluation import (
        evaluate_selection_accuracy,
        calculate_metrics,
        generate_evaluation_report
    )
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Set, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import json
import os
from datetime import datetime

from stage0_config import RESULT_DIR, _convert_for_json


# ============================================================================
# 真实供应商获取
# ============================================================================

def get_actual_suppliers(
    target_entity_id: str,
    decision_year: int,
    all_dfs: Dict[str, pd.DataFrame],
    look_ahead_years: int = 1
) -> Set[str]:
    """
    获取实际建立的供应商关系（用于验证）
    
    Args:
        target_entity_id: 目标公司ID
        decision_year: 决策年份
        all_dfs: 所有数据集
        look_ahead_years: 向前看的年份数（默认1年）
    
    Returns:
        实际供应商ID集合
    """
    df_rev = all_dfs.get('factset_revere_relationship')
    if df_rev is None or df_rev.empty:
        return set()
    
    # 定义时间窗口
    start_date = pd.Timestamp(f'{decision_year}-01-01')
    end_date = pd.Timestamp(f'{decision_year + look_ahead_years}-12-31')
    
    # 通过 company_factset 查出数值型 company_id
    df_cf = all_dfs.get('company_factset')
    target_numeric_id = None
    cf_numeric = None
    if df_cf is not None and 'fs_entity_id' in df_cf.columns and 'company_id' in df_cf.columns:
        cf_numeric = df_cf.copy()
        cf_numeric['_cid'] = pd.to_numeric(cf_numeric['company_id'], errors='coerce')
        row = cf_numeric[cf_numeric['fs_entity_id'] == target_entity_id]
        if not row.empty:
            target_numeric_id = row.iloc[0]['_cid']

    if target_numeric_id is None or pd.isna(target_numeric_id):
        return set()

    # FactSet REVERE 语义: source=供应商(卖方), target=买方/客户
    # 目标公司作为买方在 target 列，其供应商在 source 列
    supplier_rels = df_rev[
        (df_rev['rel_type'] == 'SUPPLIER') &
        (df_rev['target_company_id'] == target_numeric_id)
    ].copy()

    if supplier_rels.empty:
        return set()

    # 确保日期格式
    if 'start_' in supplier_rels.columns:
        supplier_rels['start_'] = pd.to_datetime(supplier_rels['start_'], errors='coerce')
        actual_rels = supplier_rels[
            (supplier_rels['start_'] >= start_date) &
            (supplier_rels['start_'] <= end_date)
        ]
    else:
        actual_rels = supplier_rels

    # 供应商在 source_company_id 列，映射回 fs_entity_id
    if 'source_company_id' in actual_rels.columns and cf_numeric is not None:
        numeric_ids = set(actual_rels['source_company_id'].dropna().unique())
        mapped = cf_numeric[cf_numeric['_cid'].isin(numeric_ids)]['fs_entity_id']
        return set(mapped.dropna().unique())

    return set()


def get_historical_suppliers(
    target_entity_id: str,
    decision_year: int,
    all_dfs: Dict[str, pd.DataFrame]
) -> Set[str]:
    """
    获取历史供应商（决策年份之前）
    
    Args:
        target_entity_id: 目标公司ID
        decision_year: 决策年份
        all_dfs: 所有数据集
    
    Returns:
        历史供应商ID集合
    """
    df_rev = all_dfs.get('factset_revere_relationship')
    if df_rev is None or df_rev.empty:
        return set()
    
    cutoff_date = pd.Timestamp(f'{decision_year}-01-01')
    
    # 通过 company_factset 查出数值型 company_id
    df_cf = all_dfs.get('company_factset')
    target_numeric_id = None
    cf_numeric = None
    if df_cf is not None and 'fs_entity_id' in df_cf.columns and 'company_id' in df_cf.columns:
        cf_numeric = df_cf.copy()
        cf_numeric['_cid'] = pd.to_numeric(cf_numeric['company_id'], errors='coerce')
        row = cf_numeric[cf_numeric['fs_entity_id'] == target_entity_id]
        if not row.empty:
            target_numeric_id = row.iloc[0]['_cid']

    if target_numeric_id is None or pd.isna(target_numeric_id):
        return set()

    # FactSet REVERE 语义: source=供应商(卖方), target=买方/客户
    supplier_rels = df_rev[
        (df_rev['rel_type'] == 'SUPPLIER') &
        (df_rev['target_company_id'] == target_numeric_id)
    ].copy()

    if supplier_rels.empty:
        return set()

    if 'start_' in supplier_rels.columns:
        supplier_rels['start_'] = pd.to_datetime(supplier_rels['start_'], errors='coerce')
        historical_rels = supplier_rels[supplier_rels['start_'] < cutoff_date]
    else:
        historical_rels = supplier_rels

    if 'source_company_id' in historical_rels.columns and cf_numeric is not None:
        numeric_ids = set(historical_rels['source_company_id'].dropna().unique())
        mapped = cf_numeric[cf_numeric['_cid'].isin(numeric_ids)]['fs_entity_id']
        return set(mapped.dropna().unique())

    return set()


# ============================================================================
# 评估指标计算
# ============================================================================

def calculate_metrics(
    selected_ids: Set[str],
    actual_ids: Set[str],
    candidate_ids: Set[str]
) -> Dict[str, float]:
    """
    计算评估指标
    
    Args:
        selected_ids: LLM选择的供应商ID集合
        actual_ids: 实际建立关系的供应商ID集合
        candidate_ids: 所有候选供应商ID集合
    
    Returns:
        指标字典
    """
    # 基本集合运算
    true_positives = selected_ids & actual_ids
    false_positives = selected_ids - actual_ids
    false_negatives = actual_ids - selected_ids
    true_negatives = candidate_ids - selected_ids - actual_ids
    
    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)
    tn = len(true_negatives)
    
    # 精确率
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # 召回率
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1分数
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 准确率
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    # 命中率 (Hit Rate @ K)
    hit_rate = 1.0 if tp > 0 else 0.0
    
    # Jaccard相似度
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'accuracy': round(accuracy, 4),
        'hit_rate': round(hit_rate, 4),
        'jaccard_similarity': round(jaccard, 4),
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn,
        'selected_count': len(selected_ids),
        'actual_count': len(actual_ids),
        'candidate_count': len(candidate_ids)
    }


def calculate_ranking_metrics(
    ranked_selections: List[Tuple[str, float]],
    actual_ids: Set[str],
    k_values: List[int] = None
) -> Dict[str, float]:
    """
    计算排序相关指标
    
    Args:
        ranked_selections: 排序后的选择列表 [(supplier_id, score), ...]
        actual_ids: 实际供应商ID集合
        k_values: 评估的K值列表
    
    Returns:
        排序指标字典
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20]
    
    metrics = {}
    
    for k in k_values:
        top_k_ids = set([sid for sid, _ in ranked_selections[:k]])
        hits = top_k_ids & actual_ids
        
        # Precision@K
        metrics[f'precision@{k}'] = round(len(hits) / k, 4) if k > 0 else 0.0
        
        # Recall@K
        metrics[f'recall@{k}'] = round(len(hits) / len(actual_ids), 4) if actual_ids else 0.0
        
        # Hit@K
        metrics[f'hit@{k}'] = 1.0 if hits else 0.0
    
    # Mean Reciprocal Rank (MRR)
    mrr = 0.0
    for i, (sid, _) in enumerate(ranked_selections):
        if sid in actual_ids:
            mrr = 1.0 / (i + 1)
            break
    metrics['mrr'] = round(mrr, 4)
    
    # NDCG (Normalized Discounted Cumulative Gain)
    dcg = 0.0
    for i, (sid, _) in enumerate(ranked_selections):
        if sid in actual_ids:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1)=0
    
    # Ideal DCG
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual_ids), len(ranked_selections))))
    
    ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    metrics['ndcg'] = round(ndcg, 4)
    
    return metrics


# ============================================================================
# 评估执行
# ============================================================================

def evaluate_selection_accuracy(
    target_entity_id: str,
    decision_year: int,
    selected_suppliers: List[Dict],
    all_dfs: Dict[str, pd.DataFrame],
    candidate_ids: Set[str] = None,
    look_ahead_years: int = 1
) -> Dict[str, Any]:
    """
    评估供应商选择的准确性
    
    Args:
        target_entity_id: 目标公司ID
        decision_year: 决策年份
        selected_suppliers: LLM选择的供应商列表
        all_dfs: 所有数据集
        candidate_ids: 候选ID集合（可选）
        look_ahead_years: 向前看的年份数
    
    Returns:
        评估结果字典
    """
    # 获取实际供应商
    actual_suppliers = get_actual_suppliers(
        target_entity_id, decision_year, all_dfs, look_ahead_years
    )
    
    # 获取历史供应商（用于排除）
    historical_suppliers = get_historical_suppliers(
        target_entity_id, decision_year, all_dfs
    )
    
    # 新建立的供应商（实际 - 历史）
    new_actual_suppliers = actual_suppliers - historical_suppliers
    
    # LLM选择的ID
    selected_ids = set(sup.get('factset_entity_id') for sup in selected_suppliers)
    
    # 候选ID（如果没有提供，使用选择ID + 实际ID）
    if candidate_ids is None:
        candidate_ids = selected_ids | new_actual_suppliers
    
    # 计算基本指标
    basic_metrics = calculate_metrics(selected_ids, new_actual_suppliers, candidate_ids)
    
    # 计算排序指标
    ranked_selections = [
        (sup.get('factset_entity_id'), sup.get('overall_score', 0))
        for sup in selected_suppliers
    ]
    ranking_metrics = calculate_ranking_metrics(ranked_selections, new_actual_suppliers)
    
    # 找出正确选择的供应商
    correct_selections = []
    for sup in selected_suppliers:
        if sup.get('factset_entity_id') in new_actual_suppliers:
            correct_selections.append({
                'factset_entity_id': sup.get('factset_entity_id'),
                'entity_name': sup.get('entity_name'),
                'overall_score': sup.get('overall_score')
            })
    
    # 找出遗漏的实际供应商
    missed_suppliers = list(new_actual_suppliers - selected_ids)
    
    return {
        'target_entity_id': target_entity_id,
        'decision_year': decision_year,
        'look_ahead_years': look_ahead_years,
        'basic_metrics': basic_metrics,
        'ranking_metrics': ranking_metrics,
        'selected_count': len(selected_ids),
        'actual_new_supplier_count': len(new_actual_suppliers),
        'historical_supplier_count': len(historical_suppliers),
        'correct_selections': correct_selections,
        'missed_supplier_ids': missed_suppliers[:20],  # 只保留前20个
    }


# ============================================================================
# 批量评估
# ============================================================================

def batch_evaluate(
    evaluations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    汇总多个评估结果
    
    Args:
        evaluations: 评估结果列表
    
    Returns:
        汇总统计
    """
    if not evaluations:
        return {'error': 'No evaluations provided'}
    
    # 收集各项指标
    precisions = []
    recalls = []
    f1_scores = []
    hit_rates = []
    mrrs = []
    
    for eval_result in evaluations:
        basic = eval_result.get('basic_metrics', {})
        ranking = eval_result.get('ranking_metrics', {})
        
        precisions.append(basic.get('precision', 0))
        recalls.append(basic.get('recall', 0))
        f1_scores.append(basic.get('f1_score', 0))
        hit_rates.append(basic.get('hit_rate', 0))
        mrrs.append(ranking.get('mrr', 0))
    
    return {
        'num_evaluations': len(evaluations),
        'aggregated_metrics': {
            'mean_precision': round(np.mean(precisions), 4),
            'std_precision': round(np.std(precisions), 4),
            'mean_recall': round(np.mean(recalls), 4),
            'std_recall': round(np.std(recalls), 4),
            'mean_f1': round(np.mean(f1_scores), 4),
            'std_f1': round(np.std(f1_scores), 4),
            'mean_hit_rate': round(np.mean(hit_rates), 4),
            'mean_mrr': round(np.mean(mrrs), 4),
        },
        'individual_results': evaluations
    }


# ============================================================================
# 报告生成
# ============================================================================

def generate_evaluation_report(
    evaluation_result: Dict[str, Any],
    self_profile: Dict = None,
    output_dir: str = None
) -> str:
    """
    生成评估报告并保存
    
    Args:
        evaluation_result: 评估结果
        self_profile: 目标公司画像
        output_dir: 输出目录
    
    Returns:
        报告文件路径
    """
    if output_dir is None:
        output_dir = RESULT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建报告
    report = {
        'report_time': datetime.now().isoformat(),
        'target_company': self_profile if self_profile else {},
        'evaluation': evaluation_result
    }
    
    # 生成文件名
    entity_id = evaluation_result.get('target_entity_id', 'unknown')
    year = evaluation_result.get('decision_year', 'unknown')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    filename = f"evaluation_{entity_id}_{year}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # 转换为JSON可序列化格式
    def convert_report(obj):
        if isinstance(obj, dict):
            return {k: convert_report(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_report(v) for v in obj]
        elif isinstance(obj, set):
            return list(obj)
        else:
            return _convert_for_json(obj)
    
    report = convert_report(report)
    
    # 保存
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    return filepath


def format_evaluation_text(evaluation_result: Dict[str, Any]) -> str:
    """
    格式化评估结果为文本
    
    Args:
        evaluation_result: 评估结果
    
    Returns:
        格式化文本
    """
    lines = []
    
    lines.append("=" * 60)
    lines.append("供应商选择评估报告")
    lines.append("=" * 60)
    lines.append("")
    
    # 基本信息
    lines.append(f"目标公司ID: {evaluation_result.get('target_entity_id', 'Unknown')}")
    lines.append(f"决策年份: {evaluation_result.get('decision_year', 'Unknown')}")
    lines.append(f"评估期间: {evaluation_result.get('look_ahead_years', 1)} 年")
    lines.append("")
    
    # 数量统计
    lines.append("-" * 40)
    lines.append("数量统计")
    lines.append("-" * 40)
    lines.append(f"LLM选择数量: {evaluation_result.get('selected_count', 0)}")
    lines.append(f"实际新供应商数量: {evaluation_result.get('actual_new_supplier_count', 0)}")
    lines.append(f"历史供应商数量: {evaluation_result.get('historical_supplier_count', 0)}")
    lines.append("")
    
    # 基本指标
    basic = evaluation_result.get('basic_metrics', {})
    lines.append("-" * 40)
    lines.append("基本评估指标")
    lines.append("-" * 40)
    lines.append(f"精确率 (Precision): {basic.get('precision', 0):.4f}")
    lines.append(f"召回率 (Recall): {basic.get('recall', 0):.4f}")
    lines.append(f"F1分数: {basic.get('f1_score', 0):.4f}")
    lines.append(f"命中率 (Hit Rate): {basic.get('hit_rate', 0):.4f}")
    lines.append(f"Jaccard相似度: {basic.get('jaccard_similarity', 0):.4f}")
    lines.append("")
    
    # 排序指标
    ranking = evaluation_result.get('ranking_metrics', {})
    lines.append("-" * 40)
    lines.append("排序评估指标")
    lines.append("-" * 40)
    lines.append(f"MRR (Mean Reciprocal Rank): {ranking.get('mrr', 0):.4f}")
    lines.append(f"NDCG: {ranking.get('ndcg', 0):.4f}")
    for k in [1, 3, 5, 10]:
        lines.append(f"Precision@{k}: {ranking.get(f'precision@{k}', 0):.4f}")
    lines.append("")
    
    # 正确选择
    correct = evaluation_result.get('correct_selections', [])
    lines.append("-" * 40)
    lines.append(f"正确选择的供应商 ({len(correct)}个)")
    lines.append("-" * 40)
    for sup in correct[:10]:
        lines.append(f"  - {sup.get('entity_name', 'Unknown')} (score: {sup.get('overall_score', 0)})")
    lines.append("")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


# ============================================================================
# 主函数 - 测试用
# ============================================================================

if __name__ == "__main__":
    print("=== Stage 6: 实验评估测试 ===\n")
    
    # 模拟数据
    mock_selected = [
        {'factset_entity_id': 'sup_001', 'entity_name': 'Supplier A', 'overall_score': 8.5},
        {'factset_entity_id': 'sup_002', 'entity_name': 'Supplier B', 'overall_score': 7.8},
        {'factset_entity_id': 'sup_003', 'entity_name': 'Supplier C', 'overall_score': 7.2},
        {'factset_entity_id': 'sup_004', 'entity_name': 'Supplier D', 'overall_score': 6.5},
        {'factset_entity_id': 'sup_005', 'entity_name': 'Supplier E', 'overall_score': 6.0},
    ]
    
    mock_actual_ids = {'sup_002', 'sup_005', 'sup_010', 'sup_015'}
    mock_candidate_ids = set([f'sup_{i:03d}' for i in range(1, 101)])
    
    # 测试指标计算
    selected_ids = set(s['factset_entity_id'] for s in mock_selected)
    metrics = calculate_metrics(selected_ids, mock_actual_ids, mock_candidate_ids)
    
    print("基本指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # 测试排序指标
    ranked = [(s['factset_entity_id'], s['overall_score']) for s in mock_selected]
    ranking_metrics = calculate_ranking_metrics(ranked, mock_actual_ids)
    
    print("\n排序指标:")
    for key, value in ranking_metrics.items():
        print(f"  {key}: {value}")
    
    # 模拟完整评估结果
    mock_evaluation = {
        'target_entity_id': '000C7F-E',
        'decision_year': 2020,
        'look_ahead_years': 1,
        'basic_metrics': metrics,
        'ranking_metrics': ranking_metrics,
        'selected_count': len(mock_selected),
        'actual_new_supplier_count': len(mock_actual_ids),
        'historical_supplier_count': 50,
        'correct_selections': [
            {'factset_entity_id': 'sup_002', 'entity_name': 'Supplier B', 'overall_score': 7.8},
            {'factset_entity_id': 'sup_005', 'entity_name': 'Supplier E', 'overall_score': 6.0},
        ],
        'missed_supplier_ids': ['sup_010', 'sup_015']
    }
    
    print("\n" + format_evaluation_text(mock_evaluation))
