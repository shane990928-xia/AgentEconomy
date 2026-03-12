#!/usr/bin/env python3
"""
Stage 5: 供应商候选筛选与选择
============================

功能：
- 从数据库中筛选供应商候选
- 应用评估结果进行最终选择
- 生成选择推荐列表

用法：
    from stage5_selection import (
        get_supplier_candidates,
        select_top_suppliers,
        generate_selection_report
    )
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set

from stage4_assessment import rank_assessments, filter_assessments


# ============================================================================
# 候选筛选
# ============================================================================

def get_supplier_candidates(
    target_entity_id: str,
    decision_year: int,
    all_dfs: Dict[str, pd.DataFrame],
    max_candidates: int = 100,
    include_existing: bool = False,
    industry_filter: str = None
) -> pd.DataFrame:
    """
    获取供应商候选列表
    
    Args:
        target_entity_id: 目标公司ID
        decision_year: 决策年份
        all_dfs: 所有数据集
        max_candidates: 最大候选数量
        include_existing: 是否包含已有供应商
        industry_filter: 行业过滤条件
    
    Returns:
        候选供应商DataFrame
    """
    cutoff_date = pd.Timestamp(f'{decision_year}-01-01')
    
    # 获取供应商关系数据
    df_rev = all_dfs.get('factset_revere_relationship')
    if df_rev is None or df_rev.empty:
        return pd.DataFrame()
    
    # 筛选供应商关系
    supplier_rels = df_rev[df_rev['rel_type'] == 'SUPPLIER'].copy()
    
    if supplier_rels.empty:
        return pd.DataFrame()
    
    # 确保日期列格式正确
    if 'start_' in supplier_rels.columns:
        supplier_rels['start_'] = pd.to_datetime(supplier_rels['start_'], errors='coerce')
    
    # factset_revere_relationship 使用数值型 company_id。
    # company_factset.company_id 可能是字符串，需要用 pd.to_numeric 统一转换。
    df_cf = all_dfs.get('company_factset')
    target_numeric_id = None
    cf_numeric = None

    if df_cf is not None and 'fs_entity_id' in df_cf.columns and 'company_id' in df_cf.columns:
        cf_numeric = df_cf.copy()
        cf_numeric['_cid'] = pd.to_numeric(cf_numeric['company_id'], errors='coerce')
        row = cf_numeric[cf_numeric['fs_entity_id'] == target_entity_id]
        if not row.empty:
            target_numeric_id = row.iloc[0]['_cid']

    # 辅助函数：数值 company_id → fs_entity_id
    def numeric_ids_to_entity_ids(numeric_ids):
        if cf_numeric is None:
            return set()
        mapped = cf_numeric[cf_numeric['_cid'].isin(set(numeric_ids))]['fs_entity_id']
        return set(mapped.dropna().unique())

    # FactSet REVERE 语义: source=供应商(卖方), target=买方/客户
    # Apple 作为买方在 target 列，其供应商在 source 列

    # 获取目标公司的历史供应商（供应商在 source_company_id, 买方在 target_company_id）
    existing_suppliers = set()
    if not include_existing and target_numeric_id is not None:
        existing_rels = supplier_rels[supplier_rels['target_company_id'] == target_numeric_id]
        existing_numeric = set(existing_rels['source_company_id'].dropna().unique())
        existing_suppliers = numeric_ids_to_entity_ids(existing_numeric)

    # 最小改动但更合理的候选池：
    # 用“同行业买方（peer buyers）的历史供应商池”作为候选来源。
    # 这样候选更接近目标公司未来可能新增的供应商。

    # 获取公司基本信息（用于识别同行业 peer buyers）
    df_entity = all_dfs.get('standard_entity')
    if df_entity is None or df_entity.empty:
        # 回退：如果没有实体表，只返回全局供应商池
        all_supplier_numeric_ids = set(supplier_rels['source_company_id'].dropna().unique())
        if target_numeric_id is not None:
            all_supplier_numeric_ids.discard(target_numeric_id)
        all_candidate_entity_ids = numeric_ids_to_entity_ids(all_supplier_numeric_ids)
        if existing_suppliers:
            all_candidate_entity_ids -= existing_suppliers
        return pd.DataFrame({'factset_entity_id': list(all_candidate_entity_ids)[:max_candidates]})

    target_info = df_entity[df_entity['factset_entity_id'] == target_entity_id]
    if target_info.empty:
        return pd.DataFrame()

    target_sic = str(target_info.iloc[0].get('primary_sic_code', '') or '')
    target_sic2 = target_sic[:2] if target_sic else ''

    # 先找同行业 peer buyers
    peer_entities = set()
    if target_sic2 and 'primary_sic_code' in df_entity.columns:
        peer_entities = set(
            df_entity[
                df_entity['primary_sic_code'].astype(str).str[:2] == target_sic2
            ]['factset_entity_id'].dropna().unique()
        ) - {target_entity_id}

    peer_candidate_entity_ids = set()
    if peer_entities and cf_numeric is not None:
        peer_cids = set(cf_numeric[cf_numeric['fs_entity_id'].isin(peer_entities)]['_cid'].dropna().unique())

        # peer buyers 在决策年前的历史供应商
        peer_hist_rels = supplier_rels[supplier_rels['target_company_id'].isin(peer_cids)].copy()
        if 'start_' in peer_hist_rels.columns:
            peer_hist_rels = peer_hist_rels[peer_hist_rels['start_'] < cutoff_date]

        peer_supplier_numeric = set(peer_hist_rels['source_company_id'].dropna().unique())
        # 排除目标公司自身和其历史供应商
        if target_numeric_id is not None:
            peer_supplier_numeric.discard(target_numeric_id)
        existing_numeric = set(existing_rels['source_company_id'].dropna().unique()) if target_numeric_id is not None else set()
        peer_supplier_numeric -= existing_numeric

        peer_candidate_entity_ids = numeric_ids_to_entity_ids(peer_supplier_numeric)

    # 如果同行业候选池为空，则回退到全局供应商池（保持鲁棒性）
    if not peer_candidate_entity_ids:
        all_supplier_numeric_ids = set(supplier_rels['source_company_id'].dropna().unique())
        if target_numeric_id is not None:
            all_supplier_numeric_ids.discard(target_numeric_id)
        peer_candidate_entity_ids = numeric_ids_to_entity_ids(all_supplier_numeric_ids)
        if existing_suppliers:
            peer_candidate_entity_ids -= existing_suppliers

    if not peer_candidate_entity_ids:
        return pd.DataFrame()

    candidate_ids = list(peer_candidate_entity_ids)

    # 匹配公司信息
    candidates = df_entity[df_entity['factset_entity_id'].isin(candidate_ids)].copy()

    # 额外行业过滤（如果显式要求）
    if industry_filter and 'primary_sic_code' in candidates.columns:
        candidates = candidates[
            candidates['primary_sic_code'].astype(str).str[:2] == str(industry_filter)[:2]
        ]

    # 让同行业、同国家候选优先排在前面（轻量排序，不改变整体框架）
    target_country = target_info.iloc[0].get('iso_country') if not target_info.empty else None
    if 'primary_sic_code' in candidates.columns:
        candidates['_same_sic2'] = (candidates['primary_sic_code'].astype(str).str[:2] == target_sic2).astype(int)
    else:
        candidates['_same_sic2'] = 0
    if target_country and 'iso_country' in candidates.columns:
        candidates['_same_country'] = (candidates['iso_country'] == target_country).astype(int)
    else:
        candidates['_same_country'] = 0

    candidates = candidates.sort_values(
        by=['_same_sic2', '_same_country'],
        ascending=[False, False]
    ).head(max_candidates).copy()

    # 清理辅助列
    for col in ['_same_sic2', '_same_country']:
        if col in candidates.columns:
            del candidates[col]

    return candidates


def get_same_industry_candidates(
    target_entity_id: str,
    decision_year: int,
    all_dfs: Dict[str, pd.DataFrame],
    max_candidates: int = 100
) -> pd.DataFrame:
    """
    获取同行业的供应商候选
    
    Args:
        target_entity_id: 目标公司ID
        decision_year: 决策年份
        all_dfs: 所有数据集
        max_candidates: 最大候选数量
    
    Returns:
        候选供应商DataFrame
    """
    # 获取目标公司的行业
    df_entity = all_dfs.get('standard_entity')
    if df_entity is None or df_entity.empty:
        return pd.DataFrame()
    
    target_info = df_entity[df_entity['factset_entity_id'] == target_entity_id]
    if target_info.empty:
        return pd.DataFrame()
    
    target_sic = target_info.iloc[0].get('primary_sic_code', '')
    if not target_sic:
        return get_supplier_candidates(target_entity_id, decision_year, all_dfs, max_candidates)
    
    # 获取同行业公司
    sic_prefix = str(target_sic)[:2]
    
    same_industry = df_entity[
        (df_entity['primary_sic_code'].astype(str).str[:2] == sic_prefix) &
        (df_entity['factset_entity_id'] != target_entity_id)
    ].head(max_candidates)
    
    return same_industry


def get_geographic_candidates(
    target_entity_id: str,
    decision_year: int,
    all_dfs: Dict[str, pd.DataFrame],
    max_candidates: int = 100,
    same_country: bool = True,
    same_region: bool = False
) -> pd.DataFrame:
    """
    获取地理相关的供应商候选
    
    Args:
        target_entity_id: 目标公司ID
        decision_year: 决策年份
        all_dfs: 所有数据集
        max_candidates: 最大候选数量
        same_country: 同一国家
        same_region: 同一区域（优先级低于same_country）
    
    Returns:
        候选供应商DataFrame
    """
    df_entity = all_dfs.get('standard_entity')
    if df_entity is None or df_entity.empty:
        return pd.DataFrame()
    
    target_info = df_entity[df_entity['factset_entity_id'] == target_entity_id]
    if target_info.empty:
        return pd.DataFrame()
    
    target_country = target_info.iloc[0].get('iso_country', '')
    
    # 定义区域映射
    REGIONS = {
        'North America': ['US', 'CA', 'MX'],
        'Europe': ['DE', 'FR', 'GB', 'IT', 'ES', 'NL', 'BE', 'CH', 'AT', 'SE', 'NO', 'DK', 'FI', 'PL', 'IE'],
        'Asia Pacific': ['CN', 'JP', 'KR', 'TW', 'SG', 'HK', 'AU', 'IN', 'TH', 'MY', 'VN', 'PH', 'ID'],
    }
    
    if same_country and target_country:
        candidates = df_entity[
            (df_entity['iso_country'] == target_country) &
            (df_entity['factset_entity_id'] != target_entity_id)
        ].head(max_candidates)
        return candidates
    
    if same_region:
        target_region = None
        for region, countries in REGIONS.items():
            if target_country in countries:
                target_region = region
                break
        
        if target_region:
            region_countries = REGIONS[target_region]
            candidates = df_entity[
                (df_entity['iso_country'].isin(region_countries)) &
                (df_entity['factset_entity_id'] != target_entity_id)
            ].head(max_candidates)
            return candidates
    
    # 默认返回所有候选
    return df_entity[
        df_entity['factset_entity_id'] != target_entity_id
    ].head(max_candidates)


# ============================================================================
# 最终选择
# ============================================================================

def select_top_suppliers(
    assessments: Dict[str, Dict],
    supplier_profiles: Dict[str, Dict],
    top_k: int = 10,
    min_score: float = 5.0,
    require_recommendation: List[str] = None
) -> List[Dict]:
    """
    选择最佳供应商
    
    Args:
        assessments: {supplier_id: assessment} 字典
        supplier_profiles: {supplier_id: profile} 字典
        top_k: 选择数量
        min_score: 最低分数要求
        require_recommendation: 要求的推荐等级
    
    Returns:
        选中的供应商列表，包含画像和评估信息
    """
    if require_recommendation is None:
        require_recommendation = ['strong_yes', 'yes']
    
    # 过滤
    filtered = filter_assessments(assessments, min_score, require_recommendation)
    
    # 排序
    ranked = rank_assessments(filtered)
    
    # 构建结果
    selected = []
    for supplier_id, assessment in ranked[:top_k]:
        profile = supplier_profiles.get(supplier_id, {})
        selected.append({
            'factset_entity_id': supplier_id,
            'entity_name': profile.get('entity_name', 'Unknown'),
            'industry': profile.get('industry', 'Unknown'),
            'country': profile.get('country', 'Unknown'),
            'overall_score': assessment.get('overall_score', 0),
            'recommendation': assessment.get('recommendation', 'unknown'),
            'financial_stability': assessment.get('financial_stability', {}).get('score', 0),
            'scale_fit': assessment.get('scale_fit', {}).get('score', 0),
            'industry_relevance': assessment.get('industry_relevance', {}).get('score', 0),
            'geographic_alignment': assessment.get('geographic_alignment', {}).get('score', 0),
            'risk_profile': assessment.get('risk_profile', {}).get('score', 0),
            'summary': assessment.get('summary', ''),
            'profile': profile,
            'assessment': assessment
        })
    
    return selected


def diversify_selection(
    selected: List[Dict],
    diversity_factor: str = 'country',
    max_per_group: int = 3
) -> List[Dict]:
    """
    多样化选择结果（避免同一国家/行业过度集中）
    
    Args:
        selected: 已选择的供应商列表
        diversity_factor: 多样化因子 ('country' 或 'industry')
        max_per_group: 每组最大数量
    
    Returns:
        多样化后的列表
    """
    if not selected:
        return []
    
    groups = {}
    diversified = []
    
    for supplier in selected:
        group_key = supplier.get(diversity_factor, 'unknown')
        
        if group_key not in groups:
            groups[group_key] = 0
        
        if groups[group_key] < max_per_group:
            diversified.append(supplier)
            groups[group_key] += 1
    
    return diversified


# ============================================================================
# 选择报告生成
# ============================================================================

def generate_selection_report(
    self_profile: Dict,
    selected_suppliers: List[Dict],
    assessments: Dict[str, Dict],
    total_candidates: int
) -> Dict:
    """
    生成选择报告
    
    Args:
        self_profile: 目标公司画像
        selected_suppliers: 选中的供应商列表
        assessments: 所有评估结果
        total_candidates: 总候选数量
    
    Returns:
        报告字典
    """
    # 评估统计
    scores = [a.get('overall_score', 0) for a in assessments.values()]
    recommendations = [a.get('recommendation', 'unknown') for a in assessments.values()]
    
    rec_counts = {}
    for rec in recommendations:
        rec_counts[rec] = rec_counts.get(rec, 0) + 1
    
    # 选中供应商国家分布
    country_dist = {}
    for sup in selected_suppliers:
        country = sup.get('country', 'Unknown')
        country_dist[country] = country_dist.get(country, 0) + 1
    
    # 选中供应商行业分布
    industry_dist = {}
    for sup in selected_suppliers:
        industry = sup.get('industry', 'Unknown')
        # 简化行业名称
        simple_industry = industry.split(',')[0].strip() if industry else 'Unknown'
        industry_dist[simple_industry] = industry_dist.get(simple_industry, 0) + 1
    
    report = {
        'target_company': {
            'factset_entity_id': self_profile.get('factset_entity_id'),
            'entity_name': self_profile.get('entity_name'),
            'industry': self_profile.get('industry'),
            'decision_year': self_profile.get('decision_year')
        },
        'candidate_statistics': {
            'total_candidates': total_candidates,
            'total_assessed': len(assessments),
            'total_selected': len(selected_suppliers)
        },
        'score_statistics': {
            'mean_score': round(np.mean(scores), 2) if scores else 0,
            'median_score': round(np.median(scores), 2) if scores else 0,
            'max_score': round(max(scores), 2) if scores else 0,
            'min_score': round(min(scores), 2) if scores else 0,
            'std_score': round(np.std(scores), 2) if scores else 0
        },
        'recommendation_distribution': rec_counts,
        'selection_diversity': {
            'countries': country_dist,
            'industries': industry_dist
        },
        'selected_suppliers': [
            {
                'rank': i + 1,
                'factset_entity_id': sup.get('factset_entity_id'),
                'entity_name': sup.get('entity_name'),
                'country': sup.get('country'),
                'overall_score': sup.get('overall_score'),
                'recommendation': sup.get('recommendation')
            }
            for i, sup in enumerate(selected_suppliers)
        ]
    }
    
    return report


def format_report_text(report: Dict) -> str:
    """
    格式化报告为文本
    
    Args:
        report: 报告字典
    
    Returns:
        格式化的文本报告
    """
    lines = []
    
    # 标题
    lines.append("=" * 60)
    lines.append("供应商选择报告")
    lines.append("=" * 60)
    lines.append("")
    
    # 目标公司
    target = report.get('target_company', {})
    lines.append(f"目标公司: {target.get('entity_name', 'Unknown')}")
    lines.append(f"公司ID: {target.get('factset_entity_id', 'Unknown')}")
    lines.append(f"行业: {target.get('industry', 'Unknown')}")
    lines.append(f"决策年份: {target.get('decision_year', 'Unknown')}")
    lines.append("")
    
    # 候选统计
    stats = report.get('candidate_statistics', {})
    lines.append("-" * 40)
    lines.append("候选统计")
    lines.append("-" * 40)
    lines.append(f"总候选数量: {stats.get('total_candidates', 0)}")
    lines.append(f"评估数量: {stats.get('total_assessed', 0)}")
    lines.append(f"选中数量: {stats.get('total_selected', 0)}")
    lines.append("")
    
    # 评分统计
    scores = report.get('score_statistics', {})
    lines.append("-" * 40)
    lines.append("评分统计")
    lines.append("-" * 40)
    lines.append(f"平均分: {scores.get('mean_score', 0)}")
    lines.append(f"中位数: {scores.get('median_score', 0)}")
    lines.append(f"最高分: {scores.get('max_score', 0)}")
    lines.append(f"最低分: {scores.get('min_score', 0)}")
    lines.append("")
    
    # 推荐分布
    rec_dist = report.get('recommendation_distribution', {})
    lines.append("-" * 40)
    lines.append("推荐分布")
    lines.append("-" * 40)
    for rec, count in sorted(rec_dist.items(), key=lambda x: -x[1]):
        lines.append(f"  {rec}: {count}")
    lines.append("")
    
    # 选中供应商
    selected = report.get('selected_suppliers', [])
    lines.append("-" * 40)
    lines.append("选中供应商 Top 10")
    lines.append("-" * 40)
    for sup in selected[:10]:
        lines.append(f"  {sup.get('rank')}. {sup.get('entity_name', 'Unknown')}")
        lines.append(f"     国家: {sup.get('country')} | 评分: {sup.get('overall_score')} | 推荐: {sup.get('recommendation')}")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


# ============================================================================
# 主函数 - 测试用
# ============================================================================

if __name__ == "__main__":
    print("=== Stage 5: 供应商选择测试 ===\n")
    
    # 模拟评估结果
    mock_assessments = {
        'supplier_001': {'overall_score': 8.5, 'recommendation': 'strong_yes', 'financial_stability': {'score': 9}},
        'supplier_002': {'overall_score': 7.8, 'recommendation': 'yes', 'financial_stability': {'score': 8}},
        'supplier_003': {'overall_score': 7.2, 'recommendation': 'yes', 'financial_stability': {'score': 7}},
        'supplier_004': {'overall_score': 6.5, 'recommendation': 'neutral', 'financial_stability': {'score': 6}},
        'supplier_005': {'overall_score': 5.1, 'recommendation': 'neutral', 'financial_stability': {'score': 5}},
        'supplier_006': {'overall_score': 4.2, 'recommendation': 'no', 'financial_stability': {'score': 4}},
    }
    
    mock_profiles = {
        'supplier_001': {'entity_name': 'Best Supplier Co', 'country': 'US', 'industry': 'Tech'},
        'supplier_002': {'entity_name': 'Good Supplier Inc', 'country': 'TW', 'industry': 'Electronics'},
        'supplier_003': {'entity_name': 'Solid Supplier Ltd', 'country': 'JP', 'industry': 'Manufacturing'},
        'supplier_004': {'entity_name': 'Average Supplier', 'country': 'CN', 'industry': 'Tech'},
        'supplier_005': {'entity_name': 'Basic Supplier', 'country': 'KR', 'industry': 'Materials'},
        'supplier_006': {'entity_name': 'Poor Supplier', 'country': 'VN', 'industry': 'Components'},
    }
    
    mock_self_profile = {
        'factset_entity_id': '000C7F-E',
        'entity_name': 'Apple Inc.',
        'industry': 'Technology',
        'decision_year': 2020
    }
    
    # 测试选择
    selected = select_top_suppliers(
        mock_assessments,
        mock_profiles,
        top_k=5,
        min_score=5.0
    )
    
    print("选中的供应商:")
    for i, sup in enumerate(selected):
        print(f"  {i+1}. {sup['entity_name']} (score: {sup['overall_score']}, rec: {sup['recommendation']})")
    
    # 测试报告
    report = generate_selection_report(
        mock_self_profile,
        selected,
        mock_assessments,
        total_candidates=100
    )
    
    print("\n" + format_report_text(report))
