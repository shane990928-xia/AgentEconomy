#!/usr/bin/env python3
"""
Stage 4: 供应商评估
==================

功能：
- LLM驱动的供应商评估
- 多维度评分
- 并行评估处理

用法：
    from stage4_assessment import (
        assess_single_supplier,
        parallel_assess_suppliers
    )
"""

import json
import concurrent.futures
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm

from stage0_config import LLMProcessor, _convert_for_json
from stage3_personality import build_personality_prompt


# ============================================================================
# 评估 Prompts
# ============================================================================

SUPPLIER_ASSESSMENT_PROMPT = '''You are a senior procurement manager evaluating potential suppliers.

{personality_prompt}

**Your Company Profile:**
- Company: {self_name}
- Industry: {self_industry}
- Annual Sales: ${self_sales:,.0f}
- Business Focus: {self_segments}

**Candidate Supplier:**
- Company: {supplier_name}
- Industry: {supplier_industry}
- Country: {supplier_country}
- Annual Sales: ${supplier_sales:,.0f}
- Net Margin: {supplier_margin:.1f}%
- Debt/Equity: {supplier_debt:.2f}
- Business Segments: {supplier_segments}
- Sales Ratio (Supplier/Your Company): {sales_ratio:.2f}

**Evaluation Task:**
Assess this supplier candidate on the following dimensions (score 1-10, where 10 is best):

1. **Financial Stability**: Can this supplier maintain consistent supply? Consider their financial health.
2. **Scale Fit**: Is their scale appropriate for your needs? Too small may lack capacity, too large may not prioritize you.
3. **Industry Relevance**: How relevant is their industry/expertise to your supply needs?
4. **Geographic Alignment**: Consider logistics, timezone, and trade relationships.
5. **Risk Profile**: Overall supply chain risk assessment.

Respond in JSON format:
{{
    "financial_stability": {{"score": 5, "reasoning": "..."}},
    "scale_fit": {{"score": 5, "reasoning": "..."}},
    "industry_relevance": {{"score": 5, "reasoning": "..."}},
    "geographic_alignment": {{"score": 5, "reasoning": "..."}},
    "risk_profile": {{"score": 5, "reasoning": "..."}},
    "overall_score": 5.0,
    "recommendation": "strong_yes|yes|neutral|no|strong_no",
    "summary": "One sentence summary of this supplier's suitability"
}}'''


# ============================================================================
# 默认评估 - 当LLM不可用时使用
# ============================================================================

def _generate_default_assessment(
    supplier_profile: Dict, 
    self_profile: Dict
) -> Dict:
    """基于规则生成默认评估"""
    supplier_summary = supplier_profile.get('financial_summary', {})
    self_summary = self_profile.get('financial_summary', {})
    relative = supplier_profile.get('relative_metrics', {})
    
    # 财务稳定性评分
    margin = supplier_summary.get('avg_margin')
    debt_eq = supplier_summary.get('avg_debt_equity')
    
    fin_score = 5
    if margin is not None:
        if margin > 10:
            fin_score += 2
        elif margin > 5:
            fin_score += 1
        elif margin < 0:
            fin_score -= 2
    if debt_eq is not None:
        if debt_eq < 0.5:
            fin_score += 1
        elif debt_eq > 2:
            fin_score -= 1
    fin_score = max(1, min(10, fin_score))
    
    # 规模匹配评分
    sales_ratio = relative.get('sales_ratio', 1.0)
    if sales_ratio is None:
        sales_ratio = 1.0
    
    scale_score = 5
    if 0.05 <= sales_ratio <= 0.5:  # 理想范围：供应商规模为客户的5%-50%
        scale_score = 8
    elif 0.01 <= sales_ratio <= 1.0:
        scale_score = 6
    elif sales_ratio > 1.0:  # 供应商比客户大
        scale_score = 4
    else:  # 供应商太小
        scale_score = 3
    
    # 行业相关性评分
    supplier_industry = supplier_profile.get('industry', '')
    self_industry = self_profile.get('industry', '')
    
    industry_score = 5
    if supplier_industry and self_industry:
        # 简单的行业匹配检查
        supplier_words = set(supplier_industry.lower().split())
        self_words = set(self_industry.lower().split())
        common = supplier_words & self_words
        if len(common) >= 2:
            industry_score = 8
        elif len(common) >= 1:
            industry_score = 6
    
    # 地理匹配评分
    supplier_country = supplier_profile.get('country', '')
    self_country = self_profile.get('country', '')
    
    geo_score = 5
    if supplier_country == self_country:
        geo_score = 8
    elif supplier_country in ['US', 'CA'] and self_country in ['US', 'CA']:
        geo_score = 7
    elif supplier_country in ['DE', 'FR', 'GB', 'IT', 'ES'] and self_country in ['DE', 'FR', 'GB', 'IT', 'ES']:
        geo_score = 7
    
    # 风险评分 (综合)
    risk_score = (fin_score + scale_score + geo_score) / 3
    risk_score = round(max(1, min(10, risk_score)))
    
    # 总体评分
    overall = (fin_score + scale_score + industry_score + geo_score + risk_score) / 5
    
    # 推荐等级
    if overall >= 7.5:
        recommendation = 'strong_yes'
    elif overall >= 6.0:
        recommendation = 'yes'
    elif overall >= 4.5:
        recommendation = 'neutral'
    elif overall >= 3.0:
        recommendation = 'no'
    else:
        recommendation = 'strong_no'
    
    return {
        'financial_stability': {'score': fin_score, 'reasoning': 'Rule-based assessment'},
        'scale_fit': {'score': scale_score, 'reasoning': f'Sales ratio: {sales_ratio:.2f}'},
        'industry_relevance': {'score': industry_score, 'reasoning': 'Industry keyword matching'},
        'geographic_alignment': {'score': geo_score, 'reasoning': f'{supplier_country} to {self_country}'},
        'risk_profile': {'score': risk_score, 'reasoning': 'Composite risk assessment'},
        'overall_score': round(overall, 2),
        'recommendation': recommendation,
        'summary': f'Default rule-based assessment for {supplier_profile.get("entity_name", "Unknown")}',
        'assessment_method': 'default_rules'
    }


# ============================================================================
# 单个供应商评估
# ============================================================================

def assess_single_supplier(
    supplier_profile: Dict,
    self_profile: Dict,
    personality: Dict,
    llm_processor: LLMProcessor = None,
    debug_mode: bool = False
) -> Dict:
    """
    评估单个供应商
    
    Args:
        supplier_profile: 供应商画像
        self_profile: 公司自身画像
        personality: 决策者性格
        llm_processor: LLM处理器
        debug_mode: 调试模式
    
    Returns:
        评估结果字典
    """
    # 如果没有LLM或调试模式，使用默认评估
    if llm_processor is None or debug_mode:
        return _generate_default_assessment(supplier_profile, self_profile)
    
    # 准备数据
    supplier_summary = supplier_profile.get('financial_summary', {})
    self_summary = self_profile.get('financial_summary', {})
    relative = supplier_profile.get('relative_metrics', {})
    
    supplier_segments = supplier_profile.get('segments', [])
    supplier_segment_names = ', '.join([s.get('seg_descr', 'Unknown') for s in supplier_segments[:3]]) if supplier_segments else 'N/A'
    
    self_segments = self_profile.get('segments', [])
    self_segment_names = ', '.join([s.get('seg_descr', 'Unknown') for s in self_segments[:3]]) if self_segments else 'N/A'
    
    # 构建prompt
    personality_prompt = build_personality_prompt(personality)
    
    prompt = SUPPLIER_ASSESSMENT_PROMPT.format(
        personality_prompt=personality_prompt,
        self_name=self_profile.get('entity_name', 'Unknown'),
        self_industry=self_profile.get('industry', 'Unknown'),
        self_sales=self_summary.get('avg_sales', 0) or 0,
        self_segments=self_segment_names,
        supplier_name=supplier_profile.get('entity_name', 'Unknown'),
        supplier_industry=supplier_profile.get('industry', 'Unknown'),
        supplier_country=supplier_profile.get('country', 'Unknown'),
        supplier_sales=supplier_summary.get('avg_sales', 0) or 0,
        supplier_margin=supplier_summary.get('avg_margin', 0) or 0,
        supplier_debt=supplier_summary.get('avg_debt_equity', 0) or 0,
        supplier_segments=supplier_segment_names,
        sales_ratio=relative.get('sales_ratio', 1.0) or 1.0
    )
    
    try:
        response = llm_processor.call(prompt)
        
        # 解析JSON响应
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                assessment = json.loads(json_str)
                assessment['assessment_method'] = 'llm'
                return assessment
        except json.JSONDecodeError:
            pass
            
        # 解析失败，返回默认评估
        return _generate_default_assessment(supplier_profile, self_profile)
        
    except Exception as e:
        result = _generate_default_assessment(supplier_profile, self_profile)
        result['error'] = str(e)
        return result


# ============================================================================
# 并行评估
# ============================================================================

def _assess_supplier_wrapper(args: Tuple) -> Tuple[str, Dict]:
    """单个供应商评估包装函数 - 用于并行处理"""
    supplier_id, supplier_profile, self_profile, personality = args
    
    # 在子进程中不使用LLM，使用默认评估
    assessment = _generate_default_assessment(supplier_profile, self_profile)
    return supplier_id, assessment


def parallel_assess_suppliers(
    supplier_profiles: Dict[str, Dict],
    self_profile: Dict,
    personality: Dict,
    llm_processor: LLMProcessor = None,
    max_workers: int = 16,
    debug_mode: bool = False
) -> Dict[str, Dict]:
    """
    并行评估多个供应商
    
    注意：由于LLM API调用的特性，这里使用ThreadPoolExecutor而非ProcessPoolExecutor
    
    Args:
        supplier_profiles: {supplier_id: profile} 字典
        self_profile: 公司自身画像
        personality: 决策者性格
        llm_processor: LLM处理器
        max_workers: 最大并行数
        debug_mode: 调试模式
    
    Returns:
        {supplier_id: assessment} 字典
    """
    if not supplier_profiles:
        return {}
    
    print(f"  -> 开始评估 {len(supplier_profiles)} 个供应商...")
    
    assessments = {}
    
    # 如果使用LLM，用线程池并行（LLM API调用是IO密集型）
    if llm_processor is not None and not debug_mode:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for supplier_id, profile in supplier_profiles.items():
                future = executor.submit(
                    assess_single_supplier,
                    profile, self_profile, personality, llm_processor, False
                )
                futures[future] = supplier_id
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                              total=len(futures),
                              desc="Assessing Suppliers"):
                supplier_id = futures[future]
                try:
                    assessment = future.result()
                    assessments[supplier_id] = assessment
                except Exception as e:
                    assessments[supplier_id] = {
                        'error': str(e),
                        'overall_score': 0,
                        'recommendation': 'no'
                    }
    else:
        # 调试模式或无LLM：串行执行默认评估
        for supplier_id, profile in tqdm(supplier_profiles.items(), 
                                         desc="Assessing Suppliers"):
            assessment = _generate_default_assessment(profile, self_profile)
            assessments[supplier_id] = assessment
    
    print(f"  -> 完成 {len(assessments)} 个供应商评估")
    
    return assessments


def rank_assessments(assessments: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
    """
    根据评估结果排序
    
    Args:
        assessments: {supplier_id: assessment} 字典
    
    Returns:
        排序后的 [(supplier_id, assessment), ...] 列表
    """
    return sorted(
        assessments.items(),
        key=lambda x: x[1].get('overall_score', 0),
        reverse=True
    )


def filter_assessments(
    assessments: Dict[str, Dict],
    min_score: float = 0,
    recommendations: List[str] = None
) -> Dict[str, Dict]:
    """
    过滤评估结果
    
    Args:
        assessments: {supplier_id: assessment} 字典
        min_score: 最低分数阈值
        recommendations: 允许的推荐等级列表
    
    Returns:
        过滤后的字典
    """
    filtered = {}
    
    for supplier_id, assessment in assessments.items():
        score = assessment.get('overall_score', 0)
        rec = assessment.get('recommendation', '')
        
        if score < min_score:
            continue
        
        if recommendations and rec not in recommendations:
            continue
        
        filtered[supplier_id] = assessment
    
    return filtered


# ============================================================================
# 主函数 - 测试用
# ============================================================================

if __name__ == "__main__":
    print("=== Stage 4: 供应商评估测试 ===\n")
    
    # 模拟数据
    mock_self_profile = {
        'factset_entity_id': '000C7F-E',
        'entity_name': 'Apple Inc.',
        'industry': 'Technology Hardware, Storage & Peripherals',
        'country': 'US',
        'segments': [{'seg_descr': 'iPhone'}],
        'financial_summary': {
            'avg_sales': 274515000000,
            'avg_margin': 21.35,
            'avg_debt_equity': 1.73
        }
    }
    
    mock_supplier_profile = {
        'factset_entity_id': '001234-E',
        'entity_name': 'Test Supplier Inc.',
        'industry': 'Electronic Components',
        'country': 'TW',
        'segments': [{'seg_descr': 'Semiconductors'}],
        'financial_summary': {
            'avg_sales': 50000000000,
            'avg_margin': 15.5,
            'avg_debt_equity': 0.8
        },
        'relative_metrics': {
            'sales_ratio': 0.18
        }
    }
    
    mock_personality = {
        'risk_tolerance': {'score': 5},
        'cost_focus': {'score': 6},
        'quality_priority': {'score': 8}
    }
    
    # 测试默认评估
    assessment = assess_single_supplier(
        mock_supplier_profile,
        mock_self_profile,
        mock_personality,
        debug_mode=True
    )
    
    print("评估结果:")
    for key, value in assessment.items():
        if isinstance(value, dict):
            print(f"  {key}: score={value.get('score')}, reasoning={value.get('reasoning', '')[:50]}")
        else:
            print(f"  {key}: {value}")
    
    # 测试排序
    mock_assessments = {
        'supplier_1': {'overall_score': 7.5, 'recommendation': 'yes'},
        'supplier_2': {'overall_score': 8.2, 'recommendation': 'strong_yes'},
        'supplier_3': {'overall_score': 4.1, 'recommendation': 'no'},
    }
    
    print("\n排序结果:")
    ranked = rank_assessments(mock_assessments)
    for i, (sid, ass) in enumerate(ranked):
        print(f"  {i+1}. {sid}: score={ass['overall_score']}, rec={ass['recommendation']}")
