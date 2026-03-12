#!/usr/bin/env python3
"""
Stage 3: LLM 性格生成
=====================

功能：
- 基于公司画像生成AI决策者性格
- 管理性格一致性缓存
- 提供性格提示词构建

用法：
    from stage3_personality import (
        generate_decision_maker_personality,
        build_personality_prompt
    )
"""

import json
import os
import hashlib
from typing import Dict, Optional, Any

from stage0_config import LLMCache, LLMProcessor, CACHE_DIR


# ============================================================================
# 性格生成 Prompts
# ============================================================================

PERSONALITY_GENERATION_PROMPT = '''You are an expert in business psychology and organizational behavior.

Based on the following company profile, generate a realistic personality profile for a senior procurement/supply chain decision-maker at this company. Consider the company's industry, size, financial health, and strategic position.

Company Profile:
- Company Name: {company_name}
- Industry: {industry}
- Country: {country}
- Annual Sales: {sales}
- Net Margin: {margin}
- Debt/Equity Ratio: {debt_equity}
- Business Segments: {segments}

Generate a decision-maker personality with the following attributes (rate each 1-10 and provide brief explanation):

1. Risk Tolerance: How willing to take calculated risks in supplier selection
2. Cost Focus: Priority on cost reduction vs. other factors
3. Innovation Orientation: Preference for innovative vs. established suppliers
4. Relationship Value: Importance of long-term relationships vs. transactional approach
5. Quality Priority: Emphasis on quality over cost
6. ESG Consciousness: Environmental and social responsibility consideration
7. Domestic Preference: Preference for local/domestic suppliers

Respond in JSON format:
{{
    "risk_tolerance": {{"score": 5, "reasoning": "..."}},
    "cost_focus": {{"score": 5, "reasoning": "..."}},
    "innovation_orientation": {{"score": 5, "reasoning": "..."}},
    "relationship_value": {{"score": 5, "reasoning": "..."}},
    "quality_priority": {{"score": 5, "reasoning": "..."}},
    "esg_consciousness": {{"score": 5, "reasoning": "..."}},
    "domestic_preference": {{"score": 5, "reasoning": "..."}},
    "overall_profile": "Brief 2-3 sentence summary of this decision-maker's approach"
}}'''


# ============================================================================
# 默认性格 - 当LLM不可用时使用
# ============================================================================

DEFAULT_PERSONALITY = {
    "risk_tolerance": {"score": 5, "reasoning": "Average risk tolerance for established company"},
    "cost_focus": {"score": 6, "reasoning": "Moderate focus on cost efficiency"},
    "innovation_orientation": {"score": 5, "reasoning": "Balanced approach to innovation"},
    "relationship_value": {"score": 6, "reasoning": "Values stable supplier relationships"},
    "quality_priority": {"score": 7, "reasoning": "Quality is important for brand reputation"},
    "esg_consciousness": {"score": 5, "reasoning": "Growing awareness of ESG factors"},
    "domestic_preference": {"score": 5, "reasoning": "No strong preference for location"},
    "overall_profile": "A balanced decision-maker who weighs multiple factors with slight emphasis on quality and cost."
}


# ============================================================================
# 性格缓存管理
# ============================================================================

class PersonalityCache:
    """性格缓存 - 确保同一公司同一年份的性格一致性"""
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.path.join(CACHE_DIR, 'personality')
        os.makedirs(self.cache_dir, exist_ok=True)
        self._memory_cache: Dict[str, Dict] = {}
    
    def _get_cache_key(self, entity_id: str, year: int) -> str:
        """生成缓存键"""
        return f"{entity_id}_{year}"
    
    def _get_cache_file(self, cache_key: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def get(self, entity_id: str, year: int) -> Optional[Dict]:
        """获取缓存的性格"""
        cache_key = self._get_cache_key(entity_id, year)
        
        # 先检查内存缓存
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
        
        # 检查文件缓存
        cache_file = self._get_cache_file(cache_key)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    personality = json.load(f)
                    self._memory_cache[cache_key] = personality
                    return personality
            except (json.JSONDecodeError, IOError):
                pass
        
        return None
    
    def set(self, entity_id: str, year: int, personality: Dict):
        """设置性格缓存"""
        cache_key = self._get_cache_key(entity_id, year)
        
        # 内存缓存
        self._memory_cache[cache_key] = personality
        
        # 文件缓存
        cache_file = self._get_cache_file(cache_key)
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(personality, f, ensure_ascii=False, indent=2)
        except IOError:
            pass


# 全局性格缓存实例
_personality_cache = PersonalityCache()


# ============================================================================
# 性格生成函数
# ============================================================================

def generate_decision_maker_personality(
    self_profile: Dict,
    llm_processor: LLMProcessor = None,
    use_cache: bool = True,
    debug_mode: bool = False
) -> Dict:
    """
    生成公司决策者性格
    
    Args:
        self_profile: 公司画像
        llm_processor: LLM处理器 (可选)
        use_cache: 是否使用缓存
        debug_mode: 调试模式
    
    Returns:
        性格字典
    """
    entity_id = self_profile.get('factset_entity_id', '')
    year = self_profile.get('decision_year', 2020)
    
    # 检查缓存
    if use_cache:
        cached = _personality_cache.get(entity_id, year)
        if cached:
            return cached
    
    # 构建prompt
    financial_summary = self_profile.get('financial_summary', {})
    segments = self_profile.get('segments', [])
    segment_names = [s.get('seg_descr', 'Unknown') for s in segments[:3]] if segments else ['N/A']
    
    prompt = PERSONALITY_GENERATION_PROMPT.format(
        company_name=self_profile.get('entity_name', 'Unknown'),
        industry=self_profile.get('industry', 'Unknown'),
        country=self_profile.get('country', 'Unknown'),
        sales=financial_summary.get('avg_sales', 'N/A'),
        margin=financial_summary.get('avg_margin', 'N/A'),
        debt_equity=financial_summary.get('avg_debt_equity', 'N/A'),
        segments=', '.join(segment_names)
    )
    
    # 如果没有LLM处理器或在调试模式，返回默认性格
    if llm_processor is None or debug_mode:
        personality = DEFAULT_PERSONALITY.copy()
        personality['generation_method'] = 'default'
    else:
        try:
            response = llm_processor.call(prompt)
            
            # 尝试解析JSON响应
            try:
                # 查找JSON部分
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    personality = json.loads(json_str)
                    personality['generation_method'] = 'llm'
                else:
                    personality = DEFAULT_PERSONALITY.copy()
                    personality['generation_method'] = 'default_parse_error'
            except json.JSONDecodeError:
                personality = DEFAULT_PERSONALITY.copy()
                personality['generation_method'] = 'default_json_error'
                
        except Exception as e:
            personality = DEFAULT_PERSONALITY.copy()
            personality['generation_method'] = 'default_exception'
            personality['error'] = str(e)
    
    # 缓存结果
    if use_cache:
        _personality_cache.set(entity_id, year, personality)
    
    return personality


def build_personality_prompt(personality: Dict) -> str:
    """
    将性格转换为提示词片段
    
    Args:
        personality: 性格字典
    
    Returns:
        性格描述提示词
    """
    if not personality:
        return "You are a balanced procurement decision-maker."
    
    # 提取分数
    scores = {
        'risk_tolerance': personality.get('risk_tolerance', {}).get('score', 5),
        'cost_focus': personality.get('cost_focus', {}).get('score', 5),
        'innovation_orientation': personality.get('innovation_orientation', {}).get('score', 5),
        'relationship_value': personality.get('relationship_value', {}).get('score', 5),
        'quality_priority': personality.get('quality_priority', {}).get('score', 5),
        'esg_consciousness': personality.get('esg_consciousness', {}).get('score', 5),
        'domestic_preference': personality.get('domestic_preference', {}).get('score', 5),
    }
    
    # 构建性格描述
    traits = []
    
    if scores['risk_tolerance'] >= 7:
        traits.append("willing to take calculated risks")
    elif scores['risk_tolerance'] <= 3:
        traits.append("risk-averse and preferring proven suppliers")
    
    if scores['cost_focus'] >= 7:
        traits.append("highly cost-conscious")
    elif scores['cost_focus'] <= 3:
        traits.append("less concerned with cost, prioritizing other factors")
    
    if scores['innovation_orientation'] >= 7:
        traits.append("seeking innovative solutions")
    elif scores['innovation_orientation'] <= 3:
        traits.append("preferring established and reliable approaches")
    
    if scores['relationship_value'] >= 7:
        traits.append("valuing long-term supplier relationships")
    elif scores['relationship_value'] <= 3:
        traits.append("transactional in supplier dealings")
    
    if scores['quality_priority'] >= 7:
        traits.append("prioritizing quality above all")
    
    if scores['esg_consciousness'] >= 7:
        traits.append("highly conscious of ESG factors")
    
    if scores['domestic_preference'] >= 7:
        traits.append("preferring domestic suppliers")
    elif scores['domestic_preference'] <= 3:
        traits.append("open to global supplier options")
    
    overall = personality.get('overall_profile', '')
    
    if traits:
        prompt = f"You are a procurement decision-maker who is {', '.join(traits)}."
        if overall:
            prompt += f" {overall}"
    elif overall:
        prompt = f"You are a procurement decision-maker. {overall}"
    else:
        prompt = "You are a balanced procurement decision-maker."
    
    return prompt


def get_personality_scores(personality: Dict) -> Dict[str, int]:
    """
    提取性格分数
    
    Args:
        personality: 性格字典
    
    Returns:
        {trait_name: score} 字典
    """
    return {
        'risk_tolerance': personality.get('risk_tolerance', {}).get('score', 5),
        'cost_focus': personality.get('cost_focus', {}).get('score', 5),
        'innovation_orientation': personality.get('innovation_orientation', {}).get('score', 5),
        'relationship_value': personality.get('relationship_value', {}).get('score', 5),
        'quality_priority': personality.get('quality_priority', {}).get('score', 5),
        'esg_consciousness': personality.get('esg_consciousness', {}).get('score', 5),
        'domestic_preference': personality.get('domestic_preference', {}).get('score', 5),
    }


# ============================================================================
# 主函数 - 测试用
# ============================================================================

if __name__ == "__main__":
    print("=== Stage 3: LLM 性格生成测试 ===\n")
    
    # 测试用的模拟画像
    test_profile = {
        'factset_entity_id': '000C7F-E',
        'entity_name': 'Apple Inc.',
        'industry': 'Technology Hardware, Storage & Peripherals',
        'country': 'US',
        'decision_year': 2020,
        'segments': [
            {'seg_descr': 'iPhone'},
            {'seg_descr': 'Services'},
            {'seg_descr': 'Mac'}
        ],
        'financial_summary': {
            'avg_sales': 274515000000,
            'avg_margin': 21.35,
            'avg_debt_equity': 1.73
        }
    }
    
    # 生成性格 (调试模式，不使用实际LLM)
    personality = generate_decision_maker_personality(test_profile, debug_mode=True)
    
    print("生成的性格:")
    for key, value in personality.items():
        if isinstance(value, dict):
            print(f"  {key}: score={value.get('score')}, reasoning={value.get('reasoning', '')[:50]}...")
        else:
            print(f"  {key}: {value}")
    
    print("\n性格提示词:")
    prompt = build_personality_prompt(personality)
    print(f"  {prompt}")
    
    print("\n性格分数:")
    scores = get_personality_scores(personality)
    for trait, score in scores.items():
        print(f"  {trait}: {score}")
