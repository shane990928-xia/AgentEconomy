#!/usr/bin/env python3
"""
Stage 0: 配置和工具类
====================

功能：
- 全局配置参数
- LLM代理创建
- 缓存管理类
- LLM处理器

用法：
    from stage0_config import (
        DATA_BASE_PATH, RESULT_DIR, CACHE_DIR,
        create_llm_agent, LLMCache, ResultCache, LLMProcessor
    )
"""

import os
import json
import pandas as pd
from typing import Dict, Optional, Any
from datetime import datetime

# ============================================================================
# 全局配置
# ============================================================================

import pathlib as _pathlib
# __file__ = supply_chain/code/supplier/stage0_config.py
# parents[2] = supply_chain/
_SUPPLY_CHAIN_DIR = _pathlib.Path(__file__).resolve().parents[2]
DATA_BASE_PATH = str(_SUPPLY_CHAIN_DIR / "data" / "factset")
RESULT_DIR     = str(_SUPPLY_CHAIN_DIR / "result")
CACHE_DIR      = str(_SUPPLY_CHAIN_DIR / "result" / "cache")

# ============================================================================
# Mock LLM类 - 用于Debug模式
# ============================================================================

class MockLLM:
    """可序列化的Mock LLM类，用于debug模式"""
    def __init__(self, api_key: str = None):
        if api_key: 
            self.api_key = api_key
        print("LLM (Placeholder) Initialized. API calls will be mocked.")

    def chat(self, prompt: str) -> str:
        # Supplier-specific mock responses
        
        # 1. Historical Supplier Selection Profile
        if "Historical Supplier Selection Profile(s)" in prompt:
            return json.dumps({
                "decision_context": "[Behaviorally-Informed Context] Based on a consistent pattern of selecting financially stable, Tier-1 partners, this company is a Process-Driven Risk Avoider."
            })
        
        # 2. Corporate DNA Context
        elif "corporate DNA" in prompt:
            return json.dumps({
                "decision_context": "[Profile-Only Context] This is a Power-Driven technology leader."
            })
        
        # 3. Supplier Profile Assessment  
        elif "Supplier Profile" in prompt and "should_select" not in prompt:
            return json.dumps({
                "supplier_name": "Mock Supplier Inc.", 
                "supplier_entity_id": "MOCK-EID-01",
                "Supplier_Tier": "Tier 1: Strategic", 
                "Financial_Risk_Score": "Low",
                "Single_Source_Flag": "N", 
                "Concentration_Risk_Flag": "N",
                "Geographic_Risk": "Low",
                "Innovation_Score": "High",
                "Quality_Alignment": "Excellent",
                "Capacity_Adequacy": "Adequate",
                "Negotiation_Leverage_Score": "Medium",
                "Cost_Position": "Competitive",
                "Sourcing_Recommendation": "Deepen Partnership"
            })
        
        # 4. Supplier Selection Quantity  
        elif "estimated_supplier_count" in prompt:
            return json.dumps({"estimated_supplier_count": 25})
        
        # 5. Supplier Selection Decision (One-step)
        elif "select your top" in prompt.lower() or "selected_supplier_ids" in prompt:
            return json.dumps({
                "selected_supplier_ids": ["MOCK-SUPP-001", "MOCK-SUPP-002", "MOCK-SUPP-003"]
            })
        
        # 6. Portfolio Adjustment Decision
        elif "should_terminate" in prompt or "should_initiate" in prompt:
            return json.dumps({
                "should_terminate": "No", 
                "should_initiate": "Yes",
                "confidence_score": 0.82,
                "reasoning": "Strong supplier performance and strategic alignment"
            })
        
        # Default fallback
        return json.dumps({
            "status": "mock_response",
            "message": "Debug mode active - mock data returned"
        })


def create_llm_agent(debug_mode: bool = False):
    """创建LLM代理，支持debug模式"""
    if debug_mode:
        print("[DEBUG] Debug模式已启用，使用placeholder LLM")
        return MockLLM()
    else:
        try:
            from llm import LLM
            print("使用真实LLM API")
            return LLM()
        except ImportError:
            print("[WARNING] llm.py not found. Falling back to placeholder LLM.")
            return MockLLM()


# ============================================================================
# 缓存管理类
# ============================================================================

class LLMCache:
    """LLM响应缓存"""
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.cache = self._load()
    
    def _load(self) -> Dict:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def save(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
    def get(self, key: str) -> Optional[str]:
        return self.cache.get(key)
    
    def set(self, key: str, value: str):
        self.cache[key] = value


class ResultCache:
    """结果缓存"""
    def __init__(self, cache_path: str):
        self.cache_path = cache_path.replace('.json', '_results.json')
        self.cache = self._load()
    
    def _load(self) -> Dict:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def save(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
    def get_year_result(self, year: int, entity_id: str, config_str: str) -> Optional[Dict]:
        key = f"{year}_{entity_id}_{config_str}"
        return self.cache.get(key)
    
    def set_year_result(self, year: int, entity_id: str, config_str: str, result: Dict):
        key = f"{year}_{entity_id}_{config_str}"
        self.cache[key] = result


# ============================================================================
# LLM处理器
# ============================================================================

class LLMProcessor:
    """LLM处理器，整合缓存和调用"""
    def __init__(self, agent, cache: LLMCache, result_cache: ResultCache = None):
        self.agent = agent
        self.cache = cache
        self.result_cache = result_cache
    
    def call(self, prompt: str, cache_key: str = None) -> str:
        """调用LLM，支持缓存"""
        if cache_key:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        
        response = self.agent.chat(prompt)
        
        if cache_key and response:
            self.cache.set(cache_key, response)
        
        return response
    
    def parse_json_response(self, response: str) -> Optional[Dict]:
        """解析LLM的JSON响应"""
        if not response:
            return None
        
        import re
        # 尝试提取JSON块
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # 尝试找到第一个 { 和最后一个 }
        try:
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                return json.loads(response[start:end+1])
        except json.JSONDecodeError:
            pass
        
        return None


# ============================================================================
# 辅助函数
# ============================================================================

def _convert_for_json(value):
    """JSON序列化辅助函数"""
    import numpy as np
    if pd.isna(value): 
        return None
    if isinstance(value, np.generic): 
        return value.item()
    if isinstance(value, pd.Timestamp): 
        return value.strftime('%Y-%m-%d')
    return value
