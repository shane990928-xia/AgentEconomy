#!/usr/bin/env python3
"""
Stage 3a: LLM Procurement Personality Generation (Evolution Mode)
==================================================================

功能：为所有Hub节点（核心决策者）生成采购决策性格

输入：
- stage1_metadata_{year}.json (获取Hub节点列表，可选start_year)
- stage2_profiles_{year}.pkl (获取Hub画像)
- stage5a_llm_selection_{prev_years}.pkl (可选，用于PROFILE_WITH_PREV_YEAR/ALL_PRIOR模式)

输出：
- stage3a_personalities_{year}.pkl
  格式：{
      'hub_id': {
          'decision_context': str,
          'decision_context_prompt': str,
          'company_name': str,
          'personality_mode': str,  # 实际使用的模式
          'generated_at': str,
          'llm_response_full': dict
      }
  }

三种性格生成模式：
1. PROFILE_ONLY: 仅基于当年画像生成
2. PROFILE_WITH_PREV_YEAR: 基于当年画像 + 前一年选择行为
3. PROFILE_WITH_ALL_PRIOR: 基于当年画像 + 所有历史选择行为

自动退化逻辑：
- 第一年（year == start_year+1）: 强制使用PROFILE_ONLY
- 第二年（year == start_year+2）: 强制使用PROFILE_WITH_PREV_YEAR
- 第三年及以后: 可以使用PROFILE_WITH_ALL_PRIOR

start_year自动推测：
1. 优先从metadata读取start_year字段
2. 否则扫描历史stage5a_llm_selection_*.pkl文件，取最早年份-1
3. 否则默认为当前年份-1

固定使用LLM方法：
- 历史选择档案固定从stage5a_llm_selection_{year}.pkl加载

依赖关系：
- ⚠️ 需要Stage 1和Stage 2的输出
- ⚠️ 需要LLM API（可选debug模式使用Mock）
- ⚠️ PROFILE_WITH_PREV_YEAR/ALL_PRIOR模式需要前面年份的Stage 5a输出

参考：
- network/code/stage3a_llm_personality.py
- network_continuous_simulation.py: generate_procurement_personality()
"""

import pandas as pd
import numpy as np
import os
import pickle
import argparse
import json
import sys
import time
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')

# 添加父目录到sys.path以导入llm模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# ============================================================================
# 全局配置
# ============================================================================

RESULT_DIR = "../result"
CACHE_DIR = "../cache"

# ============================================================================
# 辅助函数
# ============================================================================

def _convert_for_json(value):
    """JSON转换辅助函数"""
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.strftime('%Y-%m-%d')
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def _parse_json_from_llm_response(response_str: str) -> Optional[Dict]:
    """
    从LLM响应中解析JSON
    支持多种格式：纯JSON、markdown包裹的JSON等
    """
    if not response_str:
        return None

    # 尝试1：直接解析
    try:
        return json.loads(response_str)
    except json.JSONDecodeError:
        pass

    # 尝试2：提取markdown代码块中的JSON
    json_match = re.search(r'```json\s*\n(.*?)\n```', response_str, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试3：提取任意代码块
    json_match = re.search(r'```\s*\n(.*?)\n```', response_str, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试4：查找{}包裹的JSON
    json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ============================================================================
# LLM相关类
# ============================================================================

class LLMCache:
    """LLM响应缓存（线程安全）"""
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.cache = self._load()
        self.dirty = False
        self.lock = threading.Lock()

    def _load(self) -> Dict[str, Any]:
        """加载缓存"""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"  [WARNING] 加载缓存失败: {e}")
                return {}
        return {}

    def get(self, key: str) -> Optional[Any]:
        """获取缓存（线程安全）"""
        with self.lock:
            return self.cache.get(key)

    def set(self, key: str, value: Any):
        """设置缓存（线程安全）"""
        with self.lock:
            self.cache[key] = value
            self.dirty = True

    def save(self):
        """保存缓存（线程安全）"""
        with self.lock:
            if not self.dirty:
                return

            try:
                os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
                with open(self.cache_path, 'w', encoding='utf-8') as f:
                    json.dump(self.cache, f, indent=2, ensure_ascii=False)
                print(f"  ✓ 缓存已保存到: {self.cache_path}")
                self.dirty = False
            except Exception as e:
                print(f"  [WARNING] 保存缓存失败: {e}")


class LLMProcessor:
    """LLM处理器，包含重试逻辑"""
    def __init__(self, agent: Any, cache: LLMCache):
        self.agent = agent
        self.cache = cache

    def _create_cache_key(self, prompt_type: str, entity_id: str, year: int, personality_mode: str) -> str:
        """创建缓存键"""
        return f"{prompt_type}_{entity_id}_{year}_{personality_mode}"

    def chat_with_retry(
        self,
        prompt: str,
        entity_id: str,
        year: int,
        personality_mode: str,
        retry_delay: int = 5
    ) -> Dict:
        """
        调用LLM API并进行重试（无限重试直到成功）

        Args:
            prompt: prompt文本
            entity_id: 实体ID
            year: 年份
            personality_mode: 性格生成模式
            retry_delay: 重试延迟（秒）

        Returns:
            解析后的JSON字典
        """
        cache_key = self._create_cache_key("personality", entity_id, year, personality_mode)

        # 检查缓存
        cached_response = self.cache.get(cache_key)
        if cached_response is not None:
            return cached_response

        # 无限重试直到成功
        attempt = 0
        while True:
            attempt += 1
            try:
                response_content = self.agent.chat(prompt)

                if response_content:
                    parsed_json = _parse_json_from_llm_response(response_content)
                    if parsed_json and 'decision_context' in parsed_json:
                        # 缓存成功的响应
                        self.cache.set(cache_key, parsed_json)
                        return parsed_json
                    else:
                        print(f"  [WARNING] {entity_id}: JSON解析失败或缺少decision_context字段 (尝试 #{attempt})")
                else:
                    print(f"  [WARNING] {entity_id}: LLM返回空响应 (尝试 #{attempt})")

            except Exception as e:
                print(f"  [ERROR] {entity_id}: LLM调用异常 (尝试 #{attempt}): {e}")

            # 等待后重试
            print(f"  → 将在 {retry_delay} 秒后重试...")
            time.sleep(retry_delay)


def create_llm_agent(debug_mode: bool = False):
    """
    创建LLM Agent

    Args:
        debug_mode: 是否使用调试模式（Mock LLM）

    Returns:
        LLM实例
    """
    if debug_mode:
        print("  [DEBUG] 使用Mock LLM")

        class MockLLM:
            def chat(self, prompt: str) -> str:
                """返回Mock响应"""
                return json.dumps({
                    "decision_context": "This is a mock procurement personality generated for testing purposes. "
                                      "The company prioritizes quality and reliability over cost."
                })

        return MockLLM()
    else:
        try:
            from llm import LLM
            print("  ✓ 使用真实LLM API")
            return LLM()
        except ImportError as e:
            print(f"  [WARNING] 无法导入llm模块: {e}")
            print("  → 回退到Mock LLM")
            return create_llm_agent(debug_mode=True)


# ============================================================================
# Prompt模板
# ============================================================================

def create_personality_generation_prompt_base(self_profile: Dict) -> str:
    """
    基础采购性格生成prompt（只基于公司画像）
    """
    self_profile_json = json.dumps(self_profile, indent=2, default=_convert_for_json)
    return f"""
# IDENTITY AND GOAL
You are a senior partner at a top-tier management consulting firm, an expert in analyzing corporate DNA. Your task is to analyze objective company data and translate it into a clear, actionable set of supplier selection decision principles.

# INPUT DATA: Corporate Profile
This is the data for the company you are analyzing. Pay attention to its industry, scale (FF_SALES), profitability (FF_NET_MGN), and investment patterns (FF_RD_EXP, FF_CAPEX).
{self_profile_json}

# YOUR TASK
Based on the provided corporate profile, derive a **core decision principle** for each of the following dimensions:

- **Risk Tolerance Framework (risk_tolerance)**: Based on the company's financial strength and industry position, what risk thresholds should it set?
- **Value Trade-off Priorities (value_priorities)**: How should the company create a clear ranking of cost, quality, innovation, and reliability?
- **Supplier Relationship Strategy (relationship_strategy)**: Should the company favor deep partnerships with fewer suppliers or diversified risk distribution?
- **Geographic Layout Preferences (geographic_preferences)**: Based on business characteristics, should the company prioritize local or global suppliers?
- **Business Synergy Requirements (synergy_requirements)**: Should the company prioritize suppliers with high business synergy or seek complementary capabilities?

**CRITICAL INSTRUCTION**: For every principle you derive, you MUST explicitly state the data points it is based on in the `"rationale"`. For example, write "Based on the high FF_RD_EXP spending...". Do NOT make generic statements unsupported by data.

# OUTPUT SPECIFICATION
You MUST respond with a single, valid JSON object containing one key: "decision_context". The value should be a brief, summary paragraph describing the company's overall supplier selection personality and guiding principles as a string.

Example output:
{{
  "decision_context": "This is an industry leader driven by technological innovation and financial strength. The core of its supply chain decisions is to secure its technological lead and maintain production stability, rather than cost minimization. The company prioritizes suppliers with 'Low' or 'Medium' financial risk and values Innovation > Quality > Reliability > Cost due to its high R&D spending."
}}
"""


def create_personality_generation_prompt_behavioral_prev_year(self_profile: Dict, historical_profiles: List[Dict]) -> str:
    """
    采购性格生成prompt（基于画像 + 前一年选择行为）

    Args:
        self_profile: 公司画像
        historical_profiles: 历史选择行为档案列表（只使用最后一年）
    """
    self_profile_json = json.dumps(self_profile, indent=2, default=_convert_for_json)

    # 只取最后一年的数据
    latest_profile = historical_profiles[-1]
    year = latest_profile['year']
    selection_data = latest_profile['selection_profile']

    selection_summary_str = f"""
### {year} Year Supplier Selection Summary:
- Total Selected Suppliers: {selection_data.get('total_selected', 0)}

#### Selected Supplier Characteristics Distribution:
- Supplier Tier Distribution: {selection_data.get('tier_distribution', {})}
- Financial Risk Distribution: {selection_data.get('financial_risk_distribution', {})}
- Single Source Distribution: {selection_data.get('single_source_distribution', {})}
- Concentration Risk Distribution: {selection_data.get('concentration_risk_distribution', {})}
- Negotiation Leverage Distribution: {selection_data.get('negotiation_leverage_distribution', {})}
- Sourcing Recommendation Distribution: {selection_data.get('sourcing_recommendation_distribution', {})}
- Country Distribution: {selection_data.get('country_distribution', {})}
- Business Segment Distribution: {selection_data.get('segment_distribution', {})}
"""

    return f"""
# IDENTITY AND GOAL
You are a senior partner at a top-tier management consulting firm, an expert in analyzing corporate DNA. Your task is to synthesize a company's static profile with its previous year's actual supplier selection behavior to reveal the company's true supply chain decision-making philosophy.

# INPUT DATA 1: Corporate Profile (As of end of {year})
This data shows the company characteristics based on financial metrics and business structure.
{self_profile_json}

# INPUT DATA 2: {year} Year Actual Supplier Selection Behavior
This data shows the company's **actual** choice pattern over the previous year, which directly reveals the company's decision priorities.
{selection_summary_str}

# YOUR TASK
By synthesizing company profile with actual selection behavior from the previous year, extract the company's **true and revealed** supplier selection philosophy:

- **Risk Tolerance (revealed truth)**: What risk levels did the company actually choose, transcending verbal statements?
- **Value Trade-off (behavioral proof)**: What value priorities are proven by actual choices, not what the profile suggests?
- **Supplier Relationship (practical strategy)**: What relationship model is embodied in actual selections?
- **Geographic Layout (actual preferences)**: What geographic preferences are revealed by distribution data?
- **Business Synergy (selection reality)**: What synergy strategy is reflected in actual segment distributions?

**Remember: behavior is more reliable than static profile. Let actual selections guide your extraction while balancing fundamental characteristics.**

# OUTPUT SPECIFICATION
You MUST respond with a single, valid JSON object containing one key: "decision_context". The value should be the descriptive paragraph as a string.
"""


def create_personality_generation_prompt_behavioral_all_prior(self_profile: Dict, historical_profiles: List[Dict]) -> str:
    """
    采购性格生成prompt（基于画像 + 所有历史选择行为）

    Args:
        self_profile: 公司画像
        historical_profiles: 所有历史选择行为档案列表
    """
    self_profile_json = json.dumps(self_profile, indent=2, default=_convert_for_json)

    historical_summary_str = ""
    for profile in historical_profiles:
        year = profile['year']
        selection_data = profile['selection_profile']

        historical_summary_str += f"""
### {year} Year Supplier Selection Profile:
- Total Selected: {selection_data.get('total_selected', 0)}
- Supplier Tier Distribution: {selection_data.get('tier_distribution', {})}
- Financial Risk Distribution: {selection_data.get('financial_risk_distribution', {})}
- Single Source Distribution: {selection_data.get('single_source_distribution', {})}
- Concentration Risk Distribution: {selection_data.get('concentration_risk_distribution', {})}
- Negotiation Leverage Distribution: {selection_data.get('negotiation_leverage_distribution', {})}
- Sourcing Recommendation Distribution: {selection_data.get('sourcing_recommendation_distribution', {})}
- Country Distribution: {selection_data.get('country_distribution', {})}
- Business Segment Distribution: {selection_data.get('segment_distribution', {})}
"""

    return f"""
# IDENTITY AND GOAL
You are a senior partner at a top-tier management consulting firm, an expert in analyzing corporate DNA. Your task is to synthesize a company's static profile with its multi-year behavioral evolution to generate a deep, insightful summary of its supply chain decision-making philosophy.

# INPUT DATA 1: Corporate Profile (As of end of {historical_profiles[-1]['year']})
This data shows the company's current financial and business structure.
{self_profile_json}

# INPUT DATA 2: Multi-Year Supplier Selection Evolution
This data shows how the company's supplier selection behavior has evolved over time, revealing strategic shifts and consistent patterns.
{historical_summary_str}

# YOUR TASK
By analyzing multi-year selection trajectories, extract the company's **mature and stable supplier selection philosophy** and develop **long-term strategic guiding principles**:

- **Risk Tolerance Evolution**: Identify stable risk management principles and adaptive adjustment patterns from multi-year changes
- **Value Trade-off Maturation**: What value priorities have been time-tested and should serve as long-term guidance?
- **Supplier Relationship Deepening**: What relationship management maturity model does multi-year development reveal?
- **Geographic Layout Strategization**: What long-term strategic direction does the evolution of geographic distribution reveal?
- **Business Synergy Systematization**: What synergy strategy does the development path of business segment selection point toward?

**Please develop supplier selection guiding principles that maintain core consistency while possessing adaptability, based on these long-term validated strategic patterns.**

# OUTPUT SPECIFICATION
You MUST respond with a single, valid JSON object containing one key: "decision_context". The value should be the descriptive paragraph as a string.
"""


# ============================================================================
# 历史选择档案加载
# ============================================================================

def load_historical_selection_profiles(
    hub_id: str,
    current_year: int,
    start_year: int,
    result_dir: str
) -> List[Dict]:
    """
    加载指定Hub的历史选择档案（固定使用LLM方法）

    Args:
        hub_id: Hub ID
        current_year: 当前年份
        start_year: 初始年份
        result_dir: 结果目录

    Returns:
        历史选择档案列表，每个元素为：
        {
            'year': year,
            'selection_profile': {
                'total_selected': ...,
                'tier_distribution': {},
                ...
            }
        }
    """
    historical_profiles = []

    # 从start_year+1到current_year-1的所有年份
    for year in range(start_year + 1, current_year):
        # 固定使用LLM方法
        selection_file = os.path.join(result_dir, f"stage5a_llm_selection_{year}.pkl")

        if not os.path.exists(selection_file):
            # 静默跳过（不打印警告，因为可能是第一年）
            continue

        try:
            with open(selection_file, 'rb') as f:
                selection_data = pickle.load(f)

            # 提取该hub的选择结果
            hub_selections = selection_data['hub_selections']
            if hub_id not in hub_selections:
                continue

            hub_selection = hub_selections[hub_id]

            # 构建selection_profile
            selection_profile = _extract_selection_profile(hub_selection, year)

            historical_profiles.append({
                'year': year,
                'selection_profile': selection_profile
            })

        except Exception as e:
            # 静默跳过错误
            continue

    return historical_profiles


def _extract_selection_profile(hub_selection: Dict, year: int) -> Dict:
    """
    从hub_selection中提取selection_profile

    Args:
        hub_selection: Hub的选择结果
        year: 年份

    Returns:
        selection_profile字典
    """
    # 默认结构
    profile = {
        'total_selected': len(hub_selection.get('selected_suppliers', [])),
        'tier_distribution': {},
        'financial_risk_distribution': {},
        'single_source_distribution': {},
        'concentration_risk_distribution': {},
        'negotiation_leverage_distribution': {},
        'sourcing_recommendation_distribution': {},
        'country_distribution': {},
        'segment_distribution': {}
    }

    # 如果有selection_details或prediction_details，提取详细统计
    details = hub_selection.get('selection_details') or hub_selection.get('prediction_details')
    if details:
        # 对于llm方法，可能有assessment信息（需要从stage4加载）
        # 对于ml/random方法，可能没有详细的tier/risk等信息
        # 这里简化处理，只返回基本统计
        pass

    return profile


# ============================================================================
# 性格生成主函数
# ============================================================================

def generate_personality_for_hub(
    hub_id: str,
    hub_profile: Dict,
    llm_processor: LLMProcessor,
    year: int,
    personality_mode: str,
    historical_profiles: List[Dict]
) -> Dict:
    """
    为单个Hub生成采购决策性格

    Args:
        hub_id: Hub ID
        hub_profile: Hub的画像（来自Stage 2）
        llm_processor: LLM处理器
        year: 当前年份
        personality_mode: 实际使用的性格生成模式
        historical_profiles: 历史选择档案列表

    Returns:
        性格字典
    """
    # 根据personality_mode生成prompt
    if personality_mode == 'PROFILE_ONLY':
        prompt = create_personality_generation_prompt_base(hub_profile)
    elif personality_mode == 'PROFILE_WITH_PREV_YEAR':
        if historical_profiles:
            prompt = create_personality_generation_prompt_behavioral_prev_year(hub_profile, historical_profiles[-1:])
        else:
            print(f"  [WARNING] {hub_id}: 没有前一年行为数据，退回到PROFILE_ONLY")
            prompt = create_personality_generation_prompt_base(hub_profile)
            personality_mode = 'PROFILE_ONLY'  # 更新实际使用的模式
    elif personality_mode == 'PROFILE_WITH_ALL_PRIOR':
        if historical_profiles:
            prompt = create_personality_generation_prompt_behavioral_all_prior(hub_profile, historical_profiles)
        else:
            print(f"  [WARNING] {hub_id}: 没有历史行为数据，退回到PROFILE_ONLY")
            prompt = create_personality_generation_prompt_base(hub_profile)
            personality_mode = 'PROFILE_ONLY'
    else:
        raise ValueError(f"未知的personality_mode: {personality_mode}")

    # 调用LLM
    response = llm_processor.chat_with_retry(
        prompt=prompt,
        entity_id=hub_id,
        year=year,
        personality_mode=personality_mode
    )

    # 构建返回结果
    return {
        'decision_context': response.get('decision_context', ''),
        'decision_context_prompt': prompt,
        'company_name': hub_profile.get('entity_proper_name', hub_id),
        'personality_mode': personality_mode,  # 记录实际使用的模式
        'generated_at': datetime.now().isoformat(),
        'llm_response_full': response
    }


# ============================================================================
# 主处理流程
# ============================================================================

def generate_procurement_personality_batch(
    hub_profiles: Dict[str, Dict],
    year: int,
    start_year: int,
    personality_mode: str,
    result_dir: str,
    llm_processor: LLMProcessor,
    max_workers: int = 1
) -> Dict[str, Dict]:
    """
    批量生成采购性格（支持三种模式，固定使用LLM方法）

    Args:
        hub_profiles: {hub_id: profile_dict}
        year: 当前年份
        start_year: 初始年份
        personality_mode: 性格生成模式（会自动退化）
        result_dir: 结果目录
        llm_processor: LLM处理器
        max_workers: 并行工作线程数

    Returns:
        {hub_id: {'decision_context_prompt': prompt_str, 'decision_context': llm_response, ...}}
    """
    # 根据年份自动退化personality_mode
    actual_mode = personality_mode
    years_since_start = year - start_year

    if years_since_start == 1:
        # 第一年：强制PROFILE_ONLY
        actual_mode = 'PROFILE_ONLY'
        if personality_mode != 'PROFILE_ONLY':
            print(f"  [INFO] 第一年自动退化为PROFILE_ONLY模式")
    elif years_since_start == 2:
        # 第二年：最多PROFILE_WITH_PREV_YEAR
        if personality_mode == 'PROFILE_WITH_ALL_PRIOR':
            actual_mode = 'PROFILE_WITH_PREV_YEAR'
            print(f"  [INFO] 第二年自动退化为PROFILE_WITH_PREV_YEAR模式")

    print(f"\n  → 为 {len(hub_profiles)} 个Hub生成采购性格")
    print(f"  → 模式: {actual_mode}")
    print(f"  → 并行度: {max_workers} 个线程")

    personality_data = {}
    success_count = 0
    lock = threading.Lock()

    def process_single_hub(hub_id: str, hub_profile: Dict) -> tuple:
        """处理单个Hub（在线程中执行）"""
        try:
            # 加载历史selection profiles（如果需要）
            historical_profiles = []
            if actual_mode in ['PROFILE_WITH_PREV_YEAR', 'PROFILE_WITH_ALL_PRIOR']:
                historical_profiles = load_historical_selection_profiles(
                    hub_id=hub_id,
                    current_year=year,
                    start_year=start_year,
                    result_dir=result_dir
                )

            # 生成personality
            result = generate_personality_for_hub(
                hub_id=hub_id,
                hub_profile=hub_profile,
                llm_processor=llm_processor,
                year=year,
                personality_mode=actual_mode,
                historical_profiles=historical_profiles
            )

            return (hub_id, result, None)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            # 记录错误信息
            error_result = {
                'hub_id': hub_id,
                'error': str(e),
                'personality_mode': actual_mode
            }
            return (hub_id, error_result, str(e))

    # 使用ThreadPoolExecutor并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_hub = {
            executor.submit(process_single_hub, hub_id, hub_profile): hub_id
            for hub_id, hub_profile in hub_profiles.items()
        }

        # 使用tqdm显示进度
        try:
            with tqdm(total=len(hub_profiles), desc="  生成性格") as pbar:
                for future in as_completed(future_to_hub):
                    hub_id = future_to_hub[future]
                    try:
                        hub_id_result, result, error = future.result()

                        with lock:
                            personality_data[hub_id_result] = result
                            if error is None:
                                success_count += 1

                        pbar.update(1)

                    except KeyboardInterrupt:
                        print(f"\n\n⚠ 用户中断！已处理 {success_count}/{len(hub_profiles)} 个Hub")
                        print(f"  已生成的结果已保存到缓存，重新运行将继续处理")
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise
                    except Exception as e:
                        print(f"\n  [ERROR] {hub_id}: 处理失败 - {e}")
                        import traceback
                        traceback.print_exc()
                        pbar.update(1)

        except KeyboardInterrupt:
            raise

    print(f"  ✓ 成功生成 {success_count}/{len(hub_profiles)} 个性格")
    return personality_data


# ============================================================================
# 主处理流程
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Stage 3a: LLM采购性格生成（演化模式，支持三种模式）'
    )
    parser.add_argument(
        '--year',
        type=int,
        required=True,
        help='处理年份（必需）'
    )
    parser.add_argument(
        '--personality_mode',
        type=str,
        default='PROFILE_ONLY',
        choices=['PROFILE_ONLY', 'PROFILE_WITH_PREV_YEAR', 'PROFILE_WITH_ALL_PRIOR'],
        help='性格生成模式（默认：PROFILE_ONLY）'
    )
    parser.add_argument(
        '--result_dir',
        type=str,
        default=RESULT_DIR,
        help='结果保存目录'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=CACHE_DIR,
        help='LLM缓存目录'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='使用Mock LLM（不调用真实API）'
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=16,
        help='并行工作线程数（默认16）'
    )

    args = parser.parse_args()

    # 打印配置
    print("=" * 80)
    print("Stage 3a: LLM采购性格生成（演化模式）")
    print("=" * 80)
    print(f"处理年份: {args.year}")
    print(f"模式: {args.personality_mode}")
    print(f"结果目录: {args.result_dir}")
    if args.debug:
        print("【DEBUG模式 - 使用Mock LLM】")
    print(f"并行度: {args.max_workers} 个线程")
    print("=" * 80)

    # 步骤1: 检查Stage 1和Stage 2输出
    print(f"\n[1/5] 检查Stage 1和Stage 2输出")

    metadata_file = os.path.join(
        args.result_dir,
        f"stage1_metadata_{args.year}.json"
    )

    profiles_file = os.path.join(
        args.result_dir,
        f"stage2_profiles_{args.year}.pkl"
    )

    if not os.path.exists(metadata_file):
        raise FileNotFoundError(
            f"找不到Stage 1元数据: {metadata_file}\n"
            f"请先运行: python stage1_candidate_pool.py --year {args.year}"
        )

    if not os.path.exists(profiles_file):
        raise FileNotFoundError(
            f"找不到Stage 2画像: {profiles_file}\n"
            f"请先运行: python stage2_profile_builder.py --year {args.year}"
        )

    # 加载Stage 1元数据
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    hub_nodes = metadata['decision_makers']  # Hub节点即核心决策者
    print(f"  ✓ 检测到 {len(hub_nodes)} 个Hub节点")

    # 步骤2: 自动推测start_year
    print(f"\n[2/5] 自动推测初始年份")

    start_year = None

    # 方法1: 从metadata读取
    if 'start_year' in metadata:
        start_year = metadata['start_year']
        print(f"  ✓ 从metadata读取start_year: {start_year}")
    else:
        # 方法2: 扫描历史stage5a文件
        import glob
        selection_files = glob.glob(os.path.join(args.result_dir, "stage5a_llm_selection_*.pkl"))
        if selection_files:
            # 提取所有年份
            years = []
            for file_path in selection_files:
                filename = os.path.basename(file_path)
                match = re.search(r'stage5a_llm_selection_(\d{4})\.pkl', filename)
                if match:
                    years.append(int(match.group(1)))

            if years:
                start_year = min(years) - 1  # 最早的selection年份 - 1
                print(f"  ✓ 从历史文件推测start_year: {start_year}")

        # 方法3: 默认为当前年份-1
        if start_year is None:
            start_year = args.year - 1
            print(f"  ✓ 使用默认start_year: {start_year} (当前年份-1)")

    print(f"  → 最终确定start_year: {start_year}")

    # 加载Stage 2画像
    print(f"\n[3/5] 加载Stage 2画像")
    with open(profiles_file, 'rb') as f:
        stage2_data = pickle.load(f)

    hub_independent_profiles = stage2_data['hub_independent_profiles']
    print(f"  ✓ 检测到 {len(hub_independent_profiles)} 个Hub无关画像")

    # 步骤3: 提取Hub节点的画像
    print(f"\n[4/5] 提取Hub节点的画像")

    hub_profiles = {}
    missing_hubs = []

    for hub_id in hub_nodes:
        if hub_id in hub_independent_profiles:
            hub_profiles[hub_id] = hub_independent_profiles[hub_id]
        else:
            missing_hubs.append(hub_id)

    if missing_hubs:
        print(f"  ⚠ {len(missing_hubs)} 个Hub缺少画像: {missing_hubs[:5]}...")

    print(f"  ✓ 实际用于生成性格的Hub数: {len(hub_profiles)}")

    # 步骤4: 初始化LLM组件并生成性格
    print(f"\n[5/5] 初始化LLM组件并生成采购性格")

    # 创建缓存目录
    os.makedirs(args.cache_dir, exist_ok=True)
    cache_path = os.path.join(args.cache_dir, 'llm_personality_cache.json')

    # 初始化LLM组件
    print("  → 初始化LLM组件...")
    llm_agent = create_llm_agent(debug_mode=args.debug)
    llm_cache = LLMCache(cache_path)
    llm_processor = LLMProcessor(llm_agent, llm_cache)

    # 生成采购性格
    personality_data = generate_procurement_personality_batch(
        hub_profiles=hub_profiles,
        year=args.year,
        start_year=start_year,
        personality_mode=args.personality_mode,
        result_dir=args.result_dir,
        llm_processor=llm_processor,
        max_workers=args.max_workers
    )

    # 保存缓存
    llm_cache.save()

    # 步骤5: 保存结果
    print(f"\n保存结果")

    output_file = os.path.join(
        args.result_dir,
        f"stage3a_personalities_{args.year}.pkl"
    )

    os.makedirs(args.result_dir, exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(personality_data, f)

    print(f"  ✓ 结果已保存到: {output_file}")

    # 统计
    success_count = sum(1 for p in personality_data.values() if 'error' not in p)
    error_count = sum(1 for p in personality_data.values() if 'error' in p)

    # 打印完成信息
    print(f"\n{'='*80}")
    print(f"✓ Stage 3a 完成！")
    print(f"{'='*80}")
    print(f"处理年份: {args.year}")
    print(f"初始年份: {start_year}")
    print(f"Personality模式: {args.personality_mode}")
    print(f"Hub总数: {len(hub_nodes)}")
    print(f"有画像Hub数: {len(hub_profiles)}")
    print(f"成功生成性格: {success_count}")
    print(f"失败数: {error_count}")
    print(f"\n{'='*80}")
    print(f"下一步: 运行 stage4_llm_assessment.py --year {args.year}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
