#!/usr/bin/env python3
"""
Stage 4: LLM Supplier Assessment
==================================

功能：使用LLM对候选供应商进行画像评估

输入：
- Stage1: hub_candidate_pools (候选池)
- Stage2: hub_independent_profiles (供应商profiles)
- Stage3a: llm_personalities (Hub性格)

输出：
- stage4_llm_assessments_*.pkl
  {
    year: {
      'hub_id': {
        'personality': {...},
        'assessments': {
          'supplier_id': {
            'Supplier_Tier': ...,
            'Financial_Risk_Score': ...,
            'Sourcing_Recommendation': ...,
            ...
          }
        }
      }
    }
  }
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

# 添加父目录到sys.path以导入llm模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# ============================================================================
# 全局配置
# ============================================================================

STAGE1_RESULT_DIR = "../result"
STAGE2_RESULT_DIR = "../result"
STAGE3A_RESULT_DIR = "../result"
RESULT_DIR = "../result"

# ============================================================================
# LLM相关组件（从Stage3a复制）
# ============================================================================

def _convert_for_json(value):
    """JSON转换辅助函数"""
    if pd.isna(value): return None
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, pd.Timestamp): return value.strftime('%Y-%m-%d')
    if isinstance(value, (pd.Timestamp, datetime)): return value.isoformat()
    if isinstance(value, (np.integer, int)): return int(value)
    if isinstance(value, (np.floating, float)): return float(value)
    return value


def _parse_json_from_llm_response(response_str: str) -> Optional[Dict]:
    """从LLM响应中解析JSON"""
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


class LLMCache:
    """LLM响应缓存（线程安全）"""
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.cache = self._load()
        self.dirty = False
        self.lock = threading.Lock()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"  [WARNING] 加载缓存失败: {e}")
                return {}
        return {}

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            return self.cache.get(key)

    def set(self, key: str, value: Any):
        with self.lock:
            self.cache[key] = value
            self.dirty = True

    def save(self):
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

    def _create_cache_key(self, prompt_type: str, hub_id: str, supplier_id: str, year: int) -> str:
        return f"{prompt_type}_{hub_id}_{supplier_id}_{year}"

    def chat_with_retry(
        self,
        prompt: str,
        hub_id: str,
        supplier_id: str,
        year: int,
        retry_delay: int = 5
    ) -> Dict:
        """调用LLM API并进行重试（无限重试直到成功）"""
        cache_key = self._create_cache_key("assessment", hub_id, supplier_id, year)

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
                    if parsed_json and 'Sourcing_Recommendation' in parsed_json:
                        # 缓存成功的响应
                        self.cache.set(cache_key, parsed_json)
                        return parsed_json
                    else:
                        print(f"  [WARNING] {hub_id}->{supplier_id}: JSON解析失败或缺少字段 (尝试 #{attempt})")
                else:
                    print(f"  [WARNING] {hub_id}->{supplier_id}: LLM返回空响应 (尝试 #{attempt})")

            except Exception as e:
                print(f"  [ERROR] {hub_id}->{supplier_id}: LLM调用异常 (尝试 #{attempt}): {e}")

            # 等待后重试
            if attempt % 5 == 0:
                print(f"  → 已重试{attempt}次，将在 {retry_delay} 秒后继续...")
            time.sleep(retry_delay)


def create_llm_agent(debug_mode: bool = False):
    """创建LLM Agent"""
    if debug_mode:
        print("  [DEBUG] 使用Mock LLM")

        class MockLLM:
            def chat(self, prompt: str) -> str:
                return json.dumps({
                    "supplier_name": "Mock Supplier",
                    "supplier_entity_id": "MOCK-E",
                    "Supplier_Tier": "Tier 2: Core",
                    "Financial_Risk_Score": "Medium",
                    "Single_Source_Flag": "N",
                    "Concentration_Risk_Flag": "N",
                    "Negotiation_Leverage_Score": "Medium",
                    "Sourcing_Recommendation": "Maintain & Monitor"
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
# Prompt生成（从network_test.py复制并调整）
# ============================================================================

def _format_self_identity_for_prompt(self_profile: dict, decision_context: str) -> str:
    """格式化buyer身份（对齐network_test.py的NAME_AND_CONTEXT模式）"""
    name = self_profile.get("entity_proper_name", "Our Company")
    return f"You are a strategist for **{name}**. Your company's core decision-making philosophy is as follows:\n**{decision_context}**"


def create_supplier_assessment_prompt(
    buyer_profile: Dict,
    supplier_profile: Dict,
    cutoff_date: pd.Timestamp,
    decision_context: str
) -> str:
    """创建供应商评估prompt（完全对齐network_test.py）"""

    # 格式化buyer身份（使用network_test.py的格式）
    self_identity_str = _format_self_identity_for_prompt(buyer_profile, decision_context)

    supplier_profile_json = json.dumps(supplier_profile, indent=2, default=_convert_for_json)

    # 计算关系持续时间
    duration_years = None
    if 'start_' in supplier_profile and pd.notna(supplier_profile.get('start_')):
        start_date = pd.to_datetime(supplier_profile['start_'])
        if pd.notna(start_date):
            duration_years = (cutoff_date - start_date).days / 365.25

    supplier_entity_id = supplier_profile.get('factset_entity_id', 'N/A')
    supplier_name = supplier_profile.get('entity_proper_name', 'N/A')

    return f"""
# IDENTITY, CONTEXT, AND GOAL
{self_identity_str}
Your task is to analyze your relationship with a SINGLE supplier based on the comprehensive FactSet data provided, and generate strategic assessments in a specific JSON format. Your final judgment MUST be tempered by the provided decision-making philosophy.

# INPUT DATA
## Supplier Profile (The Company You Are Analyzing):
This profile includes basic info (including "iso_country"), a history of latest financials, relationship data, and top business segments.
{supplier_profile_json}

# DECISION LOGIC GUIDELINES
You must follow these rules as a baseline, but your final recommendation should reflect your core philosophy. For instance, a "Power-Driven" company might still choose "Seek Alternatives" for a "Tier 1" supplier to maintain competitive tension, even if the rules suggest "Deepen Partnership".

1.  **Supplier Tiering**:
    *   Relationship duration is approx. **{f"{duration_years:.1f}" if duration_years is not None else "N/A"} years**.
    *   **Tier 1: Strategic**: `revenue_percent` is not null AND `revenue_percent` >= 5 AND duration is not null AND duration >= 3.
    *   **Tier 2: Core**: (`revenue_percent` is not null AND `revenue_percent` >= 1) OR (duration is not null AND duration >= 1).
    *   **Tier 3: Tactical**: Default.
2.  **Financial Risk Assessment**: (Use the MOST RECENT record in `financial_data`)
    *   **High**: `FF_DEBT_EQ` > 2.5 OR `FF_NET_MGN` < 0 OR `FF_CURR_RATIO` < 1.
    *   **Medium**: (`FF_DEBT_EQ` > 1.5) OR (`FF_NET_MGN` < 0.03). Default if data is missing.
    *   **Low**: Otherwise.
3.  **Single Source Flag**: **Y** if `is_single_source` in the profile is true, else **N**.
4.  **Concentration Risk Flag**: **Y** if `revenue_percent` is not null AND `revenue_percent` >= 30, else **N**.
5.  **Negotiation Leverage Score (for Self)**:
    *   **High**: (`revenue_percent` is not null AND `revenue_percent` >= 10) OR (`is_single_source` is false AND Self's latest `FF_SALES` is not null AND Supplier's latest `FF_SALES` is not null AND Self's latest `FF_SALES` > 10 * Supplier's latest `FF_SALES`).
    *   **Low**: `is_single_source` is true OR (`revenue_percent` is not null AND `revenue_percent` < 1 AND Self's latest `FF_SALES` is not null AND Supplier's latest `FF_SALES` is not null AND Self's latest `FF_SALES` < Supplier's latest `FF_SALES`).
    *   **Medium**: Default.
6.  **Sourcing Recommendation (Enhanced Logic)**:
    *   **"Diversify Now"**: `is_single_source` is true AND (Financial Risk is 'High').
    *   **"Seek Alternatives"**: Financial Risk is 'High'.
    *   **"Deepen Partnership"**: Tier is 'Tier 1: Strategic' AND Financial Risk is 'Low'.
    *   **"Maintain & Monitor"**: Tier is 'Tier 2: Core' OR (Tier is 'Tier 3: Tactical' AND Financial Risk is 'Low').
    *   **"Seek Alternatives"**: Tier is 'Tier 3: Tactical' AND Financial Risk is 'Medium'.

# BUSINESS SUPPORT EVALUATION
**CRITICAL**: Before making any sourcing recommendation, evaluate business support compatibility:
- **Direct Business Support**: Does this supplier's business segments align with your company's core operations?
- **Value Chain Integration**: Does this supplier play a necessary role in your value creation process?
- **Operational Necessity**: Are this supplier's services/products essential for your business continuity?

**IMPORTANT**: Even suppliers with moderate financial risk or lower tiers should be considered for "Maintain & Monitor" if they provide essential business support. Avoid being overly conservative - focus on business necessity over perfect financial metrics.

# OUTPUT SPECIFICATION
You MUST respond with a single, valid JSON object with ALL the following keys.
{{
  "supplier_name": "{supplier_name}",
  "supplier_entity_id": "{supplier_entity_id}",
  "Supplier_Tier": "Tier 1: Strategic" | "Tier 2: Core" | "Tier 3: Tactical",
  "Financial_Risk_Score": "High" | "Medium" | "Low",
  "Single_Source_Flag": "Y" | "N",
  "Concentration_Risk_Flag": "Y" | "N",
  "Negotiation_Leverage_Score": "High" | "Medium" | "Low",
  "Sourcing_Recommendation": "Diversify Now" | "Deepen Partnership" | "Maintain & Monitor" | "Seek Alternatives"
}}
"""


# ============================================================================
# 主流程
# ============================================================================

def assess_candidates_for_hub(
    hub_id: str,
    hub_profile: Dict,
    personality: Dict,
    candidate_pool: pd.DataFrame,
    supplier_profiles: Dict[str, Dict],
    year: int,
    llm_processor: LLMProcessor,
    max_workers: int = 1
) -> Dict[str, Dict]:
    """
    为单个Hub评估所有候选供应商

    Args:
        hub_id: Hub节点ID
        hub_profile: Hub的profile
        personality: Hub的性格（decision_context）
        candidate_pool: 候选池DataFrame
        supplier_profiles: 供应商profiles字典
        year: 年份
        llm_processor: LLM处理器
        max_workers: 并行工作线程数

    Returns:
        评估结果字典 {supplier_id: assessment}
    """
    decision_context = personality.get('decision_context', 'No decision context available.')
    cutoff_date = pd.Timestamp(f"{year - 1}-12-31")

    assessments = {}
    lock = threading.Lock()

    def assess_single_supplier(supplier_id: str) -> tuple:
        """评估单个供应商"""
        try:
            supplier_profile = supplier_profiles.get(supplier_id)
            if not supplier_profile:
                return (supplier_id, None, f"Profile not found")

            # 生成prompt
            prompt = create_supplier_assessment_prompt(
                hub_profile,
                supplier_profile,
                cutoff_date,
                decision_context
            )

            # 调用LLM API
            assessment = llm_processor.chat_with_retry(
                prompt=prompt,
                hub_id=hub_id,
                supplier_id=supplier_id,
                year=year
            )

            return (supplier_id, assessment, None)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            return (supplier_id, None, str(e))

    # 并行评估
    supplier_ids = candidate_pool['factset_entity_id'].unique()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_supplier = {
            executor.submit(assess_single_supplier, supplier_id): supplier_id
            for supplier_id in supplier_ids
        }

        try:
            with tqdm(total=len(supplier_ids), desc=f"  Assessing {hub_id}") as pbar:
                for future in as_completed(future_to_supplier):
                    supplier_id = future_to_supplier[future]
                    try:
                        supplier_id_result, assessment, error = future.result()

                        with lock:
                            if error is None and assessment:
                                assessments[supplier_id_result] = assessment

                        pbar.update(1)

                    except KeyboardInterrupt:
                        print(f"\n\n⚠ 用户中断！")
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise
                    except Exception as e:
                        print(f"\n  [ERROR] {supplier_id}: 评估失败 - {e}")
                        pbar.update(1)

        except KeyboardInterrupt:
            raise

    return assessments


def process_single_year(
    year: int,
    stage1_data: Dict,
    stage2_data: Dict,
    stage3a_data: Dict,
    llm_processor: LLMProcessor,
    max_workers: int = 1
) -> Dict:
    """处理单个年份"""
    print(f"\n{'='*80}")
    print(f"处理 {year} 年")
    print(f"{'='*80}")

    # 获取数据
    hub_nodes = stage1_data['hub_nodes']
    hub_candidate_pools = stage1_data['hub_candidate_pools']
    hub_independent_profiles = stage2_data['hub_independent_profiles']
    personalities = stage3a_data['personalities']

    print(f"  Hub节点数: {len(hub_nodes)}")
    print(f"  Hub画像数: {len(hub_independent_profiles)}")
    print(f"  Hub性格数: {len(personalities)}")

    # 为每个Hub评估候选供应商
    all_hub_assessments = {}

    for hub_id in tqdm(hub_nodes, desc="Processing hubs"):
        try:
            # 检查必需数据
            if hub_id not in hub_candidate_pools:
                print(f"  [WARNING] {hub_id}: 无候选池，跳过")
                continue

            if hub_id not in hub_independent_profiles:
                print(f"  [WARNING] {hub_id}: 无profile，跳过")
                continue

            if hub_id not in personalities:
                print(f"  [WARNING] {hub_id}: 无性格，跳过")
                continue

            if 'error' in personalities[hub_id]:
                print(f"  [WARNING] {hub_id}: 性格生成失败，跳过")
                continue

            hub_profile = hub_independent_profiles[hub_id]
            personality = personalities[hub_id]
            candidate_pool = hub_candidate_pools[hub_id]

            # 评估候选供应商
            assessments = assess_candidates_for_hub(
                hub_id,
                hub_profile,
                personality,
                candidate_pool,
                hub_independent_profiles,
                year,
                llm_processor,
                max_workers=max_workers
            )

            all_hub_assessments[hub_id] = {
                'personality': personality,
                'assessments': assessments,
                'statistics': {
                    'total_candidates': len(candidate_pool),
                    'assessed_count': len(assessments)
                }
            }

        except Exception as e:
            print(f"  [ERROR] {hub_id}: 处理失败 - {e}")
            import traceback
            traceback.print_exc()
            continue

    # 统计
    total_hubs = len(hub_nodes)
    assessed_hubs = len(all_hub_assessments)
    total_assessments = sum(len(h['assessments']) for h in all_hub_assessments.values())

    print(f"\n{year} 年统计:")
    print(f"  成功处理Hub数: {assessed_hubs}/{total_hubs}")
    print(f"  总评估数: {total_assessments}")

    return {
        'year': year,
        'hub_assessments': all_hub_assessments,
        'statistics': {
            'total_hubs': total_hubs,
            'assessed_hubs': assessed_hubs,
            'total_assessments': total_assessments
        }
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Stage 4: LLM供应商评估')
    parser.add_argument('--stage1_file', type=str, default=None)
    parser.add_argument('--stage2_file', type=str, default=None)
    parser.add_argument('--stage3a_file', type=str, default=None)
    parser.add_argument('--result_dir', type=str, default=RESULT_DIR)
    parser.add_argument('--cache_dir', type=str, default='../cache')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--max_workers', type=int, default=16)

    args = parser.parse_args()

    print("=" * 80)
    print("Stage 4: LLM供应商评估")
    if args.debug:
        print("【DEBUG模式 - 使用Mock LLM】")
    print(f"并行度: {args.max_workers} 个线程")
    print("=" * 80)

    # 创建缓存目录
    os.makedirs(args.cache_dir, exist_ok=True)
    cache_path = os.path.join(args.cache_dir, 'llm_assessment_cache.json')

    # 初始化LLM组件
    print("\n初始化LLM组件...")
    llm_agent = create_llm_agent(debug_mode=args.debug)
    llm_cache = LLMCache(cache_path)
    llm_processor = LLMProcessor(llm_agent, llm_cache)

    # 查找输入文件
    if args.stage1_file:
        stage1_file = args.stage1_file
    else:
        stage1_files = list(Path(STAGE1_RESULT_DIR).glob("stage1_candidate_pools_*.pkl"))
        if not stage1_files:
            print("❌ 未找到Stage1输出文件")
            return
        stage1_file = str(sorted(stage1_files)[-1])

    if args.stage2_file:
        stage2_file = args.stage2_file
    else:
        stage2_files = list(Path(STAGE2_RESULT_DIR).glob("stage2_company_profiles_*.pkl"))
        if not stage2_files:
            print("❌ 未找到Stage2输出文件")
            return
        stage2_file = str(sorted(stage2_files)[-1])

    if args.stage3a_file:
        stage3a_file = args.stage3a_file
    else:
        stage3a_files = list(Path(STAGE3A_RESULT_DIR).glob("stage3a_llm_personalities_*.pkl"))
        if not stage3a_files:
            print("❌ 未找到Stage3a输出文件")
            return
        stage3a_file = str(sorted(stage3a_files)[-1])

    print(f"Stage1文件: {stage1_file}")
    print(f"Stage2文件: {stage2_file}")
    print(f"Stage3a文件: {stage3a_file}")

    # 加载数据
    with open(stage1_file, 'rb') as f:
        stage1_results = pickle.load(f)

    with open(stage2_file, 'rb') as f:
        stage2_results = pickle.load(f)

    with open(stage3a_file, 'rb') as f:
        stage3a_results = pickle.load(f)

    # 确保年份一致
    years = sorted(
        set(stage1_results.keys()) &
        set(stage2_results.keys()) &
        set(stage3a_results.keys())
    )
    print(f"✓ 共同年份: {len(years)} 个 - {years}")

    # 处理所有年份
    all_year_results = {}
    for year in years:
        try:
            year_result = process_single_year(
                year,
                stage1_results[year],
                stage2_results[year],
                stage3a_results[year],
                llm_processor,
                max_workers=args.max_workers
            )
            all_year_results[year] = year_result

            # 每处理完一个年份就保存缓存
            llm_cache.save()

        except Exception as e:
            print(f"\n❌ {year} 年处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 最终保存缓存
    llm_cache.save()

    # 保存结果
    stage1_basename = os.path.basename(stage1_file)
    hub_pct_match = stage1_basename.split('_hub')[-1].split('pct')[0] if '_hub' in stage1_basename else '20'

    output_file = os.path.join(
        args.result_dir,
        f'stage4_llm_assessments_hub{hub_pct_match}pct.pkl'
    )

    print(f"\n{'='*80}")
    print(f"保存结果到: {output_file}")

    with open(output_file, 'wb') as f:
        pickle.dump(all_year_results, f)

    print(f"✓ Stage 4 完成！")

    # 统计
    print(f"\n{'='*80}")
    print("汇总统计")
    print(f"{'='*80}")
    print(f"成功处理年份: {len(all_year_results)}/{len(years)}\n")

    print("年份 | Hub总数 | 成功评估Hub | 总评估数")
    print("-" * 80)
    for year, result in sorted(all_year_results.items()):
        stats = result['statistics']
        print(f"{year} | {stats['total_hubs']:>7} | {stats['assessed_hubs']:>11} | "
              f"{stats['total_assessments']:>8}")


if __name__ == "__main__":
    main()
