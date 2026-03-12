#!/usr/bin/env python3
"""
Stage 4: LLM Supplier Assessment (Evolution Mode)
==================================================

功能：使用LLM对候选供应商进行评估（演化模式）

输入：
- stage1_candidate_pools_{year}/ (候选池，包含所有候选者)
- stage2_profiles_{year}.pkl (画像，包含hub_profiles)
- stage3a_personalities_{year}.pkl (性格，包含decision_context)

输出：
- stage4_llm_assessments_{year}.pkl
  格式：{
    hub_id: {
      'personality': {...},
      'assessments': {
        supplier_id: {
          'Supplier_Tier': 'Tier 1: Strategic',
          'Financial_Risk_Score': 'Low',
          'Single_Source_Flag': 'N',
          'Concentration_Risk_Flag': 'N',
          'Negotiation_Leverage_Score': 'Medium',
          'Sourcing_Recommendation': 'Deepen Partnership'
        }
      }
    }
  }

依赖关系：
- ⚠️ 需要Stage 1、2、3a的输出
- ⚠️ 需要LLM API（可选debug模式使用Mock）

参考：
- supplier_test.py: create_single_supplier_analysis_prompt()
- network/code/stage4_llm_assessment.py
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

    def _create_cache_key(self, prompt_type: str, hub_id: str, supplier_id: str, year: int) -> str:
        """创建缓存键"""
        return f"{prompt_type}_{hub_id}_{supplier_id}_{year}"

    def chat_with_retry(
        self,
        prompt: str,
        hub_id: str,
        supplier_id: str,
        year: int,
        retry_delay: int = 5
    ) -> Dict:
        """
        调用LLM API并进行重试（无限重试直到成功）

        Args:
            prompt: prompt文本
            hub_id: Hub ID
            supplier_id: 供应商ID
            year: 年份
            retry_delay: 重试延迟（秒）

        Returns:
            解析后的JSON字典
        """
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
                        print(f"  [WARNING] {hub_id}->{supplier_id}: JSON解析失败或缺少Sourcing_Recommendation字段 (尝试 #{attempt})")
                else:
                    print(f"  [WARNING] {hub_id}->{supplier_id}: LLM返回空响应 (尝试 #{attempt})")

            except Exception as e:
                print(f"  [ERROR] {hub_id}->{supplier_id}: LLM调用异常 (尝试 #{attempt}): {e}")

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
                    "supplier_name": "Mock Supplier Inc.",
                    "supplier_entity_id": "MOCK-EID-01",
                    "Supplier_Tier": "Tier 1: Strategic",
                    "Financial_Risk_Score": "Low",
                    "Single_Source_Flag": "N",
                    "Concentration_Risk_Flag": "N",
                    "Negotiation_Leverage_Score": "Medium",
                    "Sourcing_Recommendation": "Deepen Partnership"
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
# Prompt模板（完全对齐supplier_test.py）
# ============================================================================

def _format_self_identity_for_prompt(self_profile: dict, decision_context: str) -> str:
    """
    格式化buyer身份（对齐supplier_test.py的NAME_AND_CONTEXT模式）
    """
    name = self_profile.get("entity_proper_name", self_profile.get("company_name", "Our Company"))
    return f"You are a strategist for **{name}**. Your company's core decision-making philosophy is as follows:\n**{decision_context}**"


def create_single_supplier_analysis_prompt(
    self_profile: dict,
    supplier_profile: dict,
    cutoff_date: pd.Timestamp,
    decision_context: str
) -> str:
    """
    创建单个供应商分析prompt（完全对齐supplier_test.py）
    """
    self_identity_str = _format_self_identity_for_prompt(self_profile, decision_context)
    supplier_profile_json = json.dumps(supplier_profile, indent=2, default=_convert_for_json)

    # 计算关系持续时间
    duration_years = None
    if 'start_' in supplier_profile and pd.notna(supplier_profile.get('start_')):
        start_date = pd.to_datetime(supplier_profile['start_'])
        if pd.notna(start_date):
            duration_years = (cutoff_date - start_date).days / 365.25

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
2.  **Financial Risk Assessment**: (Use the MOST RECENT record in `financial_history`)
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
  "supplier_name": "{supplier_profile.get('entity_proper_name', 'N/A')}",
  "supplier_entity_id": "{supplier_profile['factset_entity_id']}",
  "Supplier_Tier": "Tier 1: Strategic" | "Tier 2: Core" | "Tier 3: Tactical",
  "Financial_Risk_Score": "High" | "Medium" | "Low",
  "Single_Source_Flag": "Y" | "N",
  "Concentration_Risk_Flag": "Y" | "N",
  "Negotiation_Leverage_Score": "High" | "Medium" | "Low",
  "Sourcing_Recommendation": "Diversify Now" | "Deepen Partnership" | "Maintain & Monitor" | "Seek Alternatives"
}}
"""


# ============================================================================
# 评估主流程
# ============================================================================

def assess_candidates_for_hub(
    hub_id: str,
    hub_supplier_profiles: Dict[str, Dict],
    personality: Dict,
    year: int,
    llm_processor: LLMProcessor,
    hub_independent_profiles: Dict[str, Dict],
    max_workers: int = 1
) -> Dict[str, Dict]:
    """
    为单个Hub评估所有候选供应商

    Args:
        hub_id: Hub节点ID
        hub_supplier_profiles: Hub特定的供应商profiles字典 {supplier_id: profile}
        personality: Hub的性格（decision_context）
        year: 年份
        llm_processor: LLM处理器
        hub_independent_profiles: Hub无关的profiles字典（用于获取hub自己的profile）
        max_workers: 并行工作线程数

    Returns:
        评估结果字典 {supplier_id: assessment}
    """
    decision_context = personality.get('decision_context', 'No decision context available.')
    cutoff_date = pd.Timestamp(f"{year - 1}-12-31")

    # 从hub_independent_profiles中获取hub自己的profile（作为buyer）
    hub_profile = hub_independent_profiles.get(hub_id)
    if not hub_profile:
        print(f"  [WARNING] {hub_id}: Hub自己的profile不存在，无法进行评估")
        return {}

    assessments = {}
    lock = threading.Lock()

    def assess_single_supplier(supplier_id: str) -> tuple:
        """评估单个供应商"""
        try:
            # 跳过hub自己
            if supplier_id == hub_id:
                return (supplier_id, None, "Skipped self-assessment")

            supplier_profile = hub_supplier_profiles.get(supplier_id)
            if not supplier_profile:
                return (supplier_id, None, f"Profile not found")

            # 生成prompt
            prompt = create_single_supplier_analysis_prompt(
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
    supplier_ids = [sid for sid in hub_supplier_profiles.keys() if sid != hub_id]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_supplier = {
            executor.submit(assess_single_supplier, supplier_id): supplier_id
            for supplier_id in supplier_ids
        }

        try:
            with tqdm(total=len(supplier_ids), desc=f"  评估 {hub_id}") as pbar:
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


# ============================================================================
# 主处理流程
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Stage 4: LLM供应商评估（演化模式）'
    )
    parser.add_argument(
        '--year',
        type=int,
        required=True,
        help='处理年份（必需）'
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
    print("Stage 4: LLM供应商评估（演化模式）")
    print("=" * 80)
    print(f"处理年份: {args.year}")
    print(f"结果目录: {args.result_dir}")
    if args.debug:
        print("【DEBUG模式 - 使用Mock LLM】")
    print(f"并行度: {args.max_workers} 个线程")
    print("=" * 80)

    # 步骤1: 检查Stage 1、2、3a输出
    print(f"\n[1/4] 检查Stage 1、2、3a输出")

    candidate_pools_dir = os.path.join(
        args.result_dir,
        f"stage1_candidate_pools_{args.year}"
    )

    profiles_file = os.path.join(
        args.result_dir,
        f"stage2_profiles_{args.year}.pkl"
    )

    personalities_file = os.path.join(
        args.result_dir,
        f"stage3a_personalities_{args.year}.pkl"
    )

    if not os.path.exists(candidate_pools_dir):
        raise FileNotFoundError(
            f"找不到Stage 1候选池目录: {candidate_pools_dir}\n"
            f"请先运行: python stage1_candidate_pool.py --year {args.year}"
        )

    if not os.path.exists(profiles_file):
        raise FileNotFoundError(
            f"找不到Stage 2画像: {profiles_file}\n"
            f"请先运行: python stage2_profile_builder.py --year {args.year}"
        )

    if not os.path.exists(personalities_file):
        raise FileNotFoundError(
            f"找不到Stage 3a性格: {personalities_file}\n"
            f"请先运行: python stage3a_llm_personality.py --year {args.year}"
        )

    # 加载Stage 2画像
    print(f"  → 加载Stage 2画像...")
    with open(profiles_file, 'rb') as f:
        stage2_data = pickle.load(f)

    hub_profiles = stage2_data['hub_profiles']
    hub_independent_profiles = stage2_data['hub_independent_profiles']
    print(f"  ✓ 加载了 {len(hub_profiles)} 个Hub的画像")
    print(f"  ✓ 加载了 {len(hub_independent_profiles)} 个Hub无关画像")

    # 加载Stage 3a性格
    print(f"  → 加载Stage 3a性格...")
    with open(personalities_file, 'rb') as f:
        personalities = pickle.load(f)

    print(f"  ✓ 加载了 {len(personalities)} 个Hub的性格")

    # 步骤2: 初始化LLM组件
    print(f"\n[2/4] 初始化LLM组件")

    # 创建缓存目录
    os.makedirs(args.cache_dir, exist_ok=True)
    cache_path = os.path.join(args.cache_dir, 'llm_assessment_cache.json')

    # 初始化LLM组件
    print("  → 初始化LLM组件...")
    llm_agent = create_llm_agent(debug_mode=args.debug)
    llm_cache = LLMCache(cache_path)
    llm_processor = LLMProcessor(llm_agent, llm_cache)

    # 步骤3: 为每个Hub评估候选供应商
    print(f"\n[3/4] 为每个Hub评估候选供应商")

    all_hub_assessments = {}
    success_count = 0

    # 获取所有Hub节点
    hub_nodes = list(hub_profiles.keys())
    print(f"  → 共有 {len(hub_nodes)} 个Hub需要处理")

    for hub_id in tqdm(hub_nodes, desc="  处理Hub"):
        try:
            # 检查必需数据
            if hub_id not in hub_profiles:
                print(f"  [WARNING] {hub_id}: 无画像，跳过")
                continue

            if hub_id not in personalities:
                print(f"  [WARNING] {hub_id}: 无性格，跳过")
                continue

            if 'error' in personalities[hub_id]:
                print(f"  [WARNING] {hub_id}: 性格生成失败，跳过")
                continue

            hub_supplier_profiles = hub_profiles[hub_id]
            personality = personalities[hub_id]

            # 评估候选供应商
            assessments = assess_candidates_for_hub(
                hub_id,
                hub_supplier_profiles,
                personality,
                args.year,
                llm_processor,
                hub_independent_profiles,
                max_workers=args.max_workers
            )

            all_hub_assessments[hub_id] = {
                'personality': personality,
                'assessments': assessments,
                'statistics': {
                    'total_candidates': len(hub_supplier_profiles) - 1,  # 减去hub自己
                    'assessed_count': len(assessments)
                }
            }

            success_count += 1

            # 每处理10个Hub就保存一次缓存
            if success_count % 10 == 0:
                llm_cache.save()

        except Exception as e:
            print(f"  [ERROR] {hub_id}: 处理失败 - {e}")
            import traceback
            traceback.print_exc()
            continue

    # 保存缓存
    llm_cache.save()

    # 统计
    total_assessments = sum(len(h['assessments']) for h in all_hub_assessments.values())

    print(f"\n{args.year} 年统计:")
    print(f"  成功处理Hub数: {success_count}/{len(hub_nodes)}")
    print(f"  总评估数: {total_assessments}")

    # 步骤4: 保存结果
    print(f"\n[4/4] 保存结果")

    output_file = os.path.join(
        args.result_dir,
        f"stage4_llm_assessments_{args.year}.pkl"
    )

    os.makedirs(args.result_dir, exist_ok=True)

    result = {
        'year': args.year,
        'hub_assessments': all_hub_assessments,
        'statistics': {
            'total_hubs': len(hub_nodes),
            'assessed_hubs': success_count,
            'total_assessments': total_assessments
        }
    }

    with open(output_file, 'wb') as f:
        pickle.dump(result, f)

    print(f"  ✓ 结果已保存到: {output_file}")

    # 打印完成信息
    print(f"\n{'='*80}")
    print(f"✓ Stage 4 完成！")
    print(f"{'='*80}")
    print(f"处理年份: {args.year}")
    print(f"Hub总数: {len(hub_nodes)}")
    print(f"成功处理Hub数: {success_count}")
    print(f"总评估数: {total_assessments}")
    print(f"\n{'='*80}")
    print(f"下一步: 运行 stage5a_llm_selection.py --year {args.year}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
