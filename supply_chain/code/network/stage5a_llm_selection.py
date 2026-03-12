#!/usr/bin/env python3
"""
Stage 5a: LLM-Based Supplier Selection
========================================

功能：使用LLM进行供应商选择决策（PER_CANDIDATE模式）

输入：
- Stage1: hub_candidate_pools
- Stage2: hub_independent_profiles
- Stage3a: llm_personalities
- Stage4: llm_assessments

输出：
- stage5a_llm_selections_*.pkl
  {
    year: {
      'hub_selections': {
        'hub_id': {
          'selected_suppliers': [...],
          'selection_details': {...},
          'statistics': {...}
        }
      }
    }
  }

选择模式：PER_CANDIDATE（逐个评估，selection_probability >= 0.5 即选择）
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
from typing import Dict, List, Optional, Any, Set
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
STAGE4_RESULT_DIR = "../result"
RESULT_DIR = "../result"

SELECTION_PROBABILITY_THRESHOLD = 0.5  # 固定阈值

# ============================================================================
# LLM相关组件（从Stage3a/Stage4复制）
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
        cache_key = self._create_cache_key("selection", hub_id, supplier_id, year)

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
                    if parsed_json and 'decision' in parsed_json:
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
                # 随机决策
                decision = "SELECT" if np.random.random() > 0.5 else "NOT SELECT"
                prob = np.random.uniform(0.3, 0.9)
                return json.dumps({
                    "decision": decision,
                    "selection_probability": prob,
                    "reasoning": "Mock selection decision for testing"
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
# Prompt生成（从network_test.py复制）
# ============================================================================

def _format_self_identity_for_prompt(self_profile: dict, decision_context: str) -> str:
    """格式化buyer身份"""
    name = self_profile.get("entity_proper_name", "Our Company")
    return f"You are a strategist for **{name}**. Your company's core decision-making philosophy is as follows:\n**{decision_context}**"


def create_single_supplier_selection_prompt(self_profile: dict, dossier: Dict, decision_context: str) -> str:
    """创建供应商选择prompt（PER_CANDIDATE模式，对齐network_test.py）"""
    self_identity_str = _format_self_identity_for_prompt(self_profile, decision_context)

    # 获取目标公司的segment信息
    target_segments = self_profile.get('top_segments', [])
    target_segments_str = ", ".join(target_segments) if target_segments else "Not available"

    entity_id = dossier['factset_entity_id']
    assessment = dossier['llm_assessment']
    supplier_name = assessment.get('supplier_name', 'Unknown')

    # 获取供应商的segment信息
    supplier_segments = dossier.get('data_profile', {}).get('top_segments', [])
    supplier_segments_str = ", ".join(supplier_segments) if supplier_segments else "Not available"

    return f"""
# IDENTITY AND CONTEXT
{self_identity_str}

# TARGET COMPANY BUSINESS SEGMENTS
Your company operates in the following business segments: {target_segments_str}

# CORE SELECTION PHILOSOPHY: STRATEGIC INSIGHTS FROM ORIGINAL DATA ARE PARAMOUNT
As a strategist, your primary focus must be on drawing strategic conclusions from **original, factual data** (like business segments, geography) rather than relying solely on pre-calculated **assessed labels** (like Risk Score, Supplier Tier). Your decision-making process MUST follow this philosophy and the strict, two-stage logic below:

**Stage 1: Business Relevance Analysis (Gatekeeper based on Your Judgment of Original Data)**
First, you MUST analyze the supplier's **business segments** (original data) in relation to our company's segments. It is your task to judge how relevant and critical their business is to our core operations.
- **Apply the Gate:**
    - If you conclude there is **NO strong and direct business relevance**, you MUST decide to **NOT SELECT**. Its other positive attributes (including good assessed labels like 'Low Risk') are irrelevant.
    - If you conclude there **IS a strong and direct business relevance**, the supplier passes this gate. Proceed to Stage 2.

**Stage 2: Strategic Evaluation (For Relevant Suppliers Only)**
For suppliers you have identified as relevant, you must now perform a strategic evaluation.
- **Prioritize Original Data:** Your decision should be driven by strategic insights you derive from other original data points. For example, does the supplier's **geography (`iso_country`)** align with our strategic goal of diversifying our supply chain?
- **Use Assessed Labels as Supporting Evidence:** The pre-calculated labels (`Financial_Risk_Score`, `Supplier_Tier`) are useful secondary indicators, but they should NOT be the primary driver of your decision.
- **Crucially, strategic needs derived from original data can OVERRIDE assessed labels.** For example, even if a supplier has a 'High' `Financial_Risk_Score`, if they are a perfect business match AND are located in a strategically vital region for us, you should still strongly consider selecting them. Your reasoning must explain this trade-off.

# TASK: SINGLE SUPPLIER SELECTION DECISION
Apply the philosophy above to the following candidate. Based on its profile, decide whether to SELECT or NOT SELECT this supplier for next year's portfolio.

# SUPPLIER PROFILE
- **Supplier**: {supplier_name} (ID: {entity_id})
- **Original Data Point (Geography)**: {dossier.get('data_profile', {}).get('iso_country', 'N/A')}
- **Original Data Point (Segments)**: {supplier_segments_str}
- **Assessed Labels (for reference)**:
  - Supplier Tier: {assessment.get('Supplier_Tier', 'N/A')}
  - Financial Risk: {assessment.get('Financial_Risk_Score', 'N/A')}
  - Sourcing Recommendation: {assessment.get('Sourcing_Recommendation', 'N/A')}

# OUTPUT FORMAT
Return a JSON object with your decision:
{{
  "decision": "SELECT" | "NOT SELECT",
  "selection_probability": 0.0-1.0,
  "reasoning": "Brief explanation of your decision, explicitly referencing the Core Selection Philosophy."
}}
"""


# ============================================================================
# 主流程
# ============================================================================

def select_suppliers_for_hub(
    hub_id: str,
    hub_profile: Dict,
    personality: Dict,
    candidate_pool: pd.DataFrame,
    supplier_profiles: Dict[str, Dict],
    assessments: Dict[str, Dict],
    year: int,
    llm_processor: LLMProcessor,
    max_workers: int = 1
) -> Dict:
    """
    为单个Hub执行供应商选择（PER_CANDIDATE模式）

    Args:
        hub_id: Hub节点ID
        hub_profile: Hub的profile
        personality: Hub的性格
        candidate_pool: 候选池DataFrame
        supplier_profiles: 供应商profiles字典
        assessments: Stage4的评估结果字典
        year: 年份
        llm_processor: LLM处理器
        max_workers: 并行工作线程数

    Returns:
        选择结果字典
    """
    decision_context = personality.get('decision_context', 'No decision context available.')

    # 构建dossiers
    dossiers = []
    for _, row in candidate_pool.iterrows():
        supplier_id = row['factset_entity_id']
        source_label = row['source_label']

        # 获取supplier的profile和assessment
        supplier_profile = supplier_profiles.get(supplier_id)
        assessment = assessments.get(supplier_id)

        if not supplier_profile or not assessment:
            continue

        dossier = {
            'factset_entity_id': supplier_id,
            'source_label': source_label,
            'data_profile': supplier_profile,
            'llm_assessment': assessment
        }
        dossiers.append(dossier)

    if len(dossiers) == 0:
        return {
            'selected_suppliers': [],
            'selection_details': {},
            'statistics': {
                'total_candidates': 0,
                'selected_count': 0
            }
        }

    # 并行执行选择决策
    selection_details = {}
    lock = threading.Lock()

    def select_single_supplier(dossier: Dict) -> tuple:
        """为单个供应商做选择决策"""
        supplier_id = dossier['factset_entity_id']
        try:
            # 生成prompt
            prompt = create_single_supplier_selection_prompt(
                hub_profile,
                dossier,
                decision_context
            )

            # 调用LLM API
            response = llm_processor.chat_with_retry(
                prompt=prompt,
                hub_id=hub_id,
                supplier_id=supplier_id,
                year=year
            )

            return (supplier_id, response, None)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            return (supplier_id, None, str(e))

    # 并行评估
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_supplier = {
            executor.submit(select_single_supplier, dossier): dossier['factset_entity_id']
            for dossier in dossiers
        }

        try:
            with tqdm(total=len(dossiers), desc=f"  Selecting for {hub_id}") as pbar:
                for future in as_completed(future_to_supplier):
                    supplier_id = future_to_supplier[future]
                    try:
                        supplier_id_result, response, error = future.result()

                        with lock:
                            if error is None and response:
                                selection_details[supplier_id_result] = response

                        pbar.update(1)

                    except KeyboardInterrupt:
                        print(f"\n\n⚠ 用户中断！")
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise
                    except Exception as e:
                        print(f"\n  [ERROR] {supplier_id}: 选择失败 - {e}")
                        pbar.update(1)

        except KeyboardInterrupt:
            raise

    # 基于阈值确定选择结果（固定阈值0.5）
    selected_suppliers = []
    for supplier_id, details in selection_details.items():
        decision = details.get('decision', 'NOT SELECT')
        prob = details.get('selection_probability', 0.0)

        # 固定阈值0.5
        if prob >= SELECTION_PROBABILITY_THRESHOLD:
            selected_suppliers.append(supplier_id)

    # 统计
    statistics = {
        'total_candidates': len(candidate_pool),
        'total_dossiers': len(dossiers),
        'evaluated_count': len(selection_details),
        'selected_count': len(selected_suppliers),
        'selection_rate': len(selected_suppliers) / len(dossiers) if len(dossiers) > 0 else 0.0
    }

    return {
        'selected_suppliers': selected_suppliers,
        'selection_details': selection_details,
        'statistics': statistics
    }


def process_single_year(
    year: int,
    stage1_data: Dict,
    stage2_data: Dict,
    stage3a_data: Dict,
    stage4_data: Dict,
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
    hub_assessments = stage4_data['hub_assessments']

    print(f"  Hub节点数: {len(hub_nodes)}")
    print(f"  Hub画像数: {len(hub_independent_profiles)}")
    print(f"  Hub性格数: {len(personalities)}")
    print(f"  Hub评估数: {len(hub_assessments)}")

    # 为每个Hub执行选择
    all_hub_selections = {}

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

            if hub_id not in hub_assessments:
                print(f"  [WARNING] {hub_id}: 无评估结果，跳过")
                continue

            hub_profile = hub_independent_profiles[hub_id]
            personality = personalities[hub_id]
            candidate_pool = hub_candidate_pools[hub_id]
            assessments = hub_assessments[hub_id].get('assessments', {})

            # 执行选择
            selection_result = select_suppliers_for_hub(
                hub_id,
                hub_profile,
                personality,
                candidate_pool,
                hub_independent_profiles,
                assessments,
                year,
                llm_processor,
                max_workers=max_workers
            )

            all_hub_selections[hub_id] = selection_result

        except Exception as e:
            print(f"  [ERROR] {hub_id}: 处理失败 - {e}")
            import traceback
            traceback.print_exc()
            continue

    # 统计
    total_hubs = len(hub_nodes)
    processed_hubs = len(all_hub_selections)
    total_selected = sum(len(s['selected_suppliers']) for s in all_hub_selections.values())

    print(f"\n{year} 年统计:")
    print(f"  成功处理Hub数: {processed_hubs}/{total_hubs}")
    print(f"  总选择供应商数: {total_selected}")

    return {
        'year': year,
        'hub_selections': all_hub_selections,
        'statistics': {
            'total_hubs': total_hubs,
            'processed_hubs': processed_hubs,
            'total_selected': total_selected
        }
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Stage 5a: LLM供应商选择')
    parser.add_argument('--stage1_file', type=str, default=None)
    parser.add_argument('--stage2_file', type=str, default=None)
    parser.add_argument('--stage3a_file', type=str, default=None)
    parser.add_argument('--stage4_file', type=str, default=None)
    parser.add_argument('--result_dir', type=str, default=RESULT_DIR)
    parser.add_argument('--cache_dir', type=str, default='../cache')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--max_workers', type=int, default=16)

    args = parser.parse_args()

    print("=" * 80)
    print("Stage 5a: LLM供应商选择 (PER_CANDIDATE模式, 阈值=0.5)")
    if args.debug:
        print("【DEBUG模式 - 使用Mock LLM】")
    print(f"并行度: {args.max_workers} 个线程")
    print("=" * 80)

    # 创建缓存目录
    os.makedirs(args.cache_dir, exist_ok=True)
    cache_path = os.path.join(args.cache_dir, 'llm_selection_cache.json')

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

    if args.stage4_file:
        stage4_file = args.stage4_file
    else:
        stage4_files = list(Path(STAGE4_RESULT_DIR).glob("stage4_llm_assessments_*.pkl"))
        if not stage4_files:
            print("❌ 未找到Stage4输出文件")
            return
        stage4_file = str(sorted(stage4_files)[-1])

    print(f"Stage1文件: {stage1_file}")
    print(f"Stage2文件: {stage2_file}")
    print(f"Stage3a文件: {stage3a_file}")
    print(f"Stage4文件: {stage4_file}")

    # 加载数据
    with open(stage1_file, 'rb') as f:
        stage1_results = pickle.load(f)

    with open(stage2_file, 'rb') as f:
        stage2_results = pickle.load(f)

    with open(stage3a_file, 'rb') as f:
        stage3a_results = pickle.load(f)

    with open(stage4_file, 'rb') as f:
        stage4_results = pickle.load(f)

    # 确保年份一致
    years = sorted(
        set(stage1_results.keys()) &
        set(stage2_results.keys()) &
        set(stage3a_results.keys()) &
        set(stage4_results.keys())
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
                stage4_results[year],
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
        f'stage5a_llm_selections_hub{hub_pct_match}pct.pkl'
    )

    print(f"\n{'='*80}")
    print(f"保存结果到: {output_file}")

    with open(output_file, 'wb') as f:
        pickle.dump(all_year_results, f)

    print(f"✓ Stage 5a 完成！")

    # 统计
    print(f"\n{'='*80}")
    print("汇总统计")
    print(f"{'='*80}")
    print(f"成功处理年份: {len(all_year_results)}/{len(years)}\n")

    print("年份 | Hub总数 | 处理Hub | 总选择供应商")
    print("-" * 80)
    for year, result in sorted(all_year_results.items()):
        stats = result['statistics']
        print(f"{year} | {stats['total_hubs']:>7} | {stats['processed_hubs']:>7} | "
              f"{stats['total_selected']:>12}")


if __name__ == "__main__":
    main()
