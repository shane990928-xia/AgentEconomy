#!/usr/bin/env python3
"""
Stage 3a: LLM Procurement Personality Generation
==================================================

功能：为所有Hub节点生成采购决策性格

从network_test.py复制性格生成相关代码
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
RESULT_DIR = "../result"

# ============================================================================
# 辅助函数（从network_test.py复制）
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
        self.lock = threading.Lock()  # 线程锁

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

    def _create_cache_key(self, prompt_type: str, entity_id: str, year: int) -> str:
        """创建缓存键"""
        return f"{prompt_type}_{entity_id}_{year}_PROFILE_ONLY"

    def chat_with_retry(
        self,
        prompt: str,
        entity_id: str,
        year: int,
        retry_delay: int = 5
    ) -> Dict:
        """
        调用LLM API并进行重试（无限重试直到成功）

        Args:
            prompt: prompt文本
            entity_id: 实体ID
            year: 年份
            retry_delay: 重试延迟（秒）

        Returns:
            解析后的JSON字典
        """
        cache_key = self._create_cache_key("personality", entity_id, year)

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
# Prompt模板（从network_test.py复制）
# ============================================================================

def create_personality_generation_prompt_base(self_profile: Dict) -> str:
    """
    基础采购性格生成prompt（只基于公司画像）
    从network_test.py复制
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




# ============================================================================
# 性格生成主函数
# ============================================================================

def generate_procurement_personality_batch(
    hub_profiles: Dict[str, Dict],
    year: int,
    llm_processor: LLMProcessor,
    max_workers: int = 1
) -> Dict[str, Dict]:
    """
    批量生成采购性格（调用真实LLM API，支持并行）

    Args:
        hub_profiles: {hub_id: profile_dict}
        year: 年份
        llm_processor: LLM处理器
        max_workers: 并行工作线程数（默认1=串行）

    Returns:
        {hub_id: {'decision_context_prompt': prompt_str, 'decision_context': llm_response, ...}}
    """
    print(f"\n--- [批量生成] 为 {len(hub_profiles)} 个Hub生成采购性格 ---")
    print(f"  并行度: {max_workers} 个线程")

    personality_data = {}
    success_count = 0
    lock = threading.Lock()  # 用于保护personality_data和success_count

    def process_single_hub(hub_id: str, hub_profile: Dict) -> tuple:
        """处理单个Hub（在线程中执行）"""
        try:
            # 生成prompt
            prompt = create_personality_generation_prompt_base(hub_profile)

            # 调用LLM API（无限重试直到成功）
            llm_response = llm_processor.chat_with_retry(
                prompt=prompt,
                entity_id=hub_id,
                year=year
            )

            # chat_with_retry保证返回成功结果
            result = {
                'hub_id': hub_id,
                'company_name': hub_profile.get('company_name', 'Unknown'),
                'decision_context_prompt': prompt,
                'decision_context': llm_response['decision_context'],
                'generated_at': datetime.now().isoformat(),
                'llm_response_full': llm_response
            }
            return (hub_id, result, None)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            # 记录错误信息
            error_result = {
                'hub_id': hub_id,
                'error': str(e)
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
            with tqdm(total=len(hub_profiles), desc="Generating personalities") as pbar:
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

    print(f"  ✓ 成功生成 {success_count} 个性格")
    return personality_data


# ============================================================================
# 主流程
# ============================================================================

def process_single_year(
    year: int,
    stage1_data: Dict,
    stage2_data: Dict,
    llm_processor: LLMProcessor,
    max_workers: int = 1
) -> Dict:
    """处理单个年份"""
    print(f"\n{'='*80}")
    print(f"处理 {year} 年")
    print(f"{'='*80}")

    # 从Stage1获取Hub节点列表
    hub_nodes = stage1_data['hub_nodes']
    print(f"  Hub节点数: {len(hub_nodes)}")

    # 从Stage2获取Hub无关的画像
    hub_independent_profiles = stage2_data['hub_independent_profiles']
    print(f"  Hub无关画像总数: {len(hub_independent_profiles)}")

    # 提取Hub节点的画像
    hub_profiles = {}
    missing_hubs = []
    for hub_id in hub_nodes:
        if hub_id in hub_independent_profiles:
            hub_profiles[hub_id] = hub_independent_profiles[hub_id]
        else:
            missing_hubs.append(hub_id)

    if missing_hubs:
        print(f"  ⚠ {len(missing_hubs)} 个Hub缺少画像: {missing_hubs[:5]}...")

    print(f"  实际用于生成性格的Hub数: {len(hub_profiles)}")

    # 生成采购性格（调用LLM API，支持并行）
    personality_data = generate_procurement_personality_batch(
        hub_profiles,
        year,
        llm_processor,
        max_workers=max_workers
    )

    return {
        'year': year,
        'personalities': personality_data,
        'statistics': {
            'total_hubs': len(hub_nodes),
            'available_profiles': len(hub_profiles),
            'success_count': sum(1 for p in personality_data.values() if 'error' not in p),
            'error_count': sum(1 for p in personality_data.values() if 'error' in p)
        }
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Stage 3a: LLM采购性格生成')
    parser.add_argument('--stage1_file', type=str, default=None,
                       help='Stage1输出文件路径')
    parser.add_argument('--stage2_file', type=str, default=None,
                       help='Stage2输出文件路径')
    parser.add_argument('--result_dir', type=str, default=RESULT_DIR)
    parser.add_argument('--cache_dir', type=str, default='../cache',
                       help='LLM缓存目录')
    parser.add_argument('--debug', action='store_true',
                       help='使用Mock LLM（不调用真实API）')
    parser.add_argument('--max_workers', type=int, default=16,
                       help='并行工作线程数（默认1=串行，建议5-10）')

    args = parser.parse_args()

    print("=" * 80)
    print("Stage 3a: LLM采购性格生成 (PROFILE_ONLY模式)")
    if args.debug:
        print("【DEBUG模式 - 使用Mock LLM】")
    print(f"并行度: {args.max_workers} 个线程")
    print("=" * 80)

    # 创建缓存目录
    os.makedirs(args.cache_dir, exist_ok=True)
    cache_path = os.path.join(args.cache_dir, 'llm_personality_cache.json')

    # 初始化LLM组件
    print("\n初始化LLM组件...")
    llm_agent = create_llm_agent(debug_mode=args.debug)
    llm_cache = LLMCache(cache_path)
    llm_processor = LLMProcessor(llm_agent, llm_cache)

    # 查找Stage1输出文件
    if args.stage1_file:
        stage1_file = args.stage1_file
    else:
        stage1_files = list(Path(STAGE1_RESULT_DIR).glob("stage1_candidate_pools_*.pkl"))
        if not stage1_files:
            print("❌ 未找到Stage1输出文件")
            return
        stage1_file = str(sorted(stage1_files)[-1])

    print(f"Stage1文件: {stage1_file}")

    # 查找Stage2输出文件
    if args.stage2_file:
        stage2_file = args.stage2_file
    else:
        stage2_files = list(Path(STAGE2_RESULT_DIR).glob("stage2_company_profiles_*.pkl"))
        if not stage2_files:
            print("❌ 未找到Stage2输出文件")
            return
        stage2_file = str(sorted(stage2_files)[-1])

    print(f"Stage2文件: {stage2_file}")

    # 加载Stage1结果
    with open(stage1_file, 'rb') as f:
        stage1_results = pickle.load(f)

    # 加载Stage2结果
    with open(stage2_file, 'rb') as f:
        stage2_results = pickle.load(f)

    # 确保两个stage有相同的年份
    years = sorted(set(stage1_results.keys()) & set(stage2_results.keys()))
    print(f"✓ 共同年份: {len(years)} 个 - {years}")

    # 处理所有年份
    all_year_results = {}
    for year in years:
        try:
            year_result = process_single_year(
                year,
                stage1_results[year],
                stage2_results[year],
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
        f'stage3a_llm_personalities_hub{hub_pct_match}pct.pkl'
    )

    print(f"\n{'='*80}")
    print(f"保存结果到: {output_file}")

    with open(output_file, 'wb') as f:
        pickle.dump(all_year_results, f)

    print(f"✓ Stage 3a 完成！")

    # 统计
    print(f"\n{'='*80}")
    print("汇总统计")
    print(f"{'='*80}")
    print(f"成功处理年份: {len(all_year_results)}/{len(years)}\n")

    print("年份 | Hub总数 | 有画像 | 成功 | 失败")
    print("-" * 80)
    for year, result in sorted(all_year_results.items()):
        stats = result['statistics']
        print(f"{year} | {stats['total_hubs']:>7} | {stats['available_profiles']:>6} | "
              f"{stats['success_count']:>4} | {stats['error_count']:>4}")


if __name__ == "__main__":
    main()