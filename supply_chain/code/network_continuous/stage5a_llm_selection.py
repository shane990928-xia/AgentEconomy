#!/usr/bin/env python3
"""
Stage 5a: LLM Supplier Selection (Evolution Mode)
==================================================

功能：使用LLM进行供应商选择决策并构建演化网络（演化模式）

输入：
- stage1_candidate_pools_{year}/ (候选池)
- stage2_profiles_{year}.pkl (画像，包含hub_profiles)
- stage3a_personalities_{year}.pkl (性格)
- stage4_llm_assessments_{year}.pkl (评估结果)

输出：
- stage5a_llm_selection_{year}.pkl
  格式：{
    'year': year,
    'hub_selections': {
      hub_id: {
        'selected_suppliers': [...],
        'selection_details': {...},
        'statistics': {...}
      }
    },
    'statistics': {...}
  }

- stage5a_evolving_graph_{year}.pkl
  格式：NetworkX有向图（供应商 -> 客户）

选择模式：PER_CANDIDATE（逐个评估，selection_probability >= 0.5 即选择）

参考：
- network/code/stage5a_llm_selection.py
- network_test.py: create_single_supplier_selection_prompt()
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
import networkx as nx
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


# ============================================================================
# 辅助函数：加载真实网络
# ============================================================================

def load_sandbox_network(sandbox_folder: str, year: int) -> nx.DiGraph:
    """
    加载指定年份的沙盒网络图（真实网络）

    Args:
        sandbox_folder: 沙盒数据文件夹路径
        year: 年份

    Returns:
        NetworkX有向图（供应商 -> 客户）
    """
    nodes_file = os.path.join(sandbox_folder, f"{year}_nodes.csv")
    edges_file = os.path.join(sandbox_folder, f"{year}_edges.csv")

    if not os.path.exists(nodes_file) or not os.path.exists(edges_file):
        raise FileNotFoundError(f"{year}年的沙盒网络数据不存在")

    # 加载节点和边
    nodes_df = pd.read_csv(nodes_file)
    edges_df = pd.read_csv(edges_file)

    # 创建有向图
    G = nx.DiGraph()

    # 添加节点（适配真实数据字段名'Id'）
    for _, row in nodes_df.iterrows():
        node_id = str(row['Id'])  # 确保节点ID为字符串
        G.add_node(node_id)

    # 添加边（适配真实数据字段名'Source', 'Target'）
    for _, row in edges_df.iterrows():
        source = str(row['Source'])
        target = str(row['Target'])
        if source in G.nodes and target in G.nodes:
            G.add_edge(source, target)

    print(f"  ✓ 加载{year}年真实网络: {G.number_of_nodes()}个节点, {G.number_of_edges()}条边")

    return G
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
                        print(f"  [WARNING] {hub_id}->{supplier_id}: JSON解析失败或缺少decision字段 (尝试 #{attempt})")
                else:
                    print(f"  [WARNING] {hub_id}->{supplier_id}: LLM返回空响应 (尝试 #{attempt})")

            except Exception as e:
                print(f"  [ERROR] {hub_id}->{supplier_id}: LLM调用异常 (尝试 #{attempt}): {e}")

            # 等待后重试
            if attempt % 5 == 0:
                print(f"  → 已重试{attempt}次，将在 {retry_delay} 秒后继续...")
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
# Ground Truth 辅助函数
# ============================================================================

def get_ground_truth_supplier_count(candidate_pool_df: pd.DataFrame) -> int:
    """
    从候选池获取 ground truth 供应商数量

    Ground truth 数量 = historical_suppliers（继续存在的） + future_opportunities（新增的）

    Args:
        candidate_pool_df: 候选池DataFrame，包含 'source_label' 列

    Returns:
        Ground truth 供应商数量
    """
    if candidate_pool_df is None or candidate_pool_df.empty:
        return 0

    # 统计 historical_suppliers 和 future_opportunities
    ground_truth_suppliers = candidate_pool_df[
        candidate_pool_df['source_label'].isin(['historical_suppliers', 'future_opportunities'])
    ]

    return len(ground_truth_suppliers)


# ============================================================================
# Prompt模板（完全对齐network_test.py）
# ============================================================================

def _format_self_identity_for_prompt(self_profile: dict, decision_context: str) -> str:
    """
    格式化buyer身份（对齐network_test.py）
    """
    name = self_profile.get("entity_proper_name", self_profile.get("company_name", "Our Company"))
    return f"You are a strategist for **{name}**. Your company's core decision-making philosophy is as follows:\n**{decision_context}**"


def create_single_supplier_selection_prompt(
    self_profile: dict,
    dossier: Dict,
    decision_context: str
) -> str:
    """
    创建供应商选择prompt（PER_CANDIDATE模式，完全对齐network_test.py）
    """
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
# 选择主流程
# ============================================================================

def select_suppliers_for_hub(
    hub_id: str,
    hub_supplier_profiles: Dict[str, Dict],
    personality: Dict,
    assessments: Dict[str, Dict],
    year: int,
    llm_processor: LLMProcessor,
    hub_independent_profiles: Dict[str, Dict],
    candidate_pool_df: Optional[pd.DataFrame] = None,
    max_workers: int = 1
) -> Dict:
    """
    为单个Hub执行供应商选择（PER_CANDIDATE模式 + Equal_adjustment阈值调整）

    Args:
        hub_id: Hub节点ID
        hub_supplier_profiles: Hub特定的supplier profiles字典（包含hub自己）
        personality: Hub的性格
        assessments: Stage4的评估结果字典
        year: 年份
        llm_processor: LLM处理器
        hub_independent_profiles: Hub无关的profiles字典（用于获取hub自己的profile）
        candidate_pool_df: 候选池DataFrame（用于获取ground truth数量）
        max_workers: 并行工作线程数

    Returns:
        选择结果字典
    """
    decision_context = personality.get('decision_context', 'No decision context available.')

    # 从hub_independent_profiles中获取hub自己的profile（作为buyer）
    hub_profile = hub_independent_profiles.get(hub_id)
    if not hub_profile:
        print(f"  [WARNING] {hub_id}: Hub自己的profile不存在，无法进行选择")
        return {
            'selected_suppliers': [],
            'selection_details': {},
            'statistics': {
                'total_candidates': 0,
                'selected_count': 0
            }
        }

    # 构建dossiers（排除hub自己）
    dossiers = []
    for supplier_id in hub_supplier_profiles.keys():
        if supplier_id == hub_id:
            continue  # 跳过hub自己

        # 获取supplier的profile和assessment
        supplier_profile = hub_supplier_profiles.get(supplier_id)
        assessment = assessments.get(supplier_id)

        if not supplier_profile or not assessment:
            continue

        dossier = {
            'factset_entity_id': supplier_id,
            'source_label': supplier_profile.get('source_label', 'unknown'),
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
            with tqdm(total=len(dossiers), desc=f"  选择 {hub_id}") as pbar:
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

    # Equal_adjustment 阈值调整（对齐supplier_test.py）
    selected_suppliers = []

    # 收集所有候选者的预测概率
    prediction_probabilities = {
        supplier_id: details.get('selection_probability', 0.0)
        for supplier_id, details in selection_details.items()
    }

    if len(prediction_probabilities) > 0:
        # 获取 ground truth 供应商数量
        n = get_ground_truth_supplier_count(candidate_pool_df)

        if n > 0:
            # 按概率从高到低排序
            sorted_probs = sorted(prediction_probabilities.values(), reverse=True)
            n = min(n, len(sorted_probs))

            # 找到唯一的候选阈值（避免重复概率导致的问题）
            unique_sorted_probs = sorted(set(sorted_probs), reverse=True)

            # 基于目标位置n，找到附近的唯一阈值进行测试
            target_threshold = sorted_probs[n - 1]

            # 收集候选阈值：目标阈值及其附近的唯一值
            candidate_thresholds = set()
            for idx in range(max(0, n - 3), min(len(sorted_probs), n + 3)):
                candidate_thresholds.add(sorted_probs[idx])

            # 如果候选阈值太少，添加更多唯一阈值
            if len(candidate_thresholds) < 5 and len(unique_sorted_probs) > 0:
                # 找到目标阈值在唯一列表中的位置
                try:
                    target_idx = unique_sorted_probs.index(target_threshold)
                    # 添加前后的唯一阈值
                    for offset in range(-2, 3):
                        idx = target_idx + offset
                        if 0 <= idx < len(unique_sorted_probs):
                            candidate_thresholds.add(unique_sorted_probs[idx])
                except ValueError:
                    # 如果找不到，使用前5个唯一阈值
                    candidate_thresholds.update(unique_sorted_probs[:5])

            best_threshold = None
            best_count_diff = float('inf')
            best_selected = []

            for threshold in candidate_thresholds:
                # 计算使用此阈值会选中多少个
                selected = [
                    supplier_id for supplier_id, prob in prediction_probabilities.items()
                    if prob >= threshold
                ]
                count_diff = abs(len(selected) - n)

                # 如果这个阈值更接近目标数量，则更新
                if count_diff < best_count_diff:
                    best_count_diff = count_diff
                    best_threshold = threshold
                    best_selected = selected

            # 使用最优阈值的选择结果
            selected_suppliers = best_selected

            print(f"  → [Equal_adjustment] n={n}, threshold={best_threshold:.6f}, "
                  f"selected={len(selected_suppliers)}/{len(prediction_probabilities)} "
                  f"(diff={abs(len(selected_suppliers) - n)})")
        else:
            # 如果没有 ground truth（候选池为空或没有正类），跳过阈值调整
            print(f"  → [Equal_adjustment] ground truth 数量为0，跳过阈值调整")

    # 统计
    statistics = {
        'total_candidates': len(dossiers),
        'evaluated_count': len(selection_details),
        'selected_count': len(selected_suppliers),
        'selection_rate': len(selected_suppliers) / len(dossiers) if len(dossiers) > 0 else 0.0,
        'ground_truth_count': get_ground_truth_supplier_count(candidate_pool_df)
    }

    return {
        'selected_suppliers': selected_suppliers,
        'selection_details': selection_details,
        'statistics': statistics
    }


# ============================================================================
# 演化网络构建
# ============================================================================

def build_evolving_graph(
    previous_graph: nx.DiGraph,
    hub_nodes: List[str],
    hub_selections: Dict[str, Dict],
    year: int
) -> nx.DiGraph:
    """
    基于前一年的网络和选择结果构建演化网络图

    Args:
        previous_graph: 前一年的演化网络图
        hub_nodes: Hub节点列表
        hub_selections: 所有Hub的选择结果
        year: 年份

    Returns:
        NetworkX有向图（供应商 -> 客户）

    流程：
    1. 复制前一年的网络
    2. 移除所有Hub的入边（要重新选择供应商）
    3. 根据选择结果添加新边和新节点
    4. 移除孤立节点（节点回收机制）
    """
    print(f"\n  → 构建{year}年演化网络图...")
    print(f"    前一年网络: {previous_graph.number_of_nodes()}个节点, {previous_graph.number_of_edges()}条边")

    # 1. 复制前一年的网络
    G = previous_graph.copy()

    # 2. 移除所有Hub的入边（准备重新选择供应商）
    edges_to_remove = []
    for hub_id in hub_nodes:
        if hub_id in G:
            in_edges = list(G.in_edges(hub_id))
            edges_to_remove.extend(in_edges)

    G.remove_edges_from(edges_to_remove)
    print(f"    移除 {len(edges_to_remove)} 条Hub入边")

    # 3. 根据选择结果添加新边和新节点
    new_nodes = set()
    new_edges = []

    for hub_id, selection_result in hub_selections.items():
        selected_suppliers = selection_result['selected_suppliers']

        for supplier_id in selected_suppliers:
            # 节点增加机制：如果供应商不在图中，添加它
            if supplier_id not in G:
                G.add_node(supplier_id)
                new_nodes.add(supplier_id)

            # 添加边（供应商 -> Hub）
            G.add_edge(supplier_id, hub_id)
            new_edges.append((supplier_id, hub_id))

    print(f"    新增 {len(new_nodes)} 个节点")
    print(f"    新增 {len(new_edges)} 条边")

    # 4. 节点回收机制：移除孤立节点（没有任何边的节点）
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    if len(isolated_nodes) > 0:
        print(f"    回收 {len(isolated_nodes)} 个孤立节点")

    print(f"  ✓ 演化网络: {G.number_of_nodes()}个节点, {G.number_of_edges()}条边")

    return G


# ============================================================================
# 主处理流程
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Stage 5a: LLM供应商选择（演化模式）'
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
        '--sandbox_folder',
        type=str,
        required=True,
        help='沙盒网络数据文件夹路径（用于加载真实网络）'
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
    print("Stage 5a: LLM供应商选择（演化模式）")
    print("=" * 80)
    print(f"处理年份: {args.year}")
    print(f"结果目录: {args.result_dir}")
    if args.debug:
        print("【DEBUG模式 - 使用Mock LLM】")
    print(f"并行度: {args.max_workers} 个线程")
    print(f"阈值策略: Equal_adjustment（动态阈值）")
    print("=" * 80)

    # 步骤1: 检查Stage 1、2、3a、4输出
    print(f"\n[1/5] 检查Stage 1、2、3a、4输出")

    profiles_file = os.path.join(
        args.result_dir,
        f"stage2_profiles_{args.year}.pkl"
    )

    personalities_file = os.path.join(
        args.result_dir,
        f"stage3a_personalities_{args.year}.pkl"
    )

    assessments_file = os.path.join(
        args.result_dir,
        f"stage4_llm_assessments_{args.year}.pkl"
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

    if not os.path.exists(assessments_file):
        raise FileNotFoundError(
            f"找不到Stage 4评估: {assessments_file}\n"
            f"请先运行: python stage4_llm_assessment.py --year {args.year}"
        )

    # 加载数据
    print(f"  → 加载Stage 2画像...")
    with open(profiles_file, 'rb') as f:
        stage2_data = pickle.load(f)

    hub_profiles = stage2_data['hub_profiles']
    hub_independent_profiles = stage2_data['hub_independent_profiles']
    print(f"  ✓ 加载了 {len(hub_profiles)} 个Hub的画像")
    print(f"  ✓ 加载了 {len(hub_independent_profiles)} 个Hub无关画像")

    print(f"  → 加载Stage 3a性格...")
    with open(personalities_file, 'rb') as f:
        personalities = pickle.load(f)

    print(f"  ✓ 加载了 {len(personalities)} 个Hub的性格")

    print(f"  → 加载Stage 4评估...")
    with open(assessments_file, 'rb') as f:
        stage4_data = pickle.load(f)

    hub_assessments = stage4_data['hub_assessments']
    print(f"  ✓ 加载了 {len(hub_assessments)} 个Hub的评估")

    # 加载Stage 1候选池（用于获取ground truth数量）
    candidate_pool_dir = os.path.join(args.result_dir, f"stage1_candidate_pools_{args.year}")

    candidate_pools = {}
    if os.path.exists(candidate_pool_dir):
        print(f"  → 加载Stage 1候选池...")
        pool_files = [f for f in os.listdir(candidate_pool_dir) if f.endswith('.pkl')]
        for pool_file in pool_files:
            hub_id = pool_file.replace('.pkl', '')
            pool_path = os.path.join(candidate_pool_dir, pool_file)
            with open(pool_path, 'rb') as f:
                candidate_pools[hub_id] = pickle.load(f)
        print(f"  ✓ 加载了 {len(candidate_pools)} 个Hub的候选池")
    else:
        print(f"  ⚠ 警告：找不到候选池目录 {candidate_pool_dir}")
        print(f"  → 将使用默认阈值调整（可能影响准确性）")

    # 步骤2: 初始化LLM组件
    print(f"\n[2/5] 初始化LLM组件")

    # 创建缓存目录
    os.makedirs(args.cache_dir, exist_ok=True)
    cache_path = os.path.join(args.cache_dir, 'llm_selection_cache.json')

    # 初始化LLM组件
    print("  → 初始化LLM组件...")
    llm_agent = create_llm_agent(debug_mode=args.debug)
    llm_cache = LLMCache(cache_path)
    llm_processor = LLMProcessor(llm_agent, llm_cache)

    # 步骤3: 为每个Hub执行供应商选择
    print(f"\n[3/5] 为每个Hub执行供应商选择")

    all_hub_selections = {}
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

            if hub_id not in hub_assessments:
                print(f"  [WARNING] {hub_id}: 无评估结果，跳过")
                continue

            hub_supplier_profiles = hub_profiles[hub_id]
            personality = personalities[hub_id]
            assessments = hub_assessments[hub_id].get('assessments', {})

            # 获取候选池
            candidate_pool_df = candidate_pools.get(hub_id, None)

            # 执行选择
            selection_result = select_suppliers_for_hub(
                hub_id,
                hub_supplier_profiles,
                personality,
                assessments,
                args.year,
                llm_processor,
                hub_independent_profiles,
                candidate_pool_df=candidate_pool_df,
                max_workers=args.max_workers
            )

            all_hub_selections[hub_id] = selection_result
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
    total_selected = sum(len(s['selected_suppliers']) for s in all_hub_selections.values())

    print(f"\n{args.year} 年统计:")
    print(f"  成功处理Hub数: {success_count}/{len(hub_nodes)}")
    print(f"  总选择供应商数: {total_selected}")

    # 步骤4: 加载前一年的演化网络
    print(f"\n[4/5] 加载前一年的演化网络")

    # 加载前一年的evolving_graph作为基础
    prev_year = args.year - 1
    prev_graph_file = os.path.join(args.result_dir, f"stage5a_llm_evolving_graph_{prev_year}.pkl")

    if os.path.exists(prev_graph_file):
        print(f"  → 加载{prev_year}年的演化网络...")
        with open(prev_graph_file, 'rb') as f:
            previous_graph = pickle.load(f)
        print(f"  ✓ 前一年网络: {previous_graph.number_of_nodes()}个节点, {previous_graph.number_of_edges()}条边")
    else:
        # 第一年：使用真实网络初始化
        print(f"  → {prev_year}年的演化网络不存在，尝试加载{prev_year}年真实网络...")
        try:
            previous_graph = load_sandbox_network(args.sandbox_folder, prev_year)
            print(f"  ✓ 使用{prev_year}年真实网络作为初始演化网络")
        except FileNotFoundError as e:
            print(f"  ⚠ 警告：{e}")
            print(f"  → 创建空图初始化")
            previous_graph = nx.DiGraph()
            for hub_id in hub_nodes:
                previous_graph.add_node(hub_id)

    # 步骤5: 构建演化网络并保存结果
    print(f"\n[5/5] 构建演化网络并保存结果")

    # 构建演化网络图
    evolving_graph = build_evolving_graph(
        previous_graph,
        hub_nodes,
        all_hub_selections,
        args.year
    )

    # 保存选择结果
    selection_output_file = os.path.join(
        args.result_dir,
        f"stage5a_llm_selection_{args.year}.pkl"
    )

    os.makedirs(args.result_dir, exist_ok=True)

    result = {
        'year': args.year,
        'hub_selections': all_hub_selections,
        'statistics': {
            'total_hubs': len(hub_nodes),
            'processed_hubs': success_count,
            'total_selected': total_selected
        }
    }

    with open(selection_output_file, 'wb') as f:
        pickle.dump(result, f)

    print(f"  ✓ 选择结果已保存到: {selection_output_file}")

    # 保存演化网络图
    graph_output_file = os.path.join(
        args.result_dir,
        f"stage5a_evolving_graph_{args.year}.pkl"
    )

    with open(graph_output_file, 'wb') as f:
        pickle.dump(evolving_graph, f)

    print(f"  ✓ 演化网络已保存到: {graph_output_file}")

    # 打印完成信息
    print(f"\n{'='*80}")
    print(f"✓ Stage 5a 完成！")
    print(f"{'='*80}")
    print(f"处理年份: {args.year}")
    print(f"Hub总数: {len(hub_nodes)}")
    print(f"成功处理Hub数: {success_count}")
    print(f"总选择供应商数: {total_selected}")
    print(f"演化网络: {evolving_graph.number_of_nodes()}个节点, {evolving_graph.number_of_edges()}条边")
    print(f"\n{'='*80}")
    print(f"下一步: 运行 stage6_evaluation_reporting.py --year {args.year}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
