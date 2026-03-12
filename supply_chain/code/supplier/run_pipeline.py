#!/usr/bin/env python3
"""
供应商选择实验 - 主运行流程
==========================

整合所有阶段，提供完整的实验执行流程。

用法：
    python run_pipeline.py --entity_id "000C7F-E" --year 2020
    python run_pipeline.py --debug  # 调试模式（不使用LLM）
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

# 导入各阶段模块
from stage0_config import (
    DATA_BASE_PATH, RESULT_DIR, CACHE_DIR,
    create_llm_agent, LLMProcessor, LLMCache, ResultCache
)
from stage1_data_loader import load_all_dataframes_parallel
from stage2_profile_builder import (
    build_self_profile, parallel_prepare_supplier_profiles
)
from stage3_personality import (
    generate_decision_maker_personality, build_personality_prompt
)
from stage4_assessment import parallel_assess_suppliers
from stage5_selection import (
    get_supplier_candidates, select_top_suppliers,
    diversify_selection, generate_selection_report, format_report_text
)
from stage6_evaluation import (
    evaluate_selection_accuracy, generate_evaluation_report, format_evaluation_text
)


DEFAULT_CONFIG = {
    'max_candidates': 100,
    'num_financial_reports': 1,
    'top_k': 10,
    'min_score': 5.0,
    'max_workers': 16,
    'look_ahead_years': 1,
    'debug_mode': False,
}


def run_single_experiment(target_entity_id, decision_year, config=None, llm_processor=None):
    """运行单个实验"""
    import pandas as pd
    
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    debug_mode = config.get('debug_mode', False)
    
    print("=" * 60)
    print("供应商选择实验")
    print(f"目标公司: {target_entity_id}")
    print(f"决策年份: {decision_year}")
    print(f"调试模式: {debug_mode}")
    print("=" * 60)
    
    experiment_result = {
        'target_entity_id': target_entity_id,
        'decision_year': decision_year,
        'config': config,
        'start_time': datetime.now().isoformat(),
        'stages': {}
    }
    
    try:
        # Stage 1: 数据加载
        print("\n[Stage 1] 数据加载...")
        t0 = time.time()
        all_dfs = load_all_dataframes_parallel()
        t1 = time.time() - t0
        print(f"  -> 完成，加载了 {len(all_dfs)} 个数据集，耗时 {t1:.2f}s")
        experiment_result['stages']['data_loading'] = {'status': 'success', 'time_seconds': round(t1, 2)}
        
        # Stage 2.1: 构建自身画像
        print("\n[Stage 2.1] 构建目标公司画像...")
        t0 = time.time()
        self_profile = build_self_profile(target_entity_id, decision_year, all_dfs, config.get('num_financial_reports', 1))
        if 'error' in self_profile:
            raise ValueError(f"无法构建目标公司画像: {self_profile['error']}")
        t1 = time.time() - t0
        print(f"  -> 公司: {self_profile.get('entity_name', 'Unknown')}")
        print(f"  -> 行业: {self_profile.get('industry', 'Unknown')}")
        experiment_result['stages']['self_profile'] = {'status': 'success', 'time_seconds': round(t1, 2)}
        
        # Stage 2.2: 获取候选供应商
        print("\n[Stage 2.2] 获取候选供应商...")
        t0 = time.time()
        candidate_df = get_supplier_candidates(target_entity_id, decision_year, all_dfs, 
                                                max_candidates=config.get('max_candidates', 100))
        t1 = time.time() - t0
        print(f"  -> 候选数量: {len(candidate_df)}")
        if len(candidate_df) == 0:
            raise ValueError("未找到任何候选供应商")
        experiment_result['stages']['candidate_retrieval'] = {'status': 'success', 'num_candidates': len(candidate_df)}
        
        # Stage 2.3: 构建供应商画像
        print("\n[Stage 2.3] 并行构建供应商画像...")
        t0 = time.time()
        cutoff_date = pd.Timestamp(f'{decision_year}-01-01')
        supplier_profiles = parallel_prepare_supplier_profiles(
            candidate_df, self_profile, cutoff_date, all_dfs,
            num_financial_reports=config.get('num_financial_reports', 1),
            max_workers=config.get('max_workers', 16)
        )
        t1 = time.time() - t0
        print(f"  -> 完成，构建了 {len(supplier_profiles)} 个画像，耗时 {t1:.2f}s")
        experiment_result['stages']['profile_building'] = {'status': 'success', 'num_profiles': len(supplier_profiles)}
        
        # Stage 3: 生成决策者性格
        print("\n[Stage 3] 生成决策者性格...")
        t0 = time.time()
        personality = generate_decision_maker_personality(self_profile, llm_processor=llm_processor, debug_mode=debug_mode)
        t1 = time.time() - t0
        print(f"  -> 生成方法: {personality.get('generation_method', 'unknown')}")
        experiment_result['stages']['personality_generation'] = {'status': 'success', 'method': personality.get('generation_method')}
        
        # Stage 4: 供应商评估
        print("\n[Stage 4] 并行评估供应商...")
        t0 = time.time()
        assessments = parallel_assess_suppliers(
            supplier_profiles, self_profile, personality,
            llm_processor=llm_processor,
            max_workers=config.get('max_workers', 16),
            debug_mode=debug_mode
        )
        t1 = time.time() - t0
        print(f"  -> 完成 {len(assessments)} 个评估，耗时 {t1:.2f}s")
        experiment_result['stages']['assessment'] = {'status': 'success', 'num_assessments': len(assessments)}
        
        # Stage 5: 选择最佳供应商
        print("\n[Stage 5] 选择最佳供应商...")
        selected_suppliers = select_top_suppliers(
            assessments, supplier_profiles,
            top_k=config.get('top_k', 10),
            min_score=config.get('min_score', 5.0)
        )
        selected_suppliers = diversify_selection(selected_suppliers, diversity_factor='country', max_per_group=5)
        print(f"  -> 选中 {len(selected_suppliers)} 个供应商")
        
        selection_report = generate_selection_report(self_profile, selected_suppliers, assessments, total_candidates=len(candidate_df))
        experiment_result['stages']['selection'] = {'status': 'success', 'num_selected': len(selected_suppliers)}
        experiment_result['selection_report'] = selection_report
        
        print("\n" + "-" * 40)
        print("选中的供应商:")
        print("-" * 40)
        for i, sup in enumerate(selected_suppliers[:10]):
            print(f"  {i+1}. {sup.get('entity_name', 'Unknown')}")
            print(f"     国家: {sup.get('country')} | 评分: {sup.get('overall_score')} | 推荐: {sup.get('recommendation')}")
        
        # Stage 6: 评估准确性
        print("\n[Stage 6] 评估选择准确性...")
        candidate_ids = set(candidate_df['factset_entity_id'].tolist())
        evaluation_result = evaluate_selection_accuracy(
            target_entity_id, decision_year, selected_suppliers, all_dfs,
            candidate_ids=candidate_ids, look_ahead_years=config.get('look_ahead_years', 1)
        )
        experiment_result['stages']['evaluation'] = {'status': 'success'}
        experiment_result['evaluation'] = evaluation_result
        
        print("\n" + format_evaluation_text(evaluation_result))
        
        # 保存结果
        print("\n[保存结果]...")
        experiment_result['end_time'] = datetime.now().isoformat()
        experiment_result['total_time_seconds'] = sum(s.get('time_seconds', 0) for s in experiment_result['stages'].values())
        
        os.makedirs(RESULT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = os.path.join(RESULT_DIR, f"experiment_{target_entity_id}_{decision_year}_{timestamp}.json")
        
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(v) for v in obj]
            elif isinstance(obj, set):
                return list(obj)
            elif hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            else:
                try:
                    json.dumps(obj)
                    return obj
                except:
                    return str(obj)
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(clean_for_json(experiment_result), f, ensure_ascii=False, indent=2)
        print(f"  -> 结果已保存至: {result_file}")
        
        experiment_result['status'] = 'success'
        
    except Exception as e:
        import traceback
        experiment_result['status'] = 'error'
        experiment_result['error'] = str(e)
        experiment_result['traceback'] = traceback.format_exc()
        print(f"\n[错误] {e}")
        traceback.print_exc()
    
    return experiment_result


def main():
    parser = argparse.ArgumentParser(description='供应商选择实验')
    parser.add_argument('--entity_id', type=str, default='000C7F-E', help='目标公司ID')
    parser.add_argument('--year', type=int, default=2020, help='决策年份')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--use_llm', action='store_true', help='使用LLM')
    parser.add_argument('--max_candidates', type=int, default=100)
    parser.add_argument('--top_k', type=int, default=10)
    
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()
    config['debug_mode'] = args.debug
    config['max_candidates'] = args.max_candidates
    config['top_k'] = args.top_k
    
    llm_processor = None
    if args.use_llm and not args.debug:
        print("初始化LLM...")
        llm_agent = create_llm_agent(debug_mode=False)
        os.makedirs(CACHE_DIR, exist_ok=True)
        llm_cache = LLMCache(os.path.join(CACHE_DIR, 'llm_cache.json'))
        result_cache = ResultCache(os.path.join(CACHE_DIR, 'result_cache.json'))
        llm_processor = LLMProcessor(llm_agent, llm_cache, result_cache)
    
    result = run_single_experiment(args.entity_id, args.year, config, llm_processor)
    print(f"\n实验完成，状态: {result.get('status')}")


if __name__ == "__main__":
    main()
