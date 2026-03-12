#!/usr/bin/env python3
"""
Run Pipeline: Full Evolutionary Network Simulation
===================================================

功能：按顺序运行所有阶段，完成从候选池构建到评估报告的完整流程

用法：
    # 运行完整流程（所有方法）
    python run_pipeline.py --sandbox_folder ../data --start_year 2016 --end_year 2020

    # 只运行LLM方法
    python run_pipeline.py --sandbox_folder ../data --start_year 2016 --end_year 2020 --methods llm

    # 运行指定年份
    python run_pipeline.py --sandbox_folder ../data --year 2018 --methods llm ml random

    # Debug模式（使用Mock LLM）
    python run_pipeline.py --sandbox_folder ../data --year 2018 --debug

流程：
    对于每个年份：
    1. Stage 1: 构建候选池
    2. Stage 2: 准备公司画像
    3a. Stage 3a: 生成LLM性格（如果methods包含llm）
    3b. Stage 3b: 准备ML训练数据（如果methods包含ml）
    4. Stage 4: LLM评估（如果methods包含llm）
    5a. Stage 5a: LLM选择（如果methods包含llm）
    5b. Stage 5b: ML选择（如果methods包含ml）
    5c. Stage 5c: Random选择（如果methods包含random）
    6. Stage 6: 评估报告

依赖关系：
    - Stage 1 <- 前一年的Stage 5演化网络（第一年除外）
    - Stage 2 <- Stage 1
    - Stage 3a <- Stage 1, Stage 2
    - Stage 3b <- Stage 1, Stage 2
    - Stage 4 <- Stage 1, Stage 2, Stage 3a
    - Stage 5a <- Stage 1, Stage 2, Stage 3a, Stage 4
    - Stage 5b <- Stage 1, Stage 2, Stage 3b
    - Stage 5c <- Stage 2
    - Stage 6 <- Stage 5a/5b/5c (所有选择方法)
"""

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime
from typing import List, Optional

# ============================================================================
# 全局配置
# ============================================================================

RESULT_DIR = "../result"

# ============================================================================
# 辅助函数
# ============================================================================

def check_stage_output_exists(stage: str, year: int, result_dir: str, method: str = None) -> bool:
    """
    检查阶段输出文件是否已存在

    Args:
        stage: 阶段名称 ('stage1', 'stage2', 'stage3a', 'stage3b', 'stage4', 'stage5a', 'stage5b', 'stage5c', 'stage6')
        year: 年份
        result_dir: 结果目录
        method: 对于stage5b，指定模型类型；对于其他阶段可选

    Returns:
        是否存在所有必需的输出文件
    """
    if stage == 'stage1':
        # Stage 1: 目录和元数据文件
        pool_dir = os.path.join(result_dir, f"stage1_candidate_pools_{year}")
        metadata_file = os.path.join(result_dir, f"stage1_metadata_{year}.json")
        return os.path.isdir(pool_dir) and os.path.exists(metadata_file)

    elif stage == 'stage2':
        # Stage 2: profiles文件
        profile_file = os.path.join(result_dir, f"stage2_profiles_{year}.pkl")
        return os.path.exists(profile_file)

    elif stage == 'stage3a':
        # Stage 3a: personalities文件
        personality_file = os.path.join(result_dir, f"stage3a_personalities_{year}.pkl")
        return os.path.exists(personality_file)

    elif stage == 'stage3b':
        # Stage 3b: ML训练数据文件
        training_file = os.path.join(result_dir, f"stage3b_ml_training_data_{year}.pkl")
        return os.path.exists(training_file)

    elif stage == 'stage4':
        # Stage 4: LLM评估文件
        assessment_file = os.path.join(result_dir, f"stage4_llm_assessments_{year}.pkl")
        return os.path.exists(assessment_file)

    elif stage == 'stage5a':
        # Stage 5a: LLM选择和演化网络
        selection_file = os.path.join(result_dir, f"stage5a_llm_selection_{year}.pkl")
        graph_file = os.path.join(result_dir, f"stage5a_evolving_graph_{year}.pkl")
        return os.path.exists(selection_file) and os.path.exists(graph_file)

    elif stage == 'stage5b':
        # Stage 5b: ML选择（需要检查所有3个模型）
        model_types = ['logistic_regression', 'random_forest', 'xgboost']
        for model_type in model_types:
            selection_file = os.path.join(result_dir, f"stage5b_{model_type}_selection_{year}.pkl")
            graph_file = os.path.join(result_dir, f"stage5b_{model_type}_evolving_graph_{year}.pkl")
            if not (os.path.exists(selection_file) and os.path.exists(graph_file)):
                return False
        return True

    elif stage == 'stage5c':
        # Stage 5c: Random选择
        selection_file = os.path.join(result_dir, f"stage5c_random_selection_{year}.pkl")
        graph_file = os.path.join(result_dir, f"stage5c_random_evolving_graph_{year}.pkl")
        return os.path.exists(selection_file) and os.path.exists(graph_file)

    elif stage == 'stage6':
        # Stage 6: 评估报告
        report_file = os.path.join(result_dir, f"stage6_evaluation_report_{year}.json")
        return os.path.exists(report_file)

    return False


def run_command(cmd: List[str], description: str) -> bool:
    """
    运行命令并显示输出

    Args:
        cmd: 命令列表
        description: 命令描述

    Returns:
        是否成功
    """
    print(f"\n{'='*80}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {description}")
    print(f"{'='*80}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            stdout=sys.stdout,
            stderr=sys.stderr
        )

        elapsed = time.time() - start_time
        print(f"\n✓ 完成！用时: {elapsed:.2f}秒")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ 失败！用时: {elapsed:.2f}秒")
        print(f"错误代码: {e.returncode}")
        return False


def detect_available_years(sandbox_folder: str) -> List[int]:
    """
    自动检测沙盒数据文件夹中可用的年份

    Args:
        sandbox_folder: 沙盒数据文件夹路径

    Returns:
        可用年份列表（排序）
    """
    import glob
    import re

    nodes_files = glob.glob(os.path.join(sandbox_folder, "*_nodes.csv"))
    years = []

    for file_path in nodes_files:
        filename = os.path.basename(file_path)
        match = re.search(r'(\d{4})_nodes\.csv', filename)
        if match:
            years.append(int(match.group(1)))

    return sorted(years)


# ============================================================================
# 主流程
# ============================================================================

def run_pipeline_for_year(
    year: int,
    sandbox_folder: str,
    methods: List[str],
    result_dir: str,
    personality_mode: str = 'PROFILE_ONLY',
    debug: bool = False,
    max_workers: int = 16,
    n_jobs: int = 16
) -> bool:
    """
    运行指定年份的完整流程

    Args:
        year: 年份
        sandbox_folder: 沙盒数据文件夹路径
        methods: 选择方法列表 ['llm', 'ml', 'random']
        result_dir: 结果目录
        personality_mode: 性格生成模式
        debug: 是否使用debug模式（Mock LLM）
        max_workers: LLM并行线程数
        n_jobs: ML训练并行线程数

    Returns:
        是否成功
    """
    print(f"\n\n{'#'*80}")
    print(f"# 开始处理 {year} 年")
    print(f"# 方法: {', '.join(methods)}")
    print(f"{'#'*80}\n")

    # 确定selection_method（用于Stage 1加载前一年的演化网络）
    # 优先级：llm > ml (使用第一个ML模型) > random
    if 'llm' in methods:
        selection_method = 'llm'
    elif 'ml' in methods:
        selection_method = 'logistic_regression'  # 默认使用第一个ML模型
    elif 'random' in methods:
        selection_method = 'random'
    else:
        print("错误：必须指定至少一种选择方法")
        return False

    # Stage 1: 候选池构建
    if check_stage_output_exists('stage1', year, result_dir):
        print(f"\n{'='*80}")
        print(f"[跳过] Stage 1: {year}年候选池已存在")
        print(f"{'='*80}\n")
    else:
        cmd = [
            'python', 'stage1_candidate_pool.py',
            '--year', str(year),
            '--sandbox_folder', sandbox_folder,
            '--selection_method', selection_method,
            '--result_dir', result_dir
        ]
        if not run_command(cmd, f"Stage 1: 构建{year}年候选池"):
            return False

    # Stage 2: 公司画像准备
    if check_stage_output_exists('stage2', year, result_dir):
        print(f"\n{'='*80}")
        print(f"[跳过] Stage 2: {year}年公司画像已存在")
        print(f"{'='*80}\n")
    else:
        cmd = [
            'python', 'stage2_profile_builder.py',
            '--year', str(year),
            '--result_dir', result_dir
        ]
        if not run_command(cmd, f"Stage 2: 准备{year}年公司画像"):
            return False

    # Stage 3a: LLM性格生成（如果需要）
    if 'llm' in methods:
        if check_stage_output_exists('stage3a', year, result_dir):
            print(f"\n{'='*80}")
            print(f"[跳过] Stage 3a: {year}年LLM性格已存在")
            print(f"{'='*80}\n")
        else:
            cmd = [
                'python', 'stage3a_llm_personality.py',
                '--year', str(year),
                '--personality_mode', personality_mode,
                '--result_dir', result_dir,
                '--max_workers', str(max_workers)
            ]
            if debug:
                cmd.append('--debug')

            if not run_command(cmd, f"Stage 3a: 生成{year}年LLM性格"):
                return False

    # Stage 3b: ML训练数据准备（如果需要）
    if 'ml' in methods:
        if check_stage_output_exists('stage3b', year, result_dir):
            print(f"\n{'='*80}")
            print(f"[跳过] Stage 3b: {year}年ML训练数据已存在")
            print(f"{'='*80}\n")
        else:
            cmd = [
                'python', 'stage3b_ml_training_data.py',
                '--year', str(year),
                '--result_dir', result_dir
            ]
            if not run_command(cmd, f"Stage 3b: 准备{year}年ML训练数据"):
                return False

    # Stage 4: LLM评估（如果需要）
    if 'llm' in methods:
        if check_stage_output_exists('stage4', year, result_dir):
            print(f"\n{'='*80}")
            print(f"[跳过] Stage 4: {year}年LLM评估已存在")
            print(f"{'='*80}\n")
        else:
            cmd = [
                'python', 'stage4_llm_assessment.py',
                '--year', str(year),
                '--result_dir', result_dir,
                '--max_workers', str(max_workers)
            ]
            if debug:
                cmd.append('--debug')

            if not run_command(cmd, f"Stage 4: 执行{year}年LLM评估"):
                return False

    # Stage 5a: LLM选择（如果需要）
    if 'llm' in methods:
        if check_stage_output_exists('stage5a', year, result_dir):
            print(f"\n{'='*80}")
            print(f"[跳过] Stage 5a: {year}年LLM选择已存在")
            print(f"{'='*80}\n")
        else:
            cmd = [
                'python', 'stage5a_llm_selection.py',
                '--year', str(year),
                '--sandbox_folder', sandbox_folder,
                '--result_dir', result_dir,
                '--max_workers', str(max_workers)
            ]
            if debug:
                cmd.append('--debug')

            if not run_command(cmd, f"Stage 5a: 执行{year}年LLM选择"):
                return False

    # Stage 5b: ML选择（如果需要）
    if 'ml' in methods:
        if check_stage_output_exists('stage5b', year, result_dir):
            print(f"\n{'='*80}")
            print(f"[跳过] Stage 5b: {year}年ML选择已存在（所有模型）")
            print(f"{'='*80}\n")
        else:
            cmd = [
                'python', 'stage5b_ml_selection.py',
                '--year', str(year),
                '--sandbox_folder', sandbox_folder,
                '--result_dir', result_dir,
                '--n_jobs', str(n_jobs)
            ]
            if not run_command(cmd, f"Stage 5b: 执行{year}年ML选择"):
                return False

    # Stage 5c: Random选择（如果需要）
    if 'random' in methods:
        if check_stage_output_exists('stage5c', year, result_dir):
            print(f"\n{'='*80}")
            print(f"[跳过] Stage 5c: {year}年Random选择已存在")
            print(f"{'='*80}\n")
        else:
            cmd = [
                'python', 'stage5c_random_selection.py',
                '--year', str(year),
                '--sandbox_folder', sandbox_folder,
                '--result_dir', result_dir
            ]
            if not run_command(cmd, f"Stage 5c: 执行{year}年Random选择"):
                return False

    # Stage 6: 评估报告
    if check_stage_output_exists('stage6', year, result_dir):
        print(f"\n{'='*80}")
        print(f"[跳过] Stage 6: {year}年评估报告已存在")
        print(f"{'='*80}\n")
    else:
        cmd = [
            'python', 'stage6_evaluation_reporting.py',
            '--year', str(year),
            '--sandbox_folder', sandbox_folder,
            '--result_dir', result_dir
        ]
        if not run_command(cmd, f"Stage 6: 生成{year}年评估报告"):
            return False

    print(f"\n{'#'*80}")
    print(f"# ✓ {year} 年处理完成！")
    print(f"{'#'*80}\n")

    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='运行完整的演化网络模拟流程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 运行完整流程（所有方法）
  python run_pipeline.py --sandbox_folder ../data --start_year 2016 --end_year 2020

  # 只运行LLM方法
  python run_pipeline.py --sandbox_folder ../data --start_year 2016 --end_year 2020 --methods llm

  # 运行指定年份
  python run_pipeline.py --sandbox_folder ../data --year 2018 --methods llm ml random

  # Debug模式
  python run_pipeline.py --sandbox_folder ../data --year 2018 --debug
        """
    )

    # 必需参数
    parser.add_argument(
        '--sandbox_folder',
        type=str,
        required=True,
        help='沙盒网络数据文件夹路径'
    )

    # 年份参数（二选一）
    year_group = parser.add_mutually_exclusive_group(required=True)
    year_group.add_argument(
        '--year',
        type=int,
        help='处理单个年份'
    )
    year_group.add_argument(
        '--start_year',
        type=int,
        help='开始年份（与--end_year一起使用）'
    )

    parser.add_argument(
        '--end_year',
        type=int,
        help='结束年份（与--start_year一起使用）'
    )

    # 选择方法
    parser.add_argument(
        '--methods',
        type=str,
        nargs='+',
        default=['llm', 'ml', 'random'],
        choices=['llm', 'ml', 'random'],
        help='选择方法列表（默认：llm ml random）'
    )

    # Personality模式
    parser.add_argument(
        '--personality_mode',
        type=str,
        default='PROFILE_ONLY',
        choices=['PROFILE_ONLY', 'PROFILE_WITH_PREV_YEAR', 'PROFILE_WITH_ALL_PRIOR'],
        help='LLM性格生成模式（默认：PROFILE_ONLY）'
    )

    # 其他参数
    parser.add_argument(
        '--result_dir',
        type=str,
        default=RESULT_DIR,
        help='结果保存目录'
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
        help='LLM并行线程数（默认16）'
    )

    parser.add_argument(
        '--n_jobs',
        type=int,
        default=16,
        help='ML训练并行线程数（默认16）'
    )

    args = parser.parse_args()

    # 验证参数
    if args.start_year is not None and args.end_year is None:
        parser.error("--start_year 需要配合 --end_year 使用")

    # 打印配置
    print("=" * 80)
    print("演化网络模拟流程")
    print("=" * 80)
    print(f"沙盒文件夹: {args.sandbox_folder}")
    print(f"结果目录: {args.result_dir}")
    print(f"选择方法: {', '.join(args.methods)}")
    print(f"Personality模式: {args.personality_mode}")
    if args.debug:
        print("【DEBUG模式 - 使用Mock LLM】")
    print(f"LLM并行线程数: {args.max_workers}")
    print(f"ML训练并行线程数: {args.n_jobs}")
    print("=" * 80)

    # 确定年份列表
    if args.year is not None:
        years = [args.year]
        print(f"\n处理年份: {args.year}")
    else:
        # 检测可用年份
        available_years = detect_available_years(args.sandbox_folder)
        if not available_years:
            print(f"错误：在 {args.sandbox_folder} 中找不到任何年份的数据文件")
            return

        print(f"\n检测到可用年份: {available_years}")

        start_year = args.start_year
        end_year = args.end_year

        # 确保在可用范围内
        if start_year < min(available_years):
            print(f"警告：开始年份 {start_year} 小于数据最早年份 {min(available_years)}，调整为 {min(available_years)+1}")
            start_year = min(available_years) + 1

        if end_year > max(available_years):
            print(f"警告：结束年份 {end_year} 大于数据最晚年份 {max(available_years)}，调整为 {max(available_years)}")
            end_year = max(available_years)

        # 演化评估从 start_year+1 开始（需要前一年的初始化）
        if start_year == min(available_years):
            start_year = min(available_years) + 1
            print(f"注意：演化评估从 {start_year} 年开始（需要 {start_year-1} 年初始化）")

        years = list(range(start_year, end_year + 1))
        print(f"处理年份: {years}")

    # 创建结果目录
    os.makedirs(args.result_dir, exist_ok=True)

    # 运行流程
    start_time = time.time()
    success_count = 0
    failed_years = []

    for year in years:
        success = run_pipeline_for_year(
            year=year,
            sandbox_folder=args.sandbox_folder,
            methods=args.methods,
            result_dir=args.result_dir,
            personality_mode=args.personality_mode,
            debug=args.debug,
            max_workers=args.max_workers,
            n_jobs=args.n_jobs
        )

        if success:
            success_count += 1
        else:
            failed_years.append(year)
            print(f"\n⚠ {year} 年处理失败！")

            # 询问是否继续
            if len(years) > 1:
                response = input(f"\n是否继续处理下一个年份？(y/n): ")
                if response.lower() != 'y':
                    break

    # 打印总结
    total_time = time.time() - start_time

    print(f"\n\n{'='*80}")
    print("流程完成总结")
    print(f"{'='*80}")
    print(f"总用时: {total_time/60:.2f} 分钟")
    print(f"成功处理: {success_count}/{len(years)} 个年份")

    if failed_years:
        print(f"失败年份: {failed_years}")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
