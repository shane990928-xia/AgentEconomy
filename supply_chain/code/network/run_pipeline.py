#!/usr/bin/env python3
"""
Run Pipeline: 静态网络重建实验
==============================

功能：按顺序运行所有阶段，完成从候选池构建到网络重建的完整流程

用法：
    # 运行完整流程（所有方法，默认年份）
    python run_pipeline.py --result_dir ../result/network

    # 指定年份
    python run_pipeline.py --years 2018,2019,2020 --result_dir ../result/network

    # 只运行LLM方法
    python run_pipeline.py --methods llm --result_dir ../result/network

    # Debug模式（使用Mock LLM）
    python run_pipeline.py --years 2020 --debug

流程：
    1. Stage 1: 构建Hub节点候选池
    2. Stage 2: 生成公司数据画像
    3a. Stage 3a: LLM采购性格生成（如果methods包含llm）
    3b. Stage 3b: ML训练数据抽取（如果methods包含ml）
    4. Stage 4: LLM供应商评估（如果methods包含llm）
    5a. Stage 5a: LLM供应商选择（如果methods包含llm）
    5b. Stage 5b: ML供应商选择（如果methods包含ml）
    5c. Stage 5c: Random供应商选择（如果methods包含random）
    6. Stage 6: 网络重构与分析
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

RESULT_DIR = "../result/network"
CACHE_DIR = "../cache"


# ============================================================================
# 辅助函数
# ============================================================================

def get_stage_files(result_dir: str, years: str) -> dict:
    """获取各阶段的输出文件路径"""
    return {
        'stage1': os.path.join(result_dir, f"stage1_candidate_pools_{years}.pkl"),
        'stage2': os.path.join(result_dir, f"stage2_profiles_{years}.pkl"),
        'stage3a': os.path.join(result_dir, f"stage3a_personalities_{years}.pkl"),
        'stage3b': os.path.join(result_dir, f"stage3b_ml_training_data_{years}.pkl"),
        'stage4': os.path.join(result_dir, f"stage4_llm_assessments_{years}.pkl"),
        'stage5a': os.path.join(result_dir, f"stage5a_llm_selections_{years}.pkl"),
        'stage5b_lr': os.path.join(result_dir, f"stage5b_lr_selections_{years}.pkl"),
        'stage5b_rf': os.path.join(result_dir, f"stage5b_rf_selections_{years}.pkl"),
        'stage5b_xgb': os.path.join(result_dir, f"stage5b_xgb_selections_{years}.pkl"),
        'stage5c': os.path.join(result_dir, f"stage5c_random_selections_{years}.pkl"),
        'stage6': os.path.join(result_dir, f"stage6_network_reconstruction_{years}.pkl"),
    }


def check_file_exists(filepath: str) -> bool:
    """检查文件是否存在"""
    return os.path.exists(filepath)


def run_command(cmd: List[str], description: str, dry_run: bool = False) -> bool:
    """
    运行命令并显示输出
    """
    print(f"\n{'='*80}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {description}")
    print(f"{'='*80}")
    print(f"命令: {' '.join(cmd)}")
    
    if dry_run:
        print("[DRY RUN] 跳过执行")
        return True
    
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True
        )
        elapsed = time.time() - start_time
        print(f"\n[完成] 耗时: {elapsed:.1f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n[失败] 退出码: {e.returncode}, 耗时: {elapsed:.1f}s")
        return False
    except Exception as e:
        print(f"\n[错误] {e}")
        return False


# ============================================================================
# Pipeline 阶段运行函数
# ============================================================================

def run_stage1(years: str, result_dir: str, hub_pct: float, noise_pct: float, 
               skip_existing: bool, dry_run: bool) -> bool:
    """Stage 1: 构建Hub节点候选池"""
    cmd = [
        sys.executable, "stage1_candidate_pool.py",
        "--years", years,
        "--result_dir", result_dir,
        "--hub_pct", str(hub_pct),
        "--noise_pct", str(noise_pct),
    ]
    return run_command(cmd, f"Stage 1: 构建候选池 (years={years})", dry_run)


def run_stage2(stage1_file: str, result_dir: str, debug: bool, dry_run: bool) -> bool:
    """Stage 2: 生成公司数据画像"""
    cmd = [
        sys.executable, "stage2_profile_builder.py",
        "--stage1_file", stage1_file,
        "--result_dir", result_dir,
    ]
    if debug:
        cmd.append("--debug")
    return run_command(cmd, "Stage 2: 生成公司画像", dry_run)


def run_stage3a(stage1_file: str, stage2_file: str, result_dir: str, 
                cache_dir: str, debug: bool, dry_run: bool) -> bool:
    """Stage 3a: LLM采购性格生成"""
    cmd = [
        sys.executable, "stage3a_llm_personality.py",
        "--stage1_file", stage1_file,
        "--stage2_file", stage2_file,
        "--result_dir", result_dir,
        "--cache_dir", cache_dir,
    ]
    if debug:
        cmd.append("--debug")
    return run_command(cmd, "Stage 3a: LLM性格生成", dry_run)


def run_stage3b(stage1_file: str, result_dir: str, dry_run: bool) -> bool:
    """Stage 3b: ML训练数据抽取"""
    cmd = [
        sys.executable, "stage3b_ml_training_data.py",
        "--stage1_file", stage1_file,
        "--result_dir", result_dir,
    ]
    return run_command(cmd, "Stage 3b: ML训练数据抽取", dry_run)


def run_stage4(stage1_file: str, stage2_file: str, stage3a_file: str,
               result_dir: str, cache_dir: str, debug: bool, 
               max_workers: int, dry_run: bool) -> bool:
    """Stage 4: LLM供应商评估"""
    cmd = [
        sys.executable, "stage4_llm_assessment.py",
        "--stage1_file", stage1_file,
        "--stage2_file", stage2_file,
        "--stage3a_file", stage3a_file,
        "--result_dir", result_dir,
        "--cache_dir", cache_dir,
        "--max_workers", str(max_workers),
    ]
    if debug:
        cmd.append("--debug")
    return run_command(cmd, "Stage 4: LLM供应商评估", dry_run)


def run_stage5a(stage1_file: str, stage2_file: str, stage3a_file: str, 
                stage4_file: str, result_dir: str, cache_dir: str,
                debug: bool, max_workers: int, dry_run: bool) -> bool:
    """Stage 5a: LLM供应商选择"""
    cmd = [
        sys.executable, "stage5a_llm_selection.py",
        "--stage1_file", stage1_file,
        "--stage2_file", stage2_file,
        "--stage3a_file", stage3a_file,
        "--stage4_file", stage4_file,
        "--result_dir", result_dir,
        "--cache_dir", cache_dir,
        "--max_workers", str(max_workers),
    ]
    if debug:
        cmd.append("--debug")
    return run_command(cmd, "Stage 5a: LLM供应商选择", dry_run)


def run_stage5b(stage1_file: str, stage2_file: str, stage3b_file: str,
                result_dir: str, n_jobs: int, dry_run: bool) -> bool:
    """Stage 5b: ML供应商选择"""
    cmd = [
        sys.executable, "stage5b_ml_selection.py",
        "--stage1_file", stage1_file,
        "--stage2_file", stage2_file,
        "--stage3b_file", stage3b_file,
        "--result_dir", result_dir,
        "--n_jobs", str(n_jobs),
    ]
    return run_command(cmd, "Stage 5b: ML供应商选择", dry_run)


def run_stage5c(stage1_file: str, result_dir: str, random_seed: int, dry_run: bool) -> bool:
    """Stage 5c: Random供应商选择"""
    cmd = [
        sys.executable, "stage5c_random_selection.py",
        "--stage1_file", stage1_file,
        "--result_dir", result_dir,
        "--random_seed", str(random_seed),
    ]
    return run_command(cmd, "Stage 5c: Random供应商选择", dry_run)


def run_stage6(stage1_file: str, stage5a_file: str, stage5b_lr_file: str,
               stage5b_rf_file: str, stage5b_xgb_file: str, stage5c_file: str,
               result_dir: str, dry_run: bool) -> bool:
    """Stage 6: 网络重构与分析"""
    cmd = [
        sys.executable, "stage6_network_reconstruction.py",
        "--stage1_file", stage1_file,
        "--result_dir", result_dir,
    ]
    if stage5a_file and check_file_exists(stage5a_file):
        cmd.extend(["--stage5a_file", stage5a_file])
    if stage5b_lr_file and check_file_exists(stage5b_lr_file):
        cmd.extend(["--stage5b_lr_file", stage5b_lr_file])
    if stage5b_rf_file and check_file_exists(stage5b_rf_file):
        cmd.extend(["--stage5b_rf_file", stage5b_rf_file])
    if stage5b_xgb_file and check_file_exists(stage5b_xgb_file):
        cmd.extend(["--stage5b_xgb_file", stage5b_xgb_file])
    if stage5c_file and check_file_exists(stage5c_file):
        cmd.extend(["--stage5c_file", stage5c_file])
    
    return run_command(cmd, "Stage 6: 网络重构与分析", dry_run)


# ============================================================================
# 主Pipeline函数
# ============================================================================

def run_pipeline(
    years: str,
    methods: List[str],
    result_dir: str,
    cache_dir: str,
    hub_pct: float,
    noise_pct: float,
    debug: bool,
    max_workers: int,
    n_jobs: int,
    random_seed: int,
    skip_existing: bool,
    dry_run: bool,
) -> bool:
    """
    运行完整的Pipeline
    """
    print("\n" + "=" * 80)
    print("网络重建实验 Pipeline")
    print("=" * 80)
    print(f"年份: {years}")
    print(f"方法: {', '.join(methods)}")
    print(f"结果目录: {result_dir}")
    print(f"Debug模式: {debug}")
    print(f"跳过已存在: {skip_existing}")
    print("=" * 80 + "\n")
    
    # 创建目录
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    # 获取文件路径
    files = get_stage_files(result_dir, years.replace(",", "_"))
    
    # Stage 1: 构建候选池
    if not skip_existing or not check_file_exists(files['stage1']):
        if not run_stage1(years, result_dir, hub_pct, noise_pct, skip_existing, dry_run):
            print("[Pipeline] Stage 1 失败，终止")
            return False
    else:
        print(f"[跳过] Stage 1 输出已存在: {files['stage1']}")
    
    # Stage 2: 生成画像
    if not skip_existing or not check_file_exists(files['stage2']):
        if not run_stage2(files['stage1'], result_dir, debug, dry_run):
            print("[Pipeline] Stage 2 失败，终止")
            return False
    else:
        print(f"[跳过] Stage 2 输出已存在: {files['stage2']}")
    
    # Stage 3a: LLM性格（如果需要LLM方法）
    if 'llm' in methods:
        if not skip_existing or not check_file_exists(files['stage3a']):
            if not run_stage3a(files['stage1'], files['stage2'], result_dir, cache_dir, debug, dry_run):
                print("[Pipeline] Stage 3a 失败，终止")
                return False
        else:
            print(f"[跳过] Stage 3a 输出已存在: {files['stage3a']}")
    
    # Stage 3b: ML训练数据（如果需要ML方法）
    if 'ml' in methods:
        if not skip_existing or not check_file_exists(files['stage3b']):
            if not run_stage3b(files['stage1'], result_dir, dry_run):
                print("[Pipeline] Stage 3b 失败，终止")
                return False
        else:
            print(f"[跳过] Stage 3b 输出已存在: {files['stage3b']}")
    
    # Stage 4: LLM评估（如果需要LLM方法）
    if 'llm' in methods:
        if not skip_existing or not check_file_exists(files['stage4']):
            if not run_stage4(files['stage1'], files['stage2'], files['stage3a'],
                            result_dir, cache_dir, debug, max_workers, dry_run):
                print("[Pipeline] Stage 4 失败，终止")
                return False
        else:
            print(f"[跳过] Stage 4 输出已存在: {files['stage4']}")
    
    # Stage 5a: LLM选择
    if 'llm' in methods:
        if not skip_existing or not check_file_exists(files['stage5a']):
            if not run_stage5a(files['stage1'], files['stage2'], files['stage3a'],
                             files['stage4'], result_dir, cache_dir, debug, max_workers, dry_run):
                print("[Pipeline] Stage 5a 失败，终止")
                return False
        else:
            print(f"[跳过] Stage 5a 输出已存在: {files['stage5a']}")
    
    # Stage 5b: ML选择
    if 'ml' in methods:
        if not skip_existing or not check_file_exists(files['stage5b_lr']):
            if not run_stage5b(files['stage1'], files['stage2'], files['stage3b'],
                             result_dir, n_jobs, dry_run):
                print("[Pipeline] Stage 5b 失败，终止")
                return False
        else:
            print(f"[跳过] Stage 5b 输出已存在")
    
    # Stage 5c: Random选择
    if 'random' in methods:
        if not skip_existing or not check_file_exists(files['stage5c']):
            if not run_stage5c(files['stage1'], result_dir, random_seed, dry_run):
                print("[Pipeline] Stage 5c 失败，终止")
                return False
        else:
            print(f"[跳过] Stage 5c 输出已存在: {files['stage5c']}")
    
    # Stage 6: 网络重构
    if not run_stage6(files['stage1'], files['stage5a'], files['stage5b_lr'],
                     files['stage5b_rf'], files['stage5b_xgb'], files['stage5c'],
                     result_dir, dry_run):
        print("[Pipeline] Stage 6 失败，终止")
        return False
    
    print("\n" + "=" * 80)
    print("[Pipeline] 全部完成!")
    print("=" * 80)
    
    return True


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='静态网络重建实验 Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 运行完整流程
    python run_pipeline.py --years 2018,2019,2020
    
    # 只运行LLM方法
    python run_pipeline.py --years 2020 --methods llm
    
    # Debug模式
    python run_pipeline.py --years 2020 --debug
    
    # 跳过已存在的结果
    python run_pipeline.py --years 2018,2019,2020 --skip_existing
        """
    )
    
    parser.add_argument('--years', type=str, default='2018,2019,2020',
                        help='实验年份，逗号分隔 (默认: 2018,2019,2020)')
    parser.add_argument('--methods', type=str, nargs='+', default=['llm', 'ml', 'random'],
                        choices=['llm', 'ml', 'random'],
                        help='选择方法 (默认: llm ml random)')
    parser.add_argument('--result_dir', type=str, default=RESULT_DIR,
                        help=f'结果目录 (默认: {RESULT_DIR})')
    parser.add_argument('--cache_dir', type=str, default=CACHE_DIR,
                        help=f'缓存目录 (默认: {CACHE_DIR})')
    parser.add_argument('--hub_pct', type=float, default=0.20,
                        help='Hub节点百分比 (默认: 0.20)')
    parser.add_argument('--noise_pct', type=float, default=0.01,
                        help='噪声候选百分比 (默认: 0.01)')
    parser.add_argument('--debug', action='store_true',
                        help='Debug模式（使用Mock LLM）')
    parser.add_argument('--max_workers', type=int, default=16,
                        help='LLM并行线程数 (默认: 16)')
    parser.add_argument('--n_jobs', type=int, default=16,
                        help='ML训练并行数 (默认: 16)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='跳过已存在的阶段输出')
    parser.add_argument('--dry_run', action='store_true',
                        help='只打印命令，不实际执行')
    
    args = parser.parse_args()
    
    success = run_pipeline(
        years=args.years,
        methods=args.methods,
        result_dir=args.result_dir,
        cache_dir=args.cache_dir,
        hub_pct=args.hub_pct,
        noise_pct=args.noise_pct,
        debug=args.debug,
        max_workers=args.max_workers,
        n_jobs=args.n_jobs,
        random_seed=args.random_seed,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
