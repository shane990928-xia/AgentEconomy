"""
Supply Chain Module Entry Point
================================

Dispatches to one of three independent supply-chain experiments:

  network             – Static supply-chain network reconstruction
  network_continuous  – Evolutionary network simulation across years
  supplier            – Single-entity supplier selection with LLM

Usage (via unified main.py):
    python main.py --module supply_chain --experiment network \\
        --years 2018,2019,2020 --methods llm ml random

    python main.py --module supply_chain --experiment network_continuous \\
        --sandbox_folder supply_chain/data --start_year 2016 --end_year 2020

    python main.py --module supply_chain --experiment supplier \\
        --entity_id "000C7F-E" --year 2020 --use_llm

Usage (standalone):
    python supply_chain/run_supply_chain.py --experiment network --years 2020 --debug
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


# ── Paths ─────────────────────────────────────────────────────────────────────

SUPPLY_CHAIN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT     = SUPPLY_CHAIN_DIR.parent

EXPERIMENT_CODE_DIRS = {
    "network":            SUPPLY_CHAIN_DIR / "code" / "network",
    "network_continuous": SUPPLY_CHAIN_DIR / "code" / "network_continuous",
    "supplier":           SUPPLY_CHAIN_DIR / "code" / "supplier",
}

EXPERIMENT_DESCRIPTIONS = {
    "network":            "Static supply-chain network reconstruction (LLM / ML / Random)",
    "network_continuous": "Evolutionary network simulation across multiple years",
    "supplier":           "Single-entity supplier selection with LLM reasoning",
}


def _build_env(code_dir: Path) -> dict:
    """
    Build subprocess environment with supply_chain/ on PYTHONPATH so that
    ``from llm import LLM`` works in all stage scripts.

    Both the code subdirectory and supply_chain/ root are added so that:
      - supply_chain/llm.py  is found by all three experiment families
      - stage scripts can still import their siblings (stage0_config etc.)
    """
    env = os.environ.copy()
    extra_paths = [str(SUPPLY_CHAIN_DIR), str(code_dir), str(PROJECT_ROOT)]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in extra_paths + [existing] if p
    )
    return env


def run(argv: list = None):
    """
    Run a supply-chain experiment.

    Args:
        argv: Argument list (without the program name).
              If None, uses sys.argv[1:].
    """
    if argv is None:
        argv = sys.argv[1:]

    # ── parse --experiment first; pass everything else to run_pipeline.py ────
    parser = argparse.ArgumentParser(
        description="Supply Chain Module",
        add_help=False,
    )
    parser.add_argument(
        "--experiment", "-e",
        required=True,
        choices=list(EXPERIMENT_CODE_DIRS.keys()),
        help="Which experiment to run: network | network_continuous | supplier",
    )
    args, remaining = parser.parse_known_args(argv)
    experiment = args.experiment

    code_dir = EXPERIMENT_CODE_DIRS[experiment]

    # ── banner ────────────────────────────────────────────────────────────────
    banner = f"Supply Chain — {experiment}"
    print("=" * len(banner))
    print(banner)
    print(f"({EXPERIMENT_DESCRIPTIONS[experiment]})")
    print("=" * len(banner))
    print()

    # ── build the command: run run_pipeline.py inside the code dir ────────────
    cmd = [sys.executable, str(code_dir / "run_pipeline.py")] + remaining

    print(f"Working dir : {code_dir}")
    print(f"Command     : {' '.join(cmd)}")
    print()

    env = _build_env(code_dir)
    original_cwd = os.getcwd()

    try:
        os.chdir(code_dir)
        result = subprocess.run(
            cmd,
            env=env,
            check=False,  # we handle non-zero ourselves
        )
    finally:
        os.chdir(original_cwd)

    if result.returncode != 0:
        print(f"\n[supply_chain] '{experiment}' exited with code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    run()

