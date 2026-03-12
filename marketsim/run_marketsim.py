"""
MarketSim Entry Point

Launches the MarketSim stock market simulation as a standalone module.
Can be called from the unified main.py or run independently.

Usage (standalone):
    cd marketsim && python abides.py -c rsmtry_LLM3 -t JNJ -d 20250402 -s 1234 -l rmsctry_LLM3 --enable-llm-cache

Usage (from unified main.py):
    python main.py --module marketsim -c rsmtry_LLM3 -t JNJ -d 20250402 -s 1234 -l rmsctry_LLM3 --enable-llm-cache
"""

import sys
import os
import importlib
import argparse

import ray


def get_marketsim_dir() -> str:
    """Return the absolute path of the marketsim package directory."""
    return os.path.dirname(os.path.abspath(__file__))


def _ensure_ray_with_marketsim_env(marketsim_dir: str):
    """
    Ensure Ray is initialized with the marketsim directory on every worker's
    sys.path.  This is critical because MarketSim modules (Agent_FLLM, Kernel,
    agent, util, …) use bare imports and Ray workers do NOT inherit the main
    process's dynamically modified sys.path.

    If Ray is already running we update the runtime env via
    ``ray.worker.global_worker`` (best-effort); if not, we start it with the
    correct ``runtime_env``.
    """
    runtime_env = {
        "working_dir": marketsim_dir,
        "env_vars": {
            "PYTHONPATH": marketsim_dir + os.pathsep + os.environ.get("PYTHONPATH", ""),
        },
    }

    if ray.is_initialized():
        # Ray is already up (e.g. started by the unified main.py).
        # We cannot change runtime_env after init, so inject via env var +
        # a worker init hook that the config modules may pick up.
        os.environ["PYTHONPATH"] = runtime_env["env_vars"]["PYTHONPATH"]
        print(f"[MarketSim] Ray already initialized — injected PYTHONPATH={marketsim_dir}")
    else:
        ray.init(
            include_dashboard=False,
            runtime_env=runtime_env,
        )
        print(f"[MarketSim] Ray initialized with working_dir={marketsim_dir}")


def run(argv: list = None):
    """
    Run the MarketSim simulation.

    Args:
        argv: Command-line arguments to pass to the config module.
              If None, uses sys.argv[1:].
    """
    marketsim_dir = get_marketsim_dir()

    # ---- path setup ----------------------------------------------------------
    # MarketSim internal modules (Kernel, agent, util, …) use bare imports like
    # ``from Kernel import Kernel``.  We need the marketsim directory on
    # sys.path so that these imports resolve correctly.
    if marketsim_dir not in sys.path:
        sys.path.insert(0, marketsim_dir)

    # Also make sure the project root is on the path (for shared utilities)
    project_root = os.path.dirname(marketsim_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # ---- Ray setup with runtime_env for worker processes ---------------------
    _ensure_ray_with_marketsim_env(marketsim_dir)

    # ---- parse only the config name first ------------------------------------
    parser = argparse.ArgumentParser(
        description="MarketSim: Stock Market Simulation with LLM Agents",
        add_help=False,  # let the config file handle --help
    )
    parser.add_argument(
        "-c", "--config", required=True,
        help="Name of the MarketSim config module (e.g. rsmtry_LLM3)",
    )
    parser.add_argument(
        "--config-help", action="store_true",
        help="Print argument options for the specific config file.",
    )

    if argv is not None:
        args, remaining = parser.parse_known_args(argv)
    else:
        args, remaining = parser.parse_known_args()

    config_name = args.config

    # ---- banner --------------------------------------------------------------
    banner = "MarketSim: Stock Market Simulation with LLM Agents"
    print("=" * len(banner))
    print(banner)
    print("=" * len(banner))
    print(f"\nConfig: {config_name}")
    print()

    # ---- save & restore argv for config modules that call parse_known_args ---
    # Config modules (e.g. rsmtry_LLM3.py) do their own argparse at import
    # time, reading from sys.argv.  We need to set sys.argv so they see the
    # right arguments.
    original_argv = sys.argv
    sys.argv = ["marketsim"] + (["-c", config_name] + remaining)

    # ---- save & restore cwd --------------------------------------------------
    original_cwd = os.getcwd()
    os.chdir(marketsim_dir)

    try:
        # Import the config module – this triggers the entire simulation run
        # (config modules build agents, create the Kernel, and call
        # kernel.runner() at module scope).
        config_module = importlib.import_module(f"config.{config_name}")
    finally:
        os.chdir(original_cwd)
        sys.argv = original_argv

    return config_module


if __name__ == "__main__":
    run()


