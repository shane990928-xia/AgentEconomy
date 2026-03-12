"""
AgentEconomy Unified Entry Point

Supports running different simulation modules via the --module / -m flag:
  - agenteconomy  (default) : LLM-based macroeconomic simulation
  - marketsim               : LLM-based stock market simulation
  - supply_chain            : Supply-chain research experiments

Examples:
    # Run the macroeconomic simulation (default)
    python main.py --config config/config_normal.yaml

    # Resume from latest checkpoint
    python main.py --resume-latest

    # Run MarketSim stock market simulation
    python main.py --module marketsim -c rsmtry_LLM3 -t JNJ -d 20250402 -s 1234 \\
        -l rmsctry_LLM3 --enable-llm-cache

    # Supply chain — static network reconstruction
    python main.py --module supply_chain --experiment network \\
        --years 2018,2019,2020 --methods llm ml random

    # Supply chain — evolutionary network simulation
    python main.py --module supply_chain --experiment network_continuous \\
        --sandbox_folder supply_chain/data --start_year 2016 --end_year 2020

    # Supply chain — single-entity supplier selection
    python main.py --module supply_chain --experiment supplier \\
        --entity_id "000C7F-E" --year 2020 --use_llm
"""

import sys
import argparse
import asyncio

import ray

from agenteconomy.utils.logger import get_logger

logger = get_logger(name="main")


# =============================================================================
# AgentEconomy (macro-economic simulation)
# =============================================================================

def parse_agenteconomy_args(argv: list):
    """Parse CLI arguments for the AgentEconomy module."""
    parser = argparse.ArgumentParser(description="AgentEconomy simulation configuration")
    parser.add_argument("--module", type=str, default="agenteconomy", help=argparse.SUPPRESS)
    parser.add_argument("--config", type=str, default="config/config_normal.yaml",
                        help="Configuration file path")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint file path")
    parser.add_argument("--resume-latest", action="store_true",
                        help="Resume from latest checkpoint")
    return parser.parse_args(argv)


async def run_agenteconomy(args):
    """Run the AgentEconomy macro-economic simulation."""
    from config.config import SimulationConfig
    from agenteconomy.simulation.simulator import Simulator

    config = SimulationConfig.from_yaml(args.config)
    logger.info(f"AgentEconomy simulation started with config: {args.config}")

    simulator = Simulator(config)
    await simulator.setup_simulation_environment()

    if args.resume_latest:
        logger.info("Resuming from latest checkpoint...")
        await simulator.run_simulation_from_checkpoint(None)
    elif args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        await simulator.run_simulation_from_checkpoint(args.resume)
    else:
        await simulator.run_simulation()

    logger.info("AgentEconomy simulation ended")


# =============================================================================
# MarketSim (stock-market simulation)
# =============================================================================

def run_marketsim(argv: list):
    """Run the MarketSim stock-market simulation."""
    from marketsim.run_marketsim import run as marketsim_run

    logger.info("MarketSim simulation started")
    marketsim_run(argv=argv)
    logger.info("MarketSim simulation ended")


# =============================================================================
# Supply Chain
# =============================================================================

def run_supply_chain(argv: list):
    """Run a supply-chain experiment."""
    from supply_chain.run_supply_chain import run as supply_chain_run

    logger.info("Supply Chain experiment started")
    supply_chain_run(argv=argv)
    logger.info("Supply Chain experiment ended")


# =============================================================================
# Dispatcher
# =============================================================================

MODULES = {
    "agenteconomy": "LLM-based macroeconomic simulation",
    "marketsim":    "LLM-based stock market simulation",
    "supply_chain": "Supply-chain research experiments (network / network_continuous / supplier)",
}


def detect_module(argv: list) -> str:
    """Peek at argv to determine which module the user wants to run."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--module", "-m", type=str, default="agenteconomy",
                        choices=list(MODULES.keys()))
    known, _ = parser.parse_known_args(argv)
    return known.module


def strip_module_arg(argv: list) -> list:
    """Remove --module / -m and its value from argv so sub-parsers don't choke."""
    result = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg in ("--module", "-m"):
            skip_next = True
            continue
        if arg.startswith("--module=") or arg.startswith("-m="):
            continue
        result.append(arg)
    return result


def main():
    argv = sys.argv[1:]
    module = detect_module(argv)
    remaining = strip_module_arg(argv)

    # Banner
    print("=" * 60)
    print(f"  AgentEconomy Platform — module: {module}")
    print(f"  ({MODULES[module]})")
    print("=" * 60)
    print()

    if module == "agenteconomy":
        ray.init(num_cpus=128, num_gpus=2)
        args = parse_agenteconomy_args(remaining)
        asyncio.run(run_agenteconomy(args))

    elif module == "marketsim":
        # Do NOT init Ray here — run_marketsim handles it with the correct
        # runtime_env so that worker processes can find Agent_FLLM, Kernel, etc.
        run_marketsim(remaining)

    elif module == "supply_chain":
        # Supply-chain pipelines are plain subprocess-based; no Ray needed here.
        run_supply_chain(remaining)

    else:
        print(f"Unknown module: {module}")
        print(f"Available modules: {', '.join(MODULES.keys())}")
        sys.exit(1)


if __name__ == "__main__":
    main()
