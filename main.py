from agenteconomy.agent.household import *
from agenteconomy.utils.logger import get_logger
from agenteconomy.center.LaborMarket import LaborMarket
from config.config import SimulationConfig
from agenteconomy.simulation.simulator import Simulator
import argparse
import asyncio
logger = get_logger(name="main")
import ray

def parse_args():
    parser = argparse.ArgumentParser(description="Simulation configuration")
    parser.add_argument("--config", type=str, default="config/config_normal.yaml", help="Configuration file path")
    return parser.parse_args()

async def main(config: SimulationConfig):
    """
    Main simulation function.
    
    Args:
        config: SimulationConfig instance loaded from YAML
    """
    simulator = Simulator(config)
    await simulator.setup_simulation_environment()
    await simulator.run_simulation()

if __name__ == "__main__":
    ray.init(num_cpus=128, num_gpus=2)
    args = parse_args()
    
    # Load configuration from YAML file
    config = SimulationConfig.from_yaml(args.config)
    
    logger.info(f"Simulation started with config: {args.config}")
    asyncio.run(main(config))
    logger.info(f"Simulation ended")