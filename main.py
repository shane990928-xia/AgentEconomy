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
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint file path")
    parser.add_argument("--resume-latest", action="store_true", help="Resume from latest checkpoint")
    return parser.parse_args()

async def main(config: SimulationConfig, resume_checkpoint: str = None):
    """
    Main simulation function.
    
    Args:
        config: SimulationConfig instance loaded from YAML
        resume_checkpoint: Optional checkpoint path to resume from
    """
    simulator = Simulator(config)
    await simulator.setup_simulation_environment()
    
    if resume_checkpoint:
        # 从指定 checkpoint 恢复
        await simulator.run_simulation_from_checkpoint(resume_checkpoint)
    else:
        # 正常运行
        await simulator.run_simulation()

async def main_resume_latest(config: SimulationConfig):
    """
    从最新 checkpoint 恢复并继续运行
    """
    simulator = Simulator(config)
    await simulator.setup_simulation_environment()
    await simulator.run_simulation_from_checkpoint(None)  # None = 使用最新 checkpoint

if __name__ == "__main__":
    ray.init(num_cpus=128, num_gpus=2)
    args = parse_args()
    
    # Load configuration from YAML file
    config = SimulationConfig.from_yaml(args.config)
    
    logger.info(f"Simulation started with config: {args.config}")
    
    if args.resume_latest:
        logger.info("Resuming from latest checkpoint...")
        asyncio.run(main_resume_latest(config))
    elif args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        asyncio.run(main(config, resume_checkpoint=args.resume))
    else:
        asyncio.run(main(config))
    
    logger.info(f"Simulation ended")