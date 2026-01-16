from agenteconomy.agent.household import *
from agenteconomy.utils.logger import get_logger
from agenteconomy.center.LaborMarket import LaborMarket
logger = get_logger(name="main")
import ray
if __name__ == "__main__":
    ray.init(num_cpus=128, num_gpus=2)
    logger.info(f"Simulation started")
    labor_market = LaborMarket.remote()
    logger.info(f"Simulation ended")