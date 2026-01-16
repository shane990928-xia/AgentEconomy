from dataclasses import dataclass
from agenteconomy.utils.logger import get_logger
logger = get_logger(name="simulation_config")
@dataclass
class SimulationConfig:
    num_months: int = 12
    num_households: int = 100
    