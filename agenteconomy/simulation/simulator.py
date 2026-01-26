from dotenv import load_dotenv
load_dotenv()
from agenteconomy.utils.logger import get_logger
logger = get_logger(name="simulator")
from agenteconomy.center.Model import *
from config.config import SimulationConfig
from agenteconomy.center.Ecocenter import EconomicCenter
from agenteconomy.center.LaborMarket import LaborMarket
from agenteconomy.center.ProductMarket import ProductMarket
from agenteconomy.agent.firm import Firm
from agenteconomy.agent.household import Household
from agenteconomy.agent.government import Government
from agenteconomy.agent.bank import Bank
from agenteconomy.simulation.agent_loader import create_firms, create_households
from agenteconomy.market.AbstractResourceMarket import AbstractResourceMarket
from datetime import datetime
import asyncio

class Simulator:
    def __init__(self, config:SimulationConfig):
        """
        Initialize Simulator with configuration
        
        Args:
            config: SimulationConfig instance loaded from YAML
        """
        self.config: SimulationConfig = config
        
        # Basic entities and markets
        self.economic_center: Optional[EconomicCenter] = None
        self.labor_market: Optional[LaborMarket] = None
        self.product_market: Optional[ProductMarket] = None
        self.firms: Optional[List[Firm]] = None
        self.households: Optional[List[Household]] = None
        self.government: Optional[Government] = None
        self.bank: Optional[Bank] = None
        
        self.current_month = 1

        # Metrics
        
        logger.info(f"Simulator initialized with {self.config.num_months} months and {self.config.num_households} households")

    async def setup_simulation_environment(self):
        """Setup simulation environment"""
        logger.info("Setting up simulation environment...")
        
        try:
            if self.config.enable_progressive_tax_system:
                tax_policy = TaxPolicy(
                    income_tax_rate=self.config.gov_tax_brackets,
                    corporate_tax_rate=self.config.corporate_tax_rate,
                    vat_rate=self.config.vat_rate
                )
            # Initialize core components (pass in tax policy)
            self.economic_center = EconomicCenter.remote(
                tax_policy=tax_policy,
                category_profit_margins=self.config.category_profit_margins
            )
            self.product_market = ProductMarket.remote()
            # LaborMarket needs economic_center for wage transfers
            self.labor_market = LaborMarket.remote(economic_center=self.economic_center)
            
            self.government = Government(
                    government_id="gov_main_simulation",
                    initial_budget=10000000.0,
                    tax_policy=tax_policy,
                    economic_center=self.economic_center
                )

            self.government.initialize()
            
            # Initialize bank
            self.bank = Bank(
                bank_id="bank",
                initial_capital=1000000.0,
                economic_center=self.economic_center
            )
            self.bank.initialize()
            logger.info("Bank system initialized")
            
            # Load simulation data
            logger.info("Loading simulation data...")
            
            # Create households
            self._create_households()
            
            # Create firms
            self._create_firms()
            
            # Verify creation results
            if len(self.households) == 0:
                logger.error("No households created")
                return False
            
            if len(self.firms) == 0:
                logger.error("No firms created")
                return False
            
            return True

        except Exception as e:
            logger.error(f"Simulation environment setup failed: {e}")
            return False

    def _create_households(self):
        """Create households"""
        self.households = [Household(household_id=f"household_{i}", name=f"Household {i}", description=f"Household {i} description", owner=f"owner_{i}") for i in range(self.config.num_households)]

    def _create_firms(self):
        """Create firms"""
        # 初始化抽象资源市场
        # 传入 EconomicCenter 以记录所有交易
        # 传入政府 ID 以将政府服务费路由到政府账户
        abstract_resource_market = AbstractResourceMarket(
            economic_center=self.economic_center,
            government_id=self.government.government_id
        )
        self.firms = create_firms(
            economic_center=self.economic_center, 
            labor_market=self.labor_market, 
            product_market=self.product_market, 
            abstract_resource_market=abstract_resource_market
        )
        
    async def run_simulation(self):
        """Run simulation"""
        logger.info("Running simulation...")
        for month in range(1, self.config.num_months + 1):
            logger.info(f"Running month {month}...")
            await self._run_month(month)
            
    async def _run_month(self, month: int):
        """Run a single month"""
        logger.info(f"Running month {month}...")

        # step 1: household find jobs
        tasks = []
        for household in self.households:
            tasks.append(household.find_jobs())
        await asyncio.gather(*tasks)

        # step 2

        