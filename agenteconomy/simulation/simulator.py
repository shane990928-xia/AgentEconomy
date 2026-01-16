from dotenv import load_dotenv
load_dotenv()
from agenteconomy.utils.logger import get_logger
logger = get_logger(name="simulator")
from agenteconomy.center.Model import *
from agenteconomy.config.config import SimulationConfig
from agenteconomy.center.Ecocenter import EconomicCenter
from agenteconomy.center.LaborMarket import LaborMarket
from agenteconomy.center.ProductMarket import ProductMarket
from agenteconomy.agent.firm import Firm
from agenteconomy.agent.household import Household
from agenteconomy.agent.government import Government
from agenteconomy.agent.bank import Bank
from datetime import datetime
import asyncio

class Simulator:
    def __init__(self, config:SimulationConfig):
        self.config = config
        self.economic_center = None
        self.labor_market = None
        self.product_market = None
        self.firms = None
        self.households = None
        self.government = None
        self.bank = None
        
        self.current_month = 1
        logger.info(f"Simulator initialized with {self.config.num_months} months and {self.config.num_households} households")

    def initialize(self):
        self.economic_center = EconomicCenter.remote()
        self.labor_market = LaborMarket.remote()
        self.product_market = ProductMarket.remote()
        self.firms = [Firm(firm_id=f"firm_{i}", name=f"Firm {i}", description=f"Firm {i} description", owner=f"owner_{i}") for i in range(self.config.num_firms)]
        self.households = [Household(household_id=f"household_{i}", name=f"Household {i}", description=f"Household {i} description", owner=f"owner_{i}") for i in range(self.config.num_households)]
        self.government = Government.remote()
        self.bank = Bank.remote()
        logger.info(f"Simulator initialized")

    async def setup_simulation_environment(self):
        """设置仿真环境"""
        logger.info("开始设置仿真环境...")
        
        try:
            if self.config.enable_progressive_tax_system:
                # 初始化政府（从config创建TaxPolicy）
                tax_policy = TaxPolicy(
                    income_tax_rate=self.config.gov_tax_brackets,
                    corporate_tax_rate=self.config.corporate_tax_rate,
                    vat_rate=self.config.vat_rate
                )
            # 初始化核心组件（传入税率配置）
            self.economic_center = EconomicCenter.remote(
                tax_policy=tax_policy,
                category_profit_margins=self.config.category_profit_margins
            )
            self.product_market = ProductMarket.remote()
            self.labor_market = LaborMarket.remote()
            
            self.government = Government.remote(
                    government_id="gov_main_simulation",
                    initial_budget=10000000.0,
                    tax_policy=tax_policy,
                    economic_center=self.economic_center
                )

            await self.government.initialize.remote()
            
            # 初始化银行
            self.bank = Bank.remote(
                bank_id="bank_main_simulation",
                initial_capital=1000000.0,
                economic_center=self.economic_center
            )
            await self.bank.initialize.remote()
            logger.info("银行系统初始化完成")
            
            # 加载数据
            logger.info("加载仿真数据...")
            
            # 设置全局LLM并发限制（在创建家庭之前）
            from agentsociety_ecosim.consumer_modeling.consumer_decision import BudgetAllocator
            BudgetAllocator.set_global_llm_limit(self.config.max_llm_concurrent)
            
            # 创建家庭
            await self._create_households()
            
            # 创建企业
            await self._create_firms()
            
            # 验证创建结果
            if len(self.households) == 0:
                logger.error("没有成功创建任何家庭")
                return False
            
            if len(self.firms) == 0:
                logger.error("没有成功创建任何企业")
                return False
            
            return True

        except Exception as e:
            logger.error(f"仿真环境设置失败: {e}")
            return False