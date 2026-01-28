"""
Simulation Initializer Module

This module provides a unified initialization framework for the agent-based economic simulation.

Initialization Order:
    1. EconomicCenter (核心账本系统)
    2. Markets (ProductMarket, LaborMarket)
    3. Government (税收和财政)
    4. Bank (储蓄和贷款)
    5. Firms (企业)
    6. Households (家庭)
    7. Cross-registration (相互注册)

Key Principles:
    - EconomicCenter only handles transfers (经济中心只处理转账)
    - ProductMarket manages inventory (商品市场管理库存)
    - All financial flows go through EconomicCenter
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import ray
import asyncio

from agenteconomy.center.Model import TaxPolicy, TaxBracket
from agenteconomy.utils.logger import get_logger

if TYPE_CHECKING:
    from agenteconomy.center.Ecocenter import EconomicCenter
    from agenteconomy.center.LaborMarket import LaborMarket
    from agenteconomy.center.ProductMarket import ProductMarket
    from agenteconomy.market.AbstractResourceMarket import AbstractResourceMarket
    from agenteconomy.agent.firm import Firm
    from agenteconomy.agent.household import Household
    from agenteconomy.agent.government import Government
    from agenteconomy.agent.bank import Bank

logger = get_logger(name="initializer")


# =============================================================================
# Configuration Data Classes
# =============================================================================

@dataclass
class EconomyConfig:
    """Economic system configuration"""
    # Tax rates
    vat_rate: float = 0.07
    corporate_tax_rate: float = 0.21
    income_tax_brackets: List[TaxBracket] = field(default_factory=lambda: [
        TaxBracket(cutoff=0, rate=0.10),
        TaxBracket(cutoff=10000, rate=0.12),
        TaxBracket(cutoff=40000, rate=0.22),
        TaxBracket(cutoff=85000, rate=0.24),
        TaxBracket(cutoff=165000, rate=0.32),
        TaxBracket(cutoff=215000, rate=0.35),
        TaxBracket(cutoff=540000, rate=0.37),
    ])
    
    # Initial allocations
    government_budget: float = 10_000_000.0
    bank_capital: float = 1_000_000.0
    household_initial_cash: float = 5_000.0
    firm_initial_cash: float = 100_000.0
    firm_initial_capital: float = 500_000.0


@dataclass
class SimulationContext:
    """Holds all initialized components for the simulation"""
    # Core systems
    economic_center: Optional['EconomicCenter'] = None
    labor_market: Optional['LaborMarket'] = None
    product_market: Optional['ProductMarket'] = None
    abstract_resource_market: Optional['AbstractResourceMarket'] = None
    
    # Agents
    government: Optional['Government'] = None
    bank: Optional['Bank'] = None
    firms: List['Firm'] = field(default_factory=list)
    households: List['Household'] = field(default_factory=list)
    
    # Mappings for quick lookup
    firm_by_id: Dict[str, 'Firm'] = field(default_factory=dict)
    household_by_id: Dict[str, 'Household'] = field(default_factory=dict)
    
    def get_all_agent_ids(self) -> Dict[str, List[str]]:
        """Get all agent IDs by type"""
        return {
            "firms": [f.firm_id for f in self.firms],
            "households": [h.household_id for h in self.households],
            "government": [self.government.government_id] if self.government else [],
            "bank": [self.bank.bank_id] if self.bank else [],
        }


# =============================================================================
# Initialization Steps
# =============================================================================

class SimulationInitializer:
    """
    Unified initialization framework for the simulation
    
    Usage:
        initializer = SimulationInitializer(config)
        context = await initializer.initialize()
    """
    
    def __init__(self, config: EconomyConfig = None):
        self.config = config or EconomyConfig()
        self.context = SimulationContext()
        
    async def initialize(self) -> SimulationContext:
        """
        Execute full initialization sequence
        
        Returns:
            SimulationContext with all components initialized
        """
        logger.info("=" * 60)
        logger.info("Starting Simulation Initialization")
        logger.info("=" * 60)
        
        # Step 1: Initialize core systems
        await self._init_economic_center()
        await self._init_markets()
        
        # Step 2: Initialize institutional agents
        await self._init_government()
        await self._init_bank()
        
        # Step 3: Initialize economic agents
        await self._init_firms()
        await self._init_households()
        
        # Step 4: Cross-registration and final setup
        await self._register_all_agents()
        await self._allocate_initial_resources()
        
        logger.info("=" * 60)
        logger.info("Simulation Initialization Complete")
        logger.info(f"  Firms: {len(self.context.firms)}")
        logger.info(f"  Households: {len(self.context.households)}")
        logger.info("=" * 60)
        
        return self.context
    
    # -------------------------------------------------------------------------
    # Step 1: Core Systems
    # -------------------------------------------------------------------------
    
    async def _init_economic_center(self):
        """Initialize EconomicCenter (核心账本系统)"""
        logger.info("[Step 1.1] Initializing EconomicCenter...")
        
        from agenteconomy.center.Ecocenter import EconomicCenter
        
        tax_policy = TaxPolicy(
            income_tax_rate=self.config.income_tax_brackets,
            corporate_tax_rate=self.config.corporate_tax_rate,
            vat_rate=self.config.vat_rate,
        )
        
        self.context.economic_center = EconomicCenter.remote(tax_policy=tax_policy)
        logger.info("  ✓ EconomicCenter initialized (Ray Actor)")
    
    async def _init_markets(self):
        """Initialize all markets"""
        logger.info("[Step 1.2] Initializing Markets...")
        
        from agenteconomy.center.ProductMarket import ProductMarket
        from agenteconomy.center.LaborMarket import LaborMarket
        from agenteconomy.market.AbstractResourceMarket import AbstractResourceMarket
        
        # ProductMarket - manages inventory
        self.context.product_market = ProductMarket.remote()
        logger.info("  ✓ ProductMarket initialized")
        
        # LaborMarket - needs economic_center for wage transfers
        self.context.labor_market = LaborMarket.remote(
            economic_center=self.context.economic_center
        )
        ray.get(self.context.economic_center.set_labor_market.remote(self.context.labor_market))
        logger.info("  ✓ LaborMarket initialized")
        
        # AbstractResourceMarket - will be set up after government
        logger.info("  ⏳ AbstractResourceMarket pending (needs government_id)")
    
    # -------------------------------------------------------------------------
    # Step 2: Institutional Agents
    # -------------------------------------------------------------------------
    
    async def _init_government(self):
        """Initialize Government agent"""
        logger.info("[Step 2.1] Initializing Government...")
        
        from agenteconomy.agent.government import Government
        
        tax_policy = TaxPolicy(
            income_tax_rate=self.config.income_tax_brackets,
            corporate_tax_rate=self.config.corporate_tax_rate,
            vat_rate=self.config.vat_rate,
        )
        
        self.context.government = Government(
            government_id="gov_main",
            initial_budget=self.config.government_budget,
            tax_policy=tax_policy,
            economic_center=self.context.economic_center,
        )
        self.context.government.initialize()
        logger.info(f"  ✓ Government initialized (budget: ${self.config.government_budget:,.0f})")
        
        # Now initialize AbstractResourceMarket with government_id
        from agenteconomy.market.AbstractResourceMarket import AbstractResourceMarket
        self.context.abstract_resource_market = AbstractResourceMarket(
            economic_center=self.context.economic_center,
            government_id=self.context.government.government_id,
        )
        logger.info("  ✓ AbstractResourceMarket initialized")
    
    async def _init_bank(self):
        """Initialize Bank agent"""
        logger.info("[Step 2.2] Initializing Bank...")
        
        from agenteconomy.agent.bank import Bank
        
        self.context.bank = Bank(
            bank_id="bank_central",
            initial_capital=self.config.bank_capital,
            economic_center=self.context.economic_center,
        )
        await self.context.bank.initialize()
        logger.info(f"  ✓ Bank initialized (capital: ${self.config.bank_capital:,.0f})")
    
    # -------------------------------------------------------------------------
    # Step 3: Economic Agents
    # -------------------------------------------------------------------------
    
    async def _init_firms(self):
        """Initialize all firms"""
        logger.info("[Step 3.1] Initializing Firms...")
        
        from agenteconomy.simulation.agent_loader import create_firms
        
        self.context.firms = create_firms(
            economic_center=self.context.economic_center,
            labor_market=self.context.labor_market,
            product_market=self.context.product_market,
            abstract_resource_market=self.context.abstract_resource_market,
        )
        
        # Build lookup dict
        for firm in self.context.firms:
            self.context.firm_by_id[firm.firm_id] = firm
        
        # Count by type
        type_counts = {}
        for firm in self.context.firms:
            t = firm.industry_type or "unknown"
            type_counts[t] = type_counts.get(t, 0) + 1
        
        logger.info(f"  ✓ Created {len(self.context.firms)} firms:")
        for t, count in sorted(type_counts.items()):
            logger.info(f"    - {t}: {count}")
    
    async def _init_households(self, num_households: int = 100):
        """Initialize households"""
        logger.info(f"[Step 3.2] Initializing {num_households} Households...")
        
        from agenteconomy.agent.household import Household
        
        for i in range(num_households):
            household = Household(
                household_id=f"hh_{i:04d}",
                name=f"Household {i}",
                description=f"Household {i}",
                owner=f"owner_{i}",
                economic_center=self.context.economic_center,
                labor_market=self.context.labor_market,
                product_market=self.context.product_market,
            )
            self.context.households.append(household)
            self.context.household_by_id[household.household_id] = household
        
        logger.info(f"  ✓ Created {len(self.context.households)} households")
    
    # -------------------------------------------------------------------------
    # Step 4: Registration and Resource Allocation
    # -------------------------------------------------------------------------
    
    async def _register_all_agents(self):
        """Register all agents in EconomicCenter"""
        logger.info("[Step 4.1] Registering all agents in EconomicCenter...")
        
        ec = self.context.economic_center
        
        # Register firms
        firm_ids = [f.firm_id for f in self.context.firms]
        if firm_ids:
            tasks = [ec.register_id.remote(fid, "firm") for fid in firm_ids]
            await asyncio.gather(*[asyncio.wrap_future(ray.get(t, asynchronous=True)) for t in tasks])
        logger.info(f"  ✓ Registered {len(firm_ids)} firms")
        
        # Register households
        hh_ids = [h.household_id for h in self.context.households]
        if hh_ids:
            tasks = [ec.register_id.remote(hid, "household") for hid in hh_ids]
            await asyncio.gather(*[asyncio.wrap_future(ray.get(t, asynchronous=True)) for t in tasks])
        logger.info(f"  ✓ Registered {len(hh_ids)} households")
    
    async def _allocate_initial_resources(self):
        """Allocate initial cash and capital to all agents"""
        logger.info("[Step 4.2] Allocating initial resources...")
        
        ec = self.context.economic_center
        
        # Allocate to firms
        firm_allocations = {}
        for firm in self.context.firms:
            firm_allocations[firm.firm_id] = {
                "capital_stock": self.config.firm_initial_capital,
                "cash": self.config.firm_initial_cash,
            }
        
        if firm_allocations:
            result = ray.get(ec.register_firm_assets.remote(firm_allocations))
            logger.info(f"  ✓ Allocated firm assets: {result}")
        
        # Allocate to households
        for household in self.context.households:
            ray.get(ec.init_agent_ledger.remote(
                household.household_id,
                self.config.household_initial_cash
            ))
        logger.info(f"  ✓ Allocated ${self.config.household_initial_cash:,.0f} to each household")


# =============================================================================
# Convenience Functions
# =============================================================================

async def quick_init(
    num_households: int = 100,
    config: EconomyConfig = None,
) -> SimulationContext:
    """
    Quick initialization for testing
    
    Args:
        num_households: Number of households to create
        config: Optional economy configuration
    
    Returns:
        SimulationContext with all components
    """
    initializer = SimulationInitializer(config)
    
    # Override household count
    original_init_households = initializer._init_households
    async def custom_init_households():
        await original_init_households(num_households)
    initializer._init_households = custom_init_households
    
    return await initializer.initialize()


def init_ray_if_needed():
    """Initialize Ray if not already running"""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
        logger.info("Ray initialized")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def main():
        init_ray_if_needed()
        
        config = EconomyConfig(
            government_budget=10_000_000.0,
            bank_capital=1_000_000.0,
            household_initial_cash=5_000.0,
            firm_initial_cash=100_000.0,
            firm_initial_capital=500_000.0,
        )
        
        context = await quick_init(num_households=50, config=config)
        
        # Print summary
        print("\n" + "=" * 60)
        print("Initialization Summary")
        print("=" * 60)
        print(f"Economic Center: {context.economic_center}")
        print(f"Labor Market: {context.labor_market}")
        print(f"Product Market: {context.product_market}")
        print(f"Government: {context.government.government_id if context.government else None}")
        print(f"Bank: {context.bank.bank_id if context.bank else None}")
        print(f"Firms: {len(context.firms)}")
        print(f"Households: {len(context.households)}")
        
        # Test query
        if context.economic_center:
            all_ids = ray.get(context.economic_center.get_all_agent_ids.remote())
            print(f"\nRegistered in EconomicCenter:")
            for agent_type, ids in all_ids.items():
                print(f"  {agent_type}: {len(ids)}")
    
    asyncio.run(main())
