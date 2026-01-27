from typing import Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agenteconomy.center.Ecocenter import EconomicCenter
    from agenteconomy.center.LaborMarket import LaborMarket
    from agenteconomy.center.ProductMarket import ProductMarket


class Household:
    """
    Household Agent
    
    Represents a household unit in the economy that:
    - Supplies labor to firms
    - Consumes goods from the product market
    - Pays taxes and receives transfers
    - Maintains savings
    """
    
    def __init__(
        self,
        household_id: str,
        name: str = None,
        description: str = None,
        owner: str = None,
        economic_center: Optional['EconomicCenter'] = None,
        labor_market: Optional['LaborMarket'] = None,
        product_market: Optional['ProductMarket'] = None,
    ):
        # Identity
        self.household_id = household_id
        self.name = name or f"Household_{household_id}"
        self.description = description
        self.owner = owner
        
        # Market references
        self.economic_center = economic_center
        self.labor_market = labor_market
        self.product_market = product_market
        
        # Employment status
        self.employed: bool = False
        self.employer_id: Optional[str] = None
        self.wage: float = 0.0
        self.occupation: Optional[str] = None
        
        # Consumption preferences
        self.consumption_budget: float = 0.0
        self.savings_rate: float = 0.2  # 20% savings rate
        
        # History tracking
        self.income_history: List[Dict[str, Any]] = []
        self.consumption_history: List[Dict[str, Any]] = []

    def query_info(self) -> Dict[str, Any]:
        """Query basic household information"""
        return {
            "household_id": self.household_id,
            "name": self.name,
            "employed": self.employed,
            "employer_id": self.employer_id,
            "wage": self.wage,
            "occupation": self.occupation,
        }

    async def find_jobs(self):
        """Search for jobs in the labor market"""
        # TODO: Implement job search logic with LLM
        pass

    async def consume(self, month: int, budget: float):
        """
        Consume goods from the product market
        
        Args:
            month: Current simulation month
            budget: Available budget for consumption
        """
        # TODO: Implement consumption logic with LLM decision
        pass

    def receive_wage(self, amount: float, month: int):
        """Record wage receipt"""
        self.income_history.append({
            "month": month,
            "amount": amount,
            "type": "wage",
            "employer_id": self.employer_id,
        })