from typing import Optional
from agenteconomy.center.Model import *
from agenteconomy.center.Ecocenter import EconomicCenter
from agenteconomy.utils.logger import get_logger
import ray


class Government:
    """
    # Government Agent
    Distributed government entity managing fiscal policy and taxation.
    
    ## Features
    - Tax policy management
    - LLM-assisted policy updates
    - Budget tracking via EconomicCenter
    """
    
    def __init__(self,
                 government_id: str,
                 initial_budget: float = 0.0,
                 tax_policy: TaxPolicy = None,
                 economic_center: Optional[EconomicCenter] = None):
        """
        ## Initialize Government Agent
        Creates a new government agent with full tax management capabilities.
        
        ### Parameters
        - `government_id` (str): Unique identifier for the government
        - `initial_budget` (float): Starting budget allocation
        - `tax_policy` (TaxPolicy): Initial tax policy configuration
        - `llm` (LLM): Language model for policy recommendations
        - `economic_center` (EconomicCenter): Economic state manager
        
        ### Raises
        - ValueError: If government_id is empty or invalid
        """
        # Validate government ID
        if not government_id or not isinstance(government_id, str):
            raise ValueError("government_id must be a non-empty string")
        
        # Use default policy if none provided
        if tax_policy is None:
            tax_policy = TaxPolicy()
        
        # Store core state directly
        self.government_id = government_id
        self.tax_policy = tax_policy.model_copy()
        self.initial_budget = initial_budget
        # Store dependencies
        self.economic_center = economic_center
        self.logger = get_logger(name="government")

    def initialize(self):
        """
        ## Initialize Government Agent
        Asynchronously initializes the government agent.
        """
        if self.economic_center:
            try:
                ray.get([self.economic_center.init_agent_ledger.remote(self.government_id, self.initial_budget),
                                     self.economic_center.register_id.remote(self.government_id, 'government')]
                )
                self.logger.info(f"Government {self.government_id} registered in EconomicCenter")
            except Exception as e:
                self.logger.warning(f"[Government Init] Failed to register ledger for {self.government_id}: {e}")

    async def update_tax_policy(self, new_policy: TaxPolicy) -> None: 
        """
        ## Update Tax Policy
        Applies new tax policy while ensuring validation and consistency.
        
        ### Parameters
        - `new_policy` (TaxPolicy): Validated tax policy to apply
        
        ### Validation
        - Ensures non-null input
        - Performs deep copy to prevent external state modification
        
        ### Raises
        - ValueError: If new_policy is None
        """
        # Validate new policy
        if not new_policy:
            raise ValueError("new_policy cannot be None")
        
        # Update internal state
        self.tax_policy = new_policy.model_copy()
