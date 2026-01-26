from typing import Optional, Dict, List
from agenteconomy.center.Model import *
from agenteconomy.center.Ecocenter import EconomicCenter
from agenteconomy.utils.logger import get_logger
import ray


# 政府服务行业代码
GOVERNMENT_SERVICE_CODES = frozenset({"GFGD", "GFGN", "GFE", "GSLG", "GSLE"})


class Government:
    """
    # Government Agent
    Distributed government entity managing fiscal policy and taxation.
    
    ## Features
    - Tax policy management
    - LLM-assisted policy updates
    - Budget tracking via EconomicCenter
    - 政府服务费通过 EconomicCenter 记录和收取（非税收入）
    
    ## 政府服务费说明
    政府服务费（规费、牌照费、公立企业服务费等）不在 Government 内部维护，
    而是通过 EconomicCenter 的交易记录系统统一管理。
    当企业采购政府服务时，AbstractResourceMarket 会直接调用 EconomicCenter
    的 record_resource_purchase()，将资金转入政府账户。
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

    def get_service_fee_summary(self, period: Optional[int] = None) -> Dict:
        """
        从 EconomicCenter 查询政府服务费收入汇总
        
        Args:
            period: 指定期数，None 表示全部
            
        Returns:
            汇总信息 {
                "total": 总金额,
                "by_type": {行业代码: 金额},
                "count": 笔数
            }
        """
        if self.economic_center is None:
            return {"total": 0.0, "by_type": {}, "count": 0}
        
        try:
            # 从 EconomicCenter 查询交易记录
            # 政府服务费的 receiver_id 是 self.government_id，tx_type 是 "resource_purchase"
            transactions = ray.get(
                self.economic_center.get_transactions_by_receiver.remote(
                    receiver_id=self.government_id,
                    tx_type="resource_purchase",
                    month=period
                )
            )
            
            by_type: Dict[str, float] = {}
            total = 0.0
            
            for tx in transactions:
                industry_code = tx.get("metadata", {}).get("industry_code", "unknown")
                amount = tx.get("amount", 0.0)
                
                if industry_code in GOVERNMENT_SERVICE_CODES:
                    by_type[industry_code] = by_type.get(industry_code, 0.0) + amount
                    total += amount
            
            return {
                "total": total,
                "by_type": by_type,
                "count": len([t for t in transactions if t.get("metadata", {}).get("industry_code") in GOVERNMENT_SERVICE_CODES])
            }
        except Exception as e:
            self.logger.error(f"查询政府服务费失败: {e}")
            return {"total": 0.0, "by_type": {}, "count": 0}

    def get_balance(self) -> float:
        """
        从 EconomicCenter 查询政府当前余额
        
        Returns:
            政府账户余额
        """
        if self.economic_center is None:
            return self.initial_budget
        
        try:
            return ray.get(self.economic_center.query_balance.remote(self.government_id))
        except Exception as e:
            self.logger.error(f"查询政府余额失败: {e}")
            return 0.0

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
