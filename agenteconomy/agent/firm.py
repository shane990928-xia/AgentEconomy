from typing import Optional, List, TYPE_CHECKING, Dict, Any
from functools import lru_cache
from pathlib import Path
import json
import ray

from agenteconomy.center.Model import LaborHour
from agenteconomy.center.LaborMarket import LaborMarket
from agenteconomy.center.ProductMarket import ProductMarket
from agenteconomy.market.AbstractResourceMarket import AbstractResourceMarket
from agenteconomy.market.IntermediateGoodsProcurement import IntermediateGoodsProcurement
from agenteconomy.utils.logger import get_logger
from agenteconomy.utils.load_io_table import get_suppliers_for_industry, get_cost_structure
from agenteconomy.data.industry_cate_map import industry_cate_map
from agenteconomy.llm.llm import *
from agenteconomy.llm.prompt_template import build_firm_post_job_prompt

# Avoid circular import by using TYPE_CHECKING
if TYPE_CHECKING:
    from agenteconomy.center.Ecocenter import EconomicCenter

logger = get_logger(name="Firm")

@lru_cache(maxsize=1)
def _load_retail_supply_chain_map() -> Dict[str, Dict[str, Any]]:
    data_path = Path(__file__).resolve().parents[1] / "data" / "retailer_mfg.json"
    try:
        with data_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("retail_supply_chain_map", {})
    except FileNotFoundError:
        logger.warning("Retail supply chain map not found at %s", data_path)
        return {}
    except json.JSONDecodeError as exc:
        logger.warning("Retail supply chain map parse error: %s", exc)
        return {}

def _get_retail_supply_chain_entry(industry: Optional[str]) -> Dict[str, Any]:
    if not industry:
        return {}
    return _load_retail_supply_chain_map().get(industry, {})


class Firm:
    def __init__(self, 
                 firm_id: str, 
                 name: Optional[str] = None, 
                 description: Optional[str] = None, 
                 industry: Optional[str] = None,
                 industry_type: Optional[str] = None,
                 economic_center: Optional['EconomicCenter'] = None,
                 labor_market: Optional[LaborMarket] = None,
                 product_market: Optional[ProductMarket] = None,
                 abstract_resource_market: Optional[AbstractResourceMarket] = None,
                 is_agent: bool = False,
               ):
        # Firm info
        self.firm_id: str = firm_id
        self.name: str = name
        self.description: Optional[str] = description
        self.industry: Optional[str] = industry
        self.industry_type: Optional[str] = industry_type
        
        # Firm employees
        self.employee_count: int = 0 # Number of employees
        self.employee_list: List[LaborHour] = [] # List of employees

        # Firm financials
        self.capital_stock: float = 0.0 # Capital stock
        self.cash: float = 0.0 # Cash

        # Market index
        self.economic_center: Optional[EconomicCenter] = economic_center # Economic center
        self.labor_market: Optional[LaborMarket] = labor_market # Labor market
        self.product_market: Optional[ProductMarket] = product_market # Product market
        self.abstract_resource_market: Optional[AbstractResourceMarket] = abstract_resource_market # Abstract resource market

        # Is agent
        self.is_agent: bool = is_agent # Whether the firm is an agent
        
        # Production tracking
        self.current_period: int = 0  # Current simulation period
        self.production_history: List[Dict[str, Any]] = []  # Production history

    async def register(self):
        """Register the firm in the economic center"""
        await self.economic_center.register_firm.remote(self)

    def _call_economic_center(self, method_name: str, *args, **kwargs):
        if self.economic_center is None:
            return None
        method = getattr(self.economic_center, method_name, None)
        if method is None:
            return None
        if 'ActorHandle' in str(type(self.economic_center)):
            return ray.get(method.remote(*args, **kwargs))
        return method(*args, **kwargs)

    # Query info
    def query_info(self):
        """Query information about the firm"""
        return {
            "firm_id": self.firm_id,
            "name": self.name,
            "description": self.description,
            "industry": self.industry,
            "capital_stock": self.capital_stock,
            "cash": self.cash,
        }

    def query_employees(self):
        """Query the employees of the firm"""
        return self.employee_count, self.employee_list

    # Labor market operations
    async def post_jobs(self):
        """Post jobs to the labor market"""
        # 根据企业自身的情况 用llm决策是否要发布岗位 发布什么岗位
        prompt = build_firm_post_job_prompt(self)
        response = await call_llm(prompt)
        return response

    def evaluate_candidates(self):
        """Evaluate candidates for the job"""
        pass

    def hire_employee(self):
        """Hire an employee"""
        pass

    def fire_employee(self):
        """Fire an employee"""
        pass
    
    def add_employee(self, employee: LaborHour):
        """Add an employee to the firm"""
        self.employee_list.append(employee)
        self.employee_count += 1

    def remove_employee(self, employee: LaborHour):
        """Remove an employee from the firm"""
        self.employee_list.remove(employee)
        self.employee_count -= 1

    # Product market operations
    def publish_product(self):
        """Publish a product to the product market"""
        pass

    def production_plan(self):
        """Generate a production plan"""
        pass


class ManufactureFirm(Firm):
    """
    制造业企业
    
    负责生产具体的SKU产品，消耗中间品和抽象资源
    
    注意：税和劳动报酬不从IO表计算，而是由企业实际运营决定
    """
    # 子类默认值
    DEFAULT_INDUSTRY_TYPE = "manufacture"
    DEFAULT_DESCRIPTION = 'A producer of physical goods that consumes third-category resources and outputs concrete SKU inventories, with full pricing authority.'
    
    def __init__(self, firm_id: str, **kwargs):
        kwargs.setdefault('industry_type', self.DEFAULT_INDUSTRY_TYPE)
        kwargs.setdefault('description', self.DEFAULT_DESCRIPTION)
        super().__init__(firm_id=firm_id, **kwargs) 

        # Production specific attributes
        self.production_costs: Dict[str, float] = {}  # Cost breakdown by input industry
        self.unit_costs: Dict[str, float] = {}  # Unit cost by SKU
        
        # Initialize procurement tool (lazy initialization)
        self._procurement = None
    
    @property
    def procurement(self) -> IntermediateGoodsProcurement:
        """Lazy initialization of procurement tool"""
        if self._procurement is None and self.product_market is not None:
            self._procurement = IntermediateGoodsProcurement(self.product_market)
        return self._procurement
    
    def calculate_production_value(self, production_plan: Dict[str, int], sku_base_prices: Dict[str, float]) -> float:
        """
        计算生产价值（用于IO表技术系数计算）
        
        Args:
            production_plan: {sku_id: quantity}
            sku_base_prices: {sku_id: base_manufacturer_price}
        
        Returns:
            总生产价值（美元）
        """
        total_value = 0.0
        for sku_id, quantity in production_plan.items():
            base_price = sku_base_prices.get(sku_id, 0)
            total_value += base_price * quantity
        
        return total_value
    
    def get_io_suppliers(self, threshold: float = 0.001) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取IO表供应商列表，并分类为中间品和抽象资源
        
        Args:
            threshold: IO系数阈值，忽略小于此值的供应商
        
        Returns:
            {
                'intermediate_goods': [...],  # Category 1 制造业
                'abstract_resources': [...],  # Category 3 抽象资源
            }
        """
        if not self.industry:
            logger.warning(f"Firm {self.firm_id}: No industry specified")
            return {'intermediate_goods': [], 'abstract_resources': []}
        
        # 获取所有供应商
        all_suppliers = get_suppliers_for_industry(self.industry, threshold=threshold)
        
        # 获取Category 3代码集合
        category_3_codes = set()
        for subgroup_info in industry_cate_map["category_3_cost_drivers"]["subgroups"].values():
            category_3_codes.update(subgroup_info["industries"].keys())
        
        # 分类
        intermediate_goods = []
        abstract_resources = []
        
        for supplier in all_suppliers:
            if supplier['supplier'] in category_3_codes:
                abstract_resources.append(supplier)
            else:
                intermediate_goods.append(supplier)
        
        logger.info(
            f"Firm {self.firm_id} IO suppliers: "
            f"{len(intermediate_goods)} intermediate goods, "
            f"{len(abstract_resources)} abstract resources"
        )
        
        return {
            'intermediate_goods': intermediate_goods,
            'abstract_resources': abstract_resources,
        }
    
    def procure_intermediate_goods(
        self,
        production_value: float,
        suppliers: List[Dict[str, Any]],
        period: int,
        strategy: str = "random"
    ) -> Dict[str, Any]:
        """
        采购中间品（从其他制造商购买具体SKU）
        
        Args:
            production_value: 生产价值
            suppliers: 中间品供应商列表
            period: 当前期数
            strategy: 采购策略 ("random", "cheapest", "balanced")
        
        Returns:
            {
                'total_cost': float,
                'by_industry': {industry_code: cost},
                'items': [PurchaseItem, ...]
            }
        """
        if self.procurement is None:
            logger.warning(f"Firm {self.firm_id}: No product market available for procurement")
            return {'total_cost': 0.0, 'by_industry': {}, 'items': []}
        
        result = self.procurement.procure_intermediate_goods(
            manufacturer_id=self.firm_id,
            production_value=production_value,
            io_suppliers=suppliers,
            period=period
        )

        # 更新成本记录
        self.production_costs.update(result['by_industry'])

        if self.economic_center is not None and result.get('total_cost', 0.0) > 0:
            items_payload = [
                {
                    "sku_id": item.sku_id,
                    "quantity": item.quantity,
                    "unit_price": item.unit_price,
                    "total_cost": item.total_cost,
                }
                for item in result.get('items', [])
            ]
            self._call_economic_center(
                "record_intermediate_goods_purchase",
                month=period,
                buyer_id=self.firm_id,
                total_cost=result['total_cost'],
                costs_by_industry=result.get('by_industry', {}),
                items=items_payload,
            )
        
        logger.info(
            f"Firm {self.firm_id} procured intermediate goods: "
            f"${result['total_cost']:.2f} from {len(result['by_industry'])} industries"
        )
        
        return result
    
    def procure_abstract_resources(
        self,
        production_value: float,
        io_suppliers: List[Dict[str, Any]],
        period: int
    ) -> Dict[str, float]:
        """
        采购抽象资源（电力、运输等）
        
        Args:
            production_value: 生产价值
            io_suppliers: IO表供应商列表 [{supplier: code, coefficient: value, name: str}]
            period: 当前期数
        
        Returns:
            成本明细 {industry_code: cost}
        """
        if self.abstract_resource_market is None:
            logger.warning(f"Firm {self.firm_id}: No abstract resource market available")
            return {}
        
        costs = {}
        
        for supplier in io_suppliers:
            supplier_code = supplier['supplier']
            
            try:
                # 检查是否为抽象资源
                self.abstract_resource_market.get_resource_info(supplier_code)
                
                # 计算物理需求
                physical_qty, unit = self.abstract_resource_market.calculate_physical_demand(
                    industry_code=supplier_code,
                    production_value=production_value,
                    io_coefficient=supplier['coefficient']
                )
                
                # 采购
                transaction = self.abstract_resource_market.purchase(
                    industry_code=supplier_code,
                    buyer_id=self.firm_id,
                    quantity=physical_qty,
                    period=period
                )

                if transaction:
                    costs[supplier_code] = transaction['total_cost']
                    if self.economic_center is not None:
                        self._call_economic_center(
                            "record_resource_purchase",
                            month=period,
                            buyer_id=self.firm_id,
                            industry_code=supplier_code,
                            quantity=physical_qty,
                            unit_price=transaction.get('unit_price', 0.0),
                            total_cost=transaction.get('total_cost', 0.0),
                            unit=transaction.get('unit'),
                            base_price=transaction.get('base_price'),
                        )

                    supplier_name = supplier.get('name', supplier_code)
                    logger.info(
                        f"Firm {self.firm_id} purchased {supplier_name}: "
                        f"{physical_qty:.2f}{unit} × ${transaction['unit_price']:.4f} = ${transaction['total_cost']:.2f}"
                    )
                
            except ValueError:
                # 不是抽象资源，跳过（可能是中间品）
                continue
            except Exception as e:
                logger.error(f"Firm {self.firm_id} failed to procure {supplier_code}: {e}")
                continue
        
        # 更新成本记录
        self.production_costs.update(costs)
        
        logger.info(
            f"Firm {self.firm_id} procured abstract resources: "
            f"${sum(costs.values()):.2f} from {len(costs)} resources"
        )
        
        return costs
    
    def calculate_labor_cost(self) -> float:
        """
        计算劳动成本（基于实际雇佣的员工）
        
        从 LaborMarket 获取本企业的工资支出总额
        
        Returns:
            劳动成本（月度工资总额）
        """
        if self.labor_market is None:
            logger.warning(f"Firm {self.firm_id}: No labor market available for labor cost calculation")
            return 0.0
        
        try:
            # 调用 LaborMarket 获取本企业的劳动成本
            if 'ActorHandle' in str(type(self.labor_market)):
                labor_cost = ray.get(self.labor_market.get_firm_labor_cost.remote(self.firm_id))
            else:
                labor_cost = self.labor_market.get_firm_labor_cost(self.firm_id)
            
            logger.info(f"Firm {self.firm_id} labor cost: ${labor_cost:.2f}")
            return labor_cost
            
        except Exception as e:
            logger.error(f"Firm {self.firm_id} failed to calculate labor cost: {e}")
            return 0.0
    
    def calculate_tax_cost(self, revenue: float = 0.0, tax_rate: float = 0.0) -> float:
        """
        计算税收成本
        
        Args:
            revenue: 收入
            tax_rate: 税率
        
        Returns:
            税收成本
        """
        # TODO: 实现基于税收系统的计算
        # 目前返回0，等待税收系统集成
        return 0.0
    
    def calculate_total_cost(
        self,
        intermediate_goods_cost: float,
        abstract_resources_cost: float,
        labor_cost: float = 0.0,
        tax_cost: float = 0.0
    ) -> Dict[str, float]:
        """
        计算总生产成本
        
        注意：税和劳动报酬不从IO表计算，而是由企业实际运营决定
        
        Args:
            intermediate_goods_cost: 中间品成本
            abstract_resources_cost: 抽象资源成本
            labor_cost: 劳动成本（由企业雇佣决定，默认0）
            tax_cost: 税收成本（由税收系统决定，默认0）
        
        Returns:
            {
                'intermediate_goods': float,
                'abstract_resources': float,
                'labor': float,
                'taxes': float,
                'total_cost': float
            }
        """
        # 总成本 = 中间品 + 抽象资源 + 劳动 + 税收
        total_cost = (
            intermediate_goods_cost +
            abstract_resources_cost +
            labor_cost +
            tax_cost
        )
        
        logger.info(
            f"Firm {self.firm_id} total cost: ${total_cost:.2f} "
            f"(intermediate: ${intermediate_goods_cost:.2f}, "
            f"abstract: ${abstract_resources_cost:.2f}, "
            f"labor: ${labor_cost:.2f}, "
            f"taxes: ${tax_cost:.2f})"
        )
        
        return {
            'intermediate_goods': intermediate_goods_cost,
            'abstract_resources': abstract_resources_cost,
            'labor': labor_cost,
            'taxes': tax_cost,
            'total_cost': total_cost
        }
    
    def produce(
        self,
        production_plan: Dict[str, int],
        sku_base_prices: Dict[str, float],
        period: int,
        update_inventory: bool = True,
        labor_cost: float = None,
        tax_cost: float = None
    ) -> Dict[str, Any]:
        """
        完整的生产流程
        
        注意：税和劳动报酬不从IO表计算，而是由企业实际运营决定
        
        Args:
            production_plan: {sku_id: quantity}
            sku_base_prices: {sku_id: base_manufacturer_price}
            period: 当前期数
            update_inventory: 是否更新库存
            labor_cost: 劳动成本（None则自动计算，默认0）
            tax_cost: 税收成本（None则自动计算，默认0）
        
        Returns:
            {
                'production_value': float,
                'total_cost': float,
                'unit_costs': {sku_id: unit_cost},
                'cost_breakdown': {...},
                'success': bool
            }
        """
        logger.info(f"Firm {self.firm_id} starting production for period {period}")
        
        try:
            # 1. 计算生产价值
            production_value = self.calculate_production_value(production_plan, sku_base_prices)
            
            # 2. 获取IO表供应商
            io_info = self.get_io_suppliers(threshold=0.001)
            
            # 3. 采购中间品
            intermediate_result = self.procure_intermediate_goods(
                production_value=production_value,
                suppliers=io_info['intermediate_goods'],
                period=period,
                strategy="random"
            )
            
            # 4. 采购抽象资源
            abstract_result = self.procure_abstract_resources(
                production_value=production_value,
                io_suppliers=io_info['abstract_resources'],
                period=period
            )
            
            # 5. 计算劳动和税收成本（如果未提供）
            if labor_cost is None:
                labor_cost = self.calculate_labor_cost()
            
            if tax_cost is None:
                tax_cost = self.calculate_tax_cost()
            
            # 6. 计算总成本
            cost_breakdown = self.calculate_total_cost(
                intermediate_goods_cost=intermediate_result['total_cost'],
                abstract_resources_cost=sum(abstract_result.values()),
                labor_cost=labor_cost,
                tax_cost=tax_cost
            )
            
            # 7. 计算单位成本
            total_quantity = sum(production_plan.values())
            unit_costs = {}
            for sku_id, quantity in production_plan.items():
                unit_cost = (cost_breakdown['total_cost'] / total_quantity) if total_quantity > 0 else 0
                unit_costs[sku_id] = unit_cost
                self.unit_costs[sku_id] = unit_cost
            
            # 8. 更新库存（如果需要）
            if update_inventory and self.product_market is not None:
                for sku_id, quantity in production_plan.items():
                    ray.get(self.product_market.update_stock.remote(sku_id, quantity))
            
            # 9. 记录生产历史
            production_record = {
                'period': period,
                'production_plan': production_plan.copy(),
                'production_value': production_value,
                'total_cost': cost_breakdown['total_cost'],
                'unit_costs': unit_costs.copy(),
                'cost_breakdown': cost_breakdown.copy(),
                'intermediate_items': len(intermediate_result['items']),
                'abstract_resources': len(abstract_result)
            }
            self.production_history.append(production_record)
            
            logger.info(
                f"Firm {self.firm_id} production completed: "
                f"{total_quantity} units, ${cost_breakdown['total_cost']:.2f} total cost"
            )
            
            return {
                'production_value': production_value,
                'total_cost': cost_breakdown['total_cost'],
                'unit_costs': unit_costs,
                'cost_breakdown': cost_breakdown,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Firm {self.firm_id} production failed: {e}")
            return {
                'production_value': 0,
                'total_cost': 0,
                'unit_costs': {},
                'cost_breakdown': {},
                'success': False,
                'error': str(e)
            }


class RetailFirm(Firm):
    # 子类默认值
    DEFAULT_INDUSTRY_TYPE = "retail"
    DEFAULT_DESCRIPTION = 'A physical goods retail channel that procures products from first-category suppliers and sells them to households, earning a channel margin.'
    def __init__(self, firm_id: str, **kwargs):
        kwargs.setdefault('industry_type', self.DEFAULT_INDUSTRY_TYPE)
        kwargs.setdefault('description', self.DEFAULT_DESCRIPTION)
        super().__init__(firm_id=firm_id, **kwargs)
        self.supply_chain_entry = _get_retail_supply_chain_entry(self.industry)
        self.supplier_industries: List[str] = list(self.supply_chain_entry.get("suppliers", []))


class ServiceFirm(Firm):
    # 子类默认值
    DEFAULT_INDUSTRY_TYPE = "service"
    DEFAULT_DESCRIPTION = 'A virtual resource and service provider that does not produce SKUs, but supplies abstract units (currency per unit), and requires hired labor.'
    def __init__(self, firm_id: str, **kwargs):
        kwargs.setdefault('industry_type', self.DEFAULT_INDUSTRY_TYPE)
        kwargs.setdefault('description', self.DEFAULT_DESCRIPTION)
        super().__init__(firm_id=firm_id, **kwargs)


if __name__ == "__main__":
    sample = RetailFirm(firm_id="1", description="Retail Firm", industry="441", economic_center=None, labor_market=None, product_market=None, is_agent=False)
    print(sample.supplier_industries)
    print(sample.supply_chain_entry)
