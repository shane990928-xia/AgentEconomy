"""
抽象资源市场 - 管理电力、运输等同质化资源的供需和价格

核心功能：
1. 维护各类抽象资源的基准价格和当期价格
2. 处理基于"单位"的交易（度、吨公里等）
3. 根据供需动态调整价格
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ResourceUnit(Enum):
    """资源单位类型"""
    # 能源
    ELECTRICITY_KWH = "度"  # 电力：千瓦时
    GAS_CUBIC_METER = "立方米"  # 天然气
    
    # 运输
    TON_KILOMETER = "吨公里"  # 运输距离
    
    # 金融服务
    DOLLAR_SERVICE = "美元服务额"  # 金融、保险等按服务金额
    
    # 其他服务
    LABOR_HOUR = "人工时"  # 劳务
    SERVICE_UNIT = "服务单位"  # 通用服务单位


@dataclass
class ResourceInfo:
    """抽象资源信息"""
    industry_code: str  # IO表行业代码
    name: str  # 资源名称
    unit: ResourceUnit  # 单位类型
    
    # 价格
    base_price: float  # 基准价格（美元/单位），用于技术计算
    current_price: float  # 当期市场价格（美元/单位）
    
    # 供需
    supply_capacity: float  # 供给能力（单位/期）
    total_demand: float  # 当期总需求（单位/期）
    
    # 历史
    price_history: List[float]  # 价格历史


# 政府服务行业代码
GOVERNMENT_INDUSTRY_CODES = frozenset({"GFGD", "GFGN", "GFE", "GSLG", "GSLE"})


class AbstractResourceMarket:
    """
    抽象资源市场
    
    管理第三类行业（Category 3）提供的同质化资源/服务
    这些资源没有具体SKU，按单位交易
    
    资金流向规则：
    - 政府服务行业（GFGD, GFGN, GFE, GSLG, GSLE）→ Government Agent
    - 其他行业 → 对应的 ServiceFirm（通过 industry_code 查找）
    - 所有交易通过 EconomicCenter 记录，确保完整的审计轨迹
    """
    
    def __init__(self, economic_center=None, government_id: str = None):
        """
        初始化抽象资源市场
        
        Args:
            economic_center: EconomicCenter Ray Actor 引用，用于记录交易
            government_id: 政府 Agent 的 ID，用于接收政府服务费
        """
        self.resources: Dict[str, ResourceInfo] = {}
        self.transactions: List[Dict] = []  # 本地交易记录（备份）
        self.economic_center = economic_center  # EconomicCenter 引用
        self.government_id = government_id or "gov_main_simulation"  # 政府 ID
        
        # 行业代码 → 企业ID 映射（用于将资金路由到真实的 ServiceFirm）
        self.industry_to_firm: Dict[str, str] = {}
        
        # 行业代码到单位类型的映射
        self.industry_unit_mapping = {
            "22": ResourceUnit.ELECTRICITY_KWH,  # 电力
            "211": ResourceUnit.GAS_CUBIC_METER,  # 石油天然气
            "481": ResourceUnit.TON_KILOMETER,  # 航空运输
            "484": ResourceUnit.TON_KILOMETER,  # 卡车运输
            "485": ResourceUnit.SERVICE_UNIT,  # 地面客运
            "521CI": ResourceUnit.DOLLAR_SERVICE,  # 金融服务
            "523": ResourceUnit.DOLLAR_SERVICE,  # 证券投资
            "524": ResourceUnit.DOLLAR_SERVICE,  # 保险
            # 政府服务 - 按服务金额计费
            "GFGD": ResourceUnit.DOLLAR_SERVICE,  # 联邦政府（国防）
            "GFGN": ResourceUnit.DOLLAR_SERVICE,  # 联邦政府（非国防）
            "GFE": ResourceUnit.DOLLAR_SERVICE,  # 联邦政府企业
            "GSLG": ResourceUnit.DOLLAR_SERVICE,  # 州和地方政府
            "GSLE": ResourceUnit.DOLLAR_SERVICE,  # 州和地方政府企业
            # 更多映射...
        }
    
    def register_firm(self, industry_code: str, firm_id: str):
        """
        注册行业对应的企业
        
        ServiceFirm 创建后调用此方法，建立 industry_code → firm_id 映射
        这样当有人采购该行业的资源时，资金会流向真实的企业
        
        Args:
            industry_code: 行业代码（如 "HS", "621" 等）
            firm_id: 企业ID（如 "svc_HS_abc12345"）
        """
        self.industry_to_firm[industry_code] = firm_id
        logger.info(f"注册行业企业映射: {industry_code} → {firm_id}")
    
    def get_receiver_id(self, industry_code: str) -> str:
        """
        获取资源采购的资金接收方 ID
        
        优先级：
        1. 政府行业 → government_id
        2. 已注册的企业 → firm_id
        3. 兜底 → 虚拟市场账户 market_resource_{industry_code}
        
        Args:
            industry_code: 行业代码
            
        Returns:
            接收方的 agent_id
        """
        if industry_code in GOVERNMENT_INDUSTRY_CODES:
            return self.government_id
        
        if industry_code in self.industry_to_firm:
            return self.industry_to_firm[industry_code]
        
        # 兜底：虚拟市场账户（不应该走到这里，说明企业未注册）
        logger.warning(f"行业 {industry_code} 未注册企业，使用虚拟账户")
        return f"market_resource_{industry_code}"
    
    def initialize_resource(
        self,
        industry_code: str,
        name: str,
        base_price: float,
        initial_supply_capacity: float,
        firm_id: Optional[str] = None
    ):
        """
        初始化一个抽象资源
        
        Args:
            industry_code: IO表行业代码
            name: 资源名称
            base_price: 基准价格（用于技术计算）
            initial_supply_capacity: 初始供给能力
            firm_id: 提供该资源的企业ID（可选，用于资金路由）
        """
        unit = self.industry_unit_mapping.get(
            industry_code, 
            ResourceUnit.SERVICE_UNIT
        )
        
        self.resources[industry_code] = ResourceInfo(
            industry_code=industry_code,
            name=name,
            unit=unit,
            base_price=base_price,
            current_price=base_price,  # 初始价格等于基准价格
            supply_capacity=initial_supply_capacity,
            total_demand=0.0,
            price_history=[base_price]
        )
        
        # 如果提供了 firm_id，自动注册映射
        if firm_id:
            self.register_firm(industry_code, firm_id)
        
        logger.info(
            f"初始化资源 {name} ({industry_code}): "
            f"基准价={base_price}, 单位={unit.value}"
            f"{f', 企业={firm_id}' if firm_id else ''}"
        )
    
    def get_base_price(self, industry_code: str) -> float:
        """
        获取基准价格（用于技术计算）
        
        这个价格在模拟期间保持不变，用于：
        1. IO表技术系数转换为物理单位
        2. 保持技术关系的稳定性
        """
        if industry_code not in self.resources:
            raise ValueError(f"未找到资源: {industry_code}")
        
        return self.resources[industry_code].base_price
    
    def get_current_price(self, industry_code: str) -> float:
        """
        获取当期市场价格（用于实际交易）
        
        这个价格会根据供需动态调整
        """
        if industry_code not in self.resources:
            raise ValueError(f"未找到资源: {industry_code}")
        
        return self.resources[industry_code].current_price
    
    def calculate_physical_demand(
        self,
        industry_code: str,
        production_value: float,
        io_coefficient: float
    ) -> Tuple[float, str]:
        """
        从IO技术系数计算物理需求量
        
        这是核心转换逻辑！
        
        Args:
            industry_code: 资源行业代码
            production_value: 生产价值（美元）
            io_coefficient: IO表技术系数（投入/产出比）
        
        Returns:
            (物理需求量, 单位名称)
        
        示例:
            生产价值 = $1,974
            IO系数 = 0.0100 (每$1产出需要$0.01的电力)
            基准电价 = $0.12/度
            
            技术需求值 = $1,974 × 0.0100 = $19.74
            物理需求量 = $19.74 / $0.12 = 164.5度
            
            → 返回 (164.5, "度")
        """
        if industry_code not in self.resources:
            raise ValueError(f"未找到资源: {industry_code}")
        
        resource = self.resources[industry_code]
        
        # 第1步：计算技术需求值（价值）
        value_needed = production_value * io_coefficient
        
        # 第2步：转换为物理单位（使用基准价格）
        # 关键：这里用base_price，所以技术需求量是固定的
        physical_quantity = value_needed / resource.base_price
        
        return physical_quantity, resource.unit.value
    
    def purchase(
        self,
        industry_code: str,
        buyer_id: str,
        quantity: float,
        period: int
    ) -> Dict:
        """
        购买抽象资源
        
        资金流向规则（由 EconomicCenter 执行）：
        1. 政府服务行业 → Government Agent
        2. 其他行业 → 对应的 ServiceFirm
        3. 未注册的行业 → 虚拟市场账户（兜底）
        
        Args:
            industry_code: 资源行业代码
            buyer_id: 购买者ID
            quantity: 需求数量（物理单位）
            period: 当前期数
        
        Returns:
            交易详情 {
                "quantity": 购买数量,
                "unit": 单位,
                "unit_price": 单价,
                "total_cost": 总成本,
                "receiver_id": 资金接收方ID,
                "is_government_fee": bool  # 是否为政府服务费
            }
        """
        if industry_code not in self.resources:
            raise ValueError(f"未找到资源: {industry_code}")
        
        resource = self.resources[industry_code]
        
        # 使用当期市场价格计算成本
        unit_price = resource.current_price
        total_cost = quantity * unit_price
        
        # 累积需求（用于后续价格调整）
        resource.total_demand += quantity
        
        # 获取资金接收方（政府/企业/虚拟账户）
        receiver_id = self.get_receiver_id(industry_code)
        is_government_fee = industry_code in GOVERNMENT_INDUSTRY_CODES
        
        # 记录交易（本地备份）
        transaction = {
            "period": period,
            "resource": industry_code,
            "buyer": buyer_id,
            "receiver": receiver_id,
            "quantity": quantity,
            "unit": resource.unit.value,
            "unit_price": unit_price,
            "total_cost": total_cost,
            "base_price": resource.base_price,
            "is_government_fee": is_government_fee
        }
        self.transactions.append(transaction)
        
        # 通过 EconomicCenter 记录交易并执行转账
        # EconomicCenter 负责：1) 扣减买方余额 2) 增加卖方余额 3) 记录交易
        if self.economic_center is not None:
            try:
                import ray
                ray.get(self.economic_center.record_resource_purchase.remote(
                    month=period,
                    buyer_id=buyer_id,
                    industry_code=industry_code,
                    quantity=quantity,
                    unit_price=unit_price,
                    total_cost=total_cost,
                    unit=resource.unit.value,
                    base_price=resource.base_price,
                    receiver_id=receiver_id  # 资金流向真实的企业或政府
                ))
                logger.info(
                    f"资源采购: {buyer_id} → {receiver_id} ${total_cost:.2f} ({industry_code})"
                    f"{' [政府服务费]' if is_government_fee else ''}"
                )
            except Exception as e:
                logger.error(f"EconomicCenter 记录交易失败: {e}")
        
        logger.debug(
            f"{buyer_id} 购买 {resource.name}: "
            f"{quantity:.2f}{resource.unit.value} × ${unit_price:.4f} = ${total_cost:.2f}"
        )
        
        return transaction
    
    def adjust_prices(self, period: int):
        """
        根据供需调整所有资源的价格
        
        在每期结束时调用
        """
        for industry_code, resource in self.resources.items():
            # 计算供需比
            if resource.supply_capacity > 0:
                demand_supply_ratio = resource.total_demand / resource.supply_capacity
            else:
                demand_supply_ratio = 0
            
            old_price = resource.current_price
            
            # 价格调整逻辑（平滑调整）
            if demand_supply_ratio > 1.1:  # 供不应求
                adjustment = 1.05  # 涨价5%
            elif demand_supply_ratio > 1.02:
                adjustment = 1.02  # 涨价2%
            elif demand_supply_ratio < 0.85:  # 供过于求
                adjustment = 0.97  # 降价3%
            elif demand_supply_ratio < 0.95:
                adjustment = 0.99  # 降价1%
            else:
                adjustment = 1.0  # 维持
            
            # 更新价格（但不能偏离基准价太远，比如±50%）
            new_price = resource.current_price * adjustment
            min_price = resource.base_price * 0.5
            max_price = resource.base_price * 1.5
            new_price = max(min_price, min(max_price, new_price))
            
            resource.current_price = new_price
            resource.price_history.append(new_price)
            
            # 重置需求统计
            old_demand = resource.total_demand
            resource.total_demand = 0.0
            
            # 记录
            price_change_pct = (new_price - old_price) / old_price * 100
            logger.info(
                f"期{period} {resource.name}价格调整: "
                f"${old_price:.4f} → ${new_price:.4f} ({price_change_pct:+.2f}%), "
                f"需求/供给={demand_supply_ratio:.2f}, "
                f"总需求={old_demand:.2f}{resource.unit.value}"
            )
    
    def get_resource_info(self, industry_code: str) -> ResourceInfo:
        """获取资源信息"""
        if industry_code not in self.resources:
            raise ValueError(f"未找到资源: {industry_code}")
        return self.resources[industry_code]
    
    def get_all_resources(self) -> Dict[str, ResourceInfo]:
        """获取所有资源"""
        return self.resources.copy()
    
    def get_transactions_by_period(self, period: int) -> List[Dict]:
        """获取某期的所有交易"""
        return [t for t in self.transactions if t["period"] == period]
    
    def get_market_statistics(self, period: int) -> Dict:
        """
        获取市场统计信息
        """
        stats = {
            "period": period,
            "resources": {}
        }
        
        for industry_code, resource in self.resources.items():
            period_transactions = [
                t for t in self.transactions 
                if t["period"] == period and t["resource"] == industry_code
            ]
            
            total_volume = sum(t["quantity"] for t in period_transactions)
            total_value = sum(t["total_cost"] for t in period_transactions)
            
            stats["resources"][industry_code] = {
                "name": resource.name,
                "current_price": resource.current_price,
                "base_price": resource.base_price,
                "price_ratio": resource.current_price / resource.base_price,
                "total_demand": total_volume,
                "total_value": total_value,
                "supply_capacity": resource.supply_capacity,
                "utilization": total_volume / resource.supply_capacity if resource.supply_capacity > 0 else 0
            }
        
        return stats


if __name__ == "__main__":
    # 测试示例
    logging.basicConfig(level=logging.INFO)
    
    print("=== 抽象资源市场测试 ===\n")
    
    # 创建市场
    market = AbstractResourceMarket()
    
    # 初始化资源
    market.initialize_resource(
        industry_code="22",
        name="电力",
        base_price=0.12,  # $0.12/度
        initial_supply_capacity=1000000  # 100万度/期
    )
    
    market.initialize_resource(
        industry_code="484",
        name="卡车运输",
        base_price=0.50,  # $0.50/吨公里
        initial_supply_capacity=500000  # 50万吨公里/期
    )
    
    print("\n=== 场景：制造商生产T恤 ===")
    production_value = 1974  # 生产价值$1,974
    io_coeff_electricity = 0.0100  # 电力系数
    io_coeff_transport = 0.0087  # 运输系数
    
    # 计算物理需求
    electricity_qty, electricity_unit = market.calculate_physical_demand(
        "22", production_value, io_coeff_electricity
    )
    print(f"\n电力需求: {electricity_qty:.2f} {electricity_unit}")
    print(f"  (生产价值${production_value} × IO系数{io_coeff_electricity} ÷ 基准价${market.get_base_price('22')})")
    
    transport_qty, transport_unit = market.calculate_physical_demand(
        "484", production_value, io_coeff_transport
    )
    print(f"\n运输需求: {transport_qty:.2f} {transport_unit}")
    
    # 第1期：采购（使用初始价格）
    print("\n=== 第1期采购 ===")
    elec_trans = market.purchase("22", "manufacturer_001", electricity_qty, period=1)
    print(f"电力采购: {elec_trans['quantity']:.2f}{elec_trans['unit']} × ${elec_trans['unit_price']:.4f} = ${elec_trans['total_cost']:.2f}")
    
    trans_trans = market.purchase("484", "manufacturer_001", transport_qty, period=1)
    print(f"运输采购: {trans_trans['quantity']:.2f}{trans_trans['unit']} × ${trans_trans['unit_price']:.4f} = ${trans_trans['total_cost']:.2f}")
    
    # 模拟其他制造商也在采购，造成需求上升
    for i in range(10):
        market.purchase("22", f"other_manufacturer_{i}", 50000, period=1)
    
    # 价格调整
    print("\n=== 期末价格调整 ===")
    market.adjust_prices(period=1)
    
    # 第2期：采购（使用调整后的价格）
    print("\n=== 第2期采购（相同技术需求） ===")
    elec_trans_2 = market.purchase("22", "manufacturer_001", electricity_qty, period=2)
    print(f"电力采购: {elec_trans_2['quantity']:.2f}{elec_trans_2['unit']} × ${elec_trans_2['unit_price']:.4f} = ${elec_trans_2['total_cost']:.2f}")
    print(f"  需求量不变: {electricity_qty:.2f}{electricity_unit}（技术固定）")
    print(f"  成本变化: ${elec_trans['total_cost']:.2f} → ${elec_trans_2['total_cost']:.2f} (+{(elec_trans_2['total_cost']/elec_trans['total_cost']-1)*100:.1f}%)")
    
    # 统计
    print("\n=== 市场统计 ===")
    stats = market.get_market_statistics(period=2)
    for code, info in stats["resources"].items():
        print(f"\n{info['name']} ({code}):")
        print(f"  基准价: ${info['base_price']:.4f}")
        print(f"  当期价: ${info['current_price']:.4f} ({info['price_ratio']:.2%})")
        print(f"  需求: {info['total_demand']:.2f} ({info['utilization']:.2%}利用率)")


