"""
交易系统数据结构

定义账户、统计等核心数据结构
注意：Transaction 和 TransactionStatus 已移至 Model.py 以避免重复
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum


class EntityType(str, Enum):
    """经济主体类型"""
    FIRM = "firm"                # 企业
    HOUSEHOLD = "household"      # 家庭
    GOVERNMENT = "government"    # 政府
    BANK = "bank"                # 银行
    MARKET = "market"            # 市场（抽象资源市场等）


@dataclass
class Account:
    """
    账户
    
    每个经济主体在经济中心都有一个账户
    """
    entity_id: str                      # 主体ID
    entity_type: EntityType             # 主体类型
    balance: float                      # 当前余额
    created_at: int                     # 创建时间（期数）
    
    # 统计信息
    total_income: float = 0.0           # 累计收入
    total_expense: float = 0.0          # 累计支出
    transaction_count: int = 0          # 交易次数
    
    # 信用信息（用于贷款等）
    credit_limit: float = 0.0           # 信用额度
    debt: float = 0.0                   # 负债
    
    def __post_init__(self):
        """初始化后验证"""
        if self.balance < 0:
            raise ValueError(f"Initial balance cannot be negative: {self.balance}")
    
    def can_afford(self, amount: float) -> bool:
        """检查是否有足够余额"""
        return self.balance >= amount
    
    def debit(self, amount: float) -> bool:
        """扣款"""
        if not self.can_afford(amount):
            return False
        self.balance -= amount
        self.total_expense += amount
        self.transaction_count += 1
        return True
    
    def credit(self, amount: float):
        """入账"""
        self.balance += amount
        self.total_income += amount
        self.transaction_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'entity_id': self.entity_id,
            'entity_type': self.entity_type.value,
            'balance': self.balance,
            'total_income': self.total_income,
            'total_expense': self.total_expense,
            'transaction_count': self.transaction_count,
            'credit_limit': self.credit_limit,
            'debt': self.debt
        }


@dataclass
class PeriodStatistics:
    """
    期间统计
    
    记录每个期间的经济指标
    """
    period: int                         # 期数
    
    # 交易统计
    total_transactions: int = 0         # 总交易数
    total_volume: float = 0.0           # 总交易额
    
    # 按类型统计
    product_volume: float = 0.0         # 商品交易额
    wage_volume: float = 0.0            # 工资总额
    resource_volume: float = 0.0        # 资源交易额
    tax_volume: float = 0.0             # 税收总额
    
    # 经济指标
    gdp: float = 0.0                    # GDP
    total_consumption: float = 0.0      # 总消费
    total_investment: float = 0.0       # 总投资
    total_government_spending: float = 0.0  # 政府支出
    
    # 货币统计
    money_supply: float = 0.0           # 货币供应量
    velocity: float = 0.0               # 货币流通速度
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'period': self.period,
            'total_transactions': self.total_transactions,
            'total_volume': self.total_volume,
            'product_volume': self.product_volume,
            'wage_volume': self.wage_volume,
            'resource_volume': self.resource_volume,
            'tax_volume': self.tax_volume,
            'gdp': self.gdp,
            'total_consumption': self.total_consumption,
            'total_investment': self.total_investment,
            'total_government_spending': self.total_government_spending,
            'money_supply': self.money_supply,
            'velocity': self.velocity
        }


# 辅助函数

def generate_transaction_id(period: int, sequence: int) -> str:
    """生成交易ID"""
    return f"TXN_{period:04d}_{sequence:08d}"


def generate_account_id(entity_type: EntityType, entity_id: str) -> str:
    """生成账户ID"""
    return f"ACC_{entity_type.value}_{entity_id}"
