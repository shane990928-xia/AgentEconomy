"""
Market module - 市场组件

包含:
- AbstractResourceMarket: 抽象资源市场（电力、运输等）
- ProductMarket: 产品市场（具体SKU）
"""

from agenteconomy.market.AbstractResourceMarket import (
    AbstractResourceMarket,
    ResourceUnit,
    ResourceInfo
)

__all__ = [
    "AbstractResourceMarket",
    "ResourceUnit",
    "ResourceInfo",
]


