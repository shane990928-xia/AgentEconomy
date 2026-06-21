"""
Auditable pricing policy for macro simulation markets.

The policy combines independent price pressures in log space so each driver can
be inspected without hiding the total adjustment inside one multiplier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Optional


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _positive(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric > 0 else None


def _log_ratio(numerator: Optional[float], denominator: float) -> float:
    numerator = _positive(numerator)
    if numerator is None or denominator <= 0:
        return 0.0
    return math.log(numerator / denominator)


@dataclass(frozen=True)
class PricingPolicyInput:
    """Inputs for one price update."""

    current_price: float
    unit_cost: Optional[float] = None
    inventory_ratio: Optional[float] = 1.0
    demand_supply_ratio: Optional[float] = 1.0
    competitor_price: Optional[float] = None
    industry_benchmark: Optional[float] = None
    markup: float = 0.0
    stickiness: float = 0.0
    max_change: Optional[float] = None
    min_price: float = 0.01

    cost_weight: float = 1.0
    inventory_sensitivity: float = 0.05
    demand_sensitivity: float = 0.05
    benchmark_weight: float = 0.0
    mean_reversion_target: Optional[float] = None
    mean_reversion_strength: float = 0.0


@dataclass(frozen=True)
class PricingPolicyResult:
    """Result of one price update plus an auditable component breakdown."""

    new_price: float
    components: Dict[str, Any] = field(default_factory=dict)


class PricingPolicy:
    """
    Compute a price update from cost, inventory, demand/supply, benchmark, and
    stickiness drivers.

    Ratio conventions:
    - inventory_ratio > 1 means inventory is high and lowers price pressure.
    - demand_supply_ratio > 1 means demand exceeds supply and raises pressure.
    - stickiness is in [0, 1], where 1 means no movement this period.
    - max_change is a one-period relative cap, e.g. 0.05 for +/-5%.
    """

    def apply(self, params: PricingPolicyInput) -> PricingPolicyResult:
        min_price = max(float(params.min_price or 0.01), 0.000001)
        current_price = max(float(params.current_price or 0.0), min_price)
        markup = max(float(params.markup or 0.0), -0.95)

        components: Dict[str, Any] = {
            "current_price": current_price,
            "min_price": min_price,
            "markup": markup,
        }

        unit_cost = _positive(params.unit_cost)
        cost_target = unit_cost * (1.0 + markup) if unit_cost is not None else None
        cost_weight = max(float(params.cost_weight or 0.0), 0.0)
        cost_pressure = cost_weight * _log_ratio(cost_target, current_price)
        components.update(
            {
                "unit_cost": unit_cost,
                "cost_target_price": cost_target,
                "cost_weight": cost_weight,
                "cost_pressure": cost_pressure,
                "cost_factor": math.exp(cost_pressure),
            }
        )

        inventory_ratio = _positive(params.inventory_ratio) or 1.0
        inventory_sensitivity = max(float(params.inventory_sensitivity or 0.0), 0.0)
        inventory_pressure = -inventory_sensitivity * math.log(inventory_ratio)
        components.update(
            {
                "inventory_ratio": inventory_ratio,
                "inventory_sensitivity": inventory_sensitivity,
                "inventory_pressure": inventory_pressure,
                "inventory_factor": math.exp(inventory_pressure),
            }
        )

        demand_supply_ratio = _positive(params.demand_supply_ratio) or 1.0
        demand_sensitivity = max(float(params.demand_sensitivity or 0.0), 0.0)
        demand_pressure = demand_sensitivity * math.log(demand_supply_ratio)
        components.update(
            {
                "demand_supply_ratio": demand_supply_ratio,
                "demand_sensitivity": demand_sensitivity,
                "demand_pressure": demand_pressure,
                "demand_factor": math.exp(demand_pressure),
            }
        )

        competitor_price = _positive(params.competitor_price)
        industry_benchmark = _positive(params.industry_benchmark)
        benchmark_target = competitor_price if competitor_price is not None else industry_benchmark
        benchmark_source = (
            "competitor_price"
            if competitor_price is not None
            else "industry_benchmark"
            if industry_benchmark is not None
            else None
        )
        benchmark_weight = max(float(params.benchmark_weight or 0.0), 0.0)
        benchmark_pressure = benchmark_weight * _log_ratio(benchmark_target, current_price)
        components.update(
            {
                "competitor_price": competitor_price,
                "industry_benchmark": industry_benchmark,
                "benchmark_target": benchmark_target,
                "benchmark_source": benchmark_source,
                "benchmark_weight": benchmark_weight,
                "benchmark_pressure": benchmark_pressure,
                "benchmark_factor": math.exp(benchmark_pressure),
            }
        )

        mean_reversion_target = _positive(params.mean_reversion_target)
        mean_reversion_strength = max(float(params.mean_reversion_strength or 0.0), 0.0)
        mean_reversion_pressure = mean_reversion_strength * _log_ratio(
            mean_reversion_target, current_price
        )
        components.update(
            {
                "mean_reversion_target": mean_reversion_target,
                "mean_reversion_strength": mean_reversion_strength,
                "mean_reversion_pressure": mean_reversion_pressure,
                "mean_reversion_factor": math.exp(mean_reversion_pressure),
            }
        )

        total_pressure = (
            cost_pressure
            + inventory_pressure
            + demand_pressure
            + benchmark_pressure
            + mean_reversion_pressure
        )
        raw_target_price = current_price * math.exp(total_pressure)

        stickiness = _clamp(float(params.stickiness or 0.0), 0.0, 1.0)
        sticky_price = current_price * stickiness + raw_target_price * (1.0 - stickiness)

        max_change = params.max_change
        max_change_ratio = None if max_change is None else max(float(max_change), 0.0)
        limited_price = sticky_price
        limit_applied = False
        if max_change_ratio is not None:
            lower = current_price * (1.0 - max_change_ratio)
            upper = current_price * (1.0 + max_change_ratio)
            limited_price = _clamp(sticky_price, lower, upper)
            limit_applied = limited_price != sticky_price

        new_price = max(min_price, limited_price)
        components.update(
            {
                "total_pressure": total_pressure,
                "raw_target_price": raw_target_price,
                "stickiness": stickiness,
                "sticky_price": sticky_price,
                "max_change": max_change_ratio,
                "limit_applied": limit_applied,
                "limited_price": limited_price,
                "new_price": new_price,
                "change_ratio": (new_price - current_price) / current_price,
            }
        )

        return PricingPolicyResult(new_price=new_price, components=components)


def apply_pricing_policy(params: PricingPolicyInput) -> PricingPolicyResult:
    """Convenience wrapper for callers that do not need a policy instance."""

    return PricingPolicy().apply(params)
