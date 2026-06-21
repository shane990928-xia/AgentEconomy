from __future__ import annotations

from dataclasses import dataclass, field
from math import inf, isfinite
from typing import Any, Dict, Mapping, Optional, Sequence, Union


QuantityLike = Union[int, float, str, Mapping[str, Any]]


@dataclass(frozen=True)
class ProductionPlanInput:
    """Inputs for an aggregate firm production planning decision."""

    sales_history: Sequence[QuantityLike] = field(default_factory=list)
    unmet_demand_history: Sequence[QuantityLike] = field(default_factory=list)
    current_inventory: QuantityLike = 0.0
    target_inventory_months: float = 1.0
    ema_alpha: float = 0.5
    include_unmet_demand: bool = True
    fallback_expected_demand: float = 0.0
    available_labor: Optional[float] = None
    labor_productivity: float = 1.0
    capital_stock: Optional[float] = None
    capital_productivity: float = 1.0
    cash: Optional[float] = None
    unit_cash_cost: Optional[float] = None
    cash_reserve: float = 0.0
    approved_credit: Optional[float] = None
    credit_limit: Optional[float] = None
    credit_outstanding: float = 0.0


@dataclass(frozen=True)
class ProductionPlan:
    expected_demand: float
    target_inventory: float
    desired_output: float
    feasible_output: float
    current_inventory: float
    production_gap: float
    max_output_by_constraint: Dict[str, Optional[float]]
    diagnostics: Dict[str, Any]


class ProductionPlanningPolicy:
    """Aggregate production planner for firms.

    The policy separates unconstrained production intent from feasible output.
    `desired_output` is demand plus target inventory minus current inventory;
    `feasible_output` applies labor, capital, and optional cash constraints.
    """

    def build_plan(self, plan_input: ProductionPlanInput) -> ProductionPlan:
        alpha = self._validate_alpha(plan_input.ema_alpha)
        target_inventory_months = self._validate_nonnegative(
            plan_input.target_inventory_months,
            "target_inventory_months",
        )

        sales_history = self._coerce_series(plan_input.sales_history)
        unmet_history = self._coerce_series(plan_input.unmet_demand_history)
        demand_history = self._combine_histories(
            sales_history,
            unmet_history,
            include_unmet=plan_input.include_unmet_demand,
        )

        expected_demand = self._ema(
            demand_history,
            alpha=alpha,
            fallback=self._validate_nonnegative(
                plan_input.fallback_expected_demand,
                "fallback_expected_demand",
            ),
        )
        current_inventory = self._coerce_quantity(plan_input.current_inventory)
        target_inventory = expected_demand * target_inventory_months
        desired_output = max(0.0, expected_demand + target_inventory - current_inventory)

        constraints = {
            "labor": self._capacity_constraint(
                available=plan_input.available_labor,
                productivity=plan_input.labor_productivity,
                desired_output=desired_output,
            ),
            "capital": self._capacity_constraint(
                available=plan_input.capital_stock,
                productivity=plan_input.capital_productivity,
                desired_output=desired_output,
            ),
            "cash": self._cash_constraint(
                cash=plan_input.cash,
                unit_cash_cost=plan_input.unit_cash_cost,
                cash_reserve=plan_input.cash_reserve,
                approved_credit=plan_input.approved_credit,
                credit_limit=plan_input.credit_limit,
                credit_outstanding=plan_input.credit_outstanding,
                desired_output=desired_output,
            ),
        }

        enabled_limits = [
            c["max_output"]
            for c in constraints.values()
            if c.get("enabled") and isfinite(float(c.get("max_output", inf)))
        ]
        feasible_output = min([desired_output] + enabled_limits)
        feasible_output = max(0.0, feasible_output)

        limiting_constraints = []
        binding_constraints = []
        for name, constraint in constraints.items():
            max_output = constraint.get("max_output")
            limited = bool(
                constraint.get("enabled")
                and max_output is not None
                and float(max_output) < desired_output
            )
            binding = bool(limited and abs(float(max_output) - feasible_output) <= 1e-9)
            constraint["limited"] = limited
            constraint["binding"] = binding
            if limited:
                limiting_constraints.append(name)
            if binding:
                binding_constraints.append(name)

        max_output_by_constraint = {
            name: (
                float(constraint["max_output"])
                if constraint.get("enabled") and isfinite(float(constraint["max_output"]))
                else None
            )
            for name, constraint in constraints.items()
        }

        diagnostics = {
            "sales_history": sales_history,
            "unmet_demand_history": unmet_history,
            "demand_history": demand_history,
            "ema_alpha": alpha,
            "include_unmet_demand": bool(plan_input.include_unmet_demand),
            "unconstrained_desired_output": desired_output,
            "is_constrained": feasible_output < desired_output,
            "limiting_constraints": limiting_constraints,
            "binding_constraints": binding_constraints,
            "constraints": constraints,
        }

        return ProductionPlan(
            expected_demand=expected_demand,
            target_inventory=target_inventory,
            desired_output=desired_output,
            feasible_output=feasible_output,
            current_inventory=current_inventory,
            production_gap=max(0.0, desired_output - feasible_output),
            max_output_by_constraint=max_output_by_constraint,
            diagnostics=diagnostics,
        )

    @classmethod
    def _validate_alpha(cls, alpha: float) -> float:
        value = float(alpha)
        if value <= 0.0 or value > 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        return value

    @classmethod
    def _validate_nonnegative(cls, value: float, field_name: str) -> float:
        result = float(value or 0.0)
        if result < 0.0:
            raise ValueError(f"{field_name} must be non-negative")
        return result

    @classmethod
    def _coerce_quantity(cls, value: QuantityLike) -> float:
        if isinstance(value, Mapping):
            return sum(cls._coerce_quantity(v) for v in value.values())
        try:
            result = float(value or 0.0)
        except (TypeError, ValueError):
            result = 0.0
        return max(0.0, result)

    @classmethod
    def _coerce_series(cls, values: Optional[Sequence[QuantityLike]]) -> list[float]:
        if values is None:
            return []
        if isinstance(values, Mapping):
            return [cls._coerce_quantity(values)]
        return [cls._coerce_quantity(value) for value in values]

    @classmethod
    def _combine_histories(
        cls,
        sales_history: Sequence[float],
        unmet_history: Sequence[float],
        include_unmet: bool,
    ) -> list[float]:
        if not include_unmet:
            return list(sales_history)

        length = max(len(sales_history), len(unmet_history))
        if length == 0:
            return []

        sales = [0.0] * (length - len(sales_history)) + list(sales_history)
        unmet = [0.0] * (length - len(unmet_history)) + list(unmet_history)
        return [max(0.0, sale + missed) for sale, missed in zip(sales, unmet)]

    @classmethod
    def _ema(cls, values: Sequence[float], alpha: float, fallback: float) -> float:
        if not values:
            return fallback
        ema_value = float(values[0])
        for value in values[1:]:
            ema_value = alpha * float(value) + (1.0 - alpha) * ema_value
        return max(0.0, ema_value)

    @classmethod
    def _capacity_constraint(
        cls,
        available: Optional[float],
        productivity: float,
        desired_output: float,
    ) -> Dict[str, Any]:
        if available is None:
            return {
                "enabled": False,
                "available": None,
                "productivity": float(productivity or 0.0),
                "max_output": inf,
                "reason": "not_configured",
            }

        available_value = cls._validate_nonnegative(available, "constraint_available")
        productivity_value = float(productivity or 0.0)
        if productivity_value <= 0.0:
            max_output = 0.0
            reason = "non_positive_productivity"
        else:
            max_output = available_value * productivity_value
            reason = None

        return {
            "enabled": True,
            "available": available_value,
            "productivity": productivity_value,
            "max_output": max(0.0, max_output),
            "reason": reason,
            "desired_output": desired_output,
        }

    @classmethod
    def _cash_constraint(
        cls,
        cash: Optional[float],
        unit_cash_cost: Optional[float],
        cash_reserve: float,
        approved_credit: Optional[float],
        credit_limit: Optional[float],
        credit_outstanding: float,
        desired_output: float,
    ) -> Dict[str, Any]:
        reserve = cls._validate_nonnegative(cash_reserve, "cash_reserve")
        if cash is None:
            available_credit = cls._available_credit(
                approved_credit=approved_credit,
                credit_limit=credit_limit,
                credit_outstanding=credit_outstanding,
            )
            return {
                "enabled": False,
                "cash": None,
                "unit_cash_cost": unit_cash_cost,
                "cash_reserve": reserve,
                "spendable_cash": None,
                "approved_credit": cls._coerce_optional_nonnegative(approved_credit),
                "credit_limit": cls._coerce_optional_nonnegative(credit_limit),
                "credit_outstanding": cls._validate_nonnegative(credit_outstanding, "credit_outstanding"),
                "available_credit": available_credit,
                "funding_available": available_credit,
                "max_output": inf,
                "reason": "not_configured",
            }
        if unit_cash_cost is None:
            cash_value = cls._coerce_cash(cash)
            spendable_cash = max(0.0, cash_value - reserve)
            available_credit = cls._available_credit(
                approved_credit=approved_credit,
                credit_limit=credit_limit,
                credit_outstanding=credit_outstanding,
            )
            return {
                "enabled": False,
                "cash": cash_value,
                "unit_cash_cost": None,
                "cash_reserve": reserve,
                "spendable_cash": spendable_cash,
                "approved_credit": cls._coerce_optional_nonnegative(approved_credit),
                "credit_limit": cls._coerce_optional_nonnegative(credit_limit),
                "credit_outstanding": cls._validate_nonnegative(credit_outstanding, "credit_outstanding"),
                "available_credit": available_credit,
                "funding_available": spendable_cash + available_credit,
                "max_output": inf,
                "reason": "unit_cash_cost_not_configured",
            }

        cash_value = cls._coerce_cash(cash)
        spendable_cash = max(0.0, cash_value - reserve)
        available_credit = cls._available_credit(
            approved_credit=approved_credit,
            credit_limit=credit_limit,
            credit_outstanding=credit_outstanding,
        )
        unit_cost = float(unit_cash_cost or 0.0)
        if unit_cost <= 0.0:
            return {
                "enabled": False,
                "cash": cash_value,
                "unit_cash_cost": unit_cost,
                "cash_reserve": reserve,
                "spendable_cash": spendable_cash,
                "approved_credit": cls._coerce_optional_nonnegative(approved_credit),
                "credit_limit": cls._coerce_optional_nonnegative(credit_limit),
                "credit_outstanding": cls._validate_nonnegative(credit_outstanding, "credit_outstanding"),
                "available_credit": available_credit,
                "funding_available": spendable_cash + available_credit,
                "max_output": inf,
                "reason": "non_positive_unit_cash_cost",
            }

        funding_available = spendable_cash + available_credit
        max_output = funding_available / unit_cost

        return {
            "enabled": True,
            "cash": cash_value,
            "unit_cash_cost": unit_cost,
            "cash_reserve": reserve,
            "spendable_cash": spendable_cash,
            "approved_credit": cls._coerce_optional_nonnegative(approved_credit),
            "credit_limit": cls._coerce_optional_nonnegative(credit_limit),
            "credit_outstanding": cls._validate_nonnegative(credit_outstanding, "credit_outstanding"),
            "available_credit": available_credit,
            "funding_available": funding_available,
            "required_funding": desired_output * unit_cost,
            "max_output": max(0.0, max_output),
            "reason": None,
            "desired_output": desired_output,
        }

    @classmethod
    def _coerce_cash(cls, cash: float) -> float:
        try:
            return float(cash or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _coerce_optional_nonnegative(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return cls._validate_nonnegative(value, "credit")

    @classmethod
    def _available_credit(
        cls,
        *,
        approved_credit: Optional[float],
        credit_limit: Optional[float],
        credit_outstanding: float,
    ) -> float:
        approved = (
            cls._validate_nonnegative(approved_credit, "approved_credit")
            if approved_credit is not None
            else None
        )
        if approved_credit is not None:
            if credit_limit is None:
                return float(approved)
            limit_available = cls._available_credit(
                approved_credit=None,
                credit_limit=credit_limit,
                credit_outstanding=credit_outstanding,
            )
            return min(float(approved), limit_available)
        if credit_limit is None:
            return 0.0

        limit = cls._validate_nonnegative(credit_limit, "credit_limit")
        outstanding = cls._validate_nonnegative(credit_outstanding, "credit_outstanding")
        return max(0.0, limit - outstanding)


def build_production_plan(plan_input: ProductionPlanInput) -> ProductionPlan:
    return ProductionPlanningPolicy().build_plan(plan_input)
