from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional


@dataclass(frozen=True)
class FirmInitCalibrationInput:
    firm_id: str
    industry_code: str
    industry_type: str
    expected_monthly_revenue: float = 0.0
    expected_monthly_cost: float = 0.0
    expected_monthly_sales_units: float = 0.0
    expected_unit_price: Optional[float] = None
    inventory_cover_months: float = 1.0
    cash_multiplier: float = 1.5
    min_cash: float = 10000.0
    capital_output_ratio: float = 3.0
    inventory_value_share: float = 0.5


@dataclass(frozen=True)
class FirmInitCalibrationResult:
    firm_id: str
    industry_code: str
    industry_type: str
    initial_cash: float
    initial_capital_stock: float
    target_inventory_units: float
    target_inventory_value: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "firm_id": self.firm_id,
            "industry_code": self.industry_code,
            "industry_type": self.industry_type,
            "initial_cash": self.initial_cash,
            "initial_capital_stock": self.initial_capital_stock,
            "target_inventory_units": self.target_inventory_units,
            "target_inventory_value": self.target_inventory_value,
            "diagnostics": dict(self.diagnostics),
        }


class FirmInitializationCalibrator:
    """
    Low-intrusion calibration helper for firm balance sheet seeds.

    The simulator can use this to convert demand discovery into a consistent
    initial cash/capital/inventory target without hard-coding a single rule.
    """

    def calibrate(self, payload: FirmInitCalibrationInput) -> FirmInitCalibrationResult:
        expected_monthly_revenue = max(0.0, float(payload.expected_monthly_revenue or 0.0))
        expected_monthly_cost = max(0.0, float(payload.expected_monthly_cost or 0.0))
        expected_monthly_sales_units = max(0.0, float(payload.expected_monthly_sales_units or 0.0))
        inventory_cover_months = max(0.0, float(payload.inventory_cover_months or 0.0))
        cash_multiplier = max(0.0, float(payload.cash_multiplier or 0.0))
        min_cash = max(0.0, float(payload.min_cash or 0.0))
        capital_output_ratio = max(0.1, float(payload.capital_output_ratio or 0.1))
        expected_unit_price = payload.expected_unit_price
        if expected_unit_price is None and expected_monthly_sales_units > 0:
            expected_unit_price = expected_monthly_revenue / expected_monthly_sales_units
        expected_unit_price = max(0.0, float(expected_unit_price or 0.0))
        inventory_value_share = _clamp(float(payload.inventory_value_share or 0.0), 0.0, 1.0)

        working_capital_requirement = expected_monthly_cost * cash_multiplier
        initial_cash = max(min_cash, working_capital_requirement)
        target_inventory_units = expected_monthly_sales_units * inventory_cover_months
        if target_inventory_units > 0.0 and expected_unit_price > 0.0:
            target_inventory_value = target_inventory_units * expected_unit_price * inventory_value_share
        else:
            target_inventory_value = expected_monthly_revenue * inventory_cover_months * inventory_value_share

        annualized_output = max(expected_monthly_revenue * 12.0, expected_monthly_cost * 12.0, 0.0)
        initial_capital_stock = max(
            target_inventory_value,
            annualized_output * capital_output_ratio,
        )

        return FirmInitCalibrationResult(
            firm_id=payload.firm_id,
            industry_code=payload.industry_code,
            industry_type=payload.industry_type,
            initial_cash=round(initial_cash, 2),
            initial_capital_stock=round(initial_capital_stock, 2),
            target_inventory_units=round(target_inventory_units, 2),
            target_inventory_value=round(target_inventory_value, 2),
            diagnostics={
                "expected_monthly_revenue": round(expected_monthly_revenue, 2),
                "expected_monthly_cost": round(expected_monthly_cost, 2),
                "expected_monthly_sales_units": round(expected_monthly_sales_units, 2),
                "expected_unit_price": round(expected_unit_price, 4),
                "inventory_cover_months": inventory_cover_months,
                "cash_multiplier": cash_multiplier,
                "working_capital_requirement": round(working_capital_requirement, 2),
                "capital_output_ratio": capital_output_ratio,
                "annualized_output": round(annualized_output, 2),
                "inventory_value_share": inventory_value_share,
            },
        )

    def calibrate_many(
        self,
        inputs: Iterable[FirmInitCalibrationInput],
    ) -> Dict[str, FirmInitCalibrationResult]:
        return {item.firm_id: self.calibrate(item) for item in inputs}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
