from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class CreditApplication:
    """Minimal firm credit application used before wiring bank ledgers."""

    firm_id: str
    requested_amount: float
    cash: float = 0.0
    monthly_revenue: float = 0.0
    monthly_operating_cost: float = 0.0
    existing_debt: float = 0.0
    capital_stock: float = 0.0
    inventory_value: float = 0.0
    distress_months: int = 0
    sales_volatility: float = 0.0
    industry_risk: float = 0.5


@dataclass(frozen=True)
class CreditDecision:
    firm_id: str
    approved: bool
    approved_amount: float
    credit_limit: float
    annual_interest_rate: float
    risk_score: float
    reasons: list[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def approved_credit(self) -> float:
        return max(0.0, float(self.approved_amount or 0.0)) if self.approved else 0.0

    @property
    def rejection_reasons(self) -> list[str]:
        return list(self.reasons)


@dataclass
class CreditFacility:
    firm_id: str
    credit_limit: float
    outstanding_balance: float = 0.0
    annual_interest_rate: float = 0.08
    defaulted: bool = False

    @property
    def available_credit(self) -> float:
        if self.defaulted:
            return 0.0
        return max(0.0, self.credit_limit - self.outstanding_balance)

    def draw(self, amount: float) -> float:
        if amount <= 0.0 or self.defaulted:
            return 0.0
        draw_amount = min(float(amount), self.available_credit)
        self.outstanding_balance += draw_amount
        return draw_amount

    def accrue_monthly_interest(self) -> float:
        if self.defaulted or self.outstanding_balance <= 0.0:
            return 0.0
        interest = self.outstanding_balance * max(0.0, self.annual_interest_rate) / 12.0
        self.outstanding_balance += interest
        return interest

    def repay(self, amount: float) -> float:
        if amount <= 0.0 or self.outstanding_balance <= 0.0:
            return 0.0
        payment = min(float(amount), self.outstanding_balance)
        self.outstanding_balance -= payment
        return payment


class SimpleFirmCreditPolicy:
    """
    Rule-based working-capital credit policy.

    This is intentionally independent from EconomicCenter so simulator wiring can
    later replace free firm overdrafts with explicit credit draws.
    """

    def __init__(
        self,
        *,
        base_annual_rate: float = 0.06,
        max_spread: float = 0.12,
        revenue_limit_multiplier: float = 1.5,
        collateral_advance_rate: float = 0.35,
        min_debt_service_coverage: float = 1.05,
    ) -> None:
        self.base_annual_rate = max(0.0, float(base_annual_rate))
        self.max_spread = max(0.0, float(max_spread))
        self.revenue_limit_multiplier = max(0.0, float(revenue_limit_multiplier))
        self.collateral_advance_rate = max(0.0, float(collateral_advance_rate))
        self.min_debt_service_coverage = max(0.0, float(min_debt_service_coverage))

    def decide(self, application: CreditApplication) -> CreditDecision:
        requested = max(0.0, float(application.requested_amount or 0.0))
        risk_score = self._risk_score(application)
        credit_limit = self._credit_limit(application, risk_score)
        interest_rate = self.base_annual_rate + self.max_spread * risk_score

        reasons: list[str] = []
        if requested <= 0.0:
            reasons.append("non_positive_request")
        if application.distress_months >= 3:
            reasons.append("extended_distress")

        projected_debt = max(0.0, application.existing_debt) + requested
        dscr = self._debt_service_coverage(application, projected_debt, interest_rate)
        if dscr < self.min_debt_service_coverage:
            reasons.append("insufficient_debt_service_coverage")

        approved_amount = min(requested, credit_limit)
        if requested > credit_limit:
            reasons.append("request_exceeds_credit_limit")

        approved = approved_amount > 0.0 and "extended_distress" not in reasons and dscr >= self.min_debt_service_coverage
        if not approved:
            approved_amount = 0.0

        return CreditDecision(
            firm_id=application.firm_id,
            approved=approved,
            approved_amount=round(approved_amount, 2),
            credit_limit=round(credit_limit, 2),
            annual_interest_rate=round(interest_rate, 6),
            risk_score=round(risk_score, 4),
            reasons=reasons,
            diagnostics={
                "requested_amount": requested,
                "projected_debt": round(projected_debt, 2),
                "debt_service_coverage": round(dscr, 4),
                "monthly_cash_flow": round(
                    float(application.monthly_revenue or 0.0)
                    - float(application.monthly_operating_cost or 0.0),
                    2,
                ),
            },
        )

    def open_facility(self, decision: CreditDecision) -> Optional[CreditFacility]:
        if not decision.approved:
            return None
        return CreditFacility(
            firm_id=decision.firm_id,
            credit_limit=decision.credit_limit,
            annual_interest_rate=decision.annual_interest_rate,
        )

    def _credit_limit(self, application: CreditApplication, risk_score: float) -> float:
        revenue_limit = max(0.0, application.monthly_revenue) * self.revenue_limit_multiplier
        collateral_value = max(0.0, application.capital_stock) + max(0.0, application.inventory_value)
        collateral_limit = collateral_value * self.collateral_advance_rate
        raw_limit = revenue_limit + collateral_limit + max(0.0, application.cash) * 0.25
        risk_haircut = max(0.05, 1.0 - 0.75 * risk_score)
        return max(0.0, raw_limit * risk_haircut - max(0.0, application.existing_debt))

    def _risk_score(self, application: CreditApplication) -> float:
        revenue = max(0.0, application.monthly_revenue)
        cost = max(0.0, application.monthly_operating_cost)
        debt = max(0.0, application.existing_debt)
        assets = max(0.0, application.capital_stock) + max(0.0, application.inventory_value) + max(0.0, application.cash)

        margin = (revenue - cost) / max(revenue, 1.0)
        leverage = debt / max(assets, revenue, 1.0)
        liquidity_gap = max(0.0, cost - max(0.0, application.cash)) / max(cost, 1.0)
        distress = min(max(int(application.distress_months), 0), 6) / 6.0
        volatility = _clamp(float(application.sales_volatility or 0.0), 0.0, 1.0)
        industry_risk = _clamp(float(application.industry_risk or 0.0), 0.0, 1.0)

        risk = (
            0.24 * _clamp(1.0 - (margin + 0.25) / 0.75, 0.0, 1.0)
            + 0.22 * _clamp(leverage, 0.0, 1.0)
            + 0.20 * liquidity_gap
            + 0.16 * distress
            + 0.10 * volatility
            + 0.08 * industry_risk
        )
        return _clamp(risk, 0.0, 1.0)

    @staticmethod
    def _debt_service_coverage(
        application: CreditApplication,
        projected_debt: float,
        annual_interest_rate: float,
    ) -> float:
        monthly_cash_flow = float(application.monthly_revenue or 0.0) - float(application.monthly_operating_cost or 0.0)
        monthly_interest = projected_debt * max(0.0, annual_interest_rate) / 12.0
        if monthly_interest <= 0.0:
            return float("inf")
        return monthly_cash_flow / monthly_interest


class BankCreditPolicy(SimpleFirmCreditPolicy):
    """Stable public name for the rule-based firm working-capital policy."""


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
