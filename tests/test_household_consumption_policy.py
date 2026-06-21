import unittest
import asyncio

from agenteconomy.agent.household import Household
from agenteconomy.agent.household_consumption_policy import (
    build_constrained_llm_consumption_plan,
    build_rule_based_consumption_plan,
)


FOOD = "Food and beverage and tobacco products"
COMPUTERS = "Computer and electronic products"


def test_rule_budget_respects_available_budget_and_cash_buffer():
    plan = build_rule_based_consumption_plan(
        household_state={
            "ER82017": 2,
            "ER85692": 12000,
            "ER85768": 2600,
        },
        available_budget=5000,
        expected_income=3500,
        persona_params={
            "price_sensitivity": 0.5,
            "liquidity_preference": 0.8,
            "habit_strength": 0.2,
        },
        categories=[FOOD, COMPUTERS],
    )

    assert plan.total_budget <= 5000
    assert plan.reserved_cash >= plan.target_cash_buffer
    assert round(sum(plan.major_budgets.values()), 2) == plan.total_budget
    assert plan.category_budgets[FOOD] > plan.category_budgets[COMPUTERS]


def test_liquidity_preference_reduces_spending_and_raises_buffer():
    common = {
        "household_state": {"ER82017": 2, "ER85692": 30000, "ER85768": 0},
        "available_budget": 7000,
        "expected_income": 4000,
        "categories": [FOOD, COMPUTERS],
    }

    low_liquidity = build_rule_based_consumption_plan(
        **common,
        persona_params={
            "price_sensitivity": 0.5,
            "liquidity_preference": 0.1,
            "habit_strength": 0.0,
        },
    )
    high_liquidity = build_rule_based_consumption_plan(
        **common,
        persona_params={
            "price_sensitivity": 0.5,
            "liquidity_preference": 0.9,
            "habit_strength": 0.0,
        },
    )

    assert high_liquidity.target_cash_buffer > low_liquidity.target_cash_buffer
    assert high_liquidity.total_budget < low_liquidity.total_budget


def test_habit_strength_moves_budget_toward_last_consumption():
    common = {
        "household_state": {
            "ER82017": 1,
            "ER85692": 15000,
            "last_month_consumption": 5000,
        },
        "available_budget": 8000,
        "expected_income": 2500,
        "categories": [FOOD],
    }

    weak_habit = build_rule_based_consumption_plan(
        **common,
        persona_params={
            "price_sensitivity": 0.5,
            "liquidity_preference": 0.4,
            "habit_strength": 0.0,
        },
    )
    strong_habit = build_rule_based_consumption_plan(
        **common,
        persona_params={
            "price_sensitivity": 0.5,
            "liquidity_preference": 0.4,
            "habit_strength": 1.0,
        },
    )

    assert strong_habit.total_budget > weak_habit.total_budget


def test_price_sensitivity_changes_product_preference_order():
    candidates = {
        FOOD: [
            {
                "product_id": "premium",
                "name": "Premium option",
                "current_price": 20,
                "storage": 20,
                "score": 1.0,
            },
            {
                "product_id": "cheap",
                "name": "Cheap option",
                "current_price": 5,
                "storage": 20,
                "score": 0.2,
            },
        ]
    }
    common = {
        "household_state": {"ER82017": 1, "ER85692": 10000},
        "available_budget": 4000,
        "expected_income": 3000,
        "candidate_products_by_category": candidates,
        "categories": [FOOD],
    }

    low_price_sensitivity = build_rule_based_consumption_plan(
        **common,
        persona_params={
            "price_sensitivity": 0.0,
            "liquidity_preference": 0.4,
            "habit_strength": 0.0,
        },
    )
    high_price_sensitivity = build_rule_based_consumption_plan(
        **common,
        persona_params={
            "price_sensitivity": 1.0,
            "liquidity_preference": 0.4,
            "habit_strength": 0.0,
        },
    )

    assert low_price_sensitivity.product_preferences[FOOD][0]["product_id"] == "premium"
    assert high_price_sensitivity.product_preferences[FOOD][0]["product_id"] == "cheap"


def test_rule_plan_spreads_large_budget_across_more_skus():
    candidates = {
        FOOD: [
            {
                "product_id": f"sku_{idx}",
                "name": f"Food option {idx}",
                "current_price": 10,
                "storage": 100,
                "score": 1.0 / (idx + 1),
            }
            for idx in range(20)
        ]
    }

    plan = build_rule_based_consumption_plan(
        household_state={"ER82017": 2, "ER85692": 50000, "ER85768": 0},
        available_budget=15000,
        expected_income=6000,
        candidate_products_by_category=candidates,
        categories=[FOOD],
        persona_params={
            "price_sensitivity": 0.4,
            "liquidity_preference": 0.2,
            "habit_strength": 0.0,
        },
    )

    selected = [p for p in plan.purchases if p["category"] == FOOD]

    assert len(selected) > 3
    assert max(p["allocated_budget"] for p in selected) <= 350.0


def test_high_cash_low_income_budget_is_capped_by_monthly_flow_guard():
    plan = build_rule_based_consumption_plan(
        household_state={"ER82017": 1, "ER85692": 1_000_000, "ER85768": 0},
        available_budget=1_000_000,
        expected_income=0,
        categories=[FOOD],
        persona_params={
            "price_sensitivity": 0.5,
            "liquidity_preference": 0.1,
            "habit_strength": 0.0,
        },
    )

    assert plan.total_budget <= plan.diagnostics["monthly_flow_cap"]
    assert plan.total_budget < 3000.0
    assert plan.diagnostics["wealth_component"] < plan.diagnostics["wealth_component_uncapped"]


def test_unemployment_pressure_raises_precautionary_saving():
    common = {
        "available_budget": 50_000,
        "expected_income": 2_000,
        "categories": [FOOD],
        "persona_params": {
            "price_sensitivity": 0.5,
            "liquidity_preference": 0.8,
            "habit_strength": 0.0,
        },
    }
    normal_labor_market = build_rule_based_consumption_plan(
        **common,
        household_state={
            "ER82017": 2,
            "ER85692": 50_000,
            "ER85768": 0,
            "macro_indicators": {"unemployment_rate": 0.06},
        },
    )
    weak_labor_market = build_rule_based_consumption_plan(
        **common,
        household_state={
            "ER82017": 2,
            "ER85692": 50_000,
            "ER85768": 0,
            "macro_indicators": {"unemployment_rate": 0.45},
        },
    )

    assert weak_labor_market.diagnostics["unemployment_pressure"] > normal_labor_market.diagnostics["unemployment_pressure"]
    assert weak_labor_market.diagnostics["precautionary_discount"] < normal_labor_market.diagnostics["precautionary_discount"]
    assert weak_labor_market.total_budget < normal_labor_market.total_budget


def test_household_default_available_budget_does_not_add_expected_income():
    household = Household(
        household_id="household_1",
        name="Test",
        description="",
        owner="test",
        load_profile=False,
    )

    result = asyncio.run(
        household.consume_rule_based(
            available_balance=5000.0,
            expected_income=3000.0,
            candidate_products_by_category={},
            persona_params={"liquidity_preference": 0.5, "habit_strength": 0.0},
        )
    )

    assert result["step0"]["available_budget"] == 5000.0
    assert result["policy_diagnostics"]["available_budget"] == 5000.0


def test_constrained_llm_budget_uses_empirical_anchor_not_llm_total():
    anchor = build_rule_based_consumption_plan(
        household_state={
            "ER82017": 2,
            "ER85629": 3500,
            "ER85692": 12000,
            "ER85768": 2600,
            "expenditure_retail_merchandise": 900,
            "ER85701": 800,
            "ER85747": 200,
            "expenditure_transportation": 300,
            "expenditure_utilities": 180,
            "expenditure_insurance": 220,
        },
        available_budget=5000,
        expected_income=3500,
        categories=[FOOD, COMPUTERS],
    )

    constrained = build_constrained_llm_consumption_plan(
        household_state={
            "ER82017": 2,
            "ER85629": 3500,
            "ER85692": 12000,
            "ER85768": 2600,
            "expenditure_retail_merchandise": 900,
            "ER85701": 800,
            "ER85747": 200,
            "expenditure_transportation": 300,
            "expenditure_utilities": 180,
            "expenditure_insurance": 220,
        },
        available_budget=5000,
        expected_income=3500,
        categories=[FOOD, COMPUTERS],
        llm_major_budgets={
            "Retail merchandise": 90_000,
            "housing": 20_000,
            "healthcare": 10_000,
            "transportation": 8_000,
            "utilities": 5_000,
            "insurance": 4_000,
        },
        llm_category_plans=[
            {"category": FOOD, "budget_amount": 80_000, "need_descriptions": ["luxury food"]},
            {"category": COMPUTERS, "budget_amount": 30_000, "need_descriptions": ["new electronics"]},
        ],
    )

    assert constrained.total_budget == anchor.total_budget
    assert constrained.total_budget <= 5000
    assert round(sum(constrained.major_budgets.values()), 2) == constrained.total_budget
    assert constrained.diagnostics["policy_mode"] == "llm_constrained"
    assert "ER85768" in constrained.diagnostics["empirical_anchor_fields"]


def test_constrained_llm_rejects_invalid_product_ids():
    candidates = {
        FOOD: {
            "candidates": [
                {
                    "product_id": "food_1",
                    "name": "Staple groceries",
                    "current_price": 10.0,
                    "storage": 100.0,
                    "score": 0.8,
                }
            ]
        }
    }

    constrained = build_constrained_llm_consumption_plan(
        household_state={"ER82017": 1, "ER85629": 3000, "ER85692": 8000, "ER85768": 1800},
        available_budget=4000,
        expected_income=3000,
        candidate_products_by_category=candidates,
        categories=[FOOD],
        llm_major_budgets={"Retail merchandise": 2000},
        llm_category_plans=[
            {"category": FOOD, "budget_amount": 2000, "need_descriptions": ["weekly groceries"]},
        ],
        llm_purchases_by_category={
            FOOD: [
                {"product_id": "not_in_market", "budget_share": 0.9, "reason": "invalid"},
                {"product_id": "food_1", "budget_share": 0.1, "reason": "valid"},
            ]
        },
    )

    product_ids = {purchase["product_id"] for purchase in constrained.purchases}
    assert "food_1" in product_ids
    assert "not_in_market" not in product_ids
    assert all(purchase["allocated_budget"] <= 350.0 for purchase in constrained.purchases)


def _make_consumption_test_household() -> Household:
    household = Household(
        household_id="household_1",
        name="Test",
        description="",
        owner="test",
        load_profile=False,
    )
    household.csv_values.update(
        {
            "ER82017": 2,
            "ER85629": 3500,
            "ER85692": 12000,
            "ER85768": 2600,
            "expenditure_retail_merchandise": 900,
            "ER85701": 800,
            "ER85747": 200,
            "expenditure_transportation": 300,
            "expenditure_utilities": 180,
            "expenditure_insurance": 220,
        }
    )
    return household


def test_consume_v2_llm_path_returns_constrained_plan_with_fake_llm():
    household = _make_consumption_test_household()
    candidates = {
        FOOD: {
            "candidates": [
                {
                    "product_id": "food_1",
                    "name": "Staple groceries",
                    "current_price": 10.0,
                    "storage": 100.0,
                    "score": 0.9,
                }
            ]
        }
    }
    responses = iter(
        [
            '{"total_budget": 999999, "budgets": {"Retail merchandise": 999999, "housing": 1, "healthcare": 1, "transportation": 1, "utilities": 1, "insurance": 1}, "note": "large"}',
            '{"category_plans": [{"category": "Food and beverage and tobacco products", "budget_amount": 999999, "need_descriptions": ["weekly groceries"]}], "note": "food"}',
            '{"categories": [{"category": "Food and beverage and tobacco products", "purchases": [{"product_id": "food_1", "allocated_budget": 5000, "reason": "chosen"}]}], "note": "pick"}',
        ]
    )

    async def fake_llm_chat(*args, **kwargs):
        return next(responses)

    household._llm_chat = fake_llm_chat

    result = asyncio.run(
        household.consume_v2(
            use_llm=True,
            llm_mode="monthly",
            available_balance=5000.0,
            expected_income=3500.0,
            available_budget=5000.0,
            candidate_products_by_category=candidates,
            persona_params={"liquidity_preference": 0.5, "habit_strength": 0.0},
        )
    )

    assert result["is_llm_consumption"]
    assert not result["is_rule_based"]
    assert result["step0"]["total_budget"] <= 5000.0
    assert result["policy_diagnostics"]["policy_mode"] == "llm_constrained"
    assert result["step3"]["purchases"][0]["product_id"] == "food_1"


def test_consume_v2_profile_mode_uses_cached_llm_profile_without_step_llms():
    household = _make_consumption_test_household()
    candidates = {
        FOOD: {
            "candidates": [
                {
                    "product_id": "food_1",
                    "name": "Staple groceries",
                    "current_price": 10.0,
                    "storage": 100.0,
                    "score": 0.9,
                }
            ]
        }
    }
    calls = {"profile": 0, "step0": 0, "step1": 0, "step3": 0}

    async def fake_llm_chat(*args, **kwargs):
        calls["profile"] += 1
        return (
            '{"price_sensitivity": 0.9, "liquidity_preference": 0.8, '
            '"habit_strength": 0.4, "essential_bias": 0.6, '
            '"strategy_type": "cautious essentials", "explanation": "test"}'
        )

    async def fail_step0(*args, **kwargs):
        calls["step0"] += 1
        raise AssertionError("profile mode must not run Step0 monthly LLM")

    async def fail_step1(*args, **kwargs):
        calls["step1"] += 1
        raise AssertionError("profile mode must not run Step1 monthly LLM")

    async def fail_step3(*args, **kwargs):
        calls["step3"] += 1
        raise AssertionError("profile mode must not run Step3 monthly LLM")

    household._llm_chat = fake_llm_chat
    household.consumption_step0_major_budget_allocation = fail_step0
    household.consumption_step1_needs_by_category = fail_step1
    household.consumption_step3_purchase_llm = fail_step3

    first = asyncio.run(
        household.consume_v2(
            use_llm=True,
            llm_mode="profile",
            current_month=1,
            profile_refresh_months=12,
            available_balance=5000.0,
            expected_income=3500.0,
            available_budget=5000.0,
            candidate_products_by_category=candidates,
        )
    )
    second = asyncio.run(
        household.consume_v2(
            use_llm=True,
            llm_mode="profile",
            current_month=2,
            profile_refresh_months=12,
            available_balance=5000.0,
            expected_income=3500.0,
            available_budget=5000.0,
            candidate_products_by_category=candidates,
        )
    )

    assert calls == {"profile": 1, "step0": 0, "step1": 0, "step3": 0}
    assert first["is_llm_consumption"]
    assert first["is_profile_consumption"]
    assert not first["is_rule_based"]
    assert first["policy_diagnostics"]["policy_mode"] == "llm_profile_constrained"
    assert second["policy_diagnostics"]["consumption_profile"]["price_sensitivity"] == 0.9


class HouseholdConsumptionPolicyTests(unittest.TestCase):
    def test_rule_budget_respects_available_budget_and_cash_buffer(self):
        test_rule_budget_respects_available_budget_and_cash_buffer()

    def test_liquidity_preference_reduces_spending_and_raises_buffer(self):
        test_liquidity_preference_reduces_spending_and_raises_buffer()

    def test_habit_strength_moves_budget_toward_last_consumption(self):
        test_habit_strength_moves_budget_toward_last_consumption()

    def test_price_sensitivity_changes_product_preference_order(self):
        test_price_sensitivity_changes_product_preference_order()

    def test_rule_plan_spreads_large_budget_across_more_skus(self):
        test_rule_plan_spreads_large_budget_across_more_skus()

    def test_high_cash_low_income_budget_is_capped_by_monthly_flow_guard(self):
        test_high_cash_low_income_budget_is_capped_by_monthly_flow_guard()

    def test_unemployment_pressure_raises_precautionary_saving(self):
        test_unemployment_pressure_raises_precautionary_saving()

    def test_household_default_available_budget_does_not_add_expected_income(self):
        test_household_default_available_budget_does_not_add_expected_income()

    def test_constrained_llm_budget_uses_empirical_anchor_not_llm_total(self):
        test_constrained_llm_budget_uses_empirical_anchor_not_llm_total()

    def test_constrained_llm_rejects_invalid_product_ids(self):
        test_constrained_llm_rejects_invalid_product_ids()

    def test_consume_v2_llm_path_returns_constrained_plan_with_fake_llm(self):
        test_consume_v2_llm_path_returns_constrained_plan_with_fake_llm()

    def test_consume_v2_profile_mode_uses_cached_llm_profile_without_step_llms(self):
        test_consume_v2_profile_mode_uses_cached_llm_profile_without_step_llms()


if __name__ == "__main__":
    unittest.main()
