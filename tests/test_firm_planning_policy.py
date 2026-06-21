import ast
import unittest
from pathlib import Path

from agenteconomy.agent.firm_planning_policy import (
    ProductionPlanInput,
    ProductionPlanningPolicy,
)


class ProductionPlanningPolicyTest(unittest.TestCase):
    def test_inventory_target_triggers_positive_output_without_current_shortage(self):
        policy = ProductionPlanningPolicy()

        plan = policy.build_plan(
            ProductionPlanInput(
                sales_history=[100.0, 100.0, 100.0],
                unmet_demand_history=[0.0, 0.0, 0.0],
                current_inventory=120.0,
                target_inventory_months=2.0,
                available_labor=500.0,
                capital_stock=500.0,
            )
        )

        self.assertAlmostEqual(plan.expected_demand, 100.0)
        self.assertAlmostEqual(plan.target_inventory, 200.0)
        self.assertAlmostEqual(plan.desired_output, 180.0)
        self.assertAlmostEqual(plan.feasible_output, 180.0)

    def test_labor_capital_and_cash_constraints_reduce_feasible_output(self):
        policy = ProductionPlanningPolicy()

        plan = policy.build_plan(
            ProductionPlanInput(
                sales_history=[100.0],
                current_inventory=0.0,
                target_inventory_months=0.0,
                available_labor=40.0,
                labor_productivity=1.0,
                capital_stock=80.0,
                capital_productivity=1.0,
                cash=50.0,
                unit_cash_cost=2.0,
            )
        )

        self.assertAlmostEqual(plan.desired_output, 100.0)
        self.assertAlmostEqual(plan.feasible_output, 25.0)
        self.assertAlmostEqual(plan.production_gap, 75.0)
        self.assertEqual(
            set(plan.diagnostics["limiting_constraints"]),
            {"labor", "capital", "cash"},
        )
        self.assertEqual(plan.diagnostics["binding_constraints"], ["cash"])
        self.assertTrue(plan.diagnostics["constraints"]["labor"]["limited"])
        self.assertTrue(plan.diagnostics["constraints"]["capital"]["limited"])
        self.assertTrue(plan.diagnostics["constraints"]["cash"]["binding"])

    def test_unmet_demand_contributes_to_ema_expected_demand(self):
        policy = ProductionPlanningPolicy()

        plan = policy.build_plan(
            ProductionPlanInput(
                sales_history=[80.0, 100.0],
                unmet_demand_history=[20.0, 0.0],
                current_inventory=0.0,
                target_inventory_months=0.0,
                ema_alpha=0.5,
                available_labor=200.0,
                capital_stock=200.0,
            )
        )

        self.assertEqual(plan.diagnostics["demand_history"], [100.0, 100.0])
        self.assertAlmostEqual(plan.expected_demand, 100.0)

    def test_cash_constraint_requires_positive_unit_cost(self):
        policy = ProductionPlanningPolicy()

        plan = policy.build_plan(
            ProductionPlanInput(
                sales_history=[50.0],
                current_inventory=0.0,
                target_inventory_months=0.0,
                available_labor=100.0,
                capital_stock=100.0,
                cash=-10.0,
                unit_cash_cost=0.0,
            )
        )

        self.assertAlmostEqual(plan.desired_output, 50.0)
        self.assertAlmostEqual(plan.feasible_output, 50.0)
        self.assertFalse(plan.diagnostics["constraints"]["cash"]["enabled"])
        self.assertEqual(
            plan.diagnostics["constraints"]["cash"]["reason"],
            "non_positive_unit_cash_cost",
        )

    def test_no_cash_and_no_credit_blocks_cash_funded_production(self):
        policy = ProductionPlanningPolicy()

        plan = policy.build_plan(
            ProductionPlanInput(
                sales_history=[100.0],
                current_inventory=0.0,
                target_inventory_months=0.0,
                available_labor=200.0,
                capital_stock=200.0,
                cash=0.0,
                unit_cash_cost=2.0,
            )
        )

        self.assertAlmostEqual(plan.desired_output, 100.0)
        self.assertAlmostEqual(plan.feasible_output, 0.0)
        self.assertEqual(plan.diagnostics["binding_constraints"], ["cash"])
        self.assertAlmostEqual(plan.max_output_by_constraint["cash"], 0.0)

    def test_cash_covers_partial_output_without_negative_balance(self):
        policy = ProductionPlanningPolicy()

        plan = policy.build_plan(
            ProductionPlanInput(
                sales_history=[100.0],
                current_inventory=0.0,
                target_inventory_months=0.0,
                available_labor=200.0,
                capital_stock=200.0,
                cash=60.0,
                unit_cash_cost=2.0,
            )
        )

        self.assertAlmostEqual(plan.feasible_output, 30.0)
        cash_constraint = plan.diagnostics["constraints"]["cash"]
        self.assertAlmostEqual(cash_constraint["spendable_cash"], 60.0)
        self.assertAlmostEqual(cash_constraint["funding_available"], 60.0)

    def test_cash_plus_approved_credit_covers_more_output(self):
        policy = ProductionPlanningPolicy()

        plan = policy.build_plan(
            ProductionPlanInput(
                sales_history=[100.0],
                current_inventory=0.0,
                target_inventory_months=0.0,
                available_labor=200.0,
                capital_stock=200.0,
                cash=60.0,
                unit_cash_cost=2.0,
                approved_credit=80.0,
            )
        )

        self.assertAlmostEqual(plan.feasible_output, 70.0)
        cash_constraint = plan.diagnostics["constraints"]["cash"]
        self.assertAlmostEqual(cash_constraint["available_credit"], 80.0)
        self.assertAlmostEqual(cash_constraint["funding_available"], 140.0)

    def test_cash_reserve_reduces_spendable_cash_before_credit(self):
        policy = ProductionPlanningPolicy()

        plan = policy.build_plan(
            ProductionPlanInput(
                sales_history=[100.0],
                current_inventory=0.0,
                target_inventory_months=0.0,
                available_labor=200.0,
                capital_stock=200.0,
                cash=60.0,
                cash_reserve=20.0,
                unit_cash_cost=2.0,
                credit_limit=100.0,
                credit_outstanding=60.0,
            )
        )

        self.assertAlmostEqual(plan.feasible_output, 40.0)
        cash_constraint = plan.diagnostics["constraints"]["cash"]
        self.assertAlmostEqual(cash_constraint["spendable_cash"], 40.0)
        self.assertAlmostEqual(cash_constraint["available_credit"], 40.0)
        self.assertAlmostEqual(cash_constraint["funding_available"], 80.0)

    def test_firm_api_surface_is_backward_compatible(self):
        firm_path = Path(__file__).resolve().parents[1] / "agenteconomy" / "agent" / "firm.py"
        tree = ast.parse(firm_path.read_text(encoding="utf-8"))
        classes = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
        }

        firm_methods = {
            node.name
            for node in classes["Firm"].body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("build_production_plan", firm_methods)

        manufacture_methods = {
            node.name: node
            for node in classes["ManufactureFirm"].body
            if isinstance(node, ast.FunctionDef)
        }
        produce_args = [arg.arg for arg in manufacture_methods["produce"].args.args]
        self.assertEqual(
            produce_args,
            [
                "self",
                "production_plan",
                "sku_base_prices",
                "period",
                "update_inventory",
                "labor_cost",
                "tax_cost",
            ],
        )


if __name__ == "__main__":
    unittest.main()
