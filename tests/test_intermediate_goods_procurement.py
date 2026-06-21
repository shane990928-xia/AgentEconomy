import unittest

from agenteconomy.market.IntermediateGoodsProcurement import (
    IntermediateGoodsProcurement,
)


class IntermediateGoodsProcurementPlanTests(unittest.TestCase):
    def setUp(self):
        self.procurement = IntermediateGoodsProcurement(product_market=None)

    def test_full_plan_is_feasible_when_quotes_and_budget_cover_inputs(self):
        plan = self.procurement.plan_intermediate_goods_procurement(
            target_output=100.0,
            io_suppliers=[
                {"supplier": "313TT", "coefficient": 0.2},
                {"supplier": "325", "coefficient": 0.1},
            ],
            supplier_quotes={
                "313TT": [{"sku_id": "textile_1", "unit_price": 2.0, "available_stock": 20}],
                "325": [{"sku_id": "chem_1", "unit_price": 5.0, "available_stock": 5}],
            },
            budget=50.0,
        )

        self.assertAlmostEqual(plan.planned_cost, 30.0)
        self.assertAlmostEqual(plan.feasible_scale, 1.0)
        self.assertAlmostEqual(plan.scaled_cost, 30.0)
        self.assertFalse(plan.shortage)
        self.assertEqual(plan.shortages, [])
        self.assertEqual(plan.reason, "fully_feasible")
        self.assertAlmostEqual(plan.by_industry["313TT"]["scaled_cost"], 20.0)
        self.assertAlmostEqual(plan.by_industry["325"]["scaled_cost"], 10.0)
        self.assertAlmostEqual(sum(r.reserved_cost for r in plan.reservations), 30.0)

    def test_single_input_shortage_scales_all_input_costs(self):
        plan = self.procurement.plan_intermediate_goods_procurement(
            target_output=100.0,
            io_suppliers=[
                {"supplier": "313TT", "coefficient": 0.2},
                {"supplier": "325", "coefficient": 0.1},
            ],
            supplier_quotes={
                "313TT": [{"sku_id": "textile_1", "unit_price": 2.0, "available_stock": 5}],
                "325": [{"sku_id": "chem_1", "unit_price": 5.0, "available_stock": 5}],
            },
        )

        self.assertAlmostEqual(plan.planned_cost, 30.0)
        self.assertAlmostEqual(plan.feasible_scale, 0.5)
        self.assertAlmostEqual(plan.scaled_cost, 15.0)
        self.assertTrue(plan.shortage)
        self.assertEqual(plan.reason, "input_shortage")
        self.assertEqual(plan.shortages[0]["supplier"], "313TT")
        self.assertAlmostEqual(plan.by_industry["313TT"]["scaled_cost"], 10.0)
        self.assertAlmostEqual(plan.by_industry["325"]["scaled_cost"], 5.0)
        self.assertAlmostEqual(sum(r.reserved_cost for r in plan.reservations), 15.0)

    def test_budget_shortage_scales_all_input_costs(self):
        plan = self.procurement.plan_intermediate_goods_procurement(
            target_output=100.0,
            io_suppliers=[
                {"supplier": "313TT", "coefficient": 0.2},
                {"supplier": "325", "coefficient": 0.1},
            ],
            supplier_quotes={
                "313TT": [{"sku_id": "textile_1", "unit_price": 2.0, "available_stock": 20}],
                "325": [{"sku_id": "chem_1", "unit_price": 5.0, "available_stock": 5}],
            },
            budget=12.0,
        )

        self.assertAlmostEqual(plan.planned_cost, 30.0)
        self.assertAlmostEqual(plan.feasible_scale, 0.4)
        self.assertAlmostEqual(plan.scaled_cost, 12.0)
        self.assertTrue(plan.shortage)
        self.assertEqual(plan.reason, "budget_shortage")
        self.assertEqual(plan.shortages[0]["type"], "budget")
        self.assertAlmostEqual(plan.by_industry["313TT"]["scaled_cost"], 8.0)
        self.assertAlmostEqual(plan.by_industry["325"]["scaled_cost"], 4.0)
        self.assertAlmostEqual(sum(r.reserved_cost for r in plan.reservations), 12.0)

    def test_zero_target_output_has_no_cost_or_shortage(self):
        plan = self.procurement.plan_intermediate_goods_procurement(
            target_output=0.0,
            io_suppliers=[{"supplier": "313TT", "coefficient": 0.2}],
            supplier_quotes={
                "313TT": [{"sku_id": "textile_1", "unit_price": 2.0, "available_stock": 5}]
            },
            budget=0.0,
        )

        self.assertAlmostEqual(plan.planned_cost, 0.0)
        self.assertAlmostEqual(plan.feasible_scale, 0.0)
        self.assertAlmostEqual(plan.scaled_cost, 0.0)
        self.assertFalse(plan.shortage)
        self.assertEqual(plan.shortages, [])
        self.assertEqual(plan.reservations, [])
        self.assertEqual(plan.reason, "zero_target_output")


if __name__ == "__main__":
    unittest.main()
