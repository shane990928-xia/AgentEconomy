import unittest

from agenteconomy.market.pricing_policy import PricingPolicy, PricingPolicyInput


class PricingPolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = PricingPolicy()

    def test_sustained_cost_increase_pushes_price_up(self):
        price = 100.0
        prices = []

        for unit_cost in (100.0, 110.0, 120.0):
            result = self.policy.apply(PricingPolicyInput(
                current_price=price,
                unit_cost=unit_cost,
                markup=0.10,
                stickiness=0.0,
                max_change=None,
                inventory_sensitivity=0.0,
                demand_sensitivity=0.0,
                benchmark_weight=0.0,
                mean_reversion_strength=0.0,
            ))
            prices.append(result.new_price)
            price = result.new_price

        self.assertGreater(prices[0], 100.0)
        self.assertGreater(prices[1], prices[0])
        self.assertGreater(prices[2], prices[1])
        self.assertIn("cost_pressure", result.components)
        self.assertIn("new_price", result.components)

    def test_high_inventory_lowers_markup(self):
        neutral = self.policy.apply(PricingPolicyInput(
            current_price=120.0,
            unit_cost=100.0,
            inventory_ratio=1.0,
            demand_supply_ratio=1.0,
            markup=0.20,
            inventory_sensitivity=0.20,
            demand_sensitivity=0.0,
            benchmark_weight=0.0,
            mean_reversion_strength=0.0,
        ))
        overstock = self.policy.apply(PricingPolicyInput(
            current_price=120.0,
            unit_cost=100.0,
            inventory_ratio=3.0,
            demand_supply_ratio=1.0,
            markup=0.20,
            inventory_sensitivity=0.20,
            demand_sensitivity=0.0,
            benchmark_weight=0.0,
            mean_reversion_strength=0.0,
        ))

        self.assertAlmostEqual(neutral.new_price, 120.0)
        self.assertLess(overstock.new_price, neutral.new_price)
        self.assertLess(overstock.new_price / 100.0 - 1.0, 0.20)
        self.assertLess(overstock.components["inventory_pressure"], 0.0)

    def test_stickiness_and_max_change_limit_single_period_move(self):
        sticky = self.policy.apply(PricingPolicyInput(
            current_price=100.0,
            unit_cost=200.0,
            markup=0.0,
            stickiness=0.75,
            max_change=None,
            inventory_sensitivity=0.0,
            demand_sensitivity=0.0,
            benchmark_weight=0.0,
            mean_reversion_strength=0.0,
        ))
        capped = self.policy.apply(PricingPolicyInput(
            current_price=100.0,
            unit_cost=200.0,
            markup=0.0,
            stickiness=0.0,
            max_change=0.05,
            inventory_sensitivity=0.0,
            demand_sensitivity=0.0,
            benchmark_weight=0.0,
            mean_reversion_strength=0.0,
        ))

        self.assertAlmostEqual(sticky.new_price, 125.0)
        self.assertAlmostEqual(capped.new_price, 105.0)
        self.assertTrue(capped.components["limit_applied"])

    def test_disabling_mean_reversion_does_not_pull_price_back(self):
        result = self.policy.apply(PricingPolicyInput(
            current_price=150.0,
            unit_cost=150.0,
            industry_benchmark=100.0,
            markup=0.0,
            stickiness=0.0,
            max_change=None,
            inventory_sensitivity=0.0,
            demand_sensitivity=0.0,
            benchmark_weight=0.0,
            mean_reversion_target=100.0,
            mean_reversion_strength=0.0,
        ))

        self.assertAlmostEqual(result.new_price, 150.0)
        self.assertEqual(result.components["mean_reversion_pressure"], 0.0)
        self.assertEqual(result.components["benchmark_pressure"], 0.0)


if __name__ == "__main__":
    unittest.main()
