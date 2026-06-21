import asyncio
import unittest

from agenteconomy.agent.household import CategoryPlan, Household


class CountingProductMarket:
    def __init__(self):
        self.queries = []

    def search_by_vector(self, query, top_k=10):
        self.queries.append((query, top_k))
        return []


class HouseholdVectorSearchTests(unittest.TestCase):
    def test_merge_need_queries_reduces_rule_mode_search_calls(self):
        household = Household(
            household_id="hh_1",
            name="HH 1",
            description="",
            owner="owner",
            load_profile=False,
        )
        market = CountingProductMarket()
        category_plans = [
            CategoryPlan(
                category="Food and beverage and tobacco products",
                budget_amount=1000.0,
                need_descriptions=["Basic recurring needs", "Budget replenishment"],
            )
        ]

        result = household._consumption_step2_vector_match_sync(
            category_plans=category_plans,
            top_k=5,
            product_market=market,
            merge_need_queries=True,
        )

        self.assertEqual(len(market.queries), 1)
        self.assertIn("Basic recurring needs", market.queries[0][0])
        self.assertIn("Budget replenishment", market.queries[0][0])
        self.assertEqual(result["Food and beverage and tobacco products"]["candidates"], [])


class HouseholdPrecomputedCandidateTests(unittest.TestCase):
    def test_rule_mode_uses_precomputed_candidates_without_market_search(self):
        household = Household(
            household_id="hh_1",
            name="HH 1",
            description="",
            owner="owner",
            load_profile=False,
        )
        market = CountingProductMarket()
        candidates = {
            "Food and beverage and tobacco products": {
                "candidates": [
                    {
                        "product_id": "food_1",
                        "name": "Food product",
                        "current_price": 10.0,
                        "storage": 100.0,
                    }
                ]
            }
        }

        result = asyncio.run(
            household.consume_v2(
                use_llm=False,
                product_market=market,
                available_balance=5000.0,
                expected_income=1000.0,
                available_budget=5000.0,
                candidate_products_by_category=candidates,
            )
        )

        self.assertEqual(market.queries, [])
        self.assertTrue(result["is_rule_based"])
        self.assertEqual(result["step3"]["purchases"][0]["product_id"], "food_1")


if __name__ == "__main__":
    unittest.main()
