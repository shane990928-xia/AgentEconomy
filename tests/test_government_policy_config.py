import unittest
from types import SimpleNamespace

from agenteconomy.agent.government import Government


class FakeLaborMarketForPublicEmployment:
    def __init__(self, total_labor=20, total_matched_jobs=0, gov_matched_jobs=0):
        self.total_labor = total_labor
        self.total_matched_jobs = total_matched_jobs
        self.gov_matched_jobs = gov_matched_jobs

    def summary(self):
        return {
            "total_labor_hours": self.total_labor,
            "total_matched_jobs": self.total_matched_jobs,
            "gov_matched_jobs": self.gov_matched_jobs,
        }


class FakeEconomicCenterForGovernmentBudget:
    def __init__(self, balance):
        self.balance = balance

    def query_balance(self, agent_id):
        return self.balance


class FakeEconomicCenterForGovernmentProcurement:
    def __init__(self):
        self.transactions = []

    def add_government_procurement_transaction(self, **kwargs):
        self.transactions.append(dict(kwargs))
        return f"tx_{len(self.transactions)}"


class FakeProductMarketForGovernmentPlan:
    def __init__(self):
        self.purchase_calls = []
        self.restore_calls = []
        self.skus = {
            "325": [
                SimpleNamespace(
                    product_id="sku_chem",
                    manufacturer_price=10.0,
                    base_manufacturer_price=12.0,
                    available_stock=0.0,
                    firm_id="mfg_325",
                )
            ]
        }

    def get_skus_by_industry(self, industry_code, available_only=True):
        skus = list(self.skus.get(str(industry_code), []))
        if available_only:
            skus = [sku for sku in skus if float(getattr(sku, "available_stock", 0.0) or 0.0) > 0.0]
        return skus

    def get_available_skus(self, industry=None, period=None):
        return self.get_skus_by_industry(industry, available_only=True)

    def purchase_manufacturer_stock(self, product_id, quantity):
        self.purchase_calls.append((product_id, quantity))
        for skus in self.skus.values():
            for sku in skus:
                if sku.product_id == product_id:
                    actual = min(int(quantity), int(sku.available_stock))
                    sku.available_stock -= actual
                    return {"actual_quantity": actual}
        return {"actual_quantity": 0}

    def restore_manufacturer_stock(self, product_id, quantity):
        self.restore_calls.append((product_id, quantity))

    def get_seller_id(self, product_id):
        return "mfg_325"


class GovernmentPolicyConfigTests(unittest.TestCase):
    def test_procurement_budget_scales_fixed_floor_by_household_count(self):
        gov = Government(
            government_id="gov_main",
            household_count=8,
            demand_injection_ratio=0.30,
            min_procurement_budget=150000.0,
            max_procurement_budget=350000.0,
            min_procurement_budget_per_household=500.0,
            max_procurement_budget_per_household=1166.6667,
        )

        budget = gov._compute_procurement_budget(household_consumption_budget=1000.0)

        self.assertEqual(budget, 4000.0)

    def test_procurement_budget_keeps_baseline_floor_at_reference_scale(self):
        gov = Government(
            government_id="gov_main",
            household_count=300,
            demand_injection_ratio=0.30,
            min_procurement_budget=150000.0,
            max_procurement_budget=350000.0,
            min_procurement_budget_per_household=500.0,
            max_procurement_budget_per_household=1166.6667,
        )

        budget = gov._compute_procurement_budget(household_consumption_budget=1000.0)

        self.assertEqual(budget, 150000.0)

    def test_procurement_budget_uses_configured_ceiling(self):
        gov = Government(
            government_id="gov_main",
            household_count=8,
            demand_injection_ratio=0.30,
            min_procurement_budget=150000.0,
            max_procurement_budget=350000.0,
            min_procurement_budget_per_household=500.0,
            max_procurement_budget_per_household=1166.6667,
        )

        budget = gov._compute_procurement_budget(household_consumption_budget=100000.0)

        self.assertAlmostEqual(budget, 9333.3336, places=4)

    def test_procurement_plan_creates_zero_stock_demand_without_purchase(self):
        product_market = FakeProductMarketForGovernmentPlan()
        gov = Government(government_id="gov_main")
        gov.set_product_market(product_market)
        gov._get_government_procurement_weights = lambda: {"325": 1.0}

        plan = gov.plan_procurement_demand(period=1, budget_override=100.0)

        self.assertTrue(plan["success"])
        self.assertEqual(plan["demand_by_product"], {"sku_chem": 10.0})
        self.assertEqual(plan["items_count"], 1)
        self.assertEqual(plan["total_planned_value"], 100.0)
        self.assertEqual(product_market.purchase_calls, [])

    def test_procurement_plan_reweights_to_modeled_sku_industries(self):
        product_market = FakeProductMarketForGovernmentPlan()
        gov = Government(government_id="gov_main")
        gov.set_product_market(product_market)
        gov._get_government_procurement_weights = lambda: {"missing": 0.8, "325": 0.2}

        plan = gov.plan_procurement_demand(period=1, budget_override=100.0)

        self.assertTrue(plan["success"])
        self.assertEqual(plan["demand_by_product"], {"sku_chem": 10.0})
        self.assertEqual(plan["total_planned_value"], 100.0)
        self.assertEqual(plan["industries_attempted"], 2)
        self.assertEqual(plan["industries_planned"], 1)

    def test_procurement_prefers_planned_skus_when_stock_exists(self):
        product_market = FakeProductMarketForGovernmentPlan()
        product_market.skus["325"][0].available_stock = 5.0
        economic_center = FakeEconomicCenterForGovernmentProcurement()
        gov = Government(government_id="gov_main", economic_center=economic_center)
        gov.set_product_market(product_market)
        gov._get_government_procurement_weights = lambda: {"325": 1.0}

        result = gov.procure_goods_and_services(
            period=2,
            budget_override=100.0,
            planned_demand_by_product={"sku_chem": 3.0},
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["total_spent"], 50.0)
        self.assertEqual(result["items_count"], 2)
        self.assertEqual(product_market.purchase_calls[0], ("sku_chem", 3))
        self.assertEqual(economic_center.transactions[0]["product_id"], "sku_chem")
        self.assertEqual(economic_center.transactions[0]["quantity"], 3)

    def test_procurement_executes_planned_skus_before_missing_industry_weights(self):
        product_market = FakeProductMarketForGovernmentPlan()
        product_market.skus["325"][0].available_stock = 10.0
        economic_center = FakeEconomicCenterForGovernmentProcurement()
        gov = Government(government_id="gov_main", economic_center=economic_center)
        gov.set_product_market(product_market)
        gov._get_government_procurement_weights = lambda: {"missing": 0.8, "325": 0.2}

        result = gov.procure_goods_and_services(
            period=2,
            budget_override=100.0,
            planned_demand_by_product={"sku_chem": 10.0},
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["total_spent"], 100.0)
        self.assertEqual(result["items_count"], 1)
        self.assertEqual(result["by_industry"], {"325": 100.0})
        self.assertEqual(product_market.purchase_calls, [("sku_chem", 10)])

    def test_public_employment_is_limited_by_scale_budget_and_monthly_share(self):
        gov = Government(
            government_id="gov_main",
            household_count=8,
            public_employment_target_unemployment=0.15,
            public_employment_min_wage=15.0,
            public_employment_max_budget=150000.0,
            public_employment_max_budget_per_household=500.0,
            public_employment_start_period=2,
            public_employment_warmup_max_monthly_jobs=20,
            public_employment_max_monthly_jobs=40,
            public_employment_max_monthly_job_share=0.10,
        )
        gov.set_labor_market(FakeLaborMarketForPublicEmployment(total_labor=20, total_matched_jobs=0))

        jobs = gov._create_public_employment_jobs(period=4)

        self.assertEqual(sum(job.positions_available for job in jobs), 1)

    def test_public_employment_stock_cap_blocks_additional_jobs(self):
        gov = Government(
            government_id="gov_main",
            household_count=20,
            public_employment_target_unemployment=0.15,
            public_employment_min_wage=15.0,
            public_employment_max_budget=150000.0,
            public_employment_max_budget_per_household=500.0,
            public_employment_start_period=2,
            public_employment_warmup_max_monthly_jobs=20,
            public_employment_max_monthly_jobs=40,
            public_employment_max_monthly_job_share=0.10,
            public_employment_max_stock_share=0.20,
        )
        gov.set_labor_market(
            FakeLaborMarketForPublicEmployment(
                total_labor=40,
                total_matched_jobs=10,
                gov_matched_jobs=8,
            )
        )

        jobs = gov._create_public_employment_jobs(period=4)

        self.assertEqual(jobs, [])

    def test_public_employment_start_period_blocks_early_jobs(self):
        gov = Government(
            government_id="gov_main",
            public_employment_start_period=3,
        )
        gov.set_labor_market(FakeLaborMarketForPublicEmployment(total_labor=20, total_matched_jobs=0))

        self.assertEqual(gov._create_public_employment_jobs(period=2), [])

    def test_regular_government_labor_budget_scales_by_household_count(self):
        gov = Government(
            government_id="gov_main",
            household_count=8,
            economic_center=FakeEconomicCenterForGovernmentBudget(balance=10_000_000.0),
            government_labor_budget_share_of_balance=0.15,
            government_min_labor_budget=5000.0,
            government_max_labor_budget=120000.0,
            government_max_labor_budget_per_household=400.0,
        )

        self.assertEqual(gov._compute_labor_budget(), 3200.0)

    def test_regular_government_labor_budget_keeps_reference_scale_ceiling(self):
        gov = Government(
            government_id="gov_main",
            household_count=300,
            economic_center=FakeEconomicCenterForGovernmentBudget(balance=10_000_000.0),
            government_labor_budget_share_of_balance=0.15,
            government_min_labor_budget=5000.0,
            government_max_labor_budget=120000.0,
            government_max_labor_budget_per_household=400.0,
        )

        self.assertEqual(gov._compute_labor_budget(), 120000.0)


if __name__ == "__main__":
    unittest.main()
