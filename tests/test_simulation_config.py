import unittest
import tempfile
from pathlib import Path

from config.config import SimulationConfig


class SimulationConfigTests(unittest.TestCase):
    def test_default_production_capacity_constraints_are_enabled(self):
        config = SimulationConfig()

        self.assertTrue(config.production_apply_capacity_constraints)
        self.assertTrue(config.production_value_calibrated_labor_productivity)
        self.assertGreater(config.production_labor_productivity, 1.0)
        self.assertEqual(config.government_demand_injection_ratio, 0.30)
        self.assertEqual(config.government_min_procurement_budget, 150000.0)
        self.assertEqual(config.government_max_labor_budget, 120000.0)
        self.assertEqual(config.government_max_labor_budget_per_household, 400.0)
        self.assertEqual(config.public_employment_target_unemployment, 0.15)
        self.assertEqual(config.public_employment_max_monthly_jobs, 40)
        self.assertEqual(config.public_employment_max_new_job_share, 0.10)
        self.assertEqual(config.public_employment_max_stock_share, 0.20)
        self.assertTrue(config.consumption_use_llm)
        self.assertEqual(config.consumption_llm_mode, "monthly")
        self.assertEqual(config.consumption_profile_refresh_months, 12)
        self.assertEqual(config.labor_match_offer_backups, 3)
        self.assertEqual(config.labor_offer_acceptance_policy, "best_loss")
        self.assertEqual(config.labor_offer_demand_wage_bonus, 0.0)
        self.assertEqual(config.firm_labor_backlog_demand_share, 0.5)
        self.assertTrue(config.retail_channel_diversification_enabled)
        self.assertEqual(config.retail_channel_max_household_share, 0.70)
        self.assertEqual(config.retail_channel_min_purchase_count, 3)

    def test_normal_yaml_keeps_capacity_constraints_enabled(self):
        config = SimulationConfig.from_yaml("config/config_normal.yaml")

        self.assertTrue(config.production_apply_capacity_constraints)
        self.assertGreater(config.production_labor_productivity, 1.0)
        self.assertGreater(config.production_unit_cash_cost_share, 0.0)
        self.assertGreaterEqual(config.labor_match_top_k, 30)
        self.assertGreaterEqual(config.labor_match_offer_backups, 10)
        self.assertEqual(config.labor_offer_acceptance_policy, "demand_adjusted_wage")
        self.assertGreater(config.labor_offer_demand_wage_bonus, 0.0)
        self.assertGreater(config.firm_labor_backlog_demand_share, 0.0)
        self.assertGreater(config.firm_min_part_time_hours_per_month, 0.0)
        self.assertGreater(config.firm_min_job_budget_coverage, 0.0)
        self.assertLess(config.firm_min_job_budget_coverage, 1.0)
        self.assertFalse(config.firm_allow_cash_based_startup_hiring)
        self.assertEqual(config.firm_layoff_min_wage_cap, 0.0)
        self.assertEqual(config.firm_layoff_min_employees_to_keep, 0)
        self.assertGreater(config.firm_layoff_wage_cap_tolerance, 0.0)
        self.assertTrue(config.retail_channel_diversification_enabled)
        self.assertLess(config.retail_channel_max_household_share, 1.0)
        self.assertGreaterEqual(config.retail_channel_min_purchase_count, 2)
        self.assertEqual(config.government_demand_injection_ratio, 0.30)
        self.assertEqual(config.government_min_procurement_budget_per_household, 500.0)
        self.assertEqual(config.government_max_labor_budget, 120000.0)
        self.assertEqual(config.government_max_labor_budget_per_household, 400.0)
        self.assertEqual(config.public_employment_start_period, 2)
        self.assertEqual(config.public_employment_max_monthly_job_share, 0.10)
        self.assertEqual(config.public_employment_max_new_job_share, 0.05)
        self.assertEqual(config.public_employment_max_stock_share, 0.20)
        self.assertTrue(config.consumption_use_llm)
        self.assertEqual(config.consumption_llm_mode, "profile")
        self.assertEqual(config.consumption_profile_refresh_months, 12)

    def test_yaml_parses_consumption_llm_mode_and_profile_refresh_months(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(
                "simulation:\n"
                "  consumption_llm_mode: profile\n"
                "  consumption_profile_refresh_months: 6\n",
                encoding="utf-8",
            )

            config = SimulationConfig.from_yaml(str(path))

        self.assertTrue(config.consumption_use_llm)
        self.assertEqual(config.consumption_llm_mode, "profile")
        self.assertEqual(config.consumption_profile_refresh_months, 6)


if __name__ == "__main__":
    unittest.main()
