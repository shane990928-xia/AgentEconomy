import unittest

from agenteconomy.agent.firm import ManufactureFirm
from agenteconomy.center.LaborMarket import LaborMarket
from agenteconomy.center.Model import Job, JobApplication, LaborHour, MatchedJob


LaborMarketClass = LaborMarket.__ray_metadata__.modified_class


class LaborWageHoursTests(unittest.TestCase):
    def _market_with_match(self, hours_per_period):
        market = LaborMarketClass()
        job = Job.create(
            soc="51-0000",
            title="Production Worker",
            wage_per_hour=20.0,
            firm_id="firm_a",
            hours_per_period=hours_per_period,
        )
        market.matched_jobs = [
            MatchedJob.create(
                job=job,
                average_wage=20.0,
                household_id="hh_1",
                lh_type="head",
                firm_id="firm_a",
            )
        ]
        return market

    def test_monthly_hours_are_not_multiplied_by_weeks(self):
        market = self._market_with_match(hours_per_period=160.0)

        wage_info = market.calculate_wage_for_matched_job(market.matched_jobs[0])
        self.assertEqual(wage_info["hours_per_period"], 160.0)
        self.assertEqual(wage_info["monthly_gross_wage"], 3200.0)
        self.assertEqual(market.get_firm_labor_cost("firm_a"), 3200.0)

        _, firm_totals = market.calculate_all_wages()
        self.assertEqual(firm_totals["firm_a"], 3200.0)

    def test_missing_hours_per_period_keeps_weekly_legacy_default(self):
        market = self._market_with_match(hours_per_period=None)

        wage_info = market.calculate_wage_for_matched_job(market.matched_jobs[0])
        self.assertEqual(wage_info["hours_per_period"], 160.0)
        self.assertEqual(wage_info["monthly_gross_wage"], 3200.0)
        self.assertEqual(market.get_firm_labor_cost("firm_a"), 3200.0)

    def test_firm_labor_cost_uses_labor_market_monthly_hours(self):
        market = self._market_with_match(hours_per_period=160.0)
        firm = ManufactureFirm(
            firm_id="firm_a",
            name="Food Producer",
            industry="311FT",
            labor_market=market,
        )

        labor_cost = firm.calculate_labor_cost()
        cost_breakdown = firm.calculate_total_cost(
            intermediate_goods_cost=0.0,
            abstract_resources_cost=0.0,
            labor_cost=labor_cost,
            tax_cost=0.0,
        )

        self.assertEqual(labor_cost, 3200.0)
        self.assertEqual(cost_breakdown["labor"], 3200.0)
        self.assertEqual(cost_breakdown["total_cost"], 3200.0)

    def test_layoff_to_zero_budget_releases_worker(self):
        market = self._market_with_match(hours_per_period=160.0)
        worker = market.matched_jobs[0]
        market.matched_workers.add((worker.household_id, worker.lh_type))

        result = market.layoff_to_budget(
            firm_id="firm_a",
            target_wage_cap=0.0,
            reason="no_revenue",
            month=2,
        )

        self.assertEqual(len(result["layoffs"]), 1)
        self.assertEqual(result["new_wage_bill"], 0.0)
        self.assertEqual(market.get_firm_wage_bill("firm_a")["employee_count"], 0)
        self.assertNotIn((worker.household_id, worker.lh_type), market.matched_workers)

    def test_layoff_treats_small_budget_as_upper_cap_not_floor(self):
        market = self._market_with_match(hours_per_period=160.0)

        result = market.layoff_to_budget(
            firm_id="firm_a",
            target_wage_cap=10.0,
            reason="tiny_revenue",
            month=2,
        )

        self.assertEqual(len(result["layoffs"]), 1)
        self.assertEqual(result["new_wage_bill"], 0.0)

    def test_highest_wage_offer_policy_overrides_best_loss(self):
        market = LaborMarketClass()
        market.offers = {
            "low_loss": {
                "offer_id": "low_loss",
                "wage_per_hour": 20.0,
                "loss": 0.0,
            },
            "high_wage": {
                "offer_id": "high_wage",
                "wage_per_hour": 35.0,
                "loss": 100.0,
            },
        }

        self.assertEqual(market._select_offer(["low_loss", "high_wage"], "highest_wage"), "high_wage")
        self.assertEqual(market._select_offer(["low_loss", "high_wage"], "best_loss"), "low_loss")

    def test_private_offer_beats_public_fallback_even_with_higher_loss(self):
        market = LaborMarketClass()
        public_job = Job.create(
            soc="43-9061",
            title="Public Service Worker",
            wage_per_hour=15.0,
            firm_id="gov_main",
            job_id="pub_gov_main_2_43-9061",
        )
        private_job = Job.create(
            soc="51-0000",
            title="Production Worker",
            wage_per_hour=20.0,
            firm_id="mfg_food",
        )
        market.job_openings = [public_job, private_job]
        market.offers = {
            "public": {
                "offer_id": "public",
                "job_id": public_job.job_id,
                "firm_id": "gov_main",
                "wage_per_hour": 15.0,
                "loss": 0.0,
            },
            "private": {
                "offer_id": "private",
                "job_id": private_job.job_id,
                "firm_id": "mfg_food",
                "wage_per_hour": 20.0,
                "loss": 1000.0,
            },
        }

        self.assertEqual(market._select_offer(["public", "private"], "best_loss"), "private")

    def test_demand_adjusted_wage_can_choose_high_demand_offer(self):
        market = LaborMarketClass()
        market.offers = {
            "high_wage_low_demand": {
                "offer_id": "high_wage_low_demand",
                "wage_per_hour": 32.0,
                "loss": 10.0,
                "demand_priority": 0.0,
                "demand_wage_bonus": 12.0,
            },
            "lower_wage_high_demand": {
                "offer_id": "lower_wage_high_demand",
                "wage_per_hour": 24.0,
                "loss": 10.0,
                "demand_priority": 1.0,
                "demand_wage_bonus": 12.0,
            },
        }

        self.assertEqual(
            market._select_offer(["high_wage_low_demand", "lower_wage_high_demand"], "highest_wage"),
            "high_wage_low_demand",
        )
        self.assertEqual(
            market._select_offer(["high_wage_low_demand", "lower_wage_high_demand"], "demand_adjusted_wage"),
            "lower_wage_high_demand",
        )

    def test_public_and_regular_government_jobs_with_same_soc_do_not_merge(self):
        market = LaborMarketClass()
        public_job = Job.create(
            soc="43-9061",
            title="Public Service Worker",
            wage_per_hour=15.0,
            firm_id="gov_main",
            job_id="pub_gov_main_2_43-9061",
        )
        regular_job = Job.create(
            soc="43-9061",
            title="Government Clerk",
            wage_per_hour=30.0,
            firm_id="gov_main",
            job_id="regular_gov_main_43-9061",
        )

        market.add_job_position("gov_main", public_job)
        market.add_job_position("gov_main", regular_job)

        self.assertEqual(len(market.job_openings), 2)

    def test_summary_counts_application_records_not_only_jobs(self):
        market = LaborMarketClass()
        job = Job.create(
            soc="51-0000",
            title="Production Worker",
            wage_per_hour=20.0,
            firm_id="mfg_1",
            required_skills={},
            required_abilities={},
        )
        job.positions_available = 1
        market.add_job_position("mfg_1", job)

        for idx in range(3):
            app = JobApplication.create(
                job_id=job.job_id,
                household_id=f"household_{idx}",
                lh_type="head",
                expected_wage=20.0,
                worker_skills={},
                worker_abilities={},
                month=1,
            )
            self.assertTrue(market.submit_application(app))

        summary = market.summary()

        self.assertEqual(summary["total_job_applications"], 3)
        self.assertEqual(summary["jobs_with_applications"], 1)

    def test_public_employment_worker_can_transition_to_private_offer(self):
        market = LaborMarketClass()
        public_job = Job.create(
            soc="43-9061",
            title="Public Service Worker",
            wage_per_hour=15.0,
            firm_id="gov_main",
            hours_per_period=160.0,
            job_id="pub_gov_main_1_43-9061",
            matching_loss_floor=20000.0,
        )
        private_job = Job.create(
            soc="51-0000",
            title="Production Worker",
            wage_per_hour=22.0,
            firm_id="mfg_food",
            hours_per_period=160.0,
        )
        public_job.positions_available = 0
        private_job.positions_available = 1
        market.job_openings = [public_job, private_job]
        labor_hour = LaborHour.create(
            agent_id="hh_1",
            total_hours=160.0,
            template="head",
            skill_profile={},
            ability_profile={},
            lh_type="head",
        )
        labor_hour.is_valid = False
        labor_hour.firm_id = "gov_main"
        market._register_labor_hour(labor_hour)
        market.matched_jobs = [
            MatchedJob.create(
                job=public_job,
                average_wage=15.0,
                household_id="hh_1",
                lh_type="head",
                firm_id="gov_main",
            )
        ]
        market.matched_workers.add(("hh_1", "head"))

        app = JobApplication.create(
            job_id=private_job.job_id,
            household_id="hh_1",
            lh_type="head",
            expected_wage=22.0,
            worker_skills={},
            worker_abilities={},
            month=2,
        )
        self.assertTrue(market.submit_application(app))
        market.make_offers(month=2, reset_existing=True)
        result = market.resolve_offers(month=2, acceptance_policy="highest_wage")

        self.assertEqual(result["accepted"], 1)
        self.assertEqual(len(market.matched_jobs), 1)
        self.assertEqual(market.matched_jobs[0].firm_id, "mfg_food")
        self.assertEqual(market.get_firm_wage_bill("gov_main")["employee_count"], 0)

    def test_matching_loss_floor_keeps_public_fallback_behind_private_job(self):
        market = LaborMarketClass()
        private_job = Job.create(
            soc="51-0000",
            title="Production Worker",
            wage_per_hour=20.0,
            firm_id="mfg_food",
            required_skills={"skill_a": {"mean": 1.0, "std": 1.0, "importance": 1.0}},
            required_abilities={},
        )
        public_job = Job.create(
            soc="43-9061",
            title="Public Service Worker",
            wage_per_hour=15.0,
            firm_id="gov_main",
            job_id="pub_gov_main_2_43-9061",
            matching_loss_floor=20000.0,
        )
        market.job_openings = [public_job, private_job]
        labor_hour = LaborHour.create(
            agent_id="hh_1",
            total_hours=160.0,
            template="head",
            skill_profile={"skill_a": 1.0},
            ability_profile={},
            lh_type="head",
        )

        ranked = market.rank_jobs_for_labor(labor_hour, loss_threshold=float("inf"))

        self.assertEqual(ranked[0][0].firm_id, "mfg_food")
        self.assertGreaterEqual(ranked[-1][1], 20000.0)


if __name__ == "__main__":
    unittest.main()
