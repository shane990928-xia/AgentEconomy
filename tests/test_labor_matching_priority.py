import asyncio
import unittest

from agenteconomy.agent.firm import Firm
from agenteconomy.agent.household import Household, JobMatch
from agenteconomy.center.Model import Job, LaborHour
from agenteconomy.simulation.simulator import Simulator
from config.config import SimulationConfig


def _run(coro):
    """Drive an async firm method to completion in a sync test."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class FakeLaborMarket:
    def __init__(self, matched):
        self._matched = matched

    def get_matched_jobs(self):
        return self._matched


class LaborMatchingPriorityTests(unittest.TestCase):
    def test_demand_priority_beats_government_fallback_and_wage_when_loss_is_close(self):
        sim = Simulator(
            SimulationConfig(
                num_households=1,
                labor_match_demand_priority_weight=5000.0,
            )
        )
        sim._firm_labor_priority = {"mfg_food": 1.0, "svc_low": 0.1}
        gov_job = Job.create(soc="00-0000", title="Public Work", wage_per_hour=50.0, firm_id="gov_main")
        service_job = Job.create(soc="11-0000", title="Service", wage_per_hour=40.0, firm_id="svc_low")
        mfg_job = Job.create(soc="51-0000", title="Production", wage_per_hour=20.0, firm_id="mfg_food")

        matches = sim._prioritize_labor_matches(
            [
                JobMatch(job=gov_job, loss=0.0),
                JobMatch(job=service_job, loss=0.0),
                JobMatch(job=mfg_job, loss=1000.0),
            ]
        )

        self.assertEqual(matches[0].job.firm_id, "mfg_food")
        self.assertEqual(matches[-1].job.firm_id, "gov_main")

    def test_prioritize_labor_jobs_stamps_priority_and_keeps_government_after_private_jobs(self):
        config = SimulationConfig(num_households=1)
        config.labor_offer_demand_wage_bonus = 12.0
        sim = Simulator(config)
        sim._firm_labor_priority = {"mfg_food": 1.0, "svc_low": 0.1}
        gov_job = Job.create(soc="00-0000", title="Public Work", wage_per_hour=80.0, firm_id="gov_main")
        service_job = Job.create(soc="11-0000", title="Service", wage_per_hour=40.0, firm_id="svc_low")
        mfg_job = Job.create(soc="51-0000", title="Production", wage_per_hour=20.0, firm_id="mfg_food")

        jobs = sim._prioritize_labor_jobs([gov_job, service_job, mfg_job])

        self.assertEqual([job.firm_id for job in jobs], ["mfg_food", "svc_low", "gov_main"])
        self.assertEqual(mfg_job.demand_priority, 1.0)
        self.assertEqual(service_job.demand_priority, 0.1)
        self.assertEqual(gov_job.demand_priority, 0.0)
        self.assertEqual(mfg_job.demand_wage_bonus, 12.0)

    def test_refresh_firm_employee_count_preserves_matched_job_hours(self):
        firm = Firm(firm_id="mfg_food", name="Food Producer", industry="311FT")
        sim = Simulator(SimulationConfig(num_households=1))
        sim.firms = [firm]
        sim.government = None
        sim.labor_market = FakeLaborMarket(
            [
                {
                    "firm_id": "mfg_food",
                    "household_id": "hh_1",
                    "lh_type": "head",
                    "hours_per_period": 48.0,
                    "soc": "51-0000",
                    "title": "Production",
                }
            ]
        )

        sim._refresh_firm_employee_count()

        self.assertEqual(firm.employee_count, 1)
        self.assertEqual(len(firm.employee_list), 1)
        self.assertEqual(firm.employee_list[0].total_hours, 48.0)

    def test_refresh_status_keeps_public_employment_available_for_private_search(self):
        sim = Simulator(SimulationConfig(num_households=1))
        household = Household(
            household_id="hh_1",
            name="Household 1",
            description="",
            owner="",
            load_profile=False,
        )
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
        household.labor_hours = [labor_hour]
        sim.households = [household]

        class FakeLaborStatusMarket:
            def get_labor_status_snapshot(self):
                return {
                    "hh_1": {
                        "head": {
                            "employed": True,
                            "firm_id": "gov_main",
                            "job_id": "pub_gov_main_1_43-9061",
                            "public_employment": True,
                        }
                    }
                }

        sim.labor_market = FakeLaborStatusMarket()

        sim._refresh_household_employment_status()

        self.assertEqual(household.csv_values["ER82433"], household._NOT_EMPLOYED_CODE)
        self.assertTrue(labor_hour.is_valid)
        self.assertIsNone(labor_hour.firm_id)
        self.assertEqual(household.list_job_seekers(household.labor_hours), [labor_hour])


class FirmPartTimePostingTests(unittest.TestCase):
    def test_tiny_demand_does_not_get_raised_to_minimum_payroll(self):
        firm = Firm(firm_id="mfg_food", name="Food Producer", industry="311FT")
        firm.compensation_ratio = 0.2

        jobs = _run(firm._decide_job_postings_from_data(
            period=2,
            current_demand_value=50.0,
            min_part_time_hours_per_month=20.0,
            max_startup_part_time_hours_per_month=160.0,
        ))

        self.assertEqual(jobs, [])

    def test_budget_below_full_time_can_post_part_time_startup_job(self):
        firm = Firm(firm_id="mfg_food", name="Food Producer", industry="311FT")
        firm.compensation_ratio = 0.2

        jobs = _run(firm._decide_job_postings_from_data(
            period=2,
            current_demand_value=2000.0,
            min_part_time_hours_per_month=10.0,
            max_startup_part_time_hours_per_month=160.0,
        ))

        self.assertTrue(jobs)
        self.assertEqual(jobs[0].positions_available, 1)
        self.assertGreaterEqual(jobs[0].hours_per_period, 10.0)
        self.assertLess(jobs[0].hours_per_period, 160.0)

    def test_budget_must_cover_minimum_job_threshold(self):
        firm = Firm(firm_id="mfg_food", name="Food Producer", industry="311FT")
        firm.compensation_ratio = 0.2

        jobs = _run(firm._decide_job_postings_from_data(
            period=2,
            current_demand_value=2000.0,
            min_part_time_hours_per_month=10.0,
            max_startup_part_time_hours_per_month=160.0,
            min_job_budget_coverage=3.0,
        ))

        self.assertEqual(jobs, [])

    def test_residual_budget_posts_multiple_affordable_roles_without_exceeding_budget(self):
        firm = Firm(firm_id="mfg_food", name="Food Producer", industry="311FT")
        firm.compensation_ratio = 0.2

        jobs = _run(firm._decide_job_postings_from_data(
            period=2,
            current_demand_value=40000.0,
            min_part_time_hours_per_month=10.0,
            max_startup_part_time_hours_per_month=160.0,
        ))

        planned_wage = sum(
            job.wage_per_hour * (job.hours_per_period or 160.0) * job.positions_available
            for job in jobs
        )
        self.assertGreaterEqual(sum(job.positions_available for job in jobs), 2)
        self.assertLessEqual(planned_wage, 40000.0 * firm.compensation_ratio + 1e-6)
        self.assertEqual(len({job.SOC for job in jobs}), len(jobs))

    def test_cash_is_not_used_for_startup_hiring_by_default(self):
        firm = Firm(firm_id="mfg_food", name="Food Producer", industry="311FT")
        firm.cash = 10000.0
        firm.compensation_ratio = 0.2

        jobs = _run(firm._decide_job_postings_from_data(
            period=1,
            current_demand_value=0.0,
            min_part_time_hours_per_month=10.0,
            max_startup_part_time_hours_per_month=160.0,
        ))

        self.assertEqual(jobs, [])

    def test_cash_startup_hiring_requires_explicit_flag(self):
        firm = Firm(firm_id="mfg_food", name="Food Producer", industry="311FT")
        firm.cash = 10000.0
        firm.compensation_ratio = 0.2

        jobs = _run(firm._decide_job_postings_from_data(
            period=1,
            current_demand_value=0.0,
            min_part_time_hours_per_month=10.0,
            max_startup_part_time_hours_per_month=160.0,
            allow_cash_based_startup_hiring=True,
        ))

        self.assertTrue(jobs)


class FakeLaborMarketForOpenings:
    def __init__(self, snapshot=None, employees=None):
        self.snapshot = {"51-0000": 2, "53-0000": 1} if snapshot is None else dict(snapshot)
        self.employees = list(employees or [])
        self.reductions = []
        self.applied = []

    def get_firm_job_snapshot(self, firm_id):
        return dict(self.snapshot)

    def get_firm_wage_bill(self, firm_id):
        return {
            "total_wage": 0.0,
            "employee_count": len(self.employees),
            "employees": list(self.employees),
        }

    def reduce_job_positions(self, firm_id, soc, reduce_by):
        self.reductions.append((firm_id, soc, reduce_by))
        self.snapshot[soc] = max(0, self.snapshot.get(soc, 0) - reduce_by)
        return reduce_by

    def apply_job_plan(self, firm_id, jobs):
        self.applied.extend(jobs)
        return {job.SOC: int(job.positions_available or 0) for job in jobs}


class FirmJobPostingAlignmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_zero_labor_budget_clears_stale_open_positions(self):
        labor_market = FakeLaborMarketForOpenings()
        firm = Firm(
            firm_id="mfg_food",
            name="Food Producer",
            industry="311FT",
            labor_market=labor_market,
        )

        jobs = await firm.post_jobs(period=2, current_demand_value=0.0, use_llm=False)

        self.assertEqual(jobs, [])
        self.assertEqual(
            sorted(labor_market.reductions),
            [("mfg_food", "51-0000", 2), ("mfg_food", "53-0000", 1)],
        )
        self.assertEqual(labor_market.snapshot, {"51-0000": 0, "53-0000": 0})

    async def test_job_plan_closes_obsolete_soc_openings(self):
        labor_market = FakeLaborMarketForOpenings()
        firm = Firm(
            firm_id="mfg_food",
            name="Food Producer",
            industry="311FT",
            labor_market=labor_market,
        )
        desired_job = Job.create(
            soc="51-0000",
            title="Production Worker",
            wage_per_hour=20.0,
            firm_id="mfg_food",
            hours_per_period=160.0,
        )
        desired_job.positions_available = 3
        async def _fake_decide(**kwargs):
            return [desired_job]
        firm._decide_job_postings_from_data = _fake_decide

        jobs = await firm.post_jobs(period=2, current_demand_value=10000.0, use_llm=False)

        self.assertEqual(labor_market.reductions, [("mfg_food", "53-0000", 1)])
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].SOC, "51-0000")
        self.assertEqual(jobs[0].positions_available, 1)

    async def test_existing_employees_offset_desired_openings(self):
        labor_market = FakeLaborMarketForOpenings(
            snapshot={},
            employees=[
                {"job_SOC": "51-0000", "household_id": "hh_1"},
                {"job_SOC": "51-0000", "household_id": "hh_2"},
            ],
        )
        firm = Firm(
            firm_id="mfg_food",
            name="Food Producer",
            industry="311FT",
            labor_market=labor_market,
        )
        desired_job = Job.create(
            soc="51-0000",
            title="Production Worker",
            wage_per_hour=20.0,
            firm_id="mfg_food",
            hours_per_period=160.0,
        )
        desired_job.positions_available = 2
        async def _fake_decide(**kwargs):
            return [desired_job]
        firm._decide_job_postings_from_data = _fake_decide

        jobs = await firm.post_jobs(period=2, current_demand_value=10000.0, use_llm=False)

        self.assertEqual(jobs, [])
        self.assertEqual(labor_market.applied, [])

    async def test_small_demand_can_post_part_time_job_with_partial_coverage(self):
        labor_market = FakeLaborMarketForOpenings(snapshot={})
        firm = Firm(
            firm_id="mfg_food",
            name="Food Producer",
            industry="311FT",
            labor_market=labor_market,
        )

        jobs = await firm.post_jobs(
            period=2,
            current_demand_value=1200.0,
            min_part_time_hours_per_month=8.0,
            min_job_budget_coverage=0.4,
            use_llm=False,
        )

        self.assertEqual(len(jobs), 1)
        self.assertGreaterEqual(jobs[0].hours_per_period, 8.0)
        self.assertEqual(labor_market.applied, jobs)


if __name__ == "__main__":
    unittest.main()
