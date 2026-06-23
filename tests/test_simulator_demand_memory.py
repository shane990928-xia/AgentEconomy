import unittest

from agenteconomy.simulation.simulator import Simulator
from config.config import SimulationConfig


class DemandCapturingFirm:
    industry = "311FT"

    def __init__(self, firm_id="mfg_food"):
        self.firm_id = firm_id
        self.seen_demand_value = None

    async def post_jobs(self, period, current_demand_value=None, use_llm=False, **kwargs):
        self.seen_demand_value = current_demand_value
        return []


class SimulatorDemandMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_post_jobs_uses_last_demand_stats_when_no_current_demand_passed(self):
        firm = DemandCapturingFirm()
        sim = Simulator(SimulationConfig(num_households=1))
        sim.firms = [firm]
        sim.government = None
        sim._last_demand_stats = {
            "by_mfg_firm": {
                "mfg_food": {"qty": 10.0, "value": 1234.5},
            }
        }

        await sim._post_jobs(month=2)

        self.assertEqual(firm.seen_demand_value, 1234.5)

    async def test_post_jobs_passes_current_service_consumption_to_service_firm(self):
        firm = DemandCapturingFirm("svc_HS")
        firm.industry = "HS"
        sim = Simulator(SimulationConfig(num_households=1))
        sim.firms = [firm]
        sim.government = None

        await sim._post_jobs(
            month=2,
            demand_stats={"by_mfg_firm": {}},
            service_stats={"by_industry": {"HS": 2500.0}},
        )

        self.assertEqual(firm.seen_demand_value, 2500.0)

    async def test_post_jobs_uses_retail_demand_for_retail_firm(self):
        firm = DemandCapturingFirm("ret_452")
        firm.industry = "452"
        sim = Simulator(SimulationConfig(num_households=1))
        sim.firms = [firm]
        sim.government = None

        await sim._post_jobs(
            month=2,
            demand_stats={
                "by_mfg_firm": {},
                "by_retail_firm": {
                    "ret_452": {"qty": 10.0, "value": 4321.0},
                },
            },
        )

        self.assertEqual(firm.seen_demand_value, 4321.0)
        self.assertEqual(sim._firm_labor_priority, {"ret_452": 1.0})

    async def test_post_jobs_uses_production_gap_value_as_labor_demand_signal(self):
        firm = DemandCapturingFirm()
        config = SimulationConfig(num_households=1)
        config.firm_labor_backlog_demand_share = 0.5
        sim = Simulator(config)
        sim.firms = [firm]
        sim.government = None
        sim._last_demand_stats = {
            "by_mfg_firm": {
                "mfg_food": {"qty": 1.0, "value": 25.0},
            }
        }
        sim._last_production_gap_value_by_firm = {"mfg_food": 4000.0}

        await sim._post_jobs(month=3)

        self.assertEqual(firm.seen_demand_value, 2000.0)
        self.assertEqual(sim._firm_labor_priority, {"mfg_food": 1.0})

    async def test_post_jobs_normalizes_priority_from_current_demand_and_backlog(self):
        firm_a = DemandCapturingFirm("mfg_a")
        firm_b = DemandCapturingFirm("mfg_b")
        config = SimulationConfig(num_households=1)
        config.firm_labor_backlog_demand_share = 0.5
        sim = Simulator(config)
        sim.firms = [firm_a, firm_b]
        sim.government = None
        sim._last_demand_stats = {
            "by_mfg_firm": {
                "mfg_a": {"qty": 1.0, "value": 1000.0},
                "mfg_b": {"qty": 1.0, "value": 2500.0},
            }
        }
        sim._last_production_gap_value_by_firm = {"mfg_a": 6000.0, "mfg_bad": "not-a-number"}

        await sim._post_jobs(month=3)

        self.assertEqual(firm_a.seen_demand_value, 3000.0)
        self.assertEqual(firm_b.seen_demand_value, 2500.0)
        self.assertAlmostEqual(sim._firm_labor_priority["mfg_a"], 1.0)
        self.assertAlmostEqual(sim._firm_labor_priority["mfg_b"], 2500.0 / 3000.0)
        self.assertNotIn("mfg_bad", sim._firm_labor_priority)

    async def test_production_demand_signal_includes_current_and_history(self):
        sim = Simulator(SimulationConfig(num_households=1))
        # 上月计划需求不再纳入信号(打破产量棘轮)；信号 = max(当前需求, 实际销售, 未满足缺口)。
        sim._last_planned_demand_by_product = {
            "sku_history": 4.0,
            "sku_overlap": 2.0,
        }
        sim._last_sales_by_product = {
            "sku_sales": 3.0,
        }
        sim._last_unmet_demand_by_product = {
            "sku_unmet": 5.0,
            "sku_overlap": 9.0,
        }

        signal = sim._build_production_demand_signal(
            {
                "sku_current": 7.0,
                "sku_overlap": 6.0,
            }
        )

        self.assertEqual(signal["sku_current"], 7.0)
        self.assertNotIn("sku_history", signal)  # 仅来自 planned → 不再进信号
        self.assertEqual(signal["sku_sales"], 3.0)
        self.assertEqual(signal["sku_unmet"], 5.0)
        # sku_overlap: max(current=6, unmet=9) = 9（planned=2 被忽略）
        self.assertEqual(signal["sku_overlap"], 9.0)


if __name__ == "__main__":
    unittest.main()
