import asyncio
import unittest
from types import SimpleNamespace

from agenteconomy.agent.firm import ManufactureFirm
from agenteconomy.center.Model import Job, MatchedJob
from agenteconomy.simulation.simulator import Simulator
from config.config import SimulationConfig


class FakeEconomicCenter:
    def get_all_balances(self):
        return {"firm_a": -25.0}

    def get_transactions(self, month):
        return []

    def get_all_firm_debt_balances(self):
        return {"firm_a": 100.0}


class FakeProductMarket:
    def get_all_products_snapshot(self):
        return [{"product_id": "sku_1", "available_stock": 1.0}]


class FakeProductMarketWithSnapshots:
    def __init__(self, snapshots):
        self.snapshots = dict(snapshots)
        self.requested = []

    def get_product_snapshot(self, product_id):
        self.requested.append(product_id)
        snapshot = self.snapshots.get(product_id)
        return dict(snapshot) if isinstance(snapshot, dict) else None


class FakeSupplyDemandProductMarket(FakeProductMarketWithSnapshots):
    def __init__(self, snapshots):
        super().__init__(snapshots)
        self.supply_demand = {}
        self.adjusted = []

    def record_demand(self, manufacturer_code, demand_qty):
        stats = self.supply_demand.setdefault(str(manufacturer_code), {"demand": 0.0, "supply": 0.0})
        stats["demand"] += float(demand_qty or 0.0)

    def record_supply(self, manufacturer_code, supply_qty):
        stats = self.supply_demand.setdefault(str(manufacturer_code), {"demand": 0.0, "supply": 0.0})
        stats["supply"] += float(supply_qty or 0.0)

    def get_all_supply_demand_stats(self):
        return {code: dict(stats) for code, stats in self.supply_demand.items()}

    def adjust_prices_by_supply_demand(self, manufacturer_code, base_adjustment, max_adjustment):
        self.adjusted.append(str(manufacturer_code))
        return 1


class FakeLaborMarketWithLayoff:
    def __init__(self, firm_id="firm_a"):
        job = Job.create(
            soc="51-0000",
            title="Production Worker",
            wage_per_hour=20.0,
            firm_id=firm_id,
            hours_per_period=160.0,
        )
        self.matched_jobs = [
            MatchedJob.create(
                job=job,
                average_wage=20.0,
                household_id="hh_1",
                lh_type="head",
                firm_id=firm_id,
            )
        ]
        self.target_caps = []

    def get_firm_wage_bill(self, firm_id):
        employees = []
        total = 0.0
        for match in self.matched_jobs:
            if match.firm_id != firm_id:
                continue
            monthly_wage = match.average_wage * float(match.job.hours_per_period or 160.0)
            employees.append(
                {
                    "household_id": match.household_id,
                    "lh_type": match.lh_type,
                    "monthly_wage": monthly_wage,
                }
            )
            total += monthly_wage
        return {"total_wage": total, "employee_count": len(employees), "employees": employees}

    def layoff_to_budget(self, firm_id, target_wage_cap, reason, month, strategy="highest_wage"):
        self.target_caps.append(target_wage_cap)
        before = self.get_firm_wage_bill(firm_id)["total_wage"]
        layoffs = []
        kept = []
        for match in self.matched_jobs:
            monthly_wage = match.average_wage * float(match.job.hours_per_period or 160.0)
            if match.firm_id == firm_id and target_wage_cap <= 0.0:
                layoffs.append({"household_id": match.household_id, "lh_type": match.lh_type})
            else:
                kept.append(match)
        self.matched_jobs = kept
        after = self.get_firm_wage_bill(firm_id)["total_wage"]
        return {"layoffs": layoffs, "saved_wage": before - after, "new_wage_bill": after}

    def get_labor_stats(self):
        return {"total_labor_hours": 1, "total_matched_jobs": len(self.matched_jobs)}


class FakeLaborMarketForWageFailure:
    def get_matched_jobs(self):
        return [
            {
                "firm_id": "firm_a",
                "household_id": "hh_1",
                "lh_type": "head",
                "wage_per_hour": 20.0,
                "hours_per_period": 160.0,
            }
        ]


class FakeEconomicCenterWageFailure:
    def process_wage(self, *args, **kwargs):
        return None


class FakeEconomicCenterForDemandMemory:
    def __init__(self):
        self.records = {
            "sku_1@retailer_a": {"product_id": "sku_1", "qty_short": 4.0},
            "sku_2@retailer_b": {"qty_short": 3.0},
        }

    def query_unmet_demand(self, month):
        return self.records if month == 2 else {}


class FakeEconomicCenterForDefaultedFirms:
    def get_all_firm_credit_defaulted(self):
        return {"firm_a": True}


class FakeEconomicCenterZeroFirmIncome:
    def query_firm_monthly_financials(self, firm_id, month):
        return {"monthly_income": 0.0}


class FakeHouseholdForWageFailure:
    def __init__(self):
        self.rp_income_updates = []
        self.sp_income_updates = []

    def update_rp_income(self, amount):
        self.rp_income_updates.append(amount)

    def update_sp_income(self, amount):
        self.sp_income_updates.append(amount)


class FakeProductionFirm(ManufactureFirm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_plan_kwargs = None

    def build_production_plan(self, **kwargs):
        self.last_plan_kwargs = kwargs
        return SimpleNamespace(
            expected_demand=10.0,
            target_inventory=0.0,
            current_inventory=0.0,
            desired_output=10.0,
            feasible_output=10.0,
            production_gap=0.0,
            diagnostics={"limiting_constraints": [], "binding_constraints": []},
        )

    def produce(self, production_plan, sku_base_prices, period, update_inventory=True):
        return {
            "production_value": 10.0,
            "total_cost": 130.0,
            "cost_breakdown": {
                "intermediate_goods": 20.0,
                "abstract_resources": 10.0,
                "labor": 100.0,
                "taxes": 0.0,
                "total_cost": 130.0,
            },
            "intermediate_by_industry": {},
            "success": True,
        }


class FakeLaborMarketForDefaultClosure:
    def __init__(self):
        self.closed = []
        self.layoff_calls = []
        self.matched_jobs = [
            {
                "firm_id": "firm_a",
                "household_id": "hh_1",
                "lh_type": "head",
                "wage_per_hour": 20.0,
                "hours_per_period": 160.0,
            }
        ]

    def close_firm_positions(self, firm_id, reason="firm_inactive"):
        self.closed.append((firm_id, reason))
        return 2

    def layoff_to_budget(self, firm_id, target_wage_cap, reason, month, strategy="highest_wage"):
        self.layoff_calls.append((firm_id, target_wage_cap, reason, month, strategy))
        self.matched_jobs = [job for job in self.matched_jobs if job["firm_id"] != firm_id]
        return {
            "layoffs": [{"household_id": "hh_1", "lh_type": "head"}],
            "saved_wage": 3200.0,
            "new_wage_bill": 0.0,
        }

    def get_labor_status_snapshot(self):
        return {}

    def get_matched_jobs(self):
        return list(self.matched_jobs)


class FakeLaborMarketForDefaultJobPosting:
    def __init__(self):
        self.closed = []

    def close_firm_positions(self, firm_id, reason="firm_inactive"):
        self.closed.append((firm_id, reason))
        return 1


class FakeAsyncFirm:
    def __init__(self, firm_id, industry="322"):
        self.firm_id = firm_id
        self.industry = industry
        self.post_job_calls = 0

    async def post_jobs(self, **kwargs):
        self.post_job_calls += 1
        return []


class FakeGovernmentForProcurementPlan:
    def plan_procurement_demand(self, period, household_consumption_budget=None):
        return {
            "budget": 50.0,
            "total_planned_value": 40.0,
            "total_planned_qty": 4.0,
            "demand_by_product": {"sku_gov": 4.0, "sku_bad": "bad"},
            "by_industry": {"325": {"planned_value": 40.0}},
            "items_count": 1,
            "success": True,
        }


class FakeGovernmentForProcurementExecution:
    def __init__(self):
        self.last_kwargs = None

    def procure_goods_and_services(self, **kwargs):
        self.last_kwargs = kwargs
        return {"total_spent": 1.0, "by_industry": {}, "items_count": 1, "success": True}


class SimulatorAccountingInvariantTests(unittest.TestCase):
    def test_simulator_passes_firm_debt_balances_to_invariant_check(self):
        sim = Simulator(SimulationConfig(num_households=1))
        sim.economic_center = FakeEconomicCenter()
        sim.product_market = FakeProductMarket()
        sim.firms = [SimpleNamespace(firm_id="firm_a")]

        result = sim._run_accounting_invariants(1)

        self.assertTrue(result["ok"])
        self.assertEqual(result["metrics"]["negative_firm_cash_with_loan_count"], 1)

    def test_firm_profit_pressure_decomposes_wage_and_production_gaps(self):
        sim = Simulator(SimulationConfig(num_households=1))
        sim._firm_by_id = {"firm_a": SimpleNamespace(firm_id="firm_a")}

        result = sim._summarize_firm_profit_pressure(
            production_stats={
                "firm_production_value": {"firm_a": 80.0},
                "firm_production_cost": {"firm_a": 120.0},
            },
            wage_stats={"by_firm": {"firm_a": 100.0}},
            firm_financials={
                "firm_a": {
                    "monthly_income": 60.0,
                    "monthly_expenses": 220.0,
                    "monthly_profit": -160.0,
                }
            },
        )

        aggregate = result["aggregate"]
        row = result["by_firm"]["firm_a"]
        self.assertEqual(aggregate["firm_count"], 1)
        self.assertEqual(aggregate["firms_income_below_wages"], 1)
        self.assertEqual(aggregate["firms_production_cost_above_output"], 1)
        self.assertAlmostEqual(row["sales_gap_to_wages"], 40.0)
        self.assertAlmostEqual(row["sales_gap_to_wages_and_production_cost"], 160.0)
        self.assertAlmostEqual(row["inventory_adjusted_income"], 140.0)
        self.assertAlmostEqual(row["inventory_adjusted_gap_to_wages"], 0.0)
        self.assertAlmostEqual(row["inventory_adjusted_gap_to_wages_and_production_cost"], 80.0)
        self.assertAlmostEqual(row["output_gap_to_production_cost"], 40.0)
        self.assertAlmostEqual(row["income_to_wage_ratio"], 0.6)
        self.assertAlmostEqual(row["inventory_adjusted_income_to_wage_ratio"], 1.4)
        self.assertAlmostEqual(row["production_cost_to_output_ratio"], 1.5)
        self.assertEqual(result["top_sales_gap_to_wages"][0]["firm_id"], "firm_a")

    def test_firm_profit_pressure_uses_none_for_zero_denominator_ratios(self):
        sim = Simulator(SimulationConfig(num_households=1))

        result = sim._summarize_firm_profit_pressure(
            production_stats={"by_firm": {"firm_zero": {"qty": 0.0, "value": 0.0}}},
            wage_stats={"by_firm": {"firm_zero": 0.0}},
            firm_financials={"firm_zero": {"monthly_income": 0.0, "monthly_expenses": 0.0}},
        )

        aggregate = result["aggregate"]
        row = result["by_firm"]["firm_zero"]
        self.assertIsNone(row["income_to_wage_ratio"])
        self.assertIsNone(row["wage_to_income_ratio"])
        self.assertIsNone(row["production_cost_to_output_ratio"])
        self.assertIsNone(aggregate["income_to_wage_ratio"])
        self.assertEqual(result["top_sales_gap_to_wages"], [])

    def test_estimated_unit_cash_cost_uses_manufacturer_price_share(self):
        config = SimulationConfig(num_households=1)
        config.production_unit_cash_cost_share = 0.5
        sim = Simulator(config)

        result = sim._estimate_unit_cash_cost_for_targets(
            [
                {"manufacturer_price": 10.0},
                {"base_manufacturer_price": 20.0},
                {"retail_price": 30.0},
                {"manufacturer_price": 0.0},
            ]
        )

        self.assertAlmostEqual(result, 10.0)

    def test_labor_productivity_is_value_calibrated_for_low_price_skus(self):
        config = SimulationConfig(num_households=1)
        config.production_labor_productivity = 2.5
        config.production_value_calibrated_labor_productivity = True
        sim = Simulator(config)
        firm = ManufactureFirm(firm_id="firm_a", name="Chemical", industry="325")
        firm.compensation_ratio = 0.2
        firm.employee_list = [
            SimpleNamespace(total_hours=160.0, wage_per_hour=10.0)
        ]

        result = sim._estimate_labor_productivity_for_targets(
            firm,
            [{"manufacturer_price": 1.0}],
        )

        self.assertAlmostEqual(result, 50.0)

    def test_labor_productivity_calibration_can_be_disabled(self):
        config = SimulationConfig(num_households=1)
        config.production_labor_productivity = 2.5
        config.production_value_calibrated_labor_productivity = False
        sim = Simulator(config)
        firm = ManufactureFirm(firm_id="firm_a", name="Chemical", industry="325")
        firm.compensation_ratio = 0.2
        firm.employee_list = [
            SimpleNamespace(total_hours=160.0, wage_per_hour=10.0)
        ]

        result = sim._estimate_labor_productivity_for_targets(
            firm,
            [{"manufacturer_price": 1.0}],
        )

        self.assertAlmostEqual(result, 2.5)

    def test_production_stats_exclude_labor_from_intermediate_cost(self):
        config = SimulationConfig(num_households=1)
        config.active_production_planning = True
        config.production_apply_capacity_constraints = False
        sim = Simulator(config)
        firm = FakeProductionFirm(firm_id="mfg_322", name="Paper", industry="322")
        sim.manufacturers_by_industry = {"322": firm}
        sim._firm_by_id = {firm.firm_id: firm}
        sim.product_market = SimpleNamespace()
        snapshot_cache = {
            "sku_1": {
                "product_id": "sku_1",
                "available_stock": 0,
                "manufacturer_code": "322",
                "manufacturer_price": 1.0,
            }
        }

        stats = sim._ensure_production(
            {"sku_1": 10.0},
            snapshot_cache,
            month=1,
            record_transactions=True,
        )

        self.assertAlmostEqual(stats["firm_production_value"]["mfg_322"], 10.0)
        self.assertAlmostEqual(stats["firm_production_cost"]["mfg_322"], 30.0)
        self.assertAlmostEqual(stats["total_production_cost"], 30.0)
        self.assertAlmostEqual(stats["firm_labor_cost_in_production"]["mfg_322"], 100.0)
        self.assertAlmostEqual(stats["firm_total_production_cost_with_labor"]["mfg_322"], 130.0)

    def test_production_stats_use_actual_plan_after_intermediate_bottleneck(self):
        class ClippedProductionFirm(FakeProductionFirm):
            def produce(self, production_plan, sku_base_prices, period, update_inventory=True):
                return {
                    "production_plan": {"sku_1": 4},
                    "production_value": 4.0,
                    "total_cost": 52.0,
                    "cost_breakdown": {
                        "intermediate_goods": 20.0,
                        "abstract_resources": 12.0,
                        "labor": 20.0,
                        "taxes": 0.0,
                        "total_cost": 52.0,
                    },
                    "intermediate_by_industry": {"311FT": 20.0},
                    "success": True,
                }

        config = SimulationConfig(num_households=1)
        config.active_production_planning = True
        config.production_apply_capacity_constraints = False
        sim = Simulator(config)
        firm = ClippedProductionFirm(firm_id="mfg_322", name="Paper", industry="322")
        sim.manufacturers_by_industry = {"322": firm}
        sim._firm_by_id = {firm.firm_id: firm}
        sim.product_market = SimpleNamespace(record_raw_material_demand=lambda industry_code, demand_value: None)
        snapshot_cache = {
            "sku_1": {
                "product_id": "sku_1",
                "available_stock": 0,
                "manufacturer_code": "322",
                "manufacturer_price": 1.0,
            }
        }

        stats = sim._ensure_production(
            {"sku_1": 10.0},
            snapshot_cache,
            month=1,
            record_transactions=True,
        )

        self.assertEqual(stats["by_firm"]["mfg_322"], {"qty": 4.0, "value": 4.0})
        self.assertAlmostEqual(stats["total_qty"], 4.0)
        self.assertAlmostEqual(stats["total_value"], 4.0)
        self.assertAlmostEqual(stats["firm_production_value"]["mfg_322"], 4.0)
        self.assertAlmostEqual(stats["total_output_value"], 4.0)

    def test_production_planning_uses_sales_and_unmet_demand_history(self):
        config = SimulationConfig(num_households=1)
        config.active_production_planning = True
        config.production_apply_capacity_constraints = False
        sim = Simulator(config)
        firm = FakeProductionFirm(firm_id="mfg_322", name="Paper", industry="322")
        sim.manufacturers_by_industry = {"322": firm}
        sim._firm_by_id = {firm.firm_id: firm}
        sim.product_market = SimpleNamespace()
        snapshot_cache = {
            "sku_1": {
                "product_id": "sku_1",
                "available_stock": 0,
                "manufacturer_code": "322",
                "manufacturer_price": 1.0,
            }
        }

        sim._ensure_production(
            {"sku_1": 10.0},
            snapshot_cache,
            month=2,
            record_transactions=False,
            sales_history_by_product={"sku_1": 6.0},
            unmet_demand_by_product={"sku_1": 4.0},
        )

        self.assertEqual(firm.last_plan_kwargs["sales_history"], [6.0])
        self.assertEqual(firm.last_plan_kwargs["unmet_demand_history"], [4.0])

    def test_production_planning_records_gap_value_for_labor_backlog_signal(self):
        class GapProductionFirm(FakeProductionFirm):
            def build_production_plan(self, **kwargs):
                self.last_plan_kwargs = kwargs
                return SimpleNamespace(
                    expected_demand=20.0,
                    target_inventory=20.0,
                    current_inventory=0.0,
                    desired_output=20.0,
                    feasible_output=5.0,
                    production_gap=15.0,
                    diagnostics={"limiting_constraints": ["labor"], "binding_constraints": ["labor"]},
                )

        config = SimulationConfig(num_households=1)
        config.active_production_planning = True
        config.production_apply_capacity_constraints = False
        sim = Simulator(config)
        firm = GapProductionFirm(firm_id="mfg_322", name="Paper", industry="322")
        sim.manufacturers_by_industry = {"322": firm}
        sim._firm_by_id = {firm.firm_id: firm}
        sim.product_market = SimpleNamespace()
        snapshot_cache = {
            "sku_1": {
                "product_id": "sku_1",
                "available_stock": 0,
                "manufacturer_code": "322",
                "manufacturer_price": 3.0,
            }
        }

        stats = sim._ensure_production(
            {"sku_1": 10.0},
            snapshot_cache,
            month=3,
            record_transactions=False,
        )

        planning = stats["planning"]["by_firm"]["mfg_322"]
        self.assertAlmostEqual(planning["production_gap"], 15.0)
        self.assertAlmostEqual(planning["production_gap_value"], 45.0)
        self.assertEqual(sim._last_production_gap_value_by_firm, {"mfg_322": 45.0})

    def test_demand_memory_records_planned_and_unmet_by_product(self):
        sim = Simulator(SimulationConfig(num_households=1))
        sim.economic_center = FakeEconomicCenterForDemandMemory()

        sim._update_demand_memory({"sku_1": 10, "sku_2": 0, "sku_3": "bad"}, econ_month=2)

        self.assertEqual(sim._last_planned_demand_by_product, {"sku_1": 10.0})
        self.assertEqual(sim._last_unmet_demand_by_product, {"sku_1": 4.0, "sku_2": 3.0})

    def test_production_demand_signal_prefers_last_planned_demand_over_low_sales(self):
        sim = Simulator(SimulationConfig(num_households=1))
        sim._last_planned_demand_by_product = {"sku_1": 10.0}
        sim._last_sales_by_product = {"sku_1": 2.0}

        result = sim._build_production_demand_signal({"sku_1": 1.0})

        self.assertEqual(result, {"sku_1": 10.0})

    def test_product_demand_merge_sums_valid_sources(self):
        sim = Simulator(SimulationConfig(num_households=1))

        result = sim._merge_product_demands(
            {"sku_1": 2, "sku_2": "bad"},
            {"sku_1": 3.5, "sku_3": -1},
            None,
        )

        self.assertEqual(result, {"sku_1": 5.5})

    def test_supply_demand_tracking_uses_manufacturer_code_for_supply_and_price_adjustment(self):
        sim = Simulator(SimulationConfig(num_households=1))
        market = FakeSupplyDemandProductMarket(
            {
                "sku_1": {
                    "product_id": "sku_1",
                    "available_stock": 0,
                    "manufacturer_code": "322",
                    "manufacturer_price": 1.0,
                }
            }
        )
        firm = FakeProductionFirm(firm_id="mfg_322", name="Paper", industry="322")
        sim.product_market = market
        sim._firm_by_id = {firm.firm_id: firm}
        sim._industry_code_to_name = {"322": "Paper products"}
        snapshot_cache = {}

        sim._record_demand_to_market({"sku_1": 10.0}, snapshot_cache)
        sim._record_supply_and_adjust_prices(
            {
                "by_firm": {"mfg_322": {"qty": 4.0, "value": 4.0}},
                "total_qty": 4.0,
                "raw_material_demand": {},
            },
            econ_month=2,
        )

        self.assertEqual(market.supply_demand, {"322": {"demand": 10.0, "supply": 4.0}})
        self.assertEqual(market.adjusted, ["322"])

    def test_government_procurement_plan_is_normalized_for_production_demand(self):
        sim = Simulator(SimulationConfig(num_households=1))
        sim.government = FakeGovernmentForProcurementPlan()

        result = sim._plan_government_procurement_demand(
            month=2,
            household_consumption_budget=100.0,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["demand_by_product"], {"sku_gov": 4.0})
        self.assertEqual(result["items_count"], 1)

    def test_government_procurement_execution_receives_planned_demand(self):
        sim = Simulator(SimulationConfig(num_households=1))
        sim.government = FakeGovernmentForProcurementExecution()

        result = asyncio.run(
            sim._execute_government_procurement(
                month=3,
                household_consumption_budget=100.0,
                planned_demand_by_product={"sku_gov": 4.0},
            )
        )

        self.assertTrue(result["success"])
        self.assertEqual(
            sim.government.last_kwargs["planned_demand_by_product"],
            {"sku_gov": 4.0},
        )

    def test_government_plan_sku_missing_from_household_cache_can_be_produced(self):
        config = SimulationConfig(num_households=1)
        config.active_production_planning = True
        config.production_apply_capacity_constraints = False
        sim = Simulator(config)
        firm = FakeProductionFirm(firm_id="mfg_325", name="Chemical", industry="325")
        sim.manufacturers_by_industry = {"325": firm}
        sim._firm_by_id = {firm.firm_id: firm}
        sim.product_market = FakeProductMarketWithSnapshots(
            {
                "sku_gov": {
                    "product_id": "sku_gov",
                    "available_stock": 0,
                    "manufacturer_code": "325",
                    "manufacturer_price": 10.0,
                }
            }
        )
        snapshot_cache = {}

        stats = sim._ensure_production(
            {"sku_gov": 4.0},
            snapshot_cache,
            month=2,
            record_transactions=False,
        )

        self.assertEqual(sim.product_market.requested, ["sku_gov"])
        self.assertEqual(firm.last_plan_kwargs["fallback_expected_demand"], 4.0)
        self.assertEqual(stats["by_firm"]["mfg_325"]["qty"], 8.0)

    def test_zero_revenue_private_firm_can_layoff_to_zero_budget(self):
        config = SimulationConfig(num_households=1)
        config.firm_layoff_min_wage_cap = 0.0
        config.firm_layoff_min_employees_to_keep = 0
        sim = Simulator(config)
        firm = SimpleNamespace(firm_id="firm_a", compensation_ratio=0.2, employee_count=1)
        sim.firms = [firm]
        sim.labor_market = FakeLaborMarketWithLayoff()
        sim.government = None
        sim.economic_center = None

        result = asyncio.run(sim._process_layoffs(month=1))

        self.assertEqual(result["total_layoffs"], 1)
        self.assertEqual(firm.employee_count, 0)
        self.assertEqual(sim.labor_market.target_caps, [0.0])

    def test_recent_demand_supports_layoff_wage_cap_when_sales_are_zero(self):
        config = SimulationConfig(num_households=1)
        sim = Simulator(config)
        firm = SimpleNamespace(firm_id="firm_a", compensation_ratio=0.2, employee_count=1)
        sim.firms = [firm]
        sim.labor_market = FakeLaborMarketWithLayoff()
        sim.government = None
        sim.economic_center = FakeEconomicCenterZeroFirmIncome()
        sim._last_demand_stats = {
            "by_mfg_firm": {
                "firm_a": {"value": 20000.0},
            }
        }

        result = asyncio.run(sim._process_layoffs(month=3))

        self.assertEqual(result["total_layoffs"], 0)
        self.assertEqual(firm.employee_count, 1)
        self.assertEqual(sim.labor_market.target_caps, [])

    def test_layoff_tolerance_prevents_churn_on_small_wage_cap_gap(self):
        config = SimulationConfig(num_households=1)
        config.firm_layoff_wage_cap_tolerance = 0.25
        sim = Simulator(config)
        firm = SimpleNamespace(firm_id="firm_a", industry="311FT", compensation_ratio=0.2, employee_count=1)
        sim.firms = [firm]
        sim.labor_market = FakeLaborMarketWithLayoff()
        sim.government = None
        sim.economic_center = None
        sim._last_demand_stats = {"by_mfg_firm": {"firm_a": {"value": 14000.0}}}

        result = asyncio.run(sim._process_layoffs(month=3))

        self.assertEqual(result["total_layoffs"], 0)
        self.assertEqual(firm.employee_count, 1)
        self.assertEqual(sim.labor_market.target_caps, [])

    def test_last_service_revenue_protects_service_firm_layoff_budget(self):
        config = SimulationConfig(num_households=1)
        sim = Simulator(config)
        firm = SimpleNamespace(firm_id="svc_HS", industry="HS", compensation_ratio=0.2, employee_count=1)
        sim.firms = [firm]
        sim.labor_market = FakeLaborMarketWithLayoff(firm_id="svc_HS")
        sim.government = None
        sim.economic_center = None
        sim._last_service_value_by_industry = {"HS": 20000.0}

        result = asyncio.run(sim._process_layoffs(month=3))

        self.assertEqual(result["total_layoffs"], 0)
        self.assertEqual(firm.employee_count, 1)
        self.assertEqual(sim.labor_market.target_caps, [])

    def test_pay_wages_counts_only_successful_transactions(self):
        sim = Simulator(SimulationConfig(num_households=1))
        hh = FakeHouseholdForWageFailure()
        sim._household_by_id = {"hh_1": hh}
        sim.labor_market = FakeLaborMarketForWageFailure()
        sim.economic_center = FakeEconomicCenterWageFailure()

        result = sim._pay_wages(month=1, record_transactions=True)

        self.assertEqual(result["total"], 0.0)
        self.assertEqual(result["by_firm"], {})
        self.assertEqual(hh.rp_income_updates, [])

    def test_credit_default_labor_closure_lays_off_workers_and_closes_openings(self):
        sim = Simulator(SimulationConfig(num_households=1))
        firm = SimpleNamespace(firm_id="firm_a", employee_count=1, employee_list=["worker"])
        sim._firm_by_id = {"firm_a": firm}
        sim.firms = [firm]
        sim.households = []
        sim.labor_market = FakeLaborMarketForDefaultClosure()
        sim.economic_center = FakeEconomicCenterForDefaultedFirms()

        result = sim._apply_credit_default_labor_closure(month=3, firm_credit_stats={})

        self.assertEqual(result["defaulted_firms"], ["firm_a"])
        self.assertEqual(result["total_layoffs"], 1)
        self.assertEqual(result["closed_positions"], 2)
        self.assertEqual(firm.employee_count, 0)
        self.assertEqual(firm.employee_list, [])
        self.assertEqual(
            sim.labor_market.layoff_calls,
            [("firm_a", 0.0, "credit_default", 3, "highest_wage")],
        )

    def test_defaulted_firm_is_skipped_for_job_posting(self):
        sim = Simulator(SimulationConfig(num_households=1))
        defaulted_firm = FakeAsyncFirm("firm_a")
        active_firm = FakeAsyncFirm("firm_b")
        sim.firms = [defaulted_firm, active_firm]
        sim.economic_center = FakeEconomicCenterForDefaultedFirms()
        sim.labor_market = FakeLaborMarketForDefaultJobPosting()
        sim.government = None

        asyncio.run(sim._post_jobs(month=4, demand_stats={}))

        self.assertEqual(defaulted_firm.post_job_calls, 0)
        self.assertEqual(active_firm.post_job_calls, 1)
        self.assertEqual(sim.labor_market.closed, [("firm_a", "credit_default")])

    def test_laid_off_firm_with_no_demand_is_skipped_for_same_month_job_posting(self):
        sim = Simulator(SimulationConfig(num_households=1))
        laid_off_firm = FakeAsyncFirm("firm_a")
        active_firm = FakeAsyncFirm("firm_b")
        sim.firms = [laid_off_firm, active_firm]
        sim.economic_center = None
        sim.labor_market = FakeLaborMarketForDefaultJobPosting()
        sim.government = None

        asyncio.run(
            sim._post_jobs(
                month=4,
                demand_stats={
                    "by_mfg_firm": {
                        "firm_b": {"value": 10000.0},
                    }
                },
                skip_firm_ids={"firm_a"},
            )
        )

        self.assertEqual(laid_off_firm.post_job_calls, 0)
        self.assertEqual(active_firm.post_job_calls, 1)
        self.assertEqual(sim.labor_market.closed, [("firm_a", "layoff_cooldown")])

    def test_laid_off_firm_with_current_demand_can_repost_jobs(self):
        sim = Simulator(SimulationConfig(num_households=1))
        laid_off_firm = FakeAsyncFirm("firm_a")
        active_firm = FakeAsyncFirm("firm_b")
        sim.firms = [laid_off_firm, active_firm]
        sim.economic_center = None
        sim.labor_market = FakeLaborMarketForDefaultJobPosting()
        sim.government = None

        asyncio.run(
            sim._post_jobs(
                month=4,
                demand_stats={
                    "by_mfg_firm": {
                        "firm_a": {"value": 10000.0},
                        "firm_b": {"value": 10000.0},
                    }
                },
                skip_firm_ids={"firm_a"},
            )
        )

        self.assertEqual(laid_off_firm.post_job_calls, 1)
        self.assertEqual(active_firm.post_job_calls, 1)
        self.assertEqual(sim.labor_market.closed, [])

    def test_defaulted_manufacturer_is_skipped_for_production(self):
        config = SimulationConfig(num_households=1)
        config.active_production_planning = True
        sim = Simulator(config)
        firm = FakeProductionFirm(firm_id="firm_a", name="Paper", industry="322")
        sim.manufacturers_by_industry = {"322": firm}
        sim._firm_by_id = {firm.firm_id: firm}
        sim.economic_center = FakeEconomicCenterForDefaultedFirms()
        sim.product_market = SimpleNamespace()
        snapshot_cache = {
            "sku_1": {
                "product_id": "sku_1",
                "available_stock": 0,
                "manufacturer_code": "322",
                "manufacturer_price": 1.0,
            }
        }

        stats = sim._ensure_production({"sku_1": 10.0}, snapshot_cache, month=4, record_transactions=False)

        self.assertIsNone(firm.last_plan_kwargs)
        self.assertEqual(stats["total_qty"], 0.0)
        self.assertTrue(stats["planning"]["by_firm"]["firm_a"]["defaulted"])


if __name__ == "__main__":
    unittest.main()
