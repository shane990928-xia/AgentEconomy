import unittest

from agenteconomy.simulation.simulator import Simulator
from config.config import SimulationConfig


class FakeProductMarket:
    def __init__(self, snapshots):
        self._snapshots = snapshots
        self.seller_stock = {}
        self.purchase_calls = []
        self.manufacturer_purchase_calls = []
        self.retailer_inventory_receipts = []
        self.restored_retailer = []
        self.restored_manufacturer = []

    def get_product_snapshot(self, product_id):
        return self._snapshots.get(product_id)

    def get_seller_stock(self, product_id, seller_id=None, seller_inventory_required=False):
        if seller_inventory_required:
            return float(self.seller_stock.get((seller_id, product_id), 0.0))
        snapshot = self._snapshots.get(product_id) or {}
        return float(snapshot.get("available_stock", 0.0) or 0.0)

    def purchase_from_seller_stock(self, product_id, seller_id, requested_quantity, seller_inventory_required=False):
        self.purchase_calls.append((product_id, seller_id, requested_quantity, seller_inventory_required))
        available = self.get_seller_stock(product_id, seller_id, seller_inventory_required)
        actual = min(float(requested_quantity or 0.0), available)
        if seller_inventory_required:
            self.seller_stock[(seller_id, product_id)] = max(0.0, available - actual)
            source = "retailer"
        else:
            snapshot = self._snapshots.get(product_id) or {}
            snapshot["available_stock"] = max(0.0, available - actual)
            source = "manufacturer"
        return {"actual_quantity": actual, "source": source}

    def purchase_manufacturer_stock(self, product_id, quantity):
        self.manufacturer_purchase_calls.append((product_id, quantity))
        snapshot = self._snapshots.get(product_id) or {}
        available = float(snapshot.get("available_stock", 0.0) or 0.0)
        actual = min(float(quantity or 0.0), available)
        snapshot["available_stock"] = max(0.0, available - actual)
        return {
            "actual_quantity": actual,
            "available_after": snapshot["available_stock"],
        }

    def receive_retailer_inventory(self, retailer_id, product_id, quantity, unit_cost=None):
        self.retailer_inventory_receipts.append((retailer_id, product_id, quantity))
        key = (retailer_id, product_id)
        self.seller_stock[key] = float(self.seller_stock.get(key, 0.0) or 0.0) + float(quantity or 0.0)
        return {"success": True, "retailer_id": retailer_id, "product_id": product_id, "quantity": quantity}

    def restore_retailer_inventory(self, retailer_id, product_id, quantity):
        self.restored_retailer.append((retailer_id, product_id, quantity))
        key = (retailer_id, product_id)
        self.seller_stock[key] = float(self.seller_stock.get(key, 0.0) or 0.0) + float(quantity or 0.0)
        return {"success": True, "retailer_id": retailer_id, "product_id": product_id, "quantity": quantity}

    def restore_manufacturer_stock(self, product_id, quantity):
        self.restored_manufacturer.append((product_id, quantity))
        snapshot = self._snapshots.get(product_id) or {}
        snapshot["available_stock"] = float(snapshot.get("available_stock", 0.0) or 0.0) + float(quantity or 0.0)
        return {"success": True, "available_after": snapshot["available_stock"]}


class FakeEconomicCenterForUnmet:
    def __init__(
        self,
        balances=None,
        purchase_success=True,
        wholesale_success=True,
        raise_on_purchase=False,
        raise_on_wholesale=False,
    ):
        self.balances = balances or {}
        self.purchase_success = purchase_success
        self.wholesale_success = wholesale_success
        self.raise_on_purchase = raise_on_purchase
        self.raise_on_wholesale = raise_on_wholesale
        self.purchases = []
        self.wholesale_calls = []
        self.unmet = []

    def query_balance(self, household_id):
        return self.balances.get(household_id, 0.0)

    def record_unmet_demand(
        self,
        month,
        buyer_id,
        seller_id,
        product_id,
        product_name,
        quantity_requested,
        available_stock,
    ):
        self.unmet.append(
            {
                "month": month,
                "buyer_id": buyer_id,
                "seller_id": seller_id,
                "product_id": product_id,
                "quantity_requested": quantity_requested,
                "available_stock": available_stock,
            }
        )

    def process_purchase(
        self,
        month,
        buyer_id,
        seller_id,
        amount,
        quantity,
        product_id,
        product_name,
        unit_price,
        base_unit_price=None,
    ):
        if self.raise_on_purchase:
            raise RuntimeError("purchase settlement failed")
        self.purchases.append((buyer_id, seller_id, product_id, amount, quantity))
        return "tx_1" if self.purchase_success else None

    def process_wholesale(
        self,
        month,
        retailer_id,
        manufacturer_id,
        amount,
        quantity,
        product_id,
        product_name,
        unit_price,
    ):
        if self.raise_on_wholesale:
            raise RuntimeError("wholesale settlement failed")
        self.wholesale_calls.append((retailer_id, manufacturer_id, product_id, amount, quantity))
        return "tx_wholesale" if self.wholesale_success else None


class FakeHousehold:
    def __init__(self, household_id="hh_1", wealth=1000.0):
        self.household_id = household_id
        self.csv_values = {"ER85692": float(wealth)}
        self.apply_consumption_calls = 0
        self.last_month_consumption = None

    def apply_consumption(self, spending_by_bucket):
        self.apply_consumption_calls += 1
        raise AssertionError("planned budgets must not mutate household consumption state")

    def update_last_month_consumption(self, actual_consumption):
        self.last_month_consumption = float(actual_consumption)


class SimulatorConsumptionStateTests(unittest.TestCase):
    def test_build_orders_does_not_apply_planned_consumption(self):
        sim = Simulator(SimulationConfig(num_households=1, debug_logging=False))
        sim.product_market = FakeProductMarket(
            {
                "sku_1": {
                    "product_id": "sku_1",
                    "name": "Test SKU",
                    "retail_price": 10.0,
                    "available_stock": 10,
                    "manufacturer_code": "311FT",
                    "retailer_code": "445",
                }
            }
        )
        hh = FakeHousehold(wealth=1000.0)
        result = {
            "step0": {
                "budgets": {
                    "Retail merchandise": 100.0,
                    "housing": 300.0,
                }
            },
            "step3": {
                "purchases": [
                    {"product_id": "sku_1", "allocated_budget": 100.0},
                ]
            },
        }

        demand_by_product, orders_by_household, _, _ = sim._build_orders([(hh, result)])

        self.assertEqual(hh.apply_consumption_calls, 0)
        self.assertEqual(hh.csv_values["ER85692"], 1000.0)
        self.assertEqual(dict(demand_by_product), {"sku_1": 10.0})
        self.assertEqual(orders_by_household[0][2], 100.0)

    def test_build_orders_diversifies_retail_channel_when_household_basket_is_concentrated(self):
        config = SimulationConfig(num_households=1, debug_logging=False)
        config.retail_channel_diversification_enabled = True
        config.retail_channel_max_household_share = 0.50
        config.retail_channel_min_purchase_count = 2
        sim = Simulator(config)
        sim.product_market = FakeProductMarket(
            {
                "sku_a": {
                    "product_id": "sku_a",
                    "name": "A",
                    "retail_price": 10.0,
                    "base_retail_price": 10.0,
                    "manufacturer_price": 6.0,
                    "base_manufacturer_price": 6.0,
                    "available_stock": 100,
                    "manufacturer_code": "311FT",
                    "retailer_code": "452",
                },
                "sku_b": {
                    "product_id": "sku_b",
                    "name": "B",
                    "retail_price": 10.0,
                    "base_retail_price": 10.0,
                    "manufacturer_price": 6.0,
                    "base_manufacturer_price": 6.0,
                    "available_stock": 100,
                    "manufacturer_code": "311FT",
                    "retailer_code": "452",
                },
            }
        )
        sim.retailers_by_industry = {
            "452": type("Retailer", (), {"firm_id": "ret_452", "cash": 0.0})(),
            "445": type("Retailer", (), {"firm_id": "ret_445", "cash": 0.0})(),
        }
        sim.manufacturers_by_industry = {
            "311FT": type("Manufacturer", (), {"firm_id": "mfg_311", "cash": 0.0})(),
        }
        hh = FakeHousehold(wealth=1000.0)
        result = {
            "step0": {"budgets": {"Retail merchandise": 200.0}},
            "step3": {
                "purchases": [
                    {"product_id": "sku_a", "allocated_budget": 100.0},
                    {"product_id": "sku_b", "allocated_budget": 100.0},
                ]
            },
        }

        demand_by_product, orders_by_household, _, demand_summary = sim._build_orders([(hh, result)])

        self.assertEqual(dict(demand_by_product), {"sku_a": 10.0, "sku_b": 10.0})
        retailer_codes = [order["retailer_code"] for order in orders_by_household[0][1]]
        self.assertIn("452", retailer_codes)
        self.assertIn("445", retailer_codes)
        self.assertEqual(demand_summary["retail_channel_diversified_orders"], 1)
        self.assertEqual(demand_summary["by_retailer_product"]["452"], {"sku_a": 10.0})
        self.assertEqual(demand_summary["by_retailer_product"]["445"], {"sku_b": 10.0})
        self.assertIn("ret_452", demand_summary["by_retail_firm"])
        self.assertIn("ret_445", demand_summary["by_retail_firm"])

    def test_consumption_history_uses_actual_execution_stats(self):
        sim = Simulator(SimulationConfig(num_households=1, debug_logging=False))
        hh = FakeHousehold(wealth=1000.0)
        planned_result = {
            "step0": {
                "budgets": {
                    "Retail merchandise": 500.0,
                    "housing": 200.0,
                    "utilities": 100.0,
                }
            }
        }

        sim._update_household_consumption_history(
            consumption_stats={
                "by_household": {
                    "hh_1": {"qty": 2.0, "value": 20.0},
                }
            },
            service_consumption_stats={
                "total_service_consumption": 7.0,
                "by_household": {
                    "hh_1": {
                        "value": 7.0,
                        "by_category": {"housing": 5.0, "utilities": 2.0},
                    }
                },
            },
            consumption_results=[(hh, planned_result)],
        )

        self.assertEqual(hh.last_month_consumption, 27.0)
        self.assertEqual(hh.csv_values["expenditure_retail_merchandise"], 20.0)
        self.assertEqual(hh.csv_values["ER85701"], 5.0)
        self.assertEqual(hh.csv_values["expenditure_utilities"], 2.0)
        self.assertEqual(hh.csv_values["ER85768"], 27.0)
        self.assertEqual(hh.csv_values["ER85692"], 973.0)

    def test_retail_order_does_not_fall_back_to_manufacturer_stock(self):
        sim = Simulator(SimulationConfig(num_households=1, debug_logging=False))
        sim.product_market = FakeProductMarket(
            {
                "sku_1": {
                    "product_id": "sku_1",
                    "name": "Test SKU",
                    "retail_price": 10.0,
                    "available_stock": 10,
                    "manufacturer_code": "311FT",
                    "retailer_code": "445",
                }
            }
        )
        seller = type("Retailer", (), {"firm_id": "ret_445", "cash": 0.0})()
        sim.retailers_by_industry = {"445": seller}
        sim.economic_center = FakeEconomicCenterForUnmet({"hh_1": 1000.0})
        hh = FakeHousehold()
        order = {
            "product_id": "sku_1",
            "desired_qty": 5,
            "unit_price": 10.0,
            "product_name": "Test SKU",
            "retailer_code": "445",
            "manufacturer_code": "311FT",
        }

        stats = sim._execute_orders([(hh, [order], 100.0)], {"sku_1": sim.product_market.get_product_snapshot("sku_1")}, 1, True)

        self.assertEqual(stats["total_qty"], 0.0)
        self.assertEqual(sim.economic_center.purchases, [])
        self.assertEqual(len(sim.economic_center.unmet), 1)
        self.assertEqual(sim.economic_center.unmet[0]["available_stock"], 0.0)
        self.assertEqual(sim.product_market._snapshots["sku_1"]["available_stock"], 10)
        self.assertEqual(sim.product_market.purchase_calls, [])

    def test_execute_orders_restores_retailer_stock_when_purchase_settlement_fails(self):
        sim = Simulator(SimulationConfig(num_households=1, debug_logging=False))
        sim.product_market = FakeProductMarket(
            {
                "sku_1": {
                    "product_id": "sku_1",
                    "name": "Test SKU",
                    "retail_price": 10.0,
                    "available_stock": 0,
                    "manufacturer_code": "311FT",
                    "retailer_code": "445",
                }
            }
        )
        sim.product_market.seller_stock[("ret_445", "sku_1")] = 5.0
        seller = type("Retailer", (), {"firm_id": "ret_445", "cash": 0.0})()
        sim.retailers_by_industry = {"445": seller}
        sim.economic_center = FakeEconomicCenterForUnmet({"hh_1": 1000.0}, raise_on_purchase=True)
        hh = FakeHousehold()
        order = {
            "product_id": "sku_1",
            "desired_qty": 4,
            "unit_price": 10.0,
            "product_name": "Test SKU",
            "retailer_code": "445",
            "manufacturer_code": "311FT",
        }

        stats = sim._execute_orders([(hh, [order], 100.0)], {"sku_1": sim.product_market.get_product_snapshot("sku_1")}, 1, True)

        self.assertEqual(stats["total_qty"], 0.0)
        self.assertEqual(sim.product_market.get_seller_stock("sku_1", "ret_445", True), 5.0)
        self.assertEqual(sim.product_market.restored_retailer, [("ret_445", "sku_1", 4)])
        self.assertEqual(sim.economic_center.purchases, [])

    def test_retailer_procurement_uses_channel_specific_demand(self):
        sim = Simulator(SimulationConfig(num_households=1, debug_logging=False))
        sim.product_market = FakeProductMarket(
            {
                "sku_1": {
                    "product_id": "sku_1",
                    "name": "Test SKU",
                    "retail_price": 10.0,
                    "base_retail_price": 10.0,
                    "manufacturer_price": 6.0,
                    "base_manufacturer_price": 6.0,
                    "available_stock": 10,
                    "manufacturer_code": "311FT",
                    "retailer_code": "452",
                }
            }
        )
        sim.retailers_by_industry = {
            "452": type("Retailer", (), {"firm_id": "ret_452", "cash": 1000.0})(),
            "445": type("Retailer", (), {"firm_id": "ret_445", "cash": 1000.0})(),
        }
        sim.manufacturers_by_industry = {
            "311FT": type("Manufacturer", (), {"firm_id": "mfg_311", "cash": 0.0})(),
        }
        snapshot_cache = {}

        stats = sim._retailer_procurement(
            {"sku_1": 10.0},
            snapshot_cache,
            month=1,
            record_transactions=False,
            demand_by_retailer_product={"445": {"sku_1": 6.0}, "452": {"sku_1": 4.0}},
        )

        self.assertEqual(stats["total_qty"], 10.0)
        self.assertEqual(sim.product_market.retailer_inventory_receipts, [
            ("ret_445", "sku_1", 6.0),
            ("ret_452", "sku_1", 4.0),
        ])
        self.assertEqual(sim.product_market.get_seller_stock("sku_1", "ret_445", True), 6.0)
        self.assertEqual(sim.product_market.get_seller_stock("sku_1", "ret_452", True), 4.0)

    def test_retailer_procurement_allocates_scarce_stock_proportionally_across_channels(self):
        sim = Simulator(SimulationConfig(num_households=1, debug_logging=False))
        sim.product_market = FakeProductMarket(
            {
                "sku_1": {
                    "product_id": "sku_1",
                    "name": "Test SKU",
                    "retail_price": 10.0,
                    "base_retail_price": 10.0,
                    "manufacturer_price": 6.0,
                    "base_manufacturer_price": 6.0,
                    "available_stock": 5,
                    "manufacturer_code": "311FT",
                    "retailer_code": "452",
                }
            }
        )
        sim.retailers_by_industry = {
            "452": type("Retailer", (), {"firm_id": "ret_452", "cash": 1000.0})(),
            "445": type("Retailer", (), {"firm_id": "ret_445", "cash": 1000.0})(),
        }
        sim.manufacturers_by_industry = {
            "311FT": type("Manufacturer", (), {"firm_id": "mfg_311", "cash": 0.0})(),
        }

        stats = sim._retailer_procurement(
            {"sku_1": 10.0},
            {},
            month=1,
            record_transactions=False,
            demand_by_retailer_product={"452": {"sku_1": 8.0}, "445": {"sku_1": 2.0}},
        )

        self.assertEqual(stats["total_qty"], 5.0)
        self.assertEqual(sim.product_market.get_seller_stock("sku_1", "ret_452", True), 4.0)
        self.assertEqual(sim.product_market.get_seller_stock("sku_1", "ret_445", True), 1.0)

    def test_retailer_procurement_restores_manufacturer_stock_when_wholesale_declines(self):
        sim = Simulator(SimulationConfig(num_households=1, debug_logging=False))
        sim.product_market = FakeProductMarket(
            {
                "sku_1": {
                    "product_id": "sku_1",
                    "name": "Test SKU",
                    "retail_price": 10.0,
                    "base_retail_price": 10.0,
                    "manufacturer_price": 6.0,
                    "base_manufacturer_price": 6.0,
                    "available_stock": 10,
                    "manufacturer_code": "311FT",
                    "retailer_code": "452",
                }
            }
        )
        sim.retailers_by_industry = {
            "445": type("Retailer", (), {"firm_id": "ret_445", "cash": 1000.0})(),
        }
        sim.manufacturers_by_industry = {
            "311FT": type("Manufacturer", (), {"firm_id": "mfg_311", "cash": 0.0})(),
        }
        sim.economic_center = FakeEconomicCenterForUnmet(wholesale_success=False)
        snapshot_cache = {"sku_1": sim.product_market.get_product_snapshot("sku_1")}

        stats = sim._retailer_procurement(
            {"sku_1": 6.0},
            snapshot_cache,
            month=1,
            record_transactions=True,
            demand_by_retailer_product={"445": {"sku_1": 6.0}},
        )

        self.assertEqual(stats["total_qty"], 0.0)
        self.assertEqual(sim.product_market._snapshots["sku_1"]["available_stock"], 10.0)
        self.assertEqual(snapshot_cache["sku_1"]["available_stock"], 10.0)
        self.assertEqual(sim.product_market.retailer_inventory_receipts, [])
        self.assertEqual(sim.product_market.restored_manufacturer, [("sku_1", 6.0)])

    def test_retailer_procurement_restores_manufacturer_stock_when_wholesale_raises(self):
        sim = Simulator(SimulationConfig(num_households=1, debug_logging=False))
        sim.product_market = FakeProductMarket(
            {
                "sku_1": {
                    "product_id": "sku_1",
                    "name": "Test SKU",
                    "retail_price": 10.0,
                    "base_retail_price": 10.0,
                    "manufacturer_price": 6.0,
                    "base_manufacturer_price": 6.0,
                    "available_stock": 10,
                    "manufacturer_code": "311FT",
                    "retailer_code": "452",
                }
            }
        )
        sim.retailers_by_industry = {
            "445": type("Retailer", (), {"firm_id": "ret_445", "cash": 1000.0})(),
        }
        sim.manufacturers_by_industry = {
            "311FT": type("Manufacturer", (), {"firm_id": "mfg_311", "cash": 0.0})(),
        }
        sim.economic_center = FakeEconomicCenterForUnmet(raise_on_wholesale=True)
        snapshot_cache = {"sku_1": sim.product_market.get_product_snapshot("sku_1")}

        stats = sim._retailer_procurement(
            {"sku_1": 6.0},
            snapshot_cache,
            month=1,
            record_transactions=True,
            demand_by_retailer_product={"445": {"sku_1": 6.0}},
        )

        self.assertEqual(stats["total_qty"], 0.0)
        self.assertEqual(sim.product_market._snapshots["sku_1"]["available_stock"], 10.0)
        self.assertEqual(snapshot_cache["sku_1"]["available_stock"], 10.0)
        self.assertEqual(sim.product_market.retailer_inventory_receipts, [])
        self.assertEqual(sim.product_market.restored_manufacturer, [("sku_1", 6.0)])


if __name__ == "__main__":
    unittest.main()
