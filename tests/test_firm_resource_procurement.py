import unittest

from agenteconomy.agent.firm import ManufactureFirm


class FakeEconomicCenter:
    def __init__(self):
        self.resource_purchase_calls = []

    def record_resource_purchase(self, *args, **kwargs):
        self.resource_purchase_calls.append((args, kwargs))


class FakeAbstractResourceMarket:
    def __init__(self):
        self.purchase_calls = []

    def get_resource_info(self, industry_code):
        if industry_code != "22":
            raise ValueError("unknown resource")
        return {"industry_code": industry_code}

    def calculate_physical_demand(self, industry_code, production_value, io_coefficient):
        return production_value * io_coefficient, "kWh"

    def purchase(self, industry_code, buyer_id, quantity, period):
        self.purchase_calls.append(
            {
                "industry_code": industry_code,
                "buyer_id": buyer_id,
                "quantity": quantity,
                "period": period,
            }
        )
        return {
            "total_cost": quantity * 2.0,
            "unit_price": 2.0,
            "unit": "kWh",
            "base_price": 1.5,
        }


class FirmResourceProcurementTests(unittest.TestCase):
    def test_abstract_resource_purchase_records_once_through_market(self):
        economic_center = FakeEconomicCenter()
        resource_market = FakeAbstractResourceMarket()
        firm = ManufactureFirm(
            firm_id="firm_a",
            economic_center=economic_center,
            abstract_resource_market=resource_market,
        )

        costs = firm.procure_abstract_resources(
            production_value=100.0,
            io_suppliers=[{"supplier": "22", "coefficient": 0.1, "name": "Utilities"}],
            period=3,
        )

        self.assertEqual(costs, {"22": 20.0})
        self.assertEqual(len(resource_market.purchase_calls), 1)
        self.assertEqual(economic_center.resource_purchase_calls, [])


if __name__ == "__main__":
    unittest.main()
