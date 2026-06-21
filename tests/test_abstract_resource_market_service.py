import unittest

from agenteconomy.market.AbstractResourceMarket import AbstractResourceMarket


class FakeEconomicCenterForServicePurchase:
    def __init__(self, result="tx_1", fail=False):
        self.result = result
        self.fail = fail
        self.calls = []

    def record_resource_purchase(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise ValueError("insufficient balance")
        return self.result


class AbstractResourceMarketServiceTests(unittest.TestCase):
    def _market(self, economic_center=None):
        market = AbstractResourceMarket(economic_center=economic_center)
        market.initialize_resource(
            industry_code="HS",
            name="Housing service",
            base_price=2.0,
            initial_supply_capacity=1000.0,
            firm_id="svc_housing",
        )
        return market

    def test_purchase_by_budget_records_only_confirmed_ledger_transfer(self):
        economic_center = FakeEconomicCenterForServicePurchase(result="tx_1")
        market = self._market(economic_center=economic_center)

        tx = market.purchase_by_budget(
            industry_code="HS",
            buyer_id="household_1",
            budget=20.0,
            period=3,
        )

        self.assertIsNotNone(tx)
        self.assertEqual(tx["tx_id"], "tx_1")
        self.assertEqual(len(market.transactions), 1)
        self.assertEqual(market.resources["HS"].total_demand, 10.0)

    def test_purchase_by_budget_returns_none_when_transfer_fails(self):
        economic_center = FakeEconomicCenterForServicePurchase(fail=True)
        market = self._market(economic_center=economic_center)

        tx = market.purchase_by_budget(
            industry_code="HS",
            buyer_id="household_1",
            budget=20.0,
            period=3,
        )

        self.assertIsNone(tx)
        self.assertEqual(market.transactions, [])
        self.assertEqual(market.resources["HS"].total_demand, 0.0)

    def test_purchase_by_budget_returns_none_when_transfer_declines(self):
        economic_center = FakeEconomicCenterForServicePurchase(result=None)
        market = self._market(economic_center=economic_center)

        tx = market.purchase_by_budget(
            industry_code="HS",
            buyer_id="household_1",
            budget=20.0,
            period=3,
        )

        self.assertIsNone(tx)
        self.assertEqual(market.transactions, [])
        self.assertEqual(market.resources["HS"].total_demand, 0.0)


if __name__ == "__main__":
    unittest.main()
