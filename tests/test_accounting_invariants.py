import unittest
from types import SimpleNamespace

from agenteconomy.utils.accounting_invariants import check_accounting_invariants


class AccountingInvariantTests(unittest.TestCase):
    def test_negative_firm_cash_without_loan_is_reported(self):
        center = SimpleNamespace(
            ledger={"firm_a": SimpleNamespace(amount=-25.0)},
            firm_id=["firm_a"],
            tx_history=[],
        )

        result = check_accounting_invariants(center)

        self.assertFalse(result.ok)
        self.assertEqual(result.metrics["negative_firm_cash_without_loan_count"], 1)
        self.assertTrue(
            any(
                issue.code == "negative_cash_without_loan" and issue.subject == "firm_a"
                for issue in result.errors
            )
        )

    def test_negative_firm_cash_with_credit_draw_is_explained(self):
        center = {
            "ledger": {"firm_a": {"amount": -25.0}},
            "firm_id": ["firm_a"],
            "tx_history": [
                {
                    "id": "tx_credit",
                    "type": "credit_draw",
                    "sender_id": "bank_credit_system",
                    "receiver_id": "firm_a",
                    "amount": 100.0,
                    "month": 2,
                    "metadata": {"reason": "wage_payment"},
                }
            ],
        }

        result = check_accounting_invariants(center)

        self.assertTrue(result.ok)
        self.assertEqual(result.metrics["negative_firm_cash_with_loan_count"], 1)

    def test_negative_product_inventory_is_reported(self):
        product_market = {
            "products_by_id": {
                "sku_1": {"product_id": "sku_1", "available_stock": -3.0},
            }
        }

        result = check_accounting_invariants(product_market=product_market)

        self.assertFalse(result.ok)
        self.assertEqual(result.metrics["negative_inventory_count"], 1)
        self.assertTrue(
            any(
                issue.code == "negative_inventory" and issue.subject == "sku_1"
                for issue in result.errors
            )
        )

    def test_duplicate_resource_purchase_for_same_firm_resource_month_warns(self):
        txs = [
            {
                "id": "tx_1",
                "type": "resource_purchase",
                "sender_id": "firm_a",
                "receiver_id": "market_resource_22",
                "amount": 100.0,
                "month": 2,
                "metadata": {"industry_code": "22", "quantity": 10.0, "unit_price": 10.0},
            },
            {
                "id": "tx_2",
                "type": "resource_purchase",
                "sender_id": "firm_a",
                "receiver_id": "market_resource_22",
                "amount": 100.0,
                "month": 2,
                "metadata": {"industry_code": "22", "quantity": 10.0, "unit_price": 10.0},
            },
        ]
        center = {
            "ledger": {"firm_a": {"amount": 0.0}},
            "firm_id": ["firm_a"],
            "tx_history": txs,
        }

        result = check_accounting_invariants(center)

        self.assertTrue(result.ok)
        self.assertEqual(result.metrics["resource_purchase_duplicate_groups"], 1)
        self.assertTrue(
            any(
                issue.code == "duplicate_resource_purchase"
                and issue.subject == "firm_a:22:2"
                for issue in result.warnings
            )
        )

    def test_department_flow_summary_tracks_basic_sfc_flows(self):
        center = {
            "ledger": {
                "firm_a": {"amount": 0.0},
                "household_1": {"amount": 0.0},
                "gov_main_simulation": {"amount": 0.0},
            },
            "firm_id": ["firm_a"],
            "household_id": ["household_1"],
            "government_id": ["gov_main_simulation"],
            "tx_history": [
                {
                    "id": "tx_wage",
                    "type": "labor_payment",
                    "sender_id": "firm_a",
                    "receiver_id": "household_1",
                    "amount": 80.0,
                    "month": 1,
                },
                {
                    "id": "tx_purchase",
                    "type": "purchase",
                    "sender_id": "household_1",
                    "receiver_id": "firm_a",
                    "amount": 50.0,
                    "month": 1,
                },
                {
                    "id": "tx_tax",
                    "type": "labor_tax",
                    "sender_id": "household_1",
                    "receiver_id": "gov_main_simulation",
                    "amount": 20.0,
                    "month": 1,
                },
                {
                    "id": "tx_credit",
                    "type": "credit_draw",
                    "sender_id": "bank_credit_system",
                    "receiver_id": "firm_a",
                    "amount": 30.0,
                    "month": 1,
                },
            ],
        }

        result = check_accounting_invariants(center, month=1)

        self.assertTrue(result.ok)
        summary = result.metrics["department_flow_summary"]
        departments = summary["departments"]
        self.assertEqual(summary["transaction_count"], 4)
        self.assertAlmostEqual(summary["total_net_flow_residual"], 0.0)
        self.assertAlmostEqual(departments["firm"]["inflow"], 80.0)
        self.assertAlmostEqual(departments["firm"]["outflow"], 80.0)
        self.assertAlmostEqual(departments["firm"]["net_flow"], 0.0)
        self.assertAlmostEqual(departments["household"]["net_flow"], 10.0)
        self.assertAlmostEqual(departments["government"]["net_flow"], 20.0)
        self.assertAlmostEqual(departments["bank_credit"]["net_flow"], -30.0)
        self.assertAlmostEqual(departments["market_or_external"]["net_flow"], 0.0)
        self.assertAlmostEqual(departments["unknown"]["net_flow"], 0.0)

    def test_explicit_flow_ids_do_not_change_negative_cash_ok_semantics(self):
        result = check_accounting_invariants(
            ledger={"household_1": {"amount": -5.0}},
            transactions=[],
            household_ids=["household_1"],
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.metrics["registered_non_firm_negative_cash_count"], 0)


if __name__ == "__main__":
    unittest.main()
