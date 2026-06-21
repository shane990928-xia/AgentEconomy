"""Regression tests for the macro-closure correctness fixes (2026-06).

Locks four result-poisoning bugs that were fixed for the paper submission:
  * bug1 — firm industry attribution (was always 'Unknown')
  * bug2 — bank-credit double-entry / total-cash conservation
  * bug3 — dividend double-taxation (corporate tax already in monthly expenses)
  * bug4 — FICA payroll-tax withholding in wage payment
"""

import unittest

from agenteconomy.center.Ecocenter import EconomicCenter
from agenteconomy.center.Model import Ledger


EconomicCenterClass = EconomicCenter.__ray_metadata__.modified_class


def _total_ledger_cash(center) -> float:
    return float(sum(float(l.amount or 0.0) for l in center.ledger.values()))


class IndustryAttributionTests(unittest.TestCase):
    """bug1: _get_firm_industry must resolve registered industries, not 'Unknown'."""

    def test_register_firm_industries_then_resolve(self):
        center = EconomicCenterClass()
        center.firm_id.append("mfg_food")
        registered = center.register_firm_industries({"mfg_food": "311", "ret_x": "  "})
        # blank industry is ignored
        self.assertEqual(registered, 1)
        self.assertEqual(center._get_firm_industry("mfg_food"), "311")

    def test_unregistered_firm_falls_back_to_unknown(self):
        center = EconomicCenterClass()
        self.assertEqual(center._get_firm_industry("ghost_firm"), "Unknown")


class BankCreditConservationTests(unittest.TestCase):
    """bug2: credit draw + repayment must keep total ledger cash invariant."""

    def _make_firm(self, center, firm_id="mfg_a", cash=0.0, capital=100000.0):
        center.firm_id.append(firm_id)
        center.ledger[firm_id] = Ledger.create(firm_id, cash)
        center.register_firm_assets({firm_id: {"cash": cash, "capital_stock": capital}})

    def test_credit_draw_is_zero_sum_across_ledgers(self):
        center = EconomicCenterClass()
        self._make_firm(center, cash=0.0)
        total_before = _total_ledger_cash(center)

        drawn = center._draw_firm_credit_if_needed("mfg_a", required_amount=500.0, month=1, reason="test")

        self.assertGreater(drawn, 0.0)
        self.assertAlmostEqual(_total_ledger_cash(center), total_before, places=6)
        self.assertAlmostEqual(center.ledger["bank_credit_system"].amount, -drawn, places=6)
        self.assertAlmostEqual(center.firm_debt_balance["mfg_a"], drawn, places=6)

    def test_repayment_is_zero_sum_across_ledgers(self):
        center = EconomicCenterClass()
        self._make_firm(center, cash=0.0)
        drawn = center._draw_firm_credit_if_needed("mfg_a", required_amount=500.0, month=1, reason="test")
        # fund the firm so it can repay, then settle with no interest / no buffer
        center.ledger["mfg_a"].amount += 1000.0
        total_before = _total_ledger_cash(center)

        center.settle_firm_credit_month(
            month=2,
            annual_interest_rate=0.0,
            repayment_cash_buffer=0.0,
            default_distress_months=3,
        )

        self.assertAlmostEqual(_total_ledger_cash(center), total_before, places=6)
        self.assertAlmostEqual(center.firm_debt_balance["mfg_a"], 0.0, places=6)
        self.assertAlmostEqual(center.ledger["bank_credit_system"].amount, 0.0, places=6)


class DividendCorporateTaxTests(unittest.TestCase):
    """bug3: corporate tax is booked into monthly expenses, so monthly_profit is
    already after-tax. The dividend path must NOT re-apply (1 - corporate_tax_rate).
    This locks the root-cause invariant the dividend fix relies on.
    """

    def test_corporate_tax_lands_in_monthly_expenses(self):
        center = EconomicCenterClass()
        center.firm_id.append("mfg_a")
        center.ledger["mfg_a"] = Ledger.create("mfg_a", 100000.0)
        center.ledger["gov_main_simulation"] = Ledger.create("gov_main_simulation", 0.0)
        center.record_firm_monthly_income("mfg_a", 1, 1000.0)
        center.record_firm_monthly_expense("mfg_a", 1, 200.0)  # pre-tax expenses

        before = center.query_firm_monthly_financials("mfg_a", 1)
        self.assertAlmostEqual(before["monthly_profit"], 800.0, places=4)

        center.settle_monthly_corporate_tax(1)

        expected_tax = 800.0 * float(center.corporate_tax_rate or 0.0)
        after = center.query_firm_monthly_financials("mfg_a", 1)
        self.assertAlmostEqual(after["monthly_expenses"], 200.0 + expected_tax, places=4)
        # monthly_profit is already net of corporate tax -> dividend uses it as-is
        self.assertAlmostEqual(after["monthly_profit"], 800.0 - expected_tax, places=4)


class FicaWithholdingTests(unittest.TestCase):
    """bug4: process_wage must withhold FICA when fica_tax_rate > 0, and be a no-op at 0."""

    def _prepare(self, center, firm_id="mfg_a"):
        center.firm_id.append(firm_id)
        center.ledger[firm_id] = Ledger.create(firm_id, 1_000_000.0)
        center.register_firm_assets({firm_id: {"cash": 1_000_000.0, "capital_stock": 100000.0}})
        center.ledger["hh_1"] = Ledger.create("hh_1", 0.0)
        center.ledger["gov_main_simulation"] = Ledger.create("gov_main_simulation", 0.0)

    def test_fica_withheld_when_rate_positive(self):
        center = EconomicCenterClass()
        center.fica_tax_rate = 0.0765
        self._prepare(center)

        center.process_wage(
            month=1, wage_hour=20.0, household_id="hh_1", firm_id="mfg_a",
            hours_per_period=40.0, periods_per_month=4.0,
        )

        gross = 20.0 * 40.0 * 4.0
        income_tax = center.calculate_progressive_income_tax(gross)
        fica = gross * 0.0765
        net = gross - income_tax - fica

        self.assertAlmostEqual(center.ledger["hh_1"].amount, net, places=4)
        self.assertAlmostEqual(
            center.ledger["gov_main_simulation"].amount, income_tax + fica, places=4
        )
        fica_txs = [tx for tx in center.tx_history if tx.type == "fica_tax"]
        self.assertEqual(len(fica_txs), 1)
        self.assertAlmostEqual(fica_txs[0].amount, fica, places=4)

    def test_no_fica_tx_when_rate_zero(self):
        center = EconomicCenterClass()
        center.fica_tax_rate = 0.0
        self._prepare(center)

        center.process_wage(
            month=1, wage_hour=20.0, household_id="hh_1", firm_id="mfg_a",
            hours_per_period=40.0, periods_per_month=4.0,
        )

        self.assertEqual([tx for tx in center.tx_history if tx.type == "fica_tax"], [])
        gross = 20.0 * 40.0 * 4.0
        income_tax = center.calculate_progressive_income_tax(gross)
        self.assertAlmostEqual(center.ledger["hh_1"].amount, gross - income_tax, places=4)


if __name__ == "__main__":
    unittest.main()
