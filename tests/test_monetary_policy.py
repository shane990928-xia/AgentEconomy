"""Tests for the monetary-policy block (2026-06): Taylor-rule policy rate,
interest-rate transmission, and closure-safe fixed-capital investment.

Covers:
  * Taylor rule math: inflation/unemployment response, inertia, clamping.
  * record_capital_formation: increments capital stock WITHOUT touching cash
    (the money-conservation-safe capex path).
  * process_purchase with tx_type='capex_purchase': moves cash buyer→seller and
    records a capex_purchase transaction (so GDP routes it into fixed investment).
  * Config defaults keep every mechanism OFF.
"""

import unittest

from agenteconomy.center.Ecocenter import EconomicCenter
from agenteconomy.center.Model import Ledger
from config.config import SimulationConfig


EconomicCenterClass = EconomicCenter.__ray_metadata__.modified_class


def _total_ledger_cash(center) -> float:
    return float(sum(float(l.amount or 0.0) for l in center.ledger.values()))


def _taylor_rate(infl_monthly, u, prev, cfg):
    """Reference implementation mirroring Simulator._update_policy_rate."""
    natural = cfg["natural"]
    target_monthly = cfg["target_annual"] / 12.0
    infl_gap_annual = (infl_monthly - target_monthly) * 12.0
    i_target = (
        natural
        + cfg["phi_pi"] * infl_gap_annual
        - cfg["phi_u"] * (u - cfg["u_star"])
        + cfg.get("shock", 0.0)
    )
    rho = cfg["rho"]
    rate = rho * prev + (1.0 - rho) * i_target
    return max(cfg["rmin"], min(cfg["rmax"], rate))


BASE_CFG = dict(
    natural=0.03, target_annual=0.02, phi_pi=1.5, phi_u=0.5, u_star=0.05,
    rho=0.0, rmin=0.0, rmax=0.20,
)


class TaylorRuleMathTests(unittest.TestCase):
    def test_high_inflation_hikes(self):
        # Inflation well above target with no inertia → rate above natural.
        rate = _taylor_rate(0.02, 0.05, 0.03, BASE_CFG)  # 2%/mo ≈ 24%/yr
        self.assertGreater(rate, BASE_CFG["natural"])

    def test_high_unemployment_cuts(self):
        # Unemployment above u* pulls the target down (toward the floor).
        at_target_infl = BASE_CFG["target_annual"] / 12.0
        rate = _taylor_rate(at_target_infl, 0.30, 0.03, BASE_CFG)
        self.assertLess(rate, BASE_CFG["natural"])

    def test_at_target_stays_near_natural(self):
        at_target_infl = BASE_CFG["target_annual"] / 12.0
        rate = _taylor_rate(at_target_infl, BASE_CFG["u_star"], 0.03, BASE_CFG)
        self.assertAlmostEqual(rate, BASE_CFG["natural"], places=6)

    def test_clamp_max(self):
        rate = _taylor_rate(0.10, 0.0, 0.20, BASE_CFG)  # huge inflation
        self.assertEqual(rate, BASE_CFG["rmax"])

    def test_clamp_min(self):
        rate = _taylor_rate(-0.05, 0.50, 0.0, BASE_CFG)  # deflation + mass unemployment
        self.assertEqual(rate, BASE_CFG["rmin"])

    def test_inertia_smooths(self):
        cfg = dict(BASE_CFG, rho=0.8)
        prev = 0.03
        # One step toward a high target moves only part-way under inertia.
        target_cfg = dict(cfg, rho=0.0)
        target = _taylor_rate(0.02, 0.05, prev, target_cfg)
        smoothed = _taylor_rate(0.02, 0.05, prev, cfg)
        self.assertTrue(prev < smoothed < target)


class CapitalFormationConservationTests(unittest.TestCase):
    """record_capital_formation must NOT touch the ledger (cash already left via
    the capital-goods purchase). This was the money-conservation bug fix."""

    def test_record_capital_formation_does_not_touch_cash(self):
        center = EconomicCenterClass()
        center.ledger["mfg_a"] = Ledger.create("mfg_a", 5000.0)
        before = _total_ledger_cash(center)
        out = center.record_capital_formation("mfg_a", 1200.0, month=3)
        self.assertEqual(out, 1200.0)
        # Cash unchanged
        self.assertAlmostEqual(_total_ledger_cash(center), before, places=6)
        # Capital stock incremented
        self.assertAlmostEqual(float(center.firm_capital_stock.get("mfg_a", 0.0)), 1200.0, places=6)

    def test_record_capital_formation_rejects_nonpositive(self):
        center = EconomicCenterClass()
        self.assertEqual(center.record_capital_formation("mfg_a", 0.0, month=3), 0.0)
        self.assertEqual(center.record_capital_formation("mfg_a", 100.0, month=0), 0.0)


class CapexPurchaseTransactionTests(unittest.TestCase):
    """process_purchase with tx_type='capex_purchase' transfers cash buyer→seller
    and tags the transaction so GDP routes it into fixed investment, not C."""

    def test_capex_purchase_moves_cash_and_tags_tx(self):
        center = EconomicCenterClass()
        center.firm_id.extend(["buyer_f", "mfg_seller"])
        center.ledger["buyer_f"] = Ledger.create("buyer_f", 10000.0)
        center.ledger["mfg_seller"] = Ledger.create("mfg_seller", 0.0)
        total_before = _total_ledger_cash(center)

        tx_id = center.process_purchase(
            month=2, buyer_id="buyer_f", seller_id="mfg_seller",
            amount=1000.0, quantity=10.0, product_id="sku1",
            product_name="machine", unit_price=100.0, base_unit_price=100.0,
            tx_type="capex_purchase",
        )
        self.assertIsNotNone(tx_id)
        # Total ledger cash conserved (buyer pays base+VAT, seller gets base, gov gets VAT).
        self.assertAlmostEqual(_total_ledger_cash(center), total_before, places=6)
        # A capex_purchase tx exists with the firm as sender.
        capex_txs = center.get_transactions(tx_type="capex_purchase")
        self.assertEqual(len(capex_txs), 1)
        self.assertEqual(capex_txs[0].sender_id, "buyer_f")
        self.assertEqual(capex_txs[0].receiver_id, "mfg_seller")

    def test_default_tx_type_is_purchase(self):
        center = EconomicCenterClass()
        center.household_id.append("hh1")
        center.ledger["hh1"] = Ledger.create("hh1", 10000.0)
        center.ledger["mfg_seller"] = Ledger.create("mfg_seller", 0.0)
        center.process_purchase(
            month=1, buyer_id="hh1", seller_id="mfg_seller",
            amount=500.0, quantity=5.0, product_id="sku1",
            unit_price=100.0, base_unit_price=100.0,
        )
        purchase_txs = center.get_transactions(tx_type="purchase")
        self.assertEqual(len(purchase_txs), 1)


class MonetaryConfigDefaultsTests(unittest.TestCase):
    def test_all_monetary_mechanisms_default_off(self):
        c = SimulationConfig()
        self.assertFalse(c.taylor_rule_enabled)
        self.assertFalse(c.fixed_investment_enabled)
        self.assertEqual(c.consumption_rate_sensitivity, 0.0)

    def test_taylor_params_have_sane_defaults(self):
        c = SimulationConfig()
        self.assertEqual(c.taylor_phi_pi, 1.5)
        self.assertGreater(c.taylor_rate_max, c.taylor_rate_min)
        self.assertGreaterEqual(c.taylor_rate_inertia, 0.0)
        self.assertLessEqual(c.taylor_rate_inertia, 1.0)


if __name__ == "__main__":
    unittest.main()
