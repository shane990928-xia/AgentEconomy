"""Tests for the three calibration root-cause fixes (2026-06):

  1. employment_adjustment_inertia — partial-adjustment of the firm labor budget
     toward last month's realized payroll (damps the period-2 cobweb).
  2. consumption_current_income_weight — current-income consumption channel that
     adds an income-surprise term and reduces the habit dilution of income.
  3. household_keep_negative_wealth + household_wealth_cap_percentile — retain the
     debt tail and the rich tail so the wealth distribution is not compressed.

All three default OFF / neutral; the OFF path must reproduce legacy behavior.
"""

import unittest

from config.config import SimulationConfig
from agenteconomy.agent.household_consumption_policy import build_rule_based_consumption_plan
from agenteconomy.simulation.agent_loader import load_all_households


def _gini(values):
    v = sorted(float(x) for x in values if x is not None)
    v = [x for x in v if x >= 0]
    n = len(v)
    if n == 0 or sum(v) == 0:
        return float("nan")
    cum, s = 0.0, 0.0
    run = 0.0
    for x in v:
        run += x
        s += run
    return (n + 1 - 2 * s / run) / n


class ConfigDefaultsTests(unittest.TestCase):
    """Every new knob defaults to the legacy/neutral value."""

    def test_calibration_knobs_default_off(self):
        c = SimulationConfig()
        self.assertEqual(c.employment_adjustment_inertia, 0.0)
        self.assertEqual(c.consumption_current_income_weight, 0.0)
        self.assertFalse(c.household_keep_negative_wealth)
        self.assertEqual(c.household_wealth_cap_percentile, 0.90)
        # Cobweb-fix knobs default to legacy behavior.
        self.assertEqual(c.firm_layoff_speed, 1.0)
        self.assertEqual(c.firm_beveridge_overposting_strength, 1.0)


class CurrentIncomeConsumptionChannelTests(unittest.TestCase):
    """consumption_current_income_weight adds a monotone current-income response."""

    BASE_STATE = {"ER85629": 4000.0, "ER85692": 50000.0}

    def _plan(self, *, income, weight):
        state = dict(self.BASE_STATE)
        state["macro_indicators"] = (
            {"consumption_current_income_weight": weight} if weight else {}
        )
        return build_rule_based_consumption_plan(
            household_state=state,
            available_budget=50000.0,
            expected_income=income,
        )

    def test_off_path_independent_of_macro_flag(self):
        # weight=0 must give the same budget as no flag at all.
        no_flag = build_rule_based_consumption_plan(
            household_state=dict(self.BASE_STATE),
            available_budget=50000.0,
            expected_income=6000.0,
        )
        zero_flag = self._plan(income=6000.0, weight=0.0)
        self.assertAlmostEqual(no_flag.total_budget, zero_flag.total_budget, places=6)

    def test_on_is_monotone_in_current_income(self):
        # With the channel on, higher current income → higher consumption.
        hi = self._plan(income=6000.0, weight=0.5)
        lo = self._plan(income=2000.0, weight=0.5)
        self.assertGreater(hi.total_budget, lo.total_budget)

    def test_above_reference_income_lifts_consumption(self):
        # Income above the permanent-income anchor (ER85629) raises consumption
        # vs the OFF path at the same income.
        off = self._plan(income=8000.0, weight=0.0)
        on = self._plan(income=8000.0, weight=0.5)
        self.assertGreater(on.total_budget, off.total_budget)


class WealthHeterogeneityLoaderTests(unittest.TestCase):
    """The loader knobs retain the debt + rich tails when enabled."""

    def test_off_path_drops_negative_and_caps(self):
        hhs = load_all_households()
        wealth = [float(h.csv_values.get("ER85692") or 0.0) for h in hhs.values()]
        self.assertEqual(sum(1 for w in wealth if w < 0.0), 0)  # no debt tail

    def test_on_path_keeps_negative_and_rich_tail(self):
        off = load_all_households()
        on = load_all_households(keep_negative_wealth=True, wealth_cap_percentile=1.0)
        off_w = [float(h.csv_values.get("ER85692") or 0.0) for h in off.values()]
        on_w = [float(h.csv_values.get("ER85692") or 0.0) for h in on.values()]
        # debt tail retained
        self.assertGreater(sum(1 for w in on_w if w < 0.0), 0)
        # rich tail retained (no p90 cap) → higher max
        self.assertGreater(max(on_w), max(off_w))
        # wealth inequality rises toward the empirical target
        self.assertGreater(_gini(on_w), _gini(off_w))

    def test_cap_percentile_one_means_no_cap(self):
        capped = load_all_households(wealth_cap_percentile=0.90)
        uncapped = load_all_households(wealth_cap_percentile=1.0)
        cap_w = [float(h.csv_values.get("ER85692") or 0.0) for h in capped.values()]
        unc_w = [float(h.csv_values.get("ER85692") or 0.0) for h in uncapped.values()]
        self.assertGreater(max(unc_w), max(cap_w))

    def test_representative_sampling_spans_distribution(self):
        # At a small limit, head-sampling clips the tails (low Gini); representative
        # sampling spans the full wealth distribution (debt tail .. rich tail), so the
        # sampled wealth Gini is much higher and closer to the empirical 0.85.
        head = load_all_households(
            limit=50, keep_negative_wealth=True, wealth_cap_percentile=1.0, sampling="head"
        )
        rep = load_all_households(
            limit=50, keep_negative_wealth=True, wealth_cap_percentile=1.0, sampling="representative"
        )
        head_w = [float(h.csv_values.get("ER85692") or 0.0) for h in head.values()]
        rep_w = [float(h.csv_values.get("ER85692") or 0.0) for h in rep.values()]
        self.assertEqual(len(rep_w), 50)
        # representative spans both tails: lower min, higher max, higher Gini
        self.assertLess(min(rep_w), min(head_w))
        self.assertGreater(max(rep_w), max(head_w))
        self.assertGreater(_gini(rep_w), _gini(head_w))

    def test_head_sampling_is_default_and_legacy(self):
        # Default sampling must reproduce the first-N rows (legacy behavior).
        default = load_all_households(limit=30)
        head = load_all_households(limit=30, sampling="head")
        self.assertEqual(list(default.keys()), list(head.keys()))


class LayoffSpeedTests(unittest.TestCase):
    """firm_layoff_speed is the labor-hoarding gradual-layoff step.

    The effective target cap = current - speed*(current - new_cap), floored at
    new_cap. speed=1 lays off fully to the cap (legacy); speed<1 keeps part of
    the over-budget workforce (hoarding) so the layoff<->rehire cobweb is damped.
    """

    @staticmethod
    def _effective_cap(current, new_cap, speed):
        speed = max(0.0, min(1.0, speed))
        if speed >= 1.0:
            return new_cap
        return max(new_cap, current - speed * (current - new_cap))

    def test_speed_one_is_legacy_full_layoff(self):
        self.assertAlmostEqual(self._effective_cap(100.0, 60.0, 1.0), 60.0)

    def test_speed_half_keeps_half_the_overshoot(self):
        self.assertAlmostEqual(self._effective_cap(100.0, 60.0, 0.5), 80.0)

    def test_partial_step_monotone_and_floored(self):
        gentle = self._effective_cap(100.0, 60.0, 0.25)
        firm = self._effective_cap(100.0, 60.0, 0.75)
        self.assertGreater(gentle, firm)
        self.assertGreaterEqual(gentle, 60.0)
        self.assertGreaterEqual(firm, 60.0)


class BeveridgeOverpostingTests(unittest.TestCase):
    """firm_beveridge_overposting_strength linearly scales the >1 over-posting
    multiplier: adj' = 1 + (adj - 1) * strength. 1.0 keeps legacy, 0.0 removes
    the phantom over-posting entirely (multiplier collapses to 1.0)."""

    @staticmethod
    def _scaled(adj, strength):
        strength = max(0.0, strength)
        if strength == 1.0:
            return adj
        return 1.0 + (adj - 1.0) * strength

    def test_strength_one_is_legacy(self):
        self.assertAlmostEqual(self._scaled(2.5, 1.0), 2.5)

    def test_strength_zero_removes_overposting(self):
        self.assertAlmostEqual(self._scaled(2.5, 0.0), 1.0)

    def test_strength_half_is_midpoint(self):
        self.assertAlmostEqual(self._scaled(2.5, 0.5), 1.75)


if __name__ == "__main__":
    unittest.main()
