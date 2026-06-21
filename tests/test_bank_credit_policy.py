import unittest

from agenteconomy.agent.bank_credit_policy import (
    BankCreditPolicy,
    CreditApplication,
    SimpleFirmCreditPolicy,
)


class BankCreditPolicyTest(unittest.TestCase):
    def test_credit_policy_approves_viable_working_capital_request(self):
        policy = SimpleFirmCreditPolicy(base_annual_rate=0.06)
        decision = policy.decide(
            CreditApplication(
                firm_id="firm_a",
                requested_amount=5_000.0,
                cash=4_000.0,
                monthly_revenue=20_000.0,
                monthly_operating_cost=14_000.0,
                capital_stock=50_000.0,
                inventory_value=10_000.0,
            )
        )

        self.assertTrue(decision.approved)
        self.assertEqual(decision.approved_amount, 5_000.0)
        self.assertGreaterEqual(decision.credit_limit, decision.approved_amount)
        self.assertGreater(decision.annual_interest_rate, 0.0)

    def test_credit_policy_rejects_extended_distress(self):
        policy = SimpleFirmCreditPolicy()
        decision = policy.decide(
            CreditApplication(
                firm_id="firm_b",
                requested_amount=2_000.0,
                monthly_revenue=10_000.0,
                monthly_operating_cost=9_000.0,
                capital_stock=25_000.0,
                distress_months=3,
            )
        )

        self.assertFalse(decision.approved)
        self.assertIn("extended_distress", decision.reasons)

    def test_credit_facility_draw_interest_and_repay(self):
        policy = SimpleFirmCreditPolicy()
        decision = policy.decide(
            CreditApplication(
                firm_id="firm_c",
                requested_amount=1_000.0,
                cash=1_000.0,
                monthly_revenue=8_000.0,
                monthly_operating_cost=5_000.0,
                capital_stock=10_000.0,
            )
        )
        facility = policy.open_facility(decision)

        self.assertIsNotNone(facility)
        drawn = facility.draw(500.0)
        interest = facility.accrue_monthly_interest()
        repaid = facility.repay(100.0)

        self.assertEqual(drawn, 500.0)
        self.assertGreater(interest, 0.0)
        self.assertEqual(repaid, 100.0)
        self.assertGreater(facility.outstanding_balance, 0.0)

    def test_bank_credit_policy_public_interface_exposes_limit_and_rejection_reasons(self):
        policy = BankCreditPolicy()
        decision = policy.decide(
            CreditApplication(
                firm_id="firm_d",
                requested_amount=1_000.0,
                monthly_revenue=0.0,
                monthly_operating_cost=10_000.0,
            )
        )

        self.assertFalse(decision.approved)
        self.assertEqual(decision.approved_credit, 0.0)
        self.assertIsInstance(decision.credit_limit, float)
        self.assertIn("insufficient_debt_service_coverage", decision.rejection_reasons)


if __name__ == "__main__":
    unittest.main()
