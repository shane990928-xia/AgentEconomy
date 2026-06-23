import unittest

from agenteconomy.center.Ecocenter import EconomicCenter
from agenteconomy.center.Model import Ledger


EconomicCenterClass = EconomicCenter.__ray_metadata__.modified_class


class EconomicCenterCreditTests(unittest.TestCase):
    def test_resource_purchase_draws_credit_before_firm_payment(self):
        center = EconomicCenterClass()
        center.firm_id.append("firm_a")
        center.ledger["firm_a"] = Ledger.create("firm_a", 0.0)
        center.register_firm_assets({"firm_a": {"cash": 0.0, "capital_stock": 100000.0}})

        tx_id = center.record_resource_purchase(
            month=1,
            buyer_id="firm_a",
            industry_code="22",
            quantity=10.0,
            unit_price=10.0,
            total_cost=100.0,
            unit="kWh",
            receiver_id="svc_22",
        )

        self.assertIsNotNone(tx_id)
        self.assertEqual(center.firm_debt_balance["firm_a"], 100.0)
        self.assertAlmostEqual(center.ledger["firm_a"].amount, 0.0)
        credit_txs = [tx for tx in center.tx_history if tx.type == "credit_draw"]
        self.assertEqual(len(credit_txs), 1)
        self.assertEqual(credit_txs[0].receiver_id, "firm_a")
        self.assertEqual(credit_txs[0].metadata["reason"], "resource_purchase")

    def test_intermediate_goods_purchase_records_seller_income_and_credit(self):
        center = EconomicCenterClass()
        center.firm_id.extend(["buyer_firm", "seller_firm"])
        center.household_id.append("household_1")
        center.ledger["buyer_firm"] = Ledger.create("buyer_firm", 40.0)
        center.ledger["seller_firm"] = Ledger.create("seller_firm", 5.0)
        center.register_firm_assets(
            {
                "buyer_firm": {"cash": 40.0, "capital_stock": 100000.0},
                "seller_firm": {"cash": 5.0, "capital_stock": 100000.0},
            }
        )

        tx_id = center.record_intermediate_goods_purchase(
            month=2,
            buyer_id="buyer_firm",
            total_cost=100.0,
            costs_by_industry={"31": 100.0},
            items=[{"sku_id": "sku_31", "quantity": 2.0, "unit_price": 50.0, "total_cost": 100.0}],
            receiver_id="seller_firm",
        )

        self.assertIsNotNone(tx_id)
        self.assertAlmostEqual(center.ledger["buyer_firm"].amount, 0.0)
        self.assertAlmostEqual(center.ledger["seller_firm"].amount, 105.0)
        self.assertAlmostEqual(center.firm_debt_balance["buyer_firm"], 60.0)
        self.assertAlmostEqual(center.firm_monthly_data["buyer_firm"][2]["expenses"], 100.0)
        self.assertAlmostEqual(center.firm_monthly_data["buyer_firm"][2]["production_cost"], 100.0)
        self.assertAlmostEqual(center.firm_monthly_data["seller_firm"][2]["income"], 100.0)

        product_txs = [tx for tx in center.tx_history if tx.id == tx_id]
        self.assertEqual(len(product_txs), 1)
        self.assertEqual(product_txs[0].type, "product_sale")
        self.assertEqual(product_txs[0].metadata["purchase_category"], "intermediate_goods")
        self.assertEqual(center.summarize_households_monthly(2)["aggregate"]["consumption"]["purchase"], 0.0)

        credit_txs = [tx for tx in center.tx_history if tx.type == "credit_draw"]
        self.assertEqual(len(credit_txs), 1)
        self.assertEqual(credit_txs[0].metadata["reason"], "intermediate_goods_purchase")
        self.assertAlmostEqual(
            credit_txs[0].amount,
            center.firm_debt_balance["buyer_firm"],
        )

    def test_wholesale_purchase_draws_credit_for_retailer(self):
        center = EconomicCenterClass()
        center.firm_id.extend(["ret_452", "mfg_322"])
        center.ledger["ret_452"] = Ledger.create("ret_452", 20.0)
        center.ledger["mfg_322"] = Ledger.create("mfg_322", 5.0)
        center.register_firm_assets(
            {
                "ret_452": {"cash": 20.0, "capital_stock": 100000.0},
                "mfg_322": {"cash": 5.0, "capital_stock": 100000.0},
            }
        )

        tx_id = center.process_wholesale(
            month=2,
            retailer_id="ret_452",
            manufacturer_id="mfg_322",
            amount=100.0,
            quantity=10.0,
            product_id="sku_1",
            unit_price=10.0,
        )

        self.assertIsNotNone(tx_id)
        self.assertAlmostEqual(center.firm_debt_balance["ret_452"], 80.0)
        self.assertAlmostEqual(center.ledger["ret_452"].amount, 0.0)
        self.assertAlmostEqual(center.ledger["mfg_322"].amount, 105.0)
        # 进销存会计：进货是购入存货(资产)，不计当期费用；COGS 在销售时(process_purchase)结转。
        self.assertAlmostEqual(center.firm_monthly_data["ret_452"][2].get("expenses", 0.0), 0.0)
        self.assertAlmostEqual(center.firm_monthly_data["mfg_322"][2]["income"], 100.0)
        credit_txs = [tx for tx in center.tx_history if tx.type == "credit_draw"]
        self.assertEqual(len(credit_txs), 1)
        self.assertEqual(credit_txs[0].metadata["reason"], "wholesale_purchase")

    def test_service_transaction_draws_credit_for_firm_sender(self):
        center = EconomicCenterClass()
        center.firm_id.append("firm_a")
        center.ledger["firm_a"] = Ledger.create("firm_a", 10.0)
        center.ledger["svc_22"] = Ledger.create("svc_22", 0.0)
        center.register_firm_assets({"firm_a": {"cash": 10.0, "capital_stock": 100000.0}})

        tx_id = center.add_tx_service(
            month=1,
            sender_id="firm_a",
            receiver_id="svc_22",
            amount=50.0,
        )

        self.assertIsNotNone(tx_id)
        self.assertEqual(center.firm_debt_balance["firm_a"], 40.0)
        self.assertAlmostEqual(center.ledger["firm_a"].amount, 0.0)

    def test_defaulted_firm_wage_payment_fails_without_transactions(self):
        center = EconomicCenterClass()
        center.firm_id.append("firm_a")
        center.household_id.append("household_1")
        center.ledger["firm_a"] = Ledger.create("firm_a", 0.0)
        center.ledger["household_1"] = Ledger.create("household_1", 0.0)
        center.register_firm_assets({"firm_a": {"cash": 0.0, "capital_stock": 100000.0}})
        center.firm_credit_defaulted["firm_a"] = True

        tx_id = center.process_wage(
            month=3,
            wage_hour=20.0,
            household_id="household_1",
            firm_id="firm_a",
            hours_per_period=160.0,
            periods_per_month=1.0,
        )

        self.assertIsNone(tx_id)
        self.assertEqual(center.ledger["firm_a"].amount, 0.0)
        self.assertEqual(center.ledger["household_1"].amount, 0.0)
        self.assertEqual([tx.type for tx in center.tx_history], [])
        self.assertEqual(center.wage_history, [])

    def test_credit_exhausted_resource_purchase_fails_without_negative_balance(self):
        center = EconomicCenterClass()
        center.firm_id.append("firm_a")
        center.ledger["firm_a"] = Ledger.create("firm_a", 0.0)
        center.ledger["svc_22"] = Ledger.create("svc_22", 0.0)
        center.firm_credit_limit["firm_a"] = 10.0
        center.firm_debt_balance["firm_a"] = 10.0

        tx_id = center.record_resource_purchase(
            month=1,
            buyer_id="firm_a",
            industry_code="22",
            quantity=1.0,
            unit_price=100.0,
            total_cost=100.0,
            receiver_id="svc_22",
        )

        self.assertIsNone(tx_id)
        self.assertEqual(center.ledger["firm_a"].amount, 0.0)
        self.assertEqual(center.ledger["svc_22"].amount, 0.0)
        self.assertEqual([tx.type for tx in center.tx_history], [])

    def test_credit_exhausted_intermediate_purchase_fails_without_seller_income(self):
        center = EconomicCenterClass()
        center.firm_id.extend(["buyer_firm", "seller_firm"])
        center.ledger["buyer_firm"] = Ledger.create("buyer_firm", 0.0)
        center.ledger["seller_firm"] = Ledger.create("seller_firm", 5.0)
        center.firm_credit_limit["buyer_firm"] = 10.0
        center.firm_debt_balance["buyer_firm"] = 10.0

        tx_id = center.record_intermediate_goods_purchase(
            month=2,
            buyer_id="buyer_firm",
            total_cost=100.0,
            receiver_id="seller_firm",
        )

        self.assertIsNone(tx_id)
        self.assertEqual(center.ledger["buyer_firm"].amount, 0.0)
        self.assertEqual(center.ledger["seller_firm"].amount, 5.0)
        self.assertEqual(center.firm_monthly_data["seller_firm"][2]["income"], 0.0)

    def test_credit_exhausted_wholesale_purchase_fails_without_manufacturer_income(self):
        center = EconomicCenterClass()
        center.firm_id.extend(["ret_452", "mfg_322"])
        center.ledger["ret_452"] = Ledger.create("ret_452", 0.0)
        center.ledger["mfg_322"] = Ledger.create("mfg_322", 5.0)
        center.firm_credit_limit["ret_452"] = 10.0
        center.firm_debt_balance["ret_452"] = 10.0

        tx_id = center.process_wholesale(
            month=2,
            retailer_id="ret_452",
            manufacturer_id="mfg_322",
            amount=100.0,
        )

        self.assertIsNone(tx_id)
        self.assertEqual(center.ledger["ret_452"].amount, 0.0)
        self.assertEqual(center.ledger["mfg_322"].amount, 5.0)
        self.assertEqual(center.firm_monthly_data["mfg_322"][2]["income"], 0.0)

    def test_credit_exhausted_service_transaction_fails(self):
        center = EconomicCenterClass()
        center.firm_id.append("firm_a")
        center.ledger["firm_a"] = Ledger.create("firm_a", 0.0)
        center.ledger["svc_22"] = Ledger.create("svc_22", 0.0)
        center.firm_credit_limit["firm_a"] = 10.0
        center.firm_debt_balance["firm_a"] = 10.0

        tx_id = center.add_tx_service(month=1, sender_id="firm_a", receiver_id="svc_22", amount=50.0)

        self.assertIsNone(tx_id)
        self.assertEqual(center.ledger["firm_a"].amount, 0.0)
        self.assertEqual(center.ledger["svc_22"].amount, 0.0)
        self.assertEqual([tx.type for tx in center.tx_history], [])

    def test_credit_month_settlement_accrues_interest_and_repays_from_cash_surplus(self):
        center = EconomicCenterClass()
        center.firm_id.append("firm_a")
        center.ledger["firm_a"] = Ledger.create("firm_a", 200.0)
        center.register_firm_assets({"firm_a": {"cash": 200.0, "capital_stock": 100000.0}})
        center.firm_debt_balance["firm_a"] = 100.0

        result = center.settle_firm_credit_month(
            month=2,
            annual_interest_rate=0.12,
            repayment_cash_buffer=50.0,
            default_distress_months=3,
        )

        self.assertGreater(result["interest_total"], 0.0)
        self.assertGreater(result["repayment_total"], 0.0)
        self.assertGreater(center.ledger["firm_a"].amount, 50.0)
        self.assertAlmostEqual(center.firm_debt_balance["firm_a"], 0.0)
        financial_txs = [tx for tx in center.tx_history if tx.type == "financial"]
        self.assertEqual(
            {tx.metadata.get("subtype") for tx in financial_txs},
            {"credit_interest_accrual", "credit_repayment"},
        )

    def test_credit_month_settlement_marks_default_after_distress_threshold(self):
        center = EconomicCenterClass()
        center.firm_id.append("firm_a")
        center.ledger["firm_a"] = Ledger.create("firm_a", 0.0)
        center.register_firm_assets({"firm_a": {"cash": 0.0, "capital_stock": 100000.0}})
        center.firm_debt_balance["firm_a"] = 100.0

        center.settle_firm_credit_month(month=1, annual_interest_rate=0.0, default_distress_months=2)
        result = center.settle_firm_credit_month(month=2, annual_interest_rate=0.0, default_distress_months=2)

        self.assertTrue(center.firm_credit_defaulted["firm_a"])
        self.assertEqual(result["defaulted_count"], 1)

    def test_credit_limit_snapshot_initializes_registered_firms(self):
        center = EconomicCenterClass()
        center.firm_id.append("firm_a")
        center.ledger["firm_a"] = Ledger.create("firm_a", 100.0)
        center.firm_capital_stock["firm_a"] = 100000.0

        limits = center.get_all_firm_credit_limits()

        self.assertIn("firm_a", limits)
        self.assertGreater(limits["firm_a"], 0.0)

    def test_firm_credit_state_snapshot_round_trips(self):
        center = EconomicCenterClass()
        center.firm_credit_limit["firm_a"] = 123.0
        center.firm_debt_balance["firm_a"] = 45.0
        center.firm_credit_distress_months["firm_a"] = 2
        center.firm_credit_defaulted["firm_a"] = True

        snapshot = center.get_firm_credit_state_snapshot()

        restored = EconomicCenterClass()
        count = restored.restore_firm_credit_state(snapshot)

        self.assertEqual(count, 1)
        self.assertEqual(restored.firm_credit_limit["firm_a"], 123.0)
        self.assertEqual(restored.firm_debt_balance["firm_a"], 45.0)
        self.assertEqual(restored.firm_credit_distress_months["firm_a"], 2)
        self.assertTrue(restored.firm_credit_defaulted["firm_a"])

    def test_interest_and_redistribution_transactions_update_ledgers(self):
        center = EconomicCenterClass()
        center.ledger["bank"] = Ledger.create("bank", 1000.0)
        center.ledger["gov_main_simulation"] = Ledger.create("gov_main_simulation", 1000.0)
        center.ledger["household_1"] = Ledger.create("household_1", 100.0)

        center.add_interest_tx(month=1, sender_id="bank", receiver_id="household_1", amount=10.0)
        center.add_redistribution_tx(
            month=1,
            sender_id="gov_main_simulation",
            receiver_id="household_1",
            amount=20.0,
        )

        self.assertAlmostEqual(center.ledger["bank"].amount, 990.0)
        self.assertAlmostEqual(center.ledger["gov_main_simulation"].amount, 980.0)
        self.assertAlmostEqual(center.ledger["household_1"].amount, 130.0)
        self.assertEqual([tx.type for tx in center.tx_history], ["interest", "redistribution"])

    def test_gdp_compensation_uses_gross_wages_not_net_labor_payment(self):
        center = EconomicCenterClass()
        center.firm_id.append("firm_a")
        center.household_id.append("household_1")
        center._record_transaction(
            sender_id="firm_a",
            receiver_id="household_1",
            amount=80.0,
            tx_type="labor_payment",
            month=1,
            metadata={"gross_wage": 100.0},
        )

        result = center.calculate_gdp_comprehensive(
            month=1,
            production_stats={"total_output_value": 100.0, "total_production_cost": 0.0},
        )

        compensation = result["income_components"]["compensation_of_employees"]
        self.assertEqual(compensation["total"], 100.0)
        self.assertEqual(compensation["wages_gross"], 100.0)
        self.assertEqual(compensation["wages_net"], 80.0)
        self.assertEqual(compensation["private_wages"], 100.0)

    def test_household_service_purchase_counts_as_service_output_in_gdp(self):
        center = EconomicCenterClass()
        center.household_id.append("household_1")
        center.firm_id.append("svc_HS")
        center.ledger["household_1"] = Ledger.create("household_1", 500.0)
        center.ledger["svc_HS"] = Ledger.create("svc_HS", 0.0)

        tx_id = center.record_resource_purchase(
            month=1,
            buyer_id="household_1",
            industry_code="HS",
            quantity=100.0,
            unit_price=1.0,
            total_cost=100.0,
            unit="service_unit",
            receiver_id="svc_HS",
        )

        self.assertIsNotNone(tx_id)
        result = center.calculate_gdp_comprehensive(month=1, production_stats={})

        self.assertAlmostEqual(result["gdp_by_method"]["expenditure"], 100.0)
        self.assertAlmostEqual(result["gdp_by_method"]["production"], 100.0)
        self.assertAlmostEqual(result["gdp_by_method"]["income"], 100.0)
        self.assertAlmostEqual(
            result["gdp_by_method"]["discrepancy"]["exp_vs_prod"], 0.0
        )
        hs_stats = result["production_components"]["by_industry"]["HS"]
        self.assertAlmostEqual(hs_stats["output"], 100.0)
        self.assertAlmostEqual(hs_stats["value_added"], 100.0)

    def test_goods_sold_from_inventory_do_not_create_current_period_gdp(self):
        center = EconomicCenterClass()
        center.household_id.append("household_1")
        center.firm_id.append("ret_452")
        center._record_transaction(
            sender_id="household_1",
            receiver_id="ret_452",
            amount=100.0,
            tx_type="purchase",
            month=1,
            metadata={"amount_ex_tax": 100.0, "industry": "452"},
        )

        result = center.calculate_gdp_comprehensive(month=1, production_stats={})

        self.assertAlmostEqual(result["expenditure_components"]["consumption"]["total"], 100.0)
        self.assertAlmostEqual(
            result["expenditure_components"]["investment"]["inventory_investment"],
            -100.0,
        )
        self.assertAlmostEqual(result["gdp_by_method"]["expenditure"], 0.0)
        self.assertAlmostEqual(result["gdp_by_method"]["production"], 0.0)
        self.assertAlmostEqual(result["gdp_by_method"]["income"], 0.0)

    def test_retail_margin_is_distribution_output_not_negative_goods_inventory(self):
        center = EconomicCenterClass()
        center.household_id.append("household_1")
        center.firm_id.append("ret_452")
        center._record_transaction(
            sender_id="household_1",
            receiver_id="ret_452",
            amount=100.0,
            tx_type="purchase",
            month=1,
            metadata={
                "amount_ex_tax": 100.0,
                "industry": "452",
                "quantity": 10.0,
                "unit_price": 10.0,
                "base_unit_price": 6.0,
                "base_amount": 60.0,
                "retail_margin": 40.0,
            },
        )

        result = center.calculate_gdp_comprehensive(
            month=1,
            production_stats={"total_output_value": 60.0, "total_production_cost": 0.0},
        )

        consumption = result["expenditure_components"]["consumption"]
        investment = result["expenditure_components"]["investment"]
        self.assertAlmostEqual(consumption["household_goods_ex_tax"], 100.0)
        self.assertAlmostEqual(consumption["household_goods_base_value"], 60.0)
        self.assertAlmostEqual(consumption["retail_distribution_margin"], 40.0)
        self.assertAlmostEqual(investment["goods_final_sales_base_value"], 60.0)
        self.assertAlmostEqual(investment["inventory_investment"], 0.0)
        self.assertAlmostEqual(investment["retail_distribution_output"], 40.0)
        self.assertAlmostEqual(result["gdp_by_method"]["expenditure"], 100.0)
        self.assertAlmostEqual(result["gdp_by_method"]["production"], 100.0)
        self.assertAlmostEqual(result["gdp_by_method"]["income"], 100.0)

    def test_purchase_amount_defaults_to_ex_tax_when_vat_is_separate(self):
        center = EconomicCenterClass()
        center.household_id.append("household_1")
        center.firm_id.append("ret_452")
        center._record_transaction(
            sender_id="household_1",
            receiver_id="ret_452",
            amount=100.0,
            tx_type="purchase",
            month=1,
            metadata={"industry": "452"},
        )
        center._record_transaction(
            sender_id="household_1",
            receiver_id="gov_main_simulation",
            amount=8.0,
            tx_type="consume_tax",
            month=1,
        )

        result = center.calculate_gdp_comprehensive(
            month=1,
            production_stats={"total_output_value": 100.0, "total_production_cost": 0.0},
        )

        consumption = result["expenditure_components"]["consumption"]
        self.assertAlmostEqual(consumption["household_consumption_ex_tax"], 100.0)
        self.assertAlmostEqual(consumption["vat_paid_by_household"], 8.0)
        self.assertAlmostEqual(consumption["total"], 108.0)
        self.assertAlmostEqual(result["gdp_by_method"]["expenditure"], 108.0)
        self.assertAlmostEqual(result["gdp_by_method"]["production"], 108.0)
        self.assertAlmostEqual(result["gdp_by_method"]["income"], 108.0)

    def test_business_service_purchase_counts_as_output_and_intermediate_input(self):
        center = EconomicCenterClass()
        center.firm_id.extend(["firm_a", "svc_22"])
        center.ledger["firm_a"] = Ledger.create("firm_a", 1000.0)
        center.ledger["svc_22"] = Ledger.create("svc_22", 0.0)

        tx_id = center.record_resource_purchase(
            month=1,
            buyer_id="firm_a",
            industry_code="22",
            quantity=40.0,
            unit_price=1.0,
            total_cost=40.0,
            unit="kWh",
            receiver_id="svc_22",
        )

        self.assertIsNotNone(tx_id)
        result = center.calculate_gdp_comprehensive(
            month=1,
            production_stats={"total_output_value": 100.0, "total_production_cost": 40.0},
        )

        self.assertAlmostEqual(result["gdp_by_method"]["expenditure"], 100.0)
        self.assertAlmostEqual(result["gdp_by_method"]["production"], 100.0)
        self.assertAlmostEqual(result["gdp_by_method"]["income"], 100.0)
        self.assertAlmostEqual(result["production_components"]["total_output"], 140.0)
        self.assertAlmostEqual(result["production_components"]["intermediate_consumption"], 40.0)
        self.assertAlmostEqual(result["production_components"]["by_industry"]["22"]["output"], 40.0)

    def test_corporate_tax_is_not_double_counted_in_income_gdp(self):
        center = EconomicCenterClass()
        center.firm_id.append("firm_a")
        center.household_id.append("household_1")
        center._record_transaction(
            sender_id="household_1",
            receiver_id="firm_a",
            amount=100.0,
            tx_type="purchase",
            month=1,
            metadata={"amount_ex_tax": 100.0, "industry": "31"},
        )
        center._record_transaction(
            sender_id="firm_a",
            receiver_id="gov_main_simulation",
            amount=20.0,
            tx_type="corporate_tax",
            month=1,
        )

        result = center.calculate_gdp_comprehensive(
            month=1,
            production_stats={"total_output_value": 100.0, "total_production_cost": 0.0},
        )

        self.assertAlmostEqual(result["gdp_by_method"]["income"], 100.0)
        self.assertEqual(
            result["income_components"]["taxes_on_production"]["corporate_tax_memo"],
            20.0,
        )


if __name__ == "__main__":
    unittest.main()
