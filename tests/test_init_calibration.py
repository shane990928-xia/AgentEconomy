import unittest

from agenteconomy.simulation.init_calibration import (
    FirmInitCalibrationInput,
    FirmInitializationCalibrator,
)


class FirmInitializationCalibrationTests(unittest.TestCase):
    def test_cash_capital_inventory_targets_are_calibrated(self):
        calibrator = FirmInitializationCalibrator()
        result = calibrator.calibrate(
            FirmInitCalibrationInput(
                firm_id="firm_a",
                industry_code="22",
                industry_type="service",
                expected_monthly_revenue=10000.0,
                expected_monthly_cost=7000.0,
                expected_monthly_sales_units=120.0,
                inventory_cover_months=1.5,
                cash_multiplier=1.5,
                min_cash=10000.0,
                capital_output_ratio=4.0,
                inventory_value_share=0.4,
            )
        )

        self.assertEqual(result.initial_cash, 10500.0)
        self.assertAlmostEqual(result.target_inventory_units, 180.0)
        self.assertAlmostEqual(result.target_inventory_value, 6000.0)
        self.assertAlmostEqual(result.initial_capital_stock, 480000.0)
        self.assertAlmostEqual(result.diagnostics["expected_unit_price"], 83.3333, places=4)
        self.assertAlmostEqual(result.diagnostics["working_capital_requirement"], 10500.0)

    def test_min_cash_floor_applies_when_costs_are_low(self):
        calibrator = FirmInitializationCalibrator()
        result = calibrator.calibrate(
            FirmInitCalibrationInput(
                firm_id="firm_b",
                industry_code="31",
                industry_type="manufacture",
                expected_monthly_revenue=0.0,
                expected_monthly_cost=1000.0,
                expected_monthly_sales_units=0.0,
                cash_multiplier=1.0,
                min_cash=5000.0,
            )
        )

        self.assertEqual(result.initial_cash, 5000.0)
        self.assertEqual(result.target_inventory_units, 0.0)


if __name__ == "__main__":
    unittest.main()
