import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agenteconomy.simulation.checkpoint import CheckpointManager


class CheckpointStateTests(unittest.TestCase):
    def test_save_and_restore_production_gap_value_by_firm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, compress=False)
            source = SimpleNamespace(
                config=SimpleNamespace(num_months=12, num_households=1, preheat_months=0),
                households=[],
                firms=[],
                government=None,
                bank=None,
                economic_center=None,
                product_market=None,
                labor_market=None,
                abstract_resource_market=None,
                current_month=4,
                _last_price_index=None,
                _last_inflation_rate=None,
                _last_balance_by_household={},
                _last_expected_income_by_household={},
                _last_sales_by_product={},
                _last_planned_demand_by_product={},
                _last_unmet_demand_by_product={},
                _last_production_gap_value_by_firm={"mfg_322": 45.0},
                _last_production_value_by_firm={"mfg_322": 120.0},
                _last_service_value_by_industry={"HS": 340.0},
                _fixed_consumption_basket=None,
            )

            path = manager.save_checkpoint(source, month=4)
            self.assertTrue(Path(path).exists())

            restored = SimpleNamespace(
                households=[],
                firms=[],
                government=None,
                bank=None,
                economic_center=None,
                product_market=None,
                labor_market=None,
                abstract_resource_market=None,
            )
            manager.restore_simulator(restored, manager.load_checkpoint(path))

        self.assertEqual(restored._last_production_gap_value_by_firm, {"mfg_322": 45.0})
        self.assertEqual(restored._last_production_value_by_firm, {"mfg_322": 120.0})
        self.assertEqual(restored._last_service_value_by_industry, {"HS": 340.0})


if __name__ == "__main__":
    unittest.main()
