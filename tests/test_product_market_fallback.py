import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agenteconomy.center.ProductMarket import (
    ProductMarket,
    get_retailer_from_manufacturer,
    manufacturer_firm_id_for_code,
    normalize_manufacturer_code,
)
from agenteconomy.market.pricing_policy import PricingPolicy


ProductMarketClass = ProductMarket.__ray_metadata__.modified_class


class ProductMarketFallbackTests(unittest.TestCase):
    def test_manufacturer_code_normalization_keeps_product_keys_canonical(self):
        self.assertEqual(normalize_manufacturer_code("Paper products"), "322")
        self.assertEqual(normalize_manufacturer_code("322"), "322")
        self.assertEqual(manufacturer_firm_id_for_code("Paper products"), "mfg_322")
        self.assertEqual(get_retailer_from_manufacturer("Food and beverage and tobacco products"), "445")
        self.assertEqual(get_retailer_from_manufacturer("311FT"), "445")

    def test_add_product_canonicalizes_manufacturer_code_and_default_seller(self):
        market = ProductMarketClass.__new__(ProductMarketClass)
        market.products = []
        market.products_by_id = {}
        market.products_by_industry = {}
        market.products_by_retailer = {}
        market.logger = MagicMock()
        product = SimpleNamespace(
            product_id="sku_paper",
            name="paper towels",
            manufacturer_code="Paper products",
            retailer_code="452",
            owner_id="Paper products",
            classification="Paper products",
        )

        market.add_product(product)

        self.assertEqual(product.manufacturer_code, "322")
        self.assertEqual(product.owner_id, "mfg_322")
        self.assertEqual(market.get_seller_id("sku_paper"), "mfg_322")
        self.assertEqual(product.classification, "Paper products")
        self.assertEqual(list(market.products_by_industry.keys()), ["322"])

    def test_get_seller_id_preserves_explicit_firm_owner(self):
        market = ProductMarketClass.__new__(ProductMarketClass)
        product = SimpleNamespace(
            product_id="sku_new_entrant",
            owner_id="mfg_new_entrant",
            manufacturer_code="322",
        )
        market.products_by_id = {"sku_new_entrant": product}

        self.assertEqual(market.get_seller_id("sku_new_entrant"), "mfg_new_entrant")

    def test_search_uses_text_fallback_without_embedding_model_path(self):
        market = ProductMarketClass.__new__(ProductMarketClass)
        market.products = [
            SimpleNamespace(
                product_id="sku_food",
                name="apple snack",
                brand="brand",
                classification="Food and beverage and tobacco products",
                description="fresh apple snack",
                available_stock=10.0,
            )
        ]
        market.client = object()
        market._require_active_filter = False
        market._active_sku_set = set()

        with patch.dict(os.environ, {"MODEL_PATH": ""}, clear=False):
            with patch("agenteconomy.center.ProductMarket.embedding") as embedding_mock:
                results = market.search_by_vector("apple", top_k=3)

        embedding_mock.assert_not_called()
        self.assertEqual([p.product_id for p in results], ["sku_food"])

    def test_text_fallback_rotates_candidates_by_household_query_seed(self):
        market = ProductMarketClass.__new__(ProductMarketClass)
        market.products = [
            SimpleNamespace(
                product_id=f"sku_food_{idx}",
                name=f"food pantry item {idx}",
                brand="brand",
                classification="Food and beverage and tobacco products",
                description="basic food replenishment",
                available_stock=10.0,
            )
            for idx in range(8)
        ]
        market.client = object()
        market._require_active_filter = False
        market._active_sku_set = set()

        with patch.dict(os.environ, {"MODEL_PATH": ""}, clear=False):
            household_1 = market.search_by_vector(
                "Food and beverage and tobacco products: basic food household:1",
                top_k=5,
            )
            household_2 = market.search_by_vector(
                "Food and beverage and tobacco products: basic food household:2",
                top_k=5,
            )

        self.assertEqual(len(household_1), 5)
        self.assertEqual(len(household_2), 5)
        self.assertNotEqual(
            [p.product_id for p in household_1],
            [p.product_id for p in household_2],
        )

    def test_search_cache_reuses_product_ids_and_rechecks_stock(self):
        market = ProductMarketClass.__new__(ProductMarketClass)
        product = SimpleNamespace(
            product_id="sku_food",
            name="apple snack",
            brand="brand",
            classification="Food and beverage and tobacco products",
            description="fresh apple snack",
            available_stock=10.0,
        )
        market.products = [product]
        market.products_by_id = {"sku_food": product}
        market.client = object()
        market._require_active_filter = False
        market._active_sku_set = set()
        market._qdrant_mode = "local"
        market._search_cache = {}
        market._search_cache_max_entries = 16

        with patch.dict(os.environ, {"MODEL_PATH": ""}, clear=False):
            with patch("agenteconomy.center.ProductMarket.embedding") as embedding_mock:
                first = market.search_by_vector("apple", top_k=3)
                product.available_stock = 0.0
                second = market.search_by_vector("apple", top_k=3)

        embedding_mock.assert_not_called()
        self.assertEqual([p.product_id for p in first], ["sku_food"])
        self.assertEqual(second, [])

    def test_activate_skus_skips_unknown_local_products(self):
        market = ProductMarketClass.__new__(ProductMarketClass)
        market.products_by_id = {"sku_a": object(), "sku_b": object()}
        market._active_sku_set = set()
        market._collection_name = "products"
        market.logger = MagicMock()
        market.client = MagicMock()

        activated = market.activate_skus(["sku_a", "missing", "sku_a", "sku_b"])

        self.assertEqual(activated, 2)
        self.assertEqual(market._active_sku_set, {"sku_a", "sku_b"})
        market.client.set_payload.assert_called_once_with(
            collection_name="products",
            payload={"is_active": True},
            points=[
                ProductMarketClass._qdrant_point_id_for_product_id("sku_a"),
                ProductMarketClass._qdrant_point_id_for_product_id("sku_b"),
            ],
        )
        market.logger.warning.assert_called_once()

    def test_activate_skus_keeps_local_activation_when_qdrant_fails(self):
        market = ProductMarketClass.__new__(ProductMarketClass)
        market.products_by_id = {"sku_a": object(), "sku_b": object()}
        market._active_sku_set = set()
        market._collection_name = "products"
        market.logger = MagicMock()
        market.client = MagicMock()
        market.client.set_payload.side_effect = RuntimeError("qdrant unavailable")

        activated = market.activate_skus(["sku_a", "sku_b"])

        self.assertEqual(activated, 2)
        self.assertEqual(market._active_sku_set, {"sku_a", "sku_b"})
        self.assertEqual(market.client.set_payload.call_count, 3)
        market.logger.warning.assert_called()

    def test_apply_initial_stock_targets_sets_active_and_inactive_stock(self):
        market = ProductMarketClass.__new__(ProductMarketClass)
        sku_a = SimpleNamespace(product_id="sku_a", available_stock=100.0, manufacturer_price=2.0)
        sku_b = SimpleNamespace(product_id="sku_b", available_stock=100.0, manufacturer_price=3.0)
        sku_c = SimpleNamespace(product_id="sku_c", available_stock=100.0, manufacturer_price=4.0)
        market.products = [sku_a, sku_b, sku_c]
        market.products_by_id = {"sku_a": sku_a, "sku_b": sku_b, "sku_c": sku_c}
        market._search_cache = {("old",): ["sku_a"]}
        market.logger = MagicMock()

        stats = market.apply_initial_stock_targets(
            {"sku_a": 2.5, "sku_b": 0.0, "missing": 9.0},
            inactive_stock=0.0,
            min_active_stock=1.0,
        )

        self.assertEqual(sku_a.available_stock, 2.5)
        self.assertEqual(sku_b.available_stock, 1.0)
        self.assertEqual(sku_c.available_stock, 0.0)
        self.assertEqual(stats["active_sku_count"], 2)
        self.assertEqual(stats["missing_sku_count"], 1)
        self.assertEqual(stats["total_stock_before"], 300.0)
        self.assertEqual(stats["total_stock_after"], 3.5)
        self.assertAlmostEqual(stats["target_inventory_value"], 8.0)
        self.assertEqual(market._search_cache, {})

    def test_available_skus_respect_active_filter_mode(self):
        market = ProductMarketClass.__new__(ProductMarketClass)
        active = SimpleNamespace(product_id="active", available_stock=5.0)
        inactive = SimpleNamespace(product_id="inactive", available_stock=5.0)
        market.products = [active, inactive]
        market.products_by_industry = {"Food": [active, inactive]}
        market._require_active_filter = True
        market._active_sku_set = {"active"}

        by_industry = market.get_available_skus("Food")
        all_available = market.get_available_skus()

        self.assertEqual([p.product_id for p in by_industry], ["active"])
        self.assertEqual([p.product_id for p in all_available], ["active"])

    def test_price_adjustment_uses_active_skus_for_inventory_pressure(self):
        market = ProductMarketClass.__new__(ProductMarketClass)
        active = SimpleNamespace(
            product_id="active",
            manufacturer_price=10.0,
            retail_price=12.0,
            base_manufacturer_price=10.0,
            base_retail_price=12.0,
            available_stock=10.0,
        )
        inactive = SimpleNamespace(
            product_id="inactive",
            manufacturer_price=10.0,
            retail_price=12.0,
            base_manufacturer_price=10.0,
            base_retail_price=12.0,
            available_stock=10_000.0,
        )
        market.products_by_industry = {"Food": [active, inactive]}
        market.industry_supply_demand = {"Food": {"demand": 10.0, "supply": 10.0}}
        market._active_sku_set = {"active"}
        market.pricing_policy = PricingPolicy()
        market.price_policy_audit = {}
        market.logger = MagicMock()

        count = market.adjust_prices_by_supply_demand(
            "Food",
            base_adjustment=0.05,
            max_adjustment=0.15,
            mean_reversion_strength=0.0,
            inventory_sensitivity=0.03,
        )

        self.assertEqual(count, 1)
        self.assertAlmostEqual(active.manufacturer_price, 10.0)
        self.assertAlmostEqual(active.retail_price, 12.0)
        self.assertEqual(inactive.manufacturer_price, 10.0)
        self.assertEqual(inactive.retail_price, 12.0)

    def test_raw_material_price_adjustment_uses_active_skus(self):
        market = ProductMarketClass.__new__(ProductMarketClass)
        active = SimpleNamespace(
            product_id="active",
            manufacturer_price=10.0,
            retail_price=12.0,
            base_manufacturer_price=10.0,
            base_retail_price=12.0,
            available_stock=10.0,
        )
        inactive = SimpleNamespace(
            product_id="inactive",
            manufacturer_price=10.0,
            retail_price=12.0,
            base_manufacturer_price=10.0,
            base_retail_price=12.0,
            available_stock=10_000.0,
        )
        market.products_by_industry = {"311FT": [active, inactive]}
        market.raw_material_demand = {"311FT": {"current": 20.0, "previous": 10.0}}
        market._active_sku_set = {"active"}
        market.pricing_policy = PricingPolicy()
        market.price_policy_audit = {}
        market.logger = MagicMock()

        count = market.adjust_raw_material_prices(
            "Food and beverage and tobacco products",
            base_adjustment=0.05,
            max_adjustment=0.15,
            mean_reversion_strength=0.0,
        )

        self.assertEqual(count, 1)
        self.assertGreater(active.manufacturer_price, 10.0)
        self.assertGreater(active.retail_price, 12.0)
        self.assertEqual(inactive.manufacturer_price, 10.0)
        self.assertEqual(inactive.retail_price, 12.0)

    def test_retail_sale_is_clipped_to_retailer_owned_inventory(self):
        market = ProductMarketClass.__new__(ProductMarketClass)
        product = SimpleNamespace(product_id="sku_a", available_stock=5.0)
        market.products_by_id = {"sku_a": product}
        market.retailer_inventory = {}

        wholesale = market.purchase_manufacturer_stock("sku_a", 3.0)
        self.assertEqual(wholesale["actual_quantity"], 3.0)
        self.assertEqual(product.available_stock, 2.0)

        received = market.receive_retailer_inventory("retailer_1", "sku_a", wholesale["actual_quantity"])
        self.assertTrue(received["success"])
        self.assertEqual(market.get_seller_stock("sku_a", "retailer_1", True), 3.0)

        sale = market.purchase_from_seller_stock("sku_a", "retailer_1", 10.0, True)

        self.assertTrue(sale["success"])
        self.assertEqual(sale["actual_quantity"], 3.0)
        self.assertEqual(sale["shortage_quantity"], 7.0)
        self.assertEqual(market.retailer_inventory["retailer_1"]["sku_a"], 0.0)
        self.assertEqual(product.available_stock, 2.0)

        second_sale = market.purchase_from_seller_stock("sku_a", "retailer_1", 1.0, True)
        self.assertFalse(second_sale["success"])
        self.assertEqual(second_sale["actual_quantity"], 0.0)
        self.assertEqual(market.retailer_inventory["retailer_1"]["sku_a"], 0.0)
        self.assertEqual(product.available_stock, 2.0)

    def test_market_state_snapshot_preserves_retailer_inventory(self):
        market = ProductMarketClass.__new__(ProductMarketClass)
        market._require_active_filter = True
        market._active_sku_set = {"sku_a"}
        market.retailer_inventory = {
            "retailer_1": {"sku_a": 2.5, "sku_zero": 0.0},
        }
        market.raw_material_demand = {}
        market.industry_supply_demand = {}

        snapshot = market.get_market_state_snapshot()

        restored = ProductMarketClass.__new__(ProductMarketClass)
        restored._require_active_filter = False
        restored.retailer_inventory = {}
        restored.raw_material_demand = {}
        restored.industry_supply_demand = {}
        restored.logger = MagicMock()
        restored.restore_market_state(snapshot)

        self.assertEqual(restored.retailer_inventory, {"retailer_1": {"sku_a": 2.5}})


if __name__ == "__main__":
    unittest.main()
