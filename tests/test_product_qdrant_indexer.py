import unittest

from scripts.index_products_qdrant import (
    build_payload,
    build_product_embedding_text,
    stable_point_id,
)


def test_product_embedding_text_uses_search_relevant_fields():
    row = {
        "Product Name": "Apple Snack",
        "Brand": "Acme",
        "Category": "Grocery",
        "Industry_fixed": "Food and beverage and tobacco products",
        "Description": "Fresh dried apple slices",
    }

    text = build_product_embedding_text(row)

    assert "Apple Snack" in text
    assert "Food and beverage and tobacco products" in text
    assert "Fresh dried apple slices" in text


def test_stable_point_id_is_repeatable_integer():
    point_id = stable_point_id("sku-123")

    assert isinstance(point_id, int)
    assert point_id == stable_point_id("sku-123")
    assert point_id != stable_point_id("sku-124")


def test_payload_keeps_product_lookup_and_filter_fields():
    payload = build_payload(
        {
            "Uniq Id": "sku-123",
            "Product Name": "Apple Snack",
            "Category": "Grocery",
            "Industry_fixed": "Food and beverage and tobacco products",
            "Retailer_Code": "",
        }
    )

    assert payload["product_id"] == "sku-123"
    assert payload["industry"] == "311FT"
    assert payload["industry_name"] == "Food and beverage and tobacco products"
    assert payload["retailer_code"] == "445"
    assert payload["is_active"] is True


class ProductQdrantIndexerTests(unittest.TestCase):
    def test_product_embedding_text_uses_search_relevant_fields(self):
        test_product_embedding_text_uses_search_relevant_fields()

    def test_stable_point_id_is_repeatable_integer(self):
        test_stable_point_id_is_repeatable_integer()

    def test_payload_keeps_product_lookup_and_filter_fields(self):
        test_payload_keeps_product_lookup_and_filter_fields()


if __name__ == "__main__":
    unittest.main()
