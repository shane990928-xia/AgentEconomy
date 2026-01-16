import json
import os
from functools import lru_cache
from typing import Any, Dict, Optional

from agenteconomy.utils.logger import get_logger

logger = get_logger(name="product_attribute_loader")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "consumer_modeling", "family_attribute_config.json")


@lru_cache(maxsize=1)
def _load_attribute_map(config_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Load product attribute mapping and index by product_id."""
    path = config_path or DEFAULT_CONFIG_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as exc:
        logger.error(f"Failed to load attribute config ({path}): {exc}")
        return {}

    attribute_file = config.get("product_attribute_file")
    if not attribute_file or not os.path.exists(attribute_file):
        logger.warning(f"Product attribute file not found: {attribute_file}")
        return {}

    try:
        with open(attribute_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        mappings = data.get("product_mappings", [])
        indexed = {}
        for item in mappings:
            product_id = item.get("product_id")
            if product_id:
                indexed[product_id] = item
        logger.info(f"Loaded {len(indexed)} product attribute records from {attribute_file}")
        return indexed
    except Exception as exc:
        logger.error(f"Failed to load product attribute file ({attribute_file}): {exc}")
    return {}


def get_product_attributes(product_id: Optional[str], config_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Return attribute payload for a product if available."""
    if not product_id:
        return None
    attr_map = _load_attribute_map(config_path)
    return attr_map.get(product_id)


def inject_product_attributes(product_kwargs: Dict[str, Any], product_id: Optional[str],
                              config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Merge attribute payload into product kwargs if available.
    Existing values in product_kwargs take precedence.
    """
    attrs = get_product_attributes(product_id, config_path)
    if not attrs:
        return product_kwargs

    enriched = dict(product_kwargs)
    enriched.setdefault("attributes", attrs)
    enriched.setdefault("is_food", attrs.get("is_food"))

    if attrs.get("nutrition_supply"):
        enriched.setdefault("nutrition_supply", attrs.get("nutrition_supply"))
    if attrs.get("satisfaction_attributes"):
        enriched.setdefault("satisfaction_attributes", attrs.get("satisfaction_attributes"))
    if attrs.get("duration_months") is not None:
        enriched.setdefault("duration_months", attrs.get("duration_months"))

    return enriched

