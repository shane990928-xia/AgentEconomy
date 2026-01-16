from dotenv import load_dotenv
load_dotenv()
from typing import List, Optional, Dict, Any
import ray
from agenteconomy.center.Model import *
from agenteconomy.utils.logger import get_logger
from agenteconomy.utils.embedding import embedding
from agenteconomy.utils.product_attribute_loader import get_product_attributes
import os


@ray.remote(num_cpus=32)
class ProductMarket:
    def __init__(self):
        self.products: List[Product] = []
        self.client = []
        self.purchase_records: Dict[str, List[PurchaseRecord]] = {}
        self.logger = get_logger(name="product_market")
        self.logger.info(f"ProductMarket initialized")

    def add_product(self, product: Product):
        self.products.append(product)
        self.logger.info(f"Product {product.product_id} added to the market")

    def publish_product(self, product: Product):

        if not getattr(product, "attributes", None) and getattr(product, "product_id", None):
            attrs = get_product_attributes(product.product_id)
            if attrs:
                product.attributes = attrs
                if product.is_food is None:
                    product.is_food = attrs.get("is_food")
                if product.nutrition_supply is None:
                    product.nutrition_supply = attrs.get("nutrition_supply")
                if product.satisfaction_attributes is None:
                    product.satisfaction_attributes = attrs.get("satisfaction_attributes")
                if product.duration_months is None:
                    product.duration_months = attrs.get("duration_months")

        self.products.append(product)

    def search_by_vector(self, query: str, top_k: int = 20, must_contain: Optional[str] = None) -> List[Product]:
        """
        Use vector search to find products similar to the query.
        """
        def _fallback_from_published() -> List[Product]:
            candidates = [p for p in (self.products or []) if float(getattr(p, "amount", 0.0) or 0.0) > 0]
            if must_contain:
                mc = must_contain.lower()
                candidates = [p for p in candidates if mc in (getattr(p, "classification", "") or "").lower()]
            ql = (query or "").lower()
            if ql:
                ranked = []
                for p in candidates:
                    text = " ".join(
                        [
                            str(getattr(p, "name", "") or ""),
                            str(getattr(p, "brand", "") or ""),
                            str(getattr(p, "classification", "") or ""),
                            str(getattr(p, "description", "") or ""),
                        ]
                    ).lower()
                    score = 1 if ql in text else 0
                    ranked.append((score, p))
                ranked.sort(key=lambda x: x[0], reverse=True)
                out = [p for s, p in ranked if s > 0][:top_k]
                if len(out) < top_k:
                    out.extend([p for s, p in ranked if s <= 0][: (top_k - len(out))])
                return out[:top_k]
            return candidates[:top_k]

        search_limit = top_k * 3
        query_embedding = embedding(query)
        if not getattr(self, "client", None):
            return _fallback_from_published()

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=search_limit,    
        )
        results = []
        for hit in hits:
            payload = hit.payload
        for product in self.products:
            if product.attributes:
                product_embedding = product.attributes.get("embedding")
                if product_embedding:
                    similarity = cosine_similarity(query_embedding, product_embedding)
                    results.append((product, similarity))
        results.sort(key=lambda x: x[1], reverse=True)
        return [product for product, similarity in results[:top_k]]