from dotenv import load_dotenv
load_dotenv()
from typing import List, Optional, Dict, Any
import ray
import pandas as pd
from agenteconomy.center.Model import *
from agenteconomy.utils.logger import get_logger
from agenteconomy.utils.embedding import embedding
from agenteconomy.utils.product_attribute_loader import get_product_attributes
from agenteconomy.utils.load_qdrant_client import load_client
from agenteconomy.data.industry_cate_map import industry_cate_map
import os

@ray.remote(num_cpus=8)
class ProductMarket:
    """
    产品市场（Product Market）
    
    管理所有SKU级别的具体产品交易：
    - Category 1（制造业）生产的具体产品
    - Category 2（零售业）销售的具体产品
    
    NOT包括：
    - Category 3（抽象资源）由AbstractResourceMarket管理
    """
    
    def __init__(self):
        self.products: List[Product] = []  # 所有SKU
        self.products_by_id: Dict[str, Product] = {}  # product_id -> Product
        self.products_by_industry: Dict[str, List[Product]] = {}  # industry_code -> [Product]
        
        self.client = load_client()
        self.purchase_records: Dict[str, List[PurchaseRecord]] = {}
        self.logger = get_logger(name="product_market")
        
        # 行业平均价格缓存（用于中间品采购的等价单位计算）
        self.industry_avg_prices: Dict[str, Dict[str, float]] = {}
        # 格式: {"manufacturer": {"315AL": 50.0}, "retail": {"441": 80.0}}
        
        self.logger.info(f"ProductMarket initialized")

    def initialize_products(self, csv_path: Optional[str] = None):
        """
        从CSV初始化所有产品
        
        Args:
            csv_path: CSV文件路径，默认使用 products_with_supply_chain_prices.csv
        """
        if csv_path is None:
            csv_path = os.path.join(
                os.path.dirname(__file__),
                "../data/products_with_supply_chain_prices.csv"
            )
        
        self.logger.info(f"Loading products from {csv_path}")
        products_df = pd.read_csv(csv_path)
        for _, row in products_df.iterrows():
            try:
                product = Product.create(
                    name=row['Product Name'],
                    product_id=row['Uniq Id'],
                    manufacturer_price=float(row['Manufacturer_Price']),
                    base_manufacturer_price=float(row['Manufacturer_Price']),
                    wholesale_price=float(row['Wholesale_Price']) if pd.notna(row['Wholesale_Price']) else None,
                    base_wholesale_price=float(row['Wholesale_Price']) if pd.notna(row['Wholesale_Price']) else None,
                    retail_price=float(row['List Price']),
                    base_retail_price=float(row['List Price']),
                    has_wholesale_layer=bool(row['Has_Wholesale_Layer']),
                    manufacturer_code=row['Manufacturer_Code'],
                    retailer_code=row['Retailer_Code'],
                    owner_id=row['Manufacturer_Code'],  # 初始拥有者为制造商
                    amount=1000,
                    classification=row.get('Industry', None),
                    description=row.get('Description', None),
                    brand=row.get('Brand', None),
                    available_stock=100,  # 初始库存为0，由制造商生产后添加
                )
                self.add_product(product)
            except Exception as e:
                self.logger.error(f"Failed to create product from row: {e}")
                continue
        
        # 计算行业平均价格
        self._calculate_industry_avg_prices()
        
        self.logger.info(f"Loaded {len(self.products)} products")
        self.logger.info(f"Covered {len(self.products_by_industry)} industries")

    def add_product(self, product: Product):
        """添加产品到市场"""
        self.products.append(product)
        self.products_by_id[product.product_id] = product
        
        # 按制造商行业分类
        mfg_code = product.manufacturer_code
        if mfg_code not in self.products_by_industry:
            self.products_by_industry[mfg_code] = []
        self.products_by_industry[mfg_code].append(product)
        
        self.logger.debug(f"Product {product.product_id} ({product.name}) added to market")
    
    def _calculate_industry_avg_prices(self):
        """
        计算各行业的平均价格
        用于中间品采购的等价单位换算
        """
        # 按制造商行业分组
        manufacturer_prices: Dict[str, List[float]] = {}
        retail_prices: Dict[str, List[float]] = {}
        
        for product in self.products:
            mfg_code = product.manufacturer_code
            ret_code = product.retailer_code
            
            # 制造商价格
            if mfg_code not in manufacturer_prices:
                manufacturer_prices[mfg_code] = []
            manufacturer_prices[mfg_code].append(product.base_manufacturer_price)
            
            # 零售商价格
            if ret_code not in retail_prices:
                retail_prices[ret_code] = []
            retail_prices[ret_code].append(product.base_retail_price)
        
        # 计算平均值
        self.industry_avg_prices = {
            "manufacturer": {
                code: sum(prices) / len(prices)
                for code, prices in manufacturer_prices.items()
            },
            "retailer": {
                code: sum(prices) / len(prices)
                for code, prices in retail_prices.items()
            }
        }
        
        self.logger.info(f"Calculated average prices for {len(self.industry_avg_prices['manufacturer'])} manufacturer industries")
    
    def get_industry_avg_price(self, industry_code: str, price_type: str = "manufacturer") -> float:
        """
        获取行业平均价格
        
        Args:
            industry_code: 行业代码
            price_type: "manufacturer" 或 "retailer"
            
        Returns:
            行业平均价格，如果找不到则返回1.0
        """
        return self.industry_avg_prices.get(price_type, {}).get(industry_code, 1.0)
    
    def get_skus_by_industry(
        self,
        industry_code: str,
        available_only: bool = True
    ) -> List[Product]:
        """
        按行业代码获取SKU列表
        
        Args:
            industry_code: 制造商行业代码
            available_only: 是否只返回有库存的
            
        Returns:
            产品列表
        """
        products = self.products_by_industry.get(industry_code, [])
        
        if available_only:
            products = [p for p in products if p.available_stock > 0]
        
        return products
    
    def get_available_skus(
        self,
        industry: Optional[str] = None,
        period: Optional[int] = None
    ) -> List[Product]:
        """
        获取可用SKU
        
        Args:
            industry: 行业代码（可选）
            period: 期数（可选，暂未使用）
            
        Returns:
            可用产品列表
        """
        if industry:
            return self.get_skus_by_industry(industry, available_only=True)
        else:
            return [p for p in self.products if p.available_stock > 0]

    def publish_product(self, product: Product):
        """
        发布产品到市场（加载完整属性）
        """
        # 如果产品没有属性，尝试从数据库加载
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

        self.add_product(product)
    
    def update_stock(self, product_id: str, quantity_change: float):
        """
        更新库存
        
        Args:
            product_id: 产品ID
            quantity_change: 库存变化量（正数增加，负数减少）
        """
        product = self.products_by_id.get(product_id)
        if product:
            product.available_stock += quantity_change
            if product.available_stock < 0:
                self.logger.warning(f"Product {product_id} has negative stock: {product.available_stock}")
        else:
            self.logger.error(f"Product {product_id} not found")
    
    def reserve_stock(self, product_id: str, quantity: float) -> bool:
        """
        预留库存
        
        Args:
            product_id: 产品ID
            quantity: 预留数量
            
        Returns:
            是否成功
        """
        product = self.products_by_id.get(product_id)
        if not product:
            return False
        
        if product.available_stock >= quantity:
            product.available_stock -= quantity
            product.reserved_stock += quantity
            return True
        else:
            return False
    
    def release_reservation(self, product_id: str, quantity: float):
        """
        释放预留库存
        
        Args:
            product_id: 产品ID
            quantity: 释放数量
        """
        product = self.products_by_id.get(product_id)
        if product:
            product.reserved_stock -= quantity
            product.available_stock += quantity
    
    def confirm_reservation(self, product_id: str, quantity: float):
        """
        确认预留（完成交易）
        
        Args:
            product_id: 产品ID
            quantity: 确认数量
        """
        product = self.products_by_id.get(product_id)
        if product:
            product.reserved_stock -= quantity
            # 不增加available_stock，因为已经卖出了

    def search_by_vector(self, query: str, top_k: int = 20, must_contain: Optional[str] = None) -> List[Product]:
        """
        使用向量搜索查找相似产品
        
        Args:
            query: 搜索查询
            top_k: 返回结果数量
            must_contain: 必须包含的分类
            
        Returns:
            产品列表
        """
        def _fallback_from_published() -> List[Product]:
            """Fallback逻辑：使用简单的文本匹配"""
            candidates = [p for p in (self.products or []) if float(getattr(p, "available_stock", 0.0) or 0.0) > 0]
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

        # 如果没有向量数据库，使用fallback
        if not getattr(self, "client", None):
            return _fallback_from_published()

        if top_k <= 0:
            return []

        query = query or ""
        if not query.strip():
            return _fallback_from_published()

        must_contain_lc = must_contain.lower() if must_contain else None
        collection_name = os.getenv("QDRANT_COLLECTION_NAME", "products")
        search_limit = max(top_k * 3, 50)
        max_fetch = max(top_k * 10, search_limit)
        try:
            query_embedding = embedding(query)

            results: List[Product] = []
            seen_ids = set()
            offset = 0
            products_by_id = self.products_by_id

            while len(results) < top_k and offset < max_fetch:
                hits = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    limit=search_limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                if not hits:
                    break

                offset += len(hits)

                for hit in hits:
                    payload = hit.payload or {}
                    product_id = payload.get("product_id")
                    if not product_id:
                        hit_id = getattr(hit, "id", None)
                        if hit_id is not None:
                            product_id = str(hit_id)

                    if not product_id or product_id in seen_ids:
                        continue

                    product = products_by_id.get(product_id)
                    if not product:
                        continue

                    if must_contain_lc:
                        classification = product.classification or ""
                        if must_contain_lc not in classification.lower():
                            continue

                    if product.available_stock <= 0:
                        continue

                    results.append(product)
                    seen_ids.add(product_id)
                    if len(results) >= top_k:
                        break

            if results:
                return results
            return _fallback_from_published()
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return _fallback_from_published()
    
    def get_product(self, product_id: str) -> Optional[Product]:
        """获取单个产品"""
        return self.products_by_id.get(product_id)
    
    def get_market_stats(self) -> Dict[str, Any]:
        """
        获取市场统计信息
        
        Returns:
            统计数据字典
        """
        total_products = len(self.products)
        available_products = sum(1 for p in self.products if p.available_stock > 0)
        total_stock = sum(p.available_stock for p in self.products)
        total_value = sum(p.available_stock * p.manufacturer_price for p in self.products)
        
        return {
            "total_skus": total_products,
            "available_skus": available_products,
            "total_stock": total_stock,
            "total_value": total_value,
            "industries_covered": len(self.products_by_industry),
            "avg_manufacturer_price": sum(p.manufacturer_price for p in self.products) / total_products if total_products > 0 else 0,
            "avg_retail_price": sum(p.retail_price for p in self.products) / total_products if total_products > 0 else 0,
        }


if __name__ == "__main__":
    pro_m = ProductMarket.remote()
    pro_m.initialize_products.remote()
    print(ray.get(pro_m.get_market_stats.remote()))