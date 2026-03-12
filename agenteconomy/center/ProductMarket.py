from dotenv import load_dotenv
load_dotenv()
from typing import List, Optional, Dict, Any, Set
import warnings
import threading
import ray
import pandas as pd
import numpy as np
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointIdsList
from agenteconomy.center.Model import *
from agenteconomy.utils.logger import get_logger
from agenteconomy.utils.embedding import embedding
from agenteconomy.utils.product_attribute_loader import get_product_attributes
from agenteconomy.utils.load_qdrant_client import load_client
import os

# 抑制 Qdrant 本地模式的数值计算警告（不影响功能）
warnings.filterwarnings("ignore", category=RuntimeWarning, module="qdrant_client.local")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

# 制造业名称 -> 零售商代码 映射
# 基于 Industry_fixed 列的行业名称（不是IO代码）
# 注意：零售商只有 441, 445, 452, 4A0, 722 这几个
MANUFACTURER_TO_RETAILER = {
    # 食品饮料相关 -> 445 Food and beverage stores
    "Food and beverage and tobacco products": "445",
    "Farms": "445",
    
    # 汽车相关 -> 441 Motor vehicle and parts dealers
    "Motor vehicles, bodies and trailers, and parts": "441",
    "Other transportation equipment": "441",
    
    # 其他所有制造业 -> 452 General merchandise stores
    "Apparel and leather and allied products": "452",
    "Textile mills and textile product mills": "452",
    "Computer and electronic products": "452",
    "Electrical equipment, appliances, and components": "452",
    "Chemical products": "452",
    "Miscellaneous manufacturing": "452",
    "Plastics and rubber products": "452",
    "Fabricated metal products": "452",
    "Machinery": "452",
    "Furniture and related products": "452",
    "Paper products": "452",
    "Wood products": "452",
    "Nonmetallic mineral products": "452",
    
    # 工业/专业产品 -> 4A0 Other retail
    "Forestry, fishing, and related activities": "4A0",
    "Petroleum and coal products": "4A0",
    "Publishing industries, except internet (includes software)": "4A0",
}

DEFAULT_RETAILER_CODE = "452"  # 默认综合百货


def get_retailer_from_manufacturer(manufacturer_code: str) -> str:
    """根据制造商行业代码获取对应的零售商代码"""
    if not manufacturer_code:
        return DEFAULT_RETAILER_CODE
    return MANUFACTURER_TO_RETAILER.get(str(manufacturer_code), DEFAULT_RETAILER_CODE)

# 控制 Ray actor 并发：不再无限制放开，避免 Qdrant 被打爆
# 实际 Qdrant 查询并发由内部信号量 _qdrant_semaphore 控制
@ray.remote(num_cpus=8, max_concurrency=200)
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
        self.products_by_industry: Dict[str, List[Product]] = {}  # manufacturer_code -> [Product]
        self.products_by_retailer: Dict[str, List[Product]] = {}  # retailer_code -> [Product]
        
        self.client = load_client()
        self.purchase_records: Dict[str, List[PurchaseRecord]] = {}
        self.logger = get_logger(name="product_market")
        
        # Qdrant 并发控制：限制同时发往 Qdrant 的查询数量，防止超时
        # 默认 20，可通过环境变量调整
        self._qdrant_max_concurrency = int(os.getenv("QDRANT_MAX_CONCURRENCY", "20"))
        self._qdrant_semaphore = threading.Semaphore(self._qdrant_max_concurrency)
        
        # 行业平均价格缓存（用于中间品采购的等价单位计算）
        self.industry_avg_prices: Dict[str, Dict[str, float]] = {}
        # 格式: {"manufacturer": {"315AL": 50.0}, "retail": {"441": 80.0}}
        
        # Qdrant collection name
        self._collection_name = os.getenv("QDRANT_COLLECTION_NAME", "products")
        
        # Qdrant 模式：cloud/docker 使用原生 filter，local 使用本地 filter
        self._qdrant_mode = os.getenv("QDRANT_MODE", "local")
        self._use_qdrant_filter = self._qdrant_mode in ("cloud", "docker")
        
        # 活跃SKU追踪
        self._active_sku_set: Set[str] = set()
        self._require_active_filter: bool = False  # 是否在搜索时强制过滤is_active
        
        # 零售商库存追踪 (retailer_id -> {product_id -> stock})
        self.retailer_inventory: Dict[str, Dict[str, float]] = {}
        
        # 供需追踪（按行业）：用于价格调整
        # {manufacturer_code: {"demand": float, "supply": float}}
        self.industry_supply_demand: Dict[str, Dict[str, float]] = {}

        # 原材料需求追踪（按行业，以价值为单位）
        # {industry_code: {"current": float, "previous": float}}
        self.raw_material_demand: Dict[str, Dict[str, float]] = {}

        filter_strategy = "Qdrant native filter" if self._use_qdrant_filter else "local Python filter"
        self.logger.info(f"ProductMarket initialized (mode={self._qdrant_mode}, filter_strategy={filter_strategy})")

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
        
        # 统计零售商分布
        retailer_counts = {}
        manufacturer_counts = {}
        
        for _, row in products_df.iterrows():
            try:
                # 使用 Industry_fixed 作为制造商代码
                manufacturer_code = str(row['Industry_fixed']) if pd.notna(row.get('Industry_fixed')) else None
                
                if not manufacturer_code:
                    self.logger.warning(f"Skipping product {row['Uniq Id']}: no Industry_fixed")
                    continue
                
                # 根据制造商代码映射零售商代码
                retailer_code = get_retailer_from_manufacturer(manufacturer_code)
                
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
                    manufacturer_code=manufacturer_code,
                    retailer_code=retailer_code,
                    owner_id=manufacturer_code,  # 初始拥有者为制造商
                    amount=1000,
                    classification=manufacturer_code,  # classification 也用 Industry_fixed
                    description=str(row['Description']) if pd.notna(row.get('Description')) else None,
                    brand=str(row['Brand']) if pd.notna(row.get('Brand')) else None,
                    available_stock=100,  # 初始库存
                    category=None,  # 不再使用 Category 字段
                )
                self.add_product(product)
                
                # 统计
                retailer_counts[retailer_code] = retailer_counts.get(retailer_code, 0) + 1
                manufacturer_counts[manufacturer_code] = manufacturer_counts.get(manufacturer_code, 0) + 1
            except Exception as e:
                self.logger.error(f"Failed to create product from row: {e}")
                continue
        
        # 计算行业平均价格
        self._calculate_industry_avg_prices()
        
        self.logger.info(f"Loaded {len(self.products)} products")
        self.logger.info(f"Covered {len(self.products_by_industry)} manufacturer industries: {list(self.products_by_industry.keys())}")
        self.logger.info(f"Retailer distribution: {retailer_counts}")

    def add_product(self, product: Product):
        """添加产品到市场"""
        self.products.append(product)
        self.products_by_id[product.product_id] = product
        
        # 按制造商行业分类
        mfg_code = product.manufacturer_code
        if mfg_code not in self.products_by_industry:
            self.products_by_industry[mfg_code] = []
        self.products_by_industry[mfg_code].append(product)
        
        # 按零售商分类
        retailer_code = product.retailer_code
        if retailer_code:
            if retailer_code not in self.products_by_retailer:
                self.products_by_retailer[retailer_code] = []
            self.products_by_retailer[retailer_code].append(product)
        
        self.logger.debug(f"Product {product.product_id} ({product.name}) added to market, retailer={retailer_code}")
    
    def get_price(self, product_id: str) -> float:
        """
        Get current retail price for a product.
        """
        product = self.products_by_id.get(product_id)
        if not product:
            return 0.0
        return float(getattr(product, "retail_price", 0.0) or 0.0)

    def get_product_snapshot(self, product_id: str) -> Optional[Dict[str, Any]]:
        """
        Return a lightweight snapshot for a product (price/stock/metadata).
        """
        product = self.products_by_id.get(product_id)
        if not product:
            return None
        return {
            "product_id": product.product_id,
            "name": product.name,
            "description": product.description,
            "retail_price": float(getattr(product, "retail_price", 0.0) or 0.0),
            "base_retail_price": float(getattr(product, "base_retail_price", 0.0) or 0.0),
            "manufacturer_price": float(getattr(product, "manufacturer_price", 0.0) or 0.0),
            "base_manufacturer_price": float(getattr(product, "base_manufacturer_price", 0.0) or 0.0),
            "available_stock": float(getattr(product, "available_stock", 0.0) or 0.0),
            "manufacturer_code": getattr(product, "manufacturer_code", None),
            "retailer_code": getattr(product, "retailer_code", None),
            "category": getattr(product, "category", None),
        }

    # ========== 零售商相关方法 ==========
    
    def get_products_by_retailer(self, retailer_code: str) -> List[Dict[str, Any]]:
        """
        获取某零售商负责销售的所有产品列表
        Returns: List of product snapshots
        """
        products = self.products_by_retailer.get(retailer_code, [])
        return [self.get_product_snapshot(p.product_id) for p in products]
    
    def get_retailer_product_ids(self, retailer_code: str) -> List[str]:
        """
        获取某零售商负责销售的所有产品ID列表
        """
        products = self.products_by_retailer.get(retailer_code, [])
        return [p.product_id for p in products]
    
    def get_retailer_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        获取各零售商的统计信息
        Returns: {retailer_code: {product_count, total_stock, avg_price, categories}}
        """
        stats = {}
        for retailer_code, products in self.products_by_retailer.items():
            total_stock = sum(p.available_stock for p in products)
            prices = [p.retail_price for p in products if p.retail_price > 0]
            categories = set(p.category for p in products if p.category)
            
            stats[retailer_code] = {
                "product_count": len(products),
                "total_stock": total_stock,
                "avg_price": sum(prices) / len(prices) if prices else 0.0,
                "categories": list(categories)
            }
        return stats

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

    # ========== 价格动态调整方法 ==========
    
    def update_manufacturer_price(
        self,
        product_id: str,
        new_price: float,
        reason: Optional[str] = None
    ) -> bool:
        """
        更新制造商价格（批发价）
        
        Args:
            product_id: 产品ID
            new_price: 新的制造商价格
            reason: 调价原因（用于日志）
            
        Returns:
            是否成功更新
        """
        product = self.products_by_id.get(product_id)
        if not product:
            self.logger.warning(f"Product {product_id} not found for price update")
            return False
        
        old_price = product.manufacturer_price
        product.manufacturer_price = max(0.01, new_price)  # 价格不能低于0.01
        
        self.logger.debug(
            f"Updated manufacturer price for {product_id}: "
            f"{old_price:.2f} -> {new_price:.2f} ({reason or 'no reason'})"
        )
        return True
    
    def update_retail_price(
        self,
        product_id: str,
        new_price: float,
        reason: Optional[str] = None
    ) -> bool:
        """
        更新零售价格
        
        Args:
            product_id: 产品ID
            new_price: 新的零售价格
            reason: 调价原因（用于日志）
            
        Returns:
            是否成功更新
        """
        product = self.products_by_id.get(product_id)
        if not product:
            self.logger.warning(f"Product {product_id} not found for price update")
            return False
        
        old_price = product.retail_price
        product.retail_price = max(0.01, new_price)  # 价格不能低于0.01
        
        self.logger.debug(
            f"Updated retail price for {product_id}: "
            f"{old_price:.2f} -> {new_price:.2f} ({reason or 'no reason'})"
        )
        return True
    
    def update_prices_by_cost(
        self,
        product_id: str,
        actual_unit_cost: float,
        manufacturer_margin: float = 0.15,
        retail_margin: float = 0.25
    ) -> bool:
        """
        根据实际成本更新制造商价格和零售价格
        
        价格计算逻辑：
        - manufacturer_price = actual_unit_cost * (1 + manufacturer_margin)
        - retail_price = manufacturer_price * (1 + retail_margin)
        
        Args:
            product_id: 产品ID
            actual_unit_cost: 实际单位成本（中间品 + 劳动力）
            manufacturer_margin: 制造商利润率（默认15%）
            retail_margin: 零售商利润率（默认25%）
            
        Returns:
            是否成功更新
        """
        product = self.products_by_id.get(product_id)
        if not product:
            return False
        
        # 计算新价格
        new_mfg_price = actual_unit_cost * (1 + manufacturer_margin)
        new_retail_price = new_mfg_price * (1 + retail_margin)
        
        # 更新产品的 unit_cost 记录
        product.unit_cost = actual_unit_cost
        
        # 更新价格
        old_mfg = product.manufacturer_price
        old_retail = product.retail_price
        
        product.manufacturer_price = max(0.01, new_mfg_price)
        product.retail_price = max(0.01, new_retail_price)
        
        self.logger.debug(
            f"Updated prices for {product_id} based on cost {actual_unit_cost:.2f}: "
            f"mfg {old_mfg:.2f}->{new_mfg_price:.2f}, retail {old_retail:.2f}->{new_retail_price:.2f}"
        )
        return True
    
    def batch_update_prices_by_industry(
        self,
        manufacturer_code: str,
        avg_unit_cost: float,
        manufacturer_margin: float = 0.15,
        retail_margin: float = 0.25,
        smoothing_factor: float = 0.15,
        max_change_ratio: float = 0.05
    ) -> int:
        """
        批量更新某制造业行业所有产品的价格
        
        用于制造商在生产后统一调整价格
        
        价格调整机制：
        1. 根据实际成本计算目标价格
        2. 使用平滑因子避免价格剧烈波动：new = old * (1-α) + target * α
        3. 限制单次调整幅度不超过 max_change_ratio
        
        Args:
            manufacturer_code: 制造商行业代码
            avg_unit_cost: 平均单位成本
            manufacturer_margin: 制造商利润率
            retail_margin: 零售商利润率
            smoothing_factor: 平滑因子（0-1），越大越接近目标价格
            max_change_ratio: 单次最大调整比例（如0.2表示最多涨跌20%）
            
        Returns:
            更新的产品数量
        """
        products = self.products_by_industry.get(manufacturer_code, [])
        if not products:
            return 0
        
        updated_count = 0
        for product in products:
            # 根据基准价格比例调整（保持产品间的相对价格差异）
            base_mfg = product.base_manufacturer_price
            if base_mfg > 0:
                # 计算基准成本（基准价格 / (1 + margin)）
                base_cost = base_mfg / (1 + manufacturer_margin)
                # 成本变化比例
                cost_ratio = avg_unit_cost / base_cost if base_cost > 0 else 1.0
                # 目标价格 = 基准价格 * 成本变化比例
                target_mfg_price = base_mfg * cost_ratio
                target_retail_price = product.base_retail_price * cost_ratio
            else:
                target_mfg_price = avg_unit_cost * (1 + manufacturer_margin)
                target_retail_price = target_mfg_price * (1 + retail_margin)
            
            # 平滑调整：new = old * (1-α) + target * α
            old_mfg = product.manufacturer_price
            old_retail = product.retail_price
            
            smoothed_mfg = old_mfg * (1 - smoothing_factor) + target_mfg_price * smoothing_factor
            smoothed_retail = old_retail * (1 - smoothing_factor) + target_retail_price * smoothing_factor
            
            # 限制单次调整幅度
            if old_mfg > 0:
                change_ratio_mfg = (smoothed_mfg - old_mfg) / old_mfg
                if abs(change_ratio_mfg) > max_change_ratio:
                    if change_ratio_mfg > 0:
                        smoothed_mfg = old_mfg * (1 + max_change_ratio)
                    else:
                        smoothed_mfg = old_mfg * (1 - max_change_ratio)
            
            if old_retail > 0:
                change_ratio_retail = (smoothed_retail - old_retail) / old_retail
                if abs(change_ratio_retail) > max_change_ratio:
                    if change_ratio_retail > 0:
                        smoothed_retail = old_retail * (1 + max_change_ratio)
                    else:
                        smoothed_retail = old_retail * (1 - max_change_ratio)
            
            # 更新价格
            product.unit_cost = avg_unit_cost
            product.manufacturer_price = max(0.01, smoothed_mfg)
            product.retail_price = max(0.01, smoothed_retail)
            updated_count += 1
        
        self.logger.info(
            f"Batch updated {updated_count} products for industry {manufacturer_code}, "
            f"avg_cost={avg_unit_cost:.2f}"
        )
        return updated_count

    # ========== 供需追踪与价格调整 ==========
    
    def record_demand(self, manufacturer_code: str, demand_qty: float):
        """
        记录某行业的需求量
        
        Args:
            manufacturer_code: 制造商行业代码
            demand_qty: 需求数量
        """
        if manufacturer_code not in self.industry_supply_demand:
            self.industry_supply_demand[manufacturer_code] = {"demand": 0.0, "supply": 0.0}
        self.industry_supply_demand[manufacturer_code]["demand"] += demand_qty
    
    def record_supply(self, manufacturer_code: str, supply_qty: float):
        """
        记录某行业的供给量（生产量）
        
        Args:
            manufacturer_code: 制造商行业代码
            supply_qty: 供给数量
        """
        if manufacturer_code not in self.industry_supply_demand:
            self.industry_supply_demand[manufacturer_code] = {"demand": 0.0, "supply": 0.0}
        self.industry_supply_demand[manufacturer_code]["supply"] += supply_qty
    
    def get_supply_demand_ratio(self, manufacturer_code: str) -> float:
        """
        获取某行业的供需比
        
        Returns:
            供需比（supply/demand），>1表示供过于求，<1表示供不应求
            如果没有需求记录或没有供给记录，返回1.0（均衡状态，不调价）
        """
        stats = self.industry_supply_demand.get(manufacturer_code, {})
        demand = stats.get("demand", 0.0)
        supply = stats.get("supply", 0.0)
        
        if demand <= 0 or supply <= 0:
            return 1.0  # 无需求或无供给时视为均衡，不调价
        return supply / demand
    
    def get_all_supply_demand_stats(self) -> Dict[str, Dict[str, float]]:
        """
        获取所有行业的供需统计数据
        
        Returns:
            {industry_code: {demand: float, supply: float, ratio: float}}
        """
        result = {}
        for code, stats in self.industry_supply_demand.items():
            demand = stats.get("demand", 0.0)
            supply = stats.get("supply", 0.0)
            ratio = supply / demand if demand > 0 else 1.0
            result[code] = {
                "demand": demand,
                "supply": supply,
                "ratio": ratio
            }
        return result
    
    def reset_supply_demand_tracking(self):
        """
        重置供需追踪数据（每月初调用）
        """
        self.industry_supply_demand = {}
        self.logger.debug("Supply-demand tracking reset")

    # ========== 原材料价格调整（基于需求变化） ==========

    def record_raw_material_demand(self, industry_code: str, demand_value: float):
        """
        记录原材料行业的需求（以价值为单位）

        原材料行业（作为中间品供应商）的需求来自下游制造商的采购
        这里记录的是采购金额，不是数量

        Args:
            industry_code: 原材料行业代码
            demand_value: 需求价值（美元）
        """
        if industry_code not in self.raw_material_demand:
            self.raw_material_demand[industry_code] = {"current": 0.0, "previous": 0.0}
        self.raw_material_demand[industry_code]["current"] += demand_value

    def get_raw_material_demand_change_ratio(self, industry_code: str) -> float:
        """
        获取原材料行业的需求变化率

        Returns:
            需求变化率（current/previous）
            - > 1: 需求增加
            - < 1: 需求减少
            - = 1: 需求不变或无历史数据
        """
        stats = self.raw_material_demand.get(industry_code, {})
        current = stats.get("current", 0.0)
        previous = stats.get("previous", 0.0)

        if previous <= 0:
            return 1.0  # 无历史数据时视为均衡
        return current / previous

    def adjust_raw_material_prices(
        self,
        industry_code: str,
        base_adjustment: float = 0.03,
        max_adjustment: float = 0.10
    ) -> int:
        """
        根据需求变化调整原材料行业的价格

        价格调整逻辑：
        - 需求增加（change_ratio > 1）：涨价
        - 需求减少（change_ratio < 1）：降价
        - 调整幅度 = base_adjustment * |ln(change_ratio)|，最大不超过 max_adjustment

        Args:
            industry_code: 原材料行业代码
            base_adjustment: 基础调整系数（默认3%，比消费品更保守）
            max_adjustment: 最大调整幅度（默认10%）

        Returns:
            更新的产品数量
        """
        change_ratio = self.get_raw_material_demand_change_ratio(industry_code)
        products = self.products_by_industry.get(industry_code, [])

        if not products:
            return 0

        # 如果没有历史数据或变化很小，不调整
        if change_ratio == 1.0 or abs(change_ratio - 1.0) < 0.02:
            self.logger.debug(
                f"Raw material {industry_code}: no price adjustment (change_ratio={change_ratio:.3f})"
            )
            return 0

        # 计算调整幅度：使用对数函数使调整更平滑
        import math
        log_ratio = math.log(change_ratio) if change_ratio > 0 else 0
        adjustment = base_adjustment * abs(log_ratio)
        adjustment = min(adjustment, max_adjustment)

        # 需求增加时涨价，需求减少时降价
        if change_ratio > 1:
            price_multiplier = 1 + adjustment  # 涨价
        else:
            price_multiplier = 1 - adjustment  # 降价

        updated_count = 0
        for product in products:
            old_price = product.manufacturer_price
            product.manufacturer_price = max(0.01, product.manufacturer_price * price_multiplier)
            product.retail_price = max(0.01, product.retail_price * price_multiplier)
            updated_count += 1

        self.logger.info(
            f"Raw material {industry_code} price adjusted: "
            f"change_ratio={change_ratio:.3f}, multiplier={price_multiplier:.3f}, "
            f"products={updated_count}"
        )
        return updated_count

    def finalize_raw_material_demand(self):
        """
        结束当期原材料需求记录，将当期需求转为历史需求

        在每月结束时调用，为下一期的需求变化计算做准备
        """
        for industry_code in self.raw_material_demand:
            current = self.raw_material_demand[industry_code].get("current", 0.0)
            self.raw_material_demand[industry_code]["previous"] = current
            self.raw_material_demand[industry_code]["current"] = 0.0
        self.logger.debug(f"Raw material demand finalized for {len(self.raw_material_demand)} industries")

    def get_raw_material_stats(self) -> Dict[str, Dict[str, float]]:
        """
        获取所有原材料行业的需求统计

        Returns:
            {industry_code: {"current": float, "previous": float, "change_ratio": float}}
        """
        result = {}
        for industry_code, stats in self.raw_material_demand.items():
            current = stats.get("current", 0.0)
            previous = stats.get("previous", 0.0)
            change_ratio = current / previous if previous > 0 else 1.0
            result[industry_code] = {
                "current": current,
                "previous": previous,
                "change_ratio": change_ratio
            }
        return result

    def adjust_prices_by_supply_demand(
        self,
        manufacturer_code: str,
        base_adjustment: float = 0.015,
        max_adjustment: float = 0.04
    ) -> int:
        """
        根据供需比调整价格，并施加均值回归力
        
        价格调整逻辑：
        - 供需比 > 1（供过于求）：降价
        - 供需比 < 1（供不应求）：涨价
        - 供需比 = 1.0（均衡/无数据）：仅施加均值回归
        - 调整幅度 = base_adjustment * |ln(供需比)|，最大不超过 max_adjustment
        - 均值回归：每期强制向基准价回归，偏离越大回归越强
        
        Args:
            manufacturer_code: 制造商行业代码
            base_adjustment: 基础调整系数
            max_adjustment: 最大调整幅度
            
        Returns:
            更新的产品数量
        """
        ratio = self.get_supply_demand_ratio(manufacturer_code)
        products = self.products_by_industry.get(manufacturer_code, [])
        
        if not products:
            return 0
        
        import math
        
        # 计算供需调整乘数
        if ratio == 1.0:
            price_multiplier = 1.0  # 均衡/无数据，不调价（仅靠均值回归）
        elif ratio > 1:
            log_ratio = math.log(ratio)
            adjustment = min(base_adjustment * log_ratio, max_adjustment)
            price_multiplier = 1 - adjustment  # 供过于求，降价
        else:
            log_ratio = math.log(ratio)
            adjustment = min(base_adjustment * abs(log_ratio), max_adjustment)
            price_multiplier = 1 + adjustment  # 供不应求，涨价
        
        # 均值回归：每期强制向基准价回归
        # 基础回归力 6%/期，偏离超过 15% 时加速到最大 20%
        MEAN_REVERSION_BASE = 0.08
        MEAN_REVERSION_MAX = 0.30
        DEVIATION_THRESHOLD = 0.10
        
        updated_count = 0
        for product in products:
            new_mfg = product.manufacturer_price * price_multiplier
            new_ret = product.retail_price * price_multiplier
            
            base_mfg = product.base_manufacturer_price
            base_ret = product.base_retail_price
            
            # 计算偏离度
            deviation_mfg = abs(new_mfg - base_mfg) / base_mfg if base_mfg > 0 else 0
            deviation_ret = abs(new_ret - base_ret) / base_ret if base_ret > 0 else 0
            
            # 动态回归系数：偏离越大回归越强
            reversion_mfg = min(MEAN_REVERSION_MAX, 
                               MEAN_REVERSION_BASE + max(0, deviation_mfg - DEVIATION_THRESHOLD) * 0.3)
            reversion_ret = min(MEAN_REVERSION_MAX,
                               MEAN_REVERSION_BASE + max(0, deviation_ret - DEVIATION_THRESHOLD) * 0.3)
            
            # 应用均值回归
            new_mfg = new_mfg * (1 - reversion_mfg) + base_mfg * reversion_mfg
            new_ret = new_ret * (1 - reversion_ret) + base_ret * reversion_ret
            
            product.manufacturer_price = max(0.01, new_mfg)
            product.retail_price = max(0.01, new_ret)
            updated_count += 1
        
        self.logger.info(
            f"Adjusted prices for {manufacturer_code}: ratio={ratio:.2f}, "
            f"multiplier={price_multiplier:.3f}, products={updated_count}"
        )
        return updated_count

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
            # 如果启用了活跃过滤，只返回活跃的SKU
            if self._require_active_filter and self._active_sku_set:
                candidates = [p for p in candidates if p.product_id in self._active_sku_set]
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
        collection_name = self._collection_name
        
        # 根据 Qdrant 模式选择过滤策略：
        # - Cloud/Docker 模式：使用 Qdrant 原生 filter（有索引优化，性能好）
        # - Local 模式：使用本地 Python filter（本地模式 filter 性能极差）
        use_qdrant_filter = self._use_qdrant_filter and self._require_active_filter
        use_local_filter = (not self._use_qdrant_filter) and self._require_active_filter and self._active_sku_set
        
        # 设置搜索参数
        if use_local_filter:
            # 本地过滤模式：需要搜索更多候选
            search_limit = max(top_k * 20, 200)
            max_fetch = max(top_k * 50, 500)
        else:
            # Qdrant 过滤或无过滤：正常搜索量
            search_limit = max(top_k * 3, 50)
            max_fetch = max(top_k * 10, search_limit)
        
        try:
            query_embedding = embedding(query)

            results: List[Product] = []
            seen_ids = set()
            offset = 0
            products_by_id = self.products_by_id
            active_sku_set = self._active_sku_set if use_local_filter else None
            
            # 构建 Qdrant filter（仅 Cloud/Docker 模式）
            search_filter = None
            if use_qdrant_filter:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="is_active",
                            match=MatchValue(value=True)
                        )
                    ]
                )
            
            # 重试配置（针对云端网络问题）
            max_retries = 3 if self._qdrant_mode == "cloud" else 1
            retry_delay = 1.0  # 秒

            while len(results) < top_k and offset < max_fetch:
                # 带重试的 Qdrant 查询（通过信号量限制并发）
                hits_resp = None
                last_error = None
                
                for attempt in range(max_retries):
                    try:
                        # 信号量控制：限制同时访问 Qdrant 的线程数
                        self._qdrant_semaphore.acquire()
                        try:
                            hits_resp = self.client.query_points(
                                collection_name=collection_name,
                                query=query_embedding,
                                query_filter=search_filter,  # Cloud/Docker 用 Qdrant filter，Local 用 None
                                limit=search_limit,
                                offset=offset,
                                with_payload=True,
                                with_vectors=False,
                            )
                        finally:
                            self._qdrant_semaphore.release()
                        break  # 成功，退出重试循环
                    except Exception as e:
                        last_error = e
                        if attempt < max_retries - 1:
                            import time
                            self.logger.warning(
                                f"Qdrant query attempt {attempt + 1}/{max_retries} failed: {e}, retrying..."
                            )
                            time.sleep(retry_delay * (attempt + 1))  # 指数退避
                        else:
                            self.logger.error(f"Qdrant query failed after {max_retries} attempts: {e}")
                            raise last_error
                
                if hits_resp is None:
                    break

                if hasattr(hits_resp, "points"):
                    hits_list = list(getattr(hits_resp, "points") or [])
                else:
                    hits_list = list(hits_resp or [])

                if not hits_list:
                    break

                offset += len(hits_list)

                for hit in hits_list:
                    payload = hit.payload or {}
                    product_id = payload.get("product_id")
                    if not product_id:
                        hit_id = getattr(hit, "id", None)
                        if hit_id is not None:
                            product_id = str(hit_id)

                    if not product_id or product_id in seen_ids:
                        continue
                    
                    # 本地活跃过滤（仅 Local 模式使用）
                    if active_sku_set and product_id not in active_sku_set:
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

    # ============ Active SKU Management (Qdrant Payload Filter) ============
    
    def set_active_filter_mode(self, enabled: bool):
        """
        设置是否在搜索时强制过滤活跃SKU
        
        Args:
            enabled: True表示只搜索is_active=true的SKU
        """
        self._require_active_filter = enabled
        self.logger.info(f"Active filter mode set to: {enabled}")
    
    def activate_skus(self, sku_ids: List[str]) -> int:
        """
        批量激活SKU（在Qdrant中设置is_active=true）
        
        Args:
            sku_ids: 要激活的SKU ID列表
            
        Returns:
            成功激活的SKU数量
        """
        if not sku_ids:
            return 0
        
        try:
            # 更新Qdrant payload
            self.client.set_payload(
                collection_name=self._collection_name,
                payload={"is_active": True},
                points=sku_ids,  # Qdrant支持直接传string ID列表
            )
            
            # 同步更新本地追踪集合
            self._active_sku_set.update(sku_ids)
            
            self.logger.info(f"Activated {len(sku_ids)} SKUs in Qdrant")
            return len(sku_ids)
        except Exception as e:
            self.logger.error(f"Failed to activate SKUs: {e}")
            return 0
    
    def deactivate_all_skus(self) -> bool:
        """
        重置所有SKU为非活跃状态（is_active=false）
        
        Returns:
            是否成功
        """
        try:
            # 使用scroll遍历所有点并更新
            # 或者使用filter更新所有is_active=true的点
            self.client.set_payload(
                collection_name=self._collection_name,
                payload={"is_active": False},
                points=Filter(
                    must=[
                        FieldCondition(
                            key="is_active",
                            match=MatchValue(value=True)
                        )
                    ]
                ),
            )
            
            # 清空本地追踪
            self._active_sku_set.clear()
            
            self.logger.info("Deactivated all SKUs in Qdrant")
            return True
        except Exception as e:
            self.logger.error(f"Failed to deactivate all SKUs: {e}")
            return False
    
    def get_active_sku_count(self) -> int:
        """获取当前活跃SKU数量"""
        return len(self._active_sku_set)
    
    def get_active_sku_ids(self) -> Set[str]:
        """获取当前活跃SKU ID集合的副本"""
        return self._active_sku_set.copy()
    
    def is_sku_active(self, sku_id: str) -> bool:
        """检查某个SKU是否活跃"""
        return sku_id in self._active_sku_set

    # =========================================================================
    # Checkpoint Support (用于断点续跑)
    # =========================================================================
    def get_all_products_snapshot(self) -> List[Dict[str, Any]]:
        """
        获取所有产品的快照数据（用于 checkpoint）
        
        Returns:
            产品数据列表
        """
        products_snapshot = []
        for product in self.products:
            products_snapshot.append({
                "product_id": product.product_id,
                "name": product.name,
                "manufacturer_price": float(product.manufacturer_price or 0.0),
                "retail_price": float(product.retail_price or 0.0),
                "available_stock": int(product.available_stock or 0),
                "manufacturer_code": product.manufacturer_code,
                "retailer_code": getattr(product, "retailer_code", None),
                "is_active": getattr(product, "is_active", True),
            })
        return products_snapshot
    
    def get_market_state_snapshot(self) -> Dict[str, Any]:
        """
        获取市场状态快照（用于 checkpoint）
        
        Returns:
            市场状态字典，包含:
            - require_active_filter: 是否启用活跃SKU过滤
            - active_sku_count: 活跃SKU数量
            - raw_material_demand: 原材料需求历史（用于价格调整）
            - industry_supply_demand: 行业供需数据
        """
        return {
            "require_active_filter": self._require_active_filter,
            "active_sku_count": len(self._active_sku_set),
            # 原材料需求历史（关键：用于跨月的价格调整计算）
            "raw_material_demand": dict(self.raw_material_demand),
            # 行业供需数据（通常在月初被重置，但checkpoint可能在月中保存）
            "industry_supply_demand": dict(self.industry_supply_demand),
        }
    
    def restore_market_state(self, state_data: Dict[str, Any]) -> None:
        """
        从 checkpoint 恢复市场状态设置
        
        Args:
            state_data: 市场状态数据
        """
        if state_data:
            self._require_active_filter = bool(state_data.get("require_active_filter", False))
            
            # 恢复原材料需求历史（关键：用于跨月的价格调整计算）
            raw_material_demand = state_data.get("raw_material_demand", {})
            if raw_material_demand:
                self.raw_material_demand = {
                    k: {"current": v.get("current", 0.0), "previous": v.get("previous", 0.0)}
                    for k, v in raw_material_demand.items()
                }
                self.logger.info(f"Restored raw_material_demand for {len(self.raw_material_demand)} industries")
            
            # 恢复行业供需数据
            industry_supply_demand = state_data.get("industry_supply_demand", {})
            if industry_supply_demand:
                self.industry_supply_demand = {
                    k: {"demand": v.get("demand", 0.0), "supply": v.get("supply", 0.0)}
                    for k, v in industry_supply_demand.items()
                }
                self.logger.info(f"Restored industry_supply_demand for {len(self.industry_supply_demand)} industries")
            
            self.logger.info(f"Restored market state: require_active_filter={self._require_active_filter}")
    
    def restore_products_snapshot(self, products_data: List[Dict[str, Any]]) -> int:
        """
        从 checkpoint 恢复产品库存和价格
        
        Args:
            products_data: 产品数据列表
            
        Returns:
            恢复的产品数量
        """
        restored = 0
        product_data_by_id = {d["product_id"]: d for d in products_data}
        
        # 重建 _active_sku_set
        self._active_sku_set.clear()
        
        for product in self.products:
            if product.product_id in product_data_by_id:
                data = product_data_by_id[product.product_id]
                # 恢复价格：优先使用 retail_price，兼容旧版 price 字段
                product.retail_price = float(data.get("retail_price") or data.get("price") or product.retail_price or 0.0)
                # 恢复库存：优先使用 available_stock，兼容旧版 stock 字段
                product.available_stock = float(data.get("available_stock") or data.get("stock") or product.available_stock or 0.0)
                restored += 1
                
                # 重建活跃SKU集合（Product 没有 is_active 字段，用 checkpoint 数据判断）
                is_active = bool(data.get("is_active", True))
                if is_active:
                    self._active_sku_set.add(product.product_id)
        
        self.logger.info(f"Restored {restored} product states from checkpoint, {len(self._active_sku_set)} active SKUs")
        return restored


if __name__ == "__main__":
    pro_m = ProductMarket.remote()
    pro_m.initialize_products.remote()
    print(ray.get(pro_m.get_market_stats.remote()))
