import csv
import json
import os
from pathlib import Path
from uuid import uuid4
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from agenteconomy.data.industry_cate_map import industry_cate_map
from agenteconomy.agent.firm import ManufactureFirm, RetailFirm, ServiceFirm, Firm
from agenteconomy.agent.household import Household

if TYPE_CHECKING:
    from agenteconomy.center.Ecocenter import EconomicCenter
    from agenteconomy.center.LaborMarket import LaborMarket
    from agenteconomy.center.ProductMarket import ProductMarket
    from agenteconomy.market.AbstractResourceMarket import AbstractResourceMarket

def create_households(
    *,
    data_dir: Optional[str] = None,
    persona_mapping_csv: str = "J357328_merged_household_persona_mapping.csv",
    codebook_json: str = "codebook.json",
    personas_json: str = "personas_final.json",
    census2010_to_soc2010_csv: str = "census2010_to_soc2010_exploded.csv",
    household_id_prefix: str = "household_",
    limit: Optional[int] = None,
    household_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Household]:
    households = load_all_households(
        data_dir=data_dir,
        persona_mapping_csv=persona_mapping_csv,
        codebook_json=codebook_json,
        personas_json=personas_json,
        census2010_to_soc2010_csv=census2010_to_soc2010_csv,
        household_id_prefix=household_id_prefix,
        limit=limit,
        household_kwargs=household_kwargs,
    )
    return list(households.values())


def load_household_rows(csv_path: str) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """
    Load persona-mapped household CSV.
    Returns:
      - by_household_idx: household_idx(int) -> row(dict[str, str])
      - by_fid: fid(int) -> row(dict[str, str])
    """
    by_idx: Dict[int, Dict[str, Any]] = {}
    by_fid: Dict[int, Dict[str, Any]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hh_idx_raw = row.get("household_idx")
            fid_raw = row.get("fid")
            if hh_idx_raw is not None and str(hh_idx_raw).strip() != "":
                by_idx[int(float(hh_idx_raw))] = row
            if fid_raw is not None and str(fid_raw).strip() != "":
                by_fid[int(float(fid_raw))] = row
    return by_idx, by_fid


def load_codebook_by_var(codebook_json_path: str) -> Dict[str, Dict[str, Any]]:
    raw = json.loads(Path(codebook_json_path).read_text(encoding="utf-8"))
    by_var: Dict[str, Dict[str, Any]] = {}
    for v in (raw or {}).get("variables", []) or []:
        var = v.get("var")
        if var:
            by_var[str(var)] = v
    return by_var


def load_personas_by_name(personas_json_path: str) -> Dict[str, Dict[str, Any]]:
    raw = json.loads(Path(personas_json_path).read_text(encoding="utf-8"))
    by_name: Dict[str, Dict[str, Any]] = {}
    for rec in raw or []:
        name = rec.get("persona_name")
        if name:
            by_name[str(name)] = rec
    return by_name


def load_census2010_to_soc2010(mapping_csv_path: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Load mapping from 2010 Census occupation code (4-digit string) -> list[(SOC2010, occupation_title)].
    """
    m: Dict[str, List[Tuple[str, str]]] = {}
    with open(mapping_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            census_code = str(row.get("census_code") or "").strip()
            soc = str(row.get("soc2010_code") or "").strip()
            title = str(row.get("occupation_title") or "").strip()
            if not census_code or not soc:
                continue
            m.setdefault(census_code, []).append((soc, title))
    return m


def build_preloaded_bundle(
    *,
    data_dir: str,
    persona_mapping_csv: str,
    codebook_json: str,
    personas_json: str,
    census2010_to_soc2010_csv: str,
) -> Dict[str, Any]:
    """
    Read all required files once and return a dict bundle that can be passed into Household(preloaded_data=...).
    """
    d = Path(data_dir)
    rows_by_idx, rows_by_fid = load_household_rows(str(d / persona_mapping_csv))
    return {
        "household_row_by_household_idx": rows_by_idx,
        "household_row_by_fid": rows_by_fid,
        "codebook_by_var": load_codebook_by_var(str(d / codebook_json)),
        "persona_by_name": load_personas_by_name(str(d / personas_json)),
        "census2010_to_soc2010": load_census2010_to_soc2010(str(d / census2010_to_soc2010_csv)),
    }


def load_all_households(
    *,
    data_dir: Optional[str] = None,
    persona_mapping_csv: str = "J357328_merged_household_persona_mapping.csv",
    codebook_json: str = "codebook.json",
    personas_json: str = "personas_final.json",
    census2010_to_soc2010_csv: str = "census2010_to_soc2010_exploded.csv",
    household_id_prefix: str = "household_",
    limit: Optional[int] = None,
    household_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Household]:
    """
    Bulk loader:
    - reads files once (outside Household)
    - iterates all households
    - instantiates one Household per row

    Returns: household_id -> Household instance
    """
    base_dir = data_dir or os.getenv(
        "AGENTECO_DATA_DIR",
        str(Path(__file__).resolve().parents[1] / "data" / "household"),
    )
    bundle = build_preloaded_bundle(
        data_dir=base_dir,
        persona_mapping_csv=persona_mapping_csv,
        codebook_json=codebook_json,
        personas_json=personas_json,
        census2010_to_soc2010_csv=census2010_to_soc2010_csv,
    )
    by_idx = bundle["household_row_by_household_idx"]
    out: Dict[str, Household] = {}
    kwargs = dict(household_kwargs or {})
    n = 0
    for hh_idx in sorted(by_idx.keys()):
        hid = f"{household_id_prefix}{hh_idx}"
        out[hid] = Household(
            household_id=hid,
            name=hid,
            description="",
            owner="",
            data_dir=base_dir,
            load_profile=True,
            preloaded_data=bundle,
            **kwargs,
        )
        n += 1
        if limit is not None and n >= int(limit):
            break
    return out

def create_firms(
    economic_center: Optional['EconomicCenter'] = None,
    labor_market: Optional['LaborMarket'] = None,
    product_market: Optional['ProductMarket'] = None,
    abstract_resource_market: Optional['AbstractResourceMarket'] = None,
) -> List[Firm]:
    """
    根据 industry_cate_map 创建所有行业的企业
    
    Args:
        economic_center: 经济中心
        labor_market: 劳动力市场
        product_market: 商品市场
        abstract_resource_market: 抽象资源市场（制造商生产时需要）
    
    Returns:
        List[Firm]: 包含制造商、零售商、服务商的企业列表
    """
    firms = []
    
    # Category 1: 制造商
    cat1 = industry_cate_map.get("category_1_manufacturers", {})
    for industry_code, industry_name in cat1.get("industries", {}).items():
        firm = ManufactureFirm(
            firm_id=f"mfg_{industry_code}_{uuid4().hex[:8]}",
            name=industry_name,
            industry=industry_code,
            industry_type="category_1_manufacturers",
            economic_center=economic_center,
            labor_market=labor_market,
            product_market=product_market,
            abstract_resource_market=abstract_resource_market,
        )
        firms.append(firm)
    
    # Category 2: 零售商
    cat2 = industry_cate_map.get("category_2_retailers", {})
    for industry_code, industry_name in cat2.get("industries", {}).items():
        firm = RetailFirm(
            firm_id=f"ret_{industry_code}_{uuid4().hex[:8]}",
            name=industry_name,
            industry=industry_code,
            industry_type="category_2_retailers",
            economic_center=economic_center,
            labor_market=labor_market,
            product_market=product_market,
            abstract_resource_market=abstract_resource_market,
        )
        firms.append(firm)
    
    # Category 3: 服务商（有 subgroups）
    # 注意：government_sectors 不创建为独立企业，其费用由 Government Agent 收取
    cat3 = industry_cate_map.get("category_3_cost_drivers", {})
    GOVERNMENT_SUBGROUP = "government_sectors"  # 政府行业子组，跳过不创建企业
    
    for subgroup_name, subgroup_info in cat3.get("subgroups", {}).items():
        # 跳过政府行业，这些由 Government Agent 处理
        if subgroup_name == GOVERNMENT_SUBGROUP:
            continue
            
        for industry_code, industry_name in subgroup_info.get("industries", {}).items():
            firm_id = f"svc_{industry_code}_{uuid4().hex[:8]}"
            firm = ServiceFirm(
                firm_id=firm_id,
                name=industry_name,
                industry=industry_code,
                industry_type=f"category_3_{subgroup_name}",
                economic_center=economic_center,
                labor_market=labor_market,
                product_market=product_market,
                abstract_resource_market=abstract_resource_market,
            )
            firms.append(firm)
            
            # 注册到 AbstractResourceMarket，建立 industry_code → firm_id 映射
            # 这样当有人采购该行业资源时，资金会流向真实的 ServiceFirm
            if abstract_resource_market is not None:
                abstract_resource_market.register_firm(industry_code, firm_id)
    
    return firms


# 政府行业代码集合（用于其他模块判断）
GOVERNMENT_INDUSTRY_CODES = frozenset(
    industry_cate_map.get("category_3_cost_drivers", {})
    .get("subgroups", {})
    .get("government_sectors", {})
    .get("industries", {})
    .keys()
)


def is_government_industry(industry_code: str) -> bool:
    """
    判断行业代码是否为政府行业
    
    Args:
        industry_code: 行业代码
        
    Returns:
        True if government industry, False otherwise
    """
    return industry_code in GOVERNMENT_INDUSTRY_CODES


def create_firms_by_category(
    category: str,
    economic_center: Optional['EconomicCenter'] = None,
    labor_market: Optional['LaborMarket'] = None,
    product_market: Optional['ProductMarket'] = None,
    abstract_resource_market: Optional['AbstractResourceMarket'] = None,
) -> List[Firm]:
    """
    创建指定类别的企业
    
    Args:
        category: "manufacturers", "retailers", or "services"
        economic_center: 经济中心
        labor_market: 劳动力市场
        product_market: 商品市场
        abstract_resource_market: 抽象资源市场
    
    Returns:
        List[Firm]: 指定类别的企业列表
    """
    firms = []
    
    if category == "manufacturers":
        cat = industry_cate_map.get("category_1_manufacturers", {})
        for industry_code, industry_name in cat.get("industries", {}).items():
            firm = ManufactureFirm(
                firm_id=f"mfg_{industry_code}_{uuid4().hex[:8]}",
                name=industry_name,
                industry=industry_code,
                industry_type="category_1_manufacturers",
                economic_center=economic_center,
                labor_market=labor_market,
                product_market=product_market,
                abstract_resource_market=abstract_resource_market,
            )
            firms.append(firm)
            
    elif category == "retailers":
        cat = industry_cate_map.get("category_2_retailers", {})
        for industry_code, industry_name in cat.get("industries", {}).items():
            firm = RetailFirm(
                firm_id=f"ret_{industry_code}_{uuid4().hex[:8]}",
                name=industry_name,
                industry=industry_code,
                industry_type="category_2_retailers",
                economic_center=economic_center,
                labor_market=labor_market,
                product_market=product_market,
                abstract_resource_market=abstract_resource_market,
            )
            firms.append(firm)
            
    elif category == "services":
        cat = industry_cate_map.get("category_3_cost_drivers", {})
        for subgroup_name, subgroup_info in cat.get("subgroups", {}).items():
            for industry_code, industry_name in subgroup_info.get("industries", {}).items():
                firm = ServiceFirm(
                    firm_id=f"svc_{industry_code}_{uuid4().hex[:8]}",
                    name=industry_name,
                    industry=industry_code,
                    industry_type=f"category_3_{subgroup_name}",
                    economic_center=economic_center,
                    labor_market=labor_market,
                    product_market=product_market,
                    abstract_resource_market=abstract_resource_market,
                )
                firms.append(firm)
    
    return firms


if __name__ == "__main__":
    # 测试
    all_firms = create_firms()
    print(f"Total firms: {len(all_firms)}")
    
    # 按类型统计
    mfg = [f for f in all_firms if isinstance(f, ManufactureFirm)]
    ret = [f for f in all_firms if isinstance(f, RetailFirm)]
    svc = [f for f in all_firms if isinstance(f, ServiceFirm)]
    
    print(f"Manufacturers: {len(mfg)}")
    print(f"Retailers: {len(ret)}")
    print(f"Services: {len(svc)}")
    
    # 打印几个示例
    print("\nSample firms:")
    for f in all_firms[:3]:
        print(f"  {f.firm_id}: {f.name} ({f.industry_type})")
