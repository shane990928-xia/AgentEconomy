import csv
import json
import os
from pathlib import Path
from uuid import uuid4
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

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
    economic_center=None,
    labor_market=None,
    product_market=None,
    total_hours: float = 160.0,
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
    out = list(households.values())

    # Household dollar-scale: shrink PSID income/wealth/expenditure so the household
    # sector is consistent with the (BLS-wage) firm scale — i.e. household income is
    # comparable to wage-earning capacity, so employment drives income. Applied before
    # initialize_in_system so the scaled ER85692 seeds the ledger cash.
    hh_scale = float(os.getenv("AGENTECO_HOUSEHOLD_SCALE", "1.0") or 1.0)
    if hh_scale != 1.0:
        _dollar_fields = (
            "ER85629", "ER85692", "ER85701", "ER85747", "ER85768",
            "expenditure_insurance", "expenditure_retail_merchandise",
            "expenditure_transportation", "expenditure_utilities",
        )
        for household in out:
            cv = getattr(household, "csv_values", None)
            if not isinstance(cv, dict):
                continue
            for f in _dollar_fields:
                v = cv.get(f)
                if v is None:
                    continue
                try:
                    cv[f] = float(v) * hh_scale
                except (TypeError, ValueError):
                    continue
            for f in ("ER85629", "ER85692"):
                if hasattr(household, f):
                    try:
                        setattr(household, f, float(cv.get(f) or 0.0))
                    except (TypeError, ValueError):
                        pass

    if economic_center is not None or labor_market is not None or product_market is not None:
        for household in out:
            household.initialize_in_system(
                economic_center=economic_center,
                labor_market=labor_market,
                product_market=product_market,
                total_hours=total_hours,
            )
    return out


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
    by_fid = bundle["household_row_by_fid"]

    # Filter negative net wealth households and cap extreme wealth at 90th percentile.
    wealth_rows: List[Tuple[int, Dict[str, Any], float]] = []
    for hh_idx, row in (by_idx or {}).items():
        raw = row.get("ER85692")
        try:
            wealth = float(raw)
        except Exception:
            wealth = 0.0
        if wealth < 0.0:
            continue
        wealth_rows.append((hh_idx, row, wealth))

    wealth_values = [w for _, _, w in wealth_rows]
    p90 = None
    if wealth_values:
        wealth_values.sort()
        p90 = wealth_values[int((len(wealth_values) - 1) * 0.9)]

    filtered_by_idx: Dict[int, Dict[str, Any]] = {}
    kept_row_ids = set()
    for hh_idx, row, wealth in wealth_rows:
        if p90 is not None and wealth > p90:
            row["ER85692"] = str(p90)
        filtered_by_idx[hh_idx] = row
        kept_row_ids.add(id(row))

    filtered_by_fid: Dict[int, Dict[str, Any]] = {}
    for fid, row in (by_fid or {}).items():
        if id(row) in kept_row_ids:
            filtered_by_fid[fid] = row

    bundle["household_row_by_household_idx"] = filtered_by_idx
    bundle["household_row_by_fid"] = filtered_by_fid
    by_idx = filtered_by_idx
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
    limit: Optional[int] = None,
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
    limit_count = None if limit is None else max(0, int(limit))

    def _take_items(items: Sequence[Tuple[str, str]], count: Optional[int]) -> List[Tuple[str, str]]:
        if count is None:
            return list(items)
        return list(items)[:max(0, int(count))]

    cat1_items = list(industry_cate_map.get("category_1_manufacturers", {}).get("industries", {}).items())
    cat2_items = list(industry_cate_map.get("category_2_retailers", {}).get("industries", {}).items())
    cat3_items: List[Tuple[str, str, str]] = []
    cat3 = industry_cate_map.get("category_3_cost_drivers", {})
    GOVERNMENT_SUBGROUP = "government_sectors"  # 政府行业子组，跳过不创建企业
    for subgroup_name, subgroup_info in cat3.get("subgroups", {}).items():
        if subgroup_name == GOVERNMENT_SUBGROUP:
            continue
        for industry_code, industry_name in subgroup_info.get("industries", {}).items():
            cat3_items.append((subgroup_name, industry_code, industry_name))

    if limit_count is not None:
        if limit_count <= 0:
            return []
        full_count = len(cat1_items) + len(cat2_items) + len(cat3_items)
        if limit_count < full_count:
            retail_count = min(len(cat2_items), limit_count)
            remaining = max(0, limit_count - retail_count)
            service_count = min(len(cat3_items), max(1, round(limit_count * 0.20))) if remaining >= 2 else 0
            service_count = min(service_count, remaining)
            mfg_count = max(0, remaining - service_count)
            cat1_items = _take_items(cat1_items, mfg_count)
            cat2_items = _take_items(cat2_items, retail_count)
            cat3_items = list(cat3_items)[:service_count]
    
    # Category 1: 制造商
    for industry_code, industry_name in cat1_items:
        firm = ManufactureFirm(
            firm_id=f"mfg_{industry_code}",
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
    for industry_code, industry_name in cat2_items:
        firm = RetailFirm(
            firm_id=f"ret_{industry_code}",
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
    for subgroup_name, industry_code, industry_name in cat3_items:
        firm_id = f"svc_{industry_code}"
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
