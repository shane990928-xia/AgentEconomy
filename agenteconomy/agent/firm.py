from typing import Optional, List, TYPE_CHECKING, Dict, Any, Iterable, Tuple
from functools import lru_cache
from pathlib import Path
import json
import ast
import csv
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
import ray

from agenteconomy.center.Model import LaborHour, Job
from agenteconomy.center.LaborMarket import LaborMarket
from agenteconomy.center.ProductMarket import ProductMarket
from agenteconomy.market.AbstractResourceMarket import AbstractResourceMarket
from agenteconomy.market.IntermediateGoodsProcurement import IntermediateGoodsProcurement
from agenteconomy.utils.logger import get_logger
from agenteconomy.utils.load_io_table import get_suppliers_for_industry, get_cost_structure
from agenteconomy.data.industry_cate_map import industry_cate_map
from agenteconomy.llm.llm import *
from agenteconomy.llm.prompt_template import build_firm_post_job_prompt

# Avoid circular import by using TYPE_CHECKING
if TYPE_CHECKING:
    from agenteconomy.center.Ecocenter import EconomicCenter

logger = get_logger(name="Firm")

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_OCCUPATION_XLSX = _DATA_DIR / "occupation.xlsx"
_JOB_SKILLS_CSV = _DATA_DIR / "jobs_with_skills_abilities_IM_merged.csv"
_IO_TABLE_CSV = _DATA_DIR / "Direct Total Requirements, After Redefinitions - Summary.csv"

_TEXT_STOPWORDS = {
    "and", "or", "of", "the", "for", "in", "to", "with",
    "industry", "industries", "services", "service", "products", "product",
    "manufacturing", "manufacture", "except", "other", "miscellaneous",
    "activities", "related",
}


def _normalize_tokens(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [t for t in text.split() if t and t not in _TEXT_STOPWORDS]
    return tokens


def _token_similarity(tokens_a: Iterable[str], tokens_b: Iterable[str]) -> float:
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    return (2.0 * inter) / (len(set_a) + len(set_b))


def _col_to_index(col: str) -> int:
    idx = 0
    for ch in col:
        if "A" <= ch <= "Z":
            idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _cell_value(cell: ET.Element, shared: List[str], ns: Dict[str, str]) -> str:
    cell_type = cell.attrib.get("t")
    v = cell.find("ns:v", ns)
    if cell_type == "s":
        if v is None or v.text is None:
            return ""
        idx = int(v.text)
        return shared[idx] if 0 <= idx < len(shared) else ""
    if cell_type == "inlineStr":
        t = cell.find(".//ns:t", ns)
        return t.text if t is not None else ""
    return v.text if v is not None else ""


def _read_xlsx_sheet(path: Path, sheet_name: str) -> List[List[str]]:
    if not path.exists():
        return []
    with zipfile.ZipFile(path) as zf:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            with zf.open("xl/sharedStrings.xml") as f:
                tree = ET.parse(f)
            root = tree.getroot()
            ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for si in root.findall(".//ns:si", ns):
                texts = []
                for t in si.findall(".//ns:t", ns):
                    texts.append(t.text or "")
                shared.append("".join(texts))

        with zf.open("xl/workbook.xml") as f:
            tree = ET.parse(f)
        root = tree.getroot()
        ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rels: Dict[str, str] = {}
        with zf.open("xl/_rels/workbook.xml.rels") as f:
            rel_tree = ET.parse(f)
        rel_root = rel_tree.getroot()
        rel_ns = {"ns": "http://schemas.openxmlformats.org/package/2006/relationships"}
        for rel in rel_root.findall("ns:Relationship", rel_ns):
            rels[rel.attrib["Id"]] = rel.attrib["Target"]

        target_sheet = None
        for sheet in root.findall(".//ns:sheets/ns:sheet", ns):
            if sheet.attrib.get("name") == sheet_name:
                rid = sheet.attrib.get(
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
                )
                target_sheet = rels.get(rid)
                break
        if not target_sheet:
            return []

        sheet_path = "xl/" + target_sheet
        with zf.open(sheet_path) as f:
            tree = ET.parse(f)
        root = tree.getroot()
        ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

        rows: List[List[str]] = []
        for row in root.findall(".//ns:sheetData/ns:row", ns):
            row_values: Dict[int, str] = {}
            max_col = -1
            for cell in row.findall("ns:c", ns):
                ref = cell.attrib.get("r", "")
                col = "".join(ch for ch in ref if ch.isalpha())
                idx = _col_to_index(col) if col else len(row_values)
                row_values[idx] = _cell_value(cell, shared, ns).strip()
                if idx > max_col:
                    max_col = idx
            if max_col >= 0:
                values = [""] * (max_col + 1)
                for idx, val in row_values.items():
                    values[idx] = val
                rows.append(values)
        return rows


@lru_cache(maxsize=1)
def _load_io_industry_names() -> Dict[str, str]:
    if not _IO_TABLE_CSV.exists():
        return {}
    with _IO_TABLE_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        first_row = next(reader, [])
    if len(header) < 3 or len(first_row) < 3:
        return {}
    codes = header[2:]
    names = first_row[2:]
    return {code: name for code, name in zip(codes, names) if code and name}


@lru_cache(maxsize=1)
def _load_job_skill_data() -> Dict[str, Dict[str, Any]]:
    if not _JOB_SKILLS_CSV.exists():
        return {}
    data: Dict[str, Dict[str, Any]] = {}
    with _JOB_SKILLS_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            soc = (row.get("O*NET-SOC Code") or "").strip()
            if not soc or soc in data:
                continue
            try:
                wage = float(row.get("Average_Wage") or 0.0)
            except ValueError:
                wage = 0.0
            skills_raw = row.get("skills") or "{}"
            abilities_raw = row.get("abilities") or "{}"
            try:
                skills = ast.literal_eval(skills_raw) if skills_raw else {}
            except Exception:
                skills = {}
            try:
                abilities = ast.literal_eval(abilities_raw) if abilities_raw else {}
            except Exception:
                abilities = {}
            data[soc] = {
                "soc": soc,
                "title": (row.get("Title") or "").strip(),
                "description": (row.get("Description") or "").strip(),
                "wage": wage,
                "skills": skills if isinstance(skills, dict) else {},
                "abilities": abilities if isinstance(abilities, dict) else {},
            }
    return data


def _find_header_row(rows: List[List[str]], required: Iterable[str]) -> Tuple[int, Dict[str, int]]:
    required_lower = [r.lower() for r in required]
    for idx, row in enumerate(rows):
        lowered = [c.lower() for c in row if c]
        if all(r in lowered for r in required_lower):
            mapping = {}
            for col_idx, value in enumerate(row):
                if not value:
                    continue
                mapping[value.lower()] = col_idx
            return idx, mapping
    return -1, {}


@lru_cache(maxsize=1)
def _load_soc_distribution() -> Dict[str, float]:
    rows = _read_xlsx_sheet(_OCCUPATION_XLSX, "Table 1.2")
    if not rows:
        return {}
    header_idx, mapping = _find_header_row(
        rows,
        [
            "2024 national employment matrix code",
            "employment distribution, percent, 2024",
        ],
    )
    if header_idx < 0:
        return {}
    code_idx = mapping.get("2024 national employment matrix code")
    dist_idx = mapping.get("employment distribution, percent, 2024")
    occ_type_idx = mapping.get("occupation type")
    if code_idx is None or dist_idx is None:
        return {}

    distribution: Dict[str, float] = {}
    for row in rows[header_idx + 1 :]:
        if code_idx >= len(row):
            continue
        soc = row[code_idx].strip()
        if not soc or soc == "00-0000":
            continue
        if occ_type_idx is not None and occ_type_idx < len(row):
            occ_type = row[occ_type_idx].strip().lower()
            if occ_type and occ_type != "line item":
                continue
        if dist_idx >= len(row):
            continue
        try:
            dist = float(row[dist_idx])
        except ValueError:
            continue
        distribution[soc] = dist
    return distribution


@lru_cache(maxsize=1)
def _load_naics_to_soc() -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    rows = _read_xlsx_sheet(_OCCUPATION_XLSX, "Table 1.12")
    if not rows:
        return {}, {}
    header_idx, mapping = _find_header_row(
        rows,
        [
            "2024 national employment matrix occupation code",
            "2024 national employment matrix industry title",
            "2024 national employment matrix industry code",
        ],
    )
    if header_idx < 0:
        return {}, {}
    occ_idx = mapping.get("2024 national employment matrix occupation code")
    ind_title_idx = mapping.get("2024 national employment matrix industry title")
    ind_code_idx = mapping.get("2024 national employment matrix industry code")
    if occ_idx is None or ind_title_idx is None or ind_code_idx is None:
        return {}, {}

    naics_to_soc: Dict[str, List[str]] = defaultdict(list)
    naics_titles: Dict[str, str] = {}
    for row in rows[header_idx + 1 :]:
        if occ_idx >= len(row) or ind_code_idx >= len(row) or ind_title_idx >= len(row):
            continue
        soc = row[occ_idx].strip()
        ind_code = row[ind_code_idx].strip()
        ind_title = row[ind_title_idx].strip()
        if not soc or not ind_code:
            continue
        if ind_title.lower().startswith("total, all industries"):
            continue
        naics_to_soc[ind_code].append(soc)
        if ind_code not in naics_titles and ind_title:
            naics_titles[ind_code] = ind_title
    return naics_to_soc, naics_titles


def _match_naics_for_io(industry_code: str, industry_name: str) -> Tuple[Optional[str], Optional[str], float]:
    naics_to_soc, naics_titles = _load_naics_to_soc()
    if not naics_titles:
        return None, None, 0.0

    tokens_io = _normalize_tokens(industry_name or "")
    best_score = 0.0
    best_code = None
    best_title = None

    digits = re.match(r"^\\d+", str(industry_code or ""))
    prefix = digits.group(0) if digits else ""
    candidates = naics_titles.items()
    if prefix:
        candidates = [(c, t) for c, t in naics_titles.items() if str(c).startswith(prefix)]
        if not candidates:
            candidates = naics_titles.items()

    for code, title in candidates:
        score = _token_similarity(tokens_io, _normalize_tokens(title))
        if score > best_score:
            best_score = score
            best_code = code
            best_title = title

    if best_code and best_code in naics_to_soc:
        return best_code, best_title, best_score
    return None, None, best_score
@lru_cache(maxsize=1)
def _load_retail_supply_chain_map() -> Dict[str, Dict[str, Any]]:
    data_path = Path(__file__).resolve().parents[1] / "data" / "retailer_mfg.json"
    try:
        with data_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("retail_supply_chain_map", {})
    except FileNotFoundError:
        logger.warning("Retail supply chain map not found at %s", data_path)
        return {}
    except json.JSONDecodeError as exc:
        logger.warning("Retail supply chain map parse error: %s", exc)
        return {}

def _get_retail_supply_chain_entry(industry: Optional[str]) -> Dict[str, Any]:
    if not industry:
        return {}
    return _load_retail_supply_chain_map().get(industry, {})


class Firm:
    def __init__(self, 
                 firm_id: str, 
                 name: Optional[str] = None, 
                 description: Optional[str] = None, 
                 industry: Optional[str] = None,
                 industry_type: Optional[str] = None,
                 economic_center: Optional['EconomicCenter'] = None,
                 labor_market: Optional[LaborMarket] = None,
                 product_market: Optional[ProductMarket] = None,
                 abstract_resource_market: Optional[AbstractResourceMarket] = None,
                 is_agent: bool = False,
               ):
        # Firm info
        self.firm_id: str = firm_id
        self.name: str = name
        self.description: Optional[str] = description
        self.industry: Optional[str] = industry
        self.industry_type: Optional[str] = industry_type
        
        # Firm employees
        self.employee_count: int = 0 # Number of employees
        self.employee_list: List[LaborHour] = [] # List of employees

        # Firm financials
        self.capital_stock: float = 0.0 # Capital stock
        self.cash: float = 0.0 # Cash

        # Market index
        self.economic_center: Optional[EconomicCenter] = economic_center # Economic center
        self.labor_market: Optional[LaborMarket] = labor_market # Labor market
        self.product_market: Optional[ProductMarket] = product_market # Product market
        self.abstract_resource_market: Optional[AbstractResourceMarket] = abstract_resource_market # Abstract resource market

        # Is agent
        self.is_agent: bool = is_agent # Whether the firm is an agent
        
        # Production tracking
        self.current_period: int = 0  # Current simulation period
        self.production_history: List[Dict[str, Any]] = []  # Production history

    async def register(self):
        """Register the firm in the economic center"""
        await self.economic_center.register_firm.remote(self)

    def _call_labor_market(self, method_name: str, *args, **kwargs):
        if self.labor_market is None:
            return None
        method = getattr(self.labor_market, method_name, None)
        if method is None:
            return None
        if 'ActorHandle' in str(type(self.labor_market)):
            return ray.get(method.remote(*args, **kwargs))
        return method(*args, **kwargs)

    def _call_economic_center(self, method_name: str, *args, **kwargs):
        if self.economic_center is None:
            return None
        method = getattr(self.economic_center, method_name, None)
        if method is None:
            return None
        if 'ActorHandle' in str(type(self.economic_center)):
            return ray.get(method.remote(*args, **kwargs))
        return method(*args, **kwargs)

    def _compute_labor_budget(self, period: Optional[int] = None) -> float:
        base_value = 0.0
        current_period = int(period if period is not None else self.current_period or 0)
        if current_period > 0:
            stats = self._call_economic_center(
                "query_firm_monthly_financials",
                firm_id=self.firm_id,
                month=current_period - 1,
            )
            if isinstance(stats, dict):
                base_value = float(stats.get("monthly_income", 0.0) or 0.0)
        if base_value <= 0 and self.production_history:
            try:
                base_value = float(self.production_history[-1].get("production_value", 0.0) or 0.0)
            except Exception:
                base_value = 0.0
        if base_value <= 0:
            base_value = float(self.cash or 0.0)

        compensation_ratio = 0.2
        try:
            cost_structure = get_cost_structure(self.industry)
            if cost_structure:
                compensation_ratio = float(cost_structure.get("compensation", compensation_ratio) or compensation_ratio)
        except Exception:
            compensation_ratio = 0.2

        budget = base_value * compensation_ratio
        return max(0.0, budget)

    def _decide_job_postings_from_data(self, period: Optional[int] = None, max_job_types: int = 10) -> List[Job]:
        if not self.industry:
            return []

        job_data = _load_job_skill_data()
        if not job_data:
            return []

        io_names = _load_io_industry_names()
        io_name = io_names.get(self.industry, "") if io_names else ""

        naics_code, naics_title, score = _match_naics_for_io(self.industry, io_name)
        naics_to_soc, _ = _load_naics_to_soc()
        candidate_socs = []
        if naics_code and naics_code in naics_to_soc:
            candidate_socs = list(dict.fromkeys(naics_to_soc[naics_code]))

        if not candidate_socs:
            distribution = _load_soc_distribution()
            candidate_socs = [
                soc for soc, _ in sorted(
                    distribution.items(), key=lambda kv: kv[1], reverse=True
                )
            ][:max_job_types]

        candidate_socs = [soc for soc in candidate_socs if soc in job_data]
        if not candidate_socs:
            return []

        distribution = _load_soc_distribution()
        weights = {}
        total_weight = 0.0
        for soc in candidate_socs:
            w = float(distribution.get(soc, 1.0) or 1.0)
            weights[soc] = w
            total_weight += w
        if total_weight <= 0:
            total_weight = float(len(candidate_socs))

        labor_budget = self._compute_labor_budget(period)
        if labor_budget <= 0:
            return []

        hours_per_week = 40.0
        weeks_per_month = 4.0
        hours_per_period = hours_per_week * weeks_per_month

        jobs: List[Job] = []
        remaining_budget = labor_budget

        ranked_socs = sorted(candidate_socs, key=lambda s: weights.get(s, 0.0), reverse=True)
        for soc in ranked_socs[:max_job_types]:
            info = job_data.get(soc)
            if not info:
                continue
            hourly_wage = float(info.get("wage", 0.0) or 0.0)
            if hourly_wage <= 0:
                continue
            monthly_wage = hourly_wage * hours_per_period
            if monthly_wage <= 0:
                continue
            budget_share = labor_budget * (weights.get(soc, 1.0) / total_weight)
            positions = int(budget_share // monthly_wage)
            if positions <= 0:
                continue
            max_affordable = int(remaining_budget // monthly_wage)
            if max_affordable <= 0:
                continue
            positions = min(positions, max_affordable)
            remaining_budget -= positions * monthly_wage

            job = Job.create(
                soc=soc,
                title=info.get("title") or soc,
                wage_per_hour=hourly_wage,
                firm_id=self.firm_id,
                description=info.get("description"),
                hours_per_period=hours_per_period,
                required_skills=info.get("skills") or {},
                required_abilities=info.get("abilities") or {},
            )
            job.positions_available = positions
            jobs.append(job)

        if not jobs:
            # Fallback: try to hire 1 lowest-wage role
            cheapest_soc = None
            cheapest_monthly = 0.0
            for soc in ranked_socs:
                info = job_data.get(soc)
                if not info:
                    continue
                hourly_wage = float(info.get("wage", 0.0) or 0.0)
                if hourly_wage <= 0:
                    continue
                monthly_wage = hourly_wage * hours_per_period
                if monthly_wage <= 0:
                    continue
                if cheapest_soc is None or monthly_wage < cheapest_monthly:
                    cheapest_soc = soc
                    cheapest_monthly = monthly_wage
            if cheapest_soc and remaining_budget >= cheapest_monthly:
                info = job_data[cheapest_soc]
                job = Job.create(
                    soc=cheapest_soc,
                    title=info.get("title") or cheapest_soc,
                    wage_per_hour=float(info.get("wage", 0.0) or 0.0),
                    firm_id=self.firm_id,
                    description=info.get("description"),
                    hours_per_period=hours_per_period,
                    required_skills=info.get("skills") or {},
                    required_abilities=info.get("abilities") or {},
                )
                job.positions_available = 1
                jobs.append(job)

        if jobs and naics_title:
            logger.info(
                f"Firm {self.firm_id} matched IO '{io_name}' to NAICS '{naics_title}' (score={score:.2f})"
            )

        return jobs

    # Query info
    def query_info(self):
        """Query information about the firm"""
        return {
            "firm_id": self.firm_id,
            "name": self.name,
            "description": self.description,
            "industry": self.industry,
            "capital_stock": self.capital_stock,
            "cash": self.cash,
        }

    def query_employees(self):
        """Query the employees of the firm"""
        return self.employee_count, self.employee_list

    # Labor market operations
    async def post_jobs(self, period: Optional[int] = None):
        """Post jobs to the labor market"""
        jobs = self._decide_job_postings_from_data(period=period)
        if jobs:
            snapshot = self._call_labor_market("get_firm_job_snapshot", self.firm_id)
            if not isinstance(snapshot, dict):
                snapshot = {}
            to_post: List[Job] = []
            for job in jobs:
                desired = int(job.positions_available or 0)
                existing = int(snapshot.get(job.SOC, 0) or 0)
                delta = desired - existing
                if delta <= 0:
                    continue
                job.positions_available = delta
                to_post.append(job)
            if to_post:
                self._call_labor_market("apply_job_plan", self.firm_id, to_post)
            return to_post

        prompt = build_firm_post_job_prompt(self)
        response = await call_llm(prompt)
        return response

    def evaluate_candidates(self):
        """Evaluate candidates for the job"""
        pass

    def hire_employee(self):
        """Hire an employee"""
        pass

    def fire_employee(self):
        """Fire an employee"""
        pass
    
    def add_employee(self, employee: LaborHour):
        """Add an employee to the firm"""
        self.employee_list.append(employee)
        self.employee_count += 1

    def remove_employee(self, employee: LaborHour):
        """Remove an employee from the firm"""
        self.employee_list.remove(employee)
        self.employee_count -= 1

    # Product market operations
    def publish_product(self):
        """Publish a product to the product market"""
        pass

    def production_plan(self):
        """Generate a production plan"""
        pass


class ManufactureFirm(Firm):
    """
    制造业企业
    
    负责生产具体的SKU产品，消耗中间品和抽象资源
    
    注意：税和劳动报酬不从IO表计算，而是由企业实际运营决定
    """
    # 子类默认值
    DEFAULT_INDUSTRY_TYPE = "manufacture"
    DEFAULT_DESCRIPTION = 'A producer of physical goods that consumes third-category resources and outputs concrete SKU inventories, with full pricing authority.'
    
    def __init__(self, firm_id: str, **kwargs):
        kwargs.setdefault('industry_type', self.DEFAULT_INDUSTRY_TYPE)
        kwargs.setdefault('description', self.DEFAULT_DESCRIPTION)
        super().__init__(firm_id=firm_id, **kwargs) 

        # Production specific attributes
        self.production_costs: Dict[str, float] = {}  # Cost breakdown by input industry
        self.unit_costs: Dict[str, float] = {}  # Unit cost by SKU
        
        # Initialize procurement tool (lazy initialization)
        self._procurement = None
    
    @property
    def procurement(self) -> IntermediateGoodsProcurement:
        """Lazy initialization of procurement tool"""
        if self._procurement is None and self.product_market is not None:
            self._procurement = IntermediateGoodsProcurement(self.product_market)
        return self._procurement
    
    def calculate_production_value(self, production_plan: Dict[str, int], sku_base_prices: Dict[str, float]) -> float:
        """
        计算生产价值（用于IO表技术系数计算）
        
        Args:
            production_plan: {sku_id: quantity}
            sku_base_prices: {sku_id: base_manufacturer_price}
        
        Returns:
            总生产价值（美元）
        """
        total_value = 0.0
        for sku_id, quantity in production_plan.items():
            base_price = sku_base_prices.get(sku_id, 0)
            total_value += base_price * quantity
        
        return total_value
    
    def get_io_suppliers(self, threshold: float = 0.001) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取IO表供应商列表，并分类为中间品和抽象资源
        
        Args:
            threshold: IO系数阈值，忽略小于此值的供应商
        
        Returns:
            {
                'intermediate_goods': [...],  # Category 1 制造业
                'abstract_resources': [...],  # Category 3 抽象资源
            }
        """
        if not self.industry:
            logger.warning(f"Firm {self.firm_id}: No industry specified")
            return {'intermediate_goods': [], 'abstract_resources': []}
        
        # 获取所有供应商
        all_suppliers = get_suppliers_for_industry(self.industry, threshold=threshold)
        
        # 获取Category 3代码集合
        category_3_codes = set()
        for subgroup_info in industry_cate_map["category_3_cost_drivers"]["subgroups"].values():
            category_3_codes.update(subgroup_info["industries"].keys())
        
        # 分类
        intermediate_goods = []
        abstract_resources = []
        
        for supplier in all_suppliers:
            if supplier['supplier'] in category_3_codes:
                abstract_resources.append(supplier)
            else:
                intermediate_goods.append(supplier)
        
        logger.info(
            f"Firm {self.firm_id} IO suppliers: "
            f"{len(intermediate_goods)} intermediate goods, "
            f"{len(abstract_resources)} abstract resources"
        )
        
        return {
            'intermediate_goods': intermediate_goods,
            'abstract_resources': abstract_resources,
        }
    
    def procure_intermediate_goods(
        self,
        production_value: float,
        suppliers: List[Dict[str, Any]],
        period: int,
        strategy: str = "random"
    ) -> Dict[str, Any]:
        """
        采购中间品（从其他制造商购买具体SKU）
        
        Args:
            production_value: 生产价值
            suppliers: 中间品供应商列表
            period: 当前期数
            strategy: 采购策略 ("random", "cheapest", "balanced")
        
        Returns:
            {
                'total_cost': float,
                'by_industry': {industry_code: cost},
                'items': [PurchaseItem, ...]
            }
        """
        if self.procurement is None:
            logger.warning(f"Firm {self.firm_id}: No product market available for procurement")
            return {'total_cost': 0.0, 'by_industry': {}, 'items': []}
        
        result = self.procurement.procure_intermediate_goods(
            manufacturer_id=self.firm_id,
            production_value=production_value,
            io_suppliers=suppliers,
            period=period
        )

        # 更新成本记录
        self.production_costs.update(result['by_industry'])

        if self.economic_center is not None and result.get('total_cost', 0.0) > 0:
            items_payload = [
                {
                    "sku_id": item.sku_id,
                    "quantity": item.quantity,
                    "unit_price": item.unit_price,
                    "total_cost": item.total_cost,
                }
                for item in result.get('items', [])
            ]
            self._call_economic_center(
                "record_intermediate_goods_purchase",
                month=period,
                buyer_id=self.firm_id,
                total_cost=result['total_cost'],
                costs_by_industry=result.get('by_industry', {}),
                items=items_payload,
            )
        
        logger.info(
            f"Firm {self.firm_id} procured intermediate goods: "
            f"${result['total_cost']:.2f} from {len(result['by_industry'])} industries"
        )
        
        return result
    
    def procure_abstract_resources(
        self,
        production_value: float,
        io_suppliers: List[Dict[str, Any]],
        period: int
    ) -> Dict[str, float]:
        """
        采购抽象资源（电力、运输等）
        
        Args:
            production_value: 生产价值
            io_suppliers: IO表供应商列表 [{supplier: code, coefficient: value, name: str}]
            period: 当前期数
        
        Returns:
            成本明细 {industry_code: cost}
        """
        if self.abstract_resource_market is None:
            logger.warning(f"Firm {self.firm_id}: No abstract resource market available")
            return {}
        
        costs = {}
        
        for supplier in io_suppliers:
            supplier_code = supplier['supplier']
            
            try:
                # 检查是否为抽象资源
                self.abstract_resource_market.get_resource_info(supplier_code)
                
                # 计算物理需求
                physical_qty, unit = self.abstract_resource_market.calculate_physical_demand(
                    industry_code=supplier_code,
                    production_value=production_value,
                    io_coefficient=supplier['coefficient']
                )
                
                # 采购
                transaction = self.abstract_resource_market.purchase(
                    industry_code=supplier_code,
                    buyer_id=self.firm_id,
                    quantity=physical_qty,
                    period=period
                )

                if transaction:
                    costs[supplier_code] = transaction['total_cost']
                    if self.economic_center is not None:
                        self._call_economic_center(
                            "record_resource_purchase",
                            month=period,
                            buyer_id=self.firm_id,
                            industry_code=supplier_code,
                            quantity=physical_qty,
                            unit_price=transaction.get('unit_price', 0.0),
                            total_cost=transaction.get('total_cost', 0.0),
                            unit=transaction.get('unit'),
                            base_price=transaction.get('base_price'),
                        )

                    supplier_name = supplier.get('name', supplier_code)
                    logger.info(
                        f"Firm {self.firm_id} purchased {supplier_name}: "
                        f"{physical_qty:.2f}{unit} × ${transaction['unit_price']:.4f} = ${transaction['total_cost']:.2f}"
                    )
                
            except ValueError:
                # 不是抽象资源，跳过（可能是中间品）
                continue
            except Exception as e:
                logger.error(f"Firm {self.firm_id} failed to procure {supplier_code}: {e}")
                continue
        
        # 更新成本记录
        self.production_costs.update(costs)
        
        logger.info(
            f"Firm {self.firm_id} procured abstract resources: "
            f"${sum(costs.values()):.2f} from {len(costs)} resources"
        )
        
        return costs
    
    def calculate_labor_cost(self) -> float:
        """
        计算劳动成本（基于实际雇佣的员工）
        
        从 LaborMarket 获取本企业的工资支出总额
        
        Returns:
            劳动成本（月度工资总额）
        """
        if self.labor_market is None:
            logger.warning(f"Firm {self.firm_id}: No labor market available for labor cost calculation")
            return 0.0
        
        try:
            # 调用 LaborMarket 获取本企业的劳动成本
            if 'ActorHandle' in str(type(self.labor_market)):
                labor_cost = ray.get(self.labor_market.get_firm_labor_cost.remote(self.firm_id))
            else:
                labor_cost = self.labor_market.get_firm_labor_cost(self.firm_id)
            
            logger.info(f"Firm {self.firm_id} labor cost: ${labor_cost:.2f}")
            return labor_cost
            
        except Exception as e:
            logger.error(f"Firm {self.firm_id} failed to calculate labor cost: {e}")
            return 0.0
    
    def calculate_tax_cost(self, revenue: float = 0.0, tax_rate: float = 0.0) -> float:
        """
        计算税收成本
        
        Args:
            revenue: 收入
            tax_rate: 税率
        
        Returns:
            税收成本
        """
        # TODO: 实现基于税收系统的计算
        # 目前返回0，等待税收系统集成
        return 0.0
    
    def calculate_total_cost(
        self,
        intermediate_goods_cost: float,
        abstract_resources_cost: float,
        labor_cost: float = 0.0,
        tax_cost: float = 0.0
    ) -> Dict[str, float]:
        """
        计算总生产成本
        
        注意：税和劳动报酬不从IO表计算，而是由企业实际运营决定
        
        Args:
            intermediate_goods_cost: 中间品成本
            abstract_resources_cost: 抽象资源成本
            labor_cost: 劳动成本（由企业雇佣决定，默认0）
            tax_cost: 税收成本（由税收系统决定，默认0）
        
        Returns:
            {
                'intermediate_goods': float,
                'abstract_resources': float,
                'labor': float,
                'taxes': float,
                'total_cost': float
            }
        """
        # 总成本 = 中间品 + 抽象资源 + 劳动 + 税收
        total_cost = (
            intermediate_goods_cost +
            abstract_resources_cost +
            labor_cost +
            tax_cost
        )
        
        logger.info(
            f"Firm {self.firm_id} total cost: ${total_cost:.2f} "
            f"(intermediate: ${intermediate_goods_cost:.2f}, "
            f"abstract: ${abstract_resources_cost:.2f}, "
            f"labor: ${labor_cost:.2f}, "
            f"taxes: ${tax_cost:.2f})"
        )
        
        return {
            'intermediate_goods': intermediate_goods_cost,
            'abstract_resources': abstract_resources_cost,
            'labor': labor_cost,
            'taxes': tax_cost,
            'total_cost': total_cost
        }
    
    def produce(
        self,
        production_plan: Dict[str, int],
        sku_base_prices: Dict[str, float],
        period: int,
        update_inventory: bool = True,
        labor_cost: float = None,
        tax_cost: float = None
    ) -> Dict[str, Any]:
        """
        完整的生产流程
        
        注意：税和劳动报酬不从IO表计算，而是由企业实际运营决定
        
        Args:
            production_plan: {sku_id: quantity}
            sku_base_prices: {sku_id: base_manufacturer_price}
            period: 当前期数
            update_inventory: 是否更新库存
            labor_cost: 劳动成本（None则自动计算，默认0）
            tax_cost: 税收成本（None则自动计算，默认0）
        
        Returns:
            {
                'production_value': float,
                'total_cost': float,
                'unit_costs': {sku_id: unit_cost},
                'cost_breakdown': {...},
                'success': bool
            }
        """
        logger.info(f"Firm {self.firm_id} starting production for period {period}")
        
        try:
            # 1. 计算生产价值
            production_value = self.calculate_production_value(production_plan, sku_base_prices)
            
            # 2. 获取IO表供应商
            io_info = self.get_io_suppliers(threshold=0.001)
            
            # 3. 采购中间品
            intermediate_result = self.procure_intermediate_goods(
                production_value=production_value,
                suppliers=io_info['intermediate_goods'],
                period=period,
                strategy="random"
            )
            
            # 4. 采购抽象资源
            abstract_result = self.procure_abstract_resources(
                production_value=production_value,
                io_suppliers=io_info['abstract_resources'],
                period=period
            )
            
            # 5. 计算劳动和税收成本（如果未提供）
            if labor_cost is None:
                labor_cost = self.calculate_labor_cost()
            
            if tax_cost is None:
                tax_cost = self.calculate_tax_cost()
            
            # 6. 计算总成本
            cost_breakdown = self.calculate_total_cost(
                intermediate_goods_cost=intermediate_result['total_cost'],
                abstract_resources_cost=sum(abstract_result.values()),
                labor_cost=labor_cost,
                tax_cost=tax_cost
            )
            
            # 7. 计算单位成本
            total_quantity = sum(production_plan.values())
            unit_costs = {}
            for sku_id, quantity in production_plan.items():
                unit_cost = (cost_breakdown['total_cost'] / total_quantity) if total_quantity > 0 else 0
                unit_costs[sku_id] = unit_cost
                self.unit_costs[sku_id] = unit_cost
            
            # 8. 更新库存（如果需要）
            if update_inventory and self.product_market is not None:
                for sku_id, quantity in production_plan.items():
                    ray.get(self.product_market.update_stock.remote(sku_id, quantity))
            
            # 9. 记录生产历史
            production_record = {
                'period': period,
                'production_plan': production_plan.copy(),
                'production_value': production_value,
                'total_cost': cost_breakdown['total_cost'],
                'unit_costs': unit_costs.copy(),
                'cost_breakdown': cost_breakdown.copy(),
                'intermediate_items': len(intermediate_result['items']),
                'abstract_resources': len(abstract_result)
            }
            self.production_history.append(production_record)
            
            logger.info(
                f"Firm {self.firm_id} production completed: "
                f"{total_quantity} units, ${cost_breakdown['total_cost']:.2f} total cost"
            )
            
            return {
                'production_value': production_value,
                'total_cost': cost_breakdown['total_cost'],
                'unit_costs': unit_costs,
                'cost_breakdown': cost_breakdown,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Firm {self.firm_id} production failed: {e}")
            return {
                'production_value': 0,
                'total_cost': 0,
                'unit_costs': {},
                'cost_breakdown': {},
                'success': False,
                'error': str(e)
            }


class RetailFirm(Firm):
    # 子类默认值
    DEFAULT_INDUSTRY_TYPE = "retail"
    DEFAULT_DESCRIPTION = 'A physical goods retail channel that procures products from first-category suppliers and sells them to households, earning a channel margin.'
    def __init__(self, firm_id: str, **kwargs):
        kwargs.setdefault('industry_type', self.DEFAULT_INDUSTRY_TYPE)
        kwargs.setdefault('description', self.DEFAULT_DESCRIPTION)
        super().__init__(firm_id=firm_id, **kwargs)
        self.supply_chain_entry = _get_retail_supply_chain_entry(self.industry)
        self.supplier_industries: List[str] = list(self.supply_chain_entry.get("suppliers", []))


class ServiceFirm(Firm):
    # 子类默认值
    DEFAULT_INDUSTRY_TYPE = "service"
    DEFAULT_DESCRIPTION = 'A virtual resource and service provider that does not produce SKUs, but supplies abstract units (currency per unit), and requires hired labor.'
    def __init__(self, firm_id: str, **kwargs):
        kwargs.setdefault('industry_type', self.DEFAULT_INDUSTRY_TYPE)
        kwargs.setdefault('description', self.DEFAULT_DESCRIPTION)
        super().__init__(firm_id=firm_id, **kwargs)


if __name__ == "__main__":
    sample = RetailFirm(firm_id="1", description="Retail Firm", industry="441", economic_center=None, labor_market=None, product_market=None, is_agent=False)
    print(sample.supplier_industries)
    print(sample.supply_chain_entry)
