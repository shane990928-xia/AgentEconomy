from typing import Optional, List, TYPE_CHECKING, Dict, Any, Iterable, Tuple
from functools import lru_cache
from pathlib import Path
import json
import ast
import csv
import os
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
from agenteconomy.agent.firm_planning_policy import (
    ProductionPlan,
    ProductionPlanInput,
    ProductionPlanningPolicy,
)
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

        # 工资竞价(B):企业自身的出价工资溢价(相对 BLS×全局标量)。劳动紧张(空缺填不满)
        # 时上调以吸引工人,满员时回落。提供 wage→cost→price 的真实菲利普斯传导。
        self.wage_premium: float = 1.0

        # Firm financials
        self.capital_stock: float = 0.0 # Capital stock
        self.cash: float = 0.0 # Cash
        self.cost_structure: Optional[Dict[str, float]] = None
        self.compensation_ratio: float = 0.2

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
        if self.industry:
            try:
                cost_structure = get_cost_structure(self.industry)
                if cost_structure:
                    self.cost_structure = cost_structure
                    self.compensation_ratio = float(
                        cost_structure.get("compensation", self.compensation_ratio) or self.compensation_ratio
                    )
            except Exception:
                self.cost_structure = None
                self.compensation_ratio = 0.2

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

    def _compute_labor_budget(
        self,
        period: Optional[int] = None,
        current_demand_value: Optional[float] = None,
        allow_cash_based_startup_hiring: bool = False,
        retail_labor_value_share: float = 0.25,
        employment_adjustment_inertia: float = 0.0,
        beveridge_overposting_strength: float = 1.0,
    ) -> float:
        """
        计算本期劳动预算
        
        优先级：
        1. 当前需求价值（如果传入且 > 0）× compensation_ratio
        2. 上月收入的 compensation_ratio 比例
        3. 生产历史中最近一次的生产价值
        4. 可选：当前现金（仅当配置显式允许 startup hiring 时）
        
        Args:
            period: 当前期间
            current_demand_value: 本月的需求/生产价值（从 simulator 传入），可以是 0 或 None
            allow_cash_based_startup_hiring: 是否允许第一期用现金余额代理启动期收入
            retail_labor_value_share: 零售历史总销售额转附加值/毛利基数的份额
        """
        import logging
        _logger = logging.getLogger(__name__)
        
        # Keep labor budgets proportional to observed demand or income. A fixed
        # dollar floor would turn tiny demand signals into real payroll costs.
        MIN_COMPENSATION_RATIO = 0.20   # 回退到基线（劳动成本杠杆经实测为收缩性，就业靠需求侧解决）
        
        base_value = 0.0
        source = "none"
        
        # 优先使用当前需求价值（仅当 > 0 时）
        if current_demand_value is not None and current_demand_value > 0:
            base_value = float(current_demand_value)
            source = "current_demand"
        
        # 其次查询上月收入
        if base_value <= 0:
            current_period = int(period if period is not None else self.current_period or 0)
            if current_period > 0:
                stats = self._call_economic_center(
                    "query_firm_monthly_financials",
                    firm_id=self.firm_id,
                    month=current_period - 1,
                )
                if isinstance(stats, dict):
                    income = float(stats.get("monthly_income", 0.0) or 0.0)
                    if income > 0:
                        base_value = income
                        source = "last_month_income"
        
        # 再次查看生产历史
        if base_value <= 0 and self.production_history:
            try:
                prod_val = float(self.production_history[-1].get("production_value", 0.0) or 0.0)
                if prod_val > 0:
                    base_value = prod_val
                    source = "production_history"
            except Exception:
                pass
        
        # 使用当前现金（仅在第一期，即还未有收入历史时）
        if base_value <= 0:
            current_period = int(period if period is not None else self.current_period or 0)
            if allow_cash_based_startup_hiring and current_period <= 1:
                industry_type = str(getattr(self, "industry_type", "") or "")
                is_service_provider = industry_type.startswith("category_3_") or str(self.firm_id).startswith("svc_")
                if not is_service_provider:
                    # 第一期制造商/零售商允许使用初始现金启动；服务企业等待真实服务收入。
                    cash_val = float(self.cash or 0.0)
                    if cash_val > 0:
                        base_value = cash_val
                        source = "current_cash"
        
        # 如果所有收入来源均为0（需求=0，上月收入=0，生产历史=0）
        # 说明该企业没有市场需求，不应继续招人消耗资金
        if base_value <= 0:
            _logger.debug(
                f"[劳动预算] {self.firm_id}: 无收入来源，不发布岗位 "
                f"(demand={current_demand_value}, cash={self.cash}, history={len(self.production_history or [])})"
            )
            return 0.0

        industry_type = str(getattr(self, "industry_type", "") or "")
        is_retailer = industry_type == "retail" or str(self.firm_id).startswith("ret_")
        # 零售商的劳动基数应是渠道增加值(毛利)，而非全额销售/需求(GMV)。
        # GMV 里大部分是上游进货成本，按 GMV×comp_ratio 定工资会让工资超过毛利→必然亏损破产。
        # 故对 current_demand / last_month_income / current_cash 等以销售额计的来源都折算为毛利基数。
        if is_retailer and source in {"current_demand", "last_month_income", "current_cash", "production_history"}:
            share = max(0.0, min(1.0, float(retail_labor_value_share or 0.0)))
            if share > 0.0:
                base_value *= share
                source = f"{source}_retail_value_added"

        # 使用 compensation_ratio，但确保不低于最低比率
        actual_ratio = float(getattr(self, "compensation_ratio", 0.2) or 0.2)
        effective_ratio = max(actual_ratio, MIN_COMPENSATION_RATIO)
        
        budget = base_value * effective_ratio

        # 就业部分调整惯性:有效劳动预算向上月实际工资支出(已匹配的真实雇佣存量,
        # 同为月美元量纲)收敛。模拟招聘/解雇粘性——企业一个月内不会从满员砍到零、也不会
        # 翻倍扩张。这阻尼 demand→hiring→production→demand 的周期-2 蛛网震荡频率,但保留
        # 需求→就业的方向(区别于平滑需求信号本身,后者会削弱 Okun/Phillips 传导)。0=关闭。
        _emp_lambda = max(0.0, min(1.0, float(employment_adjustment_inertia or 0.0)))
        if _emp_lambda > 0.0 and self.labor_market is not None:
            try:
                wage_bill = self._call_labor_market("get_firm_wage_bill", self.firm_id)
                prev_payroll = float((wage_bill or {}).get("total_wage", 0.0) or 0.0)
            except Exception:
                prev_payroll = 0.0
            if prev_payroll > 0.0:
                budget = (1.0 - _emp_lambda) * budget + _emp_lambda * prev_payroll

        # ========== 贝弗里奇曲线修正：根据失业率调整职位发布倍数 ==========
        # 核心经济逻辑：
        #   低失业率 → 企业预期招人困难 → 超额发布职位（但大部分填不满）→ 高空缺率
        #   高失业率 → 企业容易招到人 → 只发布刚需职位 → 低空缺率
        #
        # 重要：adjustment 的下限是 1.0，不能低于 1.0
        # 贝弗里奇修正只影响"超额发布倍数"，不压缩实际劳动需求
        # 否则高失业率 → 企业不招人 → 失业率不降 → 死循环
        self._beveridge_adjustment = 1.0
        if self.labor_market is not None:
            try:
                labor_summary = self._call_labor_market("summary")
                if labor_summary and isinstance(labor_summary, dict):
                    unemployment_rate = float(labor_summary.get("unemployment_rate", 0.5) or 0.5)
                    if unemployment_rate < 0.05:
                        # 极低失业：企业疯狂抢人，大幅超额发布（2.5-3.0倍）
                        self._beveridge_adjustment = 2.5 + (0.05 - unemployment_rate) * 10.0
                    elif unemployment_rate < 0.10:
                        # 低失业：超额发布（1.5-2.5倍）
                        self._beveridge_adjustment = 2.5 - (unemployment_rate - 0.05) * 20.0
                    elif unemployment_rate < 0.20:
                        # 中等失业：略微超额发布（1.0-1.5倍）
                        self._beveridge_adjustment = 1.5 - (unemployment_rate - 0.10) * 5.0
                    else:
                        # 高失业：正常发布（1.0倍），不压缩
                        self._beveridge_adjustment = 1.0
                    # 超发强度线性缩放:adj=1+(adj-1)*strength。strength=1 保留旧行为;
                    # strength<1 收敛向 1.0;strength=0 完全关闭幻影超发(消除劳动市场周期-2 蛛网)。
                    _bev_strength = max(0.0, float(beveridge_overposting_strength if beveridge_overposting_strength is not None else 1.0))
                    if _bev_strength != 1.0:
                        self._beveridge_adjustment = 1.0 + (self._beveridge_adjustment - 1.0) * _bev_strength
                    budget = budget * self._beveridge_adjustment
                    _logger.debug(
                        f"[劳动预算-贝弗里奇] {self.firm_id}: unemp_rate={unemployment_rate:.3f}, "
                        f"adjustment={self._beveridge_adjustment:.3f}, budget={budget:.2f}"
                    )
            except Exception as e:
                _logger.debug(f"[劳动预算] 获取失业率失败: {e}")
        
        _logger.debug(
            f"[劳动预算] {self.firm_id}: base={base_value:.2f} (source={source}), "
            f"ratio={actual_ratio:.2f}→{effective_ratio:.2f}, budget={budget:.2f}"
        )
        return budget

    async def _llm_reweight_hiring(self, *, candidate_socs, anchor_weights,
                                   labor_budget, demand_value, job_data):
        """Ask the LLM for a hiring-priority weight per candidate SOC.

        Returns a {soc: weight>=0} dict restricted to candidate_socs, or None on
        any failure (caller then keeps the rule weights). Constrained decision:
        the LLM only sets relative emphasis; feasibility/positions stay rule-based.
        """
        socs = [s for s in (candidate_socs or []) if s]
        if not socs:
            return None
        # attach occupation titles for the prompt
        self._soc_titles = {
            s: (job_data.get(s, {}) or {}).get("title", "") for s in socs
        }
        prompt = build_firm_post_job_prompt(
            self,
            candidate_socs=socs,
            labor_budget=float(labor_budget or 0.0),
            demand_value=float(demand_value or 0.0),
            anchor_weights=anchor_weights,
        )
        raw = await call_llm(prompt)
        if not raw:
            return None
        text = str(raw).strip()
        # strip code fences / extract the JSON object
        if "```" in text:
            text = text.split("```")[1] if len(text.split("```")) > 1 else text
            text = text.replace("json", "", 1).strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        parsed = json.loads(text[start:end + 1])
        if not isinstance(parsed, dict):
            return None
        out = {}
        for s in socs:
            try:
                w = float(parsed.get(s, anchor_weights.get(s, 0.0)))
            except (TypeError, ValueError):
                w = float(anchor_weights.get(s, 0.0) or 0.0)
            out[s] = max(0.0, w)
        if sum(out.values()) <= 0.0:
            return None
        return out

    async def _decide_job_postings_from_data(
        self,
        period: Optional[int] = None,
        max_job_types: int = 10,
        current_demand_value: Optional[float] = None,
        min_part_time_hours_per_month: Optional[float] = None,
        max_startup_part_time_hours_per_month: Optional[float] = None,
        min_job_budget_coverage: float = 1.0,
        allow_cash_based_startup_hiring: bool = False,
        retail_labor_value_share: float = 0.25,
        employment_adjustment_inertia: float = 0.0,
        beveridge_overposting_strength: float = 1.0,
        use_llm: bool = False,
    ) -> List[Job]:
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

        labor_budget = self._compute_labor_budget(
            period,
            current_demand_value=current_demand_value,
            allow_cash_based_startup_hiring=allow_cash_based_startup_hiring,
            retail_labor_value_share=retail_labor_value_share,
            employment_adjustment_inertia=employment_adjustment_inertia,
            beveridge_overposting_strength=beveridge_overposting_strength,
        )
        if labor_budget <= 0:
            return []

        # LLM hiring decision (constrained): the LLM reweights the candidate-SOC
        # hiring emphasis given the firm's industry/headcount/budget/demand; the
        # rule distribution serves as the anchor. Output flows through the same
        # allocation machinery below, so positions/budget math stays valid (only
        # the occupation MIX is LLM-driven). On any failure the rule weights are
        # kept, so use_llm never breaks hiring.
        if use_llm:
            try:
                llm_weights = await self._llm_reweight_hiring(
                    candidate_socs=candidate_socs,
                    anchor_weights=weights,
                    labor_budget=labor_budget,
                    demand_value=float(current_demand_value or 0.0),
                    job_data=job_data,
                )
                if llm_weights:
                    weights = llm_weights
                    total_weight = sum(weights.values()) or float(len(candidate_socs))
            except Exception as e:
                logger.debug(f"[岗位发布-LLM] {self.firm_id}: reweight failed, keep rule weights: {e}")

        hours_per_week = 40.0
        weeks_per_month = 4.0
        hours_per_period = hours_per_week * weeks_per_month
        if min_part_time_hours_per_month is None:
            try:
                min_part_time_hours = float(os.getenv("FIRM_MIN_PART_TIME_HOURS_PER_MONTH", "20.0") or 20.0)
            except ValueError:
                min_part_time_hours = 20.0
        else:
            min_part_time_hours = float(min_part_time_hours_per_month or 20.0)
        if max_startup_part_time_hours_per_month is None:
            try:
                max_startup_part_time_hours = float(os.getenv("FIRM_MAX_STARTUP_PART_TIME_HOURS_PER_MONTH", "160.0") or 160.0)
            except ValueError:
                max_startup_part_time_hours = 160.0
        else:
            max_startup_part_time_hours = float(max_startup_part_time_hours_per_month or 160.0)
        min_part_time_hours = max(1.0, min(min_part_time_hours, hours_per_period))
        max_startup_part_time_hours = max(min_part_time_hours, min(max_startup_part_time_hours, hours_per_period))

        ranked_socs = sorted(candidate_socs, key=lambda s: weights.get(s, 0.0), reverse=True)
        soc_infos: List[Dict[str, Any]] = []
        cheapest_min_job_cost = 0.0
        for soc in ranked_socs[:max_job_types]:
            info = job_data.get(soc)
            if not info:
                continue
            hourly_wage = float(info.get("wage", 0.0) or 0.0) * float(os.getenv("AGENTECO_WAGE_SCALE", "1.0") or 1.0) * float(getattr(self, "wage_premium", 1.0) or 1.0)
            if hourly_wage <= 0.0:
                continue
            min_job_cost = hourly_wage * min_part_time_hours
            full_job_cost = hourly_wage * hours_per_period
            if min_job_cost <= 0.0:
                continue
            soc_infos.append(
                {
                    "soc": soc,
                    "info": info,
                    "hourly_wage": hourly_wage,
                    "min_job_cost": min_job_cost,
                    "full_job_cost": full_job_cost,
                    "weight": float(weights.get(soc, 1.0) or 1.0),
                }
            )
            if cheapest_min_job_cost <= 0.0 or min_job_cost < cheapest_min_job_cost:
                cheapest_min_job_cost = min_job_cost
        if not soc_infos:
            return []

        coverage = max(0.0, float(min_job_budget_coverage or 0.0))
        if cheapest_min_job_cost > 0.0 and labor_budget < cheapest_min_job_cost * coverage:
            logger.debug(
                f"[劳动预算] {self.firm_id}: budget={labor_budget:.2f} below minimum "
                f"job cost threshold={cheapest_min_job_cost * coverage:.2f}"
            )
            return []

        jobs: List[Job] = []
        remaining_budget = labor_budget
        planned_socs: set[str] = set()

        def _make_job(soc_info: Dict[str, Any], hours: float, positions: int) -> Job:
            info = soc_info["info"]
            job = Job.create(
                soc=soc_info["soc"],
                title=info.get("title") or soc_info["soc"],
                wage_per_hour=float(soc_info["hourly_wage"] or 0.0),
                firm_id=self.firm_id,
                description=info.get("description"),
                hours_per_period=hours,
                required_skills=info.get("skills") or {},
                required_abilities=info.get("abilities") or {},
            )
            job.positions_available = int(positions)
            return job

        def _add_full_position(soc_info: Dict[str, Any]) -> None:
            soc = str(soc_info["soc"])
            if soc in planned_socs:
                for job in jobs:
                    if job.SOC == soc:
                        job.positions_available += 1
                        return
            jobs.append(_make_job(soc_info, hours_per_period, 1))
            planned_socs.add(soc)

        for soc_info in soc_infos:
            monthly_wage = float(soc_info["full_job_cost"] or 0.0)
            if monthly_wage <= 0:
                continue
            budget_share = labor_budget * (float(soc_info["weight"] or 1.0) / total_weight)
            positions = int(budget_share // monthly_wage)
            if positions <= 0:
                continue
            max_affordable = int(remaining_budget // monthly_wage)
            if max_affordable <= 0:
                continue
            positions = min(positions, max_affordable)
            remaining_budget -= positions * monthly_wage

            jobs.append(_make_job(soc_info, hours_per_period, positions))
            planned_socs.add(str(soc_info["soc"]))

        # Use the residual aggregate budget instead of letting SOC weight
        # fragmentation suppress all hiring. Full positions are added one round at
        # a time across SOCs to keep a broad applicant pool.
        while True:
            allocated = False
            for soc_info in soc_infos:
                monthly_wage = float(soc_info["full_job_cost"] or 0.0)
                if monthly_wage <= 0.0 or remaining_budget < monthly_wage:
                    continue
                _add_full_position(soc_info)
                remaining_budget -= monthly_wage
                allocated = True
            if not allocated:
                break

        # If the remaining budget cannot buy a full position, post one part-time
        # job only when the firm can fund the configured minimum hours.
        for soc_info in soc_infos:
            if remaining_budget < float(soc_info["min_job_cost"] or 0.0):
                continue
            if str(soc_info["soc"]) in planned_socs:
                continue

            hourly_wage = float(soc_info["hourly_wage"] or 0.0)
            if hourly_wage <= 0.0:
                continue
            affordable_hours = min(max_startup_part_time_hours, remaining_budget / hourly_wage)
            if affordable_hours >= min_part_time_hours:
                jobs.append(_make_job(soc_info, affordable_hours, 1))
                planned_socs.add(str(soc_info["soc"]))
                remaining_budget -= affordable_hours * hourly_wage

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
    def _update_wage_premium(
        self,
        bid_up: float = 0.04,
        bid_down: float = 0.02,
        premium_min: float = 0.5,
        premium_max: float = 2.5,
    ) -> float:
        """
        工资竞价(B):按企业自身的空缺填补情况内生调整出价工资溢价。

        机制(无 reduced-form 捷径):
        - 读取本企业当前仍未填补的空缺(get_firm_job_snapshot,即上轮 post→match 后
          还剩下的开放岗位)。
        - 有未填补空缺 → 招不到人 → 劳动紧张 → 上调溢价吸引工人(bid_up)。
        - 无未填补空缺 → 已招满 → 溢价向 1.0 回落(bid_down),避免单调上行。
        这样劳动市场紧张直接体现为企业加薪 → 真实劳动成本上升 → 单位成本上升 →
        价格上升,菲利普斯曲线从供需传导中涌现,而非由公式硬编码。
        """
        try:
            snapshot = self._call_labor_market("get_firm_job_snapshot", self.firm_id)
        except Exception:
            snapshot = None
        open_positions = 0
        if isinstance(snapshot, dict):
            for positions in snapshot.values():
                try:
                    open_positions += int(positions or 0)
                except (TypeError, ValueError):
                    continue
        premium = float(getattr(self, "wage_premium", 1.0) or 1.0)
        if open_positions > 0:
            premium *= (1.0 + max(0.0, float(bid_up)))
        else:
            # 满员:向中性 1.0 线性回落(对 >1 和 <1 都收敛)
            premium += (1.0 - premium) * max(0.0, min(1.0, float(bid_down)))
        premium = max(float(premium_min), min(float(premium_max), premium))
        self.wage_premium = premium
        return premium

    async def post_jobs(
        self,
        period: Optional[int] = None,
        current_demand_value: Optional[float] = None,
        min_part_time_hours_per_month: Optional[float] = None,
        max_startup_part_time_hours_per_month: Optional[float] = None,
        min_job_budget_coverage: float = 1.0,
        allow_cash_based_startup_hiring: bool = False,
        retail_labor_value_share: float = 0.25,
        use_llm: bool = False,
        wage_bidding_enabled: bool = False,
        wage_bid_up: float = 0.04,
        wage_bid_down: float = 0.02,
        wage_premium_min: float = 0.5,
        wage_premium_max: float = 2.5,
        employment_adjustment_inertia: float = 0.0,
        beveridge_overposting_strength: float = 1.0,
    ):
        """
        Post jobs to the labor market

        支持双向调整：
        - desired > existing: 发布增量职位
        - desired < existing: 缩减空缺职位（不影响已雇佣的员工）

        Args:
            period: 当前期间
            current_demand_value: 本月的需求/生产价值（用于计算劳动预算）
            use_llm: If True, the LLM sets the hiring emphasis across candidate
                occupations (constrained SOC reweight inside
                _decide_job_postings_from_data); positions/budget stay rule-based.
            wage_bidding_enabled: 若开启,先按上轮空缺填补情况内生调整 self.wage_premium。
        """
        if wage_bidding_enabled:
            self._update_wage_premium(
                bid_up=wage_bid_up,
                bid_down=wage_bid_down,
                premium_min=wage_premium_min,
                premium_max=wage_premium_max,
            )
        jobs = await self._decide_job_postings_from_data(
            period=period,
            current_demand_value=current_demand_value,
            min_part_time_hours_per_month=min_part_time_hours_per_month,
            max_startup_part_time_hours_per_month=max_startup_part_time_hours_per_month,
            min_job_budget_coverage=min_job_budget_coverage,
            allow_cash_based_startup_hiring=allow_cash_based_startup_hiring,
            retail_labor_value_share=retail_labor_value_share,
            employment_adjustment_inertia=employment_adjustment_inertia,
            beveridge_overposting_strength=beveridge_overposting_strength,
            use_llm=use_llm,
        )
        if jobs:
            snapshot = self._call_labor_market("get_firm_job_snapshot", self.firm_id)
            if not isinstance(snapshot, dict):
                snapshot = {}
            employed_by_soc: Dict[str, int] = {}
            wage_bill = self._call_labor_market("get_firm_wage_bill", self.firm_id)
            if isinstance(wage_bill, dict):
                for employee in wage_bill.get("employees", []) or []:
                    if not isinstance(employee, dict):
                        continue
                    soc = employee.get("job_SOC") or employee.get("soc")
                    if not soc:
                        continue
                    employed_by_soc[str(soc)] = employed_by_soc.get(str(soc), 0) + 1
            desired_socs = {job.SOC for job in jobs}
            for soc, positions in snapshot.items():
                if soc in desired_socs:
                    continue
                reduce_by = int(positions or 0)
                if reduce_by > 0:
                    self._call_labor_market("reduce_job_positions", self.firm_id, soc, reduce_by)
            to_post: List[Job] = []
            for job in jobs:
                desired = int(job.positions_available or 0)
                current_staff = int(employed_by_soc.get(str(job.SOC), 0) or 0)
                existing = int(snapshot.get(job.SOC, 0) or 0)
                target_open = max(0, desired - current_staff)
                delta = target_open - existing
                if delta > 0:
                    # 需要增加职位
                    job.positions_available = delta
                    to_post.append(job)
                elif delta < 0:
                    # 需要缩减空缺职位（市场宽松时减少不必要的空缺）
                    self._call_labor_market(
                        "reduce_job_positions", self.firm_id, job.SOC, abs(delta)
                    )
            if to_post:
                self._call_labor_market("apply_job_plan", self.firm_id, to_post)
            return to_post

        snapshot = self._call_labor_market("get_firm_job_snapshot", self.firm_id)
        if isinstance(snapshot, dict):
            for soc, positions in snapshot.items():
                reduce_by = int(positions or 0)
                if reduce_by > 0:
                    self._call_labor_market("reduce_job_positions", self.firm_id, soc, reduce_by)

        # No fundable job plan this month → post nothing. (LLM hiring is applied
        # inside _decide_job_postings_from_data as a constrained SOC reweight, not
        # as a separate free-text fallback.)
        return []

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

    def _estimate_available_labor_for_planning(self) -> float:
        total_hours = 0.0
        for employee in self.employee_list or []:
            hours = getattr(employee, "total_hours", None)
            if hours is None:
                hours = getattr(employee, "hours_per_period", None)
            try:
                total_hours += max(0.0, float(hours or 0.0))
            except (TypeError, ValueError):
                continue

        if total_hours <= 0.0 and self.employee_count > 0:
            total_hours = float(self.employee_count) * 160.0
        return total_hours

    def build_production_plan(
        self,
        sales_history: Optional[Iterable[Any]] = None,
        unmet_demand_history: Optional[Iterable[Any]] = None,
        current_inventory: Any = 0.0,
        target_inventory_months: float = 1.0,
        ema_alpha: float = 0.5,
        include_unmet_demand: bool = True,
        fallback_expected_demand: float = 0.0,
        available_labor: Optional[float] = None,
        labor_productivity: float = 1.0,
        capital_stock: Optional[float] = None,
        capital_productivity: float = 1.0,
        cash: Optional[float] = None,
        unit_cash_cost: Optional[float] = None,
        cash_reserve: float = 0.0,
        approved_credit: Optional[float] = None,
        credit_limit: Optional[float] = None,
        credit_outstanding: float = 0.0,
        use_current_state_constraints: bool = True,
    ) -> ProductionPlan:
        """
        Build an aggregate active production plan without executing production.

        Existing `produce()` callers remain unchanged. This method is a planning
        surface for the simulator to translate aggregate intent into SKU plans.
        """
        if use_current_state_constraints:
            if available_labor is None:
                available_labor = self._estimate_available_labor_for_planning()
            if capital_stock is None:
                capital_stock = self.capital_stock
            if cash is None and unit_cash_cost is not None:
                cash = self.cash

        plan_input = ProductionPlanInput(
            sales_history=[] if sales_history is None else sales_history,
            unmet_demand_history=[] if unmet_demand_history is None else unmet_demand_history,
            current_inventory=current_inventory,
            target_inventory_months=target_inventory_months,
            ema_alpha=ema_alpha,
            include_unmet_demand=include_unmet_demand,
            fallback_expected_demand=fallback_expected_demand,
            available_labor=available_labor,
            labor_productivity=labor_productivity,
            capital_stock=capital_stock,
            capital_productivity=capital_productivity,
            cash=cash,
            unit_cash_cost=unit_cash_cost,
            cash_reserve=cash_reserve,
            approved_credit=approved_credit,
            credit_limit=credit_limit,
            credit_outstanding=credit_outstanding,
        )
        return ProductionPlanningPolicy().build_plan(plan_input)


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
            self._procurement = IntermediateGoodsProcurement(
                self.product_market,
                receiver_id_resolver=self._resolve_intermediate_receiver_id,
            )
        return self._procurement

    def _resolve_intermediate_receiver_id(self, sku_id: str, industry_code: Optional[str], sku_obj: Optional[Any] = None) -> Optional[str]:
        if self.economic_center is None:
            return None
        if sku_obj is not None:
            owner_id = getattr(sku_obj, "owner_id", None)
            if owner_id:
                owner_id = str(owner_id)
                if owner_id.startswith(("mfg_", "ret_", "svc_")):
                    return owner_id
                firm_id = self._call_economic_center("resolve_market_price_id", "intermediate_goods", owner_id)
                if firm_id:
                    return firm_id
        if industry_code:
            firm_id = self._call_economic_center("resolve_market_price_id", "intermediate_goods", industry_code)
            if firm_id:
                return firm_id
        return None
    
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
                    "supplier_industry": getattr(item, "supplier_industry", None),
                    "receiver_id": getattr(item, "receiver_id", None),
                }
                for item in result.get('items', [])
            ]
            items_by_receiver: Dict[Optional[str], List[Dict[str, Any]]] = defaultdict(list)
            for item in items_payload:
                items_by_receiver[item.get("receiver_id")].append(item)

            for receiver_id, group_items in items_by_receiver.items():
                group_total = float(sum(float(it.get("total_cost") or 0.0) for it in group_items))
                if group_total <= 0:
                    continue
                group_by_industry: Dict[str, float] = defaultdict(float)
                for it in group_items:
                    code = it.get("supplier_industry")
                    if code:
                        group_by_industry[str(code)] += float(it.get("total_cost") or 0.0)
                self._call_economic_center(
                    "record_intermediate_goods_purchase",
                    month=period,
                    buyer_id=self.firm_id,
                    total_cost=group_total,
                    costs_by_industry=dict(group_by_industry),
                    items=group_items,
                    receiver_id=receiver_id,
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
            # 1. 计算生产价值（计划值）
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

            # 木桶效应：中间品供给不足则按最短板降低产量
            bottleneck_ratio = float(intermediate_result.get("bottleneck_ratio", 1.0) or 1.0)
            if bottleneck_ratio < 1.0:
                scaled_plan: Dict[str, int] = {}
                for sku_id, qty in production_plan.items():
                    new_qty = int(float(qty) * bottleneck_ratio)
                    if new_qty > 0:
                        scaled_plan[sku_id] = new_qty
                production_plan = scaled_plan
                production_value = self.calculate_production_value(production_plan, sku_base_prices)
                logger.info(
                    f"Firm {self.firm_id} bottleneck_ratio={bottleneck_ratio:.4f} "
                    f"scaled production to {sum(production_plan.values())} units"
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
            
            # 9. 根据实际成本更新价格
            if update_inventory and self.product_market is not None and total_quantity > 0:
                avg_unit_cost = cost_breakdown['total_cost'] / total_quantity
                # 成本推动渠道(Phillips)：把当前工资水平并入单位成本基准。劳动市场紧张时
                # 内生工资上升 → 单位成本上升 → 价格上升 → 通胀。
                # NOTE: 这是一条 reduced-form 捷径(直接把全局工资标量乘进成本)，用 env gate
                # AGENTECO_COSTPUSH 控制，便于消融实验区分"涌现"vs"硬编码"。默认开(=1)。
                if os.getenv("AGENTECO_COSTPUSH", "1") == "1":
                    try:
                        _wscale = float(os.getenv("AGENTECO_WAGE_SCALE", "1.0") or 1.0)
                    except (TypeError, ValueError):
                        _wscale = 1.0
                    # 劳动成本份额约 0.55(IO 补偿均值)；按工资相对基准(0.3 起调)的偏离放大单位成本。
                    _labor_share = 0.55
                    _wage_ref = 0.3
                    if _wage_ref > 0:
                        avg_unit_cost = avg_unit_cost * (
                            (1.0 - _labor_share) + _labor_share * (_wscale / _wage_ref)
                        )
                # 批量更新该行业所有产品的价格
                ray.get(self.product_market.batch_update_prices_by_industry.remote(
                    manufacturer_code=self.industry,
                    avg_unit_cost=avg_unit_cost,
                    manufacturer_margin=0.15,  # 制造商利润率15%
                    retail_margin=0.45         # 零售商加价45%：覆盖零售商工资+管理费用，避免负毛利破产
                ))
            
            # 10. 记录生产历史
            production_record = {
                'period': period,
                'production_plan': production_plan.copy(),
                'production_value': production_value,
                'total_cost': cost_breakdown['total_cost'],
                'unit_costs': unit_costs.copy(),
                'cost_breakdown': cost_breakdown.copy(),
                'intermediate_items': len(intermediate_result['items']),
                'abstract_resources': len(abstract_result),
                'avg_unit_cost': cost_breakdown['total_cost'] / total_quantity if total_quantity > 0 else 0
            }
            self.production_history.append(production_record)
            
            logger.info(
                f"Firm {self.firm_id} production completed: "
                f"{total_quantity} units, ${cost_breakdown['total_cost']:.2f} total cost"
            )

            return {
                'production_plan': production_plan.copy(),
                'production_value': production_value,
                'total_cost': cost_breakdown['total_cost'],
                'unit_costs': unit_costs,
                'cost_breakdown': cost_breakdown,
                'intermediate_by_industry': intermediate_result.get('by_industry', {}),
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
