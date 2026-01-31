from __future__ import annotations

import ast
import asyncio
import copy
import csv
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import ray

_JOB_SKILLS_CSV = Path(__file__).resolve().parents[1] / "data" / "jobs_with_skills_abilities_IM_merged.csv"

from agenteconomy.center.Model import Job, JobApplication, LaborHour, Product
from agenteconomy.llm.llm import call_llm
from agenteconomy.llm.prompt_template import (
    CONSUMPTION_MAJOR_BUDGET_PROMPT,
    CONSUMPTION_NEEDS_BY_CATEGORY_PROMPT,
    PURCHASE_BY_CATEGORY_PROMPT,
    JOB_APPLICATION_DECISION_PROMPT,
    JOB_OFFER_DECISION_PROMPT,
    PERSONA_UPDATE_PROMPT,
)
from agenteconomy.utils.logger import get_logger

logger = get_logger(name="household")


# =============================================================================
# Extensible IO models (minimal, but runnable)
# =============================================================================

@dataclass
class CategoryPlan:
    category: str
    budget_amount: float
    need_descriptions: List[str] = field(default_factory=list)


@dataclass
class CategoryNeedsOutput:
    total_budget: float = 0.0
    category_plans: List[CategoryPlan] = field(default_factory=list)
    note: str = ""
    raw_llm_output: Optional[str] = None


@dataclass
class MajorBudgetOutput:
    total_budget: float = 0.0
    budgets: Dict[str, float] = field(default_factory=dict)
    note: str = ""
    raw_llm_output: Optional[str] = None


@dataclass
class BudgetedPurchase:
    category: str
    product_id: str
    allocated_budget: float
    reason: str = ""


@dataclass
class BudgetedPurchasePlan:
    purchases: List[BudgetedPurchase] = field(default_factory=list)
    note: str = ""
    raw_llm_output: Optional[str] = None


@dataclass
class JobMatch:
    job: Job
    loss: float


@dataclass
class JobApplicationDecision:
    apply_job_ids: List[str] = field(default_factory=list)
    note: str = ""
    raw_llm_output: Optional[str] = None


@dataclass
class JobOfferDecision:
    accept_job_id: Optional[str] = None
    note: str = ""
    raw_llm_output: Optional[str] = None


# =============================================================================
# Household agent (new workflow)
# =============================================================================


class Household:
    """
    Household agent (minimal, extensible).

    Two major capabilities:
    1) Consumption decision (v2 only):
       - Step1: LLM budgets + need descriptions by category
       - Step2: Vector search (Qdrant) per need description
       - Step3: LLM chooses products + allocates per-product budgets
       - Step4: validation placeholder
    2) Job matching decision (household-side):
       - list_job_seekers (rule)
       - match_jobs_topk_by_loss (rule)
       - decide_job_applications (LLM; rule fallback)
       - decide_offer (LLM; rule fallback)
    """

    # Default dataset location (override via env to keep it configurable)
    DEFAULT_DATA_DIR = os.getenv(
        "AGENTECO_DATA_DIR",
        str(Path(__file__).resolve().parents[1] / "data" / "household"),
    )

    # Minimal, class-level caches (avoid re-reading large json/csv for each instance)
    _HOUSEHOLD_ROW_BY_HOUSEHOLD_IDX: Optional[Dict[int, Dict[str, Any]]] = None
    _HOUSEHOLD_ROW_BY_FID: Optional[Dict[int, Dict[str, Any]]] = None
    _CODEBOOK_BY_VAR: Optional[Dict[str, Dict[str, Any]]] = None
    _PERSONA_BY_NAME: Optional[Dict[str, Dict[str, Any]]] = None
    _CENSUS2010_TO_SOC2010: Optional[Dict[str, List[Tuple[str, str]]]] = None

    # Fields that look "categorical" in metadata but are actually numeric measures (should NOT decode).
    # Example: ER82017 is a count (household size); ER85780 is completed grades.
    # Also: ER82181 (RP) / ER82500 (SP) are 2010 Census occupation codes; we will map them via
    # census2010_to_soc2010_exploded.csv instead of using codebook big categories.
    NO_DECODE_FIELDS: set[str] = {"ER82017", "ER85780", "ER82181", "ER82500"}

    MONTHLY_FLOW_FIELDS: set[str] = {"ER85629", "ER85701", "ER85747", "ER85768"}

    MONTHLY_RESET_FIELDS: Tuple[str, ...] = (
        "ER85629",
        "RP_income",
        "SP_income",
        "ER85768",
        "ER85701",
        "ER85747",
        "expenditure_retail_merchandise",
        "expenditure_insurance",
        "expenditure_utilities",
        "expenditure_transportation",
    )

    SIMULATION_STATE_FIELDS: set[str] = {
        "RP_income",
        "SP_income",
        "expenditure_retail_merchandise",
        "expenditure_insurance",
        "expenditure_utilities",
        "expenditure_transportation",
    }

    PERSONA_PROMPT_FIELDS: Tuple[str, ...] = (
        "persona_name",
        "core_characteristics",
        "behavior_patterns",
    )

    PAST_STATUS_LABEL_OVERRIDES: Dict[str, str] = {
        "ER82017": "Household size (number of people)",
        "ER82027": "Life satisfaction",
        "ER84520": "General health status (reference person)",
        "ER82800": "Time pressure frequency (reference person)",
        "ER85629": "Monthly household income",
        "ER85666": "Credit card / store card debt",
        "ER85692": "Net wealth (including home equity)",
        "ER85701": "Monthly housing expenditure",
        "ER85747": "Monthly health care expenditure",
        "ER85768": "Monthly total expenditure",
        "ER85780": "Education level completed (reference person)",
        "ER82181": "Occupation (reference person, main job)",
        "ER82500": "Occupation (spouse/partner, main job)",
        "ER82433": "Currently working (reference person)",
        "SP_employment_status": "Currently working (spouse/partner)",
        "RP_income": "Monthly income (reference person)",
        "SP_income": "Monthly income (spouse/partner)",
        "expenditure_retail_merchandise": "Monthly retail merchandise expenditure",
        "expenditure_insurance": "Monthly insurance expenditure",
        "expenditure_utilities": "Monthly utilities expenditure",
        "expenditure_transportation": "Monthly transportation expenditure",
    }

    # Employment status codes (PSID-style; keep numeric codes as requested)
    _EMPLOYED_CODE: int = 1
    _NOT_EMPLOYED_CODE: int = 5

    @classmethod
    def set_preloaded_data(
        cls,
        *,
        household_row_by_household_idx: Optional[Dict[int, Dict[str, Any]]] = None,
        household_row_by_fid: Optional[Dict[int, Dict[str, Any]]] = None,
        codebook_by_var: Optional[Dict[str, Dict[str, Any]]] = None,
        persona_by_name: Optional[Dict[str, Dict[str, Any]]] = None,
        census2010_to_soc2010: Optional[Dict[str, List[Tuple[str, str]]]] = None,
    ) -> None:
        """
        Inject preloaded dataset artifacts (no file I/O).
        External loaders can read CSV/JSON once and reuse across many Household instances.
        """
        if household_row_by_household_idx is not None:
            cls._HOUSEHOLD_ROW_BY_HOUSEHOLD_IDX = household_row_by_household_idx
        if household_row_by_fid is not None:
            cls._HOUSEHOLD_ROW_BY_FID = household_row_by_fid
        if codebook_by_var is not None:
            cls._CODEBOOK_BY_VAR = codebook_by_var
        if persona_by_name is not None:
            cls._PERSONA_BY_NAME = persona_by_name
        if census2010_to_soc2010 is not None:
            cls._CENSUS2010_TO_SOC2010 = census2010_to_soc2010

    def __init__(
        self,
        household_id: str,
        name: str,
        description: str,
        owner: str,
        *,
        data_dir: Optional[str] = None,
        persona_mapping_csv: str = "J357328_merged_household_persona_mapping.csv",
        codebook_json: str = "codebook.json",
        personas_json: str = "personas_final.json",
        census2010_to_soc2010_csv: str = "census2010_to_soc2010_exploded.csv",
        load_profile: bool = True,
        preloaded_data: Optional[Dict[str, Any]] = None,
    ):
        self.household_id = household_id
        self.name = name
        self.description = description
        self.owner = owner

        # Broad, extensible info blobs.
        # NOTE: we intentionally keep this minimal; persona + past status will fill in rich description.
        self.household_info: Dict[str, Any] = {
            "household_id": household_id,
            "name": name,
        }

        # Consumption categories (industry-like labels; store category NAME only)
        # Based on the user-specified set for step1 budgeting.
        self.consumption_categories: List[str] = [
            "Farms",
            "Forestry, fishing, and related activities",
            "Food and beverage and tobacco products",
            "Textile mills and textile product mills",
            "Apparel and leather and allied products",
            "Wood products",
            "Paper products",
            "Petroleum and coal products",
            "Chemical products",
            "Plastics and rubber products",
            "Nonmetallic mineral products",
            "Fabricated metal products",
            "Machinery",
            "Computer and electronic products",
            "Electrical equipment, appliances, and components",
            "Motor vehicles, bodies and trailers, and parts",
            "Other transportation equipment",
            "Furniture and related products",
            "Miscellaneous manufacturing",
            "Publishing industries, except internet (includes software)",
        ]

        # Profile from dataset (optional)
        self.data_dir = str(data_dir or self.DEFAULT_DATA_DIR)
        self.profile_row: Optional[Dict[str, Any]] = None  # raw CSV row (strings)
        self.csv_raw: Dict[str, Any] = {}
        self.csv_values: Dict[str, Any] = {}  # parsed/normalized values
        # NOTE: we store *codes* in csv_values / member variables.
        # Any decoding to human-readable category text should happen via decode helpers (for prompts/UI).
        self.csv_decoded: Dict[str, Any] = {}  # optional cache; not a source of truth

        self.persona: Optional[Dict[str, Any]] = None  # persona record from personas_final.json
        self.past_household_status: Dict[str, Any] = {}
        self.past_household_status_text: str = ""

        # Employment status (requested):
        # - RP employment status uses ER82433 (we will force it to "not employed" during initialization)
        # - SP employment status is a new field (not in the CSV) and defaults to "not employed"
        self.ER82433 = self._NOT_EMPLOYED_CODE
        self.SP_employment_status = self._NOT_EMPLOYED_CODE
        self.csv_values["ER82433"] = self._NOT_EMPLOYED_CODE
        self.csv_values["SP_employment_status"] = self._NOT_EMPLOYED_CODE

        if load_profile:
            self._ensure_profile_loaded(
                persona_mapping_csv=persona_mapping_csv,
                codebook_json=codebook_json,
                personas_json=personas_json,
                census2010_to_soc2010_csv=census2010_to_soc2010_csv,
                preloaded_data=preloaded_data,
            )

        # Dependencies (can be Ray ActorHandles or local instances)
        self.economic_center = None
        self.product_market = None
        self.labor_market = None
        self.labor_hours: List[LaborHour] = []

        # LLM config: use agenteconomy.llm.llm Router ("simple" / "strong")
        self.llm_model_type: Literal["simple", "strong"] = "simple"

    @classmethod
    def _normalize_census_code_4(cls, code: Any) -> Optional[str]:
        """
        Normalize a census occupation code to 4-digit string with leading zeros.
        CSV data may not include leading zeros; mapping keys are guaranteed 4 digits.
        """
        if code is None:
            return None
        s = str(code).strip()
        if s == "":
            return None
        try:
            i = int(float(s))
        except Exception:
            # already a string code?
            digits = re.sub(r"\D", "", s)
            if digits == "":
                return None
            i = int(digits)
        if i <= 0:
            return None
        return str(i).zfill(4)

    @classmethod
    def census2010_to_soc2010(cls, census_code: Any) -> Tuple[Optional[str], Optional[str]]:
        """
        Map a 2010 Census occupation code to SOC2010 code and occupation_title.
        If multiple SOC codes exist for one census_code, return the first.
        """
        cc4 = cls._normalize_census_code_4(census_code)
        if not cc4:
            return None, None
        rows = (cls._CENSUS2010_TO_SOC2010 or {}).get(cc4)
        if not rows:
            return None, None
        soc, title = rows[0]
        return soc, title or None

    # -------------------------------------------------------------------------
    # Dataset-backed profile loading
    # (File I/O is intentionally moved out of Household; see `agent_loader.py`.)
    # -------------------------------------------------------------------------

    @staticmethod
    @lru_cache(maxsize=4096)
    def _parse_value_range(value_range: str) -> Optional[Tuple[int, int]]:
        """
        Parse codebook "value_range" into an inclusive integer interval.

        Examples:
        - "1" -> (1, 1)
        - "1 - 20" -> (1, 20)
        - "1,000 - 1,240" -> (1000, 1240)
        - "-999,997 - -1" -> (-999997, -1)
        - "9,999" -> (9999, 9999)
        """
        s = (value_range or "").strip()
        if not s:
            return None
        # Extract integers (supports commas and negative sign)
        nums = re.findall(r"-?\d[\d,]*", s)
        if not nums:
            return None
        try:
            vals = [int(n.replace(",", "")) for n in nums]
        except Exception:
            return None
        if len(vals) == 1:
            return (vals[0], vals[0])
        # For strings that contain multiple numbers, treat the first two as interval bounds.
        lo, hi = vals[0], vals[1]
        if lo > hi:
            lo, hi = hi, lo
        return (lo, hi)

    @classmethod
    def _decode_codebook_value(cls, var: str, raw_value: Any) -> Optional[str]:
        """
        Decode a categorical code using codebook distributions when possible.
        """
        if cls._CODEBOOK_BY_VAR is None:
            return None
        if var in cls.NO_DECODE_FIELDS:
            return None
        meta = cls._CODEBOOK_BY_VAR.get(var)
        if not meta:
            return None
        # Only decode fields marked categorical in codebook.json, to avoid mapping continuous values
        if meta.get("field_type") not in (None, "categorical"):
            return None
        # Convert raw to int when possible
        try:
            code = int(float(raw_value))
        except Exception:
            return None
        for d in meta.get("distribution", []) or []:
            interval = cls._parse_value_range(str(d.get("value_range", "")))
            if interval is None:
                continue
            lo, hi = interval
            if lo <= code <= hi:
                return str(d.get("text") or "")
        return None

    @classmethod
    def decode_field_value(cls, field_name: str, value: Any) -> Optional[str]:
        """
        Convert a stored code/value into a human-readable string.

        Policy:
        - We store numeric codes in csv_values / member variables.
        - For ERxxxx categorical fields, use codebook distribution (only exact single-code mappings are supported).
        - For continuous values or unmapped codes, return None (caller can fallback to str(value)).
        """
        if value is None:
            return None
        fname = str(field_name)
        if not fname.startswith("ER"):
            return None
        if fname in cls.NO_DECODE_FIELDS:
            return None
        # Reuse existing decoder but pass in the numeric code (not raw string).
        return cls._decode_codebook_value(fname, value)

    def get_field_text(self, field_name: str) -> str:
        """
        Prompt-facing value: prefer decoded category text, fallback to stored code/value.
        """
        # Occupation special-case: prefer fine-grained occupation_title, not codebook big category.
        if field_name == "ER82181":
            if self.csv_values.get("ER82433") == self._NOT_EMPLOYED_CODE:
                return "Not working"
            title = getattr(self, "ER82181_occupation_title", None)
            soc = getattr(self, "ER82181", None)
            if title:
                return str(title)
            return str(soc)
        if field_name == "ER82500":
            if self.csv_values.get("SP_employment_status") == self._NOT_EMPLOYED_CODE:
                return "Not working"
            title = getattr(self, "ER82500_occupation_title", None)
            soc = getattr(self, "ER82500", None)
            if title:
                return str(title)
            return str(soc)
        v = self.csv_values.get(field_name, getattr(self, field_name, None))
        decoded = self.decode_field_value(field_name, v)
        return decoded if decoded else str(v)

    def get_persona_prompt_view(self) -> Dict[str, Any]:
        """
        Prompt-facing persona view: only keep selected fields to reduce noise.
        """
        p = self.persona if isinstance(self.persona, dict) else {}
        out: Dict[str, Any] = {k: p.get(k) for k in self.PERSONA_PROMPT_FIELDS if k in p}
        if "persona_name" not in out:
            pname = self.csv_values.get("persona_name_current") or self.csv_values.get("persona_name")
            out["persona_name"] = pname
        return out

    @classmethod
    def _parse_csv_value(cls, field_name: str, raw_value: Any) -> Any:
        """
        Parse value from CSV based on codebook.json field_type when available.
        Falls back to numeric parsing when possible.
        """
        if raw_value is None:
            return None
        s = str(raw_value).strip()
        if s == "":
            return None

        # booleans in mapping columns
        if field_name in {"is_validated"}:
            if s.lower() in {"true", "1"}:
                return True
            if s.lower() in {"false", "0"}:
                return False
            return s

        meta = (cls._CODEBOOK_BY_VAR or {}).get(field_name) if cls._CODEBOOK_BY_VAR else None
        ftype = (meta or {}).get("field_type")

        if ftype == "continuous":
            try:
                return float(s)
            except Exception:
                return s

        if ftype == "categorical":
            # many categorical codes are numeric; store as int when possible
            try:
                v = float(s)
                if v.is_integer():
                    return int(v)
                return v
            except Exception:
                return s

        # fallback heuristic
        try:
            v = float(s)
            if v.is_integer():
                return int(v)
            return v
        except Exception:
            return s

    def _ingest_csv_row_as_members(self, row: Dict[str, Any]) -> None:
        """
        Requirement: treat each CSV column as a member variable.
        We also keep structured dicts: csv_raw / csv_values / csv_decoded.
        """
        self.csv_raw = dict(row or {})
        self.csv_values = {}
        self.csv_decoded = {}

        for k, raw_v in (row or {}).items():
            parsed = self._parse_csv_value(str(k), raw_v)
            self.csv_values[str(k)] = parsed
            # Make each CSV field a member variable
            setattr(self, str(k), parsed)
            # Optional cache for prompt/UI (not source of truth)
            decoded = self.decode_field_value(str(k), parsed)
            if decoded:
                self.csv_decoded[str(k)] = decoded

        # Convert annual flow variables to monthly values (/12) for monthly simulation prompts.
        for field in self.MONTHLY_FLOW_FIELDS:
            v = self.csv_values.get(field)
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            mv = fv / 12.0
            self.csv_values[field] = mv
            setattr(self, field, mv)

    def _normalize_household_lookup_keys(self) -> List[int]:
        """
        Try to map Household(household_id=...) to dataset row keys.
        Supports:
        - "123" -> 123 (fid or household_idx)
        - "household_123" -> 123
        """
        s = str(self.household_id)
        keys: List[int] = []
        if s.isdigit():
            keys.append(int(s))
        if "_" in s:
            tail = s.split("_")[-1]
            if tail.isdigit():
                keys.append(int(tail))
        return keys

    def _ensure_profile_loaded(
        self,
        *,
        persona_mapping_csv: str,
        codebook_json: str,
        personas_json: str,
        census2010_to_soc2010_csv: str,
        preloaded_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        data_dir = Path(self.data_dir)
        csv_path = str(data_dir / persona_mapping_csv)
        codebook_path = str(data_dir / codebook_json)
        personas_path = str(data_dir / personas_json)
        census_to_soc_path = str(data_dir / census2010_to_soc2010_csv)

        # Preferred: use injected/preloaded artifacts (no file I/O in Household).
        if preloaded_data:
            self.set_preloaded_data(
                household_row_by_household_idx=preloaded_data.get("household_row_by_household_idx"),
                household_row_by_fid=preloaded_data.get("household_row_by_fid"),
                codebook_by_var=preloaded_data.get("codebook_by_var"),
                persona_by_name=preloaded_data.get("persona_by_name"),
                census2010_to_soc2010=preloaded_data.get("census2010_to_soc2010"),
            )

        # Backward compatible: if caches are missing, load via class-external helpers.
        if (
            self._HOUSEHOLD_ROW_BY_HOUSEHOLD_IDX is None
            or self._HOUSEHOLD_ROW_BY_FID is None
            or self._CODEBOOK_BY_VAR is None
            or self._PERSONA_BY_NAME is None
            or self._CENSUS2010_TO_SOC2010 is None
        ):
            from agenteconomy.simulation.agent_loader import (
                load_census2010_to_soc2010,
                load_codebook_by_var,
                load_household_rows,
                load_personas_by_name,
            )

            by_idx, by_fid = load_household_rows(csv_path)
            self.set_preloaded_data(
                household_row_by_household_idx=by_idx,
                household_row_by_fid=by_fid,
                codebook_by_var=load_codebook_by_var(codebook_path),
                persona_by_name=load_personas_by_name(personas_path),
                census2010_to_soc2010=load_census2010_to_soc2010(census_to_soc_path),
            )

        keys = self._normalize_household_lookup_keys()
        row = None
        for k in keys:
            row = (self._HOUSEHOLD_ROW_BY_HOUSEHOLD_IDX or {}).get(k)
            if row is not None:
                break
            row = (self._HOUSEHOLD_ROW_BY_FID or {}).get(k)
            if row is not None:
                break
        self.profile_row = row

        if row:
            # Treat each CSV field as a member variable (parsed)
            self._ingest_csv_row_as_members(row)

            persona_name = row.get("persona_name_current") or row.get("persona_name")
            self.persona = (self._PERSONA_BY_NAME or {}).get(str(persona_name)) if persona_name else None

            # Occupation (RP) mapping: 2010 Census occupation code -> SOC2010 + fine-grained title
            census_code = self.csv_values.get("ER82181")
            soc, title = self.census2010_to_soc2010(census_code)
            # Save final occupation code as SOC code (as requested), while keeping original census code in a dedicated member
            setattr(self, "ER82181_census_code", self._normalize_census_code_4(census_code))
            if soc:
                setattr(self, "ER82181", soc)
                self.csv_values["ER82181"] = soc
            setattr(self, "ER82181_occupation_title", title)

            # Occupation (SP) mapping: 2010 Census occupation code -> SOC2010 + fine-grained title
            sp_census_code = self.csv_values.get("ER82500")
            sp_soc, sp_title = self.census2010_to_soc2010(sp_census_code)
            setattr(self, "ER82500_census_code", self._normalize_census_code_4(sp_census_code))
            if sp_soc:
                setattr(self, "ER82500", sp_soc)
                self.csv_values["ER82500"] = sp_soc
            setattr(self, "ER82500_occupation_title", sp_title)

            # Force employment status at init (requested)
            # RP: ER82433 -> not employed
            setattr(self, "ER82433", self._NOT_EMPLOYED_CODE)
            self.csv_values["ER82433"] = self._NOT_EMPLOYED_CODE
            self.csv_raw["ER82433"] = str(self._NOT_EMPLOYED_CODE)
            # SP: new field -> not employed
            setattr(self, "SP_employment_status", self._NOT_EMPLOYED_CODE)
            self.csv_values["SP_employment_status"] = self._NOT_EMPLOYED_CODE
            self.csv_raw["SP_employment_status"] = str(self._NOT_EMPLOYED_CODE)

            for k in self.SIMULATION_STATE_FIELDS:
                setattr(self, k, None)
                self.csv_values[k] = None

        # Build "past household status" (both structured and text)
        self.past_household_status = self._build_past_household_status()
        self.past_household_status_text = self.past_household_status.get("summary_text") or ""

        # Inject into household_info for downstream prompts (persona == description)
        if self.profile_row:
            self.household_info["persona"] = {
                "persona_name": self.profile_row.get("persona_name_current") or self.profile_row.get("persona_name"),
                "persona_id": self.profile_row.get("persona_id_current") or self.profile_row.get("persona_id"),
                "persona_record": self.persona,
            }
        else:
            self.household_info["persona"] = {"persona_name": None, "persona_id": None, "persona_record": None}
        self.household_info["past_household_status"] = self.past_household_status

    def _build_past_household_status(self) -> Dict[str, Any]:
        """
        Construct an extensible, human-readable view of the household's recent situation.
        In prompts, we describe this as: "Recent household situation: ...".
        """
        row = self.csv_values or {}
        if not row:
            return {"summary_text": "Recent household situation: no historical profile data (no match in persona mapping CSV)."}

        all_fields: Dict[str, Dict[str, Any]] = {}
        for k, raw_v in (row or {}).items():
            label = None
            if str(k) in self.PAST_STATUS_LABEL_OVERRIDES:
                label = self.PAST_STATUS_LABEL_OVERRIDES[str(k)]
            elif self._CODEBOOK_BY_VAR and k in self._CODEBOOK_BY_VAR:
                label = self._CODEBOOK_BY_VAR[k].get("label") or self._CODEBOOK_BY_VAR[k].get("name")
            decoded = self.decode_field_value(k, self.csv_values.get(str(k))) if str(k).startswith("ER") else None
            all_fields[str(k)] = {
                "raw": self.csv_raw.get(str(k)) if hasattr(self, "csv_raw") else raw_v,
                "value": self.csv_values.get(str(k)),
                "decoded": decoded,
                "label": label,
            }

        for k in sorted(self.SIMULATION_STATE_FIELDS):
            label = self.PAST_STATUS_LABEL_OVERRIDES.get(k)
            v = self.csv_values.get(k)
            all_fields[k] = {
                "raw": None,
                "value": v,
                "decoded": None,
                "label": label,
            }

        focus_vars = [
            "ER82017",
            "ER82027",
            "ER84520",
            "ER82800",
            "ER85629",
            "ER85666",
            "ER85692",
            "ER85701",
            "ER85747",
            "ER85768",
            "ER85780",
            "ER82181",
            "ER82500",
            "RP_income",
            "SP_income",
            "expenditure_retail_merchandise",
            "expenditure_insurance",
            "expenditure_utilities",
            "expenditure_transportation",
        ]

        facts: List[Dict[str, Any]] = []
        lines: List[str] = []
        for var in focus_vars:
            raw_val = row.get(var) if var in row else None
            label = None
            if var in self.PAST_STATUS_LABEL_OVERRIDES:
                label = self.PAST_STATUS_LABEL_OVERRIDES[var]
            elif self._CODEBOOK_BY_VAR and var in self._CODEBOOK_BY_VAR:
                label = self._CODEBOOK_BY_VAR[var].get("label") or self._CODEBOOK_BY_VAR[var].get("name")
            stored_v = self.csv_values.get(var)
            if var == "ER82181":
                value_view = self.get_field_text("ER82181")
                decoded = value_view
            elif var == "ER82500":
                value_view = self.get_field_text("ER82500")
                decoded = value_view
            else:
                decoded = self.decode_field_value(var, stored_v)
                value_view = decoded if decoded else stored_v
            if var not in self.SIMULATION_STATE_FIELDS:
                if raw_val is None or str(raw_val).strip() == "":
                    continue
            if value_view is None:
                value_view = "unknown"
            facts.append({"var": var, "label": label, "raw": raw_val, "value": stored_v, "decoded": decoded})
            display_label = label or var
            lines.append(f"- {display_label}: {value_view}")

        persona_name = row.get("persona_name_current") or row.get("persona_name")
        persona_core = None
        if self.persona:
            persona_core = self.persona.get("core_characteristics")

        total_assets = self.csv_values.get("ER85692")
        head_occ = self.csv_values.get("ER82181")
        spouse_occ = self.csv_values.get("ER82500")
        rp_employment_status = self.csv_values.get("ER82433")
        sp_employment_status = self.csv_values.get("SP_employment_status")
        rp_income = self.csv_values.get("RP_income")
        sp_income = self.csv_values.get("SP_income")
        exp_retail = self.csv_values.get("expenditure_retail_merchandise")
        exp_ins = self.csv_values.get("expenditure_insurance")
        exp_util = self.csv_values.get("expenditure_utilities")
        exp_trans = self.csv_values.get("expenditure_transportation")

        summary = "Recent household situation:\n"
        if persona_name:
            summary += f"- persona: {persona_name}\n"
        if persona_core:
            summary += f"- persona core characteristics: {persona_core}\n"
        summary += "\n".join(lines)

        return {
            "persona_name": persona_name,
            "derived": {
                "total_assets": total_assets,
                "head_occupation_code": head_occ,
                "spouse_occupation_code": spouse_occ,
                "rp_employment_status": rp_employment_status,
                "sp_employment_status": sp_employment_status,
                "rp_income": rp_income,
                "sp_income": sp_income,
                "expenditure_retail_merchandise": exp_retail,
                "expenditure_insurance": exp_ins,
                "expenditure_utilities": exp_util,
                "expenditure_transportation": exp_trans,
            },
            "all_fields": all_fields,
            "facts": facts,
            "summary_text": summary,
        }

    # -------------------------------------------------------------------------
    # Public state update APIs (requested)
    # -------------------------------------------------------------------------

    @staticmethod
    def _as_float(v: Any) -> float:
        try:
            return float(v)
        except Exception:
            return 0.0

    @staticmethod
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
                    "skills": skills if isinstance(skills, dict) else {},
                    "abilities": abilities if isinstance(abilities, dict) else {},
                }
        return data

    @classmethod
    def _mean_profile(cls, requirements: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        profile: Dict[str, float] = {}
        for name, meta in (requirements or {}).items():
            if not isinstance(meta, dict):
                continue
            mean = meta.get("mean")
            if mean is None:
                continue
            try:
                profile[name] = float(mean)
            except Exception:
                continue
        return profile

    @classmethod
    def _sample_profile(cls, requirements: Dict[str, Dict[str, Any]], rng: random.Random) -> Dict[str, float]:
        profile: Dict[str, float] = {}
        for name, meta in (requirements or {}).items():
            if not isinstance(meta, dict):
                continue
            mean = meta.get("mean")
            std = meta.get("std")
            if mean is None:
                continue
            try:
                mean_v = float(mean)
            except Exception:
                continue
            try:
                std_v = float(std) if std is not None else 0.0
            except Exception:
                std_v = 0.0
            if std_v > 0.0:
                value = rng.gauss(mean_v, std_v)
            else:
                value = mean_v
            profile[name] = float(value)
        return profile

    def _build_labor_hour_from_soc(self, soc: Optional[str], lh_type: str, total_hours: float) -> Optional[LaborHour]:
        if not soc:
            return None
        data = self._load_job_skill_data().get(str(soc))
        rng = random.Random()
        skill_profile = self._sample_profile(data.get("skills") or {}, rng) if data else {}
        ability_profile = self._sample_profile(data.get("abilities") or {}, rng) if data else {}
        labor_hour = LaborHour.create(
            agent_id=self.household_id,
            total_hours=float(total_hours),
            template=f"labor_{lh_type}",
            skill_profile=skill_profile,
            ability_profile=ability_profile,
            lh_type=lh_type,
        )
        labor_hour.job_SOC = str(soc)
        labor_hour.job_title = (data.get("title") if data else None) or labor_hour.job_title
        return labor_hour

    def build_labor_hours(self, total_hours: float = 160.0) -> List[LaborHour]:
        labor_hours: List[LaborHour] = []
        head_soc = self.get_rp_soc_occupation_code()
        spouse_soc = self.get_sp_soc_occupation_code()
        head = self._build_labor_hour_from_soc(head_soc, "head", total_hours)
        if head is not None:
            labor_hours.append(head)
        spouse = self._build_labor_hour_from_soc(spouse_soc, "spouse", total_hours)
        if spouse is not None:
            labor_hours.append(spouse)
        self.labor_hours = labor_hours
        return labor_hours

    def get_rp_soc_occupation_code(self) -> Optional[str]:
        """
        Public getter: RP (head) SOC occupation code.
        Note: stored in csv_values["ER82181"] after census2010->SOC mapping.
        """
        v = self.csv_values.get("ER82181")
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    def get_sp_soc_occupation_code(self) -> Optional[str]:
        """
        Public getter: SP (spouse) SOC occupation code.
        Note: stored in csv_values["ER82500"] after census2010->SOC mapping.
        """
        v = self.csv_values.get("ER82500")
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    def update_total_assets(self, delta: float) -> float:
        """
        Update total assets by delta (can be positive or negative).
        Returns the updated total assets.
        """
        current = self.csv_values.get("ER85692")
        try:
            cur_v = float(current or 0.0)
        except Exception:
            cur_v = 0.0
        new_v = cur_v + float(delta or 0.0)

        # update member variable + dicts
        setattr(self, "ER85692", new_v)
        self.csv_values["ER85692"] = new_v
        return new_v

    def update_rp_income(self, wage: float) -> float:
        """
        Record RP wage payment for this month.
        """
        w = float(wage or 0.0)
        new_rp = w
        setattr(self, "RP_income", new_rp)
        self.csv_values["RP_income"] = new_rp

        self.csv_values["ER85629"] = self._as_float(self.csv_values.get("ER85629")) + w
        setattr(self, "ER85629", self.csv_values["ER85629"])

        self.csv_values["ER85692"] = self._as_float(self.csv_values.get("ER85692")) + w
        setattr(self, "ER85692", self.csv_values["ER85692"])
        return new_rp

    def update_sp_income(self, wage: float) -> float:
        """
        Record SP wage payment for this month.
        """
        w = float(wage or 0.0)
        new_sp = w
        setattr(self, "SP_income", new_sp)
        self.csv_values["SP_income"] = new_sp

        self.csv_values["ER85629"] = self._as_float(self.csv_values.get("ER85629")) + w
        setattr(self, "ER85629", self.csv_values["ER85629"])

        self.csv_values["ER85692"] = self._as_float(self.csv_values.get("ER85692")) + w
        setattr(self, "ER85692", self.csv_values["ER85692"])
        return new_sp

    def apply_consumption(self, spending_by_bucket: Dict[str, float]) -> Dict[str, float]:
        """
        Set monthly consumption spending by bucket (absolute totals) and propagate to totals.
        """
        total_spend = 0.0
        updated: Dict[str, float] = {}

        bucket_to_field: Dict[str, str] = {
            "Retail merchandise": "expenditure_retail_merchandise",
            "insurance": "expenditure_insurance",
            "utilities": "expenditure_utilities",
            "transportation": "expenditure_transportation",
            "housing": "ER85701",
            "healthcare": "ER85747",
        }

        for bucket, new_total in (spending_by_bucket or {}).items():
            if bucket not in bucket_to_field:
                continue
            field = bucket_to_field[bucket]
            new_v = float(new_total or 0.0)
            self.csv_values[field] = new_v
            setattr(self, field, new_v)
            updated[field] = new_v
            total_spend += new_v

        self.csv_values["ER85768"] = float(total_spend)
        setattr(self, "ER85768", self.csv_values["ER85768"])
        updated["ER85768"] = self.csv_values["ER85768"]

        self.csv_values["ER85692"] = self._as_float(self.csv_values.get("ER85692")) - total_spend
        setattr(self, "ER85692", self.csv_values["ER85692"])
        updated["ER85692"] = self.csv_values["ER85692"]
        return updated

    def update_head_occupation(self, new_occupation_code: str) -> None:
        """
        Update head's occupation based on 2010 Census occupation code.
        The stored final code will be SOC2010 (as requested).
        """
        census = self._normalize_census_code_4(new_occupation_code)
        soc, title = self.census2010_to_soc2010(census)
        setattr(self, "ER82181_census_code", census)
        if soc:
            setattr(self, "ER82181", soc)
            self.csv_values["ER82181"] = soc
            # If occupation is set, mark employed (requested)
            setattr(self, "ER82433", self._EMPLOYED_CODE)
            self.csv_values["ER82433"] = self._EMPLOYED_CODE
        else:
            setattr(self, "ER82181", None)
            self.csv_values["ER82181"] = None
            setattr(self, "ER82433", self._NOT_EMPLOYED_CODE)
            self.csv_values["ER82433"] = self._NOT_EMPLOYED_CODE
        setattr(self, "ER82181_occupation_title", title)

    def update_spouse_occupation(self, new_occupation_code: str) -> None:
        """
        Update spouse occupation based on 2010 Census occupation code.
        The stored final code will be SOC2010 (as requested).
        """
        census = self._normalize_census_code_4(new_occupation_code)
        soc, title = self.census2010_to_soc2010(census)
        setattr(self, "ER82500_census_code", census)
        setattr(self, "ER82500", soc)
        self.csv_values["ER82500"] = soc
        setattr(self, "ER82500_occupation_title", title)
        # If occupation is set, mark employed (requested)
        if soc:
            setattr(self, "SP_employment_status", self._EMPLOYED_CODE)
            self.csv_values["SP_employment_status"] = self._EMPLOYED_CODE
        else:
            setattr(self, "SP_employment_status", self._NOT_EMPLOYED_CODE)
            self.csv_values["SP_employment_status"] = self._NOT_EMPLOYED_CODE

    @staticmethod
    def _deep_merge_dict(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep-merge a JSON patch dict into base dict. Nested dicts are merged recursively.
        """
        out = dict(base or {})
        for k, v in (patch or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = Household._deep_merge_dict(out.get(k) or {}, v)
            else:
                out[k] = v
        return out

    async def refresh_past_household_status_text_and_persona(self) -> Dict[str, Any]:
        """
        Rebuild past_household_status_text from the CURRENT internal variables, then ask the LLM
        whether the persona should be updated given the changes.
        """
        old_status_text = str(self.past_household_status_text or "")
        persona_before = copy.deepcopy(self.persona) if isinstance(self.persona, dict) else None

        self.past_household_status_text = (self._build_past_household_status() or {}).get("summary_text") or ""

        prompt = PERSONA_UPDATE_PROMPT.format(
            persona_before=json.dumps(self.get_persona_prompt_view() if persona_before else {}, ensure_ascii=False),
            past_household_status_text_before=old_status_text,
            past_household_status_text_after=self.past_household_status_text,
        )
        raw = await self._llm_chat(system="Return strict JSON only.", user=prompt, temperature=0.2)
        parsed = self._json_loads_loose(raw) or {}
        persona_patch = parsed.get("persona_patch") or {}

        if isinstance(persona_before, dict):
            self.persona = self._deep_merge_dict(persona_before, persona_patch if isinstance(persona_patch, dict) else {})
        else:
            self.persona = persona_patch if isinstance(persona_patch, dict) and persona_patch else self.persona

        if "persona" not in self.household_info:
            self.household_info["persona"] = {"persona_name": None, "persona_id": None, "persona_record": None}
        self.household_info["persona"]["persona_record"] = self.persona
        if isinstance(persona_patch, dict) and "persona_name" in persona_patch:
            self.household_info["persona"]["persona_name"] = persona_patch.get("persona_name")

        result = {
            "persona_patch": persona_patch if isinstance(persona_patch, dict) else {},
            "note": str(parsed.get("note") or ""),
            "raw_llm_output": raw,
        }

        for k in self.MONTHLY_RESET_FIELDS:
            self.csv_values[k] = 0.0
            setattr(self, k, 0.0)

        return result

    # -------------------------------------------------------------------------
    # Dependency wiring
    # -------------------------------------------------------------------------

    def set_dependencies(self, *, economic_center=None, product_market=None, labor_market=None):
        self.economic_center = economic_center
        self.product_market = product_market
        self.labor_market = labor_market

    def _call_market_method(self, market, method_name: str, *args, **kwargs):
        if market is None:
            return None
        method = getattr(market, method_name, None)
        if method is None:
            return None
        if hasattr(method, "remote"):
            return ray.get(method.remote(*args, **kwargs))
        return method(*args, **kwargs)

    def _call_economic_center(self, method_name: str, *args, **kwargs):
        return self._call_market_method(self.economic_center, method_name, *args, **kwargs)

    def _call_product_market(self, method_name: str, *args, **kwargs):
        return self._call_market_method(self.product_market, method_name, *args, **kwargs)

    def _call_labor_market(self, method_name: str, *args, **kwargs):
        return self._call_market_method(self.labor_market, method_name, *args, **kwargs)

    def initialize_in_system(
        self,
        *,
        economic_center=None,
        labor_market=None,
        product_market=None,
        total_hours: float = 160.0,
    ) -> List[LaborHour]:
        if economic_center is not None:
            self.economic_center = economic_center
        if labor_market is not None:
            self.labor_market = labor_market
        if product_market is not None:
            self.product_market = product_market

        if self.economic_center is not None:
            self._call_economic_center("register_id", self.household_id, "household")
            savings = self._as_float(self.csv_values.get("ER85692"))
            self._call_economic_center("init_agent_ledger", self.household_id, savings)

        if not self.labor_hours:
            self.build_labor_hours(total_hours=total_hours)
        if self.labor_market is not None and self.labor_hours:
            self._call_labor_market("register_labor_hours", self.labor_hours)

        return list(self.labor_hours)

    def set_household_info(self, info: Dict[str, Any]):
        self.household_info = dict(info or {})
        self.household_info.setdefault("household_id", self.household_id)
        self.household_info.setdefault("name", self.name)

    # -------------------------------------------------------------------------
    # Product info helpers (placeholders for now)
    # -------------------------------------------------------------------------

    def get_product_price(self, product_id: str) -> float:
        """
        Placeholder: query current price by product_id.
        For now returns a default value; later you can wire it to EconomicCenter/ProductMarket.
        """
        snapshot = self._call_product_market("get_product_snapshot", product_id)
        if isinstance(snapshot, dict):
            price = snapshot.get("retail_price")
            if price is not None:
                return float(price)
        return 1.0

    def get_product_stock(self, product_id: str) -> float:
        """
        Placeholder: query current stock by product_id.
        For now returns a default value; later you can wire it to EconomicCenter/ProductMarket.
        """
        snapshot = self._call_product_market("get_product_snapshot", product_id)
        if isinstance(snapshot, dict):
            stock = snapshot.get("available_stock")
            if stock is not None:
                return float(stock)
        return 100.0

    @staticmethod
    def _format_product_line(
        *,
        product_id: str,
        name: Optional[str],
        description: Optional[str],
        current_price: float,
        storage: float,
    ) -> str:
        """
        Required product description format for purchase prompts:
        id + name + description + current_price + storage
        """
        return (
            f"id={product_id} | name={name or ''} | description={description or ''} | "
            f"current_price={current_price} | storage={storage}"
        )

    # -------------------------------------------------------------------------
    # LLM helper (minimal)
    # -------------------------------------------------------------------------

    async def _llm_chat(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        # temperature is currently ignored by agenteconomy.llm.llm.call_llm; kept for extensibility.
        content = await call_llm(prompt=user, system_prompt=system, model_type=self.llm_model_type)
        return (content or "").strip()

    @staticmethod
    def _json_loads_loose(text: str) -> Any:
        """
        Parse JSON from LLM output that may be wrapped in markdown fences like:
          ```json
          {...}
          ```
        Also handles common LLM JSON errors like trailing commas.
        """
        import re
        
        s = (text or "").strip()
        if s.startswith("```"):
            # strip leading fence line
            lines = s.splitlines()
            if lines:
                lines = lines[1:]
            # strip trailing fence
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            s = "\n".join(lines).strip()
            # if first token is "json", drop it
            if s.lower().startswith("json"):
                s = s[4:].strip()
        
        def try_parse(json_str: str) -> Any:
            """Try to parse JSON, fixing common errors."""
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Fix trailing commas before } or ]
                fixed = re.sub(r',\s*([}\]])', r'\1', json_str)
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    # Fix missing commas between values (e.g., "key": value\n"key2")
                    fixed2 = re.sub(r'(\d+|"[^"]*"|true|false|null)\s*\n\s*"', r'\1,\n"', fixed)
                    return json.loads(fixed2)
        
        try:
            return try_parse(s)
        except Exception:
            # fallback: extract first JSON object/array substring
            start_obj = s.find("{")
            start_arr = s.find("[")
            if start_obj == -1 and start_arr == -1:
                raise
            start = start_obj if (start_obj != -1 and (start_arr == -1 or start_obj < start_arr)) else start_arr
            end = s.rfind("}") if start == start_obj else s.rfind("]")
            if end == -1:
                raise
            return try_parse(s[start : end + 1])

    # -------------------------------------------------------------------------
    # Consumption decision (NEW workflow): step1-4
    # -------------------------------------------------------------------------

    async def consumption_step1_needs_by_category(
        self,
        *,
        total_budget: float,
        categories: Optional[List[str]] = None,
        available_balance: Optional[float] = None,
        expected_income: Optional[float] = None,
        available_budget: Optional[float] = None,
    ) -> CategoryNeedsOutput:
        """
        Step1 (default LLM):
        For each category name, allocate a budget and provide multiple need descriptions.
        """
        cats = categories or self.consumption_categories

        llm_start = time.perf_counter()
        prompt = CONSUMPTION_NEEDS_BY_CATEGORY_PROMPT.format(
            persona=json.dumps(self.get_persona_prompt_view(), ensure_ascii=False),
            past_household_status=self.past_household_status_text,
            total_budget=json.dumps(float(total_budget), ensure_ascii=False),
            categories=json.dumps(list(cats), ensure_ascii=False),
            available_balance=json.dumps(available_balance, ensure_ascii=False),
            expected_income=json.dumps(expected_income, ensure_ascii=False),
            available_budget=json.dumps(available_budget, ensure_ascii=False),
        )
        raw = await self._llm_chat(system="Return strict JSON only.", user=prompt, temperature=0.2)
        llm_elapsed = time.perf_counter() - llm_start
        parsed = self._json_loads_loose(raw)

        # Normalize: ensure all categories exist, budgets sum to total_budget, and each has >=1 description.
        incoming: Dict[str, CategoryPlan] = {}
        for rec in (parsed.get("category_plans") or []):
            cat = str(rec.get("category") or "").strip()
            if not cat:
                continue
            if cat not in cats:
                continue
            incoming[cat] = CategoryPlan(
                category=cat,
                budget_amount=float(rec.get("budget_amount") or 0.0),
                need_descriptions=[str(x) for x in (rec.get("need_descriptions") or [])],
            )

        plans: List[CategoryPlan] = []
        for cat in cats:
            cp = incoming.get(cat) or CategoryPlan(category=cat, budget_amount=0.0, need_descriptions=[])
            if not cp.need_descriptions:
                cp.need_descriptions = [f"General needs for {cat} (be specific in future iterations)."]
            plans.append(cp)

        # Budget normalization to match total_budget (scale all non-negative budgets)
        tb = float(total_budget or 0.0)
        if tb >= 0:
            total_alloc = sum(max(0.0, float(cp.budget_amount or 0.0)) for cp in plans)
            if total_alloc <= 0:
                # If model returned zeros, allocate evenly.
                per = tb / max(1, len(plans))
                for cp in plans:
                    cp.budget_amount = per
            else:
                scale = tb / total_alloc
                for cp in plans:
                    cp.budget_amount = max(0.0, float(cp.budget_amount or 0.0)) * scale


        return CategoryNeedsOutput(
            total_budget=float(tb),
            category_plans=plans,
            note=str(parsed.get("note") or ""),
            raw_llm_output=raw,
        )

    async def consumption_step0_major_budget_allocation(
        self,
        *,
        available_balance: Optional[float] = None,
        expected_income: Optional[float] = None,
        available_budget: Optional[float] = None,
    ) -> MajorBudgetOutput:
        """
        Step0 (default LLM):
        Allocate major budget buckets. Retail merchandise budget will be used as step1 total_budget.
        """
        prompt = CONSUMPTION_MAJOR_BUDGET_PROMPT.format(
            persona=json.dumps(self.get_persona_prompt_view(), ensure_ascii=False),
            past_household_status=self.past_household_status_text,
            available_balance=json.dumps(available_balance, ensure_ascii=False),
            expected_income=json.dumps(expected_income, ensure_ascii=False),
            available_budget=json.dumps(available_budget, ensure_ascii=False),
        )
        raw = await self._llm_chat(system="Return strict JSON only.", user=prompt, temperature=0.2)
        parsed = self._json_loads_loose(raw)
        total_budget = float(parsed.get("total_budget") or 0.0)
        budgets = dict(parsed.get("budgets") or {})

        # Ensure required keys exist and normalize sums to total_budget
        keys = [
            "Retail merchandise",
            "housing",
            "healthcare",
            "transportation",
            "utilities",
            "insurance",
        ]
        norm: Dict[str, float] = {k: float(budgets.get(k) or 0.0) for k in keys}
        s = sum(max(0.0, v) for v in norm.values())
        if total_budget >= 0:
            if s <= 0:
                per = total_budget / max(1, len(keys))
                norm = {k: per for k in keys}
            else:
                scale = total_budget / s
                norm = {k: max(0.0, v) * scale for k, v in norm.items()}

        return MajorBudgetOutput(
            total_budget=float(total_budget),
            budgets=norm,
            note=str(parsed.get("note") or ""),
            raw_llm_output=raw,
        )

    def _get_qdrant_client(self):
        from agenteconomy.utils.load_qdrant_client import load_client

        return load_client()

    def _consumption_step2_vector_match_sync(
        self,
        *,
        category_plans: List[CategoryPlan],
        top_k: int = 10,
        product_market=None,
    ) -> Dict[str, Dict[str, Any]]:
        """Sync helper for step2 vector search."""
        out: Dict[str, Dict[str, Any]] = {}
        market = product_market or self.product_market
        if market is None:
            logger.warning(f"[消费检索] {self.household_id} no_product_market")
            return out

        search_method = getattr(market, "search_by_vector", None)
        is_actor = hasattr(search_method, "remote")

        for cp in category_plans:
            cat = cp.category
            cat_candidates: List[Dict[str, Any]] = []
            need_descs = list(cp.need_descriptions or [])

            products_by_desc: List[Tuple[str, List[Any]]] = []
            if is_actor:
                futures = []
                for desc in need_descs:
                    query = f"{cat}: {desc}"
                    futures.append((desc, search_method.remote(query, top_k=int(top_k))))
                if futures:
                    results = ray.get([fut for _, fut in futures])
                    for (desc, _), products_raw in zip(futures, results):
                        products_by_desc.append((desc, list(products_raw or [])))
            else:
                for desc in need_descs:
                    query = f"{cat}: {desc}"
                    products_raw = self._call_market_method(market, "search_by_vector", query, top_k=int(top_k))
                    products_by_desc.append((desc, list(products_raw or [])))

            for desc, products in products_by_desc:
                for product in products:
                    product_id = getattr(product, "product_id", None)
                    if not product_id:
                        continue
                    current_price = float(
                        getattr(product, "retail_price", None)
                        or self.get_product_price(str(product_id))
                    )
                    storage = float(
                        getattr(product, "available_stock", None)
                        or self.get_product_stock(str(product_id))
                    )
                    product_line = self._format_product_line(
                        product_id=str(product_id),
                        name=getattr(product, "name", None),
                        description=getattr(product, "description", None),
                        current_price=current_price,
                        storage=storage,
                    )
                    cat_candidates.append(
                        {
                            "category": cat,
                            "need_description": desc,
                            "product_id": str(product_id),
                            "name": getattr(product, "name", None),
                            "description": getattr(product, "description", None),
                            "current_price": current_price,
                            "storage": storage,
                            "product_line": product_line,
                            "score": None,
                        }
                    )
            out[cat] = {
                "category": cat,
                "budget_amount": float(cp.budget_amount or 0.0),
                "need_descriptions": list(cp.need_descriptions or []),
                "candidates": cat_candidates,
            }

        return out

    async def consumption_step2_vector_match(
        self,
        *,
        category_plans: List[CategoryPlan],
        top_k: int = 10,
        product_market=None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Step2:
        For each category, for each need description, retrieve top_k products via ProductMarket
        when available (preferred), otherwise fallback to Qdrant.
        """
        return await asyncio.to_thread(
            self._consumption_step2_vector_match_sync,
            category_plans=category_plans,
            top_k=top_k,
            product_market=product_market,
        )

    async def consumption_step3_purchase_llm(
        self,
        *,
        category_bundles: Dict[str, Dict[str, Any]],
    ) -> BudgetedPurchasePlan:
        """
        Step3:
        Call LLM once per category. Each call chooses product_ids from that category's candidate list
        and allocates per-product budgets. Then we merge all categories into one purchase plan.
        """
        purchases: List[BudgetedPurchase] = []
        notes: List[str] = []
        raw_map: Dict[str, str] = {}

        # Deterministic order for easier debugging
        for cat in sorted(category_bundles.keys()):
            bundle = category_bundles.get(cat) or {}
            cat_budget = float(bundle.get("budget_amount") or 0.0)
            need_descs = list(bundle.get("need_descriptions") or [])
            candidates = list(bundle.get("candidates") or [])

            prompt = PURCHASE_BY_CATEGORY_PROMPT.format(
                persona=json.dumps(self.get_persona_prompt_view(), ensure_ascii=False),
                past_household_status=self.past_household_status_text,
                category=json.dumps(cat, ensure_ascii=False),
                category_budget=json.dumps(cat_budget, ensure_ascii=False),
                need_descriptions=json.dumps(need_descs, ensure_ascii=False),
                candidates=json.dumps(candidates, ensure_ascii=False),
            )
            raw = await self._llm_chat(system="Return strict JSON only.", user=prompt, temperature=0.2)
            raw_map[cat] = raw
            parsed = self._json_loads_loose(raw)
            notes.append(f"{cat}: {str(parsed.get('note') or '').strip()}")
            recs = list(parsed.get("purchases") or [])
            shares: List[float] = []
            for rec in recs:
                share = float(rec.get("budget_share") or 0.0)
                if share < 0.0:
                    share = 0.0
                if share > 1.0:
                    share = 1.0
                shares.append(share)

            # If shares sum to > 1, scale down so total spend does not exceed category budget.
            s = sum(shares)
            if s > 1e-9 and s > 1.0:
                shares = [x / s for x in shares]

            for rec, share in zip(recs, shares):
                purchases.append(
                    BudgetedPurchase(
                        category=cat,
                        product_id=str(rec.get("product_id") or ""),
                        allocated_budget=float(cat_budget) * float(share),
                        reason=str(rec.get("reason") or ""),
                    )
                )

        return BudgetedPurchasePlan(purchases=purchases, note=" | ".join([n for n in notes if n]), raw_llm_output=json.dumps(raw_map))

    def consumption_step4_validate(self, *args, **kwargs) -> bool:
        """
        Step4 (placeholder): validate purchase reasonableness. Not implemented yet.
        """
        return True

    async def consume_v2(
        self,
        *,
        top_k: int = 10,
        product_market=None,
        available_balance: Optional[float] = None,
        expected_income: Optional[float] = None,
        available_budget: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        New end-to-end consumption flow (step1-4).
        Step4 is stubbed.
        """

        avail_balance = None if available_balance is None else float(available_balance)
        exp_income = None if expected_income is None else float(expected_income)
        avail_budget = available_budget
        if avail_budget is None:
            if avail_balance is not None:
                base = float(avail_balance)
                if exp_income is not None and exp_income > 0.0:
                    base += float(exp_income)
                avail_budget = base
        if avail_budget is not None:
            avail_budget = max(0.0, float(avail_budget))

        step0 = await self.consumption_step0_major_budget_allocation(
            available_balance=avail_balance,
            expected_income=exp_income,
            available_budget=avail_budget,
        )
        if avail_budget is not None:
            if step0.total_budget > avail_budget and step0.total_budget > 0:
                scale = avail_budget / float(step0.total_budget)
                step0.budgets = {k: float(v) * scale for k, v in (step0.budgets or {}).items()}
                step0.total_budget = float(avail_budget)
            elif avail_budget <= 0.0:
                step0.budgets = {k: 0.0 for k in (step0.budgets or {}).keys()}
                step0.total_budget = 0.0
        retail_budget = float((step0.budgets or {}).get("Retail merchandise") or 0.0)
        step1 = await self.consumption_step1_needs_by_category(
            total_budget=retail_budget,
            available_balance=avail_balance,
            expected_income=exp_income,
            available_budget=avail_budget,
        )
        step2 = await self.consumption_step2_vector_match(
            category_plans=step1.category_plans, top_k=top_k, product_market=product_market
        )
        step3 = await self.consumption_step3_purchase_llm(category_bundles=step2)
        # _ = self.consumption_step4_validate(step1, step2, step3)

        return {
            "step0": {
                "total_budget": step0.total_budget,
                "budgets": step0.budgets,
                "note": step0.note,
                "available_balance": avail_balance,
                "expected_income": exp_income,
                "available_budget": avail_budget,
            },
            "step1": {
                "total_budget": step1.total_budget,
                "category_plans": [cp.__dict__ for cp in step1.category_plans],
                "note": step1.note,
            },
            "step2": {"category_bundles": step2},
            "step3": {"purchases": [p.__dict__ for p in step3.purchases], "note": step3.note},
        }

    # -------------------------------------------------------------------------
    # Job matching workflow
    # -------------------------------------------------------------------------

    def list_job_seekers(self, labor_hours: Sequence[LaborHour]) -> List[LaborHour]:
        """
        Step: household gives the list of individuals seeking jobs.
        Current version alignment:
        - The simulation should provide each household's labor supplies (RP/SP) as LaborHour objects.
        - This function only filters the provided list.
        - We do NOT construct default RP/SP labor hours inside Household.
        Rule (requested): only non-employed RP/SP will seek jobs.
        - RP employment status: ER82433 (1=employed, 5=not employed)
        - SP employment status: SP_employment_status (1=employed, 5=not employed)
        """
        seekers: List[LaborHour] = []
        for lh in labor_hours or []:
            if not getattr(lh, "is_valid", True):
                continue
            if getattr(lh, "firm_id", None):
                continue
            lh_type = getattr(lh, "lh_type", None)
            if lh_type == "head":
                if self.csv_values.get("ER82433") != self._NOT_EMPLOYED_CODE:
                    continue
            elif lh_type == "spouse":
                if self.csv_values.get("SP_employment_status") != self._NOT_EMPLOYED_CODE:
                    continue
            seekers.append(lh)
        return seekers

    def match_jobs_topk_by_loss(self, *, labor_hour: LaborHour, jobs: Sequence[Job], top_k: int = 3) -> List[JobMatch]:
        """
        Step: use loss to match topk jobs (rule-based).
        This is the skill/ability-based matching interface:
        - Caller must provide candidate jobs (e.g., already filtered by SOC elsewhere).
        - Loss is computed from LaborHour.skill_profile/ability_profile vs Job.required_skills/required_abilities.
        """
        if self.labor_market is not None:
            method = getattr(self.labor_market, "rank_jobs_for_labor", None)
            if method is None:
                ranked = []
            elif hasattr(method, "remote"):
                ranked = ray.get(method.remote(labor_hour, loss_threshold=float("inf")))
            else:
                ranked = method(labor_hour, loss_threshold=float("inf"))
            job_set = {j.job_id for j in (jobs or [])}
            matches = [JobMatch(job=j, loss=loss) for j, loss in ranked if j.job_id in job_set]
            return matches[: max(1, int(top_k))]

        ## Todo: SOC->Skill mapping
        if getattr(labor_hour, "skill_profile", None) is None or getattr(labor_hour, "ability_profile", None) is None:
            return []

        matches: List[JobMatch] = []
        for job in jobs or []:
            if not getattr(job, "is_valid", True):
                continue
            if int(getattr(job, "positions_available", 0) or 0) <= 0:
                continue
            matches.append(JobMatch(job=job, loss=0.0))
        return matches[: max(1, int(top_k))]

    async def decide_job_applications(
        self,
        *,
        month: int,
        labor_hour: LaborHour,
        top_matches: Sequence[JobMatch],
        use_llm: bool = True,
    ) -> List[JobApplication]:
        """
        Step: member chooses some jobs in topk to apply, or apply to none (LLM).
        Output: list of JobApplication objects for LaborMarket / firm-side processing.
        """
        if not top_matches:
            return []
        if not use_llm:
            # minimal rule: apply to the best one
            j0 = top_matches[0].job
            return [
                JobApplication.create(
                    job_id=j0.job_id,
                    household_id=self.household_id,
                    lh_type=labor_hour.lh_type,
                    expected_wage=float(getattr(j0, "wage_per_hour", 0.0) or 0.0),
                    worker_skills=dict(labor_hour.skill_profile or {}),
                    worker_abilities=dict(labor_hour.ability_profile or {}),
                    month=month,
                )
            ]

        jobs_view = [
            {
                "job_id": m.job.job_id,
                "firm_id": getattr(m.job, "firm_id", None),
                "title": getattr(m.job, "title", None),
                "soc": getattr(m.job, "SOC", None),
                "wage_per_hour": getattr(m.job, "wage_per_hour", None),
                "loss": m.loss,
            }
            for m in top_matches
        ]
        prompt = JOB_APPLICATION_DECISION_PROMPT.format(
            persona=json.dumps(self.get_persona_prompt_view(), ensure_ascii=False),
            past_household_status=self.past_household_status_text,
            member_info=json.dumps(
                {
                    "lh_type": getattr(labor_hour, "lh_type", None),
                    "skill_profile": getattr(labor_hour, "skill_profile", None),
                    "ability_profile": getattr(labor_hour, "ability_profile", None),
                    "total_hours": getattr(labor_hour, "total_hours", None),
                },
                ensure_ascii=False,
            ),
            jobs=json.dumps(jobs_view, ensure_ascii=False),
            month=json.dumps(month, ensure_ascii=False),
        )
        raw = await self._llm_chat(system="Return strict JSON only.", user=prompt, temperature=0.2)
        try:
            parsed = self._json_loads_loose(raw)
            chosen = [str(x) for x in (parsed.get("apply_job_ids") or [])]
        except Exception:
            chosen = [top_matches[0].job.job_id]

        apps: List[JobApplication] = []
        by_id = {m.job.job_id: m.job for m in top_matches}
        for job_id in chosen:
            if job_id not in by_id:
                continue
            job = by_id[job_id]
            apps.append(
                JobApplication.create(
                    job_id=job.job_id,
                    household_id=self.household_id,
                    lh_type=labor_hour.lh_type,
                    expected_wage=float(getattr(job, "wage_per_hour", 0.0) or 0.0),
                    worker_skills=dict(labor_hour.skill_profile or {}),
                    worker_abilities=dict(labor_hour.ability_profile or {}),
                    month=month,
                )
            )
        return apps

    async def decide_offer(
        self,
        *,
        month: int,
        labor_hour: LaborHour,
        offers: Sequence[Job],
        use_llm: bool = True,
    ) -> Optional[str]:
        """
        Step: member receives firm offers and decides where to work (LLM).
        Returns accepted job_id or None.
        """
        if not offers:
            return None
        if not use_llm:
            # minimal rule: accept the highest wage offer
            best = sorted(offers, key=lambda j: float(getattr(j, "wage_per_hour", 0.0) or 0.0), reverse=True)[0]
            return best.job_id

        offers_view = [
            {
                "job_id": j.job_id,
                "firm_id": getattr(j, "firm_id", None),
                "title": getattr(j, "title", None),
                "soc": getattr(j, "SOC", None),
                "wage_per_hour": getattr(j, "wage_per_hour", None),
                "hours_per_period": getattr(j, "hours_per_period", None),
            }
            for j in offers
        ]
        prompt = JOB_OFFER_DECISION_PROMPT.format(
            persona=json.dumps(self.get_persona_prompt_view(), ensure_ascii=False),
            past_household_status=self.past_household_status_text,
            member_info=json.dumps(
                {
                    "lh_type": getattr(labor_hour, "lh_type", None),
                    "skill_profile": getattr(labor_hour, "skill_profile", None),
                    "ability_profile": getattr(labor_hour, "ability_profile", None),
                    "total_hours": getattr(labor_hour, "total_hours", None),
                },
                ensure_ascii=False,
            ),
            offers=json.dumps(offers_view, ensure_ascii=False),
            month=json.dumps(month, ensure_ascii=False),
        )
        raw = await self._llm_chat(system="Return strict JSON only.", user=prompt, temperature=0.2)
        try:
            parsed = self._json_loads_loose(raw)
            accept_job_id = parsed.get("accept_job_id")
        except Exception:
            accept_job_id = sorted(offers, key=lambda j: float(getattr(j, "wage_per_hour", 0.0) or 0.0), reverse=True)[0].job_id
        if accept_job_id is None:
            return None
        return str(accept_job_id)


# =============================================================================
# Assumptions / interfaces used (for you to review & adjust)
# =============================================================================
#
# 1) ProductMarket interface (vector matching step):
#    - search_by_vector(query: str, top_k: int = 20, must_contain: Optional[str] = None) -> List[Product]
#    - If it's a Ray actor: product_market.search_by_vector.remote(...)
#
# 2) EconomicCenter interface (purchase execution step):
#    - process_batch_purchases(month: int, buyer_id: str, purchase_list: List[Dict]) -> List[Optional[str]]
#      where each purchase dict is:
#        {"seller_id": str, "product": Product, "quantity": float, "reservation_id": Optional[str]}
#    - If it's a Ray actor: economic_center.process_batch_purchases.remote(...)
#
# 3) Job/Labor interfaces:
#    - LaborHour fields used: is_valid, firm_id, lh_type, skill_profile, ability_profile, total_hours
#    - Job fields used: job_id, firm_id, SOC, title, wage_per_hour, required_skills, required_abilities, is_valid, positions_available
#
# 4) LLM env vars used:
#    - DEEPSEEK_API_KEY, BASE_URL, MODEL
