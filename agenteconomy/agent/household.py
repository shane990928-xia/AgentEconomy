from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from agenteconomy.center.Model import Job, JobApplication, LaborHour, Product
from agenteconomy.llm.llm import call_llm
from agenteconomy.llm.prompt_template import (
    CONSUMPTION_MAJOR_BUDGET_PROMPT,
    CONSUMPTION_NEEDS_BY_CATEGORY_PROMPT,
    PURCHASE_BY_CATEGORY_PROMPT,
    JOB_APPLICATION_DECISION_PROMPT,
    JOB_OFFER_DECISION_PROMPT,
)


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
            title = getattr(self, "ER82181_occupation_title", None)
            soc = getattr(self, "ER82181", None)
            if title:
                return str(title)
            return str(soc)
        if field_name == "ER82500":
            title = getattr(self, "ER82500_occupation_title", None)
            soc = getattr(self, "ER82500", None)
            if title:
                return str(title)
            return str(soc)
        v = self.csv_values.get(field_name, getattr(self, field_name, None))
        decoded = self.decode_field_value(field_name, v)
        return decoded if decoded else str(v)

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
        row = self.profile_row or {}
        if not row:
            return {"summary_text": "Recent household situation: no historical profile data (no match in persona mapping CSV)."}

        # For extensibility: keep *all* CSV fields in structured form
        all_fields: Dict[str, Dict[str, Any]] = {}
        for k, raw_v in (row or {}).items():
            label = None
            if self._CODEBOOK_BY_VAR and k in self._CODEBOOK_BY_VAR:
                label = self._CODEBOOK_BY_VAR[k].get("label") or self._CODEBOOK_BY_VAR[k].get("name")
            # Decode based on stored parsed code (not raw string), to keep behavior consistent after updates
            decoded = self.decode_field_value(k, self.csv_values.get(str(k))) if str(k).startswith("ER") else None
            all_fields[str(k)] = {
                "raw": raw_v,
                "value": self.csv_values.get(str(k)),
                "decoded": decoded,
                "label": label,
            }

        # A small, high-signal subset for text summary (prompt-friendly)
        focus_vars = [
            "ER82017",
            "ER82027",
            "ER82150",
            "ER84520",
            "ER82800",
            "ER85629",
            "ER85666",
            "ER85692",
            "ER85701",
            "ER85747",
            "ER85768",
            "ER85780",
            "ER82181",  # RP occupation code
            "ER82500",  # SP occupation code
        ]

        facts: List[Dict[str, Any]] = []
        lines: List[str] = []
        for var in focus_vars:
            raw_val = row.get(var)
            if raw_val is None or str(raw_val).strip() == "":
                continue
            label = None
            if self._CODEBOOK_BY_VAR and var in self._CODEBOOK_BY_VAR:
                label = self._CODEBOOK_BY_VAR[var].get("label") or self._CODEBOOK_BY_VAR[var].get("name")
            # Prompt-facing value should be category text when possible
            stored_v = self.csv_values.get(var)
            if var == "ER82181":
                # Show fine-grained occupation title rather than big category
                value_view = self.get_field_text("ER82181")
                decoded = value_view
            elif var == "ER82500":
                value_view = self.get_field_text("ER82500")
                decoded = value_view
            else:
                decoded = self.decode_field_value(var, stored_v)
                value_view = decoded if decoded else stored_v
            facts.append({"var": var, "label": label, "raw": raw_val, "value": stored_v, "decoded": decoded})
            if label:
                lines.append(f"- {label} ({var}): {value_view}")
            else:
                lines.append(f"- {var}: {value_view}")

        persona_name = row.get("persona_name_current") or row.get("persona_name")
        persona_core = None
        if self.persona:
            persona_core = self.persona.get("core_characteristics")

        # Explicit derived keys that downstream logic may use (not duplicated in the English lines below)
        total_assets = self.csv_values.get("ER85692")
        head_occ = self.csv_values.get("ER82181")  # SOC code after mapping
        spouse_occ = self.csv_values.get("ER82500")  # SOC code after mapping
        rp_employment_status = self.csv_values.get("ER82433")
        sp_employment_status = self.csv_values.get("SP_employment_status")

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
            },
            "all_fields": all_fields,
            "facts": facts,
            "summary_text": summary,
        }

    # -------------------------------------------------------------------------
    # Public state update APIs (requested)
    # -------------------------------------------------------------------------

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
        self.csv_raw["ER85692"] = str(new_v)
        self.past_household_status = self._build_past_household_status()
        self.past_household_status_text = self.past_household_status.get("summary_text") or ""
        self.household_info["past_household_status"] = self.past_household_status
        return new_v

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
            self.csv_raw["ER82433"] = str(self._EMPLOYED_CODE)
        else:
            setattr(self, "ER82181", None)
            self.csv_values["ER82181"] = None
            setattr(self, "ER82433", self._NOT_EMPLOYED_CODE)
            self.csv_values["ER82433"] = self._NOT_EMPLOYED_CODE
            self.csv_raw["ER82433"] = str(self._NOT_EMPLOYED_CODE)
        setattr(self, "ER82181_occupation_title", title)
        self.csv_raw["ER82181"] = str(new_occupation_code)
        self.past_household_status = self._build_past_household_status()
        self.past_household_status_text = self.past_household_status.get("summary_text") or ""
        self.household_info["past_household_status"] = self.past_household_status

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
            self.csv_raw["SP_employment_status"] = str(self._EMPLOYED_CODE)
        else:
            setattr(self, "SP_employment_status", self._NOT_EMPLOYED_CODE)
            self.csv_values["SP_employment_status"] = self._NOT_EMPLOYED_CODE
            self.csv_raw["SP_employment_status"] = str(self._NOT_EMPLOYED_CODE)
        self.past_household_status = self._build_past_household_status()
        self.past_household_status_text = self.past_household_status.get("summary_text") or ""
        self.household_info["past_household_status"] = self.past_household_status

    # -------------------------------------------------------------------------
    # Dependency wiring
    # -------------------------------------------------------------------------

    def set_dependencies(self, *, economic_center=None, product_market=None, labor_market=None):
        self.economic_center = economic_center
        self.product_market = product_market
        self.labor_market = labor_market

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
        _ = product_id
        return 1.0

    def get_product_stock(self, product_id: str) -> float:
        """
        Placeholder: query current stock by product_id.
        For now returns a default value; later you can wire it to EconomicCenter/ProductMarket.
        """
        _ = product_id
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
        """
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
        try:
            return json.loads(s)
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
            return json.loads(s[start : end + 1])

    # -------------------------------------------------------------------------
    # Consumption decision (NEW workflow): step1-4
    # -------------------------------------------------------------------------

    async def consumption_step1_needs_by_category(
        self,
        *,
        total_budget: float,
        categories: Optional[List[str]] = None,
    ) -> CategoryNeedsOutput:
        """
        Step1 (default LLM):
        For each category name, allocate a budget and provide multiple need descriptions.
        """
        cats = categories or self.consumption_categories

        prompt = CONSUMPTION_NEEDS_BY_CATEGORY_PROMPT.format(
            household_info=json.dumps(self.household_info, ensure_ascii=False),
            past_household_status=self.past_household_status_text,
            total_budget=json.dumps(float(total_budget), ensure_ascii=False),
            categories=json.dumps(list(cats), ensure_ascii=False),
        )
        raw = await self._llm_chat(system="Return strict JSON only.", user=prompt, temperature=0.2)
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

    async def consumption_step0_major_budget_allocation(self) -> MajorBudgetOutput:
        """
        Step0 (default LLM):
        Allocate major budget buckets. Retail merchandise budget will be used as step1 total_budget.
        """
        prompt = CONSUMPTION_MAJOR_BUDGET_PROMPT.format(
            household_info=json.dumps(self.household_info, ensure_ascii=False),
            past_household_status=self.past_household_status_text,
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

    async def consumption_step2_vector_match(
        self,
        *,
        category_plans: List[CategoryPlan],
        top_k: int = 10,
        product_market=None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Step2:
        For each category, for each need description, retrieve top_k products via Qdrant,
        and build a per-category candidate list.

        Vector search code follows load_qdrant.py main (query_points).
        Note: We do not hard-filter by category here because the Qdrant payload does not contain
        the industry/category label; instead we bias retrieval by including the category name in the query text.
        """
        from agenteconomy.utils.embedding import embedding

        client = self._get_qdrant_client()
        collection = os.getenv("QDRANT_COLLECTION_NAME", "products")
        _ = product_market  # kept for extensibility

        out: Dict[str, Dict[str, Any]] = {}
        for cp in category_plans:
            cat = cp.category
            cat_candidates: List[Dict[str, Any]] = []
            for desc in cp.need_descriptions:
                # Include category name in query to bias retrieval toward that category.
                qv = embedding(f"{cat}: {desc}")
                resp = client.query_points(collection_name=collection, query=qv, limit=int(top_k))
                for pt in getattr(resp, "points", []) or []:
                    payload = pt.payload or {}
                    product_id = payload.get("product_id") or str(getattr(pt, "id", "") or "")
                    if not product_id:
                        continue

                    # payload["price"] is a reference price; DO NOT pass it into prompts (use queried current_price instead)
                    current_price = float(self.get_product_price(str(product_id)))
                    storage = float(self.get_product_stock(str(product_id)))
                    product_line = self._format_product_line(
                        product_id=str(product_id),
                        name=payload.get("name"),
                        description=payload.get("description"),
                        current_price=current_price,
                        storage=storage,
                    )
                    cat_candidates.append(
                        {
                            "category": cat,
                            "need_description": desc,
                            "product_id": str(product_id),
                            "name": payload.get("name"),
                            "description": payload.get("description"),
                            "current_price": current_price,
                            "storage": storage,
                            "product_line": product_line,
                            "score": getattr(pt, "score", None),
                        }
                    )
            out[cat] = {
                "category": cat,
                "budget_amount": float(cp.budget_amount or 0.0),
                "need_descriptions": list(cp.need_descriptions or []),
                "candidates": cat_candidates,
            }
        try:
            client.close()
        except Exception:
            pass
        return out

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
                household_info=json.dumps(self.household_info, ensure_ascii=False),
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
    ) -> Dict[str, Any]:
        """
        New end-to-end consumption flow (step1-4).
        Step4 is stubbed.
        """
        step0 = await self.consumption_step0_major_budget_allocation()
        retail_budget = float((step0.budgets or {}).get("Retail merchandise") or 0.0)
        step1 = await self.consumption_step1_needs_by_category(total_budget=retail_budget)
        step2 = await self.consumption_step2_vector_match(
            category_plans=step1.category_plans, top_k=top_k, product_market=product_market
        )
        step3 = await self.consumption_step3_purchase_llm(category_bundles=step2)
        _ = self.consumption_step4_validate(step1, step2, step3)
        return {
            "step0": {
                "total_budget": step0.total_budget,
                "budgets": step0.budgets,
                "note": step0.note,
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

    def _compute_matching_loss(self, worker_profile: List[Dict[str, float]], required_profile: List[Dict[str, Dict[str, float]]]) -> float:
        """
        Mirror of LaborMarket._compute_matching_loss (rule-based).
        Lower is better.
        """
        total_loss = 0.0
        if len(worker_profile) != len(required_profile):
            return float("inf")
        for i in range(len(worker_profile)):
            w = worker_profile[i] or {}
            r = required_profile[i] or {}
            for key, req in r.items():
                mean = (req or {}).get("mean")
                std = (req or {}).get("std")
                importance = (req or {}).get("importance", 1.0)
                if importance is None or float(importance) <= 0:
                    continue
                if std is None or float(std) <= 0:
                    continue
                if mean is None:
                    continue
                worker_value = float(w.get(key, 0.0) or 0.0)
                distance = (worker_value - float(mean)) / float(std)
                loss = float(importance) * (distance**2)
                if distance > 0:
                    loss *= 0.2
                total_loss += loss
        return total_loss

    def match_jobs_topk_by_loss(self, *, labor_hour: LaborHour, jobs: Sequence[Job], top_k: int = 3) -> List[JobMatch]:
        """
        Step: use loss to match topk jobs (rule-based).
        This is the skill/ability-based matching interface:
        - Caller must provide candidate jobs (e.g., already filtered by SOC elsewhere).
        - Loss is computed from LaborHour.skill_profile/ability_profile vs Job.required_skills/required_abilities.
        """

        ## Todo: SOC->Skill mapping
        if getattr(labor_hour, "skill_profile", None) is None or getattr(labor_hour, "ability_profile", None) is None:
            return []

        worker_profile = [dict(labor_hour.skill_profile or {}), dict(labor_hour.ability_profile or {})]
        matches: List[JobMatch] = []
        for job in jobs or []:
            if not getattr(job, "is_valid", True):
                continue
            if int(getattr(job, "positions_available", 0) or 0) <= 0:
                continue
            req_profile = [dict(getattr(job, "required_skills", {}) or {}), dict(getattr(job, "required_abilities", {}) or {})]
            loss = self._compute_matching_loss(worker_profile, req_profile)
            matches.append(JobMatch(job=job, loss=loss))
        matches.sort(key=lambda m: m.loss)
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
            household_info=json.dumps(self.household_info, ensure_ascii=False),
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
            parsed = json.loads(raw)
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
            household_info=json.dumps(self.household_info, ensure_ascii=False),
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
            parsed = json.loads(raw)
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
