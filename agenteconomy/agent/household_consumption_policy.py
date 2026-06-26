from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


MAJOR_BUDGET_KEYS: Tuple[str, ...] = (
    "Retail merchandise",
    "housing",
    "healthcare",
    "transportation",
    "utilities",
    "insurance",
)


ESSENTIAL_CATEGORY_KEYWORDS: Tuple[str, ...] = (
    "farm",
    "food",
    "beverage",
    "tobacco",
    "apparel",
    "leather",
    "chemical",
    "petroleum",
    "coal",
)

SEMI_ESSENTIAL_CATEGORY_KEYWORDS: Tuple[str, ...] = (
    "textile",
    "paper",
    "rubber",
    "plastic",
    "furniture",
    "wood",
    "fabricated metal",
    "motor vehicles",
    "transportation",
)


@dataclass(frozen=True)
class ConsumptionPersonaParams:
    """Rule knobs in normalized 0..1 form."""

    price_sensitivity: float = 0.5
    liquidity_preference: float = 0.5
    habit_strength: float = 0.35
    essential_bias: float = 0.0

    @classmethod
    def from_mapping(cls, raw: Optional[Mapping[str, Any]]) -> "ConsumptionPersonaParams":
        if not isinstance(raw, Mapping):
            return cls()

        return cls(
            price_sensitivity=_read_unit_value(raw, "price_sensitivity", default=0.5),
            liquidity_preference=_read_unit_value(raw, "liquidity_preference", default=0.5),
            habit_strength=_read_unit_value(raw, "habit_strength", default=0.35),
            essential_bias=_read_signed_unit_value(raw, "essential_bias", default=0.0),
        )


@dataclass
class NormalizedCandidate:
    category: str
    product_id: str
    name: str = ""
    description: str = ""
    current_price: float = 0.0
    storage: float = 0.0
    score: Optional[float] = None
    rank: int = 0
    raw: Any = None


@dataclass
class ConsumptionPolicyPlan:
    total_budget: float
    target_cash_buffer: float
    reserved_cash: float
    major_budgets: Dict[str, float]
    category_budgets: Dict[str, float]
    category_plans: List[Dict[str, Any]]
    product_preferences: Dict[str, List[Dict[str, Any]]]
    purchases: List[Dict[str, Any]]
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_budget": self.total_budget,
            "target_cash_buffer": self.target_cash_buffer,
            "reserved_cash": self.reserved_cash,
            "major_budgets": dict(self.major_budgets),
            "category_budgets": dict(self.category_budgets),
            "category_plans": list(self.category_plans),
            "product_preferences": {
                cat: list(items) for cat, items in self.product_preferences.items()
            },
            "purchases": list(self.purchases),
            "diagnostics": dict(self.diagnostics),
        }


def build_constrained_llm_consumption_plan(
    *,
    household_state: Optional[Mapping[str, Any]] = None,
    available_budget: Optional[float] = None,
    expected_income: Optional[float] = None,
    candidate_products_by_category: Optional[Any] = None,
    persona_params: Optional[Mapping[str, Any]] = None,
    categories: Optional[Sequence[str]] = None,
    llm_major_budgets: Optional[Mapping[str, Any]] = None,
    llm_category_plans: Optional[Sequence[Mapping[str, Any]]] = None,
    llm_purchases_by_category: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
    llm_note: str = "",
) -> ConsumptionPolicyPlan:
    """
    Build an LLM-assisted plan with hard empirical and accounting constraints.

    Empirical anchoring is supplied by the same PSID-backed state used by the
    deterministic policy: monthly income (ER85629), wealth (ER85692), household
    size (ER82017), historical monthly expenditure (ER85768), and observed
    category expenditure fields. The LLM can move shares and rank products, but
    cannot change the total budget envelope or create invalid purchases.
    """

    anchor = build_rule_based_consumption_plan(
        household_state=household_state,
        available_budget=available_budget,
        expected_income=expected_income,
        candidate_products_by_category=candidate_products_by_category,
        persona_params=persona_params,
        categories=categories,
    )

    major_budgets = _constrain_major_budgets(
        anchor_major_budgets=anchor.major_budgets,
        llm_major_budgets=llm_major_budgets,
        total_budget=anchor.total_budget,
    )
    retail_budget = float(major_budgets.get("Retail merchandise") or 0.0)
    category_budgets = _constrain_category_budgets(
        anchor_category_budgets=anchor.category_budgets,
        llm_category_plans=llm_category_plans,
        retail_budget=retail_budget,
    )
    category_plans = _build_category_plans_from_llm(
        anchor_category_plans=anchor.category_plans,
        category_budgets=category_budgets,
        llm_category_plans=llm_category_plans,
    )

    candidates = normalize_candidates_by_category(candidate_products_by_category)
    product_preferences, purchases = _select_llm_constrained_purchases(
        candidates_by_category=candidates,
        category_budgets=category_budgets,
        llm_purchases_by_category=llm_purchases_by_category,
        fallback_preferences=anchor.product_preferences,
        fallback_purchases=anchor.purchases,
    )

    diagnostics = dict(anchor.diagnostics)
    diagnostics.update(
        {
            "policy_mode": "llm_constrained",
            "llm_note": str(llm_note or ""),
            "anchor_total_budget": anchor.total_budget,
            "empirical_anchor_fields": [
                "ER85629",
                "ER85692",
                "ER85768",
                "ER82017",
                "ER85701",
                "ER85747",
                "expenditure_retail_merchandise",
                "expenditure_transportation",
                "expenditure_utilities",
                "expenditure_insurance",
            ],
            "major_budget_anchor": dict(anchor.major_budgets),
            "category_budget_anchor": dict(anchor.category_budgets),
        }
    )

    return ConsumptionPolicyPlan(
        total_budget=anchor.total_budget,
        target_cash_buffer=anchor.target_cash_buffer,
        reserved_cash=anchor.reserved_cash,
        major_budgets=major_budgets,
        category_budgets=category_budgets,
        category_plans=category_plans,
        product_preferences=product_preferences,
        purchases=purchases,
        diagnostics=diagnostics,
    )


def build_rule_based_consumption_plan(
    *,
    household_state: Optional[Mapping[str, Any]] = None,
    available_budget: Optional[float] = None,
    expected_income: Optional[float] = None,
    candidate_products_by_category: Optional[Any] = None,
    persona_params: Optional[Mapping[str, Any]] = None,
    categories: Optional[Sequence[str]] = None,
) -> ConsumptionPolicyPlan:
    """
    Build a deterministic household consumption plan.

    The rule path intentionally avoids LLM/Ray dependencies. It combines a cash
    buffer constraint, income/wealth consumption propensity, basic-vs-optional
    category allocation, price sensitivity, and habit/persona adjustments.
    """

    state = dict(household_state or {})
    persona = ConsumptionPersonaParams.from_mapping(
        persona_params
        or state.get("consumption_persona_params")
        or state.get("persona_params")
        or state.get("persona")
    )

    available_cash = _nonnegative_float(
        available_budget,
        fallback=_first_number(
            state,
            ("available_budget", "available_balance", "cash", "ER85692", "net_wealth"),
            default=0.0,
        ),
    )
    income = _nonnegative_float(
        expected_income,
        fallback=_first_number(state, ("expected_income", "ER85629", "monthly_income"), default=0.0),
    )
    wealth = _nonnegative_float(
        _first_number(state, ("net_wealth", "wealth", "ER85692"), default=available_cash),
        fallback=available_cash,
    )
    household_size = max(1.0, _nonnegative_float(_first_number(state, ("ER82017", "household_size"), default=1.0), fallback=1.0))
    historical_total = _nonnegative_float(
        _first_number(
            state,
            ("historical_monthly_expenditure", "ER85768", "monthly_expenditure"),
            default=0.0,
        ),
        fallback=0.0,
    )
    last_consumption = _nonnegative_float(
        _first_number(state, ("last_month_consumption", "_last_month_consumption"), default=0.0),
        fallback=0.0,
    )
    macro_indicators = state.get("macro_indicators") if isinstance(state.get("macro_indicators"), Mapping) else {}
    unemployment_rate = _clamp(
        _nonnegative_float(
            _first_number(macro_indicators, ("unemployment_rate",), default=0.0),
            fallback=0.0,
        ),
        0.0,
        1.0,
    )
    unemployment_pressure = _clamp((unemployment_rate - 0.08) / 0.32, 0.0, 1.0)

    basic_floor = 1000.0 + 450.0 * max(0.0, household_size - 1.0)
    target_cash_buffer = _target_cash_buffer(
        available_cash=available_cash,
        expected_income=income,
        basic_floor=basic_floor,
        liquidity_preference=persona.liquidity_preference,
    )
    spendable_after_buffer = max(0.0, available_cash - target_cash_buffer)

    mpc = _marginal_propensity_to_consume(
        expected_income=income,
        wealth=wealth,
        basic_floor=basic_floor,
        liquidity_preference=persona.liquidity_preference,
    )
    # 消费利率敏感性:实际利率上升 → MPC 下降(储蓄增加)。仅在 consumption_rate_sensitivity>0
    # 时生效(默认 0 = 不改变行为)。利率取 macro_indicators 的月度利率,折年化偏离自然利率。
    rate_sensitivity = _nonnegative_float(
        macro_indicators.get("consumption_rate_sensitivity")
        if isinstance(macro_indicators, Mapping) else None,
        fallback=0.0,
    )
    if rate_sensitivity > 0.0:
        monthly_rate = _nonnegative_float(
            _first_number(macro_indicators, ("interest_rate",), default=0.0), fallback=0.0
        )
        natural_monthly = _nonnegative_float(
            _first_number(macro_indicators, ("natural_rate_monthly",), default=0.005 / 12.0),
            fallback=0.005 / 12.0,
        )
        rate_gap_annual = (monthly_rate - natural_monthly) * 12.0
        mpc = _clamp(mpc * (1.0 - rate_sensitivity * rate_gap_annual), 0.30, 0.92)
    income_component = income * mpc
    shortfall_ratio = _clamp((basic_floor - income) / basic_floor if basic_floor > 0 else 0.0, 0.0, 1.0)
    excess_liquidity = max(0.0, available_cash - target_cash_buffer)
    precautionary_discount = 1.0 - 0.35 * unemployment_pressure * (0.45 + 0.55 * persona.liquidity_preference)
    wealth_draw_rate = (
        0.003
        + 0.012 * (1.0 - persona.liquidity_preference)
        + 0.030 * shortfall_ratio
    ) * precautionary_discount
    income_flow_cap = (
        income * (1.05 + 0.20 * (1.0 - persona.liquidity_preference))
        + basic_floor * (0.75 + 0.35 * (1.0 - persona.liquidity_preference))
    )
    safety_flow_cap = basic_floor * (
        1.05
        + 0.75 * (1.0 - persona.liquidity_preference)
        + 0.25 * shortfall_ratio
    )
    monthly_flow_cap = max(income_flow_cap, safety_flow_cap)
    monthly_flow_cap *= 1.0 - 0.12 * unemployment_pressure * persona.liquidity_preference
    monthly_flow_cap = max(basic_floor * 0.65, monthly_flow_cap)
    wealth_draw_cap = max(0.0, monthly_flow_cap - income_component)
    wealth_component_uncapped = excess_liquidity * wealth_draw_rate
    wealth_component = min(wealth_component_uncapped, wealth_draw_cap)
    desired_budget = income_component + wealth_component

    habit_reference = last_consumption or historical_total
    if habit_reference > 0.0:
        habit_reference = min(habit_reference, monthly_flow_cap)
        desired_budget = (
            desired_budget * (1.0 - persona.habit_strength)
            + habit_reference * persona.habit_strength
        )

    if spendable_after_buffer > 0.0:
        minimum_basic_spend = min(spendable_after_buffer, basic_floor * 0.35)
        desired_budget = max(desired_budget, minimum_basic_spend)

    total_budget = _round_money(
        min(max(0.0, desired_budget), monthly_flow_cap, spendable_after_buffer, available_cash)
    )
    reserved_cash = _round_money(max(0.0, available_cash - total_budget))

    tightness = 1.0 - _clamp(total_budget / max(basic_floor, 1.0), 0.0, 1.0)
    major_budgets = _allocate_major_budgets(
        total_budget=total_budget,
        tightness=tightness,
        persona=persona,
    )

    retail_budget = major_budgets.get("Retail merchandise", 0.0)
    category_list = list(categories or [])
    if not category_list:
        category_list = _categories_from_candidates(candidate_products_by_category)
    category_budgets = _allocate_retail_categories(
        categories=category_list,
        retail_budget=retail_budget,
        tightness=tightness,
        persona=persona,
    )

    category_plans = [
        {
            "category": category,
            "budget_amount": budget,
            "need_descriptions": _need_descriptions_for_category(category),
            "need_tier": _category_tier(category),
        }
        for category, budget in category_budgets.items()
    ]

    candidates = normalize_candidates_by_category(candidate_products_by_category)
    product_preferences, purchases = _rank_product_preferences(
        candidates_by_category=candidates,
        category_budgets=category_budgets,
        persona=persona,
    )

    return ConsumptionPolicyPlan(
        total_budget=total_budget,
        target_cash_buffer=_round_money(target_cash_buffer),
        reserved_cash=reserved_cash,
        major_budgets=major_budgets,
        category_budgets=category_budgets,
        category_plans=category_plans,
        product_preferences=product_preferences,
        purchases=purchases,
        diagnostics={
            "available_budget": _round_money(available_cash),
            "expected_income": _round_money(income),
            "wealth": _round_money(wealth),
            "basic_floor": _round_money(basic_floor),
            "unemployment_rate": round(unemployment_rate, 4),
            "unemployment_pressure": round(unemployment_pressure, 4),
            "mpc": round(mpc, 4),
            "precautionary_discount": round(precautionary_discount, 4),
            "wealth_draw_rate": round(wealth_draw_rate, 4),
            "wealth_component_uncapped": _round_money(wealth_component_uncapped),
            "wealth_component": _round_money(wealth_component),
            "monthly_flow_cap": _round_money(monthly_flow_cap),
            "habit_reference": _round_money(habit_reference),
            "persona_params": {
                "price_sensitivity": persona.price_sensitivity,
                "liquidity_preference": persona.liquidity_preference,
                "habit_strength": persona.habit_strength,
                "essential_bias": persona.essential_bias,
            },
            "tightness": round(tightness, 4),
        },
    )


def normalize_candidates_by_category(candidate_products_by_category: Optional[Any]) -> Dict[str, List[NormalizedCandidate]]:
    if candidate_products_by_category is None:
        return {}

    grouped: Dict[str, List[NormalizedCandidate]] = {}

    if isinstance(candidate_products_by_category, Mapping):
        for category, payload in candidate_products_by_category.items():
            if isinstance(payload, Mapping) and "candidates" in payload:
                raw_candidates = payload.get("candidates") or []
            else:
                raw_candidates = payload or []
            if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
                continue
            for rank, raw in enumerate(raw_candidates):
                cand = _normalize_candidate(raw, default_category=str(category), rank=rank)
                if cand is None:
                    continue
                grouped.setdefault(cand.category, []).append(cand)
        return grouped

    if isinstance(candidate_products_by_category, Sequence) and not isinstance(candidate_products_by_category, (str, bytes)):
        for rank, raw in enumerate(candidate_products_by_category):
            cand = _normalize_candidate(raw, default_category="", rank=rank)
            if cand is None:
                continue
            grouped.setdefault(cand.category, []).append(cand)

    return grouped


def _rank_product_preferences(
    *,
    candidates_by_category: Mapping[str, Sequence[NormalizedCandidate]],
    category_budgets: Mapping[str, float],
    persona: ConsumptionPersonaParams,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    preferences: Dict[str, List[Dict[str, Any]]] = {}
    purchases: List[Dict[str, Any]] = []

    for category, budget in category_budgets.items():
        cat_budget = max(0.0, float(budget or 0.0))
        candidates = [
            c
            for c in candidates_by_category.get(category, [])
            if c.product_id and c.current_price > 0.0 and c.storage > 0.0
        ]
        if not candidates or cat_budget <= 0.0:
            preferences[category] = []
            continue

        prices = sorted(c.current_price for c in candidates if c.current_price > 0.0)
        median_price = prices[len(prices) // 2] if prices else 1.0
        scored: List[Tuple[float, NormalizedCandidate]] = []
        for cand in candidates:
            price_score = _clamp(median_price / max(cand.current_price, 0.01), 0.0, 2.0) / 2.0
            stock_score = _clamp(cand.storage / 20.0, 0.0, 1.0)
            search_score = _coerce_search_score(cand.score)
            rank_score = 1.0 / (1.0 + max(0, cand.rank))
            affordability = 1.0 if cand.current_price <= max(cat_budget * 0.5, 1.0) else max(cat_budget / cand.current_price, 0.0)
            tier_bonus = 0.10 if _category_tier(category) == "basic" else 0.0

            score = (
                persona.price_sensitivity * price_score
                + (1.0 - persona.price_sensitivity) * (0.42 * search_score + 0.28 * stock_score + 0.30 * rank_score)
                + 0.18 * affordability
                + tier_bonus
            )
            scored.append((score, cand))

        scored.sort(key=lambda item: item[0], reverse=True)
        max_selected = min(12, len(scored))
        if median_price > 0.0:
            budget_scaled = int(cat_budget / max(median_price * 25.0, 1.0))
            max_selected = min(max_selected, max(3, budget_scaled))
        top = scored[:max_selected]
        score_sum = sum(max(0.0, score) for score, _ in top)
        if score_sum <= 0.0:
            score_sum = float(len(top))

        cat_preferences: List[Dict[str, Any]] = []
        for score, cand in scored:
            preferred_share = 0.0
            allocated_budget = 0.0
            if any(cand is selected for _, selected in top):
                preferred_share = max(0.0, score) / score_sum if score_sum > 0.0 else 1.0 / len(top)
                per_household_stock_cap = cand.current_price * cand.storage * 0.35
                allocated_budget = min(
                    cat_budget * preferred_share,
                    per_household_stock_cap,
                    cand.current_price * cand.storage,
                )
            rec = {
                "category": category,
                "product_id": cand.product_id,
                "name": cand.name,
                "current_price": _round_money(cand.current_price),
                "storage": _round_money(cand.storage),
                "preference_score": round(score, 4),
                "preferred_budget_share": round(preferred_share, 4),
                "allocated_budget": _round_money(allocated_budget),
                "reason": _preference_reason(persona=persona, category=category, candidate=cand),
            }
            cat_preferences.append(rec)
            if allocated_budget > 0.0:
                purchases.append(
                    {
                        "category": category,
                        "product_id": cand.product_id,
                        "allocated_budget": _round_money(allocated_budget),
                        "reason": rec["reason"],
                    }
                )
        preferences[category] = cat_preferences

    return preferences, purchases


def _constrain_major_budgets(
    *,
    anchor_major_budgets: Mapping[str, float],
    llm_major_budgets: Optional[Mapping[str, Any]],
    total_budget: float,
) -> Dict[str, float]:
    if total_budget <= 0.0:
        return {key: 0.0 for key in MAJOR_BUDGET_KEYS}
    if not isinstance(llm_major_budgets, Mapping):
        return _money_allocation(total_budget, anchor_major_budgets, MAJOR_BUDGET_KEYS)

    raw = {
        key: _nonnegative_float(llm_major_budgets.get(key), fallback=anchor_major_budgets.get(key, 0.0))
        for key in MAJOR_BUDGET_KEYS
    }
    if sum(raw.values()) <= 0.0:
        return _money_allocation(total_budget, anchor_major_budgets, MAJOR_BUDGET_KEYS)

    anchor_shares = _shares(anchor_major_budgets, MAJOR_BUDGET_KEYS)
    llm_shares = _shares(raw, MAJOR_BUDGET_KEYS)
    blended = {
        key: _clamp(
            0.72 * anchor_shares.get(key, 0.0) + 0.28 * llm_shares.get(key, 0.0),
            max(0.0, anchor_shares.get(key, 0.0) * 0.60),
            min(1.0, anchor_shares.get(key, 0.0) * 1.45 + 0.02),
        )
        for key in MAJOR_BUDGET_KEYS
    }
    return _money_allocation(total_budget, blended, MAJOR_BUDGET_KEYS)


def _constrain_category_budgets(
    *,
    anchor_category_budgets: Mapping[str, float],
    llm_category_plans: Optional[Sequence[Mapping[str, Any]]],
    retail_budget: float,
) -> Dict[str, float]:
    categories = list(anchor_category_budgets.keys())
    if retail_budget <= 0.0:
        return {category: 0.0 for category in categories}
    if not categories:
        return {}

    llm_weights = {category: 0.0 for category in categories}
    if isinstance(llm_category_plans, Sequence) and not isinstance(llm_category_plans, (str, bytes)):
        for rec in llm_category_plans:
            if not isinstance(rec, Mapping):
                continue
            category = str(rec.get("category") or "")
            if category not in llm_weights:
                continue
            llm_weights[category] = _nonnegative_float(
                rec.get("budget_amount"),
                fallback=0.0,
            )

    if sum(llm_weights.values()) <= 0.0:
        return _money_allocation(retail_budget, anchor_category_budgets, categories)

    anchor_shares = _shares(anchor_category_budgets, categories)
    llm_shares = _shares(llm_weights, categories)
    blended = {
        category: _clamp(
            0.68 * anchor_shares.get(category, 0.0) + 0.32 * llm_shares.get(category, 0.0),
            max(0.0, anchor_shares.get(category, 0.0) * 0.45),
            min(1.0, anchor_shares.get(category, 0.0) * 1.80 + 0.02),
        )
        for category in categories
    }
    return _money_allocation(retail_budget, blended, categories)


def _build_category_plans_from_llm(
    *,
    anchor_category_plans: Sequence[Mapping[str, Any]],
    category_budgets: Mapping[str, float],
    llm_category_plans: Optional[Sequence[Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    llm_by_category: Dict[str, Mapping[str, Any]] = {}
    if isinstance(llm_category_plans, Sequence) and not isinstance(llm_category_plans, (str, bytes)):
        for rec in llm_category_plans:
            if isinstance(rec, Mapping) and rec.get("category"):
                llm_by_category[str(rec.get("category"))] = rec

    plans: List[Dict[str, Any]] = []
    for anchor in anchor_category_plans:
        category = str(anchor.get("category") or "")
        if not category:
            continue
        llm_rec = llm_by_category.get(category, {})
        needs = [
            str(item)
            for item in (llm_rec.get("need_descriptions") or anchor.get("need_descriptions") or [])
            if str(item).strip()
        ]
        if not needs:
            needs = _need_descriptions_for_category(category)
        plans.append(
            {
                "category": category,
                "budget_amount": _round_money(float(category_budgets.get(category, 0.0) or 0.0)),
                "need_descriptions": needs[:4],
                "need_tier": _category_tier(category),
            }
        )
    return plans


def _select_llm_constrained_purchases(
    *,
    candidates_by_category: Mapping[str, Sequence[NormalizedCandidate]],
    category_budgets: Mapping[str, float],
    llm_purchases_by_category: Optional[Mapping[str, Sequence[Mapping[str, Any]]]],
    fallback_preferences: Mapping[str, Sequence[Mapping[str, Any]]],
    fallback_purchases: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    if not isinstance(llm_purchases_by_category, Mapping):
        return (
            {category: list(items) for category, items in fallback_preferences.items()},
            [dict(item) for item in fallback_purchases],
        )

    fallback_by_category: Dict[str, List[Mapping[str, Any]]] = {}
    for item in fallback_purchases or []:
        category = str(item.get("category") or "")
        if category:
            fallback_by_category.setdefault(category, []).append(item)

    preferences: Dict[str, List[Dict[str, Any]]] = {}
    purchases: List[Dict[str, Any]] = []

    for category, budget in category_budgets.items():
        cat_budget = max(0.0, float(budget or 0.0))
        valid_candidates = {
            cand.product_id: cand
            for cand in candidates_by_category.get(category, [])
            if cand.product_id and cand.current_price > 0.0 and cand.storage > 0.0
        }
        llm_recs = list(llm_purchases_by_category.get(category) or [])
        selected: List[Tuple[Mapping[str, Any], NormalizedCandidate, float]] = []
        for rec in llm_recs:
            if not isinstance(rec, Mapping):
                continue
            product_id = str(rec.get("product_id") or "")
            candidate = valid_candidates.get(product_id)
            if candidate is None:
                continue
            share = _nonnegative_float(rec.get("budget_share"), fallback=0.0)
            if share > 1.0:
                share = share / 100.0 if share <= 100.0 else 1.0
            selected.append((rec, candidate, _clamp(share, 0.0, 1.0)))

        if cat_budget > 0.0 and not selected:
            for fallback in fallback_by_category.get(category, []):
                product_id = str(fallback.get("product_id") or "")
                candidate = valid_candidates.get(product_id)
                if candidate is None:
                    continue
                allocated = _nonnegative_float(fallback.get("allocated_budget"), fallback=0.0)
                share = allocated / cat_budget if cat_budget > 0 else 0.0
                selected.append((fallback, candidate, _clamp(share, 0.0, 1.0)))

        share_sum = sum(share for _, _, share in selected)
        if selected and share_sum <= 0.0:
            selected = [(rec, cand, 1.0 / len(selected)) for rec, cand, _ in selected]
            share_sum = 1.0
        if share_sum > 1.0:
            selected = [(rec, cand, share / share_sum) for rec, cand, share in selected]
            share_sum = 1.0

        cat_preferences: List[Dict[str, Any]] = []
        for rec, cand, share in selected:
            allocated_budget = min(
                cat_budget * share,
                cand.current_price * cand.storage,
                cand.current_price * cand.storage * 0.35,
            )
            if allocated_budget <= 0.0:
                continue
            reason = str(rec.get("reason") or "llm_preference_constrained")
            item = {
                "category": category,
                "product_id": cand.product_id,
                "name": cand.name,
                "current_price": _round_money(cand.current_price),
                "storage": _round_money(cand.storage),
                "preference_score": round(share, 4),
                "preferred_budget_share": round(share, 4),
                "allocated_budget": _round_money(allocated_budget),
                "reason": reason,
            }
            cat_preferences.append(item)
            purchases.append(
                {
                    "category": category,
                    "product_id": cand.product_id,
                    "allocated_budget": _round_money(allocated_budget),
                    "reason": reason,
                }
            )
        if not cat_preferences:
            cat_preferences = [dict(item) for item in fallback_preferences.get(category, [])]
        preferences[category] = cat_preferences

    return preferences, purchases


def _shares(values: Mapping[str, float], keys: Sequence[str]) -> Dict[str, float]:
    total = sum(max(0.0, float(values.get(key, 0.0) or 0.0)) for key in keys)
    if total <= 0.0:
        return {key: 0.0 for key in keys}
    return {
        key: max(0.0, float(values.get(key, 0.0) or 0.0)) / total
        for key in keys
    }


def _target_cash_buffer(
    *,
    available_cash: float,
    expected_income: float,
    basic_floor: float,
    liquidity_preference: float,
) -> float:
    if available_cash <= 0.0:
        return 0.0
    income_buffer = expected_income * (0.08 + 0.32 * liquidity_preference)
    basic_buffer = basic_floor * (0.12 + 0.30 * liquidity_preference)
    target = max(150.0, income_buffer, basic_buffer)
    return min(available_cash, target)


def _marginal_propensity_to_consume(
    *,
    expected_income: float,
    wealth: float,
    basic_floor: float,
    liquidity_preference: float,
) -> float:
    income_pressure = basic_floor / (basic_floor + max(expected_income, 0.0))
    wealth_cushion = _clamp(wealth / max(expected_income * 12.0, basic_floor * 12.0, 1.0), 0.0, 2.0)
    mpc = 0.58 + 0.25 * income_pressure + 0.04 * wealth_cushion - 0.12 * liquidity_preference
    return _clamp(mpc, 0.45, 0.92)


def _allocate_major_budgets(
    *,
    total_budget: float,
    tightness: float,
    persona: ConsumptionPersonaParams,
) -> Dict[str, float]:
    if total_budget <= 0.0:
        return {k: 0.0 for k in MAJOR_BUDGET_KEYS}

    ratios = {
        "Retail merchandise": 0.36 - 0.04 * tightness,
        "housing": 0.25 + 0.07 * tightness + 0.03 * persona.liquidity_preference,
        "healthcare": 0.10 + 0.02 * tightness,
        "transportation": 0.12,
        "utilities": 0.08 + 0.03 * tightness,
        "insurance": 0.09 + 0.02 * persona.liquidity_preference,
    }
    return _money_allocation(total_budget, ratios, MAJOR_BUDGET_KEYS)


def _allocate_retail_categories(
    *,
    categories: Sequence[str],
    retail_budget: float,
    tightness: float,
    persona: ConsumptionPersonaParams,
) -> Dict[str, float]:
    clean_categories = [str(c) for c in categories if str(c).strip()]
    if not clean_categories:
        return {}
    if retail_budget <= 0.0:
        return {category: 0.0 for category in clean_categories}

    essential_share = _clamp(
        0.58
        + 0.22 * tightness
        + 0.08 * persona.liquidity_preference
        + 0.10 * persona.essential_bias,
        0.45,
        0.88,
    )

    essential: List[str] = []
    semi: List[str] = []
    optional: List[str] = []
    for category in clean_categories:
        tier = _category_tier(category)
        if tier == "basic":
            essential.append(category)
        elif tier == "semi":
            semi.append(category)
        else:
            optional.append(category)

    weights: Dict[str, float] = {}
    if essential:
        per = essential_share / len(essential)
        weights.update({c: per for c in essential})
    if semi:
        per = (1.0 - essential_share) * 0.55 / len(semi)
        weights.update({c: per for c in semi})
    if optional:
        per = (1.0 - essential_share) * 0.45 / len(optional)
        weights.update({c: per for c in optional})

    if not weights:
        weights = {c: 1.0 for c in clean_categories}
    return _money_allocation(retail_budget, weights, clean_categories)


def _money_allocation(total: float, weights: Mapping[str, float], ordered_keys: Sequence[str]) -> Dict[str, float]:
    total = _round_money(max(0.0, total))
    clean_weights = {k: max(0.0, float(weights.get(k, 0.0) or 0.0)) for k in ordered_keys}
    weight_sum = sum(clean_weights.values())
    if weight_sum <= 0.0:
        clean_weights = {k: 1.0 for k in ordered_keys}
        weight_sum = float(len(ordered_keys))

    allocation: Dict[str, float] = {}
    running = 0.0
    for key in ordered_keys[:-1]:
        amount = _round_money(total * clean_weights[key] / weight_sum)
        allocation[key] = amount
        running += amount
    if ordered_keys:
        allocation[ordered_keys[-1]] = _round_money(max(0.0, total - running))
    return allocation


def _category_tier(category: str) -> str:
    text = str(category or "").lower()
    if any(keyword in text for keyword in ESSENTIAL_CATEGORY_KEYWORDS):
        return "basic"
    if any(keyword in text for keyword in SEMI_ESSENTIAL_CATEGORY_KEYWORDS):
        return "semi"
    return "optional"


def _need_descriptions_for_category(category: str) -> List[str]:
    tier = _category_tier(category)
    if tier == "basic":
        return [f"Basic recurring needs in {category}", f"Budget-conscious replenishment for {category}"]
    if tier == "semi":
        return [f"Maintenance and replacement needs in {category}"]
    return [f"Optional purchases in {category} only after core needs are covered"]


def _preference_reason(
    *,
    persona: ConsumptionPersonaParams,
    category: str,
    candidate: NormalizedCandidate,
) -> str:
    parts = []
    if persona.price_sensitivity >= 0.65:
        parts.append("price_sensitive")
    elif persona.price_sensitivity <= 0.35:
        parts.append("quality_or_match_weighted")
    else:
        parts.append("balanced_price_match")
    parts.append(_category_tier(category))
    if candidate.storage > 0:
        parts.append("in_stock")
    return ";".join(parts)


def _normalize_candidate(raw: Any, *, default_category: str, rank: int) -> Optional[NormalizedCandidate]:
    product_id = _read_field(raw, "product_id", "id", default="")
    if product_id in (None, ""):
        return None

    category = str(_read_field(raw, "category", "classification", default=default_category) or default_category or "Uncategorized")
    price = _nonnegative_float(
        _read_field(raw, "current_price", "price", "retail_price", "base_retail_price", default=0.0),
        fallback=0.0,
    )
    storage = _nonnegative_float(
        _read_field(raw, "storage", "available_stock", "stock", "amount", default=1.0),
        fallback=1.0,
    )
    return NormalizedCandidate(
        category=category,
        product_id=str(product_id),
        name=str(_read_field(raw, "name", default="") or ""),
        description=str(_read_field(raw, "description", default="") or ""),
        current_price=price,
        storage=storage,
        score=_optional_float(_read_field(raw, "score", "similarity", "relevance", default=None)),
        rank=rank,
        raw=raw,
    )


def _categories_from_candidates(candidate_products_by_category: Optional[Any]) -> List[str]:
    candidates = normalize_candidates_by_category(candidate_products_by_category)
    return list(candidates.keys())


def _read_field(raw: Any, *names: str, default: Any = None) -> Any:
    if isinstance(raw, Mapping):
        for name in names:
            if name in raw:
                return raw.get(name)
        return default
    for name in names:
        if hasattr(raw, name):
            return getattr(raw, name)
    return default


def _read_unit_value(raw: Mapping[str, Any], key: str, *, default: float) -> float:
    return _normalize_unit_value(_find_nested_value(raw, key), default=default)


def _read_signed_unit_value(raw: Mapping[str, Any], key: str, *, default: float) -> float:
    value = _find_nested_value(raw, key)
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if -1.0 <= f <= 1.0:
        return _clamp(f, -1.0, 1.0)
    if -100.0 <= f <= 100.0:
        return _clamp(f / 100.0, -1.0, 1.0)
    return default


def _normalize_unit_value(value: Any, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, str):
        label = value.strip().lower()
        if label in {"very low", "low"}:
            return 0.15
        if label in {"medium", "moderate", "neutral"}:
            return 0.5
        if label in {"high", "very high"}:
            return 0.85
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if 0.0 <= f <= 1.0:
        return f
    if 1.0 < f <= 5.0:
        return (f - 1.0) / 4.0
    if 5.0 < f <= 100.0:
        return f / 100.0
    return default


def _find_nested_value(raw: Mapping[str, Any], key: str) -> Any:
    if key in raw:
        return raw[key]
    for value in raw.values():
        if isinstance(value, Mapping):
            found = _find_nested_value(value, key)
            if found is not None:
                return found
    return None


def _first_number(raw: Mapping[str, Any], keys: Sequence[str], *, default: float) -> float:
    for key in keys:
        if key not in raw:
            continue
        value = _optional_float(raw.get(key))
        if value is not None:
            return value
    return default


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nonnegative_float(value: Any, *, fallback: float) -> float:
    parsed = _optional_float(value)
    if parsed is None:
        parsed = fallback
    return max(0.0, float(parsed or 0.0))


def _coerce_search_score(score: Optional[float]) -> float:
    if score is None:
        return 0.5
    value = float(score)
    if value < 0.0:
        return 0.0
    if value <= 1.0:
        return value
    return _clamp(value / 100.0, 0.0, 1.0)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _round_money(value: float) -> float:
    return round(float(value or 0.0), 2)
