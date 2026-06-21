"""
Centralized prompt templates for LLM-powered decisions.

Design principle:
- Keep inputs broad and extensible (persona/member_info) rather than narrow fields.
- Enforce strict JSON outputs to make downstream logic reliable.
"""


def build_firm_post_job_prompt(firm):
    prompt = f"""
    """
    return prompt


# =============================================================================
# Consumption: step0 (major budget allocation)
# =============================================================================

CONSUMPTION_PROFILE_PROMPT = """\
You are assigning stable household consumption behavior parameters for a
long-running macroeconomic agent-based simulation.

Inputs:
- persona (selected fields): {persona}
- household_state: {household_state}
- macro_indicators: {macro_indicators}

Task:
Infer a stable consumption profile. Do not choose products, do not set taxes,
do not clear markets, and do not override accounting or liquidity constraints.
The simulator will use these parameters as preference inputs to a constrained
monthly consumption function.

Parameter meanings:
- price_sensitivity: 0.0=quality/fit focused, 1.0=strongly price focused.
- liquidity_preference: 0.0=willing to spend down cash, 1.0=strong cash buffer.
- habit_strength: 0.0=adjusts quickly, 1.0=strongly follows past spending.
- essential_bias: -1.0=more discretionary, 1.0=more essential/basic needs.

Output STRICT JSON ONLY with this schema:
{{
  "price_sensitivity": 0.0,
  "liquidity_preference": 0.0,
  "habit_strength": 0.0,
  "essential_bias": 0.0,
  "strategy_type": "brief stable strategy label",
  "explanation": "one short sentence"
}}
"""


CONSUMPTION_MAJOR_BUDGET_PROMPT = """\
You are allocating a household's monthly consumption budget. Make a realistic and economically rational decision.

## Financial Status
- available_balance (savings): {available_balance}
- expected_income (this month): {expected_income}
- last_month_consumption: {last_month_consumption}
- historical_monthly_expenditure: {historical_expenditure}

## Household Profile
- persona: {persona}
- Recent situation: {past_household_status}

## Macroeconomic Environment
{macro_indicators}

## Empirical Budget Anchor
{empirical_constraints}

## Budget Allocation Principles (IMPORTANT)
1. Treat the empirical anchor as the binding budget envelope. Do not invent a larger monthly budget.
2. You may express household preferences by shifting category shares, but keep shares close to the anchor.
3. Income, wealth, household size, and historical expenditure are already embedded in the anchor.
4. Preserve a cash buffer and avoid spending more than available_balance.
5. Use the persona and macro environment for narrative and small preference adjustments only.

## Task
Decide total_budget and allocate it into these categories:
- Retail merchandise (food, clothing, household goods)
- housing (rent, mortgage, maintenance)
- healthcare (medical, insurance copays)
- transportation (car, gas, transit)
- utilities (electricity, water, internet)
- insurance (life, property, other)

## Output (STRICT JSON ONLY)
{{
  "total_budget": 0.0,
  "budgets": {{
    "Retail merchandise": 0.0,
    "housing": 0.0,
    "healthcare": 0.0,
    "transportation": 0.0,
    "utilities": 0.0,
    "insurance": 0.0
  }},
  "reasoning": "brief explanation of your budget decision",
  "note": "string"
}}
"""

# =============================================================================
# Consumption: step1 (category budgets + need descriptions)
# =============================================================================

CONSUMPTION_NEEDS_BY_CATEGORY_PROMPT = """\
You are designing a household consumption plan for this period.
Time period: one month. This consumption decision corresponds to the household's spending behavior for this month.

Inputs:
- persona (selected fields): {persona}
- Recent household situation: {past_household_status}
- total_budget: {total_budget}
- categories: {categories}
- available_balance: {available_balance}
- expected_income: {expected_income}
- available_budget: {available_budget}
- empirical_category_anchors: {empirical_category_anchors}

Task:
For EACH category (by its category name), do both:
1) Allocate a budget_amount (float) for this category.
2) Provide a list of need_descriptions (strings) that are comprehensive and specific enough to reflect the household's full needs in that category.

Rules:
- Budgets MUST sum exactly to total_budget (allow a small floating error <= 0.01), and should stay close to empirical_category_anchors.
- total_budget MUST be <= available_budget.
- Each category must have at least 1 need description.
- Use English only.

Output STRICT JSON ONLY with this schema:
{{
  "category_plans": [
    {{
      "category": "string (must be one of the categories input list)",
      "budget_amount": 0.0,
      "need_descriptions": ["string", "..."]
    }}
  ],
  "note": "string"
}}
"""

# =============================================================================
# Consumption: step3 (per-category purchase decision; one LLM call per category)
# =============================================================================

PURCHASE_BY_CATEGORY_PROMPT = """\
You are selecting specific products to buy for ONE category given candidate lists and the category budget.
Time period: one month. This purchase decision corresponds to the household's spending behavior for this month.

Inputs:
- persona (selected fields): {persona}
- Recent household situation: {past_household_status}
- category: {category}
- category_budget: {category_budget}
- need_descriptions: {need_descriptions}
- candidates: {candidates}

Task:
Choose products to purchase from candidates and allocate a share of the category budget per chosen product.

Rules:
- You may choose 0..N products.
- Each chosen product MUST come from the provided candidates list.
- You MUST fully cover ALL need_descriptions for this category for this month with your selected products.
- Assume the household will NOT buy any other food, daily necessities, or goods beyond the items you select here (within this retail plan).
- budget_share is a fraction of category_budget (0.0 ~ 1.0).
- The actual spend SHOULD be close to the budget (neither too high nor too low), i.e. sum(budget_share) should be close to 1.0.
- Cover essential needs first; prefer lower price for similar suitability.
- Each product in candidates includes a product_line string with: id + name + description + current_price + storage.
- Use English only.

Output STRICT JSON ONLY with this schema:
{{
  "purchases": [
    {{
      "product_id": "string",
      "budget_share": 0.0,
      "reason": "string"
    }}
  ],
  "note": "string"
}}
"""


PURCHASE_ALL_CATEGORIES_PROMPT = """\
You are selecting specific products to buy for ALL categories given candidate lists and budgets.
Time period: one month. This purchase decision corresponds to the household's spending behavior for this month.

Inputs:
- persona (selected fields): {persona}
- Recent household situation: {past_household_status}
- categories_data: {categories_data}

Each category in categories_data contains:
- category: category name
- budget: budget for this category (if budget > 0, you MUST spend it)
- need_descriptions: list of needs to fulfill
- candidates: list of candidate products (each has product_id, name, price)

Task:
For EACH category with budget > 0, choose products from its candidates and allocate budget shares.

Rules:
- IMPORTANT: For each category with budget > 0, you MUST choose at least 1 product. Do NOT leave purchases empty.
- Each chosen product MUST come from that category's candidates list (use exact product_id).
- budget_share is a fraction of that category's budget (0.0 ~ 1.0).
- Sum of budget_share per category should equal 1.0 (spend the entire budget).
- If a category has budget = 0, you may leave purchases empty for that category.
- Cover essential needs first; prefer lower price for similar suitability.
- Use English only.

Output STRICT JSON ONLY with this schema:
{{
  "categories": [
    {{
      "category": "category_name",
      "purchases": [
        {{
          "product_id": "string",
          "budget_share": 0.0,
          "reason": "string"
        }}
      ],
      "note": "string"
    }}
  ]
}}
"""


# =============================================================================
# Labor: apply-to-jobs decision (member chooses subset of topk)
# =============================================================================

JOB_APPLICATION_DECISION_PROMPT = """\
You are advising a household member on whether to apply to jobs.

Inputs:
- persona (selected fields): {persona}
- Recent household situation: {past_household_status}
- member_info: {member_info}
- jobs (top matches): {jobs}
- month: {month}

Rules:
- You may apply to none, one, or multiple jobs.
- Prefer jobs with better fit (lower loss) and reasonable wage.

Output STRICT JSON ONLY with this schema:
{{
  "apply_job_ids": ["job_id1", "job_id2"],
  "note": "string"
}}
"""


# =============================================================================
# Labor: choose final offer decision (member chooses where to work)
# =============================================================================

JOB_OFFER_DECISION_PROMPT = """\
You are advising a household member who received job offers.

Inputs:
- persona (selected fields): {persona}
- Recent household situation: {past_household_status}
- member_info: {member_info}
- offers: {offers}
- month: {month}

Rules:
- Choose at most one job to accept, or accept none.
- Prefer higher wage given similar suitability; consider hours_per_period if present.

Output STRICT JSON ONLY with this schema:
{{
  "accept_job_id": "job_id_or_null",
  "note": "string"
}}
"""


# =============================================================================
# Persona update: reconcile persona with updated household state
# =============================================================================

PERSONA_UPDATE_PROMPT = """\
You are maintaining a household persona profile for a simulation.

Inputs:
- persona_before: {persona_before}
- past_household_status_text_before: {past_household_status_text_before}
- past_household_status_text_after: {past_household_status_text_after}

Task:
Decide whether the persona profile should be updated given the new household state.

Rules:
- If no persona change is needed, output an empty object {{}} for persona_patch.
- If changes are needed, output ONLY the fields that should change (do NOT repeat unchanged fields).
- The output should be a JSON patch-style object that can be deep-merged into persona_before.
- Use English only.

Output STRICT JSON ONLY with this schema:
{{
  "persona_patch": {{}},
  "note": "string"
}}
"""
