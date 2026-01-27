"""
Centralized prompt templates for LLM-powered decisions.

Design principle:
- Keep inputs broad and extensible (household_info/member_info) rather than narrow fields.
- Enforce strict JSON outputs to make downstream logic reliable.
"""

# =============================================================================
# Consumption: step0 (major budget allocation)
# =============================================================================

CONSUMPTION_MAJOR_BUDGET_PROMPT = """\
  You are allocating a household's total consumption budget into major buckets for this period.
Time period: one month. This budget allocation corresponds to the household's spending behavior for this month.

Inputs:
- household_info: {household_info}
- Recent household situation: {past_household_status}

Task:
1) Infer the household's total consumption budget (total_budget).
2) Allocate that total_budget into the following budget buckets:
   - Retail merchandise
   - housing
   - healthcare
   - transportation
   - utilities
   - insurance

Rules:
- All budgets are floats.
- The sum of all bucket budgets MUST equal total_budget (allow a small floating error <= 0.01).
- Use English only.

Output STRICT JSON ONLY with this schema:
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
- household_info: {household_info}
- Recent household situation: {past_household_status}
- total_budget: {total_budget}
- categories: {categories}

Task:
For EACH category (by its category name), do both:
1) Allocate a budget_amount (float) for this category.
2) Provide a list of need_descriptions (strings) that are comprehensive and specific enough to reflect the household's full needs in that category.

Rules:
- Budgets MUST sum exactly to total_budget (allow a small floating error <= 0.01).
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
- household_info: {household_info}
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


# =============================================================================
# Labor: apply-to-jobs decision (member chooses subset of topk)
# =============================================================================

JOB_APPLICATION_DECISION_PROMPT = """\
You are advising a household member on whether to apply to jobs.

Inputs:
- household_info: {household_info}
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
- household_info: {household_info}
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


def build_firm_post_job_prompt(firm):
    pass
