# LLM-Constrained Household Consumption Design

Date: 2026-06-17

## Decision Boundary

Normal simulation should call an LLM for household consumption when
`simulation.consumption_use_llm` is enabled. The LLM is used for soft behavioral
variation:

- interpreting persona and recent household situation,
- expressing consumption needs by category,
- choosing among already available candidate products,
- producing short explanations for preference shifts.

Hard economic mechanics stay deterministic:

- total spending envelope,
- cash buffer and liquidity constraint,
- tax, ledger, and transaction clearing,
- inventory mutation and stock caps,
- product-market candidate validity,
- GDP/accounting aggregation.

This keeps the model "large-model agent based" without letting the model invent
cash, bypass markets, or decide accounting identities.

## Empirical Anchor

The consumption validator first builds a rule anchor from household state and
PSID-style fields already loaded into `Household.csv_values`:

- `ER85629`: monthly family income,
- `ER85692`: net wealth / balance anchor,
- `ER85768`: historical monthly total expenditure,
- `ER82017`: household size,
- `ER85701`: housing expenditure,
- `ER85747`: healthcare expenditure,
- `expenditure_retail_merchandise`,
- `expenditure_transportation`,
- `expenditure_utilities`,
- `expenditure_insurance`.

The rule anchor computes:

- available monthly budget from current ledger balance,
- target cash buffer from income, household size, and liquidity preference,
- total monthly budget from MPC, historical expenditure, wealth draw, and habit,
- major spending buckets,
- retail category budgets,
- candidate product preferences under price/stock constraints.

## LLM Flow

The active LLM path is:

1. Build the rule anchor.
2. Ask the LLM for major budget preferences with the anchor in the prompt.
3. Constrain major budgets back to the rule anchor total.
4. Ask the LLM for category needs using constrained retail budget and category
   anchors.
5. Search or reuse product candidates.
6. Ask the LLM to select among candidates.
7. Validate purchases:
   - only candidate `product_id`s are accepted,
   - category budgets cannot exceed constrained budgets,
   - per-SKU allocation is capped by available stock,
   - total budget remains the empirical anchor.

The LLM can shift shares and choose products, but the final plan is
`policy_mode=llm_constrained`.

## Current Validation

Commands run:

```bash
.venv/bin/python -m unittest tests.test_household_consumption_policy tests.test_household_vector_search tests.test_simulation_config
.venv/bin/python -m unittest discover -s tests
```

Results:

- targeted tests: 14 passed,
- full test suite: 123 passed.

Real provider smoke:

- one household,
- fixed in-memory product candidates,
- no Qdrant dependency,
- `consume_v2(use_llm=True)`.

Observed result:

- `is_llm_consumption=true`,
- `is_rule_based=false`,
- `policy_mode=llm_constrained`,
- total budget `2329.53`,
- major budget sum `2329.53`,
- anchor total budget `2329.53`,
- all purchases used valid candidate IDs.

## Long-Run Emergence Status

The simulator is now closer to a long-run ABM/SFC-style economy because
household consumption is heterogeneous, budget-constrained, and connected to
ledger cash and product inventories. This is necessary but not sufficient to
claim robust macro emergence.

Long-run validation still needs:

- multi-month runs with LLM consumption enabled,
- aggregate consumption/income and budget-share checks against empirical
  anchors,
- firm revenue/wage/profit closure checks,
- inventory cycle and price response checks,
- credit/default sensitivity checks,
- seed-to-seed stability checks.

