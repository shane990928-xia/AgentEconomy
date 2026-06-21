# Macro Reasonableness Report

Date: 2026-06-18

## Scope

This report covers the latest macro-simulation pass focused on household
consumption, GDP decomposition, inventory accounting, active SKU supply, and
LLM consumption execution.

## Changes Made

- GDP inventory accounting now uses the same price basis for goods output and
  goods inventory drawdown. Household purchases keep final retail value in
  consumption, while the manufacturer/base value is used for goods inventory.
  Retail margin is counted as distribution/service output instead of being
  subtracted as negative goods inventory.
- Household purchase transactions now record `base_unit_price`, `base_amount`,
  and `retail_margin` metadata.
- Production demand memory now merges current demand, planned demand, realized
  sales, and unmet demand with a max rule instead of letting stale planned
  demand hide current-period demand.
- ProductMarket supply/demand price adjustment now acts on active/sellable SKUs
  rather than the full product catalog.
- `get_available_skus()` and industry SKU queries now respect active SKU filter
  mode. This prevents government procurement and other non-vector queries from
  buying inactive catalog inventory after Phase 0.
- Default inactive SKU initial stock was changed from `10.0` to `0.0`, so
  unactivated products do not create hidden free supply.

## Preheat Assessment

Preheat is still necessary in the current architecture.

Phase 0 discovers which SKUs households actually demand, activates those SKUs,
and calibrates initial inventory and firm assets from the discovered demand.
The warmup month then lets labor matching, prices, service consumption, taxes,
and inventories move away from artificial initial conditions.

Preheat should not be used as a formal macro observation window. It is an
initialization and burn-in mechanism. It can be shortened or replaced later only
after active SKU sets, initial inventory, firm cash/capital, and sector demand
are loaded from a deterministic calibration artifact.

## LLM Consumption Result

LLM consumption is integrated and was tested through the real runtime path.

- `output/smoke_records_macro_reasonable_llm_p0/run_20260618_012952`
  used 2 households, 1 preheat month, and 1 formal month with
  `consumption_use_llm=True`.
  Formal-month LLM Step3 completed for both households. Warmup had one
  180-second LLM timeout and fallback.
- `output/smoke_records_macro_reasonable_llm_p1/run_20260618_014105`
  used 1 household, 1 preheat month, and 1 formal month after active-SKU and
  inactive-stock fixes. All LLM stages completed.

Observed runtime cost is high: even 1-2 households take minutes because each
household can make multiple online LLM calls per consumption round. For
long-run simulations, LLM should generate or refresh behavior parameters,
persona traits, preference weights, product-consideration rules, and narrative
explanations. Monthly budget allocation, affordability, clearing, taxes,
inventory, and accounting should remain deterministic/rule-based.

## Macro Validation

### Before active-stock fix

Run: `output/smoke_records_macro_reasonable_llm_p0/run_20260618_012952`

Formal month:

- Nominal GDP: `$9,582.67`
- Consumption rate: `105.6%`
- Inventory investment rate: `-11.4%`
- Goods consumption: `$1,570.91`
- Service consumption: `$8,421.12`
- Wage expense: `$1,182.27`
- Firm aggregate income-to-wage ratio: `8.90`
- No aggregate wage-payment crisis; zero inventory-adjusted aggregate gap.

Issue remaining: inactive SKU stock still existed at `10.0` per inactive SKU,
leaving `296,176` total stock units after initialization.

### After active-stock fix

Run: `output/smoke_records_macro_reasonable_llm_p1/run_20260618_014105`

Formal month:

- Nominal GDP: `$3,564.43`
- Consumption rate: `100.4%`
- Government rate: `0.0%`
- Inventory investment rate: `-0.4%`
- Inventory investment: `$-14.24`
- Goods consumption: `$41.97`
- Service consumption: `$3,533.34`
- Inflation: `0.079%`
- Wage expense: `$353.59`
- Firm aggregate income-to-wage ratio: `10.15`
- Inventory initialization: `53` active SKUs, inactive stock `0.0`, total stock
  after initialization `117` units, target inventory value `$615.55`.

This is the first smoke run where household consumption and GDP decomposition
are in a reasonable range. The previous "consumption over 100% of GDP" problem
was primarily an inventory and accounting-basis problem, not an LLM budget
problem.

## Remaining Issues

- Government procurement became too low after strict active-SKU filtering
  because it only buys current on-hand active inventory. It should be converted
  into an ex-ante demand signal that participates in production planning before
  procurement clears.
- In the p1 formal month, goods demand was small and production did not fire.
  This is plausible for a 1-household smoke run but should be rechecked with
  more households and more formal months.
- A single firm still had a local sales-to-wage gap in the p1 formal month, but
  the aggregate firm income-to-wage ratio was healthy. This should be monitored
  with firm exit/default logic rather than treated as a macro failure.
- Online LLM latency is too high for long simulations at full population size.
  The architecture should cache LLM outputs and move recurrent monthly choice
  to deterministic policy functions.

## Verification

- `.venv/bin/python -m unittest discover -s tests`
  passed: `127` tests OK.
- Real LLM runtime smoke completed:
  `output/smoke_records_macro_reasonable_llm_p1/run_20260618_014105`.

## Recommended Next Implementation Tasks

1. Add government procurement demand to pre-production demand planning, then
   clear purchases after production and retail procurement.
2. Add a cached household behavior-profile layer: LLM produces persona
   parameters and preference weights occasionally; monthly consumption uses a
   deterministic budget/utility/affordability policy.
3. Run a non-LLM policy-scale smoke with 8-20 households and 6-12 formal months
   to validate convergence of consumption rate, wage share, employment, CPI,
   firm defaults, and inventory investment.
4. Run a small LLM-refresh smoke where LLM updates behavior profiles every N
   months rather than every household every month.
