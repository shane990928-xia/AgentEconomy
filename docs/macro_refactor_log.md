# Macro Refactor Log

This file records local implementation steps for the AgentEconomy macroeconomic
simulation refactor. Entries are append-only notes for coordination across
agent/subagent sessions.

## 2026-06-19, Government Procurement Seller Attribution

- Added ProductMarket seller attribution for direct manufacturer-stock sales:
  canonical manufacturing industries now resolve to `mfg_{industry_code}` via
  `get_seller_id(...)` instead of falling through to virtual `market_*`
  receivers.
- Product initialization and `add_product(...)` now normalize industry-code or
  industry-name owners to the default modeled manufacturer firm while preserving
  explicit entrant/seller firm ids.
- Kept intermediate-goods procurement compatible with real owner firm ids by
  returning `mfg_*`, `ret_*`, or `svc_*` owners directly before consulting the
  market price registry.
- Added regression coverage for seller attribution and explicit firm-owner
  preservation in `tests/test_product_market_fallback.py`.
- Validation: `.venv/bin/python -m unittest discover tests` passed with 175
  tests.
- Runtime validation, 20 households / 2 preheat / 4 formal months, LLM profile
  mode:
  `output/validation_macro_profile_p20_m4_seller_income/run_20260619_181850`.
  Final month metrics: nominal GDP `$50.6K`, real GDP `$43.3K`, employment
  `57.5%`, wages `$21.5K`, labor share `42.5%`, household goods consumption
  `$8.5K`, service consumption `$30.9K`, goods demand fulfillment `51.6%`,
  production `$13.5K`, government procurement `$12.6K` on a `$14.2K` budget,
  and no accounting invariant errors/warnings. Sales logs confirm government
  procurement now credits real manufacturers such as `mfg_325`.

## 2026-06-19, Procurement Reweighting and Labor Demand Stabilization

- Government procurement planning now reweights IO procurement demand over SKU
  industries that actually exist in the modeled active catalog.
- Government procurement execution first clears ex-ante planned SKU demand with
  the full procurement budget, then reallocates residual budget only across
  currently available modeled industries.
- Retail demand now enters firm job-posting demand values and labor-market
  priority, so retail firms hire from household-facing demand rather than only
  lagged income.
- Firm job posting now offsets existing employees by SOC before opening new
  positions, preventing duplicate vacancies when current staff already covers
  desired staffing.
- Same-month layoff cooldown now blocks only firms with no current demand
  signal; firms with fresh demand/backlog/production signals can repost jobs.
- Added regression coverage in government policy, labor priority, demand memory,
  and simulator accounting-invariant tests.
- Validation before seller attribution: `.venv/bin/python -m unittest discover
  tests` passed with 174 tests. A 20-household LLM-profile validation reached
  final-month nominal GDP `$54.6K`, employment `60.0%`, labor share `39.4%`,
  government procurement `$13.0K`, and no accounting invariant errors.

## 2026-06-19, Product Industry Normalization and Active Price Scope

- Normalized product manufacturer industries to canonical IO/manufacturing codes
  while keeping human-readable names for product classification/search text.
- Made ProductMarket industry lookup, retailer mapping, demand/supply tracking,
  and Qdrant product indexing use the same canonical manufacturer code.
- Restricted inventory-pressure price adjustment to active or currently sellable
  SKUs instead of the full inactive catalog, avoiding artificial deflation from
  thousands of inactive zero-demand products.
- Added/updated regression coverage for code/name normalization, ProductMarket
  fallback behavior, active SKU filtering, price adjustment scope, and product
  Qdrant indexing.
- Validation at that point: `.venv/bin/python -m unittest discover tests`
  passed with 169 tests.

## 2026-06-19, Actual Production Metrics and Government Demand Ordering

- Changed production accounting to report actual post-production output rather
  than the pre-execution plan when intermediate-goods procurement clips
  production.
- Added the final `production_plan` to `Firm.produce()` results so simulator
  production statistics, GDP output value, and production-gap diagnostics use
  the executed quantities.
- Fixed ProductMarket supply/demand tracking to use manufacturer codes
  consistently. Demand was recorded by manufacturer code while supply and price
  adjustment could be recorded by industry name, which made produced goods look
  like zero supply in supply-demand logs.
- Moved government procurement execution ahead of retail household-channel
  procurement. Government demand is pre-declared final demand and already feeds
  production planning; clearing it after household retail procurement caused
  household channels to consume the same finished-goods stock first.
- Cleared the product snapshot cache after government procurement so retail
  procurement sees post-government inventory.
- Added regression coverage for actual clipped production stats,
  manufacturer-code supply-demand tracking, and inventory rollback on failed
  wholesale/purchase settlement.
- Validation: `.venv/bin/python -m unittest discover -s tests` passed with
  166 tests.

## 2026-06-19, Inventory Rollback Safety

- Added settlement-failure rollback around retailer wholesale procurement and
  household purchase execution. If accounting settlement returns no transaction
  or raises after stock has been reserved, the simulator restores the reserved
  manufacturer or retailer inventory before continuing.
- Added regression coverage for wholesale decline, wholesale exception, and
  purchase exception paths in `tests/test_simulator_consumption_state.py`.
- Validation: `.venv/bin/python -m unittest discover -s tests` passed with
  164 tests.

## 2026-06-19, Retail Channel Demand and Stock Allocation

- Added household retail-channel diversification in `Simulator._build_orders`.
  LLM/profile consumption still chooses SKU and budget; simulator now assigns
  the selling retail channel deterministically when a household basket would be
  too concentrated in one retailer.
- Added config knobs:
  `retail_channel_diversification_enabled`,
  `retail_channel_max_household_share`, and
  `retail_channel_min_purchase_count`.
- Added channel-level demand output under
  `details.demand.by_retailer_product` and
  `details.demand.retail_channel_diversified_orders`.
- Changed retailer procurement to use channel-level demand, so the same SKU can
  be stocked by multiple retail firms when demand was assigned to multiple
  channels.
- Added proportional allocation of scarce manufacturer stock across retailer
  channels by SKU, avoiding fixed dict-order priority for `ret_452`.
- Regression tests added for retail-channel diversification, channel-specific
  procurement, scarce-stock proportional allocation, and config parsing.
- Validation: `.venv/bin/python -m unittest discover -s tests` passed with
  161 tests.
- Runtime validation, 20 households / 2 preheat / 4 formal months, LLM profile
  mode:
  `output/validation_macro_profile_p20_m4_retail_stock_alloc/run_20260619_155256`.
  Final month metrics: nominal GDP `$47.1K`, real GDP `$60.7K`, employment
  `62.5%`, wages `$23.3K`, household goods consumption `$11.7K`, service
  consumption `$32.4K`, goods demand fulfillment `68.1%`, production `$3.0K`,
  procurement `$5.6K`, and 629 diversified retail-channel orders.
  Non-452 retail sales were active in final month:
  `ret_441=$1.2K`, `ret_445=$0.86K`, `ret_4A0=$0.70K`, `ret_722=$0.70K`.

## 2026-06-16

- Continued macro simulation refactor from the prior rule-consumption,
  production-planning, pricing-policy, accounting-invariant, and credit-policy
  scaffolds.
- Fixed duplicate abstract resource purchase recording by keeping
  `AbstractResourceMarket.purchase()` as the single `record_resource_purchase`
  caller.
- Added non-blocking monthly accounting invariant diagnostics to simulator
  records under `details.accounting_invariants`.
- Added firm initialization calibration scaffolding and wired Phase 0 demand
  discovery into initial cash/capital registration via
  `EconomicCenter.register_firm_assets`.
- Validation: `.venv/bin/python -m unittest discover -s tests` passed with
  15 tests; simulator import smoke test passed.

## 2026-06-16, Credit Draw Wiring

- Added a lightweight firm credit facility state inside `EconomicCenter`:
  `firm_credit_limit`, `firm_debt_balance`, and `firm_credit_draws`.
- Added `_draw_firm_credit_if_needed(...)` so firm payment shortfalls are
  explicitly funded by `credit_draw` transactions instead of unexplained free
  negative balances.
- Wired credit draws into firm intermediate-goods purchases, abstract resource
  purchases, wage payments, and generic service transactions.
- Extended `Transaction.type` to allow `credit_draw`.
- Added `EconomicCenter.get_all_firm_debt_balances()` and passed the debt
  snapshot into simulator accounting invariant diagnostics, so negative firm
  cash can be classified as debt-backed rather than unexplained.
- Added regression tests for `EconomicCenter` credit draws, accounting invariant
  credit explanations, and simulator invariant debt-balance forwarding.
- Validation: `.venv/bin/python -m unittest discover -s tests` passed with
  19 tests.

## 2026-06-16, Firm Credit Month-End Settlement

- Added monthly firm credit settlement in `EconomicCenter`:
  interest accrual, automatic repayment from cash above a configured buffer,
  distress-month tracking, and default flags.
- Added config fields:
  `firm_credit_annual_interest_rate`,
  `firm_credit_repayment_cash_buffer`, and
  `firm_credit_default_distress_months`.
- Wired firm credit settlement into both warmup and normal month-end flows before
  redistribution/dividends, so cash can service debt before profit distribution.
- Added monthly record output under `details.firm_credit`.
- Added regression tests for interest accrual, repayment, and default marking.
- Validation: `.venv/bin/python -m unittest discover -s tests` passed with
  21 tests.

## 2026-06-16, Runtime Smoke Test Setup

- Began end-to-end smoke testing with a 1-month, 5-household, 5-firm
  configuration.
- Ray requires `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` in this container because its
  uv runtime-env hook cannot inspect the driver PID.
- Ray also needs `TMPDIR=/tmp` / `RAY_TMPDIR=/tmp` here so local Unix socket
  paths stay below the AF_UNIX path length limit.
- Fixed `load_qdrant_client.load_client()` to default `QDRANT_MODE` to
  `local` and `QDRANT_PATH` to the repository `qdrant_database`, matching
  `ProductMarket`'s local default.

## 2026-06-16, Rules-First Runtime Fixes

- Added `firm_job_posting_use_llm` config, defaulting to `false`.
- Changed `Firm.post_jobs()` so the legacy LLM fallback is only used when
  explicitly enabled; rules-first job posting now returns an empty plan instead
  of calling an API key-dependent model.
- Added baseline firm cash/capital initialization for no-preheat runs, so the
  first month can generate labor demand without requiring Phase 0 demand
  discovery.
- Wired `num_firms` into `create_firms(..., limit=...)` for controlled smoke
  tests while keeping full industry coverage when the limit exceeds available
  industry firms.
- Fixed `EconomicCenter.add_interest_tx()` and `add_redistribution_tx()` to
  update ledgers as well as transaction history.
- Validation: `.venv/bin/python -m unittest discover -s tests` passed with
  24 tests.

## 2026-06-16, LLM Environment Configuration

- Added repository `.env` values for LiteLLM/OpenAI-compatible routing:
  `OPENAI_API_KEY`, `BASE_URL`, `SIMPLE_MODEL`, `STRONG_MODEL`, and `MODEL`.
- Kept macro simulation hard-rule defaults as rules-first:
  `consumption_use_llm=false` and `firm_job_posting_use_llm=false`.

## 2026-06-17, LLM-Constrained Consumption Default

- Changed normal household consumption to LLM-assisted by default:
  `SimulationConfig.consumption_use_llm=true` and
  `config/config_normal.yaml` sets `consumption_use_llm: true`.
- Reworked the LLM consumption path so model outputs are behavioral signals, not
  accounting authority. The final plan is validated by
  `build_constrained_llm_consumption_plan(...)`.
- Added empirical anchors from household profile fields: income (`ER85629`),
  wealth (`ER85692`), historical monthly expenditure (`ER85768`), household size
  (`ER82017`), housing (`ER85701`), healthcare (`ER85747`), and derived retail /
  transportation / utilities / insurance expenditure fields.
- LLM now participates in:
  major budget preference, category needs, and candidate product selection.
  Deterministic code enforces the total budget, category budgets, candidate IDs,
  stock caps, and ledger liquidity.
- Simulator consumption candidate prefetch now applies to both rule and LLM
  modes, so LLM mode can reuse category-level product candidates instead of
  forcing repeated per-household vector searches.
- Added regression tests for constrained LLM budget anchoring, invalid product
  filtering, fake-LLM `consume_v2(use_llm=True)`, and config defaults.
- Validation:
  `.venv/bin/python -m unittest tests.test_household_consumption_policy tests.test_household_vector_search tests.test_simulation_config`
  passed with 14 tests.
- Validation:
  `.venv/bin/python -m unittest discover -s tests` passed with 123 tests.
- Real provider smoke with one household and in-memory candidates passed:
  `is_llm_consumption=true`, `policy_mode=llm_constrained`,
  total budget and anchor budget both `$2,329.53`.

## 2026-06-16, Smoke-Test Calibration Fixes

- Adjusted explicit small-firm sampling so the full retail layer is kept before
  manufacturing/service samples are added. This prevents small smoke configs
  from dropping all household goods purchases because no retailer seller exists.
- Added ProductMarket fallback behavior that skips embedding/vector search when
  `MODEL_PATH` is absent and uses local text/stock matching instead.
- Corrected comprehensive GDP income distribution so `labor_share` uses total
  employee compensation, including government employment compensation, while
  leaving the expenditure-side GDP main metric unchanged.
- Validation: `.venv/bin/python -m unittest discover -s tests` passed with
  26 tests.

## 2026-06-16, No-Embedding Consumption Diversification

- Clarified runtime diagnosis: the no-vector smoke issue is caused by missing
  local embedding model configuration (`MODEL_PATH`), not by a missing Qdrant
  service. The simulation can continue in rules-first fallback mode.
- Improved `ProductMarket.search_by_vector()` fallback so it uses token scores
  plus deterministic query-seeded rotation instead of returning the same first
  SKUs for every household.
- Added household ID to fallback queries only when `MODEL_PATH` is absent, so
  candidate diversification does not alter real embedding searches.
- Spread large rule-based retail budgets over up to 12 SKUs and capped each
  household's per-SKU allocation to avoid the first household exhausting shared
  stock in small smoke tests.
- Corrected comprehensive GDP wage accounting to keep household disposable
  income as net `labor_payment` while using transaction `gross_wage` for income
  components and labor share.
- Added regression tests for SKU diversification, query-seeded fallback
  rotation, and gross-vs-net wage accounting.
- Validation: `.venv/bin/python -m unittest discover -s tests` passed with
  28 tests.

## 2026-06-16, Product Embedding Index Setup

- Added `.env` configuration for the local embedding model:
  `MODEL_PATH=/home/dataset-assist-0/xiaxu/data/model/embedding`.
- Kept `QDRANT_MODE=local` as the default and added local/cloud Qdrant
  connection values for explicit switching.
- Added `scripts/index_products_qdrant.py`, a repeatable product indexing tool
  that reads the product CSV in chunks, generates batched embeddings, and upserts
  lightweight product payloads into Qdrant.
- Added `--start-row` support so interrupted indexing jobs can resume without
  recomputing earlier rows.
- Built the local `products` collection with 29,659 product vectors
  (384-dimensional cosine vectors) under `qdrant_database`.
- Sanity check: Qdrant query for `fresh apple snack food` returned food product
  hits from the local vector collection.
- Validation: `.venv/bin/python -m unittest discover -s tests` passed with
  36 tests after converting the household consumption and indexer tests to
  `unittest` collection style.

## 2026-06-16, Vector-Search Smoke Test

- Ran a 1-month end-to-end smoke test with:
  5 households, `num_firms=100` (66 actual industry firms), no preheat,
  rules-first consumption/job posting, local embedding model, and local Qdrant.
- Command used the repository `.venv` plus Ray container settings:
  `TMPDIR=/tmp RAY_TMPDIR=/tmp RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`.
- Record written to
  `output/smoke_records_full/run_20260616_024502/month_0001.json`.
- Smoke metrics:
  nominal GDP `$112,555.26`, consumption `$72,330.87`, government spending
  `$40,224.39`, gross wages `$23,752.00`, net wages `$20,317.50`, labor share
  `21.10%`.
- Goods demand/execution was active:
  goods orders value `$21,300.36`, service consumption `$51,030.51`, production
  output value `$4,339.20`, and 7 manufacturing firms produced.
- Accounting invariants passed with no errors or warnings:
  82 ledger accounts, 66 registered firms, 29,659 inventory records checked,
  195 resource-purchase transactions checked, zero duplicate resource purchase
  groups, and zero negative balances.
- Runtime note: local Qdrant warns that 20k+ point collections should use Docker
  or Cloud for performance. In this smoke test, the 5-household consumption
  decision stage took about 84 seconds.

## 2026-06-16, Runtime Hardening and Default-Scale Smoke

- Fixed concurrent embedding calls by serializing local tokenizer/model loading
  and forward passes with a re-entrant lock. ProductMarket now also guards
  embedding/vector search failures so the first failure switches to text/stock
  fallback instead of repeatedly failing.
- Reduced `EconomicCenter` and `ProductMarket` Ray actor CPU reservations from
  8 CPUs to 1 CPU each. With 8 CPUs, `EconomicCenter` could consume all local
  Ray resources and prevent `ProductMarket` from scheduling during setup.
- Added shared rule-mode consumption candidate prefetch in
  `Simulator._collect_consumption_plans()`. In rules-first mode, the simulator
  now performs one category-level vector prefetch per consumption collection and
  passes those candidates into each household, instead of issuing repeated
  per-household vector searches for the same categories.
- Added ProductMarket SKU activation hardening:
  active SKUs are updated in the local `_active_sku_set` first, Qdrant payload
  updates use the same stable integer point IDs as `scripts/index_products_qdrant.py`,
  and Qdrant payload update failures no longer make local active filtering
  semantically inactive.
- Added regression coverage for:
  rule-mode precomputed candidates without market search, concurrent ProductMarket
  vector fallback, unknown SKU activation skipping, Qdrant activation point IDs,
  and local activation when Qdrant payload updates fail.
- Validation:
  `.venv/bin/python -m unittest discover -s tests` passed with 47 tests.
- Ray/ProductMarket validation:
  direct Qdrant activation probe for SKU
  `cc76792ea076836235b332c4f085481b` returned `ACTIVATED 1` and
  `LOCAL_ACTIVE True`.

## 2026-06-16, Smoke Results After Hardening

- 5-household resource-fix smoke:
  `output/smoke_records_preheat_resource_fix/run_20260616_142754/month_0001.json`.
  Production value `$514.93`, production quantity `143`, 1 producing firm,
  employment rate `100.0%`, firm wages `$16,406.36`, accounting invariants OK.
- 20-household medium smoke:
  `output/smoke_records_medium_resource_fix/run_20260616_143424/month_0001.json`.
  Production value `$7,039.76`, production quantity `1,883`, 7 producing firms,
  employment rate `60.0%`, firm wages `$60,113.08`, accounting invariants OK.
- Default-household-count smoke:
  `output/smoke_records_default_households_resource_fix/run_20260616_143641/month_0001.json`.
  Configuration was 300 households, 100 requested firms (66 actual industry
  firms), 1 preheat month plus 1 normal month, rules-first consumption/job
  posting, local embedding model, and local Qdrant.
- Default smoke metrics:
  nominal GDP `$6.84M`, consumption `$6.77M`, government spending `$67.98K`,
  production value `$80,231.49`, production quantity `21,557`, 8 producing
  firms, household goods consumption `$109,116.09`, service consumption
  `$6.67M`, gross wages `$439,655.64`, employment rate `30.3%`.
- Default smoke accounting invariants passed:
  377 ledger accounts, 66 registered firms, 29,659 inventory records checked,
  3,567 resource-purchase transactions checked, zero duplicate resource
  purchase groups, zero negative balances, zero errors, zero warnings.
- Remaining calibration note:
  one-month startup runs still show low labor share and negative operating
  surplus because service consumption and initial inventories dominate the
  first measured month. This is now a calibration target rather than a runtime
  blocker.

## 2026-06-16, Firm Initialization Calibration Pass

- Corrected `FirmInitializationCalibrator` formula semantics:
  initial cash is a cost runway (`expected_monthly_cost * cash_multiplier`)
  subject to `firm_min_initial_cash`; target inventory value is based on
  expected revenue, inventory-cover months, and inventory value share; capital
  stock now uses the standard capital-output ratio interpretation (`K/Y`) as
  `annualized_output * firm_capital_output_ratio`.
- Added diagnostics to firm initialization records:
  expected unit price, working-capital requirement, annualized output, cash
  multiplier, inventory cover, capital-output ratio, and inventory value share.
- Added Phase 0 product inventory calibration:
  demanded active SKUs are set to
  `phase0_demand_qty * firm_initial_inventory_cover_months` with a minimum
  active stock floor; non-active SKUs are reset to a small inactive buffer.
- Added config fields:
  `firm_initial_inactive_sku_stock` and
  `firm_initial_min_active_sku_stock`. The default inactive buffer is `10`
  units per SKU, which is far below the old fixed `100` but preserves long-tail
  and government procurement supply.
- Monthly JSON records now include:
  `details.firm_initialization_calibration` and
  `details.initial_inventory_calibration`.
- Added `docs/firm_initialization_calibration.md` with the equations and tuning
  loop for cash, capital, and product inventory calibration.
- Validation:
  `.venv/bin/python -m unittest discover -s tests` passed with 48 tests.
- Smoke validation:
  `output/smoke_records_init_calibration_buffer10/run_20260616_151615/month_0001.json`.
  Phase 0 stock reset changed total product stock from `2,965,900` to `301,999`,
  with 65 active SKUs, zero missing SKU IDs, and inactive stock `10`.

## 2026-06-16, Profit-Closure Planning and First Worker Wave

- Completed four read-only subagent assessments for macro/SFC closure,
  firm production and labor decisions, household consumption and market
  clearing, and banking/agent lifecycle.
- Added `docs/macro_profit_closure_execution_plan.md` to record the staged
  implementation plan.
- Root diagnosis:
  closed-economy firm losses are expected when household saving is positive and
  no government deficit, fixed investment, net exports, or explicit credit
  expansion offsets it; current implementation also has measurable distortions
  that overstate costs or understate sales.
- First P0 worker wave launched:
  wage-hour semantics, intermediate-goods seller income, SFC flow diagnostics,
  and consumption-plan vs actual-consumption state separation.
- Integration policy:
  review each worker patch, run focused tests, run full unittest discovery, and
  append the verified changes here.

## 2026-06-16, Profit-Closure P0 Integration

- Integrated the first P0 worker wave.
- Wage-hour semantics:
  `Job.hours_per_period` is now treated as monthly hours inside
  `LaborMarket`; a 160-hour/month job at `$20/hour` is counted as `$3,200`,
  not `$12,800`, in wage summaries and firm labor-cost queries.
- Intermediate-goods accounting:
  `EconomicCenter.record_intermediate_goods_purchase(...)` now records seller
  firm income when the receiver is a registered firm, while preserving buyer
  production cost and a distinct intermediate-goods transaction category.
- SFC flow diagnostics:
  `accounting_invariants` now attaches a department-level flow summary covering
  household, firm, government, bank_credit, market_or_external, and unknown
  sectors. This is diagnostic-only and does not change invariant pass/fail
  semantics.
- Consumption state separation:
  `_build_orders(...)` no longer applies planned budgets as realized household
  consumption. Household consumption history and lightweight CSV wealth fields
  are updated from actual goods and service execution stats.
- Added `docs/macro_profit_closure_execution_plan.md` with staged phases and
  subagent/worker ownership.
- Validation:
  focused P0 worker tests passed, and
  `.venv/bin/python -m unittest discover -s tests` passed with 56 tests.
- Smoke validation:
  `output/smoke_records_profit_closure_p0/run_20260616_154451/month_0001.json`.
  Accounting invariants passed with zero errors and zero warnings, SFC flow
  residual was `0.0`, gross wages were `$17,016.81`, net wages `$14,805.95`,
  household goods purchases `$7,940.51`, government procurement `$29,134.43`,
  nominal GDP `$109,559.72`, and labor share `15.53%`.
  Normal month government procurement remained active at `$29,136.42` across
  263 items; accounting invariants passed with zero errors and zero warnings.

## 2026-06-16, Profit-Closure P1 Labor Budget Floor

- Removed the fixed dollar floor from firm labor-budget calculation. Labor
  budgets now remain proportional to observed demand, prior income, production
  history, or startup cash where applicable; tiny demand is no longer promoted
  into a minimum payroll commitment.
- Added a regression test that a firm with only `$50` of current demand does
  not post even a part-time job when the demand-implied budget cannot cover the
  configured minimum part-time hours.
- Validation:
  `.venv/bin/python -m unittest discover -s tests` passed with 57 tests.
- Smoke validation:
  `output/smoke_records_profit_closure_p1_laborbudget/run_20260616_155442/month_0001.json`.
  Accounting invariants passed with zero errors and zero warnings, SFC flow
  residual was `1.4551915228366852e-11`, gross wages were `$16,211.79`, net
  wages `$14,113.35`, household goods purchases `$2,460.81`, government
  procurement `$29,132.95`, nominal GDP `$72,162.58`, and labor share `22.47%`.
- Remaining diagnosis:
  this removes one mechanical payroll floor, but the small smoke still shows a
  producing firm with monthly payroll charged against very small realized
  output. The next closure task should align staffing/payroll allocation with
  desired production and add production cash/credit constraints before tuning
  prices.

## 2026-06-16, Profit-Closure P1 Cash, IO, and Production-Cost Diagnostics

- Integrated subagent P1-B:
  production planning now accepts explicit cash, cash reserve, credit limit,
  credit outstanding, and approved credit. Cash-constrained production is capped
  by spendable cash plus available credit when a unit cash cost is available.
- Added a minimal public `BankCreditPolicy` name and stable credit-decision
  fields (`approved_credit`, `rejection_reasons`) for future bank wiring.
- Integrated subagent P1-C:
  `IntermediateGoodsProcurement` now exposes a pure quote/scale planning
  interface (`plan_intermediate_goods_procurement` /
  `quote_intermediate_goods_plan`). It returns planned cost, feasible scale,
  scaled cost, shortages, and planning-only reservations without mutating stock
  or ledgers.
- Wired production planning to current EconomicCenter snapshots:
  simulator reads firm cash balances, credit limits, and debt balances before
  building production plans. If `production_unit_cash_cost` is not explicitly
  configured, it estimates unit cash cost from targeted manufacturer prices and
  `production_unit_cash_cost_share`.
- Added `details.firm_profit_pressure` to monthly records. This diagnostic
  decomposes realized firm income, expenses, wages, production output,
  production input cost, and sales gaps by firm.
- Corrected production-cost accounting for production statistics:
  `firm_production_cost` / `total_production_cost` now represent intermediate
  and abstract-resource input costs for value-added/GDP diagnostics. Labor is
  still paid and recorded separately through wage transactions and is preserved
  under `firm_labor_cost_in_production` and
  `firm_total_production_cost_with_labor` for diagnostics.
- Added config field:
  `production_unit_cash_cost_share` (default `0.6`), used only when no explicit
  `production_unit_cash_cost` is configured.
- Validation:
  `.venv/bin/python -m unittest discover -s tests` passed with 74 tests.
- Smoke validation:
  `output/smoke_records_profit_closure_p1_cash_io_costfix/run_20260616_162739/month_0001.json`.
  Accounting invariants passed with zero errors and zero warnings, SFC flow
  residual was `0.0`, gross wages were `$16,211.91`, net wages `$14,421.16`,
  household goods purchases `$2,460.82`, government procurement `$29,131.27`,
  nominal GDP `$72,167.03`, and labor share `22.46%`.
- Production-cost diagnostic change in the final smoke:
  total production input cost was `$33.67`, production labor diagnostic was
  `$1,158.05`, and total cost with labor was `$1,191.72`. Firms with production
  input cost above output fell to `0`; aggregate production-cost-to-output ratio
  fell to `0.233`. Firms with realized income below wages remain `6`, so the
  next issue is demand/sales coverage and staffing adjustment, not duplicated
  labor in production cost.

## 2026-06-16, Runtime Web Dashboard

- Added a Flask dashboard under `agenteconomy/web/` for runtime monitoring and
  replay from `output/**/run_*` records.
- The dashboard exposes:
  `/api/health`, `/api/runs`, `/api/latest`, and `/api/run?path=...`.
- Added a structured simulator phase stream:
  every `Simulator._time_block(...)` now writes best-effort `start` and `end`
  events to `stage_events.jsonl` in the current run directory.
- The page shows the current decision phase when a simulation is running, plus
  historical monthly replay, macro paths, firm wage/sales pressure, credit
  state, and accounting invariant diagnostics.
- Added `tests/test_web_dashboard.py` for record summarization, stage-event
  parsing, and Flask API routing.
- Validation:
  `.venv/bin/python -m unittest discover -s tests` passed with 75 tests.
- Stage-event smoke:
  `output/web_dashboard_stage_smoke/run_20260616_165041/stage_events.jsonl`
  contains paired `dashboard-smoke` start/end events.

## 2026-06-16, Labor Lifecycle Profit-Closure Guardrails

- Switched private-firm layoff protection from hard-coded floors to config:
  `firm_layoff_min_wage_cap` and `firm_layoff_min_employees_to_keep`, both
  defaulting to zero in `config_normal.yaml`.
- Changed `Simulator._process_layoffs()` so zero-revenue firms are no longer
  skipped and can be laid off down to the configured wage cap.
- Changed `LaborMarket.layoff_to_budget()` to treat `target_wage_cap` as a wage
  spending ceiling, not a lower bound. Because employees are indivisible, the
  resulting wage bill may fall below the cap.
- Changed `Firm.post_jobs()` to synchronize stale open positions even when the
  new desired job plan is empty, and to close obsolete SOC openings when a new
  plan only covers part of the previous openings.
- Added regression coverage for:
  zero-budget layoffs, tiny nonzero wage-cap layoffs, stale opening cleanup,
  obsolete SOC cleanup, simulator zero-revenue layoffs, and config defaults.
- Validation:
  `.venv/bin/python -m unittest discover -s tests` passed with 80 tests.
- Remaining next closure issue from subagent review:
  execution-layer credit still lets wages and purchases continue after credit
  is exhausted or a firm has defaulted; the next patch should make wage/resource/
  intermediate payments fail cleanly when no approved funding remains, and then
  make `_pay_wages()` count only successful wage transfers.

## 2026-06-16, Execution-Layer Credit Hard Constraints

- Added `EconomicCenter._ensure_firm_payment_funding(...)` as a shared funding
  gate for firm payments.
- Firm credit draws are now blocked once a firm is marked defaulted.
- Wage payments, abstract-resource purchases, intermediate-goods purchases, and
  firm service payments now fail cleanly when cash plus available credit cannot
  cover the payment. Failed payments do not move ledger balances and do not
  create income/expense/tax/wage transactions.
- Updated `Simulator._pay_wages()` so household income and wage statistics only
  include wages whose `EconomicCenter.process_wage(...)` call succeeded.
- Added regression coverage for defaulted wage-payment failure, exhausted-credit
  resource/intermediate/service payment failure, and simulator wage statistics
  excluding failed wage transfers.
- Validation:
  `.venv/bin/python -m unittest discover -s tests` passed with 85 tests.
- Remaining next closure issues:
  retail sales still use global SKU stock rather than retailer-owned procured
  inventory, and production demand expectation should include unmet demand
  rather than only realized sales.

## 2026-06-16, Production Demand Memory, Default Closure, and Retail Inventory

- Added production demand memory to `Simulator`: production now uses planned
  demand plus realized sales and recorded unmet demand, so stockouts do not
  mechanically depress next-period production expectations.
- Extended unmet-demand records with product and seller metadata, allowing
  downstream aggregation by product for production planning.
- Propagated firm credit default into the real economy:
  defaulted firms have open labor positions closed, are laid off to zero wage
  budget, are skipped during job posting, and are skipped during production.
- Added firm credit checkpoint support:
  credit limits, debt balances, distress month counts, and default flags are
  saved and restored with simulator checkpoints.
- Added retailer-owned inventory to `ProductMarket` and made household retail
  purchases consume the selected retailer's inventory rather than falling back
  to global manufacturer stock. Failed purchase transactions roll inventory
  reservations back.
- Updated retailer procurement, government procurement, and intermediate-goods
  procurement to clip actual purchases by available manufacturer stock and to
  avoid creating transactions for unavailable quantities.
- Added regression coverage for defaulted-firm labor/production closure,
  production planning with unmet demand, demand-memory persistence, retailer
  inventory clipping, retailer-inventory checkpoint restore, and no fallback
  from retailer orders to manufacturer stock.
- Validation:
  `.venv/bin/python -m unittest discover -s tests` passed with 95 tests.
- Smoke validation:
  `output/smoke_records_macro_closure_p2/run_20260616_180220/month_0001.json`.
  Accounting invariants passed with zero errors and zero warnings, SFC flow
  residual was `0.0`, negative firm cash count was `0`, and negative inventory
  count was `0`.
- Smoke macro diagnostics:
  gross wages were `$10,114.58`, household goods purchases were `$14,030.90`,
  household service consumption was `$147,519.66`, government procurement was
  `$29,132.75`, nominal GDP was `$190,683.31`, and labor share was `5.30%`.
- Firm pressure diagnostic:
  aggregate realized firm income was `$166,409.14` against wage expense
  `$10,114.58`. Only one firm had realized income below wages in the small
  smoke run (`mfg_325`, gap `$801.55`). The remaining closure issue is
  production/staffing scale calibration and service/GDP composition, not
  unconstrained negative balances.

## 2026-06-16, Labor Budget Gates and Value-Calibrated Production Capacity

- Added firm labor-budget controls:
  `firm_min_job_budget_coverage` and
  `firm_allow_cash_based_startup_hiring`.
- Changed firm job posting so current cash is no longer treated as startup
  revenue by default. Cash-based startup hiring is now opt-in.
- Added a minimum affordable-job threshold before posting any private-firm job.
  This blocks tiny demand signals from creating a real monthly payroll.
- Added `LaborHour.wage_per_hour` and populated it from matched labor-market
  jobs when syncing firm employee lists. Production planning can now use the
  actual matched wage and hours instead of only employee counts.
- Added `production_value_calibrated_labor_productivity`. When enabled, the
  simulator converts the fixed labor capacity floor into a value-consistent
  units-per-hour rate for the targeted SKU mix:
  `(hourly wage / compensation ratio) / average target unit price`.
  This prevents low-price SKUs from being constrained to a few hundred dollars
  of output while paying a full monthly wage.
- Extended `firm_profit_pressure` diagnostics with inventory-adjusted income
  and gaps. The old cash-sales gap remains visible, but the new diagnostic
  distinguishes a true payroll/revenue problem from produced output sitting in
  inventory.
- Validation:
  `.venv/bin/python -m unittest discover -s tests` passed with 100 tests.
- Smoke validation:
  `output/smoke_records_macro_closure_p4/run_20260616_182502/month_0001.json`.
  Accounting invariants passed with zero errors and zero warnings, negative
  firm cash count was `0`, negative inventory count was `0`, and defaulted firm
  count was `0`.
- Smoke macro diagnostics:
  gross wages were `$14,370.30`, household goods purchases were `$47,545.30`,
  household service consumption was `$181,932.00`, government procurement was
  `$29,130.58`, production output was `$9,871.74`, and production input cost
  was `$1,337.49`.
- Closure result:
the prior `mfg_325` low-price-SKU wage/output mismatch disappeared. The only
remaining cash-sales gap was `mfg_334`, but its inventory-adjusted income was
`$5,753.40` against wage plus production-input cost of `$4,026.34`, so the
adjusted gap was `0`. This is unsold product/inventory timing rather than an
unlimited-loss wage mechanism.

## 2026-06-17, Small-Sample Demand Closure and GDP Service Accounting

- Added a same-month layoff cooldown for private job posting. Firms that lay
  off workers during the month have open positions closed and cannot
  immediately repost jobs in the same labor-market round.
- Parameterized government procurement, regular government labor budget, and
  public employment with household-count scaling. Small smoke simulations no
  longer inherit full-size economy floors that overwhelm household demand.
- Added household monthly flow caps and changed default rule-based consumption
  budgets to use current liquid balance only. Expected income is still an
  input to the consumption function, but it is no longer double-counted as
  immediately spendable cash.
- Made abstract service/resource purchases commit local demand and transaction
  records only after the EconomicCenter transfer succeeds.
- Fixed comprehensive GDP accounting:
  household service consumption is counted as final service output; business
  resource purchases are counted as service output and buyer intermediate
  input; goods sold from initial or previous inventory reduce inventory
  investment instead of creating current-period output; VAT is counted once;
  corporate tax remains a fiscal memo rather than a second income-side GDP
  component; and purchase transaction amounts default to ex-tax values when
  VAT is recorded separately.
- Validation:
  `.venv/bin/python -m unittest discover -s tests` passed with 118 tests.
- Smoke validation:
  `output/smoke_records_macro_closure_p10/run_20260617_142726/month_0001.json`.
  Comprehensive GDP closed across expenditure, production, and income:
  `$33,477.8336268481`, `$33,477.833626848085`, and
  `$33,477.833626848085`; max discrepancy was `1.4551915228366852e-11`.
  Accounting invariants passed with zero errors and zero warnings, SFC flow
  residual was `0.0`, negative firm cash count was `0`, and negative inventory
  count was `0`.
- Firm pressure diagnostic:
  aggregate realized firm income was `$43,104.50` against wage expense
  `$5,366.44` (`income_to_wage_ratio=8.03`). No firm had realized income below
  wages, and no inventory-adjusted wage gap remained in the smoke run.

## 2026-06-17, Public Runtime Dashboard Deployment

- Updated the dashboard run-selection policy:
  active simulation runs still sort first, but when no run is active the
  default latest run now prefers replayable runs with monthly records over
  inactive stage-only smoke runs.
- Added API metadata for run activity:
  `stage`, `active_stage`, and `is_active` are included in `/api/runs` and
  reused by both Flask and dependency-free standalone dashboard servers.
- Added regression coverage for dashboard latest-run selection:
  inactive stage-only runs no longer hide replayable monthly runs, while an
  actually active stage run still becomes the default live view.
- Synced `agenteconomy/web` and the replayable p10 smoke run to
  `root@192.236.145.33:/opt/agenteconomy-dashboard`.
- Restarted the public standalone dashboard:
  `python3 -m agenteconomy.web.standalone_dashboard --host 0.0.0.0 --port 7860
  --output-root /opt/agenteconomy-dashboard/output`, PID `146089`.
- Public URL:
  `http://192.236.145.33:7860/`.
- Validation:
  `.venv/bin/python -m unittest tests.test_web_dashboard` passed with 3 tests.
  Public `/api/health` returned ok with output root
  `/opt/agenteconomy-dashboard/output`.
  Public `/api/runs?limit=3` now returns
  `smoke_records_macro_closure_p10/run_20260617_142726` first.
  Public `/api/latest` returns 3 replay records and 78 stage events for the
  p10 run. Public HTML contains the runtime, replay, and decision-stage UI,
  and public `dashboard.js` serves the replay and live-refresh handlers.

## 2026-06-17, LLM Provider Smoke Test

- Ran a real provider smoke test against the configured LiteLLM router.
- Initial result failed because `BASE_URL=https://zgc.apihy.com` returned the
  gateway HTML page instead of the OpenAI-compatible chat-completions API.
- Updated `.env` to use `BASE_URL=https://zgc.apihy.com/v1`.
- Retested both configured routes:
  `simple` and `strong` both returned strict JSON for the same minimal prompt.
  Observed latencies were about `3.06s` and `2.40s`.

## 2026-06-18, LLM Consumption and Macro Reasonableness Pass

- Fixed GDP inventory accounting so goods inventory drawdown uses manufacturer
  or base value while household consumption remains at retail value. Retail
  margin is now recorded as distribution/service output rather than negative
  inventory investment.
- Added purchase metadata for `base_unit_price`, `base_amount`, and
  `retail_margin`.
- Changed production demand memory to include current demand, planned demand,
  realized sales, and unmet demand together.
- Restricted ProductMarket price adjustment and available-SKU queries to the
  active/sellable SKU set when active filtering is enabled.
- Changed inactive SKU initial stock from `10.0` to `0.0`.
- Real LLM smoke:
  `output/smoke_records_macro_reasonable_llm_p1/run_20260618_014105`.
  Formal-month GDP was `$3,564.43`, consumption rate was `100.4%`, inventory
  investment was `$-14.24`, and inactive stock was `0.0`.
- Validation:
  `.venv/bin/python -m unittest discover -s tests` passed with 127 tests.
- Full report:
  `docs/macro_reasonableness_report.md`.

## 2026-06-18, Low-Frequency LLM Consumption Profile Mode

- Diagnosed the consumption bottleneck:
  full LLM consumption used Step0/Step1/Step3 online calls for every household
  in Phase0, every preheat month, and every formal month. Previous real-LLM
  smoke timings were dominated by Step3 and reached tens of seconds per
  single household consumption round, with a prior 180s timeout.
- Added `consumption_llm_mode` and `consumption_profile_refresh_months`:
  `monthly` preserves the old full Step0/Step1/Step3 behavior, `profile`
  generates a cached LLM preference profile and uses monthly budget/price/stock
  constrained consumption, and `off` runs the rule path only.
- Added a consumption profile prompt and household profile cache for:
  `price_sensitivity`, `liquidity_preference`, `habit_strength`, and
  `essential_bias`. LLM profile failures fall back to deterministic persona and
  household-state heuristics without stopping the simulation.
- Set `config/config_normal.yaml` to the long-run profile mode while keeping
  direct `Household.consume_v2(use_llm=True)` default behavior compatible with
  the old monthly LLM path.
- Validation:
  `.venv/bin/python -m unittest discover -s tests` passed with 129 tests.
- Real-provider profile smoke:
  `output/smoke_records_consumption_profile_p2/run_20260618_141428`.
  Phase0 demand discovery took `13.64s` for 2 households; preheat consumption
  decision took `1.58s`; formal-month consumption decision took `0.04s` after
  profile caching. Formal-month comprehensive GDP was `$8,593.96`,
  consumption rate was `104.7%`, inventory investment rate was `-5.6%`, and
  accounting invariants reported 0 errors and 0 warnings.

## 2026-06-18, Production Backlog Labor Signal

- Added a persisted firm-level production backlog signal:
  `_last_production_gap_value_by_firm`.
- Production planning now records `desired_output_value`,
  `feasible_output_value`, and `production_gap_value` by firm. The gap is
  valued from the targeted SKU mix so labor demand can compare current sales
  demand and unsatisfied production capacity in the same dollar units.
- Job posting now uses
  `max(current_demand_value, production_gap_value * firm_labor_backlog_demand_share)`
  for manufacturing labor demand, with malformed checkpoint values ignored.
- Labor matching priority now includes backlog-derived demand, so firms with
  unresolved production gaps have a better chance of filling jobs in the next
  month.
- Checkpoints now save and restore `_last_production_gap_value_by_firm`.
- Added regression coverage for multi-firm backlog priority normalization,
  priority stamping on job objects, production gap value generation, and
  checkpoint save/restore.
- Validation:
  `.venv/bin/python -m unittest discover -s tests` passed with 146 tests.
- Six-month validation run:
  `output/validation_macro_profile_p20_m6_backlog_labor/run_20260618_162012`.
  Compared with the previous p20 profile run
  `output/validation_macro_profile_p20_m6_consumption_labor/run_20260618_155757`,
  month 6 improved from employment `40.0%` to `52.5%`, production output
  `$4,426.5` to `$9,590.8`, C/GDP `101.7%` to `77.8%`, and inventory
  investment/GDP `-8.1%` to `+3.5%`.
- Month 6 comprehensive GDP was `$68,425.5`, labor share `52.8%`, household
  goods consumption `$14,886.9`, household service consumption `$37,180.6`,
  government spending `$12,803.2`, and accounting invariants passed with zero
  errors and zero warnings.
- Remaining issue: some individual firms still have realized-sales-to-wage
  pressure in a given month, but aggregate realized firm income/wage was `2.54`
  and inventory-adjusted wage gaps were zero. Next pass should improve retail
  margin/wage sizing and government procurement as an ex-ante demand signal.
