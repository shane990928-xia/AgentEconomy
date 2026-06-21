# Macro Profit-Closure Execution Plan

This plan tracks the next refactor pass for the recurring issue where firm
sales are too small relative to wage and production costs. The goal is not to
force every firm to be profitable. The goal is to make losses explainable by
auditable macro flows, production plans, inventory investment, and explicit
credit constraints.

## Diagnosis

The current gap has three layers:

1. Real macro closure: in a closed economy, if households save part of wages
   and there is no offsetting government deficit, fixed investment, net exports,
   or sustained credit expansion, the firm sector must absorb the deficit.
2. Implementation distortions: some flows currently overstate costs or understate
   sales, especially wage-hour accounting and intermediate-goods seller income.
3. Decision coupling: hiring, production, inventory, pricing, household demand,
   and credit are present as partial policies, but they are not yet tied into one
   strict monthly constraint loop.

## Target Architecture

The simulation should converge on a rules-first ABM plus SFC kernel:

- Household decisions use deterministic budget constraints, MPC, cash buffers,
  actual ledger balances, history, and persona soft parameters.
- Firm decisions use demand expectations, target inventory, labor, capital, IO
  inputs, cash, and approved credit before committing production or payroll.
- Product markets clear deterministic purchase intents, inventory, substitution,
  shortages, seller assignment, and price feedback.
- The bank provides explicit credit facilities, interest, repayment, default, and
  eventually a real balance sheet.
- Government spending, transfers, taxes, and debt are represented as a budget
  constraint, with both balanced-budget and deficit-financed modes.
- LLMs are limited to soft behavior parameters, strategy narratives, shocks, and
  explanations. They must not decide taxes, ledger movements, clearing, inventory
  updates, credit balances, or prices.

## Phase 0: Accounting Truth Before Behavior

These tasks can run first because they fix measurement and obvious accounting
distortions without changing the high-level model.

### P0-A Wage-Hour Semantics

Files:
- `agenteconomy/center/LaborMarket.py`
- tests

Goal:
- Treat `Job.hours_per_period` as monthly hours across wage payment, labor-cost
  queries, and production-cost accounting.
- Preserve fallback behavior for legacy jobs without `hours_per_period`.

Acceptance:
- A 160-hour/month job at 20 per hour costs exactly 3200 in wage payment and
  firm labor-cost queries.

### P0-B Intermediate-Goods Seller Income

Files:
- `agenteconomy/center/Ecocenter.py`
- tests

Goal:
- Intermediate-goods purchases must book buyer production cost and seller firm
  income.
- The transaction remains distinguishable from household final consumption.

Acceptance:
- A 100 intermediate purchase raises buyer expense/production cost by 100,
  seller monthly income by 100, and preserves ledger balance.

### P0-C SFC Flow Diagnostics

Files:
- `agenteconomy/utils/accounting_invariants.py`
- `agenteconomy/simulation/simulator.py`
- tests

Goal:
- Add department-level flow diagnostics for household, firm, government,
  bank_credit, market_or_external, and unknown.
- Do not block simulation; expose residuals in accounting diagnostics.

Acceptance:
- Toy transactions have department net flows whose sum is near zero.

### P0-D Consumption Plan vs Actual State

Files:
- `agenteconomy/simulation/simulator.py`
- tests

Goal:
- Do not mutate household wealth/state when only building planned orders.
- Update consumption history from actual executed sales/service spending.

Acceptance:
- `_build_orders` does not apply planned consumption as realized spending.

## Phase 1: One-Period Decision Constraint Loop

After P0 is green, wire existing policies into a consistent monthly loop.

Tasks:
- Enable production cash and credit constraints using rolling unit cost.
- Make job posting budgets depend on expected demand and feasible financing.
- Move IO procurement toward quote/reserve/scale/execute instead of purchasing
  before bottleneck scaling.
- Use current manufacturer price for wholesale and clarify retail inventory.
- Add explicit diagnostics for wage bill, expected revenue, planned production
  cost, credit draw, and realized sales by firm.

Acceptance:
- A no-demand firm does not continue expanding payroll.
- A cash-constrained firm cannot produce beyond cash plus approved credit.
- A demand shock leads payroll to adjust within one to two periods.

## Phase 2: Market Clearing and Sales Coverage

Tasks:
- Introduce canonical consumption categories and hard product-category filters.
- Add deterministic substitution rules for stockouts.
- Replace simulator-order sequential stock decrement with ProductMarket batch
  clearing.
- Split monthly sales stats into household retail sales, manufacturer wholesale
  sales, government procurement, services/resources, and intermediate sales.

Acceptance:
- No overselling under excess demand.
- Firm ledger income can be reconciled against sales categories.
- Unmet demand is explicit and feeds demand expectations.

## Phase 3: SFC Closure Instruments

Tasks:
- Add government balanced-budget and deficit-financed modes.
- Make transfers, procurement, public wages, taxes, and government debt use one
  treasury rule.
- Convert fixed capital investment from a cash-to-capital mutation into a
  payment to capital-goods firms.
- Add COGS/inventory accounting so unsold inventory is not treated as the same
  thing as realized sales failure.

Acceptance:
- With household saving and no G/I/X, the test explicitly shows firm-sector
  deficit.
- With government deficit or fixed investment enabled, the sales gap moves by
  the expected amount and appears in another sector's balance sheet.

## Phase 4: Banking, Default, Entry, and Exit

Tasks:
- Replace free overdrafts with `ensure_liquidity()` and explicit credit denial.
- Use `bank_credit_policy.py` to compute limits, rates, and rejection reasons.
- Persist credit/debt/default state in checkpoints.
- Add firm lifecycle states: active, distressed, defaulted, bankrupt/exited.
- Add staged firm entry based on unmet demand, industry profitability, and exit
  replacement.
- Add a minimal economic-agent identity/status adapter before lifecycle work
  needs shared metadata.

Acceptance:
- Defaulted firms cannot get new credit or expand production.
- Exited firms cannot hire, produce, sell, or hold active product listings.
- Entrants register with EconomicCenter, markets, and LaborMarket, then can
  participate in the next month.

## First Worker Wave

The first active worker wave is:

- Wage-hour semantics: `019ecf59-914a-74e2-ad15-13d1fffc76f8` integrated.
- Intermediate-goods accounting: `019ecf59-cd2a-7bb0-889f-1f6bcf85090f` integrated.
- SFC flow diagnostics: `019ecf5a-989d-77a0-86bb-81fe1dcbf41f` integrated.
- Consumption plan/actual state separation: `019ecf5a-cce4-7ed2-98cd-168fb5da21bf` integrated.

Integration rule:
- Review each returned patch before running full tests.
- Resolve conflicts conservatively.
- Run focused tests from each worker, then full `unittest discover`.
- Record each completed integration in `docs/macro_refactor_log.md`.

First-wave validation:
- `.venv/bin/python -m unittest discover -s tests` passed with 56 tests.
- Ray smoke record:
  `output/smoke_records_profit_closure_p0/run_20260616_154451/month_0001.json`.
- Smoke accounting: `ok=True`, errors `0`, warnings `0`, SFC residual `0.0`.

## P1-A Integrated Follow-Up

Labor-budget floor:
- Removed the fixed minimum dollar payroll budget from firm job planning.
  Demand, income, production history, and startup cash can still support jobs,
  but weak demand is not raised into an artificial minimum wage bill.
- Added a regression test for tiny demand not producing part-time postings.

Validation:
- `.venv/bin/python -m unittest discover -s tests` passed with 57 tests.
- Ray smoke record:
  `output/smoke_records_profit_closure_p1_laborbudget/run_20260616_155442/month_0001.json`.
- Smoke accounting: `ok=True`, errors `0`, warnings `0`, SFC residual
  `1.4551915228366852e-11`.

Implication:
- The fixed payroll floor is no longer a source of persistent losses. Remaining
  loss pressure is now concentrated in the production/staffing chain: existing
  monthly payroll can still be charged against small output batches, and firms
  are not yet constrained by explicit cash/credit feasibility before production.

## P1-B/C/D Integrated Follow-Up

Cash and credit production planning:
- `ProductionPlanningPolicy` accepts cash, reserve, credit limit, outstanding
  debt, and approved credit. Feasible output is capped by spendable cash plus
  available credit whenever a unit cash cost is available.
- Simulator now reads current firm cash/debt/credit snapshots from
  `EconomicCenter` before planning production, instead of relying only on stale
  `firm.cash` values.
- When `production_unit_cash_cost` is not configured, simulator estimates a
  conservative unit cash-cost proxy from targeted SKU manufacturer prices and
  `production_unit_cash_cost_share`.

IO quote/scale planning:
- `IntermediateGoodsProcurement` has a pure planning interface that returns
  planned cost, feasible scale, scaled cost, shortages, and planning
  reservations without mutating market stock or ledgers.
- This is ready for the next simulator wiring step where production should
  quote inputs, scale output, then execute only the scaled purchase set.

Production-cost diagnostic/accounting:
- Monthly records now include `details.firm_profit_pressure`.
- Production statistics separate intermediate/resource input cost from labor.
  GDP/value-added diagnostics use input cost, while labor remains recorded by
  wage transactions and separate diagnostic fields.

Validation:
- `.venv/bin/python -m unittest discover -s tests` passed with 74 tests.
- Ray smoke record:
  `output/smoke_records_profit_closure_p1_cash_io_costfix/run_20260616_162739/month_0001.json`.
- Smoke accounting: `ok=True`, errors `0`, warnings `0`, SFC residual `0.0`.
- Smoke diagnostics:
  production input cost `$33.67`, labor-in-production diagnostic `$1,158.05`,
  total cost with labor `$1,191.72`, production-cost-to-output ratio `0.233`,
  and firms with production input cost above output `0`.

Implication:
- The prior small-smoke finding that production cost exceeded output was mostly
  a duplicated/aggregated labor-cost accounting issue. Remaining firm losses are
  now concentrated in realized sales coverage relative to payroll and in the
  macro closure identity when household saving is not offset by G/I/credit.

## Recommended Second Worker Wave

Start these after reviewing the P0 smoke metrics:

1. Production cash and credit constraints.
   Files: `firm_planning_policy.py`, `firm.py`, `simulator.py`,
   `config/config.py`, `config/config_normal.yaml`, tests.
   Status: policy and simulator planning integration completed. Further work
   should replace automatic credit draws with explicit application/approval
   before production and wage payments.

2. IO quote/reserve/scale/execute.
   Files: `IntermediateGoodsProcurement.py`, `firm.py`, tests.
   Status: quote/scale planning interface completed. Next work should wire the
   simulator/firm production execution path to execute only scaled purchases.

3. Wholesale and retail inventory chain.
   Files: `simulator.py`, `ProductMarket.py`, `Ecocenter.py`, tests.
   Acceptance: wholesale uses current manufacturer price, transfers stock to
   retail inventory, and household purchases cannot sell stock that retailers
   failed to procure.

4. Government budget mode.
   Files: `government.py`, `Ecocenter.py`, `simulator.py`, config, tests.
   Acceptance: balanced-budget mode hard-caps outlays; deficit-financed mode
   records government debt and preserves SFC residual.
