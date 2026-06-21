# Macro Closure Backlog Report

Date: 2026-06-18

## Summary

This pass focused on the remaining macro-closure problem where firms were
posting and paying wages without a strong enough production response. The main
change is to carry unmet production capacity from one month into the next
month's labor demand signal.

The validated run is:

`output/validation_macro_profile_p20_m6_backlog_labor/run_20260618_162012`

Configuration highlights:

- 20 households
- 2 preheat months
- 6 formal months
- `consumption_use_llm=True`
- `consumption_llm_mode="profile"`
- `firm_labor_backlog_demand_share=0.5`

## Implementation

- `Simulator` now keeps `_last_production_gap_value_by_firm`.
- `_ensure_production()` values each firm's production gap from the targeted
  SKU price mix and writes `production_gap_value` into monthly planning
  diagnostics.
- `_post_jobs()` combines current manufacturing demand with the previous
  production gap value, using:
  `max(current_demand_value, production_gap_value * firm_labor_backlog_demand_share)`.
- Labor matching priority uses the same combined demand signal.
- Checkpoints persist and restore the production gap value cache.

## Validation Metrics

Month 6 of the new p20 validation run:

- Nominal GDP: `$68,425.5`
- C/GDP: `77.8%`
- G/GDP: `18.7%`
- I/GDP: `3.5%`
- Labor share: `52.8%`
- Employment rate: `52.5%`
- Gross wages: `$36,101.3`
- Production output: `$9,590.8`
- Producing firms: `3`
- Household goods consumption: `$14,886.9`
- Household service consumption: `$37,180.6`
- Accounting invariants: zero errors, zero warnings

Compared with the previous p20 profile validation month 6:

- Employment improved from `40.0%` to `52.5%`.
- Production output improved from `$4,426.5` to `$9,590.8`.
- C/GDP moved from `101.7%` to `77.8%`.
- Inventory investment moved from `-8.1%` of GDP to `+3.5%`.

## Month Sequence

| Month | GDP | C/GDP | G/GDP | I/GDP | Labor Share | Employment | Production |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 53,527.8 | 87.7% | 5.4% | 6.9% | 40.8% | 27.5% | 7,656.5 |
| 2 | 57,070.7 | 83.6% | 9.7% | 6.7% | 48.6% | 45.0% | 9,424.2 |
| 3 | 61,226.3 | 83.5% | 16.9% | -0.4% | 34.5% | 27.5% | 6,607.9 |
| 4 | 62,129.5 | 85.5% | 20.5% | -6.0% | 58.1% | 50.0% | 3,610.0 |
| 5 | 65,047.2 | 82.6% | 19.6% | -2.2% | 37.8% | 35.0% | 5,965.5 |
| 6 | 68,425.5 | 77.8% | 18.7% | 3.5% | 52.8% | 52.5% | 9,590.8 |

## Remaining Risks

- Individual firm pressure remains: in month 6, three firms had realized income
  below wages. Aggregate realized income/wage was `2.54`, and the
  inventory-adjusted wage gap was zero, so this is no longer a system-wide wage
  black hole.
- Government procurement still clears after production and is constrained by
  active on-hand SKUs. It should become an ex-ante demand signal before
  production.
- Retail labor sizing remains coarse. `ret_452` still had high wage expense
  relative to realized sales in month 6. The next pass should tune retail
  compensation ratios, inventory turnover, and job-hour sizing separately from
  manufacturing.
- Labor remains volatile at small scale because 20 households means one job is
  2.5 percentage points of total labor. Larger validation runs are needed
  before treating month-to-month cycles as stable macro dynamics.

## Verification

- `.venv/bin/python -m unittest discover -s tests`
  passed with `146` tests.
- Six-month p20 validation completed with the repository `.venv` and local Ray.
