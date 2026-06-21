# Firm Initialization Calibration

This note defines the current calibration logic for firm opening balance sheets
and product-market initial inventory.

## Inputs

The calibration starts from Phase 0 demand discovery:

- `expected_monthly_revenue`: product-level household demand aggregated to each
  manufacturer or retailer.
- `expected_monthly_cost`: expected operating cost inferred from revenue and the
  firm's cost share.
- `expected_monthly_sales_units`: demanded product units aggregated to each
  firm.
- `demand_by_product`: SKU-level demanded units.

Configuration knobs live under `simulation` in `config/config_normal.yaml`:

- `firm_initial_capital_multiplier`: cash runway in months of expected cost.
- `firm_min_initial_cash`: cash floor for every firm.
- `firm_initial_inventory_cover_months`: SKU stock cover relative to Phase 0
  monthly demand.
- `firm_initial_min_active_sku_stock`: minimum stock for any demanded active SKU.
- `firm_initial_inactive_sku_stock`: small buffer for non-household-demanded
  SKUs, used by government and long-tail demand.
- `firm_capital_output_ratio`: capital-output ratio, interpreted as `K / Y`.
- `firm_inventory_value_share`: inventory valuation as a share of sales value.

## Equations

For each firm:

```text
working_capital_requirement = expected_monthly_cost * firm_initial_capital_multiplier
initial_cash = max(firm_min_initial_cash, working_capital_requirement)

target_inventory_units = expected_monthly_sales_units * firm_initial_inventory_cover_months
target_inventory_value = expected_monthly_revenue
                         * firm_initial_inventory_cover_months
                         * firm_inventory_value_share

annualized_output = max(expected_monthly_revenue, expected_monthly_cost) * 12
initial_capital_stock = max(target_inventory_value,
                            annualized_output * firm_capital_output_ratio)
```

For products after Phase 0:

```text
active_sku_stock[sku] = max(phase0_demand_units[sku] * firm_initial_inventory_cover_months,
                            firm_initial_min_active_sku_stock)
inactive_sku_stock = firm_initial_inactive_sku_stock
```

This replaces the old mechanical `100 units per SKU` seed. The default inactive
buffer is intentionally small (`10`) so government procurement has a long-tail
source without giving households unlimited hidden supply.

## Audit Trail

Monthly records include:

- `details.firm_initialization_calibration`: per-firm cash, capital, inventory
  targets, and diagnostics.
- `details.initial_inventory_calibration`: SKU stock reset totals, active SKU
  count, missing SKU count, and target inventory value.

## Calibration Loop

Use this loop for tuning:

1. Run one small smoke with `preheat_months >= 1`.
2. Check `initial_inventory_calibration.total_stock_after` against expected
   first-month goods demand. It should be close to active demand cover plus the
   inactive buffer, not catalogue size times 100.
3. Check production planning. If first-month production is near zero while
   goods demand is high, active SKU stock cover is too high.
4. Check government procurement. If it is zero in normal months, inactive SKU
   stock is too low or public procurement should be routed through active SKUs.
5. Check accounting invariants. No firm should need unexplained negative cash;
   shortfalls should become explicit credit draws.
