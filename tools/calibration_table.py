#!/usr/bin/env python3
"""
Formal calibration / validation table: simulated moments vs. US empirical
targets, with a normalized distance for the alignable moments.

This is the moment-matching evidence a top-tier ABM/macro reviewer expects —
beyond "the stylized-fact signs are correct", it reports HOW CLOSE the
simulated second-moment / distributional moments are to post-war US values.

Design (per the project decision — hybrid):
  * ALIGNABLE moments (business-cycle second moments, persistence, relative
    volatilities, distributional shape: firm-size skew, income/wealth Gini,
    MPC) are compared to an empirical target band and scored by how many
    target standard-deviations away the pooled simulated mean lies.
  * LEVEL moments that are KNOWN small-economy structural deviations
    (unemployment level, labor share) are REPORTED but flagged "structural
    deviation — not scored", per the honest-reporting decision. The 50-household
    IO-constrained economy is not calibrated to US levels; its contribution is
    emergent dynamics, not level-matching.

Moments are pooled across a seed x repeat grid (mean +/- std over runs), the
same convention as grid_table.py. Cyclical components are linear-detrended;
inflation/growth use the monthly series.

Empirical targets (post-war US, monthly-equivalent where noted; sources are
standard RBC/labor/inequality references — adjust in TARGETS as needed):

Usage:
    python tools/calibration_table.py 'output/bidgrid_*' [--burnin 12]
"""
import sys
import glob
import re
import argparse
import statistics as st
import json

import numpy as np


# -- empirical target bands (US post-war). (target, tol, note, scored?) --
# tol = one "empirical standard deviation" used to normalize the distance.
TARGETS = {
    # business-cycle second moments (cyclical, detrended)
    "u_volatility":        (1.3,  0.6,  "Unemployment cyclical std (pp)",         True),
    "infl_persistence":    (0.6,  0.25, "Inflation AC(1)",                        True),
    "output_persistence":  (0.3,  0.3,  "Real-GDP-growth AC(1)",                  True),
    "cons_rel_volatility": (0.6,  0.25, "Consumption vol / output vol (<1)",      True),
    "inv_rel_volatility":  (3.0,  1.5,  "Investment vol / output vol (>1)",       True),
    # distributional / cross-section
    "firm_revenue_skew":   (2.5,  1.5,  "Firm revenue right-skew (>0 heavy tail)", True),
    "income_gini":         (0.39, 0.06, "Household income Gini",                  True),
    "wealth_gini":         (0.85, 0.10, "Household wealth Gini",                  True),
    "mpc":                 (0.40, 0.20, "Marginal propensity to consume (0,1)",   True),
    # level moments — KNOWN small-economy structural deviations, NOT scored
    "u_mean":              (0.05, 0.02, "Unemployment level",                     False),
    "labor_share":         (0.58, 0.04, "Labor share (comp/GDP)",                 False),
}


def _detrend(x):
    x = np.asarray(x, float)
    t = np.arange(len(x))
    m = ~np.isnan(x)
    if m.sum() < 3:
        return x - np.nanmean(x)
    b = np.polyfit(t[m], x[m], 1)
    return x - np.polyval(b, t)


def _ac1(x):
    x = np.asarray(x, float)
    m = ~np.isnan(x)
    x = x[m]
    if len(x) < 4 or np.std(x) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x[1:], x[:-1])[0, 1])


def _gini(v):
    v = np.sort(np.asarray([x for x in v if x is not None and x == x], float))
    v = v[v >= 0]
    n = len(v)
    if n == 0 or v.sum() == 0:
        return float("nan")
    cum = np.cumsum(v)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def _skew(v):
    v = np.asarray([x for x in v if x is not None and x == x], float)
    if len(v) < 3 or np.std(v) < 1e-12:
        return float("nan")
    return float(np.mean(((v - v.mean()) / v.std()) ** 3))


def measure_run(run_dir, burnin):
    files = sorted(glob.glob(f"{run_dir}/month_*.json"))[burnin:]
    if len(files) < 8:
        return None
    u, inf, rgdp, cons_agg, ls = [], [], [], [], []
    for f in files:
        d = json.load(open(f))
        lm = d.get("labor_market", {}); m = d.get("macro", {})
        u.append(lm.get("unemployment_rate", np.nan))
        inf.append(m.get("inflation_rate", np.nan))
        rgdp.append(m.get("real_gdp", np.nan))
        ls.append(m.get("labor_share", np.nan))
        # aggregate consumption (expenditure-side C) for relative volatility
        cc = (m.get("gdp_comprehensive", {}) or {}).get("expenditure_components", {}).get("consumption", {})
        cons_agg.append(cc.get("total", np.nan) if isinstance(cc, dict) else np.nan)

    u = np.array(u, float); inf = np.array(inf, float)
    rgdp = np.array(rgdp, float); cons_agg = np.array(cons_agg, float)
    gg = np.diff(np.log(np.where(rgdp > 0, rgdp, np.nan))) * 100
    cg = np.diff(np.log(np.where(cons_agg > 0, cons_agg, np.nan))) * 100

    # investment growth volatility from FCF+inventory (total investment)
    inv = []
    for f in files:
        d = json.load(open(f))
        ic = (d.get("macro", {}).get("gdp_comprehensive", {}) or {}).get("expenditure_components", {}).get("investment", {}) or {}
        inv.append(ic.get("total_investment", ic.get("inventory_investment", np.nan)))
    inv = np.array(inv, float)
    ig = np.diff(inv)

    out = {
        "u_volatility": float(np.nanstd(_detrend(u)) * 100),  # pp
        "infl_persistence": _ac1(inf),
        "output_persistence": _ac1(gg),
        "cons_rel_volatility": (float(np.nanstd(cg) / np.nanstd(gg)) if np.nanstd(gg) > 1e-9 else np.nan),
        "inv_rel_volatility": (float(np.nanstd(ig) / (np.nanstd(np.diff(rgdp)) + 1e-9))),
        "u_mean": float(np.nanmean(u)),
        "labor_share": float(np.nanmean(ls)),
    }

    # cross-section (final formal month)
    d = files[-1]
    d = json.load(open(d))
    ff = d.get("details", {}).get("firm_financials", {})
    rev = [v.get("monthly_income", 0.0) for v in ff.values()]
    hb = d.get("household", {}).get("by_household", {})
    inc = [h.get("income", {}).get("total", np.nan) for h in hb.values()]
    wealth = [h.get("balance", np.nan) for h in hb.values()]
    consh = [h.get("consumption", {}).get("total", np.nan) for h in hb.values()]
    out["firm_revenue_skew"] = _skew(rev)
    out["income_gini"] = _gini(inc)
    out["wealth_gini"] = _gini(wealth)
    ai = np.asarray(inc, float); ac = np.asarray(consh, float)
    mm = ~(np.isnan(ai) | np.isnan(ac)) & (ai > 0)
    out["mpc"] = float(np.polyfit(ai[mm], ac[mm], 1)[0]) if mm.sum() > 5 else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glob_pat")
    ap.add_argument("--burnin", type=int, default=12)
    a = ap.parse_args()

    runs = []
    for d in sorted(glob.glob(a.glob_pat)):
        inner = sorted(glob.glob(f"{d}/run_*"))
        rd = inner[-1] if inner else d
        r = measure_run(rd, a.burnin)
        if r:
            runs.append(r)
    if not runs:
        sys.exit(f"no runs matched {a.glob_pat}")

    def pooled(key):
        vals = [r[key] for r in runs if r.get(key) == r.get(key)]
        if not vals:
            return float("nan"), float("nan")
        return st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)

    print(f"\nFormal calibration: simulated moments vs US empirical targets "
          f"(pooled over n={len(runs)} runs, burn-in={a.burnin})")
    print("=" * 92)
    print(f"{'moment':<26}{'sim mean±std':>18}{'US target':>14}{'|z|':>7}  {'note'}")
    print("-" * 92)

    scored_z = []
    for key, (tgt, tol, note, scored) in TARGETS.items():
        m, s = pooled(key)
        if m != m:
            print(f"{key:<26}{'n/a':>18}{tgt:>14.2f}{'—':>7}  {note}")
            continue
        z = abs(m - tgt) / tol if tol > 0 else float("nan")
        if scored:
            scored_z.append(z)
            zs = f"{z:>6.2f}"
            flag = "" if z <= 1.5 else ("  (loose)" if z <= 3 else "  (FAR)")
            print(f"{key:<26}{m:>9.3f}±{s:<7.3f}{tgt:>14.2f}{zs:>7}{flag}  {note}")
        else:
            print(f"{key:<26}{m:>9.3f}±{s:<7.3f}{tgt:>14.2f}{'—':>7}  {note} [structural dev, not scored]")
    print("-" * 92)
    if scored_z:
        within = sum(1 for z in scored_z if z <= 1.5)
        print(f"Scored moments within 1.5 target-SD: {within}/{len(scored_z)}   "
              f"mean |z| = {st.mean(scored_z):.2f}")
    print("\n|z| = target-standardized distance of the pooled simulated mean from the US value.")
    print("Level moments (unemployment, labor share) are reported but NOT scored: the 50-household")
    print("IO-constrained economy is calibrated for emergent dynamics, not US levels (see paper).")


if __name__ == "__main__":
    main()
