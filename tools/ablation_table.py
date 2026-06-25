#!/usr/bin/env python3
"""
Decompose which emergent regularities depend on which mechanisms.

Scans the ablation cells output/abl_<cell>/run_*/ (one or more repeats each),
measures Beveridge / Okun(lag1) / Phillips(best-neg-lag 0..3) per run, and
prints a per-cell mean±std table. The cells are:

    baseline    endogenous wage formula ON,  cost-push ON
    nocostpush  endogenous wage formula ON,  cost-push OFF (run w/ AGENTECO_COSTPUSH=0)
    noendog     endogenous wage formula OFF (wage_scale fixed → cost-push auto-neutral)
    neither     same as noendog (control)

Reads month_*.json directly so it does not depend on tools/stylized_facts
printing. Phillips uses the most-negative lag in 0..3 because U leads
inflation by ~1 month in this economy.

Usage:
    python tools/ablation_table.py [--burnin 12]
"""
import glob
import json
import argparse
import statistics as st

import numpy as np

CELLS = ["baseline", "nocostpush", "noendog", "neither"]


def _detrend(x):
    x = np.asarray(x, float)
    t = np.arange(len(x))
    m = ~np.isnan(x)
    if m.sum() < 3:
        return x - np.nanmean(x)
    b = np.polyfit(t[m], x[m], 1)
    return x - np.polyval(b, t)


def _corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 4 or np.std(a[m]) < 1e-9 or np.std(b[m]) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _best_neg_lag(x, y, lags=(0, 1, 2, 3)):
    vals = []
    for L in lags:
        vals.append(_corr(x, y) if L == 0 else _corr(x[L:], y[:-L]))
    cand = [v for v in vals if v == v]
    return min(cand) if cand else float("nan")


def measure_run(run_dir, burnin):
    files = sorted(glob.glob(f"{run_dir}/month_*.json"))[burnin:]
    if len(files) < 8:
        return None
    u, v, inf, rgdp = [], [], [], []
    for f in files:
        d = json.load(open(f))
        lm = d.get("labor_market", {}); m = d.get("macro", {})
        u.append(lm.get("unemployment_rate", np.nan))
        v.append(lm.get("vacancy_rate", np.nan))
        inf.append(m.get("inflation_rate", np.nan))
        rgdp.append(m.get("real_gdp", np.nan))
    u = np.array(u, float); v = np.array(v, float)
    inf = np.array(inf, float); rgdp = np.array(rgdp, float)
    uc, vc, infc = _detrend(u), _detrend(v), _detrend(inf)
    gg = np.diff(np.log(np.where(rgdp > 0, rgdp, np.nan))) * 100
    du = np.diff(u)
    return {
        "beveridge": _corr(uc, vc),
        "okun_lag1": _corr(du[1:], gg[:-1]),
        "phillips_best": _best_neg_lag(uc, infc),
        "u_mean": float(np.nanmean(u)),
        "inf_mean": float(np.nanmean(inf)),
    }


def fmt(vals):
    vals = [x for x in vals if x == x]
    if not vals:
        return "       n/a"
    m = st.mean(vals)
    s = st.pstdev(vals) if len(vals) > 1 else 0.0
    neg = sum(1 for x in vals if x < 0)
    return f"{m:+.2f}±{s:.2f} ({neg}/{len(vals)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--burnin", type=int, default=12)
    a = ap.parse_args()

    keys = ["okun_lag1", "beveridge", "phillips_best", "u_mean", "inf_mean"]
    print(f"\n{'cell':>11} | {'n':>2} | {'Okun(lag1)':>16} | {'Beveridge':>16} | "
          f"{'Phillips(best)':>16} | {'U mean':>8} | {'infl':>8}")
    print("-" * 100)
    for cell in CELLS:
        runs = []
        for rd in sorted(glob.glob(f"output/abl_{cell}/run_*")):
            r = measure_run(rd, a.burnin)
            if r:
                runs.append(r)
        if not runs:
            print(f"{cell:>11} |  0 | (no runs)")
            continue
        row = {k: [r[k] for r in runs] for k in keys}
        umean = st.mean(row["u_mean"])
        infm = st.mean(row["inf_mean"])
        print(f"{cell:>11} | {len(runs):>2} | {fmt(row['okun_lag1']):>16} | "
              f"{fmt(row['beveridge']):>16} | {fmt(row['phillips_best']):>16} | "
              f"{umean:>8.3f} | {infm:>8.4f}")
    print("-" * 100)
    print("(neg/n = runs with empirically-correct negative sign)")
    print("\nReading: compare phillips_best across rows.")
    print("  baseline vs nocostpush  → cost-push shortcut's contribution to Phillips")
    print("  baseline vs noendog     → endogenous-wage formula's contribution")
    print("  if Phillips survives in noendog/neither → it is genuinely emergent")


if __name__ == "__main__":
    main()
