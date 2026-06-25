#!/usr/bin/env python3
"""
Build the paper's emergence-regularity table from a seed x repeat grid.

Scans output/grid_<seed>_r<rep>/ dirs, measures each run, and prints a
per-seed mean+/-std table plus a pooled (all-runs) row. Phillips uses the
best (most negative) lag in 0..3 since the U->inflation lead is ~1 month.

Usage:
    python tools/grid_table.py 'output/grid_*'  [--burnin 12]
"""
import sys
import glob
import re
import argparse
import statistics as st
import json

import numpy as np


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
    vals = {}
    for L in lags:
        vals[L] = _corr(x, y) if L == 0 else _corr(x[L:], y[:-L])
    cand = [v for v in vals.values() if v == v]
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
    okun = _corr(du[1:], gg[:-1])  # lag1
    return {
        "beveridge": _corr(uc, vc),
        "okun_lag1": okun,
        "phillips_best": _best_neg_lag(uc, infc),
        "u_mean": float(np.nanmean(u)),
    }


def fmt(vals):
    vals = [x for x in vals if x == x]
    if not vals:
        return "  n/a"
    m = st.mean(vals)
    s = st.pstdev(vals) if len(vals) > 1 else 0.0
    neg = sum(1 for x in vals if x < 0)
    return f"{m:+.2f}±{s:.2f} ({neg}/{len(vals)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glob_pat")
    ap.add_argument("--burnin", type=int, default=12)
    a = ap.parse_args()

    by_seed = {}
    for d in sorted(glob.glob(a.glob_pat)):
        mobj = re.search(r"grid_(\d+)_r(\d+)", d)
        if not mobj:
            continue
        seed = mobj.group(1)
        inner = sorted(glob.glob(f"{d}/run_*"))
        rd = inner[-1] if inner else d
        r = measure_run(rd, a.burnin)
        if r:
            by_seed.setdefault(seed, []).append(r)

    keys = ["okun_lag1", "beveridge", "phillips_best", "u_mean"]
    print(f"\n{'seed':>8} | {'n':>2} | {'Okun(lag1)':>18} | {'Beveridge':>18} | {'Phillips(best)':>18} | {'U mean':>12}")
    print("-" * 95)
    pooled = {k: [] for k in keys}
    for seed in sorted(by_seed):
        runs = by_seed[seed]
        row = {k: [r[k] for r in runs] for k in keys}
        for k in keys:
            pooled[k].extend(row[k])
        print(f"{seed:>8} | {len(runs):>2} | {fmt(row['okun_lag1']):>18} | {fmt(row['beveridge']):>18} | {fmt(row['phillips_best']):>18} | {fmt(row['u_mean']):>12}")
    print("-" * 95)
    print(f"{'POOLED':>8} | {sum(len(v) for v in by_seed.values()):>2} | {fmt(pooled['okun_lag1']):>18} | {fmt(pooled['beveridge']):>18} | {fmt(pooled['phillips_best']):>18} | {fmt(pooled['u_mean']):>12}")
    print("\n(neg/n = fraction of runs with the empirically-correct negative sign)")


if __name__ == "__main__":
    main()
