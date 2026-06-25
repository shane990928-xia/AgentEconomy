#!/usr/bin/env python3
"""
Aggregate emergence regularities across multiple runs (same or different seeds)
to report mean +/- std — the publication-form result for a parallel ABM whose
single-run output carries floating-point/scheduling noise.

Usage:
    python tools/aggregate_emergence.py output/rep_12345_*  [--burnin 12]
"""
import sys
import glob
import argparse
import statistics as st

sys.path.insert(0, "tools")
from stylized_facts import measure  # reuse the single-run measurer


def _run_dir(path):
    # accept either a record dir or its parent containing run_*
    inner = sorted(glob.glob(f"{path}/run_*"))
    return inner[-1] if inner else path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="record dirs (each holds run_* or month_*.json)")
    ap.add_argument("--burnin", type=int, default=12)
    a = ap.parse_args()

    keys = ["beveridge", "okun_lag1", "phillips_lag0", "u_mean"]
    acc = {k: [] for k in keys}
    n = 0
    for p in a.paths:
        rd = _run_dir(p)
        if not glob.glob(f"{rd}/month_*.json"):
            continue
        try:
            r = measure(rd, a.burnin)
        except SystemExit:
            continue
        for k in keys:
            v = r.get(k)
            if v is not None and v == v:  # not NaN
                acc[k].append(v)
        n += 1

    print(f"\n{'='*60}\nAGGREGATE over {n} runs\n{'='*60}")
    for k in keys:
        vals = acc[k]
        if not vals:
            print(f"  {k:16s}: no data")
            continue
        m = st.mean(vals)
        s = st.pstdev(vals) if len(vals) > 1 else 0.0
        sign_ok = sum(1 for v in vals if v < 0)
        print(f"  {k:16s}: mean={m:+.3f}  std={s:.3f}  n={len(vals)}  "
              f"neg={sign_ok}/{len(vals)}  range=[{min(vals):+.2f},{max(vals):+.2f}]")


if __name__ == "__main__":
    main()
