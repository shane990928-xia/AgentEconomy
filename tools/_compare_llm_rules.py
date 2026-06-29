#!/usr/bin/env python3
"""Throwaway: compare the calibrated economy under LLM-monthly vs rules mode.

Measures the cobweb-fix + regularity metrics on a single run dir (or glob) and
prints them side-by-side with the rules-mode ls035 grid baseline, so the
LLM-robustness of the calibrated economy (commit 110c180) can be judged.

Usage: python tools/_compare_llm_rules.py <llm_run_glob>
  e.g. python tools/_compare_llm_rules.py '/home/dataset-local/bendi/xiaxu/calibllm_out/calibllm_*'
"""
import json
import glob
import os
import sys
import statistics as st


def _detrend(x):
    n = len(x)
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(x) / n
    den = sum((xs[i] - mx) ** 2 for i in range(n))
    b = sum((xs[i] - mx) * (x[i] - my) for i in range(n)) / den if den else 0.0
    a = my - b * mx
    return [x[i] - (a + b * xs[i]) for i in range(n)]


def _corr(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n < 4:
        return float("nan")
    ma = sum(a) / n
    mb = sum(b) / n
    va = sum((v - ma) ** 2 for v in a)
    vb = sum((v - mb) ** 2 for v in b)
    if va <= 0 or vb <= 0:
        return float("nan")
    return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / (va * vb) ** 0.5


def _lag1(x):
    n = len(x)
    m = sum(x) / n
    var = sum((v - m) ** 2 for v in x)
    if var == 0:
        return float("nan")
    return sum((x[i] - m) * (x[i + 1] - m) for i in range(n - 1)) / var


def _best_neg(a, b, ml=3):
    best = None
    for L in range(ml + 1):
        x = a[L:]
        y = b[: len(b) - L] if L > 0 else b
        c = _corr(x, y)
        if c == c and (best is None or c < best[1]):
            best = (L, round(c, 2))
    return best


def measure(run_dir, burnin=12):
    ms = sorted(glob.glob(os.path.join(run_dir, "month_*.json")))[burnin:]
    if len(ms) < 8:
        return None
    U, V, OUT, INFL = [], [], [], []
    for mf in ms:
        d = json.load(open(mf))
        lm = d.get("labor_market", {})
        mac = d.get("macro", {})
        U.append(lm.get("unemployment_rate", float("nan")))
        V.append(lm.get("vacancy_rate", float("nan")))
        OUT.append(mac.get("real_gdp", float("nan")))
        INFL.append(mac.get("inflation_rate", float("nan")))
    g = [OUT[i] - OUT[i - 1] for i in range(1, len(OUT))]
    du = [U[i] - U[i - 1] for i in range(1, len(U))]
    return {
        "n_months": len(ms),
        "u_mean_pct": st.mean(U) * 100,
        "u_vol_pp": st.pstdev(U) * 100,
        "u_ac1": _lag1(U),
        "infl_ac1": _lag1(INFL),
        "gdpg_ac1": _lag1(g),
        "okun": _best_neg(du, g),
        "phillips": _best_neg(U, INFL),
        "beveridge": round(_corr(_detrend(U), _detrend(V)), 2),
    }


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "/home/dataset-local/bendi/xiaxu/calibllm_out/calibllm_*"
    dirs = sorted(glob.glob(pat))
    print(f"LLM-monthly calibrated economy — {len(dirs)} run(s) matching {pat}")
    print("=" * 84)
    print("RULES-mode baseline (ls035 grid, pooled n=5): "
          "u_vol 2.93pp, mean|z| 1.57, Okun -0.24(5/5), Phillips -0.29(5/5), Beveridge +0.30")
    print("-" * 84)
    for d in dirs:
        inner = sorted(glob.glob(os.path.join(d, "run_*")))
        rd = inner[-1] if inner else d
        m = measure(rd)
        if not m:
            print(f"{os.path.basename(d)}: <{m} insufficient months>")
            continue
        print(f"{os.path.basename(d)} ({m['n_months']}mo post-burnin):")
        print(f"   u_mean={m['u_mean_pct']:.1f}%  u_vol={m['u_vol_pp']:.2f}pp  "
              f"u_ac1={m['u_ac1']:+.2f}  infl_ac1={m['infl_ac1']:+.2f}  gdpg_ac1={m['gdpg_ac1']:+.2f}")
        print(f"   Okun(du,g)={m['okun']}  Phillips(U,infl)={m['phillips']}  Beveridge(U,vac)={m['beveridge']:+.2f}")


if __name__ == "__main__":
    main()
