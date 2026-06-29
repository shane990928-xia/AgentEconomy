#!/usr/bin/env python3
"""Sensitivity analysis: sweep one config knob and measure how the emergent
regularities + key moments respond.

This backs the paper's core methodological claim — the macro regularities
(Okun/Phillips/Beveridge) and the calibrated moments are EMERGENT, not
parameter-fitted: if a regularity's sign survives a wide sweep of a knob, it is
robust to that knob rather than tuned to one value.

It runs the model once per knob value (rules mode by default, for speed) from a
base config, then prints a table of Okun/Phillips/Beveridge (best-negative lag,
burn-in dropped, linear-detrended — the SAME convention as grid_table.py /
calibration_table.py so results are comparable) plus u_volatility and mpc.

Usage:
  python tools/sensitivity_sweep.py --config config/config_calib.yaml \
      --knob firm_layoff_speed --values 0.2,0.35,0.5,0.75,1.0 \
      [--seed 12345] [--burnin 12] [--out-root /tmp/sens]

Throwaway tool (leading underscore not used so it can be a keeper); writes each
run under <out-root>/<knob>_<value>/ and reuses them if present.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def _detrend(x):
    n = len(x)
    if n < 3:
        return [v - (sum(x) / n) for v in x]
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


def _best_neg(a, b, ml=3):
    best = None
    for L in range(ml + 1):
        x = a[L:]
        y = b[: len(b) - L] if L > 0 else b
        c = _corr(x, y)
        if c == c and (best is None or c < best[1]):
            best = (L, c)
    return best if best else (0, float("nan"))


def _pstdev(x):
    n = len(x)
    if n < 2:
        return float("nan")
    m = sum(x) / n
    return (sum((v - m) ** 2 for v in x) / n) ** 0.5


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
    okun = _best_neg(du, g)
    php = _best_neg(U, INFL)
    bev = _corr(_detrend(U), _detrend(V))
    return {
        "u_vol_pp": _pstdev(U) * 100,
        "okun": okun[1],
        "okun_lag": okun[0],
        "phillips": php[1],
        "phillips_lag": php[0],
        "beveridge": bev,
    }


def run_one(base_cfg, knob, value, seed, out_root):
    rec = os.path.join(out_root, f"{knob}_{value}")
    if glob.glob(os.path.join(rec, "run_*", "month_*.json")):
        return rec  # reuse existing
    text = base_cfg
    text = re.sub(r"random_seed:\s*\d+", f"random_seed: {seed}", text)
    text = re.sub(r'local_record_dir:\s*"[^"]*"', f'local_record_dir: "{rec}"', text)
    # set the knob (replace existing line or inject under the simulation block)
    if re.search(rf"^\s*{re.escape(knob)}:", text, re.M):
        text = re.sub(rf"(^\s*{re.escape(knob)}:).*$", rf"\1 {value}", text, count=1, flags=re.M)
    else:
        text = re.sub(r"(^simulation:\s*$)", rf"\1\n  {knob}: {value}", text, count=1, flags=re.M)
    cfg = Path(out_root) / f"_sweep_{knob}_{value}.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(text)
    env = dict(os.environ)
    env.update(RAY_ENABLE_UV_RUN_RUNTIME_ENV="0", TMPDIR="/tmp", RAY_TMPDIR="/tmp", AGENTECO_COSTPUSH="0")
    print(f"  running {knob}={value} (seed {seed}) ...", flush=True)
    r = subprocess.run([".venv/bin/python", "main.py", "--config", str(cfg)],
                       env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode != 0:
        print(f"  WARN: {knob}={value} exited {r.returncode}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config_calib.yaml")
    ap.add_argument("--knob", required=True)
    ap.add_argument("--values", required=True, help="comma-separated values")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--burnin", type=int, default=12)
    ap.add_argument("--out-root", default="/tmp/sens")
    a = ap.parse_args()

    base_cfg = Path(a.config).read_text()
    values = [v.strip() for v in a.values.split(",") if v.strip()]
    Path(a.out_root).mkdir(parents=True, exist_ok=True)

    print(f"Sensitivity sweep: {a.knob} in {values} (base={a.config}, seed={a.seed})")
    rows = []
    for v in values:
        rec = run_one(base_cfg, a.knob, v, a.seed, a.out_root)
        inner = sorted(glob.glob(os.path.join(rec, "run_*")))
        rd = inner[-1] if inner else rec
        m = measure(rd, a.burnin)
        rows.append((v, m))

    print("\n" + "=" * 78)
    print(f"{a.knob:>22} | {'u_vol':>6} | {'Okun(lag)':>12} | {'Phillips(lag)':>14} | {'Beveridge':>10}")
    print("-" * 78)
    for v, m in rows:
        if not m:
            print(f"{v:>22} | <insufficient data>")
            continue
        print(f"{v:>22} | {m['u_vol_pp']:>5.2f}p | "
              f"{m['okun']:>+7.2f}(L{m['okun_lag']}) | "
              f"{m['phillips']:>+9.2f}(L{m['phillips_lag']}) | "
              f"{m['beveridge']:>+10.2f}")
    print("-" * 78)
    print("Robustness read: a regularity whose SIGN holds across the sweep is")
    print("emergent/robust to this knob, not fitted to one value.")


if __name__ == "__main__":
    main()
