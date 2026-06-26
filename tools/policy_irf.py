#!/usr/bin/env python3
"""
Monetary-policy impulse-response (IRF) experiment for the AgentEconomy macro sim.

Runs the Taylor-rule economy twice — a baseline and a +shock scenario where the
central bank's policy rate carries an exogenous additive shock
(`taylor_rate_shock`) — and plots the response of the policy rate, fixed
investment, real output, unemployment, and inflation.

Empirically-correct directions for a monetary tightening (rate hike):
    policy_rate ↑  →  fixed investment ↓  →  real output ↓  →  unemployment ↑
    and (with a lag) inflation ↓.

Both runs use the SAME seed and config; the only difference is the rate shock,
so the difference in paths is the impulse response.

Usage:
    python tools/policy_irf.py --config config/config_taylor.yaml --shock 0.05 \
        --months 48 --out output/policy_irf
Requires the two runs to already be produced, OR pass --run to launch them
(launches main.py twice with a tweaked config).
"""
import argparse
import glob
import json
import subprocess
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_series(run_glob, burnin=0):
    dirs = sorted(glob.glob(run_glob))
    if not dirs:
        return None
    inner = sorted(glob.glob(f"{dirs[-1]}/run_*"))
    rd = inner[-1] if inner else dirs[-1]
    files = sorted(glob.glob(f"{rd}/month_*.json"))[burnin:]
    if not files:
        return None
    out = {k: [] for k in ["policy_rate", "fcf", "real_gdp", "u", "infl"]}
    for f in files:
        d = json.load(open(f))
        m = d.get("macro", {})
        lm = d.get("labor_market", {})
        inv = (m.get("gdp_comprehensive", {}) or {}).get("expenditure_components", {}).get("investment", {}) or {}
        out["policy_rate"].append(m.get("policy_rate", np.nan))
        out["fcf"].append(inv.get("fixed_capital_formation", np.nan))
        out["real_gdp"].append(m.get("real_gdp", np.nan))
        out["u"].append(lm.get("unemployment_rate", np.nan))
        out["infl"].append(m.get("inflation_rate", np.nan))
    return {k: np.asarray(v, float) for k, v in out.items()}


def _write_shocked_config(base_config, shock, out_dir, record_dir):
    """Read the YAML as text, override taylor_rate_shock + local_record_dir."""
    import re
    text = Path(base_config).read_text()
    if "taylor_rate_shock" in text:
        text = re.sub(r"taylor_rate_shock:\s*[-0-9.]+", f"taylor_rate_shock: {shock}", text)
    else:
        text = text.replace("taylor_rule_enabled: true",
                            f"taylor_rule_enabled: true\n  taylor_rate_shock: {shock}")
    text = re.sub(r'local_record_dir:\s*"[^"]*"', f'local_record_dir: "{record_dir}"', text)
    p = Path(out_dir) / f"_irf_cfg_shock_{shock}.yaml"
    p.write_text(text)
    return str(p)


def _run(config_path):
    env = dict(os.environ)
    env.update(RAY_ENABLE_UV_RUN_RUNTIME_ENV="0", TMPDIR="/tmp", RAY_TMPDIR="/tmp",
               AGENTECO_COSTPUSH="0")
    subprocess.run([".venv/bin/python", "main.py", "--config", config_path],
                   env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config_taylor.yaml")
    ap.add_argument("--shock", type=float, default=0.05)
    ap.add_argument("--burnin", type=int, default=6)
    ap.add_argument("--out", default="output/policy_irf")
    ap.add_argument("--run", action="store_true",
                    help="Launch the baseline + shock runs before plotting.")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    base_rec = f"{a.out}/baseline"
    shock_rec = f"{a.out}/shock"
    if a.run:
        base_cfg = _write_shocked_config(a.config, 0.0, a.out, base_rec)
        shock_cfg = _write_shocked_config(a.config, a.shock, a.out, shock_rec)
        print(f"running baseline (shock=0) → {base_rec}")
        _run(base_cfg)
        print(f"running shock (shock={a.shock}) → {shock_rec}")
        _run(shock_cfg)

    base = _load_series(base_rec, a.burnin)
    shock = _load_series(shock_rec, a.burnin)
    if base is None or shock is None:
        print("missing runs — pass --run to generate them first")
        return

    panels = [
        ("policy_rate", "Policy rate", 1.0),
        ("fcf", "Fixed investment ($)", 1.0),
        ("real_gdp", "Real GDP", 1.0),
        ("u", "Unemployment", 100.0),
        ("infl", "Inflation (%)", 100.0),
    ]
    n = min(len(base["u"]), len(shock["u"]))
    t = np.arange(n)
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 13), sharex=True)
    for ax, (key, label, scale) in zip(axes, panels):
        b = base[key][:n] * scale
        s = shock[key][:n] * scale
        ax.plot(t, b, "b-o", ms=3, label="baseline")
        ax.plot(t, s, "r-s", ms=3, label=f"+{a.shock} rate shock")
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    axes[-1].set_xlabel("Formal month (post burn-in)", fontsize=10)
    fig.suptitle(f"Monetary-policy IRF: +{a.shock} policy-rate shock vs baseline",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    p = out / "policy_irf.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p}")

    # Quick text summary of mean differences (shock − baseline).
    print("\nmean(shock − baseline) over the horizon:")
    for key, label, _ in panels:
        diff = np.nanmean(shock[key][:n] - base[key][:n])
        print(f"  {label:24s} {diff:+.4f}")


if __name__ == "__main__":
    main()
