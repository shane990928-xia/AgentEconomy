#!/usr/bin/env python3
"""LLM-decision result figures (the platform's core selling point).

Uses the LLM-mode calibrated economy (calibllm_* = consumption_use_llm on,
monthly mode) vs the rules-mode calibrated grid (calibgrid_*_ls035).

Generates into output/paper_figures/:
  06_llm_regularities.png   Okun + Phillips scatter, LLM economy (emergent under LLM)
  07_llm_vs_rules.png       side-by-side bars: u_vol / Okun / Phillips / MPC
  08_llm_timeseries.png     u / inflation / GDP-growth over time, an LLM run

Same measurement convention as the tables: 12-month burn-in, linear detrend,
best-negative lag for Okun/Phillips.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BURNIN = 12
LLM_DIRS = sorted(glob.glob("/home/dataset-local/bendi/xiaxu/calibllm_out/calibllm_[0-9]*"))
RULE_SEEDS = [12345, 2026, 7777, 31337, 90210]
FIGDIR = "output/paper_figures"


def _detrend(x):
    x = np.asarray(x, float)
    t = np.arange(len(x))
    m = ~np.isnan(x)
    if m.sum() < 3:
        return x - np.nanmean(x)
    b = np.polyfit(t[m], x[m], 1)
    return x - np.polyval(b, t)


def _pearson(x, y):
    n = min(len(x), len(y)); x, y = x[:n], y[:n]
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 4:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def _best_neg_lag(x, y, lags=(0, 1, 2, 3)):
    best = None
    for L in lags:
        xx = x[L:] if L else x
        yy = y[:len(y) - L] if L else y
        r = _pearson(xx, yy)
        if r == r and (best is None or r < best[1]):
            best = (L, r)
    return best if best else (0, np.nan)


def _load_dir(rundir):
    files = sorted(glob.glob(os.path.join(rundir, "month_*.json")))[BURNIN:]
    u, inf, rgdp = [], [], []
    for f in files:
        d = json.load(open(f))
        u.append(d["labor_market"]["unemployment_rate"])
        inf.append(d["macro"]["inflation_rate"])
        rgdp.append(d["macro"]["real_gdp"])
    return np.array(u), np.array(inf), np.array(rgdp)


def _run(path):
    inner = sorted(glob.glob(os.path.join(path, "run_*")))
    return inner[-1] if inner else path


def _regularities(u, inf, rgdp):
    g = np.diff(np.log(np.where(rgdp > 0, rgdp, np.nan))) * 100
    du = np.diff(u) * 100
    ud = _detrend(u) * 100
    infd = _detrend(inf) * 100
    return _best_neg_lag(du, g), _best_neg_lag(ud, infd), g, du, ud, infd


# ---- Figure 06: LLM regularity scatter -------------------------------------
def fig_llm_regularities():
    okun_x, okun_y, phil_x, phil_y = [], [], [], []
    okun_rs, phil_rs = [], []
    for d in LLM_DIRS:
        u, inf, rgdp = _load_dir(_run(d))
        (Lo, ro), (Lp, rp), g, du, ud, infd = _regularities(u, inf, rgdp)
        xo = du[Lo:] if Lo else du; yo = g[:len(g) - Lo] if Lo else g
        n = min(len(xo), len(yo)); okun_x += list(xo[:n]); okun_y += list(yo[:n]); okun_rs.append(ro)
        xp = ud[Lp:] if Lp else ud; yp = infd[:len(infd) - Lp] if Lp else infd
        n = min(len(xp), len(yp)); phil_x += list(xp[:n]); phil_y += list(yp[:n]); phil_rs.append(rp)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("LLM-agent economy — emergent negative regularities (monthly-LLM, seeds pooled)",
                 fontsize=13)

    def panel(ax, X, Y, rs, title, xl, yl):
        X = np.array(X); Y = np.array(Y)
        m = ~(np.isnan(X) | np.isnan(Y)); X, Y = X[m], Y[m]
        ax.scatter(X, Y, s=16, alpha=0.4, color="#59a14f", edgecolors="none")
        b, a = np.polyfit(X, Y, 1)
        xs = np.linspace(X.min(), X.max(), 50)
        ax.plot(xs, a + b * xs, "r-", lw=2.5, label=f"slope={b:+.2f}\npooled r={_pearson(X, Y):+.2f}")
        neg = sum(1 for r in rs if r < 0)
        ax.axhline(0, color="gray", lw=0.6, ls=":"); ax.axvline(0, color="gray", lw=0.6, ls=":")
        ax.set_title(f"{title}\nper-seed r={np.nanmean(rs):+.2f}  ({neg}/{len(rs)} negative)", fontsize=11)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.legend(loc="upper right", fontsize=9)

    panel(a1, okun_x, okun_y, okun_rs, "Okun's law (LLM)", "Δ Unemployment (pp)", "Real GDP growth (%)")
    panel(a2, phil_x, phil_y, phil_rs, "Phillips curve (LLM)", "Unemployment (detrended, pp)", "Inflation (detrended, %)")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    p = f"{FIGDIR}/06_llm_regularities.png"; plt.savefig(p, dpi=130); plt.close(); print("saved", p)


# ---- Figure 07: LLM vs rules bars ------------------------------------------
def _metrics_over(dirs_or_seeds, is_llm):
    uvol, okun, phil, mpc = [], [], [], []
    for item in dirs_or_seeds:
        rd = _run(item) if is_llm else _run(f"output/calibgrid_{item}_ls035")
        u, inf, rgdp = _load_dir(rd)
        (Lo, ro), (Lp, rp), *_ = _regularities(u, inf, rgdp)
        uvol.append(float(np.nanstd(_detrend(u)) * 100)); okun.append(ro); phil.append(rp)
        # MPC cross-section from final month
        files = sorted(glob.glob(os.path.join(rd, "month_*.json")))
        d = json.load(open(files[-1])); hb = d.get("household", {}).get("by_household", {})
        inc = np.array([h.get("income", {}).get("total", np.nan) for h in hb.values()], float)
        con = np.array([h.get("consumption", {}).get("total", np.nan) for h in hb.values()], float)
        m = ~(np.isnan(inc) | np.isnan(con)) & (inc > 0)
        mpc.append(float(np.polyfit(inc[m], con[m], 1)[0]) if m.sum() > 5 else np.nan)
    return {k: (np.nanmean(v), np.nanstd(v)) for k, v in
            [("uvol", uvol), ("okun", okun), ("phil", phil), ("mpc", mpc)]}


def fig_llm_vs_rules():
    llm = _metrics_over(LLM_DIRS, True)
    rules = _metrics_over(RULE_SEEDS, False)
    labels = ["U volatility (pp)", "Okun", "Phillips", "MPC"]
    keys = ["uvol", "okun", "phil", "mpc"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    fig.suptitle("LLM-agent decisions vs rule-based decisions (calibrated economy)", fontsize=13)
    for ax, lab, k in zip(axes, labels, keys):
        rm, rs = rules[k]; lm, ls = llm[k]
        ax.bar([0, 1], [rm, lm], yerr=[rs, ls], capsize=5,
               color=["#4e79a7", "#59a14f"], width=0.6)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Rules", "LLM"])
        ax.set_title(lab, fontsize=11)
        ax.axhline(0, color="gray", lw=0.6)
        for i, v in enumerate([rm, lm]):
            ax.text(i, v, f"{v:+.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    p = f"{FIGDIR}/07_llm_vs_rules.png"; plt.savefig(p, dpi=130); plt.close(); print("saved", p)


# ---- Figure 08: LLM timeseries ---------------------------------------------
def fig_llm_timeseries():
    rd = _run(LLM_DIRS[0])
    files = sorted(glob.glob(os.path.join(rd, "month_*.json")))[BURNIN:]
    u, inf, rgdp = _load_dir(rd)
    g = np.diff(np.log(np.where(rgdp > 0, rgdp, np.nan))) * 100
    t = np.arange(len(u))
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("LLM-agent economy: macro time series (monthly-LLM, seed 12345)", fontsize=13)
    a1.plot(t, u * 100, color="#4e79a7"); a1.set_ylabel("Unemployment (%)"); a1.grid(alpha=0.3)
    colors = ["#59a14f" if v >= 0 else "#e15759" for v in inf]
    a2.bar(t, inf * 100, color=colors); a2.set_ylabel("Inflation (%)"); a2.grid(alpha=0.3)
    a3.plot(t[1:], g, color="#b07aa1"); a3.axhline(0, color="gray", lw=0.6)
    a3.set_ylabel("Real GDP growth (%)"); a3.set_xlabel("Month"); a3.grid(alpha=0.3)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    p = f"{FIGDIR}/08_llm_timeseries.png"; plt.savefig(p, dpi=130); plt.close(); print("saved", p)


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    fig_llm_regularities()
    fig_llm_vs_rules()
    fig_llm_timeseries()
