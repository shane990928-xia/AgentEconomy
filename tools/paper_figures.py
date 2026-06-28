#!/usr/bin/env python3
"""
Build the paper's emergence figures from the on-disk run grids.

Uses the SAME measurement convention as tools/grid_table.py and
tools/ablation_table.py (drop first `burnin` formal months, linear-detrend the
cyclical component, Okun/Phillips reported at the best/most-negative lag in
0..3) so the figures and the tables agree exactly.

Regimes (each = a set of run dirs holding month_*.json):
    hardcoded   output/grid_*        committed economy, both reduced-form
                                     shortcuts ON (endogenous-wage formula +
                                     cost-push) — the OLD result.
    bid_rules   output/bidgrid_*     firm wage bidding, BOTH shortcuts OFF,
                                     rules-mode — the emergent mechanism.
    bid_llm     output/bidllm_*      firm wage bidding, shortcuts OFF, LLM
                                     consumption layer.

Figures written to output/paper_figures/:
    01_regime_comparison.png   grouped bars: Okun/Beveridge/Phillips mean±std
                               per regime (the hardcoded-vs-emergent headline).
    02_emergence_scatter.png   pooled detrended scatter (Beveridge / Okun /
                               Phillips) for the emergent rules-mode economy,
                               every run overlaid + pooled regression line.
    03_firm_size_distribution.png  firm revenue/wage-bill rank-size (final month).
    04_timeseries.png          U / inflation / real-GDP-growth for a sample run.

Usage:
    python tools/paper_figures.py [--burnin 12]
"""
import glob
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False

REGIMES = [
    ("hardcoded", "output/grid_*", "Hardcoded\n(shortcuts ON)", "#e15759"),
    ("bid_rules", "output/bidgrid_*", "Wage bidding\n(rules)", "#4e79a7"),
    ("bid_llm", "output/bidllm_*", "Wage bidding\n(LLM)", "#59a14f"),
    ("calibrated", "output/calibgrid_*_ls035", "Calibrated\n(+hoarding+Phase2/3)", "#b07aa1"),
]
OUT = Path("output/paper_figures")


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
    vals = [_corr(x, y) if L == 0 else _corr(x[L:], y[:-L]) for L in lags]
    cand = [v for v in vals if v == v]
    return min(cand) if cand else float("nan")


def _run_dir(path):
    inner = sorted(glob.glob(f"{path}/run_*"))
    return inner[-1] if inner else path


def _load_series(run_dir, burnin):
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
    return (np.array(u, float), np.array(v, float),
            np.array(inf, float), np.array(rgdp, float))


def measure(run_dir, burnin):
    s = _load_series(run_dir, burnin)
    if s is None:
        return None
    u, v, inf, rgdp = s
    uc, vc, infc = _detrend(u), _detrend(v), _detrend(inf)
    gg = np.diff(np.log(np.where(rgdp > 0, rgdp, np.nan))) * 100
    du = np.diff(u)
    ok = [_corr(du, gg)] + [_corr(du[L:], gg[:-L]) for L in (1, 2, 3)]
    okc = [z for z in ok if z == z]
    return {
        "beveridge": _corr(uc, vc),
        "okun": min(okc) if okc else float("nan"),
        "phillips": _best_neg_lag(uc, infc),
    }


def regime_runs(glob_pat):
    dirs = sorted(glob.glob(glob_pat))
    return [_run_dir(d) for d in dirs]


def fig_regime_comparison(burnin):
    metrics = ["okun", "beveridge", "phillips"]
    labels = ["Okun (best lag)", "Beveridge", "Phillips (best lag)"]
    means = {}; stds = {}; negfrac = {}
    for key, pat, _, _ in REGIMES:
        vals = {mt: [] for mt in metrics}
        for rd in regime_runs(pat):
            r = measure(rd, burnin)
            if not r:
                continue
            for mt in metrics:
                if r[mt] == r[mt]:
                    vals[mt].append(r[mt])
        means[key] = [np.mean(vals[mt]) if vals[mt] else np.nan for mt in metrics]
        stds[key] = [np.std(vals[mt]) if vals[mt] else 0.0 for mt in metrics]
        negfrac[key] = [
            (sum(1 for x in vals[mt] if x < 0), len(vals[mt])) for mt in metrics
        ]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(metrics))
    n = len(REGIMES)
    w = 0.8 / n
    for i, (key, _, disp, color) in enumerate(REGIMES):
        off = (i - (n - 1) / 2) * w
        bars = ax.bar(x + off, means[key], w, yerr=stds[key], capsize=4,
                      label=disp, color=color, alpha=0.85,
                      error_kw=dict(ecolor="gray", lw=1.2))
        for j, b in enumerate(bars):
            neg, tot = negfrac[key][j]
            if tot:
                ax.annotate(f"{neg}/{tot}",
                            (b.get_x() + b.get_width() / 2, means[key][j]),
                            textcoords="offset points",
                            xytext=(0, -14 if means[key][j] < 0 else 8),
                            ha="center", fontsize=8, color="black")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Correlation (empirically correct sign: negative)", fontsize=11)
    ax.set_title("Emergent macro regularities: hardcoded shortcuts vs. genuine "
                 "wage bidding\n(annotations = runs with correct negative sign)",
                 fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = OUT / "01_regime_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}")


def _pooled_scatter(ax, xs, ys, xlabel, ylabel, title, cmap):
    allx = np.concatenate(xs) if xs else np.array([])
    ally = np.concatenate(ys) if ys else np.array([])
    for i, (xr, yr) in enumerate(zip(xs, ys)):
        ax.scatter(xr, yr, s=40, alpha=0.55, edgecolors="none",
                   color=plt.get_cmap(cmap)(i / max(1, len(xs) - 1)))
    m = ~(np.isnan(allx) | np.isnan(ally))
    if m.sum() >= 4:
        z = np.polyfit(allx[m], ally[m], 1)
        xl = np.linspace(np.nanmin(allx), np.nanmax(allx), 100)
        corr = np.corrcoef(allx[m], ally[m])[0, 1]
        ax.plot(xl, np.poly1d(z)(xl), "r--", lw=2,
                label=f"pooled slope={z[0]:.2f}, corr={corr:+.2f}")
        ax.legend(fontsize=9)
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.axvline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel(xlabel, fontsize=11); ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12); ax.grid(True, alpha=0.3)


def fig_emergence_scatter(burnin, glob_pat="output/bidgrid_*"):
    bx, by = [], []  # Beveridge
    ox, oy = [], []  # Okun
    px, py = [], []  # Phillips (best lag applied per-run)
    for rd in regime_runs(glob_pat):
        s = _load_series(rd, burnin)
        if s is None:
            continue
        u, v, inf, rgdp = s
        uc, vc, infc = _detrend(u), _detrend(v), _detrend(inf)
        bx.append(uc); by.append(vc)
        gg = np.diff(np.log(np.where(rgdp > 0, rgdp, np.nan))) * 100
        du = np.diff(u)
        # Okun at best-neg lag
        cand = {0: _corr(du, gg)}
        for L in (1, 2, 3):
            cand[L] = _corr(du[L:], gg[:-L])
        bestL = min((L for L in cand if cand[L] == cand[L]),
                    key=lambda L: cand[L], default=0)
        if bestL == 0:
            ox.append(du); oy.append(gg[:len(du)])
        else:
            ox.append(du[bestL:]); oy.append(gg[:len(du) - bestL])
        # Phillips at best-neg lag
        cand = {0: _corr(uc, infc)}
        for L in (1, 2, 3):
            cand[L] = _corr(uc[L:], infc[:-L])
        bestL = min((L for L in cand if cand[L] == cand[L]),
                    key=lambda L: cand[L], default=0)
        if bestL == 0:
            px.append(uc); py.append(infc)
        else:
            px.append(uc[bestL:]); py.append(infc[:-bestL])

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    _pooled_scatter(axes[0], bx, by, "Unemployment (detrended)",
                    "Vacancy rate (detrended)", "Beveridge curve", "viridis")
    _pooled_scatter(axes[1], ox, oy, "ΔUnemployment (best lag)",
                    "Real GDP growth (%)", "Okun's law", "plasma")
    _pooled_scatter(axes[2], px, py, "Unemployment (detrended, best lag)",
                    "Inflation (detrended)", "Phillips curve", "cividis")
    fig.suptitle("Emergent regularities — wage-bidding economy, no shortcuts "
                 "(all runs pooled)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = OUT / "02_emergence_scatter.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}")


def fig_firm_size(glob_pat="output/bidgrid_*"):
    rd = regime_runs(glob_pat)
    if not rd:
        print("no runs for firm-size fig"); return
    files = sorted(glob.glob(f"{rd[0]}/month_*.json"))
    if not files:
        return
    d = json.load(open(files[-1]))
    ff = d.get("details", {}).get("firm_financials", {})
    rev = sorted([max(0.0, float(v.get("monthly_income", 0.0) or 0.0))
                  for v in ff.values()], reverse=True)
    wb = d.get("details", {}).get("wages", {}).get("by_firm", {})
    wbv = sorted([max(0.0, float(x or 0.0)) for x in wb.values()], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, data, name, color in (
        (axes[0], rev, "Firm revenue", "#4e79a7"),
        (axes[1], wbv, "Firm wage bill", "#59a14f"),
    ):
        data = [x for x in data if x > 0]
        if len(data) >= 3:
            rank = np.arange(1, len(data) + 1)
            ax.loglog(rank, data, "o", color=color, alpha=0.8, ms=6)
            ax.set_xlabel("Rank (log)", fontsize=11)
            ax.set_ylabel(f"{name} ($, log)", fontsize=11)
            ax.set_title(f"{name} rank-size (final month, n={len(data)})",
                         fontsize=12)
            ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Firm-size distribution (right-skewed, heavy-tailed)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = OUT / "03_firm_size_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}")


def fig_timeseries(burnin, glob_pat="output/bidgrid_*"):
    rd = regime_runs(glob_pat)
    if not rd:
        print("no runs for timeseries"); return
    s = _load_series(rd[0], burnin)
    if s is None:
        return
    u, v, inf, rgdp = s
    t = np.arange(len(u))
    gg = np.concatenate([[np.nan], np.diff(np.log(np.where(rgdp > 0, rgdp, np.nan))) * 100])

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(t, u * 100, "b-o", ms=4); axes[0].set_ylabel("Unemployment (%)")
    axes[0].grid(True, alpha=0.3); axes[0].set_title("Sample run — wage-bidding economy")
    axes[1].bar(t, inf * 100, color=["#59a14f" if x >= 0 else "#e15759" for x in inf], alpha=0.8)
    axes[1].axhline(0, color="k", lw=0.5); axes[1].set_ylabel("Inflation (%)")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(t, gg, "m-s", ms=4); axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_ylabel("Real GDP growth (%)"); axes[2].set_xlabel("Formal month (post burn-in)")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    p = OUT / "04_timeseries.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--burnin", type=int, default=12)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_regime_comparison(a.burnin)
    fig_emergence_scatter(a.burnin)
    fig_firm_size()
    fig_timeseries(a.burnin)
    print(f"\nAll paper figures in {OUT}/")


if __name__ == "__main__":
    main()
