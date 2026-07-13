#!/usr/bin/env python3
"""Plot the two robust NEGATIVE macro regularities (Okun's law, Phillips curve)
on the calibrated economy as clean pooled scatter + regression lines.

Uses the calibrated grid (output/calibgrid_*_ls035, 5 seeds). Same measurement
convention as the tables: drop 12-month burn-in, linear-detrend the cyclical
components, use the best-negative lag for each regularity. Points from all seeds
are pooled; a single OLS fit line + Pearson r are drawn.

Output: output/paper_figures/05_negative_regularities.png
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BURNIN = 12
SEEDS = [12345, 2026, 7777, 31337, 90210]
OUT = "output/paper_figures/05_negative_regularities.png"


def _detrend(x):
    x = np.asarray(x, float)
    t = np.arange(len(x))
    m = ~np.isnan(x)
    if m.sum() < 3:
        return x - np.nanmean(x)
    b = np.polyfit(t[m], x[m], 1)
    return x - np.polyval(b, t)


def _load(seed):
    rd = sorted(glob.glob(f"output/calibgrid_{seed}_ls035/run_*"))[-1]
    files = sorted(glob.glob(os.path.join(rd, "month_*.json")))[BURNIN:]
    u, inf, rgdp = [], [], []
    for f in files:
        d = json.load(open(f))
        u.append(d["labor_market"]["unemployment_rate"])
        inf.append(d["macro"]["inflation_rate"])
        rgdp.append(d["macro"]["real_gdp"])
    return np.array(u), np.array(inf), np.array(rgdp)


def _pearson(x, y):
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 4:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def _best_neg_lag(x, y, lags=(0, 1, 2, 3)):
    """Return (lag, r) with the most-negative correlation of x_t vs y_{t-lag}."""
    best = None
    for L in lags:
        xx = x[L:] if L else x
        yy = y[:len(y) - L] if L else y
        r = _pearson(xx, yy)
        if r == r and (best is None or r < best[1]):
            best = (L, r)
    return best if best else (0, np.nan)


def main():
    # collect pooled points for Okun and Phillips at their best-neg lag
    okun_x, okun_y = [], []
    phil_x, phil_y = [], []
    okun_rs, phil_rs = [], []
    for s in SEEDS:
        u, inf, rgdp = _load(s)
        g = np.diff(np.log(np.where(rgdp > 0, rgdp, np.nan))) * 100  # GDP growth %
        du = np.diff(u) * 100  # change in unemployment, pp
        ud = _detrend(u) * 100  # detrended unemployment, pp (for Phillips)
        infd = _detrend(inf) * 100  # detrended inflation, %

        # Okun: corr(Δu, GDP growth), best negative lag
        Lo, ro = _best_neg_lag(du, g)
        xo = du[Lo:] if Lo else du
        yo = g[:len(g) - Lo] if Lo else g
        n = min(len(xo), len(yo))
        okun_x += list(xo[:n]); okun_y += list(yo[:n]); okun_rs.append(ro)

        # Phillips: corr(u, inflation), best negative lag
        Lp, rp = _best_neg_lag(ud, infd)
        xp = ud[Lp:] if Lp else ud
        yp = infd[:len(infd) - Lp] if Lp else infd
        n = min(len(xp), len(yp))
        phil_x += list(xp[:n]); phil_y += list(yp[:n]); phil_rs.append(rp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Emergent negative macro regularities — calibrated economy (5 seeds pooled)",
                 fontsize=13)

    def _panel(ax, X, Y, rs, title, xlabel, ylabel):
        X = np.array(X); Y = np.array(Y)
        m = ~(np.isnan(X) | np.isnan(Y))
        X, Y = X[m], Y[m]
        ax.scatter(X, Y, s=14, alpha=0.35, color="#4e79a7", edgecolors="none")
        # pooled OLS fit
        b, a = np.polyfit(X, Y, 1)
        xs = np.linspace(X.min(), X.max(), 50)
        ax.plot(xs, a + b * xs, "r-", lw=2.5,
                label=f"fit slope={b:+.2f}\npooled r={_pearson(X, Y):+.2f}")
        r_mean = np.nanmean(rs); r_std = np.nanstd(rs)
        neg = sum(1 for r in rs if r < 0)
        ax.axhline(0, color="gray", lw=0.6, ls=":")
        ax.axvline(0, color="gray", lw=0.6, ls=":")
        ax.set_title(f"{title}\nper-seed r = {r_mean:+.2f}±{r_std:.2f}  ({neg}/{len(rs)} negative)",
                     fontsize=11)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize=9)

    _panel(ax1, okun_x, okun_y, okun_rs,
           "Okun's law", "Δ Unemployment (pp, best lag)", "Real GDP growth (%)")
    _panel(ax2, phil_x, phil_y, phil_rs,
           "Phillips curve", "Unemployment (detrended, pp, best lag)", "Inflation (detrended, %)")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.savefig(OUT, dpi=130)
    print(f"saved {OUT}")
    print(f"Okun pooled r={_pearson(np.array(okun_x), np.array(okun_y)):+.2f}, "
          f"Phillips pooled r={_pearson(np.array(phil_x), np.array(phil_y)):+.2f}")


if __name__ == "__main__":
    main()
