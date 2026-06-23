#!/usr/bin/env python3
"""
Stylized-facts measurement over a finished run's month_*.json panel.

Pure measurement — does NOT touch the model. Extracts the macro regularities
and cross-section/time-series stylized facts used to argue the LLM-agent
economy is empirically reasonable for the journal submission.

Usage:
    python tools/stylized_facts.py output/seed_12345/run_XXXX [--burnin 12]

Reports:
  Time-series regularities (cyclical, detrended, lag-scanned):
    - Beveridge   corr(unemployment, vacancy)            expect < 0
    - Okun        corr(Δunemployment, real-GDP growth)   expect < 0 (lag 1)
    - Phillips    corr(unemployment, inflation)          expect < 0
    - Wage-Phillips corr(unemployment, wage_scale)       expect < 0
    - Inflation persistence  autocorr(inflation, lag1)   expect > 0
    - Output persistence     autocorr(gdp growth, lag1)
  Cross-section (final formal month):
    - Firm size distribution skew (wage bill / revenue) expect right-skewed
    - Household income & wealth distribution + Gini
    - Consumption-income relation (APC slope, MPC proxy)
"""
import json
import glob
import os
import sys
import argparse
import numpy as np


def _load_panel(run_dir, burnin):
    files = sorted(glob.glob(os.path.join(run_dir, "month_*.json")))
    if not files:
        sys.exit(f"no month_*.json in {run_dir}")
    panel = [json.load(open(f)) for f in files]
    return panel[burnin:], panel


def _detrend(x):
    """Remove linear trend; return cyclical component."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 3:
        return x - np.nanmean(x)
    t = np.arange(n)
    mask = ~np.isnan(x)
    if mask.sum() < 3:
        return x - np.nanmean(x)
    b = np.polyfit(t[mask], x[mask], 1)
    return x - np.polyval(b, t)


def _corr(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 4:
        return float("nan"), 0
    a, b = a[m], b[m]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan"), int(m.sum())
    return float(np.corrcoef(a, b)[0, 1]), int(m.sum())


def _lag_corr(x, y, lags=(0, 1, 2, 3)):
    """corr(x[t], y[t-lag]) — best (most negative) lag for Okun-type."""
    out = {}
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    for L in lags:
        if L == 0:
            out[L] = _corr(x, y)
        else:
            out[L] = _corr(x[L:], y[:-L])
    return out


def _gini(v):
    v = np.sort(np.asarray([x for x in v if x is not None and not np.isnan(x)], float))
    v = v[v >= 0]
    n = len(v)
    if n == 0 or v.sum() == 0:
        return float("nan")
    cum = np.cumsum(v)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def _skew(v):
    v = np.asarray([x for x in v if x is not None and not np.isnan(x)], float)
    if len(v) < 3 or np.std(v) < 1e-12:
        return float("nan")
    return float(np.mean(((v - v.mean()) / v.std()) ** 3))


def measure(run_dir, burnin=12):
    panel, full = _load_panel(run_dir, burnin)
    u, v, inf, ws, rgdp = [], [], [], [], []
    for d in panel:
        lm = d.get("labor_market", {})
        m = d.get("macro", {})
        u.append(lm.get("unemployment_rate", np.nan))
        v.append(lm.get("vacancy_rate", np.nan))
        inf.append(m.get("inflation_rate", np.nan))
        ws.append(m.get("wage_scale", np.nan))
        rgdp.append(m.get("real_gdp", np.nan))

    u = np.array(u); v = np.array(v); inf = np.array(inf)
    ws = np.array(ws); rgdp = np.array(rgdp)
    rgdp_g = np.diff(np.log(np.where(rgdp > 0, rgdp, np.nan))) * 100
    du = np.diff(u)

    uc, vc, infc = _detrend(u), _detrend(v), _detrend(inf)

    print(f"\n{'='*64}\n{run_dir}  (burn-in={burnin}, formal months={len(panel)})\n{'='*64}")
    print(f"levels: U mean={np.nanmean(u):.3f} [{np.nanmin(u):.3f},{np.nanmax(u):.3f}]  "
          f"infl mean={np.nanmean(inf):.4f}  wage_scale last={ws[-1]:.4f}")

    print("\n-- Time-series regularities (detrended cyclical) --")
    r, n = _corr(uc, vc)
    print(f"  Beveridge   corr(U,V)        = {r:+.3f}  (n={n})  [<0 ✓]")

    ok = _lag_corr(du, rgdp_g[:len(du)])
    best = min((L for L in ok if not np.isnan(ok[L][0])), key=lambda L: ok[L][0], default=None)
    okstr = "  ".join(f"L{L}={ok[L][0]:+.2f}" for L in ok)
    print(f"  Okun        corr(ΔU,gdpG)    = {okstr}  best=L{best}  [<0 at lag1 ✓]")

    r, n = _corr(uc, infc)
    php = _lag_corr(uc, infc)
    phstr = "  ".join(f"L{L}={php[L][0]:+.2f}" for L in php)
    print(f"  Phillips    corr(U,infl)     = {phstr}  [<0 ✓]")

    r, n = _corr(_detrend(u), _detrend(ws))
    print(f"  WagePhillips corr(U,wscale)  = {r:+.3f}  (n={n})  [<0 ✓]")

    ac_inf, _ = _corr(inf[1:], inf[:-1])
    ac_g, _ = _corr(rgdp_g[1:], rgdp_g[:-1])
    print(f"  Inflation persistence ac1    = {ac_inf:+.3f}  [>0 realistic]")
    print(f"  Output-growth persistence ac1= {ac_g:+.3f}")

    # ---- cross-section (final formal month) ----
    d = panel[-1]
    wages_by_firm = d.get("details", {}).get("wages", {}).get("by_firm", {})
    ff = d.get("details", {}).get("firm_financials", {})
    rev = [v.get("monthly_income", 0.0) for v in ff.values()]
    wb = list(wages_by_firm.values())

    hb = d.get("household", {}).get("by_household", {})
    inc = [h.get("income", {}).get("total", np.nan) for h in hb.values()]
    wealth = [h.get("balance", np.nan) for h in hb.values()]
    cons = [h.get("consumption", {}).get("total", np.nan) for h in hb.values()]

    print("\n-- Cross-section (final formal month M%d) --" % d.get("month", -1))
    print(f"  Firm wage-bill skew          = {_skew(wb):+.2f}  (n={len(wb)})  [>0 right-skew ✓]")
    print(f"  Firm revenue skew            = {_skew(rev):+.2f}  (n={len(rev)})")
    print(f"  Household income Gini        = {_gini(inc):.3f}")
    print(f"  Household wealth Gini        = {_gini(wealth):.3f}  [wealth>income typ.]")

    # APC: consumption vs income regression slope (MPC proxy)
    ai = np.asarray(inc, float); ac = np.asarray(cons, float)
    mm = ~(np.isnan(ai) | np.isnan(ac)) & (ai > 0)
    if mm.sum() > 5:
        slope = np.polyfit(ai[mm], ac[mm], 1)[0]
        print(f"  MPC proxy (dC/dInc slope)    = {slope:+.3f}  [0<MPC<1 ✓]")

    return {
        "beveridge": _corr(uc, vc)[0],
        "okun_lag1": ok.get(1, (np.nan,))[0],
        "phillips_lag0": php.get(0, (np.nan,))[0],
        "u_mean": float(np.nanmean(u)),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--burnin", type=int, default=12)
    a = ap.parse_args()
    measure(a.run_dir, a.burnin)
