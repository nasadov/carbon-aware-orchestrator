#!/usr/bin/env python3
"""Pool all regime_*/tidy.csv rows and fit a simple predictive summary for the
certified grid-relief benefit. Reports per-knob monotone trends and a standardized
OLS regression of grid_relief_pct on the candidate drivers, so we can say which
knob(s) most strongly predict large certified benefit and which are dead ends.
"""
from __future__ import annotations
import csv
import glob
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# Spatial CI arbitrage spread (mean max-min across regions, g/kWh), precomputed per signal dir.
SIGNAL_SPREAD = {
    "timealigned": 393.0, "timealigned_realci": 442.0,
    "timealigned_2017": 356.0, "timealigned_s0520": 336.0,
}


def load_all(paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
    return rows


def f(x, default=np.nan):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def ols_standardized(X, y, feats, label):
    mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    Xd = np.column_stack([np.ones(len(Xs)), Xs])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    yhat = Xd @ beta
    ss_res = np.sum((y - yhat) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"\n[{label}] standardized OLS  R^2={r2:.3f}  (n={len(y)}, intercept={beta[0]:.3f})")
    order = np.argsort(-np.abs(beta[1:]))
    for i in order:
        r = np.corrcoef(X[:, i], y)[0, 1] if X[:, i].std() > 0 else float("nan")
        print(f"    {feats[i]:>16}: beta={beta[i+1]:+.3f}  (univariate r={r:+.3f})")


def main():
    paths = sorted(glob.glob(str(REPO / "experiments" / "regime_*" / "tidy.csv")))
    rows = load_all(paths)
    print(f"pooled {len(rows)} runs from {len(paths)} sweeps")

    # Drivers. headroom = (1 - oversub_real) clipped at 0 -> spare clean capacity to migrate into.
    # oversub^2 captures the inverted-U in grid relief (peaks at moderate load).
    feats = ["flexible_pct", "oversub_real", "oversub_sq", "headroom", "mean_flex_slack", "signal_spread"]
    X = {"grid": [], "carbon": [], "water": []}
    M, keep = [], []
    for r in rows:
        spread = SIGNAL_SPREAD.get(r.get("signals", ""), np.nan)
        ov = f(r.get("oversub_real"))
        xrow = [f(r.get("flexible_pct")), ov, ov * ov, max(0.0, 1.0 - ov),
                f(r.get("mean_flex_slack")), spread]
        yg = f(r.get("grid_relief_pct")); yc = f(r.get("carbon_pct")); yw = f(r.get("water_pct"))
        if any(np.isnan(v) for v in xrow) or np.isnan(yg):
            continue
        M.append(xrow); keep.append(r)
        X["grid"].append(yg); X["carbon"].append(yc); X["water"].append(yw)
    M = np.array(M)

    print("\n=== predictive summary: standardized OLS per benefit axis ===")
    ols_standardized(M, np.array(X["grid"]), feats, "grid_relief_pct")
    ols_standardized(M, np.array(X["carbon"]), feats, "carbon_pct (more negative = bigger co-benefit)")
    ols_standardized(M, np.array(X["water"]), feats, "water_pct (more negative = bigger co-benefit)")

    # Certification across ALL pooled runs.
    cert = np.array([f(r.get("certified")) for r in rows])
    print(f"\nenvelope certified in {int(np.nansum(cert))}/{len(cert)} pooled runs "
          f"({100*np.nanmean(cert):.0f}%)")
    bh = np.array([f(r.get("n_baselines_harm")) for r in rows])
    print(f"mean action-baselines harming per run: {np.nanmean(bh):.2f}/3")


if __name__ == "__main__":
    sys.exit(main())
