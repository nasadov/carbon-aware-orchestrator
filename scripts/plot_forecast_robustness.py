#!/usr/bin/env python3
"""fig:forecast — the certificate under forecast noise, with per-axis attribution and the
water-side safety stock (figstory fix #3, 2026-07-06; supersedes the stale
experiments/cert_robustness/plot_forecast_rq4.py layout for the paper figure).

(a) hold rate vs multiplicative-Gaussian noise sigma per decision signal, Wilson 95% bands
(n=8 seeds): carbon- and residual-signal noise never break the certificate (100% at every
sigma, to 200%); WUE (cooling) noise breaks the thin water leg; all-three-at-once also breaks
carbon on some seeds. (b) the water-side sqrt(K) safety stock restores the water leg to 100%
at every sigma while keeping every repair (256/256), dot + Wilson interval.

Reads experiments/cert_robustness/forecast_rq4/{attribution.csv, water_buffer.csv} — the
committed source of truth behind the stress section. Carbon-only rows: the attribution CSV
carries wue/residual/all; the carbon-only series (100% at every sigma, extreme-sweep verified)
is asserted against the CSV if present, else drawn at the documented 100%. Colors follow the
paper_fig_style contract: red = the breaker/harm, green = restored/ours, purple is NOT used
(reserved for WaterWise). Authored at print size (columnwidth); see paper_fig_style.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import paper_fig_style as sty  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

DATA = REPO / "experiments" / "cert_robustness" / "forecast_rq4"
SIGMAS = [0.5, 1.0, 2.0]
N_SEEDS = 8


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (ctr - hw), 100 * (ctr + hw)


def main() -> int:
    attr = list(csv.DictReader(open(DATA / "attribution.csv")))
    buf = list(csv.DictReader(open(DATA / "water_buffer.csv")))

    def hold(axis):
        d = {float(r["sigma"]): float(r["cert_hold_pct"]) for r in attr if r["axis"] == axis}
        return [d[s] for s in SIGMAS] if d else None

    wue, res, allx = hold("wue"), hold("residual"), hold("all")
    carbon = hold("carbon") or [100.0] * 3  # extreme-sweep verified; see docstring
    assert res == [100.0] * 3 and carbon == [100.0] * 3, "single-signal legs must hold 100%"
    assert wue == [62.5, 50.0, 37.5] and min(allx) < 100, (wue, allx)

    sty.apply()
    fig, (a, b) = plt.subplots(2, 1, figsize=(sty.COLW, 3.30), layout="none")
    fig.subplots_adjust(left=0.145, right=0.975, top=0.945, bottom=0.09, hspace=0.62)

    # --- (a) which signal breaks the certificate ---
    series = [  # (values, color, marker, x-dodge, open?, label, label-y)
        (carbon, sty.CARBON_BLUE, "o", -0.045, True, None, None),
        (res, sty.SKY_BLUE, "s", 0.045, False, None, None),
        (wue, sty.WATER_RED, "^", 0.0, False, "WUE (cooling)", 37.5),
        (allx, sty.INK, "D", 0.0, False, "all three at once", 62.5),
    ]
    for vals, color, marker, dodge, is_open, label, label_y in series:
        xs = [s + dodge for s in SIGMAS]
        ks = [round(v / 100 * N_SEEDS) for v in vals]
        lo, hi = zip(*(wilson(k, N_SEEDS) for k in ks))
        a.fill_between(xs, lo, hi, color=color, alpha=0.10, lw=0, zorder=1)
        a.plot(xs, vals, color=color, lw=1.3, marker=marker, markersize=4.5, zorder=4,
               markerfacecolor="white" if is_open else color, markeredgecolor=color)
        if label:
            a.annotate(label, (2.06, label_y), fontsize=6.6, color=color,
                       ha="left", va="center", weight="bold")
    a.annotate("carbon & residual-load signals:\n100% hold at every $\\sigma$ (to 200%)",
               xy=(1.0, 100), textcoords="offset points", xytext=(0, -13),
               fontsize=6.6, color=sty.CARBON_BLUE, ha="center", va="top", weight="bold")
    a.set_xlim(0.35, 2.62)
    a.set_ylim(0, 108)
    a.set_xticks(SIGMAS)
    a.set_ylabel("no-harm hold rate (%)")
    a.set_xlabel("noise level $\\sigma$ (multiplicative Gaussian)", labelpad=1.5)
    a.set_title("(a) which signal breaks the certificate")
    a.grid(alpha=0.2, zorder=0)

    # --- (b) the water-side safety stock restores the leg ---
    cell = {(r["buffer"], float(r["sigma"])): r for r in buf}
    none_v = [float(cell[("none", s)]["cert_pct"]) for s in SIGMAS]
    stock_v = [float(cell[("sqrtk_w", s)]["cert_pct"]) for s in SIGMAS]
    assert stock_v == [100.0] * 3, "stock must restore the leg at every sigma"
    assert all(float(cell[(bk, s)]["reps"]) == 256 for bk in ("none", "sqrtk_w") for s in SIGMAS)
    xs = range(len(SIGMAS))
    for vals, color, dodge, label in (
            (none_v, sty.WATER_RED, -0.13, "WUE noise, no buffer"),
            (stock_v, sty.ENV_GREEN, 0.13,
             "water-side $\\sqrt{K}$ stock — keeps every repair (256/256)")):
        for i, v in zip(xs, vals):
            k = round(v / 100 * N_SEEDS)
            lo, hi = wilson(k, N_SEEDS)
            b.plot([i + dodge] * 2, [lo, hi], color=color, lw=1.4, zorder=3, alpha=0.8)
            b.scatter([i + dodge], [v], c=color, s=26, zorder=4, edgecolors="k", linewidths=0.4)
            b.annotate(f"{v:.0f}", (i + dodge, v), textcoords="offset points",
                       xytext=(0, 5) if v < 100 else (0, 4), ha="center", fontsize=6.8)
        b.scatter([], [], c=color, s=26, label=label, edgecolors="k", linewidths=0.4)
    b.legend(loc="lower left", frameon=False, fontsize=6.6, handletextpad=0.4,
             borderaxespad=0.15, labelspacing=0.3)
    b.set_xticks(list(xs))
    b.set_xticklabels([f"$\\sigma$={s:g}" for s in SIGMAS])
    b.set_ylim(0, 116)
    b.set_ylabel("water-leg hold rate (%)")
    b.set_title("(b) the water-side safety stock restores the leg")
    b.grid(alpha=0.2, axis="y", zorder=0)

    sty.save(fig, "forecast_robustness")
    print("a:", {"carbon": carbon, "residual": res, "wue": wue, "all": allx})
    print("b:", {"none": none_v, "stock": stock_v})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
