#!/usr/bin/env python3
"""fig:frontier — the certified frontier: the ranking, not the guard, carries the value (RQ3b).

Two stacked panels, certified grid-stress relief (x) vs carbon delta (y). Every plotted point ships
the identical ex-post guard and certifies; only the acceptance RANKING differs — so the spread IS
the ranking's contribution. The relief-first and footprint-first members span the frontier
(Prop. 2); the random-order control is dominated everywhere. Panel (a) also locates the unguarded
WaterWise weight sweep: every weight raises carbon (harm band), certifying 0/15 — the bang-bang
failure the guard exists to prevent.

Reads the committed guarded-control digests (experiments/mc1_guarded/{gpu_w1587_v3,
strong_favorable_v3}/mc1_digest.json — the source of truth behind tab:guarded). The WaterWise
sweep points are constants below, verified against the published sweep (16-node GPU fleet, 5
seeds, 0/15 certified): w=0.25 -> +6.6% C at 2.8 kWh; w=0.5 -> +2.5% C at 30.4 kWh;
w=0.75 -> +2.1% C at 26.3 kWh. Authored at print size (columnwidth); see paper_fig_style.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import paper_fig_style as sty  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

GPU = REPO / "experiments" / "mc1_guarded" / "gpu_w1587_v3" / "mc1_digest.json"
DROUGHT = REPO / "experiments" / "mc1_guarded" / "strong_german_v3" / "mc1_digest.json"  # v3 = German-basin fleet (Depth-1.5 fold 2026-07-07)

# mode -> (label, color, marker). Envelope members are stars: filled = relief-first,
# open = footprint-first (same green family — the ranking, not the hue, is the variable;
# purple stays WaterWise-only per the paper_fig_style contract).
STYLE = {
    "combined":          ("relief-first member", sty.ENV_GREEN, "*"),
    "combined_v2":       ("footprint-first member", sty.ENV_GREEN, "*"),
    "guarded_carbon":    ("guarded-carbon", sty.CARBON_BLUE, "o"),
    "guarded_waterwise": ("guarded-WaterWise", sty.WW_PURPLE, "D"),
    "guarded_random":    ("guarded-random (no ranking)", sty.RANDOM_GRAY, "v"),
}
OPEN_MARKER = {"combined_v2"}  # open star = footprint-first
WW_SWEEP = [(2.8, 6.6, "0.25"), (30.4, 2.5, "0.5"), (26.3, 2.1, "0.75")]  # (kWh, +C%, w)


def modes(path):
    d = json.load(open(path))
    return {m["mode"]: m for m in d["modes"]}


def draw(ax, recs, keys, label_off, ylim, xlim):
    ax.axhspan(0, ylim[1], color=sty.HARM_BG, alpha=0.06, zorder=0)
    ax.axhspan(ylim[0], 0, color=sty.HOLD_BG, alpha=0.07, zorder=0)
    ax.axhline(0, color="k", lw=0.8, zorder=2)
    for k in keys:
        m = recs[k]
        label, color, marker = STYLE[k]
        x, y = m["relief_kwh_mean"], m["carbon_pct_mean"]
        assert m["cert_rate"] == 1.0, f"{k} must certify (shared guard)"
        if k in OPEN_MARKER:
            ax.scatter([x], [y], facecolors="white", marker=marker, s=150,
                       edgecolors=color, linewidths=1.2, zorder=4)
        else:
            ax.scatter([x], [y], c=color, marker=marker, s=150 if marker == "*" else 45,
                       edgecolors="k", linewidths=0.5, zorder=4)
        dx, dy, *ha = label_off[k]
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=6.8, color=sty.ENV_DARK if color == sty.ENV_GREEN else color,
                    ha=ha[0] if ha else ("left" if dx >= 0 else "right"),
                    weight="bold", zorder=5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.2, zorder=1)
    ax.set_ylabel("carbon $\\Delta$ vs $B$ (%)")


def main() -> int:
    sty.apply()
    gpu, drt = modes(GPU), modes(DROUGHT)
    # long in-axes annotations confuse constrained layout; create with layout disabled
    # (set_layout_engine after creation is too late: subplots_adjust would be ignored)
    fig, (a, b) = plt.subplots(2, 1, figsize=(sty.COLW, 3.35), layout="none")
    fig.subplots_adjust(left=0.19, right=0.972, top=0.945, bottom=0.105, hspace=0.58)

    # (a) saturated GPU, divergent window: relief spans 0 -> 8.9 kWh at near-neutral carbon
    keys = ["combined", "combined_v2", "guarded_carbon", "guarded_waterwise", "guarded_random"]
    off = {"combined": (-8, -13), "combined_v2": (-31, -12, "left"),
           "guarded_carbon": (8, 5), "guarded_waterwise": (7, -11), "guarded_random": (11, -10)}
    draw(a, gpu, keys, off, ylim=(-1.12, 0.62), xlim=(-0.5, 10.2))
    fx = [gpu["combined_v2"]["relief_kwh_mean"], gpu["combined"]["relief_kwh_mean"]]
    fy = [gpu["combined_v2"]["carbon_pct_mean"], gpu["combined"]["carbon_pct_mean"]]
    a.plot(fx, fy, ls="--", color="gray", lw=0.9, zorder=2)  # the frontier (named in caption)
    a.text(0.2, 0.56, "unguarded WaterWise sweep:\n"
                      "every weight raises carbon — 0/15 certify\n"
                      "(+2.1…+6.6% C at 2.8–30.4 kWh)",
           fontsize=6.4, color=sty.WATER_RED, ha="left", va="top", weight="bold")
    a.annotate("", xy=(10.1, 0.30), xytext=(8.6, 0.30),
               arrowprops=dict(arrowstyle="-|>", color=sty.WATER_RED, lw=1.2))
    # the quotable ratio, ON the figure (figstory fix #4): guarded-random vs relief-first,
    # in its own bottom lane so it collides with nothing
    rr = gpu["combined"]["relief_kwh_mean"] / gpu["guarded_random"]["relief_kwh_mean"]
    assert 27 < rr < 31, f"random-vs-relief ratio drifted: {rr:.1f}"
    a.annotate("", xy=(gpu["combined"]["relief_kwh_mean"], -1.02),
               xytext=(gpu["guarded_random"]["relief_kwh_mean"], -1.02),
               arrowprops=dict(arrowstyle="-|>", color=sty.INK, lw=0.9))
    a.text(8.55, -0.99, "no ranking: 1/29 of the relief",
           fontsize=6.4, color=sty.INK, ha="right", va="bottom")
    a.set_title("(a) saturated GPU, divergent window (w1587)")
    a.set_xlabel("certified grid-stress relief (kWh/window)", labelpad=1.5)

    # (b) favorable drought regime: ranking decides whether the -31% carbon is captured
    keys_b = ["combined", "guarded_carbon", "guarded_waterwise", "guarded_random"]
    off_b = {"combined": (-8, -9), "guarded_carbon": (-8, 3), "guarded_waterwise": (6, -3),
             "guarded_random": (5, -3)}
    draw(b, drt, keys_b, off_b, ylim=(-35.5, 3.5), xlim=(-0.06, 1.22))
    b.set_title("(b) favorable drought regime (7 seeds)")
    b.set_xlabel("certified grid-stress relief (kWh/window)")

    sty.save(fig, "certified_frontier")
    print("gpu:", {k: (round(gpu[k]["relief_kwh_mean"], 2), round(gpu[k]["carbon_pct_mean"], 2))
                   for k in keys})
    print("drought:", {k: (round(drt[k]["relief_kwh_mean"], 2), round(drt[k]["carbon_pct_mean"], 2))
                       for k in keys_b})
    assert abs(gpu["combined"]["relief_kwh_mean"] - 8.90) < 0.05
    # v3 German-basin drought fleet (Depth-1.5 fold 2026-07-07): envelope trades carbon for
    # water/relief in the richer six-basin geometry; guarded-carbon carries the -28.3C member.
    assert abs(drt["combined"]["carbon_pct_mean"] + 17.87) < 0.1
    assert abs(drt["guarded_carbon"]["carbon_pct_mean"] + 28.31) < 0.1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
