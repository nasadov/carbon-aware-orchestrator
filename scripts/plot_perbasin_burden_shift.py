#!/usr/bin/env python3
"""fig:burden — "the sum hides the harm": per-watershed scarcity-water deltas on the Azure
headline window (d0) as a ranked dumbbell/dot plot (BP §11 form; figstory fix #1, 2026-07-06).
Basins on y sorted by worst rise across the two aggregate-winners, fleet aggregate separated
below; x = scarcity-water Δ% with the zero no-harm line. The aggregate wins (-6.0 / -8.8%) hide
a rising watershed (Seine +7.5 / Tagus +9.4) — those two dots wear the red ring that means
"fails Eq. (1)" everywhere in the paper (same vocabulary as fig:plane); the per-basin guard
turns every dot non-positive while landing the LARGEST aggregate cut (-10.5%).

Reads experiments/water_basin/basin_certification.json (the committed source of truth behind
tab:realtrace_perbasin). The DE basin bears no charged scarcity in this window (base 0) and is
omitted. Markers: carbon-greedy o (blue), aggregate-guarded member s (orange), per-basin-guarded
member * (green) — shape-redundant for grayscale. Authored at print size; see paper_fig_style.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import paper_fig_style as sty  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.lines as mlines  # noqa: E402

WINDOW = "azure_packing_2020_d0_s42_200"
BASIN_NAME = {"FR": "Seine (FR)", "ES": "Tagus (ES)", "IT-NO": "Po (IT-NO)"}


def main() -> int:
    data = json.load(open(REPO / "experiments" / "water_basin" / "basin_certification.json"))
    d0 = next(e for e in data if e["window"] == WINDOW)
    cg = d0["methods"]["carbon"]
    agg = d0["methods"]["no_harm_flex"]
    pbg = d0["per_basin_guarded"]["no_harm_flex"]
    # SEMANTICS ASSERTIONS -- the figure must encode the paper's certificate-of-record story:
    # both aggregate-winners raise a watershed (fail Eq. 1), the per-basin-guarded member
    # certifies with every watershed non-positive and the largest aggregate cut of the three.
    assert not cg["per_basin_cert"] and max(cg["per_basin_delta_pct"].values()) > 0
    assert not agg["per_basin_cert"] and max(agg["per_basin_delta_pct"].values()) > 0
    assert pbg["per_basin_cert"] and max(pbg["per_basin_delta_pct"].values()) <= 0
    assert pbg["agg_delta_pct"] < min(cg["agg_delta_pct"], agg["agg_delta_pct"])
    methods = [  # (label, color, marker, per-basin dict, aggregate)
        ("carbon-greedy", sty.CARBON_BLUE, "o", cg["per_basin_delta_pct"], cg["agg_delta_pct"]),
        ("aggregate-guarded member", sty.AGG_ORANGE, "s", agg["per_basin_delta_pct"], agg["agg_delta_pct"]),
        ("per-basin-guarded member", sty.ENV_GREEN, "*", pbg["per_basin_delta_pct"], pbg["agg_delta_pct"]),
    ]
    # rows: basins sorted by the worst rise across the two aggregate-winners (descending),
    # then the fleet aggregate, separated
    basins = sorted(BASIN_NAME, key=lambda b: max(cg["per_basin_delta_pct"][b],
                                                  agg["per_basin_delta_pct"][b]), reverse=True)
    rows = basins + ["AGG"]
    ypos = {r: y for r, y in zip(rows, [3.35, 2.45, 1.55, 0.45])}

    sty.apply()
    fig, ax = plt.subplots(figsize=(sty.COLW, 1.95))
    xmin, xmax = -104, 16
    ax.axvspan(0, xmax, color=sty.HARM_BG, alpha=0.06, zorder=0)
    ax.axvspan(xmin, 0, color=sty.HOLD_BG, alpha=0.07, zorder=0)
    ax.axvline(0, color="k", lw=0.8, zorder=2)
    ax.axhline(1.0, color="gray", lw=0.7, ls=":", zorder=2)  # basins | aggregate separator
    dodge = {"o": +0.21, "s": 0.0, "*": -0.21}  # near-zero values overlap on a -100..+16 axis
    for r in rows:
        vals = [(agg_d if r == "AGG" else pb[r]) for _, _, _, pb, agg_d in methods]
        y = ypos[r]
        ax.plot([min(vals), max(vals)], [y, y], color="gray", lw=0.7, zorder=2, alpha=0.6)
        for (label, color, marker, pb, agg_d), v in zip(methods, vals):
            size = 120 if marker == "*" else 34
            ym = y + dodge[marker]
            ax.scatter([v], [ym], c=color, marker=marker, s=size,
                       edgecolors="k", linewidths=0.4, zorder=4)
            if v > 0.3:  # a rising watershed = the burden-shift; the fig:plane fail-ring
                ax.scatter([v], [ym], s=size * 2.3, facecolors="none",
                           edgecolors=sty.WATER_RED, linewidths=1.1, zorder=3)
                ax.annotate(f"+{v:.1f}", (v, ym), textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=6.8, color=sty.WATER_RED, weight="bold")
            elif v <= xmin + 8:  # Po -100 under the aggregate member: label right of the dot
                ax.annotate(f"{v:.0f}", (v, ym), textcoords="offset points", xytext=(7, -2.4),
                            ha="left", fontsize=6.8, color="k")
            elif abs(v) >= 2.0:
                off = {"o": ((0, 5.5), "center"), "s": ((-7, -2.6), "right"),
                       "*": ((0, -10), "center")}[marker]
                if r == "IT-NO" and marker == "*":  # keep clear of the aggregate row's labels
                    off = ((-7, -2.6), "right")
                ax.annotate(f"{v:.1f}", (v, ym), textcoords="offset points", xytext=off[0],
                            ha=off[1], fontsize=6.8, color="k")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.35, 4.38)
    ax.set_yticks([ypos[r] for r in rows])
    ax.set_yticklabels([BASIN_NAME.get(r, "fleet aggregate") for r in rows], fontsize=7.5)
    ax.set_xlabel("scarcity-water $\\Delta$ vs $B$ (%)   ($\\leq$ 0 = no harm)")
    ax.text(0.02, 0.985, "ringed: watershed rises $\\Rightarrow$ fails Eq. (1)",
            transform=ax.transAxes, fontsize=6.8, color=sty.WATER_RED, ha="left", va="top")
    handles = [mlines.Line2D([], [], color=c, marker=m, linestyle="None",
                             markersize=8 if m == "*" else 4.6, markeredgecolor="k",
                             markeredgewidth=0.4)
               for _, c, m, _, _ in methods]
    ax.legend(handles, [lbl for lbl, *_ in methods], loc="lower left", frameon=False,
              fontsize=6.6, handlelength=1.0, handletextpad=0.5, labelspacing=0.25,
              borderaxespad=0.1)
    ax.grid(alpha=0.18, axis="x", zorder=1)
    sty.save(fig, "perbasin_burden_shift")
    # sanity print against tab:realtrace_perbasin
    for label, _, _, pb, aggd in methods:
        print(f"{label:28} agg {aggd:+.2f}%  basins "
              f"{ {k: round(v, 2) for k, v in pb.items()} }")
    assert all(v <= 0.0 for v in pbg["per_basin_delta_pct"].values()), "pbg must be all-non-positive"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
