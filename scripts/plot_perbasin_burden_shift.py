#!/usr/bin/env python3
"""fig:burden — "the sum hides the harm": per-watershed scarcity-water deltas on the Azure
headline window (d0), for carbon-greedy, the aggregate-guarded member, and the per-basin-guarded
member. The aggregate wins (-6.0 / -8.8%) hide a rising watershed (FR +7.5 / ES +9.4); only the
per-basin guard turns every bar non-positive while landing the LARGEST aggregate cut (-10.5%).

Reads experiments/water_basin/basin_certification.json (the committed source of truth behind
tab:realtrace_perbasin). The DE basin bears no charged scarcity in this window (base 0) and is
omitted. Authored at print size (columnwidth); see paper_fig_style.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import paper_fig_style as sty  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

WINDOW = "azure_packing_2020_d0_s42_200"
BASINS = [("FR", "Seine\n(FR)"), ("ES", "Tagus\n(ES)"), ("IT-NO", "Po\n(IT-NO)")]
YCLIP = -24.0  # IT-NO hits -100% under the aggregate member; clip and label


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
    methods = [
        ("carbon-greedy", sty.CARBON_BLUE, cg["per_basin_delta_pct"], cg["agg_delta_pct"]),
        ("aggregate-guarded member", sty.AGG_LILAC, agg["per_basin_delta_pct"], agg["agg_delta_pct"]),
        ("per-basin-guarded member", sty.ENV_GREEN, pbg["per_basin_delta_pct"], pbg["agg_delta_pct"]),
    ]

    sty.apply()
    fig, ax = plt.subplots(figsize=(sty.COLW, 2.30))
    ymax = 14.0
    ax.axhspan(0, ymax, color=sty.HARM_BG, alpha=0.06, zorder=0)
    ax.axhspan(YCLIP, 0, color=sty.HOLD_BG, alpha=0.07, zorder=0)
    groups = [b for b, _ in BASINS] + ["AGG"]
    w = 0.26
    for mi, (label, color, per_basin, agg_delta) in enumerate(methods):
        xs, ys, real = [], [], []
        for gi, g in enumerate(groups):
            v = agg_delta if g == "AGG" else per_basin[g]
            xs.append(gi + (mi - 1) * w)
            ys.append(max(v, YCLIP + 0.8))
            real.append(v)
        ax.bar(xs, ys, w, color=color, label=label, zorder=3,
               edgecolor="k", linewidth=0.4)
        for x, y, v in zip(xs, ys, real):
            if v > 0.3:  # a rising watershed = the burden-shift; call it out in red
                ax.annotate(f"+{v:.1f}", (x, y), textcoords="offset points", xytext=(0, 2),
                            ha="center", fontsize=6.8, color=sty.WATER_RED, weight="bold")
            elif v < YCLIP:  # clipped bar (IT-NO -100% under the aggregate member)
                ax.annotate(f"{v:.0f}", (x, y), textcoords="offset points", xytext=(0, -9),
                            ha="center", fontsize=6.8, color="k")
            elif abs(v) >= 2.0:
                ax.annotate(f"{v:.1f}", (x, y), textcoords="offset points", xytext=(0, -9),
                            ha="center", fontsize=6.8, color="k")
    ax.axhline(0, color="k", lw=0.8, zorder=2)
    ax.axvline(len(BASINS) - 0.5, color="gray", lw=0.7, ls=":", zorder=2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([n for _, n in BASINS] + ["fleet\naggregate"], fontsize=7.5)
    ax.set_ylim(YCLIP, ymax)
    ax.set_ylabel("scarcity-water $\\Delta$ vs $B$ (%)")
    ax.text(0.99, 0.965, "watershed rises $\\Rightarrow$ fails Eq. (1)", transform=ax.transAxes,
            fontsize=6.8, color=sty.WATER_RED, ha="right", va="top")
    ax.legend(loc="lower left", frameon=False, fontsize=6.8, handlelength=1.2,
              bbox_to_anchor=(0.0, -0.02))
    ax.grid(alpha=0.18, axis="y", zorder=1)
    sty.save(fig, "perbasin_burden_shift")
    # sanity print against tab:realtrace_perbasin
    for label, _, pb, aggd in methods:
        print(f"{label:28} agg {aggd:+.2f}%  basins "
              f"{ {k: round(v, 2) for k, v in pb.items()} }")
    assert all(v <= 0.0 for v in pbg["per_basin_delta_pct"].values()), "pbg must be all-non-positive"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
