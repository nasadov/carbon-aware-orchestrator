#!/usr/bin/env python3
"""fig:divergence — the cleanest grid is not the least-scarce site (corrected, Option-A axis).

Mean measured carbon intensity (x) vs mean OPERATIONAL scarcity-water intensity (y) per region,
computed exactly as the certificate charges it (Option A): direct WUE x native-watershed AWARE CF
+ EWIF x generation-country AWARE CF, per facility-kWh, averaged over the window's slots — all from
the committed realci2 signals. Authored at print size (columnwidth); see paper_fig_style.

Run: python scripts/plot_carbon_water_divergence.py
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

import paper_fig_style as sty
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
TA = REPO / "pkg" / "carbon-aware" / "data" / "timealigned_realci2"
WATER = REPO / "pkg" / "carbon-aware" / "data" / "water"
REGIONS = ["DE", "FR", "ES", "IT-NO"]
NAME = {"DE": "Germany", "FR": "France", "ES": "Spain", "IT-NO": "N. Italy"}
MONTH = "jul_cf"
# label offsets in points so no text clips at the canvas edge (Germany sits far right)
LABEL_OFF = {"DE": (-4, 9), "FR": (0, -14), "ES": (2, -14), "IT-NO": (4, 10)}


def cf(path, key):
    out = {}
    for r in csv.DictReader(open(path)):
        k = (r.get(key) or "").strip().upper()
        try:
            out[k] = float(r[MONTH])
        except (TypeError, ValueError):
            continue
    return out


def main() -> int:
    basin = cf(WATER / "aware20_basin_nonagri_factors.csv", "region")
    country = cf(WATER / "aware20_country_nonagri_factors.csv", "country_code")
    c4r = lambda r: "IT" if r == "IT-NO" else r.split("-")[0]
    fc = json.load(open(TA / "forecasts.json"))
    ci = {r: st.mean(e["carbonIntensity"] for e in fc[r]["forecast"]) for r in REGIONS}
    wue, ewif = {r: [] for r in REGIONS}, {r: [] for r in REGIONS}
    for row in csv.DictReader(open(TA / "wue_region_slot.csv")):
        if row["region"] in wue:
            wue[row["region"]].append(float(row["direct_wue_l_per_kwh"]))
    for row in csv.DictReader(open(TA / "ewif_region_slot.csv")):
        if row["region"] in ewif:
            ewif[row["region"]].append(float(row["ewif_l_per_kwh"]))
    sw = {r: st.mean(wue[r]) * basin[r] + st.mean(ewif[r]) * country[c4r(r)] for r in REGIONS}

    sty.apply()
    fig, ax = plt.subplots(figsize=(sty.COLW, 2.30))
    for r in REGIONS:
        ax.scatter(ci[r], sw[r], s=110, color="#4A4A8C", alpha=0.9, edgecolor="white",
                   lw=1.0, zorder=3)
        dx, dy = LABEL_OFF[r]
        ax.annotate(NAME[r], (ci[r], sw[r]), textcoords="offset points", xytext=(dx, dy),
                    color="#2c2c55", fontsize=7.5, fontweight="bold", ha="center", zorder=4)
    # carbon-greedy arrow: toward the cleanest grid (France) -> scarcity-water RISES
    ax.annotate("", xy=(ci["FR"] + 22, sw["FR"] + 2), xytext=(ci["DE"] - 30, sw["DE"] + 4),
                arrowprops=dict(arrowstyle="-|>", color=sty.WATER_RED, lw=1.6), zorder=2)
    ax.text(0.47 * (ci["DE"] + ci["FR"]), 0.42 * (sw["FR"] + sw["DE"]) + 12,
            "carbon-greedy shift toward the\ncleanest grid raises scarcity-water",
            color=sty.WATER_RED, fontsize=7.0, ha="center", va="bottom")
    # the measured consequence, on the figure (figstory fix: gradient -> backfire);
    # values = \PNcfCulpritCarbon / \PNcfCulpritWater (paper_numbers.tex, Table 1 window);
    # placed in the empty upper-left zone, clear of the arrow and country labels
    ax.text(12, 84, "measured on the July-2018 window:\n$-24.0\\%$ C, $\\mathbf{+63.8\\%}$ W (Table 1)",
            color=sty.WATER_RED, fontsize=6.4, ha="left", va="top")
    ax.set_xlabel("measured carbon intensity (gCO$_2$/kWh)  $\\rightarrow$ dirtier")
    ax.set_ylabel("scarcity-water (L$_{\\mathrm{eq}}$/kWh)\n$\\rightarrow$ thirstier")
    ax.grid(alpha=0.2)
    ax.set_xlim(0, 600)
    ax.set_ylim(-8, 122)
    for name in ("carbon_water_divergence",):
        sty.save(fig, name)
    print("regions:", {r: (round(ci[r]), round(sw[r], 1)) for r in REGIONS})
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
