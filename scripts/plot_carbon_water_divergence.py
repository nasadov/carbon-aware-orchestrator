#!/usr/bin/env python3
"""fig:divergence — the cleanest grid is not the least-scarce site (corrected, Option-A axis).

Mean measured carbon intensity (x) vs mean OPERATIONAL scarcity-water intensity (y) per region,
computed exactly as the certificate charges it (Option A): direct WUE x native-watershed AWARE CF
+ EWIF x generation-country AWARE CF, per facility-kWh, averaged over the window's slots — all from
the committed realci2 signals. Regenerable; replaces the ad-hoc pre-revamp figure.

Run: python scripts/plot_carbon_water_divergence.py
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
TA = REPO / "pkg" / "carbon-aware" / "data" / "timealigned_realci2"
WATER = REPO / "pkg" / "carbon-aware" / "data" / "water"
OUTS = [REPO / "experiments" / "figures", REPO / "docs" / "paper-three" / "Paper" / "figures"]
REGIONS = ["DE", "FR", "ES", "IT-NO"]
NAME = {"DE": "Germany", "FR": "France", "ES": "Spain", "IT-NO": "N. Italy"}
MONTH = "jul_cf"


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

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for r in REGIONS:
        ax.scatter(ci[r], sw[r], s=1500, color="#4A4A8C", alpha=0.88, edgecolor="white",
                   lw=2, zorder=3)
        ax.annotate(NAME[r], (ci[r], sw[r]), color="white", fontsize=10, fontweight="bold",
                    ha="center", va="center", zorder=4)
    # carbon-greedy arrow: toward the cleanest grid (France) -> scarcity-water RISES
    ax.annotate("", xy=(ci["FR"] + 28, sw["FR"] - 4), xytext=(ci["DE"] - 40, sw["DE"] + 6),
                arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=2.4), zorder=2)
    ax.text(0.42 * (ci["DE"] + ci["FR"]), 0.55 * (sw["FR"] + sw["DE"]),
            "carbon-greedy shift toward\nthe cleanest grid raises\nscarcity-weighted water",
            color="#C0392B", fontsize=9.5, ha="center")
    ax.set_xlabel("mean measured carbon intensity (gCO$_2$/kWh)  $\\rightarrow$ dirtier")
    ax.set_ylabel("operational scarcity-water intensity\n(L$_{\\mathrm{eq}}$/kWh, Option-A charging)  $\\rightarrow$ thirstier")
    ax.grid(alpha=0.2)
    ax.set_xlim(0, 600)
    ax.set_ylim(-8, 122)
    fig.tight_layout()
    for out in OUTS:
        out.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"carbon_water_divergence.{ext}", dpi=170, bbox_inches="tight")
    print("regions:", {r: (round(ci[r]), round(sw[r], 1)) for r in REGIONS})
    print(f"wrote carbon_water_divergence to {len(OUTS)} dirs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
