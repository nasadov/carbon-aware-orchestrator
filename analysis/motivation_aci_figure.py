#!/usr/bin/env python3
"""Motivation figure: hourly average carbon intensity (ACI) for the four evaluated
regions over the 24-hour scheduling horizon, from the shared Electricity Maps
forecast snapshot. Shows the cross-region spread and within-day variation that
spatio-temporal carbon-aware placement exploits."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
FORECASTS = (
    REPO / "experiments/resubmission_baseline_matrix_sweep_v5/matrix_20260515_111056"
    / "seed_42_pods_200/all_forecasts.json"
)
OUT_DIRS = [
    REPO / "docs/paper-two/figures/CompositeEvaluation/core_current",
    REPO / "docs/paper-two/figures/CompositeEvaluation/current",
    REPO / "figures/CompositeEvaluation/core_current",
    REPO / "figures/CompositeEvaluation/current",
]

STYLE = {
    "FR": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
    "ES": {"color": "#009E73", "linestyle": (0, (5, 2)), "marker": "s"},
    "IT-NO": {"color": "#E69F00", "linestyle": (0, (4, 1.5, 1, 1.5)), "marker": "^"},
    "DE": {"color": "#D55E00", "linestyle": "-", "marker": "v"},
}
HOURS = 24


def main() -> int:
    plt.rcParams.update({
        "font.size": 7.6, "axes.labelsize": 7.8, "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0, "legend.fontsize": 6.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.linestyle": "--", "grid.linewidth": 0.45,
        "grid.alpha": 0.38, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    data = json.load(open(FORECASTS))
    fig, ax = plt.subplots(figsize=(3.5, 1.58))
    for zone in ("DE", "IT-NO", "ES", "FR"):
        ci = [p["carbonIntensity"] for p in data[zone]["forecast"][:HOURS]]
        ax.plot(range(len(ci)), ci, label=zone, linewidth=1.3, markersize=2.8,
                markevery=3, **STYLE[zone])
    ax.set_xlabel("Hour of scheduling horizon")
    ax.set_ylabel("ACI (gCO$_2$e/kWh)")
    ax.set_xlim(0, HOURS - 1)
    ax.set_ylim(0, None)
    ax.set_xticks(range(0, HOURS, 6))
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16),
              columnspacing=1.0, handlelength=1.8)
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.21, top=0.88)
    for d in OUT_DIRS:
        if d.exists():
            for ext in ("pdf", "png"):
                fig.savefig(d / f"motivation_aci.{ext}", bbox_inches="tight", facecolor="white")
            print(f"wrote {d}/motivation_aci.[pdf,png]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
