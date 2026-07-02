#!/usr/bin/env python3
"""Light radiation no-harm demo (Seminar #7): does the 4th axis (ionising radiation) catch harm the
first three (carbon / scarcity-water / SLO) miss, and does WaterWise leak it?

Runs the no-harm pilot on the coherent corrected signals (timealigned_realci2, Option A water +
in-window EWIF + continuous cooling) over the DE/FR/ES/IT-NO fleet, with the committed per-region
ionising-radiation CFs (radiation_region.csv; FR ~0.44 kBq U235eq/kWh, ~10x DE). Two passes per window:

  * MEASURE  (radiation_enabled=True, guard off): every method's radiation delta is reported. The
    baselines (carbon-greedy, WaterWise) and the 3-axis envelope can pass carbon+water+SLO yet LEAK
    radiation (delta_rad > 0) when they push work onto the clean-but-nuclear French grid.
  * GUARD    (radiation_guard=True): the envelope additionally enforces radiation <= baseline -> 4-axis.

Emits experiments/radiation_demo/radiation_demo.json and a seminar figure
docs/paper-three/Brainstorm/PhDSeminar#7/assets/fig_radiation_demo.png.

Run:
  PYTHONPATH=pkg/carbon-aware/server-python python scripts/run_radiation_demo.py [--light]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SERVER = REPO_ROOT / "pkg" / "carbon-aware" / "server-python"
for p in (str(SERVER), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_water_basin_certification as wb  # reuse the realci2 fleet/config builders  # noqa: E402
from carbon_aware.no_harm_flexibility import run_no_harm_flexibility_pilot  # noqa: E402

OUT = REPO_ROOT / "experiments" / "radiation_demo"
ASSETS = REPO_ROOT / "docs" / "paper-three" / "Brainstorm" / "PhDSeminar#7" / "assets"

# (label, method_key, source-pass) -- which run each displayed bar is read from.
BASELINES = [("carbon-greedy", "carbon"), ("WaterWise", "waterwise")]


def _summ(res) -> Dict[str, dict]:
    return {r["method_key"]: r for r in res["summary_rows"]}


def run_window(testbed: str, window: str, seed: int) -> dict:
    run_dir = OUT / window
    run_dir.mkdir(parents=True, exist_ok=True)
    if testbed == "alibaba":
        pc, n_nodes, n_pods = wb._build_gpu_config(window, run_dir)
    else:
        pc, n_nodes, n_pods = wb._build_azure_config(window, run_dir, seed=seed)

    # Pass 1: MEASURE radiation for every method (envelope is 3-axis, may leak).
    measure = _summ(run_no_harm_flexibility_pilot(replace(pc, radiation_enabled=True)))
    # Pass 2: GUARD radiation (envelope becomes 4-axis).
    guard = _summ(run_no_harm_flexibility_pilot(replace(pc, output_dir=run_dir / "out_guard",
                                                         radiation_guard=True)))

    def row(s):
        return {
            "carbon_delta_pct": s.get("carbon_delta_pct"),
            "scarcity_delta_pct": s.get("scarcity_delta_pct"),
            "radiation_delta_pct": s.get("radiation_delta_pct"),
            "no_harm_3axis": bool(s.get("no_harm_certificate")),
            "no_harm_4axis": bool(s.get("no_harm_certificate_with_radiation")),
            "radiation_nonincrease": bool(s.get("radiation_nonincrease", False)),
        }

    out = {"testbed": testbed, "window": window, "n_nodes": n_nodes, "n_pods": n_pods, "methods": {}}
    for label, key in BASELINES:
        if key in measure:
            out["methods"][label] = row(measure[key])
    out["methods"]["envelope (3-axis)"] = row(measure["no_harm_flex"])
    out["methods"]["envelope + radiation (4-axis)"] = row(guard["no_harm_flex"])
    return out


def make_figure(results: List[dict]) -> None:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ["carbon-greedy", "WaterWise", "envelope (3-axis)", "envelope + radiation (4-axis)"]
    axes = [("carbon", "carbon_delta_pct"), ("scarcity-water", "scarcity_delta_pct"),
            ("ionising radiation", "radiation_delta_pct")]

    # mean delta across windows per method/axis
    agg = {m: {a: [] for a, _ in axes} for m in methods}
    cert4 = {m: [] for m in methods}
    for r in results:
        for m in methods:
            mm = r["methods"].get(m)
            if not mm:
                continue
            for a, k in axes:
                v = mm.get(k)
                if v is not None:
                    agg[m][a].append(v)
            cert4[m].append(mm["no_harm_4axis"])
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")

    INK = "#2B2B45"; GREEN = "#2E8B3D"; RED = "#C0392B"; AXc = {"carbon": "#555588",
            "scarcity-water": "#3AA0C0", "ionising radiation": "#D98C00"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "text.color": INK,
                         "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 200, "savefig.dpi": 200, "savefig.facecolor": "white",
                         "figure.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    x = np.arange(len(methods)); w = 0.26
    for i, (a, _) in enumerate(axes):
        vals = [mean(agg[m][a]) for m in methods]
        ax.bar(x + (i - 1) * w, vals, w, label=a, color=AXc[a], edgecolor="white", zorder=3)
    ax.axhline(0, color=INK, lw=1.1)
    # annotate 4-axis certificate status under each group
    for j, m in enumerate(methods):
        ok = all(cert4[m]) if cert4[m] else False
        ax.annotate(("4-axis ✓" if ok else "4-axis ✗"), (x[j], 0),
                    textcoords="offset points", xytext=(0, -28), ha="center",
                    color=(GREEN if ok else RED), fontweight="bold", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("mean $\\Delta$ vs packing baseline (%)")
    ax.set_title("WaterWise meets carbon+water but leaks ionising radiation; the 4th guard closes it",
                 fontsize=12.5, color=INK)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    ax.margins(y=0.22)
    ASSETS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(ASSETS / f"fig_radiation_demo.{ext}", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {ASSETS/'fig_radiation_demo.png'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--light", action="store_true", help="3 windows (d0,d2 Azure + w1587 GPU) instead of all 6")
    args = ap.parse_args()
    logging.disable(logging.CRITICAL)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.light:
        plan = [("azure", "azure_packing_2020_d0_s42_200", 42),
                ("azure", "azure_packing_2020_d2_s43_200", 43),
                ("alibaba", "alibaba_gpu_2020_w1587_s43_200", 43)]
    else:
        plan = ([("alibaba", w, 42) for w in wb.ALIBABA_WINDOWS]
                + [("azure", w, s) for (w, s) in wb.AZURE_WINDOWS])

    results: List[dict] = []
    for testbed, window, seed in plan:
        print(f"\n##### radiation demo: {testbed} / {window} #####", flush=True)
        r = run_window(testbed, window, seed)
        results.append(r)
        for label, mm in r["methods"].items():
            print(f"  {label:30s} C={mm['carbon_delta_pct']:+7.2f}%  W={mm['scarcity_delta_pct']:+7.2f}%  "
                  f"RAD={mm['radiation_delta_pct']:+7.2f}%  3ax={mm['no_harm_3axis']!s:5} 4ax={mm['no_harm_4axis']!s:5}")

    (OUT / "radiation_demo.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT/'radiation_demo.json'}")
    make_figure(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
