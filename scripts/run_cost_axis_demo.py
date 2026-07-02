#!/usr/bin/env python3
"""Electricity-cost fifth-axis demo (P3.1): does grid-supportive flexibility raise the bill, and
does one more inequality guard it?

Runs the no-harm pilot on the six real-trace windows (realci2 signals + in-window day-ahead prices
from price_region_slot.csv) in two passes per window, mirroring run_radiation_demo.py:

  * MEASURE (cost_enabled=True): every method's operational-cost delta vs the packing baseline is
    reported; a method can pass the 3-axis certificate yet RAISE the operator's bill.
  * GUARD  (cost_guard=True): the envelope additionally enforces cost <= baseline -> the
    "flex for the grid, provably without raising cost, carbon, water, or SLO violations" claim.

Emits experiments/cost_axis/cost_axis_demo.json.

Run:
  PYTHONPATH=pkg/carbon-aware/server-python python scripts/run_cost_axis_demo.py [--light]
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

OUT = REPO_ROOT / "experiments" / "cost_axis"
PRICE_CSV = wb.REALCI / "price_region_slot.csv"
METHODS = [("carbon-greedy", "carbon"), ("WaterWise", "waterwise"),
           ("envelope (3-axis)", "no_harm_flex")]


def _summ(res) -> Dict[str, dict]:
    return {r["method_key"]: r for r in res["summary_rows"]}


def run_window(testbed: str, window: str, seed: int) -> dict:
    run_dir = OUT / window
    run_dir.mkdir(parents=True, exist_ok=True)
    if testbed == "alibaba":
        pc, n_nodes, n_pods = wb._build_gpu_config(window, run_dir)
    else:
        pc, n_nodes, n_pods = wb._build_azure_config(window, run_dir, seed=seed)

    measure = _summ(run_no_harm_flexibility_pilot(replace(pc, cost_enabled=True, price_csv=PRICE_CSV)))
    guard = _summ(run_no_harm_flexibility_pilot(replace(pc, output_dir=run_dir / "out_guard",
                                                         cost_guard=True, price_csv=PRICE_CSV)))

    def row(s):
        return {
            "carbon_delta_pct": s.get("carbon_delta_pct"),
            "scarcity_delta_pct": s.get("scarcity_delta_pct"),
            "cost_delta_pct": s.get("cost_delta_pct"),
            "cost_delta_eur": s.get("cost_delta_eur"),
            "no_harm_3axis": bool(s.get("no_harm_certificate")),
            "no_harm_with_cost": bool(s.get("no_harm_certificate_with_cost")),
        }

    out = {"testbed": testbed, "window": window, "n_nodes": n_nodes, "n_pods": n_pods, "methods": {}}
    for label, key in METHODS:
        if key in measure:
            out["methods"][label] = row(measure[key])
    out["methods"]["envelope + cost (4th ineq.)"] = row(guard["no_harm_flex"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--light", action="store_true")
    args = ap.parse_args()
    logging.disable(logging.CRITICAL)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.light:
        plan = [("azure", "azure_packing_2020_d0_s42_200", 42),
                ("alibaba", "alibaba_gpu_2020_w1587_s43_200", 43)]
    else:
        plan = ([("alibaba", w, 42) for w in wb.ALIBABA_WINDOWS]
                + [("azure", w, s) for (w, s) in wb.AZURE_WINDOWS])
    results: List[dict] = []
    for testbed, window, seed in plan:
        print(f"\n##### cost axis: {testbed} / {window} #####", flush=True)
        r = run_window(testbed, window, seed)
        results.append(r)
        for label, mm in r["methods"].items():
            print(f"  {label:28s} C={mm['carbon_delta_pct']:+7.2f}%  W={mm['scarcity_delta_pct']:+7.2f}%  "
                  f"COST={mm['cost_delta_pct']:+7.2f}%  3ax={mm['no_harm_3axis']!s:5} +cost={mm['no_harm_with_cost']!s:5}")
    (OUT / "cost_axis_demo.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT/'cost_axis_demo.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
