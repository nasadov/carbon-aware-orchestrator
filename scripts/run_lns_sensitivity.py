#!/usr/bin/env python3
"""F3 sensitivity of the guarded RELIEF move-set / LNS (B2): where does the atomic move-set add
certified relief the single-move greedy cannot reach?

Two axes on real Alibaba GPU (PAI) w1587 instances, measured-CI signals (timealigned_realci2):
  (1) DEADLINE SLACK  -- fixed fleet (npr=4), fixed_slack in {1,3,6,12,24}. slack=1 is the near-floor
      (deadline = duration+1h; below the 2h flex threshold most pods are firm -> relief -> 0).
  (2) FLEET SIZE      -- trace slack, npr in {2,4,8} at oversubscription-matched pod counts.

Each cell runs the production envelope (score_mode="combined") with relief_move_set OFF and ON, on the
SAME packing baseline B, rematerializing both under idle-once accounting before scoring. Reports the
no-harm certificate, carbon/water deltas, and the headline weighted grid-stress relief for OFF vs ON,
plus the LNS relief gain. The certificate must hold in BOTH columns (the move-set is a guarded pass).

Run: PYTHONPATH=pkg/carbon-aware/server-python python scripts/run_lns_sensitivity.py
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[0]
for p in (str(REPO / "pkg" / "carbon-aware" / "server-python"), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import logging
logging.disable(logging.CRITICAL)

import alibaba_common as ac  # noqa: E402
from carbon_aware.no_harm_flexibility import (  # noqa: E402
    load_flavours_for_pilot, apply_pilot_scenario_to_flavours, build_action_signals, load_pods,
    build_greedy_schedule, repair_schedule_no_harm, summarize_against_reference,
    _rematerialize_under_realized,
)

SIGNALS = REPO / "pkg" / "carbon-aware" / "data" / "timealigned_realci2"
WINDOW_START_H = 1587.0
SEED = 43


def _run_cell(run_dir: Path, npr: int, fixed_slack: int) -> dict:
    target_pods = ac.pods_for_fleet(npr)
    stats = ac.build_window(run_dir, target_pods, SEED, window_start_hours=WINDOW_START_H,
                            fixed_slack=fixed_slack)
    nodes_file, n_nodes, _ = ac.build_fleet(run_dir / "fleet", npr)
    cfg = ac.make_pilot_config(nodes_file, stats["workloads_dir"], run_dir / "out", signals=SIGNALS)
    flavours = load_flavours_for_pilot(cfg)
    apply_pilot_scenario_to_flavours(flavours, cfg)
    signals = build_action_signals(flavours, cfg)
    pods = load_pods(stats["workloads_dir"])
    packing, _, _ = build_greedy_schedule(method_key="packing", pods=pods, flavours=flavours, config=cfg)

    def envelope(relief_move_set: bool):
        rep = repair_schedule_no_harm(method_key="no_harm_flex", baseline=packing, pods=pods,
                                      flavours=flavours, signals=signals,
                                      config=replace(cfg, relief_move_set=relief_move_set),
                                      score_mode="combined")
        _rematerialize_under_realized(rep, flavours, cfg)
        return summarize_against_reference(rep, packing, signals)

    off = envelope(False)
    on = envelope(True)
    return {
        "npr": npr, "n_nodes": n_nodes, "fixed_slack": fixed_slack, "pods": len(pods),
        "flexible_pct": round(stats["flexible_pct"], 1),
        "off_cert": off["no_harm_certificate"], "off_relief_kwh": off["weighted_stress_kwh_avoided"],
        "off_carbon_pct": off["carbon_delta_pct"], "off_water_pct": off["scarcity_delta_pct"],
        "off_repairs": off["repairs_applied"],
        "on_cert": on["no_harm_certificate"], "on_relief_kwh": on["weighted_stress_kwh_avoided"],
        "on_carbon_pct": on["carbon_delta_pct"], "on_water_pct": on["scarcity_delta_pct"],
        "on_repairs": on["repairs_applied"],
        "lns_relief_gain_kwh": on["weighted_stress_kwh_avoided"] - off["weighted_stress_kwh_avoided"],
    }


def main() -> int:
    out_root = REPO / "experiments" / "lns_sensitivity"
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []

    print("=== (1) LNS relief gain vs DEADLINE SLACK (fleet npr=4) ===")
    print(f"{'slack':>5} {'pods':>5} {'flex%':>5} | {'off_relief':>10} {'on_relief':>10} {'gain':>8} | "
          f"{'off_cert':>8} {'on_cert':>7} {'C%on':>7} {'W%on':>7}")
    for slack in (1, 3, 6, 12, 24):
        r = _run_cell(out_root / f"slack{slack}", npr=4, fixed_slack=slack)
        rows.append(r)
        print(f"{slack:>5} {r['pods']:>5} {r['flexible_pct']:>5.0f} | {r['off_relief_kwh']:>10.3f} "
              f"{r['on_relief_kwh']:>10.3f} {r['lns_relief_gain_kwh']:>8.3f} | {str(r['off_cert']):>8} "
              f"{str(r['on_cert']):>7} {r['on_carbon_pct']:>7.2f} {r['on_water_pct']:>7.2f}", flush=True)

    print("\n=== (2) LNS relief gain vs FLEET SIZE (trace slack) ===")
    print(f"{'npr':>5} {'nodes':>5} {'pods':>5} | {'off_relief':>10} {'on_relief':>10} {'gain':>8} | "
          f"{'off_cert':>8} {'on_cert':>7} {'C%on':>7} {'W%on':>7}")
    for npr in (2, 4, 8):
        r = _run_cell(out_root / f"npr{npr}", npr=npr, fixed_slack=0)
        rows.append(r)
        print(f"{npr:>5} {r['n_nodes']:>5} {r['pods']:>5} | {r['off_relief_kwh']:>10.3f} "
              f"{r['on_relief_kwh']:>10.3f} {r['lns_relief_gain_kwh']:>8.3f} | {str(r['off_cert']):>8} "
              f"{str(r['on_cert']):>7} {r['on_carbon_pct']:>7.2f} {r['on_water_pct']:>7.2f}", flush=True)

    with (out_root / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out_root / "summary.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out_root}/summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
