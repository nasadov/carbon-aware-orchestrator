#!/usr/bin/env python3
"""Evaluate the full published-baseline field + the no-harm envelope on both testbeds.

Runs every method (packing, carbon-greedy, water-greedy, the WaterWise weight-frontier,
CIC, Wait Awhile, GreenSlot, and the envelope) on:
  * the committed headroom fleet (pkg/carbon-aware/nodes.yaml + workloads/ + real-CI
    signals timealigned_realci), and
  * the realistic Alibaba GPU v2020 fleet (built via scripts/alibaba_common.py),

with ONE shared ex-post no-harm certificate (summarize_against_reference). Emits per
testbed a baseline table CSV, a certified Pareto-frontier CSV (carbon vs water, with the
cert flag), and a JSON digest, all under experiments/baselines_*/.

Usage:
    python scripts/run_published_baselines.py                 # both fleets, defaults
    python scripts/run_published_baselines.py --skip-alibaba  # headroom fleet only
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SERVER = REPO_ROOT / "pkg" / "carbon-aware" / "server-python"
for _p in (str(SERVER), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from carbon_aware.no_harm_flexibility import PilotConfig  # noqa: E402
from published_baselines import run_all_baselines  # noqa: E402

TIMEALIGNED_REALCI = REPO_ROOT / "pkg" / "carbon-aware" / "data" / "timealigned_realci"
CONFIG_FILE = REPO_ROOT / "pkg" / "carbon-aware" / "infra-workload-config.yaml"

# Columns reported per method in the baseline table.
TABLE_FIELDS = [
    "method_key", "no_harm_certificate", "carbon_nonincrease", "scarcity_nonincrease",
    "slo_non_decrease", "carbon_delta_pct", "scarcity_delta_pct",
    "weighted_stress_kwh_avoided_pct", "stress_kwh_avoided_pct",
    "placed_pods", "unplaced_pods", "repairs_applied",
]

WW_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _write_table(out_dir: Path, rows: Sequence[Dict]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "baseline_table.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TABLE_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in TABLE_FIELDS})
    return path


def _write_frontier(out_dir: Path, rows: Sequence[Dict]) -> Path:
    """Certified Pareto-frontier data: each method as a (carbon%, water%) point with its
    cert flag, plus a 'dominated' flag computed among certified methods only."""
    pts = [
        {
            "method_key": r["method_key"],
            "carbon_delta_pct": float(r["carbon_delta_pct"]),
            "scarcity_delta_pct": float(r["scarcity_delta_pct"]),
            "weighted_stress_kwh_avoided_pct": float(r["weighted_stress_kwh_avoided_pct"]),
            "certified": bool(r["no_harm_certificate"]),
            "placed_pods": int(r["placed_pods"]),
            "unplaced_pods": int(r["unplaced_pods"]),
        }
        for r in rows
    ]
    # Pareto dominance on (carbon%, water%) where LOWER (more negative) is better, among
    # certified points: p is dominated if some other certified q is <= on both and < on one.
    cert = [p for p in pts if p["certified"]]
    for p in pts:
        dom = False
        if p["certified"]:
            for q in cert:
                if q is p:
                    continue
                le = (q["carbon_delta_pct"] <= p["carbon_delta_pct"] + 1e-9 and
                      q["scarcity_delta_pct"] <= p["scarcity_delta_pct"] + 1e-9)
                lt = (q["carbon_delta_pct"] < p["carbon_delta_pct"] - 1e-9 or
                      q["scarcity_delta_pct"] < p["scarcity_delta_pct"] - 1e-9)
                if le and lt:
                    dom = True
                    break
        p["on_certified_frontier"] = bool(p["certified"] and not dom)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "certified_frontier.csv"
    with path.open("w", newline="") as fh:
        fields = ["method_key", "carbon_delta_pct", "scarcity_delta_pct",
                  "weighted_stress_kwh_avoided_pct", "certified", "on_certified_frontier",
                  "placed_pods", "unplaced_pods"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for p in pts:
            w.writerow({k: p[k] for k in fields})
    return path, pts


def _print_table(title: str, rows: Sequence[Dict]) -> None:
    print(f"\n=== {title} ===")
    hdr = f"{'method':>16} {'cert':>5} {'carbon%':>9} {'water%':>9} {'wstress%':>9} {'placed':>7} {'unpl':>5} {'repairs':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['method_key']:>16} {str(r['no_harm_certificate']):>5} "
              f"{float(r['carbon_delta_pct']):>+9.2f} {float(r['scarcity_delta_pct']):>+9.2f} "
              f"{float(r['weighted_stress_kwh_avoided_pct']):>+9.2f} "
              f"{int(r['placed_pods']):>7} {int(r['unplaced_pods']):>5} {int(r.get('repairs_applied') or 0):>8}")


def run_one(name: str, config: PilotConfig, out_dir: Path) -> Dict:
    res = run_all_baselines(config, waterwise_weights=WW_WEIGHTS)
    rows = res["summary_rows"]
    table_path = _write_table(out_dir, rows)
    frontier_path, pts = _write_frontier(out_dir, rows)
    _print_table(name, rows)

    cert_frontier = [p["method_key"] for p in pts if p["on_certified_frontier"]]
    # A "threat" = any competitor that (a) sits on the certified frontier alongside/instead of
    # the envelope, OR (b) beats the envelope on carbon and/or water on an axis -- REGARDLESS of
    # whether it certifies. We deliberately include UNcertified axis-beaters too, because a
    # reviewer will ask "why not just run CIC, it cuts more carbon?" -- and the honest answer
    # (it drops SLO / inflates the other axis -> uncertified) is the whole no-harm argument, so
    # it must be surfaced, not hidden behind the cert filter.
    threats = []
    env = next((p for p in pts if p["method_key"] == "no_harm_flex"), None)
    if env is not None:
        for p in pts:
            if p["method_key"] == "no_harm_flex":
                continue
            beats_carbon = p["carbon_delta_pct"] < env["carbon_delta_pct"] - 1e-9
            beats_water = p["scarcity_delta_pct"] < env["scarcity_delta_pct"] - 1e-9
            if beats_carbon or beats_water or p["on_certified_frontier"]:
                threats.append({
                    "method_key": p["method_key"], "certified": p["certified"],
                    "carbon_delta_pct": p["carbon_delta_pct"],
                    "scarcity_delta_pct": p["scarcity_delta_pct"],
                    "placed_pods": p["placed_pods"], "unplaced_pods": p["unplaced_pods"],
                    "beats_envelope_carbon": beats_carbon, "beats_envelope_water": beats_water,
                    "on_certified_frontier": p["on_certified_frontier"],
                })
    digest = {
        "testbed": name,
        "nodes_file": str(config.nodes_file),
        "workloads_dir": str(config.workloads_dir),
        "n_methods": len(rows),
        "certified_methods": [r["method_key"] for r in rows if r["no_harm_certificate"]],
        "certified_frontier_methods": cert_frontier,
        "envelope": {k: env.get(k) for k in (
            "carbon_delta_pct", "scarcity_delta_pct", "weighted_stress_kwh_avoided_pct",
            "certified", "on_certified_frontier")} if env else None,
        "threats_to_envelope": threats,
        "table_csv": str(table_path),
        "frontier_csv": str(frontier_path),
    }
    (out_dir / "digest.json").write_text(json.dumps(digest, indent=2))
    print(f"  certified frontier: {cert_frontier}")
    if threats:
        print(f"  THREATS to envelope: {[t['method_key'] for t in threats]}")
    print(f"  -> {out_dir}")
    return digest


def headroom_config(out_dir: Path) -> PilotConfig:
    return PilotConfig(
        repo_root=REPO_ROOT,
        nodes_file=REPO_ROOT / "pkg" / "carbon-aware" / "nodes.yaml",
        workloads_dir=REPO_ROOT / "pkg" / "carbon-aware" / "workloads",
        forecasts_file=TIMEALIGNED_REALCI / "forecasts.json",
        config_file=CONFIG_FILE,
        output_dir=out_dir,
        max_timeslots=24,
        max_pods=80,
        scenario="heatwave-drought",
        lever_mode="both",
        grid_signal_csv=TIMEALIGNED_REALCI / "grid_residual_region_slot.csv",
        wue_csv=TIMEALIGNED_REALCI / "wue_region_slot.csv",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="baselines_published")
    ap.add_argument("--skip-alibaba", action="store_true")
    ap.add_argument("--nodes-per-region", type=int, default=4)
    ap.add_argument("--alibaba-pods", type=int, default=200)
    ap.add_argument("--alibaba-seed", type=int, default=42)
    ap.add_argument("--alibaba-max-timeslots", type=int, default=48)
    args = ap.parse_args()
    logging.disable(logging.CRITICAL)

    root = REPO_ROOT / "experiments" / args.run_name
    digests = {}

    # --- Headroom fleet (committed small fleet, real-CI signals) ---------------
    hd_dir = root / "headroom_fleet"
    digests["headroom_fleet"] = run_one("HEADROOM FLEET (committed, real-CI)", headroom_config(hd_dir / "out"), hd_dir)

    # --- Alibaba GPU fleet -----------------------------------------------------
    if not args.skip_alibaba:
        import alibaba_common as ac  # noqa: E402
        ab_dir = root / "alibaba_gpu_fleet"
        nodes_file, n_nodes, n_gpus = ac.build_fleet(ab_dir / "fleet", args.nodes_per_region)
        win = ac.build_window(ab_dir / "inputs", target_pods=args.alibaba_pods, seed=args.alibaba_seed)
        ab_cfg = ac.make_pilot_config(
            nodes_file=nodes_file, workloads_dir=win["workloads_dir"], output_dir=ab_dir / "out",
            signals=ac.TIMEALIGNED, max_timeslots=args.alibaba_max_timeslots,
            scenario="heatwave-drought", lever_mode="both",
        )
        name = (f"ALIBABA GPU FLEET ({n_nodes} V100 DGX-1 nodes, {win['n_pods']} pods, "
                f"{win['flexible_pct']:.0f}% flexible)")
        digests["alibaba_gpu_fleet"] = run_one(name, ab_cfg, ab_dir)

    (root / "all_digests.json").write_text(json.dumps(digests, indent=2))
    print(f"\nALL DIGESTS -> {root / 'all_digests.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
