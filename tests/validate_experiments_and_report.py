#!/usr/bin/env python3
"""
Batch placement validation across experiment folders.

Validates capacity and timeslot constraints for each experiment's placement CSV
and outputs a timestamped CSV and TXT summary in tests/reports/.
"""

import argparse
import csv
import datetime as dt
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from placement_constraint_validator import (  # type: ignore
    load_node_capacities,
    build_pod_timeslot_map,
    validate_capacity_constraints,
    validate_timeslot_constraints,
)


def discover_experiments(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isdir(p) and not name.lower().startswith("archive"):
            out.append(p)
    return out


def detect_algorithm(exp_dir: str) -> Optional[str]:
    b = os.path.basename(exp_dir)
    if b.startswith("vanilla_"):
        return "vanilla"
    if b.startswith("heuristic_"):
        return "heuristic"
    if b.startswith("global-optimal_"):
        return "global-optimal"
    return None


def detect_pods(exp_dir: str) -> Optional[int]:
    b = os.path.basename(exp_dir)
    m = re.search(r"_(\d+)pods_", b)
    return int(m.group(1)) if m else None


def find_csv(exp_dir: str, algo: str) -> Optional[str]:
    names = {
        "vanilla": ["vanilla_placement_session.csv"],
        "heuristic": ["heuristic_placements_session.csv", "heuristic_op_placements_session.csv"],
        "global-optimal": ["global_optimal_placements_session.csv"],
    }.get(algo, [])
    for fname in names:
        p = os.path.join(exp_dir, fname)
        if os.path.isfile(p):
            return p
    return None


def validate_one(csv_path: str, nodes_file: str, workloads_dir: str) -> Tuple[Dict[str, int], bool, bool]:
    df = pd.read_csv(csv_path)
    for col in ("start_slot", "duration", "cpu_request", "ram_request"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Capacity
    cap_counts = {"cpu_violations": 0, "memory_violations": 0, "unknown_nodes": 0}
    cap_ok = True
    if os.path.isfile(nodes_file):
        caps = load_node_capacities(nodes_file)
        ok, violations = validate_capacity_constraints(df, caps)
        cap_ok = ok
        for v in violations:
            t = v.get("type")
            if t == "cpu_violation":
                cap_counts["cpu_violations"] += 1
            elif t == "memory_violation":
                cap_counts["memory_violations"] += 1
            elif t == "unknown_node":
                cap_counts["unknown_nodes"] += 1

    # Timeslot
    ts_ok = True
    ts_count = 0
    if os.path.isdir(workloads_dir):
        pod_ts = build_pod_timeslot_map(workloads_dir)
        ok, violations = validate_timeslot_constraints(df, pod_ts)
        ts_ok = ok
        ts_count = len(violations)

    cap_counts["timeslot_violations"] = ts_count
    return cap_counts, cap_ok, ts_ok


def main():
    ap = argparse.ArgumentParser(description="Validate all experiments and write timestamped report")
    ap.add_argument("--experiments-dir", default="/root/carbon-aware-orchestrator/pkg/carbon-aware/server-python/experiments")
    ap.add_argument("--nodes-file", default="/root/carbon-aware-orchestrator/pkg/carbon-aware/nodes.yaml")
    ap.add_argument("--workloads-dir", default="/root/carbon-aware-orchestrator/pkg/carbon-aware/workloads")
    ap.add_argument("--workloads-vanilla-dir", default="/root/carbon-aware-orchestrator/pkg/carbon-aware/workloads-vanilla")
    ap.add_argument("--output-dir", default="/root/carbon-aware-orchestrator/tests/reports")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(args.output_dir, f"placement_validation_report_{ts}.csv")
    out_txt = os.path.join(args.output_dir, f"placement_validation_report_{ts}.txt")

    rows: List[Dict[str, object]] = []
    totals = {k: 0 for k in [
        "experiments_scanned", "experiments_with_csv", "capacity_ok", "timeslot_ok",
        "cpu_violations", "memory_violations", "unknown_nodes", "timeslot_violations"
    ]}

    for exp in discover_experiments(args.experiments_dir):
        algo = detect_algorithm(exp)
        if not algo:
            continue
        totals["experiments_scanned"] += 1
        pods = detect_pods(exp)
        csv_path = find_csv(exp, algo)
        if not csv_path:
            rows.append({
                "experiment": os.path.basename(exp),
                "algorithm": algo,
                "pods": pods,
                "csv_found": False,
            })
            continue
        totals["experiments_with_csv"] += 1
        workloads = args.workloads_dir if algo != "vanilla" else args.workloads_vanilla_dir
        try:
            counts, cap_ok, ts_ok = validate_one(csv_path, args.nodes_file, workloads)
        except Exception as e:
            rows.append({
                "experiment": os.path.basename(exp),
                "algorithm": algo,
                "pods": pods,
                "csv_found": True,
                "capacity_ok": False,
                "timeslot_ok": False,
                "cpu_violations": -1,
                "memory_violations": -1,
                "unknown_nodes": -1,
                "timeslot_violations": -1,
                "error": str(e),
            })
            continue

        if cap_ok:
            totals["capacity_ok"] += 1
        if ts_ok:
            totals["timeslot_ok"] += 1
        totals["cpu_violations"] += counts.get("cpu_violations", 0)
        totals["memory_violations"] += counts.get("memory_violations", 0)
        totals["unknown_nodes"] += counts.get("unknown_nodes", 0)
        totals["timeslot_violations"] += counts.get("timeslot_violations", 0)

        rows.append({
            "experiment": os.path.basename(exp),
            "algorithm": algo,
            "pods": pods,
            "csv_found": True,
            "capacity_ok": cap_ok,
            "timeslot_ok": ts_ok,
            **counts,
        })

    # Write CSV
    headers = [
        "experiment", "algorithm", "pods", "csv_found",
        "capacity_ok", "timeslot_ok", "cpu_violations", "memory_violations",
        "unknown_nodes", "timeslot_violations"
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in headers})

    # Write TXT summary
    with open(out_txt, "w") as f:
        f.write("PLACEMENT VALIDATION REPORT\n")
        f.write(f"Generated: {ts}\n")
        f.write(f"Experiments: {args.experiments_dir}\n")
        f.write(f"Nodes: {args.nodes_file}\n")
        f.write(f"Workloads(custom): {args.workloads_dir}\n")
        f.write(f"Workloads(vanilla): {args.workloads_vanilla_dir}\n\n")
        for k, v in totals.items():
            f.write(f"{k}: {v}\n")

    print(f"✅ Report written:\n  CSV: {out_csv}\n  TXT: {out_txt}")


if __name__ == "__main__":
    main()


