#!/usr/bin/env python3
"""
Summarize placement results across scheduler experiment directories.

The script evaluates all placements with the same post-hoc accounting model:
node idle power is charged once per active node-slot, dynamic power is charged by
aggregate utilization, and embodied emissions are amortized once per active
node-slot. This makes carbon-aware baselines comparable even when their decision
objectives differ.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import yaml


def parse_cpu(cpu_value) -> float:
    text = str(cpu_value).strip()
    if text.endswith("m"):
        return float(text[:-1]) / 1000.0
    return float(text)


def parse_memory_mb(memory_value) -> float:
    text = str(memory_value).strip()
    match = re.match(r"^(\d+(?:\.\d+)?)([KMG]i?|)$", text)
    if not match:
        return float(text)
    value = float(match.group(1))
    unit = match.group(2)
    if unit.startswith("K"):
        return value / 1024.0
    if unit.startswith("G"):
        return value * 1024.0
    return value


def load_nodes(nodes_file: Path) -> Dict[str, dict]:
    nodes: Dict[str, dict] = {}
    with nodes_file.open("r") as handle:
        for doc in yaml.safe_load_all(handle):
            if not isinstance(doc, dict) or doc.get("kind") != "Node":
                continue
            metadata = doc.get("metadata", {})
            annotations = metadata.get("annotations", {})
            labels = metadata.get("labels", {})
            allocatable = doc.get("status", {}).get("allocatable", {})
            node_id = metadata.get("name")
            if not node_id:
                continue
            region = labels.get("topology.kubernetes.io/region", "")
            if not region:
                parts = node_id.split("-")
                region = parts[2].upper() if len(parts) >= 3 else "DE"
            nodes[node_id] = {
                "region": region.upper(),
                "cpu": parse_cpu(allocatable.get("cpu", "0")),
                "memory_mb": parse_memory_mb(allocatable.get("memory", "0")),
                "embodied_g": float(annotations.get("hardware.carbon/embodied_emissions", "0")) * 1000.0,
                "lifetime_h": float(annotations.get("hardware.carbon/lifetime_years", "3")) * 365.0 * 24.0,
                "idle_w": float(annotations.get("hardware.power/idle_watts", "100")),
                "max_w": float(annotations.get("hardware.power/max_watts", "400")),
            }
    return nodes


def load_forecasts(forecasts_file: Path) -> Dict[str, Dict[int, float]]:
    import json

    with forecasts_file.open("r") as handle:
        raw = json.load(handle)
    forecasts: Dict[str, Dict[int, float]] = {}
    for region, payload in raw.items():
        forecasts[region.upper()] = {
            idx: float(entry["carbonIntensity"])
            for idx, entry in enumerate(payload.get("forecast", []))
        }
    return forecasts


def count_workload_pods(workloads_dir: Path) -> int:
    total = 0
    for path in sorted(workloads_dir.glob("timeslot_*.yaml")):
        with path.open("r") as handle:
            for doc in yaml.safe_load_all(handle):
                if isinstance(doc, dict) and doc.get("kind") == "Deployment":
                    total += int(doc.get("spec", {}).get("replicas", 1) or 1)
    return total


def read_experiment_pod_count(exp_dir: Path, fallback: int) -> int:
    pods_file = exp_dir / "pods.txt"
    if pods_file.exists():
        try:
            text = pods_file.read_text().strip()
            match = re.search(r"pods\s*=\s*(\d+)", text)
            if match:
                return int(match.group(1))
        except Exception:
            pass
    match = re.search(r"_(\d+)pods_", exp_dir.name)
    if match:
        return int(match.group(1))
    return fallback


def infer_matrix_context(root: Path, exp_dir: Path) -> Tuple[str, str]:
    try:
        parts = exp_dir.relative_to(root).parts[:-1]
    except ValueError:
        parts = exp_dir.parts[:-1]
    parts = (root.name,) + tuple(parts)
    context = "/".join(parts)
    seed = ""
    target_pods = ""
    for part in parts:
        seed_match = re.search(r"seed[_-](\d+)", part)
        pods_match = re.search(r"pods[_-](\d+)", part)
        if seed_match:
            seed = seed_match.group(1)
        if pods_match:
            target_pods = pods_match.group(1)
    return seed, target_pods or context


def discover_experiment_dirs(root: Path) -> List[Path]:
    dirs: List[Path] = []
    for current, dirnames, filenames in os.walk(root):
        path = Path(current)
        if find_placement_csv(path):
            dirs.append(path)
            dirnames[:] = []
    return sorted(dirs)


def find_placement_csv(exp_dir: Path) -> Path | None:
    candidates = []
    for pattern in ("*placements_session.csv", "*placement_session.csv"):
        candidates.extend(Path(p) for p in glob.glob(str(exp_dir / pattern)))
    if not candidates:
        return None
    return sorted(candidates)[0]


def infer_algorithm(exp_dir: Path, csv_path: Path) -> str:
    name = exp_dir.name
    if name.startswith("heuristic_op_"):
        return "TotEm-OpOnly"
    if csv_path.name.startswith("heuristic_op_"):
        return "TotEm-OpOnly"
    if name.startswith("heuristic_") or csv_path.name.startswith("heuristic_"):
        return "TotEm"
    if name.startswith("vanilla_") or csv_path.name.startswith("vanilla_"):
        if "most_allocated" in name or "most-allocated" in name:
            return "Vanilla-MostAllocated"
        if "least_allocated" in name or "least-allocated" in name:
            return "Vanilla-LeastAllocated"
        try:
            sample = pd.read_csv(csv_path, nrows=1)
            score_mode = str(sample.get("node_score_mode", [""]).iloc[0])
            if score_mode == "most_allocated":
                return "Vanilla-MostAllocated"
            if score_mode == "least_allocated":
                return "Vanilla-LeastAllocated"
        except Exception:
            pass
        return "Vanilla"
    if name.startswith("piontek-temporal_") or csv_path.name.startswith("piontek_temporal_"):
        return "Piontek-Temporal-K8s"
    if name.startswith("green-mlfq_") or csv_path.name.startswith("green_mlfq_"):
        return "GREEN-MLFQ-K8s"
    if name.startswith("caspian-operational_") or csv_path.name.startswith("caspian_operational_"):
        return "Caspian-style"
    if name.startswith("global-optimal_"):
        return "MILP"
    return name


def load_run_times(path: Path | None) -> Dict[str, float]:
    if not path or not path.exists():
        return {}
    out: Dict[str, float] = {}
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            exp = row.get("experiment")
            seconds = row.get("elapsed_seconds")
            if exp and seconds:
                out[exp] = float(seconds)
    return out


def evaluate_placements(
    placement_csv: Path,
    nodes: Dict[str, dict],
    forecasts: Dict[str, Dict[int, float]],
) -> dict:
    df = pd.read_csv(placement_csv)
    if df.empty:
        return {
            "placed_pods": 0,
            "operational_kg": 0.0,
            "embodied_kg": 0.0,
            "total_kg": 0.0,
            "active_node_slots": 0,
            "avg_active_cpu_util": 0.0,
            "nodes_used": 0,
        }

    occupancy: Dict[Tuple[str, int], List[Tuple[int, float]]] = defaultdict(list)
    for idx, row in df.iterrows():
        node_id = str(row.get("node_id"))
        if node_id not in nodes:
            continue
        try:
            start_slot = int(float(row.get("start_slot", 0)))
            duration = int(float(row.get("duration", 0)))
            cpu = float(row.get("cpu_request", 0.0))
        except Exception:
            continue
        for slot in range(start_slot, start_slot + max(duration, 0)):
            occupancy[(node_id, slot)].append((idx, cpu))

    operational_g = 0.0
    embodied_g = 0.0
    utilization_values: List[float] = []
    nodes_used = set()

    for (node_id, slot), entries in occupancy.items():
        node = nodes[node_id]
        nodes_used.add(node_id)
        total_cpu = max(float(node["cpu"]), 1e-9)
        cpu_used = sum(cpu for _, cpu in entries)
        utilization = max(cpu_used / total_cpu, 0.0)
        if utilization <= 0.0:
            continue
        utilization_values.append(utilization)
        region = str(node["region"]).upper()
        intensity = forecasts.get(region, {}).get(slot, 200.0)
        dynamic_w = (node["max_w"] - node["idle_w"]) * utilization
        operational_g += intensity * ((node["idle_w"] + dynamic_w) / 1000.0)
        embodied_g += node["embodied_g"] / max(node["lifetime_h"], 1e-9)

    placed_pods = int(df["pod_id"].nunique()) if "pod_id" in df.columns else len(df)
    decision_operational_g = None
    if "decision_operational_emissions_g" in df.columns:
        decision_operational_g = pd.to_numeric(
            df["decision_operational_emissions_g"], errors="coerce"
        ).fillna(0.0).sum()

    return {
        "placed_pods": placed_pods,
        "operational_kg": operational_g / 1000.0,
        "embodied_kg": embodied_g / 1000.0,
        "total_kg": (operational_g + embodied_g) / 1000.0,
        "active_node_slots": len(occupancy),
        "avg_active_cpu_util": (
            sum(utilization_values) / len(utilization_values) if utilization_values else 0.0
        ),
        "nodes_used": len(nodes_used),
        "decision_operational_kg": (
            decision_operational_g / 1000.0 if decision_operational_g is not None else ""
        ),
    }


def write_markdown(rows: List[dict], output_md: Path) -> None:
    headers = [
        "seed",
        "target_pods",
        "algorithm",
        "placed_pods",
        "success_rate_pct",
        "total_kg",
        "operational_kg",
        "embodied_kg",
        "per_pod_g",
        "elapsed_seconds",
    ]
    with output_md.open("w") as handle:
        handle.write("# Baseline Comparison Summary\n\n")
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            values = []
            for header in headers:
                value = row.get(header, "")
                if isinstance(value, float):
                    values.append(f"{value:.6g}")
                else:
                    values.append(str(value))
            handle.write("| " + " | ".join(values) + " |\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", required=True)
    parser.add_argument("--nodes-file", default="pkg/carbon-aware/nodes.yaml")
    parser.add_argument("--forecasts-file", default="pkg/carbon-aware/server-python/all_forecasts.json")
    parser.add_argument("--workloads-dir", default="pkg/carbon-aware/workloads")
    parser.add_argument("--run-times-csv", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.experiments_root).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    nodes_file = (repo_root / args.nodes_file).resolve() if not os.path.isabs(args.nodes_file) else Path(args.nodes_file)
    forecasts_file = (repo_root / args.forecasts_file).resolve() if not os.path.isabs(args.forecasts_file) else Path(args.forecasts_file)
    workloads_dir = (repo_root / args.workloads_dir).resolve() if not os.path.isabs(args.workloads_dir) else Path(args.workloads_dir)
    output_csv = Path(args.output_csv).resolve() if args.output_csv else root / "baseline_comparison_summary.csv"
    output_md = Path(args.output_md).resolve() if args.output_md else root / "baseline_comparison_summary.md"
    run_times = load_run_times(Path(args.run_times_csv).resolve() if args.run_times_csv else None)

    nodes = load_nodes(nodes_file)
    forecasts = load_forecasts(forecasts_file)
    total_pods = count_workload_pods(workloads_dir)

    rows: List[dict] = []
    for exp_dir in discover_experiment_dirs(root):
        csv_path = find_placement_csv(exp_dir)
        if not csv_path:
            continue
        row = evaluate_placements(csv_path, nodes, forecasts)
        if row["placed_pods"] == 0:
            continue
        row["algorithm"] = infer_algorithm(exp_dir, csv_path)
        row["experiment"] = exp_dir.name
        row["placement_csv"] = str(csv_path)
        exp_total_pods = read_experiment_pod_count(exp_dir, total_pods)
        seed, target_pods = infer_matrix_context(root, exp_dir)
        row["seed"] = seed
        row["target_pods"] = target_pods or exp_total_pods
        row["total_workload_pods"] = exp_total_pods
        row["success_rate_pct"] = row["placed_pods"] / max(exp_total_pods, 1) * 100.0
        row["per_pod_g"] = row["total_kg"] * 1000.0 / max(row["placed_pods"], 1)
        row["elapsed_seconds"] = run_times.get(exp_dir.name, "")
        rows.append(row)

    rows.sort(key=lambda r: (str(r.get("algorithm", "")), str(r.get("experiment", ""))))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    write_markdown(rows, output_md)
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
