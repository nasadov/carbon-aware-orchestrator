#!/usr/bin/env python3
"""
Compare schedulers on common placed pods using per-pod attributed emissions.

The regular baseline summary charges node idle and embodied emissions once per
active node-slot. That is correct for whole-schedule totals, but it is not a
good way to compare a subset of pods: filtering rows can recharge a full idle
node-slot to a smaller subset. This script instead evaluates the full schedule,
attributes each active node-slot's operational and embodied emissions to active
pods in proportion to requested CPU, then sums only the pod IDs shared by the
compared schedulers.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from baseline_comparison_summary import (
    discover_experiment_dirs,
    find_placement_csv,
    infer_algorithm,
    infer_matrix_context,
    load_forecasts,
    load_nodes,
)


def attribute_pod_emissions(
    placement_csv: Path,
    nodes: Dict[str, dict],
    forecasts: Dict[str, Dict[int, float]],
) -> Dict[str, dict]:
    df = pd.read_csv(placement_csv)
    pod_totals: Dict[str, dict] = defaultdict(
        lambda: {"operational_g": 0.0, "embodied_g": 0.0}
    )
    occupancy: Dict[Tuple[str, int], List[Tuple[str, float]]] = defaultdict(list)

    for _, row in df.iterrows():
        pod_id = str(row.get("pod_id"))
        node_id = str(row.get("node_id"))
        if node_id not in nodes or not pod_id:
            continue
        try:
            start_slot = int(float(row.get("start_slot", 0)))
            duration = int(float(row.get("duration", 0)))
            cpu = float(row.get("cpu_request", 0.0))
        except Exception:
            continue
        for slot in range(start_slot, start_slot + max(duration, 0)):
            occupancy[(node_id, slot)].append((pod_id, max(cpu, 0.0)))

    for (node_id, slot), entries in occupancy.items():
        if not entries:
            continue
        node = nodes[node_id]
        total_cpu = max(float(node["cpu"]), 1e-9)
        requested_cpu = sum(cpu for _, cpu in entries)
        if requested_cpu <= 0.0:
            share = 1.0 / len(entries)
            shares = [(pod_id, share) for pod_id, _ in entries]
        else:
            shares = [(pod_id, cpu / requested_cpu) for pod_id, cpu in entries]

        utilization = max(requested_cpu / total_cpu, 0.0)
        region = str(node["region"]).upper()
        intensity = forecasts.get(region, {}).get(slot, 200.0)
        dynamic_w = (node["max_w"] - node["idle_w"]) * utilization
        operational_g = intensity * ((node["idle_w"] + dynamic_w) / 1000.0)
        embodied_g = node["embodied_g"] / max(node["lifetime_h"], 1e-9)

        for pod_id, pod_share in shares:
            pod_totals[pod_id]["operational_g"] += operational_g * pod_share
            pod_totals[pod_id]["embodied_g"] += embodied_g * pod_share

    return pod_totals


def summarize_subset(
    attributed: Dict[str, dict],
    pod_ids: Iterable[str],
) -> dict:
    pod_set = set(pod_ids)
    operational_g = sum(attributed[pod_id]["operational_g"] for pod_id in pod_set)
    embodied_g = sum(attributed[pod_id]["embodied_g"] for pod_id in pod_set)
    total_g = operational_g + embodied_g
    count = len(pod_set)
    return {
        "common_pods": count,
        "operational_kg": operational_g / 1000.0,
        "embodied_kg": embodied_g / 1000.0,
        "total_kg": total_g / 1000.0,
        "per_pod_g": total_g / max(count, 1),
    }


def write_markdown(rows: List[dict], output_md: Path) -> None:
    headers = [
        "seed",
        "target_pods",
        "comparison",
        "algorithm",
        "common_pods",
        "total_kg",
        "operational_kg",
        "embodied_kg",
        "per_pod_g",
    ]
    with output_md.open("w") as handle:
        handle.write("# Attributed Common-Pod Comparison\n\n")
        handle.write(
            "Node-slot operational and embodied emissions are attributed to active pods "
            "in proportion to requested CPU before common pod sets are summed.\n\n"
        )
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
    parser.add_argument(
        "--forecasts-file",
        default="pkg/carbon-aware/server-python/all_forecasts.json",
    )
    parser.add_argument("--reference-algorithm", default="TotEm")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    root = Path(args.experiments_root).resolve()
    nodes_file = (
        repo_root / args.nodes_file
        if not Path(args.nodes_file).is_absolute()
        else Path(args.nodes_file)
    )
    forecasts_file = (
        repo_root / args.forecasts_file
        if not Path(args.forecasts_file).is_absolute()
        else Path(args.forecasts_file)
    )
    output_csv = Path(args.output_csv).resolve() if args.output_csv else root / "attributed_common_pod_comparison.csv"
    output_md = Path(args.output_md).resolve() if args.output_md else root / "attributed_common_pod_comparison.md"

    nodes = load_nodes(nodes_file)
    forecasts = load_forecasts(forecasts_file)

    runs = []
    for exp_dir in discover_experiment_dirs(root):
        placement_csv = find_placement_csv(exp_dir)
        if not placement_csv:
            continue
        algorithm = infer_algorithm(exp_dir, placement_csv)
        seed, target_pods = infer_matrix_context(root, exp_dir)
        attributed = attribute_pod_emissions(placement_csv, nodes, forecasts)
        if not attributed:
            continue
        runs.append(
            {
                "algorithm": algorithm,
                "experiment": exp_dir.name,
                "seed": seed,
                "target_pods": target_pods,
                "pod_ids": set(attributed.keys()),
                "attributed": attributed,
            }
        )

    rows: List[dict] = []
    if not runs:
        raise SystemExit(f"No placement CSVs found under {root}")

    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for run in runs:
        grouped[(str(run["seed"]), str(run["target_pods"]))].append(run)

    for (seed, target_pods), group_runs in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        all_common = set.intersection(*(run["pod_ids"] for run in group_runs))
        for run in group_runs:
            row = summarize_subset(run["attributed"], all_common)
            row.update(
                {
                    "seed": seed,
                    "target_pods": target_pods,
                    "comparison": "all_algorithms_common",
                    "algorithm": run["algorithm"],
                    "experiment": run["experiment"],
                }
            )
            rows.append(row)

        reference_runs = [
            run for run in group_runs if run["algorithm"] == args.reference_algorithm
        ]
        if reference_runs:
            reference = sorted(reference_runs, key=lambda run: run["experiment"])[0]
            for run in group_runs:
                if run is reference:
                    continue
                common = reference["pod_ids"] & run["pod_ids"]
                comparison = f"{args.reference_algorithm}_vs_{run['algorithm']}"
                for current in (reference, run):
                    row = summarize_subset(current["attributed"], common)
                    row.update(
                        {
                            "seed": seed,
                            "target_pods": target_pods,
                            "comparison": comparison,
                            "algorithm": current["algorithm"],
                            "experiment": current["experiment"],
                        }
                    )
                    rows.append(row)

    rows.sort(key=lambda item: (str(item["target_pods"]), str(item["seed"]), item["comparison"], item["algorithm"]))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    write_markdown(rows, output_md)
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
