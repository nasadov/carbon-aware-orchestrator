#!/usr/bin/env python3
"""
Run capacity-sensitivity experiments for paper-two resubmission.

The experiment keeps the workload, regions, hardware types, and forecast data
fixed, and replicates the node set by a capacity multiplier. This isolates
whether baseline behavior changes when cluster capacity is less constrained.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = REPO_ROOT / "pkg/carbon-aware/infra-workload-config.yaml"
GENERATOR = REPO_ROOT / "pkg/carbon-aware/infra_workload_gen.py"
CARBON_AWARE_DIR = REPO_ROOT / "pkg/carbon-aware"
SERVER_DIR = REPO_ROOT / "pkg/carbon-aware/server-python"
SERVER_MAIN = SERVER_DIR / "main.py"
FORECASTS_FILE = SERVER_DIR / "all_forecasts.json"
NODES_FILE = CARBON_AWARE_DIR / "nodes.yaml"
WORKLOADS_DIR = CARBON_AWARE_DIR / "workloads"
WORKLOADS_VANILLA_DIR = CARBON_AWARE_DIR / "workloads-vanilla"

sys.path.insert(0, str(REPO_ROOT / "analysis"))
from baseline_comparison_summary import (  # noqa: E402
    find_placement_csv,
    infer_algorithm,
    load_forecasts,
    load_nodes,
)


BASELINES = [
    {
        "label": "vanilla-least-allocated",
        "algorithm": "vanilla",
        "extra": ["--vanilla-score-mode", "least_allocated"],
        "prefix": "vanilla_least_allocated_",
    },
    {
        "label": "vanilla-most-allocated",
        "algorithm": "vanilla",
        "extra": ["--vanilla-score-mode", "most_allocated"],
        "prefix": "vanilla_most_allocated_",
    },
    {
        "label": "totem-oponly",
        "algorithm": "heuristic",
        "extra": ["--operational-only"],
        "prefix": "heuristic_op_",
    },
    {
        "label": "piontek-temporal",
        "algorithm": "piontek-temporal",
        "extra": ["--piontek-node-score-mode", "most_allocated"],
        "prefix": "piontek-temporal_",
    },
    {
        "label": "wait-awhile",
        "algorithm": "wait-awhile",
        "extra": ["--wait-awhile-node-score-mode", "most_allocated"],
        "prefix": "wait-awhile_",
    },
    {
        "label": "greencourier-spatial",
        "algorithm": "greencourier-spatial",
        "extra": ["--greencourier-node-score-mode", "most_allocated"],
        "prefix": "greencourier-spatial_",
    },
    {
        "label": "caspian-style",
        "algorithm": "caspian-operational",
        "extra": [],
        "prefix": "caspian-operational_",
    },
    {
        "label": "green-mlfq",
        "algorithm": "green-mlfq",
        "extra": [],
        "prefix": "green-mlfq_",
    },
    {
        "label": "totem",
        "algorithm": "heuristic",
        "extra": ["--embodied-mode", "proportional"],
        "prefix": "heuristic_proportional_",
    },
]


def parse_int_list(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def update_config(seed: int, pod_count: int) -> None:
    with CONFIG_FILE.open("r") as handle:
        cfg = yaml.safe_load(handle) or {}
    cfg.setdefault("nodes", {})["random_seed"] = seed
    workload = cfg.setdefault("workload", {})
    workload["random_seed"] = seed
    workload["generation_strategy"] = "exact_total"
    workload["exact_total_pods"] = int(pod_count)
    with CONFIG_FILE.open("w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def run_command(cmd: List[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=stdout, stderr=stderr)
    return proc.returncode


def copytree_fresh(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def expand_nodes_file(source: Path, destination: Path, multiplier: int) -> None:
    with source.open("r") as handle:
        docs = [
            doc
            for doc in yaml.safe_load_all(handle)
            if isinstance(doc, dict) and doc.get("kind") == "Node"
        ]
    if multiplier <= 1:
        shutil.copy2(source, destination)
        return

    expanded = []
    for replica_idx in range(multiplier):
        for doc in docs:
            clone = yaml.safe_load(yaml.safe_dump(doc))
            metadata = clone.setdefault("metadata", {})
            labels = metadata.setdefault("labels", {})
            base_name = str(metadata.get("name", "node"))
            new_name = f"{base_name}-r{replica_idx}"
            metadata["name"] = new_name
            labels["kubernetes.io/hostname"] = new_name
            expanded.append(clone)

    with destination.open("w") as handle:
        for doc in expanded:
            yaml.safe_dump(doc, handle, sort_keys=False)
            handle.write("---\n")


def snapshot_inputs(combo_dir: Path, multiplier: int) -> None:
    combo_dir.mkdir(parents=True, exist_ok=True)
    expand_nodes_file(NODES_FILE, combo_dir / "nodes.yaml", multiplier)
    shutil.copy2(FORECASTS_FILE, combo_dir / "all_forecasts.json")
    copytree_fresh(WORKLOADS_DIR, combo_dir / "workloads")
    copytree_fresh(WORKLOADS_VANILLA_DIR, combo_dir / "workloads-vanilla")


def latest_session_dir(combo_dir: Path, prefix: str) -> Path | None:
    candidates = sorted(
        [path for path in combo_dir.iterdir() if path.is_dir() and path.name.startswith(prefix)],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def run_baseline(
    baseline: dict,
    combo_dir: Path,
    multiplier: int,
    seed: int,
    pod_count: int,
    loglevel: str,
    run_times_writer: csv.DictWriter,
) -> None:
    label = baseline["label"]
    cmd = [
        "python3",
        str(SERVER_MAIN),
        "--algorithm",
        baseline["algorithm"],
        "--precompute",
        "--workloads-dir",
        str(combo_dir / "workloads"),
        "--nodes-file",
        str(combo_dir / "nodes.yaml"),
        "--forecasts-file",
        str(combo_dir / "all_forecasts.json"),
        "--experiment-dir",
        str(combo_dir),
        "--loglevel",
        loglevel,
    ] + list(baseline["extra"])

    begin = datetime.now().isoformat(timespec="seconds")
    start = time.perf_counter()
    status = "success"
    stdout_path = combo_dir / "command_logs" / f"{label}.stdout.log"
    stderr_path = combo_dir / "command_logs" / f"{label}.stderr.log"
    return_code = run_command(cmd, SERVER_DIR, stdout_path, stderr_path)
    elapsed = time.perf_counter() - start
    if return_code != 0:
        status = f"failed:{return_code}"

    session_dir = latest_session_dir(combo_dir, baseline["prefix"])
    if session_dir:
        (session_dir / "pods.txt").write_text(f"pods={pod_count}\n")

    run_times_writer.writerow(
        {
            "capacity_multiplier": multiplier,
            "seed": seed,
            "target_pods": pod_count,
            "algorithm": label,
            "experiment": session_dir.name if session_dir else "",
            "session_dir": str(session_dir) if session_dir else "",
            "begin_time": begin,
            "end_time": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": f"{elapsed:.3f}",
            "status": status,
        }
    )


def run_cell_reports(combo_dir: Path, run_times_csv: Path) -> None:
    summary_cmd = [
        "python3",
        str(REPO_ROOT / "analysis/baseline_comparison_summary.py"),
        "--experiments-root",
        str(combo_dir),
        "--nodes-file",
        str(combo_dir / "nodes.yaml"),
        "--forecasts-file",
        str(combo_dir / "all_forecasts.json"),
        "--workloads-dir",
        str(combo_dir / "workloads"),
        "--run-times-csv",
        str(run_times_csv),
        "--output-csv",
        str(combo_dir / "baseline_comparison_summary.csv"),
        "--output-md",
        str(combo_dir / "baseline_comparison_summary.md"),
    ]
    subprocess.run(summary_cmd, cwd=str(REPO_ROOT), check=True)

    common_cmd = [
        "python3",
        str(REPO_ROOT / "analysis/attributed_common_pod_comparison.py"),
        "--experiments-root",
        str(combo_dir),
        "--nodes-file",
        str(combo_dir / "nodes.yaml"),
        "--forecasts-file",
        str(combo_dir / "all_forecasts.json"),
        "--output-csv",
        str(combo_dir / "attributed_common_pod_comparison.csv"),
        "--output-md",
        str(combo_dir / "attributed_common_pod_comparison.md"),
    ]
    subprocess.run(common_cmd, cwd=str(REPO_ROOT), check=True)

    validation_cmd = [
        "python3",
        str(REPO_ROOT / "tests/validate_experiments_and_report.py"),
        "--experiments-dir",
        str(combo_dir),
        "--nodes-file",
        str(combo_dir / "nodes.yaml"),
        "--workloads-dir",
        str(combo_dir / "workloads"),
        "--workloads-vanilla-dir",
        str(combo_dir / "workloads-vanilla"),
        "--output-dir",
        str(combo_dir / "validation"),
    ]
    subprocess.run(validation_cmd, cwd=str(REPO_ROOT), check=True)


def parse_start_duration(row: pd.Series) -> tuple[int, int]:
    start = int(float(row.get("start_slot", 0)))
    duration = max(1, int(float(row.get("duration", 1))))
    return start, duration


def write_diagnostics(rows: list[dict], output_csv: Path) -> None:
    pd.DataFrame(rows).sort_values(
        ["capacity_multiplier", "target_pods", "seed", "algorithm"]
    ).to_csv(output_csv, index=False)


def collect_combo_diagnostics(combo_dir: Path, multiplier: int, seed: int, pod_count: int) -> list[dict]:
    nodes = load_nodes(combo_dir / "nodes.yaml")
    forecasts = load_forecasts(combo_dir / "all_forecasts.json")
    rows: list[dict] = []
    for exp_dir in sorted(path for path in combo_dir.iterdir() if path.is_dir()):
        placement_csv = find_placement_csv(exp_dir)
        if not placement_csv:
            continue
        algorithm = infer_algorithm(exp_dir, placement_csv)
        placements = pd.read_csv(placement_csv)
        if placements.empty:
            continue
        placements["cpu_request"] = pd.to_numeric(placements["cpu_request"], errors="coerce").fillna(0.0)

        region_cpu_hours: defaultdict[str, float] = defaultdict(float)
        weighted_ci_sum = 0.0
        total_cpu_hours = 0.0
        occupancy: defaultdict[tuple[str, int], float] = defaultdict(float)
        for _, row in placements.iterrows():
            node_id = str(row.get("node_id"))
            if node_id not in nodes:
                continue
            start, duration = parse_start_duration(row)
            cpu = float(row.get("cpu_request", 0.0))
            region = str(nodes[node_id]["region"]).upper()
            for slot in range(start, start + duration):
                region_cpu_hours[region] += cpu
                total_cpu_hours += cpu
                weighted_ci_sum += forecasts.get(region, {}).get(slot, 200.0) * cpu
                occupancy[(node_id, slot)] += cpu

        active_utils = []
        for (node_id, _slot), cpu_used in occupancy.items():
            active_utils.append(cpu_used / max(float(nodes[node_id]["cpu"]), 1e-9))

        rows.append(
            {
                "capacity_multiplier": multiplier,
                "seed": seed,
                "target_pods": pod_count,
                "algorithm": algorithm,
                "placed_pods": int(placements["pod_id"].nunique()) if "pod_id" in placements.columns else len(placements),
                "cpu_hours": total_cpu_hours,
                "weighted_carbon_intensity": weighted_ci_sum / max(total_cpu_hours, 1e-9),
                "fr_es_cpu_hour_share_pct": 100.0
                * (region_cpu_hours["FR"] + region_cpu_hours["ES"])
                / max(total_cpu_hours, 1e-9),
                "de_cpu_hour_share_pct": 100.0
                * region_cpu_hours["DE"]
                / max(total_cpu_hours, 1e-9),
                "it_no_cpu_hour_share_pct": 100.0
                * region_cpu_hours["IT-NO"]
                / max(total_cpu_hours, 1e-9),
                "active_node_slots": len(occupancy),
                "avg_active_cpu_util": sum(active_utils) / len(active_utils) if active_utils else 0.0,
            }
        )
    return rows


def fmt(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(df: pd.DataFrame, path: Path, title: str) -> None:
    cols = list(df.columns)
    with path.open("w") as handle:
        handle.write(f"# {title}\n\n")
        handle.write("| " + " | ".join(cols) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for _, row in df.iterrows():
            handle.write("| " + " | ".join(fmt(row[col]) for col in cols) + " |\n")


def aggregate_outputs(root: Path, combo_dirs: list[tuple[int, int, int, Path]]) -> None:
    summary_frames = []
    common_frames = []
    diagnostic_rows: list[dict] = []
    for multiplier, seed, pod_count, combo_dir in combo_dirs:
        summary_path = combo_dir / "baseline_comparison_summary.csv"
        common_path = combo_dir / "attributed_common_pod_comparison.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            df["capacity_multiplier"] = multiplier
            df["seed"] = seed
            df["target_pods"] = pod_count
            summary_frames.append(df)
        if common_path.exists():
            common = pd.read_csv(common_path)
            common["capacity_multiplier"] = multiplier
            common["seed"] = seed
            common["target_pods"] = pod_count
            common_frames.append(common)
        diagnostic_rows.extend(collect_combo_diagnostics(combo_dir, multiplier, seed, pod_count))

    if not summary_frames:
        return

    summary = pd.concat(summary_frames, ignore_index=True)
    summary["elapsed_seconds"] = pd.to_numeric(summary["elapsed_seconds"], errors="coerce")
    summary.to_csv(root / "capacity_baseline_comparison_summary.csv", index=False)
    whole = (
        summary.groupby(["capacity_multiplier", "target_pods", "algorithm"], as_index=False)
        .agg(
            runs=("experiment", "count"),
            mean_success=("success_rate_pct", "mean"),
            mean_placed_pods=("placed_pods", "mean"),
            mean_total_kg=("total_kg", "mean"),
            mean_operational_kg=("operational_kg", "mean"),
            mean_embodied_kg=("embodied_kg", "mean"),
            mean_per_pod_g=("per_pod_g", "mean"),
            std_per_pod_g=("per_pod_g", "std"),
            mean_runtime_s=("elapsed_seconds", "mean"),
            std_runtime_s=("elapsed_seconds", "std"),
        )
        .sort_values(["capacity_multiplier", "target_pods", "algorithm"])
    )
    whole.to_csv(root / "capacity_baseline_aggregate_by_pods.csv", index=False)
    write_markdown(whole, root / "capacity_baseline_aggregate_by_pods.md", "Capacity Baseline Aggregate")

    if common_frames:
        common_detail = pd.concat(common_frames, ignore_index=True)
        common_detail.to_csv(root / "capacity_attributed_common_pod_comparison.csv", index=False)
        all_common = common_detail[common_detail["comparison"] == "all_algorithms_common"]
        common_agg = (
            all_common.groupby(["capacity_multiplier", "target_pods", "algorithm"], as_index=False)
            .agg(
                runs=("experiment", "count"),
                mean_common_pods=("common_pods", "mean"),
                mean_total_kg=("total_kg", "mean"),
                mean_per_pod_g=("per_pod_g", "mean"),
                std_per_pod_g=("per_pod_g", "std"),
            )
            .sort_values(["capacity_multiplier", "target_pods", "algorithm"])
        )
        common_agg.to_csv(root / "capacity_attributed_common_all_algorithms_aggregate.csv", index=False)
        write_markdown(
            common_agg,
            root / "capacity_attributed_common_all_algorithms_aggregate.md",
            "Capacity Attributed Common-Pod Aggregate",
        )

    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(root / "capacity_region_diagnostics.csv", index=False)
    diagnostics_agg = (
        diagnostics.groupby(["capacity_multiplier", "target_pods", "algorithm"], as_index=False)
        .agg(
            mean_weighted_carbon_intensity=("weighted_carbon_intensity", "mean"),
            mean_fr_es_cpu_hour_share_pct=("fr_es_cpu_hour_share_pct", "mean"),
            mean_de_cpu_hour_share_pct=("de_cpu_hour_share_pct", "mean"),
            mean_it_no_cpu_hour_share_pct=("it_no_cpu_hour_share_pct", "mean"),
            mean_active_node_slots=("active_node_slots", "mean"),
            mean_avg_active_cpu_util=("avg_active_cpu_util", "mean"),
        )
        .sort_values(["capacity_multiplier", "target_pods", "algorithm"])
    )
    diagnostics_agg.to_csv(root / "capacity_region_diagnostics_aggregate.csv", index=False)
    write_markdown(
        diagnostics_agg,
        root / "capacity_region_diagnostics_aggregate.md",
        "Capacity Region Diagnostics Aggregate",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-counts", default="160,200")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--multipliers", default="1,2,4")
    parser.add_argument(
        "--experiment-root",
        default=str(REPO_ROOT / "experiments/resubmission_capacity_sensitivity"),
    )
    parser.add_argument("--loglevel", default="ERROR")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pod_counts = parse_int_list(args.pod_counts)
    seeds = parse_int_list(args.seeds)
    multipliers = parse_int_list(args.multipliers)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.experiment_root).resolve() / f"capacity_{timestamp}"
    root.mkdir(parents=True, exist_ok=False)
    run_times_path = root / "capacity_run_times.csv"
    combo_dirs: list[tuple[int, int, int, Path]] = []

    original_config = CONFIG_FILE.read_text()
    try:
        with run_times_path.open("w", newline="") as handle:
            fieldnames = [
                "capacity_multiplier",
                "seed",
                "target_pods",
                "algorithm",
                "experiment",
                "session_dir",
                "begin_time",
                "end_time",
                "elapsed_seconds",
                "status",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

            for seed in seeds:
                for pod_count in pod_counts:
                    update_config(seed=seed, pod_count=pod_count)
                    workload_snapshot = root / "workload_snapshots" / f"seed_{seed}_pods_{pod_count}"
                    workload_snapshot.mkdir(parents=True, exist_ok=True)
                    gen_rc = run_command(
                        ["python3", str(GENERATOR)],
                        CARBON_AWARE_DIR,
                        workload_snapshot / "command_logs/generator.stdout.log",
                        workload_snapshot / "command_logs/generator.stderr.log",
                    )
                    if gen_rc != 0:
                        raise RuntimeError(f"workload generation failed for seed={seed}, pods={pod_count}")

                    for multiplier in multipliers:
                        combo_dir = (
                            root
                            / f"capacity_{multiplier}x"
                            / f"seed_{seed}_pods_{pod_count}"
                        )
                        snapshot_inputs(combo_dir, multiplier)
                        combo_dirs.append((multiplier, seed, pod_count, combo_dir))
                        for baseline in BASELINES:
                            print(
                                f"[capacity] {multiplier}x seed={seed} pods={pod_count} baseline={baseline['label']}",
                                flush=True,
                            )
                            run_baseline(
                                baseline=baseline,
                                combo_dir=combo_dir,
                                multiplier=multiplier,
                                seed=seed,
                                pod_count=pod_count,
                                loglevel=args.loglevel,
                                run_times_writer=writer,
                            )
                            handle.flush()
                        run_cell_reports(combo_dir, run_times_path)
    finally:
        CONFIG_FILE.write_text(original_config)

    aggregate_outputs(root, combo_dirs)
    print(f"Capacity sensitivity complete: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
