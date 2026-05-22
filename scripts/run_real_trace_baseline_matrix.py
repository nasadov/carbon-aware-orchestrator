#!/usr/bin/env python3
"""
Run a trace-derived baseline matrix using Azure Trace for Packing 2020.

This runner converts fixed 24-hour windows from the Azure VM request trace into
the same timeslot_*.yaml format used by the synthetic experiments, then runs the
existing scheduler/baseline harness unchanged.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_ROOT / "pkg/carbon-aware/server-python"
SERVER_MAIN = SERVER_DIR / "main.py"
FORECASTS_FILE = SERVER_DIR / "all_forecasts.json"
NODES_FILE = REPO_ROOT / "pkg/carbon-aware/nodes.yaml"
CONVERTER = REPO_ROOT / "scripts/real_traces/convert_azure_packing_trace.py"


BASELINES = [
    {
        "label": "vanilla-most-allocated",
        "algorithm": "vanilla",
        "extra": ["--vanilla-score-mode", "most_allocated"],
        "prefix": "vanilla_most_allocated_",
    },
    {
        "label": "vanilla-least-allocated",
        "algorithm": "vanilla",
        "extra": ["--vanilla-score-mode", "least_allocated"],
        "prefix": "vanilla_least_allocated_",
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


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_str_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def run_command(cmd: list[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> int:
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
        docs = [doc for doc in yaml.safe_load_all(handle) if isinstance(doc, dict) and doc.get("kind") == "Node"]
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


def latest_session_dir(combo_dir: Path, prefix: str) -> Path | None:
    candidates = sorted(
        [path for path in combo_dir.iterdir() if path.is_dir() and path.name.startswith(prefix)],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def selected_baselines(labels: Iterable[str]) -> list[dict]:
    by_label = {baseline["label"]: baseline for baseline in BASELINES}
    wanted = list(labels)
    unknown = sorted(set(wanted) - set(by_label))
    if unknown:
        raise ValueError(f"Unknown baselines: {unknown}")
    return [by_label[label] for label in wanted]


def convert_window(
    sqlite_path: Path,
    trace_root: Path,
    window_day: int,
    seed: int,
    target_pods: int,
    horizon_slots: int,
    log_dir: Path,
) -> Path:
    output_root = trace_root / f"window_day_{window_day}_seed_{seed}_{target_pods}pods"
    cmd = [
        "python3",
        str(CONVERTER),
        "--sqlite-path",
        str(sqlite_path),
        "--output-root",
        str(output_root),
        "--window-day",
        str(window_day),
        "--target-pods",
        str(target_pods),
        "--horizon-slots",
        str(horizon_slots),
        "--seed",
        str(seed),
    ]
    rc = run_command(
        cmd,
        REPO_ROOT,
        log_dir / f"convert_window_{window_day}_seed_{seed}.stdout.log",
        log_dir / f"convert_window_{window_day}_seed_{seed}.stderr.log",
    )
    if rc != 0:
        raise RuntimeError(f"Trace conversion failed for window_day={window_day}, seed={seed}")
    return output_root


def snapshot_inputs(combo_dir: Path, trace_window_dir: Path, multiplier: int) -> None:
    combo_dir.mkdir(parents=True, exist_ok=True)
    expand_nodes_file(NODES_FILE, combo_dir / "nodes.yaml", multiplier)
    shutil.copy2(FORECASTS_FILE, combo_dir / "all_forecasts.json")
    shutil.copy2(trace_window_dir / "canonical_trace_workload.csv", combo_dir / "canonical_trace_workload.csv")
    shutil.copy2(trace_window_dir / "manifest.json", combo_dir / "trace_manifest.json")
    shutil.copy2(trace_window_dir / "trace_diagnostics.csv", combo_dir / "trace_diagnostics.csv")
    shutil.copy2(trace_window_dir / "trace_resource_diagnostics.csv", combo_dir / "trace_resource_diagnostics.csv")
    copytree_fresh(trace_window_dir / "workloads", combo_dir / "workloads")
    copytree_fresh(trace_window_dir / "workloads-vanilla", combo_dir / "workloads-vanilla")


def run_baseline(
    baseline: dict,
    combo_dir: Path,
    capacity_multiplier: int,
    window_day: int,
    seed: int,
    target_pods: int,
    loglevel: str,
    writer: csv.DictWriter,
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
    stdout_path = combo_dir / "command_logs" / f"{label}.stdout.log"
    stderr_path = combo_dir / "command_logs" / f"{label}.stderr.log"
    rc = run_command(cmd, SERVER_DIR, stdout_path, stderr_path)
    elapsed = time.perf_counter() - start
    status = "success" if rc == 0 else f"failed:{rc}"

    session_dir = latest_session_dir(combo_dir, baseline["prefix"])
    if session_dir:
        (session_dir / "pods.txt").write_text(f"pods={target_pods}\n")

    writer.writerow(
        {
            "capacity_multiplier": capacity_multiplier,
            "window_day": window_day,
            "seed": seed,
            "target_pods": target_pods,
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
    commands = [
        [
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
        ],
        [
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
        ],
        [
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
        ],
    ]
    for cmd in commands:
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


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


def aggregate_outputs(root: Path, combo_dirs: list[tuple[int, int, int, int, Path]]) -> None:
    summary_frames = []
    common_frames = []
    for capacity_multiplier, window_day, seed, target_pods, combo_dir in combo_dirs:
        summary_path = combo_dir / "baseline_comparison_summary.csv"
        common_path = combo_dir / "attributed_common_pod_comparison.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            df["capacity_multiplier"] = capacity_multiplier
            df["window_day"] = window_day
            df["seed"] = seed
            df["target_pods"] = target_pods
            summary_frames.append(df)
        if common_path.exists():
            common = pd.read_csv(common_path)
            common["capacity_multiplier"] = capacity_multiplier
            common["window_day"] = window_day
            common["seed"] = seed
            common["target_pods"] = target_pods
            common_frames.append(common)

    if summary_frames:
        summary = pd.concat(summary_frames, ignore_index=True)
        summary["elapsed_seconds"] = pd.to_numeric(summary["elapsed_seconds"], errors="coerce")
        summary.to_csv(root / "real_trace_baseline_comparison_summary.csv", index=False)
        whole = (
            summary.groupby(["capacity_multiplier", "target_pods", "algorithm"], as_index=False)
            .agg(
                runs=("experiment", "count"),
                mean_success=("success_rate_pct", "mean"),
                mean_placed_pods=("placed_pods", "mean"),
                mean_total_kg=("total_kg", "mean"),
                mean_per_pod_g=("per_pod_g", "mean"),
                std_per_pod_g=("per_pod_g", "std"),
                mean_runtime_s=("elapsed_seconds", "mean"),
                std_runtime_s=("elapsed_seconds", "std"),
            )
            .sort_values(["capacity_multiplier", "target_pods", "algorithm"])
        )
        whole.to_csv(root / "real_trace_baseline_aggregate.csv", index=False)
        write_markdown(whole, root / "real_trace_baseline_aggregate.md", "Real Trace Baseline Aggregate")

    if common_frames:
        common_detail = pd.concat(common_frames, ignore_index=True)
        common_detail.to_csv(root / "real_trace_attributed_common_pod_comparison.csv", index=False)
        all_common = common_detail[common_detail["comparison"] == "all_algorithms_common"]
        common_agg = (
            all_common.groupby(["capacity_multiplier", "target_pods", "algorithm"], as_index=False)
            .agg(
                runs=("experiment", "count"),
                mean_common_pods=("common_pods", "mean"),
                mean_per_pod_g=("per_pod_g", "mean"),
                std_per_pod_g=("per_pod_g", "std"),
            )
            .sort_values(["capacity_multiplier", "target_pods", "algorithm"])
        )
        common_agg.to_csv(root / "real_trace_common_aggregate.csv", index=False)
        write_markdown(common_agg, root / "real_trace_common_aggregate.md", "Real Trace Common-Pod Aggregate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sqlite-path",
        default=str(REPO_ROOT / "external/real-traces/raw/azure-packing-2020/packing_trace_zone_a_v1.sqlite"),
    )
    parser.add_argument("--target-pods", type=int, default=200)
    parser.add_argument("--window-days", default="0,2,4,6,8")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--capacity-multipliers", default="1,4")
    parser.add_argument("--horizon-slots", type=int, default=24)
    parser.add_argument("--experiment-root", default=str(REPO_ROOT / "experiments/real_trace_baseline_matrix"))
    parser.add_argument("--trace-workload-root", default=str(REPO_ROOT / "experiments/real_trace_workloads/azure_packing_2020"))
    parser.add_argument(
        "--algorithms",
        default=",".join(b["label"] for b in BASELINES),
        help="Comma-separated baseline labels. Defaults to all practical baselines.",
    )
    parser.add_argument("--loglevel", default="ERROR")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sqlite_path = Path(args.sqlite_path).resolve()
    if not sqlite_path.exists():
        raise FileNotFoundError(f"Missing Azure trace sqlite: {sqlite_path}")

    window_days = parse_int_list(args.window_days)
    seeds = parse_int_list(args.seeds)
    multipliers = parse_int_list(args.capacity_multipliers)
    algorithms = selected_baselines(parse_str_list(args.algorithms))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.experiment_root).resolve() / f"azure_trace_{timestamp}"
    trace_root = Path(args.trace_workload_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    run_times_csv = root / "real_trace_run_times.csv"
    combo_dirs: list[tuple[int, int, int, int, Path]] = []

    with run_times_csv.open("w", newline="") as handle:
        fieldnames = [
            "capacity_multiplier",
            "window_day",
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

        for idx, window_day in enumerate(window_days):
            seed = seeds[idx % len(seeds)]
            trace_window_dir = convert_window(
                sqlite_path=sqlite_path,
                trace_root=trace_root,
                window_day=window_day,
                seed=seed,
                target_pods=args.target_pods,
                horizon_slots=args.horizon_slots,
                log_dir=root / "command_logs",
            )
            for multiplier in multipliers:
                combo_dir = root / f"capacity_{multiplier}x" / f"window_day_{window_day}_seed_{seed}_{args.target_pods}pods"
                snapshot_inputs(combo_dir, trace_window_dir, multiplier)
                combo_dirs.append((multiplier, window_day, seed, args.target_pods, combo_dir))
                for baseline in algorithms:
                    print(
                        f"[real-trace] {multiplier}x day={window_day} seed={seed} pods={args.target_pods} baseline={baseline['label']}",
                        flush=True,
                    )
                    run_baseline(
                        baseline=baseline,
                        combo_dir=combo_dir,
                        capacity_multiplier=multiplier,
                        window_day=window_day,
                        seed=seed,
                        target_pods=args.target_pods,
                        loglevel=args.loglevel,
                        writer=writer,
                    )
                    handle.flush()
                run_cell_reports(combo_dir, run_times_csv)

    aggregate_outputs(root, combo_dirs)
    print(f"Real-trace matrix complete: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
