#!/usr/bin/env python3
"""
Run the frozen paper-two resubmission baseline matrix.

The runner regenerates workloads for each (seed, pod-count) cell, snapshots the
generated inputs, runs the frozen baseline set, and validates each cell against
its own workload snapshot. MILP is optional because it is much slower than the
practical schedulers.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import time
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

MILP_BASELINE = {
    "label": "milp",
    "algorithm": "global-optimal",
    "extra": ["--embodied-mode", "proportional"],
    "prefix": "global-optimal_proportional_",
}


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


def snapshot_inputs(combo_dir: Path) -> None:
    shutil.copy2(NODES_FILE, combo_dir / "nodes.yaml")
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
        str(WORKLOADS_DIR),
        "--nodes-file",
        str(NODES_FILE),
        "--forecasts-file",
        str(FORECASTS_FILE),
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


def run_cell_reports(combo_dir: Path) -> None:
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
        str(combo_dir.parent / "run_times.csv"),
        "--output-csv",
        str(combo_dir / "baseline_comparison_summary.csv"),
        "--output-md",
        str(combo_dir / "baseline_comparison_summary.md"),
    ]
    subprocess.run(summary_cmd, cwd=str(REPO_ROOT), check=False)

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
    subprocess.run(common_cmd, cwd=str(REPO_ROOT), check=False)

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
    subprocess.run(validation_cmd, cwd=str(REPO_ROOT), check=False)


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


def aggregate_outputs(root: Path, combo_dirs: list[Path]) -> None:
    summary_frames = []
    common_frames = []
    for combo_dir in combo_dirs:
        summary_path = combo_dir / "baseline_comparison_summary.csv"
        common_path = combo_dir / "attributed_common_pod_comparison.csv"
        if summary_path.exists():
            summary_frames.append(pd.read_csv(summary_path))
        if common_path.exists():
            common_frames.append(pd.read_csv(common_path))

    if summary_frames:
        summary = pd.concat(summary_frames, ignore_index=True)
        summary["target_pods"] = pd.to_numeric(summary["target_pods"], errors="coerce")
        summary["elapsed_seconds"] = pd.to_numeric(summary["elapsed_seconds"], errors="coerce")
        summary.to_csv(root / "baseline_comparison_summary.csv", index=False)
        whole = (
            summary.groupby(["target_pods", "algorithm"], as_index=False)
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
            .sort_values(["target_pods", "algorithm"])
        )
        whole.to_csv(root / "baseline_comparison_aggregate_by_pods.csv", index=False)
        write_markdown(
            whole,
            root / "baseline_comparison_aggregate_by_pods.md",
            "Baseline Comparison Aggregate",
        )

    if common_frames:
        common_detail = pd.concat(common_frames, ignore_index=True)
        common_detail["target_pods"] = pd.to_numeric(common_detail["target_pods"], errors="coerce")
        common_detail.to_csv(root / "attributed_common_pod_comparison.csv", index=False)

        all_common = common_detail[common_detail["comparison"] == "all_algorithms_common"]
        common_agg = (
            all_common.groupby(["target_pods", "algorithm"], as_index=False)
            .agg(
                runs=("experiment", "count"),
                mean_common_pods=("common_pods", "mean"),
                mean_total_kg=("total_kg", "mean"),
                mean_per_pod_g=("per_pod_g", "mean"),
                std_per_pod_g=("per_pod_g", "std"),
            )
            .sort_values(["target_pods", "algorithm"])
        )
        common_agg.to_csv(root / "attributed_common_all_algorithms_aggregate.csv", index=False)
        write_markdown(
            common_agg,
            root / "attributed_common_all_algorithms_aggregate.md",
            "Attributed Common-Pod Aggregate",
        )

        pairwise = common_detail[common_detail["comparison"] != "all_algorithms_common"]
        pairwise_agg = (
            pairwise.groupby(["target_pods", "comparison", "algorithm"], as_index=False)
            .agg(
                runs=("experiment", "count"),
                mean_common_pods=("common_pods", "mean"),
                mean_total_kg=("total_kg", "mean"),
                mean_per_pod_g=("per_pod_g", "mean"),
                std_per_pod_g=("per_pod_g", "std"),
            )
            .sort_values(["target_pods", "comparison", "algorithm"])
        )
        pairwise_agg.to_csv(root / "attributed_common_totem_pairwise_aggregate.csv", index=False)
        write_markdown(
            pairwise_agg,
            root / "attributed_common_totem_pairwise_aggregate.md",
            "Attributed TotEm Pairwise Aggregate",
        )


def selected_baselines(include_milp: bool) -> Iterable[dict]:
    yield from BASELINES
    if include_milp:
        yield MILP_BASELINE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-counts", default="80,140,200")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument(
        "--experiment-root",
        default=str(REPO_ROOT / "experiments/resubmission_baseline_matrix"),
    )
    parser.add_argument("--include-milp", action="store_true")
    parser.add_argument("--loglevel", default="WARNING")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pod_counts = parse_int_list(args.pod_counts)
    seeds = parse_int_list(args.seeds)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.experiment_root).resolve() / f"matrix_{timestamp}"
    root.mkdir(parents=True, exist_ok=False)
    run_times_path = root / "run_times.csv"
    combo_dirs: list[Path] = []

    original_config = CONFIG_FILE.read_text()
    try:
        with run_times_path.open("w", newline="") as handle:
            fieldnames = [
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
                    combo_dir = root / f"seed_{seed}_pods_{pod_count}"
                    combo_dir.mkdir(parents=True, exist_ok=True)
                    combo_dirs.append(combo_dir)
                    update_config(seed=seed, pod_count=pod_count)
                    gen_stdout = combo_dir / "command_logs/generator.stdout.log"
                    gen_stderr = combo_dir / "command_logs/generator.stderr.log"
                    gen_rc = run_command(
                        ["python3", str(GENERATOR)],
                        CARBON_AWARE_DIR,
                        gen_stdout,
                        gen_stderr,
                    )
                    if gen_rc != 0:
                        raise RuntimeError(f"workload generation failed for seed={seed}, pods={pod_count}")
                    snapshot_inputs(combo_dir)

                    for baseline in selected_baselines(args.include_milp):
                        print(f"[matrix] seed={seed} pods={pod_count} baseline={baseline['label']}", flush=True)
                        run_baseline(
                            baseline=baseline,
                            combo_dir=combo_dir,
                            seed=seed,
                            pod_count=pod_count,
                            loglevel=args.loglevel,
                            run_times_writer=writer,
                        )
                        handle.flush()

                    run_cell_reports(combo_dir)
    finally:
        CONFIG_FILE.write_text(original_config)

    aggregate_outputs(root, combo_dirs)
    print(f"Matrix complete: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
