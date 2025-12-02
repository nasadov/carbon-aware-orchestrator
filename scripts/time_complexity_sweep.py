#!/usr/bin/env python3
"""Parameter sweep for precompute runtime scaling (heuristic, vanilla, oracle).

This script generates synthetic infrastructures and workloads, runs the selected
algorithm's precomputation pipeline for a grid of node/pod configurations, and
produces a publication-ready plot summarizing how runtime scales with workload size.

Example usage (from repository root):
    python scripts/time_complexity_sweep.py \
        --algorithm heuristic \
        --node-counts 8,16,32,64 \
        --pod-counts 50,100,200,400 \
        --replicates 3
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import logging
import math
import shutil
import sys
import time
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


@dataclass
class SweepConfig:
    """Runtime configuration for a single sweep run."""

    node_count: int
    total_pods: int
    pods_per_node: float
    replicate: int
    seed: int
    run_label: str
    artifact_dir: Path
    workloads_dir: Path
    nodes_file: Path
    log_dir: Path


def parse_int_series(series: str) -> List[int]:
    """Parse a comma-separated list of integers."""
    values = []
    for chunk in series.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.append(int(chunk))
        except ValueError as exc:  # pragma: no cover - defensive
            raise argparse.ArgumentTypeError(f"Could not parse integer from '{chunk}'") from exc
    if not values:
        raise argparse.ArgumentTypeError("At least one integer value is required")
    return values


def add_repo_modules_to_path() -> Tuple[Path, Path]:
    """Ensure repository-local modules are importable."""
    repo_root = Path(__file__).resolve().parents[1]
    generator_dir = repo_root / "pkg" / "carbon-aware"
    server_python_dir = generator_dir / "server-python"

    for path in (generator_dir, server_python_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    return repo_root, server_python_dir


# Delay heavy imports until sys.path is updated.
DEFAULT_CONFIG = None
generate_nodes_file = None
generate_timeslot_files = None
run_precompute = None
PerformanceLogger = None


def lazy_import_dependencies(algorithm: str) -> None:
    """Import tooling after sys.path has been populated.

    Select the appropriate precompute runner based on the algorithm.
    """
    global DEFAULT_CONFIG, generate_nodes_file, generate_timeslot_files
    global run_precompute, PerformanceLogger

    if DEFAULT_CONFIG is not None:
        return

    from infra_workload_gen import (  # type: ignore
        DEFAULT_CONFIG as GEN_DEFAULT_CONFIG,
        generate_nodes_file as gen_nodes,
        generate_timeslot_files as gen_timeslots,
    )
    # Select precompute entrypoint per algorithm
    if algorithm == "heuristic":
        from carbon_aware.precompute_heuristic import (
            run_heuristic_precomputation as _run_precompute,  # type: ignore
        )
    elif algorithm == "vanilla":
        from carbon_aware.precompute_vanilla import (
            run_vanilla_precomputation as _run_precompute,  # type: ignore
        )
    else:
        from carbon_aware.precompute_global_optimal import (
            run_global_optimal_precomputation as _run_precompute,  # type: ignore
        )
    from carbon_aware.utils import PerformanceLogger as PerfLogger  # type: ignore

    DEFAULT_CONFIG = GEN_DEFAULT_CONFIG
    generate_nodes_file = gen_nodes
    generate_timeslot_files = gen_timeslots
    run_precompute = _run_precompute
    PerformanceLogger = PerfLogger


class ProgressTracker:
    """Minimal progress bar with ETA for environments without tqdm."""

    def __init__(self, total: int):
        self.total = max(total, 1)
        self.completed = 0
        self.start_time = time.perf_counter()
        self.last_message_len = 0

    def _format_duration(self, seconds: float) -> str:
        if math.isnan(seconds) or seconds < 0:
            return "--:--"
        seconds = int(seconds)
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours > 0:
            return f"{hours:d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def update(self, step: int = 1, status: str = "") -> None:
        self.completed = min(self.total, self.completed + step)
        elapsed = time.perf_counter() - self.start_time
        if self.completed > 0:
            rate = elapsed / self.completed
            remaining = max((self.total - self.completed) * rate, 0.0)
        else:
            remaining = float('nan')
        bar_width = 30
        filled = int(bar_width * self.completed / self.total)
        bar = f"{'=' * filled}{'-' * (bar_width - filled)}"
        msg = (
            f"[{bar}] {self.completed}/{self.total} runs | "
            f"elapsed {self._format_duration(elapsed)} | "
            f"ETA {self._format_duration(remaining)}"
        )
        if status:
            msg += f" | last: {status}"
        padding = max(self.last_message_len - len(msg), 0)
        sys.stdout.write('\r' + msg + ' ' * padding)
        sys.stdout.flush()
        self.last_message_len = len(msg)

    def close(self) -> None:
        sys.stdout.write('\n')
        sys.stdout.flush()


def build_sweep_plan(
    node_counts: Sequence[int],
    pod_counts: Sequence[int],
    replicates: int,
    seed: int,
    min_density: float,
    max_density: float,
    base_output_dir: Path,
) -> List[SweepConfig]:
    """Construct the list of runs to execute."""
    plan: List[SweepConfig] = []

    for node_count in node_counts:
        for total_pods in pod_counts:
            pods_per_node = total_pods / max(node_count, 1)
            if pods_per_node < min_density or pods_per_node > max_density:
                continue
            for replicate_idx in range(replicates):
                run_label = (
                    f"nodes{node_count}_pods{total_pods}_rep{replicate_idx + 1}"
                )
                artifact_dir = base_output_dir / "artifacts" / run_label
                workloads_dir = artifact_dir / "workloads"
                nodes_file = artifact_dir / "nodes.yaml"
                log_dir = base_output_dir / "logs" / run_label

                plan.append(
                    SweepConfig(
                        node_count=node_count,
                        total_pods=total_pods,
                        pods_per_node=pods_per_node,
                        replicate=replicate_idx + 1,
                        seed=seed + replicate_idx,
                        run_label=run_label,
                        artifact_dir=artifact_dir,
                        workloads_dir=workloads_dir,
                        nodes_file=nodes_file,
                        log_dir=log_dir,
                    )
                )

    if not plan:
        raise ValueError(
            "No parameter combinations remain after applying density bounds. "
            "Adjust --min-density/--max-density or provide different node/pod lists."
        )

    return plan


def synthesize_inputs(
    config: SweepConfig,
    timeslots: int,
    homogeneous_pods: bool = False,
    pod_cpu: str | None = None,
    pod_mem: str | None = None,
) -> None:
    """Generate nodes/workloads for a single run."""
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    vanilla_dir = config.artifact_dir / "workloads-vanilla"

    generator_nodes_cfg = copy.deepcopy(DEFAULT_CONFIG["nodes"])
    generator_nodes_cfg.update(
        {
            "filename": str(config.nodes_file),
            "num_nodes": config.node_count,
            "random_seed": config.seed,
        }
    )

    generator_workload_cfg = copy.deepcopy(DEFAULT_CONFIG["workload"])
    generator_workload_cfg.update(
        {
            "output_dir": str(config.workloads_dir),
            "vanilla_output_dir": str(vanilla_dir),
            "generation_strategy": "exact_total",
            "exact_total_pods": config.total_pods,
            "num_timeslots": timeslots,
            "random_seed": config.seed,
        }
    )

    # Optional homogeneous pod configuration for scaling to higher pod counts
    if homogeneous_pods:
        if pod_cpu:
            generator_workload_cfg["cpu_options"] = [pod_cpu]
        if pod_mem:
            generator_workload_cfg["mem_options"] = [pod_mem]
        generator_workload_cfg["duration_assignment_method"] = "cycle"
        generator_workload_cfg["deadline_strategy"] = "flexible"
        generator_workload_cfg["deadline_flexibility_hours"] = [1]

    generate_nodes_file(generator_nodes_cfg)
    generate_timeslot_files(generator_workload_cfg)


def run_single_precompute(
    config: SweepConfig,
    forecasts_file: Path,
    prioritize_efficiency: bool,
    operational_only: bool,
    embodied_mode: str,
    algorithm: str,
) -> Tuple[bool, float, Path]:
    """Execute the precompute run for the given algorithm and capture elapsed wall time."""
    config.log_dir.mkdir(parents=True, exist_ok=True)
    algo_key = algorithm.replace("-", "_")
    perf_logger = PerformanceLogger(algorithm)
    perf_filename = f"{algo_key}_performance.csv"
    perf_log_path = config.log_dir / perf_filename
    perf_logger.set_new_log_file(str(config.log_dir), perf_filename)

    start = time.perf_counter()
    # Invoke the selected precompute function with the right signature
    if algorithm == "vanilla":
        success = run_precompute(
            workloads_dir=str(config.workloads_dir),
            nodes_file=str(config.nodes_file),
            forecasts_file=str(forecasts_file),
            session_log_dir=str(config.log_dir),
            perf_logger=perf_logger,
            prioritize_efficiency=prioritize_efficiency,
            operational_only=operational_only,
        )
    elif algorithm == "global-optimal":
        success = run_precompute(
            workloads_dir=str(config.workloads_dir),
            nodes_file=str(config.nodes_file),
            forecasts_file=str(forecasts_file),
            session_log_dir=str(config.log_dir),
            perf_logger=perf_logger,
            prioritize_efficiency=prioritize_efficiency,
            operational_only=operational_only,
            embodied_mode=embodied_mode,
        )
    else:  # heuristic
        success = run_precompute(
            workloads_dir=str(config.workloads_dir),
            nodes_file=str(config.nodes_file),
            forecasts_file=str(forecasts_file),
            session_log_dir=str(config.log_dir),
            perf_logger=perf_logger,
            prioritize_efficiency=prioritize_efficiency,
            operational_only=operational_only,
            embodied_mode=embodied_mode,
        )
    elapsed = time.perf_counter() - start

    return success, elapsed, perf_log_path


def aggregate_run_metrics(perf_log_path: Path) -> Tuple[int, float, float, float]:
    """Extract summary statistics from the performance log."""
    if not perf_log_path.exists() or perf_log_path.stat().st_size == 0:
        return 0, math.nan, math.nan, math.nan

    df = pd.read_csv(perf_log_path)
    if df.empty:
        return 0, math.nan, math.nan, math.nan

    total_calls = len(df)
    total_ms = df["execution_time_ms"].sum()
    mean_ms = df["execution_time_ms"].mean()
    max_ms = df["execution_time_ms"].max()
    return total_calls, total_ms, mean_ms, max_ms


def create_publication_plot(
    results: pd.DataFrame,
    output_dir: Path,
    algorithm: str,
    *,
    knee_detection: bool = False,
    knee_slope_multiplier: float = 3.0,
    time_budget_seconds: float | None = None,
) -> Path:
    """Create and save the runtime scaling plot for the specified algorithm."""
    if results.empty:
        raise ValueError("No successful runs available to plot.")

    summary = (
        results[results["status"] == "success"]
        .groupby(["node_count", "total_pods"], as_index=False)
        .agg(
            median_elapsed=("elapsed_seconds", "median"),
            mean_elapsed=("elapsed_seconds", "mean"),
            std_elapsed=("elapsed_seconds", "std"),
            pods_per_node=("pods_per_node", "mean"),
        )
        .sort_values(["node_count", "total_pods"])
    )

    if summary.empty:
        raise ValueError("No successful runs available to plot.")

    plt.style.use("seaborn-v0_8-colorblind")
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    knees_records = []

    for node_count, group in summary.groupby("node_count"):
        group = group.sort_values("total_pods")
        ax.errorbar(
            group["total_pods"],
            group["median_elapsed"],
            yerr=group["std_elapsed"].fillna(0.0),
            label=f"{node_count} nodes",
            marker="o",
            linewidth=2,
            capsize=3,
        )

        # Optional: knee detection per series (time vs pods)
        if knee_detection and group.shape[0] >= 3:
            xs = group["total_pods"].to_list()
            ys = group["median_elapsed"].to_list()
            # Compute adjacent slopes Δy/Δx
            slopes: List[float] = []
            for i in range(1, len(xs)):
                dx = float(xs[i] - xs[i - 1]) or 1.0
                slopes.append((ys[i] - ys[i - 1]) / dx)
            knee_idx = None
            for i in range(1, len(slopes)):
                prev = slopes[:i]
                baseline = float(pd.Series(prev).median()) if prev else 0.0
                if baseline <= 0:
                    # Fall back to mean if median is zero or negative
                    baseline = float(pd.Series(prev).mean()) if prev else 0.0
                if baseline > 0 and slopes[i] >= knee_slope_multiplier * baseline:
                    knee_idx = i + 1  # slopes index i corresponds to point i+1
                    break
            if knee_idx is not None and 0 <= knee_idx < len(xs):
                kx = xs[knee_idx]
                ky = ys[knee_idx]
                ax.scatter([kx], [ky], s=80, marker="X", color="#d62728", zorder=5)
                ax.axvline(kx, linestyle=":", color="#d62728", alpha=0.6)
                knees_records.append(
                    {
                        "algorithm": algorithm,
                        "node_count": int(node_count),
                        "total_pods": int(kx),
                        "median_elapsed": float(ky),
                        "index": int(knee_idx),
                        "time_budget_hit": False,
                    }
                )

        # Optional: mark time-budget hits
        if time_budget_seconds is not None and time_budget_seconds > 0:
            over = group[group["median_elapsed"] >= time_budget_seconds]
            if not over.empty:
                ax.scatter(
                    over["total_pods"],
                    over["median_elapsed"],
                    s=70,
                    marker="^",
                    color="#9467bd",
                    edgecolor="black",
                    zorder=6,
                    label=None,
                )
                # If we also recorded knees, update time budget flags for matching points
                for _, row in over.iterrows():
                    knees_records.append(
                        {
                            "algorithm": algorithm,
                            "node_count": int(node_count),
                            "total_pods": int(row["total_pods"]),
                            "median_elapsed": float(row["median_elapsed"]),
                            "index": -1,
                            "time_budget_hit": True,
                        }
                    )

    ax.set_xlabel("Total pods scheduled", fontsize=12)
    ax.set_ylabel("Runtime (s)", fontsize=12)
    # Title per algorithm
    algo_label = (
        "TotEm" if algorithm == "heuristic" else
        "Oracle" if algorithm == "global-optimal" else
        "Carbon-Agnostic"
    )
    ax.set_title(f"{algo_label} precompute runtime scaling", fontsize=13)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(title="Infrastructure size", fontsize=10)
    ax.set_ylim(bottom=0)

    note = "Median over replicates; error bars denote ±1σ"
    ax.text(
        0.02,
        0.02,
        note,
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="bottom",
        color="#444444",
    )

    fig.tight_layout()

    base = (
        "heuristic_time_complexity" if algorithm == "heuristic" else
        "global_optimal_time_complexity" if algorithm == "global-optimal" else
        "vanilla_time_complexity"
    )
    png_path = output_dir / f"{base}.png"
    pdf_path = output_dir / f"{base}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Write knees file if requested
    if knee_detection or (time_budget_seconds is not None and time_budget_seconds > 0):
        try:
            knees_df = pd.DataFrame(knees_records)
            if not knees_df.empty:
                knees_path = output_dir / f"{base}_knees.csv"
                knees_df.to_csv(knees_path, index=False)
        except Exception:  # pragma: no cover - non-critical
            pass

    return png_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm",
        choices=["heuristic", "vanilla", "global-optimal"],
        default="heuristic",
        help="Algorithm to evaluate in the sweep (default: heuristic)",
    )
    parser.add_argument(
        "--preset",
        choices=["standard", "highscale"],
        default="standard",
        help="Optional preset for node/pod ranges (default: standard)",
    )
    parser.add_argument(
        "--node-counts",
        type=parse_int_series,
        default="8,16,32,64",
        help="Comma-separated list of node counts to evaluate (default: 8,16,32,64)",
    )
    parser.add_argument(
        "--pod-counts",
        type=parse_int_series,
        default="50,100,200,400",
        help="Comma-separated list of total pod counts to evaluate (default: 50,100,200,400)",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=3,
        help="Number of replicates per parameter combination (default: 3)",
    )
    parser.add_argument(
        "--timeslots",
        type=int,
        default=12,
        help="Number of timeslots per synthetic workload (default: 12)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Base random seed for reproducible generation (default: 2024)",
    )
    parser.add_argument(
        "--min-density",
        type=float,
        default=2.0,
        help="Minimum pods per node to accept a configuration (default: 2.0)",
    )
    parser.add_argument(
        "--max-density",
        type=float,
        default=40.0,
        help="Maximum pods per node to accept a configuration (default: 40.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/time_complexity",
        help="Directory (relative to repo root or absolute) to store results (default: experiments/time_complexity)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional custom name for this sweep run (default: timestamped)",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Preserve generated nodes/workloads; otherwise they are cleaned after each run",
    )
    parser.add_argument(
        "--operational-only",
        action="store_true",
        help="Use operational emissions only during heuristic evaluation",
    )
    parser.add_argument(
        "--embodied-mode",
        type=str,
        default="proportional",
        choices=["proportional", "uniform"],
        help="Embodied emissions allocation mode (default: proportional)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging output to warnings only",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.replicates <= 0:
        raise ValueError("--replicates must be positive")
    if args.timeslots <= 0:
        raise ValueError("--timeslots must be positive")
    if args.min_density <= 0 or args.max_density <= 0:
        raise ValueError("Density bounds must be positive")
    if args.min_density > args.max_density:
        raise ValueError("--min-density cannot exceed --max-density")

    # Normalize node/pod counts in case defaults were not converted by argparse
    if isinstance(args.node_counts, str):
        args.node_counts = parse_int_series(args.node_counts)
    if isinstance(args.pod_counts, str):
        args.pod_counts = parse_int_series(args.pod_counts)

    # Apply presets only if user did not override defaults
    if args.preset == "highscale":
        default_nodes = parse_int_series("8,16,32,64")
        default_pods = parse_int_series("50,100,200,400")
        if args.node_counts == default_nodes:
            args.node_counts = [8, 16, 32, 64, 128, 256]
        if args.pod_counts == default_pods:
            args.pod_counts = [200, 400, 800, 1600, 3200, 6400, 12800]

    repo_root, server_python_dir = add_repo_modules_to_path()
    lazy_import_dependencies(args.algorithm)

    forecasts_file = server_python_dir / "all_forecasts.json"
    if not forecasts_file.exists():
        raise FileNotFoundError(f"Missing carbon forecast file: {forecasts_file}")

    output_dir_arg = Path(args.output_dir)
    if output_dir_arg.is_absolute():
        output_root = output_dir_arg
    else:
        output_root = repo_root / output_dir_arg
    output_root.mkdir(parents=True, exist_ok=True)

    run_stamp = args.run_name or datetime.now(timezone.utc).strftime("sweep_%Y%m%d_%H%M%S")
    base_output_dir = output_root / run_stamp
    base_output_dir.mkdir(parents=True, exist_ok=False)

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    plan = build_sweep_plan(
        node_counts=args.node_counts,
        pod_counts=args.pod_counts,
        replicates=args.replicates,
        seed=args.seed,
        min_density=args.min_density,
        max_density=args.max_density,
        base_output_dir=base_output_dir,
    )

    total_runs = len(plan)
    logging.info(
        "Planned %s runs across %s node counts and %s pod targets",
        total_runs,
        len(args.node_counts),
        len(args.pod_counts),
    )

    label = (
        "Heuristic" if args.algorithm == "heuristic" else
        "Oracle" if args.algorithm == "global-optimal" else
        "Carbon-Agnostic"
    )
    if tqdm is not None:
        progress = tqdm(total=total_runs, desc=f"{label} sweep", unit="run")
    else:
        progress = ProgressTracker(total_runs)

    records = []
    for idx, sweep_cfg in enumerate(plan, start=1):
        logging.info(
            "[%s/%s] nodes=%s pods=%s (%.2f per node) replicate=%s",
            idx,
            total_runs,
            sweep_cfg.node_count,
            sweep_cfg.total_pods,
            sweep_cfg.pods_per_node,
            sweep_cfg.replicate,
        )

        try:
            # Use homogeneous pods automatically for highscale preset
            use_homogeneous = args.preset == "highscale"
            synthesize_inputs(
                sweep_cfg,
                args.timeslots,
                homogeneous_pods=use_homogeneous,
                pod_cpu="100m" if use_homogeneous else None,
                pod_mem="128Mi" if use_homogeneous else None,
            )
            success, elapsed, perf_log_path = run_single_precompute(
                sweep_cfg,
                forecasts_file,
                False,
                args.operational_only,
                args.embodied_mode,
                args.algorithm,
            )
            total_calls, total_ms, mean_ms, max_ms = aggregate_run_metrics(perf_log_path)
            status = "success" if success else "failed"
        except Exception as exc:  # pragma: no cover - runtime guard
            logging.exception("Run %s failed", sweep_cfg.run_label)
            elapsed = float("nan")
            total_calls = 0
            total_ms = float("nan")
            mean_ms = float("nan")
            max_ms = float("nan")
            status = "error"
            perf_log_path = sweep_cfg.log_dir / "heuristic_performance.csv"
            if not perf_log_path.exists():
                perf_log_path = None
            error_message = str(exc)
        else:
            error_message = ""
        finally:
            if not args.keep_artifacts:
                with contextlib.suppress(Exception):
                    shutil.rmtree(sweep_cfg.artifact_dir)

        records.append(
            {
                "run_label": sweep_cfg.run_label,
                "node_count": sweep_cfg.node_count,
                "total_pods": sweep_cfg.total_pods,
                "pods_per_node": sweep_cfg.pods_per_node,
                "replicate": sweep_cfg.replicate,
                "elapsed_seconds": elapsed,
                "status": status,
                "performance_log": (
                    str(perf_log_path.relative_to(base_output_dir))
                    if perf_log_path and perf_log_path.exists()
                    else ""
                ),
                "placement_calls": total_calls,
                "total_execution_ms": total_ms,
                "mean_execution_ms": mean_ms,
                "max_execution_ms": max_ms,
                "error": error_message,
            }
        )

        if tqdm is not None:
            progress.update(1)
            progress.set_postfix_str(status)
        else:
            progress.update(status=status)

    progress.close()

    results_df = pd.DataFrame(records)
    algo_key = args.algorithm.replace("-", "_")
    results_csv = base_output_dir / f"{algo_key}_time_complexity_results.csv"
    results_df.to_csv(results_csv, index=False)

    # Write sweep-level metadata capturing adjustable parameters
    meta = {
        "run_id": run_stamp,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "time_complexity_sweep.py",
        "algorithm": args.algorithm,
        "preset": args.preset,
        "node_counts": list(args.node_counts),
        "pod_counts": list(args.pod_counts),
        "replicates": int(args.replicates),
        "timeslots": int(args.timeslots),
        "seed": int(args.seed),
        "min_density": float(args.min_density),
        "max_density": float(args.max_density),
        "output_dir": str(base_output_dir),
        "keep_artifacts": bool(args.keep_artifacts),
        "operational_only": bool(args.operational_only),
        "embodied_mode": str(args.embodied_mode),
        "quiet": bool(args.quiet),
        "repo_root": str(repo_root),
        "forecasts_file": str(forecasts_file),
        "git_commit": (
            shutil.which("git")
            and __import__("subprocess")
            .run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, capture_output=True, text=True)
            .stdout.strip()
        )
        or "unknown",
    }

    # Attach key generator/workload configuration from DEFAULT_CONFIG
    if DEFAULT_CONFIG is not None:
        try:
            nodes_cfg = DEFAULT_CONFIG.get("nodes", {})
            work_cfg = DEFAULT_CONFIG.get("workload", {})
        except Exception:  # pragma: no cover - defensive
            nodes_cfg = {}
            work_cfg = {}

        regions_cfg = nodes_cfg.get("regions")
        if isinstance(regions_cfg, dict) and "region_counts" in regions_cfg:
            regions = sorted(regions_cfg["region_counts"].keys())
        elif isinstance(regions_cfg, dict) and "region_list" in regions_cfg:
            regions = sorted(regions_cfg["region_list"])
        elif isinstance(regions_cfg, list):
            regions = sorted(regions_cfg)
        else:
            regions = regions_cfg

        hardware_cfg = nodes_cfg.get("hardware_subcategories") or {}
        hardware_categories = sorted(hardware_cfg.keys())

        meta["nodes_config"] = {
            "regions": regions,
            "hardware_categories": hardware_categories,
        }
        meta["workload_config"] = {
            "num_timeslots": work_cfg.get("num_timeslots"),
            "generation_strategy": work_cfg.get("generation_strategy"),
            "exact_total_pods": work_cfg.get("exact_total_pods"),
            "durations": work_cfg.get("durations"),
            "deadline_strategy": work_cfg.get("deadline_strategy"),
            "deadline_flexibility_hours": work_cfg.get("deadline_flexibility_hours"),
            "cpu_options": work_cfg.get("cpu_options"),
            "mem_options": work_cfg.get("mem_options"),
        }
    metadata_path = base_output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2, sort_keys=True)

    successful_runs = results_df[results_df["status"] == "success"].shape[0]
    if successful_runs == 0:
        logging.warning("All runs failed; no plot will be generated")
        return 1

    try:
        # Always enable knee detection with a sensible default; omit time budget annotations
        plot_path = create_publication_plot(
            results_df,
            base_output_dir,
            args.algorithm,
            knee_detection=True,
            knee_slope_multiplier=3.0,
            time_budget_seconds=None,
        )
        logging.info("Saved runtime plot to %s", plot_path)
    except ValueError as exc:
        logging.warning("Plot generation skipped: %s", exc)
        plot_path = None

    logging.info("Collected sweep results in %s", results_csv)
    if plot_path:
        logging.info("Primary plot: %s", plot_path)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())


