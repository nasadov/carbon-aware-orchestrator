#!/usr/bin/env python3
"""Evaluate scheduler behavior across CPU-size, offered-load, and deadline regimes."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CARBON_AWARE_DIR = REPO_ROOT / "pkg/carbon-aware"
CONFIG_FILE = CARBON_AWARE_DIR / "infra-workload-config.yaml"
GENERATOR = CARBON_AWARE_DIR / "infra_workload_gen.py"
NODES_FILE = CARBON_AWARE_DIR / "nodes.yaml"
WORKLOADS_DIR = CARBON_AWARE_DIR / "workloads"
WORKLOADS_VANILLA_DIR = CARBON_AWARE_DIR / "workloads-vanilla"
SERVER_DIR = CARBON_AWARE_DIR / "server-python"
SERVER_MAIN = SERVER_DIR / "main.py"
FORECASTS_FILE = SERVER_DIR / "all_forecasts.json"

import sys

sys.path.insert(0, str(REPO_ROOT / "analysis"))
from attributed_common_pod_comparison import (  # noqa: E402
    attribute_pod_emissions,
    find_placement_csv,
    load_forecasts,
    load_nodes,
    summarize_subset,
)


ALGORITHMS = {
    "totem-oponly": {
        "algorithm": "heuristic",
        "extra": ["--operational-only"],
        "prefix": "heuristic_op_",
    },
    "totem": {
        "algorithm": "heuristic",
        "extra": ["--embodied-mode", "proportional"],
        "prefix": "heuristic_proportional_",
    },
    "vanilla-most-allocated": {
        "algorithm": "vanilla",
        "extra": ["--vanilla-score-mode", "most_allocated"],
        "prefix": "vanilla_most_allocated_",
    },
    "greencourier-spatial": {
        "algorithm": "greencourier-spatial",
        "extra": ["--greencourier-node-score-mode", "most_allocated"],
        "prefix": "greencourier-spatial_",
    },
    "caspian-style": {
        "algorithm": "caspian-operational",
        "extra": [],
        "prefix": "caspian-operational_",
    },
    "green-mlfq": {
        "algorithm": "green-mlfq",
        "extra": [],
        "prefix": "green-mlfq_",
    },
}

RELOCATED_ASSIGNMENTS = {
    "FR": {"Server": 1},
    "ES": {"Laptop": 1},
    "IT-NO": {"Smartphone": 1},
    "DE": {"IoT": 1},
}


def parse_csv(text: str, cast):
    return [cast(part.strip()) for part in text.split(",") if part.strip()]


def cpu_string(cores: float) -> str:
    milli = int(round(cores * 1000))
    return f"{milli}m"


def duration_sum(pods: int, durations: list[int]) -> int:
    return sum(durations[index % len(durations)] for index in range(pods))


def pod_count_for_load(
    cpu_cores: float,
    target_load: float,
    cluster_cores: int,
    horizon_slots: int,
    durations: list[int],
) -> tuple[int, float]:
    capacity_core_hours = cluster_cores * horizon_slots
    best = None
    for pods in range(1, 2001):
        load = cpu_cores * duration_sum(pods, durations) / capacity_core_hours
        candidate = (abs(load - target_load), pods, load)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best[1], best[2]


def update_config(
    original: dict,
    seed: int,
    cpu_cores: float,
    pods: int,
    deadline: str,
) -> None:
    cfg = yaml.safe_load(yaml.safe_dump(original))
    nodes = cfg.setdefault("nodes", {})
    nodes["explicit_assignments"] = RELOCATED_ASSIGNMENTS
    nodes["random_seed"] = seed
    workload = cfg.setdefault("workload", {})
    workload["random_seed"] = seed
    workload["generation_strategy"] = "exact_total"
    workload["exact_total_pods"] = pods
    workload["cpu_options"] = [cpu_string(cpu_cores)]
    workload["deadline_strategy"] = deadline
    with CONFIG_FILE.open("w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def run_logged(cmd: list[str], cwd: Path, stdout: Path, stderr: Path) -> float:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with stdout.open("w") as out, stderr.open("w") as err:
        subprocess.run(cmd, cwd=str(cwd), stdout=out, stderr=err, check=True)
    return time.perf_counter() - start


def copytree_fresh(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def snapshot_inputs(cell: Path) -> None:
    shutil.copy2(NODES_FILE, cell / "nodes.yaml")
    shutil.copy2(FORECASTS_FILE, cell / "all_forecasts.json")
    copytree_fresh(WORKLOADS_DIR, cell / "workloads")
    copytree_fresh(WORKLOADS_VANILLA_DIR, cell / "workloads-vanilla")


def latest_session(cell: Path, prefix: str) -> Path:
    matches = sorted(
        (path for path in cell.iterdir() if path.is_dir() and path.name.startswith(prefix)),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise RuntimeError(f"No session with prefix {prefix} under {cell}")
    return matches[0]


def run_algorithm(label: str, cell: Path, loglevel: str) -> tuple[Path, float]:
    spec = ALGORITHMS[label]
    cmd = [
        "python3",
        str(SERVER_MAIN),
        "--algorithm",
        spec["algorithm"],
        "--precompute",
        "--workloads-dir",
        str(WORKLOADS_DIR),
        "--nodes-file",
        str(NODES_FILE),
        "--forecasts-file",
        str(FORECASTS_FILE),
        "--experiment-dir",
        str(cell),
        "--loglevel",
        loglevel,
    ] + spec["extra"]
    elapsed = run_logged(
        cmd,
        SERVER_DIR,
        cell / "command_logs" / f"{label}.stdout.log",
        cell / "command_logs" / f"{label}.stderr.log",
    )
    return latest_session(cell, spec["prefix"]), elapsed


def validate_cell(cell: Path) -> None:
    cmd = [
        "python3",
        str(REPO_ROOT / "tests/validate_experiments_and_report.py"),
        "--experiments-dir",
        str(cell),
        "--nodes-file",
        str(cell / "nodes.yaml"),
        "--workloads-dir",
        str(cell / "workloads"),
        "--workloads-vanilla-dir",
        str(cell / "workloads-vanilla"),
        "--output-dir",
        str(cell / "validation"),
    ]
    run_logged(
        cmd,
        REPO_ROOT,
        cell / "command_logs" / "validation.stdout.log",
        cell / "command_logs" / "validation.stderr.log",
    )


def summarize_cell(
    sessions: dict[str, Path],
    cell: Path,
    context: dict,
    runtimes: dict[str, float],
) -> list[dict]:
    nodes = load_nodes(cell / "nodes.yaml")
    forecasts = load_forecasts(cell / "all_forecasts.json")
    attributed = {}
    for label, session in sessions.items():
        placement_csv = find_placement_csv(session)
        if placement_csv:
            attributed[label] = attribute_pod_emissions(placement_csv, nodes, forecasts)
    common = set.intersection(*(set(values) for values in attributed.values()))
    rows = []
    for label, values in attributed.items():
        row = summarize_subset(values, common)
        row.update(context)
        row.update(
            {
                "algorithm": label,
                "common_pods": len(common),
                "runtime_s": runtimes[label],
                "session": sessions[label].name,
            }
        )
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-cores", default="0.5,1,2,4")
    parser.add_argument("--offered-loads", default="0.25,0.5,0.75,0.9")
    parser.add_argument("--deadlines", default="tight,flexible")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--algorithms", default="totem-oponly,totem")
    parser.add_argument(
        "--experiment-root",
        default=str(REPO_ROOT / "experiments/workload_regime_sweep"),
    )
    parser.add_argument("--loglevel", default="WARNING")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cpu_sizes = parse_csv(args.cpu_cores, float)
    offered_loads = parse_csv(args.offered_loads, float)
    deadlines = parse_csv(args.deadlines, str)
    seeds = parse_csv(args.seeds, int)
    algorithms = parse_csv(args.algorithms, str)
    unknown = sorted(set(algorithms) - set(ALGORITHMS))
    if unknown:
        raise ValueError(f"Unknown algorithms: {unknown}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.experiment_root).resolve() / f"sweep_{timestamp}"
    root.mkdir(parents=True, exist_ok=False)

    original_text = CONFIG_FILE.read_text()
    original_cfg = yaml.safe_load(original_text)
    workload = original_cfg["workload"]
    durations = [int(value) for value in workload["durations"]]
    horizon_slots = int(workload["num_timeslots"])
    hardware_profiles = original_cfg["nodes"]["hardware_subcategories"]
    cluster_cores = sum(
        int(hardware_profiles[profile]["cpu"]) * count
        for profile, count in original_cfg["nodes"]["hardware_assignment"]["hardware_counts"].items()
    )

    plan_rows = []
    for cpu_cores in cpu_sizes:
        for target_load in offered_loads:
            pods, actual_load = pod_count_for_load(
                cpu_cores, target_load, cluster_cores, horizon_slots, durations
            )
            plan_rows.append(
                {
                    "cpu_cores": cpu_cores,
                    "target_offered_load": target_load,
                    "actual_offered_load": actual_load,
                    "pods": pods,
                    "cluster_cores": cluster_cores,
                    "horizon_slots": horizon_slots,
                }
            )
    plan = pd.DataFrame(plan_rows)
    plan.to_csv(root / "sweep_plan.csv", index=False)

    rows = []
    try:
        for plan_row in plan_rows:
            for deadline in deadlines:
                for seed in seeds:
                    cpu_cores = plan_row["cpu_cores"]
                    target_load = plan_row["target_offered_load"]
                    pods = plan_row["pods"]
                    regime = (
                        f"cpu_{cpu_cores:g}_load_{target_load:.2f}_"
                        f"{deadline}_seed_{seed}"
                    )
                    cell = root / regime
                    cell.mkdir(parents=True, exist_ok=False)
                    print(f"[sweep] {regime} pods={pods}", flush=True)
                    update_config(original_cfg, seed, cpu_cores, pods, deadline)
                    run_logged(
                        ["python3", str(GENERATOR)],
                        CARBON_AWARE_DIR,
                        cell / "command_logs" / "generator.stdout.log",
                        cell / "command_logs" / "generator.stderr.log",
                    )
                    snapshot_inputs(cell)
                    sessions = {}
                    runtimes = {}
                    for label in algorithms:
                        sessions[label], runtimes[label] = run_algorithm(label, cell, args.loglevel)
                    validate_cell(cell)
                    context = {
                        **plan_row,
                        "deadline": deadline,
                        "seed": seed,
                        "regime": regime,
                    }
                    rows.extend(summarize_cell(sessions, cell, context, runtimes))
                    pd.DataFrame(rows).to_csv(root / "common_pod_emissions.csv", index=False)
    finally:
        CONFIG_FILE.write_text(original_text)
        subprocess.run(
            ["python3", str(GENERATOR)],
            cwd=str(CARBON_AWARE_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

    pd.DataFrame(rows).to_csv(root / "common_pod_emissions.csv", index=False)
    print(f"Sweep complete: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
