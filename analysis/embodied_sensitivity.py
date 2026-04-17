#!/usr/bin/env python3
"""
Embodied-emissions and hardware-lifetime sensitivity analysis for the carbon-aware orchestrator.

This driver:
  1. Fixes the workload to a given exact pod count (default: 80 pods).
  2. Scales the embodied emissions and lifetime parameters used in `nodes.yaml`
     by literature-backed factors.
  3. Re-runs the heuristic and MILP (global-optimal) schedulers under each
     parameter setting using the same workloads and carbon-intensity forecasts.
  4. Re-scores placements with the corresponding hardware metadata to compute
     total emissions.
  5. Produces a CSV summary and a compact plot for inclusion in the paper.

Sensitivity ranges (Boavizta-informed, see docs/paper-two/Notes/boavizta_sku_spread_notes.md):
  - Embodied emissions scaling factors: [0.60, 0.80, 1.00, 1.20, 1.60]
      → span the 10th–90th percentile band of SKU variability across Server,
        Laptop, Smartphone, IoT classes (≈0.56–1.73× mean), with midpoints to
        probe symmetric perturbations around baseline.
  - Lifetime scaling factors: [0.75, 1.0, 1.25]
      → ≈ -25%, 0, +25% around the baseline lifetimes in Table~\\ref{tab:embodied-carbon},
        reflecting recent CS work on lifetime uncertainty and extended lifetimes
        in sustainability-aware scheduling and hardware management
        [ji2025, hewage2025, zhang2025, panteleaki2025].

We vary embodied and lifetime one at a time (keeping the other fixed at 1.0)
to isolate their individual influence on total emissions, and compare the
heuristic against the MILP oracle.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import contextlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from repo_paths import (
    EXPERIMENTS_ROOT,
    FIGURES_ROOT,
    FORECASTS_FILE,
    NODES_FILE as DEFAULT_NODES_FILE,
    PKG_ROOT as DEFAULT_PKG_ROOT,
    REPO_ROOT as DEFAULT_REPO_ROOT,
    SERVER_PYTHON_DIR as DEFAULT_SERVER_PYTHON_DIR,
    WORKLOAD_CONFIG as DEFAULT_WORKLOAD_CONFIG,
    WORKLOAD_GENERATOR as DEFAULT_WORKLOAD_GENERATOR,
    WORKLOADS_DIR as DEFAULT_WORKLOADS_DIR,
)

REPO_ROOT = str(DEFAULT_REPO_ROOT)
PKG_ROOT = str(DEFAULT_PKG_ROOT)
SERVER_PYTHON_DIR = str(DEFAULT_SERVER_PYTHON_DIR)
SERVER_MAIN = os.path.join(SERVER_PYTHON_DIR, "main.py")
WORKLOAD_GENERATOR = str(DEFAULT_WORKLOAD_GENERATOR)
WORKLOAD_CONFIG = str(DEFAULT_WORKLOAD_CONFIG)
WORKLOADS_DIR = str(DEFAULT_WORKLOADS_DIR)
NODES_FILE = str(DEFAULT_NODES_FILE)
GROUND_TRUTH_FORECAST = str(FORECASTS_FILE)

DEFAULT_EXPERIMENT_ROOT = str(EXPERIMENTS_ROOT / "embodied_sensitivity")
DEFAULT_FIGURE_DIR = str(FIGURES_ROOT / "EmbodiedSensitivity")

# Add carbon-aware modules to path
if SERVER_PYTHON_DIR not in os.sys.path:
    os.sys.path.append(SERVER_PYTHON_DIR)

from carbon_aware.placement_summary import get_total_pods_from_workloads  # noqa: E402
from carbon_aware.forecast_noise import load_forecasts  # noqa: E402


# ---------------------------------------------------------------------------
# Simple helpers for YAML I/O
# ---------------------------------------------------------------------------


def _load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        # For multi-document structures we expect `data` to be a list
        if isinstance(data, list):
            yaml.safe_dump_all(data, f, sort_keys=False)
        else:
            yaml.safe_dump(data, f, sort_keys=False)


# ---------------------------------------------------------------------------
# Config managers (borrowed pattern from forecast_error_sensitivity)
# ---------------------------------------------------------------------------


class WorkloadConfigManager:
    """Context manager that restores the original infra-workload config file on exit."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self._backup_path: Optional[str] = None

    def __enter__(self):
        fd, tmp_path = tempfile.mkstemp(prefix="embodied_config_backup_", suffix=".yaml")
        os.close(fd)
        shutil.copyfile(self.config_path, tmp_path)
        self._backup_path = tmp_path
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._backup_path and os.path.exists(self._backup_path):
            shutil.copyfile(self._backup_path, self.config_path)
            os.remove(self._backup_path)


class NodesConfigManager:
    """Context manager that restores the original nodes.yaml on exit."""

    def __init__(self, nodes_path: str):
        self.nodes_path = nodes_path
        self._backup_path: Optional[str] = None

    def __enter__(self):
        fd, tmp_path = tempfile.mkstemp(prefix="embodied_nodes_backup_", suffix=".yaml")
        os.close(fd)
        shutil.copyfile(self.nodes_path, tmp_path)
        self._backup_path = tmp_path
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._backup_path and os.path.exists(self._backup_path):
            shutil.copyfile(self._backup_path, self.nodes_path)
            os.remove(self._backup_path)


# ---------------------------------------------------------------------------
# Workload config helpers
# ---------------------------------------------------------------------------


def ensure_exact_total_strategy(config_path: str) -> None:
    """Ensure workload.generation_strategy == 'exact_total'."""
    data = _load_yaml(config_path)
    workload = data.setdefault("workload", {})
    if workload.get("generation_strategy") != "exact_total":
        workload["generation_strategy"] = "exact_total"
        _write_yaml(config_path, data)


def set_exact_total_pods(config_path: str, pods: int) -> None:
    data = _load_yaml(config_path)
    workload = data.setdefault("workload", {})
    if workload.get("exact_total_pods") != pods:
        workload["exact_total_pods"] = pods
        _write_yaml(config_path, data)


def set_random_seed(config_path: str, seed: int) -> None:
    """Update both workload.random_seed and nodes.random_seed if present."""
    data = _load_yaml(config_path)
    workload = data.setdefault("workload", {})
    nodes = data.setdefault("nodes", {})
    workload["random_seed"] = seed
    nodes["random_seed"] = seed
    _write_yaml(config_path, data)


def regenerate_workloads(generator_path: str) -> None:
    subprocess.run(
        ["python3", generator_path],
        cwd=PKG_ROOT,
        check=True,
    )


# ---------------------------------------------------------------------------
# Nodes scaling helpers
# ---------------------------------------------------------------------------


def load_baseline_nodes(nodes_file: str) -> List[dict]:
    """Load the original multi-document nodes.yaml as a list of documents."""
    with open(nodes_file, "r", encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))
    return docs


def write_scaled_nodes(
    nodes_file: str,
    base_docs: List[dict],
    embodied_scale: float,
    lifetime_scale: float,
) -> None:
    """
    Write a scaled version of nodes.yaml where:
      embodied_emissions := embodied_emissions * embodied_scale
      lifetime_years    := lifetime_years    * lifetime_scale

    Scaling is applied uniformly across hardware categories. Baseline values
    are taken from Boavizta-derived category averages as encoded in nodes.yaml.
    """
    scaled_docs: List[dict] = []
    for doc in base_docs:
        if not isinstance(doc, dict) or doc.get("kind") != "Node":
            scaled_docs.append(doc)
            continue

        doc = yaml.safe_load(yaml.safe_dump(doc))  # deep copy
        metadata = doc.setdefault("metadata", {})
        annotations = metadata.setdefault("annotations", {})

        # Embodied emissions (kg CO2e)
        emb_key = "hardware.carbon/embodied_emissions"
        if emb_key in annotations:
            try:
                base_emb = float(annotations[emb_key])
                annotations[emb_key] = f"{base_emb * embodied_scale:.6f}"
            except (TypeError, ValueError):
                # Leave as-is on parse errors
                pass

        # Lifetime (years)
        life_key = "hardware.carbon/lifetime_years"
        if life_key in annotations:
            try:
                base_life = float(annotations[life_key])
                annotations[life_key] = f"{base_life * lifetime_scale:.4f}"
            except (TypeError, ValueError):
                pass

        scaled_docs.append(doc)

    _write_yaml(nodes_file, scaled_docs)


# ---------------------------------------------------------------------------
# Emissions computation (adapted from forecast_error_sensitivity)
# ---------------------------------------------------------------------------


def load_nodes_with_forecasts(nodes_file: str, forecasts_file: str) -> Dict[str, dict]:
    """Load nodes.yaml and enrich with forecasts for emissions recomputation."""
    with open(nodes_file, "r", encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))
    nodes = []
    for doc in docs:
        if not doc or doc.get("kind") != "Node":
            continue
        nodes.append(doc)
    forecasts = load_forecasts(forecasts_file)
    nodes_dict: Dict[str, dict] = {}
    for node in nodes:
        node_id = node.get("metadata", {}).get("name")
        if not node_id:
            continue
        region = (
            node.get("metadata", {})
            .get("labels", {})
            .get("topology.kubernetes.io/region", "")
            .upper()
        )
        power = {
            "idle": float(
                node.get("metadata", {})
                .get("annotations", {})
                .get("hardware.power/idle_watts", 100)
            ),
            "max": float(
                node.get("metadata", {})
                .get("annotations", {})
                .get("hardware.power/max_watts", 300)
            ),
        }
        total_cpu = node.get("status", {}).get("allocatable", {}).get("cpu", "0")
        if isinstance(total_cpu, str) and total_cpu.endswith("m"):
            total_cpu_value = float(total_cpu[:-1]) / 1000.0
        else:
            total_cpu_value = float(total_cpu)
        ram_str = node.get("status", {}).get("allocatable", {}).get("memory", "0")
        if isinstance(ram_str, str) and ram_str.endswith("Ki"):
            total_ram = float(ram_str[:-2]) / 1024.0
        elif isinstance(ram_str, str) and ram_str.endswith("Mi"):
            total_ram = float(ram_str[:-2])
        elif isinstance(ram_str, str) and ram_str.endswith("Gi"):
            total_ram = float(ram_str[:-2]) * 1024.0
        else:
            total_ram = float(ram_str)
        nodes_dict[node_id] = {
            "region": region,
            "power": power,
            "totalCpu": total_cpu_value,
            "totalRam": total_ram,
            "embodied": float(
                node.get("metadata", {})
                .get("annotations", {})
                .get("hardware.carbon/embodied_emissions", 0)
            )
            * 1000.0,  # kg → g
            "lifetime_hours": float(
                node.get("metadata", {})
                .get("annotations", {})
                .get("hardware.carbon/lifetime_years", 3)
            )
            * 365
            * 24,
            "forecast": forecasts.get(region, {}),
        }
    return nodes_dict


def compute_total_emissions_kg(placement_csv: str, nodes_dict: dict) -> float:
    """Recompute total emissions (kgCO2e) for a placement CSV."""
    df = pd.read_csv(placement_csv)
    if df.empty:
        return 0.0

    from collections import defaultdict

    occupancy: Dict[Tuple[str, int], List[float]] = defaultdict(list)

    def parse_cpu(val) -> float:
        s = str(val)
        if s.endswith("m"):
            return float(s[:-1]) / 1000.0
        try:
            return float(s)
        except ValueError:
            return 0.0

    for _, row in df.iterrows():
        node_id = row.get("node_id")
        duration = int(float(row.get("duration", 0)))
        if duration <= 0 or node_id not in nodes_dict:
            continue
        start_slot = int(float(row.get("start_slot", 0)))
        cpu_request = parse_cpu(row.get("cpu_request", 0))
        for offset in range(duration):
            occupancy[(node_id, start_slot + offset)].append(cpu_request)

    total_kg = 0.0
    for (node_id, slot), cpu_requests in occupancy.items():
        node = nodes_dict[node_id]
        total_cpu = max(node["totalCpu"], 1e-6)
        utilization = sum(cpu_requests) / total_cpu
        if utilization <= 0:
            continue
        idle_w = node["power"]["idle"]
        max_w = node["power"]["max"]
        dynamic_k = max_w - idle_w
        intensity = node["forecast"].get(slot, 200.0)
        embodied_per_hour = (
            node["embodied"] / max(node["lifetime_hours"], 1) if node["lifetime_hours"] else 0.0
        )
        # Operational emissions
        idle_g = intensity * (idle_w / 1000.0)
        dynamic_g = intensity * (dynamic_k * utilization / 1000.0)
        total_kg += (idle_g + dynamic_g + embodied_per_hour) / 1000.0
    return total_kg


# ---------------------------------------------------------------------------
# Running algorithms
# ---------------------------------------------------------------------------


def find_placement_csv(session_dir: str, algorithm: str) -> Optional[str]:
    candidates: List[str] = []
    if algorithm == "global-optimal":
        candidates.append("global_optimal_placements_session.csv")
    elif algorithm == "heuristic":
        candidates.extend(
            [
                "heuristic_prop_placements_session.csv",
                "heuristic_uniform_placements_session.csv",
                "heuristic_op_placements_session.csv",
                "heuristic_placements_session.csv",
            ]
        )
    else:
        return None

    for name in candidates:
        path = os.path.join(session_dir, name)
        if os.path.exists(path):
            return path
    return None


def run_algorithm(
    algorithm: str,
    workloads_dir: str,
    nodes_file: str,
    forecasts_file: str,
    experiment_dir: str,
    embodied_mode: str = "proportional",
    loglevel: str = "WARNING",
    reuse_existing: bool = False,
) -> Tuple[str, str]:
    """
    Run one algorithm in precomputation mode and return (session_dir, placement_csv_path).
    """
    os.makedirs(experiment_dir, exist_ok=True)

    if reuse_existing:
        # Try to find the latest session directory with a placement CSV
        session_dirs = [
            os.path.join(experiment_dir, entry)
            for entry in sorted(os.listdir(experiment_dir))
            if os.path.isdir(os.path.join(experiment_dir, entry))
        ]
        session_dirs = sorted(session_dirs, key=os.path.getmtime, reverse=True)
        for sd in session_dirs:
            placement_csv = find_placement_csv(sd, algorithm)
            if placement_csv:
                return sd, placement_csv

    before = set(os.listdir(experiment_dir))
    cmd = [
        "python3",
        SERVER_MAIN,
        "--algorithm",
        algorithm,
        "--precompute",
        "--workloads-dir",
        workloads_dir,
        "--nodes-file",
        nodes_file,
        "--forecasts-file",
        forecasts_file,
        "--experiment-dir",
        experiment_dir,
        "--loglevel",
        loglevel,
    ]
    if algorithm in ("heuristic", "global-optimal"):
        cmd.extend(["--embodied-mode", embodied_mode])
    subprocess.run(cmd, cwd=SERVER_PYTHON_DIR, check=True)
    after = set(os.listdir(experiment_dir))
    new_entries = sorted(after - before)
    if not new_entries:
        raise RuntimeError(f"No session directory created for {algorithm} under {experiment_dir}")
    session_dirs = [
        os.path.join(experiment_dir, entry)
        for entry in new_entries
        if os.path.isdir(os.path.join(experiment_dir, entry))
    ]
    if not session_dirs:
        # Fallback: pick latest directory
        session_dirs = [
            os.path.join(experiment_dir, entry)
            for entry in sorted(os.listdir(experiment_dir))
            if os.path.isdir(os.path.join(experiment_dir, entry))
        ]
    session_dir = max(session_dirs, key=os.path.getmtime)
    placement_csv = find_placement_csv(session_dir, algorithm)
    if not placement_csv:
        raise RuntimeError(f"Could not locate placement CSV for {algorithm} in {session_dir}")
    return session_dir, placement_csv


def find_existing_placement(experiment_dir: str, algorithm: str) -> Optional[str]:
    if not os.path.isdir(experiment_dir):
        return None
    session_dirs = [
        os.path.join(experiment_dir, entry)
        for entry in sorted(os.listdir(experiment_dir))
        if os.path.isdir(os.path.join(experiment_dir, entry))
    ]
    session_dirs = sorted(session_dirs, key=os.path.getmtime, reverse=True)
    for sd in session_dirs:
        placement_csv = find_placement_csv(sd, algorithm)
        if placement_csv:
            return placement_csv
    return None


# ---------------------------------------------------------------------------
# Main experiment logic
# ---------------------------------------------------------------------------


@dataclass
class SensitivityRecord:
    algorithm: str
    pods: int
    seed: int
    embodied_scale: float
    lifetime_scale: float
    total_kg: float


def run_sensitivity(args) -> pd.DataFrame:
    records: List[SensitivityRecord] = []

    # Fix workloads to the requested pod count and iterate over seeds
    workload_ctx = (
        WorkloadConfigManager(args.workload_config)
        if not args.analyze_only
        else contextlib.nullcontext()
    )

    with workload_ctx:
        if not args.analyze_only:
            ensure_exact_total_strategy(args.workload_config)
            set_exact_total_pods(args.workload_config, args.pods)

        # Cache baseline nodes once
        baseline_docs = load_baseline_nodes(args.nodes_file)

        for seed in args.seeds:
            print(f"\n==============================\nSeed {seed}\n==============================")
            if not args.analyze_only:
                set_random_seed(args.workload_config, seed)
                regenerate_workloads(args.generator)
                total_pods = get_total_pods_from_workloads(WORKLOADS_DIR)
                print(f"Generated workloads for {total_pods} pods (target {args.pods}).")

            with NodesConfigManager(args.nodes_file):
                # Embodied scaling sweep (lifetime fixed at 1.0)
                for emb_scale in args.embodied_scales:
                    life_scale = 1.0
                    print(f"\n==> Embodied scaling factor {emb_scale:.2f}, lifetime_scale={life_scale:.2f}")
                    write_scaled_nodes(args.nodes_file, baseline_docs, emb_scale, life_scale)
                    nodes_dict = load_nodes_with_forecasts(args.nodes_file, args.ground_truth_forecast)
                    for algorithm in args.algorithms:
                        scenario_dir = os.path.join(
                            args.experiment_root,
                            f"pods_{args.pods}",
                            f"seed_{seed}",
                            f"emb_{emb_scale:.2f}_life_{life_scale:.2f}",
                            algorithm,
                        )
                        if args.analyze_only:
                            placement_csv = find_existing_placement(scenario_dir, algorithm)
                            if not placement_csv:
                                raise RuntimeError(f"No existing placement for {algorithm} in {scenario_dir}")
                        else:
                            _, placement_csv = run_algorithm(
                                algorithm=algorithm,
                                workloads_dir=args.workloads_dir,
                                nodes_file=args.nodes_file,
                                forecasts_file=args.ground_truth_forecast,
                                experiment_dir=scenario_dir,
                                embodied_mode=args.embodied_mode,
                                loglevel=args.loglevel,
                                reuse_existing=args.reuse_existing,
                            )
                        total_kg = compute_total_emissions_kg(placement_csv, nodes_dict)
                        print(f"    Total emissions: {total_kg:.4f} kg CO2e")
                        records.append(
                            SensitivityRecord(
                                algorithm=algorithm,
                                pods=args.pods,
                                seed=seed,
                                embodied_scale=emb_scale,
                                lifetime_scale=life_scale,
                                total_kg=total_kg,
                            )
                        )

                # Lifetime scaling sweep (embodied fixed at 1.0)
                for life_scale in args.lifetime_scales:
                    emb_scale = 1.0
                    if life_scale == 1.0:
                        continue
                    print(f"\n==> Lifetime scaling factor {life_scale:.2f}, embodied_scale={emb_scale:.2f}")
                    write_scaled_nodes(args.nodes_file, baseline_docs, emb_scale, life_scale)
                    nodes_dict = load_nodes_with_forecasts(args.nodes_file, args.ground_truth_forecast)
                    for algorithm in args.algorithms:
                        scenario_dir = os.path.join(
                            args.experiment_root,
                            f"pods_{args.pods}",
                            f"seed_{seed}",
                            f"emb_{emb_scale:.2f}_life_{life_scale:.2f}",
                            algorithm,
                        )
                        if args.analyze_only:
                            placement_csv = find_existing_placement(scenario_dir, algorithm)
                            if not placement_csv:
                                raise RuntimeError(f"No existing placement for {algorithm} in {scenario_dir}")
                        else:
                            _, placement_csv = run_algorithm(
                                algorithm=algorithm,
                                workloads_dir=args.workloads_dir,
                                nodes_file=args.nodes_file,
                                forecasts_file=args.ground_truth_forecast,
                                experiment_dir=scenario_dir,
                                embodied_mode=args.embodied_mode,
                                loglevel=args.loglevel,
                                reuse_existing=args.reuse_existing,
                            )
                        total_kg = compute_total_emissions_kg(placement_csv, nodes_dict)
                        print(f"    Total emissions: {total_kg:.4f} kg CO2e")
                        records.append(
                            SensitivityRecord(
                                algorithm=algorithm,
                                pods=args.pods,
                                seed=seed,
                                embodied_scale=emb_scale,
                                lifetime_scale=life_scale,
                                total_kg=total_kg,
                            )
                        )

    # Convert to DataFrame and derive per-pod and deltas
    df = pd.DataFrame([r.__dict__ for r in records])
    if df.empty:
        return df
    df["kg_per_pod"] = df["total_kg"] / df["pods"]

    # Baseline rows: embodied_scale==1.0 and lifetime_scale==1.0
    baseline_mask = (df["embodied_scale"] == 1.0) & (df["lifetime_scale"] == 1.0)
    if baseline_mask.any():
        baseline = df[baseline_mask].set_index(["algorithm", "seed"])["kg_per_pod"]

        def delta_vs_baseline(row):
            key = (row["algorithm"], row["seed"])
            if key not in baseline:
                return 0.0
            base_val = baseline[key]
            if base_val == 0:
                return 0.0
            return (row["kg_per_pod"] - base_val) / base_val * 100.0

        df["delta_vs_baseline_pct"] = df.apply(delta_vs_baseline, axis=1)
    else:
        df["delta_vs_baseline_pct"] = 0.0

    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _latest_timestamp_for_pods(experiment_root: str, pods: int) -> Optional[str]:
    """Find the most recent modification timestamp under experiments/pods_<pods> (as filename-safe slug)."""
    pods_dir = os.path.join(experiment_root, f"pods_{pods}")
    if not os.path.isdir(pods_dir):
        return None
    latest_mtime = None
    for dirpath, _, filenames in os.walk(pods_dir):
        for name in filenames:
            try:
                mtime = os.path.getmtime(os.path.join(dirpath, name))
                if (latest_mtime is None) or (mtime > latest_mtime):
                    latest_mtime = mtime
            except OSError:
                continue
    if latest_mtime is None:
        return None
    ts = datetime.fromtimestamp(latest_mtime, tz=timezone.utc)
    return ts.strftime("%Y%m%d_%H%MUTC")


def plot_sensitivity(df: pd.DataFrame, figure_dir: str, pods: int) -> str:
    os.makedirs(figure_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.8), sharex=False)

    # Aggregate across seeds to plot mean ± std
    grouped_cols_emb = ["algorithm", "embodied_scale"]
    grouped_cols_life = ["algorithm", "lifetime_scale"]

    emb_agg = (
        df[df["lifetime_scale"] == 1.0]
        .groupby(grouped_cols_emb)
        .agg(
            mean_kg=("kg_per_pod", "mean"),
            std_kg=("kg_per_pod", "std"),
            mean_delta=("delta_vs_baseline_pct", "mean"),
            count=("kg_per_pod", "count"),
        )
        .reset_index()
    )
    life_agg = (
        df[df["embodied_scale"] == 1.0]
        .groupby(grouped_cols_life)
        .agg(
            mean_kg=("kg_per_pod", "mean"),
            std_kg=("kg_per_pod", "std"),
            mean_delta=("delta_vs_baseline_pct", "mean"),
            count=("kg_per_pod", "count"),
        )
        .reset_index()
    )

    # Styling aligned with other paper figures (Fig. 7/8 palette)
    style = {
        # TotEm solid orange, Oracle dashed green to match runtime overlays
        "heuristic": {"label": "TotEm", "color": "#ff7f0e", "marker": "s", "linestyle": "-"},
        "global-optimal": {"label": "Oracle", "color": "#2ca02c", "marker": "o", "linestyle": "--"},
    }
    algorithms = [algo for algo in ["heuristic", "global-optimal"] if algo in df["algorithm"].unique()]

    # Left subplot: embodied scaling (lifetime_scale == 1.0)
    for algo in algorithms:
        sub = emb_agg[emb_agg["algorithm"] == algo].sort_values("embodied_scale")
        if sub.empty:
            continue
        s = style.get(algo, {"label": algo, "color": "#1f77b4", "marker": "o"})
        ax1.errorbar(
            sub["embodied_scale"],
            sub["mean_kg"],
            yerr=sub["std_kg"],
            marker=s["marker"],
            label=s["label"],
            color=s["color"],
            linestyle=s.get("linestyle", "-"),
            linewidth=2.0,
            markersize=6.5,
            capsize=3,
        )
    ax1.set_xlabel("Embodied scaling factor", fontsize=8)
    ax1.set_ylabel("Emissions per pod (kg CO$_2$e)", fontsize=8)
    ax1.set_title("")
    ax1.grid(True, linestyle="--", alpha=0.35, linewidth=0.6)
    ax1.tick_params(labelsize=8)

    # Right subplot: lifetime scaling (embodied_scale == 1.0)
    for algo in algorithms:
        sub = life_agg[life_agg["algorithm"] == algo].sort_values("lifetime_scale")
        if sub.empty:
            continue
        s = style.get(algo, {"label": algo, "color": "#1f77b4", "marker": "o"})
        ax2.errorbar(
            sub["lifetime_scale"],
            sub["mean_kg"],
            yerr=sub["std_kg"],
            marker=s["marker"],
            label=s["label"],
            color=s["color"],
            linestyle=s.get("linestyle", "-"),
            linewidth=2.0,
            markersize=6.5,
            capsize=3,
        )
    ax2.set_xlabel("Lifetime scaling factor", fontsize=8)
    ax2.set_ylabel("Emissions per pod (kg CO$_2$e)", fontsize=8)
    ax2.set_title("")
    ax2.grid(True, linestyle="--", alpha=0.35, linewidth=0.6)
    ax2.tick_params(labelsize=8)

    # Shared legend
    # Align y-ticks/formatting: coarser ticks, mirrored onto upper, 3-dec precision
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ticks = ax2.get_yticks()
    ax1.set_yticks(ticks)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend(handles, labels, loc="upper left", fontsize=8, frameon=False)

    plt.subplots_adjust(bottom=0.12, top=0.95, hspace=0.32)

    # Timestamped filename based on data recency
    ts = _latest_timestamp_for_pods(DEFAULT_EXPERIMENT_ROOT, pods)
    suffix = f"_{ts}" if ts else ""
    pdf_path = os.path.join(figure_dir, f"embodied_sensitivity_{pods}pods{suffix}.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved sensitivity plot: {pdf_path}")
    return pdf_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Embodied and lifetime sensitivity driver.")
    parser.add_argument(
        "--pods",
        type=int,
        default=80,
        help="Exact pod count to generate workloads for.",
    )
    parser.add_argument(
        "--embodied-scales",
        type=float,
        nargs="+",
        default=[0.60, 0.80, 1.00, 1.20, 1.60],
        help="Scaling factors for embodied emissions.",
    )
    parser.add_argument(
        "--lifetime-scales",
        type=float,
        nargs="+",
        default=[0.75, 1.0, 1.25],
        help="Scaling factors for hardware lifetimes.",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=["heuristic", "global-optimal"],
        choices=["heuristic", "global-optimal"],
        help="Algorithms to run.",
    )
    parser.add_argument(
        "--experiment-root",
        type=str,
        default=DEFAULT_EXPERIMENT_ROOT,
        help="Directory to store experiment runs.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="Random seeds for workload/node generation (mirrors emissions sweeps).",
    )
    parser.add_argument(
        "--figure-dir",
        type=str,
        default=DEFAULT_FIGURE_DIR,
        help="Directory to store plots.",
    )
    parser.add_argument("--workload-config", type=str, default=WORKLOAD_CONFIG)
    parser.add_argument("--generator", type=str, default=WORKLOAD_GENERATOR)
    parser.add_argument("--workloads-dir", type=str, default=WORKLOADS_DIR)
    parser.add_argument("--nodes-file", type=str, default=NODES_FILE)
    parser.add_argument("--ground-truth-forecast", type=str, default=GROUND_TRUTH_FORECAST)
    parser.add_argument(
        "--embodied-mode",
        type=str,
        default="proportional",
        choices=["proportional", "uniform"],
        help="Embodied allocation mode for heuristic and oracle.",
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level for scheduler runs (lower verbosity accelerates runs).",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing placement outputs in experiment_dir instead of re-running schedulers.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip running algorithms and only analyze existing CSV (if provided).",
    )
    parser.add_argument(
        "--runs-csv",
        type=str,
        default=None,
        help="Optional path to an existing runs CSV to analyze.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Do not run schedulers; reuse existing placement logs for all scenarios.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.skip_run and not args.runs_csv:
        raise SystemExit("--skip-run requires --runs-csv to be set.")

    if not args.skip_run:
        df = run_sensitivity(args)
        if df.empty:
            print("No records generated; aborting.")
            return
        os.makedirs(args.experiment_root, exist_ok=True)
        runs_csv = args.runs_csv or os.path.join(
            args.experiment_root, f"embodied_sensitivity_runs_{args.pods}pods.csv"
        )
        df.to_csv(runs_csv, index=False)
        print(f"Saved runs CSV: {runs_csv}")
    else:
        runs_csv = args.runs_csv
        if not os.path.exists(runs_csv):
            raise SystemExit(f"Runs CSV not found: {runs_csv}")
        df = pd.read_csv(runs_csv)

    plot_sensitivity(df, args.figure_dir, args.pods)


if __name__ == "__main__":
    main()
