#!/usr/bin/env python3
"""
Forecast-error sensitivity pipeline for the carbon-aware orchestrator.

Features:
1. Generates noisy carbon-intensity forecasts at configurable MAPE levels.
2. Re-runs vanilla, heuristic, and MILP (global-optimal) schedulers using those forecasts.
3. Re-scores every placement with the ground-truth trace to quantify actual emissions.
4. Produces a "Carbon Savings vs Forecast Error" plot plus CSV summaries.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import sys

# Absolute paths (per instructions)
REPO_ROOT = "/root/carbon-aware-orchestrator"
PKG_ROOT = os.path.join(REPO_ROOT, "pkg", "carbon-aware")
SERVER_PYTHON_DIR = os.path.join(PKG_ROOT, "server-python")
SERVER_MAIN = os.path.join(SERVER_PYTHON_DIR, "main.py")
WORKLOAD_GENERATOR = os.path.join(PKG_ROOT, "infra_workload_gen.py")
WORKLOAD_CONFIG = os.path.join(PKG_ROOT, "infra-workload-config.yaml")
WORKLOADS_DIR = os.path.join(PKG_ROOT, "workloads")
NODES_FILE = os.path.join(PKG_ROOT, "nodes.yaml")
GROUND_TRUTH_FORECAST = os.path.join(SERVER_PYTHON_DIR, "all_forecasts.json")

DEFAULT_EXPERIMENT_ROOT = os.path.join(REPO_ROOT, "experiments", "forecast_error_sensitivity")
DEFAULT_FIGURE_DIR = os.path.join(REPO_ROOT, "figures", "ForecastErrorSensitivity")

CARBON_AWARE_SERVER_PY_PATH = os.path.join(PKG_ROOT, "server-python")
if CARBON_AWARE_SERVER_PY_PATH not in sys.path:
    sys.path.append(CARBON_AWARE_SERVER_PY_PATH)

from carbon_aware.placement_summary import get_total_pods_from_workloads  # noqa: E402
from carbon_aware.forecast_noise import compute_mape, generate_and_save, load_forecasts  # noqa: E402


# ---------------------------------------------------------------------------
# Metadata handling
# ---------------------------------------------------------------------------


def _record_key(record: dict) -> Tuple[str, int, float, int]:
    return (
        record["algorithm"],
        int(record["pods"]),
        float(record["noise_mape"]),
        int(record["seed"]),
    )


class MetadataStore:
    def __init__(self, path: str):
        self.path = path
        self._records: Dict[Tuple[str, int, float, int], dict] = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for record in data.get("runs", []):
                self._records[_record_key(record)] = record
        self._dirty = False

    def get(self, algorithm: str, pods: int, noise_mape: float, seed: int) -> Optional[dict]:
        return self._records.get((algorithm, pods, noise_mape, seed))

    def upsert(self, record: dict) -> None:
        self._records[_record_key(record)] = record
        self._dirty = True

    def to_list(self) -> List[dict]:
        return sorted(self._records.values(), key=lambda r: (r["noise_mape"], r["seed"], r["pods"], r["algorithm"]))

    def save(self) -> None:
        if not self._dirty:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"runs": self.to_list()}, f, indent=2)
        self._dirty = False


# ---------------------------------------------------------------------------
# Workload config helpers
# ---------------------------------------------------------------------------


class WorkloadConfigManager:
    """Context manager that restores the original infra-workload config file on exit."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self._backup_path = None

    def __enter__(self):
        fd, tmp_path = tempfile.mkstemp(prefix="forecast_config_backup_", suffix=".yaml")
        os.close(fd)
        shutil.copyfile(self.config_path, tmp_path)
        self._backup_path = tmp_path
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._backup_path and os.path.exists(self._backup_path):
            shutil.copyfile(self._backup_path, self.config_path)
            os.remove(self._backup_path)


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def ensure_exact_total_strategy(config_path: str) -> None:
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


def regenerate_workloads(generator_path: str) -> None:
    subprocess.run(
        ["python3", generator_path],
        cwd=PKG_ROOT,
        check=True,
    )


# ---------------------------------------------------------------------------
# Forecast noise helpers
# ---------------------------------------------------------------------------


def ensure_noisy_forecast(
    target_mape: float,
    seed: int,
    output_path: str,
    regenerate: bool,
    base_cache: Optional[Dict[str, dict]] = None,
) -> Tuple[str, float]:
    """
    Generate (or reuse) a noisy forecast file.

    Returns tuple (path, achieved_mape).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not regenerate and os.path.exists(output_path):
        base = base_cache or load_forecasts(GROUND_TRUTH_FORECAST)
        noisy = load_forecasts(output_path)
        gt_values = [entry["carbonIntensity"] for region in base.values() for entry in region.get("forecast", [])]
        noisy_values = [entry["carbonIntensity"] for region in noisy.values() for entry in region.get("forecast", [])]
        achieved = compute_mape(gt_values, noisy_values)
        return output_path, achieved

    result = generate_and_save(
        target_mape=target_mape,
        seed=seed,
        output_path=output_path,
        base_path=GROUND_TRUTH_FORECAST,
    )
    return output_path, result.achieved_mape


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
    elif algorithm == "vanilla":
        candidates.extend(
            [
                "vanilla_placement_session_presence.csv",
                "vanilla_placement_session_bind.csv",
                "vanilla_placement_session_fixed.csv",
                "vanilla_placement_session.csv",
                "vanilla_placements.csv",
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
) -> Tuple[str, str]:
    os.makedirs(experiment_dir, exist_ok=True)
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
        "INFO",
    ]
    if algorithm in ("heuristic", "global-optimal"):
        cmd.extend(["--embodied-mode", embodied_mode])
    subprocess.run(cmd, cwd=SERVER_PYTHON_DIR, check=True)
    after = set(os.listdir(experiment_dir))
    new_entries = sorted(after - before)
    if not new_entries:
        raise RuntimeError(f"No session directory created for {algorithm} under {experiment_dir}")
    session_dirs = [os.path.join(experiment_dir, entry) for entry in new_entries if os.path.isdir(os.path.join(experiment_dir, entry))]
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


# ---------------------------------------------------------------------------
# Emissions computation (ground truth scoring)
# ---------------------------------------------------------------------------


def load_nodes_with_forecasts(nodes_file: str, forecasts_file: str):
    with open(nodes_file, "r", encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))
    nodes = []
    for doc in docs:
        if not doc or doc.get("kind") != "Node":
            continue
        nodes.append(doc)
    forecasts = load_forecasts(forecasts_file)
    nodes_dict = {}
    for node in nodes:
        node_id = node.get("metadata", {}).get("name")
        region = (
            node.get("metadata", {})
            .get("labels", {})
            .get("topology.kubernetes.io/region", "")
            .upper()
        )
        power = {
            "idle": float(node.get("metadata", {}).get("annotations", {}).get("hardware.power/idle_watts", 100)),
            "max": float(node.get("metadata", {}).get("annotations", {}).get("hardware.power/max_watts", 300)),
        }
        total_cpu = node.get("status", {}).get("allocatable", {}).get("cpu", "0")
        if isinstance(total_cpu, str) and total_cpu.endswith("m"):
            total_cpu_value = float(total_cpu[:-1]) / 1000.0
        else:
            total_cpu_value = float(total_cpu)
        ram_str = node.get("status", {}).get("allocatable", {}).get("memory", "0")
        if ram_str.endswith("Ki"):
            total_ram = float(ram_str[:-2]) / 1024.0
        elif ram_str.endswith("Mi"):
            total_ram = float(ram_str[:-2])
        elif ram_str.endswith("Gi"):
            total_ram = float(ram_str[:-2]) * 1024.0
        else:
            total_ram = float(ram_str)
        nodes_dict[node_id] = {
            "region": region,
            "power": power,
            "totalCpu": total_cpu_value,
            "totalRam": total_ram,
            "embodied": float(node.get("metadata", {}).get("annotations", {}).get("hardware.carbon/embodied_emissions", 0)) * 1000.0,
            "lifetime_hours": float(node.get("metadata", {}).get("annotations", {}).get("hardware.carbon/lifetime_years", 3)) * 365 * 24,
            "forecast": forecasts.get(region, {}),
        }
    return nodes_dict


def compute_total_emissions_kg(placement_csv: str, nodes_dict: dict, ignore_idle: bool = False) -> float:
    df = pd.read_csv(placement_csv)
    if df.empty:
        return 0.0
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
        embodied_per_hour = node["embodied"] / max(node["lifetime_hours"], 1) if node["lifetime_hours"] else 0.0
        idle_g = 0.0 if ignore_idle else intensity * (idle_w / 1000.0)
        dynamic_g = intensity * (dynamic_k * utilization / 1000.0)
        total_kg += (idle_g + dynamic_g + embodied_per_hour) / 1000.0
    return total_kg


# ---------------------------------------------------------------------------
# Running and analyzing experiments
# ---------------------------------------------------------------------------


def run_experiments(args, metadata: MetadataStore):
    base_forecasts_cache = load_forecasts(GROUND_TRUTH_FORECAST)
    with WorkloadConfigManager(args.workload_config):
        ensure_exact_total_strategy(args.workload_config)
        for pods in args.pod_counts:
            print(f"==> Generating workloads for {pods} pods")
            set_exact_total_pods(args.workload_config, pods)
            regenerate_workloads(args.generator)
            total_pods = get_total_pods_from_workloads(WORKLOADS_DIR)
            print(f"    Confirmed workloads contain {total_pods} pods")
            for noise in args.noise_levels:
                for seed in args.seeds:
                    noise_dir = os.path.join(args.experiment_root, f"pods_{pods}", f"mape_{noise:.1f}", f"seed_{seed}")
                    os.makedirs(noise_dir, exist_ok=True)
                    noise_file = os.path.join(noise_dir, "noisy_forecasts.json")
                    forecast_path, achieved_mape = ensure_noisy_forecast(
                        target_mape=noise,
                        seed=seed,
                        output_path=noise_file,
                        regenerate=args.regenerate_noise,
                        base_cache=base_forecasts_cache,
                    )
                    print(f"    Noise {noise}% (seed {seed}) -> {achieved_mape:.2f}% MAPE [{forecast_path}]")
                    for algorithm in args.algorithms:
                        existing = metadata.get(algorithm, pods, noise, seed)
                        if existing and not args.force_rerun and os.path.exists(existing.get("placement_csv", "")):
                            print(f"      Skipping {algorithm}: placement already recorded ({existing['placement_csv']})")
                            continue
                        print(f"      Running {algorithm}...")
                        session_dir, placement_csv = run_algorithm(
                            algorithm=algorithm,
                            workloads_dir=args.workloads_dir,
                            nodes_file=args.nodes_file,
                            forecasts_file=forecast_path,
                            experiment_dir=noise_dir,
                        )
                        timestamp_utc = datetime.now(timezone.utc).isoformat()
                        record = {
                            "algorithm": algorithm,
                            "pods": pods,
                            "noise_mape": noise,
                            "achieved_mape": achieved_mape,
                            "seed": seed,
                            "forecast_file": forecast_path,
                            "session_dir": session_dir,
                            "placement_csv": placement_csv,
                            "timestamp": timestamp_utc,
                        }
                        metadata.upsert(record)
                        metadata.save()


def analyze_results(args, metadata: MetadataStore):
    records = metadata.to_list()
    if not records:
        print("⚠️ No metadata available; skipping analysis.")
        return
    noise_values = sorted({float(r["noise_mape"]) for r in records})
    seed_values = sorted({int(r["seed"]) for r in records})
    pod_values = sorted({int(r["pods"]) for r in records})
    algorithms_present = sorted({r["algorithm"] for r in records})
    nodes = load_nodes_with_forecasts(args.nodes_file, args.ground_truth_forecast)
    per_run_emissions: Dict[Tuple[str, int, float, int], float] = {}
    for record in records:
        csv_path = record.get("placement_csv")
        if not csv_path or not os.path.exists(csv_path):
            print(f"⚠️ Missing placement CSV for {record}")
            continue
        total_kg = compute_total_emissions_kg(csv_path, nodes)
        per_run_emissions[_record_key(record)] = total_kg

    savings_rows = []
    baseline_algorithm = "vanilla"
    for noise in noise_values:
        for seed in seed_values:
            for pods in pod_values:
                baseline_key = ("vanilla", pods, noise, seed)
                baseline = per_run_emissions.get(baseline_key)
                if baseline is None or baseline == 0:
                    continue
                for algorithm in algorithms_present:
                    if algorithm == baseline_algorithm:
                        continue
                    key = (algorithm, pods, noise, seed)
                    algo_val = per_run_emissions.get(key)
                    if algo_val is None:
                        continue
                    savings = (baseline - algo_val) / baseline * 100.0
                    savings_rows.append(
                        {
                            "noise_mape": noise,
                            "seed": seed,
                            "pods": pods,
                            "algorithm": algorithm,
                            "baseline_kg": baseline,
                            "actual_kg": algo_val,
                            "carbon_savings_pct": savings,
                        }
                    )

    if not savings_rows:
        print("⚠️ No savings data computed.")
        return

    runs_df = pd.DataFrame(savings_rows)
    summary_csv = os.path.join(args.experiment_root, "forecast_error_sensitivity_runs.csv")
    runs_df.to_csv(summary_csv, index=False)
    print(f"Saved per-run summary: {summary_csv}")

    agg = (
        runs_df.groupby(["noise_mape", "algorithm"])
        .agg(
            mean_savings_pct=("carbon_savings_pct", "mean"),
            std_savings_pct=("carbon_savings_pct", "std"),
            count=("carbon_savings_pct", "count"),
        )
        .reset_index()
    )
    agg["std_savings_pct"] = agg["std_savings_pct"].fillna(0.0)
    summary_agg_csv = os.path.join(args.experiment_root, "forecast_error_sensitivity_summary.csv")
    agg.to_csv(summary_agg_csv, index=False)
    print(f"Saved aggregate summary: {summary_agg_csv}")

    plot_carbon_savings(runs_df, agg, args.figure_dir)


def plot_carbon_savings(runs_df: pd.DataFrame, summary_df: pd.DataFrame, figure_dir: str) -> None:
    os.makedirs(figure_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))

    # Per-pod lines for heuristic: mean savings vs MAPE, averaged over seeds
    heur_df = runs_df[runs_df["algorithm"] == "heuristic"]
    all_pods = sorted(heur_df["pods"].unique())
    # Pick at most 4 representative pod counts spanning the range
    if len(all_pods) <= 4:
        pod_values = all_pods
    else:
        idxs = np.linspace(0, len(all_pods) - 1, 4, dtype=int)
        pod_values = [all_pods[i] for i in idxs]
    x_values = sorted(heur_df["noise_mape"].unique())
    cmap = plt.get_cmap("tab10")

    for idx, pods in enumerate(pod_values):
        sub = heur_df[heur_df["pods"] == pods]
        if sub.empty:
            continue
        grouped = sub.groupby("noise_mape")["carbon_savings_pct"].mean().reset_index()
        x = grouped["noise_mape"].values
        y = grouped["carbon_savings_pct"].values
        color = cmap(idx % 10)
        label = f"TotEm {int(pods)} pods"
        plt.plot(x, y, marker="o", linewidth=1.8, color=color, alpha=0.85, label=label)

    # Optional: overlay global-optimal mean as a thick reference line
    oracle_df = summary_df[summary_df["algorithm"] == "global-optimal"]
    if not oracle_df.empty:
        x_o = oracle_df["noise_mape"].values
        y_o = oracle_df["mean_savings_pct"].values
        plt.plot(
            x_o,
            y_o,
            color="black",
            linestyle="--",
            linewidth=2.5,
            label="Oracle (mean over pods)",
        )
    plt.xlabel("Forecast Error (MAPE %)")
    plt.ylabel("Carbon Savings vs Carbon-Agnostic (%)")
    plt.title("TotEm Robustness to Forecast Error\nPer-Pod-Count Savings vs Carbon-Agnostic")
    plt.grid(True, linestyle="--", alpha=0.4)

    # Use only the actual noise levels as x-ticks (no 2.5, 7.5, etc.)
    xticks = sorted(heur_df["noise_mape"].unique())
    plt.xticks(xticks, [f"{int(x)}" if float(x).is_integer() else f"{x:g}" for x in xticks])

    # Place legend inside the plot in the upper left corner
    plt.legend(
        fontsize=8,
        ncol=1,
        loc="upper left",
        frameon=True,
    )

    plt.tight_layout()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(figure_dir, f"carbon_savings_vs_forecast_error_{timestamp}.pdf")
    png_path = os.path.join(figure_dir, f"carbon_savings_vs_forecast_error_{timestamp}.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"Saved plot: {pdf_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Forecast-error sensitivity driver.")
    parser.add_argument("--pod-counts", type=int, nargs="+", default=[100, 140, 180], help="Exact pod counts to regenerate workloads for.")
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 5.0, 10.0, 15.0, 20.0], help="MAPE percentages to test.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds for noise generation.")
    parser.add_argument("--algorithms", type=str, nargs="+", default=["vanilla", "heuristic", "global-optimal"], choices=["vanilla", "heuristic", "global-optimal"], help="Algorithms to run.")
    parser.add_argument("--experiment-root", type=str, default=DEFAULT_EXPERIMENT_ROOT, help="Directory to store experiment runs.")
    parser.add_argument("--figure-dir", type=str, default=DEFAULT_FIGURE_DIR, help="Directory to store plots.")
    parser.add_argument("--workload-config", type=str, default=WORKLOAD_CONFIG)
    parser.add_argument("--generator", type=str, default=WORKLOAD_GENERATOR)
    parser.add_argument("--workloads-dir", type=str, default=WORKLOADS_DIR)
    parser.add_argument("--nodes-file", type=str, default=NODES_FILE)
    parser.add_argument("--ground-truth-forecast", type=str, default=GROUND_TRUTH_FORECAST)
    parser.add_argument("--metadata-file", type=str, default=os.path.join(DEFAULT_EXPERIMENT_ROOT, "metadata.json"))
    parser.add_argument("--regenerate-noise", action="store_true", help="Force regeneration of noisy forecast files even if cached.")
    parser.add_argument("--force-rerun", action="store_true", help="Re-run algorithms even if metadata already exists.")
    parser.add_argument("--skip-run", action="store_true", help="Skip the run phase and only analyze existing metadata.")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis/plotting.")
    return parser.parse_args()


def main():
    args = parse_args()
    if "vanilla" not in args.algorithms:
        print("⚠️ Vanilla baseline required for savings; adding it automatically.")
        args.algorithms.append("vanilla")
    metadata = MetadataStore(args.metadata_file)
    if not args.skip_run:
        run_experiments(args, metadata)
    else:
        print("Skipping run phase.")
    if not args.skip_analysis:
        analyze_results(args, metadata)
    else:
        print("Skipping analysis phase.")


if __name__ == "__main__":
    main()

