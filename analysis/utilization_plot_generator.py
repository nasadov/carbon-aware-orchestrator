#!/usr/bin/env python3
"""
Cluster Utilization Plot Generator

Generates utilization vs. pod count comparisons for Vanilla, Heuristic,
and Global-Optimal by parsing experiment placement CSVs and cluster
capacity from nodes.yaml.

For CPU: avg_over_slots( sum_active_pods(cpu_request_cores) / total_cores ).
For Memory: avg_over_slots( sum_active_pods(mem_request_bytes) / total_bytes ).

Default: proportional-only (plus vanilla). Use flags to change selection:
  --uniform  Plot only uniform (plus vanilla)
  --both     Plot both proportional and uniform (plus vanilla)

Outputs three figures under figures/Utilization:
  - CPU-only plot
  - Memory-only plot
  - Combined two-panel (CPU, Memory)
"""

import csv
import math
import os
import re
import random
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import argparse
import yaml


EXPERIMENTS_ROOT = "/root/carbon-aware-orchestrator/experiments"
NODES_YAML_PATH = "/root/carbon-aware-orchestrator/pkg/carbon-aware/nodes.yaml"
WORKLOAD_CONFIG_PATH = "/root/carbon-aware-orchestrator/pkg/carbon-aware/infra-workload-config.yaml"


def parse_cpu_to_cores(cpu_str: str) -> float:
    """Parse Kubernetes CPU quantity into cores.

    Examples:
      - "2000m" -> 2.0
      - "500m"  -> 0.5
      - "2"     -> 2.0
      - "0.5"   -> 0.5
    """
    if cpu_str is None:
        return 0.0
    s = str(cpu_str).strip()
    if not s:
        return 0.0
    try:
        if s.endswith("m"):
            milli = float(s[:-1])
            return milli / 1000.0
        return float(s)
    except Exception:
        return 0.0


def parse_memory_to_bytes(mem_str: str) -> float:
    """Parse Kubernetes memory quantity into bytes.

    Supports Ki, Mi, Gi, Ti suffixes and plain numbers (bytes).
    """
    if mem_str is None:
        return 0.0
    s = str(mem_str).strip()
    if not s:
        return 0.0
    try:
        # Normalize case
        s_up = s.upper()
        # Binary suffixes
        for suf, mul in (
            ("KIB", 1024.0),
            ("MIB", 1024.0**2),
            ("GIB", 1024.0**3),
            ("TIB", 1024.0**4),
            ("KI", 1024.0),
            ("MI", 1024.0**2),
            ("GI", 1024.0**3),
            ("TI", 1024.0**4),
        ):
            if s_up.endswith(suf):
                return float(s_up[: -len(suf)]) * mul

        # Decimal suffixes
        for suf, mul in (
            ("KB", 1000.0),
            ("MB", 1000.0**2),
            ("GB", 1000.0**3),
            ("TB", 1000.0**4),
        ):
            if s_up.endswith(suf):
                return float(s_up[: -len(suf)]) * mul

        # No suffix -> our CSVs for heuristic/global-optimal use MiB numerics
        # Treat plain numbers as MiB to match those placements
        return float(s_up) * (1024.0**2)
    except Exception:
        return 0.0


def _iter_allocatable_value_lines(nodes_yaml_path: str, resource_key: str):
    """Yield CPU capacity strings found under status.allocatable.cpu in nodes.yaml.

    Lightweight parser to avoid external YAML deps, assumes consistent indentation
    as produced by our generator.
    """
    if not os.path.exists(nodes_yaml_path):
        return

    try:
        with open(nodes_yaml_path, "r") as fh:
            lines = fh.readlines()
    except Exception:
        return

    in_alloc_block = False
    alloc_indent = None

    for raw in lines:
        line = raw.rstrip("\n")
        # Determine indentation
        leading_spaces = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        # Reset block if indent dedents beyond current alloc block
        if in_alloc_block and alloc_indent is not None and leading_spaces <= alloc_indent:
            in_alloc_block = False
            alloc_indent = None

        # Enter allocatable block
        if stripped.startswith("allocatable:"):
            in_alloc_block = True
            alloc_indent = leading_spaces
            continue

        # Within allocatable block, look for desired resource value
        if in_alloc_block and re.match(rf"^{re.escape(resource_key)}:\s*", stripped):
            # Value after colon
            parts = stripped.split(":", 1)
            if len(parts) == 2:
                yield parts[1].strip().strip("'")


def parse_total_cluster_cores(nodes_yaml_path: str) -> float:
    total_cores = 0.0
    for cpu_str in _iter_allocatable_value_lines(nodes_yaml_path, "cpu"):
        total_cores += parse_cpu_to_cores(cpu_str)
    return total_cores


def parse_total_cluster_memory_bytes(nodes_yaml_path: str) -> float:
    total_bytes = 0.0
    for mem_str in _iter_allocatable_value_lines(nodes_yaml_path, "memory"):
        total_bytes += parse_memory_to_bytes(mem_str)
    return total_bytes


def _simulate_total_core_hours_exact_total(
    pods_count: int,
    workload_cfg: Dict,
) -> float:
    """
    Deterministically simulate total requested CPU-core*hours for a workload
    with generation_strategy == 'exact_total' and exact_total_pods == pods_count,
    using the same randomization logic as infra_workload_gen.py but without
    writing any YAML files.
    """
    wl = workload_cfg or {}
    durations = wl.get("durations", [1, 3, 6])
    duration_assignment_method = wl.get("duration_assignment_method", "cycle")
    cpu_options = wl.get("cpu_options", ["2000m", "1000m", "500m", "250m", "100m"])
    random_seed = wl.get("random_seed", 42)

    # Mirror generator seeding
    random.seed(random_seed)
    np.random.seed(random_seed)

    total_services = int(pods_count)
    duration_counter = 0
    total_core_hours = 0.0

    for _ in range(total_services):
        cpu_req = random.choice(cpu_options)
        if duration_assignment_method == "cycle":
            duration_hours = durations[duration_counter % len(durations)]
            duration_counter += 1
        else:
            duration_hours = random.choice(durations)

        cpu_cores = parse_cpu_to_cores(cpu_req)
        if cpu_cores <= 0 or duration_hours <= 0:
            continue
        total_core_hours += cpu_cores * float(duration_hours)

    return total_core_hours


def compute_pods_to_24h_cpu_utilization(
    experiments_root: str,
    nodes_yaml_path: str,
    prefer_algo: str = "vanilla",  # kept for backward compatibility; unused now
    horizon_hours: float = 24.0,
) -> Dict[int, float]:
    """
    Compute a mapping from pod count -> average CPU utilization over a fixed horizon,
    based on requested capacity (workload), not on placements.

    For each pod count P that appears under experiments_root, we:
      - Simulate the exact_total workload with P pods using the workload
        configuration from infra-workload-config.yaml.
      - Sum cpu_request_cores * duration_hours over all pods.
      - Divide by (total_cluster_cores * horizon_hours) and convert to percent.

    Because the generator is deterministic with a fixed seed and exact_total,
    total requested CPU-core*hours is monotone in P, so utilization is monotone.
    """
    total_cores = parse_total_cluster_cores(nodes_yaml_path)
    if total_cores <= 0 or horizon_hours <= 0:
        return {}

    if not os.path.isdir(experiments_root):
        return {}

    # Discover pod counts present under this experiments_root
    discovered = list(_discover_experiments(experiments_root, selection="proportional"))
    pod_counts = sorted({pods for (_, pods, _, _) in discovered}) if discovered else []
    if not pod_counts:
        return {}

    # Load workload configuration (same one used by infra_workload_gen.py)
    workload_cfg: Dict = {}
    try:
        with open(WORKLOAD_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        workload_cfg = cfg.get("workload", {})
    except Exception:
        workload_cfg = {}

    pods_to_util: Dict[int, float] = {}
    for pods_count in pod_counts:
        total_core_hours = _simulate_total_core_hours_exact_total(pods_count, workload_cfg)
        if total_core_hours <= 0:
            continue
        util_percent = (total_core_hours / (total_cores * horizon_hours)) * 100.0
        pods_to_util[pods_count] = float(util_percent)

    return pods_to_util


def _placement_csv_candidates(algo_key: str) -> List[str]:
    base = algo_key.split("-", 1)[0]
    suffix = algo_key[len(base):]
    if base == "global-optimal":
        return ["global_optimal_placements_session.csv"]
    if base == "heuristic":
        # Prefer mode-specific names if the algo key encodes them
        if suffix == "-proportional":
            return [
                "heuristic_prop_placements_session.csv",
                "heuristic_placements_session.csv",
            ]
        if suffix == "-uniform":
            return [
                "heuristic_uniform_placements_session.csv",
                "heuristic_placements_session.csv",
            ]
        return [
            "heuristic_prop_placements_session.csv",
            "heuristic_uniform_placements_session.csv",
            "heuristic_placements_session.csv",
        ]
    if base == "vanilla":
        return [
            "vanilla_placement_session.csv",
            "vanilla_placement_session_bind.csv",
            "vanilla_placement_session_presence.csv",
            "vanilla_placement_session_fixed.csv",
            "vanilla_placement_session.csv",
            "vanilla_placements.csv",
        ]
    # Fallback for unknown naming
    return [
        "global_optimal_placements_session.csv",
        "heuristic_placements_session.csv",
        "vanilla_placement_session.csv",
        "vanilla_placement_session_bind.csv",
        "vanilla_placement_session_presence.csv",
        "vanilla_placement_session_fixed.csv",
        "vanilla_placements.csv",
    ]


def _find_placement_csv(entry_path: str, algo_key: str) -> Optional[str]:
    for name in _placement_csv_candidates(algo_key):
        p = os.path.join(entry_path, name)
        if os.path.exists(p):
            return p
    return None


def _parse_experiment_name(entry: str) -> Optional[Tuple[str, int]]:
    """Return (algo_key, pods_count) from directory name or None if no match."""
    m = re.match(r"^(?P<algo>[^_]+?)(?:_(?P<mode>op|proportional|uniform))?_(?P<pods>\d+)pods_", entry)
    if not m:
        return None
    algo_key = m.group("algo")
    mode = m.group("mode") or ""
    if mode:
        algo_key = f"{algo_key}-{mode}"
    try:
        pods_count = int(m.group("pods"))
    except Exception:
        return None
    return algo_key, pods_count


def _should_include_algo(algo_key: str, selection: str) -> bool:
    """Return True if this algo key should be included under selection.

    selection in { 'proportional', 'uniform', 'both' }
    """
    if algo_key == 'vanilla':
        return True
    if selection == 'both':
        return algo_key.endswith('-proportional') or algo_key.endswith('-uniform')
    if selection == 'uniform':
        return algo_key.endswith('-uniform')
    return algo_key.endswith('-proportional')


def _compute_run_avg_cpu_util_percent(csv_path: str, total_cluster_cores: float) -> Optional[float]:
    if total_cluster_cores <= 0:
        return None
    segments: List[Tuple[float, float, float]] = []  # (start_slot, end_slot, cores)

    try:
        with open(csv_path, "r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                cpu_cores = parse_cpu_to_cores(row.get("cpu_request", "0"))
                try:
                    start_slot = float(row.get("start_slot", 0.0))
                except Exception:
                    start_slot = 0.0
                try:
                    duration = float(row.get("duration", 0.0))
                except Exception:
                    duration = 0.0
                if cpu_cores <= 0 or duration <= 0:
                    continue
                start = max(0.0, start_slot)
                end = max(start, start + duration)
                segments.append((start, end, cpu_cores))
    except Exception:
        return None

    if not segments:
        return None

    max_end = max(end for (_, end, _) in segments)
    if max_end <= 0:
        return None
    horizon_slots = int(math.ceil(max_end))
    if horizon_slots <= 0:
        return None

    # Discrete per-slot accumulation
    utilizations: List[float] = []
    for s in range(horizon_slots):
        used = 0.0
        for (st, en, cores) in segments:
            if st <= s < en:
                used += cores
        utilizations.append((used / total_cluster_cores) * 100.0)

    if not utilizations:
        return None
    return float(np.mean(utilizations))


def _compute_run_avg_mem_util_percent(csv_path: str, total_cluster_bytes: float) -> Optional[float]:
    if total_cluster_bytes <= 0:
        return None
    segments: List[Tuple[float, float, float]] = []  # (start_slot, end_slot, bytes)

    try:
        with open(csv_path, "r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                # Try several possible memory request column names
                mem_val = None
                for k in ("ram_request", "memory_request", "mem_request", "ram", "memory"):
                    if k in row and row.get(k):
                        mem_val = row.get(k)
                        break
                bytes_req = parse_memory_to_bytes(mem_val or "0")
                try:
                    start_slot = float(row.get("start_slot", 0.0))
                except Exception:
                    start_slot = 0.0
                try:
                    duration = float(row.get("duration", 0.0))
                except Exception:
                    duration = 0.0
                if bytes_req <= 0 or duration <= 0:
                    continue
                start = max(0.0, start_slot)
                end = max(start, start + duration)
                segments.append((start, end, bytes_req))
    except Exception:
        return None

    if not segments:
        return None

    max_end = max(end for (_, end, _) in segments)
    if max_end <= 0:
        return None
    horizon_slots = int(math.ceil(max_end))
    if horizon_slots <= 0:
        return None

    utilizations: List[float] = []
    for s in range(horizon_slots):
        used = 0.0
        for (st, en, bytes_req) in segments:
            if st <= s < en:
                used += bytes_req
        utilizations.append((used / total_cluster_bytes) * 100.0)

    if not utilizations:
        return None
    return float(np.mean(utilizations))


def _discover_experiments(experiments_root: str, selection: str):
    """Yield (algo_key, pods_count, dir_path, mtime) for matching experiments."""
    if not os.path.isdir(experiments_root):
        return
    for current_root, dirnames, _ in os.walk(experiments_root):
        dirnames.sort()
        next_level = []
        for entry in dirnames:
            if entry.lower().startswith("archive"):
                continue
            entry_path = os.path.join(current_root, entry)
            parsed = _parse_experiment_name(entry)
            if not parsed:
                next_level.append(entry)
                continue
            algo_key, pods_count = parsed
            if not _should_include_algo(algo_key, selection):
                continue
            try:
                mtime = os.path.getmtime(entry_path)
            except Exception:
                mtime = 0.0
            yield algo_key, pods_count, entry_path, mtime
        dirnames[:] = next_level


def parse_utilizations_from_experiments(experiments_root: str, nodes_yaml_path: str, selection: str = 'proportional', include_all: bool = False):
    """Return two mappings (cpu_results, mem_results): algo -> pods -> list of utilization percent per run.

    If include_all is False, only the latest directory per (algo_key, pods) is used.
    """
    cpu_results: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    mem_results: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))

    total_cores = parse_total_cluster_cores(nodes_yaml_path)
    total_bytes = parse_total_cluster_memory_bytes(nodes_yaml_path)
    if total_cores <= 0:
        print(f"⚠️ Total cluster cores parsed as {total_cores} from {nodes_yaml_path}")
    if total_bytes <= 0:
        print(f"⚠️ Total cluster memory bytes parsed as {total_bytes} from {nodes_yaml_path}")

    if not os.path.isdir(experiments_root):
        print(f"⚠️ Experiments directory not found: {experiments_root}")
        return cpu_results, mem_results

    discovered = list(_discover_experiments(experiments_root, selection))
    if not include_all:
        latest_map: Dict[Tuple[str, int], Tuple[str, float]] = {}
        for algo_key, pods_count, entry_path, mtime in discovered:
            key = (algo_key, pods_count)
            prev = latest_map.get(key)
            if prev is None or mtime > prev[1]:
                latest_map[key] = (entry_path, mtime)
        selected = [(a, p, path) for (a, p), (path, _) in latest_map.items()]
    else:
        selected = [(a, p, path) for (a, p, path, _) in discovered]

    for algo_key, pods_count, entry_path in selected:
        csv_path = _find_placement_csv(entry_path, algo_key)
        if not csv_path:
            continue
        cpu_util = _compute_run_avg_cpu_util_percent(csv_path, total_cores)
        if cpu_util is not None:
            cpu_results[algo_key][pods_count].append(cpu_util)
        mem_util = _compute_run_avg_mem_util_percent(csv_path, total_bytes)
        if mem_util is not None:
            mem_results[algo_key][pods_count].append(mem_util)

    return cpu_results, mem_results


def _aggregate_results(results):
    mean_vals: Dict[str, Dict[int, float]] = defaultdict(dict)
    std_errs: Dict[str, Dict[int, float]] = defaultdict(dict)

    pod_counts = set()
    for algo, by_pods in results.items():
        pod_counts.update(by_pods.keys())
    pod_counts_sorted = sorted(pod_counts)

    known_order = [
        "vanilla",
        "heuristic-proportional", "heuristic-uniform",
        "global-optimal-proportional", "global-optimal-uniform",
        "heuristic", "global-optimal",
    ]
    algos_present = list(results.keys())
    algos_order = [a for a in known_order if a in algos_present] + [a for a in algos_present if a not in known_order]

    for algo, by_pods in results.items():
        for pods, vals in by_pods.items():
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            mean_vals[algo][pods] = float(np.mean(arr))
            if arr.size > 1:
                std_errs[algo][pods] = float(np.std(arr, ddof=1))
            else:
                std_errs[algo][pods] = 0.0

    return algos_order, pod_counts_sorted, mean_vals, std_errs


def _plot_utilization(ax, algos_order, pod_counts_sorted, mean_vals, std_errs, ylabel: str, legend_loc: str = 'lower right'):
    colors = {
        'vanilla': '#2ca02c',
        'heuristic': '#ff7f0e',
        'heuristic-proportional': '#ff7f0e',
        'heuristic-uniform': '#ffbb78',
        'global-optimal': '#d62728',
        'global-optimal-proportional': '#d62728',
        'global-optimal-uniform': '#ff9896',
    }
    markers = {
        'vanilla': '^',
        'heuristic': 's',
        'heuristic-proportional': 's',
        'heuristic-uniform': 'D',
        'global-optimal': 'o',
        'global-optimal-proportional': 'o',
        'global-optimal-uniform': '^',
    }
    labels = {
        'vanilla': 'Carbon-Agnostic (Baseline)',
        'heuristic': 'TotEm (Carbon-Aware)',
        'heuristic-proportional': 'TotEm (Proportional)',
        'heuristic-uniform': 'TotEm (Uniform)',
        'global-optimal': 'Oracle',
        'global-optimal-proportional': 'Oracle (Proportional)',
        'global-optimal-uniform': 'Oracle (Uniform)',
    }

    for algo in algos_order:
        color = colors.get(algo, '#1f77b4')
        marker = markers.get(algo, 'o')
        label = labels.get(algo, algo.replace('-', ' ').title())

        x_vals, y_vals, y_errs = [], [], []
        for pods in pod_counts_sorted:
            if pods in mean_vals[algo]:
                x_vals.append(pods)
                y_vals.append(mean_vals[algo][pods])
                y_errs.append(std_errs[algo][pods])

        if not x_vals:
            continue

        if any(e > 0 for e in y_errs):
            ax.errorbar(
                x_vals, y_vals, yerr=y_errs,
                marker=marker, color=color, label=label,
                linewidth=3, markersize=10, alpha=0.9, capsize=4,
            )
        else:
            ax.plot(
                x_vals, y_vals,
                marker=marker, color=color, label=label,
                linewidth=3, markersize=10, alpha=0.9,
            )

    ax.set_xlabel('Number of Pods to Schedule', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    if pod_counts_sorted:
        ax.set_xlim(min(pod_counts_sorted) - 5, max(pod_counts_sorted) + 10)
    ax.set_ylim(0, 105)
    ax.set_xticks(pod_counts_sorted)
    ax.set_yticks(list(range(0, 101, 10)))
    ax.legend(fontsize=12, loc=legend_loc, framealpha=0.9, shadow=True, fancybox=True)


def create_utilization_plots(selection: str = 'proportional', include_all: bool = False):
    cpu_results, mem_results = parse_utilizations_from_experiments(EXPERIMENTS_ROOT, NODES_YAML_PATH, selection, include_all)
    any_results = any(cpu_results.values()) or any(mem_results.values())
    if not any_results:
        print(f"❌ No experiment results found in: {EXPERIMENTS_ROOT}")
        return

    cpu_algos_order, cpu_pods_sorted, cpu_means, cpu_errs = _aggregate_results(cpu_results)
    mem_algos_order, mem_pods_sorted, mem_means, mem_errs = _aggregate_results(mem_results)

    output_dir = "/root/carbon-aware-orchestrator/figures/Utilization"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CPU-only figure
    fig_cpu, ax_cpu = plt.subplots(figsize=(12, 8))
    _plot_utilization(ax_cpu, cpu_algos_order, cpu_pods_sorted, cpu_means, cpu_errs, 'Average Cluster CPU Utilization (%)', legend_loc='lower right')
    fig_cpu.suptitle('CPU Utilization vs. Pod Count\nComparison of Three Algorithms', fontsize=16, fontweight='bold', y=0.98)
    fig_cpu.tight_layout(rect=[0, 0.03, 1, 0.95])
    cpu_path = f"{output_dir}/cpu_utilization_vs_pods_{timestamp}.pdf"
    fig_cpu.savefig(cpu_path, bbox_inches='tight', facecolor='white')
    plt.close(fig_cpu)

    # Memory-only figure
    fig_mem, ax_mem = plt.subplots(figsize=(12, 8))
    _plot_utilization(ax_mem, mem_algos_order, mem_pods_sorted, mem_means, mem_errs, 'Average Cluster Memory Utilization (%)', legend_loc='lower right')
    fig_mem.suptitle('Memory Utilization vs. Pod Count\nComparison of Three Algorithms', fontsize=16, fontweight='bold', y=0.98)
    fig_mem.tight_layout(rect=[0, 0.03, 1, 0.95])
    mem_path = f"{output_dir}/memory_utilization_vs_pods_{timestamp}.pdf"
    fig_mem.savefig(mem_path, bbox_inches='tight', facecolor='white')
    plt.close(fig_mem)

    # Combined two-panel figure (side-by-side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    _plot_utilization(ax1, cpu_algos_order, cpu_pods_sorted, cpu_means, cpu_errs, 'CPU Utilization (%)', legend_loc='lower right')
    _plot_utilization(ax2, mem_algos_order, mem_pods_sorted, mem_means, mem_errs, 'Memory Utilization (%)', legend_loc='lower right')
    fig.suptitle('Cluster Utilization vs. Pod Count\nCPU and Memory', fontsize=16, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    combined_path = f"{output_dir}/utilization_cpu_mem_combined_{timestamp}.pdf"
    fig.savefig(combined_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print("✅ UTILIZATION PLOTS GENERATED!")
    print(f"📊 CPU PDF: {cpu_path}")
    print(f"📊 MEM PDF: {mem_path}")
    print(f"📊 COMBINED PDF: {combined_path}")

    print("\n📈 CPU RESULTS SUMMARY:")
    for algo in cpu_algos_order:
        vals_list = [cpu_means[algo][p] for p in cpu_pods_sorted if p in cpu_means[algo]]
        if not vals_list:
            continue
        print(f"   {algo}: avg={np.mean(vals_list):.1f}% range={min(vals_list):.1f}%-{max(vals_list):.1f}%")

    print("\n📈 MEMORY RESULTS SUMMARY:")
    for algo in mem_algos_order:
        vals_list = [mem_means[algo][p] for p in mem_pods_sorted if p in mem_means[algo]]
        if not vals_list:
            continue
        print(f"   {algo}: avg={np.mean(vals_list):.1f}% range={min(vals_list):.1f}%-{max(vals_list):.1f}%")


if __name__ == "__main__":
    print("🚀 UTILIZATION PLOT GENERATOR")
    print("=" * 50)
    parser = argparse.ArgumentParser(description="Cluster Utilization Plot Generator")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--uniform', action='store_true', help='Plot uniform embodied allocation only (plus vanilla)')
    group.add_argument('--both', action='store_true', help='Plot both proportional and uniform (plus vanilla)')
    parser.add_argument('--all', action='store_true', help='Include all experiments (default: latest-only per algo and pod count)')
    args = parser.parse_args()
    selection = 'both' if args.both else ('uniform' if args.uniform else 'proportional')
    create_utilization_plots(selection, include_all=args.all)
