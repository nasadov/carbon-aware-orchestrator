#!/usr/bin/env python3
"""
Emissions vs Pod Count (Common Pods Only)

Generates plots like the standard emissions_vs_pods_plot_generator, but only
counts emissions from pods that were scheduled by ALL THREE algorithms
(vanilla, heuristic, global-optimal) for a given pod count experiment.

Usage:
    python analysis/emissions_vs_pods_common_pods_plot_generator.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import re
import csv
from collections import defaultdict
import json
import yaml
import sys

import utilization_plot_generator as util_mod

# Ensure carbon_aware modules are importable
CARBON_AWARE_SERVER_PY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pkg", "carbon-aware", "server-python"
)
sys.path.append(CARBON_AWARE_SERVER_PY_PATH)

try:
    from carbon_aware.models import CarbonAwareFlavour
except Exception as e:
    print(f"⚠️ Warning: could not import carbon_aware modules: {e}")
    CarbonAwareFlavour = None

EXPERIMENTS_ROOT = "/root/carbon-aware-orchestrator/experiments"
NODES_FILE = "/root/carbon-aware-orchestrator/pkg/carbon-aware/nodes.yaml"
FORECASTS_FILE = "/root/carbon-aware-orchestrator/pkg/carbon-aware/server-python/all_forecasts.json"


def _load_nodes_from_yaml(nodes_file: str):
    nodes = []
    try:
        with open(nodes_file, 'r') as f:
            docs = list(yaml.safe_load_all(f))
        for doc in docs:
            if not doc or doc.get('kind') != 'Node':
                continue
            metadata = doc.get('metadata', {})
            annotations = metadata.get('annotations', {})
            status = doc.get('status', {})
            alloc = status.get('allocatable', {})

            node_id = metadata.get('name', 'unknown')
            region = 'DE'
            if '-' in node_id:
                parts = node_id.split('-')
                if len(parts) >= 5 and parts[2] == 'it' and parts[3] == 'no':
                    region = 'IT-NO'
                elif len(parts) >= 3:
                    region = parts[2].upper()

            cpu_str = alloc.get('cpu', '0')
            if cpu_str.endswith('m'):
                total_cpu = float(cpu_str[:-1]) / 1000.0
            else:
                total_cpu = float(cpu_str)

            ram_str = alloc.get('memory', '0')
            if 'Ki' in ram_str:
                total_ram = float(ram_str.replace('Ki', '')) / 1024.0
            elif 'Mi' in ram_str:
                total_ram = float(ram_str.replace('Mi', ''))
            elif 'Gi' in ram_str:
                total_ram = float(ram_str.replace('Gi', '')) * 1024.0
            else:
                total_ram = float(ram_str)

            embodied_carbon = float(annotations.get("hardware.carbon/embodied_emissions", "0")) * 1000.0
            lifetime_years = float(annotations.get("hardware.carbon/lifetime_years", "3"))
            lifetime_hours = lifetime_years * 365 * 24

            power_settings = {
                "idle": float(annotations.get("hardware.power/idle_watts", "100")),
                # active no longer used in calculations; keep for backward compatibility if present
                "active": float(annotations.get("hardware.power/active_watts", annotations.get("hardware.power/idle_watts", "200"))),
                "max": float(annotations.get("hardware.power/max_watts", "300")),
            }

            if CarbonAwareFlavour is None:
                node = type("Node", (), {})()
                node.id = node_id
                node.embodiedCarbon = embodied_carbon
                node.lifetime = lifetime_hours
                node.totalCpu = total_cpu
                node.totalRam = total_ram
                node.totalStorage = 0
                node.forecast = {}
                node.power = power_settings
                node.region = region
            else:
                node = CarbonAwareFlavour(
                    id=node_id,
                    embodiedCarbon=embodied_carbon,
                    lifetime=lifetime_hours,
                    totalCpu=total_cpu,
                    totalRam=total_ram,
                    totalStorage=1000 * 1024 * 1024,
                    forecast={},
                    power=power_settings,
                )
                node.region = region

            nodes.append(node)
        return nodes
    except Exception as e:
        print(f"❌ Error loading nodes from {nodes_file}: {e}")
        return []


def _load_carbon_forecasts(forecasts_file: str):
    try:
        with open(forecasts_file, 'r') as f:
            data = json.load(f)
        forecasts = {}
        for region, region_data in data.items():
            forecasts[region] = {}
            for i, entry in enumerate(region_data.get('forecast', [])):
                forecasts[region][i] = entry['carbonIntensity']
        return forecasts
    except Exception as e:
        print(f"❌ Error loading forecasts from {forecasts_file}: {e}")
        return {}


def _attach_forecasts(nodes, forecasts):
    for node in nodes:
        if hasattr(node, 'region') and node.region in forecasts:
            node.forecast = forecasts[node.region]
        else:
            if forecasts:
                node.forecast = forecasts[next(iter(forecasts))]
    return nodes


def _discover_experiments(experiments_root: str):
    """Yield (algo, pods, dirpath) for each experiment directory."""
    pattern = re.compile(r"^(?P<algo>[^_]+?)(?:_(?P<mode>op|proportional|uniform))?_(?P<pods>\d+)pods_")
    for current_root, dirnames, _ in os.walk(experiments_root):
        dirnames.sort()
        next_level = []
        for entry in dirnames:
            if entry.lower().startswith("archive"):
                continue
            dir_path = os.path.join(current_root, entry)
            m = pattern.match(entry)
            if m:
                algo = m.group('algo')
                pods = int(m.group('pods'))
                mode = m.group('mode') or ''
                if mode:
                    algo = f"{algo}-{mode}"
                yield algo, pods, dir_path
            else:
                next_level.append(entry)
        dirnames[:] = next_level


def _find_placement_csv(algo: str, dir_path: str):
    if algo.startswith('global-optimal'):
        p = os.path.join(dir_path, 'global_optimal_placements_session.csv')
        return p if os.path.exists(p) else None
    if algo.startswith('heuristic'):
        for name in (
            'heuristic_prop_placements_session.csv',
            'heuristic_uniform_placements_session.csv',
            'heuristic_placements_session.csv',
        ):
            p = os.path.join(dir_path, name)
            if os.path.exists(p):
                return p
        return None
    if algo == 'vanilla':
        for name in (
            'vanilla_placements_session.csv',
            'vanilla_placement_session_presence.csv',
            'vanilla_placement_session_bind.csv',
            'vanilla_placement_session_fixed.csv',
            'vanilla_placement_session.csv',
            'vanilla_placement_session.csv',
            'vanilla_placements.csv',
        ):
            p = os.path.join(dir_path, name)
            if os.path.exists(p):
                return p
        return None
    return None


def _parse_cpu_to_cores(val) -> float:
    s = str(val).strip() if val is not None else ""
    if not s:
        return 0.0
    try:
        return float(s)
    except Exception:
        pass
    if s.endswith('m'):
        try:
            return float(s[:-1]) / 1000.0
        except Exception:
            return 0.0
    import re as _re
    m = _re.match(r"^([0-9]+(?:\.[0-9]+)?)", s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return 0.0
    return 0.0


def _collect_pod_sets(csv_path: str):
    pods = set()
    try:
        with open(csv_path, 'r') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                pod_id = row.get('pod_id') or row.get('name') or row.get('pod')
                if pod_id:
                    pods.add(pod_id)
    except Exception:
        pass
    return pods


def _compute_total_emissions_for_common_pods(csv_path: str, nodes_dict: dict, allowed_pods: set):
    total_kg = 0.0
    placed_rows = 0
    with open(csv_path, 'r') as fh:
        reader = csv.DictReader(fh)
        occupancy = defaultdict(list)  # (node_id, slot) -> list[cpu cores]
        for row in reader:
            pod_id = row.get('pod_id') or row.get('name') or row.get('pod')
            if not pod_id or pod_id not in allowed_pods:
                continue
            node_id = row.get('node_id')
            if not node_id or node_id not in nodes_dict:
                continue
            try:
                start_slot = int(float(row.get('start_slot', 0)))
                duration = int(float(row.get('duration', 0)))
                cpu_req = _parse_cpu_to_cores(row.get('cpu_request', 0))
            except Exception:
                continue
            placed_rows += 1
            for offset in range(duration):
                occupancy[(node_id, start_slot + offset)].append(cpu_req)

        for (node_id, slot), cpu_list in occupancy.items():
            node = nodes_dict.get(node_id)
            if not node:
                continue
            total_cpu = getattr(node, 'totalCpu', 0.0) or 1e-6
            U = sum(cpu_list) / total_cpu
            if U <= 0:
                continue
            idle_w = node.power.get('idle', 0.0)
            k_watts = (node.power.get('max', 0.0) - node.power.get('active', 0.0))
            intensity = 200.0
            try:
                intensity = node.forecast.get(slot, 200.0)
            except Exception:
                pass
            lifetime_hours = getattr(node, 'lifetime', 0.0) or 1e-6
            embodied_per_h = getattr(node, 'embodiedCarbon', 0.0) / lifetime_hours
            idle_oper_g = intensity * (idle_w / 1000.0)
            dynamic_g = intensity * (k_watts * U / 1000.0)
            total_kg += (idle_oper_g + dynamic_g + embodied_per_h) / 1000.0
    return total_kg, placed_rows


def create_common_pods_emissions_plot(x_axis: str = 'utilization'):
    nodes = _load_nodes_from_yaml(NODES_FILE)
    forecasts = _load_carbon_forecasts(FORECASTS_FILE)
    nodes = _attach_forecasts(nodes, forecasts)
    nodes_dict = {getattr(n, 'id', None): n for n in nodes}

    totals = defaultdict(lambda: defaultdict(list))
    per_pod = defaultdict(lambda: defaultdict(list))

    if not os.path.isdir(EXPERIMENTS_ROOT):
        print(f"❌ Experiments directory not found: {EXPERIMENTS_ROOT}")
        return

    # Group directories by pods so we can intersect pod IDs across algos per pod count
    by_pods = defaultdict(lambda: defaultdict(list))  # pods -> algo -> [dir]
    for algo, pods, dir_path in _discover_experiments(EXPERIMENTS_ROOT):
        by_pods[pods][algo].append(dir_path)

    for pods, alg_map in by_pods.items():
        # Require vanilla presence
        if 'vanilla' not in alg_map:
            continue

        # Build mode-indexed keys for heuristic and global-optimal
        def _mode_of(key: str) -> str:
            parts = key.split('-', 1)
            return parts[1] if len(parts) == 2 else ''

        heur_keys = [k for k in alg_map.keys() if k.startswith('heuristic')]
        glob_keys = [k for k in alg_map.keys() if k.startswith('global-optimal')]
        if not heur_keys or not glob_keys:
            continue

        heur_by_mode = { _mode_of(k): k for k in heur_keys }
        glob_by_mode = { _mode_of(k): k for k in glob_keys }
        modes = sorted(set(heur_by_mode.keys()) & set(glob_by_mode.keys()))
        if not modes:
            continue

        for mode in modes:
            vanilla_key = 'vanilla'
            heur_key = heur_by_mode[mode]
            glob_key = glob_by_mode[mode]

            # Pick latest directory for each
            latest_dirs = {
                vanilla_key: sorted(alg_map[vanilla_key])[-1],
                heur_key: sorted(alg_map[heur_key])[-1],
                glob_key: sorted(alg_map[glob_key])[-1],
            }

            # Resolve CSVs
            csvs = {}
            for a, d in latest_dirs.items():
                p = _find_placement_csv(a, d)
                if not p:
                    csvs = {}
                    break
                csvs[a] = p
            if len(csvs) < 3:
                continue

            # Intersect pod ids across the three for this mode
            pod_sets = {a: _collect_pod_sets(p) for a, p in csvs.items()}
            common_pods = set.intersection(*pod_sets.values())
            if not common_pods:
                continue

            # Compute emissions for each algo restricted to common pods
            for a in (vanilla_key, heur_key, glob_key):
                total_kg, rows = _compute_total_emissions_for_common_pods(csvs[a], nodes_dict, common_pods)
                if rows == 0:
                    continue
                totals[a][pods].append(total_kg)
                per_pod[a][pods].append(total_kg / len(common_pods))

    if not totals:
        print(f"⚠️ No common-pods emissions data found in experiments: {EXPERIMENTS_ROOT}")
        return

    mean_vals = defaultdict(dict)
    stderr_vals = defaultdict(dict)
    mean_per_pod = defaultdict(dict)
    stderr_per_pod = defaultdict(dict)

    all_pods = set()
    for algo, byp in totals.items():
        for pods, vals in byp.items():
            all_pods.add(pods)
            arr = np.array(vals, dtype=float)
            mean_vals[algo][pods] = float(np.mean(arr))
            stderr_vals[algo][pods] = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    for algo, byp in per_pod.items():
        for pods, vals in byp.items():
            all_pods.add(pods)
            arr = np.array(vals, dtype=float)
            mean_per_pod[algo][pods] = float(np.mean(arr))
            stderr_per_pod[algo][pods] = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    pod_counts_sorted = sorted(all_pods)

    # Map pod counts to x-axis values (default: 24h-normalized CPU utilization).
    x_axis_mode = x_axis or 'utilization'
    pods_to_x = {}
    if x_axis_mode == 'utilization':
        pods_to_x = util_mod.compute_pods_to_24h_cpu_utilization(
            EXPERIMENTS_ROOT,
            NODES_FILE,
            prefer_algo='vanilla',
            horizon_hours=24.0,
        )
        if not pods_to_x:
            print(f"⚠️ Could not compute utilization mapping from experiments at {EXPERIMENTS_ROOT}; falling back to pod count on x-axis.")
            x_axis_mode = 'pods'
    if x_axis_mode != 'utilization':
        pods_to_x = {p: float(p) for p in pod_counts_sorted}

    # Plot 1: total emissions (kg) for common pods only
    plt.figure(figsize=(12, 8))

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
        'vanilla': 'Vanilla (Common Pods)',
        'heuristic': 'Heuristic (Common Pods)',
        'heuristic-proportional': 'Heuristic (Common Pods, Proportional)',
        'heuristic-uniform': 'Heuristic (Common Pods, Uniform)',
        'global-optimal': 'Global-Optimal (Common Pods)',
        'global-optimal-proportional': 'Global-Optimal (Common Pods, Proportional)',
        'global-optimal-uniform': 'Global-Optimal (Common Pods, Uniform)',
    }

    preferred = ['vanilla', 'heuristic-proportional', 'heuristic-uniform', 'global-optimal-proportional', 'global-optimal-uniform']
    algos_order = [a for a in preferred if a in totals.keys()] + [a for a in totals.keys() if a not in preferred]

    for algo in algos_order:
        x_vals, y_vals, y_errs = [], [], []
        for pods in pod_counts_sorted:
            if pods in mean_vals[algo]:
                x_vals.append(pods_to_x.get(pods, float(pods)))
                y_vals.append(mean_vals[algo][pods])
                y_errs.append(stderr_vals[algo][pods])
        if not x_vals:
            continue
        color = colors.get(algo, '#1f77b4')
        marker = markers.get(algo, 'o')
        label = labels.get(algo, algo.replace('-', ' ').title())
        if any(e > 0 for e in y_errs):
            plt.errorbar(x_vals, y_vals, yerr=y_errs, marker=marker, color=color, label=label, linewidth=3, markersize=10, alpha=0.9, capsize=4)
        else:
            plt.plot(x_vals, y_vals, marker=marker, color=color, label=label, linewidth=3, markersize=10, alpha=0.9)

    if x_axis_mode == 'utilization':
        plt.xlabel('Implied Average Cluster CPU Utilization over 24h (%)', fontsize=14, fontweight='bold')
    else:
        plt.xlabel('Number of Pods to Schedule', fontsize=14, fontweight='bold')
    plt.ylabel('Total Carbon Emissions for Common Pods (kg CO₂e)', fontsize=14, fontweight='bold')
    plt.title('Total Emissions vs. Cluster Utilization (Common Pods Only)\nVanilla vs Heuristic vs Global-Optimal', fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    if pod_counts_sorted:
        if x_axis_mode == 'utilization':
            x_coords = [pods_to_x.get(p, float(p)) for p in pod_counts_sorted]
            xmin = min(x_coords)
            xmax = max(x_coords)
            lo = max(0, 5 * int(xmin // 5))
            hi = 5 * int((xmax + 4.9999) // 5)
            if hi <= lo:
                hi = lo + 5
            hi = min(100, hi)
            plt.xlim(lo, hi)
            plt.xticks(list(range(lo, hi + 1, 5)), fontsize=12)
        else:
            plt.xlim(min(pod_counts_sorted) - 5, max(pod_counts_sorted) + 10)
            plt.xticks(pod_counts_sorted, fontsize=12)
    plt.ylim(bottom=0)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc='best', framealpha=0.9, shadow=True, fancybox=True)
    plt.tight_layout()

    output_dir = "/root/carbon-aware-orchestrator/figures/EmissionsVsPods"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"emissions_vs_pods_common_pods_{timestamp}"
    pdf_path = os.path.join(output_dir, base + ".pdf")
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

    print("✅ COMMON-PODS EMISSIONS VS PODS PLOT GENERATED!")
    print(f"📊 PDF saved: {pdf_path}")

    # Plot 2: emissions per common pod (kg/pod)
    plt.figure(figsize=(12, 8))
    for algo in algos_order:
        x_vals, y_vals, y_errs = [], [], []
        for pods in pod_counts_sorted:
            if pods in mean_per_pod[algo]:
                x_vals.append(pods_to_x.get(pods, float(pods)))
                y_vals.append(mean_per_pod[algo][pods])
                y_errs.append(stderr_per_pod[algo][pods])
        if not x_vals:
            continue
        color = colors.get(algo, '#1f77b4')
        marker = markers.get(algo, 'o')
        label = labels.get(algo, algo.replace('-', ' ').title())
        if any(e > 0 for e in y_errs):
            plt.errorbar(x_vals, y_vals, yerr=y_errs, marker=marker, color=color, label=label, linewidth=3, markersize=10, alpha=0.9, capsize=4)
        else:
            plt.plot(x_vals, y_vals, marker=marker, color=color, label=label, linewidth=3, markersize=10, alpha=0.9)

    if x_axis_mode == 'utilization':
        plt.xlabel('Implied Average Cluster CPU Utilization over 24h (%)', fontsize=14, fontweight='bold')
    else:
        plt.xlabel('Number of Pods to Schedule', fontsize=14, fontweight='bold')
    plt.ylabel('Emissions per Common Pod (kg CO₂e/pod)', fontsize=14, fontweight='bold')
    plt.title('Emissions per Pod vs. Cluster Utilization (Common Pods Only)\nVanilla vs Heuristic vs Global-Optimal', fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    if pod_counts_sorted:
        if x_axis_mode == 'utilization':
            x_coords = [pods_to_x.get(p, float(p)) for p in pod_counts_sorted]
            xmin = min(x_coords)
            xmax = max(x_coords)
            lo = max(0, 5 * int(xmin // 5))
            hi = 5 * int((xmax + 4.9999) // 5)
            if hi <= lo:
                hi = lo + 5
            hi = min(100, hi)
            plt.xlim(lo, hi)
            plt.xticks(list(range(lo, hi + 1, 5)), fontsize=12)
        else:
            plt.xlim(min(pod_counts_sorted) - 5, max(pod_counts_sorted) + 10)
            plt.xticks(pod_counts_sorted, fontsize=12)
    plt.ylim(bottom=0)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc='best', framealpha=0.9, shadow=True, fancybox=True)
    plt.tight_layout()

    base_perpod = f"emissions_per_pod_vs_pods_common_pods_{timestamp}"
    pdf_path_perpod = os.path.join(output_dir, base_perpod + ".pdf")
    plt.savefig(pdf_path_perpod, bbox_inches='tight', facecolor='white')
    print(f"📊 PDF saved: {pdf_path_perpod}")


if __name__ == "__main__":
    print("🚀 COMMON-PODS EMISSIONS VS PODS PLOT GENERATOR")
    print("=" * 50)
    create_common_pods_emissions_plot()
