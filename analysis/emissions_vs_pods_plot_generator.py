#!/usr/bin/env python3
"""
Carbon Emissions vs Pod Count Plot Generator

Generates a plot with three curves (Vanilla, Heuristic, Global-Optimal),
with x-axis as number of pods in the experiment and y-axis as total carbon
emissions (kg CO2e) per experiment. Data is parsed dynamically from
/root/carbon-aware-orchestrator/pkg/carbon-aware/server-python/experiments.

Usage:
    python analysis/emissions_vs_pods_plot_generator.py
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

# Ensure carbon_aware modules are importable
CARBON_AWARE_SERVER_PY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pkg", "carbon-aware", "server-python"
)
sys.path.append(CARBON_AWARE_SERVER_PY_PATH)

try:
    from carbon_aware.models import CarbonAwareFlavour, CarbonAwarePod
    from carbon_aware.utils import compute_emissions
except Exception as e:
    print(f"⚠️ Warning: could not import carbon_aware modules, emissions computation for heuristic/vanilla may fail: {e}")
    CarbonAwareFlavour = None
    CarbonAwarePod = None
    compute_emissions = None

EXPERIMENTS_ROOT = "/root/carbon-aware-orchestrator/pkg/carbon-aware/server-python/experiments"
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
                "active": float(annotations.get("hardware.power/active_watts", "200")),
                "max": float(annotations.get("hardware.power/max_watts", "300")),
            }

            if CarbonAwareFlavour is None:
                # Minimal placeholder object if import failed
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
            # fallback to any available region
            if forecasts:
                node.forecast = forecasts[next(iter(forecasts))]
    return nodes


def _compute_total_emissions_for_csv(algo: str, csv_path: str, nodes_dict: dict):
    """Compute total emissions (kg) for a given placement CSV.
    - global-optimal: sums 'total_carbon_emissions' column (already kg)
    - heuristic/vanilla: compute using compute_emissions for each row
    """
    total_kg = 0.0
    rows = 0
    def _parse_cpu_to_cores(val) -> float:
        s = str(val).strip() if val is not None else ""
        if not s:
            return 0.0
        try:
            return float(s)
        except Exception:
            pass
        # Handle Kubernetes milli-CPU, e.g., "500m" => 0.5 cores
        if s.endswith('m'):
            try:
                return float(s[:-1]) / 1000.0
            except Exception:
                return 0.0
        # Fallback: remove any non-digit suffix and try again
        import re
        m = re.match(r"^([0-9]+(?:\.[0-9]+)?)", s)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return 0.0
        return 0.0

    with open(csv_path, 'r') as fh:
        reader = csv.DictReader(fh)
        # Compute node-level emissions by grouping per (node, slot) for all algorithms
        # Build occupancy: (node_id, slot) -> list of cpu_request (cores)
        occupancy = defaultdict(list)
        for row in reader:
            rows += 1
            node_id = row.get('node_id')
            if not node_id or node_id not in nodes_dict:
                continue
            try:
                start_slot = int(float(row.get('start_slot', 0)))
                duration = int(float(row.get('duration', 0)))
                cpu_req = _parse_cpu_to_cores(row.get('cpu_request', 0))
            except Exception:
                continue
            for offset in range(duration):
                occupancy[(node_id, start_slot + offset)].append(cpu_req)

        # Sum node emissions per slot and convert to kg
        for (node_id, slot), cpu_list in occupancy.items():
            node = nodes_dict.get(node_id)
            if not node:
                continue
            total_cpu = getattr(node, 'totalCpu', 0.0) or 1e-6
            U = sum(cpu_list) / total_cpu
            if U <= 0:
                continue
            # Node parameters
            idle_w = node.power.get('idle', 0.0)
            k_watts = (node.power.get('max', 0.0) - node.power.get('active', 0.0))
            # Emissions per hour (g)
            intensity = 200.0
            try:
                intensity = node.forecast.get(slot, 200.0)
            except Exception:
                pass
            embodied_per_h = 0.0
            lifetime_hours = getattr(node, 'lifetime', 0.0) or 1e-6
            embodied_per_h = getattr(node, 'embodiedCarbon', 0.0) / lifetime_hours
            idle_oper_g = intensity * (idle_w / 1000.0)
            dynamic_g = intensity * (k_watts * U / 1000.0)
            total_kg += (idle_oper_g + dynamic_g + embodied_per_h) / 1000.0
    return total_kg, rows


def _discover_experiments(experiments_root: str):
    """Yield (algo, pods, dirpath) for each experiment directory."""
    for entry in os.listdir(experiments_root):
        dir_path = os.path.join(experiments_root, entry)
        if not os.path.isdir(dir_path):
            continue
        m = re.match(r"^(?P<algo>[^_]+?)(?:_op)?_(?P<pods>\d+)pods_", entry)
        if not m:
            continue
        algo = m.group('algo')
        pods = int(m.group('pods'))
        yield algo, pods, dir_path


def _find_placement_csv(algo: str, dir_path: str):
    if algo == 'global-optimal':
        p = os.path.join(dir_path, 'global_optimal_placements_session.csv')
        return p if os.path.exists(p) else None
    if algo == 'heuristic':
        p = os.path.join(dir_path, 'heuristic_placements_session.csv')
        return p if os.path.exists(p) else None
    if algo == 'vanilla':
        # Prefer presence/bind reconstructed CSVs for accurate durations
        for name in (
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
    # Unknown algo naming
    return None


def create_emissions_plot():
    # Prepare nodes and forecasts for computed emissions
    nodes = _load_nodes_from_yaml(NODES_FILE)
    forecasts = _load_carbon_forecasts(FORECASTS_FILE)
    nodes = _attach_forecasts(nodes, forecasts)
    nodes_dict = {getattr(n, 'id', None): n for n in nodes}

    # Aggregate total emissions (kg) and per-pod emissions (kg/pod) per (algo, pods)
    totals = defaultdict(lambda: defaultdict(list))      # algo -> pods -> [kg]
    per_pod = defaultdict(lambda: defaultdict(list))     # algo -> pods -> [kg/pod]

    if not os.path.isdir(EXPERIMENTS_ROOT):
        print(f"❌ Experiments directory not found: {EXPERIMENTS_ROOT}")
        return

    for algo, pods, dir_path in _discover_experiments(EXPERIMENTS_ROOT):
        csv_path = _find_placement_csv(algo, dir_path)
        if not csv_path:
            continue
        total_kg, rows = _compute_total_emissions_for_csv(algo, csv_path, nodes_dict)
        if rows == 0:
            continue
        totals[algo][pods].append(total_kg)
        per_pod[algo][pods].append(total_kg / rows)

    if not totals:
        print(f"⚠️ No emissions data found in experiments: {EXPERIMENTS_ROOT}")
        return

    # Aggregate mean and std error (totals)
    mean_vals = defaultdict(dict)
    stderr_vals = defaultdict(dict)
    # Aggregate mean and std error (per pod)
    mean_per_pod = defaultdict(dict)
    stderr_per_pod = defaultdict(dict)

    all_pods = set()
    for algo, by_pods in totals.items():
        for pods, vals in by_pods.items():
            all_pods.add(pods)
            arr = np.array(vals, dtype=float)
            mean_vals[algo][pods] = float(np.mean(arr))
            stderr_vals[algo][pods] = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    for algo, by_pods in per_pod.items():
        for pods, vals in by_pods.items():
            all_pods.add(pods)
            arr = np.array(vals, dtype=float)
            mean_per_pod[algo][pods] = float(np.mean(arr))
            stderr_per_pod[algo][pods] = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    pod_counts_sorted = sorted(all_pods)

    # Plot
    plt.figure(figsize=(12, 8))

    colors = {
        'vanilla': '#2ca02c',
        'heuristic': '#ff7f0e',
        'global-optimal': '#d62728',
    }
    markers = {
        'vanilla': '^',
        'heuristic': 's',
        'global-optimal': 'o',
    }
    labels = {
        'vanilla': 'Vanilla (Baseline)',
        'heuristic': 'Heuristic (Carbon-Aware)',
        'global-optimal': 'Global-Optimal (MILP Oracle)',
    }

    algos_order = [a for a in ('vanilla', 'heuristic', 'global-optimal') if a in totals.keys()] + [a for a in totals.keys() if a not in ('vanilla', 'heuristic', 'global-optimal')]

    for algo in algos_order:
        x_vals, y_vals, y_errs = [], [], []
        for pods in pod_counts_sorted:
            if pods in mean_vals[algo]:
                x_vals.append(pods)
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

    plt.xlabel('Number of Pods to Schedule', fontsize=14, fontweight='bold')
    plt.ylabel('Total Carbon Emissions per Experiment (kg CO₂e)', fontsize=14, fontweight='bold')
    plt.title('Total Carbon Emissions vs. Pod Count\nVanilla vs Heuristic vs Global-Optimal', fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    if pod_counts_sorted:
        plt.xlim(min(pod_counts_sorted) - 5, max(pod_counts_sorted) + 10)
    plt.ylim(bottom=0)
    plt.xticks(pod_counts_sorted, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc='best', framealpha=0.9, shadow=True, fancybox=True)
    plt.tight_layout()

    output_dir = "/root/carbon-aware-orchestrator/figures/EmissionsVsPods"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"emissions_vs_pods_{timestamp}"
    pdf_path = os.path.join(output_dir, base + ".pdf")
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

    print("✅ EMISSIONS VS PODS PLOT GENERATED!")
    print(f"📊 PDF saved: {pdf_path}")

    # Summary log
    print("\n📈 Aggregated total emissions (kg) by algorithm and pods:")
    for algo in algos_order:
        line = []
        for pods in pod_counts_sorted:
            if pods in mean_vals[algo]:
                m = mean_vals[algo][pods]
                se = stderr_vals[algo][pods]
                line.append(f"{pods}:{m:.3f}{'±'+str(round(se,3)) if se>0 else ''}")
        if line:
            print(f"  {algo}: ", ", ".join(line))

    # Second plot: emissions per placed pod (kg/pod)
    plt.figure(figsize=(12, 8))
    for algo in algos_order:
        x_vals, y_vals, y_errs = [], [], []
        for pods in pod_counts_sorted:
            if pods in mean_per_pod[algo]:
                x_vals.append(pods)
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

    plt.xlabel('Number of Pods to Schedule', fontsize=14, fontweight='bold')
    plt.ylabel('Emissions per Placed Pod (kg CO₂e/pod)', fontsize=14, fontweight='bold')
    plt.title('Emissions per Pod vs. Pod Count\nVanilla vs Heuristic vs Global-Optimal', fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    if pod_counts_sorted:
        plt.xlim(min(pod_counts_sorted) - 5, max(pod_counts_sorted) + 10)
    plt.ylim(bottom=0)
    plt.xticks(pod_counts_sorted, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc='best', framealpha=0.9, shadow=True, fancybox=True)
    plt.tight_layout()

    base_perpod = f"emissions_per_pod_vs_pods_{timestamp}"
    pdf_path_perpod = os.path.join(output_dir, base_perpod + ".pdf")
    plt.savefig(pdf_path_perpod, bbox_inches='tight', facecolor='white')
    print(f"📊 PDF saved: {pdf_path_perpod}")


if __name__ == "__main__":
    print("🚀 EMISSIONS VS PODS PLOT GENERATOR")
    print("=" * 50)
    create_emissions_plot()
