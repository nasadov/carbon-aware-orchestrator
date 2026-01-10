#!/usr/bin/env python3
"""
Carbon Emissions vs Pod Count Plot Generator

Generates a plot with three curves (Carbon-Agnostic, Heuristic, Oracle),
with x-axis as number of pods in the experiment and y-axis as total carbon
emissions (kg CO2e) per experiment. Data is parsed dynamically from
/root/carbon-aware-orchestrator/experiments and its subdirectories.

Default behavior is to plot proportional embodied allocation only. Idle power is
included by default in operational emissions; pass --exclude-idle to drop it.
Embodied emissions are included by default; pass --exclude-emb to drop them.
Flags:
  --uniform  Plot only uniform embodied allocation (plus vanilla)
  --both     Plot both proportional and uniform (plus vanilla)
  --experiments-roots  Provide multiple experiment folders (e.g., different seeds)

Usage:
    python analysis/emissions_vs_pods_plot_generator.py [--uniform | --both]
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import re
import csv
import argparse
from collections import defaultdict
import json
import yaml
import sys

import utilization_plot_generator as util_mod
import success_rate_plot_generator as sr_plot

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

DEFAULT_EXPERIMENTS_ROOT = "/root/carbon-aware-orchestrator/experiments"
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


def _compute_total_emissions_for_csv(algo: str, csv_path: str, nodes_dict: dict, ignore_idle: bool = False, ignore_embodied: bool = False):
    """Compute total emissions (kg) for a given placement CSV.
    - global-optimal: sums 'total_carbon_emissions' column (already kg)
    - heuristic/vanilla: compute node-slot emissions from placements

    If ignore_idle is True, the idle power term is excluded from operational emissions.
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
            # Dynamic coefficient (W). Prefer max-idle; fall back to max-active for compatibility.
            k_watts = (node.power.get('max', 0.0) - node.power.get('idle', node.power.get('active', 0.0)))
            # Emissions per hour (g)
            intensity = 200.0
            try:
                intensity = node.forecast.get(slot, 200.0)
            except Exception:
                pass
            lifetime_hours = getattr(node, 'lifetime', 0.0) or 1e-6
            embodied_per_h = 0.0 if ignore_embodied else (getattr(node, 'embodiedCarbon', 0.0) / lifetime_hours)
            idle_oper_g = 0.0 if ignore_idle else intensity * (idle_w / 1000.0)
            dynamic_g = intensity * (k_watts * U / 1000.0)
            total_kg += (idle_oper_g + dynamic_g + embodied_per_h) / 1000.0
    return total_kg, rows


def _discover_experiments(experiments_root: str):
    """Yield (algo, pods, dirpath, mtime) for each experiment directory."""
    pattern = re.compile(r"^(?P<algo>[^_]+?)(?:_(?P<mode>op|proportional|uniform))?_(?P<pods>\d+)pods_")
    fallback_pattern = re.compile(r"^(?P<algo>heuristic|global-optimal|vanilla)(?:_(?P<mode>op|proportional|uniform))?_experiment_session_")
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
                try:
                    mtime = os.path.getmtime(dir_path)
                except Exception:
                    mtime = 0.0
                yield algo, pods, dir_path, mtime
            else:
                # Fallback: support session dirs like 'heuristic_op_experiment_session_...'
                fm = fallback_pattern.match(entry)
                if not fm:
                    next_level.append(entry)
                    continue
                algo = fm.group('algo')
                mode = fm.group('mode') or ''
                if mode:
                    algo = f"{algo}-{mode}"
                # Try to parse pods from placement_summary.log inside this dir
                pods = None
                summary_path = os.path.join(dir_path, 'placement_summary.log')
                try:
                    if os.path.isfile(summary_path):
                        with open(summary_path, 'r') as sf:
                            for line in sf:
                                # Example lines:
                                # '📈 Placed: 192/200 pods (96.0%)' or 'Dynamically calculated total pods (sum of replicas): 200'
                                m1 = re.search(r"Placed:\s*\d+/(\d+)\s+pods", line)
                                if m1:
                                    pods = int(m1.group(1))
                                    break
                                m2 = re.search(r"total\s+pods.*?:\s*(\d+)", line, flags=re.I)
                                if m2:
                                    pods = int(m2.group(1))
                                    break
                except Exception:
                    pods = None
                if pods is None:
                    # As a last resort, skip if we cannot determine pod count
                    next_level.append(entry)
                    continue
                try:
                    mtime = os.path.getmtime(dir_path)
                except Exception:
                    mtime = 0.0
                yield algo, pods, dir_path, mtime
        dirnames[:] = next_level


def _should_include_algo(algo: str, selection: str) -> bool:
    """Return True if this algo key should be included under selection.

    selection in { 'proportional', 'uniform', 'both', 'op' }
    """
    if algo == 'vanilla':
        return True
    if selection == 'both':
        return algo.endswith('-proportional') or algo.endswith('-uniform')
    if selection == 'uniform':
        return algo.endswith('-uniform')
    if selection == 'op':
        return algo.endswith('-op')
    # Default: proportional-only
    return algo.endswith('-proportional')


def _find_placement_csv(algo: str, dir_path: str):
    if algo.startswith('global-optimal'):
        p = os.path.join(dir_path, 'global_optimal_placements_session.csv')
        return p if os.path.exists(p) else None
    if algo.startswith('heuristic'):
        # try mode-specific file first
        candidates = [
            'heuristic_op_placements_session.csv',
            'heuristic_prop_placements_session.csv',
            'heuristic_uniform_placements_session.csv',
            'heuristic_placements_session.csv',
        ]
        for name in candidates:
            p = os.path.join(dir_path, name)
            if os.path.exists(p):
                return p
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


def _prepare_roots(experiments_root: str | None, experiments_roots: list[str] | None):
    if experiments_roots:
        roots = list(experiments_roots)
    elif experiments_root:
        roots = [experiments_root]
    else:
        roots = [DEFAULT_EXPERIMENTS_ROOT]

    # Deduplicate while preserving order
    normalized = []
    seen = set()
    for r in roots:
        abs_path = os.path.abspath(r)
        if abs_path in seen:
            continue
        seen.add(abs_path)
        normalized.append(abs_path)
    return normalized


def _build_output_suffix(roots: list[str]):
    """Derive a short, repeatable suffix for the output directory."""
    if len(roots) == 1:
        folder_name = os.path.basename(os.path.normpath(roots[0]))
        return folder_name.replace("sweep_", "") if folder_name.startswith("sweep_") else folder_name

    parts = []
    for r in roots:
        name = os.path.basename(os.path.normpath(r))
        name = name.replace("sweep_", "") if name.startswith("sweep_") else name
        parts.append(name)
    suffix = "multi_" + "__".join(parts)
    # Keep path short enough for most filesystems
    return suffix[:120]


def create_emissions_plot(
    selection: str = 'proportional',
    include_all: bool = False,
    ignore_idle: bool = False,
    ignore_embodied: bool = False,
    experiments_root: str | None = None,
    experiments_roots: list[str] | None = None,
    x_axis: str = 'utilization',
    run_success_rate: bool = False,
):
    roots = _prepare_roots(experiments_root, experiments_roots)
    valid_roots = []
    for r in roots:
        if os.path.isdir(r):
            valid_roots.append(r)
        else:
            print(f"⚠️ Experiments directory not found: {r}")

    if not valid_roots:
        print("❌ No valid experiments directories provided.")
        return

    print("📂 Scanning experiment roots:")
    for r in valid_roots:
        print(f"   - {r}")

    # Prepare nodes and forecasts for computed emissions
    nodes = _load_nodes_from_yaml(NODES_FILE)
    forecasts = _load_carbon_forecasts(FORECASTS_FILE)
    nodes = _attach_forecasts(nodes, forecasts)
    nodes_dict = {getattr(n, 'id', None): n for n in nodes}

    # Aggregate total emissions (kg) and per-pod emissions (kg/pod) per (algo, pods)
    totals = defaultdict(lambda: defaultdict(list))      # algo -> pods -> [kg]
    per_pod = defaultdict(lambda: defaultdict(list))     # algo -> pods -> [kg/pod]

    # Optionally filter to latest-only per (algo,pods) within each root; always keep all distinct roots
    selected = []
    for root in valid_roots:
        discovered = list(_discover_experiments(root))
        if not include_all:
            latest_map = {}  # (algo,pods) -> (dir_path, mtime)
            for algo, pods, dir_path, mtime in discovered:
                if not _should_include_algo(algo, selection):
                    continue
                key = (algo, pods)
                prev = latest_map.get(key)
                if prev is None or mtime > prev[1]:
                    latest_map[key] = (dir_path, mtime)
            selected.extend([(algo, pods, dir_path) for (algo, pods), (dir_path, _) in latest_map.items()])
        else:
            selected.extend([(algo, pods, dir_path) for (algo, pods, dir_path, _) in discovered if _should_include_algo(algo, selection)])

    for algo, pods, dir_path in selected:
        csv_path = _find_placement_csv(algo, dir_path)
        if not csv_path:
            continue
        total_kg, rows = _compute_total_emissions_for_csv(algo, csv_path, nodes_dict, ignore_idle=ignore_idle, ignore_embodied=ignore_embodied)
        if rows == 0:
            continue
        totals[algo][pods].append(total_kg)
        per_pod[algo][pods].append(total_kg / rows)

    if not totals:
        print(f"⚠️ No emissions data found in experiments: {', '.join(valid_roots)}")
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
            stderr_vals[algo][pods] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    for algo, by_pods in per_pod.items():
        for pods, vals in by_pods.items():
            all_pods.add(pods)
            arr = np.array(vals, dtype=float)
            mean_per_pod[algo][pods] = float(np.mean(arr))
            stderr_per_pod[algo][pods] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    pod_counts_sorted = sorted(all_pods)

    # Map pod counts to a shared x-axis (default: 24h-normalized CPU utilization).
    x_axis_mode = x_axis or 'utilization'
    pods_to_x = {}
    if x_axis_mode == 'utilization':
        for root in valid_roots:
            pods_to_x = util_mod.compute_pods_to_24h_cpu_utilization(
                root,
                NODES_FILE,
                prefer_algo='vanilla',
                horizon_hours=24.0,
            )
            if pods_to_x:
                break
        if not pods_to_x:
            joined_roots = ", ".join(valid_roots)
            print(f"⚠️ Could not compute utilization mapping from experiments at {joined_roots}; falling back to pod count on x-axis.")
            x_axis_mode = 'pods'
    if x_axis_mode != 'utilization':
        pods_to_x = {p: float(p) for p in pod_counts_sorted}

    # Compute percentage improvement vs carbon-agnostic baseline (per-pod emissions)
    improvement_vs_vanilla = defaultdict(dict)
    vanilla_key = None
    for key in mean_per_pod.keys():
        if key == 'vanilla':
            vanilla_key = key
            break
    if vanilla_key is None:
        print("⚠️ Carbon-agnostic (vanilla) baseline missing; skipping percentage improvement plot.")
    else:
        for algo, by_pods in mean_per_pod.items():
            if algo == vanilla_key:
                continue
            for pods, algo_mean in by_pods.items():
                baseline = mean_per_pod[vanilla_key].get(pods)
                if baseline is None or baseline <= 0:
                    continue
                improvement = (baseline - algo_mean) / baseline * 100.0
                improvement_vs_vanilla[algo][pods] = improvement

    # Plot
    plt.figure(figsize=(3.5, 2.7))

    colors = {
        'vanilla': '#d62728',
        'vanilla-op': '#d62728',
        'heuristic': '#ff7f0e',
        'heuristic-proportional': '#ff7f0e',
        'heuristic-uniform': '#ffbb78',
        'heuristic-op': '#ff7f0e',  # operational-only uses heuristic's classic orange
        'global-optimal': '#2ca02c',
        'global-optimal-proportional': '#2ca02c',
        'global-optimal-uniform': '#98df8a',
        'global-optimal-op': '#2ca02c',  # operational-only uses oracle's classic green
    }
    linestyles = {
        # Match runtime overlays (Fig. 7/8): TotEm solid, Oracle dashed, baseline dash-dot
        'vanilla': '-.',
        'vanilla-op': '-.',
        'heuristic': '-',
        'heuristic-proportional': '-',
        'heuristic-uniform': '-',
        'heuristic-op': '-',
        'global-optimal': '--',
        'global-optimal-proportional': '--',
        'global-optimal-uniform': '--',
        'global-optimal-op': '--',
    }
    markers = {
        'vanilla': '^',
        'vanilla-op': '^',
        'heuristic': 's',
        'heuristic-proportional': 's',
        'heuristic-uniform': 'D',
        'heuristic-op': 's',
        'global-optimal': 'o',
        'global-optimal-proportional': 'o',
        'global-optimal-uniform': '^',
        'global-optimal-op': 'o',
    }
    labels = {
        'vanilla': 'Carbon-Agnostic',
        'vanilla-op': 'Carbon-Agnostic Operational-Only',
        'heuristic': 'TotEm',
        'heuristic-proportional': 'TotEm',
        'heuristic-uniform': 'TotEm (Uniform)',
        'heuristic-op': 'TotEm (Operational-Only)',
        'global-optimal': 'Oracle',
        'global-optimal-proportional': 'Oracle',
        'global-optimal-uniform': 'Oracle (Uniform)',
        'global-optimal-op': 'Oracle (Operational-Only)',
    }

    preferred = [
        'vanilla', 'vanilla-op',
        'heuristic-op', 'global-optimal-op',
        'heuristic-proportional', 'heuristic-uniform',
        'global-optimal-proportional', 'global-optimal-uniform',
        'heuristic', 'global-optimal',
    ]
    present = list(totals.keys())
    algos_order = [a for a in preferred if a in present] + [a for a in present if a not in preferred]

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
        linestyle = linestyles.get(algo, '-')
        if any(e > 0 for e in y_errs):
            plt.errorbar(x_vals, y_vals, yerr=y_errs, marker=marker, color=color, label=label, linewidth=2.0, markersize=6.5, alpha=0.9, capsize=3, linestyle=linestyle)
        else:
            plt.plot(x_vals, y_vals, marker=marker, color=color, label=label, linewidth=2.0, markersize=6.5, alpha=0.9, linestyle=linestyle)

    if x_axis_mode == 'utilization':
        plt.xlabel('Implied Avg. CPU Utilization\n(24h, %)', fontsize=8, labelpad=6)
    else:
        plt.xlabel('Number of Pods to Schedule', fontsize=8)
    plt.ylabel('Total Emissions\n(kg CO₂e)', fontsize=8, labelpad=4)
    title_flags = []
    if ignore_idle:
        title_flags.append('Idle Excluded')
    if ignore_embodied:
        title_flags.append('Embodied Excluded')
    title_suffix = f" ({', '.join(title_flags)})" if title_flags else ''
    plt.title('')
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
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
            step = 10
            plt.xticks(list(range(lo, hi + 1, step)), fontsize=8)
        else:
            plt.xlim(min(pod_counts_sorted) - 5, max(pod_counts_sorted) + 10)
            plt.xticks(pod_counts_sorted, fontsize=8)
    plt.ylim(bottom=0)
    plt.yticks(fontsize=8)
    plt.legend(fontsize=8, loc='best', framealpha=0.9, shadow=False, fancybox=False)
    plt.tight_layout()

    output_dir_base = "/root/carbon-aware-orchestrator/figures/EmissionsVsPods"
    
    # Derive an output subdirectory based on the provided experiment roots
    ts_suffix = _build_output_suffix(valid_roots)
    output_dir = os.path.join(output_dir_base, ts_suffix)
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
    plt.figure(figsize=(3.5, 2.7))
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
        linestyle = linestyles.get(algo, '-')
        if any(e > 0 for e in y_errs):
            plt.errorbar(x_vals, y_vals, yerr=y_errs, marker=marker, color=color, label=label, linewidth=2.0, markersize=6.5, alpha=0.9, capsize=3, linestyle=linestyle)
        else:
            plt.plot(x_vals, y_vals, marker=marker, color=color, label=label, linewidth=2.0, markersize=6.5, alpha=0.9, linestyle=linestyle)

    if x_axis_mode == 'utilization':
        plt.xlabel('Implied Avg. CPU Utilization\n(24h, %)', fontsize=8, labelpad=6)
    else:
        plt.xlabel('Number of Pods to Schedule', fontsize=8)
    plt.ylabel('Emissions per Pod\n(kg CO₂e/pod)', fontsize=8, labelpad=4)
    plt.title('')
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
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
            step = 10
            plt.xticks(list(range(lo, hi + 1, step)), fontsize=8)
        else:
            plt.xlim(min(pod_counts_sorted) - 5, max(pod_counts_sorted) + 10)
            plt.xticks(pod_counts_sorted, fontsize=8)
    plt.ylim(bottom=0)
    plt.yticks(fontsize=8)
    plt.legend(fontsize=8, loc='best', framealpha=0.9, shadow=False, fancybox=False)
    plt.tight_layout()

    base_perpod = f"emissions_per_pod_vs_pods_{timestamp}"
    pdf_path_perpod = os.path.join(output_dir, base_perpod + ".pdf")
    plt.savefig(pdf_path_perpod, bbox_inches='tight', facecolor='white')
    print(f"📊 PDF saved: {pdf_path_perpod}")

    # Third plot: percentage improvement relative to vanilla baseline
    if vanilla_key and improvement_vs_vanilla:
        plt.figure(figsize=(3.5, 2.7))
        compare_algos = [a for a in algos_order if a in improvement_vs_vanilla]
        if not compare_algos:
            print("⚠️ No comparison algorithms available for percentage improvement plot.")
        else:
            for algo in compare_algos:
                x_vals, y_vals = [], []
                for pods in pod_counts_sorted:
                    if pods in improvement_vs_vanilla[algo]:
                        x_vals.append(pods_to_x.get(pods, float(pods)))
                        y_vals.append(improvement_vs_vanilla[algo][pods])
                if not x_vals:
                    continue
                color = colors.get(algo, '#1f77b4')
                marker = markers.get(algo, 'o')
                label = labels.get(algo, algo.replace('-', ' ').title())
                plt.plot(x_vals, y_vals, marker=marker, color=color, label=label, linewidth=2.0, markersize=6.5, alpha=0.9, linestyle=linestyles.get(algo, '-'))

            plt.axhline(0, color='gray', linestyle='--', linewidth=0.6, alpha=0.7)
            if x_axis_mode == 'utilization':
                plt.xlabel('Implied Avg. CPU Utilization\n(24h, %)', fontsize=8, labelpad=6)
            else:
                plt.xlabel('Number of Pods to Schedule', fontsize=8)
            plt.ylabel('Emissions Improvement\nvs Carbon-Agnostic (%)', fontsize=8, labelpad=4)
            plt.title('')
            plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
            if pod_counts_sorted:
                if x_axis_mode == 'utilization':
                    x_coords = [pods_to_x.get(p, float(p)) for p in pod_counts_sorted if p in improvement_vs_vanilla.get(compare_algos[0], {})]
                    if x_coords:
                        xmin = min(x_coords)
                        xmax = max(x_coords)
                        lo = max(0, 5 * int(xmin // 5))
                        hi = 5 * int((xmax + 4.9999) // 5)
                        if hi <= lo:
                            hi = lo + 5
                        hi = min(100, hi)
                        plt.xlim(lo, hi)
                        step = 10
                        plt.xticks(list(range(lo, hi + 1, step)), fontsize=8)
                        # Highlight typical edge/cloud utilization band (e.g., 10--35%)
                        typical_lo, typical_hi = 10.0, 35.0
                        band_lo = max(lo, typical_lo)
                        band_hi = min(hi, typical_hi)
                        if band_hi > band_lo:
                            # Stronger highlight for typical edge/cloud utilization band
                            plt.axvspan(band_lo, band_hi, color='#d0e2ff', alpha=0.5, zorder=0)
                else:
                    plt.xlim(min(pod_counts_sorted) - 5, max(pod_counts_sorted) + 10)
                    plt.xticks(pod_counts_sorted, fontsize=8)
            plt.yticks(fontsize=8)
            plt.legend(fontsize=8, loc='best', framealpha=0.9, shadow=False, fancybox=False)
            plt.tight_layout()

            base_improvement = f"emissions_improvement_vs_pods_{timestamp}"
            pdf_path_improvement = os.path.join(output_dir, base_improvement + ".pdf")
            plt.savefig(pdf_path_improvement, bbox_inches='tight', facecolor='white')
            print(f"📊 PDF saved: {pdf_path_improvement}")

    # Optional hook: generate success rate plots for the same experiment roots
    if run_success_rate:
        sr_selection = selection if selection in ('proportional', 'uniform', 'both') else 'proportional'
        try:
            sr_plot.create_success_rate_plot(
                selection=sr_selection,
                include_all=include_all,
                experiments_roots=valid_roots,
                x_axis=x_axis,
                show_plot=False,
                output_suffix=ts_suffix,
            )
        except Exception as e:
            print(f"⚠️ Success rate plot generation failed: {e}")


if __name__ == "__main__":
    print("🚀 EMISSIONS VS PODS PLOT GENERATOR")
    print("=" * 50)
    parser = argparse.ArgumentParser(description="Emissions vs Pods Plot Generator (idle included by default)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--uniform', action='store_true', help='Plot uniform embodied allocation only (plus vanilla)')
    group.add_argument('--both', action='store_true', help='Plot both proportional and uniform (plus vanilla)')
    group.add_argument('--op', action='store_true', help='Plot operational-only runs (suffix -op) (plus vanilla)')
    parser.add_argument('--all', action='store_true', help='Include all experiments (default: latest-only per algo and pod count)')
    parser.add_argument('--exclude-idle', action='store_true', help='Exclude idle power from operational emissions (default: included)')
    parser.add_argument('--exclude-emb', action='store_true', help='Exclude embodied emissions (default: included)')
    parser.add_argument('--experiments-root', type=str, default=None, help='Override experiments root directory (default: repository experiments)')
    parser.add_argument('--experiments-roots', type=str, nargs='+', default=None, help='List of experiment root directories (e.g., multiple seeds). Overrides --experiments-root if set.')
    parser.add_argument(
        '--x-axis',
        choices=['utilization', 'pods'],
        default='utilization',
        help='X-axis mode: 24h-normalized CPU utilization (default) or raw pod count',
    )
    parser.add_argument('--also-success-rate', action='store_true', help='Also generate success rate plot(s) using the same experiment roots.')
    args = parser.parse_args()
    selection = 'both' if args.both else ('uniform' if args.uniform else ('op' if args.op else 'proportional'))
    create_emissions_plot(
        selection,
        include_all=args.all,
        ignore_idle=args.exclude_idle,
        ignore_embodied=args.exclude_emb,
        experiments_root=args.experiments_root,
        experiments_roots=args.experiments_roots,
        x_axis=args.x_axis,
        run_success_rate=args.also_success_rate,
    )
