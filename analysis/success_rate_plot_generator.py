#!/usr/bin/env python3
"""
Success Rate Plot Generator

Generates success rate comparison plots by parsing experiment results.

Default: proportional-only (plus carbon-agnostic baseline). Use flags to change selection:
  --uniform  Plot only uniform (plus carbon-agnostic)
  --both     Plot both proportional and uniform (plus carbon-agnostic)

Usage:
    python analysis/success_rate_plot_generator.py [--uniform | --both]
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import re
import csv
import argparse
from collections import defaultdict

import utilization_plot_generator as util_mod


DEFAULT_EXPERIMENTS_ROOT = "/root/carbon-aware-orchestrator/experiments"


def _prepare_roots(experiments_root: str | None, experiments_roots: list[str] | None):
    if experiments_roots:
        roots = list(experiments_roots)
    elif experiments_root:
        roots = [experiments_root]
    else:
        roots = [DEFAULT_EXPERIMENTS_ROOT]

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
    if len(roots) == 1:
        folder_name = os.path.basename(os.path.normpath(roots[0]))
        return folder_name.replace("sweep_", "") if folder_name.startswith("sweep_") else folder_name
    parts = []
    for r in roots:
        name = os.path.basename(os.path.normpath(r))
        name = name.replace("sweep_", "") if name.startswith("sweep_") else name
        parts.append(name)
    suffix = "multi_" + "__".join(parts)
    return suffix[:120]


def _read_pods_txt_if_present(session_dir: str, pods_count_from_dir: int) -> int | None:
    pods_marker = os.path.join(session_dir, 'pods.txt')
    if os.path.exists(pods_marker):
        try:
            with open(pods_marker, 'r') as pf:
                content = pf.read()
            m = re.search(r"pods\s*=\s*(\d+)", content)
            if m:
                marker_val = int(m.group(1))
                # Sanity check: ignore clearly stale markers (e.g., 200 written into an 80pods dir)
                if marker_val != pods_count_from_dir:
                    # If difference is more than small tolerance, treat as stale and ignore
                    return None
                return marker_val
        except Exception:
            pass
    return None

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


def _discover_experiment_dirs(experiments_root: str, selection: str = 'proportional'):
    """Yield (algo_key, pods_count, entry_path, mtime) for directories matching selection."""
    if not os.path.isdir(experiments_root):
        return
    for current_root, dirnames, _ in os.walk(experiments_root):
        dirnames.sort()
        next_level = []
        for entry in dirnames:
            if entry.lower().startswith('archive'):
                continue
            entry_path = os.path.join(current_root, entry)

            m = re.match(r"^(?P<algo>[^_]+?)(?:_(?P<mode>op|proportional|uniform))?_(?P<pods>\d+)pods_", entry)
            if not m:
                next_level.append(entry)
                continue

            algo_key = m.group('algo')
            mode = m.group('mode') or ''
            if mode:
                algo_key = f"{algo_key}-{mode}"
            if not _should_include_algo(algo_key, selection):
                continue
            try:
                pods_count = int(m.group('pods'))
            except Exception:
                continue

            try:
                mtime = os.path.getmtime(entry_path)
            except Exception:
                mtime = 0.0
            yield algo_key, pods_count, entry_path, mtime

        dirnames[:] = next_level


def _parse_success_rates_from_experiments(experiments_root: str, selection: str = 'proportional', include_all: bool = False):
    """Scan experiments_root for runs and parse success rates per algorithm and pod count.

    Returns:
        results: dict[str, dict[int, list[float]]] mapping algo -> pods -> list of success rates
    """
    results = defaultdict(lambda: defaultdict(list))

    if not os.path.isdir(experiments_root):
        print(f"⚠️ Experiments directory not found: {experiments_root}")
        return results

    # Build selected directories: latest-only per (algo,pods) unless include_all
    discovered = list(_discover_experiment_dirs(experiments_root, selection))
    if not include_all:
        latest_map: dict[tuple[str, int], tuple[str, float]] = {}
        for algo_key, pods_count, entry_path, mtime in discovered:
            key = (algo_key, pods_count)
            prev = latest_map.get(key)
            if prev is None or mtime > prev[1]:
                latest_map[key] = (entry_path, mtime)
        selected = [(a, p, path) for (a, p), (path, _) in latest_map.items()]
    else:
        selected = [(a, p, path) for (a, p, path, _) in discovered]

    # Parse selected runs
    for algo_key, pods_count, entry_path in selected:
        placement_csv = None
        candidate_names = []
        if algo_key.startswith('global-optimal'):
            candidate_names = ['global_optimal_placements_session.csv']
        elif algo_key.startswith('heuristic'):
            candidate_names = [
                'heuristic_prop_placements_session.csv',
                'heuristic_uniform_placements_session.csv',
                'heuristic_placements_session.csv',
            ]
        elif algo_key == 'vanilla':
            candidate_names = [
                'vanilla_placement_session.csv',
                'vanilla_placement_session_bind.csv',
                'vanilla_placement_session_presence.csv',
                'vanilla_placement_session_fixed.csv',
                'vanilla_placements.csv',
            ]
        else:
            candidate_names = [
                'global_optimal_placements_session.csv',
                'heuristic_placements_session.csv',
                'vanilla_placement_session_fixed.csv',
                'vanilla_placement_session.csv',
                'vanilla_placements.csv',
            ]

        for name in candidate_names:
            p = os.path.join(entry_path, name)
            if os.path.exists(p):
                placement_csv = p
                break

        if placement_csv:
            try:
                with open(placement_csv, 'r') as fh:
                    reader = csv.DictReader(fh)
                    unique_pods = set()
                    for row in reader:
                        pid = row.get('pod_id')
                        if pid:
                            unique_pods.add(pid)
                placed = len(unique_pods)
                total_from_marker = _read_pods_txt_if_present(entry_path, pods_count)
                total = total_from_marker if total_from_marker is not None else pods_count
                if total > 0:
                    results[algo_key][pods_count].append(100.0 * placed / total)
            except Exception:
                pass

    return results


def _aggregate_results(results):
    """Aggregate list of success rates into mean and standard error per algo/pods.

    Returns:
        algos_order: list[str]
        pod_counts_sorted: list[int]
        mean_rates: dict[str, dict[int, float]]
        std_errs: dict[str, dict[int, float]]
    """
    mean_rates = defaultdict(dict)
    std_errs = defaultdict(dict)

    # Collect the union of pod counts across all algorithms
    pod_counts = set()
    for algo, by_pods in results.items():
        for pods in by_pods.keys():
            pod_counts.add(pods)
    pod_counts_sorted = sorted(pod_counts)

    # Stable order for known algos; unknown algos appended at end
    preferred = [
        "vanilla",
        "heuristic-proportional", "heuristic-uniform",
        "global-optimal-proportional", "global-optimal-uniform",
        "heuristic", "global-optimal",
    ]
    algos_present = list(results.keys())
    algos_order = [a for a in preferred if a in algos_present] + [a for a in algos_present if a not in preferred]

    for algo, by_pods in results.items():
        for pods, rates in by_pods.items():
            if not rates:
                continue
            arr = np.array(rates, dtype=float)
            mean_rates[algo][pods] = float(np.mean(arr))
            if arr.size > 1:
                std_errs[algo][pods] = float(np.std(arr, ddof=1))
            else:
                std_errs[algo][pods] = 0.0

    return algos_order, pod_counts_sorted, mean_rates, std_errs


def create_success_rate_plot(
    selection: str = 'proportional',
    include_all: bool = False,
    experiments_root: str | None = None,
    experiments_roots: list[str] | None = None,
    x_axis: str = 'utilization',
    show_plot: bool = True,
    output_suffix: str | None = None,
):
    """Generate success rate comparison plot by parsing experiment outputs."""
    roots = _prepare_roots(experiments_root, experiments_roots)
    valid_roots = []
    for r in roots:
        if os.path.isdir(r):
            valid_roots.append(r)
        else:
            print(f"⚠️ Experiments directory not found: {r}")

    if not valid_roots:
        print("❌ No valid experiments directories provided for success rate plotting.")
        return

    print("📂 (Success Rate) Scanning experiment roots:")
    for r in valid_roots:
        print(f"   - {r}")

    # Parse dynamic results across all roots and aggregate
    parsed = defaultdict(lambda: defaultdict(list))
    for root in valid_roots:
        root_results = _parse_success_rates_from_experiments(root, selection, include_all)
        for algo, by_pods in root_results.items():
            for pods, vals in by_pods.items():
                parsed[algo][pods].extend(vals)

    if not parsed:
        print(f"❌ No experiment results found in: {', '.join(valid_roots)}")
        return

    algos_order, pod_counts_sorted, mean_rates, std_errs = _aggregate_results(parsed)

    # Map pod counts to x-axis values (default: 24h-normalized CPU utilization).
    x_axis_mode = x_axis or 'utilization'
    pods_to_x = {}
    if x_axis_mode == 'utilization':
        for root in valid_roots:
            pods_to_x = util_mod.compute_pods_to_24h_cpu_utilization(
                root,
                util_mod.NODES_YAML_PATH,
                prefer_algo='vanilla',
                horizon_hours=24.0,
            )
            if pods_to_x:
                break
        if not pods_to_x:
            print(f"⚠️ Could not compute utilization mapping from experiments at {', '.join(valid_roots)}; falling back to pod count on x-axis.")
            x_axis_mode = 'pods'
    if x_axis_mode != 'utilization':
        pods_to_x = {p: float(p) for p in pod_counts_sorted}

    # Create publication-quality figure sized for single-column use
    plt.figure(figsize=(3.5, 2.7))

    # Define colors and markers for consistency
    colors = {
        'vanilla': '#d62728',
        'heuristic': '#ff7f0e',
        'heuristic-proportional': '#ff7f0e',
        'heuristic-uniform': '#ffbb78',
        'global-optimal': '#2ca02c',
        'global-optimal-proportional': '#2ca02c',
        'global-optimal-uniform': '#98df8a',
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
    linestyles = {
        # Align with runtime overlays: TotEm solid, Oracle dashed, baseline dash-dot
        'vanilla': '-.',
        'heuristic': '-',
        'heuristic-proportional': '-',
        'heuristic-uniform': '-',
        'global-optimal': '--',
        'global-optimal-proportional': '--',
        'global-optimal-uniform': '--',
    }

    labels = {
        'vanilla': 'Carbon-Agnostic',
        'heuristic': 'TotEm',
        'heuristic-proportional': 'TotEm',
        'heuristic-uniform': 'TotEm Uniform',
        'global-optimal': 'Oracle',
        'global-optimal-proportional': 'Oracle',
        'global-optimal-uniform': 'Oracle (Uniform)',
    }

    # Plot each algorithm
    for algo in algos_order:
        algo_label = labels.get(algo, algo.replace('-', ' ').title())
        color = colors.get(algo, '#1f77b4')
        marker = markers.get(algo, 'o')
        linestyle = linestyles.get(algo, '-')

        x_vals = []
        y_vals = []
        y_errs = []
        for pods in pod_counts_sorted:
            if pods in mean_rates[algo]:
                x_vals.append(pods_to_x.get(pods, float(pods)))
                y_vals.append(mean_rates[algo][pods])
                y_errs.append(std_errs[algo][pods])

        if not x_vals:
            continue

        if any(e > 0 for e in y_errs):
            plt.errorbar(
                x_vals, y_vals, yerr=y_errs,
                marker=marker, color=color, label=algo_label,
                linewidth=2.0, markersize=6.5, alpha=0.9, capsize=3, linestyle=linestyle
            )
        else:
            plt.plot(
                x_vals, y_vals,
                marker=marker, color=color, label=algo_label,
                linewidth=2.0, markersize=6.5, alpha=0.9, linestyle=linestyle
            )

    if x_axis_mode == 'utilization':
        plt.xlabel('Implied Avg. CPU Utilization\n(24h, %)', fontsize=8, labelpad=6)
    else:
        plt.xlabel('Number of Pods to Schedule', fontsize=8)
    plt.ylabel('Scheduling Success Rate\n(%)', fontsize=8, labelpad=4)

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
            step = 10
            plt.xticks(list(range(lo, hi + 1, step)), fontsize=8)
        else:
            plt.xlim(min(pod_counts_sorted) - 5, max(pod_counts_sorted) + 10)
            plt.xticks(pod_counts_sorted, fontsize=8)
    plt.ylim(50, 102)
    plt.yticks(range(50, 101, 10), fontsize=8)
    plt.legend(fontsize=8, loc='lower left', framealpha=0.9, shadow=False, fancybox=False)
    plt.title('')
    plt.tight_layout()

    # Save PDF only
    output_dir_base = "/root/carbon-aware-orchestrator/figures/SuccessRate"
    suffix = output_suffix or _build_output_suffix(valid_roots)
    output_dir = os.path.join(output_dir_base, suffix)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(output_dir, f"success_rate_comparison_{timestamp}.pdf")
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

    print("✅ SUCCESS RATE PLOT GENERATED!")
    print(f"📊 PDF saved: {pdf_path}")
    
    # Display summary of experimental results
    print(f"\n📈 EXPERIMENTAL RESULTS SUMMARY:")
    for algo in algos_order:
        rates_list = [mean_rates[algo][p] for p in pod_counts_sorted if p in mean_rates[algo]]
        if not rates_list:
            continue
        print(f"   {algo}: avg={np.mean(rates_list):.1f}% range={min(rates_list):.1f}%-{max(rates_list):.1f}%")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    print("🚀 SUCCESS RATE PLOT GENERATOR")
    print("=" * 50)
    parser = argparse.ArgumentParser(description="Success Rate Plot Generator")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--uniform', action='store_true', help='Plot uniform embodied allocation only (plus vanilla)')
    group.add_argument('--both', action='store_true', help='Plot both proportional and uniform (plus vanilla)')
    parser.add_argument('--all', action='store_true', help='Include all experiments (default: latest-only per algo and pod count)')
    parser.add_argument('--experiments-root', type=str, default=None, help='Override experiments root directory (default: repository experiments)')
    parser.add_argument('--experiments-roots', type=str, nargs='+', default=None, help='List of experiment root directories (e.g., multiple seeds). Overrides --experiments-root if set.')
    parser.add_argument(
        '--x-axis',
        choices=['utilization', 'pods'],
        default='utilization',
        help='X-axis mode: 24h-normalized CPU utilization (default) or raw pod count',
    )
    args = parser.parse_args()
    selection = 'both' if args.both else ('uniform' if args.uniform else 'proportional')
    create_success_rate_plot(
        selection,
        include_all=args.all,
        experiments_root=args.experiments_root,
        experiments_roots=args.experiments_roots,
        x_axis=args.x_axis,
        show_plot=True,
    ) 
