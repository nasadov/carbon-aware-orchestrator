#!/usr/bin/env python3
"""
Success Rate Plot Generator

Generates success rate comparison plots by parsing experiment results.

Default: proportional-only (plus vanilla). Use flags to change selection:
  --uniform  Plot only uniform (plus vanilla)
  --both     Plot both proportional and uniform (plus vanilla)

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


EXPERIMENTS_ROOT = "/root/carbon-aware-orchestrator/experiments"


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
                std_errs[algo][pods] = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
            else:
                std_errs[algo][pods] = 0.0

    return algos_order, pod_counts_sorted, mean_rates, std_errs


def create_success_rate_plot(selection: str = 'proportional', include_all: bool = False):
    """Generate success rate comparison plot by parsing experiment outputs."""
    # Parse dynamic results
    parsed = _parse_success_rates_from_experiments(EXPERIMENTS_ROOT, selection, include_all)
    if not parsed:
        print(f"❌ No experiment results found in: {EXPERIMENTS_ROOT}")
        return

    algos_order, pod_counts_sorted, mean_rates, std_errs = _aggregate_results(parsed)

    # Create publication-quality figure
    plt.figure(figsize=(12, 8))

    # Define colors and markers for consistency
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
        'vanilla': 'Vanilla (Baseline)',
        'heuristic': 'Heuristic (Carbon-Aware)',
        'heuristic-proportional': 'Heuristic (Proportional)',
        'heuristic-uniform': 'Heuristic (Uniform)',
        'global-optimal': 'Global-Optimal (MILP Oracle)',
        'global-optimal-proportional': 'Global-Optimal (Proportional)',
        'global-optimal-uniform': 'Global-Optimal (Uniform)',
    }

    # Plot each algorithm
    for algo in algos_order:
        algo_label = labels.get(algo, algo.replace('-', ' ').title())
        color = colors.get(algo, '#1f77b4')
        marker = markers.get(algo, 'o')

        x_vals = []
        y_vals = []
        y_errs = []
        for pods in pod_counts_sorted:
            if pods in mean_rates[algo]:
                x_vals.append(pods)
                y_vals.append(mean_rates[algo][pods])
                y_errs.append(std_errs[algo][pods])

        if not x_vals:
            continue

        if any(e > 0 for e in y_errs):
            plt.errorbar(
                x_vals, y_vals, yerr=y_errs,
                marker=marker, color=color, label=algo_label,
                linewidth=3, markersize=10, alpha=0.9, capsize=4
            )
        else:
            plt.plot(
                x_vals, y_vals,
                marker=marker, color=color, label=algo_label,
                linewidth=3, markersize=10, alpha=0.9
            )

    plt.xlabel('Number of Pods to Schedule', fontsize=14, fontweight='bold')
    plt.ylabel('Scheduling Success Rate (%)', fontsize=14, fontweight='bold')
    plt.title('Scheduling Success Rate vs. Pod Count\nComparison of Three Algorithms', 
              fontsize=16, fontweight='bold', pad=20)

    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    if pod_counts_sorted:
        plt.xlim(min(pod_counts_sorted) - 5, max(pod_counts_sorted) + 10)
    plt.ylim(0, 105)
    plt.xticks(pod_counts_sorted, fontsize=12)
    plt.yticks(range(0, 101, 10), fontsize=12)
    plt.legend(fontsize=12, loc='lower left', framealpha=0.9, shadow=True, fancybox=True)
    plt.tight_layout()

    # Save PDF only
    output_dir = "/root/carbon-aware-orchestrator/figures/SuccessRate"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = f"{output_dir}/success_rate_comparison_{timestamp}.pdf"
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
    
    plt.show()

if __name__ == "__main__":
    print("🚀 SUCCESS RATE PLOT GENERATOR")
    print("=" * 50)
    parser = argparse.ArgumentParser(description="Success Rate Plot Generator")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--uniform', action='store_true', help='Plot uniform embodied allocation only (plus vanilla)')
    group.add_argument('--both', action='store_true', help='Plot both proportional and uniform (plus vanilla)')
    parser.add_argument('--all', action='store_true', help='Include all experiments (default: latest-only per algo and pod count)')
    args = parser.parse_args()
    selection = 'both' if args.both else ('uniform' if args.uniform else 'proportional')
    create_success_rate_plot(selection, include_all=args.all) 
