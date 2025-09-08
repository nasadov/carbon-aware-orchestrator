#!/usr/bin/env python3
"""
Time Complexity Plot Generator

Parses precompute timing CSVs and generates:
- Precompute time (s) vs number of pods (x-axis)
- Precompute time (s) vs number of nodes (x-axis)

Supports plotting for `heuristic`, `global-optimal`, and `vanilla` algorithms.

Usage examples:
    # Default: plot all three algorithms, using the latest timing CSV
    python analysis/time_complexity_plot_generator.py

    # Use all available timing CSVs (aggregate across runs)
    python analysis/time_complexity_plot_generator.py --all

    # Limit to a subset of algorithms with all runs
    python analysis/time_complexity_plot_generator.py --all --algorithms heuristic global-optimal
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import csv
import glob
from collections import defaultdict
import argparse

EXPERIMENTS_ROOT = "/root/carbon-aware-orchestrator/pkg/carbon-aware/server-python/experiments"
OUTPUT_DIR = "/root/carbon-aware-orchestrator/figures/TimeComplexity"


def _iter_timing_csv_paths(experiments_root: str, latest_only: bool = True, algorithm_filter: str | None = None):
    """Yield timing CSV file paths from experiments directory.

    If latest_only is True, return only the most recent file by mtime.
    """
    candidates = [p for p in glob.glob(os.path.join(experiments_root, "precompute_timing_*.csv")) if os.path.isfile(p)]
    # Exclude vanilla timing files unless explicitly requested
    if algorithm_filter in ("heuristic", "global-optimal"):
        candidates = [p for p in candidates if "precompute_timing_vanilla_" not in os.path.basename(p)]

    # Legacy fallback name
    legacy = os.path.join(experiments_root, "precompute_timing.csv")
    if not candidates and os.path.isfile(legacy):
        candidates = [legacy]

    if latest_only and candidates:
        latest = max(candidates, key=os.path.getmtime)
        yield latest
        return

    for path in sorted(candidates):
        yield path


def _parse_timing_rows(csv_path: str, algorithm_filter: str):
    """Parse rows from a timing CSV and return list of dicts for the given algorithm."""
    rows = []
    try:
        with open(csv_path, 'r') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                algo = (row.get('algorithm') or '').strip()
                if algo != algorithm_filter:
                    continue
                try:
                    pods = int(float(row.get('pods', 0)))
                except Exception:
                    pods = 0
                try:
                    nodes = int(float(row.get('nodes', 0)))
                except Exception:
                    nodes = 0
                try:
                    elapsed = float(row.get('elapsed_seconds') or row.get('elapsed') or 0.0)
                except Exception:
                    elapsed = 0.0
                rows.append({
                    'pods': pods,
                    'nodes': nodes,
                    'elapsed_seconds': elapsed,
                    'run_id': row.get('run_id') or '',
                    'begin_time': row.get('begin_time') or row.get('timestamp') or '',
                    'end_time': row.get('end_time') or ''
                })
    except Exception as e:
        print(f"⚠️ Failed to read {csv_path}: {e}")
    return rows


def _aggregate_time(series: dict[int, list[float]]):
    """Compute mean and standard error for a mapping X -> [times]."""
    xs = sorted(series.keys())
    means = {}
    stderrs = {}
    for x in xs:
        arr = np.array(series[x], dtype=float)
        if arr.size == 0:
            continue
        means[x] = float(np.mean(arr))
        stderrs[x] = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return xs, means, stderrs


def _collect_time_data(experiments_root: str, algorithm: str, latest_only: bool = True):
    """Collect elapsed time lists per pods and per nodes for runs of a specific algorithm."""
    by_pods = defaultdict(list)   # pods -> [elapsed]
    by_nodes = defaultdict(list)  # nodes -> [elapsed]

    any_files = False
    for csv_path in _iter_timing_csv_paths(experiments_root, latest_only=latest_only, algorithm_filter=algorithm):
        any_files = True
        for row in _parse_timing_rows(csv_path, algorithm_filter=algorithm):
            if row['elapsed_seconds'] <= 0:
                continue
            by_pods[row['pods']].append(row['elapsed_seconds'])
            by_nodes[row['nodes']].append(row['elapsed_seconds'])
    if not any_files:
        print(f"❌ No timing CSV files found in: {experiments_root}")
    return by_pods, by_nodes


def _plot_with_errorbars(x_vals, mean_map, stderr_map, *, xlabel, ylabel, title, color, marker, outfile_base, label):
    plt.figure(figsize=(12, 8))

    y_vals = [mean_map[x] for x in x_vals if x in mean_map]
    y_errs = [stderr_map.get(x, 0.0) for x in x_vals if x in mean_map]

    if not y_vals:
        print("⚠️ No data to plot for", title)
        return None

    if any(e > 0 for e in y_errs):
        plt.errorbar(
            x_vals, y_vals, yerr=y_errs,
            marker=marker, color=color, label=label,
            linewidth=3, markersize=10, alpha=0.9, capsize=4
        )
    else:
        plt.plot(
            x_vals, y_vals,
            marker=marker, color=color, label=label,
            linewidth=3, markersize=10, alpha=0.9
        )

    plt.xlabel(xlabel, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1)

    if x_vals:
        x_min = min(x_vals)
        x_max = max(x_vals)
        if isinstance(x_min, (int, np.integer)) and isinstance(x_max, (int, np.integer)):
            plt.xlim(x_min - 1, x_max + 1)
            plt.xticks(x_vals, fontsize=12)
        else:
            plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(fontsize=12, loc='best', framealpha=0.9, shadow=True, fancybox=True)
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(OUTPUT_DIR, f"{outfile_base}_{timestamp}.pdf")
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

    print(f"✅ Plot saved: {pdf_path}")
    return pdf_path


def _create_time_plots_for_algorithm(algorithm: str, latest_only: bool = False):
    by_pods, by_nodes = _collect_time_data(EXPERIMENTS_ROOT, algorithm=algorithm, latest_only=latest_only)

    if not by_pods and not by_nodes:
        print(f"ℹ️ No timing data found for algorithm={algorithm}")
        return

    # Aggregate by pods
    pods_x, pods_mean, pods_stderr = _aggregate_time(by_pods)
    # Aggregate by nodes
    nodes_x, nodes_mean, nodes_stderr = _aggregate_time(by_nodes)

    # Select visuals per algorithm
    if algorithm == 'heuristic':
        color = '#ff7f0e'
        label = 'Heuristic'
        base_pods = 'heuristic_time_vs_pods'
        base_nodes = 'heuristic_time_vs_nodes'
    elif algorithm == 'global-optimal':
        color = '#1f77b4'
        label = 'Global-Optimal'
        base_pods = 'global_optimal_time_vs_pods'
        base_nodes = 'global_optimal_time_vs_nodes'
    else:
        color = '#2ca02c'
        label = 'Vanilla'
        base_pods = 'vanilla_time_vs_pods'
        base_nodes = 'vanilla_time_vs_nodes'

    # Time vs Pods
    _plot_with_errorbars(
        pods_x, pods_mean, pods_stderr,
        xlabel='Number of Pods to Schedule',
        ylabel='Precomputation Time (seconds)',
        title=f'{label} Precomputation Time vs. Pod Count',
        color=color, marker='s', outfile_base=base_pods, label=label
    )

    # Time vs Nodes (if we have at least one x)
    if nodes_x:
        _plot_with_errorbars(
            nodes_x, nodes_mean, nodes_stderr,
            xlabel='Number of Nodes in Infrastructure',
            ylabel='Precomputation Time (seconds)',
            title=f'{label} Precomputation Time vs. Node Count',
            color=color, marker='s', outfile_base=base_nodes, label=label
        )
    else:
        print(f"ℹ️ Skipping time vs nodes plot for {algorithm} (no node count variation found).")


def _collect_all_algorithms_time_data(latest_only: bool = True):
    algs = ['heuristic', 'global-optimal', 'vanilla']
    data = {}
    for alg in algs:
        by_pods, _ = _collect_time_data(EXPERIMENTS_ROOT, algorithm=alg, latest_only=latest_only)
        data[alg] = by_pods
    return data


def _plot_combined_time_vs_pods(alg_to_by_pods: dict, cap_seconds: float = 60.0):
    # Visual settings per algorithm
    styles = {
        'heuristic': {'color': '#ff7f0e', 'marker': 's', 'label': 'Heuristic'},
        'global-optimal': {'color': '#1f77b4', 'marker': 'o', 'label': 'Global-Optimal'},
        'vanilla': {'color': '#2ca02c', 'marker': 'D', 'label': 'Vanilla'},
    }

    plt.figure(figsize=(12, 8))

    for alg, series in alg_to_by_pods.items():
        if not series:
            continue
        xs, means, stderrs = _aggregate_time(series)
        y_vals = [means[x] for x in xs if x in means]
        y_errs = [stderrs.get(x, 0.0) for x in xs if x in means]

        style = styles.get(alg, {'color': 'gray', 'marker': 'o', 'label': alg})

        if any(e > 0 for e in y_errs):
            plt.errorbar(xs, y_vals, yerr=y_errs, color=style['color'], marker=style['marker'],
                         linewidth=3, markersize=10, alpha=0.9, capsize=4, label=style['label'])
        else:
            plt.plot(xs, y_vals, color=style['color'], marker=style['marker'],
                     linewidth=3, markersize=10, alpha=0.9, label=style['label'])

        # Mark likely time-limited points for global-optimal
        if alg == 'global-optimal' and cap_seconds and cap_seconds > 0:
            eps = 0.5
            for x in xs:
                vals = series.get(x, [])
                if not vals:
                    continue
                if max(vals) >= cap_seconds - eps:
                    plt.scatter([x], [means[x]], marker='^', s=120, color=style['color'], edgecolors='k', zorder=5)

    plt.xlabel('Number of Pods to Schedule', fontsize=14, fontweight='bold')
    plt.ylabel('Precomputation Time (seconds)', fontsize=14, fontweight='bold')
    plt.title('Precomputation Time vs. Pod Count (All Algorithms)', fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    plt.legend(fontsize=12, loc='best', framealpha=0.9, shadow=True, fancybox=True)
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(OUTPUT_DIR, f'time_vs_pods_all_algorithms_{ts}.pdf')
    plt.savefig(out, bbox_inches='tight', facecolor='white')
    print(f"✅ Plot saved: {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate time complexity plots from precompute timing CSVs.")
    parser.add_argument('--all', action='store_true', help='Use all available timing CSV files (default: latest only)')
    parser.add_argument('--algorithms', nargs='+', choices=['heuristic', 'global-optimal', 'vanilla'], default=['heuristic', 'global-optimal', 'vanilla'], help='Algorithms to plot')
    parser.add_argument('--cap-seconds', type=float, default=60.0, help='Time budget used by MILP; marks capped points (default: 60)')
    args = parser.parse_args()

    latest_only = not args.all
    print("🚀 TIME COMPLEXITY PLOT GENERATOR")
    print("=" * 50)
    print(f"Algorithms: {', '.join(args.algorithms)} | latest_only={latest_only}")
    for algo in args.algorithms:
        _create_time_plots_for_algorithm(algo, latest_only=latest_only)
    # Combined overlay plot across all algorithms
    alg_to_by_pods = _collect_all_algorithms_time_data(latest_only=latest_only)
    _plot_combined_time_vs_pods(alg_to_by_pods, cap_seconds=args.cap_seconds)


