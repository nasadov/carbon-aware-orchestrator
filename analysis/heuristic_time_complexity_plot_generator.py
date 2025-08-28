#!/usr/bin/env python3
"""
Heuristic Time Complexity Plot Generator

Parses precompute timing CSVs and generates:
- Precompute time (s) vs number of pods (x-axis)
- Precompute time (s) vs number of nodes (x-axis)

Only the heuristic algorithm is considered.

Usage:
    python analysis/heuristic_time_complexity_plot_generator.py
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

EXPERIMENTS_ROOT = "/root/carbon-aware-orchestrator/pkg/carbon-aware/server-python/experiments"
OUTPUT_DIR = "/root/carbon-aware-orchestrator/figures/TimeComplexity"


def _iter_timing_csv_paths(experiments_root: str):
    """Yield all timing CSV files from experiments directory."""
    # Newer format: precompute_timing_<RUN_START>-<RUN_END>.csv
    for path in glob.glob(os.path.join(experiments_root, "precompute_timing_*.csv")):
        if os.path.isfile(path):
            yield path
    # Legacy fallback name
    legacy = os.path.join(experiments_root, "precompute_timing.csv")
    if os.path.isfile(legacy):
        yield legacy


def _parse_timing_rows(csv_path: str):
    """Parse rows from a timing CSV and return list of dicts for heuristic algorithm only."""
    rows = []
    try:
        with open(csv_path, 'r') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                algo = (row.get('algorithm') or '').strip()
                if algo != 'heuristic':
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


def _collect_time_data(experiments_root: str):
    """Collect elapsed time lists per pods and per nodes for heuristic runs."""
    by_pods = defaultdict(list)   # pods -> [elapsed]
    by_nodes = defaultdict(list)  # nodes -> [elapsed]

    any_files = False
    for csv_path in _iter_timing_csv_paths(experiments_root):
        any_files = True
        for row in _parse_timing_rows(csv_path):
            if row['elapsed_seconds'] <= 0:
                continue
            by_pods[row['pods']].append(row['elapsed_seconds'])
            by_nodes[row['nodes']].append(row['elapsed_seconds'])
    if not any_files:
        print(f"❌ No timing CSV files found in: {experiments_root}")
    return by_pods, by_nodes


def _plot_with_errorbars(x_vals, mean_map, stderr_map, *, xlabel, ylabel, title, color, marker, outfile_base):
    plt.figure(figsize=(12, 8))

    y_vals = [mean_map[x] for x in x_vals if x in mean_map]
    y_errs = [stderr_map.get(x, 0.0) for x in x_vals if x in mean_map]

    if not y_vals:
        print("⚠️ No data to plot for", title)
        return None

    if any(e > 0 for e in y_errs):
        plt.errorbar(
            x_vals, y_vals, yerr=y_errs,
            marker=marker, color=color, label='Heuristic',
            linewidth=3, markersize=10, alpha=0.9, capsize=4
        )
    else:
        plt.plot(
            x_vals, y_vals,
            marker=marker, color=color, label='Heuristic',
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


def create_heuristic_time_plots():
    by_pods, by_nodes = _collect_time_data(EXPERIMENTS_ROOT)

    if not by_pods and not by_nodes:
        return

    # Aggregate by pods
    pods_x, pods_mean, pods_stderr = _aggregate_time(by_pods)
    # Aggregate by nodes
    nodes_x, nodes_mean, nodes_stderr = _aggregate_time(by_nodes)

    # Time vs Pods
    _plot_with_errorbars(
        pods_x, pods_mean, pods_stderr,
        xlabel='Number of Pods to Schedule',
        ylabel='Precomputation Time (seconds)',
        title='Heuristic Precomputation Time vs. Pod Count',
        color='#ff7f0e', marker='s', outfile_base='heuristic_time_vs_pods'
    )

    # Time vs Nodes (if we have at least one x)
    if nodes_x:
        _plot_with_errorbars(
            nodes_x, nodes_mean, nodes_stderr,
            xlabel='Number of Nodes in Infrastructure',
            ylabel='Precomputation Time (seconds)',
            title='Heuristic Precomputation Time vs. Node Count',
            color='#ff7f0e', marker='s', outfile_base='heuristic_time_vs_nodes'
        )
    else:
        print("ℹ️ Skipping time vs nodes plot (no node count variation found).")


if __name__ == "__main__":
    print("🚀 HEURISTIC TIME COMPLEXITY PLOT GENERATOR")
    print("=" * 50)
    create_heuristic_time_plots()
