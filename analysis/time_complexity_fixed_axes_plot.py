#!/usr/bin/env python3
"""
Fixed-axis multi-algorithm time complexity plots.

Generates two figures from sweep result CSVs:
1) Nodes fixed → precomputation time vs pods (all three algorithms overlaid)
2) Pods fixed → precomputation time vs nodes (all three algorithms overlaid)

It discovers result CSVs under an experiments root produced by
scripts/time_complexity_sweep.py:
  - heuristic_time_complexity_results.csv
  - global_optimal_time_complexity_results.csv
  - vanilla_time_complexity_results.csv

Usage examples:
  python analysis/time_complexity_fixed_axes_plot.py --latest-only
  python analysis/time_complexity_fixed_axes_plot.py --fixed-node 64 --latest-only
  python analysis/time_complexity_fixed_axes_plot.py --fixed-pods 3200 --latest-only
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

DEFAULT_EXPERIMENTS_ROOT = "/root/carbon-aware-orchestrator/experiments/time_complexity"
DEFAULT_OUTPUT_DIR = "/root/carbon-aware-orchestrator/figures/TimeComplexity"


ALG_STYLES: Dict[str, Dict[str, str]] = {
    "heuristic": {"color": "#ff7f0e", "marker": "s", "label": "TotEm"},
    "global-optimal": {"color": "#2ca02c", "marker": "o", "label": "Oracle"},
    "vanilla": {"color": "#d62728", "marker": "D", "label": "Carbon-Agnostic"},
}


def _find_results_csvs(root: str, algorithm: str, latest_only: bool) -> List[str]:
    key = "global_optimal" if algorithm == "global-optimal" else algorithm
    pattern = os.path.join(root, "**", f"{key}_time_complexity_results.csv")
    paths = [p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)]
    if not paths:
        return []
    if latest_only:
        return [max(paths, key=os.path.getmtime)]
    return sorted(paths)


def _load_results(paths: List[str], algorithm: str) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            if not {"node_count", "total_pods", "elapsed_seconds", "status"}.issubset(df.columns):
                continue
            df = df[df["status"] == "success"][
                ["node_count", "total_pods", "elapsed_seconds"]
            ].copy()
            df["algorithm"] = algorithm
            frames.append(df)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    agg = (
        df.groupby(["algorithm", "node_count", "total_pods"], as_index=False)
        .agg(
            median_elapsed=("elapsed_seconds", "median"),
            stderr_elapsed=("elapsed_seconds", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0),
        )
        .sort_values(["algorithm", "node_count", "total_pods"])
    )
    return agg


def _common_axes(aggs: Dict[str, pd.DataFrame]) -> Tuple[List[int], List[int]]:
    nodes_sets = []
    pods_sets = []
    for df in aggs.values():
        if df.empty:
            continue
        nodes_sets.append(set(df["node_count"].unique().tolist()))
        pods_sets.append(set(df["total_pods"].unique().tolist()))
    if not nodes_sets or not pods_sets:
        return [], []
    nodes = sorted(set.intersection(*nodes_sets))
    pods = sorted(set.intersection(*pods_sets))
    return nodes, pods


def _plot_fixed_nodes(
    aggs: Dict[str, pd.DataFrame],
    fixed_node: int,
    output_dir: str,
) -> str | None:
    plt.figure(figsize=(12, 8))
    any_series = False
    for alg, df in aggs.items():
        if df.empty:
            continue
        sub = df[df["node_count"] == fixed_node].sort_values("total_pods")
        if sub.empty:
            continue
        style = ALG_STYLES.get(alg, {"color": "gray", "marker": "o", "label": alg})
        plt.errorbar(
            sub["total_pods"],
            sub["median_elapsed"],
            yerr=sub["stderr_elapsed"],
            color=style["color"],
            marker=style["marker"],
            linewidth=3,
            markersize=9,
            alpha=0.9,
            capsize=4,
            label=style["label"],
        )
        any_series = True
    if not any_series:
        plt.close()
        return None
    plt.xlabel("Total pods scheduled", fontsize=14, fontweight="bold")
    plt.ylabel("Runtime (s)", fontsize=14, fontweight="bold")
    plt.title(f"Precompute time vs pods (nodes fixed = {fixed_node})", fontsize=16, fontweight="bold", pad=18)
    plt.grid(True, alpha=0.3, linestyle="--", linewidth=1)
    plt.legend(fontsize=12, loc="best", framealpha=0.9, shadow=True, fancybox=True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(output_dir, f"time_vs_pods_fixed_nodes_{fixed_node}_all_algorithms_{ts}.pdf")
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close()
    return out


def _plot_fixed_pods(
    aggs: Dict[str, pd.DataFrame],
    fixed_pods: int,
    output_dir: str,
) -> str | None:
    plt.figure(figsize=(12, 8))
    any_series = False
    for alg, df in aggs.items():
        if df.empty:
            continue
        sub = df[df["total_pods"] == fixed_pods].sort_values("node_count")
        if sub.empty:
            continue
        style = ALG_STYLES.get(alg, {"color": "gray", "marker": "o", "label": alg})
        plt.errorbar(
            sub["node_count"],
            sub["median_elapsed"],
            yerr=sub["stderr_elapsed"],
            color=style["color"],
            marker=style["marker"],
            linewidth=3,
            markersize=9,
            alpha=0.9,
            capsize=4,
            label=style["label"],
        )
        any_series = True
    if not any_series:
        plt.close()
        return None
    plt.xlabel("Number of nodes", fontsize=14, fontweight="bold")
    plt.ylabel("Runtime (s)", fontsize=14, fontweight="bold")
    plt.title(f"Precompute time vs nodes (pods fixed = {fixed_pods})", fontsize=16, fontweight="bold", pad=18)
    plt.grid(True, alpha=0.3, linestyle="--", linewidth=1)
    plt.legend(fontsize=12, loc="best", framealpha=0.9, shadow=True, fancybox=True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(output_dir, f"time_vs_nodes_fixed_pods_{fixed_pods}_all_algorithms_{ts}.pdf")
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close()
    return out


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate fixed-axis multi-algorithm time complexity plots.")
    parser.add_argument("--sweep-root", type=str, default=DEFAULT_EXPERIMENTS_ROOT, help="Root folder with sweep outputs")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to place figures")
    parser.add_argument("--algorithms", nargs="+", default=["heuristic", "global-optimal", "vanilla"], choices=["heuristic", "global-optimal", "vanilla"], help="Algorithms to include")
    parser.add_argument("--latest-only", action="store_true", help="Use latest results per algorithm (otherwise aggregate all)")
    parser.add_argument("--fixed-node", type=int, default=None, help="Use this node count for pods-varying plot")
    parser.add_argument("--fixed-pods", type=int, default=None, help="Use this pod count for nodes-varying plot")
    args = parser.parse_args(argv)

    # Load and aggregate per algorithm
    agg_by_alg: Dict[str, pd.DataFrame] = {}
    for alg in args.algorithms:
        paths = _find_results_csvs(args.sweep_root, alg, args.latest_only)
        df = _load_results(paths, alg)
        agg_by_alg[alg] = _aggregate(df)

    # Determine common axes across algorithms
    nodes_common, pods_common = _common_axes(agg_by_alg)
    if not nodes_common or not pods_common:
        print("❌ No overlapping nodes/pods across selected algorithms. Nothing to plot.")
        return 1

    # Pick defaults if not provided: use the maximum intersection point for better visibility
    fixed_node = args.fixed_node if args.fixed_node is not None else max(nodes_common)
    fixed_pods = args.fixed_pods if args.fixed_pods is not None else max(pods_common)

    out_a = _plot_fixed_nodes(agg_by_alg, fixed_node, args.output_dir)
    out_b = _plot_fixed_pods(agg_by_alg, fixed_pods, args.output_dir)

    if out_a:
        print(f"✅ Plot saved: {out_a}")
    else:
        print("ℹ️ Skipped time vs pods (nodes fixed) due to no data.")
    if out_b:
        print(f"✅ Plot saved: {out_b}")
    else:
        print("ℹ️ Skipped time vs nodes (pods fixed) due to no data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())










