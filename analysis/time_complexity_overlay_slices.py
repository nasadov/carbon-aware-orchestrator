#!/usr/bin/env python3
"""
Time Complexity Overlay Slices (multi-algorithm)

Generates two comparison plots across heuristic, global-optimal, and vanilla:
1) Time vs Nodes for a fixed number of Pods
2) Time vs Pods for a fixed number of Nodes

Data source: *_time_complexity_results.csv produced by scripts/time_complexity_sweep.py
Search scope: experiments/time_complexity/**/ (latest CSV per algorithm)

Usage:
    # Defaults: fixed pods=200, fixed nodes=32
    python analysis/time_complexity_overlay_slices.py

    # Explicit values
    python analysis/time_complexity_overlay_slices.py --fixed-pods 400 --fixed-nodes 16

    # Restrict to specific sweep dir (optional)
    python analysis/time_complexity_overlay_slices.py --sweep-dir experiments/time_complexity/sweep_shared_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = "/root/carbon-aware-orchestrator"
EXPERIMENTS_ROOT = f"{REPO_ROOT}/experiments/time_complexity"
OUTPUT_DIR = f"{REPO_ROOT}/figures/TimeComplexity"

# Algorithm config: label, color, marker, file key
ALGOS = {
    "heuristic": {"label": "Heuristic", "color": "#ff7f0e", "marker": "s", "key": "heuristic"},
    "global-optimal": {"label": "Global-Optimal", "color": "#1f77b4", "marker": "s", "key": "global_optimal"},
    "vanilla": {"label": "Vanilla", "color": "#2ca02c", "marker": "s", "key": "vanilla"},
}

# Floor to avoid log(0) and to keep tiny values visible in log scale
EPS_SECONDS = 1e-2


def _find_latest_results_csv(algorithm: str, sweep_dir: Optional[str]) -> Optional[str]:
    """Find the latest *_time_complexity_results.csv path for the given algorithm.

    Searches under experiments/time_complexity or a provided sweep_dir.
    """
    algo_key = ALGOS[algorithm]["key"]
    base = sweep_dir if sweep_dir else EXPERIMENTS_ROOT
    pattern = os.path.join(base, "**", f"{algo_key}_time_complexity_results.csv")
    candidates = [p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _aggregate_slice(df: pd.DataFrame, group_col: str) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Aggregate elapsed_seconds by group column (median ±1σ). Returns x, median, std."""
    ok = df[df["status"] == "success"].copy()
    if ok.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    grouped = (
        ok.groupby(group_col, as_index=True)["elapsed_seconds"]
          .agg(["median", "std"])
          .sort_index()
    )
    return grouped.index.astype(int), grouped["median"], grouped["std"].fillna(0.0)


def _plot_overlay(ax, x_vals, med, std, *, color, marker, label):
    if x_vals.empty:
        return
    med = med.clip(lower=EPS_SECONDS)
    # Ensure yerr does not push below zero on log scale; clip minimal med - EPS
    std = std.clip(lower=0.0)
    ax.errorbar(x_vals, med, yerr=std, label=label, color=color, marker=marker, linewidth=2, capsize=3)


def _load_algo_df(algorithm: str, sweep_dir: Optional[str]) -> Optional[pd.DataFrame]:
    csv_path = _find_latest_results_csv(algorithm, sweep_dir)
    if not csv_path:
        print(f"⚠️ No results CSV found for {algorithm}")
        return None
    try:
        df = pd.read_csv(csv_path)
        # Minimal required columns
        for c in ("node_count", "total_pods", "elapsed_seconds", "status"):
            if c not in df.columns:
                print(f"⚠️ Missing column '{c}' in {csv_path}; skipping {algorithm}")
                return None
        return df
    except Exception as e:
        print(f"⚠️ Failed to read {csv_path}: {e}")
        return None


def create_overlays(fixed_pods: int, fixed_nodes: int, sweep_dir: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load dataframes
    alg_to_df: Dict[str, pd.DataFrame] = {}
    for algo in ALGOS.keys():
        df = _load_algo_df(algo, sweep_dir)
        if df is not None:
            alg_to_df[algo] = df

    # 1) Time vs Nodes at fixed Pods
    fig1, ax1 = plt.subplots(figsize=(7.0, 4.5))
    for algo, cfg in ALGOS.items():
        df = alg_to_df.get(algo)
        if df is None:
            continue
        slice_df = df[df["total_pods"] == fixed_pods]
        if slice_df.empty:
            print(f"ℹ️ {algo}: no data for fixed pods={fixed_pods}")
            continue
        x, med, std = _aggregate_slice(slice_df, group_col="node_count")
        _plot_overlay(ax1, x, med, std, color=cfg["color"], marker=cfg["marker"], label=cfg["label"])

    ax1.set_xlabel("Number of Nodes", fontsize=12)
    ax1.set_ylabel("Precomputation Time (s, log scale)", fontsize=12)
    ax1.set_title(f"Time vs Nodes (fixed pods = {fixed_pods})", fontsize=13)
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax1.legend(fontsize=10)
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=EPS_SECONDS)
    fig1.tight_layout()
    out1_png = os.path.join(OUTPUT_DIR, f"overlay_time_vs_nodes_fixed_pods_{fixed_pods}_{timestamp}.png")
    out1_pdf = os.path.join(OUTPUT_DIR, f"overlay_time_vs_nodes_fixed_pods_{fixed_pods}_{timestamp}.pdf")
    fig1.savefig(out1_png, dpi=300, bbox_inches="tight")
    fig1.savefig(out1_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # 2) Time vs Pods at fixed Nodes
    fig2, ax2 = plt.subplots(figsize=(7.0, 4.5))
    for algo, cfg in ALGOS.items():
        df = alg_to_df.get(algo)
        if df is None:
            continue
        slice_df = df[df["node_count"] == fixed_nodes]
        if slice_df.empty:
            print(f"ℹ️ {algo}: no data for fixed nodes={fixed_nodes}")
            continue
        x, med, std = _aggregate_slice(slice_df, group_col="total_pods")
        _plot_overlay(ax2, x, med, std, color=cfg["color"], marker=cfg["marker"], label=cfg["label"])

    ax2.set_xlabel("Number of Pods", fontsize=12)
    ax2.set_ylabel("Precomputation Time (s, log scale)", fontsize=12)
    ax2.set_title(f"Time vs Pods (fixed nodes = {fixed_nodes})", fontsize=13)
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax2.legend(fontsize=10)
    ax2.set_yscale('log')
    ax2.set_ylim(bottom=EPS_SECONDS)
    fig2.tight_layout()
    out2_png = os.path.join(OUTPUT_DIR, f"overlay_time_vs_pods_fixed_nodes_{fixed_nodes}_{timestamp}.png")
    out2_pdf = os.path.join(OUTPUT_DIR, f"overlay_time_vs_pods_fixed_nodes_{fixed_nodes}_{timestamp}.pdf")
    fig2.savefig(out2_png, dpi=300, bbox_inches="tight")
    fig2.savefig(out2_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    print(f"✅ Saved: {out1_png}")
    print(f"✅ Saved: {out2_png}")
    return out1_png, out2_png


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create overlay time complexity plots across algorithms.")
    p.add_argument("--fixed-pods", type=int, default=200, help="Fixed pods count for the Time vs Nodes plot (default: 200)")
    p.add_argument("--fixed-nodes", type=int, default=32, help="Fixed nodes count for the Time vs Pods plot (default: 32)")
    p.add_argument("--sweep-dir", type=str, default=None, help="Optional sweep directory to search within (defaults to all under experiments/time_complexity)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_overlays(args.fixed_pods, args.fixed_nodes, args.sweep_dir)


