#!/usr/bin/env python3
"""
Time Complexity Overlay Slices (multi-algorithm)

Generates two comparison plots across heuristic, oracle, and carbon-agnostic baselines:
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

# Algorithm config: label, color, marker, linestyle, file key
ALGOS = {
    "heuristic": {"label": "TotEm", "color": "#ff7f0e", "marker": "s", "linestyle": "-", "key": "heuristic"},
    "global-optimal": {"label": "Oracle", "color": "#2ca02c", "marker": "o", "linestyle": "--", "key": "global_optimal"},
    "vanilla": {"label": "Carbon-Agnostic", "color": "#d62728", "marker": "^", "linestyle": "-.", "key": "vanilla"},
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


def _plot_overlay(ax, x_vals, med, std, *, color, marker, label, linestyle):
    if x_vals.empty:
        return
    med = med.clip(lower=EPS_SECONDS)
    # Ensure yerr does not push below zero on log scale; clip minimal med - EPS
    std = std.clip(lower=0.0)
    ax.errorbar(
        x_vals,
        med,
        yerr=std,
        label=label,
        color=color,
        marker=marker,
        linewidth=2.0,
        capsize=3,
        linestyle=linestyle,
        markersize=6.0,
    )


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


def _create_and_save_plot(
    alg_to_df: Dict[str, pd.DataFrame],
    filter_col: str,
    filter_val: int,
    group_col: str,
    xlabel: str,
    title_fmt: str,
    filename_base: str,
    timestamp: str,
    yscale: str
) -> None:
    """Helper to generate and save a single plot (log or linear)."""
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    
    has_data = False
    for algo, cfg in ALGOS.items():
        df = alg_to_df.get(algo)
        if df is None:
            continue
        slice_df = df[df[filter_col] == filter_val]
        if slice_df.empty:
            # print(f"ℹ️ {algo}: no data for {filter_col}={filter_val}")
            continue
        x, med, std = _aggregate_slice(slice_df, group_col=group_col)
        if not x.empty:
            has_data = True
            _plot_overlay(
                ax,
                x,
                med,
                std,
                color=cfg["color"],
                marker=cfg["marker"],
                label=cfg["label"],
                linestyle=cfg.get("linestyle", "-"),
            )

    if not has_data:
        print(f"⚠️ No data found for plot: {title_fmt.format(val=filter_val)}")
        plt.close(fig)
        return

    ax.set_xlabel(xlabel, fontsize=8)
    scale_str = "log scale" if yscale == "log" else "linear scale"
    ax.set_ylabel(f"Precomputation Time (s, {scale_str})", fontsize=8)
    ax.set_title("")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8)
    
    ax.set_yscale(yscale)
    if yscale == "log":
        ax.set_ylim(bottom=EPS_SECONDS)
    else:
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    
    # Naming convention: if log, keep original name style. If linear, append _linear.
    suffix = "_linear" if yscale == "linear" else ""
    out_png = os.path.join(OUTPUT_DIR, f"{filename_base}_{filter_val}{suffix}_{timestamp}.png")
    out_pdf = os.path.join(OUTPUT_DIR, f"{filename_base}_{filter_val}{suffix}_{timestamp}.pdf")
    
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {out_png}")


def create_overlays(fixed_pods: int, fixed_nodes: int, sweep_dir: Optional[str]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load dataframes
    alg_to_df: Dict[str, pd.DataFrame] = {}
    for algo in ALGOS.keys():
        df = _load_algo_df(algo, sweep_dir)
        if df is not None:
            alg_to_df[algo] = df

    # 1) Time vs Nodes at fixed Pods (Log & Linear)
    for scale in ["log", "linear"]:
        _create_and_save_plot(
            alg_to_df=alg_to_df,
            filter_col="total_pods",
            filter_val=fixed_pods,
            group_col="node_count",
            xlabel="Number of Nodes",
            title_fmt="Time vs Nodes (fixed pods = {val})",
            filename_base="overlay_time_vs_nodes_fixed_pods",
            timestamp=timestamp,
            yscale=scale
        )

    # 2) Time vs Pods at fixed Nodes (Log & Linear)
    for scale in ["log", "linear"]:
        _create_and_save_plot(
            alg_to_df=alg_to_df,
            filter_col="node_count",
            filter_val=fixed_nodes,
            group_col="total_pods",
            xlabel="Number of Pods",
            title_fmt="Time vs Pods (fixed nodes = {val})",
            filename_base="overlay_time_vs_pods_fixed_nodes",
            timestamp=timestamp,
            yscale=scale
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create overlay time complexity plots across algorithms.")
    p.add_argument("--fixed-pods", type=int, default=200, help="Fixed pods count for the Time vs Nodes plot (default: 200)")
    p.add_argument("--fixed-nodes", type=int, default=32, help="Fixed nodes count for the Time vs Pods plot (default: 32)")
    p.add_argument("--sweep-dir", type=str, default=None, help="Optional sweep directory to search within (defaults to all under experiments/time_complexity)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_overlays(args.fixed_pods, args.fixed_nodes, args.sweep_dir)
