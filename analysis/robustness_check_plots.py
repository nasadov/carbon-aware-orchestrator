#!/usr/bin/env python3
"""
Generate robustness-check plots for real-trace and scalability experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALGORITHM_ORDER = [
    "Vanilla-MostAllocated",
    "GREEN-MLFQ-K8s",
    "GreenCourier-Spatial-K8s",
    "Caspian-style",
    "TotEm-OpOnly",
    "TotEm",
]

COLORS = {
    "Vanilla-MostAllocated": "#666666",
    "GREEN-MLFQ-K8s": "#31a354",
    "GreenCourier-Spatial-K8s": "#2ca25f",
    "Caspian-style": "#3182bd",
    "TotEm-OpOnly": "#e6550d",
    "TotEm": "#de2d26",
}


def ordered_algorithms(df: pd.DataFrame) -> list[str]:
    present = set(df["algorithm"])
    return [name for name in ALGORITHM_ORDER if name in present]


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_real_trace(real_root: Path, out_dir: Path) -> list[Path]:
    written: list[Path] = []
    whole = pd.read_csv(real_root / "real_trace_baseline_aggregate.csv")
    common = pd.read_csv(real_root / "real_trace_common_aggregate.csv")

    for capacity in sorted(whole["capacity_multiplier"].unique()):
        data = whole[whole["capacity_multiplier"] == capacity].copy()
        order = ordered_algorithms(data)
        data = data.set_index("algorithm").loc[order].reset_index()

        fig, ax1 = plt.subplots(figsize=(8.4, 3.6))
        x = range(len(data))
        ax1.bar(
            x,
            data["mean_per_pod_g"],
            color=[COLORS.get(a, "#777777") for a in data["algorithm"]],
            yerr=data["std_per_pod_g"].fillna(0),
            capsize=3,
            alpha=0.88,
        )
        ax1.set_ylabel("Emissions per placed pod (gCO2e)")
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(data["algorithm"], rotation=35, ha="right")
        ax1.grid(axis="y", alpha=0.25)

        ax2 = ax1.twinx()
        ax2.plot(x, data["mean_success"], color="black", marker="o", linewidth=1.4)
        ax2.set_ylabel("Success rate (%)")
        ax2.set_ylim(0, 105)
        ax1.set_title(f"Azure trace-derived workload, {capacity}x capacity")
        path = out_dir / f"azure_trace_whole_schedule_{capacity}x.pdf"
        savefig(path)
        written.append(path)

    for capacity in sorted(common["capacity_multiplier"].unique()):
        data = common[common["capacity_multiplier"] == capacity].copy()
        order = ordered_algorithms(data)
        data = data.set_index("algorithm").loc[order].reset_index()
        plt.figure(figsize=(8.4, 3.4))
        plt.bar(
            range(len(data)),
            data["mean_per_pod_g"],
            color=[COLORS.get(a, "#777777") for a in data["algorithm"]],
            yerr=data["std_per_pod_g"].fillna(0),
            capsize=3,
            alpha=0.88,
        )
        plt.ylabel("Common-pod emissions (gCO2e/pod)")
        plt.xticks(range(len(data)), data["algorithm"], rotation=35, ha="right")
        plt.grid(axis="y", alpha=0.25)
        plt.title(f"Azure trace-derived workload, common-pod view, {capacity}x capacity")
        path = out_dir / f"azure_trace_common_pod_{capacity}x.pdf"
        savefig(path)
        written.append(path)

    return written


def plot_scalability(scale_root: Path, out_dir: Path) -> list[Path]:
    written: list[Path] = []
    df = pd.read_csv(scale_root / "capacity_baseline_aggregate_by_pods.csv")
    df["nodes"] = df["capacity_multiplier"] * 4
    df["pod_node_pairs"] = df["target_pods"] * df["nodes"]
    order = ordered_algorithms(df)
    node_labels = ", ".join(str(int(n)) for n in sorted(df["nodes"].unique()))
    slope_rows = []

    plt.figure(figsize=(7.5, 3.6))
    for algorithm in order:
        data = df[df["algorithm"] == algorithm].sort_values("target_pods")
        plt.errorbar(
            data["target_pods"],
            data["mean_runtime_s"],
            yerr=data["std_runtime_s"].fillna(0),
            marker="o",
            linewidth=1.6,
            capsize=3,
            label=algorithm,
            color=COLORS.get(algorithm),
        )
    plt.xlabel(f"Pods (nodes scale proportionally: {node_labels})")
    plt.ylabel("Runtime (s)")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    path = out_dir / "paired_scalability_runtime.pdf"
    savefig(path)
    written.append(path)

    plt.figure(figsize=(7.5, 3.6))
    for algorithm in order:
        data = df[df["algorithm"] == algorithm].sort_values("pod_node_pairs")
        if len(data) >= 2 and (data["mean_runtime_s"] > 0).all():
            beta, intercept = np.polyfit(
                np.log2(data["pod_node_pairs"]),
                np.log2(data["mean_runtime_s"]),
                deg=1,
            )
            label = f"{algorithm} (slope {beta:.2f})"
            slope_rows.append(
                {
                    "algorithm": algorithm,
                    "loglog_slope_runtime_vs_pod_node_pairs": beta,
                    "intercept_log2": intercept,
                }
            )
        else:
            label = algorithm
        plt.errorbar(
            data["pod_node_pairs"],
            data["mean_runtime_s"],
            yerr=data["std_runtime_s"].fillna(0),
            marker="o",
            linewidth=1.6,
            capsize=3,
            label=label,
            color=COLORS.get(algorithm),
        )
    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    plt.xlabel("Pod-node candidate pairs per run (P x N, log scale)")
    plt.ylabel("Runtime (s, log scale)")
    plt.grid(alpha=0.25, which="both")
    plt.legend(fontsize=7, ncol=2)
    path = out_dir / "paired_scalability_runtime_loglog.pdf"
    savefig(path)
    written.append(path)
    if slope_rows:
        slope_path = out_dir / "paired_scalability_runtime_slopes.csv"
        pd.DataFrame(slope_rows).to_csv(slope_path, index=False)
        written.append(slope_path)

    plt.figure(figsize=(7.5, 3.6))
    for algorithm in order:
        data = df[df["algorithm"] == algorithm].sort_values("target_pods").copy()
        data["throughput"] = data["mean_placed_pods"] / data["mean_runtime_s"].clip(lower=1e-9)
        plt.plot(
            data["target_pods"],
            data["throughput"],
            marker="o",
            linewidth=1.6,
            label=algorithm,
            color=COLORS.get(algorithm),
        )
    plt.xlabel(f"Pods (nodes scale proportionally: {node_labels})")
    plt.ylabel("Placed pods per second")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    path = out_dir / "paired_scalability_throughput.pdf"
    savefig(path)
    written.append(path)

    plt.figure(figsize=(7.5, 3.6))
    for algorithm in order:
        data = df[df["algorithm"] == algorithm].sort_values("target_pods")
        plt.errorbar(
            data["target_pods"],
            data["mean_per_pod_g"],
            yerr=data["std_per_pod_g"].fillna(0),
            marker="o",
            linewidth=1.6,
            capsize=3,
            label=algorithm,
            color=COLORS.get(algorithm),
        )
    plt.xlabel("Pods (nodes scale proportionally: 16, 32, 64, 128)")
    plt.ylabel("Emissions per placed pod (gCO2e)")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    path = out_dir / "paired_scalability_emissions.pdf"
    savefig(path)
    written.append(path)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-trace-root")
    parser.add_argument("--scalability-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    written = []
    if args.real_trace_root:
        written.extend(plot_real_trace(Path(args.real_trace_root).resolve(), out_dir / "RealTrace"))
    written.extend(plot_scalability(Path(args.scalability_root).resolve(), out_dir / "ScalabilityStress"))
    manifest = out_dir / "plot_manifest.md"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("# Scalability Plot Manifest\n\n" + "\n".join(f"- {p}" for p in written) + "\n")
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
