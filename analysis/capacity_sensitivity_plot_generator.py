#!/usr/bin/env python3
"""
Generate capacity-sensitivity plots for paper-two resubmission.

Input is a directory produced by scripts/run_capacity_sensitivity.py.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "docs/paper-two/figures/CapacitySensitivity"

ALGORITHM_ORDER = [
    "Vanilla-LeastAllocated",
    "Vanilla-MostAllocated",
    "Piontek-Temporal-K8s",
    "Wait-Awhile",
    "GREEN-MLFQ-K8s",
    "Caspian-style",
    "TotEm-OpOnly",
    "TotEm",
]

LABELS = {
    "Vanilla-LeastAllocated": "Vanilla LeastAllocated",
    "Vanilla-MostAllocated": "Vanilla MostAllocated",
    "Piontek-Temporal-K8s": "Piontek-style",
    "Wait-Awhile": "Wait-Awhile",
    "GREEN-MLFQ-K8s": "GREEN-style",
    "Caspian-style": "Caspian-style",
    "TotEm-OpOnly": "TotEm operational-only",
    "TotEm": "TotEm",
}

STYLES = {
    "Vanilla-LeastAllocated": {"color": "#999999", "marker": "X", "linestyle": ":"},
    "Vanilla-MostAllocated": {"color": "#666666", "marker": "o", "linestyle": "-."},
    "Piontek-Temporal-K8s": {"color": "#CC79A7", "marker": "D", "linestyle": ":"},
    "Wait-Awhile": {"color": "#56B4E9", "marker": "*", "linestyle": "--"},
    "GREEN-MLFQ-K8s": {"color": "#009E73", "marker": "v", "linestyle": "--"},
    "Caspian-style": {"color": "#0072B2", "marker": "^", "linestyle": "-"},
    "TotEm-OpOnly": {"color": "#E69F00", "marker": "s", "linestyle": "--"},
    "TotEm": {"color": "#D55E00", "marker": "P", "linestyle": "-"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--target-pods", default="160,200")
    parser.add_argument(
        "--runtime-tradeoff-capacities",
        default="2,4",
        help="Capacity multipliers to include in runtime/emissions tradeoff plots.",
    )
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.45,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, name: str, formats: Iterable[str]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        fmt = fmt.strip().lstrip(".")
        if not fmt:
            continue
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def ordered_algorithms(df: pd.DataFrame) -> list[str]:
    present = set(df["algorithm"].dropna().unique())
    out = [algo for algo in ALGORITHM_ORDER if algo in present]
    out.extend(sorted(present.difference(out)))
    return out


def mark_pareto(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    frontier = []
    for _, candidate in out.iterrows():
        dominated = False
        for _, other in out.iterrows():
            if candidate["algorithm"] == other["algorithm"]:
                continue
            better_or_equal_success = other["mean_success"] >= candidate["mean_success"]
            better_or_equal_emissions = other["mean_per_pod_g"] <= candidate["mean_per_pod_g"]
            strictly_better = (
                other["mean_success"] > candidate["mean_success"]
                or other["mean_per_pod_g"] < candidate["mean_per_pod_g"]
            )
            if better_or_equal_success and better_or_equal_emissions and strictly_better:
                dominated = True
                break
        frontier.append(not dominated)
    out["pareto_front"] = frontier
    return out


def mark_minimize_pareto(df: pd.DataFrame, metrics: tuple[str, str]) -> pd.DataFrame:
    """
    Mark non-dominated rows when both metrics should be minimized.
    """
    out = df.copy()
    frontier = []
    for candidate_idx, candidate in out.iterrows():
        dominated = False
        for other_idx, other in out.iterrows():
            if candidate_idx == other_idx:
                continue
            better_or_equal = all(other[metric] <= candidate[metric] for metric in metrics)
            strictly_better = any(other[metric] < candidate[metric] for metric in metrics)
            if better_or_equal and strictly_better:
                dominated = True
                break
        frontier.append(not dominated)
    out["pareto_front"] = frontier
    return out


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    whole = pd.read_csv(root / "capacity_baseline_aggregate_by_pods.csv")
    common = pd.read_csv(root / "capacity_attributed_common_all_algorithms_aggregate.csv")
    diagnostics = pd.read_csv(root / "capacity_region_diagnostics_aggregate.csv")
    for df in (whole, common, diagnostics):
        df["capacity_multiplier"] = df["capacity_multiplier"].astype(int)
        df["target_pods"] = df["target_pods"].astype(int)
    return whole, common, diagnostics


def plot_pareto_by_capacity(
    whole: pd.DataFrame,
    common: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    target_pods: int,
) -> list[Path]:
    success = whole[whole["target_pods"] == target_pods][
        ["capacity_multiplier", "target_pods", "algorithm", "mean_success", "mean_runtime_s"]
    ]
    emissions = common[common["target_pods"] == target_pods][
        ["capacity_multiplier", "target_pods", "algorithm", "mean_common_pods", "mean_per_pod_g", "std_per_pod_g"]
    ]
    merged = emissions.merge(
        success,
        on=["capacity_multiplier", "target_pods", "algorithm"],
        how="inner",
    )
    if merged.empty:
        return []
    rows = []
    for multiplier, chunk in merged.groupby("capacity_multiplier"):
        rows.append(mark_pareto(chunk))
    pareto = pd.concat(rows, ignore_index=True)
    pareto.to_csv(output_dir / f"carbon_success_pareto_{target_pods}pods_by_capacity.csv", index=False)

    multipliers = sorted(pareto["capacity_multiplier"].unique())
    fig, axes = plt.subplots(1, len(multipliers), figsize=(4.0 * len(multipliers), 3.3), sharey=True)
    if len(multipliers) == 1:
        axes = [axes]
    x_min = max(0.0, float(pareto["mean_success"].min()) - 2.0)
    x_max = min(101.0, float(pareto["mean_success"].max()) + 2.0)
    y_min = max(0.0, float(pareto["mean_per_pod_g"].min()) - 1.0)
    y_max = float(pareto["mean_per_pod_g"].max()) + 1.4

    for ax, multiplier in zip(axes, multipliers):
        chunk = pareto[pareto["capacity_multiplier"] == multiplier]
        frontier = chunk[chunk["pareto_front"]].sort_values("mean_success")
        if len(frontier) >= 2:
            ax.plot(
                frontier["mean_success"],
                frontier["mean_per_pod_g"],
                color="#333333",
                linestyle="--",
                linewidth=1.0,
                alpha=0.65,
                zorder=1,
            )
        for _, row in chunk.iterrows():
            algo = row["algorithm"]
            style = STYLES.get(algo, {})
            ax.scatter(
                row["mean_success"],
                row["mean_per_pod_g"],
                s=72 if not bool(row["pareto_front"]) else 88,
                color=style.get("color"),
                marker=style.get("marker", "o"),
                edgecolor="#111111" if bool(row["pareto_front"]) else "white",
                linewidth=1.0 if bool(row["pareto_front"]) else 0.5,
                zorder=3,
            )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"{multiplier}x capacity")
        ax.set_xlabel("Scheduled pods (%)")
    axes[0].set_ylabel("Common-pod emissions (gCO2e/pod)")

    handles = []
    labels = []
    for algo in ordered_algorithms(pareto):
        style = STYLES.get(algo, {})
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=style.get("marker", "o"),
                color="none",
                markerfacecolor=style.get("color"),
                markeredgecolor="white",
                markersize=6,
                linestyle="none",
            )
        )
        labels.append(LABELS.get(algo, algo))
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=4,
        frameon=False,
        columnspacing=1.0,
    )
    fig.tight_layout()
    return save_figure(fig, output_dir, f"carbon_success_pareto_{target_pods}pods_by_capacity", formats)


def plot_metric_vs_capacity(
    df: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    target_pods: int,
    metric: str,
    ylabel: str,
    filename_base: str,
) -> list[Path]:
    chunk = df[df["target_pods"] == target_pods]
    if chunk.empty:
        return []
    fig, ax = plt.subplots(figsize=(5.7, 3.15))
    for algo in ordered_algorithms(chunk):
        current = chunk[chunk["algorithm"] == algo].sort_values("capacity_multiplier")
        style = STYLES.get(algo, {})
        ax.plot(
            current["capacity_multiplier"],
            current[metric],
            label=LABELS.get(algo, algo),
            color=style.get("color"),
            marker=style.get("marker", "o"),
            linestyle=style.get("linestyle", "-"),
            linewidth=1.7,
            markersize=5,
        )
    ax.set_xticks(sorted(chunk["capacity_multiplier"].unique()))
    ax.set_xlabel("Capacity multiplier")
    ax.set_ylabel(ylabel)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        ncol=3,
        frameon=False,
        columnspacing=1.0,
    )
    fig.tight_layout()
    return save_figure(fig, output_dir, f"{filename_base}_{target_pods}pods", formats)


def plot_runtime_emissions_pareto(
    whole: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    target_pods: int,
    capacity_multipliers: list[int],
) -> list[Path]:
    """
    Plot scheduler runtime against full emissions per scheduled pod.

    The relaxed-capacity experiments mostly equalize completion rate, so this
    plot uses full per-pod emissions rather than common-pod emissions. Success
    remains in the CSV and is annotated if a point is below full completion.
    """
    cols = [
        "capacity_multiplier",
        "target_pods",
        "algorithm",
        "mean_success",
        "mean_placed_pods",
        "mean_per_pod_g",
        "std_per_pod_g",
        "mean_runtime_s",
        "std_runtime_s",
    ]
    subset = whole[
        (whole["target_pods"] == target_pods)
        & (whole["capacity_multiplier"].isin(capacity_multipliers))
    ][cols].copy()
    if subset.empty:
        return []

    rows = []
    for multiplier, chunk in subset.groupby("capacity_multiplier", sort=True):
        rows.append(mark_minimize_pareto(chunk, ("mean_runtime_s", "mean_per_pod_g")))
    pareto = pd.concat(rows, ignore_index=True)
    pareto = (
        pareto.set_index("algorithm")
        .loc[
            [
                algo
                for algo in ALGORITHM_ORDER
                if algo in set(pareto["algorithm"])
            ]
        ]
        .reset_index()
        .sort_values(["capacity_multiplier", "algorithm"])
    )
    csv_path = output_dir / f"runtime_emissions_pareto_{target_pods}pods_relaxed_capacity.csv"
    pareto.to_csv(csv_path, index=False)

    multipliers = sorted(pareto["capacity_multiplier"].unique())
    fig, axes = plt.subplots(1, len(multipliers), figsize=(4.1 * len(multipliers), 3.35), sharey=True)
    if len(multipliers) == 1:
        axes = [axes]

    x_pad = max(0.1, float(pareto["mean_runtime_s"].max() - pareto["mean_runtime_s"].min()) * 0.08)
    y_pad = max(0.8, float(pareto["mean_per_pod_g"].max() - pareto["mean_per_pod_g"].min()) * 0.08)
    x_min = max(0.0, float(pareto["mean_runtime_s"].min()) - x_pad)
    x_max = float(pareto["mean_runtime_s"].max()) + x_pad
    y_min = max(0.0, float(pareto["mean_per_pod_g"].min()) - y_pad)
    y_max = float(pareto["mean_per_pod_g"].max()) + y_pad

    for ax, multiplier in zip(axes, multipliers):
        chunk = pareto[pareto["capacity_multiplier"] == multiplier]
        frontier = chunk[chunk["pareto_front"]].sort_values("mean_runtime_s")
        if len(frontier) >= 2:
            ax.plot(
                frontier["mean_runtime_s"],
                frontier["mean_per_pod_g"],
                color="#333333",
                linestyle="--",
                linewidth=1.0,
                alpha=0.65,
                zorder=1,
            )
        for _, row in chunk.iterrows():
            algo = row["algorithm"]
            style = STYLES.get(algo, {})
            is_front = bool(row["pareto_front"])
            ax.errorbar(
                row["mean_runtime_s"],
                row["mean_per_pod_g"],
                xerr=row["std_runtime_s"],
                yerr=row["std_per_pod_g"],
                fmt="none",
                ecolor=style.get("color", "#777777"),
                elinewidth=0.7,
                capsize=2,
                alpha=0.35,
                zorder=2,
            )
            ax.scatter(
                row["mean_runtime_s"],
                row["mean_per_pod_g"],
                s=86 if is_front else 70,
                color=style.get("color"),
                marker=style.get("marker", "o"),
                edgecolor="#111111" if is_front else "white",
                linewidth=1.0 if is_front else 0.5,
                zorder=3,
            )
            if row["mean_success"] < 99.95:
                ax.annotate(
                    f"{row['mean_success']:.1f}% placed",
                    (row["mean_runtime_s"], row["mean_per_pod_g"]),
                    xytext=(6, -12),
                    textcoords="offset points",
                    fontsize=6.5,
                )
        ax.text(
            0.02,
            0.04,
            "lower-left is better",
            transform=ax.transAxes,
            fontsize=6.8,
            color="#555555",
        )
        ax.set_title(f"{multiplier}x capacity")
        ax.set_xlabel("Scheduler runtime (s)")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    axes[0].set_ylabel("Emissions per scheduled pod (gCO2e/pod)")

    handles = []
    labels = []
    for algo in ordered_algorithms(pareto):
        style = STYLES.get(algo, {})
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=style.get("marker", "o"),
                color="none",
                markerfacecolor=style.get("color"),
                markeredgecolor="white",
                markersize=6,
                linestyle="none",
            )
        )
        labels.append(LABELS.get(algo, algo))
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=4,
        frameon=False,
        columnspacing=1.0,
    )
    fig.suptitle(f"{target_pods} target pods: carbon/runtime tradeoff under relaxed capacity", y=1.16)
    fig.tight_layout()
    return save_figure(fig, output_dir, f"runtime_emissions_pareto_{target_pods}pods_relaxed_capacity", formats)


def write_manifest(output_dir: Path, root: Path, written: list[Path]) -> None:
    lines = [
        "# Capacity Sensitivity Plots",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"Source experiment root: `{root}`",
        "",
        "Input aggregate files:",
        "- `capacity_baseline_aggregate_by_pods.csv`",
        "- `capacity_attributed_common_all_algorithms_aggregate.csv`",
        "- `capacity_region_diagnostics_aggregate.csv`",
        "",
        "Figures:",
    ]
    for path in sorted(written):
        lines.append(f"- `{path.name}`")
    derived_csvs = sorted(path.name for path in output_dir.glob("*.csv"))
    if derived_csvs:
        lines.extend(["", "Derived CSV files:"])
        for name in derived_csvs:
            lines.append(f"- `{name}`")
    lines.extend(
        [
            "",
            "Interpretation notes:",
            "- Pareto plots show whether improved completion comes with higher common-pod emissions.",
            "- Runtime/emissions Pareto plots use full emissions per scheduled pod for relaxed-capacity cases where completion is nearly saturated.",
            "- Low-carbon CPU-hour share and weighted carbon intensity diagnose whether schedulers actually exploit added clean capacity.",
            "- This is a sensitivity study; it should supplement, not replace, the main heterogeneous-cluster results.",
        ]
    )
    (output_dir / "plot_manifest.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    setup_matplotlib()
    root = Path(args.capacity_root).resolve()
    formats = [fmt.strip().lstrip(".") for fmt in args.formats.split(",") if fmt.strip()]
    target_pods_values = parse_int_list(args.target_pods)
    runtime_tradeoff_capacities = parse_int_list(args.runtime_tradeoff_capacities)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT / root.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    whole, common, diagnostics = load_inputs(root)
    written: list[Path] = []
    for target_pods in target_pods_values:
        written.extend(plot_pareto_by_capacity(whole, common, output_dir, formats, target_pods))
        written.extend(
            plot_metric_vs_capacity(
                common,
                output_dir,
                formats,
                target_pods,
                metric="mean_per_pod_g",
                ylabel="Common-pod emissions (gCO2e/pod)",
                filename_base="common_pod_emissions_vs_capacity",
            )
        )
        written.extend(
            plot_metric_vs_capacity(
                whole,
                output_dir,
                formats,
                target_pods,
                metric="mean_success",
                ylabel="Scheduled pods (%)",
                filename_base="success_rate_vs_capacity",
            )
        )
        written.extend(
            plot_metric_vs_capacity(
                diagnostics,
                output_dir,
                formats,
                target_pods,
                metric="mean_fr_es_cpu_hour_share_pct",
                ylabel="FR+ES CPU-hour share (%)",
                filename_base="low_carbon_cpu_hour_share_vs_capacity",
            )
        )
        written.extend(
            plot_metric_vs_capacity(
                diagnostics,
                output_dir,
                formats,
                target_pods,
                metric="mean_weighted_carbon_intensity",
                ylabel="CPU-hour weighted CI (gCO2/kWh)",
                filename_base="weighted_ci_vs_capacity",
            )
        )
        written.extend(
            plot_runtime_emissions_pareto(
                whole,
                output_dir,
                formats,
                target_pods,
                runtime_tradeoff_capacities,
            )
        )

    write_manifest(output_dir, root, written)
    print(f"Wrote {len(written)} figure files to {output_dir}")
    print(f"Wrote {output_dir / 'plot_manifest.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
