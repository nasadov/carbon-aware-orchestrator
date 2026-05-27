#!/usr/bin/env python3
"""
Generate composite evaluation figures from aggregate experiment outputs.

The ordinary plot generators keep the full diagnostic surface. This script
creates a smaller set of manuscript-ready composite figures: baseline tradeoff,
embodied ablation, capacity mechanism, and robustness.
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

DEFAULT_MATRIX_ROOT = (
    REPO_ROOT
    / "experiments"
    / "resubmission_baseline_matrix_sweep_v5"
    / "matrix_20260515_111056"
)
DEFAULT_CAPACITY_ROOT = (
    REPO_ROOT
    / "experiments"
    / "resubmission_capacity_sensitivity_v4"
    / "capacity_20260516_211508"
)
DEFAULT_REAL_TRACE_ROOT = (
    REPO_ROOT
    / "experiments"
    / "real_trace_baseline_matrix"
    / "azure_trace_20260522_120352"
)
DEFAULT_SCALE_ROOT = (
    REPO_ROOT
    / "experiments"
    / "scalability_stress_core"
    / "capacity_20260523_095852"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "docs" / "paper-two" / "figures" / "CompositeEvaluation" / "current"
)

ALGORITHM_ORDER = [
    "Vanilla-MostAllocated",
    "GREEN-MLFQ-K8s",
    "GreenCourier-Spatial-K8s",
    "Caspian-style",
    "TotEm-OpOnly",
    "TotEm",
]

ROBUSTNESS_ORDER = [
    "Vanilla-MostAllocated",
    "GreenCourier-Spatial-K8s",
    "GREEN-MLFQ-K8s",
    "Caspian-style",
    "TotEm-OpOnly",
    "TotEm",
]

LABELS = {
    "Vanilla-MostAllocated": "Vanilla MostAllocated",
    "GREEN-MLFQ-K8s": "GREEN-style",
    "GreenCourier-Spatial-K8s": "GreenCourier-style",
    "Caspian-style": "Caspian-style",
    "TotEm-OpOnly": "TotEm operational-only",
    "TotEm": "TotEm",
}

SHORT_LABELS = {
    "Vanilla-MostAllocated": "V. Most",
    "GREEN-MLFQ-K8s": "GREEN",
    "GreenCourier-Spatial-K8s": "GreenCourier",
    "Caspian-style": "Caspian",
    "TotEm-OpOnly": "TotEm op-only",
    "TotEm": "TotEm",
}

STYLES = {
    "Vanilla-MostAllocated": {"color": "#5F6368", "marker": "o", "linestyle": "-."},
    "GREEN-MLFQ-K8s": {"color": "#009E73", "marker": "v", "linestyle": "--"},
    "GreenCourier-Spatial-K8s": {"color": "#117733", "marker": "h", "linestyle": "-."},
    "Caspian-style": {"color": "#0072B2", "marker": "^", "linestyle": "-"},
    "TotEm-OpOnly": {"color": "#E69F00", "marker": "s", "linestyle": "--"},
    "TotEm": {"color": "#D55E00", "marker": "P", "linestyle": "-"},
}

FOCUS_ALGORITHMS = set(ALGORITHM_ORDER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", default=str(DEFAULT_MATRIX_ROOT))
    parser.add_argument("--capacity-root", default=str(DEFAULT_CAPACITY_ROOT))
    parser.add_argument("--real-trace-root", default=str(DEFAULT_REAL_TRACE_ROOT))
    parser.add_argument("--scalability-root", default=str(DEFAULT_SCALE_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--formats", default="pdf,png")
    return parser.parse_args()


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 7.6,
            "axes.labelsize": 7.8,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "figure.titlesize": 8.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.38,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_matrix(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    whole = read_csv(root / "baseline_comparison_aggregate_by_pods.csv")
    common = read_csv(root / "attributed_common_all_algorithms_aggregate.csv")
    for df in (whole, common):
        df["target_pods"] = df["target_pods"].astype(int)
    return whole, common


def load_capacity(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    whole = read_csv(root / "capacity_baseline_aggregate_by_pods.csv")
    common = read_csv(root / "capacity_attributed_common_all_algorithms_aggregate.csv")
    diagnostics = read_csv(root / "capacity_region_diagnostics_aggregate.csv")
    deltas = read_csv(root / "capacity_paired_common_emission_deltas_vs_totem.csv")
    for df in (whole, common, diagnostics, deltas):
        if "capacity_multiplier" in df.columns:
            df["capacity_multiplier"] = df["capacity_multiplier"].astype(int)
        if "target_pods" in df.columns:
            df["target_pods"] = df["target_pods"].astype(int)
    return whole, common, diagnostics, deltas


def ordered_algorithms(df: pd.DataFrame, order: list[str] | None = None) -> list[str]:
    present = set(df["algorithm"].dropna().unique())
    selected_order = order or ALGORITHM_ORDER
    return [algo for algo in selected_order if algo in present]


def formats_from_text(text: str) -> list[str]:
    return [part.strip().lstrip(".") for part in text.split(",") if part.strip()]


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    name: str,
    formats: Iterable[str],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        written.append(path)
    plt.close(fig)
    return written


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="top",
    )


def lineplot(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    order: list[str],
    err_col: str | None = None,
    focus_only: bool = False,
) -> None:
    for algo in order:
        chunk = df[df["algorithm"] == algo].sort_values(x_col)
        if chunk.empty:
            continue
        style = STYLES.get(algo, {})
        is_focus = algo in FOCUS_ALGORITHMS
        if focus_only and not is_focus:
            alpha = 0.42
            linewidth = 1.0
            markersize = 3.7
        else:
            alpha = 0.96 if is_focus else 0.72
            linewidth = 2.05 if is_focus else 1.25
            markersize = 4.8 if is_focus else 4.0
        yerr = chunk[err_col].fillna(0.0) if err_col and err_col in chunk.columns else None
        ax.errorbar(
            chunk[x_col],
            chunk[y_col],
            yerr=yerr,
            color=style.get("color"),
            marker=style.get("marker", "o"),
            linestyle=style.get("linestyle", "-"),
            linewidth=linewidth,
            markersize=markersize,
            capsize=2 if yerr is not None else 0,
            elinewidth=0.65,
            alpha=alpha,
            label=LABELS.get(algo, algo),
        )


def legend_handles(order: list[str]) -> tuple[list[plt.Line2D], list[str]]:
    handles = []
    labels = []
    for algo in order:
        style = STYLES.get(algo, {})
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=style.get("marker", "o"),
                color=style.get("color", "#777777"),
                linestyle=style.get("linestyle", "-"),
                linewidth=1.6,
                markersize=5,
            )
        )
        labels.append(LABELS.get(algo, algo))
    return handles, labels


def mark_success_emission_pareto(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    frontier = []
    for _, candidate in out.iterrows():
        dominated = False
        for _, other in out.iterrows():
            if candidate["algorithm"] == other["algorithm"]:
                continue
            better_success = other["mean_success"] >= candidate["mean_success"]
            better_emissions = other["mean_per_pod_g"] <= candidate["mean_per_pod_g"]
            strict = (
                other["mean_success"] > candidate["mean_success"]
                or other["mean_per_pod_g"] < candidate["mean_per_pod_g"]
            )
            if better_success and better_emissions and strict:
                dominated = True
                break
        frontier.append(not dominated)
    out["pareto_front"] = frontier
    return out


def plot_baseline_composite(
    whole: pd.DataFrame,
    common: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
) -> list[Path]:
    order = ordered_algorithms(whole)
    fig = plt.figure(figsize=(7.25, 3.75))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.25, 1.02, 1.05],
        left=0.07,
        right=0.99,
        bottom=0.18,
        top=0.76,
        wspace=0.42,
    )
    ax_em = fig.add_subplot(gs[0, 0])
    ax_success = fig.add_subplot(gs[0, 1])
    ax_pareto = fig.add_subplot(gs[0, 2])

    lineplot(
        ax_em,
        common,
        "target_pods",
        "mean_per_pod_g",
        order=order,
        err_col="std_per_pod_g",
        focus_only=True,
    )
    ax_em.set_xlabel("Target pods")
    ax_em.set_ylabel("Common-pod emissions (gCO2e/pod)")
    ax_em.set_xticks(sorted(common["target_pods"].unique()))
    ax_em.set_title("Same pod IDs")
    panel_label(ax_em, "a")

    lineplot(
        ax_success,
        whole,
        "target_pods",
        "mean_success",
        order=order,
        focus_only=True,
    )
    ax_success.set_xlabel("Target pods")
    ax_success.set_ylabel("Scheduled pods (%)")
    ax_success.set_xticks(sorted(whole["target_pods"].unique()))
    ax_success.set_ylim(max(0, np.floor((whole["mean_success"].min() - 2) / 5) * 5), 101.2)
    ax_success.set_title("Completion")
    panel_label(ax_success, "b")

    success = whole[whole["target_pods"] == 200][
        ["target_pods", "algorithm", "mean_success", "mean_runtime_s"]
    ]
    emissions = common[common["target_pods"] == 200][
        ["target_pods", "algorithm", "mean_per_pod_g", "std_per_pod_g"]
    ]
    merged = emissions.merge(success, on=["target_pods", "algorithm"], how="inner")
    merged = (
        merged.set_index("algorithm")
        .reindex([algo for algo in ALGORITHM_ORDER if algo in set(merged["algorithm"])])
        .reset_index()
    )
    merged = mark_success_emission_pareto(merged)
    frontier = merged[merged["pareto_front"]].sort_values("mean_success")
    if len(frontier) >= 2:
        ax_pareto.plot(
            frontier["mean_success"],
            frontier["mean_per_pod_g"],
            color="#3F3F3F",
            linestyle="--",
            linewidth=1.0,
            zorder=1,
        )
    label_offsets = {
        "Vanilla-MostAllocated": (5, 5),
        "GreenCourier-Spatial-K8s": (5, -15),
        "GREEN-MLFQ-K8s": (-10, -13),
        "Caspian-style": (5, 4),
        "TotEm-OpOnly": (5, 2),
        "TotEm": (5, -11),
    }
    for _, row in merged.iterrows():
        algo = row["algorithm"]
        style = STYLES.get(algo, {})
        is_front = bool(row["pareto_front"])
        ax_pareto.scatter(
            row["mean_success"],
            row["mean_per_pod_g"],
            s=56 if is_front else 42,
            color=style.get("color"),
            marker=style.get("marker", "o"),
            edgecolor="#111111" if is_front else "white",
            linewidth=0.8 if is_front else 0.45,
            alpha=0.96 if algo in FOCUS_ALGORITHMS else 0.78,
            zorder=3,
        )
        ax_pareto.annotate(
            SHORT_LABELS.get(algo, algo),
            (row["mean_success"], row["mean_per_pod_g"]),
            xytext=label_offsets.get(algo, (4, 4)),
            textcoords="offset points",
            fontsize=6.5,
        )
    ax_pareto.set_xlabel("Scheduled pods (%)")
    ax_pareto.set_ylabel("Common-pod emissions")
    ax_pareto.set_title("200-pod tradeoff")
    ax_pareto.set_xlim(max(0, merged["mean_success"].min() - 1.4), min(101, merged["mean_success"].max() + 1.8))
    ax_pareto.set_ylim(max(0, merged["mean_per_pod_g"].min() - 0.8), merged["mean_per_pod_g"].max() + 0.9)
    panel_label(ax_pareto, "c")

    handles, labels = legend_handles(order)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.98),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.0,
    )
    return save_figure(fig, output_dir, "composite_baseline_matrix_200pods", formats)


def plot_capacity_mechanism_composite(
    common: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
) -> list[Path]:
    common_200 = common[common["target_pods"] == 200].copy()
    diagnostics_200 = diagnostics[diagnostics["target_pods"] == 200].copy()
    order = ordered_algorithms(common_200)

    fig = plt.figure(figsize=(7.25, 3.35))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.15, 1.0, 1.0],
        left=0.07,
        right=0.99,
        bottom=0.18,
        top=0.75,
        wspace=0.42,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    lineplot(
        axes[0],
        common_200,
        "capacity_multiplier",
        "mean_per_pod_g",
        order=order,
        err_col="std_per_pod_g",
        focus_only=True,
    )
    axes[0].set_ylabel("Common-pod emissions (gCO2e/pod)")
    axes[0].set_title("Carbon outcome")
    axes[0].set_xticks([1, 2, 4])
    panel_label(axes[0], "a")

    lineplot(
        axes[1],
        diagnostics_200,
        "capacity_multiplier",
        "mean_fr_es_cpu_hour_share_pct",
        order=order,
        focus_only=True,
    )
    axes[1].set_ylabel("FR+ES CPU-hour share (%)")
    axes[1].set_title("Clean-region use")
    axes[1].set_xticks([1, 2, 4])
    panel_label(axes[1], "b")

    lineplot(
        axes[2],
        diagnostics_200,
        "capacity_multiplier",
        "mean_weighted_carbon_intensity",
        order=order,
        focus_only=True,
    )
    axes[2].set_ylabel("CPU-hour weighted CI (gCO2/kWh)")
    axes[2].set_title("Selected-grid cleanliness")
    axes[2].set_xticks([1, 2, 4])
    panel_label(axes[2], "c")

    for ax in axes:
        ax.set_xlabel("Capacity multiplier")

    handles, labels = legend_handles(order)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.98),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.0,
    )
    return save_figure(fig, output_dir, "composite_capacity_mechanism_200pods", formats)


def plot_embodied_ablation_composite(
    matrix_common: pd.DataFrame,
    capacity_whole: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
) -> list[Path]:
    fig = plt.figure(figsize=(7.25, 3.05))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.18, 0.95, 1.0],
        left=0.08,
        right=0.99,
        bottom=0.20,
        top=0.84,
        wspace=0.44,
    )
    ax_left = fig.add_subplot(gs[0, 0])
    ax_mid = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[0, 2])

    subset = matrix_common[matrix_common["algorithm"].isin(["TotEm", "TotEm-OpOnly"])]
    lineplot(
        ax_left,
        subset,
        "target_pods",
        "mean_per_pod_g",
        order=["TotEm-OpOnly", "TotEm"],
        err_col="std_per_pod_g",
    )
    ax_left.set_xlabel("Target pods")
    ax_left.set_ylabel("Common-pod emissions (gCO2e/pod)")
    ax_left.set_title("Operational-only ablation")
    ax_left.set_xticks(sorted(subset["target_pods"].unique()))
    panel_label(ax_left, "a")

    component_order = ["TotEm-OpOnly", "TotEm"]
    components = capacity_whole[
        (capacity_whole["capacity_multiplier"] == 4)
        & (capacity_whole["target_pods"] == 200)
        & (capacity_whole["algorithm"].isin(component_order))
    ].copy()
    components = components.set_index("algorithm").loc[component_order].reset_index()
    components["operational_g_per_pod"] = (
        components["mean_operational_kg"] * 1000.0 / components["mean_placed_pods"]
    )
    components["embodied_g_per_pod"] = (
        components["mean_embodied_kg"] * 1000.0 / components["mean_placed_pods"]
    )
    components["total_g_per_pod"] = components["mean_per_pod_g"]

    x = np.arange(len(component_order))
    width = 0.56
    op_color = "#4C78A8"
    embodied_color = "#D55E00"
    ax_mid.bar(
        x,
        components["operational_g_per_pod"],
        width=width,
        color=op_color,
        edgecolor="white",
        linewidth=0.5,
        label="Operational",
    )
    ax_mid.bar(
        x,
        components["embodied_g_per_pod"],
        width=width,
        bottom=components["operational_g_per_pod"],
        color=embodied_color,
        edgecolor="white",
        linewidth=0.5,
        label="Embodied",
    )

    totals = components["total_g_per_pod"].to_numpy()
    for xpos, total in zip(x, totals):
        ax_mid.text(
            xpos,
            total + 0.22,
            f"{total:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.9,
            fontweight="bold",
        )

    benefit = totals[0] - totals[1]
    bracket_x = x[-1] + 0.55
    ax_mid.plot(
        [bracket_x, bracket_x],
        [totals[1], totals[0]],
        color="#333333",
        linewidth=0.8,
        clip_on=False,
    )
    ax_mid.plot(
        [bracket_x - 0.05, bracket_x + 0.05],
        [totals[0], totals[0]],
        color="#333333",
        linewidth=0.8,
        clip_on=False,
    )
    ax_mid.plot(
        [bracket_x - 0.05, bracket_x + 0.05],
        [totals[1], totals[1]],
        color="#333333",
        linewidth=0.8,
        clip_on=False,
    )
    ax_mid.text(
        bracket_x + 0.08,
        (totals[0] + totals[1]) / 2.0,
        f"-{benefit:.2f}",
        ha="left",
        va="center",
        fontsize=6.9,
        fontweight="bold",
        color="#333333",
    )

    ax_mid.set_xticks(x)
    ax_mid.set_xticklabels(["Op-only", "Embodied-aware"], rotation=15, ha="right")
    ax_mid.set_xlim(-0.55, 1.92)
    ax_mid.set_ylabel("Emissions at 4x/200 (gCO2e/pod)")
    ax_mid.set_title("Component split")
    ax_mid.legend(frameon=False, loc="lower right", fontsize=6.7)
    panel_label(ax_mid, "b")

    rate_factors = np.array([0.50, 2.0 / 3.0, 1.00, 1.50, 2.00])
    rate_labels = ["0.5x", "0.67x", "1x", "1.5x", "2x"]
    op_only = components.set_index("algorithm").loc["TotEm-OpOnly"]
    totem = components.set_index("algorithm").loc["TotEm"]
    op_only_total = op_only["operational_g_per_pod"] + rate_factors * op_only["embodied_g_per_pod"]
    totem_total = totem["operational_g_per_pod"] + rate_factors * totem["embodied_g_per_pod"]
    sensitivity = pd.DataFrame(
        {
            "embodied_rate_multiplier": rate_factors,
            "totem_oponly_g_per_pod": op_only_total,
            "totem_g_per_pod": totem_total,
            "totem_reduction_g_per_pod": op_only_total - totem_total,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(output_dir / "composite_embodied_lifecycle_sensitivity.csv", index=False)

    ax_right.axhline(0.0, color="#4d4d4d", linewidth=0.7)
    ax_right.plot(
        sensitivity["embodied_rate_multiplier"],
        sensitivity["totem_reduction_g_per_pod"],
        color=STYLES["TotEm"]["color"],
        marker=STYLES["TotEm"]["marker"],
        linewidth=1.8,
        markersize=4.8,
    )
    ax_right.fill_between(
        sensitivity["embodied_rate_multiplier"],
        0.0,
        sensitivity["totem_reduction_g_per_pod"],
        color=STYLES["TotEm"]["color"],
        alpha=0.13,
    )
    min_delta = sensitivity["totem_reduction_g_per_pod"].min()
    max_delta = sensitivity["totem_reduction_g_per_pod"].max()
    ax_right.text(
        rate_factors[0] * 1.04,
        max_delta - 0.03,
        f"positive across range\n{min_delta:.2f}-{max_delta:.2f} g/pod",
        ha="left",
        va="top",
        fontsize=6.6,
        color="#333333",
    )
    ax_right.set_xscale("log", base=2)
    ax_right.set_xlim(rate_factors[0] / 1.08, rate_factors[-1] * 1.08)
    ax_right.set_xlabel("Embodied-rate multiplier")
    ax_right.set_ylabel("TotEm reduction vs op-only (g/pod)")
    ax_right.set_title("Lifecycle-rate sensitivity")
    ax_right.set_xticks(rate_factors)
    ax_right.set_xticklabels(rate_labels)
    ax_right.minorticks_off()
    ax_right.set_ylim(0, max_delta * 1.22)
    panel_label(ax_right, "c")

    handles, labels = legend_handles(["TotEm-OpOnly", "TotEm"])
    ax_left.legend(handles, labels, frameon=False, loc="upper right", fontsize=6.8)
    return save_figure(fig, output_dir, "composite_embodied_ablation", formats)


def plot_robustness_composite(
    real_trace_root: Path,
    scale_root: Path,
    output_dir: Path,
    formats: Iterable[str],
) -> list[Path]:
    trace_common = read_csv(real_trace_root / "real_trace_common_aggregate.csv")
    scale = read_csv(scale_root / "capacity_baseline_aggregate_by_pods.csv")
    trace_common["capacity_multiplier"] = trace_common["capacity_multiplier"].astype(int)
    trace_common["target_pods"] = trace_common["target_pods"].astype(int)
    scale["target_pods"] = scale["target_pods"].astype(int)
    scale["capacity_multiplier"] = scale["capacity_multiplier"].astype(int)
    scale["nodes"] = scale["capacity_multiplier"] * 4

    order = [algo for algo in ROBUSTNESS_ORDER if algo in set(scale["algorithm"])]
    fig = plt.figure(figsize=(7.25, 4.75))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.07,
        right=0.99,
        bottom=0.13,
        top=0.82,
        wspace=0.34,
        hspace=0.62,
    )
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    for ax, capacity, label in [(axes[0], 1, "a"), (axes[1], 4, "b")]:
        chunk = trace_common[
            (trace_common["capacity_multiplier"] == capacity)
            & (trace_common["algorithm"].isin(order))
        ].copy()
        chunk = chunk.set_index("algorithm").loc[order].reset_index()
        y = np.arange(len(chunk))
        ax.barh(
            y,
            chunk["mean_per_pod_g"],
            xerr=chunk["std_per_pod_g"].fillna(0),
            color=[STYLES[a]["color"] for a in chunk["algorithm"]],
            edgecolor="white",
            linewidth=0.45,
            capsize=2.5,
        )
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT_LABELS[a] for a in chunk["algorithm"]])
        ax.invert_yaxis()
        ax.set_xlabel("Common-pod emissions")
        ax.set_title(f"Azure trace-derived, {capacity}x")
        panel_label(ax, label)

    for ax, metric, ylabel, title, label in [
        (axes[2], "mean_runtime_s", "Runtime (s)", "Paired scalability runtime", "c"),
        (axes[3], "mean_per_pod_g", "Emissions per pod", "Paired scalability emissions", "d"),
    ]:
        for algo in order:
            chunk = scale[scale["algorithm"] == algo].sort_values("target_pods")
            if chunk.empty:
                continue
            style = STYLES.get(algo, {})
            err_col = "std_runtime_s" if metric == "mean_runtime_s" else "std_per_pod_g"
            ax.errorbar(
                chunk["target_pods"],
                chunk[metric],
                yerr=chunk[err_col].fillna(0),
                color=style.get("color"),
                marker=style.get("marker", "o"),
                linestyle=style.get("linestyle", "-"),
                linewidth=1.55 if algo in FOCUS_ALGORITHMS else 1.15,
                markersize=4.4,
                capsize=2,
                alpha=0.92,
            )
        node_labels = f"{int(scale['nodes'].min())} to {int(scale['nodes'].max())}"
        ax.set_xlabel(f"Pods (nodes scale {node_labels})")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(sorted(scale["target_pods"].unique()))
        panel_label(ax, label)

    handles, labels = legend_handles(order)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.98),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.0,
    )
    return save_figure(fig, output_dir, "composite_robustness_scalability", formats)


def write_manifest(output_dir: Path, sources: dict[str, Path], written: list[Path]) -> None:
    lines = [
        "# Composite Evaluation Figures",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
        "",
        "Sources:",
    ]
    for key, path in sources.items():
        lines.append(f"- {key}: `{path}`")
    lines.extend(["", "Figures:"])
    for path in sorted(written):
        lines.append(f"- `{path.name}`")
    lines.extend(
        [
            "",
            "Figure order:",
            "1. `composite_baseline_matrix_200pods`: core baselines and the tight-capacity carbon/completion tradeoff.",
            "2. `composite_embodied_ablation`: operational-only ablation and the measured embodied-objective effect.",
            "3. `composite_capacity_mechanism_200pods`: relaxed-capacity spatial carbon mechanism.",
            "4. `composite_robustness_scalability`: trace-derived and paired pod/node scalability checks.",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plot_manifest.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    setup_matplotlib()

    matrix_root = Path(args.matrix_root).resolve()
    capacity_root = Path(args.capacity_root).resolve()
    real_trace_root = Path(args.real_trace_root).resolve()
    scale_root = Path(args.scalability_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    formats = formats_from_text(args.formats)

    matrix_whole, matrix_common = load_matrix(matrix_root)
    capacity_whole, capacity_common, capacity_diagnostics, capacity_deltas = load_capacity(capacity_root)
    del capacity_deltas

    written: list[Path] = []
    written.extend(plot_baseline_composite(matrix_whole, matrix_common, output_dir, formats))
    written.extend(
        plot_embodied_ablation_composite(
            matrix_common,
            capacity_whole,
            output_dir,
            formats,
        )
    )
    written.extend(
        plot_capacity_mechanism_composite(
            capacity_common,
            capacity_diagnostics,
            output_dir,
            formats,
        )
    )
    written.extend(plot_robustness_composite(real_trace_root, scale_root, output_dir, formats))
    write_manifest(
        output_dir,
        {
            "baseline matrix": matrix_root,
            "capacity sensitivity": capacity_root,
            "real trace": real_trace_root,
            "paired scalability": scale_root,
        },
        written,
    )
    print(f"Wrote {len(written)} files to {output_dir}")
    print(output_dir / "plot_manifest.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
