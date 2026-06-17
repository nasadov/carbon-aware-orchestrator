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
import re
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from attributed_common_pod_comparison import attribute_pod_emissions, summarize_subset
from baseline_comparison_summary import find_placement_csv, infer_algorithm, load_forecasts, load_nodes


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MATRIX_ROOT = (
    REPO_ROOT
    / "experiments"
    / "resubmission_baseline_matrix_merged_n10"
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
    / "real_trace_baseline_matrix_12h"
    / "azure_trace_20260611_083605"
)
DEFAULT_SCALE_ROOT = (
    REPO_ROOT
    / "experiments"
    / "scalability_stress_core"
    / "capacity_20260523_095852"
)
DEFAULT_REGIME_SWEEP_ROOT = (
    REPO_ROOT
    / "experiments"
    / "workload_regime_sweep"
    / "sweep_20260609_145314"
)
DEFAULT_REGIME_FOLLOWUP_ROOT = (
    REPO_ROOT
    / "experiments"
    / "workload_regime_baseline_followup_merged_n10"
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
    "TotEm-OpOnly": "TotEm-OpOnly",
    "TotEm": "TotEm",
}

STYLES = {
    "Vanilla-MostAllocated": {"color": "#5F6368", "marker": "o", "linestyle": (0, (4, 1.5, 1, 1.5))},
    "GREEN-MLFQ-K8s": {"color": "#009E73", "marker": "v", "linestyle": (0, (5, 2))},
    "GreenCourier-Spatial-K8s": {"color": "#117733", "marker": "h", "linestyle": (0, (4, 1.5, 1, 1.5))},
    "Caspian-style": {"color": "#0072B2", "marker": "^", "linestyle": "-"},
    "TotEm-OpOnly": {"color": "#E69F00", "marker": "s", "linestyle": (0, (5, 2))},
    "TotEm": {"color": "#D55E00", "marker": "P", "linestyle": "-"},
}

FOCUS_ALGORITHMS = set(ALGORITHM_ORDER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", default=str(DEFAULT_MATRIX_ROOT))
    parser.add_argument("--capacity-root", default=str(DEFAULT_CAPACITY_ROOT))
    parser.add_argument("--real-trace-root", default=str(DEFAULT_REAL_TRACE_ROOT))
    parser.add_argument("--scalability-root", default=str(DEFAULT_SCALE_ROOT))
    parser.add_argument("--regime-sweep-root", default=str(DEFAULT_REGIME_SWEEP_ROOT))
    parser.add_argument("--regime-followup-root", default=str(DEFAULT_REGIME_FOLLOWUP_ROOT))
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


def add_ci95(df: pd.DataFrame, std_col: str, out_col: str, n_col: str = "runs", n_default: int = 5) -> pd.DataFrame:
    """Add a 95% confidence-interval half-width column (t-based) from a std column.

    CI95 = t_{0.975, n-1} * std / sqrt(n). Falls back to n_default when the count
    column is absent. This is the half-width plotted as error bars; significance of
    pairwise TotEm-vs-baseline gaps is assessed separately by paired tests.
    """
    df = df.copy()
    if n_col in df.columns:
        n = pd.to_numeric(df[n_col], errors="coerce").fillna(n_default)
    else:
        n = pd.Series(n_default, index=df.index, dtype=float)
    n = n.clip(lower=2)
    df[out_col] = stats.t.ppf(0.975, n - 1) * df[std_col].fillna(0.0) / np.sqrt(n)
    return df


def load_matrix(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    whole = read_csv(root / "baseline_comparison_aggregate_by_pods.csv")
    common = read_csv(root / "attributed_common_all_algorithms_aggregate.csv")
    for df in (whole, common):
        df["target_pods"] = df["target_pods"].astype(int)
    return whole, common


def paired_pvalues_vs_totem(root: Path, target_pods: int = 200) -> dict[str, float]:
    """Paired t-test p-values of each algorithm vs TotEm on the per-seed common
    cohort at a given density. Returns {} if the per-seed file is absent so the
    figure still renders without significance marks."""
    path = root / "attributed_common_pod_comparison.csv"
    if not path.exists():
        return {}
    df = read_csv(path)
    df = df[(df["comparison"] == "all_algorithms_common") & (df["target_pods"] == target_pods)]
    piv = df.pivot_table(index="seed", columns="algorithm", values="per_pod_g")
    if "TotEm" not in piv.columns:
        return {}
    out: dict[str, float] = {}
    for algo in piv.columns:
        if algo == "TotEm":
            continue
        pair = piv[[algo, "TotEm"]].dropna()
        if len(pair) >= 2:
            out[algo] = float(stats.ttest_rel(pair[algo], pair["TotEm"]).pvalue)
    return out


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


def subset_region_diagnostics(
    placements: pd.DataFrame,
    pod_ids: set[str],
    nodes: dict[str, dict],
    forecasts: dict[str, dict[int, float]],
) -> dict[str, float]:
    """Measure spatial placement behavior for a fixed subset of pods."""
    subset = placements[placements["pod_id"].astype(str).isin(pod_ids)].copy()
    region_cpu_hours: dict[str, float] = {}
    weighted_ci_sum = 0.0
    total_cpu_hours = 0.0

    for _, row in subset.iterrows():
        node_id = str(row.get("node_id"))
        if node_id not in nodes:
            continue
        try:
            start = int(float(row.get("start_slot", 0)))
            duration = max(1, int(float(row.get("duration", 1))))
            cpu = max(0.0, float(row.get("cpu_request", 0.0)))
        except (TypeError, ValueError):
            continue
        region = str(nodes[node_id]["region"]).upper()
        for slot in range(start, start + duration):
            region_cpu_hours[region] = region_cpu_hours.get(region, 0.0) + cpu
            total_cpu_hours += cpu
            weighted_ci_sum += forecasts.get(region, {}).get(slot, 200.0) * cpu

    return {
        "fr_es_cpu_hour_share_pct": 100.0
        * (region_cpu_hours.get("FR", 0.0) + region_cpu_hours.get("ES", 0.0))
        / max(total_cpu_hours, 1e-9),
        "weighted_carbon_intensity": weighted_ci_sum / max(total_cpu_hours, 1e-9),
    }


def load_fixed_capacity_cohort(
    root: Path,
    output_dir: Path,
    target_pods: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Re-attribute one six-policy pod cohort across all capacity settings.

    Connected lines in the capacity figure represent repeated measurements only
    when every point uses the same work. The cohort is therefore intersected
    across the six reported policies and every available capacity multiplier
    within each seed.
    """
    records: list[dict] = []
    capacities: list[int] = []
    for capacity_dir in sorted(root.glob("capacity_*x")):
        capacity_match = re.fullmatch(r"capacity_(\d+)x", capacity_dir.name)
        if not capacity_match:
            continue
        capacity = int(capacity_match.group(1))
        capacities.append(capacity)
        for combo_dir in sorted(capacity_dir.glob(f"seed_*_pods_{target_pods}")):
            seed_match = re.fullmatch(rf"seed_(\d+)_pods_{target_pods}", combo_dir.name)
            if not seed_match:
                continue
            seed = int(seed_match.group(1))
            nodes = load_nodes(combo_dir / "nodes.yaml")
            forecasts = load_forecasts(combo_dir / "all_forecasts.json")
            for exp_dir in sorted(path for path in combo_dir.iterdir() if path.is_dir()):
                placement_csv = find_placement_csv(exp_dir)
                if not placement_csv:
                    continue
                algorithm = infer_algorithm(exp_dir, placement_csv)
                if algorithm not in ALGORITHM_ORDER:
                    continue
                placements = pd.read_csv(placement_csv)
                records.append(
                    {
                        "seed": seed,
                        "capacity_multiplier": capacity,
                        "algorithm": algorithm,
                        "placement_csv": placement_csv,
                        "placements": placements,
                        "pod_ids": set(placements["pod_id"].astype(str)),
                        "nodes": nodes,
                        "forecasts": forecasts,
                    }
                )

    capacities = sorted(set(capacities))
    if not records or not capacities:
        raise RuntimeError(f"No capacity placement runs found under {root}")

    detail_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    cohort_rows: list[dict] = []
    records_by_seed: dict[int, list[dict]] = {}
    for record in records:
        records_by_seed.setdefault(record["seed"], []).append(record)

    expected_pairs = {(capacity, algorithm) for capacity in capacities for algorithm in ALGORITHM_ORDER}
    for seed, seed_records in sorted(records_by_seed.items()):
        actual_pairs = {
            (record["capacity_multiplier"], record["algorithm"])
            for record in seed_records
        }
        missing = expected_pairs - actual_pairs
        if missing:
            raise RuntimeError(f"Seed {seed} is missing reported policy/capacity runs: {sorted(missing)}")
        fixed_pods = set.intersection(*(record["pod_ids"] for record in seed_records))
        if not fixed_pods:
            raise RuntimeError(f"Seed {seed} has no pods common across reported policies and capacities")
        cohort_rows.append(
            {
                "seed": seed,
                "target_pods": target_pods,
                "fixed_common_pods": len(fixed_pods),
                "fixed_common_pct": 100.0 * len(fixed_pods) / target_pods,
            }
        )

        for record in seed_records:
            attributed = attribute_pod_emissions(
                record["placement_csv"],
                record["nodes"],
                record["forecasts"],
            )
            emission_summary = summarize_subset(attributed, fixed_pods)
            detail_rows.append(
                {
                    "seed": seed,
                    "capacity_multiplier": record["capacity_multiplier"],
                    "target_pods": target_pods,
                    "algorithm": record["algorithm"],
                    **emission_summary,
                }
            )
            diagnostics = subset_region_diagnostics(
                record["placements"],
                fixed_pods,
                record["nodes"],
                record["forecasts"],
            )
            diagnostic_rows.append(
                {
                    "seed": seed,
                    "capacity_multiplier": record["capacity_multiplier"],
                    "target_pods": target_pods,
                    "algorithm": record["algorithm"],
                    "common_pods": len(fixed_pods),
                    **diagnostics,
                }
            )

    detail = pd.DataFrame(detail_rows).sort_values(
        ["capacity_multiplier", "seed", "algorithm"]
    )
    diagnostics = pd.DataFrame(diagnostic_rows).sort_values(
        ["capacity_multiplier", "seed", "algorithm"]
    )
    cohorts = pd.DataFrame(cohort_rows).sort_values("seed")
    common_aggregate = (
        detail.groupby(["capacity_multiplier", "target_pods", "algorithm"], as_index=False)
        .agg(
            runs=("seed", "nunique"),
            mean_common_pods=("common_pods", "mean"),
            mean_total_kg=("total_kg", "mean"),
            mean_per_pod_g=("per_pod_g", "mean"),
            std_per_pod_g=("per_pod_g", "std"),
        )
        .sort_values(["capacity_multiplier", "algorithm"])
    )
    diagnostics_aggregate = (
        diagnostics.groupby(["capacity_multiplier", "target_pods", "algorithm"], as_index=False)
        .agg(
            runs=("seed", "nunique"),
            mean_common_pods=("common_pods", "mean"),
            mean_fr_es_cpu_hour_share_pct=("fr_es_cpu_hour_share_pct", "mean"),
            std_fr_es_cpu_hour_share_pct=("fr_es_cpu_hour_share_pct", "std"),
            mean_weighted_carbon_intensity=("weighted_carbon_intensity", "mean"),
            std_weighted_carbon_intensity=("weighted_carbon_intensity", "std"),
        )
        .sort_values(["capacity_multiplier", "algorithm"])
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / "composite_capacity_fixed_cohort_detail.csv", index=False)
    common_aggregate.to_csv(output_dir / "composite_capacity_fixed_cohort_aggregate.csv", index=False)
    diagnostics.to_csv(output_dir / "composite_capacity_fixed_cohort_diagnostics_detail.csv", index=False)
    diagnostics_aggregate.to_csv(
        output_dir / "composite_capacity_fixed_cohort_diagnostics_aggregate.csv",
        index=False,
    )
    cohorts.to_csv(output_dir / "composite_capacity_fixed_cohort_sizes.csv", index=False)
    return common_aggregate, diagnostics_aggregate, cohorts


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


def panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
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
            linewidth = 1.4 if is_focus else 0.9
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
                linewidth=1.4,
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
    sig_vs_totem: dict[str, float] | None = None,
) -> list[Path]:
    sig_vs_totem = sig_vs_totem or {}
    order = ordered_algorithms(whole)
    common = add_ci95(common, "std_per_pod_g", "ci95_per_pod_g")
    fig = plt.figure(figsize=(7.25, 3.30))
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
        err_col="ci95_per_pod_g",
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
        "GreenCourier-Spatial-K8s": (10, -7),
        "GREEN-MLFQ-K8s": (6, -10),
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
        label = SHORT_LABELS.get(algo, algo)
        if sig_vs_totem.get(algo, 1.0) < 0.05:
            label += "*"
        ax_pareto.annotate(
            label,
            (row["mean_success"], row["mean_per_pod_g"]),
            xytext=label_offsets.get(algo, (4, 4)),
            textcoords="offset points",
            fontsize=6.5,
        )
    ax_pareto.set_xlabel("Scheduled pods (%)")
    ax_pareto.set_ylabel("Common-pod emissions (gCO2e/pod)")
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
        handlelength=3.5,
    )
    return save_figure(fig, output_dir, "composite_baseline_matrix_200pods", formats)


def plot_capacity_mechanism_composite(
    common: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
) -> list[Path]:
    common_200 = common[common["target_pods"] == 200].copy()
    common_200 = add_ci95(common_200, "std_per_pod_g", "ci95_per_pod_g")
    diagnostics_200 = diagnostics[diagnostics["target_pods"] == 200].copy()
    diagnostics_200 = add_ci95(
        diagnostics_200,
        "std_fr_es_cpu_hour_share_pct",
        "ci95_fr_es_cpu_hour_share_pct",
    )
    diagnostics_200 = add_ci95(
        diagnostics_200,
        "std_weighted_carbon_intensity",
        "ci95_weighted_carbon_intensity",
    )
    order = ordered_algorithms(common_200)

    fig = plt.figure(figsize=(7.25, 2.95))
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
        err_col="ci95_per_pod_g",
        focus_only=True,
    )
    axes[0].set_ylabel("Common-pod emissions (gCO2e/pod)")
    axes[0].set_title("Same-work emissions")
    axes[0].set_xticks([1, 2, 4])
    panel_label(axes[0], "a")

    lineplot(
        axes[1],
        diagnostics_200,
        "capacity_multiplier",
        "mean_fr_es_cpu_hour_share_pct",
        order=order,
        err_col="ci95_fr_es_cpu_hour_share_pct",
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
        err_col="ci95_weighted_carbon_intensity",
        focus_only=True,
    )
    axes[2].set_ylabel("CPU-hour weighted ACI (gCO2e/kWh)")
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
        handlelength=3.5,
    )
    return save_figure(fig, output_dir, "composite_capacity_mechanism_200pods", formats)


def plot_embodied_ablation_composite(
    regime_sweep_root: Path,
    regime_followup_root: Path,
    output_dir: Path,
    formats: Iterable[str],
) -> list[Path]:
    regime_stats = read_csv(regime_sweep_root / "paired_regime_stats.csv")
    followup = read_csv(regime_followup_root / "common_pod_emissions.csv")
    followup = followup[followup["deadline"] == "flexible"].copy()
    algorithm_names = {
        "vanilla-most-allocated": "Vanilla-MostAllocated",
        "totem-oponly": "TotEm-OpOnly",
        "greencourier-spatial": "GreenCourier-Spatial-K8s",
        "caspian-style": "Caspian-style",
        "green-mlfq": "GREEN-MLFQ-K8s",
        "totem": "TotEm",
    }
    followup["algorithm"] = followup["algorithm"].map(algorithm_names)
    followup["operational_g_per_pod"] = followup["operational_kg"] * 1000.0 / followup["common_pods"]
    followup["embodied_g_per_pod"] = followup["embodied_kg"] * 1000.0 / followup["common_pods"]

    output_dir.mkdir(parents=True, exist_ok=True)
    regime_stats.to_csv(output_dir / "composite_embodied_regime_phase_map.csv", index=False)

    components = (
        followup[followup["algorithm"].isin(["TotEm-OpOnly", "TotEm"])]
        .groupby("algorithm", as_index=False)
        .agg(
            operational_g_per_pod=("operational_g_per_pod", "mean"),
            embodied_g_per_pod=("embodied_g_per_pod", "mean"),
            total_g_per_pod=("per_pod_g", "mean"),
        )
        .set_index("algorithm")
        .loc[["TotEm-OpOnly", "TotEm"]]
        .reset_index()
    )
    components.to_csv(output_dir / "composite_embodied_representative_components.csv", index=False)

    baseline = (
        followup.groupby("algorithm", as_index=False)
        .agg(
            mean_per_pod_g=("per_pod_g", "mean"),
            std_per_pod_g=("per_pod_g", "std"),
            runs=("seed", "nunique"),
        )
    )
    baseline = add_ci95(baseline, "std_per_pod_g", "ci95_per_pod_g")
    baseline["algorithm"] = pd.Categorical(baseline["algorithm"], categories=ALGORITHM_ORDER, ordered=True)
    baseline = baseline.sort_values("algorithm").reset_index(drop=True)
    baseline.to_csv(output_dir / "composite_embodied_representative_baselines.csv", index=False)

    fig = plt.figure(figsize=(7.25, 3.00))
    outer = fig.add_gridspec(
        1,
        3,
        width_ratios=[2.25, 1.05, 1.42],
        left=0.065,
        right=0.99,
        bottom=0.19,
        top=0.86,
        wspace=0.47,
    )

    heatmap_grid = outer[0, 0].subgridspec(2, 2, height_ratios=[1.0, 0.06], hspace=1.00, wspace=0.10)
    ax_tight = fig.add_subplot(heatmap_grid[0, 0])
    ax_flexible = fig.add_subplot(heatmap_grid[0, 1])
    cax = fig.add_subplot(heatmap_grid[1, :])
    cmap = LinearSegmentedColormap.from_list("embodied_effect", ["#5477C4", "#FCFCFD", "#CC6F47"])
    norm = TwoSlopeNorm(vmin=-25, vcenter=0, vmax=45)
    cpu_order = [4.0, 2.0, 1.0, 0.5]
    load_order = [0.25, 0.50, 0.75, 0.90]

    for index, (deadline, ax) in enumerate((("tight", ax_tight), ("flexible", ax_flexible))):
        subset = regime_stats[regime_stats["deadline"] == deadline].copy()
        values = subset.pivot(index="cpu", columns="target", values="pct").reindex(index=cpu_order, columns=load_order)
        significant = subset.assign(
            annotation=subset.apply(
                lambda row: (
                    f"{0 if abs(row['pct']) < 0.5 else row['pct']:+.0f}"
                    f"{'*' if row['p'] < 0.05 and row['better'] == 5 else ''}"
                ),
                axis=1,
            )
        ).pivot(index="cpu", columns="target", values="annotation").reindex(index=cpu_order, columns=load_order)
        sns.heatmap(
            values,
            ax=ax,
            cmap=cmap,
            norm=norm,
            annot=significant,
            fmt="",
            linewidths=0.7,
            linecolor="white",
            cbar=index == 1,
            cbar_ax=cax if index == 1 else None,
            cbar_kws={"orientation": "horizontal"},
            annot_kws={"fontsize": 6.1, "fontweight": "bold"},
        )
        ax.set_title(f"{deadline.capitalize()} deadlines")
        ax.set_xlabel("")
        ax.set_xticklabels(["25%", "50%", "75%", "90%"], rotation=0)
        ax.set_yticklabels(["4", "2", "1", "0.5"], rotation=0)
        ax.tick_params(length=0)
        if index == 0:
            ax.set_ylabel("Pod CPU request (cores)")
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
    ax_flexible.add_patch(Rectangle((1, 1), 1, 1, fill=False, edgecolor="#1F2430", linewidth=1.4))
    cax.set_xlabel("Reduction vs TotEm-OpOnly (%)", labelpad=2)
    cax.xaxis.set_label_position("bottom")
    cax.tick_params(axis="x", labelsize=6.0, length=2, pad=1)
    heatmap_left = ax_tight.get_position().x0
    heatmap_right = ax_flexible.get_position().x1
    gap_bottom = cax.get_position().y1
    gap_top = min(ax_tight.get_position().y0, ax_flexible.get_position().y0)
    fig.text(
        (heatmap_left + heatmap_right) / 2,
        gap_bottom + 0.62 * (gap_top - gap_bottom),
        "Requested demand (% capacity)",
        ha="center",
        va="center",
        fontsize=7.0,
    )
    panel_label(ax_tight, "a")

    ax_components = fig.add_subplot(outer[0, 1])
    x = np.arange(2)
    op_color = "#A3BEFA"
    embodied_color = "#A3D576"
    ax_components.bar(
        x,
        components["operational_g_per_pod"],
        width=0.63,
        color=op_color,
        edgecolor="#2E4780",
        linewidth=0.6,
        label="Operational",
    )
    ax_components.bar(
        x,
        components["embodied_g_per_pod"],
        bottom=components["operational_g_per_pod"],
        width=0.63,
        color=embodied_color,
        edgecolor="#386411",
        linewidth=0.6,
        label="Embodied",
    )
    for xpos, total in zip(x, components["total_g_per_pod"]):
        ax_components.text(xpos, total + 0.55, f"{total:.1f}", ha="center", va="bottom", fontsize=6.8, fontweight="bold")
    bracket_x = 1.48
    upper = components.loc[0, "total_g_per_pod"]
    lower = components.loc[1, "total_g_per_pod"]
    ax_components.plot([bracket_x, bracket_x], [lower, upper], color="#464C55", linewidth=0.8, clip_on=False)
    ax_components.plot([bracket_x - 0.05, bracket_x + 0.05], [upper, upper], color="#464C55", linewidth=0.8)
    ax_components.plot([bracket_x - 0.05, bracket_x + 0.05], [lower, lower], color="#464C55", linewidth=0.8)
    piv = followup[followup["algorithm"].isin(["TotEm-OpOnly", "TotEm"])].pivot_table(
        index="seed", columns="algorithm", values="per_pod_g"
    )
    p_paired = stats.ttest_rel(piv["TotEm-OpOnly"], piv["TotEm"]).pvalue
    reduction_pct = 100.0 * (upper - lower) / upper
    ax_components.text(
        bracket_x + 0.06,
        (upper + lower) / 2,
        f"$-${reduction_pct:.1f}%\n$p$={p_paired:.3f}",
        ha="left",
        va="center",
        fontsize=6.2,
        fontweight="bold",
    )
    ax_components.set_xticks(x)
    ax_components.set_xticklabels(["TotEm-\nOpOnly", "TotEm"], rotation=0)
    ax_components.tick_params(axis="x", pad=1)
    ax_components.set_ylabel("Emissions (gCO2e/pod)")
    ax_components.set_title("Representative mechanism")
    ax_components.set_xlim(-0.5, 2.05)
    ax_components.set_ylim(0, 31)
    ax_components.legend(frameon=False, loc="upper left", fontsize=6.2)
    panel_label(ax_components, "b", x=-0.30)

    ax_baselines = fig.add_subplot(outer[0, 2])
    baseline_plot = baseline.sort_values("mean_per_pod_g", ascending=False).copy()
    y = np.arange(len(baseline_plot))
    colors = [
        STYLES["TotEm"]["color"]
        if algo == "TotEm"
        else STYLES["TotEm-OpOnly"]["color"]
        if algo == "TotEm-OpOnly"
        else "#C5CAD3"
        for algo in baseline_plot["algorithm"].astype(str)
    ]
    edges = [
        STYLES["TotEm"]["color"]
        if algo == "TotEm"
        else STYLES["TotEm-OpOnly"]["color"]
        if algo == "TotEm-OpOnly"
        else "#7A828F"
        for algo in baseline_plot["algorithm"].astype(str)
    ]
    ax_baselines.barh(
        y,
        baseline_plot["mean_per_pod_g"],
        xerr=baseline_plot["ci95_per_pod_g"],
        color=colors,
        edgecolor=edges,
        linewidth=0.6,
        error_kw={"ecolor": "#464C55", "elinewidth": 0.7, "capsize": 2},
    )
    ax_baselines.set_yticks(y)
    ax_baselines.set_yticklabels([SHORT_LABELS[a] for a in baseline_plot["algorithm"].astype(str)], fontsize=6.1)
    for ypos, value, ci in zip(y, baseline_plot["mean_per_pod_g"], baseline_plot["ci95_per_pod_g"].fillna(0.0)):
        ax_baselines.text(value + ci + 0.5, ypos, f"{value:.1f}", va="center", ha="left", fontsize=6.1)
    ax_baselines.set_xlim(0, 33)
    ax_baselines.set_xlabel("Common-pod emissions (gCO2e/pod)")
    ax_baselines.set_title("Six-policy comparison")
    ax_baselines.grid(axis="y", visible=False)
    panel_label(ax_baselines, "c")

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
    trace_common = add_ci95(trace_common, "std_per_pod_g", "ci95_per_pod_g")
    scale = add_ci95(scale, "std_per_pod_g", "ci95_per_pod_g")
    scale = add_ci95(scale, "std_runtime_s", "ci95_runtime_s")

    order = [algo for algo in ROBUSTNESS_ORDER if algo in set(scale["algorithm"])]
    fig = plt.figure(figsize=(7.25, 3.85))
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
            xerr=chunk["ci95_per_pod_g"].fillna(0),
            color=[STYLES[a]["color"] for a in chunk["algorithm"]],
            edgecolor="white",
            linewidth=0.45,
            capsize=2.5,
        )
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT_LABELS[a] for a in chunk["algorithm"]])
        ax.invert_yaxis()
        ax.set_xlabel("Common-pod emissions (gCO2e/pod)")
        ax.set_title(f"Azure trace-derived, {capacity}x")
        panel_label(ax, label)

    for ax, metric, ylabel, title, label in [
        (axes[2], "mean_runtime_s", "Runtime (s)", "Paired scalability runtime", "c"),
        (axes[3], "mean_per_pod_g", "gCO2e/pod", "Paired scalability emissions", "d"),
    ]:
        for algo in order:
            chunk = scale[scale["algorithm"] == algo].sort_values("target_pods")
            if chunk.empty:
                continue
            style = STYLES.get(algo, {})
            err_col = "ci95_runtime_s" if metric == "mean_runtime_s" else "ci95_per_pod_g"
            ax.errorbar(
                chunk["target_pods"],
                chunk[metric],
                yerr=chunk[err_col].fillna(0),
                color=style.get("color"),
                marker=style.get("marker", "o"),
                linestyle=style.get("linestyle", "-"),
                linewidth=1.2 if algo in FOCUS_ALGORITHMS else 0.85,
                markersize=4.4,
                capsize=2,
                alpha=0.92,
            )
        node_labels = f"{int(scale['nodes'].min())} to {int(scale['nodes'].max())}"
        ax.set_xlabel(f"Pods (nodes scale {node_labels})")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(scale["target_pods"].unique()))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
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
        handlelength=3.5,
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
            "2. `composite_embodied_ablation`: workload-regime boundary, representative mechanism, and six-policy follow-up.",
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
    regime_sweep_root = Path(args.regime_sweep_root).resolve()
    regime_followup_root = Path(args.regime_followup_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    formats = formats_from_text(args.formats)

    matrix_whole, matrix_common = load_matrix(matrix_root)
    fixed_capacity_common, fixed_capacity_diagnostics, _fixed_capacity_cohorts = load_fixed_capacity_cohort(
        capacity_root,
        output_dir,
    )

    written: list[Path] = []
    matrix_sig = paired_pvalues_vs_totem(matrix_root, target_pods=200)
    written.extend(
        plot_baseline_composite(matrix_whole, matrix_common, output_dir, formats, matrix_sig)
    )
    written.extend(
        plot_embodied_ablation_composite(
            regime_sweep_root,
            regime_followup_root,
            output_dir,
            formats,
        )
    )
    written.extend(
        plot_capacity_mechanism_composite(
            fixed_capacity_common,
            fixed_capacity_diagnostics,
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
            "workload-regime sweep": regime_sweep_root,
            "workload-regime baseline follow-up": regime_followup_root,
        },
        written,
    )
    print(f"Wrote {len(written)} files to {output_dir}")
    print(output_dir / "plot_manifest.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
