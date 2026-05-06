#!/usr/bin/env python3
"""
Generate paper-two resubmission baseline plots from the matrix aggregate CSVs.

The script expects the aggregate files produced by
scripts/run_resubmission_baseline_matrix.py and the two post-processing scripts:

  - baseline_comparison_aggregate_by_pods.csv
  - attributed_common_all_algorithms_aggregate.csv
  - attributed_common_totem_pairwise_aggregate.csv

It writes PDF and PNG figures, plus small derived CSVs for plots that compute a
secondary metric such as relative overhead against TotEm.
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
    / "resubmission_baseline_matrix_sweep_v2"
    / "matrix_20260506_090340"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "docs" / "paper-two" / "figures" / "ResubmissionBaselines"
)

ALGORITHM_ORDER = [
    "Vanilla-MostAllocated",
    "Piontek-Temporal-K8s",
    "GREEN-MLFQ-K8s",
    "Caspian-style",
    "TotEm-OpOnly",
    "TotEm",
]

LABELS = {
    "Vanilla-MostAllocated": "Vanilla MostAllocated",
    "Piontek-Temporal-K8s": "Piontek-style",
    "GREEN-MLFQ-K8s": "GREEN-style",
    "Caspian-style": "Caspian-style",
    "TotEm-OpOnly": "TotEm operational-only",
    "TotEm": "TotEm",
}

STYLES = {
    "Vanilla-MostAllocated": {
        "color": "#666666",
        "marker": "o",
        "linestyle": "-.",
    },
    "Piontek-Temporal-K8s": {
        "color": "#CC79A7",
        "marker": "D",
        "linestyle": ":",
    },
    "GREEN-MLFQ-K8s": {
        "color": "#009E73",
        "marker": "v",
        "linestyle": "--",
    },
    "Caspian-style": {
        "color": "#0072B2",
        "marker": "^",
        "linestyle": "-",
    },
    "TotEm-OpOnly": {
        "color": "#E69F00",
        "marker": "s",
        "linestyle": "--",
    },
    "TotEm": {
        "color": "#D55E00",
        "marker": "P",
        "linestyle": "-",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-root",
        default=str(DEFAULT_MATRIX_ROOT),
        help="Path to a completed resubmission baseline matrix directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to docs/paper-two/figures/ResubmissionBaselines/<matrix-name>.",
    )
    parser.add_argument(
        "--formats",
        default="pdf,png",
        help="Comma-separated output formats, for example pdf,png.",
    )
    parser.add_argument(
        "--component-pods",
        default="120,200",
        help="Comma-separated pod counts for operational/embodied split plots.",
    )
    parser.add_argument(
        "--tradeoff-pods",
        type=int,
        default=200,
        help="Pod count used for the runtime-emissions tradeoff scatter.",
    )
    parser.add_argument(
        "--pareto-pods",
        default="160,200",
        help="Comma-separated pod counts for carbon-success Pareto plots.",
    )
    return parser.parse_args()


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


def require_csv(matrix_root: Path, filename: str) -> Path:
    path = matrix_root / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required aggregate CSV: {path}")
    return path


def load_inputs(matrix_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    whole = pd.read_csv(require_csv(matrix_root, "baseline_comparison_aggregate_by_pods.csv"))
    common = pd.read_csv(
        require_csv(matrix_root, "attributed_common_all_algorithms_aggregate.csv")
    )
    pairwise = pd.read_csv(
        require_csv(matrix_root, "attributed_common_totem_pairwise_aggregate.csv")
    )
    for df in (whole, common, pairwise):
        if "target_pods" in df.columns:
            df["target_pods"] = df["target_pods"].astype(int)
    return whole, common, pairwise


def ordered_algorithms(df: pd.DataFrame) -> list[str]:
    present = set(df["algorithm"].dropna().unique())
    ordered = [algo for algo in ALGORITHM_ORDER if algo in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    name: str,
    formats: Iterable[str],
) -> list[Path]:
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


def line_with_error(
    ax: plt.Axes,
    df: pd.DataFrame,
    y_col: str,
    err_col: str | None,
    *,
    algorithms: list[str],
) -> None:
    for algo in algorithms:
        chunk = df[df["algorithm"] == algo].sort_values("target_pods")
        if chunk.empty:
            continue
        style = STYLES.get(algo, {})
        yerr = chunk[err_col] if err_col and err_col in chunk.columns else None
        ax.errorbar(
            chunk["target_pods"],
            chunk[y_col],
            yerr=yerr,
            label=LABELS.get(algo, algo),
            color=style.get("color"),
            marker=style.get("marker", "o"),
            linestyle=style.get("linestyle", "-"),
            linewidth=1.7,
            markersize=5.0,
            capsize=2.5 if yerr is not None else 0,
        )


def finish_line_plot(
    ax: plt.Axes,
    *,
    xlabel: str,
    ylabel: str,
    x_ticks: Iterable[int] | None = None,
    legend_columns: int = 3,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if x_ticks is not None:
        ax.set_xticks(sorted(x_ticks))
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=legend_columns,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.1,
    )


def plot_common_emissions(
    common: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.7, 3.35))
    line_with_error(
        ax,
        common,
        "mean_per_pod_g",
        "std_per_pod_g",
        algorithms=ordered_algorithms(common),
    )
    finish_line_plot(
        ax,
        xlabel="Target pods",
        ylabel="Common-pod emissions (gCO2e/pod)",
        x_ticks=common["target_pods"].unique(),
    )
    fig.tight_layout()
    return save_figure(fig, output_dir, "common_pod_emissions_vs_pods", formats)


def plot_success_rate(
    whole: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.7, 3.2))
    line_with_error(
        ax,
        whole,
        "mean_success",
        None,
        algorithms=ordered_algorithms(whole),
    )
    min_success = float(whole["mean_success"].min())
    bottom = max(0.0, np.floor((min_success - 3.0) / 5.0) * 5.0)
    finish_line_plot(
        ax,
        xlabel="Target pods",
        ylabel="Scheduled pods (%)",
        x_ticks=whole["target_pods"].unique(),
        y_min=bottom,
        y_max=101.5,
    )
    fig.tight_layout()
    return save_figure(fig, output_dir, "success_rate_vs_pods", formats)


def plot_runtime(
    whole: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.7, 3.2))
    line_with_error(
        ax,
        whole,
        "mean_runtime_s",
        "std_runtime_s",
        algorithms=ordered_algorithms(whole),
    )
    finish_line_plot(
        ax,
        xlabel="Target pods",
        ylabel="Wall-clock runtime (s)",
        x_ticks=whole["target_pods"].unique(),
        y_min=0.0,
    )
    fig.tight_layout()
    return save_figure(fig, output_dir, "runtime_vs_pods", formats)


def plot_relative_overhead(
    common: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
) -> list[Path]:
    pivot = common.pivot(index="target_pods", columns="algorithm", values="mean_per_pod_g")
    if "TotEm" not in pivot.columns:
        raise ValueError("Cannot compute relative overhead without TotEm rows")

    rows = []
    for pods, row in pivot.sort_index().iterrows():
        totem = float(row["TotEm"])
        for algo in ordered_algorithms(common):
            if algo == "TotEm" or algo not in pivot.columns:
                continue
            rows.append(
                {
                    "target_pods": int(pods),
                    "algorithm": algo,
                    "relative_overhead_pct": 100.0 * (float(row[algo]) - totem) / totem,
                }
            )
    rel = pd.DataFrame(rows)
    rel.to_csv(output_dir / "totem_relative_overhead_common_pods.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.7, 3.3))
    pods_values = sorted(rel["target_pods"].unique())
    algos = [algo for algo in ordered_algorithms(common) if algo != "TotEm"]
    x = np.arange(len(pods_values))
    width = 0.78 / max(len(algos), 1)

    for offset, algo in enumerate(algos):
        chunk = rel[rel["algorithm"] == algo].set_index("target_pods")
        vals = [chunk.loc[pods, "relative_overhead_pct"] for pods in pods_values]
        style = STYLES.get(algo, {})
        ax.bar(
            x + (offset - (len(algos) - 1) / 2.0) * width,
            vals,
            width=width,
            label=LABELS.get(algo, algo),
            color=style.get("color"),
            edgecolor="white",
            linewidth=0.4,
        )

    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(pods) for pods in pods_values])
    ax.set_xlabel("Target pods")
    ax.set_ylabel("Additional emissions vs TotEm (%)")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        frameon=False,
        columnspacing=1.0,
    )
    fig.tight_layout()
    return save_figure(fig, output_dir, "totem_relative_overhead_common_pods", formats)


def plot_totem_oponly_ablation(
    common: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
) -> list[Path]:
    subset = common[common["algorithm"].isin(["TotEm", "TotEm-OpOnly"])]
    pivot = subset.pivot(index="target_pods", columns="algorithm", values="mean_per_pod_g")
    if {"TotEm", "TotEm-OpOnly"}.difference(pivot.columns):
        raise ValueError("Cannot compute TotEm ablation without TotEm and TotEm-OpOnly")
    data = pd.DataFrame(
        {
            "target_pods": pivot.index.astype(int),
            "totem_g_per_pod": pivot["TotEm"].values,
            "operational_only_g_per_pod": pivot["TotEm-OpOnly"].values,
        }
    ).sort_values("target_pods")
    data["totem_reduction_vs_operational_only_pct"] = (
        100.0
        * (data["operational_only_g_per_pod"] - data["totem_g_per_pod"])
        / data["operational_only_g_per_pod"]
    )
    data.to_csv(output_dir / "totem_oponly_ablation_common_pods.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.bar(
        data["target_pods"].astype(str),
        data["totem_reduction_vs_operational_only_pct"],
        color=STYLES["TotEm"]["color"],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Target pods")
    ax.set_ylabel("TotEm reduction vs operational-only (%)")
    fig.tight_layout()
    return save_figure(fig, output_dir, "totem_oponly_ablation_common_pods", formats)


def plot_component_split(
    whole: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    pod_count: int,
) -> list[Path]:
    chunk = whole[whole["target_pods"] == pod_count].copy()
    if chunk.empty:
        return []
    chunk["operational_g_per_placed_pod"] = (
        1000.0 * chunk["mean_operational_kg"] / chunk["mean_placed_pods"]
    )
    chunk["embodied_g_per_placed_pod"] = (
        1000.0 * chunk["mean_embodied_kg"] / chunk["mean_placed_pods"]
    )
    chunk = (
        chunk.set_index("algorithm")
        .reindex([algo for algo in ALGORITHM_ORDER if algo in set(chunk["algorithm"])])
        .reset_index()
    )
    out_csv = output_dir / f"emission_component_split_{pod_count}pods.csv"
    chunk[
        [
            "target_pods",
            "algorithm",
            "operational_g_per_placed_pod",
            "embodied_g_per_placed_pod",
            "mean_per_pod_g",
            "mean_success",
        ]
    ].to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(6.7, 3.15))
    x = np.arange(len(chunk))
    op = chunk["operational_g_per_placed_pod"].to_numpy()
    emb = chunk["embodied_g_per_placed_pod"].to_numpy()
    colors = [STYLES.get(algo, {}).get("color", "#999999") for algo in chunk["algorithm"]]
    ax.bar(x, op, color=colors, edgecolor="white", linewidth=0.4, label="Operational")
    ax.bar(
        x,
        emb,
        bottom=op,
        color="#BDBDBD",
        edgecolor="white",
        linewidth=0.4,
        label="Embodied",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [LABELS.get(algo, algo) for algo in chunk["algorithm"]],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Emissions (gCO2e/placed pod)")
    ax.set_xlabel(f"{pod_count} target pods")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    return save_figure(fig, output_dir, f"emission_component_split_{pod_count}pods", formats)


def plot_tradeoff(
    whole: pd.DataFrame,
    common: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    pod_count: int,
) -> list[Path]:
    runtime = whole[whole["target_pods"] == pod_count][
        ["target_pods", "algorithm", "mean_runtime_s", "mean_success"]
    ]
    emissions = common[common["target_pods"] == pod_count][
        ["target_pods", "algorithm", "mean_per_pod_g"]
    ]
    merged = emissions.merge(runtime, on=["target_pods", "algorithm"], how="inner")
    if merged.empty:
        return []
    merged = (
        merged.set_index("algorithm")
        .reindex([algo for algo in ALGORITHM_ORDER if algo in set(merged["algorithm"])])
        .reset_index()
    )
    merged.to_csv(output_dir / f"runtime_emissions_tradeoff_{pod_count}pods.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.2, 3.35))
    label_offsets = {
        "Vanilla-MostAllocated": (-62, -7),
        "Piontek-Temporal-K8s": (-62, 7),
        "GREEN-MLFQ-K8s": (7, -12),
        "Caspian-style": (7, 5),
        "TotEm-OpOnly": (7, -12),
        "TotEm": (7, 5),
    }
    for _, row in merged.iterrows():
        algo = row["algorithm"]
        style = STYLES.get(algo, {})
        ax.scatter(
            row["mean_runtime_s"],
            row["mean_per_pod_g"],
            s=70,
            color=style.get("color"),
            marker=style.get("marker", "o"),
            edgecolor="white",
            linewidth=0.5,
            label=LABELS.get(algo, algo),
            zorder=3,
        )
        xytext = label_offsets.get(algo, (5, 4))
        ax.annotate(
            LABELS.get(algo, algo),
            (row["mean_runtime_s"], row["mean_per_pod_g"]),
            xytext=xytext,
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xlabel("Wall-clock runtime (s)")
    ax.set_ylabel("Common-pod emissions (gCO2e/pod)")
    ax.set_title(f"{pod_count} target pods")
    fig.tight_layout()
    return save_figure(fig, output_dir, f"runtime_emissions_tradeoff_{pod_count}pods", formats)


def _mark_pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark non-dominated schedulers for success-vs-emissions plots.

    A scheduler is dominated if another scheduler has at least as high success
    and at most as high emissions, with one of those inequalities strict.
    """
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


def plot_carbon_success_pareto(
    whole: pd.DataFrame,
    common: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    pod_count: int,
) -> list[Path]:
    success = whole[whole["target_pods"] == pod_count][
        ["target_pods", "algorithm", "mean_success", "mean_runtime_s"]
    ]
    emissions = common[common["target_pods"] == pod_count][
        ["target_pods", "algorithm", "mean_common_pods", "mean_per_pod_g", "std_per_pod_g"]
    ]
    merged = emissions.merge(success, on=["target_pods", "algorithm"], how="inner")
    if merged.empty:
        return []
    merged = (
        merged.set_index("algorithm")
        .reindex([algo for algo in ALGORITHM_ORDER if algo in set(merged["algorithm"])])
        .reset_index()
    )
    merged = _mark_pareto_front(merged)
    merged.to_csv(output_dir / f"carbon_success_pareto_{pod_count}pods.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.4, 3.55))
    label_offsets = {
        "Vanilla-MostAllocated": (-82, 5),
        "Piontek-Temporal-K8s": (-78, 6),
        "GREEN-MLFQ-K8s": (-76, -12),
        "Caspian-style": (7, 4),
        "TotEm-OpOnly": (7, 5),
        "TotEm": (7, -13),
    }

    frontier = merged[merged["pareto_front"]].sort_values("mean_success")
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

    for _, row in merged.iterrows():
        algo = row["algorithm"]
        style = STYLES.get(algo, {})
        edgecolor = "#111111" if bool(row["pareto_front"]) else "white"
        linewidth = 1.0 if bool(row["pareto_front"]) else 0.5
        ax.scatter(
            row["mean_success"],
            row["mean_per_pod_g"],
            s=90 if bool(row["pareto_front"]) else 72,
            color=style.get("color"),
            marker=style.get("marker", "o"),
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=3,
        )
        ax.annotate(
            LABELS.get(algo, algo),
            (row["mean_success"], row["mean_per_pod_g"]),
            xytext=label_offsets.get(algo, (5, 4)),
            textcoords="offset points",
            fontsize=7,
        )

    x_min = max(0.0, float(merged["mean_success"].min()) - 2.0)
    x_max = min(101.0, float(merged["mean_success"].max()) + 2.0)
    y_min = max(0.0, float(merged["mean_per_pod_g"].min()) - 1.0)
    y_max = float(merged["mean_per_pod_g"].max()) + 1.2
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Scheduled pods (%)")
    ax.set_ylabel("Common-pod emissions (gCO2e/pod)")
    ax.set_title(f"{pod_count} target pods")
    fig.tight_layout()
    return save_figure(fig, output_dir, f"carbon_success_pareto_{pod_count}pods", formats)


def write_manifest(
    output_dir: Path,
    matrix_root: Path,
    formats: list[str],
    written: list[Path],
) -> None:
    lines = [
        "# Resubmission Baseline Plots",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"Source matrix: `{matrix_root}`",
        f"Formats: {', '.join(formats)}",
        "",
        "Input aggregate files:",
        "- `baseline_comparison_aggregate_by_pods.csv`",
        "- `attributed_common_all_algorithms_aggregate.csv`",
        "- `attributed_common_totem_pairwise_aggregate.csv`",
        "",
        "Figures:",
    ]
    for path in sorted(written):
        lines.append(f"- `{path.name}`")
    lines.extend(
        [
            "",
            "Interpretation notes:",
            "- Use `carbon_success_pareto_*pods` as the compact summary of the carbon/completion tradeoff.",
            "- Use `common_pod_emissions_vs_pods` as the carbon-efficiency view because it compares the same placed pod IDs across algorithms.",
            "- Use `success_rate_vs_pods` next to the carbon plot when discussing high-density cases with unequal placement success.",
            "- Use `runtime_vs_pods` and `runtime_emissions_tradeoff_200pods` for the runtime/carbon tradeoff.",
            "- Use the component split and operational-only ablation plots to isolate the embodied-emissions contribution.",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plot_manifest.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    setup_matplotlib()

    matrix_root = Path(args.matrix_root).resolve()
    formats = [fmt.strip().lstrip(".") for fmt in args.formats.split(",") if fmt.strip()]
    component_pods = [
        int(item.strip())
        for item in args.component_pods.split(",")
        if item.strip()
    ]
    pareto_pods = [
        int(item.strip())
        for item in args.pareto_pods.split(",")
        if item.strip()
    ]
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT / matrix_root.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    whole, common, _pairwise = load_inputs(matrix_root)

    written: list[Path] = []
    written.extend(plot_common_emissions(common, output_dir, formats))
    written.extend(plot_success_rate(whole, output_dir, formats))
    written.extend(plot_runtime(whole, output_dir, formats))
    written.extend(plot_relative_overhead(common, output_dir, formats))
    written.extend(plot_totem_oponly_ablation(common, output_dir, formats))
    for pods in component_pods:
        written.extend(plot_component_split(whole, output_dir, formats, pods))
    written.extend(plot_tradeoff(whole, common, output_dir, formats, args.tradeoff_pods))
    for pods in pareto_pods:
        written.extend(plot_carbon_success_pareto(whole, common, output_dir, formats, pods))
    write_manifest(output_dir, matrix_root, formats, written)

    print(f"Wrote {len(written)} figure files to {output_dir}")
    print(f"Wrote {output_dir / 'plot_manifest.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
