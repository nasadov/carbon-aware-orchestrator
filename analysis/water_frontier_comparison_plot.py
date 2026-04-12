#!/usr/bin/env python3
"""
Plot carbon vs scarcity-characterized water across placement runs.

Marker area encodes total raw water, so the same figure can show when reducing
scarcity-characterized water increases physical water consumption.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
    "pod_id",
    "node_id",
    "total_carbon_emissions",
    "total_raw_water_l",
    "scarcity_characterized_water",
}
DEFAULT_FIGURE_DIR = Path(__file__).resolve().parents[1] / "experiments" / "figures"


def _parse_run_arg(value: str) -> Tuple[str, str, Path]:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--run must use LABEL:GROUP:CSV_PATH, e.g. 'Pareto heuristic:heuristic:/path/file.csv'"
        )
    label, group, csv_path = parts
    if not label.strip() or not group.strip() or not csv_path.strip():
        raise argparse.ArgumentTypeError("--run label, group, and CSV path must be non-empty")
    return label.strip(), group.strip(), Path(csv_path).expanduser().resolve()


def _summarize_run(label: str, group: str, csv_path: Path) -> Dict[str, object]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Placement CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")

    node_counts = df["node_id"].value_counts().sort_index().to_dict()

    return {
        "label": label,
        "group": group,
        "csv_path": str(csv_path),
        "placed_pods": int(df["pod_id"].nunique()),
        "rows": int(len(df)),
        "carbon_kg": float(df["total_carbon_emissions"].fillna(0).sum()),
        "raw_water_l": float(df["total_raw_water_l"].fillna(0).sum()),
        "scarcity_water": float(df["scarcity_characterized_water"].fillna(0).sum()),
        "node_distribution": "; ".join(f"{node}={count}" for node, count in node_counts.items()),
    }


def _marker_sizes(raw_water_values: List[float], min_size: float = 90.0, max_size: float = 900.0) -> List[float]:
    min_value = min(raw_water_values)
    max_value = max(raw_water_values)
    if max_value <= min_value:
        return [(min_size + max_size) / 2.0 for _ in raw_water_values]
    return [
        min_size + ((value - min_value) / (max_value - min_value)) * (max_size - min_size)
        for value in raw_water_values
    ]


def _group_styles(groups: List[str]) -> Dict[str, Dict[str, object]]:
    base = {
        "heuristic": {"color": "#1f77b4", "marker": "o"},
        "milp": {"color": "#d95f02", "marker": "s"},
        "oracle": {"color": "#d95f02", "marker": "s"},
        "baseline": {"color": "#4d4d4d", "marker": "D"},
    }
    fallback_colors = ["#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    styles: Dict[str, Dict[str, object]] = {}
    fallback_index = 0
    for group in groups:
        key = group.lower()
        if key in base:
            styles[group] = base[key]
        else:
            styles[group] = {
                "color": fallback_colors[fallback_index % len(fallback_colors)],
                "marker": "o",
            }
            fallback_index += 1
    return styles


def make_plot(summary_df: pd.DataFrame, output_path: Path, title: str) -> None:
    raw_values = summary_df["raw_water_l"].tolist()
    sizes = _marker_sizes(raw_values)
    styles = _group_styles(summary_df["group"].drop_duplicates().tolist())

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

    for idx, row in summary_df.iterrows():
        style = styles[row["group"]]
        ax.scatter(
            row["carbon_kg"],
            row["scarcity_water"],
            s=sizes[idx],
            color=style["color"],
            marker=style["marker"],
            alpha=0.78,
            edgecolor="black",
            linewidth=0.8,
            label=row["group"] if row["group"] not in ax.get_legend_handles_labels()[1] else None,
        )
        ax.annotate(
            row["label"],
            (row["carbon_kg"], row["scarcity_water"]),
            textcoords="offset points",
            xytext=(7, 5),
            fontsize=8,
        )

    ax.set_xlabel("Total Carbon Emissions (kgCO2e)")
    ax.set_ylabel("Scarcity-Characterized Water")
    ax.set_title(title)
    ax.grid(True, alpha=0.28)

    # Legend for method group.
    method_legend = ax.legend(title="Method group", loc="best", frameon=True)
    ax.add_artist(method_legend)

    # Legend for raw-water marker area.
    raw_min = min(raw_values)
    raw_max = max(raw_values)
    raw_mid = (raw_min + raw_max) / 2.0
    legend_values = [raw_min, raw_mid, raw_max]
    legend_sizes = _marker_sizes(legend_values, min_size=90.0, max_size=900.0)
    raw_handles = [
        ax.scatter([], [], s=size, color="#bdbdbd", edgecolor="black", alpha=0.6)
        for size in legend_sizes
    ]
    ax.legend(
        raw_handles,
        [f"{value:.2f} L" for value in legend_values],
        title="Raw water",
        loc="upper right",
        bbox_to_anchor=(1.0, 0.78),
        frameon=True,
        scatterpoints=1,
    )

    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a carbon-vs-scarcity comparison plot with raw water as marker size."
    )
    parser.add_argument(
        "--run",
        action="append",
        type=_parse_run_arg,
        required=True,
        help="Run descriptor as LABEL:GROUP:CSV_PATH. Repeat for each run.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_FIGURE_DIR / "water_frontier_comparison.png"),
        help="Output PNG path. Defaults under repo-root/experiments/figures/.",
    )
    parser.add_argument("--title", default="Carbon vs Scarcity-Characterized Water", help="Plot title.")
    args = parser.parse_args()

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summaries = [_summarize_run(label, group, csv_path) for label, group, csv_path in args.run]
    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df.sort_values(["group", "carbon_kg", "scarcity_water"]).reset_index(drop=True)

    summary_csv_path = output_path.with_suffix(".csv")
    summary_df.to_csv(summary_csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    make_plot(summary_df, output_path, args.title)

    print(f"Wrote plot: {output_path}")
    print(f"Wrote summary: {summary_csv_path}")


if __name__ == "__main__":
    main()
