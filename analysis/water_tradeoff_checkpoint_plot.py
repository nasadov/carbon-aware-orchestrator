#!/usr/bin/env python3
"""
Generate a simple checkpoint carbon-vs-water plot from an enriched placement CSV.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_FIGURE_DIR = Path(__file__).resolve().parents[1] / "experiments" / "figures"


def build_node_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["node_id", "region"], dropna=False)
        .agg(
            placements=("pod_id", "count"),
            carbon_kg=("total_carbon_emissions", "sum"),
            raw_water_l=("total_raw_water_l", "sum"),
            scarcity_water=("scarcity_characterized_water", "sum"),
        )
        .reset_index()
        .sort_values(["carbon_kg", "raw_water_l"])
    )
    return grouped


def make_plot(summary_df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    plots = [
        ("raw_water_l", "Raw Water (L)"),
        ("scarcity_water", "Scarcity-Characterized Water"),
    ]

    for ax, (column, ylabel) in zip(axes, plots):
        ax.scatter(
            summary_df["carbon_kg"],
            summary_df[column],
            s=summary_df["placements"] * 18,
            alpha=0.8,
        )
        for _, row in summary_df.iterrows():
            ax.annotate(
                row["node_id"],
                (row["carbon_kg"], row[column]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
            )
        ax.set_xlabel("Carbon Emissions (kgCO2e)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a checkpoint water trade-off plot from placement CSV data.")
    parser.add_argument("placement_csv", help="Path to an enriched placement CSV")
    parser.add_argument(
        "--output",
        help="Output PNG path. Defaults under repo-root/experiments/figures/.",
    )
    args = parser.parse_args()

    csv_path = Path(args.placement_csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Placement CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {
        "pod_id",
        "node_id",
        "region",
        "total_carbon_emissions",
        "total_raw_water_l",
        "scarcity_characterized_water",
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Placement CSV is missing required columns: {', '.join(missing)}")

    summary_df = build_node_summary(df)

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = DEFAULT_FIGURE_DIR / f"{csv_path.parent.name}_checkpoint_carbon_vs_water_by_node.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_csv = output_path.with_suffix(".csv")
    summary_df.to_csv(summary_csv, index=False)

    title = f"Checkpoint Carbon vs Water by Node ({len(df)} placements)"
    make_plot(summary_df, output_path, title)

    print(f"Wrote plot: {output_path}")
    print(f"Wrote node summary: {summary_csv}")


if __name__ == "__main__":
    main()
