#!/usr/bin/env python3
"""Compute simple heuristic-vs-MILP water-frontier diagnostics from a sweep summary."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


DIAGNOSTIC_COLUMNS = [
    "case_label",
    "method_key",
    "method_label",
    "carbon_kg",
    "raw_water_l",
    "scarcity_water",
    "water_metric",
    "water_value",
    "nearest_milp_key",
    "nearest_milp_carbon_kg",
    "nearest_milp_water_value",
    "nearest_water_distance",
    "best_milp_same_or_lower_water_key",
    "best_milp_same_or_lower_water_carbon_kg",
    "carbon_reduction_needed_vs_best_same_or_lower_water_pct",
    "dominated_by_any_milp_in_carbon_raw_scarcity",
]


def _as_float(value) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_column(metric: str) -> str:
    if metric == "raw":
        return "raw_water_l"
    return "scarcity_water"


def build_diagnostics(summary_df: pd.DataFrame, water_metric: str) -> pd.DataFrame:
    water_column = _metric_column(water_metric)
    required = {"method_key", "method_group", "carbon_kg", "raw_water_l", "scarcity_water", water_column}
    missing = sorted(required - set(summary_df.columns))
    if missing:
        raise ValueError(f"Summary CSV is missing required columns: {', '.join(missing)}")

    status = summary_df["status"] if "status" in summary_df.columns else pd.Series("success", index=summary_df.index)
    success_df = summary_df[status == "success"].copy()
    refs = success_df[success_df["method_group"].astype(str).str.lower() == "milp"].copy()
    heuristics = success_df[success_df["method_group"].astype(str).str.lower() == "heuristic"].copy()
    if refs.empty or heuristics.empty:
        return pd.DataFrame(columns=DIAGNOSTIC_COLUMNS)

    rows = []
    for _, heuristic in heuristics.iterrows():
        h_carbon = _as_float(heuristic.get("carbon_kg"))
        h_water = _as_float(heuristic.get(water_column))
        h_raw = _as_float(heuristic.get("raw_water_l"))
        h_scarcity = _as_float(heuristic.get("scarcity_water"))
        if h_carbon is None or h_water is None:
            continue

        refs = refs.assign(
            _water_distance=(refs[water_column].astype(float) - h_water).abs(),
            _carbon_gap=refs["carbon_kg"].astype(float) - h_carbon,
        )
        nearest = refs.sort_values(["_water_distance", "carbon_kg"]).iloc[0]

        same_or_lower_water = refs[refs[water_column].astype(float) <= h_water + 1e-12]
        if not same_or_lower_water.empty:
            best_same_or_lower = same_or_lower_water.sort_values("carbon_kg").iloc[0]
            best_same_or_lower_key = best_same_or_lower.get("method_key")
            best_same_or_lower_carbon = float(best_same_or_lower.get("carbon_kg"))
            carbon_gap_same_or_lower_pct = (
                (h_carbon - best_same_or_lower_carbon) / h_carbon * 100.0 if h_carbon > 0 else None
            )
        else:
            best_same_or_lower_key = ""
            best_same_or_lower_carbon = None
            carbon_gap_same_or_lower_pct = None

        dominated_by_any_milp = bool(
            (
                (refs["carbon_kg"].astype(float) <= h_carbon + 1e-12)
                & (refs["scarcity_water"].astype(float) <= (h_scarcity if h_scarcity is not None else float("inf")) + 1e-12)
                & (refs["raw_water_l"].astype(float) <= (h_raw if h_raw is not None else float("inf")) + 1e-12)
            ).any()
        )

        rows.append(
            {
                "case_label": heuristic.get("case_label", ""),
                "method_key": heuristic.get("method_key", ""),
                "method_label": heuristic.get("method_label", ""),
                "carbon_kg": h_carbon,
                "raw_water_l": h_raw,
                "scarcity_water": h_scarcity,
                "water_metric": water_metric,
                "water_value": h_water,
                "nearest_milp_key": nearest.get("method_key"),
                "nearest_milp_carbon_kg": float(nearest.get("carbon_kg")),
                "nearest_milp_water_value": float(nearest.get(water_column)),
                "nearest_water_distance": float(nearest.get("_water_distance")),
                "best_milp_same_or_lower_water_key": best_same_or_lower_key,
                "best_milp_same_or_lower_water_carbon_kg": best_same_or_lower_carbon,
                "carbon_reduction_needed_vs_best_same_or_lower_water_pct": carbon_gap_same_or_lower_pct,
                "dominated_by_any_milp_in_carbon_raw_scarcity": dominated_by_any_milp,
            }
        )

    return pd.DataFrame(rows, columns=DIAGNOSTIC_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csv", type=Path, help="Path to water_paper_sweep_summary.csv")
    parser.add_argument("--water-metric", choices=["scarcity", "raw"], default="scarcity")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path")
    args = parser.parse_args()

    summary_csv = args.summary_csv.resolve()
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    output = args.output.resolve() if args.output else summary_csv.with_name("water_frontier_gap_diagnostics.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    diagnostics_df = build_diagnostics(pd.read_csv(summary_csv), args.water_metric)
    diagnostics_df.to_csv(output, index=False)
    print(f"Wrote diagnostics: {output}")


if __name__ == "__main__":
    main()
