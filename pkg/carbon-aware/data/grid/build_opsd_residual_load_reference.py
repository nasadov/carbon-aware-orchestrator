#!/usr/bin/env python3
"""Build a compact OPSD residual-load reference table for pilot experiments.

The full OPSD 60-minute package is large. This script downloads only the columns
needed for the regions in the simulator, selects a high-stress 24-hour window,
and writes a small region-slot table with provenance. Residual load is computed
as load - wind - solar because those are the consistently available OPSD fields
for the selected regions.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd


OPSD_URL = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "opsd_residual_load_region_slot.csv"
REGION_TO_OPSD = {
    "DE": "DE",
    "FR": "FR",
    "ES": "ES",
    "IT-NO": "IT_NORD",
}


def _wind_columns(prefix: str, columns: Iterable[str]) -> List[str]:
    column_set = set(columns)
    total = f"{prefix}_wind_generation_actual"
    if total in column_set:
        return [total]
    return [
        col
        for col in (
            f"{prefix}_wind_onshore_generation_actual",
            f"{prefix}_wind_offshore_generation_actual",
        )
        if col in column_set
    ]


def _required_columns(region_to_opsd: Mapping[str, str], available: Sequence[str]) -> List[str]:
    required = ["utc_timestamp"]
    for prefix in region_to_opsd.values():
        required.append(f"{prefix}_load_actual_entsoe_transparency")
        solar = f"{prefix}_solar_generation_actual"
        if solar in available:
            required.append(solar)
        required.extend(_wind_columns(prefix, available))
    return list(dict.fromkeys(required))


def _window_score(frame: pd.DataFrame, region_to_opsd: Mapping[str, str], start_idx: int, hours: int) -> float:
    window = frame.iloc[start_idx : start_idx + hours]
    score = 0.0
    for region in region_to_opsd:
        series = window[f"{region}_residual_load_mw"]
        if series.max() <= series.min():
            score += float(series.mean())
        else:
            score += float(((series - series.min()) / (series.max() - series.min())).mean())
    return score


def build_reference(
    *,
    source_url: str = OPSD_URL,
    output: Path = DEFAULT_OUTPUT,
    region_to_opsd: Mapping[str, str] = REGION_TO_OPSD,
    month: Optional[int] = 7,
    hours: int = 24,
) -> Path:
    header = pd.read_csv(source_url, nrows=0)
    usecols = _required_columns(region_to_opsd, list(header.columns))
    raw = pd.read_csv(source_url, usecols=usecols, parse_dates=["utc_timestamp"])

    if month is not None:
        raw = raw[raw["utc_timestamp"].dt.month == int(month)].copy()

    for region, prefix in region_to_opsd.items():
        load_col = f"{prefix}_load_actual_entsoe_transparency"
        solar_col = f"{prefix}_solar_generation_actual"
        wind_cols = _wind_columns(prefix, raw.columns)
        renewables = raw[wind_cols].fillna(0.0).sum(axis=1)
        if solar_col in raw:
            renewables += raw[solar_col].fillna(0.0)
        raw[f"{region}_load_mw"] = raw[load_col]
        raw[f"{region}_wind_solar_mw"] = renewables
        raw[f"{region}_residual_load_mw"] = raw[f"{region}_load_mw"] - raw[f"{region}_wind_solar_mw"]

    required_residuals = [f"{region}_residual_load_mw" for region in region_to_opsd]
    raw = raw.dropna(subset=required_residuals).sort_values("utc_timestamp").reset_index(drop=True)
    if len(raw) < hours:
        raise RuntimeError(f"Not enough complete OPSD rows for a {hours}h reference window")

    best_start = max(range(0, len(raw) - hours + 1), key=lambda idx: _window_score(raw, region_to_opsd, idx, hours))
    window = raw.iloc[best_start : best_start + hours].copy()

    rows = []
    for slot, (_, row) in enumerate(window.iterrows()):
        for region, prefix in region_to_opsd.items():
            rows.append(
                {
                    "region": region,
                    "slot_index": slot,
                    "utc_timestamp": row["utc_timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "opsd_region": prefix,
                    "load_mw": round(float(row[f"{region}_load_mw"]), 6),
                    "wind_solar_generation_mw": round(float(row[f"{region}_wind_solar_mw"]), 6),
                    "residual_load_mw": round(float(row[f"{region}_residual_load_mw"]), 6),
                    "signal_model": "residual_load_lite_load_minus_wind_solar",
                    "source_dataset": "Open Power System Data time_series 2020-10-06, 60-minute singleindex",
                    "source_url": source_url,
                    "notes": "Run-of-river not subtracted because it is not consistently available in OPSD time_series_60min_singleindex.",
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-url", default=OPSD_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--hours", type=int, default=24)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output = build_reference(
        source_url=args.source_url,
        output=args.output,
        month=args.month,
        hours=args.hours,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
