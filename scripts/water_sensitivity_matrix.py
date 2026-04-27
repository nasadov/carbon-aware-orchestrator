#!/usr/bin/env python3
"""Run one-factor carbon-water sensitivity sweeps on frozen config variants.

This driver keeps the scheduler/method stack fixed and varies only the
environmental data configuration so the resulting comparison is reviewer-ready.
Each scenario gets its own generated config file and sweep output directory, and
the script writes one combined CSV across all scenarios.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd
import yaml


@dataclass(frozen=True)
class ScenarioSpec:
    label: str
    group: str
    description: str
    updates: Dict[str, Any]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def scenario_catalog() -> List[ScenarioSpec]:
    return [
        ScenarioSpec("baseline", "baseline", "Finalized baseline stack", {}),
        ScenarioSpec(
            "direct_low",
            "direct_water",
            "Low-water direct cooling scenario",
            {"water.data_sources.wue_region_slot_csv": "data/water/wue_region_slot_low.csv"},
        ),
        ScenarioSpec(
            "direct_high",
            "direct_water",
            "High-water direct cooling scenario",
            {"water.data_sources.wue_region_slot_csv": "data/water/wue_region_slot_high.csv"},
        ),
        ScenarioSpec(
            "ewif_low",
            "indirect_water",
            "Low indirect-water technology factors",
            {"water.data_sources.ewif_region_slot_csv": "data/water/ewif_region_slot_low.csv"},
        ),
        ScenarioSpec(
            "ewif_high",
            "indirect_water",
            "High indirect-water technology factors",
            {"water.data_sources.ewif_region_slot_csv": "data/water/ewif_region_slot_high.csv"},
        ),
        ScenarioSpec(
            "ewif_wri_static",
            "indirect_water",
            "Disable slot-level EWIF and fall back to static country-level WRI factors",
            {"water.data_sources.ewif_region_slot_csv": ""},
        ),
        ScenarioSpec(
            "scarcity_annual",
            "scarcity",
            "Annual operational AWARE instead of monthly operational AWARE",
            {"water.operational_scarcity_temporal_resolution": "annual"},
        ),
        ScenarioSpec(
            "embodied_water_low",
            "embodied_water",
            "Low embodied-water scenario",
            {"water.data_sources.embodied_water_reference_csv": "data/water/embodied_water_reference_low.csv"},
        ),
        ScenarioSpec(
            "embodied_water_high",
            "embodied_water",
            "High embodied-water scenario",
            {"water.data_sources.embodied_water_reference_csv": "data/water/embodied_water_reference_high.csv"},
        ),
        ScenarioSpec(
            "embodied_carbon_low",
            "embodied_carbon",
            "Low embodied-carbon scenario",
            {"carbon.data_sources.embodied_carbon_reference_csv": "data/carbon/embodied_carbon_reference_low.csv"},
        ),
        ScenarioSpec(
            "embodied_carbon_high",
            "embodied_carbon",
            "High embodied-carbon scenario",
            {"carbon.data_sources.embodied_carbon_reference_csv": "data/carbon/embodied_carbon_reference_high.csv"},
        ),
    ]


def parse_csv_list(value: str) -> List[str]:
    items = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    if not items:
        raise argparse.ArgumentTypeError("At least one value is required")
    return items


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-file",
        type=Path,
        default=root / "pkg" / "carbon-aware" / "infra-workload-config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiments" / "water",
        help="Parent directory for the matrix root.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=root / "experiments" / "figures",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--pod-counts", default="34")
    parser.add_argument("--seeds", default="44")
    parser.add_argument("--timeslots", type=int, default=12)
    parser.add_argument(
        "--methods",
        default="heuristic-carbon,heuristic-epsilon-pareto,milp-waterwise,milp-epsilon",
    )
    parser.add_argument("--epsilon-fractions", default="0.80")
    parser.add_argument("--epsilon-water-metric", choices=["scarcity", "raw"], default="scarcity")
    parser.add_argument(
        "--scenario-labels",
        type=parse_csv_list,
        default=None,
        help="Optional comma-separated subset of scenario labels to run.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip per-scenario frontier plots.")
    parser.add_argument("--target-waterwise-budget", action="store_true")
    parser.add_argument("--skip-milp", action="store_true", help="Forward --skip-milp to each scenario sweep.")
    parser.add_argument("--operational-only", action="store_true")
    parser.add_argument("--embodied-mode", choices=["proportional", "uniform"], default="proportional")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def apply_dotted_update(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor: Dict[str, Any] = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def build_variant_config(base_config: Dict[str, Any], scenario: ScenarioSpec) -> Dict[str, Any]:
    variant = copy.deepcopy(base_config)
    for dotted_key, value in scenario.updates.items():
        apply_dotted_update(variant, dotted_key, value)
    return variant


def _absolutize_data_source_paths(config: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    """Make copied scenario configs location-independent.

    The baseline infra config stores dataset paths relative to its own directory.
    Sensitivity runs write derived configs under experiments/, so those relative
    paths must be rewritten to absolute paths before the copied config is used.
    """

    def absolutize(raw: Any) -> Any:
        if not isinstance(raw, str) or not raw.strip():
            return raw
        value = raw.strip()
        # Keep env-var placeholders untouched if we ever add them later.
        if value.startswith("$"):
            return raw
        candidate = Path(value)
        if candidate.is_absolute():
            return str(candidate)
        return str((base_dir / candidate).resolve())

    for section in ("water", "carbon"):
        section_cfg = config.get(section)
        if not isinstance(section_cfg, dict):
            continue
        data_sources = section_cfg.get("data_sources")
        if not isinstance(data_sources, dict):
            continue
        for key, value in list(data_sources.items()):
            data_sources[key] = absolutize(value)

    return config


def scenario_summary_rows(summary_csv: Path, scenario: ScenarioSpec, matrix_run_name: str) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    df.insert(0, "scenario_label", scenario.label)
    df.insert(1, "scenario_group", scenario.group)
    df.insert(2, "scenario_description", scenario.description)
    df.insert(3, "matrix_run_name", matrix_run_name)
    return df


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    config_file = args.config_file.resolve()
    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle) or {}
    base_config = _absolutize_data_source_paths(base_config, config_file.parent.resolve())

    all_scenarios = scenario_catalog()
    scenario_map = {scenario.label: scenario for scenario in all_scenarios}
    if args.scenario_labels:
        unknown = sorted(set(args.scenario_labels) - set(scenario_map))
        if unknown:
            raise ValueError(f"Unknown scenario labels: {', '.join(unknown)}")
        scenarios = [scenario_map[label] for label in args.scenario_labels]
    else:
        scenarios = all_scenarios

    run_name = args.run_name or datetime.now(timezone.utc).strftime("water_sensitivity_%Y%m%d_%H%M%S")
    matrix_root = args.output_dir.resolve() / run_name
    configs_dir = matrix_root / "configs"
    scenario_runs_dir = matrix_root / "scenario_runs"
    configs_dir.mkdir(parents=True, exist_ok=False)
    scenario_runs_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.resolve().mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_name": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_config_file": str(config_file),
        "pod_counts": args.pod_counts,
        "seeds": args.seeds,
        "timeslots": args.timeslots,
        "methods": args.methods,
        "epsilon_fractions": args.epsilon_fractions,
        "epsilon_water_metric": args.epsilon_water_metric,
        "target_waterwise_budget": bool(args.target_waterwise_budget),
        "skip_milp": bool(args.skip_milp),
        "operational_only": bool(args.operational_only),
        "embodied_mode": args.embodied_mode,
        "scenarios": [
            {
                "label": scenario.label,
                "group": scenario.group,
                "description": scenario.description,
                "updates": scenario.updates,
            }
            for scenario in scenarios
        ],
        "scenario_runs": [],
    }
    (matrix_root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")

    combined_rows: List[pd.DataFrame] = []
    failures: List[str] = []
    for scenario in scenarios:
        scenario_config = build_variant_config(base_config, scenario)
        scenario_config = _absolutize_data_source_paths(scenario_config, config_file.parent.resolve())
        scenario_config_path = configs_dir / f"{scenario.label}.yaml"
        scenario_config_path.write_text(yaml.safe_dump(scenario_config, sort_keys=False), encoding="utf-8")

        scenario_run_name = f"{run_name}_{scenario.label}"
        cmd = [
            sys.executable,
            str(root / "scripts" / "water_sweep.py"),
            "--pod-counts",
            args.pod_counts,
            "--seeds",
            args.seeds,
            "--timeslots",
            str(args.timeslots),
            "--methods",
            args.methods,
            "--epsilon-fractions",
            args.epsilon_fractions,
            "--epsilon-water-metric",
            args.epsilon_water_metric,
            "--config-file",
            str(scenario_config_path),
            "--output-dir",
            str(scenario_runs_dir),
            "--figures-dir",
            str(args.figures_dir.resolve()),
            "--run-name",
            scenario_run_name,
            "--overwrite",
        ]
        if args.target_waterwise_budget:
            cmd.append("--target-waterwise-budget")
        if args.skip_milp:
            cmd.append("--skip-milp")
        if args.no_plots:
            cmd.append("--no-plots")
        if args.operational_only:
            cmd.append("--operational-only")
        if args.embodied_mode:
            cmd.extend(["--embodied-mode", args.embodied_mode])
        if args.quiet:
            cmd.append("--quiet")

        logging.info("Running sensitivity scenario %s", scenario.label)
        proc = subprocess.run(
            cmd,
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        scenario_summary = scenario_runs_dir / scenario_run_name / "water_sweep_summary.csv"
        metadata["scenario_runs"].append(
            {
                "label": scenario.label,
                "config_file": str(scenario_config_path),
                "run_name": scenario_run_name,
                "returncode": proc.returncode,
                "summary_csv": str(scenario_summary),
                "stdout_tail": proc.stdout.splitlines()[-50:],
            }
        )
        (matrix_root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

        if proc.returncode != 0:
            failures.append(scenario.label)
            logging.error("Scenario %s failed", scenario.label)
            continue
        if not scenario_summary.is_file():
            failures.append(scenario.label)
            logging.error("Scenario %s completed without summary CSV", scenario.label)
            continue

        combined_rows.append(scenario_summary_rows(scenario_summary, scenario, run_name))

    combined_csv = matrix_root / "sensitivity_matrix_summary.csv"
    if combined_rows:
        combined_df = pd.concat(combined_rows, ignore_index=True)
        combined_df.to_csv(combined_csv, index=False)

        aggregate_cols = [
            "placed_pods",
            "carbon_kg",
            "raw_water_l",
            "scarcity_water",
            "direct_water_l",
            "indirect_water_l",
            "embodied_water_l",
            "criticality_adjusted_water",
            "elapsed_seconds",
        ]
        present_cols = [col for col in aggregate_cols if col in combined_df.columns]
        grouped = (
            combined_df.groupby(["scenario_label", "method_key"], dropna=False)[present_cols]
            .mean(numeric_only=True)
            .reset_index()
        )
        grouped.to_csv(matrix_root / "sensitivity_matrix_aggregates.csv", index=False)

    (matrix_root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
