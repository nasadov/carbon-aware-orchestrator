from __future__ import annotations

from pathlib import Path

from scripts.water_sensitivity_matrix import (
    _absolutize_data_source_paths,
    build_variant_config,
    parse_args,
    ScenarioSpec,
)


def test_absolutize_data_source_paths_rewrites_relative_paths() -> None:
    base_dir = Path("/tmp/example/pkg/carbon-aware")
    config = {
        "water": {
            "data_sources": {
                "ewif_region_slot_csv": "data/water/ewif_region_slot.csv",
                "embodied_water_reference_csv": "data/water/embodied_water_reference.csv",
                "blank_value": "",
            }
        },
        "carbon": {
            "data_sources": {
                "embodied_carbon_reference_csv": "data/carbon/embodied_carbon_reference.csv",
            }
        },
    }

    hydrated = _absolutize_data_source_paths(config, base_dir)

    assert hydrated["water"]["data_sources"]["ewif_region_slot_csv"] == str(
        (base_dir / "data/water/ewif_region_slot.csv").resolve()
    )
    assert hydrated["water"]["data_sources"]["embodied_water_reference_csv"] == str(
        (base_dir / "data/water/embodied_water_reference.csv").resolve()
    )
    assert hydrated["water"]["data_sources"]["blank_value"] == ""
    assert hydrated["carbon"]["data_sources"]["embodied_carbon_reference_csv"] == str(
        (base_dir / "data/carbon/embodied_carbon_reference.csv").resolve()
    )


def test_scenario_update_paths_can_be_absolutized_after_override() -> None:
    base_dir = Path("/tmp/example/pkg/carbon-aware")
    config = {
        "water": {"data_sources": {"embodied_water_reference_csv": str((base_dir / "data/water/embodied_water_reference.csv").resolve())}},
    }
    scenario = ScenarioSpec(
        label="embodied_water_high",
        group="embodied_water",
        description="High embodied-water scenario",
        updates={"water.data_sources.embodied_water_reference_csv": "data/water/embodied_water_reference_high.csv"},
    )

    variant = build_variant_config(config, scenario)
    hydrated = _absolutize_data_source_paths(variant, base_dir)

    assert hydrated["water"]["data_sources"]["embodied_water_reference_csv"] == str(
        (base_dir / "data/water/embodied_water_reference_high.csv").resolve()
    )


def test_parse_args_accepts_skip_milp() -> None:
    args = parse_args(["--skip-milp"])

    assert args.skip_milp is True
