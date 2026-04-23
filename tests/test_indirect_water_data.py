from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_PYTHON_ROOT = REPO_ROOT / "pkg" / "carbon-aware" / "server-python"
WATER_DATA_ROOT = REPO_ROOT / "pkg" / "carbon-aware" / "data" / "water"

if str(SERVER_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_PYTHON_ROOT))

from carbon_aware.models import EnvironmentalFlavor
from carbon_aware.water_signals import attach_water_metadata, load_water_config


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _factor_map(path: Path, scenario: str) -> dict[str, float]:
    column = f"water_consumption_l_per_kwh_{scenario}"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            row["source_type"]: float(row[column])
            for row in reader
            if row.get(column) not in (None, "")
        }


def test_region_ewif_builder_generates_expected_slot_values_from_mix_json() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        forecasts_path = tmp / "all_forecasts.json"
        mix_path = tmp / "mix.json"
        output_path = tmp / "ewif_region_slot_base.csv"

        forecasts_payload = {
            "DE": {
                "zone": "DE",
                "forecast": [
                    {"datetime": "2026-01-01T00:00:00.000Z"},
                    {"datetime": "2026-01-01T01:00:00.000Z"},
                ],
            }
        }
        mix_payload = {
            "DE": {
                "zone": "DE",
                "history": [
                    {
                        "datetime": "2026-01-01T00:00:00.000Z",
                        "powerConsumptionBreakdown": {"wind": 70.0, "gas": 30.0},
                        "powerConsumptionTotal": 100.0,
                        "isEstimated": False,
                    },
                    {
                        "datetime": "2026-01-01T01:00:00.000Z",
                        "powerConsumptionBreakdown": {"battery-discharge": 20.0, "solar": 80.0},
                        "powerConsumptionTotal": 100.0,
                        "isEstimated": True,
                        "estimationMethod": "mock",
                    },
                ],
            }
        }
        forecasts_path.write_text(json.dumps(forecasts_payload), encoding="utf-8")
        mix_path.write_text(json.dumps(mix_payload), encoding="utf-8")

        subprocess.run(
            [
                sys.executable,
                "build_region_ewif_reference.py",
                "--forecasts",
                str(forecasts_path),
                "--mix-json",
                str(mix_path),
                "--factors",
                str(WATER_DATA_ROOT / "ewif_technology_factors.csv"),
                "--output",
                str(output_path),
                "--scenario",
                "base",
            ],
            cwd=WATER_DATA_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        rows = _read_rows(output_path)
        assert len(rows) == 2

        factors = _factor_map(WATER_DATA_ROOT / "ewif_technology_factors.csv", "base")
        expected_first = 0.70 * factors["wind"] + 0.30 * factors["gas"]
        expected_second = factors["solar"]

        assert math.isclose(float(rows[0]["ewif_l_per_kwh"]), expected_first, rel_tol=1e-9)
        assert math.isclose(float(rows[1]["ewif_l_per_kwh"]), expected_second, rel_tol=1e-9)
        assert math.isclose(float(rows[1]["inherit_share"]), 0.20, rel_tol=1e-9)
        assert rows[1]["is_estimated"] == "True"
        assert rows[1]["estimation_method"] == "mock"


def test_attach_water_metadata_prefers_region_slot_ewif_over_country_ewif() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        ewif_csv = tmp / "ewif_region_slot.csv"
        config_path = tmp / "infra-workload-config.yaml"

        ewif_csv.write_text(
            "\n".join(
                [
                    "region,slot_index,forecast_datetime_utc,mix_datetime_utc,ewif_l_per_kwh,ewif_scenario,power_consumption_total_mw,non_storage_share,inherit_share,is_estimated,estimation_method,mix_notes,source_dataset,source_url",
                    "DE,0,2026-01-01T00:00:00.000Z,2026-01-01T00:00:00.000Z,0.123,base,100,1.0,0.0,False,,,'test','test'",
                    "DE,1,2026-01-01T01:00:00.000Z,2026-01-01T01:00:00.000Z,0.456,base,100,1.0,0.0,False,,,'test','test'",
                ]
            ),
            encoding="utf-8",
        )

        config = {
            "water": {
                "enabled": True,
                "data_sources": {
                    "aware_country_factors_csv": "",
                    "grid_water_factors_csv": "",
                    "wue_region_slot_csv": "",
                    "ewif_region_slot_csv": str(ewif_csv),
                    "embodied_water_reference_csv": "",
                },
                "defaults": {
                    "pue": 1.2,
                    "embodied_water": 0.0,
                    "wue": 0.0,
                    "ewif": 0.0,
                    "direct_scarcity_cf": 1.0,
                    "indirect_scarcity_cf": 1.0,
                    "embodied_scarcity_cf": 1.0,
                    "criticality": 1.0,
                },
                "region_to_country": {"DE": "DE"},
                "by_region": {},
                "by_country": {"DE": {"ewif": 9.99}},
                "by_hardware_subcategory": {},
            }
        }
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

        hydrated = load_water_config(str(config_path))
        flavor = EnvironmentalFlavor(
            id="node-0",
            embodiedCarbon=0.0,
            lifetime=1.0,
            totalCpu=16.0,
            totalRam=16.0,
            totalStorage=1.0,
            forecast={0: 100.0, 1: 120.0},
        )

        attach_water_metadata(flavor, region="DE", config=hydrated, slot_count=2)

        assert flavor.ewif_by_slot[0] == 0.123
        assert flavor.ewif_by_slot[1] == 0.456
