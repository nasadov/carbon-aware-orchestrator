from __future__ import annotations

import csv
import importlib.util
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
WUE_BUILDER_PATH = WATER_DATA_ROOT / "build_region_wue_reference.py"

if str(SERVER_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_PYTHON_ROOT))

from carbon_aware.models import EnvironmentalFlavor
from carbon_aware.water_signals import attach_water_metadata, load_water_config


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_wue_builder_module():
    spec = importlib.util.spec_from_file_location("build_region_wue_reference", WUE_BUILDER_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_direct_wue_profile_logic_distinguishes_low_base_high() -> None:
    builder = _load_wue_builder_module()

    low_profile = builder.CoolingProfile(
        region="DE",
        scenario="low",
        profile_class="closed_loop_low_water",
        activation_rule="never",
        activation_logic="any",
        dry_mode_wue_l_per_kwh=0.005,
        evap_multiplier=0.0,
        evap_activation_dry_bulb_c=None,
        evap_activation_wet_bulb_c=None,
        source_basis="test",
        notes="test",
    )
    base_profile = builder.CoolingProfile(
        region="DE",
        scenario="base",
        profile_class="hybrid_economized",
        activation_rule="threshold",
        activation_logic="any",
        dry_mode_wue_l_per_kwh=0.03,
        evap_multiplier=0.5,
        evap_activation_dry_bulb_c=26.0,
        evap_activation_wet_bulb_c=18.0,
        source_basis="test",
        notes="test",
    )
    high_profile = builder.CoolingProfile(
        region="DE",
        scenario="high",
        profile_class="wet_tower",
        activation_rule="always",
        activation_logic="any",
        dry_mode_wue_l_per_kwh=0.0,
        evap_multiplier=1.0,
        evap_activation_dry_bulb_c=None,
        evap_activation_wet_bulb_c=None,
        source_basis="test",
        notes="test",
    )

    # Warm case activates the base scenario.
    evaporative_active, wet_tower_wue, _, high_value = builder._compute_direct_wue(high_profile, 30.0, 20.0)
    assert evaporative_active is True
    assert math.isclose(high_value, wet_tower_wue, rel_tol=1e-9)

    evaporative_active, wet_tower_wue, _, base_value = builder._compute_direct_wue(base_profile, 30.0, 20.0)
    assert evaporative_active is True
    assert math.isclose(base_value, 0.03 + 0.5 * wet_tower_wue, rel_tol=1e-9)

    evaporative_active, _, _, low_value = builder._compute_direct_wue(low_profile, 30.0, 20.0)
    assert evaporative_active is False
    assert math.isclose(low_value, 0.005, rel_tol=1e-9)

    # Cool case should fall back to the dry mode in the base scenario.
    evaporative_active, wet_tower_wue, _, cool_base_value = builder._compute_direct_wue(base_profile, 10.0, 5.0)
    assert evaporative_active is False
    assert math.isclose(cool_base_value, 0.03, rel_tol=1e-9)
    assert wet_tower_wue > cool_base_value


def test_region_wue_builder_generates_expected_slot_values_from_weather_json() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        forecasts_path = tmp / "all_forecasts.json"
        weather_json_path = tmp / "weather.json"
        weather_sites_path = tmp / "region_weather_sites.csv"
        cooling_profiles_path = tmp / "region_cooling_profiles.csv"
        output_path = tmp / "wue_region_slot_base.csv"

        forecasts_payload = {
            "DE": {
                "zone": "DE",
                "forecast": [
                    {"datetime": "2026-01-01T00:00:00.000Z"},
                    {"datetime": "2026-01-01T01:00:00.000Z"},
                ],
            }
        }
        weather_payload = {
            "DE": {
                "hourly": {
                    "time": ["2026-01-01T00:00", "2026-01-01T01:00"],
                    "temperature_2m": [10.0, 30.0],
                    "wet_bulb_temperature_2m": [5.0, 20.0],
                },
                "_request_url": "mock://weather",
            }
        }
        forecasts_path.write_text(json.dumps(forecasts_payload), encoding="utf-8")
        weather_json_path.write_text(json.dumps(weather_payload), encoding="utf-8")
        weather_sites_path.write_text(
            "\n".join(
                [
                    "region,site_proxy_name,latitude,longitude,source_note,confidence",
                    "DE,Frankfurt test proxy,50.0,8.0,test,high",
                ]
            ),
            encoding="utf-8",
        )
        cooling_profiles_path.write_text(
            "\n".join(
                [
                    "region,scenario,profile_class,activation_rule,activation_logic,dry_mode_wue_l_per_kwh,evap_multiplier,evap_activation_dry_bulb_c,evap_activation_wet_bulb_c,source_basis,notes",
                    "DE,base,hybrid_economized,threshold,any,0.03,0.50,26,18,test,test",
                ]
            ),
            encoding="utf-8",
        )

        subprocess.run(
            [
                sys.executable,
                str(WUE_BUILDER_PATH),
                "--forecasts",
                str(forecasts_path),
                "--weather-json",
                str(weather_json_path),
                "--weather-sites",
                str(weather_sites_path),
                "--cooling-profiles",
                str(cooling_profiles_path),
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
        assert rows[0]["evaporative_mode_active"] == "False"
        assert rows[1]["evaporative_mode_active"] == "True"
        assert math.isclose(float(rows[0]["direct_wue_l_per_kwh"]), 0.03, rel_tol=1e-9)

        builder = _load_wue_builder_module()
        _, wet_tower_wue, _, _ = builder._compute_direct_wue(
            builder.CoolingProfile(
                region="DE",
                scenario="base",
                profile_class="hybrid_economized",
                activation_rule="threshold",
                activation_logic="any",
                dry_mode_wue_l_per_kwh=0.03,
                evap_multiplier=0.50,
                evap_activation_dry_bulb_c=26.0,
                evap_activation_wet_bulb_c=18.0,
                source_basis="test",
                notes="test",
            ),
            30.0,
            20.0,
        )
        assert math.isclose(float(rows[1]["direct_wue_l_per_kwh"]), 0.03 + 0.5 * wet_tower_wue, rel_tol=1e-9)


def test_attach_water_metadata_prefers_region_slot_wue_over_region_default() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        wue_csv = tmp / "wue_region_slot.csv"
        config_path = tmp / "infra-workload-config.yaml"

        wue_csv.write_text(
            "\n".join(
                [
                    "region,reference_location,latitude,longitude,weather_site_source_note,weather_site_confidence,slot_index,forecast_datetime_utc,weather_time_utc,dry_bulb_c,wet_bulb_c,wet_bulb_f,model_wet_bulb_f,cooling_scenario,cooling_profile,activation_rule,activation_logic,evaporative_mode_active,dry_mode_wue_l_per_kwh,wet_tower_wue_l_per_kwh,evap_multiplier,evap_activation_dry_bulb_c,evap_activation_wet_bulb_c,direct_wue_l_per_kwh,model,source_dataset,source_url,cooling_profile_source_basis,cooling_profile_notes",
                    "DE,Frankfurt,50.0,8.0,test,high,0,2026-01-01T00:00:00.000Z,2026-01-01T00:00,10,5,41,41,base,hybrid,threshold,any,False,0.03,1.4,0.5,26,18,0.03,test,test,test,test,test",
                    "DE,Frankfurt,50.0,8.0,test,high,1,2026-01-01T01:00:00.000Z,2026-01-01T01:00,30,20,68,68,base,hybrid,threshold,any,True,0.03,1.0,0.5,26,18,0.53,test,test,test,test,test",
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
                    "wue_region_slot_csv": str(wue_csv),
                    "ewif_region_slot_csv": "",
                    "embodied_water_reference_csv": "",
                    "region_weather_sites_csv": "",
                    "region_cooling_profiles_csv": "",
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
                "by_region": {"DE": {"wue": 9.99}},
                "by_country": {},
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

        assert flavor.wue_by_slot[0] == 0.03
        assert flavor.wue_by_slot[1] == 0.53
