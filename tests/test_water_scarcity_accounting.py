from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml


import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_PYTHON_ROOT = REPO_ROOT / "pkg" / "carbon-aware" / "server-python"

if str(SERVER_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_PYTHON_ROOT))

from carbon_aware.footprints import compute_footprint_vector
from carbon_aware.models import CarbonAwarePod, EnvironmentalFlavor
from carbon_aware.water_signals import attach_water_metadata, load_water_config


def _write_config_bundle(tmp: Path, operational_mode: str) -> Path:
    aware_csv = tmp / "aware.csv"
    wue_csv = tmp / "wue.csv"
    ewif_csv = tmp / "ewif.csv"
    config_path = tmp / "infra-workload-config.yaml"

    aware_csv.write_text(
        "\n".join(
            [
                "country_name,country_code,annual_cf,jan_cf,feb_cf,mar_cf,apr_cf,may_cf,jun_cf,jul_cf,aug_cf,sep_cf,oct_cf,nov_cf,dec_cf,source_dataset",
                "Germany,DE,10,2,2,2,2,2,2,20,20,20,20,20,20,AWARE2.0 test",
            ]
        ),
        encoding="utf-8",
    )
    wue_csv.write_text(
        "\n".join(
            [
                "region,reference_location,latitude,longitude,weather_site_source_note,weather_site_confidence,slot_index,forecast_datetime_utc,weather_time_utc,dry_bulb_c,wet_bulb_c,wet_bulb_f,model_wet_bulb_f,cooling_scenario,cooling_profile,activation_rule,activation_logic,evaporative_mode_active,dry_mode_wue_l_per_kwh,wet_tower_wue_l_per_kwh,evap_multiplier,evap_activation_dry_bulb_c,evap_activation_wet_bulb_c,direct_wue_l_per_kwh,model,source_dataset,source_url,cooling_profile_source_basis,cooling_profile_notes",
                "DE,Frankfurt,50.0,8.0,test,high,0,2026-01-15T00:00:00.000Z,2026-01-15T00:00,10,5,41,41,base,hybrid,threshold,any,False,1.0,1.0,0.0,26,18,1.0,test,test,test,test,test",
                "DE,Frankfurt,50.0,8.0,test,high,1,2026-07-15T00:00:00.000Z,2026-07-15T00:00,30,20,68,68,base,hybrid,threshold,any,True,1.0,1.0,0.0,26,18,1.0,test,test,test,test,test",
            ]
        ),
        encoding="utf-8",
    )
    ewif_csv.write_text(
        "\n".join(
            [
                "region,slot_index,forecast_datetime_utc,mix_datetime_utc,ewif_l_per_kwh,ewif_scenario,power_consumption_total_mw,non_storage_share,inherit_share,is_estimated,estimation_method,mix_notes,source_dataset,source_url",
                "DE,0,2026-01-15T00:00:00.000Z,2026-01-15T00:00:00.000Z,1.0,base,100,1.0,0.0,False,,,test,test",
                "DE,1,2026-07-15T00:00:00.000Z,2026-07-15T00:00:00.000Z,1.0,base,100,1.0,0.0,False,,,test,test",
            ]
        ),
        encoding="utf-8",
    )

    config = {
        "water": {
            "enabled": True,
            "operational_scarcity_temporal_resolution": operational_mode,
            "data_sources": {
                "aware_country_factors_csv": str(aware_csv),
                "grid_water_factors_csv": "",
                "wue_region_slot_csv": str(wue_csv),
                "ewif_region_slot_csv": str(ewif_csv),
                "embodied_water_reference_csv": "",
                "region_weather_sites_csv": "",
                "region_cooling_profiles_csv": "",
            },
            "defaults": {
                "pue": 1.0,
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
            "by_country": {},
            "by_hardware_subcategory": {},
        }
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _build_test_flavor() -> EnvironmentalFlavor:
    return EnvironmentalFlavor(
        id="node-0",
        embodiedCarbon=0.0,
        lifetime=1.0,
        totalCpu=1.0,
        totalRam=16.0,
        totalStorage=1.0,
        forecast={0: 100.0, 1: 100.0},
        power={"idle": 0.0, "active": 0.0, "max": 1000.0},
    )


def _build_test_pod() -> CarbonAwarePod:
    return CarbonAwarePod(
        id="pod-0",
        deadline_hours=4,
        duration=2,
        powerConsumption=0.0,
        cpuRequest=1.0,
        ramRequest=1.0,
        storageRequest=1,
    )


def test_monthly_operational_scarcity_changes_with_slot_month() -> None:
    with TemporaryDirectory() as tmpdir:
        config_path = _write_config_bundle(Path(tmpdir), operational_mode="monthly")
        hydrated = load_water_config(str(config_path))
        flavor = _build_test_flavor()

        attach_water_metadata(flavor, region="DE", config=hydrated, slot_count=2)

        assert flavor.water_scarcity_direct_cf_by_slot[0] == 2.0
        assert flavor.water_scarcity_direct_cf_by_slot[1] == 20.0
        assert flavor.water_scarcity_indirect_cf_by_slot[0] == 2.0
        assert flavor.water_scarcity_indirect_cf_by_slot[1] == 20.0

        footprint = compute_footprint_vector(flavor, 0, _build_test_pod(), operational_only=True)

        assert footprint.total_raw_water_l == 4.0
        assert footprint.scarcity_characterized_water == 44.0


def test_annual_operational_scarcity_mode_keeps_constant_cf() -> None:
    with TemporaryDirectory() as tmpdir:
        config_path = _write_config_bundle(Path(tmpdir), operational_mode="annual")
        hydrated = load_water_config(str(config_path))
        flavor = _build_test_flavor()

        attach_water_metadata(flavor, region="DE", config=hydrated, slot_count=2)

        assert flavor.water_scarcity_direct_cf_by_slot[0] == 10.0
        assert flavor.water_scarcity_direct_cf_by_slot[1] == 10.0
        assert flavor.water_scarcity_indirect_cf_by_slot[0] == 10.0
        assert flavor.water_scarcity_indirect_cf_by_slot[1] == 10.0

        footprint = compute_footprint_vector(flavor, 0, _build_test_pod(), operational_only=True)

        assert footprint.total_raw_water_l == 4.0
        assert footprint.scarcity_characterized_water == 40.0


def test_load_water_config_respects_env_override() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config_path = _write_config_bundle(tmp, operational_mode="monthly")
        previous = os.environ.get("CARBON_AWARE_CONFIG_PATH")
        try:
            os.environ["CARBON_AWARE_CONFIG_PATH"] = str(config_path)
            hydrated = load_water_config()
        finally:
            if previous is None:
                os.environ.pop("CARBON_AWARE_CONFIG_PATH", None)
            else:
                os.environ["CARBON_AWARE_CONFIG_PATH"] = previous

        assert hydrated["operational_scarcity_temporal_resolution"] == "monthly"
        assert hydrated["data_sources"]["aware_country_factors_csv"] == str(tmp / "aware.csv")
