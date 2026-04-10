"""
Helpers for loading and attaching water-related metadata to schedulable nodes.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "infra-workload-config.yaml"


def _default_water_config() -> Dict[str, Any]:
    return {
        "enabled": True,
        "data_sources": {
            "aware_country_factors_csv": "",
            "grid_water_factors_csv": "",
            "wue_region_slot_csv": "",
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
        "region_to_country": {},
        "by_region": {},
        "by_country": {},
        "by_hardware_subcategory": {},
    }


def load_water_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load the water section from infra-workload-config.yaml with sane defaults."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        logging.warning("Water config path %s does not exist; using defaults only", path)
        return _default_water_config()

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    water = config.get("water", {}) or {}
    merged = _default_water_config()
    merged["enabled"] = bool(water.get("enabled", merged["enabled"]))
    merged["data_sources"].update(water.get("data_sources", {}) or {})
    merged["defaults"].update(water.get("defaults", {}) or {})
    merged["region_to_country"].update(water.get("region_to_country", {}) or {})
    merged["by_region"].update(water.get("by_region", {}) or {})
    merged["by_country"].update(water.get("by_country", {}) or {})
    merged["by_hardware_subcategory"].update(water.get("by_hardware_subcategory", {}) or {})
    _hydrate_dataset_backed_defaults(merged, base_dir=path.parent)
    return merged


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _build_slot_map(value: Any, default_value: float, slot_count: int) -> Dict[int, float]:
    """Normalize scalar, list, or dict input into an int-keyed slot map."""
    if isinstance(value, dict):
        result = {}
        for key, raw in value.items():
            try:
                result[int(key)] = float(raw)
            except (TypeError, ValueError):
                continue
        if result:
            return result

    if isinstance(value, list):
        return {idx: _as_float(raw, default_value) for idx, raw in enumerate(value[:slot_count])}

    scalar = _as_float(value, default_value)
    return {idx: scalar for idx in range(slot_count)}


def _resolve_country(region: str, water_config: Dict[str, Any]) -> str:
    if not region:
        return ""

    mapping = water_config.get("region_to_country", {})
    if region in mapping:
        return str(mapping[region]).upper()

    if "-" in region:
        return region.split("-", 1)[0].upper()

    return region.upper()


def _resolve_data_path(raw_path: Any, base_dir: Path) -> Optional[Path]:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _load_country_rows(csv_path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if csv_path is None:
        return {}
    if not csv_path.exists():
        logging.warning("Water dataset %s does not exist; skipping", csv_path)
        return {}

    result: Dict[str, Dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            country_code = (row.get("country_code") or "").strip().upper()
            if country_code:
                result[country_code] = row
    return result


def _load_region_slot_rows(csv_path: Optional[Path]) -> Dict[str, Dict[int, Dict[str, str]]]:
    if csv_path is None:
        return {}
    if not csv_path.exists():
        logging.warning("Water dataset %s does not exist; skipping", csv_path)
        return {}

    result: Dict[str, Dict[int, Dict[str, str]]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            region = (row.get("region") or "").strip().upper()
            slot_index = row.get("slot_index")
            if not region or slot_index in (None, ""):
                continue
            try:
                slot = int(slot_index)
            except (TypeError, ValueError):
                continue
            result.setdefault(region, {})[slot] = row
    return result


def _load_hardware_rows(csv_path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if csv_path is None:
        return {}
    if not csv_path.exists():
        logging.warning("Water dataset %s does not exist; skipping", csv_path)
        return {}

    result: Dict[str, Dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subcategory = (row.get("subcategory") or "").strip()
            if subcategory:
                result[subcategory] = row
    return result


def _hydrate_dataset_backed_defaults(water_config: Dict[str, Any], base_dir: Path) -> None:
    data_sources = water_config.get("data_sources", {}) or {}

    aware_rows = _load_country_rows(_resolve_data_path(data_sources.get("aware_country_factors_csv"), base_dir))
    for country_code, row in aware_rows.items():
        country_entry = water_config.setdefault("by_country", {}).setdefault(country_code, {})
        annual_cf = _as_float(row.get("annual_cf"), 1.0)
        country_entry.setdefault("direct_scarcity_cf", annual_cf)
        country_entry.setdefault("indirect_scarcity_cf", annual_cf)
        country_entry.setdefault("embodied_scarcity_cf", annual_cf)

    grid_rows = _load_country_rows(_resolve_data_path(data_sources.get("grid_water_factors_csv"), base_dir))
    for country_code, row in grid_rows.items():
        country_entry = water_config.setdefault("by_country", {}).setdefault(country_code, {})
        country_entry.setdefault("ewif", _as_float(row.get("consumption_l_per_kwh"), 0.0))

    region_wue_rows = _load_region_slot_rows(_resolve_data_path(data_sources.get("wue_region_slot_csv"), base_dir))
    for region, rows_by_slot in region_wue_rows.items():
        region_entry = water_config.setdefault("by_region", {}).setdefault(region, {})
        slot_map = region_entry.setdefault("wue_by_slot", {})
        for slot, row in rows_by_slot.items():
            if slot not in slot_map:
                slot_map[slot] = _as_float(row.get("direct_wue_l_per_kwh"), 0.0)

    hardware_rows = _load_hardware_rows(_resolve_data_path(data_sources.get("embodied_water_reference_csv"), base_dir))
    for subcategory, row in hardware_rows.items():
        hardware_entry = water_config.setdefault("by_hardware_subcategory", {}).setdefault(subcategory, {})
        hardware_entry.setdefault("embodied_water", _as_float(row.get("embodied_water_l"), 0.0))
        manufacturing_country = (row.get("manufacturing_country") or "").strip().upper()
        if manufacturing_country:
            hardware_entry.setdefault("manufacturing_country", manufacturing_country)


def attach_water_metadata(
    flavor,
    region: Optional[str] = None,
    hardware_subcategory: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
    slot_count: int = 24,
):
    """
    Attach water-related attributes to an EnvironmentalFlavor.

    The implementation is deliberately conservative:
    - it supports inline config defaults first
    - it allows later replacement with external CSV/JSON-backed loaders
    - it does not change scheduler behavior on its own
    """
    water_config = config or load_water_config(config_path)

    region_key = (region or getattr(flavor, "region", "") or "").upper()
    flavor.region = region_key

    country = _resolve_country(region_key, water_config)
    flavor.country = country

    defaults = water_config.get("defaults", {})
    region_overrides = water_config.get("by_region", {}).get(region_key, {}) or {}
    country_overrides = water_config.get("by_country", {}).get(country, {}) or {}
    hardware_overrides = water_config.get("by_hardware_subcategory", {}).get(hardware_subcategory or "", {}) or {}
    manufacturing_country = hardware_overrides.get("manufacturing_country", country)
    manufacturing_country = str(manufacturing_country).upper() if manufacturing_country else country
    manufacturing_country_overrides = water_config.get("by_country", {}).get(manufacturing_country, {}) or {}

    default_pue = _as_float(defaults.get("pue", 1.2), 1.2)
    default_embodied_water = _as_float(defaults.get("embodied_water", 0.0), 0.0)
    default_wue = _as_float(defaults.get("wue", 0.0), 0.0)
    default_ewif = _as_float(defaults.get("ewif", 0.0), 0.0)
    default_direct_cf = _as_float(defaults.get("direct_scarcity_cf", 1.0), 1.0)
    default_indirect_cf = _as_float(defaults.get("indirect_scarcity_cf", 1.0), 1.0)
    default_embodied_cf = _as_float(defaults.get("embodied_scarcity_cf", 1.0), 1.0)
    default_criticality = _as_float(defaults.get("criticality", 1.0), 1.0)

    flavor.pue = _as_float(
        region_overrides.get("pue", country_overrides.get("pue", default_pue)),
        default_pue,
    )
    flavor.embodiedWater = _as_float(
        hardware_overrides.get("embodied_water", default_embodied_water),
        default_embodied_water,
    )

    flavor.wue_by_slot = _build_slot_map(
        region_overrides.get(
            "wue_by_slot",
            region_overrides.get("wue", country_overrides.get("wue", default_wue)),
        ),
        default_wue,
        slot_count,
    )
    flavor.ewif_by_slot = _build_slot_map(
        region_overrides.get(
            "ewif_by_slot",
            region_overrides.get("ewif", country_overrides.get("ewif", default_ewif)),
        ),
        default_ewif,
        slot_count,
    )

    flavor.water_scarcity_direct_cf = _as_float(
        region_overrides.get(
            "direct_scarcity_cf",
            country_overrides.get("direct_scarcity_cf", default_direct_cf),
        ),
        default_direct_cf,
    )
    flavor.water_scarcity_indirect_cf = _as_float(
        region_overrides.get(
            "indirect_scarcity_cf",
            country_overrides.get("indirect_scarcity_cf", default_indirect_cf),
        ),
        default_indirect_cf,
    )
    flavor.water_scarcity_embodied_cf = _as_float(
        hardware_overrides.get(
            "embodied_scarcity_cf",
            manufacturing_country_overrides.get("embodied_scarcity_cf", default_embodied_cf),
        ),
        default_embodied_cf,
    )
    flavor.water_criticality = _as_float(
        region_overrides.get(
            "criticality",
            country_overrides.get("criticality", default_criticality),
        ),
        default_criticality,
    )

    flavor.manufacturing_country = manufacturing_country
    return flavor
