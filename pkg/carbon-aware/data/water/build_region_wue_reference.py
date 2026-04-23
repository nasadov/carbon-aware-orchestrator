"""
Build region-by-slot direct-water (WUE) reference tables.

The workflow is:
- read region weather-site proxies from a repo-owned CSV
- read forecast timestamps from all_forecasts.json
- fetch hourly dry-bulb and wet-bulb weather from Open-Meteo or use a cached/mock payload
- convert wet-bulb temperature to a wet-cooling WUE term using the fixed-approach
  Gupta et al. model
- apply explicit cooling-profile scenarios (wet tower, hybrid economized,
  low-water/closed-loop) to produce low/base/high direct-WUE tables

The scheduler only consumes the generated CSVs. No weather API calls happen at
runtime.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import requests
import yaml


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = THIS_DIR.parent.parent / "infra-workload-config.yaml"
DEFAULT_FORECASTS_PATH = THIS_DIR.parent.parent / "server-python" / "all_forecasts.json"
DEFAULT_WEATHER_SITES_PATH = THIS_DIR / "region_weather_sites.csv"
DEFAULT_COOLING_PROFILES_PATH = THIS_DIR / "region_cooling_profiles.csv"
DEFAULT_OUTPUT_TEMPLATE = THIS_DIR / "wue_region_slot_{scenario}.csv"
DEFAULT_BASE_OUTPUT = THIS_DIR / "wue_region_slot.csv"
DEFAULT_LEGACY_BASE_OUTPUT = THIS_DIR / "open_meteo_region_wue_by_slot.csv"
OPEN_METEO_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OPEN_METEO_SOURCE = "Open-Meteo historical forecast API"


@dataclass(frozen=True)
class WeatherSite:
    region: str
    name: str
    latitude: float
    longitude: float
    source_note: str
    confidence: str


@dataclass(frozen=True)
class CoolingProfile:
    region: str
    scenario: str
    profile_class: str
    activation_rule: str
    activation_logic: str
    dry_mode_wue_l_per_kwh: float
    evap_multiplier: float
    evap_activation_dry_bulb_c: Optional[float]
    evap_activation_wet_bulb_c: Optional[float]
    source_basis: str
    notes: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to infra-workload-config.yaml (used only as a fallback for legacy weather_reference entries).",
    )
    parser.add_argument(
        "--forecasts",
        default=str(DEFAULT_FORECASTS_PATH),
        help="Path to all_forecasts.json used to define region-slot timestamps.",
    )
    parser.add_argument(
        "--weather-sites",
        default=str(DEFAULT_WEATHER_SITES_PATH),
        help="CSV with region weather-site proxies.",
    )
    parser.add_argument(
        "--cooling-profiles",
        default=str(DEFAULT_COOLING_PROFILES_PATH),
        help="CSV with region/scenario cooling-profile assumptions.",
    )
    parser.add_argument(
        "--weather-json",
        default="",
        help="Optional cached/mock Open-Meteo-style JSON payload keyed by region to avoid live API calls.",
    )
    parser.add_argument(
        "--cache-json",
        default="",
        help="Optional path to write the raw fetched weather payload.",
    )
    parser.add_argument(
        "--scenario",
        choices=["low", "base", "high", "all"],
        default="all",
        help="Which direct-water scenario(s) to generate.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Destination CSV for a single scenario. Ignored when --scenario=all.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_forecast_slots(path: Path) -> Dict[str, List[Tuple[int, str, str]]]:
    """
    Return {region: [(slot_index, raw_iso_utc, open_meteo_hour_key), ...]}.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    result: Dict[str, List[Tuple[int, str, str]]] = {}
    for region, region_payload in data.items():
        forecasts = (region_payload or {}).get("forecast", []) or []
        region_slots: List[Tuple[int, str, str]] = []
        for slot_index, entry in enumerate(forecasts):
            raw_datetime = str(entry.get("datetime", "")).strip()
            if not raw_datetime:
                continue
            dt = datetime.fromisoformat(raw_datetime.replace("Z", "+00:00")).astimezone(timezone.utc)
            open_meteo_key = dt.strftime("%Y-%m-%dT%H:%M")
            region_slots.append((slot_index, raw_datetime, open_meteo_key))
        if region_slots:
            result[str(region).upper()] = region_slots
    return result


def _load_legacy_weather_points(config_path: Path) -> Dict[str, WeatherSite]:
    config = _load_yaml(config_path)
    by_region = (((config.get("water") or {}).get("by_region")) or {})
    weather_points: Dict[str, WeatherSite] = {}
    for region, entry in by_region.items():
        point = (entry or {}).get("weather_reference", {}) or {}
        latitude = point.get("latitude")
        longitude = point.get("longitude")
        if latitude is None or longitude is None:
            continue
        region_key = str(region).upper()
        weather_points[region_key] = WeatherSite(
            region=region_key,
            name=str(point.get("name", region)),
            latitude=float(latitude),
            longitude=float(longitude),
            source_note="Legacy weather_reference entry from infra-workload-config.yaml",
            confidence="legacy",
        )
    return weather_points


def _load_weather_sites(csv_path: Path, config_path: Path) -> Dict[str, WeatherSite]:
    if not csv_path.exists():
        return _load_legacy_weather_points(config_path)

    sites: Dict[str, WeatherSite] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            region = str(row.get("region", "")).strip().upper()
            if not region:
                continue
            sites[region] = WeatherSite(
                region=region,
                name=str(row.get("site_proxy_name") or row.get("reference_location") or region).strip(),
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                source_note=str(row.get("source_note", "")).strip(),
                confidence=str(row.get("confidence", "")).strip(),
            )
    if not sites:
        raise ValueError(f"No weather-site entries found in {csv_path}")
    return sites


def _load_cooling_profiles(csv_path: Path) -> Dict[str, Dict[str, CoolingProfile]]:
    profiles: Dict[str, Dict[str, CoolingProfile]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            region = str(row.get("region", "")).strip().upper()
            scenario = str(row.get("scenario", "")).strip().lower()
            if not region or not scenario:
                continue
            profile = CoolingProfile(
                region=region,
                scenario=scenario,
                profile_class=str(row.get("profile_class", "")).strip(),
                activation_rule=str(row.get("activation_rule", "threshold")).strip().lower(),
                activation_logic=str(row.get("activation_logic", "any")).strip().lower(),
                dry_mode_wue_l_per_kwh=float(row.get("dry_mode_wue_l_per_kwh") or 0.0),
                evap_multiplier=float(row.get("evap_multiplier") or 0.0),
                evap_activation_dry_bulb_c=_safe_optional_float(row.get("evap_activation_dry_bulb_c")),
                evap_activation_wet_bulb_c=_safe_optional_float(row.get("evap_activation_wet_bulb_c")),
                source_basis=str(row.get("source_basis", "")).strip(),
                notes=str(row.get("notes", "")).strip(),
            )
            profiles.setdefault(region, {})[scenario] = profile
    if not profiles:
        raise ValueError(f"No cooling profiles found in {csv_path}")
    return profiles


def _safe_optional_float(value: Any) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    return float(value)


def _iter_region_date_ranges(region_slots: Dict[str, List[Tuple[int, str, str]]]) -> Iterable[Tuple[str, str, str]]:
    for region, slots in region_slots.items():
        datetimes = [datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc) for _, raw, _ in slots]
        start_date = min(datetimes).strftime("%Y-%m-%d")
        end_date = max(datetimes).strftime("%Y-%m-%d")
        yield region, start_date, end_date


def _fetch_hourly_weather(latitude: float, longitude: float, start_date: str, end_date: str) -> Tuple[Dict[str, Dict[str, float]], str]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,wet_bulb_temperature_2m",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "GMT",
    }
    response = requests.get(OPEN_METEO_URL, params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly", {}) or {}
    times = hourly.get("time", []) or []
    dry_bulbs = hourly.get("temperature_2m", []) or []
    wet_bulbs = hourly.get("wet_bulb_temperature_2m", []) or []
    series: Dict[str, Dict[str, float]] = {}
    for raw_time, dry_bulb_c, wet_bulb_c in zip(times, dry_bulbs, wet_bulbs):
        if dry_bulb_c in (None, "") or wet_bulb_c in (None, ""):
            continue
        series[str(raw_time)] = {
            "dry_bulb_c": float(dry_bulb_c),
            "wet_bulb_c": float(wet_bulb_c),
        }
    return series, response.url


def _load_weather_payload(
    weather_json_path: Optional[Path],
    region_slots: Mapping[str, Sequence[Tuple[int, str, str]]],
    weather_sites: Mapping[str, WeatherSite],
    cache_json_path: Optional[Path],
) -> Dict[str, Dict[str, Any]]:
    if weather_json_path is not None:
        return json.loads(weather_json_path.read_text(encoding="utf-8"))

    payload: Dict[str, Dict[str, Any]] = {}
    for region, start_date, end_date in _iter_region_date_ranges(dict(region_slots)):
        site = weather_sites.get(region)
        if site is None:
            raise ValueError(f"Region {region} is missing a weather-site proxy")
        hourly_values, request_url = _fetch_hourly_weather(
            latitude=site.latitude,
            longitude=site.longitude,
            start_date=start_date,
            end_date=end_date,
        )
        payload[region] = {
            "hourly": {
                "time": list(hourly_values.keys()),
                "temperature_2m": [hourly_values[key]["dry_bulb_c"] for key in hourly_values],
                "wet_bulb_temperature_2m": [hourly_values[key]["wet_bulb_c"] for key in hourly_values],
            },
            "_request_url": request_url,
            "_source_dataset": OPEN_METEO_SOURCE,
        }

    if cache_json_path is not None:
        cache_json_path.parent.mkdir(parents=True, exist_ok=True)
        cache_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def _build_region_weather_cache(
    weather_payload: Mapping[str, Mapping[str, Any]],
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, str]]:
    weather_cache: Dict[str, Dict[str, Dict[str, float]]] = {}
    request_urls: Dict[str, str] = {}
    for region, region_payload in weather_payload.items():
        hourly = (region_payload or {}).get("hourly", {}) or {}
        times = hourly.get("time", []) or []
        dry_bulbs = hourly.get("temperature_2m", []) or []
        wet_bulbs = hourly.get("wet_bulb_temperature_2m", []) or []
        entries: Dict[str, Dict[str, float]] = {}
        for raw_time, dry_bulb_c, wet_bulb_c in zip(times, dry_bulbs, wet_bulbs):
            if dry_bulb_c in (None, "") or wet_bulb_c in (None, ""):
                continue
            entries[str(raw_time)] = {
                "dry_bulb_c": float(dry_bulb_c),
                "wet_bulb_c": float(wet_bulb_c),
            }
        weather_cache[str(region).upper()] = entries
        request_urls[str(region).upper()] = str(region_payload.get("_request_url", ""))
    return weather_cache, request_urls


def _c_to_f(temp_c: float) -> float:
    return temp_c * 9.0 / 5.0 + 32.0


def _fixed_approach_wue_l_per_kwh(wet_bulb_c: float) -> Tuple[float, float]:
    """
    Gupta et al. (e-Energy 2024), Eq. (2), fixed-approach cooling-tower model.

    The paper states a lower validity bound of 30 F. We clamp colder conditions
    to that limit rather than extrapolating below the fitted range.
    """
    wet_bulb_f = _c_to_f(wet_bulb_c)
    model_wet_bulb_f = max(wet_bulb_f, 30.0)
    wue = (
        -0.0001896 * (model_wet_bulb_f ** 2)
        + 0.03095 * model_wet_bulb_f
        + 0.4442
    )
    return model_wet_bulb_f, max(wue, 0.0)


def _is_evaporative_mode_active(profile: CoolingProfile, dry_bulb_c: float, wet_bulb_c: float) -> bool:
    if profile.activation_rule == "always":
        return True
    if profile.activation_rule == "never":
        return False

    conditions: List[bool] = []
    if profile.evap_activation_dry_bulb_c is not None:
        conditions.append(dry_bulb_c >= profile.evap_activation_dry_bulb_c)
    if profile.evap_activation_wet_bulb_c is not None:
        conditions.append(wet_bulb_c >= profile.evap_activation_wet_bulb_c)
    if not conditions:
        return False
    if profile.activation_logic == "all":
        return all(conditions)
    return any(conditions)


def _compute_direct_wue(profile: CoolingProfile, dry_bulb_c: float, wet_bulb_c: float) -> Tuple[bool, float, float, float]:
    model_wet_bulb_f, wet_tower_wue = _fixed_approach_wue_l_per_kwh(wet_bulb_c)
    evaporative_active = _is_evaporative_mode_active(profile, dry_bulb_c, wet_bulb_c)
    evap_component = wet_tower_wue * profile.evap_multiplier if evaporative_active else 0.0
    direct_wue = max(profile.dry_mode_wue_l_per_kwh + evap_component, 0.0)
    return evaporative_active, wet_tower_wue, model_wet_bulb_f, direct_wue


def _output_path_for_scenario(scenario: str, explicit_output: str) -> Path:
    if explicit_output:
        return Path(explicit_output).resolve()
    return Path(str(DEFAULT_OUTPUT_TEMPLATE).format(scenario=scenario)).resolve()


def _write_output(
    output_path: Path,
    scenario: str,
    region_slots: Mapping[str, Sequence[Tuple[int, str, str]]],
    weather_sites: Mapping[str, WeatherSite],
    cooling_profiles: Mapping[str, Mapping[str, CoolingProfile]],
    weather_cache: Mapping[str, Mapping[str, Mapping[str, float]]],
    request_urls: Mapping[str, str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "region",
                "reference_location",
                "latitude",
                "longitude",
                "weather_site_source_note",
                "weather_site_confidence",
                "slot_index",
                "forecast_datetime_utc",
                "weather_time_utc",
                "dry_bulb_c",
                "wet_bulb_c",
                "wet_bulb_f",
                "model_wet_bulb_f",
                "cooling_scenario",
                "cooling_profile",
                "activation_rule",
                "activation_logic",
                "evaporative_mode_active",
                "dry_mode_wue_l_per_kwh",
                "wet_tower_wue_l_per_kwh",
                "evap_multiplier",
                "evap_activation_dry_bulb_c",
                "evap_activation_wet_bulb_c",
                "direct_wue_l_per_kwh",
                "model",
                "source_dataset",
                "source_url",
                "cooling_profile_source_basis",
                "cooling_profile_notes",
            ]
        )
        for region, slots in region_slots.items():
            site = weather_sites.get(region)
            if site is None:
                raise ValueError(f"Region {region} is missing a weather-site proxy")
            profile = (cooling_profiles.get(region) or {}).get(scenario)
            if profile is None:
                raise ValueError(f"Region {region} is missing a cooling profile for scenario={scenario}")
            hourly_values = weather_cache.get(region, {})
            source_url = request_urls.get(region, "")
            for slot_index, raw_datetime, open_meteo_key in slots:
                slot_weather = hourly_values.get(open_meteo_key)
                if slot_weather is None:
                    raise ValueError(
                        f"Missing weather value for region={region} time={open_meteo_key}"
                    )
                dry_bulb_c = float(slot_weather["dry_bulb_c"])
                wet_bulb_c = float(slot_weather["wet_bulb_c"])
                wet_bulb_f = _c_to_f(wet_bulb_c)
                evaporative_active, wet_tower_wue, model_wet_bulb_f, direct_wue = _compute_direct_wue(
                    profile=profile,
                    dry_bulb_c=dry_bulb_c,
                    wet_bulb_c=wet_bulb_c,
                )
                writer.writerow(
                    [
                        region,
                        site.name,
                        site.latitude,
                        site.longitude,
                        site.source_note,
                        site.confidence,
                        slot_index,
                        raw_datetime,
                        open_meteo_key,
                        dry_bulb_c,
                        wet_bulb_c,
                        wet_bulb_f,
                        model_wet_bulb_f,
                        scenario,
                        profile.profile_class,
                        profile.activation_rule,
                        profile.activation_logic,
                        evaporative_active,
                        profile.dry_mode_wue_l_per_kwh,
                        wet_tower_wue,
                        profile.evap_multiplier,
                        profile.evap_activation_dry_bulb_c,
                        profile.evap_activation_wet_bulb_c,
                        direct_wue,
                        "Gupta wet-tower submodel + explicit cooling-profile scenario",
                        OPEN_METEO_SOURCE,
                        source_url,
                        profile.source_basis,
                        profile.notes,
                    ]
                )


def _scenario_names(raw_scenario: str) -> List[str]:
    if raw_scenario == "all":
        return ["low", "base", "high"]
    return [raw_scenario]


def _refresh_base_aliases(base_output: Path) -> None:
    alias_output = DEFAULT_BASE_OUTPUT.resolve()
    legacy_output = DEFAULT_LEGACY_BASE_OUTPUT.resolve()
    if base_output != alias_output:
        shutil.copyfile(base_output, alias_output)
    if base_output != legacy_output:
        shutil.copyfile(base_output, legacy_output)


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).resolve()
    forecasts_path = Path(args.forecasts).resolve()
    weather_sites_path = Path(args.weather_sites).resolve()
    cooling_profiles_path = Path(args.cooling_profiles).resolve()
    weather_json_path = Path(args.weather_json).resolve() if args.weather_json else None
    cache_json_path = Path(args.cache_json).resolve() if args.cache_json else None

    region_slots = _load_forecast_slots(forecasts_path)
    weather_sites = _load_weather_sites(weather_sites_path, config_path)
    cooling_profiles = _load_cooling_profiles(cooling_profiles_path)
    weather_payload = _load_weather_payload(
        weather_json_path=weather_json_path,
        region_slots=region_slots,
        weather_sites=weather_sites,
        cache_json_path=cache_json_path,
    )
    weather_cache, request_urls = _build_region_weather_cache(weather_payload)

    outputs: Dict[str, Path] = {}
    for scenario in _scenario_names(args.scenario):
        output_path = _output_path_for_scenario(scenario, args.output if args.scenario != "all" else "")
        _write_output(
            output_path=output_path,
            scenario=scenario,
            region_slots=region_slots,
            weather_sites=weather_sites,
            cooling_profiles=cooling_profiles,
            weather_cache=weather_cache,
            request_urls=request_urls,
        )
        outputs[scenario] = output_path

    if "base" in outputs:
        _refresh_base_aliases(outputs["base"])


if __name__ == "__main__":
    main()
