"""
Build a repo-owned region-by-slot direct-WUE table from forecast timestamps.

The workflow is:
- read the regions and representative weather points from infra-workload-config
- read forecast timestamps from all_forecasts.json
- fetch hourly wet-bulb temperatures from Open-Meteo for the covered dates
- convert wet-bulb temperature to direct WUE using the fixed-approach model
  from Gupta et al., e-Energy 2024

The scheduler only consumes the generated CSV. No weather API calls happen at
runtime.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests
import yaml


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = THIS_DIR.parent.parent / "infra-workload-config.yaml"
DEFAULT_FORECASTS_PATH = THIS_DIR.parent.parent / "server-python" / "all_forecasts.json"
DEFAULT_OUTPUT_PATH = THIS_DIR / "open_meteo_region_wue_by_slot.csv"
OPEN_METEO_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OPEN_METEO_SOURCE = "Open-Meteo historical forecast API"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to infra-workload-config.yaml",
    )
    parser.add_argument(
        "--forecasts",
        default=str(DEFAULT_FORECASTS_PATH),
        help="Path to all_forecasts.json",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Destination CSV path",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict:
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


def _load_weather_points(config_path: Path) -> Dict[str, Dict[str, float]]:
    config = _load_yaml(config_path)
    by_region = (((config.get("water") or {}).get("by_region")) or {})
    weather_points: Dict[str, Dict[str, float]] = {}
    for region, entry in by_region.items():
        point = (entry or {}).get("weather_reference", {}) or {}
        latitude = point.get("latitude")
        longitude = point.get("longitude")
        if latitude is None or longitude is None:
            continue
        weather_points[str(region).upper()] = {
            "name": str(point.get("name", region)),
            "latitude": float(latitude),
            "longitude": float(longitude),
        }
    return weather_points


def _iter_region_date_ranges(region_slots: Dict[str, List[Tuple[int, str, str]]]) -> Iterable[Tuple[str, str, str]]:
    for region, slots in region_slots.items():
        datetimes = [datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc) for _, raw, _ in slots]
        start_date = min(datetimes).strftime("%Y-%m-%d")
        end_date = max(datetimes).strftime("%Y-%m-%d")
        yield region, start_date, end_date


def _fetch_hourly_wet_bulb(latitude: float, longitude: float, start_date: str, end_date: str) -> Tuple[Dict[str, float], str]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "wet_bulb_temperature_2m",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "GMT",
    }
    response = requests.get(OPEN_METEO_URL, params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly", {}) or {}
    times = hourly.get("time", []) or []
    wet_bulbs = hourly.get("wet_bulb_temperature_2m", []) or []
    return {str(t): float(v) for t, v in zip(times, wet_bulbs)}, response.url


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


def _write_output(
    output_path: Path,
    region_slots: Dict[str, List[Tuple[int, str, str]]],
    weather_points: Dict[str, Dict[str, float]],
) -> None:
    weather_cache: Dict[str, Dict[str, float]] = {}
    request_urls: Dict[str, str] = {}

    for region, start_date, end_date in _iter_region_date_ranges(region_slots):
        point = weather_points.get(region)
        if point is None:
            raise ValueError(f"Region {region} is missing water.by_region.{region}.weather_reference coordinates")
        hourly_values, request_url = _fetch_hourly_wet_bulb(
            latitude=point["latitude"],
            longitude=point["longitude"],
            start_date=start_date,
            end_date=end_date,
        )
        weather_cache[region] = hourly_values
        request_urls[region] = request_url

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "region",
                "reference_location",
                "latitude",
                "longitude",
                "slot_index",
                "forecast_datetime_utc",
                "weather_time_utc",
                "wet_bulb_c",
                "wet_bulb_f",
                "model_wet_bulb_f",
                "direct_wue_l_per_kwh",
                "model",
                "source_dataset",
                "source_url",
            ]
        )
        for region, slots in region_slots.items():
            point = weather_points[region]
            hourly_values = weather_cache[region]
            source_url = request_urls[region]
            for slot_index, raw_datetime, open_meteo_key in slots:
                if open_meteo_key not in hourly_values:
                    raise ValueError(
                        f"Missing Open-Meteo wet-bulb value for region={region} time={open_meteo_key}"
                    )
                wet_bulb_c = hourly_values[open_meteo_key]
                wet_bulb_f = _c_to_f(wet_bulb_c)
                model_wet_bulb_f, direct_wue = _fixed_approach_wue_l_per_kwh(wet_bulb_c)
                writer.writerow(
                    [
                        region,
                        point["name"],
                        point["latitude"],
                        point["longitude"],
                        slot_index,
                        raw_datetime,
                        open_meteo_key,
                        wet_bulb_c,
                        wet_bulb_f,
                        model_wet_bulb_f,
                        direct_wue,
                        "Gupta et al. 2024 fixed-approach model",
                        OPEN_METEO_SOURCE,
                        source_url,
                    ]
                )


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).resolve()
    forecasts_path = Path(args.forecasts).resolve()
    output_path = Path(args.output).resolve()

    region_slots = _load_forecast_slots(forecasts_path)
    weather_points = _load_weather_points(config_path)
    _write_output(output_path, region_slots, weather_points)


if __name__ == "__main__":
    main()
