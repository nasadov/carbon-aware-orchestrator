"""
Build region-by-slot indirect-water (EWIF) reference tables.

The workflow is:
- read the slot timestamps from all_forecasts.json
- obtain a flow-traced electricity consumption mix for each region and slot
  either from Electricity Maps or from a cached/mock JSON file
- combine the regional technology shares with literature-derived operational
  water factors to compute EWIF per slot

The scheduler consumes only the generated CSVs. No Electricity Maps calls happen
at runtime.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import requests


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_FORECASTS_PATH = THIS_DIR.parent.parent / "server-python" / "all_forecasts.json"
DEFAULT_FACTORS_PATH = THIS_DIR / "ewif_technology_factors.csv"
DEFAULT_OUTPUT_TEMPLATE = THIS_DIR / "ewif_region_slot_{scenario}.csv"
DEFAULT_BASE_OUTPUT = THIS_DIR / "ewif_region_slot.csv"
DEFAULT_TOKEN_ENV = "ELECTRICITY_MAPS_API_TOKEN"
ELECTRICITY_MAPS_BASE_URL = "https://api.electricitymap.org/v3"


@dataclass(frozen=True)
class SlotRecord:
    slot_index: int
    raw_datetime: str
    dt: datetime


@dataclass(frozen=True)
class FactorRule:
    source_type: str
    factor_strategy: str
    low: Optional[float]
    base: Optional[float]
    high: Optional[float]
    source_id: str
    source_note: str

    def factor_for_scenario(self, scenario: str) -> Optional[float]:
        if scenario == "low":
            return self.low
        if scenario == "base":
            return self.base
        if scenario == "high":
            return self.high
        raise ValueError(f"Unknown scenario: {scenario}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecasts",
        default=str(DEFAULT_FORECASTS_PATH),
        help="Path to all_forecasts.json used to define region-slot timestamps.",
    )
    parser.add_argument(
        "--factors",
        default=str(DEFAULT_FACTORS_PATH),
        help="Path to the technology-level EWIF factor table.",
    )
    parser.add_argument(
        "--mix-json",
        default="",
        help="Optional pre-fetched or mock Electricity Maps JSON payload to avoid live API calls.",
    )
    parser.add_argument(
        "--cache-json",
        default="",
        help="Optional path to write the raw fetched Electricity Maps payload.",
    )
    parser.add_argument(
        "--scenario",
        choices=["low", "base", "high", "all"],
        default="all",
        help="Which EWIF scenario(s) to generate.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Destination CSV for a single scenario. Ignored when --scenario=all.",
    )
    parser.add_argument(
        "--token-env",
        default=DEFAULT_TOKEN_ENV,
        help="Environment variable containing the Electricity Maps API token.",
    )
    parser.add_argument(
        "--source-mode",
        choices=["auto", "forecast", "past-range"],
        default="auto",
        help="Choose whether to fetch forecast or historical power-breakdown data when --mix-json is not supplied.",
    )
    parser.add_argument(
        "--disable-estimations",
        action="store_true",
        help="Ask Electricity Maps to omit estimated values when live-fetching.",
    )
    return parser.parse_args()


def _parse_iso_utc(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)


def _normalize_iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _load_forecast_slots(path: Path) -> Dict[str, List[SlotRecord]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    result: Dict[str, List[SlotRecord]] = {}
    for region, region_payload in data.items():
        forecasts = (region_payload or {}).get("forecast", []) or []
        region_slots: List[SlotRecord] = []
        for slot_index, entry in enumerate(forecasts):
            raw_datetime = str(entry.get("datetime", "")).strip()
            if not raw_datetime:
                continue
            region_slots.append(
                SlotRecord(
                    slot_index=slot_index,
                    raw_datetime=raw_datetime,
                    dt=_parse_iso_utc(raw_datetime),
                )
            )
        if region_slots:
            result[str(region).upper()] = region_slots
    return result


def _load_factor_rules(path: Path) -> Dict[str, FactorRule]:
    rules: Dict[str, FactorRule] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_type = _canonicalize_source_type(row.get("source_type", ""))
            if not source_type:
                continue
            rules[source_type] = FactorRule(
                source_type=source_type,
                factor_strategy=str(row.get("factor_strategy", "fixed") or "fixed").strip(),
                low=_safe_optional_float(row.get("water_consumption_l_per_kwh_low")),
                base=_safe_optional_float(row.get("water_consumption_l_per_kwh_base")),
                high=_safe_optional_float(row.get("water_consumption_l_per_kwh_high")),
                source_id=str(row.get("source_id", "")).strip(),
                source_note=str(row.get("source_note", "")).strip(),
            )
    if not rules:
        raise ValueError(f"No EWIF technology factors found in {path}")
    return rules


def _safe_optional_float(value: Any) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    return float(value)


def _canonicalize_source_type(value: Any) -> str:
    return str(value).strip().lower().replace("_", "-").replace(" ", "-")


def _slot_range_mode(region_slots: Mapping[str, Sequence[SlotRecord]], explicit_mode: str) -> str:
    if explicit_mode != "auto":
        return explicit_mode
    now = datetime.now(timezone.utc)
    latest_slot = max(slot.dt for slots in region_slots.values() for slot in slots)
    return "forecast" if latest_slot >= now.replace(minute=0, second=0, microsecond=0) else "past-range"


def _needed_horizon_hours(region_slots: Sequence[SlotRecord]) -> int:
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    latest_slot = max(slot.dt for slot in region_slots)
    delta_hours = max(int(math.ceil((latest_slot - now).total_seconds() / 3600.0)), 0)
    if delta_hours <= 24:
        return 24
    if delta_hours <= 48:
        return 48
    if delta_hours <= 72:
        return 72
    raise ValueError(
        f"Requested slot horizon requires {delta_hours} hours, but Electricity Maps forecast access tops out at 72 hours."
    )


def _fetch_live_mix_payload(
    region_slots: Mapping[str, Sequence[SlotRecord]],
    token: str,
    source_mode: str,
    disable_estimations: bool,
) -> Dict[str, Dict[str, Any]]:
    payload: Dict[str, Dict[str, Any]] = {}
    headers = {"auth-token": token}
    mode = _slot_range_mode(region_slots, source_mode)

    for region, slots in region_slots.items():
        params: Dict[str, Any] = {
            "zone": region,
            "temporalGranularity": "hourly",
        }
        if disable_estimations:
            params["disableEstimations"] = "true"

        if mode == "forecast":
            endpoint = f"{ELECTRICITY_MAPS_BASE_URL}/power-breakdown/forecast"
            params["horizonHours"] = _needed_horizon_hours(slots)
        else:
            endpoint = f"{ELECTRICITY_MAPS_BASE_URL}/power-breakdown/past-range"
            start = min(slot.dt for slot in slots)
            end = max(slot.dt for slot in slots) + timedelta(hours=1)
            params["start"] = _normalize_iso_utc(start)
            params["end"] = _normalize_iso_utc(end)

        response = requests.get(endpoint, headers=headers, params=params, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(
                f"Electricity Maps request failed for region={region} mode={mode} "
                f"(HTTP {response.status_code}): {response.text[:300]}"
            )

        region_payload = response.json()
        if isinstance(region_payload, dict):
            region_payload.setdefault("_request_url", response.url)
            region_payload.setdefault("_request_mode", mode)
        payload[region] = region_payload

    return payload


def _load_mix_payload(
    mix_json_path: Optional[Path],
    region_slots: Mapping[str, Sequence[SlotRecord]],
    token_env: str,
    source_mode: str,
    disable_estimations: bool,
    cache_json_path: Optional[Path],
) -> Dict[str, Dict[str, Any]]:
    if mix_json_path is not None:
        return json.loads(mix_json_path.read_text(encoding="utf-8"))

    import os

    token = os.environ.get(token_env, "").strip()
    if not token:
        raise RuntimeError(
            f"No Electricity Maps token available. Set {token_env} or pass --mix-json with a cached/mock payload."
        )

    payload = _fetch_live_mix_payload(region_slots, token, source_mode, disable_estimations)
    if cache_json_path is not None:
        cache_json_path.parent.mkdir(parents=True, exist_ok=True)
        cache_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _extract_series(region_payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    for key in ("forecast", "history", "past", "data"):
        series = region_payload.get(key)
        if isinstance(series, list):
            return [entry for entry in series if isinstance(entry, Mapping)]
    if "datetime" in region_payload and "powerConsumptionBreakdown" in region_payload:
        return [region_payload]
    raise ValueError("Could not locate electricity-mix time series in payload")


def _build_entry_index(series: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    index: Dict[str, Mapping[str, Any]] = {}
    for entry in series:
        raw_datetime = str(entry.get("datetime", "")).strip()
        if not raw_datetime:
            continue
        index[_normalize_iso_utc(_parse_iso_utc(raw_datetime))] = entry
    return index


def _entry_breakdown(entry: Mapping[str, Any]) -> Dict[str, float]:
    breakdown = entry.get("powerConsumptionBreakdown")
    if not isinstance(breakdown, Mapping):
        raise ValueError("Entry is missing powerConsumptionBreakdown")
    result: Dict[str, float] = {}
    for raw_key, raw_value in breakdown.items():
        if raw_value in (None, ""):
            continue
        value = float(raw_value)
        if value <= 0.0:
            continue
        result[_canonicalize_source_type(raw_key)] = value
    return result


def _entry_total(entry: Mapping[str, Any], breakdown: Mapping[str, float]) -> float:
    total = entry.get("powerConsumptionTotal")
    if total not in (None, ""):
        total_value = float(total)
        if total_value > 0.0:
            return total_value
    return float(sum(breakdown.values()))


def _compute_slot_ewif(
    breakdown: Mapping[str, float],
    total: float,
    factor_rules: Mapping[str, FactorRule],
    scenario: str,
) -> Tuple[float, float, float, str]:
    if total <= 0.0:
        raise ValueError("Power consumption total must be positive")

    fixed_contribution = 0.0
    fixed_share = 0.0
    inherit_share = 0.0
    notes: List[str] = []

    for source_type, power in breakdown.items():
        share = float(power) / total
        rule = factor_rules.get(source_type)
        if rule is None:
            inherit_share += share
            notes.append(f"unmapped:{source_type}")
            continue

        if rule.factor_strategy == "inherit_non_storage_mix":
            inherit_share += share
            notes.append(f"inherit:{source_type}")
            continue

        factor = rule.factor_for_scenario(scenario)
        if factor is None:
            inherit_share += share
            notes.append(f"missing_factor:{source_type}")
            continue

        fixed_share += share
        fixed_contribution += share * factor

    inherited_factor = (fixed_contribution / fixed_share) if fixed_share > 0.0 else 0.0
    ewif = fixed_contribution + inherit_share * inherited_factor
    return ewif, fixed_share, inherit_share, ";".join(notes)


def _write_rows_for_scenario(
    output_path: Path,
    scenario: str,
    region_slots: Mapping[str, Sequence[SlotRecord]],
    mix_payload: Mapping[str, Mapping[str, Any]],
    factor_rules: Mapping[str, FactorRule],
    source_label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "region",
                "slot_index",
                "forecast_datetime_utc",
                "mix_datetime_utc",
                "ewif_l_per_kwh",
                "ewif_scenario",
                "power_consumption_total_mw",
                "non_storage_share",
                "inherit_share",
                "is_estimated",
                "estimation_method",
                "mix_notes",
                "source_dataset",
                "source_url",
            ]
        )

        for region, slots in region_slots.items():
            region_payload = mix_payload.get(region)
            if region_payload is None:
                raise ValueError(f"No electricity-mix payload found for region={region}")
            entry_index = _build_entry_index(_extract_series(region_payload))
            source_url = str(region_payload.get("_request_url", source_label))

            for slot in slots:
                key = _normalize_iso_utc(slot.dt)
                entry = entry_index.get(key)
                if entry is None:
                    raise ValueError(f"Missing electricity-mix entry for region={region} datetime={key}")

                breakdown = _entry_breakdown(entry)
                total = _entry_total(entry, breakdown)
                ewif, fixed_share, inherit_share, notes = _compute_slot_ewif(
                    breakdown=breakdown,
                    total=total,
                    factor_rules=factor_rules,
                    scenario=scenario,
                )
                writer.writerow(
                    [
                        region,
                        slot.slot_index,
                        slot.raw_datetime,
                        key,
                        ewif,
                        scenario,
                        total,
                        fixed_share,
                        inherit_share,
                        bool(entry.get("isEstimated", False)),
                        entry.get("estimationMethod", ""),
                        notes,
                        "Electricity Maps flow-traced power breakdown + literature-derived operational water factors",
                        source_url,
                    ]
                )


def _scenario_output_path(args: argparse.Namespace, scenario: str) -> Path:
    if args.scenario != "all" and args.output:
        return Path(args.output).resolve()
    return Path(str(DEFAULT_OUTPUT_TEMPLATE).format(scenario=scenario)).resolve()


def main() -> None:
    args = _parse_args()
    forecasts_path = Path(args.forecasts).resolve()
    factors_path = Path(args.factors).resolve()
    mix_json_path = Path(args.mix_json).resolve() if args.mix_json else None
    cache_json_path = Path(args.cache_json).resolve() if args.cache_json else None

    region_slots = _load_forecast_slots(forecasts_path)
    factor_rules = _load_factor_rules(factors_path)
    mix_payload = _load_mix_payload(
        mix_json_path=mix_json_path,
        region_slots=region_slots,
        token_env=args.token_env,
        source_mode=args.source_mode,
        disable_estimations=bool(args.disable_estimations),
        cache_json_path=cache_json_path,
    )
    source_label = str(mix_json_path) if mix_json_path is not None else "Electricity Maps API"

    scenarios = ["low", "base", "high"] if args.scenario == "all" else [args.scenario]
    for scenario in scenarios:
        output_path = _scenario_output_path(args, scenario)
        _write_rows_for_scenario(
            output_path=output_path,
            scenario=scenario,
            region_slots=region_slots,
            mix_payload=mix_payload,
            factor_rules=factor_rules,
            source_label=source_label,
        )
        if scenario == "base" and args.scenario == "all":
            _write_rows_for_scenario(
                output_path=DEFAULT_BASE_OUTPUT.resolve(),
                scenario=scenario,
                region_slots=region_slots,
                mix_payload=mix_payload,
                factor_rules=factor_rules,
                source_label=source_label,
            )


if __name__ == "__main__":
    main()
