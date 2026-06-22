"""Pilot utilities for no-harm flexibility-envelope experiments.

This module intentionally keeps the pilot separate from the production scheduler.
It reuses the existing feasible-candidate and footprint machinery, then runs a
baseline-plus-repair experiment that tests whether actionable grid/cooling/water
signals can produce grid-stress relief while preserving ex-post carbon, scarcity
water, and SLO guardrails.
"""
from __future__ import annotations

import copy
import csv
import json
import logging
import os
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from carbon_aware.algorithms.heuristic import (
    CandidatePlacement,
    _build_feasible_candidates,
    _candidate_water_value,
    _rank_candidates,
    find_ranked_candidates,
)
from carbon_aware.models import CarbonAwarePod, CarbonAwareTimeslot, EnvironmentalFlavor
from carbon_aware.precompute_heuristic import _extract_pods_from_yaml, _load_nodes_from_yaml
from carbon_aware.utils import build_timeslots, load_carbon_intensity_data


RegionSlot = Tuple[str, int]


@dataclass(frozen=True)
class ActionSignal:
    region: str
    slot: int
    carbon_intensity_g_per_kwh: float
    demand_mw: float
    residual_load_proxy_mw: float
    grid_stress_score: float
    grid_stress: bool
    clean_headroom: bool
    wet_bulb_c: float
    direct_wue_l_per_kwh: float
    scarcity_cf: float
    drought_guardrail: bool
    grid_signal_source: str


@dataclass
class Placement:
    pod: CarbonAwarePod
    candidate: CandidatePlacement
    flexibility_class: str


@dataclass
class ScheduleResult:
    method_key: str
    placements: List[Placement]
    unplaced_pods: List[CarbonAwarePod]
    elapsed_seconds: float
    repairs_applied: int = 0
    rejected_moves: int = 0
    reference_method: str = ""


@dataclass(frozen=True)
class PilotConfig:
    repo_root: Path
    nodes_file: Path
    workloads_dir: Path
    forecasts_file: Path
    config_file: Path
    output_dir: Path
    max_timeslots: int = 24
    max_pods: Optional[int] = 80
    flexibility_slack_hours: float = 2.0
    stress_quantile: float = 0.75
    headroom_quantile: float = 0.25
    drought_cf_threshold: float = 20.0
    scenario: str = "heatwave-drought"
    regret_margin: float = 0.0
    max_repairs: Optional[int] = None
    grid_signal_csv: Optional[Path] = None
    lever_mode: str = "both"  # both | temporal (same site, shift time) | spatial (same time, move site)
    wue_csv: Optional[Path] = None  # override the scenario WUE table (e.g. a time-aligned real window)
    forecast_noise: float = 0.0  # RQ3: stdev of multiplicative carbon-forecast error used for DECISIONS
    forecast_seed: int = 0
    # RQ3 robust guard: assumed carbon-forecast-error fraction. Each accepted move adds its
    # forecast-error EXPOSURE (robust_buffer x the OPERATIONAL carbon it relocates) to a
    # cumulative worst-case carbon accumulator that the no-harm guard keeps under baseline,
    # so realized carbon stays <= baseline even when the forecast is wrong. Applied to the
    # CARBON guard only (water is decided on OBSERVED wet-bulb -> no forecast error).
    # Default 0.0 -> identical to the plain forecast guard (bit-identical).
    robust_buffer: float = 0.0


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize(values: Mapping[RegionSlot, float]) -> Dict[RegionSlot, float]:
    if not values:
        return {}
    lower = min(values.values())
    upper = max(values.values())
    if upper <= lower + 1e-12:
        return {key: 0.0 for key in values}
    return {key: (value - lower) / (upper - lower) for key, value in values.items()}


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    q = min(max(float(q), 0.0), 1.0)
    idx = int(round((len(ordered) - 1) * q))
    return ordered[idx]


def _read_region_slot_csv(path: Path) -> Dict[RegionSlot, Dict[str, str]]:
    if not path.exists():
        return {}
    rows: Dict[RegionSlot, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            region = (row.get("region") or "").strip().upper()
            if not region:
                continue
            try:
                slot = int(row.get("slot_index") or 0)
            except (TypeError, ValueError):
                continue
            rows[(region, slot)] = row
    return rows


def _read_aware_rows(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    rows: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            country = (row.get("country_code") or "").strip().upper()
            if country:
                rows[country] = row
    return rows


def _country_for_region(region: str) -> str:
    if region == "IT-NO":
        return "IT"
    if "-" in region:
        return region.split("-", 1)[0].upper()
    return region.upper()


def _water_data_path(repo_root: Path, filename: str) -> Path:
    return repo_root / "pkg" / "carbon-aware" / "data" / "water" / filename


def _grid_data_path(repo_root: Path, filename: str) -> Path:
    return repo_root / "pkg" / "carbon-aware" / "data" / "grid" / filename


def scenario_wue_path(config: PilotConfig) -> Path:
    if config.wue_csv is not None:
        return config.wue_csv
    if config.scenario == "heatwave-drought":
        return _water_data_path(config.repo_root, "wue_region_slot_high.csv")
    return _water_data_path(config.repo_root, "wue_region_slot_base.csv")


def grid_signal_path(config: PilotConfig) -> Optional[Path]:
    if config.grid_signal_csv is not None:
        return config.grid_signal_csv
    default_path = _grid_data_path(config.repo_root, "opsd_residual_load_region_slot.csv")
    return default_path if default_path.exists() else None


def apply_pilot_scenario_to_flavours(flavours: Sequence[EnvironmentalFlavor], config: PilotConfig) -> None:
    """Apply the selected pilot scenario to WUE and scarcity factors.

    The heatwave-drought scenario is a transparent sensitivity replay: it uses the
    existing high-water cooling table plus July AWARE factors to mimic a stressed
    summer window. It is a pilot stress test, not a claim that the local forecast
    file itself occurred in July.
    """
    wue_rows = _read_region_slot_csv(scenario_wue_path(config))
    aware_rows = _read_aware_rows(_water_data_path(config.repo_root, "aware20_country_nonagri_factors.csv"))
    # Indirect (power-plant) water: EWIF per kWh of grid electricity, from a flow-traced
    # generation mix x literature water-consumption factors (Macknick 2012; Spang 2014).
    ewif_rows = _read_region_slot_csv(_water_data_path(config.repo_root, "ewif_region_slot.csv"))

    for flavour in flavours:
        region = (getattr(flavour, "region", "") or "").upper()
        wue_by_slot = {
            slot: _as_float(row.get("direct_wue_l_per_kwh"), 0.0)
            for (row_region, slot), row in wue_rows.items()
            if row_region == region
        }
        if wue_by_slot:
            flavour.wue_by_slot = wue_by_slot

        # Step 4 / T6: fixed per-site cooling architecture (kappa) with a temperature-
        # dependent PUE. The time-aligned tables carry a per-SLOT pue so dry cooling costs
        # energy -> carbon (facility energy = IT x PUE in the footprint) while saving water;
        # hotter hours raise dry-cooling PUE (the heatwave carbon<->water coupling). Legacy
        # tables (constant pue) still work -> pue_by_slot is constant.
        pue_by_slot = {
            slot: _as_float(row.get("pue"))
            for (row_region, slot), row in wue_rows.items()
            if row_region == region and _as_float(row.get("pue")) > 0
        }
        if pue_by_slot:
            flavour.pue_by_slot = pue_by_slot
            flavour.pue = max(pue_by_slot.values())  # scalar fallback = peak overhead

        # Indirect-water EWIF per slot (power-plant freshwater for the grid electricity drawn).
        ewif_by_slot = {
            slot: _as_float(row.get("ewif_l_per_kwh"))
            for (row_region, slot), row in ewif_rows.items()
            if row_region == region and _as_float(row.get("ewif_l_per_kwh")) > 0
        }
        if ewif_by_slot:
            flavour.ewif_by_slot = ewif_by_slot

        if config.scenario == "heatwave-drought":
            country = _country_for_region(region)
            aware_row = aware_rows.get(country, {})
            summer_cf = _as_float(aware_row.get("jul_cf"), getattr(flavour, "water_scarcity_direct_cf", 1.0))
            flavour.water_scarcity_direct_cf = summer_cf
            flavour.water_scarcity_indirect_cf = summer_cf
            flavour.water_scarcity_direct_cf_by_slot = {slot: summer_cf for slot in range(config.max_timeslots)}
            flavour.water_scarcity_indirect_cf_by_slot = {slot: summer_cf for slot in range(config.max_timeslots)}


def build_action_signals(flavours: Sequence[EnvironmentalFlavor], config: PilotConfig) -> Dict[RegionSlot, ActionSignal]:
    forecasts = load_carbon_intensity_data(str(config.forecasts_file))
    wue_rows = _read_region_slot_csv(scenario_wue_path(config))
    ewif_rows = _read_region_slot_csv(_water_data_path(config.repo_root, "ewif_region_slot.csv"))
    grid_rows = _read_region_slot_csv(grid_signal_path(config)) if grid_signal_path(config) else {}

    regions = sorted({(getattr(flv, "region", "") or "").upper() for flv in flavours if getattr(flv, "region", "")})
    carbon_values: Dict[RegionSlot, float] = {}
    demand_values: Dict[RegionSlot, float] = {}

    for region in regions:
        region_forecast = forecasts.get(region, {})
        for slot in range(config.max_timeslots):
            key = (region, slot)
            carbon_values[key] = _as_float(region_forecast.get(slot), 200.0)
            demand_values[key] = _as_float(ewif_rows.get(key, {}).get("power_consumption_total_mw"), 1.0)

    grid_signal_source = "opsd_residual_load_lite"
    residual_proxy = {
        key: _as_float(grid_rows.get(key, {}).get("residual_load_mw"), 0.0)
        for key in carbon_values
    }
    if not grid_rows or all(value == 0.0 for value in residual_proxy.values()):
        # The fallback deliberately keeps grid stress separate from the carbon
        # account. It is useful for plumbing tests, but publication evidence
        # should use the OPSD residual-load table generated by the data builder.
        total_demand_by_slot = {
            slot: sum(demand_values.get((region, slot), 0.0) for region in regions)
            for slot in range(config.max_timeslots)
        }
        residual_proxy = {key: total_demand_by_slot[key[1]] for key in carbon_values}
        grid_signal_source = "fallback_system_demand_proxy"
    stress_score = _normalize(residual_proxy)
    stress_cut = _quantile(list(stress_score.values()), config.stress_quantile)
    headroom_cut = _quantile(list(stress_score.values()), config.headroom_quantile)

    flavour_by_region = {(getattr(flv, "region", "") or "").upper(): flv for flv in flavours}
    signals: Dict[RegionSlot, ActionSignal] = {}
    for key, carbon in carbon_values.items():
        region, slot = key
        flavour = flavour_by_region[region]
        scarcity_by_slot = getattr(flavour, "water_scarcity_direct_cf_by_slot", {}) or {}
        scarcity_cf = _as_float(scarcity_by_slot.get(slot), getattr(flavour, "water_scarcity_direct_cf", 1.0))
        wue_row = wue_rows.get(key, {})
        wet_bulb = _as_float(wue_row.get("wet_bulb_c"), 0.0)
        direct_wue = _as_float(wue_row.get("direct_wue_l_per_kwh"), 0.0)
        score = stress_score.get(key, 0.0)
        signals[key] = ActionSignal(
            region=region,
            slot=slot,
            carbon_intensity_g_per_kwh=carbon,
            demand_mw=demand_values[key],
            residual_load_proxy_mw=residual_proxy[key],
            grid_stress_score=score,
            grid_stress=score >= stress_cut - 1e-12,
            clean_headroom=score <= headroom_cut + 1e-12,
            wet_bulb_c=wet_bulb,
            direct_wue_l_per_kwh=direct_wue,
            scarcity_cf=scarcity_cf,
            drought_guardrail=scarcity_cf >= config.drought_cf_threshold,
            grid_signal_source=grid_signal_source,
        )
    return signals


def load_pods(workloads_dir: Path, *, max_pods: Optional[int] = None) -> List[CarbonAwarePod]:
    pods: List[CarbonAwarePod] = []
    files = sorted(workloads_dir.glob("timeslot_*.yaml"), key=lambda path: int(path.stem.split("_")[-1]))
    for path in files:
        earliest = int(path.stem.split("_")[-1])
        for pod in _extract_pods_from_yaml(str(path)):
            pod.earliest_timeslot = earliest
            pod.calculate_deadline_slot()
            pods.append(pod)
            if max_pods is not None and len(pods) >= max_pods:
                return pods
    return pods


def order_pods_for_pilot(pods: Sequence[CarbonAwarePod]) -> List[CarbonAwarePod]:
    def key(pod: CarbonAwarePod) -> Tuple[float, float, float, float, str]:
        slack = float(getattr(pod, "deadline_hours", 24.0)) - float(getattr(pod, "duration", 1.0))
        return (
            float(getattr(pod, "earliest_timeslot", 0)),
            slack,
            -float(getattr(pod, "cpuRequest", 0.0)),
            -float(getattr(pod, "ramRequest", 0.0)),
            pod.id,
        )

    return sorted(pods, key=key)


def classify_pod(pod: CarbonAwarePod, slack_threshold_hours: float) -> str:
    # Honour an explicit firm/flexible tier when the trace provides one (e.g. Alibaba GPU
    # job roles: training=flexible, inference/serving=firm); otherwise fall back to slack.
    tier = getattr(pod, "tier", None)
    if tier == "flexible":
        return "flexible"
    if tier in ("firm", "protected"):
        return "protected"
    slack = float(getattr(pod, "deadline_hours", 0.0)) - float(getattr(pod, "duration", 0.0))
    return "flexible" if slack >= slack_threshold_hours else "protected"


def _init_resources(flavours: Sequence[EnvironmentalFlavor], max_timeslots: int) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, float]], Dict[str, Dict[int, float]]]:
    leftover_cpu = {flv.id: {slot: flv.totalCpu for slot in range(max_timeslots)} for flv in flavours}
    leftover_ram = {flv.id: {slot: flv.totalRam for slot in range(max_timeslots)} for flv in flavours}
    leftover_gpu = {flv.id: {slot: float(getattr(flv, "totalGpu", 0) or 0) for slot in range(max_timeslots)} for flv in flavours}
    return leftover_cpu, leftover_ram, leftover_gpu


def _apply_candidate_resources(
    pod: CarbonAwarePod,
    candidate: CandidatePlacement,
    leftover_cpu: Dict[str, Dict[int, float]],
    leftover_ram: Dict[str, Dict[int, float]],
    leftover_gpu: Optional[Dict[str, Dict[int, float]]] = None,
    *,
    release: bool = False,
) -> None:
    sign = 1.0 if release else -1.0
    gpu_req = float(getattr(pod, "gpuRequest", 0) or 0)
    for offset in range(int(pod.duration)):
        slot = candidate.timeslot.id + offset
        leftover_cpu[candidate.flavour.id][slot] += sign * pod.cpuRequest
        leftover_ram[candidate.flavour.id][slot] += sign * pod.ramRequest
        if leftover_gpu is not None and gpu_req:
            leftover_gpu[candidate.flavour.id][slot] += sign * gpu_req


def _candidate_sort_key(candidate: CandidatePlacement, method_key: str) -> Tuple[float, ...]:
    water_value = _candidate_water_value(candidate.footprint, "scarcity")
    if method_key == "packing":
        return (
            candidate.timeslot.id,
            candidate.pack_score,
            candidate.footprint.total_carbon_g,
            water_value,
        )
    if method_key == "water_scarcity":
        return (
            water_value,
            candidate.footprint.total_carbon_g,
            candidate.pack_score,
            candidate.timeslot.id,
        )
    return (
        candidate.footprint.total_carbon_g,
        water_value,
        candidate.pack_score,
        candidate.timeslot.id,
    )


def build_greedy_schedule(
    *,
    method_key: str,
    pods: Sequence[CarbonAwarePod],
    flavours: Sequence[EnvironmentalFlavor],
    config: PilotConfig,
) -> Tuple[ScheduleResult, Dict[str, Dict[int, float]], Dict[str, Dict[int, float]]]:
    start = time.perf_counter()
    leftover_cpu, leftover_ram, leftover_gpu = _init_resources(flavours, config.max_timeslots)
    placements: List[Placement] = []
    unplaced: List[CarbonAwarePod] = []
    timeslots = build_timeslots(config.max_timeslots)

    for pod in order_pods_for_pilot(pods):
        ranked = find_ranked_candidates(
            pod=pod,
            flavours=list(flavours),
            timeslots=timeslots,
            leftover_cpu=leftover_cpu,
            leftover_ram=leftover_ram,
            max_time_slots=config.max_timeslots,
            objective_mode="carbon",
            water_metric="scarcity",
            leftover_gpu=leftover_gpu,
        )
        if not ranked:
            unplaced.append(pod)
            continue

        candidate = sorted(ranked, key=lambda item: _candidate_sort_key(item, method_key))[0]
        _apply_candidate_resources(pod, candidate, leftover_cpu, leftover_ram, leftover_gpu)
        placements.append(
            Placement(
                pod=pod,
                candidate=candidate,
                flexibility_class=classify_pod(pod, config.flexibility_slack_hours),
            )
        )

    return (
        ScheduleResult(
            method_key=method_key,
            placements=placements,
            unplaced_pods=unplaced,
            elapsed_seconds=time.perf_counter() - start,
        ),
        leftover_cpu,
        leftover_ram,
    )


def placement_signal_metrics(placement: Placement, signals: Mapping[RegionSlot, ActionSignal]) -> Dict[str, float]:
    candidate = placement.candidate
    region = (getattr(candidate.flavour, "region", "") or "").upper()
    duration = max(int(placement.pod.duration), 1)
    energy_per_slot = candidate.footprint.operational_energy_kwh / duration
    direct_water_per_slot = candidate.footprint.direct_water_l / duration
    stress_kwh = 0.0
    weighted_stress_kwh = 0.0
    headroom_kwh = 0.0
    drought_direct_water_l = 0.0
    drought_scarcity_water = 0.0

    for offset in range(duration):
        slot = candidate.timeslot.id + offset
        signal = signals.get((region, slot))
        if signal is None:
            continue
        weighted_stress_kwh += energy_per_slot * signal.grid_stress_score
        if signal.grid_stress:
            stress_kwh += energy_per_slot
        if signal.clean_headroom:
            headroom_kwh += energy_per_slot
        if signal.drought_guardrail:
            drought_direct_water_l += direct_water_per_slot
            drought_scarcity_water += direct_water_per_slot * signal.scarcity_cf

    return {
        "stress_kwh": stress_kwh,
        "weighted_stress_kwh": weighted_stress_kwh,
        "headroom_kwh": headroom_kwh,
        "drought_direct_water_l": drought_direct_water_l,
        "drought_scarcity_water": drought_scarcity_water,
    }


def schedule_totals(result: ScheduleResult, signals: Mapping[RegionSlot, ActionSignal]) -> Dict[str, float]:
    totals = {
        "placed_pods": float(len(result.placements)),
        "unplaced_pods": float(len(result.unplaced_pods)),
        "flexible_pods": 0.0,
        "protected_pods": 0.0,
        "operational_energy_kwh": 0.0,
        "carbon_kg": 0.0,
        "operational_carbon_kg": 0.0,
        "embodied_carbon_kg": 0.0,
        "raw_water_l": 0.0,
        "direct_water_l": 0.0,
        "indirect_water_l": 0.0,
        "embodied_water_l": 0.0,
        "scarcity_water": 0.0,
        "criticality_adjusted_water": 0.0,
        "stress_kwh": 0.0,
        "weighted_stress_kwh": 0.0,
        "headroom_kwh": 0.0,
        "drought_direct_water_l": 0.0,
        "drought_scarcity_water": 0.0,
    }
    for placement in result.placements:
        fp = placement.candidate.footprint
        totals["flexible_pods" if placement.flexibility_class == "flexible" else "protected_pods"] += 1.0
        totals["operational_energy_kwh"] += fp.operational_energy_kwh
        totals["carbon_kg"] += fp.total_carbon_kg
        totals["operational_carbon_kg"] += fp.operational_carbon_kg
        totals["embodied_carbon_kg"] += fp.embodied_carbon_kg
        totals["raw_water_l"] += fp.total_raw_water_l
        totals["direct_water_l"] += fp.direct_water_l
        totals["indirect_water_l"] += fp.indirect_water_l
        totals["embodied_water_l"] += fp.embodied_water_l
        totals["scarcity_water"] += fp.scarcity_characterized_water
        totals["criticality_adjusted_water"] += fp.criticality_adjusted_water
        signal_metrics = placement_signal_metrics(placement, signals)
        for key, value in signal_metrics.items():
            totals[key] += value
    return totals


def _stress_load_by_region_slot(result: ScheduleResult, signals: Mapping[RegionSlot, ActionSignal]) -> Dict[RegionSlot, float]:
    loads: Dict[RegionSlot, float] = {}
    for placement in result.placements:
        region = (getattr(placement.candidate.flavour, "region", "") or "").upper()
        duration = max(int(placement.pod.duration), 1)
        energy_per_slot = placement.candidate.footprint.operational_energy_kwh / duration
        for offset in range(duration):
            slot = placement.candidate.timeslot.id + offset
            signal = signals.get((region, slot))
            if signal is not None and signal.grid_stress:
                loads[(region, slot)] = loads.get((region, slot), 0.0) + energy_per_slot
    return loads


def _placement_by_pod(result: ScheduleResult) -> Dict[str, Placement]:
    return {placement.pod.id: placement for placement in result.placements}


def summarize_against_reference(
    result: ScheduleResult,
    reference: ScheduleResult,
    signals: Mapping[RegionSlot, ActionSignal],
    *,
    tolerance: float = 1e-9,
) -> Dict[str, Any]:
    totals = schedule_totals(result, signals)
    reference_totals = schedule_totals(reference, signals)
    result_by_pod = _placement_by_pod(result)
    reference_by_pod = _placement_by_pod(reference)
    common_pods = sorted(set(result_by_pod) & set(reference_by_pod))
    moved = [
        pod_id
        for pod_id in common_pods
        if (
            result_by_pod[pod_id].candidate.flavour.id != reference_by_pod[pod_id].candidate.flavour.id
            or result_by_pod[pod_id].candidate.timeslot.id != reference_by_pod[pod_id].candidate.timeslot.id
        )
    ]

    reference_stress_load = _stress_load_by_region_slot(reference, signals)
    result_stress_load = _stress_load_by_region_slot(result, signals)
    stress_hours_covered = sum(
        1
        for key, base_load in reference_stress_load.items()
        if base_load > result_stress_load.get(key, 0.0) + tolerance
    )

    carbon_delta = totals["carbon_kg"] - reference_totals["carbon_kg"]
    scarcity_delta = totals["scarcity_water"] - reference_totals["scarcity_water"]
    stress_delta = totals["stress_kwh"] - reference_totals["stress_kwh"]
    headroom_delta = totals["headroom_kwh"] - reference_totals["headroom_kwh"]
    no_harm = (
        totals["placed_pods"] >= reference_totals["placed_pods"]
        and totals["unplaced_pods"] <= reference_totals["unplaced_pods"]
        and carbon_delta <= tolerance
        and scarcity_delta <= tolerance
    )

    return {
        "method_key": result.method_key,
        "reference_method": reference.method_key,
        "no_harm_certificate": bool(no_harm),
        "carbon_nonincrease": bool(carbon_delta <= tolerance),
        "scarcity_nonincrease": bool(scarcity_delta <= tolerance),
        "slo_non_decrease": bool(
            totals["placed_pods"] >= reference_totals["placed_pods"]
            and totals["unplaced_pods"] <= reference_totals["unplaced_pods"]
        ),
        "placed_pods": int(totals["placed_pods"]),
        "unplaced_pods": int(totals["unplaced_pods"]),
        "common_pods": len(common_pods),
        "moved_common_pods": len(moved),
        "flexible_pods": int(totals["flexible_pods"]),
        "protected_pods": int(totals["protected_pods"]),
        "repairs_applied": result.repairs_applied,
        "rejected_moves": result.rejected_moves,
        "elapsed_seconds": result.elapsed_seconds,
        "carbon_kg": totals["carbon_kg"],
        "carbon_delta_kg": carbon_delta,
        "carbon_delta_pct": _percent_delta(totals["carbon_kg"], reference_totals["carbon_kg"]),
        "scarcity_water": totals["scarcity_water"],
        "scarcity_delta": scarcity_delta,
        "scarcity_delta_pct": _percent_delta(totals["scarcity_water"], reference_totals["scarcity_water"]),
        "raw_water_l": totals["raw_water_l"],
        "direct_water_l": totals["direct_water_l"],
        "indirect_water_l": totals["indirect_water_l"],
        "operational_energy_kwh": totals["operational_energy_kwh"],
        "stress_kwh": totals["stress_kwh"],
        "stress_kwh_delta": stress_delta,
        "stress_kwh_avoided": max(-stress_delta, 0.0),
        "stress_kwh_avoided_pct": _percent_delta(reference_totals["stress_kwh"], totals["stress_kwh"]),
        "weighted_stress_kwh": totals["weighted_stress_kwh"],
        "headroom_kwh": totals["headroom_kwh"],
        "headroom_kwh_gain": max(headroom_delta, 0.0),
        "grid_stress_hours_covered": stress_hours_covered,
        "drought_direct_water_l": totals["drought_direct_water_l"],
        "drought_scarcity_water": totals["drought_scarcity_water"],
    }


def _percent_delta(new_value: float, old_value: float) -> float:
    if abs(old_value) <= 1e-12:
        return 0.0
    return (new_value - old_value) / old_value * 100.0


def _projected_totals(
    current: Mapping[str, float],
    old_placement: Placement,
    new_candidate: CandidatePlacement,
) -> Dict[str, float]:
    old_fp = old_placement.candidate.footprint
    new_fp = new_candidate.footprint
    projected = dict(current)
    projected["carbon_kg"] += new_fp.total_carbon_kg - old_fp.total_carbon_kg
    projected["scarcity_water"] += new_fp.scarcity_characterized_water - old_fp.scarcity_characterized_water
    return projected


def _move_signal_delta(
    old_placement: Placement,
    new_candidate: CandidatePlacement,
    signals: Mapping[RegionSlot, ActionSignal],
) -> Dict[str, float]:
    new_placement = replace(old_placement, candidate=new_candidate)
    old_metrics = placement_signal_metrics(old_placement, signals)
    new_metrics = placement_signal_metrics(new_placement, signals)
    return {key: new_metrics[key] - old_metrics[key] for key in old_metrics}


def repair_schedule_no_harm(
    *,
    method_key: str,
    baseline: ScheduleResult,
    pods: Sequence[CarbonAwarePod],
    flavours: Sequence[EnvironmentalFlavor],
    signals: Mapping[RegionSlot, ActionSignal],
    config: PilotConfig,
    score_mode: str,
) -> ScheduleResult:
    start = time.perf_counter()
    placements = [replace(placement) for placement in baseline.placements]
    leftover_cpu, leftover_ram, leftover_gpu = _init_resources(flavours, config.max_timeslots)
    for placement in placements:
        _apply_candidate_resources(placement.pod, placement.candidate, leftover_cpu, leftover_ram, leftover_gpu)

    baseline_result = ScheduleResult(
        method_key=baseline.method_key,
        placements=baseline.placements,
        unplaced_pods=baseline.unplaced_pods,
        elapsed_seconds=baseline.elapsed_seconds,
    )
    baseline_totals = schedule_totals(baseline_result, signals)
    current = ScheduleResult(method_key=method_key, placements=placements, unplaced_pods=list(baseline.unplaced_pods), elapsed_seconds=0.0)
    max_repairs = config.max_repairs if config.max_repairs is not None else max(1, int(baseline_totals["flexible_pods"]) * 2)
    rejected_moves = 0
    repairs = 0
    timeslots: List[CarbonAwareTimeslot] = build_timeslots(config.max_timeslots)

    # --- Incremental caching (bit-identical with the naive full rescan) ---------
    # The naive loop recomputed find_ranked_candidates for every flexible pod on
    # every outer iteration, even though a single repair only changes the resource
    # state of two flavours (the source and destination of the moved pod). We cache,
    # per pod, the feasible candidates grouped by flavour, plus each candidate's
    # move-invariant deltas/score, and recompute only what an applied move dirties.
    flavours_list = list(flavours)
    flavour_by_id = {flv.id: flv for flv in flavours_list}
    # Pre-sort order used inside _build_feasible_candidates (forecast is static here).
    try:
        ordered_ts = sorted(timeslots, key=lambda t: min(flv.forecast.get(t.id, 200.0) for flv in flavours_list))
    except Exception:
        ordered_ts = list(timeslots)

    margin = config.regret_margin
    robust_buffer = config.robust_buffer
    applied_unc_carbon = 0.0  # cumulative worst-case carbon-forecast exposure of applied moves
    lever_mode = config.lever_mode
    flavour_version: Dict[str, int] = {flv.id: 0 for flv in flavours_list}
    flav_cache: Dict[int, Dict[str, Dict[int, CandidatePlacement]]] = {}
    built_ver: Dict[int, Dict[str, int]] = {}
    ranked_cache: Dict[int, List[CandidatePlacement]] = {}
    delta_dirty: Dict[int, bool] = {}

    def _compute_nh(placement: Placement, old_candidate: CandidatePlacement, candidate: CandidatePlacement) -> None:
        """Attach the move-invariant deltas/score to the candidate (recomputed only
        when the pod's own placement changed or the candidate's footprint was rebuilt)."""
        is_self = (
            candidate.flavour.id == old_candidate.flavour.id
            and candidate.timeslot.id == old_candidate.timeslot.id
        )
        lever_skip = (
            (lever_mode == "temporal" and candidate.flavour.id != old_candidate.flavour.id)
            or (lever_mode == "spatial" and candidate.timeslot.id != old_candidate.timeslot.id)
        )
        d_carbon = candidate.footprint.total_carbon_kg - old_candidate.footprint.total_carbon_kg
        d_scarcity = candidate.footprint.scarcity_characterized_water - old_candidate.footprint.scarcity_characterized_water
        carbon_regret = d_carbon if d_carbon > 0.0 else 0.0
        scarcity_regret = d_scarcity if d_scarcity > 0.0 else 0.0
        signal_delta = _move_signal_delta(placement, candidate, signals)
        carbon_saved = -d_carbon
        water_saved = -d_scarcity
        if score_mode == "search_control":
            score = (
                carbon_saved,
                water_saved,
                -abs(signal_delta["stress_kwh"]),
                -candidate.timeslot.id,
            )
        else:
            score = (
                -signal_delta["weighted_stress_kwh"] + 0.5 * signal_delta["headroom_kwh"],
                -signal_delta["stress_kwh"],
                -signal_delta["drought_scarcity_water"],
                water_saved,
                carbon_saved,
            )
        candidate._nh = (
            is_self,
            lever_skip,
            d_carbon,
            d_scarcity,
            carbon_regret,
            scarcity_regret,
            signal_delta["drought_scarcity_water"],
            score,
            # Operational carbon relocated by the move = the surface exposed to carbon-forecast
            # error (embodied carbon is certain; only CI is forecast).
            candidate.footprint.operational_carbon_kg + old_candidate.footprint.operational_carbon_kg,
        )

    while repairs < max_repairs:
        current_totals = schedule_totals(current, signals)
        best: Optional[Tuple[Tuple[float, ...], int, CandidatePlacement]] = None
        base_carbon = baseline_totals["carbon_kg"]
        base_scarcity = baseline_totals["scarcity_water"]
        cur_carbon = current_totals["carbon_kg"]
        cur_scarcity = current_totals["scarcity_water"]

        for idx, placement in enumerate(current.placements):
            if placement.flexibility_class != "flexible":
                continue

            old_candidate = placement.candidate
            bv = built_ver.get(idx)
            first_build = bv is None
            if first_build:
                dirty_fids = list(flavour_version.keys())
            else:
                dirty_fids = [fid for fid, ver in flavour_version.items() if bv.get(fid) != ver]

            if dirty_fids:
                # Rebuild only the dirty flavours' candidates (pod self released, as
                # the naive path does), then reassemble + re-rank in identical order.
                _apply_candidate_resources(placement.pod, old_candidate, leftover_cpu, leftover_ram, leftover_gpu, release=True)
                cache = flav_cache.setdefault(idx, {})
                for fid in dirty_fids:
                    built = _build_feasible_candidates(
                        pod=placement.pod,
                        flavours=[flavour_by_id[fid]],
                        timeslots=timeslots,
                        leftover_cpu=leftover_cpu,
                        leftover_ram=leftover_ram,
                        max_time_slots=config.max_timeslots,
                        leftover_gpu=leftover_gpu,
                    )
                    cache[fid] = {c.timeslot.id: c for c in built}
                _apply_candidate_resources(placement.pod, old_candidate, leftover_cpu, leftover_ram, leftover_gpu)
                built_ver[idx] = dict(flavour_version)
                assembled: List[CandidatePlacement] = []
                for ts in ordered_ts:
                    tsid = ts.id
                    for flv in flavours_list:
                        c = cache[flv.id].get(tsid)
                        if c is not None:
                            assembled.append(c)
                ranked_cache[idx] = _rank_candidates(
                    assembled, objective_mode="carbon", carbon_weight=1.0, water_metric="scarcity"
                )

            ranked = ranked_cache[idx]
            # Recompute cached deltas: all candidates if the pod's own placement changed
            # (its old_candidate baseline moved); otherwise only freshly-rebuilt ones.
            recompute_all = first_build or delta_dirty.get(idx, True)
            for candidate in ranked:
                if recompute_all or not hasattr(candidate, "_nh"):
                    _compute_nh(placement, old_candidate, candidate)
            delta_dirty[idx] = False

            for candidate in ranked:
                (is_self, lever_skip, d_carbon, d_scarcity, carbon_regret, scarcity_regret,
                 drought_delta, score, unc_carbon) = candidate._nh
                if is_self or lever_skip:
                    continue
                # Carbon guard keeps the cumulative WORST-CASE realized carbon under baseline:
                # nominal current + uncertainty already committed + this move's exposure.
                if (cur_carbon + d_carbon + margin * carbon_regret
                        + applied_unc_carbon + robust_buffer * unc_carbon) > base_carbon + 1e-9:
                    rejected_moves += 1
                    continue
                if cur_scarcity + d_scarcity + margin * scarcity_regret > base_scarcity + 1e-9:
                    rejected_moves += 1
                    continue
                if drought_delta > 1e-9:
                    rejected_moves += 1
                    continue
                if score[0] <= 1e-12:
                    continue
                if best is None or score > best[0]:
                    best = (score, idx, candidate)

        if best is None:
            break

        _, placement_idx, candidate = best
        if robust_buffer:
            applied_unc_carbon += robust_buffer * candidate._nh[8]  # commit this move's carbon-forecast exposure
        old = current.placements[placement_idx]
        f_old = old.candidate.flavour.id
        f_new = candidate.flavour.id
        _apply_candidate_resources(old.pod, old.candidate, leftover_cpu, leftover_ram, leftover_gpu, release=True)
        _apply_candidate_resources(old.pod, candidate, leftover_cpu, leftover_ram, leftover_gpu)
        current.placements[placement_idx] = replace(old, candidate=candidate)
        flavour_version[f_old] += 1
        flavour_version[f_new] += 1
        delta_dirty[placement_idx] = True
        repairs += 1

    current.repairs_applied = repairs
    current.rejected_moves = rejected_moves
    current.reference_method = baseline.method_key
    current.elapsed_seconds = time.perf_counter() - start
    return current


def load_flavours_for_pilot(config: PilotConfig) -> List[EnvironmentalFlavor]:
    previous = os.environ.get("CARBON_AWARE_CONFIG_PATH")
    os.environ["CARBON_AWARE_CONFIG_PATH"] = str(config.config_file)
    try:
        flavours = _load_nodes_from_yaml(str(config.nodes_file))
    finally:
        if previous is None:
            os.environ.pop("CARBON_AWARE_CONFIG_PATH", None)
        else:
            os.environ["CARBON_AWARE_CONFIG_PATH"] = previous

    forecasts = load_carbon_intensity_data(str(config.forecasts_file))
    for flavour in flavours:
        region = (getattr(flavour, "region", "") or "").upper()
        if region in forecasts:
            flavour.forecast = forecasts[region]
    apply_pilot_scenario_to_flavours(flavours, config)
    return flavours


def write_placements_csv(path: Path, result: ScheduleResult, signals: Mapping[RegionSlot, ActionSignal]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method_key",
                "pod_id",
                "node_id",
                "region",
                "start_slot",
                "duration",
                "cpu_request",
                "ram_request",
                "flexibility_class",
                "operational_energy_kwh",
                "carbon_kg",
                "operational_carbon_kg",
                "embodied_carbon_kg",
                "raw_water_l",
                "direct_water_l",
                "indirect_water_l",
                "scarcity_water",
                "stress_kwh",
                "weighted_stress_kwh",
                "headroom_kwh",
                "drought_direct_water_l",
                "drought_scarcity_water",
            ],
        )
        writer.writeheader()
        for placement in result.placements:
            fp = placement.candidate.footprint
            signal_metrics = placement_signal_metrics(placement, signals)
            row = {
                "method_key": result.method_key,
                "pod_id": placement.pod.id,
                "node_id": placement.candidate.flavour.id,
                "region": getattr(placement.candidate.flavour, "region", ""),
                "start_slot": placement.candidate.timeslot.id,
                "duration": placement.pod.duration,
                "cpu_request": placement.pod.cpuRequest,
                "ram_request": placement.pod.ramRequest,
                "flexibility_class": placement.flexibility_class,
                "operational_energy_kwh": fp.operational_energy_kwh,
                "carbon_kg": fp.total_carbon_kg,
                "operational_carbon_kg": fp.operational_carbon_kg,
                "embodied_carbon_kg": fp.embodied_carbon_kg,
                "raw_water_l": fp.total_raw_water_l,
                "direct_water_l": fp.direct_water_l,
                "indirect_water_l": fp.indirect_water_l,
                "scarcity_water": fp.scarcity_characterized_water,
                **signal_metrics,
            }
            writer.writerow(row)


def write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_signal_csv(path: Path, signals: Mapping[RegionSlot, ActionSignal]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ActionSignal.__dataclass_fields__.keys()))
        writer.writeheader()
        for _, signal in sorted(signals.items()):
            writer.writerow(signal.__dict__)


def _forecast_flavours(flavours: Sequence[EnvironmentalFlavor], noise: float, seed: int) -> List[EnvironmentalFlavor]:
    """Decision-time view: perturb each flavour's carbon forecast by multiplicative
    Gaussian noise. Footprints built from these reflect forecast error; the realized
    flavours are kept separate for ex-post verification (RQ3)."""
    rng = random.Random(seed)
    perturbed = copy.deepcopy(list(flavours))
    for flavour in perturbed:
        forecast = getattr(flavour, "forecast", {}) or {}
        flavour.forecast = {slot: max(ci * (1.0 + rng.gauss(0.0, noise)), 1.0) for slot, ci in forecast.items()}
    return perturbed


def _rematerialize_under_realized(result: ScheduleResult, realized_flavours: Sequence[EnvironmentalFlavor], config: PilotConfig) -> None:
    """Replace each placement's (forecast) footprint with the realized footprint at the
    SAME (site, timeslot), so the no-harm certificate is verified on realized signals."""
    timeslots = build_timeslots(config.max_timeslots)
    leftover_cpu, leftover_ram, leftover_gpu = _init_resources(realized_flavours, config.max_timeslots)
    for placement in result.placements:
        ranked = find_ranked_candidates(
            pod=placement.pod,
            flavours=list(realized_flavours),
            timeslots=timeslots,
            leftover_cpu=leftover_cpu,
            leftover_ram=leftover_ram,
            max_time_slots=config.max_timeslots,
            objective_mode="carbon",
            water_metric="scarcity",
            leftover_gpu=leftover_gpu,
        )
        match = next(
            (c for c in ranked
             if c.flavour.id == placement.candidate.flavour.id and c.timeslot.id == placement.candidate.timeslot.id),
            None,
        )
        if match is not None:
            placement.candidate = match


def run_no_harm_flexibility_pilot(config: PilotConfig) -> Dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    realized_flavours = load_flavours_for_pilot(config)
    if not realized_flavours:
        raise RuntimeError(f"No nodes loaded from {config.nodes_file}")
    pods = load_pods(config.workloads_dir, max_pods=config.max_pods)
    if not pods:
        raise RuntimeError(f"No pods loaded from {config.workloads_dir}")

    # RQ3: decide on a (possibly noised) carbon forecast; verify on realized signals.
    realized_signals = build_action_signals(realized_flavours, config)
    use_forecast = bool(config.forecast_noise and config.forecast_noise > 0)
    decision_flavours = (
        _forecast_flavours(realized_flavours, config.forecast_noise, config.forecast_seed)
        if use_forecast else realized_flavours
    )
    decision_signals = build_action_signals(decision_flavours, config) if use_forecast else realized_signals

    packing, _, _ = build_greedy_schedule(method_key="packing", pods=pods, flavours=decision_flavours, config=config)
    carbon, _, _ = build_greedy_schedule(method_key="carbon", pods=pods, flavours=decision_flavours, config=config)
    water, _, _ = build_greedy_schedule(method_key="water_scarcity", pods=pods, flavours=decision_flavours, config=config)
    no_harm_flex = repair_schedule_no_harm(
        method_key="no_harm_flex",
        baseline=packing,
        pods=pods,
        flavours=decision_flavours,
        signals=decision_signals,
        config=config,
        score_mode="flexibility",
    )
    search_control_config = replace(config, max_repairs=no_harm_flex.repairs_applied)
    search_control = repair_schedule_no_harm(
        method_key="no_harm_search_control",
        baseline=packing,
        pods=pods,
        flavours=decision_flavours,
        signals=decision_signals,
        config=search_control_config,
        score_mode="search_control",
    )

    results = [packing, carbon, water, search_control, no_harm_flex]
    if use_forecast:
        # Verify the forecast-chosen schedules on the realized world.
        for result in results:
            _rematerialize_under_realized(result, realized_flavours, config)

    signals = realized_signals
    for result in results:
        write_placements_csv(config.output_dir / f"placements_{result.method_key}.csv", result, signals)
    write_signal_csv(config.output_dir / "actionable_signals.csv", signals)

    summary_rows = [summarize_against_reference(result, packing, signals) for result in results]
    write_summary_csv(config.output_dir / "summary.csv", summary_rows)

    n_sig = max(len(signals), 1)
    signal_base_rates = {
        "region_slots": len(signals),
        "grid_stress_share": sum(1 for s in signals.values() if s.grid_stress) / n_sig,
        "clean_headroom_share": sum(1 for s in signals.values() if s.clean_headroom) / n_sig,
        "drought_share": sum(1 for s in signals.values() if s.drought_guardrail) / n_sig,
    }

    certificate = {
        "reference_method": packing.method_key,
        "scenario": config.scenario,
        "regret_margin": config.regret_margin,
        "operating_envelope": {
            "baseline": packing.method_key,
            "lever_mode": config.lever_mode,
            "flexibility_slack_threshold_hours": config.flexibility_slack_hours,
            "horizon_ceiling_timeslots": config.max_timeslots,
            "stress_quantile": config.stress_quantile,
            "headroom_quantile": config.headroom_quantile,
            "drought_cf_threshold": config.drought_cf_threshold,
            "forecast_noise": config.forecast_noise,
        },
        "signal_base_rates": signal_base_rates,
        "data_strength": {
            "carbon": "local Electricity Maps forecast file used as ex-post attributional account",
            "water": "local WUE/EWIF/AWARE reference tables; heatwave-drought scenario uses high-WUE plus July AWARE factors",
            "grid": (
                "OPSD residual-load-lite table (load minus wind and solar) used when available; "
                "fallback is system-demand proxy for plumbing tests only"
            ),
        },
        "grid_signal_csv": str(grid_signal_path(config).relative_to(config.repo_root)) if grid_signal_path(config) else None,
        "methods": summary_rows,
    }
    (config.output_dir / "no_harm_certificate.json").write_text(json.dumps(certificate, indent=2), encoding="utf-8")

    metadata = {
        "nodes_file": str(config.nodes_file.relative_to(config.repo_root)),
        "workloads_dir": str(config.workloads_dir.relative_to(config.repo_root)),
        "forecasts_file": str(config.forecasts_file.relative_to(config.repo_root)),
        "config_file": str(config.config_file.relative_to(config.repo_root)),
        "max_timeslots": config.max_timeslots,
        "max_pods": config.max_pods,
        "scenario": config.scenario,
        "wue_table": str(scenario_wue_path(config).relative_to(config.repo_root)),
        "grid_signal_table": str(grid_signal_path(config).relative_to(config.repo_root)) if grid_signal_path(config) else None,
        "pod_count": len(pods),
        "node_count": len(realized_flavours),
        "created_at_unix": time.time(),
    }
    (config.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "output_dir": str(config.output_dir),
        "summary_rows": summary_rows,
        "certificate": certificate,
        "signal_base_rates": signal_base_rates,
    }
