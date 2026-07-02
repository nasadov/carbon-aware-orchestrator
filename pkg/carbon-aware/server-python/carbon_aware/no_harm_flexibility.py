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
import math
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
from carbon_aware.footprints import compute_footprint_vector
from carbon_aware.models import CarbonAwarePod, CarbonAwareTimeslot, EnvironmentalFlavor
from carbon_aware.precompute_heuristic import _extract_pods_from_yaml, _load_nodes_from_yaml
from carbon_aware.utils import build_timeslots, is_timeslot_valid, load_carbon_intensity_data


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
    ewif_csv: Optional[Path] = None  # override the EWIF table with a time-aligned in-window one (mirrors wue_csv)
    scenario_month: str = "jul"  # AWARE CF month column (jul default; "aug" for the Aug-2022 strong window)
    forecast_noise: float = 0.0  # RQ3: stdev of multiplicative carbon-forecast error used for DECISIONS
    forecast_seed: int = 0
    # RQ3 robust guard: assumed carbon-forecast-error fraction. Each accepted move adds its
    # forecast-error EXPOSURE (robust_buffer x the OPERATIONAL carbon it relocates) to a
    # cumulative worst-case carbon accumulator that the no-harm guard keeps under baseline,
    # so realized carbon stays <= baseline even when the forecast is wrong. Applied to the
    # CARBON guard only (water is decided on OBSERVED wet-bulb -> no forecast error).
    # Default 0.0 -> identical to the plain forecast guard (bit-identical).
    robust_buffer: float = 0.0
    # D2 -- safety-stock buffer scaling (default "flat" -> BIT-IDENTICAL to the legacy behaviour).
    # The legacy ("flat") buffer charges EVERY accepted move the SAME fraction `robust_buffer` of the
    # operational carbon it relocates and SUMS them, so the reserve grows LINEARLY in the number of
    # moves K. That assumes per-move carbon-forecast errors are perfectly correlated and adversarial.
    # If the per-move errors are instead independent (the realistic case: each move's slot CI is a
    # separate forecast draw), the cumulative error POOLS -- its standard deviation grows like sqrt(K),
    # not K -- so the principled reserve is the safety-stock formula
    #     total_reserve = Phi^{-1}(1-eps) * rho * s * sqrt(K) * (mean operational carbon per move)
    # i.e. the PER-MOVE buffer fraction must FALL like 1/sqrt(K). A flat fraction over-reserves at large
    # K (erasing all grid relief) and under-reserves at small K. Set robust_buffer_mode="sqrtk" to use
    # the pooled formula; the per-move charge then becomes
    #     z(eps) * rho * shape / sqrt(K_est)              (z = Phi^{-1}(1-eps))
    # applied to the SAME operational-carbon exposure the flat path uses. K_est is the decision-time
    # movable surface (number of flexible pods) unless robust_buffer_k_override > 0. rho is the relative
    # CI-forecast half-width measured from data (decide-CI vs realized-CI). This converts the EX-POST
    # certificate into a PROSPECTIVE (decision-time) guarantee at target hold-rate (1-eps): pick eps,
    # read the buffer off the formula. CARBON-ONLY (water is decided on observed wet-bulb -> no error).
    # "flat" (legacy, bit-identical) | "sqrtk" (pooled Gaussian safety-stock) | "dro" (distribution-free).
    # The "dro" mode (Tier-1-D) is the rigorous prospective upgrade: it replaces the untested Gaussian
    # quantile z(eps) with the DISTRIBUTION-FREE one-sided Cantelli factor sqrt((1-eps)/eps) (no
    # Gaussianity, no calibration sample) AND replaces the silent independence (pure sqrt(K)) pooling
    # with an explicit Bertsimas-Sim correlation budget Gamma in [sqrt(K), K] (Gamma=sqrt(K) recovers
    # independence; Gamma=K is the fully-correlated worst case). The per-move charge becomes
    #     cantelli(eps) * rho * shape * Gamma / K  (summed over K -> total reserve cantelli*rho*shape*Gamma*mean)
    # The guarantee it yields is WORST-CASE EX-POST FEASIBILITY (P(realized C > C(B)) <= eps, one-sided)
    # under the stated mean/variance (or support) + correlation-budget assumptions on n=1 -- weaker than
    # a finite-sample probability claim but EARNED without a calibration panel. Default unchanged ("flat").
    robust_buffer_mode: str = "flat"
    robust_buffer_rho: float = 0.0       # relative carbon-forecast half-width rho (measured from data)
    robust_buffer_epsilon: float = 0.05  # target ex-post violation probability eps -> z=Phi^{-1}(1-eps)
    robust_buffer_shape: float = 1.0     # error-shape factor s (1.0 = std-fraction; ~0.577 = uniform)
    robust_buffer_k_override: int = 0    # override K_est (#moves); 0 -> use the flexible-pod count
    # DRO correlation budget Gamma (used only when robust_buffer_mode == "dro").
    #   robust_buffer_gamma_mode == "frac": robust_buffer_gamma is a correlation fraction c in [0,1],
    #       mapped to Gamma = sqrt(K) + c*(K - sqrt(K)); c=0 -> sqrt(K) (independent), c=1 -> K (fully
    #       correlated). Default c=0.0 reproduces the pooled (independence) scaling -- so dro with c=0
    #       differs from sqrtk ONLY by the Cantelli-vs-Gaussian factor.
    #   robust_buffer_gamma_mode == "abs": robust_buffer_gamma is Gamma directly (clamped to [sqrt(K),K]).
    robust_buffer_gamma: float = 0.0
    robust_buffer_gamma_mode: str = "frac"  # "frac" (correlation fraction c) | "abs" (Gamma directly)
    # Symmetric WATER forecast-error buffer (carbon's counterpart). Off by default (0.0) -> the scarcity
    # guard is byte-for-byte the legacy path. It exists because TEMPORAL shifting prices a move's water at
    # a FUTURE slot, so the water axis is NOT forecast-free: direct cooling water depends on the wet-bulb
    # forecast (small -- weather is highly predictable within-day) and indirect/off-site water on the
    # grid-mix forecast (correlated with the carbon error). "flat" mode uses robust_buffer_water as the
    # per-move fraction; "sqrtk" mode uses z(eps)*rho_water*shape/sqrt(K) with its own rho_water.
    robust_buffer_water: float = 0.0
    robust_buffer_water_rho: float = 0.0  # relative water-forecast half-width (direct wet-bulb + indirect mix)
    # WaterWise-style scalarized co-optimizer baseline (method_key="waterwise"): the
    # weight on max-normalized carbon in the per-pod scalar objective; the water weight
    # is (1 - this). Default 0.5 = WaterWise's equal-weight setting. Sweeping this traces
    # the co-optimizer's carbon-water frontier. Affects ONLY the "waterwise" baseline.
    waterwise_carbon_weight: float = 0.5
    # MC1 guarded-baseline controls: seed for the deterministic per-candidate tie-break/order used by
    # the score_mode in {"guarded_random","guarded_carbon","guarded_waterwise"}. These controls share
    # the IDENTICAL no-harm guard with the envelope and differ ONLY in the acceptance RANKING, isolating
    # what the lexicographic combined ranking buys over "any safe move". Unused by every other
    # score_mode (combined/search_control/legacy) -> bit-identical when not selected.
    repair_random_seed: int = 0
    # Independent verification signal set: decisions are made on the observable decision-time CI
    # (forecasts_file, e.g. the residual-load-scaled proxy), but the no-harm certificate is
    # re-evaluated against THIS carbon-intensity set (e.g. real generation-mix CI). Default None ->
    # verify on the decision-time signals (bit-identical). This breaks the decide/verify circularity
    # of accounting carbon with the same residual-load signal the scheduler optimised.
    verify_forecasts_file: Optional[Path] = None
    # Independent verification of the WATER/scarcity side: decisions use the decision-time water config
    # (config_file, e.g. country-level AWARE CFs), but the certificate is re-evaluated against this
    # config (e.g. basin-resolution CFs). Default None -> verify on the decision-time config. Lets us
    # test whether a country-scarcity-decided schedule still does no harm at basin resolution.
    verify_config_file: Optional[Path] = None
    # D1 -- certified atomic consolidation (default OFF, bit-identical when off). The per-move greedy
    # guard rejects any single move onto a fresh (otherwise-empty) clean node-slot because that move
    # pays an idle-once activation that fails the carbon guard MID-consolidation -- even though the
    # COMPLETED consolidation (source nodes emptied -> their idle removed) net-reduces carbon. This is
    # the myopic pointwise-safety-filtering vs set-based-filtering gap (safe-control CBFs; OR LNS /
    # capacitated facility location with idle power = facility opening cost). When enabled, after the
    # per-move loop converges we run an LNS-style destroy-and-recreate post-pass that evaluates a
    # coordinated SET of moves JOINTLY against the no-harm guard using EXACT occupancy-based footprints
    # (idle charged once on the destination, removed from any source the set fully empties), and commits
    # the set atomically iff the completed schedule is Pareto non-degrading (carbon<=baseline,
    # scarcity<=baseline, SLO preserved) AND strictly improves >=1 certified axis. The harmful
    # intermediate state therefore never exists as a guard-checked state, so the certificate is intact
    # by construction. NOVEL: existing consolidators tolerate transient SLA harm (hysteresis / global
    # MILP) -- none check a move-SET against a no-harm certificate.
    consolidation_pass: bool = False
    # Bound the post-pass cost: max distinct (flavour, start-slot) destinations probed (cleanest first)
    # and max LNS rounds. 96 covers the full clean-half of a 4-node x 48-slot horizon (probing only the
    # cleanest third, 64, left certified carbon on the table at n>=16 because the cleanest node's later
    # slots sit beyond the cap); 8 rounds suffices for the schedules tested. Both are tunable for larger
    # fleets; the EXACT occupancy-order eval was narrowed to the moved node so runtime stays ~<=2s.
    consolidation_max_destinations: int = 96
    consolidation_max_rounds: int = 8
    # Per-basin (per-region == per-watershed in this testbed) water non-degradation. Default False ->
    # ONLY the aggregate scarcity guard runs (byte-identical to the legacy path). When True, the repair
    # loop additionally rejects any move that would raise ANY basin's scarcity-characterized water above
    # the baseline B's, i.e. enforces Delta W_b <= 0 for every basin b (strictly tighter than the
    # aggregate sum sum_b Delta W_b <= 0). This subsumes the single-threshold drought guard (a CF>=20
    # special case): it protects every watershed continuously rather than only those above an
    # aggregation-inflated cutoff. The guard only ADDS rejections, so it can never create a false
    # certificate; with the flag off no new branch executes (bit-identical).
    per_basin_scarcity_guard: bool = False
    # --- Ionising-radiation axis: fully opt-in so we can cleanly "go back" ---------------------
    # MASTER SWITCH. Default OFF -> radiation is NOT attached, measured, or reported anywhere;
    # the engine + evaluation are byte-identical to the pre-radiation system (experiments/eval/
    # plotting unchanged). Set True to MEASURE radiation (attach the per-region CF + report deltas)
    # without acting on it. radiation_guard implies enabled.
    radiation_enabled: bool = False
    # GUARD. When True, the repair also enforces operational ionising radiation <= baseline (a
    # fourth non-degradation constraint) and the 4-axis certificate is reported. Implies the signal
    # is attached (i.e. acts as radiation_enabled). Default OFF -> bit-identical.
    radiation_guard: bool = False
    radiation_csv: Optional[Path] = None  # override the per-region radiation CF table
    # --- Electricity-cost axis (fifth axis): fully opt-in, mirrors the radiation switches -------
    # MASTER SWITCH. Default OFF -> cost is NOT attached, measured, or reported anywhere (bit-
    # identical off path). cost_enabled=True MEASURES cost (attach day-ahead prices + report
    # deltas) without acting on it; cost_guard=True additionally enforces cost <= baseline in the
    # repair (implies enabled). price_csv points at a (region, slot) day-ahead price table
    # (price_eur_per_kwh) -- REQUIRED when the axis is on (no silent default).
    cost_enabled: bool = False
    cost_guard: bool = False
    price_csv: Optional[Path] = None
    # Exact commit-time guard mode. "node_local" re-prices only the move's source and destination
    # nodes per commit (provably equivalent: idle-once accounting never crosses node boundaries, so
    # a move changes footprints only on those two nodes) -- O(occupants of 2 nodes) instead of the
    # whole schedule. "full" re-prices the entire placement set (the original path, kept for
    # swap-and-diff validation). Both modes yield identical accepted-move decisions.
    exact_guard_mode: str = "node_local"


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


def _read_radiation_region_csv(path: Path, scenario: str = "base") -> Dict[str, float]:
    """Per-region ionising-radiation CF (kBq U-235 eq / kWh). Best-effort: missing file -> {}."""
    if not path.exists():
        return {}
    col = f"ionising_radiation_kbq_u235eq_per_kwh_{scenario}"
    out: Dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            region = (row.get("region") or "").strip().upper()
            if not region:
                continue
            out[region] = _as_float(row.get(col), 0.0)
    return out


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


def _read_basin_cf_rows(path: Path) -> Dict[str, Dict[str, str]]:
    """AWARE CF rows keyed by REGION (the native-watershed file), for on-site/direct water."""
    if not path.exists():
        return {}
    rows: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            region = (row.get("region") or "").strip().upper()
            if region:
                rows[region] = row
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
    country_aware_rows = _read_aware_rows(_water_data_path(config.repo_root, "aware20_country_nonagri_factors.csv"))
    basin_aware_rows = _read_basin_cf_rows(_water_data_path(config.repo_root, "aware20_basin_nonagri_factors.csv"))
    month_col = f"{config.scenario_month}_cf"
    # Indirect (power-plant) water: EWIF per kWh of grid electricity, from a flow-traced
    # generation mix x literature water-consumption factors (Macknick 2012; Spang 2014).
    # Redirectable to a time-aligned in-window EWIF table (mirrors wue_csv); falls back to the bundled file.
    ewif_rows = _read_region_slot_csv(config.ewif_csv or _water_data_path(config.repo_root, "ewif_region_slot.csv"))
    # Per-region ionising-radiation CF (kBq U-235 eq / kWh), region-resolved (mix-average).
    # Attached ONLY when the radiation axis is switched on (master switch / guard). When off,
    # nothing is attached -> operational_radiation_kbq stays 0 -> the system is the pre-radiation
    # engine. Best-effort: if the table is absent the factor stays 0.
    radiation_on = bool(getattr(config, "radiation_enabled", False) or getattr(config, "radiation_guard", False))
    radiation_by_region = _read_radiation_region_csv(
        config.radiation_csv or (config.repo_root / "pkg" / "carbon-aware" / "data" / "radiation" / "radiation_region.csv")
    ) if radiation_on else {}
    # Electricity-cost axis (fifth axis): attach hourly day-ahead prices ONLY when switched on.
    # No default table -- a silent 0-price would fake a free grid -- so the axis fails loudly.
    cost_on = bool(getattr(config, "cost_enabled", False) or getattr(config, "cost_guard", False))
    price_rows: Dict[RegionSlot, Dict[str, str]] = {}
    if cost_on:
        if not config.price_csv or not Path(config.price_csv).exists():
            raise SystemExit("cost axis is on (cost_enabled/cost_guard) but price_csv is missing")
        price_rows = _read_region_slot_csv(Path(config.price_csv))

    for flavour in flavours:
        region = (getattr(flavour, "region", "") or "").upper()
        if radiation_on:
            rad_cf = radiation_by_region.get(region)
            if rad_cf is not None:
                flavour.radiation_intensity = rad_cf
                flavour.radiation_by_slot = {slot: rad_cf for slot in range(config.max_timeslots)}
        if cost_on:
            price_by_slot = {
                slot: _as_float(row.get("price_eur_per_kwh"))
                for (row_region, slot), row in price_rows.items()
                if row_region == region
            }
            if price_by_slot:
                flavour.price_by_slot = price_by_slot
                flavour.electricity_price_eur_kwh = sum(price_by_slot.values()) / len(price_by_slot)
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
            country_row = country_aware_rows.get(country, {})
            basin_row = basin_aware_rows.get(region, {})
            prev_cf = getattr(flavour, "water_scarcity_direct_cf", 1.0)
            country_cf = _as_float(country_row.get(month_col), prev_cf)
            # Direct (on-site cooling) water is consumed AT the data center -> charge at its own
            # watershed (basin) CF. Indirect (power-plant) water is consumed where the electricity is
            # GENERATED -- distributed across the country's fleet, NOT the DC's local basin -- so absent
            # plant-level/flow-traced origin we charge it at the generation COUNTRY's CF (a coarse
            # domestic-generation proxy, but strictly more faithful than the DC-basin CF). [water-rigor]
            direct_cf = _as_float(basin_row.get(month_col), country_cf)  # basin for on-site; country fallback
            indirect_cf = country_cf
            flavour.water_scarcity_direct_cf = direct_cf
            flavour.water_scarcity_indirect_cf = indirect_cf
            flavour.water_scarcity_direct_cf_by_slot = {slot: direct_cf for slot in range(config.max_timeslots)}
            flavour.water_scarcity_indirect_cf_by_slot = {slot: indirect_cf for slot in range(config.max_timeslots)}


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


def _select_waterwise_candidate(
    ranked: Sequence[CandidatePlacement], carbon_weight: float
) -> CandidatePlacement:
    """WaterWise-style scalarized choice (Jiang et al. 2025): minimise a weighted
    sum of per-pod max-normalised carbon and scarcity-weighted water over the pod's
    feasible placements, with equal default weights. Unlike the single-objective
    carbon/water-greedy baselines, this *co-optimises* both axes -- and, like any
    weighted sum, it will accept a carbon increase when water falls enough (and vice
    versa), so it tolerates residual harm on an axis by construction. It is therefore
    the fair state-of-the-art comparator for the no-harm certificate, not a strawman.
    """
    carbon_weight = min(max(float(carbon_weight), 0.0), 1.0)
    water_weight = 1.0 - carbon_weight
    carbons = [c.footprint.total_carbon_g for c in ranked]
    waters = [_candidate_water_value(c.footprint, "scarcity") for c in ranked]
    # Min-max normalisation per pod (matching the codebase's weighted-sum heuristic,
    # `_normalize_to_unit_interval`): maps each axis's best feasible candidate to 0 and
    # worst to 1, so both axes get equal leverage regardless of their absolute ranges.
    # (Dividing by the max alone lets a compressed water range wash out and collapses
    # the equal-weight choice onto carbon-greedy.)
    c_min, c_max = min(carbons), max(carbons)
    w_min, w_max = min(waters), max(waters)

    def _score(candidate: CandidatePlacement) -> Tuple[float, ...]:
        water_value = _candidate_water_value(candidate.footprint, "scarcity")
        c_norm = (candidate.footprint.total_carbon_g - c_min) / (c_max - c_min) if c_max > c_min + 1e-12 else 0.0
        w_norm = (water_value - w_min) / (w_max - w_min) if w_max > w_min + 1e-12 else 0.0
        scalar = carbon_weight * c_norm + water_weight * w_norm
        # Deterministic tie-breaks, mirroring the other methods' secondary ordering.
        return (scalar, candidate.footprint.total_carbon_g, water_value,
                candidate.pack_score, candidate.timeslot.id)

    return min(ranked, key=_score)


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

        if method_key == "waterwise" or method_key.startswith("waterwise@"):
            # `waterwise@<w>` keys (the published-baselines sweep) carry the weight in the suffix.
            # The old exact-match dispatch let them FALL THROUGH to the carbon sort below, silently
            # running carbon-greedy under a WaterWise label (the published-baselines label bug,
            # fixed 2026-07-01). Suffix weight, when present, takes precedence over the config field.
            weight = config.waterwise_carbon_weight
            if "@" in method_key:
                try:
                    weight = float(method_key.split("@", 1)[1])
                except ValueError:
                    pass  # malformed suffix -> config weight
            candidate = _select_waterwise_candidate(ranked, weight)
        else:
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
        "operational_radiation_kbq": 0.0,
        "operational_cost_eur": 0.0,
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
        totals["operational_radiation_kbq"] += getattr(fp, "operational_radiation_kbq", 0.0)
        totals["operational_cost_eur"] += getattr(fp, "operational_cost_eur", 0.0)
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
    radiation_delta = totals.get("operational_radiation_kbq", 0.0) - reference_totals.get("operational_radiation_kbq", 0.0)
    cost_delta = totals.get("operational_cost_eur", 0.0) - reference_totals.get("operational_cost_eur", 0.0)
    stress_delta = totals["stress_kwh"] - reference_totals["stress_kwh"]
    weighted_stress_delta = totals["weighted_stress_kwh"] - reference_totals["weighted_stress_kwh"]
    headroom_delta = totals["headroom_kwh"] - reference_totals["headroom_kwh"]
    no_harm = (
        totals["placed_pods"] >= reference_totals["placed_pods"]
        and totals["unplaced_pods"] <= reference_totals["unplaced_pods"]
        and carbon_delta <= tolerance
        and scarcity_delta <= tolerance
    )

    radiation_nonincrease = bool(radiation_delta <= tolerance)
    cost_nonincrease = bool(cost_delta <= tolerance)
    summary = {
        "method_key": result.method_key,
        "reference_method": reference.method_key,
        "no_harm_certificate": bool(no_harm),
        "carbon_nonincrease": bool(carbon_delta <= tolerance),
        "scarcity_nonincrease": bool(scarcity_delta <= tolerance),
        "slo_non_decrease": bool(
            totals["placed_pods"] >= reference_totals["placed_pods"]
            and totals["unplaced_pods"] <= reference_totals["unplaced_pods"]
        ),
        # SLO axis = deadline-feasible admission. Infeasible (past-deadline) placements are never
        # generated (is_timeslot_valid enforces finish<=deadline), so a pod is either admitted
        # on time or unplaced -- there is no "late" state. Deadline attainment is thus the
        # admitted share; late_pods is 0 by construction (reported for transparency).
        "deadline_attainment_pct": (
            100.0 * totals["placed_pods"] / max(totals["placed_pods"] + totals["unplaced_pods"], 1.0)
        ),
        "late_pods": 0,
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
        # Headline grid-relief metric: the *continuous* residual-load-weighted relief the objective
        # actually optimizes. The binary stress_kwh_avoided above is a lossy projection (only counts
        # slots that cross the binary stress quantile) and undercounts achieved relief by ~50%.
        "weighted_stress_kwh_delta": weighted_stress_delta,
        "weighted_stress_kwh_avoided": max(-weighted_stress_delta, 0.0),
        "weighted_stress_kwh_avoided_pct": _percent_delta(
            reference_totals["weighted_stress_kwh"], totals["weighted_stress_kwh"]
        ),
        "headroom_kwh": totals["headroom_kwh"],
        "headroom_kwh_gain": max(headroom_delta, 0.0),
        "grid_stress_hours_covered": stress_hours_covered,
        "drought_direct_water_l": totals["drought_direct_water_l"],
        "drought_scarcity_water": totals["drought_scarcity_water"],
    }
    # Ionising-radiation reporting is included ONLY when the radiation axis is actually in play
    # (signal attached -> nonzero operational radiation on either schedule). When radiation is OFF
    # (the radiation_enabled/radiation_guard master switches), these keys are ABSENT and the
    # evaluation output is byte-for-byte the pre-radiation engine -> a clean "go back".
    if (totals.get("operational_radiation_kbq", 0.0) > 0.0
            or reference_totals.get("operational_radiation_kbq", 0.0) > 0.0):
        summary["no_harm_certificate_with_radiation"] = bool(no_harm and radiation_nonincrease)
        summary["radiation_nonincrease"] = radiation_nonincrease
        summary["radiation_delta_kbq"] = radiation_delta
        summary["radiation_delta_pct"] = _percent_delta(
            totals.get("operational_radiation_kbq", 0.0), reference_totals.get("operational_radiation_kbq", 0.0))
        summary["operational_radiation_kbq"] = totals.get("operational_radiation_kbq", 0.0)
    # Electricity-cost reporting, exactly parallel: keys ABSENT when the cost axis is off, so the
    # off path stays byte-for-byte identical to the pre-cost engine.
    if (totals.get("operational_cost_eur", 0.0) > 0.0
            or reference_totals.get("operational_cost_eur", 0.0) > 0.0):
        summary["no_harm_certificate_with_cost"] = bool(no_harm and cost_nonincrease)
        summary["cost_nonincrease"] = cost_nonincrease
        summary["cost_delta_eur"] = cost_delta
        summary["cost_delta_pct"] = _percent_delta(
            totals.get("operational_cost_eur", 0.0), reference_totals.get("operational_cost_eur", 0.0))
        summary["operational_cost_eur"] = totals.get("operational_cost_eur", 0.0)
    return summary


def _percent_delta(new_value: float, old_value: float) -> float:
    if abs(old_value) <= 1e-12:
        return 0.0
    return (new_value - old_value) / old_value * 100.0


def _z_score(epsilon: float) -> float:
    """One-sided normal quantile z = Phi^{-1}(1 - eps) for the safety-stock buffer.
    eps is the target ex-post violation probability (e.g. 0.05 -> z~1.645 for 95% hold)."""
    from statistics import NormalDist
    eps = min(max(float(epsilon), 1e-9), 0.5)
    return NormalDist().inv_cdf(1.0 - eps)


def _cantelli_factor(epsilon: float) -> float:
    """Distribution-free one-sided Cantelli (one-sided Chebyshev) factor k = sqrt((1-eps)/eps).

    For ANY random variable X with mean mu and standard deviation sigma, Cantelli's inequality gives
        P(X - mu >= k*sigma) <= 1/(1 + k^2).
    Setting the right-hand side to eps and solving yields k = sqrt((1-eps)/eps). Sizing the carbon
    reserve as k*sigma_C therefore bounds the realized one-sided overshoot probability by eps WITHOUT
    any distributional assumption (no Gaussianity, no calibration sample) -- the honest distribution-free
    replacement for the Gaussian quantile z = Phi^{-1}(1-eps). It is conservative: at eps=0.05 the
    Cantelli factor is ~4.36 vs the Gaussian z~1.645 (~2.65x), the price of assumption-freedom.
    """
    eps = min(max(float(epsilon), 1e-9), 0.5)
    return math.sqrt((1.0 - eps) / eps)


def _budget_gamma(*, gamma: float, gamma_mode: str, k_moves: int) -> float:
    """Bertsimas-Sim uncertainty budget Gamma in [sqrt(K), K] for the per-move error correlation.

    The pooled (sqrtk) reserve assumes per-move CI-forecast errors are INDEPENDENT, so their
    cumulative standard deviation grows like sqrt(K). That assumption is false: CI forecast errors are
    AUTOCORRELATED across contiguous slots/regions (a biased-low proxy is biased low for adjacent
    hours), so pure sqrt(K) UNDER-reserves. Bertsimas-Sim ("price of robustness") interpolates between
    the two extremes with a single budget knob Gamma:
        Gamma = sqrt(K)  -> independent errors (recovers the pooled sqrtk reserve);
        Gamma = K        -> fully (adversarially) correlated worst case (the box / flat reserve).
    A correlation fraction c in [0,1] maps to Gamma = sqrt(K) + c*(K - sqrt(K)); c=0 -> sqrt(K), c=1 -> K.
    Returns Gamma clamped to [sqrt(K), K].
    """
    k = max(int(k_moves), 1)
    sk = math.sqrt(k)
    if gamma_mode == "frac":
        c = min(max(float(gamma), 0.0), 1.0)
        g = sk + c * (float(k) - sk)
    else:  # "abs": Gamma given directly
        g = float(gamma) if gamma > 0 else sk
    return min(max(g, sk), float(k))


def sqrtk_per_move_buffer(*, rho: float, epsilon: float, shape: float, k_moves: int) -> float:
    """Pooled safety-stock PER-MOVE buffer fraction: z(eps) * rho * shape / sqrt(K).

    This is the per-move multiplier the carbon guard applies to each accepted move's operational-carbon
    exposure. Summed over K independent moves it yields the total reserve
        z(eps) * rho * shape * sqrt(K) * (mean operational carbon per move),
    so the reserve grows like sqrt(K) (error pooling) rather than the flat buffer's linear-in-K growth.
    """
    k = max(int(k_moves), 1)
    return _z_score(epsilon) * float(rho) * float(shape) / math.sqrt(k)


def dro_per_move_buffer(
    *, rho: float, epsilon: float, shape: float, k_moves: int, gamma: float, gamma_mode: str
) -> float:
    """Distribution-free DRO PER-MOVE buffer fraction: cantelli(eps) * rho * shape * Gamma / K.

    Two principled changes vs sqrtk_per_move_buffer, both flag-gated (default OFF):
      1. The Gaussian quantile z = Phi^{-1}(1-eps) is replaced by the distribution-free one-sided
         Cantelli factor sqrt((1-eps)/eps), which bounds the realized overshoot probability by eps for
         ANY error distribution (no Gaussianity, no calibration sample).
      2. The silent independence (pure sqrt(K)) pooling is replaced by an explicit Bertsimas-Sim
         budget Gamma in [sqrt(K), K]. Summed over K moves the per-move charge cantelli*rho*shape*Gamma/K
         yields the TOTAL reserve
             cantelli(eps) * rho * shape * Gamma * (mean operational carbon per move),
         so the reserve scales as Gamma: Gamma=sqrt(K) recovers the independent pooled reserve, Gamma=K
         is the fully-correlated (adversarial) box reserve. The dependence assumption is thus STATED
         (via Gamma) rather than hidden.
    """
    k = max(int(k_moves), 1)
    g = _budget_gamma(gamma=gamma, gamma_mode=gamma_mode, k_moves=k)
    return _cantelli_factor(epsilon) * float(rho) * float(shape) * g / float(k)


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


def _exact_footprints_in_occupancy_order(
    placements: Sequence[Placement],
    flavours: Sequence[EnvironmentalFlavor],
    config: PilotConfig,
) -> Dict[int, CandidatePlacement]:
    """Recompute every placement's EXACT engine footprint against the CURRENT joint occupancy of the
    given placement set, in occupancy order (idle-once charged to the first pod that activates each
    node-slot). Returns id(placement) -> rematerialized CandidatePlacement at the SAME (site, slot).

    This is the same idle-correct accounting `_rematerialize_under_realized` performs, but it operates
    on a HYPOTHETICAL placement list without mutating it -- so the consolidation pass can score a
    candidate move-set's true engine carbon/scarcity before deciding whether to commit it.
    """
    timeslots = build_timeslots(config.max_timeslots)
    flavour_by_id = {flv.id: flv for flv in flavours}
    leftover_cpu, leftover_ram, leftover_gpu = _init_resources(flavours, config.max_timeslots)
    out: Dict[int, CandidatePlacement] = {}
    # Apply pods in a stable order (start slot, then node) so idle attribution is deterministic and
    # matches a left-to-right fill; the TOTAL carbon/scarcity is order-invariant (idle is charged
    # exactly once per active node-slot regardless of which co-tenant carries it).
    ordered = sorted(
        placements,
        key=lambda pl: (pl.candidate.timeslot.id, pl.candidate.flavour.id, pl.pod.id),
    )
    for placement in ordered:
        # Build candidates for ONLY this placement's own node (we know its (flavour, slot)); this
        # yields the same footprint the full-fleet ranker would at that location but avoids scoring
        # every other node-slot, which dominates the post-pass cost.
        dest_flv = flavour_by_id.get(placement.candidate.flavour.id)
        built = _build_feasible_candidates(
            pod=placement.pod,
            flavours=[dest_flv] if dest_flv is not None else list(flavours),
            timeslots=timeslots,
            leftover_cpu=leftover_cpu,
            leftover_ram=leftover_ram,
            max_time_slots=config.max_timeslots,
            leftover_gpu=leftover_gpu,
        )
        match = next(
            (c for c in built
             if c.flavour.id == placement.candidate.flavour.id
             and c.timeslot.id == placement.candidate.timeslot.id),
            None,
        )
        if match is None:
            # Infeasible under the joint occupancy (capacity conflict) -> signal an invalid set.
            return {}
        out[id(placement)] = match
        _apply_candidate_resources(placement.pod, match, leftover_cpu, leftover_ram, leftover_gpu)
    return out


def _exact_carbon_scarcity(
    placements: Sequence[Placement],
    flavours: Sequence[EnvironmentalFlavor],
    config: PilotConfig,
) -> Optional[Tuple[float, float, Dict[int, CandidatePlacement]]]:
    """Exact engine (carbon_kg, scarcity_water) for a placement set, plus the rematerialized
    footprints (idle-once correct). Returns None if the set is infeasible under joint occupancy."""
    remat = _exact_footprints_in_occupancy_order(placements, flavours, config)
    if not remat and placements:
        return None
    carbon = sum(c.footprint.total_carbon_kg for c in remat.values())
    scarcity = sum(c.footprint.scarcity_characterized_water for c in remat.values())
    return carbon, scarcity, remat


def _exact_node_totals(
    node_placements: Sequence["Placement"],
    flavour: EnvironmentalFlavor,
    config: PilotConfig,
) -> Optional[Tuple[float, float, float, float]]:
    """Exact idle-once totals (carbon_kg, scarcity, radiation_kbq, cost_eur) for ONE node's
    placements, replayed in the same occupancy order (timeslot, pod) the global exact evaluator
    uses. Footprints depend only on the node's OWN per-slot occupancy -- idle-once attribution
    never crosses node boundaries -- so the per-node replay equals the global evaluator restricted
    to this node; summing node totals over the fleet reproduces _exact_carbon_scarcity exactly.
    Returns None if the occupants would exceed the node's CPU capacity in any slot (infeasible)."""
    ordered = sorted(node_placements, key=lambda pl: (pl.candidate.timeslot.id, pl.pod.id))
    used: Dict[int, float] = {}
    total_cpu = float(getattr(flavour, "totalCpu", 0.0) or 0.0)
    c = w = r = cost = 0.0
    for pl in ordered:
        start = pl.candidate.timeslot.id
        dur = int(pl.pod.duration)
        used_before = {start + o: used.get(start + o, 0.0) for o in range(dur)}
        fp = compute_footprint_vector(
            flavour=flavour,
            start_slot=start,
            pod=pl.pod,
            used_cpu_before_by_slot=used_before,
            embodied_allocation_mode="proportional",
            operational_only=False,
            use_pod_power_only=False,
        )
        for o in range(dur):
            s = start + o
            used[s] = used.get(s, 0.0) + pl.pod.cpuRequest
            if total_cpu and used[s] > total_cpu + 1e-9:
                return None
        c += fp.total_carbon_kg
        w += fp.scarcity_characterized_water
        r += fp.operational_radiation_kbq
        cost += fp.operational_cost_eur
    return c, w, r, cost


def _consolidation_pass(
    *,
    current: ScheduleResult,
    flavours: Sequence[EnvironmentalFlavor],
    signals: Mapping[RegionSlot, ActionSignal],
    config: PilotConfig,
    base_carbon: float,
    base_scarcity: float,
) -> int:
    """Certified atomic consolidation (D1). LNS destroy-and-recreate / capacitated-facility-location
    move evaluated against the no-harm certificate as a SET.

    Mechanism. For each clean, underutilized destination node-slot (cleanest first), greedily assemble
    a SET of flexible pods -- drawn from dirtier current locations -- that fit on the destination, then
    evaluate the COMPLETED schedule's EXACT engine carbon/scarcity (idle charged once on the destination,
    removed from any source the set fully empties). Commit the set atomically iff the completed schedule
    is Pareto non-degrading vs the packing baseline on carbon AND scarcity, keeps every pod placed
    (SLO preserved), and strictly improves carbon (the consolidation payoff). The harmful intermediate
    (a transient idle activation before the source is emptied) is never a guard-checked state, so the
    ex-post certificate holds by construction. Mutates `current.placements` in place; returns #moves.

    Returns 0 (no mutation) unless the consolidation strictly improves carbon under the certificate.
    """
    flavours_list = list(flavours)
    flavour_by_id = {flv.id: flv for flv in flavours_list}
    timeslots = build_timeslots(config.max_timeslots)
    lever_mode = config.lever_mode

    total_moves = 0
    for _round in range(max(1, config.consolidation_max_rounds)):
        # Exact current state (idle-correct) -- the incumbent we must not degrade.
        cur = _exact_carbon_scarcity(current.placements, flavours_list, config)
        if cur is None:
            break
        cur_carbon, cur_scarcity, cur_remat = cur

        # Flexible pods are the only movable surface; index them by current location.
        flex_idx = [
            i for i, pl in enumerate(current.placements)
            if pl.flexibility_class == "flexible"
        ]
        if not flex_idx:
            break

        # Candidate destinations = (flavour, start-slot) over the horizon, cleanest CI first. A pod's
        # carbon is dominated by the destination's slot CI; consolidating onto the lowest-CI active or
        # fresh node-slot is the all-or-nothing payoff the per-move guard cannot reach.
        def _slot_ci(flv: EnvironmentalFlavor, slot: int) -> float:
            return _as_float((getattr(flv, "forecast", {}) or {}).get(slot), 200.0)

        dests: List[Tuple[float, str, int]] = []
        for flv in flavours_list:
            for slot in range(config.max_timeslots):
                dests.append((_slot_ci(flv, slot), flv.id, slot))
        dests.sort(key=lambda d: (d[0], d[1], d[2]))
        dests = dests[: max(1, config.consolidation_max_destinations)]

        committed_this_round = False
        for _ci, dest_fid, dest_slot in dests:
            dest_flv = flavour_by_id[dest_fid]

            # Pods to (potentially) pull onto this destination: flexible pods NOT already there whose
            # deadline window admits dest_slot and that respect the lever mode. Sort by how dirty their
            # current slot is (dirtiest first) -- moving those yields the biggest certified carbon cut.
            movable: List[Tuple[float, int]] = []
            for i in flex_idx:
                pl = current.placements[i]
                cand = pl.candidate
                if cand.flavour.id == dest_fid and cand.timeslot.id == dest_slot:
                    continue
                if lever_mode == "temporal" and dest_fid != cand.flavour.id:
                    continue
                if lever_mode == "spatial" and dest_slot != cand.timeslot.id:
                    continue
                cur_ci = _slot_ci(cand.flavour, cand.timeslot.id)
                movable.append((cur_ci, i))
            movable.sort(key=lambda m: (-m[0], m[1]))

            if not movable:
                continue

            # Greedily grow the move-set; re-feasibility (capacity) is enforced exactly by the
            # occupancy-order rematerialization, so we only need a cheap pre-filter here.
            trial_placements = [replace(pl) for pl in current.placements]
            chosen: List[int] = []
            leftover_cpu, leftover_ram, leftover_gpu = _init_resources(flavours_list, config.max_timeslots)
            # Seed occupancy with everything NOT being considered for the move (the firm + untouched).
            # We rebuild occupancy from the trial set after each tentative add to test the destination
            # fits, but capacity is ultimately validated by _exact_carbon_scarcity below.
            for cur_ci, i in movable:
                pod = current.placements[i].pod
                dur = max(int(pod.duration), 1)
                if dest_slot + dur > config.max_timeslots:
                    continue
                ts = next((t for t in timeslots if t.id == dest_slot), None)
                if ts is None or not is_timeslot_valid(ts, pod):
                    continue
                chosen.append(i)

            if not chosen:
                continue

            # Build a candidate footprint for each chosen pod at the destination, applied jointly.
            applied: List[int] = []
            for i in chosen:
                pod = current.placements[i].pod
                built = _build_feasible_candidates(
                    pod=pod,
                    flavours=[dest_flv],
                    timeslots=timeslots,
                    leftover_cpu=leftover_cpu,
                    leftover_ram=leftover_ram,
                    max_time_slots=config.max_timeslots,
                    leftover_gpu=leftover_gpu,
                )
                new_cand = next((c for c in built if c.timeslot.id == dest_slot), None)
                if new_cand is None:
                    continue  # destination capacity exhausted for this pod; skip it, keep the set
                _apply_candidate_resources(pod, new_cand, leftover_cpu, leftover_ram, leftover_gpu)
                trial_placements[i] = replace(current.placements[i], candidate=new_cand)
                applied.append(i)

            if not applied:
                continue

            scored = _exact_carbon_scarcity(trial_placements, flavours_list, config)
            if scored is None:
                continue
            new_carbon, new_scarcity, new_remat = scored

            # No-harm certificate vs the packing baseline AND strict carbon improvement vs incumbent.
            improves_carbon = new_carbon < cur_carbon - 1e-12
            no_harm = (
                new_carbon <= base_carbon + 1e-9
                and new_scarcity <= base_scarcity + 1e-9
                and len(trial_placements) == len(current.placements)
            )
            if improves_carbon and no_harm:
                # Commit atomically: adopt the rematerialized (idle-correct) footprints for the WHOLE
                # schedule so subsequent rounds and the caller's totals stay occupancy-consistent.
                for pl in trial_placements:
                    rc = new_remat.get(id(pl))
                    if rc is not None:
                        pl.candidate = rc
                current.placements = trial_placements
                total_moves += len(applied)
                committed_this_round = True
                break  # re-evaluate the incumbent before probing more destinations

        if not committed_this_round:
            break

    return total_moves


def repair_schedule_no_harm(
    *,
    method_key: str,
    baseline: ScheduleResult,
    pods: Sequence[CarbonAwarePod],
    flavours: Sequence[EnvironmentalFlavor],
    signals: Mapping[RegionSlot, ActionSignal],
    config: PilotConfig,
    score_mode: str,
    start_schedule: Optional[ScheduleResult] = None,
) -> ScheduleResult:
    start = time.perf_counter()
    # The no-harm REFERENCE is always `baseline` (B): the guard checks carbon/scarcity <= B's. The
    # repair may START from a different schedule (`start_schedule`, e.g. phase-1 of the two-phase
    # combined_v2 envelope), so phase 2 continues from phase-1's placements while still guarding vs B.
    # Default (start_schedule=None) starts from B -> bit-identical to the single-pass behaviour.
    start_from = start_schedule if start_schedule is not None else baseline
    placements = [replace(placement) for placement in start_from.placements]
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
    current = ScheduleResult(method_key=method_key, placements=placements, unplaced_pods=list(start_from.unplaced_pods), elapsed_seconds=0.0)
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
    # Effective per-move carbon-forecast buffer fraction. In the default "flat" mode this is EXACTLY
    # config.robust_buffer (bit-identical to the legacy guard). In "sqrtk" mode it is the pooled
    # safety-stock per-move fraction z(eps)*rho*shape/sqrt(K_est), where K_est = the decision-time
    # movable surface (#flexible pods) unless overridden. K_est is computed ONCE before the loop, so
    # the per-move charge is constant across the loop (the guard stays monotone) while the TOTAL
    # reserve scales as sqrt(K). When robust_buffer_mode != "sqrtk" the sqrtk branch is never taken,
    # so the off-path behaviour is byte-for-byte the legacy path.
    if config.robust_buffer_mode == "sqrtk":
        k_est = (config.robust_buffer_k_override
                 if config.robust_buffer_k_override > 0
                 else max(1, int(baseline_totals["flexible_pods"])))
        robust_buffer = sqrtk_per_move_buffer(
            rho=config.robust_buffer_rho,
            epsilon=config.robust_buffer_epsilon,
            shape=config.robust_buffer_shape,
            k_moves=k_est,
        )
        # Symmetric water reserve: same eps/K/shape, its own (smaller) rho_water.
        robust_buffer_water = sqrtk_per_move_buffer(
            rho=config.robust_buffer_water_rho,
            epsilon=config.robust_buffer_epsilon,
            shape=config.robust_buffer_shape,
            k_moves=k_est,
        )
    elif config.robust_buffer_mode == "dro":
        # Distribution-free DRO reserve (Tier-1-D): Cantelli factor sqrt((1-eps)/eps) in place of the
        # Gaussian z(eps), and a Bertsimas-Sim correlation budget Gamma in [sqrt(K), K] in place of the
        # silent independence assumption. Same per-move operational-carbon exposure surface as flat/sqrtk.
        k_est = (config.robust_buffer_k_override
                 if config.robust_buffer_k_override > 0
                 else max(1, int(baseline_totals["flexible_pods"])))
        robust_buffer = dro_per_move_buffer(
            rho=config.robust_buffer_rho,
            epsilon=config.robust_buffer_epsilon,
            shape=config.robust_buffer_shape,
            k_moves=k_est,
            gamma=config.robust_buffer_gamma,
            gamma_mode=config.robust_buffer_gamma_mode,
        )
        robust_buffer_water = dro_per_move_buffer(
            rho=config.robust_buffer_water_rho,
            epsilon=config.robust_buffer_epsilon,
            shape=config.robust_buffer_shape,
            k_moves=k_est,
            gamma=config.robust_buffer_gamma,
            gamma_mode=config.robust_buffer_gamma_mode,
        )
    else:
        robust_buffer = config.robust_buffer
        robust_buffer_water = config.robust_buffer_water
    applied_unc_carbon = 0.0  # cumulative worst-case carbon-forecast exposure of applied moves
    applied_unc_scarcity = 0.0  # cumulative worst-case water-forecast exposure of applied moves
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
        elif score_mode == "combined":
            # Combined objective: pursue grid-stress relief AND certified carbon/water co-benefits,
            # all still inside the no-harm guard. We rank by the NUMBER of certified axes a move
            # strictly improves (Pareto progress; win-win-win first) — deliberately NOT a weighted
            # sum (that is the bang-bang scalarization we critique in WaterWise) — then by stress
            # relief, then the water and carbon co-benefits. Because score[0] is this count, the
            # existing `score[0] <= 1e-12` gate keeps any move that improves >=1 certified axis,
            # instead of discarding safe carbon/water-cutting moves that don't relieve stress.
            stress_relief = -signal_delta["weighted_stress_kwh"] + 0.5 * signal_delta["headroom_kwh"]
            eps = 1e-9
            n_improved = int(stress_relief > eps) + int(carbon_saved > eps) + int(water_saved > eps)
            score = (
                n_improved,
                stress_relief,
                water_saved,
                carbon_saved,
                -signal_delta["drought_scarcity_water"],
                -candidate.timeslot.id,
            )
        elif score_mode == "relief_only":
            # Phase 1 of the dominating two-phase envelope (combined_v2). Accept ONLY grid-relieving
            # moves (gate = stress_relief > 0), ranked by relief. It deliberately leaves every pure
            # carbon/water co-benefit move for the phase-2 co-benefit pass, so phase 1 does NOT scramble
            # the placements the co-benefit pass needs. (A single-score "do relief then co-benefit"
            # ranking failed: with an any-axis gate it churned many tiny moves and recovered almost no
            # co-benefit, and its move order broke ex-post certification.) Lexicographic, not a weighted
            # sum -> no bang-bang. On a no-relief window this pass is a no-op, so phase 2 runs from B.
            stress_relief = -signal_delta["weighted_stress_kwh"] + 0.5 * signal_delta["headroom_kwh"]
            score = (
                stress_relief,
                water_saved,
                carbon_saved,
                -signal_delta["drought_scarcity_water"],
                -candidate.timeslot.id,
            )
        elif score_mode in ("guarded_random", "guarded_carbon", "guarded_waterwise"):
            # MC1 guarded controls. IDENTICAL no-harm guard to the envelope (the three guard checks in
            # the move loop are unchanged); the ONLY difference is the acceptance ranking below. This
            # isolates the value of the lexicographic combined ranking: each control still certifies by
            # construction (guard-passing), so the experiment is the DELTA in certified grid-relief and
            # co-benefit, not the certification rate. A deterministic per-candidate key (seeded) gives a
            # reproducible "arbitrary order" and seed-varied CIs.
            rkey = random.Random(
                "%d|%s|%s|%d" % (
                    config.repair_random_seed, placement.pod.id,
                    candidate.flavour.id, candidate.timeslot.id,
                )
            ).random()
            if score_mode == "guarded_random":
                # Accept any move that strictly improves >=1 certified axis, in arbitrary (seeded) order.
                stress_relief = -signal_delta["weighted_stress_kwh"] + 0.5 * signal_delta["headroom_kwh"]
                eps = 1e-9
                n_improved = int(stress_relief > eps) + int(carbon_saved > eps) + int(water_saved > eps)
                score = (1.0 if n_improved > 0 else 0.0, rkey)
            elif score_mode == "guarded_carbon":
                # epsilon-constraint carbon-greedy: greedily maximize carbon reduction, ignore grid-stress.
                score = (carbon_saved, water_saved, rkey)
            else:  # guarded_waterwise
                # epsilon-constraint WaterWise: greedily improve the scalarized w*carbon + (1-w)*water
                # (relative to baseline), ignore grid-stress.
                w = config.waterwise_carbon_weight
                scalar = (
                    w * (carbon_saved / base_carbon if base_carbon > 0 else 0.0)
                    + (1.0 - w) * (water_saved / base_scarcity if base_scarcity > 0 else 0.0)
                )
                score = (scalar, rkey)
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
            # [8] Operational carbon relocated by the move = the surface exposed to carbon-forecast
            # error (embodied carbon is certain; only CI is forecast).
            candidate.footprint.operational_carbon_kg + old_candidate.footprint.operational_carbon_kg,
            # [9] Scarcity-water surface exposed to forecast error (wet-bulb for direct cooling water +
            # grid-mix for indirect off-site water). Conservative: uses TOTAL scarcity-characterized
            # water; the embodied-water fraction is certain, so this slightly over-reserves -> safe.
            candidate.footprint.scarcity_characterized_water + old_candidate.footprint.scarcity_characterized_water,
        )

    base_carbon = baseline_totals["carbon_kg"]
    base_scarcity = baseline_totals["scarcity_water"]
    # Ionising-radiation no-harm axis (default OFF -> the guard clause below never runs -> bit-identical).
    radiation_guard = config.radiation_guard
    base_radiation = baseline_totals.get("operational_radiation_kbq", 0.0)
    cur_radiation = base_radiation
    cost_guard = bool(getattr(config, "cost_guard", False))
    base_cost = baseline_totals.get("operational_cost_eur", 0.0)
    cur_cost = base_cost
    # Running totals maintained from CORRECT per-move deltas (idle re-attributed via the rebuilt
    # self-candidate as the removal baseline), so the guard checks the true current footprint rather
    # than a stale sum of per-pod footprints that loses a node's idle when its idle-bearer moves.
    cur_carbon = base_carbon
    cur_scarcity = base_scarcity
    # Per-basin water baseline (default-off). Region == watershed in this testbed; the key generalizes
    # to a basin id if a region spans basins. base_basin_water[reg] is the scarcity-characterized water
    # per region in B (the no-harm reference); cur_basin_water tracks the running per-region totals.
    per_basin_guard = config.per_basin_scarcity_guard
    base_basin_water: Dict[str, float] = {}
    cur_basin_water: Dict[str, float] = {}
    if per_basin_guard:
        for pl in baseline.placements:
            reg = (getattr(pl.candidate.flavour, "region", "") or "").upper()
            base_basin_water[reg] = base_basin_water.get(reg, 0.0) + \
                pl.candidate.footprint.scarcity_characterized_water
        cur_basin_water = dict(base_basin_water)
    # Candidates rejected by the EXACT commit-time guard (see below). Banned for the window: they
    # were priced against the true idle-once account and found harmful; occupancy changes could in
    # principle redeem one, but keeping the ban is conservative (never unsafe) and bounds reruns.
    exact_banned: set = set()
    # Node-local exact-guard state (exact_guard_mode="node_local"): per-node exact totals
    # (carbon, scarcity, radiation, cost), their global sum, and per-region scarcity, all under the
    # idle-once account. Initialized lazily on the first commit attempt with one full per-node pass;
    # afterwards each commit re-prices only the move's two touched nodes.
    node_local_guard = getattr(config, "exact_guard_mode", "node_local") != "full"
    node_exact: Dict[str, Tuple[float, float, float, float]] = {}
    exact_glob: Optional[Tuple[float, float, float, float]] = None
    exact_reg: Dict[str, float] = {}
    while repairs < max_repairs:
        best: Optional[Tuple[Tuple[float, ...], int, CandidatePlacement]] = None

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
            # Removal baseline = the pod's footprint at its CURRENT location recomputed against CURRENT
            # occupancy (the rebuilt self-candidate), NOT the stale stored candidate. This makes the
            # per-move delta correct: a node's idle is credited on removal only if the pod is still its
            # sole occupant (otherwise the idle stays with a remaining co-tenant).
            remove_baseline = next(
                (c for c in ranked
                 if c.flavour.id == old_candidate.flavour.id and c.timeslot.id == old_candidate.timeslot.id),
                old_candidate,
            )
            # Recompute cached deltas: all candidates if the pod's own placement changed
            # (its removal baseline moved); otherwise only freshly-rebuilt ones.
            recompute_all = first_build or delta_dirty.get(idx, True)
            for candidate in ranked:
                if recompute_all or not hasattr(candidate, "_nh"):
                    _compute_nh(placement, remove_baseline, candidate)
            delta_dirty[idx] = False

            for candidate in ranked:
                (is_self, lever_skip, d_carbon, d_scarcity, carbon_regret, scarcity_regret,
                 drought_delta, score, unc_carbon, unc_scarcity) = candidate._nh
                if is_self or lever_skip:
                    continue
                if (idx, candidate.flavour.id, candidate.timeslot.id) in exact_banned:
                    continue  # already found harmful by the exact commit-time guard
                # Carbon guard keeps the cumulative WORST-CASE realized carbon under baseline:
                # nominal current + uncertainty already committed + this move's exposure.
                if (cur_carbon + d_carbon + margin * carbon_regret
                        + applied_unc_carbon + robust_buffer * unc_carbon) > base_carbon + 1e-9:
                    rejected_moves += 1
                    continue
                # Scarcity guard, symmetric: nominal current + committed water-forecast uncertainty +
                # this move's water exposure. robust_buffer_water defaults to 0 -> byte-identical legacy.
                if (cur_scarcity + d_scarcity + margin * scarcity_regret
                        + applied_unc_scarcity + robust_buffer_water * unc_scarcity) > base_scarcity + 1e-9:
                    rejected_moves += 1
                    continue
                # Ionising-radiation guard (fourth axis): keep cumulative operational radiation <= B.
                # d_radiation is computed vs the SAME rebuilt removal baseline as carbon/water, so the
                # checked delta equals the committed delta. Gated -> off path is bit-identical.
                if radiation_guard:
                    d_radiation = (candidate.footprint.operational_radiation_kbq
                                   - remove_baseline.footprint.operational_radiation_kbq)
                    if cur_radiation + d_radiation > base_radiation + 1e-9:
                        rejected_moves += 1
                        continue
                # Electricity-cost guard (fifth axis): keep cumulative operational cost <= B.
                # Gated -> off path is bit-identical.
                if cost_guard:
                    d_cost = (candidate.footprint.operational_cost_eur
                              - remove_baseline.footprint.operational_cost_eur)
                    if cur_cost + d_cost > base_cost + 1e-9:
                        rejected_moves += 1
                        continue
                if drought_delta > 1e-9:
                    rejected_moves += 1
                    continue
                if per_basin_guard:
                    src_reg = (getattr(remove_baseline.flavour, "region", "") or "").upper()
                    dst_reg = (getattr(candidate.flavour, "region", "") or "").upper()
                    src_w = remove_baseline.footprint.scarcity_characterized_water
                    dst_w = candidate.footprint.scarcity_characterized_water
                    # Projected per-basin water if this move is taken; reject if ANY basin exceeds B.
                    proj_src = cur_basin_water.get(src_reg, 0.0) - src_w
                    proj_dst = cur_basin_water.get(dst_reg, 0.0) \
                        - (src_w if dst_reg == src_reg else 0.0) + dst_w
                    if (proj_dst > base_basin_water.get(dst_reg, 0.0) + 1e-9
                            or proj_src > base_basin_water.get(src_reg, 0.0) + 1e-9):
                        rejected_moves += 1
                        continue
                if score[0] <= 1e-12:
                    continue
                if best is None or score > best[0]:
                    best = (score, idx, candidate, remove_baseline)

        if best is None:
            break

        _, placement_idx, candidate, best_remove_baseline = best
        # ---- EXACT commit-time guard (2026-07-01) -------------------------------------------------
        # The per-move deltas above are fast FILTERS, but their incremental ledger can drift from the
        # true idle-once account when idle-bearer-ship is silently inherited (a bearer leaves, the
        # co-tenant's STORED footprint never picks the idle up; when the inheritor later moves, the
        # rebuilt removal baseline legitimately credits idle the ledger never paid for). On slack
        # fleets many spreading moves compound this into certified-looking harm (found: +11.8% carbon
        # "guarded" schedule on the 8-node headroom fleet; the ex-post check caught it, but the repair
        # should never walk there). Before committing the SELECTED move, price the whole hypothetical
        # placement set with the exact occupancy-ordered evaluator and reject if ANY certified ledger
        # would exceed B; on acceptance re-sync every running ledger from the exact account, so drift
        # cannot accumulate across moves.
        if node_local_guard:
            # ---- node-local exact pricing: only the move's two nodes change footprints ---------
            if exact_glob is None:
                # One full per-node pass (same cost as one whole-schedule re-pricing), then O(2
                # nodes) per commit forever after.
                groups: Dict[str, List[Placement]] = {}
                for pl in current.placements:
                    groups.setdefault(pl.candidate.flavour.id, []).append(pl)
                for fid, pls in groups.items():
                    tot = _exact_node_totals(pls, flavour_by_id[fid], config)
                    if tot is None:  # should be impossible for a feasibility-checked schedule
                        raise RuntimeError(f"exact node pricing found infeasible occupancy on {fid}")
                    node_exact[fid] = tot
                exact_glob = tuple(sum(t[i] for t in node_exact.values()) for i in range(4))  # type: ignore[assignment]
                exact_reg = {}
                for fid, t in node_exact.items():
                    reg = (getattr(flavour_by_id[fid], "region", "") or "").upper()
                    exact_reg[reg] = exact_reg.get(reg, 0.0) + t[1]
            old_pl = current.placements[placement_idx]
            fid_old = old_pl.candidate.flavour.id
            fid_new = candidate.flavour.id
            hypo = replace(old_pl, candidate=candidate)
            occ_new: Dict[str, List[Placement]] = {fid_old: [], fid_new: []}
            for i, pl in enumerate(current.placements):
                fid = pl.candidate.flavour.id
                if fid in occ_new and i != placement_idx:
                    occ_new[fid].append(pl)
            occ_new[fid_new].append(hypo)
            new_tot: Dict[str, Tuple[float, float, float, float]] = {}
            feasible = True
            for fid, pls in occ_new.items():
                t = _exact_node_totals(pls, flavour_by_id[fid], config)
                if t is None:
                    feasible = False
                    break
                new_tot[fid] = t
            if not feasible:
                exact_banned.add((placement_idx, candidate.flavour.id, candidate.timeslot.id))
                rejected_moves += 1
                continue
            glob = list(exact_glob)
            trial_reg = dict(exact_reg)
            for fid in occ_new:
                old_t = node_exact.get(fid, (0.0, 0.0, 0.0, 0.0))
                for i in range(4):
                    glob[i] += new_tot[fid][i] - old_t[i]
                reg = (getattr(flavour_by_id[fid], "region", "") or "").upper()
                trial_reg[reg] = trial_reg.get(reg, 0.0) + new_tot[fid][1] - old_t[1]
            exact_carbon, exact_scarcity, exact_radiation, exact_cost = glob
            exact_basin = trial_reg
        else:
            # ---- original whole-schedule re-pricing (kept for swap-and-diff validation) --------
            trial_placements = list(current.placements)
            trial_placements[placement_idx] = replace(trial_placements[placement_idx], candidate=candidate)
            exact = _exact_carbon_scarcity(trial_placements, flavours, config)
            if exact is None:
                exact_banned.add((placement_idx, candidate.flavour.id, candidate.timeslot.id))
                rejected_moves += 1
                continue  # infeasible under joint occupancy -> rescan without this candidate
            exact_carbon, exact_scarcity, exact_by_id = exact
            exact_basin = {}
            exact_radiation = 0.0
            exact_cost = 0.0
            for pl in trial_placements:
                m = exact_by_id.get(id(pl))
                if m is None:
                    continue
                reg = (getattr(m.flavour, "region", "") or "").upper()
                exact_basin[reg] = exact_basin.get(reg, 0.0) + m.footprint.scarcity_characterized_water
                exact_radiation += m.footprint.operational_radiation_kbq
                exact_cost += m.footprint.operational_cost_eur
        planned_unc_c = applied_unc_carbon + (robust_buffer * candidate._nh[8] if robust_buffer else 0.0)
        planned_unc_w = applied_unc_scarcity + (robust_buffer_water * candidate._nh[9] if robust_buffer_water else 0.0)
        harmful = (
            exact_carbon + planned_unc_c > base_carbon + 1e-9
            or exact_scarcity + planned_unc_w > base_scarcity + 1e-9
            or (radiation_guard and exact_radiation > base_radiation + 1e-9)
            or (cost_guard and exact_cost > base_cost + 1e-9)
            or (per_basin_guard and any(
                exact_basin.get(reg, 0.0) > base_basin_water.get(reg, 0.0) + 1e-9
                for reg in set(exact_basin) | set(base_basin_water)))
        )
        if harmful:
            exact_banned.add((placement_idx, candidate.flavour.id, candidate.timeslot.id))
            rejected_moves += 1
            continue  # the exact account rejects this move -> rescan for the next best
        # Commit: re-sync ALL running ledgers from the exact account (no incremental drift).
        cur_carbon = exact_carbon
        cur_scarcity = exact_scarcity
        if radiation_guard:
            cur_radiation = exact_radiation
        if cost_guard:
            cur_cost = exact_cost
        if per_basin_guard:
            cur_basin_water = dict(exact_basin)
        if node_local_guard:
            # Persist the two touched nodes' exact totals + the global/per-region aggregates.
            node_exact.update(new_tot)
            exact_glob = tuple(glob)  # type: ignore[assignment]
            exact_reg = trial_reg
        if robust_buffer:
            applied_unc_carbon += robust_buffer * candidate._nh[8]  # commit this move's carbon-forecast exposure
        if robust_buffer_water:
            applied_unc_scarcity += robust_buffer_water * candidate._nh[9]  # commit this move's water-forecast exposure
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

    # D1: certified atomic consolidation post-pass (default OFF -> bit-identical). Closes the
    # consolidation barrier the per-move guard cannot cross (an idle-once activation that fails the
    # carbon guard mid-consolidation), by evaluating a coordinated move-SET against the certificate.
    if config.consolidation_pass:
        consolidated = _consolidation_pass(
            current=current,
            flavours=flavours,
            signals=signals,
            config=config,
            base_carbon=base_carbon,
            base_scarcity=base_scarcity,
        )
        repairs += consolidated

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
            # Apply the matched candidate's resources so the NEXT pod on this node-slot sees the
            # occupancy: node idle power is charged once (to the first pod), not to every pod.
            # Without this, every pod is scored against empty capacity -> idle massively over-counted.
            _apply_candidate_resources(placement.pod, match, leftover_cpu, leftover_ram, leftover_gpu)


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
    # Independent verification: if a separate CI set is given (e.g. real generation-mix), re-evaluate
    # the certificate against it; decisions stay on the decision-time signals above. Else verify on them.
    verify_active = bool(config.verify_forecasts_file or config.verify_config_file)
    if verify_active:
        verify_flavours = load_flavours_for_pilot(replace(
            config,
            forecasts_file=config.verify_forecasts_file or config.forecasts_file,
            config_file=config.verify_config_file or config.config_file,
        ))
        verify_signals = build_action_signals(verify_flavours, config)
    else:
        verify_flavours = realized_flavours
        verify_signals = realized_signals

    packing, _, _ = build_greedy_schedule(method_key="packing", pods=pods, flavours=decision_flavours, config=config)
    carbon, _, _ = build_greedy_schedule(method_key="carbon", pods=pods, flavours=decision_flavours, config=config)
    water, _, _ = build_greedy_schedule(method_key="water_scarcity", pods=pods, flavours=decision_flavours, config=config)
    # WaterWise-style scalarized co-optimizer: the fair SOTA comparator (carbon+water
    # weighted sum), which tolerates residual harm by construction and so cannot certify.
    waterwise, _, _ = build_greedy_schedule(method_key="waterwise", pods=pods, flavours=decision_flavours, config=config)
    no_harm_flex = repair_schedule_no_harm(
        method_key="no_harm_flex",
        baseline=packing,
        pods=pods,
        flavours=decision_flavours,
        signals=decision_signals,
        config=config,
        score_mode="combined",  # pursue grid-stress relief AND certified carbon/water co-benefits
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

    results = [packing, carbon, water, waterwise, search_control, no_harm_flex]
    # Re-evaluate footprints + certificate against the verification world: an independent CI set if
    # provided (decide-on-proxy / verify-on-real-CI), else the realized (un-noised) signals for RQ3.
    # ALWAYS re-materialize footprints under the verification flavours, so the reported totals and
    # the certificate use CORRECT occupancy-based accounting (idle charged once per active node-slot),
    # not the in-place repair's stale per-pod footprints (which lose a node's idle when its idle-bearing
    # pod moves -> spurious carbon/water "savings"). verify_flavours defaults to the realized
    # (decision-time) signals, so for the headline this is an honest recompute on the same signals;
    # under RQ3 it is the realized world; under a verify_* override it is the independent CI/config set.
    for result in results:
        _rematerialize_under_realized(result, verify_flavours, config)

    signals = verify_signals
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
