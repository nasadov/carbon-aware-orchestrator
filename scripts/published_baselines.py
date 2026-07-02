#!/usr/bin/env python3
"""Faithful published-baseline schedulers for the No-Harm Flexibility Envelope.

This module adds, as STANDALONE schedulers, the additional published baselines a
T-SUSC reviewer would demand so the comparison cannot be dismissed as "weak
baselines". It does NOT edit the engine: it imports the engine's feasible-candidate
enumeration, footprint accounting, re-materialisation, and the ex-post no-harm
certificate, and only supplies each method's own *selection rule* over the
deadline-feasible candidate set. Every baseline therefore (a) sees exactly the same
feasible (site, timeslot) options the engine's own baselines see, (b) is accounted
with the same idle-once-per-node-slot footprint model, and (c) is judged by the same
`summarize_against_reference` certificate. Adaptations to the original papers are
documented inline at each function.

Baselines implemented here:
  * cic        -- Carbon-Intelligent Computing, temporal shift-to-cleanest within
                  deadline at a fixed site (Radovanovic et al., 2023, Google).
  * wait_awhile-- "Let's Wait Awhile" suspend/resume carbon-aware deferral of
                  *flexible* jobs only; firm jobs run ASAP (Wiesner et al., 2021).
  * greenslot  -- GreenSlot/GreenHadoop deadline-aware green scheduling: least-laxity
                  ordering + greenest feasible slot, fixed site (Goiri et al., 2011/12).
  * waterwise@w-- WaterWise scalarised carbon+water co-optimiser swept across weights
                  to trace its full carbon-water frontier (Jiang et al., 2025).

The engine's own baselines (packing, carbon, water_scarcity, waterwise@0.5) and the
envelope (no_harm_flex) are produced by importing the engine, so all methods share one
testbed, one set of signals, one accounting pass, and one certificate.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SERVER = REPO_ROOT / "pkg" / "carbon-aware" / "server-python"
for _p in (str(SERVER), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from carbon_aware.algorithms.heuristic import (  # noqa: E402
    CandidatePlacement,
    _candidate_water_value,
    find_ranked_candidates,
)
from carbon_aware.models import CarbonAwarePod, EnvironmentalFlavor  # noqa: E402
from carbon_aware.no_harm_flexibility import (  # noqa: E402
    ActionSignal,
    Placement,
    PilotConfig,
    RegionSlot,
    ScheduleResult,
    _apply_candidate_resources,
    _init_resources,
    _rematerialize_under_realized,
    build_action_signals,
    build_greedy_schedule,
    classify_pod,
    load_flavours_for_pilot,
    load_pods,
    order_pods_for_pilot,
    repair_schedule_no_harm,
    summarize_against_reference,
)
from carbon_aware.utils import build_timeslots  # noqa: E402


# --------------------------------------------------------------------------- helpers
def _candidate_ci(candidate: CandidatePlacement) -> float:
    """Effective average carbon intensity (gCO2/kWh) of a candidate placement.

    This is the marginal-energy-weighted CI the method actually optimises: the
    operational carbon divided by the operational energy of THIS placement. It
    automatically reflects per-slot CI x duration x PUE, so a temporal shifter
    sees the true carbon of starting at each feasible slot, not a raw lookup.
    """
    e = candidate.footprint.operational_energy_kwh
    if e <= 1e-12:
        # No operational energy (degenerate): fall back to total carbon so ties are stable.
        return candidate.footprint.total_carbon_g
    return candidate.footprint.operational_carbon_g / e


def _laxity(pod: CarbonAwarePod) -> float:
    return float(getattr(pod, "deadline_hours", 24.0)) - float(getattr(pod, "duration", 1.0))


def _site_id(candidate: CandidatePlacement) -> str:
    return candidate.flavour.id


def _ranked_for_pod(
    pod: CarbonAwarePod,
    flavours: Sequence[EnvironmentalFlavor],
    timeslots,
    leftover_cpu,
    leftover_ram,
    leftover_gpu,
    config: PilotConfig,
) -> List[CandidatePlacement]:
    return find_ranked_candidates(
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


def _earliest_feasible_site_choice(
    ranked: Sequence[CandidatePlacement],
) -> CandidatePlacement:
    """The site/slot a deadline-blind packing scheduler would pick: earliest start,
    then best packing fit, then carbon, then water -- the engine's `packing` sort key.
    Used to PIN the site for the single-site temporal methods (CIC, Wait Awhile,
    GreenSlot) so they shift *time*, not *region*, faithfully to their papers.
    """
    return min(
        ranked,
        key=lambda c: (
            c.timeslot.id,
            c.pack_score,
            c.footprint.total_carbon_g,
            _candidate_water_value(c.footprint, "scarcity"),
        ),
    )


# --------------------------------------------------------------------------- baselines
def build_baseline_schedule(
    *,
    method_key: str,
    selector: Callable[[CarbonAwarePod, str, List[CandidatePlacement]], CandidatePlacement],
    pods: Sequence[CarbonAwarePod],
    flavours: Sequence[EnvironmentalFlavor],
    config: PilotConfig,
) -> ScheduleResult:
    """Generic greedy driver: enumerate deadline-feasible candidates for each pod (in
    the engine's canonical pod order), let `selector` pick one given the pod's
    flexibility class, commit its resources, and emit an engine-compatible
    ScheduleResult. Mirrors build_greedy_schedule exactly except for the selection rule.
    """
    start = time.perf_counter()
    leftover_cpu, leftover_ram, leftover_gpu = _init_resources(flavours, config.max_timeslots)
    placements: List[Placement] = []
    unplaced: List[CarbonAwarePod] = []
    timeslots = build_timeslots(config.max_timeslots)

    for pod in order_pods_for_pilot(pods):
        ranked = _ranked_for_pod(pod, flavours, timeslots, leftover_cpu, leftover_ram, leftover_gpu, config)
        if not ranked:
            unplaced.append(pod)
            continue
        flex_class = classify_pod(pod, config.flexibility_slack_hours)
        candidate = selector(pod, flex_class, ranked)
        _apply_candidate_resources(pod, candidate, leftover_cpu, leftover_ram, leftover_gpu)
        placements.append(Placement(pod=pod, candidate=candidate, flexibility_class=flex_class))

    return ScheduleResult(
        method_key=method_key,
        placements=placements,
        unplaced_pods=unplaced,
        elapsed_seconds=time.perf_counter() - start,
    )


def _cic_selector(pod: CarbonAwarePod, flex_class: str, ranked: List[CandidatePlacement]) -> CandidatePlacement:
    """Carbon-Intelligent Computing (Radovanovic et al., 2023).

    Decision rule: temporal load shifting to the cleanest hours WITHIN the job's
    deadline, at a fixed location (Google's CICS shifts time inside a cluster; it does
    not migrate jobs across regions). We pin the site to the deadline-blind packing
    choice, then among that site's deadline-feasible start slots pick the one with the
    lowest effective carbon intensity (the "carbon cost curve" minimiser). Flexible and
    firm jobs are both shiftable in CICS within their deadline, but firm jobs in our
    trace have ~zero slack so they cannot move -- handled naturally by the feasible set.

    Adaptation: CICS optimises a day-ahead virtual-capacity-curve; with a static
    forecast over the horizon this reduces exactly to "earliest-feasible site, cleanest
    feasible hour", which is what we implement.
    """
    site = _earliest_feasible_site_choice(ranked).flavour.id
    same_site = [c for c in ranked if c.flavour.id == site]
    pool = same_site or ranked
    return min(pool, key=lambda c: (_candidate_ci(c), c.footprint.total_carbon_g, c.timeslot.id))


def _wait_awhile_selector(pod: CarbonAwarePod, flex_class: str, ranked: List[CandidatePlacement]) -> CandidatePlacement:
    """"Let's Wait Awhile" (Wiesner et al., 2021).

    Decision rule: suspend/resume carbon-aware temporal DEFERRAL. Only deferrable
    (flexible) jobs are shifted to the lowest-carbon period within their deadline;
    non-deferrable (firm) jobs run as soon as possible. Single region (the paper studies
    temporal, not geographic, shifting). We pin the site to the packing choice and:
      * flexible -> cleanest feasible start slot at that site (defer);
      * firm     -> earliest feasible start slot at that site (run ASAP).

    Adaptation: the paper models continuous suspend/resume with checkpoint overhead;
    our slot model places each job in one contiguous deadline-feasible window, so we
    apply the deferral as a start-slot choice (the dominant carbon effect; checkpoint
    overhead would only weaken deferral, not the no-harm comparison).
    """
    site = _earliest_feasible_site_choice(ranked).flavour.id
    same_site = [c for c in ranked if c.flavour.id == site]
    pool = same_site or ranked
    if flex_class == "flexible":
        return min(pool, key=lambda c: (_candidate_ci(c), c.footprint.total_carbon_g, c.timeslot.id))
    return min(pool, key=lambda c: (c.timeslot.id, c.footprint.total_carbon_g))


def _greenslot_selector(pod: CarbonAwarePod, flex_class: str, ranked: List[CandidatePlacement]) -> CandidatePlacement:
    """GreenSlot / GreenHadoop (Goiri et al., 2011/2012).

    Decision rule: deadline-aware "green" scheduling. GreenSlot predicts green-energy
    availability per slot and schedules each job into the slots with the most green
    (i.e. lowest-carbon) energy while meeting deadlines, breaking ties toward running
    earlier to leave room for later jobs. Jobs are considered in least-laxity-first
    order (enforced by the driver's pod ordering, which already sorts by slack). Single
    cluster. We pin the site to the packing choice and pick the greenest feasible start
    slot there, tie-breaking earliest.

    Adaptation: GreenSlot uses an explicit green-energy MW forecast; we use grid carbon
    intensity (lower CI == greener marginal energy) as the green-ness signal, which is
    the modern equivalent and the signal our testbed actually carries.
    """
    site = _earliest_feasible_site_choice(ranked).flavour.id
    same_site = [c for c in ranked if c.flavour.id == site]
    pool = same_site or ranked
    return min(pool, key=lambda c: (_candidate_ci(c), c.timeslot.id, c.footprint.total_carbon_g))


def _cic_spatial_selector(pod: CarbonAwarePod, flex_class: str, ranked: List[CandidatePlacement]) -> CandidatePlacement:
    """CIC with geographic load-balancing ENABLED (Radovanovic et al. discuss both
    temporal shifting and cross-cluster geographic shifting). This is the MOST charitable
    carbon-minimising version of CIC: cleanest hour across ANY feasible site -- it is, in
    effect, the engine's carbon-greedy, included so a reviewer cannot claim our single-site
    CIC handicaps the baseline. It maximises carbon savings but inherits carbon-greedy's
    water inflation / SLO risk and so still cannot certify.
    """
    return min(ranked, key=lambda c: (_candidate_ci(c), c.footprint.total_carbon_g, c.timeslot.id))


SELECTORS: Dict[str, Callable] = {
    "cic": _cic_selector,
    "cic_spatial": _cic_spatial_selector,
    "wait_awhile": _wait_awhile_selector,
    "greenslot": _greenslot_selector,
}


# --------------------------------------------------------------------------- full run
def run_all_baselines(
    config: PilotConfig,
    *,
    waterwise_weights: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> Dict[str, object]:
    """Build EVERY method on one testbed and judge them all with the same certificate.

    Returns a dict with the per-method summary rows (vs the packing reference) and the
    raw schedule results. All footprints are re-materialised under the verification
    flavours first (idle charged once per active node-slot), exactly as the engine's own
    pilot does, so the new baselines are accounted identically to packing/envelope.
    """
    realized_flavours = load_flavours_for_pilot(config)
    if not realized_flavours:
        raise RuntimeError(f"No nodes loaded from {config.nodes_file}")
    pods = load_pods(config.workloads_dir, max_pods=config.max_pods)
    if not pods:
        raise RuntimeError(f"No pods loaded from {config.workloads_dir}")
    signals = build_action_signals(realized_flavours, config)

    # Engine baselines + envelope (imported, not reimplemented).
    packing, _, _ = build_greedy_schedule(method_key="packing", pods=pods, flavours=realized_flavours, config=config)
    carbon, _, _ = build_greedy_schedule(method_key="carbon", pods=pods, flavours=realized_flavours, config=config)
    water, _, _ = build_greedy_schedule(method_key="water_scarcity", pods=pods, flavours=realized_flavours, config=config)

    results: List[ScheduleResult] = [packing, carbon, water]

    # WaterWise frontier sweep (engine selection rule, multiple weights).
    from dataclasses import replace as _dc_replace
    for w in waterwise_weights:
        ww_cfg = _dc_replace(config, waterwise_carbon_weight=float(w))
        ww, _, _ = build_greedy_schedule(method_key=f"waterwise@{w:g}", pods=pods, flavours=realized_flavours, config=ww_cfg)
        results.append(ww)

    # New published baselines (this module's selection rules).
    for key, selector in SELECTORS.items():
        results.append(
            build_baseline_schedule(
                method_key=key, selector=selector, pods=pods, flavours=realized_flavours, config=config
            )
        )

    # The envelope (engine repair, combined objective).
    no_harm_flex = repair_schedule_no_harm(
        method_key="no_harm_flex",
        baseline=packing,
        pods=pods,
        flavours=realized_flavours,
        signals=signals,
        config=config,
        score_mode="combined",
    )
    results.append(no_harm_flex)

    # Identical accounting pass for every method: re-materialise under verification flavours.
    verify_flavours = realized_flavours
    verify_signals = signals
    for result in results:
        _rematerialize_under_realized(result, verify_flavours, config)

    summary_rows = [summarize_against_reference(r, packing, verify_signals) for r in results]
    return {
        "summary_rows": summary_rows,
        "results": results,
        "pods": pods,
        "flavours": realized_flavours,
        "signals": verify_signals,
    }
