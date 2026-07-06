#!/usr/bin/env python3
"""Escrow / baseline-inflation replay: what a gamed declared baseline fabricates,
and what a recomputable declaration catches.

Threat (FERC v. Silkman / Lincoln Paper, DR baseline inflation): the certificate is
relative to the operator's DECLARED baseline B. If B may be free-form, a Lincoln-style
operator declares a wasteful-but-feasible B_pad (here: a fraction p of flexible pods
placed at their WORST-carbon feasible candidate instead of the packing choice), then
ships an ordinary certified schedule. Every certificate inequality passes; the
"certified savings" grow by exactly the padding.

Defense measured here, in two rungs:
  1. RECOMPUTE (the paper's design): B is not free-form -- it is a declared, deterministic
     POLICY (consolidation-first packing) any verifier re-runs on the declared pods+nodes.
     Padded baselines differ from the recomputed policy on >=1 placement -> caught, always.
  2. PLAUSIBILITY (aggregate-only declaration): if only totals were declared, an idle-share
     statistic (declared baseline energy vs recomputed-policy energy) flags padding above
     a threshold -- we report the statistic's separation per padding level.

Emits experiments/escrow/escrow_inflation.json.

Run:
  PYTHONPATH=pkg/carbon-aware/server-python python scripts/escrow_inflation_replay.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SERVER = REPO / "pkg" / "carbon-aware" / "server-python"
for p in (str(SERVER), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_water_basin_certification as wb  # noqa: E402
from carbon_aware.no_harm_flexibility import (  # noqa: E402
    Placement,
    ScheduleResult,
    _apply_candidate_resources,
    _candidate_sort_key,
    _init_resources,
    build_action_signals,
    build_greedy_schedule,
    build_timeslots,
    classify_pod,
    find_ranked_candidates,
    load_flavours_for_pilot,
    load_pods,
    order_pods_for_pilot,
    repair_schedule_no_harm,
    schedule_totals,
    _rematerialize_under_realized,
)

OUT = REPO / "experiments" / "escrow"
WINDOW = "azure_packing_2020_d0_s42_200"


def build_padded_baseline(pods, flavours, config, pad_fraction: float, rng: random.Random) -> ScheduleResult:
    """Mirror build_greedy_schedule('packing'), except a fraction `pad_fraction` of FLEXIBLE
    pods takes its WORST-carbon feasible candidate (Lincoln-style: waste during the
    baseline-setting period). Feasible, deterministic given the seed, and free-form-declared."""
    start = time.perf_counter()
    leftover_cpu, leftover_ram, leftover_gpu = _init_resources(flavours, config.max_timeslots)
    placements, unplaced = [], []
    timeslots = build_timeslots(config.max_timeslots)
    for pod in order_pods_for_pilot(pods):
        ranked = find_ranked_candidates(
            pod=pod, flavours=list(flavours), timeslots=timeslots,
            leftover_cpu=leftover_cpu, leftover_ram=leftover_ram,
            max_time_slots=config.max_timeslots, objective_mode="carbon",
            water_metric="scarcity", leftover_gpu=leftover_gpu,
        )
        if not ranked:
            unplaced.append(pod)
            continue
        flex = classify_pod(pod, config.flexibility_slack_hours)
        by_packing = sorted(ranked, key=lambda item: _candidate_sort_key(item, "packing"))
        if flex == "flexible" and rng.random() < pad_fraction:
            candidate = max(ranked, key=lambda item: item.footprint.total_carbon_kg)
        else:
            candidate = by_packing[0]
        _apply_candidate_resources(pod, candidate, leftover_cpu, leftover_ram, leftover_gpu)
        placements.append(Placement(pod=pod, candidate=candidate, flexibility_class=flex))
    return ScheduleResult(method_key=f"padded@{pad_fraction:.2f}", placements=placements,
                          unplaced_pods=unplaced, elapsed_seconds=time.perf_counter() - start)


def placement_map(result: ScheduleResult):
    return {p.pod.id: (getattr(p.candidate.flavour, "id", "?"), p.candidate.timeslot.id)
            for p in result.placements}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pads", type=float, nargs="*", default=[0.0, 0.1, 0.2, 0.3, 0.5, 1.0])
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    run_dir = OUT / "fleet_build"
    run_dir.mkdir(parents=True, exist_ok=True)
    pc, n_nodes, n_pods = wb._build_azure_config(WINDOW, run_dir, seed=args.seed)
    pc.output_dir.mkdir(parents=True, exist_ok=True)
    pods = load_pods(pc.workloads_dir, max_pods=pc.max_pods)
    flavours = load_flavours_for_pilot(pc)
    signals = build_action_signals(flavours, pc)
    print(f"[fleet] {WINDOW}: {n_pods} pods on {n_nodes} nodes")

    # The verifier's recomputation of the declared policy (deterministic packing).
    honest, _, _ = build_greedy_schedule(method_key="packing", pods=pods, flavours=flavours, config=pc)
    _rematerialize_under_realized(honest, flavours, pc)
    honest_t = schedule_totals(honest, signals)
    honest_map = placement_map(honest)

    rows = []
    for pad in args.pads:
        rng = random.Random(args.seed * 7919 + int(pad * 100))
        declared = build_padded_baseline(pods, flavours, pc, pad, rng)
        _rematerialize_under_realized(declared, flavours, pc)
        declared_t = schedule_totals(declared, signals)

        # the operator's certified schedule vs the DECLARED baseline
        cert = repair_schedule_no_harm(
            method_key="no_harm_flex", baseline=declared, pods=pods,
            flavours=flavours, signals=signals, config=pc, score_mode="combined")
        _rematerialize_under_realized(cert, flavours, pc)
        cert_t = schedule_totals(cert, signals)

        def pct(x, b):
            return 100.0 * (x - b) / b if b else 0.0

        # detection rung 1: exact recompute of the declared policy
        mism = sum(1 for k, v in placement_map(declared).items() if honest_map.get(k) != v)
        # detection rung 2: aggregate plausibility (energy of declared vs recomputed policy)
        energy_excess_pct = pct(declared_t["operational_energy_kwh"], honest_t["operational_energy_kwh"])

        row = {
            "pad_fraction": pad,
            "declared_baseline_inflation_pct": {
                "carbon": pct(declared_t["carbon_kg"], honest_t["carbon_kg"]),
                "scarcity": pct(declared_t["scarcity_water"], honest_t["scarcity_water"]),
                "energy": energy_excess_pct,
            },
            "certified_savings_vs_declared_pct": {
                "carbon": pct(cert_t["carbon_kg"], declared_t["carbon_kg"]),
                "scarcity": pct(cert_t["scarcity_water"], declared_t["scarcity_water"]),
            },
            "certified_savings_vs_honest_pct": {
                "carbon": pct(cert_t["carbon_kg"], honest_t["carbon_kg"]),
                "scarcity": pct(cert_t["scarcity_water"], honest_t["scarcity_water"]),
            },
            "certificate_passes_vs_declared": (
                cert_t["carbon_kg"] <= declared_t["carbon_kg"] + 1e-9
                and cert_t["scarcity_water"] <= declared_t["scarcity_water"] + 1e-9
                and len(cert.unplaced_pods) <= len(declared.unplaced_pods)),
            "recompute_placement_mismatches": mism,
            "caught_by_recompute": mism > 0,
            "energy_plausibility_statistic_pct": energy_excess_pct,
        }
        rows.append(row)
        print(f"[pad {pad:>4.0%}] baseline inflated C {row['declared_baseline_inflation_pct']['carbon']:+.1f}% | "
              f"cert 'savings' vs declared C {row['certified_savings_vs_declared_pct']['carbon']:+.1f}% "
              f"(vs honest {row['certified_savings_vs_honest_pct']['carbon']:+.1f}%) | "
              f"cert passes: {row['certificate_passes_vs_declared']} | "
              f"recompute mismatches: {mism} caught={row['caught_by_recompute']} | "
              f"energy stat {energy_excess_pct:+.2f}%")

    out = {
        "window": WINDOW, "seed": args.seed, "n_pods": n_pods, "n_nodes": n_nodes,
        "honest_baseline": {k: honest_t[k] for k in
                            ("carbon_kg", "scarcity_water", "operational_energy_kwh")},
        "rows": rows,
    }
    (OUT / "escrow_inflation.json").write_text(json.dumps(out, indent=2))
    print(f"\n[done] -> {OUT/'escrow_inflation.json'}")


if __name__ == "__main__":
    main()
