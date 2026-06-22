#!/usr/bin/env python3
"""T8 (part 1) — MILP optimality gap for the no-harm repair on small instances.

The engine's footprints are occupancy-dependent (idle attribution + proportional embodied),
which is non-linear in the assignment. To get a clean, honest optimality gap we compare the
greedy no-harm repair against an exact MILP on an IDENTICAL occupancy-independent linear cost
model (operational_only + use_pod_power_only): each pod-candidate has fixed carbon, scarcity,
and weighted grid-stress. Both methods optimise the same costs, so the gap measures the greedy's
SEARCH quality (decoupled from the footprint model). Objective: minimise total weighted
grid-stress subject to no-harm (carbon <= baseline, scarcity <= baseline), capacity, and
one-placement-per-pod (SLO).
"""
from __future__ import annotations
import argparse, sys, logging
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(REPO / "pkg" / "carbon-aware" / "server-python"))
sys.path.insert(0, str(SCRIPT_DIR))
logging.disable(logging.CRITICAL)

import pulp  # noqa: E402
from carbon_aware.models import CarbonAwareTimeslot  # noqa: E402
from carbon_aware.footprints import compute_footprint_vector  # noqa: E402
from carbon_aware.utils import is_timeslot_valid, build_timeslots  # noqa: E402
from carbon_aware.no_harm_flexibility import (  # noqa: E402
    PilotConfig, load_flavours_for_pilot, load_pods, build_action_signals,
    build_greedy_schedule, classify_pod, _candidate_water_value,
)
from no_harm_flex_matrix import _generate_case_inputs, load_generator_dependencies  # noqa: E402

TA = REPO / "pkg" / "carbon-aware" / "data" / "timealigned"


def linear_cost(flavour, t, pod, signals, max_ts):
    """Occupancy-independent (pod-power-only, operational-only) carbon/scarcity + weighted stress
    for placing `pod` on `flavour` starting at slot t. Returns None if infeasible (window)."""
    dur = max(int(pod.duration), 1)
    if t + dur > max_ts:
        return None
    fp = compute_footprint_vector(flavour, t, pod, operational_only=True, use_pod_power_only=True)
    region = (getattr(flavour, "region", "") or "").upper()
    e_per_slot = fp.operational_energy_kwh / dur
    wstress = 0.0
    for o in range(dur):
        sig = signals.get((region, t + o))
        if sig is not None:
            wstress += e_per_slot * sig.grid_stress_score
    return dict(carbon=fp.total_carbon_kg, scarcity=fp.scarcity_characterized_water,
               wstress=wstress, flavour=flavour, t=t, dur=dur)


def build_candidates(pods, flavours, timeslots, signals, max_ts):
    cands = {}
    for p in pods:
        lst = []
        for f in flavours:
            for ts in timeslots:
                if not is_timeslot_valid(ts, p):
                    continue
                c = linear_cost(f, ts.id, p, signals, max_ts)
                if c is not None:
                    lst.append(c)
        cands[p.id] = lst
    return cands


def greedy_no_harm(pods, cands, base_choice, base_carbon, base_scarcity, flavours, max_ts):
    """Mimic the engine repair on the linear cost model: start at baseline, greedily move
    flexible pods to cut weighted stress while keeping carbon/scarcity <= baseline + capacity."""
    choice = dict(base_choice)
    leftover_cpu = {f.id: {s: f.totalCpu for s in range(max_ts)} for f in flavours}
    leftover_ram = {f.id: {s: f.totalRam for s in range(max_ts)} for f in flavours}
    by_id = {p.id: p for p in pods}

    def apply(pid, c, sign):
        p = by_id[pid]
        for o in range(c["dur"]):
            leftover_cpu[c["flavour"].id][c["t"] + o] -= sign * p.cpuRequest
            leftover_ram[c["flavour"].id][c["t"] + o] -= sign * p.ramRequest

    for pid, c in choice.items():
        apply(pid, c, +1)

    def fits(pid, c):
        p = by_id[pid]
        for o in range(c["dur"]):
            if (leftover_cpu[c["flavour"].id][c["t"] + o] < p.cpuRequest or
                    leftover_ram[c["flavour"].id][c["t"] + o] < p.ramRequest):
                return False
        return True

    cur_c = sum(choice[p.id]["carbon"] for p in pods)
    cur_w = sum(choice[p.id]["scarcity"] for p in pods)
    flex = [p.id for p in pods if classify_pod(p, 2.0) == "flexible"]
    improved = True
    while improved:
        improved = True
        best = None
        for pid in flex:
            old = choice[pid]
            apply(pid, old, -1)  # release self
            for c in cands[pid]:
                if c["flavour"].id == old["flavour"].id and c["t"] == old["t"]:
                    continue
                if not fits(pid, c):
                    continue
                d_c = c["carbon"] - old["carbon"]
                d_w = c["scarcity"] - old["scarcity"]
                if cur_c + d_c > base_carbon + 1e-9 or cur_w + d_w > base_scarcity + 1e-9:
                    continue
                relief = old["wstress"] - c["wstress"]
                if relief <= 1e-12:
                    continue
                if best is None or relief > best[0]:
                    best = (relief, pid, c, d_c, d_w)
            apply(pid, old, +1)  # re-apply self
        if best is None:
            break
        _, pid, c, d_c, d_w = best
        apply(pid, choice[pid], -1)
        apply(pid, c, +1)
        choice[pid] = c
        cur_c += d_c
        cur_w += d_w
    return sum(choice[p.id]["wstress"] for p in pods), cur_c, cur_w


def solve_milp(pods, cands, base_carbon, base_scarcity, flavours, max_ts):
    prob = pulp.LpProblem("no_harm", pulp.LpMinimize)
    x = {}
    for p in pods:
        for i, c in enumerate(cands[p.id]):
            x[(p.id, i)] = pulp.LpVariable(f"x_{p.id}_{i}", cat="Binary")
    # one placement per pod
    for p in pods:
        prob += pulp.lpSum(x[(p.id, i)] for i in range(len(cands[p.id]))) == 1
    # capacity per (flavour, slot)
    by_id = {p.id: p for p in pods}
    for f in flavours:
        for s in range(max_ts):
            cpu_terms, ram_terms = [], []
            for p in pods:
                for i, c in enumerate(cands[p.id]):
                    if c["flavour"].id == f.id and c["t"] <= s < c["t"] + c["dur"]:
                        cpu_terms.append(x[(p.id, i)] * by_id[p.id].cpuRequest)
                        ram_terms.append(x[(p.id, i)] * by_id[p.id].ramRequest)
            if cpu_terms:
                prob += pulp.lpSum(cpu_terms) <= f.totalCpu
                prob += pulp.lpSum(ram_terms) <= f.totalRam
    # no-harm
    prob += pulp.lpSum(x[(p.id, i)] * cands[p.id][i]["carbon"]
                       for p in pods for i in range(len(cands[p.id]))) <= base_carbon + 1e-9
    prob += pulp.lpSum(x[(p.id, i)] * cands[p.id][i]["scarcity"]
                       for p in pods for i in range(len(cands[p.id]))) <= base_scarcity + 1e-9
    # objective: minimise weighted stress
    prob += pulp.lpSum(x[(p.id, i)] * cands[p.id][i]["wstress"]
                       for p in pods for i in range(len(cands[p.id])))
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return pulp.value(prob.objective), pulp.LpStatus[prob.status]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes-per-region", type=int, default=1)
    ap.add_argument("--pods", type=int, default=16)
    ap.add_argument("--max-timeslots", type=int, default=12)
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()

    config_file = REPO / "pkg" / "carbon-aware" / "infra-workload-config.yaml"
    base_config, gn, gt = load_generator_dependencies(REPO, config_file)
    print(f"# MILP optimality gap (linear occupancy-independent cost model)")
    print(f"# nodes/region={args.nodes_per_region} pods={args.pods} max_ts={args.max_timeslots}")
    gaps = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        paths = _generate_case_inputs(base_config=base_config, generate_nodes_file=gn,
            generate_timeslot_files=gt, input_dir=REPO / "experiments" / "flexibility" / "t8_milp" / f"s{seed}",
            pod_count=args.pods, seed=seed, timeslots=args.max_timeslots, config_file=config_file,
            deadline_flex_hours=args.max_timeslots, nodes_per_region=args.nodes_per_region)
        cfg = PilotConfig(repo_root=REPO, nodes_file=paths["nodes_file"], workloads_dir=paths["workloads_dir"],
            forecasts_file=TA / "forecasts.json", config_file=config_file,
            output_dir=REPO / "experiments" / "flexibility" / "t8_milp" / "out",
            max_timeslots=args.max_timeslots, max_pods=None, scenario="heatwave-drought", lever_mode="both",
            grid_signal_csv=TA / "grid_residual_region_slot.csv", wue_csv=TA / "wue_region_slot.csv")
        flavours = load_flavours_for_pilot(cfg)
        all_pods = load_pods(paths["workloads_dir"])
        signals = build_action_signals(flavours, cfg)
        timeslots = build_timeslots(args.max_timeslots)
        # baseline = packing placement, re-costed on the linear model. Compare over the pods
        # packing actually placed (SLO = match the baseline's placed set).
        packing, _, _ = build_greedy_schedule(method_key="packing", pods=all_pods, flavours=flavours, config=cfg)
        pods = [pl.pod for pl in packing.placements]
        cands = build_candidates(pods, flavours, timeslots, signals, args.max_timeslots)
        base_choice = {}
        ok = True
        for pl in packing.placements:
            match = next((c for c in cands[pl.pod.id]
                          if c["flavour"].id == pl.candidate.flavour.id and c["t"] == pl.candidate.timeslot.id), None)
            if match is None:
                ok = False; break
            base_choice[pl.pod.id] = match
        if not ok:
            print(f"seed={seed}: skipped (baseline candidate not representable)"); continue
        print(f"seed={seed}: placed {len(pods)}/{len(all_pods)} pods", end="  ")
        base_carbon = sum(c["carbon"] for c in base_choice.values())
        base_scar = sum(c["scarcity"] for c in base_choice.values())
        base_w = sum(c["wstress"] for c in base_choice.values())
        g_w, g_c, g_s = greedy_no_harm(pods, cands, base_choice, base_carbon, base_scar, flavours, args.max_timeslots)
        m_w, status = solve_milp(pods, cands, base_carbon, base_scar, flavours, args.max_timeslots)
        gap = (g_w - m_w) / m_w * 100.0 if m_w > 1e-12 else 0.0
        gaps.append(gap)
        print(f"seed={seed}: baseline_wstress={base_w:.4f}  greedy={g_w:.4f}  MILP*={m_w:.4f} ({status})  "
              f"greedy_gap={gap:.2f}%  (greedy relief={100*(base_w-g_w)/base_w:.1f}% vs MILP {100*(base_w-m_w)/base_w:.1f}%)")
    if gaps:
        print(f"\nmean greedy optimality gap = {sum(gaps)/len(gaps):.2f}%  (n={len(gaps)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
