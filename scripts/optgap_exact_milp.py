#!/usr/bin/env python3
"""Optimality gap of the greedy no-harm repair vs an EXACT no-harm-constrained MILP.

Unlike scripts/milp_optimality_gap.py (which uses an occupancy-INDEPENDENT linear cost
to isolate search quality), this script linearizes the engine's TRUE occupancy-dependent
footprint EXACTLY and solves the same no-harm-constrained placement to optimality.

Why the engine cost is exactly linearizable
--------------------------------------------
The engine charges, per covered (node, slot):
  * an IDLE term (idle_w -> facility energy -> carbon + direct water + indirect water +
    scarcity) ONLY for the first pod that activates an otherwise-empty node-slot
    (`total_cpu_ratio_before <= 0` gate in compute_footprint_vector), and
  * a POD-MARGINAL term (dynamic power proportional to the pod's CPU ratio, plus embodied
    carbon/water proportional to the pod's CPU/GPU share) for every placement, INDEPENDENT
    of occupancy.
We verified by differencing footprints that:
  footprint(slot empty)  ==  footprint(slot occupied)  +  idle_once(node, slot)
exactly, and the embodied/dynamic terms are occupancy-invariant. Therefore the engine's
total carbon / scarcity / weighted-grid-stress are EXACTLY:
  total = Σ_{(f,s) active} idle_cost(f,s)  +  Σ_{placements} pod_marginal_cost(p,f,t)
which is linear in activation binaries y[f,s] and placement binaries x[p,f,t]. We obtain
idle_cost and pod_marginal_cost by CALLING the engine footprint function (never re-deriving
the math): idle_cost(f,s) = footprint(empty) - footprint(occupied) for a probe pod; the
pod-marginal cost of placing pod p is footprint(occupied) i.e. with the slot already busy.

MILP
----
vars:  x[p,f,t] ∈ {0,1}   pod p placed on flavour f starting slot t (deadline-feasible only)
       y[f,s]   ∈ {0,1}   node f is active in slot s (carries the idle-once cost)
constraints:
  Σ_{f,t} x[p,f,t] = 1                                  (SLO: every baseline-placed pod placed)
  Σ_{p,(f,t) covering s} cpu_req·x ≤ totalCpu[f]        (capacity per node-slot, CPU)
  Σ_{p,(f,t) covering s} ram_req·x ≤ totalRam[f]        (capacity per node-slot, RAM)
  y[f,s] ≥ x[p,f,t]  for every placement covering (f,s) (activation forces idle)
  Σ idle·y + Σ pod_carbon·x ≤ baseline_carbon          (no-harm: carbon)
  Σ idle·y + Σ pod_scarcity·x ≤ baseline_scarcity       (no-harm: scarcity)
objectives (solved separately):
  (a) min total weighted grid-stress   (the engine's headline relief objective)
  (b) min total carbon                  (carbon-optimal certified schedule)
  (c) certified carbon-water Pareto frontier (min carbon s.t. scarcity ≤ ε, swept)
The greedy combined-objective repair is compared to (a) and (b); the certificate it produces
is checked against the certificate the solver respects.

Outputs land under experiments/optgap_<tag>/. Does not touch committed signals or the engine.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO / "pkg" / "carbon-aware" / "server-python"))
logging.disable(logging.CRITICAL)

import pulp  # noqa: E402

from carbon_aware.footprints import compute_footprint_vector  # noqa: E402
from carbon_aware.models import EnvironmentalFlavor  # noqa: E402
from carbon_aware.no_harm_flexibility import (  # noqa: E402
    PilotConfig,
    build_action_signals,
    build_greedy_schedule,
    classify_pod,
    load_flavours_for_pilot,
    load_pods,
    repair_schedule_no_harm,
    summarize_against_reference,
    _rematerialize_under_realized,
)
from carbon_aware.utils import build_timeslots, is_timeslot_valid  # noqa: E402

REALCI = REPO / "pkg" / "carbon-aware" / "data" / "timealigned_realci2"  # coherent revamp build


# ---------------------------------------------------------------------------
# Exact linear decomposition of the engine footprint (idle-once + pod-marginal)
# ---------------------------------------------------------------------------
def _idle_once_cost(flavour: EnvironmentalFlavor, start: int, dur: int, probe_pod) -> Dict[str, float]:
    """idle-once cost for activating `flavour` over [start, start+dur).
    = footprint(slot empty) - footprint(slot occupied), via the engine itself.
    Independent of which pod first activates the slot (idle is pod-independent)."""
    empty = compute_footprint_vector(
        flavour, start, probe_pod,
        used_cpu_before_by_slot={start + o: 0.0 for o in range(dur)},
    )
    occ = compute_footprint_vector(
        flavour, start, probe_pod,
        used_cpu_before_by_slot={start + o: flavour.totalCpu for o in range(dur)},
    )
    return {
        "carbon_kg": empty.total_carbon_kg - occ.total_carbon_kg,
        "scarcity": empty.scarcity_characterized_water - occ.scarcity_characterized_water,
        "op_energy": empty.operational_energy_kwh - occ.operational_energy_kwh,
    }


def _pod_marginal_cost(flavour: EnvironmentalFlavor, start: int, pod, dur: int) -> Dict[str, float]:
    """pod-marginal cost of placing `pod` on `flavour` at `start` when the slot is ALREADY
    occupied (idle not charged) — the dynamic + embodied + proportional-water surface that
    the engine attributes to this pod regardless of who carried the idle."""
    occ = compute_footprint_vector(
        flavour, start, pod,
        used_cpu_before_by_slot={start + o: flavour.totalCpu for o in range(dur)},
    )
    return {
        "carbon_kg": occ.total_carbon_kg,
        "scarcity": occ.scarcity_characterized_water,
        "op_energy": occ.operational_energy_kwh,
    }


def _weighted_stress(flavour, start: int, dur: int, op_energy_kwh: float, signals) -> float:
    """Residual-load-weighted grid stress charged to op_energy spread evenly over the span.
    Matches placement_signal_metrics' weighted_stress_kwh = Σ energy_per_slot·stress_score."""
    region = (getattr(flavour, "region", "") or "").upper()
    e_per_slot = op_energy_kwh / max(dur, 1)
    w = 0.0
    for o in range(dur):
        sig = signals.get((region, start + o))
        if sig is not None:
            w += e_per_slot * sig.grid_stress_score
    return w


def build_candidate_table(pods, flavours, signals, max_ts, timeslots):
    """For every deadline-feasible (pod, flavour, start), precompute the EXACT engine
    pod-marginal carbon/scarcity/op-energy + weighted stress. Also the per-(flavour,slot)
    idle-once carbon/scarcity/op-energy + the idle slot's weighted stress."""
    probe = pods[0]
    cand: Dict[str, List[dict]] = {}
    for p in pods:
        dur = max(int(p.duration), 1)
        lst = []
        for f in flavours:
            for ts in timeslots:
                if not is_timeslot_valid(ts, p):
                    continue
                if ts.id + dur > max_ts:
                    continue
                pm = _pod_marginal_cost(f, ts.id, p, dur)
                # pod-marginal op-energy = dynamic only (idle stripped); weighted stress on it
                ws_pod = _weighted_stress(f, ts.id, dur, pm["op_energy"], signals)
                lst.append({
                    "flavour": f, "t": ts.id, "dur": dur,
                    "carbon": pm["carbon_kg"], "scarcity": pm["scarcity"],
                    "op_energy": pm["op_energy"], "wstress": ws_pod,
                })
        cand[p.id] = lst
    # idle-once table per (flavour, slot): a single-slot probe activation
    idle: Dict[Tuple[str, int], dict] = {}
    for f in flavours:
        for s in range(max_ts):
            io = _idle_once_cost(f, s, 1, probe)
            ws_idle = _weighted_stress(f, s, 1, io["op_energy"], signals)
            idle[(f.id, s)] = {
                "carbon": io["carbon_kg"], "scarcity": io["scarcity"],
                "op_energy": io["op_energy"], "wstress": ws_idle,
            }
    return cand, idle


def engine_totals_from_assignment(choice: Dict[str, dict], idle_tab, flavours) -> Dict[str, float]:
    """Recompute exact engine totals (carbon/scarcity/weighted-stress) from an assignment
    dict pod_id -> candidate, using idle-once for activated (flavour,slot) + pod-marginals.
    This is the exact decomposition; used to score both greedy and MILP identically."""
    active: set = set()
    for c in choice.values():
        for o in range(c["dur"]):
            active.add((c["flavour"].id, c["t"] + o))
    carbon = sum(idle_tab[a]["carbon"] for a in active) + sum(c["carbon"] for c in choice.values())
    scarcity = sum(idle_tab[a]["scarcity"] for a in active) + sum(c["scarcity"] for c in choice.values())
    wstress = sum(idle_tab[a]["wstress"] for a in active) + sum(c["wstress"] for c in choice.values())
    return {"carbon": carbon, "scarcity": scarcity, "wstress": wstress, "n_active": len(active)}


# ---------------------------------------------------------------------------
# MILP
# ---------------------------------------------------------------------------
def solve_milp(pods, cand, idle_tab, flavours, max_ts, base_carbon, base_scarcity,
               objective: str, scarcity_cap: Optional[float] = None, carbon_cap: Optional[float] = None,
               time_limit: int = 120, pinned: Optional[Dict[str, Tuple[str, int]]] = None):
    """pinned: pod_id -> (flavour_id, start) that must be kept fixed (firm pods, to match the
    greedy's action space). When None, the MILP is free to re-place every pod (global optimum)."""
    prob = pulp.LpProblem("noharm_exact", pulp.LpMinimize)
    x = {}
    for p in pods:
        for i, c in enumerate(cand[p.id]):
            x[(p.id, i)] = pulp.LpVariable(f"x_{p.id}_{i}", cat="Binary")
    by_pod = {p.id: p for p in pods}
    # one placement per pod (SLO non-degradation; every baseline-placed pod stays placed)
    for p in pods:
        prob += pulp.lpSum(x[(p.id, i)] for i in range(len(cand[p.id]))) == 1
    # pin firm pods to their baseline placement (action-space-matched MILP)
    if pinned:
        for p in pods:
            tgt = pinned.get(p.id)
            if tgt is None:
                continue
            for i, c in enumerate(cand[p.id]):
                if not (c["flavour"].id == tgt[0] and c["t"] == tgt[1]):
                    prob += x[(p.id, i)] == 0
    # activation binaries
    y = {}
    fl_ids = [f.id for f in flavours]
    for fid in fl_ids:
        for s in range(max_ts):
            y[(fid, s)] = pulp.LpVariable(f"y_{fid}_{s}", cat="Binary")
    # link x -> y and accumulate capacity terms
    cover: Dict[Tuple[str, int], List[Tuple[str, int]]] = {}
    for p in pods:
        for i, c in enumerate(cand[p.id]):
            for o in range(c["dur"]):
                cover.setdefault((c["flavour"].id, c["t"] + o), []).append((p.id, i))
    fl_by_id = {f.id: f for f in flavours}
    for (fid, s), terms in cover.items():
        # activation: y >= each x (idle charged once when any pod present)
        for (pid, i) in terms:
            prob += y[(fid, s)] <= 1  # trivially bounded; real link below
        prob += pulp.lpSum(x[t] for t in terms) <= len(terms) * y[(fid, s)]
        # capacity
        prob += pulp.lpSum(by_pod[pid].cpuRequest * x[(pid, i)] for (pid, i) in terms) <= fl_by_id[fid].totalCpu
        prob += pulp.lpSum(by_pod[pid].ramRequest * x[(pid, i)] for (pid, i) in terms) <= fl_by_id[fid].totalRam

    def total_expr(field):
        pod_part = pulp.lpSum(cand[p.id][i][field] * x[(p.id, i)]
                              for p in pods for i in range(len(cand[p.id])))
        idle_part = pulp.lpSum(idle_tab[(fid, s)][field] * y[(fid, s)]
                               for fid in fl_ids for s in range(max_ts) if (fid, s) in idle_tab)
        return pod_part + idle_part

    carbon_expr = total_expr("carbon")
    scarcity_expr = total_expr("scarcity")
    wstress_expr = total_expr("wstress")

    # no-harm constraints (always enforced)
    eff_carbon_cap = carbon_cap if carbon_cap is not None else base_carbon
    eff_scarcity_cap = scarcity_cap if scarcity_cap is not None else base_scarcity
    prob += carbon_expr <= eff_carbon_cap + 1e-7
    prob += scarcity_expr <= eff_scarcity_cap + 1e-7

    if objective == "wstress":
        prob += wstress_expr
    elif objective == "carbon":
        prob += carbon_expr
    elif objective == "scarcity":
        prob += scarcity_expr
    else:
        raise ValueError(objective)

    t0 = time.perf_counter()
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
    elapsed = time.perf_counter() - t0
    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal",):
        return None, status, elapsed
    return {
        "carbon": pulp.value(carbon_expr),
        "scarcity": pulp.value(scarcity_expr),
        "wstress": pulp.value(wstress_expr),
        "obj": pulp.value(prob.objective),
    }, status, elapsed


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def make_config(n: int, out: Path, max_ts: int) -> PilotConfig:
    return PilotConfig(
        repo_root=REPO,
        nodes_file=REPO / "pkg/carbon-aware/nodes.yaml",
        workloads_dir=REPO / "pkg/carbon-aware/workloads",
        forecasts_file=REALCI / "forecasts.json",
        config_file=REPO / "pkg/carbon-aware/infra-workload-config.yaml",
        grid_signal_csv=REALCI / "grid_residual_region_slot.csv",
        wue_csv=REALCI / "wue_region_slot.csv",
        output_dir=out,
        max_pods=n, scenario="heatwave-drought", max_timeslots=max_ts,
    )


def run_instance(n: int, max_ts: int, out_root: Path, time_limit: int):
    out = out_root / f"n{n}"
    out.mkdir(parents=True, exist_ok=True)
    pc = make_config(n, out, max_ts)
    flavours = load_flavours_for_pilot(pc)
    pods_all = load_pods(pc.workloads_dir, max_pods=n)
    signals = build_action_signals(flavours, pc)
    timeslots = build_timeslots(max_ts)

    # baseline = packing; compare over the pods packing actually placed (SLO = match baseline set)
    packing, _, _ = build_greedy_schedule(method_key="packing", pods=pods_all, flavours=flavours, config=pc)
    pods = [pl.pod for pl in packing.placements]
    n_placed = len(pods)
    n_flex = sum(1 for p in pods if classify_pod(p, pc.flexibility_slack_hours) == "flexible")

    cand, idle_tab = build_candidate_table(pods, flavours, signals, max_ts, timeslots)

    # represent the packing baseline in the candidate table
    base_choice = {}
    ok = True
    for pl in packing.placements:
        m = next((c for c in cand[pl.pod.id]
                  if c["flavour"].id == pl.candidate.flavour.id and c["t"] == pl.candidate.timeslot.id), None)
        if m is None:
            ok = False
            break
        base_choice[pl.pod.id] = m
    if not ok:
        return {"n": n_placed, "skipped": "baseline not representable"}

    base = engine_totals_from_assignment(base_choice, idle_tab, flavours)

    # cross-check our exact decomposition against the engine's own re-materialized totals
    pk2, _, _ = build_greedy_schedule(method_key="packing", pods=pods_all, flavours=flavours, config=pc)
    _rematerialize_under_realized(pk2, flavours, pc)
    eng_base = summarize_against_reference(pk2, pk2, signals)
    decomp_err_carbon = abs(base["carbon"] - eng_base["carbon_kg"])
    decomp_err_scar = abs(base["scarcity"] - eng_base["scarcity_water"])

    # ----- GREEDY repairs, in EACH objective mode the engine supports -----
    # "stress"  = pure weighted-grid-stress relief under no-harm   (matches the wstress MILP)
    # "combined"= production default: rank by #certified axes improved, then stress
    # "search_control" = carbon/water-greedy ignoring stress
    def run_greedy(mode, *, relief_move_set=False):
        cfg = replace(pc, relief_move_set=True) if relief_move_set else pc
        t0 = time.perf_counter()
        rep = repair_schedule_no_harm(method_key=f"greedy_{mode}", baseline=packing, pods=pods_all,
                                      flavours=flavours, signals=signals, config=cfg, score_mode=mode)
        wall = time.perf_counter() - t0
        ch = {}
        for pl in rep.placements:
            if pl.pod.id not in cand:
                continue
            m = next((c for c in cand[pl.pod.id]
                      if c["flavour"].id == pl.candidate.flavour.id and c["t"] == pl.candidate.timeslot.id), None)
            if m is None:
                return None, wall, rep
            ch[pl.pod.id] = m
        if len(ch) != n_placed:
            return None, wall, rep
        return engine_totals_from_assignment(ch, idle_tab, flavours), wall, rep

    g_stress, wall_stress, rep_stress = run_greedy("stress")
    g_combined, wall_combined, rep_combined = run_greedy("combined")
    # B2: pure-stress greedy + guarded RELIEF move-set / LNS post-pass (default-off feature toggled on).
    g_lns, wall_lns, rep_lns = run_greedy("stress", relief_move_set=True)
    if g_stress is None or g_combined is None:
        return {"n": n_placed, "skipped": "greedy result not representable in candidate table"}

    # firm pods pinned to baseline = the greedy's exact action space (only flexible pods move)
    pinned = {}
    for pl in packing.placements:
        if classify_pod(pl.pod, pc.flexibility_slack_hours) != "flexible":
            pinned[pl.pod.id] = (pl.candidate.flavour.id, pl.candidate.timeslot.id)

    # ----- GLOBAL MILP optima (free to re-place every pod; subject to no-harm) -----
    milp_w, st_w, wall_w = solve_milp(pods, cand, idle_tab, flavours, max_ts, base["carbon"], base["scarcity"],
                                      objective="wstress", time_limit=time_limit)
    milp_c, st_c, wall_c = solve_milp(pods, cand, idle_tab, flavours, max_ts, base["carbon"], base["scarcity"],
                                      objective="carbon", time_limit=time_limit)
    milp_s, st_s, wall_s = solve_milp(pods, cand, idle_tab, flavours, max_ts, base["carbon"], base["scarcity"],
                                      objective="scarcity", time_limit=time_limit)
    # ----- ACTION-SPACE-MATCHED MILP optima (firm pods pinned, like the greedy) -----
    milp_wm, st_wm, wall_wm = solve_milp(pods, cand, idle_tab, flavours, max_ts, base["carbon"], base["scarcity"],
                                         objective="wstress", time_limit=time_limit, pinned=pinned)
    milp_cm, st_cm, wall_cm = solve_milp(pods, cand, idle_tab, flavours, max_ts, base["carbon"], base["scarcity"],
                                         objective="carbon", time_limit=time_limit, pinned=pinned)
    milp_sm, st_sm, wall_sm = solve_milp(pods, cand, idle_tab, flavours, max_ts, base["carbon"], base["scarcity"],
                                         objective="scarcity", time_limit=time_limit, pinned=pinned)

    def gap(g, m):
        return (g - m) / m * 100.0 if m and abs(m) > 1e-12 else 0.0

    def relief(b, v):
        return (b - v) / b * 100.0 if b else 0.0

    row = {
        "n_placed": n_placed, "n_flex": n_flex, "max_ts": max_ts,
        "n_xvars": sum(len(cand[p.id]) for p in pods), "n_active_idle": len(idle_tab),
        "decomp_err_carbon_kg": decomp_err_carbon, "decomp_err_scarcity": decomp_err_scar,
        "base_carbon": base["carbon"], "base_scarcity": base["scarcity"], "base_wstress": base["wstress"],
        # production combined-objective greedy
        "combined_carbon": g_combined["carbon"], "combined_scarcity": g_combined["scarcity"],
        "combined_wstress": g_combined["wstress"], "combined_repairs": rep_combined.repairs_applied,
        "combined_wall_s": wall_combined,
        # pure-stress greedy (objective-matched to the wstress MILP)
        "stress_carbon": g_stress["carbon"], "stress_scarcity": g_stress["scarcity"],
        "stress_wstress": g_stress["wstress"], "stress_repairs": rep_stress.repairs_applied,
        "stress_wall_s": wall_stress,
        # B2 relief move-set / LNS: pure-stress greedy + atomic move-set post-pass (only differs from
        # `stress` when the LNS commits a coordinated set the per-move greedy could not).
        "lns_carbon": (g_lns or {}).get("carbon"), "lns_scarcity": (g_lns or {}).get("scarcity"),
        "lns_wstress": (g_lns or {}).get("wstress"), "lns_repairs": rep_lns.repairs_applied,
        "lns_wall_s": wall_lns,
        "milp_wstress_status": st_w, "milp_wstress_wall_s": wall_w,
        "milp_carbon_status": st_c, "milp_carbon_wall_s": wall_c,
        "milp_scarcity_status": st_s, "milp_scarcity_wall_s": wall_s,
    }
    # ---- headline: pure-stress greedy vs wstress-optimal MILP (apples to apples) ----
    if milp_w:
        row["opt_wstress"] = milp_w["wstress"]
        row["gap_wstress_pct"] = gap(g_stress["wstress"], milp_w["wstress"])  # objective-matched gap
        row["stress_relief_pct"] = relief(base["wstress"], g_stress["wstress"])
        row["opt_relief_pct"] = relief(base["wstress"], milp_w["wstress"])
        # B2: LNS greedy vs the SAME wstress-optimal MILP -> does the move-set close the search gap?
        if g_lns is not None:
            row["gap_wstress_lns_pct"] = gap(g_lns["wstress"], milp_w["wstress"])
            row["lns_relief_pct"] = relief(base["wstress"], g_lns["wstress"])
            # fraction of the MILP-optimal relief captured, greedy vs greedy+LNS (stable near opt~0).
            opt_rel = relief(base["wstress"], milp_w["wstress"])
            row["stress_frac_of_opt_pct"] = (row["stress_relief_pct"] / opt_rel * 100.0) if opt_rel > 1e-9 else 0.0
            row["lns_frac_of_opt_pct"] = (row["lns_relief_pct"] / opt_rel * 100.0) if opt_rel > 1e-9 else 0.0
            row["lns_carbon_le_base"] = bool(g_lns["carbon"] <= base["carbon"] + 1e-7)
            row["lns_scarcity_le_base"] = bool(g_lns["scarcity"] <= base["scarcity"] + 1e-7)
        # combined-objective greedy's stress outcome (it trades stress for carbon/water co-benefits)
        row["combined_relief_pct"] = relief(base["wstress"], g_combined["wstress"])
        if wall_stress > 0:
            row["speedup_stress_vs_milp"] = wall_w / wall_stress
        # matched (firm pods pinned): the fair gap inside the greedy's own action space
        if milp_wm:
            row["opt_wstress_matched"] = milp_wm["wstress"]
            row["gap_wstress_matched_pct"] = gap(g_stress["wstress"], milp_wm["wstress"])
            row["opt_relief_matched_pct"] = relief(base["wstress"], milp_wm["wstress"])
    # ---- carbon: combined greedy (pursues carbon as a certified axis) vs carbon-optimal MILP ----
    if milp_c:
        row["opt_carbon"] = milp_c["carbon"]
        row["gap_carbon_pct"] = gap(g_combined["carbon"], milp_c["carbon"])
        row["combined_carbon_relief_pct"] = relief(base["carbon"], g_combined["carbon"])
        row["opt_carbon_relief_pct"] = relief(base["carbon"], milp_c["carbon"])
        if milp_cm:
            row["opt_carbon_matched"] = milp_cm["carbon"]
            row["gap_carbon_matched_pct"] = gap(g_combined["carbon"], milp_cm["carbon"])
            row["opt_carbon_relief_matched_pct"] = relief(base["carbon"], milp_cm["carbon"])
    # ---- scarcity: combined greedy vs scarcity-optimal MILP ----
    if milp_s:
        row["opt_scarcity"] = milp_s["scarcity"]
        row["gap_scarcity_pct"] = gap(g_combined["scarcity"], milp_s["scarcity"])
        row["combined_scarcity_relief_pct"] = relief(base["scarcity"], g_combined["scarcity"])
        row["opt_scarcity_relief_pct"] = relief(base["scarcity"], milp_s["scarcity"])
        if milp_sm:
            row["opt_scarcity_matched"] = milp_sm["scarcity"]
            row["gap_scarcity_matched_pct"] = gap(g_combined["scarcity"], milp_sm["scarcity"])
            row["opt_scarcity_relief_matched_pct"] = relief(base["scarcity"], milp_sm["scarcity"])
    # ---- sanity: both greedy variants respect the certificate the solver respects ----
    for tag, g in (("combined", g_combined), ("stress", g_stress)):
        row[f"{tag}_carbon_le_base"] = bool(g["carbon"] <= base["carbon"] + 1e-7)
        row[f"{tag}_scarcity_le_base"] = bool(g["scarcity"] <= base["scarcity"] + 1e-7)
    row["greedy_slo_ok"] = True

    (out / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="6,10,14,18,24,30")
    ap.add_argument("--max-ts", type=int, default=48)
    ap.add_argument("--time-limit", type=int, default=120)
    ap.add_argument("--tag", default="exact")
    args = ap.parse_args()
    out_root = REPO / "experiments" / f"optgap_{args.tag}"
    out_root.mkdir(parents=True, exist_ok=True)

    sizes = [int(s) for s in args.sizes.split(",")]
    rows = []
    print(f"# EXACT no-harm MILP optimality gap  (max_ts={args.max_ts}, time_limit={args.time_limit}s)")
    print(f"# decomp = |exact-decomposition total - engine re-materialized total| (must be ~0).")
    print(f"# GLOBAL gap = greedy vs MILP free to re-place EVERY pod (global certified optimum).")
    print(f"# MATCHED gap = greedy vs MILP with firm pods PINNED (the greedy's own action space).")
    print(f"# STRESS: greedy 'stress' mode; CARBON/SCAR: production 'combined' greedy.")
    print(f"#")
    print(f"# STRESS vs LNS: pure-stress greedy relief%, greedy+LNS relief%, MILP-optimal relief%, and")
    print(f"#   frac-of-optimum each captures (the B2 headline: does the move-set close the search gap?).")
    print(f"# {'n':>3} {'flx':>3} | {'g_rel%':>7} {'lns_rel%':>8} {'opt_rel%':>8} | "
          f"{'g/opt%':>7} {'lns/opt%':>8} | {'lns_mv':>6} {'cert':>5} {'dec':>7}")
    for n in sizes:
        r = run_instance(n, args.max_ts, out_root, args.time_limit)
        rows.append(r)
        if r.get("skipped"):
            print(f"# n={n}: SKIPPED ({r['skipped']})")
            continue
        cert = (r["combined_carbon_le_base"] and r["combined_scarcity_le_base"]
                and r["stress_carbon_le_base"] and r["stress_scarcity_le_base"] and r["greedy_slo_ok"]
                and r.get("lns_carbon_le_base", True) and r.get("lns_scarcity_le_base", True))
        g = lambda k: r.get(k, float('nan'))
        print(f"  {r['n_placed']:>3} {r['n_flex']:>3} | {g('stress_relief_pct'):>7.1f} {g('lns_relief_pct'):>8.1f} "
              f"{g('opt_relief_pct'):>8.1f} | {g('stress_frac_of_opt_pct'):>7.1f} {g('lns_frac_of_opt_pct'):>8.1f} | "
              f"{g('lns_repairs'):>6.0f} {str(cert):>5} {max(r['decomp_err_carbon_kg'],r['decomp_err_scarcity']):>7.0e}")

    with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as h:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(h, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (out_root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n# wrote {out_root}/summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
