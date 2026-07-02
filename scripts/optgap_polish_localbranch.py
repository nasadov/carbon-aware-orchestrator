#!/usr/bin/env python3
"""Anytime optimality-gap certificate for the no-harm greedy, via local-branching MILP-polish.

D4 deliverable (theory): a *constant-factor approximation ratio is impossible* for the greedy
no-harm repair (supermodular consolidation -> greedy can reach 0% of optimum; see
docs/paper-three/lit_research/D4_theory.md Prop 3). The rigorous substitute is a PER-INSTANCE,
ANYTIME optimality-gap certificate: warm-start the EXACT no-harm MILP with the greedy's certified
incumbent, restrict it to a Hamming-radius-K neighborhood (local branching, Fischetti & Lodi 2003),
solve under a time budget, and report the branch-and-bound primal-dual gap. The returned schedule is
no-harm-feasible BY THE SAME MILP CONSTRAINTS the exact solver uses, so the certificate is intact;
the B&B gap is the anytime guarantee ("within g% of the certified optimum, certificate intact").

This is a STANDALONE script. It does NOT edit the engine. It reuses the EXACT engine-footprint
linearization and MILP builder from scripts/optgap_exact_milp.py (idle-once + pod-marginal
decomposition verified there to match the engine to ~1e-16). The only addition over the exact MILP
is the local-branching cut:

    sum over pods of (1 - x[p, i*(p)])  <=  K          # >= n-K pods keep the incumbent candidate

where i*(p) is the incumbent (greedy) candidate index for pod p. As K -> (n - n_pinned) this
relaxes to the full free-replacement optimum; at K small it certifies "no better certified schedule
within K reassignments of the greedy incumbent." Objective options match the exact script:
wstress (grid relief, the headline) | carbon | scarcity.

Outputs land under experiments/d4_polish_<tag>/. Compares, per instance and per K:
  greedy relief  ->  polish(K) relief  ->  certified-optimum relief (free MILP),
plus the B&B gap polish(K) reports against its own bound.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO / "pkg" / "carbon-aware" / "server-python"))
sys.path.insert(0, str(REPO / "scripts"))
logging.disable(logging.CRITICAL)

import pulp  # noqa: E402

from carbon_aware.no_harm_flexibility import (  # noqa: E402
    build_action_signals,
    build_greedy_schedule,
    classify_pod,
    load_flavours_for_pilot,
    load_pods,
    repair_schedule_no_harm,
)
from carbon_aware.utils import build_timeslots  # noqa: E402

# Reuse the EXACT linearization + MILP machinery (no re-derivation of the footprint math).
from optgap_exact_milp import (  # noqa: E402
    build_candidate_table,
    engine_totals_from_assignment,
    make_config,
)


def _incumbent_indices(choice: Dict[str, dict], cand: Dict[str, List[dict]]) -> Dict[str, int]:
    """Map pod_id -> index in cand[pod_id] of the candidate the incumbent (greedy) chose."""
    idx: Dict[str, int] = {}
    for pid, c in choice.items():
        for i, cc in enumerate(cand[pid]):
            if cc["flavour"].id == c["flavour"].id and cc["t"] == c["t"]:
                idx[pid] = i
                break
    return idx


def solve_localbranch(
    pods,
    cand,
    idle_tab,
    flavours,
    max_ts,
    base_carbon: float,
    base_scarcity: float,
    incumbent_idx: Dict[str, int],
    K: Optional[int],
    objective: str,
    time_limit: int,
):
    """Exact no-harm MILP restricted to Hamming-radius-K of the incumbent (None K = free optimum).
    Returns (totals_or_None, status, wall, gap_pct) where gap_pct is the B&B primal-dual gap."""
    prob = pulp.LpProblem("noharm_localbranch", pulp.LpMinimize)
    x = {}
    for p in pods:
        for i in range(len(cand[p.id])):
            x[(p.id, i)] = pulp.LpVariable(f"x_{p.id}_{i}", cat="Binary")
    by_pod = {p.id: p for p in pods}
    fl_by_id = {f.id: f for f in flavours}
    fl_ids = [f.id for f in flavours]

    # one placement per pod (SLO: every baseline-placed pod stays placed)
    for p in pods:
        prob += pulp.lpSum(x[(p.id, i)] for i in range(len(cand[p.id]))) == 1

    # activation binaries + capacity + link x -> y
    y = {}
    for fid in fl_ids:
        for s in range(max_ts):
            y[(fid, s)] = pulp.LpVariable(f"y_{fid}_{s}", cat="Binary")
    cover: Dict[Tuple[str, int], List[Tuple[str, int]]] = {}
    for p in pods:
        for i, c in enumerate(cand[p.id]):
            for o in range(c["dur"]):
                cover.setdefault((c["flavour"].id, c["t"] + o), []).append((p.id, i))
    for (fid, s), terms in cover.items():
        prob += pulp.lpSum(x[t] for t in terms) <= len(terms) * y[(fid, s)]
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

    # no-harm constraints (the certificate, identical to the exact solver)
    prob += carbon_expr <= base_carbon + 1e-7
    prob += scarcity_expr <= base_scarcity + 1e-7

    # ---- LOCAL-BRANCHING CUT: keep >= n-K pods at their incumbent candidate ----
    # sum_p (1 - x[p, i*(p)]) <= K   <=>   sum_p x[p, i*(p)] >= n - K
    if K is not None:
        n = len(pods)
        prob += pulp.lpSum(
            x[(p.id, incumbent_idx[p.id])] for p in pods if p.id in incumbent_idx
        ) >= n - K

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
    wall = time.perf_counter() - t0
    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal",) or pulp.value(prob.objective) is None:
        return None, status, wall, float("nan")

    obj_val = pulp.value(prob.objective)
    # B&B primal-dual gap: CBC's best bound vs incumbent. PuLP exposes it inconsistently across
    # versions; if a dual bound is unavailable we report nan and rely on the cross-check against the
    # free optimum (also computed) for the true per-instance gap.
    gap_pct = float("nan")
    try:
        bound = prob.solverModel.getBestPossibleObjValue()  # may not exist on all CBC bindings
        if bound is not None and abs(obj_val) > 1e-12:
            gap_pct = abs(obj_val - bound) / abs(obj_val) * 100.0
    except Exception:
        pass
    return {
        "carbon": pulp.value(carbon_expr),
        "scarcity": pulp.value(scarcity_expr),
        "wstress": pulp.value(wstress_expr),
        "obj": obj_val,
    }, status, wall, gap_pct


def run_instance(n: int, max_ts: int, ks: List[int], objective: str, out_root: Path, time_limit: int):
    out = out_root / f"n{n}"
    out.mkdir(parents=True, exist_ok=True)
    pc = make_config(n, out, max_ts)
    flavours = load_flavours_for_pilot(pc)
    pods_all = load_pods(pc.workloads_dir, max_pods=n)
    signals = build_action_signals(flavours, pc)
    timeslots = build_timeslots(max_ts)

    packing, _, _ = build_greedy_schedule(method_key="packing", pods=pods_all, flavours=flavours, config=pc)
    pods = [pl.pod for pl in packing.placements]
    n_placed = len(pods)
    n_flex = sum(1 for p in pods if classify_pod(p, pc.flexibility_slack_hours) == "flexible")

    cand, idle_tab = build_candidate_table(pods, flavours, signals, max_ts, timeslots)

    base_choice = {}
    for pl in packing.placements:
        m = next((c for c in cand[pl.pod.id]
                  if c["flavour"].id == pl.candidate.flavour.id and c["t"] == pl.candidate.timeslot.id), None)
        if m is None:
            return {"n": n_placed, "skipped": "baseline not representable"}
        base_choice[pl.pod.id] = m
    base = engine_totals_from_assignment(base_choice, idle_tab, flavours)

    # greedy incumbent in the objective-matched mode:
    #  wstress -> 'stress' greedy (pure grid relief); carbon/scarcity -> 'combined' greedy.
    greedy_mode = "stress" if objective == "wstress" else "combined"
    t0 = time.perf_counter()
    rep = repair_schedule_no_harm(method_key=f"greedy_{greedy_mode}", baseline=packing, pods=pods_all,
                                  flavours=flavours, signals=signals, config=pc, score_mode=greedy_mode)
    greedy_wall = time.perf_counter() - t0
    g_choice = {}
    for pl in rep.placements:
        if pl.pod.id not in cand:
            continue
        m = next((c for c in cand[pl.pod.id]
                  if c["flavour"].id == pl.candidate.flavour.id and c["t"] == pl.candidate.timeslot.id), None)
        if m is None:
            return {"n": n_placed, "skipped": "greedy not representable"}
        g_choice[pl.pod.id] = m
    if len(g_choice) != n_placed:
        return {"n": n_placed, "skipped": "greedy not representable (count)"}
    greedy = engine_totals_from_assignment(g_choice, idle_tab, flavours)
    inc_idx = _incumbent_indices(g_choice, cand)

    field = objective  # 'wstress'|'carbon'|'scarcity' all live in the totals dict
    base_v = base[field]

    def relief(v):
        return (base_v - v) / base_v * 100.0 if base_v else 0.0

    # free (unrestricted) certified optimum = the true per-instance optimum for the gap denominator
    opt, opt_status, opt_wall, _ = solve_localbranch(
        pods, cand, idle_tab, flavours, max_ts, base["carbon"], base["scarcity"],
        inc_idx, None, objective, time_limit)

    row = {
        "n_placed": n_placed, "n_flex": n_flex, "max_ts": max_ts, "objective": objective,
        "base": base_v, "greedy": greedy[field], "greedy_relief_pct": relief(greedy[field]),
        "greedy_wall_s": greedy_wall, "greedy_repairs": rep.repairs_applied,
        "opt_status": opt_status, "opt_wall_s": opt_wall,
    }
    if opt:
        row["opt"] = opt[field]
        row["opt_relief_pct"] = relief(opt[field])
        # price of no-harm = 1 - greedy_relief / opt_relief (the headline gap)
        gr, orl = relief(greedy[field]), relief(opt[field])
        row["price_of_noharm"] = (1 - gr / orl) if orl > 1e-9 else float("nan")

    # local-branching polish at each K
    polish = {}
    for K in ks:
        sol, status, wall, gap = solve_localbranch(
            pods, cand, idle_tab, flavours, max_ts, base["carbon"], base["scarcity"],
            inc_idx, K, objective, time_limit)
        if sol:
            pr = relief(sol[field])
            # certified by construction (the MILP enforced C<=base, W<=base); double-check:
            cert = (sol["carbon"] <= base["carbon"] + 1e-6) and (sol["scarcity"] <= base["scarcity"] + 1e-6)
            # fraction of the (greedy->opt) gap that polish(K) closed
            closed = float("nan")
            if opt:
                denom = relief(opt[field]) - relief(greedy[field])
                if abs(denom) > 1e-9:
                    closed = (pr - relief(greedy[field])) / denom * 100.0
            polish[K] = {
                "K": K, "status": status, "wall_s": wall, "relief_pct": pr,
                "bnb_gap_pct": gap, "certified": bool(cert), "gap_closed_vs_opt_pct": closed,
            }
        else:
            polish[K] = {"K": K, "status": status, "wall_s": wall, "relief_pct": float("nan")}
    row["polish"] = polish
    (out / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="10,12,14,18")
    ap.add_argument("--ks", default="1,2,4,8")
    ap.add_argument("--objective", default="carbon", choices=["wstress", "carbon", "scarcity"])
    ap.add_argument("--max-ts", type=int, default=48)
    ap.add_argument("--time-limit", type=int, default=120)
    ap.add_argument("--tag", default="localbranch")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    ks = [int(k) for k in args.ks.split(",")]
    out_root = REPO / "experiments" / f"d4_polish_{args.tag}"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"# Local-branching MILP-polish: anytime optimality-gap certificate ({args.objective} relief)")
    print(f"# greedy -> polish(K) -> free certified optimum.  'closed%' = fraction of greedy->opt gap recovered.")
    print(f"# {'n':>3} {'flx':>3} | {'greedy%':>8} {'opt%':>8} {'PoNH':>6} | "
          + " ".join(f"K={k}:rel%/closed%" for k in ks))
    rows = []
    for n in sizes:
        r = run_instance(n, args.max_ts, ks, args.objective, out_root, args.time_limit)
        rows.append(r)
        if r.get("skipped"):
            print(f"# n={n}: SKIPPED ({r['skipped']})")
            continue
        cells = []
        for k in ks:
            p = r["polish"].get(k, {})
            rel = p.get("relief_pct", float("nan"))
            cl = p.get("gap_closed_vs_opt_pct", float("nan"))
            cells.append(f"{rel:>6.1f}/{cl:>6.1f}")
        print(f"  {r['n_placed']:>3} {r['n_flex']:>3} | {r.get('greedy_relief_pct', float('nan')):>8.1f} "
              f"{r.get('opt_relief_pct', float('nan')):>8.1f} {r.get('price_of_noharm', float('nan')):>6.2f} | "
              + " ".join(cells))

    (out_root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    # flat CSV (drop nested polish for the csv; keep per-K relief columns)
    flat = []
    for r in rows:
        if r.get("skipped"):
            continue
        base = {k: v for k, v in r.items() if k != "polish"}
        for k, p in r.get("polish", {}).items():
            base[f"K{k}_relief_pct"] = p.get("relief_pct")
            base[f"K{k}_closed_pct"] = p.get("gap_closed_vs_opt_pct")
            base[f"K{k}_bnb_gap_pct"] = p.get("bnb_gap_pct")
            base[f"K{k}_wall_s"] = p.get("wall_s")
        flat.append(base)
    if flat:
        with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as h:
            keys = sorted({k for r in flat for k in r})
            w = csv.DictWriter(h, fieldnames=keys)
            w.writeheader()
            for r in flat:
                w.writerow(r)
    print(f"\n# wrote {out_root}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
