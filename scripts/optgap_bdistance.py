#!/usr/bin/env python3
"""MC6 (TSUSC R2): how far does the baseline B sit from the per-axis optima?

Reviewer MC6: "no harm" means "no worse than B = lifecycle-carbon packing". If B already packs onto the
cleanest feasible nodes, "non-degradation vs B" is a STRINGENT bar (which would EXPLAIN the near-neutral
real-trace carbon and make it non-vacuous); if B is weak, the guarantee is cheap. The reviewer asks: on
the MILP-tractable instances, where does B sit relative to the carbon optimum and the water optimum?

This reuses the engine-faithful exact MILP (scripts/optgap_exact_milp.py) but solves the UNCONSTRAINED
per-axis optima (no no-harm caps -> pass huge caps), so we can measure how far B's carbon sits ABOVE the
minimum achievable carbon, and B's water above the minimum achievable water. We report both the
action-space-matched optimum (firm pods pinned, the scheduler's actual reach) and the global optimum
(every pod free). Additive: does not change optgap_exact_milp.py or any existing output.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(REPO / "pkg" / "carbon-aware" / "server-python"))
sys.path.insert(0, str(SCRIPT_DIR))

import optgap_exact_milp as ox  # noqa: E402
from carbon_aware.no_harm_flexibility import (  # noqa: E402
    build_action_signals, build_greedy_schedule, classify_pod,
    load_flavours_for_pilot, load_pods,
)
from carbon_aware.utils import build_timeslots  # noqa: E402

HUGE = 1e18  # disables the no-harm cap inside solve_milp -> true per-axis optimum


def _gap_above(b: float, opt: float):
    return (b - opt) / opt * 100.0 if opt and abs(opt) > 1e-12 else None


def bdistance_instance(n: int, max_ts: int, out_root: Path, time_limit: int = 120):
    out = out_root / f"n{n}"
    out.mkdir(parents=True, exist_ok=True)
    pc = ox.make_config(n, out, max_ts)
    flavours = load_flavours_for_pilot(pc)
    pods_all = load_pods(pc.workloads_dir, max_pods=n)
    signals = build_action_signals(flavours, pc)
    timeslots = build_timeslots(max_ts)

    packing, _, _ = build_greedy_schedule(method_key="packing", pods=pods_all, flavours=flavours, config=pc)
    pods = [pl.pod for pl in packing.placements]
    n_placed = len(pods)
    n_flex = sum(1 for p in pods if classify_pod(p, pc.flexibility_slack_hours) == "flexible")
    cand, idle_tab = ox.build_candidate_table(pods, flavours, signals, max_ts, timeslots)

    base_choice = {}
    for pl in packing.placements:
        m = next((c for c in cand[pl.pod.id]
                  if c["flavour"].id == pl.candidate.flavour.id and c["t"] == pl.candidate.timeslot.id), None)
        if m is None:
            return {"n": n_placed, "skipped": "baseline not representable"}
        base_choice[pl.pod.id] = m
    base = ox.engine_totals_from_assignment(base_choice, idle_tab, flavours)

    # firm pods pinned to baseline = the scheduler's actual action space (only flexible pods move)
    pinned = {}
    for pl in packing.placements:
        if classify_pod(pl.pod, pc.flexibility_slack_hours) != "flexible":
            pinned[pl.pod.id] = (pl.candidate.flavour.id, pl.candidate.timeslot.id)

    def opt(objective, pin, constrained=False):
        # constrained=False -> unconstrained per-axis optimum (degenerate single-axis corner);
        # constrained=True  -> CERTIFIED optimum: min this axis s.t. BOTH axes stay <= B (no-harm) ->
        # the certified headroom the footprint member can actually capture without harm.
        caps = {} if constrained else dict(carbon_cap=HUGE, scarcity_cap=HUGE)
        r, st, _w = ox.solve_milp(
            pods, cand, idle_tab, flavours, max_ts, base["carbon"], base["scarcity"],
            objective=objective, time_limit=time_limit, pinned=pin, **caps,
        )
        return r, st

    c_act, st_c_act = opt("carbon", pinned)
    s_act, st_s_act = opt("scarcity", pinned)
    c_glob, st_c_glob = opt("carbon", None)
    s_glob, st_s_glob = opt("scarcity", None)
    c_cert, st_c_cert = opt("carbon", pinned, constrained=True)   # certified carbon headroom (water<=B)
    s_cert, st_s_cert = opt("scarcity", pinned, constrained=True)  # certified water headroom (carbon<=B)

    row = {"n_placed": n_placed, "n_flex": n_flex, "max_ts": max_ts,
           "base_carbon": base["carbon"], "base_scarcity": base["scarcity"],
           "milp_status": {"carbon_action": st_c_act, "water_action": st_s_act,
                           "carbon_global": st_c_glob, "water_global": st_s_glob}}
    if c_act:
        row["carbon_opt_action"] = c_act["carbon"]
        row["B_above_carbonopt_action_pct"] = _gap_above(base["carbon"], c_act["carbon"])
    if s_act:
        row["water_opt_action"] = s_act["scarcity"]
        row["B_above_wateropt_action_pct"] = _gap_above(base["scarcity"], s_act["scarcity"])
    if c_glob:
        row["carbon_opt_global"] = c_glob["carbon"]
        row["B_above_carbonopt_global_pct"] = _gap_above(base["carbon"], c_glob["carbon"])
    if s_glob:
        row["water_opt_global"] = s_glob["scarcity"]
        row["B_above_wateropt_global_pct"] = _gap_above(base["scarcity"], s_glob["scarcity"])
    if c_cert:
        row["carbon_opt_certified"] = c_cert["carbon"]
        row["B_above_carbonopt_certified_pct"] = _gap_above(base["carbon"], c_cert["carbon"])
    if s_cert:
        row["water_opt_certified"] = s_cert["scarcity"]
        row["B_above_wateropt_certified_pct"] = _gap_above(base["scarcity"], s_cert["scarcity"])
    return row


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="6,10,14,18,24")
    ap.add_argument("--max-ts", type=int, default=48)
    ap.add_argument("--time-limit", type=int, default=120)
    args = ap.parse_args()
    import logging
    logging.disable(logging.CRITICAL)

    out_root = REPO / "experiments" / "mc6_bdistance"
    out_root.mkdir(parents=True, exist_ok=True)
    sizes = [int(s) for s in args.sizes.split(",")]
    rows = []
    print(f"# MC6 B-distance to per-axis optima (max_ts={args.max_ts})")
    print(f"{'n':>4} {'B_carbon':>9} {'C_opt':>9} {'B>Copt%':>8} {'B_scar':>9} {'W_opt':>9} {'B>Wopt%':>8}")
    for n in sizes:
        try:
            r = bdistance_instance(n, args.max_ts, out_root, args.time_limit)
        except Exception as e:  # noqa: BLE001
            r = {"n": n, "error": repr(e)}
        rows.append(r)
        if "B_above_carbonopt_action_pct" in r:
            print(f"{r['n_placed']:>4} | unconstrained: B>Copt {r['B_above_carbonopt_action_pct']:>7.1f}%  "
                  f"B>Wopt {r['B_above_wateropt_action_pct']:>7.1f}%  | CERTIFIED: B>Copt "
                  f"{r.get('B_above_carbonopt_certified_pct', float('nan')):>6.2f}%  B>Wopt "
                  f"{r.get('B_above_wateropt_certified_pct', float('nan')):>6.2f}%")
        else:
            print(f"{n:>4}  {r.get('skipped') or r.get('error')}")

    def med(key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return statistics.median(vals) if vals else None

    summary = {
        "median_B_above_carbonopt_action_pct": med("B_above_carbonopt_action_pct"),
        "median_B_above_wateropt_action_pct": med("B_above_wateropt_action_pct"),
        "median_B_above_carbonopt_certified_pct": med("B_above_carbonopt_certified_pct"),
        "median_B_above_wateropt_certified_pct": med("B_above_wateropt_certified_pct"),
        "median_B_above_carbonopt_global_pct": med("B_above_carbonopt_global_pct"),
        "median_B_above_wateropt_global_pct": med("B_above_wateropt_global_pct"),
        "n_instances": len(rows),
    }
    (out_root / "bdistance.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
    print(f"\nMEDIANS: unconstrained B>carbon-opt={summary['median_B_above_carbonopt_action_pct']:.0f}% "
          f"B>water-opt={summary['median_B_above_wateropt_action_pct']:.0f}%  |  "
          f"CERTIFIED B>carbon-opt={summary['median_B_above_carbonopt_certified_pct']}% "
          f"B>water-opt={summary['median_B_above_wateropt_certified_pct']}%")
    print(f"output={out_root/'bdistance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
