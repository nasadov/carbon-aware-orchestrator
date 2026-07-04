#!/usr/bin/env python3
"""F3-T1 flagship replay: full multi-method no-harm pilot at TRACE-NATIVE fleet scale.

The 200-pod real-trace windows are the paper's variance machinery (seeded studies). This driver
runs the SAME pilot (packing baseline B, carbon-greedy, WaterWise, water-greedy, the no-harm
envelope, and the search_control member) at a much larger, oversubscription-matched Alibaba GPU
fleet on the divergent w1587 window with MEASURED-CI signals (timealigned_realci2). It is the
headline exhibit for F3-T1: the envelope still certifies (carbon<=B, scarcity<=B, per-basin, SLO)
where the green baselines harm, at ~1k-2k pods rather than 200.

Per the F2 verdict the LITERAL full ~9k-pod trace is a ~50 h/window run-once offline artifact
(the greedy repair is ~cubic; nodes scale with pods). This driver takes a nodes-per-region knob so
the same code produces a completable in-session flagship (npr=20 -> ~1k pods) or the offline
artifact (npr=185 -> full trace) unchanged; it times every phase and writes a rich digest.

Run (background):
  PYTHONPATH=pkg/carbon-aware/server-python nohup python scripts/run_flagship_replay.py \
      --nodes-per-region 20 --window-start 1587 --run-name flagship_w1587_npr20 &
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[0]
for p in (str(REPO / "pkg" / "carbon-aware" / "server-python"), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import logging
logging.disable(logging.CRITICAL)

import alibaba_common as ac  # noqa: E402
from carbon_aware.no_harm_flexibility import load_pods, run_no_harm_flexibility_pilot  # noqa: E402

SIGNALS = REPO / "pkg" / "carbon-aware" / "data" / "timealigned_realci2"

# Every field we want out of each method's summary row (missing keys tolerated).
ROW_KEYS = ("no_harm_certificate", "carbon_delta_pct", "scarcity_delta_pct", "stress_kwh_avoided",
            "repairs_applied", "placed_pods", "unplaced_pods", "flexible_pods", "protected_pods",
            "per_basin_certificate", "radiation_delta_pct", "cost_delta_pct")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes-per-region", type=int, default=20,
                    help="20 -> 80 nodes / ~1000 pods (in-session); 40 -> ~2000; 185 -> full trace (offline)")
    ap.add_argument("--window-start", type=float, default=1587.0)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--max-timeslots", type=int, default=48)
    ap.add_argument("--run-name", default="flagship")
    args = ap.parse_args()

    run_dir = REPO / "experiments" / "flagship_replay" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    target_pods = ac.pods_for_fleet(args.nodes_per_region)

    wall0 = time.perf_counter()
    t0 = time.perf_counter()
    stats = ac.build_window(run_dir, target_pods, args.seed, window_start_hours=args.window_start)
    t_window = time.perf_counter() - t0

    nodes_file, n_nodes, total_gpus = ac.build_fleet(run_dir / "fleet", args.nodes_per_region)
    pc = ac.make_pilot_config(nodes_file, stats["workloads_dir"], run_dir / "out",
                              signals=SIGNALS, max_timeslots=args.max_timeslots)
    pods = load_pods(stats["workloads_dir"])

    print(f"[flagship] window h{args.window_start:g}  target_pods={target_pods}  loaded={len(pods)}  "
          f"nodes={n_nodes} ({total_gpus} GPUs)  flexible={stats['n_flexible']} "
          f"({stats['flexible_pct']:.1f}%)  mean_gpu_util={stats['mean_gpu_util']:.3f}", flush=True)

    t0 = time.perf_counter()
    res = run_no_harm_flexibility_pilot(pc)
    t_pilot = time.perf_counter() - t0
    rows = {r["method_key"]: r for r in res["summary_rows"]}

    digest = {
        "window_start_h": args.window_start, "seed": args.seed, "nodes_per_region": args.nodes_per_region,
        "n_nodes": n_nodes, "total_gpus": total_gpus, "target_pods": target_pods, "loaded_pods": len(pods),
        "n_flexible": stats["n_flexible"], "flexible_pct": round(stats["flexible_pct"], 2),
        "mean_gpu_util": round(stats["mean_gpu_util"], 4), "gpu_demand": round(stats["gpu_demand"], 2),
        "t_window_s": round(t_window, 1), "t_pilot_s": round(t_pilot, 1),
        "t_wall_s": round(time.perf_counter() - wall0, 1),
        "methods": {mk: {k: r.get(k) for k in ROW_KEYS if k in r} for mk, r in rows.items()},
    }
    (run_dir / "flagship_digest.json").write_text(json.dumps(digest, indent=2))

    print(f"\n=== FLAGSHIP w{args.window_start:g} npr={args.nodes_per_region} "
          f"({len(pods)} pods / {n_nodes} nodes) ===", flush=True)
    for mk, r in rows.items():
        print(f"  {mk:>26}: cert={r.get('no_harm_certificate')}  "
              f"C%={r.get('carbon_delta_pct', float('nan')):.2f}  "
              f"W%={r.get('scarcity_delta_pct', float('nan')):.3f}  "
              f"relief={r.get('stress_kwh_avoided', float('nan')):.3f}kWh  "
              f"repairs={r.get('repairs_applied', '-')}", flush=True)
    print(f"\n[flagship] pilot {t_pilot:.0f}s  wall {digest['t_wall_s']:.0f}s  -> {run_dir/'flagship_digest.json'}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
