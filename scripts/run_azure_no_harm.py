#!/usr/bin/env python3
"""T3 — run the no-harm pilot on a real Azure Packing 2020 trace window (CPU/RAM general cloud).

Canonical Azure runner. It rebuilds the window's workloads with the corrected energy/slack model
(imputed CPU utilization drives dynamic power; slack capped at 24 h) from the window's
``canonical_trace_workload.csv``, provisions a REALISTIC CPU fleet to a target utilization (no
synthetic generator, no oversized nodes), and reports the no-harm certificate, carbon/scarcity
deltas, grid-stress relief, and the firm/flexible share, beside the carbon-/water-greedy ceilings
and WaterWise. The fleet, signals, regions, and PilotConfig are shared with the Alibaba GPU testbed
(``azure_common``/``alibaba_common``), so only the workload class and binding resource differ.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import azure_common as ac  # noqa: E402
from carbon_aware.no_harm_flexibility import load_pods, run_no_harm_flexibility_pilot  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-window", default="experiments/real_traces/azure_packing_2020_d0_s42_200",
                    help="converted Azure window dir (must contain canonical_trace_workload.csv)")
    ap.add_argument("--target-util", type=float, default=0.55, help="target peak CPU utilization")
    ap.add_argument("--cpu-util", type=float, default=ac.conv.CPU_UTIL_MEAN,
                    help="imputed per-pod CPU utilization fraction (drives dynamic power)")
    ap.add_argument("--signals-dir", default=None, help="time-aligned signals dir (default heatwave)")
    ap.add_argument("--max-timeslots", type=int, default=48)
    ap.add_argument("--lever-mode", default="both")
    ap.add_argument("--scenario", default="heatwave-drought")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", default=None, help="default: t3_<source-window-basename>")
    args = ap.parse_args()
    import logging
    logging.disable(logging.CRITICAL)

    src = (ac.REPO_ROOT / args.source_window).resolve()
    canonical = src / "canonical_trace_workload.csv"
    if not canonical.exists():
        raise SystemExit(f"no canonical_trace_workload.csv under {src} (run convert_azure_packing_trace.py)")
    run_name = args.run_name or f"t3_{src.name}"
    run_dir = ac.REPO_ROOT / "experiments" / "flexibility" / run_name
    signals = (ac.REPO_ROOT / args.signals_dir).resolve() if args.signals_dir else ac.TIMEALIGNED

    # Rebuild workloads with the corrected energy/slack model, then provision the fleet to ~target util.
    stats = ac.build_window_from_canonical(canonical, run_dir, seed=args.seed, cpu_util=args.cpu_util)
    workloads = stats["workloads_dir"]
    peak = ac.workload_peak_cores(workloads)
    per_region = ac.provision_for_util(peak, args.target_util)
    nodes_file, n_nodes, cap = ac.build_fleet(run_dir / "fleet", per_region)
    achieved = 100 * peak / cap if cap else 0.0

    pc = ac.make_pilot_config(nodes_file, workloads, run_dir / "out", signals=signals,
                              max_timeslots=args.max_timeslots, scenario=args.scenario,
                              lever_mode=args.lever_mode)
    pods = load_pods(workloads)
    rows = {r["method_key"]: r for r in run_no_harm_flexibility_pilot(pc)["summary_rows"]}
    flex = rows["no_harm_flex"]
    n_flex, n_prot = flex["flexible_pods"], flex["protected_pods"]
    total = n_flex + n_prot

    print(f"\n=== T3 Azure Packing 2020 no-harm result ({src.name}) ===")
    print(f"pods={len(pods)}  placed={flex['placed_pods']:.0f} unplaced={flex['unplaced_pods']:.0f}  "
          f"fleet={n_nodes} CPU nodes ({per_region}/region x {ac.CPU_NODE['cpu_cores']} cores = {cap}) "
          f"-> peak util {achieved:.0f}% (target {100*args.target_util:.0f}%)")
    print(f"workload: imputed cpu_util_ratio={args.cpu_util}  signals={signals.name}  max_ts={args.max_timeslots}")
    print(f"flexible share (trace priority/evictability): {n_flex}/{total} = "
          f"{100*n_flex/max(total,1):.1f}%  (firm={n_prot})")
    for m, tag in [("carbon", "carbon-greedy (ceiling)"), ("water_scarcity", "water-greedy"),
                   ("waterwise", "WaterWise"), ("no_harm_search_control", "search-control"),
                   ("no_harm_flex", "NO-HARM ENVELOPE")]:
        r = rows.get(m)
        if not r:
            continue
        print(f"  {tag:24}: carbon%={r['carbon_delta_pct']:7.2f}  scar%={r['scarcity_delta_pct']:7.3f}  "
              f"cert={r['no_harm_certificate']!s:5}  stress_avoid={r.get('stress_kwh_avoided',0):6.3f}  "
              f"repairs={r.get('repairs_applied',0):.0f}")

    digest = {
        "window": src.name, "run_name": run_name, "nodes": n_nodes, "nodes_per_region": per_region,
        "total_cpu_cores": cap, "peak_concurrent_cores": round(peak, 1),
        "achieved_peak_util_pct": round(achieved, 1), "target_util_pct": round(100 * args.target_util, 1),
        "cpu_util_ratio_imputed": args.cpu_util, "signals": signals.name,
        "max_timeslots": args.max_timeslots, "pods_loaded": len(pods),
        "placed": flex["placed_pods"], "unplaced": flex["unplaced_pods"],
        "flexible_share_pct": round(100 * n_flex / max(total, 1), 2),
        "flexible_pods": int(n_flex), "protected_pods": int(n_prot),
        "no_harm_flex": {k: flex[k] for k in ("no_harm_certificate", "carbon_delta_pct",
                                              "scarcity_delta_pct", "stress_kwh_avoided",
                                              "repairs_applied", "rejected_moves")
                         if k in flex},
        "baselines": {m: {k: rows[m][k] for k in ("no_harm_certificate", "carbon_delta_pct",
                                                  "scarcity_delta_pct") if k in rows[m]}
                      for m in ("carbon", "water_scarcity", "waterwise", "no_harm_search_control")
                      if m in rows},
    }
    (run_dir / "out").mkdir(parents=True, exist_ok=True)
    (run_dir / "out" / "t3_digest.json").write_text(json.dumps(digest, indent=2))
    print(f"\noutput_dir={run_dir/'out'}\ncertificate={run_dir/'out'/'no_harm_certificate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
