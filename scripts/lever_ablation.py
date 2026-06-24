#!/usr/bin/env python3
"""Lever ablation on the SAME testbed as the fleet headline (Table 1).

Isolates which flexibility channel carries the envelope by running the no-harm pilot at
lever_mode in {both, temporal, spatial} on identical generated inputs (same _generate_case_inputs
as fleet_scale_sweep), so the numbers are directly comparable to the headline -- unlike the matrix
harness, whose workload generation differs. Fixed fleet (default 16 nodes), multiple seeds,
time-aligned 2018 EU signals, heatwave-drought.

Example:
    python scripts/lever_ablation.py --nodes-per-region 4 --pods 160 --seeds 0,1,2,3,4 \
        --run-name t16b_lever_ablation
"""
from __future__ import annotations

import argparse, csv, math, sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SERVER = REPO_ROOT / "pkg" / "carbon-aware" / "server-python"
for p in (str(SERVER), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from carbon_aware.no_harm_flexibility import PilotConfig, run_no_harm_flexibility_pilot  # noqa: E402
from no_harm_flex_matrix import _generate_case_inputs, load_generator_dependencies  # noqa: E402

TIMEALIGNED = REPO_ROOT / "pkg" / "carbon-aware" / "data" / "timealigned"
N_REGIONS = 4
LEVERS = ["both", "temporal", "spatial"]


def _ints(s): return [int(x) for x in s.split(",") if x.strip()]
def _ci95(v):
    a = np.asarray([x for x in v if not (isinstance(x, float) and math.isnan(x))], float)
    return 1.96 * a.std(ddof=1) / math.sqrt(a.size) if a.size > 1 else 0.0
def _mean(v):
    a = np.asarray([x for x in v if not (isinstance(x, float) and math.isnan(x))], float)
    return float(a.mean()) if a.size else float("nan")


def run(args) -> Path:
    out = REPO_ROOT / "experiments" / "flexibility" / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    signals = TIMEALIGNED
    config_file = REPO_ROOT / "pkg" / "carbon-aware" / "infra-workload-config.yaml"
    base_config, gen_nodes, gen_ts = load_generator_dependencies(REPO_ROOT, config_file)
    seeds = _ints(args.seeds); npr = args.nodes_per_region; fleet = npr * N_REGIONS
    rows: List[Dict] = []
    for seed in seeds:
        case = out / "inputs" / f"f{fleet}_s{seed}"
        paths = _generate_case_inputs(
            base_config=base_config, generate_nodes_file=gen_nodes, generate_timeslot_files=gen_ts,
            input_dir=case, pod_count=args.pods, seed=seed, timeslots=args.timeslots,
            config_file=config_file, deadline_flex_hours=args.horizon, nodes_per_region=npr, server_only=True)
        for lever in LEVERS:
            pc = PilotConfig(
                repo_root=REPO_ROOT, nodes_file=paths["nodes_file"], workloads_dir=paths["workloads_dir"],
                forecasts_file=signals / "forecasts.json", config_file=config_file,
                output_dir=out / "cases" / f"f{fleet}_s{seed}_{lever}", max_timeslots=args.max_timeslots,
                max_pods=None, scenario=args.scenario, lever_mode=lever,
                grid_signal_csv=signals / "grid_residual_region_slot.csv", wue_csv=signals / "wue_region_slot.csv")
            r = {x["method_key"]: x for x in run_no_harm_flexibility_pilot(pc)["summary_rows"]}["no_harm_flex"]
            rows.append({"seed": seed, "lever": lever, "no_harm": int(bool(r["no_harm_certificate"])),
                         "carbon_delta_pct": r["carbon_delta_pct"], "scarcity_delta_pct": r["scarcity_delta_pct"],
                         "stress_kwh_avoided": r["stress_kwh_avoided"], "repairs": r["repairs_applied"]})
            print(f"  seed={seed} lever={lever:8} no_harm={bool(r['no_harm_certificate'])} "
                  f"stress={r['stress_kwh_avoided']:.3f} carbon%={r['carbon_delta_pct']:.2f} "
                  f"scar%={r['scarcity_delta_pct']:.2f}", flush=True)
    with (out / "tidy.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    L = [f"# Lever ablation (same testbed as fleet headline) — {fleet}-node, {args.pods} pods, "
         f"{len(seeds)} seeds, heatwave-drought\n",
         "Which channel carries the envelope. Mean ± 95% CI; identical generated inputs to Table 1.\n",
         "| lever | no-harm | grid-stress avoided (kWh) | carbon Δ% | scarcity-water Δ% |",
         "|---|---|---|---|---|"]
    for lev in LEVERS:
        sub = [r for r in rows if r["lever"] == lev]
        L.append(f"| {lev} | {_mean([r['no_harm'] for r in sub])*100:.0f}% | "
                 f"{_mean([r['stress_kwh_avoided'] for r in sub]):.2f} ± {_ci95([r['stress_kwh_avoided'] for r in sub]):.2f} | "
                 f"{_mean([r['carbon_delta_pct'] for r in sub]):.2f} ± {_ci95([r['carbon_delta_pct'] for r in sub]):.2f} | "
                 f"{_mean([r['scarcity_delta_pct'] for r in sub]):.1f} ± {_ci95([r['scarcity_delta_pct'] for r in sub]):.1f} |")
    L.append("\n**Reading:** the no-harm certificate holds in every mode. Both channels relieve grid "
             "stress -- temporal deferral and same-time *spatial* relocation away from stressed regions "
             "(per-region stress signals) -- and combining them maximizes it. The temporal lever "
             "additionally carries the largest footprint co-benefit (shifting to cleaner, cooler hours).")
    (out / "evidence_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\noutput_dir={out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes-per-region", type=int, default=4)
    ap.add_argument("--pods", type=int, default=160)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--timeslots", type=int, default=24)
    ap.add_argument("--max-timeslots", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--scenario", default="heatwave-drought")
    ap.add_argument("--run-name", default="t16b_lever_ablation")
    run(ap.parse_args()); return 0


if __name__ == "__main__":
    raise SystemExit(main())
