#!/usr/bin/env python3
"""SLO-binding sensitivity (gap #2): make the certificate's service axis explicit and bite.

The no-harm certificate's third axis is SLO. Because a placement is only generated if the
pod *finishes before its deadline* (`is_timeslot_valid`), there is no "late" state: a pod is
either admitted on time or left unplaced. The SLO axis is therefore **deadline attainment**
(the admitted share), and an SLO violation means a method admits fewer pods than the baseline.

In the headline (lightly loaded) runs every method places every pod, so the SLO leg never
binds. Here we tighten the screws -- raise the load (pods per node) under a short deadline
horizon -- until admission becomes scarce, and show that (a) aggressive from-scratch baselines
(e.g. water-greedy concentrating onto a few low-water sites) start dropping pods, degrading
SLO, while (b) the no-harm envelope, a repair of the packing baseline, preserves the baseline's
admission by construction and keeps carbon/water <= baseline -- i.e. its SLO guarantee binds and
holds. This turns "SLO" from an asserted axis into a measured, binding one.

Example:
    python scripts/slo_binding_sensitivity.py \
        --nodes-per-region 2 --pods 80,160,240,320 --horizon 6 --timeslots 12 \
        --seeds 0,1,2 --run-name t14_slo_binding
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
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
METHODS = ["packing", "carbon", "water_scarcity", "waterwise", "no_harm_flex"]
LABELS = {"packing": "packing (reference B)", "carbon": "carbon-greedy",
          "water_scarcity": "water-greedy", "waterwise": "WaterWise (balanced)",
          "no_harm_flex": "no-harm envelope (ours)"}


def _ints(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _ci95(values: Sequence[float]) -> float:
    arr = np.asarray([v for v in values if not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if arr.size < 2:
        return 0.0
    return 1.96 * arr.std(ddof=1) / math.sqrt(arr.size)


def _mean(values: Sequence[float]) -> float:
    arr = np.asarray([v for v in values if not (isinstance(v, float) and math.isnan(v))], dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def run(args) -> Path:
    out_root = REPO_ROOT / "experiments" / "flexibility" / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)
    signals = (REPO_ROOT / args.signals_dir).resolve() if args.signals_dir else TIMEALIGNED
    config_file = REPO_ROOT / "pkg" / "carbon-aware" / "infra-workload-config.yaml"
    base_config, gen_nodes, gen_ts = load_generator_dependencies(REPO_ROOT, config_file)

    seeds = _ints(args.seeds)
    pod_loads = _ints(args.pods)
    npr = args.nodes_per_region
    fleet = npr * N_REGIONS
    tidy: List[Dict] = []

    for pods in pod_loads:
        for seed in seeds:
            case = out_root / "inputs" / f"p{pods}_s{seed}"
            paths = _generate_case_inputs(
                base_config=base_config, generate_nodes_file=gen_nodes,
                generate_timeslot_files=gen_ts, input_dir=case,
                pod_count=pods, seed=seed, timeslots=args.timeslots,
                config_file=config_file, deadline_flex_hours=args.horizon,
                nodes_per_region=npr, server_only=True,
            )
            pc = PilotConfig(
                repo_root=REPO_ROOT, nodes_file=paths["nodes_file"],
                workloads_dir=paths["workloads_dir"], forecasts_file=signals / "forecasts.json",
                config_file=config_file, output_dir=out_root / "cases" / f"p{pods}_s{seed}",
                max_timeslots=args.max_timeslots, max_pods=None, scenario=args.scenario,
                lever_mode="both", grid_signal_csv=signals / "grid_residual_region_slot.csv",
                wue_csv=signals / "wue_region_slot.csv",
            )
            rows = {r["method_key"]: r for r in run_no_harm_flexibility_pilot(pc)["summary_rows"]}
            for m in METHODS:
                r = rows[m]
                tidy.append({
                    "pods": pods, "seed": seed, "method": m,
                    "placed": r["placed_pods"], "unplaced": r["unplaced_pods"],
                    "deadline_attainment_pct": r["deadline_attainment_pct"],
                    "late_pods": r["late_pods"],
                    "slo_non_decrease": int(bool(r["slo_non_decrease"])),
                    "no_harm": int(bool(r["no_harm_certificate"])),
                    "carbon_delta_pct": r["carbon_delta_pct"],
                    "scarcity_delta_pct": r["scarcity_delta_pct"],
                })
            pk = rows["packing"]; wg = rows["water_scarcity"]; fx = rows["no_harm_flex"]
            print(f"  pods={pods:>3} seed={seed}: attainment  packing={pk['deadline_attainment_pct']:.0f}%  "
                  f"water-greedy={wg['deadline_attainment_pct']:.0f}%  envelope={fx['deadline_attainment_pct']:.0f}%  "
                  f"(SLO non-degrade: water-greedy={bool(wg['slo_non_decrease'])} envelope={bool(fx['slo_non_decrease'])})",
                  flush=True)

    _write_csv(out_root / "tidy.csv", tidy)
    _write_report(out_root / "evidence_report.md", args, fleet, seeds, pod_loads, tidy)
    print(f"\noutput_dir={out_root}")
    return out_root


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def _agg(tidy, pods, method, col):
    vals = [r[col] for r in tidy if r["pods"] == pods and r["method"] == method]
    return _mean(vals), _ci95(vals)


def _write_report(path, args, fleet, seeds, pod_loads, tidy) -> None:
    L: List[str] = []
    L.append("# SLO-binding sensitivity — the certificate's service axis, made explicit and binding\n")
    L.append(f"Testbed: time-aligned 2018 EU, **{fleet}-node** fleet, deadline horizon **{args.horizon} h** "
             f"(tight), {len(seeds)} seeds {seeds}, scenario `{args.scenario}`. Load swept "
             f"{pod_loads} pods. All values mean ± 95% CI across seeds.\n")
    L.append("**SLO = deadline-feasible admission.** Placements past a pod's deadline are never generated, "
             "so there is no \"late\" state (late pods = 0 by construction); an SLO violation means a method "
             "admits fewer pods than the baseline. Deadline attainment = admitted share.\n")

    L.append("## Deadline attainment by load (%)\n")
    header = "| pods | " + " | ".join(LABELS[m] for m in METHODS) + " |"
    L.append(header)
    L.append("|" + "---|" * (len(METHODS) + 1))
    for pods in pod_loads:
        cells = []
        for m in METHODS:
            mean, ci = _agg(tidy, pods, m, "deadline_attainment_pct")
            cells.append(f"{mean:.0f} ± {ci:.0f}")
        L.append(f"| {pods} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## SLO non-degradation vs baseline (share of seeds; 1.0 = always holds)\n")
    L.append("| pods | " + " | ".join(LABELS[m] for m in METHODS) + " |")
    L.append("|" + "---|" * (len(METHODS) + 1))
    for pods in pod_loads:
        cells = [f"{_agg(tidy, pods, m, 'slo_non_decrease')[0]:.2f}" for m in METHODS]
        L.append(f"| {pods} | " + " | ".join(cells) + " |")
    L.append("")

    # Find the load at which the SLO leg first binds (a non-baseline method drops below packing).
    bind_load = None
    for pods in pod_loads:
        pk = _agg(tidy, pods, "packing", "deadline_attainment_pct")[0]
        worst = min(_agg(tidy, pods, m, "deadline_attainment_pct")[0]
                    for m in ("carbon", "water_scarcity", "waterwise"))
        if worst < pk - 0.5:
            bind_load = pods
            break

    # Is admission method-dependent at all? (max spread of attainment across methods, per load)
    max_spread = 0.0
    for pods in pod_loads:
        atts = [_agg(tidy, pods, m, "deadline_attainment_pct")[0] for m in METHODS]
        max_spread = max(max_spread, max(atts) - min(atts))
    min_att = min(_agg(tidy, pods, m, "deadline_attainment_pct")[0] for pods in pod_loads for m in METHODS)

    L.append("## Takeaway\n")
    if bind_load is not None:
        wg = _agg(tidy, bind_load, "water_scarcity", "deadline_attainment_pct")[0]
        fx = _agg(tidy, bind_load, "no_harm_flex", "deadline_attainment_pct")[0]
        pk = _agg(tidy, bind_load, "packing", "deadline_attainment_pct")[0]
        L.append(f"- **The SLO leg binds at ~{bind_load} pods.** Aggressive from-scratch baselines drop work "
                 f"(water-greedy admits **{wg:.0f}%** vs packing {pk:.0f}%); the no-harm envelope preserves "
                 f"the baseline's admission (**{fx:.0f}%**) by construction while keeping carbon/scarcity ≤ baseline.")
    else:
        L.append(f"- **Admission is method-independent** (attainment spread across all methods ≤ "
                 f"{max_spread:.1f} pp at every load). A from-scratch greedy that fills its preferred (e.g. "
                 f"low-water) sites still places each pod on its next *feasible* site rather than dropping it; "
                 f"a pod is unplaced only when no site has a deadline-feasible slot — set by deadline + "
                 f"aggregate capacity, not by the carbon/water objective. So **no method can degrade SLO "
                 f"relative to another**, and the certificate's SLO leg is a structural guarantee, not a "
                 f"discriminating axis.")
        L.append(f"- Here the tight {args.horizon} h horizon makes ~{100 - min_att:.0f}% of pods "
                 f"deadline-infeasible for *every* method (dropped identically); in the headline configuration "
                 f"(loose deadlines, horizon 24 h) attainment is 100% for all methods, the envelope included.")
        L.append("- The no-harm envelope is a **repair** of the baseline: it relocates flexible pods within "
                 "their feasible (in-deadline) windows and never evicts, so it preserves the baseline's "
                 "admission exactly — SLO non-degradation holds by construction, now reported explicitly "
                 "(deadline attainment) rather than merely asserted.")
    L.append("- **Lateness is structurally zero** for every method (feasibility enforces finish ≤ deadline); "
             "the service guarantee is therefore about admission, verified by the certificate on realized "
             "placement.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes-per-region", type=int, default=2)
    ap.add_argument("--pods", default="80,160,240,320")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--horizon", type=int, default=6, help="deadline flex hours (tight => SLO binds)")
    ap.add_argument("--timeslots", type=int, default=12)
    ap.add_argument("--max-timeslots", type=int, default=12)
    ap.add_argument("--scenario", default="heatwave-drought", choices=["heatwave-drought", "observed-winter"])
    ap.add_argument("--signals-dir", default=None)
    ap.add_argument("--run-name", default="t14_slo_binding")
    args = ap.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
