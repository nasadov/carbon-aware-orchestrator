#!/usr/bin/env python3
"""Build the merged n=10 matrix root (seeds 42-51) with a UNIFORM common-pod cohort.

The original matrix run included nine schedulers, so its per-cell
`all_algorithms_common` cohort intersects nine placement sets; the seed-extension
run includes only the six schedulers reported in the paper. To make the
common-pod metric identical across all ten seeds, this script recomputes the
cohort per cell as the intersection of the SIX reported schedulers' placements,
re-attributes emissions with the shared model, and writes a merged experiment
root with the same file names and schemas the composite figure generator
consumes:

  baseline_comparison_summary.csv               (per-seed whole-schedule detail)
  baseline_comparison_aggregate_by_pods.csv     (whole-schedule aggregate)
  attributed_common_pod_comparison.csv          (per-seed uniform common detail)
  attributed_common_all_algorithms_aggregate.csv (uniform common aggregate)
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
from attributed_common_pod_comparison import (  # noqa: E402
    attribute_pod_emissions, find_placement_csv, load_forecasts, load_nodes,
)

OLD_ROOT = REPO / "experiments/resubmission_baseline_matrix_sweep_v5/matrix_20260515_111056"
NEW_ROOT = Path(sorted(glob.glob(str(REPO / "experiments/resubmission_baseline_matrix_seedext/matrix_*")))[-1])
OUT_ROOT = REPO / "experiments/resubmission_baseline_matrix_merged_n10"

SCHED_GLOB = {
    "Vanilla-MostAllocated": "vanilla_most_allocated_*",
    "GREEN-MLFQ-K8s": "green-mlfq_*",
    "GreenCourier-Spatial-K8s": "greencourier-spatial_*",
    "Caspian-style": "caspian-operational_*",
    "TotEm-OpOnly": "heuristic_op_*",
    "TotEm": "heuristic_proportional_*",
}
PODS = (40, 80, 120, 160, 200)
CELLS = [(OLD_ROOT, s) for s in (42, 43, 44, 45, 46)] + [(NEW_ROOT, s) for s in (47, 48, 49, 50, 51)]


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # --- whole-schedule detail: concat, restrict to the six reported policies ---
    summary = pd.concat(
        [pd.read_csv(OLD_ROOT / "baseline_comparison_summary.csv"),
         pd.read_csv(NEW_ROOT / "baseline_comparison_summary.csv")],
        ignore_index=True,
    )
    summary = summary[summary["algorithm"].isin(SCHED_GLOB)]
    summary["target_pods"] = pd.to_numeric(summary["target_pods"], errors="coerce")
    summary["elapsed_seconds"] = pd.to_numeric(summary["elapsed_seconds"], errors="coerce")
    summary.to_csv(OUT_ROOT / "baseline_comparison_summary.csv", index=False)
    whole = (
        summary.groupby(["target_pods", "algorithm"], as_index=False)
        .agg(runs=("experiment", "count"),
             mean_success=("success_rate_pct", "mean"),
             mean_placed_pods=("placed_pods", "mean"),
             mean_total_kg=("total_kg", "mean"),
             mean_operational_kg=("operational_kg", "mean"),
             mean_embodied_kg=("embodied_kg", "mean"),
             mean_per_pod_g=("per_pod_g", "mean"),
             std_per_pod_g=("per_pod_g", "std"),
             mean_runtime_s=("elapsed_seconds", "mean"),
             std_runtime_s=("elapsed_seconds", "std"))
        .sort_values(["target_pods", "algorithm"])
    )
    whole.to_csv(OUT_ROOT / "baseline_comparison_aggregate_by_pods.csv", index=False)

    # --- uniform six-policy common cohort, recomputed from placements ---
    rows = []
    for root, seed in CELLS:
        for pods in PODS:
            cell = root / f"seed_{seed}_pods_{pods}"
            nodes = load_nodes(cell / "nodes.yaml")
            forecasts = load_forecasts(cell / "all_forecasts.json")
            attrs = {}
            ok = True
            for name, pat in SCHED_GLOB.items():
                hits = sorted(cell.glob(pat))
                csvp = find_placement_csv(hits[-1]) if hits else None
                if not csvp:
                    ok = False
                    break
                attrs[name] = attribute_pod_emissions(csvp, nodes, forecasts)
            if not ok:
                print(f"[skip] {cell}")
                continue
            cohort = set.intersection(*[set(a) for a in attrs.values()])
            for name, a in attrs.items():
                op = sum(a[p]["operational_g"] for p in cohort) / 1000.0
                emb = sum(a[p]["embodied_g"] for p in cohort) / 1000.0
                rows.append(dict(common_pods=len(cohort), operational_kg=op, embodied_kg=emb,
                                 total_kg=op + emb, per_pod_g=(op + emb) * 1000.0 / len(cohort),
                                 seed=seed, target_pods=pods,
                                 comparison="all_algorithms_common", algorithm=name,
                                 experiment=f"uniform6_{seed}_{pods}"))
        print(f"[done] seed {seed}", flush=True)

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_ROOT / "attributed_common_pod_comparison.csv", index=False)
    agg = (
        detail.groupby(["target_pods", "algorithm"], as_index=False)
        .agg(runs=("experiment", "count"),
             mean_common_pods=("common_pods", "mean"),
             mean_total_kg=("total_kg", "mean"),
             mean_per_pod_g=("per_pod_g", "mean"),
             std_per_pod_g=("per_pod_g", "std"))
        .sort_values(["target_pods", "algorithm"])
    )
    agg.to_csv(OUT_ROOT / "attributed_common_all_algorithms_aggregate.csv", index=False)
    print(f"merged root: {OUT_ROOT}")
    print(agg[agg.target_pods == 200].round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
