#!/usr/bin/env python3
"""Pilot analysis for the n=10 seed extension (seeds 42-46 + 47-51).

Merges the original and extension runs WITHOUT touching the paper, and reports
whether the n=10 paired statistics strengthen or weaken the current claims:
  A. Matrix 1x/200 common-pod: TotEm vs each baseline (currently GREEN p=0.13,
     Caspian p=0.12 n.s. at n=5).
  B. Six-policy follow-up regime (flexible 2-core 50% load): TotEm vs each
     (currently GREEN delta 2.51 n.s. at n=5).
Prints n=5-original vs n=10-merged side by side.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
MATRIX_OLD = REPO / "experiments/resubmission_baseline_matrix_sweep_v5/matrix_20260515_111056"
MATRIX_NEW_GLOB = str(REPO / "experiments/resubmission_baseline_matrix_seedext/matrix_*")
FOLLOWUP_OLD = REPO / "experiments/workload_regime_baseline_followup/sweep_20260609_151612"
FOLLOWUP_NEW_GLOB = str(REPO / "experiments/workload_regime_baseline_followup_seedext/sweep_*")

BASELINES = ["Vanilla-MostAllocated", "GREEN-MLFQ-K8s", "GreenCourier-Spatial-K8s",
             "Caspian-style", "TotEm-OpOnly"]
FOLLOWUP_NAME_MAP = {
    "vanilla-most-allocated": "Vanilla-MostAllocated",
    "green-mlfq": "GREEN-MLFQ-K8s",
    "greencourier-spatial": "GreenCourier-Spatial-K8s",
    "caspian-style": "Caspian-style",
    "totem-oponly": "TotEm-OpOnly",
    "totem": "TotEm",
}


def paired_table(piv: pd.DataFrame, label: str) -> None:
    print(f"\n--- {label} (n={piv.shape[0]}) | TotEm mean={piv['TotEm'].mean():.2f} g/pod ---")
    for b in BASELINES:
        if b not in piv.columns:
            continue
        d = (piv[b] - piv["TotEm"]).dropna().values
        n = len(d)
        m, sd = d.mean(), d.std(ddof=1)
        ci = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        p = stats.ttest_rel(piv[b].dropna(), piv["TotEm"].dropna()).pvalue
        wins = bool((d > 0).all())
        sig = "SIG" if (p < 0.05 and m > 0) else ("n.s." if m > 0 else "WORSE")
        print(f"  {b:26} delta={m:6.2f}  CI95=+/-{ci:5.2f}  p={p:7.4f}  lower_all_seeds={wins}  [{sig}]")


def matrix_pivot(roots: list[Path]) -> pd.DataFrame:
    frames = []
    for root in roots:
        f = root / "attributed_common_pod_comparison.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
        else:  # aggregate file absent: collect per-cell files
            for cell in sorted(root.glob("seed_*_pods_200")):
                g = cell / "attributed_common_pod_comparison.csv"
                if g.exists():
                    frames.append(pd.read_csv(g))
    df = pd.concat(frames, ignore_index=True)
    df = df[(df.comparison == "all_algorithms_common") & (df.target_pods == 200)]
    return df.pivot_table(index="seed", columns="algorithm", values="per_pod_g")


def followup_pivot(csvs: list[Path]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    df = df[df.deadline == "flexible"].copy()
    df["algorithm"] = df["algorithm"].map(FOLLOWUP_NAME_MAP).fillna(df["algorithm"])
    return df.pivot_table(index="seed", columns="algorithm", values="per_pod_g")


def main() -> int:
    new_matrix = [Path(p) for p in sorted(glob.glob(MATRIX_NEW_GLOB))]
    new_followup = [Path(p) / "common_pod_emissions.csv" for p in sorted(glob.glob(FOLLOWUP_NEW_GLOB))]
    new_followup = [p for p in new_followup if p.exists()]

    print("=== A. Matrix 1x / 200 pods, all-algorithms common-pod ===")
    piv5 = matrix_pivot([MATRIX_OLD])
    paired_table(piv5, "original seeds 42-46")
    if new_matrix:
        piv10 = matrix_pivot([MATRIX_OLD] + new_matrix)
        paired_table(piv10, "merged seeds 42-51")
    else:
        print("  (extension matrix not found yet)")

    print("\n=== B. Six-policy follow-up, flexible 2-core 50% load ===")
    f5 = FOLLOWUP_OLD / "common_pod_emissions.csv"
    piv5f = followup_pivot([f5])
    paired_table(piv5f, "original seeds 42-46")
    if new_followup:
        piv10f = followup_pivot([f5] + new_followup)
        paired_table(piv10f, "merged seeds 42-51")
    else:
        print("  (extension follow-up not found yet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
