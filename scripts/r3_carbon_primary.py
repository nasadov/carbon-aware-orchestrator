#!/usr/bin/env python3
"""R3 / reviewer MC4-iii (Q4): FOOTPRINT CO-BENEFIT vs A CARBON-PRIMARY BASELINE.

Round-2 headline carbon co-benefit (e.g. -20% on Azure) is measured against the packing baseline B,
whose candidate sort key is (earliest-slot, pack_score, lifecycle-carbon, water) -- carbon is only the
3rd (tiebreak) key. The round-3 reviewer's sharp point: a large carbon co-benefit vs B partly just
measures how carbon-UNAWARE B is ("B doesn't move work to France"). The honest test is the co-benefit
against a CARBON-PRIMARY reference B'.

We use B' = the engine's carbon-greedy schedule (build_greedy_schedule(method_key="carbon")), whose
sort key is (lifecycle-carbon, water, pack_score, slot) -- carbon as the PRIMARY objective. This is the
spatial-carbon-arbitrage-optimal placement available to a single-objective carbon scheduler on the same
feasible-candidate machinery the envelope uses, so it is the fair "carbon-primary" reference (a
carbon-primary *packing* variant would be strictly weaker on carbon -- carbon-greedy upper-bounds the
carbon a carbon-aware baseline can remove, making it the HARDEST reference for our co-benefit claim).

For each window we run BOTH certified frontier members --
  * relief-priority envelope     (no_harm_flex,           score_mode="combined")
  * footprint-priority member    (no_harm_search_control, score_mode="search_control")
-- once with baseline=B (packing) and once with baseline=B' (carbon). The no-harm guard and the
reported co-benefit are BOTH evaluated against whichever baseline is the reference. Footprints are
re-materialized under the realized signals BEFORE scoring (mirrors run_no_harm_flexibility_pilot and
guarded_baselines_mc1.py), so totals/certificate use correct occupancy accounting.

Output: experiments/r3_carbon_primary/{results.json, RESULTS.md, table.tex}.

Run:
  PYTHONPATH=pkg/carbon-aware/server-python python scripts/r3_carbon_primary.py
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SERVER = REPO_ROOT / "pkg" / "carbon-aware" / "server-python"
for p in (str(SERVER), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import azure_common as ac  # noqa: E402
from run_alibaba_no_harm import write_gpu_fleet  # noqa: E402
from carbon_aware.no_harm_flexibility import (  # noqa: E402
    PilotConfig,
    _rematerialize_under_realized,
    build_action_signals,
    build_greedy_schedule,
    load_flavours_for_pilot,
    load_pods,
    repair_schedule_no_harm,
    summarize_against_reference,
    ScheduleResult,
)

REALCI = REPO_ROOT / "pkg" / "carbon-aware" / "data" / "timealigned_realci2"
CONFIG_FILE = REPO_ROOT / "pkg" / "carbon-aware" / "infra-workload-config.yaml"
OUT_ROOT = REPO_ROOT / "experiments" / "r3_carbon_primary"

ALIBABA_WINDOWS = [
    "alibaba_gpu_2020_w1575_s42_200",
    "alibaba_gpu_2020_w1587_s43_200",
    "alibaba_gpu_2020_w1600_s44_200",
]
AZURE_WINDOWS = [
    ("azure_packing_2020_d0_s42_200", 42),
    ("azure_packing_2020_d2_s43_200", 43),
    ("azure_packing_2020_d4_s44_200", 44),
]


def _build_gpu_config(window: str, run_dir: Path) -> Tuple[PilotConfig, int]:
    window_dir = (REPO_ROOT / "experiments" / "real_traces" / window).resolve()
    workloads = window_dir / "workloads"
    if not workloads.exists():
        raise SystemExit(f"no workloads/ under {window_dir}")
    fleet_dir = run_dir / "fleet"
    fleet_dir.mkdir(parents=True, exist_ok=True)
    nodes_file = fleet_dir / "nodes.yaml"
    n_nodes = write_gpu_fleet(nodes_file, 4)
    pc = PilotConfig(
        repo_root=REPO_ROOT, nodes_file=nodes_file, workloads_dir=workloads,
        forecasts_file=REALCI / "forecasts.json", config_file=CONFIG_FILE,
        output_dir=run_dir / "out", max_timeslots=48, max_pods=None,
        scenario="heatwave-drought", lever_mode="both",
        grid_signal_csv=REALCI / "grid_residual_region_slot.csv",
        wue_csv=REALCI / "wue_region_slot.csv",
    )
    return pc, n_nodes


def _build_azure_config(window: str, run_dir: Path, *, seed: int,
                        target_util: float = 0.55) -> Tuple[PilotConfig, int]:
    src = (REPO_ROOT / "experiments" / "real_traces" / window).resolve()
    canonical = src / "canonical_trace_workload.csv"
    if not canonical.exists():
        raise SystemExit(f"no canonical_trace_workload.csv under {src}")
    stats = ac.build_window_from_canonical(canonical, run_dir, seed=seed, cpu_util=ac.conv.CPU_UTIL_MEAN)
    workloads = stats["workloads_dir"]
    peak = ac.workload_peak_cores(workloads)
    per_region = ac.provision_for_util(peak, target_util)
    nodes_file, n_nodes, _cap = ac.build_fleet(run_dir / "fleet", per_region)
    pc = ac.make_pilot_config(
        nodes_file, workloads, run_dir / "out", signals=REALCI,
        max_timeslots=48, scenario="heatwave-drought", lever_mode="both",
    )
    return pc, n_nodes


def _run_member(score_mode: str, method_key: str, *, baseline, pods, flavours, signals, config,
                max_repairs=None):
    cfg = config if max_repairs is None else replace(config, max_repairs=max_repairs)
    result = repair_schedule_no_harm(
        method_key=method_key, baseline=baseline, pods=pods, flavours=flavours,
        signals=signals, config=cfg, score_mode=score_mode,
    )
    # Correct occupancy-based footprints under the realized signals BEFORE scoring (mirrors the pilot).
    _rematerialize_under_realized(result, flavours, config)
    summary = summarize_against_reference(result, baseline, signals)
    return result, summary


def run_window(testbed: str, window: str, *, seed: int) -> dict:
    run_dir = OUT_ROOT / window
    run_dir.mkdir(parents=True, exist_ok=True)
    if testbed == "alibaba":
        config, n_nodes = _build_gpu_config(window, run_dir)
    else:
        config, n_nodes = _build_azure_config(window, run_dir, seed=seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    flavours = load_flavours_for_pilot(config)
    pods = load_pods(config.workloads_dir, max_pods=config.max_pods)
    signals = build_action_signals(flavours, config)

    # Baselines: B = packing (round-2 reference); B' = carbon-greedy (carbon-PRIMARY reference).
    packing, _, _ = build_greedy_schedule(method_key="packing", pods=pods, flavours=flavours, config=config)
    carbon_b, _, _ = build_greedy_schedule(method_key="carbon", pods=pods, flavours=flavours, config=config)

    # The two baselines' own footprints (for context: how much carbon B' already removed vs B).
    pk_rem = _copy_schedule(packing); cb_rem = _copy_schedule(carbon_b)
    _rematerialize_under_realized(pk_rem, flavours, config)
    _rematerialize_under_realized(cb_rem, flavours, config)
    b_vs_b = summarize_against_reference(cb_rem, pk_rem, signals)  # B' measured against B
    base_ctx = {
        "carbon_primary_carbon_delta_pct_vs_packing": b_vs_b["carbon_delta_pct"],
        "carbon_primary_water_delta_pct_vs_packing": b_vs_b["scarcity_delta_pct"],
        "packing_carbon_kg": summarize_against_reference(pk_rem, pk_rem, signals)["carbon_kg"],
        "carbon_primary_carbon_kg": b_vs_b["carbon_kg"],
    }

    out: Dict[str, dict] = {}
    # Reference-anchor budget parity: search_control uses the relief member's repair count (mirrors pilot).
    for ref_name, ref in (("packing", packing), ("carbon_primary", carbon_b)):
        relief, relief_s = _run_member("combined", "no_harm_flex", baseline=ref, pods=pods,
                                       flavours=flavours, signals=signals, config=config)
        fp, fp_s = _run_member("search_control", "no_harm_search_control", baseline=ref, pods=pods,
                               flavours=flavours, signals=signals, config=config,
                               max_repairs=relief.repairs_applied)
        out[ref_name] = {
            "relief": _summary_subset(relief_s),
            "footprint": _summary_subset(fp_s),
        }

    return {
        "testbed": testbed, "window": window, "n_nodes": n_nodes, "n_pods": len(pods),
        "decide_signals": "timealigned_realci2",
        "baseline_context": base_ctx,
        "vs_baseline": out,
    }


def _copy_schedule(src: ScheduleResult) -> ScheduleResult:
    """Shallow copy so re-materializing a baseline's footprints doesn't mutate the original schedule
    used as the no-harm reference."""
    return ScheduleResult(
        method_key=src.method_key,
        placements=[replace(p) for p in src.placements],
        unplaced_pods=list(src.unplaced_pods),
        elapsed_seconds=src.elapsed_seconds,
    )


def _summary_subset(s: dict) -> dict:
    return {
        "no_harm_certificate": bool(s["no_harm_certificate"]),
        "carbon_nonincrease": bool(s["carbon_nonincrease"]),
        "scarcity_nonincrease": bool(s["scarcity_nonincrease"]),
        "carbon_delta_pct": s["carbon_delta_pct"],
        "scarcity_delta_pct": s["scarcity_delta_pct"],
        "carbon_kg": s["carbon_kg"],
        "scarcity_water": s["scarcity_water"],
        "stress_kwh_avoided": s["stress_kwh_avoided"],
        "stress_kwh_avoided_pct": s["stress_kwh_avoided_pct"],
        "weighted_stress_kwh_avoided_pct": s["weighted_stress_kwh_avoided_pct"],
        "repairs_applied": s["repairs_applied"],
        "placed_pods": s["placed_pods"],
    }


def main() -> int:
    logging.disable(logging.CRITICAL)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    plan = ([("alibaba", w, 42) for w in ALIBABA_WINDOWS]
            + [("azure", w, s) for (w, s) in AZURE_WINDOWS])
    results: List[dict] = []
    for testbed, window, seed in plan:
        print(f"\n##### r3 carbon-primary: {testbed} / {window} #####", flush=True)
        r = run_window(testbed, window, seed=seed)
        results.append(r)
        ctx = r["baseline_context"]
        print(f"  B' (carbon-greedy) vs B (packing): carbonΔ={ctx['carbon_primary_carbon_delta_pct_vs_packing']:+.2f}%  "
              f"waterΔ={ctx['carbon_primary_water_delta_pct_vs_packing']:+.2f}%")
        for member in ("relief", "footprint"):
            vb = r["vs_baseline"]["packing"][member]
            vbp = r["vs_baseline"]["carbon_primary"][member]
            print(f"  {member:9s}  vs B : cert={vb['no_harm_certificate']!s:5} carbonΔ={vb['carbon_delta_pct']:+7.2f}% waterΔ={vb['scarcity_delta_pct']:+7.2f}%  ||  "
                  f"vs B': cert={vbp['no_harm_certificate']!s:5} carbonΔ={vbp['carbon_delta_pct']:+7.2f}% waterΔ={vbp['scarcity_delta_pct']:+7.2f}%")

    digest = OUT_ROOT / "results.json"
    digest.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
