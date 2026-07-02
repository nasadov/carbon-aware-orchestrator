#!/usr/bin/env python3
"""MC3 -- re-verify the no-harm CARBON certificate on a MARGINAL (short-run) emissions signal.

Reviewer MC3: the carbon no-harm certificate is verified on the measured hourly AVERAGE
generation-mix intensity, but the grid-supportive/consequential framing concerns the MARGINAL
(price-setting) unit. This runner re-verifies, WITHOUT changing any decision, whether carbon
non-degradation still holds when the certificate is checked against a short-run MARGINAL emission
factor (SRMEF) signal derived by merit-order attribution to the price-setting fuel.

Mechanism (NO engine change): decisions are made on the measured-CI signals (timealigned_realci),
exactly as in the headline RQ1/RQ2 runs; the no-harm CERTIFICATE is then re-evaluated against the
marginal forecasts via PilotConfig.verify_forecasts_file (the existing decide-on-X / verify-on-Y
hook). For each window we run:
  (A) decide=realci, verify=realci   -> the headline AVERAGE-account certificate (sanity baseline)
  (B) decide=realci, verify=marginal -> the MC3 MARGINAL-account re-verification

It reports, per window and method, the certificate (carbon non-increase / scarcity / SLO) and the
carbon delta on each account. The marginal grid (gas-marginal in the 2018-07 EU summer) is nearly
flat, so temporal/spatial shifting has ~0 marginal-carbon delta: average non-degradation is
conservative w.r.t. marginal. Any window where the marginal certificate FAILS is reported with the
exact carbon delta.

Outputs: experiments/mc3_marginal/<window>/{average,marginal}/ run dirs + a combined
experiments/mc3_marginal/mc3_results.json digest.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SERVER = REPO_ROOT / "pkg" / "carbon-aware" / "server-python"
for p in (str(SERVER), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import azure_common as ac  # noqa: E402
from run_alibaba_no_harm import GPU_NODE, write_gpu_fleet  # noqa: E402
from carbon_aware.no_harm_flexibility import (  # noqa: E402
    PilotConfig, load_pods, run_no_harm_flexibility_pilot,
)

REALCI = REPO_ROOT / "pkg" / "carbon-aware" / "data" / "timealigned_realci2"  # coherent revamp build (Option A + in-window EWIF + continuous cooling)
MARGINAL = REPO_ROOT / "pkg" / "carbon-aware" / "data" / "timealigned_marginal"  # SRMEF verify signal (merit-order attribution; cooling-independent)
CONFIG_FILE = REPO_ROOT / "pkg" / "carbon-aware" / "infra-workload-config.yaml"
OUT_ROOT = REPO_ROOT / "experiments" / "mc3_marginal"

ALIBABA_WINDOWS = [
    "alibaba_gpu_2020_w1575_s42_200",
    "alibaba_gpu_2020_w1587_s43_200",
    "alibaba_gpu_2020_w1600_s44_200",
]
AZURE_WINDOWS = [
    "azure_packing_2020_d0_s42_200",
    "azure_packing_2020_d2_s43_200",
    "azure_packing_2020_d4_s44_200",
]
METHODS = ("packing", "carbon", "water_scarcity", "waterwise",
           "no_harm_search_control", "no_harm_flex")


def _digest_row(row: dict) -> dict:
    return {k: row.get(k) for k in (
        "method_key", "no_harm_certificate", "carbon_nonincrease", "scarcity_nonincrease",
        "slo_non_decrease", "carbon_delta_kg", "carbon_delta_pct", "scarcity_delta_pct",
        "stress_kwh_avoided", "repairs_applied", "placed_pods", "unplaced_pods",
    )}


def _build_gpu_config(window: str, run_dir: Path, *, decide_signals: Path,
                      verify_forecasts: Path | None, max_ts: int, lever_mode: str,
                      scenario: str) -> tuple[PilotConfig, int, int]:
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
        forecasts_file=decide_signals / "forecasts.json", config_file=CONFIG_FILE,
        output_dir=run_dir / "out", max_timeslots=max_ts, max_pods=None,
        scenario=scenario, lever_mode=lever_mode,
        grid_signal_csv=decide_signals / "grid_residual_region_slot.csv",
        wue_csv=decide_signals / "wue_region_slot.csv",
        verify_forecasts_file=(verify_forecasts if verify_forecasts is not None else None),
    )
    pods = load_pods(workloads)
    return pc, n_nodes, len(pods)


def _build_azure_config(window: str, run_dir: Path, *, decide_signals: Path,
                        verify_forecasts: Path | None, max_ts: int, lever_mode: str,
                        scenario: str, seed: int, target_util: float) -> tuple[PilotConfig, int, int]:
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
        nodes_file, workloads, run_dir / "out", signals=decide_signals,
        max_timeslots=max_ts, scenario=scenario, lever_mode=lever_mode,
    )
    pc = replace(pc, verify_forecasts_file=(verify_forecasts if verify_forecasts is not None else None))
    pods = load_pods(workloads)
    return pc, n_nodes, len(pods)


def run_window(testbed: str, window: str, *, max_ts: int, lever_mode: str, scenario: str,
               seed: int, target_util: float) -> dict:
    win_out = OUT_ROOT / window
    accounts = {}
    for label, verify in (("average", None), ("marginal", MARGINAL)):
        run_dir = win_out / label
        run_dir.mkdir(parents=True, exist_ok=True)
        if testbed == "alibaba":
            pc, n_nodes, n_pods = _build_gpu_config(
                window, run_dir, decide_signals=REALCI, verify_forecasts=verify,
                max_ts=max_ts, lever_mode=lever_mode, scenario=scenario)
        else:
            pc, n_nodes, n_pods = _build_azure_config(
                window, run_dir, decide_signals=REALCI, verify_forecasts=verify,
                max_ts=max_ts, lever_mode=lever_mode, scenario=scenario,
                seed=seed, target_util=target_util)
        res = run_no_harm_flexibility_pilot(pc)
        rows = {r["method_key"]: r for r in res["summary_rows"]}
        accounts[label] = {m: _digest_row(rows[m]) for m in METHODS if m in rows}
        accounts.setdefault("_meta", {})[label] = {"n_nodes": n_nodes, "n_pods": n_pods}
    return {"testbed": testbed, "window": window, "decide_signals": "timealigned_realci",
            "accounts": accounts}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-timeslots", type=int, default=48)
    ap.add_argument("--lever-mode", default="both")
    ap.add_argument("--scenario", default="heatwave-drought")
    ap.add_argument("--target-util", type=float, default=0.55)
    ap.add_argument("--only", default=None, help="comma-separated window names to restrict to")
    args = ap.parse_args()
    logging.disable(logging.CRITICAL)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    only = set(args.only.split(",")) if args.only else None
    plan = ([("alibaba", w, 42) for w in ALIBABA_WINDOWS]
            + [("azure", AZURE_WINDOWS[i], s) for i, s in enumerate((42, 43, 44))])

    all_results = []
    for testbed, window, seed in plan:
        if only and window not in only:
            continue
        print(f"\n##### MC3 re-verify: {testbed} / {window} (decide=realci) #####")
        r = run_window(testbed, window, max_ts=args.max_timeslots, lever_mode=args.lever_mode,
                       scenario=args.scenario, seed=seed, target_util=args.target_util)
        all_results.append(r)
        env_avg = r["accounts"]["average"]["no_harm_flex"]
        env_mar = r["accounts"]["marginal"]["no_harm_flex"]
        print(f"  ENVELOPE  average: cert={env_avg['no_harm_certificate']} "
              f"carbonΔ={env_avg['carbon_delta_pct']:+.3f}% ({env_avg['carbon_delta_kg']:+.4f} kg)")
        print(f"  ENVELOPE marginal: cert={env_mar['no_harm_certificate']} "
              f"carbonΔ={env_mar['carbon_delta_pct']:+.3f}% ({env_mar['carbon_delta_kg']:+.4f} kg) "
              f"carbon_nonincrease={env_mar['carbon_nonincrease']}")

    digest_path = OUT_ROOT / "mc3_results.json"
    digest_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nwrote {digest_path}")

    # Concise verdict table
    print("\n=== MARGINAL-account certificate (envelope), per window ===")
    print(f"{'window':36s} {'avg_cert':8s} {'mar_cert':8s} {'avg_carbonΔ%':>12s} {'mar_carbonΔ%':>12s}")
    n_hold = 0
    for r in all_results:
        a = r["accounts"]["average"]["no_harm_flex"]
        m = r["accounts"]["marginal"]["no_harm_flex"]
        n_hold += int(bool(m["no_harm_certificate"]))
        print(f"{r['window']:36s} {str(a['no_harm_certificate']):8s} {str(m['no_harm_certificate']):8s} "
              f"{a['carbon_delta_pct']:12.3f} {m['carbon_delta_pct']:12.3f}")
    print(f"\nmarginal certificate holds on {n_hold}/{len(all_results)} windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
