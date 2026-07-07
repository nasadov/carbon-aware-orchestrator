#!/usr/bin/env python3
"""MC1 (TSUSC Round 2): guarded-baseline comparison that ISOLATES the envelope's ranking.

The round-2 referee's central objection (MC1): "the only method that certifies" is true *by
construction* -- the envelope is the only method carrying the no-harm guard, so unguarded competitors
trivially fail it. The genuinely open question is whether the envelope's lexicographic combined ranking
(#certified-axes-improved -> grid-relief -> water -> carbon) extracts more *certified* grid-relief /
co-benefit than a method that *also* carries the guard.

This script answers it. All five modes repair from the SAME packing baseline B with the IDENTICAL
no-harm guard; they differ ONLY in the acceptance ranking:

  combined           the envelope (lexicographic combined objective)
  search_control     existing same-budget control (co-benefit greedy; ignores grid-relief)
  guarded_random     accept any >=1-axis-improving move in arbitrary (seeded) order
  guarded_carbon     epsilon-constraint carbon-greedy (guarded)
  guarded_waterwise  epsilon-constraint WaterWise scalar w*carbon+(1-w)*water (guarded)

Because every mode is guard-passing, ALL certify ~100% (this CONFIRMS MC1(a)). The result is therefore
the DELTA in certified grid-stress relief and carbon/water co-benefit, with 95% CIs over seeds for the
seed-dependent guarded_* controls. Engine is untouched except for the three additive score_mode
branches (default-off, bit-identical).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SERVER = REPO_ROOT / "pkg" / "carbon-aware" / "server-python"
for p in (str(SERVER), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dataclasses import replace  # noqa: E402

from carbon_aware.no_harm_flexibility import (  # noqa: E402
    PilotConfig,
    _rematerialize_under_realized,
    build_action_signals,
    build_greedy_schedule,
    load_flavours_for_pilot,
    load_pods,
    repair_schedule_no_harm,
    summarize_against_reference,
)
from run_alibaba_no_harm import REGIONS, write_gpu_fleet  # noqa: E402

TIMEALIGNED = REPO_ROOT / "pkg" / "carbon-aware" / "data" / "timealigned"

# combined/search_control are deterministic (no random tie-break) -> one seed each; the guarded_*
# controls depend on repair_random_seed -> run over the seed set for CIs.
DETERMINISTIC = {"combined", "combined_v2", "search_control"}
MODES = ["combined", "combined_v2", "search_control", "guarded_random", "guarded_carbon", "guarded_waterwise"]

# t_{0.975, n-1} for small n (two-sided 95%).
_T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
      9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086}


def _ci95(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    s = statistics.stdev(xs)
    t = _T.get(n - 1, 1.96)
    return t * s / math.sqrt(n)


def run_mode(mode: str, *, packing, pods, flavours, signals, config, seeds: Sequence[int]) -> Dict:
    """Repair from B with this mode over the seed set; return per-seed rows + aggregates."""
    use_seeds = [0] if mode in DETERMINISTIC else list(seeds)
    rows: List[Dict] = []
    for seed in use_seeds:
        cfg = replace(config, repair_random_seed=seed)
        if mode == "combined_v2":
            # Dominating two-phase envelope: phase 1 = co-benefit greedy (capture the full certified
            # carbon/water co-benefit), phase 2 = relief-only ON TOP (add the grid relief the
            # co-benefit greedy lacks), both guarded vs B. Co-benefit-first (not relief-first) because a
            # relief-first phase 2 chasing carbon UNDOES the relief; relief-only phase 2 only makes
            # relieving moves (guarded), so it adds relief without sacrificing the phase-1 co-benefit.
            phase1 = repair_schedule_no_harm(
                method_key="search_control", baseline=packing, pods=pods, flavours=flavours,
                signals=signals, config=cfg, score_mode="search_control",
            )
            result = repair_schedule_no_harm(
                method_key="combined_v2", baseline=packing, pods=pods, flavours=flavours,
                signals=signals, config=cfg, score_mode="relief_only", start_schedule=phase1,
            )
        else:
            result = repair_schedule_no_harm(
                method_key=mode, baseline=packing, pods=pods, flavours=flavours,
                signals=signals, config=cfg, score_mode=mode,
            )
        # Correct occupancy-based footprints under the realized signals BEFORE scoring the certificate
        # (the in-place repair keeps stale per-pod footprints that lose a node's idle when its
        # idle-bearer moves -> spurious savings). Mirrors run_no_harm_flexibility_pilot exactly.
        _rematerialize_under_realized(result, flavours, config)
        row = summarize_against_reference(result, packing, signals)
        rows.append(row)

    def col(k):
        return [float(r[k]) for r in rows]

    cert_rate = sum(1 for r in rows if r["no_harm_certificate"]) / len(rows)
    return {
        "mode": mode,
        "n_seeds": len(use_seeds),
        "cert_rate": cert_rate,
        "relief_kwh_mean": statistics.mean(col("stress_kwh_avoided")),
        "relief_kwh_ci": _ci95(col("stress_kwh_avoided")),
        "relief_pct_mean": statistics.mean(col("stress_kwh_avoided_pct")),
        "carbon_pct_mean": statistics.mean(col("carbon_delta_pct")),
        "carbon_pct_ci": _ci95(col("carbon_delta_pct")),
        "water_pct_mean": statistics.mean(col("scarcity_delta_pct")),
        "water_pct_ci": _ci95(col("scarcity_delta_pct")),
        "repairs_mean": statistics.mean(col("repairs_applied")),
        "placed_mean": statistics.mean(col("placed_pods")),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="experiments/real_traces/alibaba_gpu_2020_d10_s42_200")
    ap.add_argument("--nodes-per-region", type=int, default=4)
    ap.add_argument("--max-timeslots", type=int, default=48)
    ap.add_argument("--lever-mode", default="both")
    ap.add_argument("--scenario", default="heatwave-drought")
    ap.add_argument("--signals-dir", default="pkg/carbon-aware/data/timealigned_realci",
                    help="signals dir (default = measured-CI timealigned_realci)")
    ap.add_argument("--seeds", type=int, default=8, help="#seeds for the guarded_* controls")
    ap.add_argument("--run-name", default="mc1_guarded")
    ap.add_argument("--tag", default="alibaba_gpu")
    # Prebuilt-fleet mode (e.g. the strong/favorable CPU regime): if --nodes-file is given we use it
    # and --workloads-dir directly instead of building the GPU fleet from --window.
    ap.add_argument("--nodes-file", default=None)
    ap.add_argument("--workloads-dir", default=None)
    ap.add_argument("--config-file", default="pkg/carbon-aware/infra-workload-config.yaml")
    ap.add_argument("--max-pods", type=int, default=None)
    # Azure CPU general-cloud mode: rebuild workloads from the window's canonical_trace_workload.csv and
    # provision a realistic CPU fleet to --target-util (mirrors run_azure_no_harm.py via azure_common).
    ap.add_argument("--azure-window", default=None)
    ap.add_argument("--target-util", type=float, default=0.55)
    ap.add_argument("--basin-cf-csv", default=None,
                    help="AWARE basin-CF override table (default-off engine flag basin_cf_csv; "
                         "e.g. data/water/aware20_basin_nonagri_factors_v3.csv for the German-basin fleet)")
    args = ap.parse_args()
    import logging
    logging.disable(logging.CRITICAL)

    signals_dir = (REPO_ROOT / args.signals_dir).resolve()
    run_dir = REPO_ROOT / "experiments" / "mc1_guarded" / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    config_file = (REPO_ROOT / args.config_file).resolve()

    if args.azure_window:  # realistic CPU fleet from the canonical Azure trace
        import azure_common as ac
        src = (REPO_ROOT / args.azure_window).resolve()
        canonical = src / "canonical_trace_workload.csv"
        if not canonical.exists():
            raise SystemExit(f"no canonical_trace_workload.csv under {src}")
        stats = ac.build_window_from_canonical(canonical, run_dir, seed=42, cpu_util=ac.conv.CPU_UTIL_MEAN)
        workloads = stats["workloads_dir"]
        per_region = ac.provision_for_util(ac.workload_peak_cores(workloads), args.target_util)
        nodes_file, n_nodes, _cap = ac.build_fleet(run_dir / "fleet", per_region)
        window = src
    elif args.nodes_file:  # prebuilt fleet
        nodes_file = (REPO_ROOT / args.nodes_file).resolve()
        workloads = (REPO_ROOT / args.workloads_dir).resolve()
        n_nodes = sum(1 for _ in open(nodes_file) if _.startswith("kind: Node"))
        window = workloads.parent
    else:  # GPU fleet from the converted trace window
        window = (REPO_ROOT / args.window).resolve()
        workloads = window / "workloads"
        if not workloads.exists():
            raise SystemExit(f"no workloads/ under {window}")
        fleet_dir = run_dir / "fleet"
        fleet_dir.mkdir(parents=True, exist_ok=True)
        nodes_file = fleet_dir / "nodes.yaml"
        n_nodes = write_gpu_fleet(nodes_file, args.nodes_per_region)

    config = PilotConfig(
        repo_root=REPO_ROOT, nodes_file=nodes_file, workloads_dir=workloads,
        forecasts_file=signals_dir / "forecasts.json", config_file=config_file,
        output_dir=run_dir / "out", max_timeslots=args.max_timeslots, max_pods=args.max_pods,
        scenario=args.scenario, lever_mode=args.lever_mode,
        scenario_month=("aug" if "2022" in signals_dir.name else "jul"),  # Aug-2022 dirs -> Aug CFs
        grid_signal_csv=signals_dir / "grid_residual_region_slot.csv",
        wue_csv=signals_dir / "wue_region_slot.csv",
        # in-window off-site water (same measured mix as the CI), when present in the signals dir
        ewif_csv=((signals_dir / "ewif_region_slot.csv")
                  if (signals_dir / "ewif_region_slot.csv").exists() else None),
        basin_cf_csv=((REPO_ROOT / args.basin_cf_csv).resolve() if args.basin_cf_csv else None),
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    flavours = load_flavours_for_pilot(config)
    pods = load_pods(workloads, max_pods=config.max_pods)
    signals = build_action_signals(flavours, config)
    packing, _, _ = build_greedy_schedule(method_key="packing", pods=pods, flavours=flavours, config=config)

    seeds = list(range(args.seeds))
    results = [run_mode(m, packing=packing, pods=pods, flavours=flavours, signals=signals,
                        config=config, seeds=seeds) for m in MODES]

    env = next(r for r in results if r["mode"] == "combined")
    print(f"\n=== MC1 guarded-baseline isolation ({args.tag}, {window.name}) ===")
    print(f"fleet={n_nodes} GPU nodes  pods={len(pods)}  signals={args.signals_dir.split('/')[-1]}  "
          f"seeds(guarded_*)={args.seeds}")
    print(f"{'mode':>18} {'cert':>5} {'relief_kWh(95%CI)':>20} {'relief%':>8} "
          f"{'carbon%':>9} {'water%':>9} {'moves':>6}")
    for r in results:
        star = "  <- envelope" if r["mode"] == "combined" else ""
        print(f"{r['mode']:>18} {100*r['cert_rate']:>4.0f}% "
              f"{r['relief_kwh_mean']:>9.3f} +/-{r['relief_kwh_ci']:<6.3f} "
              f"{r['relief_pct_mean']:>7.1f}% {r['carbon_pct_mean']:>8.2f}% "
              f"{r['water_pct_mean']:>8.2f}% {r['repairs_mean']:>6.1f}{star}")

    # The headline deltas: envelope relief vs each guarded control.
    print("\n  envelope grid-relief advantage (kWh) over guarded controls:")
    for r in results:
        if r["mode"] in ("combined", "search_control"):
            continue
        d = env["relief_kwh_mean"] - r["relief_kwh_mean"]
        rel = (100 * d / r["relief_kwh_mean"]) if r["relief_kwh_mean"] > 1e-9 else float("nan")
        print(f"    vs {r['mode']:>18}: +{d:.3f} kWh ({rel:+.0f}% more relief), "
              f"both cert={100*r['cert_rate']:.0f}%")

    digest = {"tag": args.tag, "window": window.name, "n_nodes": n_nodes, "pods": len(pods),
              "signals_dir": args.signals_dir, "seeds": args.seeds, "modes": results}
    (run_dir / "mc1_digest.json").write_text(json.dumps(digest, indent=2))
    print(f"\noutput={run_dir/'mc1_digest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
