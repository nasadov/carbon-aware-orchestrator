#!/usr/bin/env python3
"""T2 — Fleet-scale sweep (headline figure).

Sweeps fleet size (nodes-per-region -> 4 regions) with proportionally-scaled pod
counts, runs the no-harm pilot for several seeds at each size, and reports how the
no-harm guarantee and the signal-driven (stress-aware) advantage over a same-budget
search control behave as the fleet grows. Emits tidy + aggregated CSVs, a multi-panel
figure, and a markdown report. Uses time-aligned 2018 EU signals, lever=both,
heatwave-drought scenario, perfect foresight (forecast error is T5).

Example:
    python scripts/fleet_scale_sweep.py \
        --nodes-per-region 1,2,4,8 --pods 40,80,160,320 \
        --seeds 0,1,2,3,4 --run-name t2_fleet_scale
"""
from __future__ import annotations

import argparse
import csv
import json
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
N_REGIONS = 4  # DE / FR / ES / IT-NO


def _int_series(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _ci95(values: Sequence[float]) -> float:
    """Half-width of the 95% CI of the mean (normal approx; 0 for n<2)."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n < 2:
        return 0.0
    sem = arr.std(ddof=1) / math.sqrt(n)
    return 1.96 * sem


def run(args) -> Path:
    out_root = REPO_ROOT / "experiments" / "flexibility" / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)
    signals = (REPO_ROOT / args.signals_dir).resolve() if args.signals_dir else TIMEALIGNED
    config_file = REPO_ROOT / "pkg" / "carbon-aware" / "infra-workload-config.yaml"
    base_config, gen_nodes, gen_ts = load_generator_dependencies(REPO_ROOT, config_file)

    npr_list = _int_series(args.nodes_per_region)
    pods_list = _int_series(args.pods)
    seeds = _int_series(args.seeds)
    if len(npr_list) != len(pods_list):
        raise SystemExit("--nodes-per-region and --pods must have equal length (paired by fleet size)")

    methods = ["no_harm_flex", "no_harm_search_control", "carbon", "water_scarcity", "packing"]
    tidy_rows: List[Dict] = []

    for npr, pods in zip(npr_list, pods_list):
        fleet = npr * N_REGIONS
        for seed in seeds:
            case = out_root / "inputs" / f"f{fleet}_s{seed}"
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
                config_file=config_file, output_dir=out_root / "cases" / f"f{fleet}_s{seed}",
                max_timeslots=args.max_timeslots, max_pods=None, scenario=args.scenario,
                lever_mode="both", grid_signal_csv=signals / "grid_residual_region_slot.csv",
                wue_csv=signals / "wue_region_slot.csv",
            )
            res = run_no_harm_flexibility_pilot(pc)
            rows = {r["method_key"]: r for r in res["summary_rows"]}
            for m in methods:
                r = rows.get(m, {})
                tidy_rows.append({
                    "fleet": fleet, "pods": pods, "seed": seed, "method": m,
                    "no_harm": int(bool(r.get("no_harm_certificate", False))),
                    "carbon_delta_pct": r.get("carbon_delta_pct", float("nan")),
                    "scarcity_delta_pct": r.get("scarcity_delta_pct", float("nan")),
                    "stress_kwh_avoided": r.get("stress_kwh_avoided", float("nan")),
                    "weighted_stress_kwh": r.get("weighted_stress_kwh", float("nan")),
                    "repairs_applied": r.get("repairs_applied", 0),
                    "rejected_moves": r.get("rejected_moves", 0),
                })
            # Per-instance signal advantage: stress-aware flex vs same-budget control.
            flex_ws = rows["no_harm_flex"]["weighted_stress_kwh"]
            ctrl_ws = rows["no_harm_search_control"]["weighted_stress_kwh"]
            tidy_rows.append({
                "fleet": fleet, "pods": pods, "seed": seed, "method": "flex_vs_control",
                "no_harm": int(flex_ws < ctrl_ws - 1e-12),  # flex wins (less residual weighted stress)
                "carbon_delta_pct": float("nan"), "scarcity_delta_pct": float("nan"),
                "stress_kwh_avoided": ctrl_ws - flex_ws,  # absolute advantage (>0 = flex better)
                "weighted_stress_kwh": (ctrl_ws - flex_ws) / ctrl_ws * 100.0 if ctrl_ws > 1e-12 else 0.0,
                "repairs_applied": rows["no_harm_flex"]["repairs_applied"],
                "rejected_moves": 0,
            })
            print(f"  fleet={fleet:>3} seed={seed}: flex no_harm={rows['no_harm_flex']['no_harm_certificate']} "
                  f"carbon%={rows['no_harm_flex']['carbon_delta_pct']:.2f} "
                  f"scar%={rows['no_harm_flex']['scarcity_delta_pct']:.3f} "
                  f"stress_avoid={rows['no_harm_flex']['stress_kwh_avoided']:.4f} "
                  f"flex>ctrl={'Y' if flex_ws < ctrl_ws - 1e-12 else 'n'} "
                  f"carbon_nh={rows['carbon']['no_harm_certificate']} water_nh={rows['water_scarcity']['no_harm_certificate']}",
                  flush=True)

    # ---- write tidy CSV ----
    tidy_path = out_root / "tidy.csv"
    with tidy_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tidy_rows[0].keys()))
        w.writeheader(); w.writerows(tidy_rows)

    # ---- aggregate mean +/- CI across seeds, per (fleet, method) ----
    fleets = sorted({r["fleet"] for r in tidy_rows})
    agg_methods = methods + ["flex_vs_control"]
    metrics = ["no_harm", "carbon_delta_pct", "scarcity_delta_pct", "stress_kwh_avoided",
               "weighted_stress_kwh", "repairs_applied"]
    agg_rows: List[Dict] = []
    for fleet in fleets:
        for m in agg_methods:
            sub = [r for r in tidy_rows if r["fleet"] == fleet and r["method"] == m]
            row = {"fleet": fleet, "method": m, "n": len(sub)}
            for metric in metrics:
                vals = [r[metric] for r in sub if not (isinstance(r[metric], float) and math.isnan(r[metric]))]
                row[f"{metric}_mean"] = float(np.mean(vals)) if vals else float("nan")
                row[f"{metric}_ci95"] = _ci95(vals) if vals else float("nan")
            agg_rows.append(row)
    agg_path = out_root / "aggregate.csv"
    with agg_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg_rows[0].keys()))
        w.writeheader(); w.writerows(agg_rows)

    label = f"signals={signals.name} · scenario={args.scenario}"
    _write_figure(agg_rows, fleets, out_root / "fleet_scale.png", label)
    _write_report(agg_rows, fleets, out_root / "evidence_report.md", seeds, label)
    print(f"\noutput_dir={out_root}\ntidy={tidy_path}\naggregate={agg_path}")
    return out_root


def _get(agg_rows, fleet, method, col):
    for r in agg_rows:
        if r["fleet"] == fleet and r["method"] == method:
            return r.get(col, float("nan"))
    return float("nan")


def _write_figure(agg_rows, fleets, path: Path, label: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array(fleets, dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Panel 1: no-harm certificate rate by method
    ax = axes[0, 0]
    for m, lab in [("no_harm_flex", "no-harm flex"), ("carbon", "carbon-only"),
                   ("water_scarcity", "water-only")]:
        y = [100 * _get(agg_rows, f, m, "no_harm_mean") for f in fleets]
        ax.plot(x, y, marker="o", label=lab)
    ax.set_title("No-harm certificate rate vs fleet size")
    ax.set_xlabel("fleet size (nodes)"); ax.set_ylabel("% configs certified no-harm")
    ax.set_ylim(-5, 105); ax.set_xscale("log", base=2); ax.set_xticks(x); ax.set_xticklabels([int(v) for v in x])
    ax.legend(); ax.grid(alpha=0.3)

    # Panel 2: signal advantage (flex beats same-budget control)
    ax = axes[0, 1]
    win = [100 * _get(agg_rows, f, "flex_vs_control", "no_harm_mean") for f in fleets]
    adv = [_get(agg_rows, f, "flex_vs_control", "weighted_stress_kwh_mean") for f in fleets]
    ax.plot(x, win, marker="o", color="C2", label="flex wins vs control (%)")
    ax.plot(x, adv, marker="s", color="C3", label="weighted-stress relief gap (%)")
    ax.set_title("Stress-aware signal advantage vs fleet size")
    ax.set_xlabel("fleet size (nodes)"); ax.set_ylabel("%")
    ax.set_xscale("log", base=2); ax.set_xticks(x); ax.set_xticklabels([int(v) for v in x])
    ax.legend(); ax.grid(alpha=0.3)

    # Panel 3: carbon & scarcity co-benefit (flex)
    ax = axes[1, 0]
    for m_col, lab, c in [("carbon_delta_pct", "carbon Δ%", "C0"), ("scarcity_delta_pct", "scarcity Δ%", "C1")]:
        y = np.array([_get(agg_rows, f, "no_harm_flex", f"{m_col}_mean") for f in fleets])
        e = np.array([_get(agg_rows, f, "no_harm_flex", f"{m_col}_ci95") for f in fleets])
        ax.errorbar(x, y, yerr=e, marker="o", capsize=3, label=lab, color=c)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("No-harm flex co-benefits vs fleet size")
    ax.set_xlabel("fleet size (nodes)"); ax.set_ylabel("Δ vs packing (%)")
    ax.set_xscale("log", base=2); ax.set_xticks(x); ax.set_xticklabels([int(v) for v in x])
    ax.legend(); ax.grid(alpha=0.3)

    # Panel 4: weighted stress (flex vs control) with CIs
    ax = axes[1, 1]
    for m, lab, c in [("no_harm_flex", "no-harm flex", "C2"), ("no_harm_search_control", "search control", "C3")]:
        y = np.array([_get(agg_rows, f, m, "weighted_stress_kwh_mean") for f in fleets])
        e = np.array([_get(agg_rows, f, m, "weighted_stress_kwh_ci95") for f in fleets])
        ax.errorbar(x, y, yerr=e, marker="o", capsize=3, label=lab, color=c)
    ax.set_title("Residual weighted grid-stress vs fleet size")
    ax.set_xlabel("fleet size (nodes)"); ax.set_ylabel("weighted stress (kWh)")
    ax.set_xscale("log", base=2); ax.set_xticks(x); ax.set_xticklabels([int(v) for v in x])
    ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle(f"No-Harm Flexibility Envelope — fleet-scale behaviour (lever=both; {label})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _write_report(agg_rows, fleets, path: Path, seeds, label: str = "") -> None:
    lines = ["# Fleet-scale sweep (headline)\n",
             f"Seeds: {seeds} · regions: {N_REGIONS} · lever=both · {label} · "
             "time-aligned EU signals · perfect foresight.\n",
             "All values are mean ± 95% CI across seeds.\n",
             "## No-harm certificate rate (the guarantee)\n",
             "| fleet | no-harm flex | carbon-only | water-only |",
             "|---|---|---|---|"]
    for f in fleets:
        lines.append(f"| {f} | {100*_get(agg_rows,f,'no_harm_flex','no_harm_mean'):.0f}% "
                     f"| {100*_get(agg_rows,f,'carbon','no_harm_mean'):.0f}% "
                     f"| {100*_get(agg_rows,f,'water_scarcity','no_harm_mean'):.0f}% |")
    lines += ["\n## No-harm flex: co-benefits + stress relief\n",
              "| fleet | carbon Δ% | scarcity Δ% | stress avoided (kWh) | repairs |",
              "|---|---|---|---|---|"]
    for f in fleets:
        lines.append(
            f"| {f} | {_get(agg_rows,f,'no_harm_flex','carbon_delta_pct_mean'):.2f}±{_get(agg_rows,f,'no_harm_flex','carbon_delta_pct_ci95'):.2f} "
            f"| {_get(agg_rows,f,'no_harm_flex','scarcity_delta_pct_mean'):.3f}±{_get(agg_rows,f,'no_harm_flex','scarcity_delta_pct_ci95'):.3f} "
            f"| {_get(agg_rows,f,'no_harm_flex','stress_kwh_avoided_mean'):.4f}±{_get(agg_rows,f,'no_harm_flex','stress_kwh_avoided_ci95'):.4f} "
            f"| {_get(agg_rows,f,'no_harm_flex','repairs_applied_mean'):.1f} |")
    lines += ["\n## Signal advantage: stress-aware flex vs same-budget search control\n",
              "| fleet | flex-wins rate | weighted-stress relief gap % | abs. gap (kWh) |",
              "|---|---|---|---|"]
    for f in fleets:
        lines.append(
            f"| {f} | {100*_get(agg_rows,f,'flex_vs_control','no_harm_mean'):.0f}% "
            f"| {_get(agg_rows,f,'flex_vs_control','weighted_stress_kwh_mean'):.2f} "
            f"| {_get(agg_rows,f,'flex_vs_control','stress_kwh_avoided_mean'):.4f} |")
    lines += ["\n_Figure: `fleet_scale.png`._\n"]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes-per-region", default="1,2,4,8")
    ap.add_argument("--pods", default="40,80,160,320")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--timeslots", type=int, default=24)
    ap.add_argument("--max-timeslots", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--run-name", default="t2_fleet_scale")
    ap.add_argument("--signals-dir", default=None,
                    help="time-aligned signal dir (default pkg/.../data/timealigned, the 2018 stress year)")
    ap.add_argument("--scenario", default="heatwave-drought", choices=["heatwave-drought", "observed-winter"])
    args = ap.parse_args()
    import logging
    logging.disable(logging.CRITICAL)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
