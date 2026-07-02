#!/usr/bin/env python3
"""Regime map of the certified no-harm benefit on the real Alibaba GPU v2020 fleet.

One knob varied at a time (>=2 seeds each), holding everything else at a fixed
reference cell. For each (knob, value, seed) it runs the full no-harm pilot and
records: envelope certified? grid relief (weighted-stress avoided %), carbon %,
water %, repairs, AND how many of the 4 action baselines certify vs harm.

Knobs:
  oversub     -- target pods / GPU capacity (load factor)
  flex_frac   -- flexible-pod fraction (promote firm->flexible to hit target)
  slack       -- fixed deferral slack (hours) for every flexible job
  signals     -- spatial CI-arbitrage / window (compare signal dirs)
  window      -- trace window start hours (workload mix over calendar)

Reference cell (overridable): oversub=0.7, flex_frac=trace(~0.5), slack=trace-dist,
signals=timealigned_realci, window=672, nodes_per_region=4, 200 pods scaled to oversub.

Outputs under experiments/regime_<name>/: tidy.csv (per seed), cells.csv (per cell
mean+ci), and a console digest. Does NOT modify the engine or committed signals.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "real_traces"))

import alibaba_common as ac  # noqa: E402
import convert_alibaba_gpu_v2020 as conv  # noqa: E402
from carbon_aware.no_harm_flexibility import run_no_harm_flexibility_pilot  # noqa: E402

ACTION_BASELINES = ["carbon", "water_scarcity", "waterwise"]
ALL_METHODS = ["no_harm_flex", "no_harm_search_control"] + ACTION_BASELINES + ["packing"]


# ----------------------------------------------------------------- window builder w/ knobs
def build_window_regime(out_dir: Path, target_pods: int, seed: int, *, window_start_hours: float,
                        fixed_slack: int, flex_frac: Optional[float], arrival_slots: int = 24,
                        horizon_hours: int = 48) -> Dict:
    """Like ac.build_window but additionally supports a flexible-fraction override.

    flex_frac=None  -> keep the trace-derived tiers (role + censored).
    flex_frac=f     -> set EXACTLY ~f of sampled pods flexible. We start from the
                       genuinely-shiftable pool (non-censored, non-interactive) and
                       promote/demote to hit the target, deterministically by seed.
                       Promoted firm jobs get the same slack treatment as flexible.
    """
    df = ac._load_window(window_start_hours, arrival_slots, horizon_hours)
    s = conv.stratified_sample(df, target_pods, seed)
    s = conv.assign_fields(s, seed, fixed_slack=fixed_slack)

    if flex_frac is not None:
        import random
        rng = random.Random(seed + 101)
        n = len(s)
        target_flex = int(round(flex_frac * n))
        idx = list(range(n))
        rng.shuffle(idx)
        # Deterministic relabel: first target_flex shuffled indices -> flexible, rest -> firm.
        tier = ["firm"] * n
        for j in idx[:target_flex]:
            tier[j] = "flexible"
        s = s.reset_index(drop=True)
        s["tier"] = tier
        # Re-derive slack consistent with the new tiers (firm=0; flexible=fixed or drawn).
        def _slack(i):
            if s["tier"].iloc[i] != "flexible":
                return 0
            return fixed_slack if fixed_slack > 0 else rng.choices(
                conv.SLACK_OPTIONS, weights=conv.SLACK_WEIGHTS)[0]
        s["slack"] = [_slack(i) for i in range(n)]
        s["deadline_slots"] = s["duration_slots"].astype(int) + s["slack"].astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)
    conv.write_workloads(s, out_dir, arrival_slots)
    n = len(s)
    n_flex = int((s["tier"] == "flexible").sum())
    n_gpu = int((s["gpu_req"] > 0).sum())
    gpu_demand = float(s["gpu_req"].sum())
    fl = s[s["tier"] == "flexible"]
    mean_flex_slack = float(fl["slack"].mean()) if len(fl) else 0.0
    mean_flex_dur = float(fl["duration_slots"].mean()) if len(fl) else 0.0
    return {
        "workloads_dir": out_dir / "workloads", "n_pods": n, "n_flexible": n_flex,
        "flexible_pct": 100.0 * n_flex / max(n, 1), "n_gpu": n_gpu, "gpu_demand": gpu_demand,
        "mean_flex_slack": mean_flex_slack, "mean_flex_dur": mean_flex_dur,
    }


@dataclass
class Cell:
    knob: str
    value: str
    oversub: float
    flex_frac: Optional[float]
    slack: int
    signals_dir: str
    window: float
    nodes_per_region: int


def run_cell(cell: Cell, seeds: List[int], out_root: Path, scenario: str,
             max_timeslots: int) -> List[Dict]:
    signals = (ac.REPO_ROOT / "pkg" / "carbon-aware" / "data" / cell.signals_dir).resolve()
    nodes_file, n_nodes, n_gpus = ac.build_fleet(
        out_root / "fleets" / f"npr{cell.nodes_per_region}_w{cell.window:g}", cell.nodes_per_region)
    target_pods = max(1, int(round(n_gpus * cell.oversub / 0.68)))
    rows: List[Dict] = []
    for seed in seeds:
        case = out_root / "inputs" / f"{cell.knob}_{cell.value}_s{seed}"
        w = build_window_regime(case, target_pods=target_pods, seed=seed,
                                window_start_hours=cell.window, fixed_slack=cell.slack,
                                flex_frac=cell.flex_frac)
        oversub_real = w["gpu_demand"] / n_gpus if n_gpus else float("nan")
        pc = ac.make_pilot_config(
            nodes_file, w["workloads_dir"], out_root / "cases" / f"{cell.knob}_{cell.value}_s{seed}",
            signals=signals, max_timeslots=max_timeslots, scenario=scenario, lever_mode="both")
        res = run_no_harm_flexibility_pilot(pc)
        rm = {r["method_key"]: r for r in res["summary_rows"]}
        flex = rm["no_harm_flex"]
        n_base_cert = sum(int(rm[m]["no_harm_certificate"]) for m in ACTION_BASELINES)
        n_base_harm = len(ACTION_BASELINES) - n_base_cert
        row = {
            "knob": cell.knob, "value": cell.value, "seed": seed,
            "oversub_target": cell.oversub, "oversub_real": round(oversub_real, 3),
            "flex_frac_target": cell.flex_frac if cell.flex_frac is not None else "trace",
            "flexible_pct": round(w["flexible_pct"], 1), "slack": cell.slack,
            "mean_flex_slack": round(w["mean_flex_slack"], 2),
            "mean_flex_dur": round(w["mean_flex_dur"], 2),
            "signals": cell.signals_dir, "window": cell.window, "npr": cell.nodes_per_region,
            "gpus": n_gpus, "pods": w["n_pods"], "placed": flex["placed_pods"],
            "certified": int(flex["no_harm_certificate"]),
            "grid_relief_pct": round(flex["weighted_stress_kwh_avoided_pct"], 3),
            "carbon_pct": round(flex["carbon_delta_pct"], 3),
            "water_pct": round(flex["scarcity_delta_pct"], 3),
            "stress_kwh_avoided": round(flex["stress_kwh_avoided"], 3),
            "repairs": flex["repairs_applied"],
            "carbon_cert": int(rm["carbon"]["no_harm_certificate"]),
            "water_cert": int(rm["water_scarcity"]["no_harm_certificate"]),
            "waterwise_cert": int(rm["waterwise"]["no_harm_certificate"]),
            "ctrl_cert": int(rm["no_harm_search_control"]["no_harm_certificate"]),
            "n_baselines_certify": n_base_cert, "n_baselines_harm": n_base_harm,
        }
        rows.append(row)
        print(f"  [{cell.knob}={cell.value} s{seed}] flex%={row['flexible_pct']} "
              f"oversub={row['oversub_real']} cert={row['certified']} "
              f"grid={row['grid_relief_pct']:+.2f}% C={row['carbon_pct']:+.2f}% "
              f"W={row['water_pct']:+.2f}% repairs={row['repairs']} "
              f"base_harm={n_base_harm}/3", flush=True)
    return rows


def aggregate(rows: List[Dict]) -> List[Dict]:
    cells = {}
    for r in rows:
        cells.setdefault((r["knob"], r["value"]), []).append(r)
    out = []
    for (knob, value), sub in cells.items():
        def m(k):
            v = [x[k] for x in sub]
            return float(np.mean(v))
        def c(k):
            v = [x[k] for x in sub]
            return 1.96 * np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        out.append({
            "knob": knob, "value": value, "n": len(sub),
            "flexible_pct": round(m("flexible_pct"), 1),
            "oversub_real": round(m("oversub_real"), 3),
            "mean_flex_slack": round(m("mean_flex_slack"), 2),
            "cert_rate": round(m("certified"), 2),
            "grid_relief_pct_mean": round(m("grid_relief_pct"), 3),
            "grid_relief_pct_ci": round(c("grid_relief_pct"), 3),
            "carbon_pct_mean": round(m("carbon_pct"), 3),
            "carbon_pct_ci": round(c("carbon_pct"), 3),
            "water_pct_mean": round(m("water_pct"), 3),
            "water_pct_ci": round(c("water_pct"), 3),
            "stress_kwh_avoided_mean": round(m("stress_kwh_avoided"), 3),
            "repairs_mean": round(m("repairs"), 1),
            "n_baselines_harm_mean": round(m("n_baselines_harm"), 2),
        })
    return out


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--knob", required=True,
                    choices=["oversub", "flex_frac", "slack", "signals", "window"])
    ap.add_argument("--values", required=True, help="comma-separated knob values")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--run-name", required=True)
    # reference cell
    ap.add_argument("--ref-oversub", type=float, default=0.7)
    ap.add_argument("--ref-flex-frac", default="trace", help="'trace' or a float")
    ap.add_argument("--ref-slack", type=int, default=0, help="0 = trace slack distribution")
    ap.add_argument("--ref-signals", default="timealigned_realci")
    ap.add_argument("--ref-window", type=float, default=672.0)
    ap.add_argument("--ref-npr", type=int, default=4)
    ap.add_argument("--scenario", default="heatwave-drought")
    ap.add_argument("--max-timeslots", type=int, default=48)
    args = ap.parse_args()
    import logging
    logging.disable(logging.CRITICAL)

    seeds = ac.ints(args.seeds)
    out_root = ac.REPO_ROOT / "experiments" / f"regime_{args.run_name}"
    out_root.mkdir(parents=True, exist_ok=True)

    def parse_ff(v):
        return None if str(v) == "trace" else float(v)

    ref = dict(oversub=args.ref_oversub, flex_frac=parse_ff(args.ref_flex_frac),
               slack=args.ref_slack, signals_dir=args.ref_signals, window=args.ref_window,
               nodes_per_region=args.ref_npr)

    values = [v.strip() for v in args.values.split(",") if v.strip()]
    all_rows: List[Dict] = []
    print(f"=== regime sweep: knob={args.knob} values={values} seeds={seeds} ===")
    print(f"ref cell: {ref}\n")
    for v in values:
        cell_kw = dict(ref)
        if args.knob == "oversub":
            cell_kw["oversub"] = float(v)
        elif args.knob == "flex_frac":
            cell_kw["flex_frac"] = parse_ff(v)
        elif args.knob == "slack":
            cell_kw["slack"] = int(v)
        elif args.knob == "signals":
            cell_kw["signals_dir"] = v
        elif args.knob == "window":
            cell_kw["window"] = float(v)
        cell = Cell(knob=args.knob, value=str(v), **cell_kw)
        all_rows += run_cell(cell, seeds, out_root, args.scenario, args.max_timeslots)

    write_csv(out_root / "tidy.csv", all_rows)
    agg = aggregate(all_rows)
    write_csv(out_root / "cells.csv", agg)
    print(f"\n=== {args.knob} cell summary ===")
    for a in agg:
        print(f"  {a['value']:>12}: flex%={a['flexible_pct']:>5} oversub={a['oversub_real']:>5} "
              f"cert={a['cert_rate']:.2f} grid={a['grid_relief_pct_mean']:+.2f}%"
              f"(±{a['grid_relief_pct_ci']:.2f}) C={a['carbon_pct_mean']:+.2f}% "
              f"W={a['water_pct_mean']:+.2f}% repairs={a['repairs_mean']:.0f} "
              f"base_harm={a['n_baselines_harm_mean']:.1f}/3")
    print(f"\noutput_dir={out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
