#!/usr/bin/env python3
"""Delivery certificate (Tier-1-D / R2) -- a post-hoc relief WITNESS on the residual-load axis.

Answers reviewer minor-concern mc1 (soundness vs delivery): the no-harm certificate is SOUND (never
falsely certifies carbon/water harm) but says nothing about whether the promised grid relief was
actually DELIVERED. This script certifies delivery, with NO engine change.

Key observation. The relief axis is decided on the OBSERVABLE residual-load nowcast (measured, not
forecast), so delivery on the relief axis is itself nearly forecast-free and can be certified EX-POST
by the same checker, by symmetry with carbon/water. The engine already computes, ex-post on the
realized residual-load signal, the relief witness
    Delta_relief = R(x) - R(B)   (weighted_stress_kwh_avoided; binary projection: stress_kwh_avoided)
where R is the residual-load-weighted stress load. We read it off summary.csv and emit a DELIVERY
CERTIFICATE: Delta_relief >= 0 (relief delivered) with its magnitude.

It also sweeps the carbon buffer to trace the SOUNDNESS<->DELIVERY FRONTIER: a bigger buffer protects
the carbon certificate but blocks relieving moves, so it REDUCES delivered relief. Over-buffering is
itself a grid harm (withheld relief). For each buffer arm we report the expected-delivery bound
    E[relief] >= (1-eps)*planned - eps*tail
where `planned` is the no-noise delivered relief, `eps` the target violation rate, and `tail` the
worst-case relief lost on a fallen window (empirically, the relief of the lowest-relief noised trial).

Usage:
    PYTHONPATH=pkg/carbon-aware/server-python python scripts/delivery_certificate.py \
        --noises 0,0.3 --seeds 0,1,2 --rho 0.237 --eps 0.05
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List

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


def _floats(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _ints(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _run_cell(*, out_dir: Path, paths, config_file, noise, seed, forecast_seed,
              max_timeslots, robust_mode, rho, eps, shape, gamma, gamma_mode) -> Dict:
    kwargs = dict(
        repo_root=REPO_ROOT, nodes_file=paths["nodes_file"], workloads_dir=paths["workloads_dir"],
        forecasts_file=TIMEALIGNED / "forecasts.json", config_file=config_file,
        output_dir=out_dir, max_timeslots=max_timeslots, max_pods=None, scenario="heatwave-drought",
        lever_mode="both", forecast_noise=noise, forecast_seed=forecast_seed,
        grid_signal_csv=TIMEALIGNED / "grid_residual_region_slot.csv",
        wue_csv=TIMEALIGNED / "wue_region_slot.csv",
        robust_buffer_mode=robust_mode, robust_buffer_rho=rho,
        robust_buffer_epsilon=eps, robust_buffer_shape=shape,
        robust_buffer_gamma=gamma, robust_buffer_gamma_mode=gamma_mode,
    )
    res = run_no_harm_flexibility_pilot(PilotConfig(**kwargs))
    flex = {r["method_key"]: r for r in res["summary_rows"]}["no_harm_flex"]
    return {
        # Delivery witnesses (ex-post, realized residual-load axis):
        "delta_relief": flex["weighted_stress_kwh_avoided"],   # continuous R(x)-R(B)
        "delta_relief_binary": flex["stress_kwh_avoided"],     # binary projection
        "delivered": int(flex["weighted_stress_kwh_avoided"] > 1e-12),
        # Soundness (carbon/water no-harm certificate, ex-post):
        "no_harm": int(bool(flex["no_harm_certificate"])),
        "carbon_delta_pct": flex["carbon_delta_pct"],
        "repairs_applied": flex["repairs_applied"],
    }


def run(args) -> Path:
    out_root = REPO_ROOT / "experiments" / "robustness_dro" / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)
    config_file = REPO_ROOT / "pkg" / "carbon-aware" / "infra-workload-config.yaml"
    base_config, gen_nodes, gen_ts = load_generator_dependencies(REPO_ROOT, config_file)

    noises = _floats(args.noises)
    seeds = _ints(args.seeds)
    fleet = args.nodes_per_region * N_REGIONS

    # Arms. Default: the soundness<->delivery FRONTIER -- sweep the DRO correlation budget Gamma from
    # sqrt(K) (c=0, independent) to K (c=1, fully correlated) at fixed rho, eps. As Gamma grows the
    # reserve grows, so relieving moves get blocked and delivered relief falls: the frontier. The `off`
    # arm anchors the no-buffer baseline. Pass --arms reserve-arms to instead compare off/sqrtk/dro.
    if args.arms == "frontier":
        arms: List[Dict] = [{"key": "off", "mode": "flat", "gamma": 0.0}]
        for c in _floats(args.gamma_fracs):
            arms.append({"key": f"dro_g{c}", "mode": "dro", "gamma": c})
    else:
        arms = [
            {"key": "off", "mode": "flat", "gamma": 0.0},
            {"key": "sqrtk", "mode": "sqrtk", "gamma": 0.0},
            {"key": "dro_corr", "mode": "dro", "gamma": args.gamma_frac},
        ]

    case_paths = {}
    for seed in seeds:
        case = out_root / "inputs" / f"s{seed}"
        case_paths[seed] = _generate_case_inputs(
            base_config=base_config, generate_nodes_file=gen_nodes, generate_timeslot_files=gen_ts,
            input_dir=case, pod_count=args.pods, seed=seed, timeslots=args.timeslots,
            config_file=config_file, deadline_flex_hours=args.horizon, nodes_per_region=args.nodes_per_region,
            server_only=True,
        )

    tidy: List[Dict] = []
    for arm in arms:
        for noise in noises:
            for seed in seeds:
                cell = _run_cell(
                    out_dir=out_root / "cases" / f"{arm['key']}_n{noise}_s{seed}",
                    paths=case_paths[seed], config_file=config_file, noise=noise, seed=seed,
                    forecast_seed=args.forecast_seed + seed, max_timeslots=args.max_timeslots,
                    robust_mode=arm["mode"], rho=args.rho, eps=args.eps, shape=args.shape,
                    gamma=arm["gamma"], gamma_mode="frac",
                )
                cell.update({"arm": arm["key"], "noise": noise, "seed": seed})
                tidy.append(cell)
            sub = [t for t in tidy if t["arm"] == arm["key"] and t["noise"] == noise]
            print(f"  arm={arm['key']:<9} noise={noise:<4}: "
                  f"delivered={100*np.mean([t['delivered'] for t in sub]):4.0f}%  "
                  f"relief={np.mean([t['delta_relief'] for t in sub]):.4f}  "
                  f"hold={100*np.mean([t['no_harm'] for t in sub]):4.0f}%", flush=True)

    tidy_path = out_root / "delivery_tidy.csv"
    with tidy_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tidy[0].keys())); w.writeheader(); w.writerows(tidy)

    # Expected-delivery bound per arm: planned = mean delivered relief at noise=0; tail = worst-case
    # relief of a fallen (noised) window; bound = (1-eps)*planned - eps*tail.
    eps = args.eps
    frontier: List[Dict] = []
    for arm in arms:
        zero = [t["delta_relief"] for t in tidy if t["arm"] == arm["key"] and t["noise"] == 0.0]
        noised = [t for t in tidy if t["arm"] == arm["key"] and t["noise"] > 0.0]
        planned = float(np.mean(zero)) if zero else 0.0
        # tail = worst-case (smallest) realized relief among noised trials that FAILED the certificate;
        # if none failed, the tail is the smallest realized relief overall (conservative).
        failed = [t["delta_relief"] for t in noised if not t["no_harm"]]
        tail = float(min(failed)) if failed else (float(min(t["delta_relief"] for t in noised)) if noised else 0.0)
        realized = float(np.mean([t["delta_relief"] for t in noised])) if noised else planned
        bound = (1.0 - eps) * planned - eps * tail
        frontier.append({
            "arm": arm["key"], "planned_relief": planned, "tail_relief": tail,
            "expected_bound": bound, "realized_mean_relief": realized,
            "hold_rate": float(np.mean([t["no_harm"] for t in (noised or zero)])),
            "delivered_rate": float(np.mean([t["delivered"] for t in (noised or zero)])),
            "bound_holds": int(realized >= bound - 1e-9),
        })
    frontier_path = out_root / "delivery_frontier.csv"
    with frontier_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(frontier[0].keys())); w.writeheader(); w.writerows(frontier)

    _report(tidy, frontier, noises, arms, out_root / "delivery_report.md", seeds, fleet, eps)
    print(f"\noutput_dir={out_root}\nfrontier={frontier_path}")
    for f in frontier:
        print(f"  {f['arm']:<9} planned={f['planned_relief']:.4f} tail={f['tail_relief']:.4f} "
              f"bound={f['expected_bound']:.4f} realized={f['realized_mean_relief']:.4f} "
              f"bound_holds={f['bound_holds']}")
    return out_root


def _mean(tidy, arm, noise, col):
    sub = [t[col] for t in tidy if t["arm"] == arm and t["noise"] == noise]
    return float(np.mean(sub)) if sub else float("nan")


def _report(tidy, frontier, noises, arms, path, seeds, fleet, eps) -> None:
    arm_keys = [a["key"] for a in arms]
    L = [f"# Delivery certificate + soundness<->delivery frontier, {fleet}-node fleet\n",
         f"Seeds: {seeds} | heatwave-drought | time-aligned 2018 EU window | eps={eps}.\n",
         "**Delivery certificate.** The relief axis is decided on the OBSERVABLE residual-load nowcast "
         "(measured, not forecast), so delivery is certified EX-POST by the same checker as carbon/water. "
         "`Delta_relief = R(x) - R(B) >= 0` is the delivery witness (continuous, residual-load-weighted).\n",
         "\n## Delivered relief Delta_relief = R(x)-R(B) (ex-post, realized residual load)\n",
         "| noise \\ arm | " + " | ".join(arm_keys) + " |", "|" + "---|" * (len(arm_keys) + 1)]
    for n in noises:
        L.append(f"| {n} | " + " | ".join(f"{_mean(tidy, a, n, 'delta_relief'):.4f}" for a in arm_keys) + " |")
    L += ["\n## No-harm certificate hold-rate (%) -- the soundness axis\n",
          "| noise \\ arm | " + " | ".join(arm_keys) + " |", "|" + "---|" * (len(arm_keys) + 1)]
    for n in noises:
        L.append(f"| {n} | " + " | ".join(f"{100*_mean(tidy, a, n, 'no_harm'):.0f}" for a in arm_keys) + " |")
    L += ["\n## Soundness<->delivery frontier (expected-delivery bound E[relief] >= (1-eps)*planned - eps*tail)\n",
          "| arm | planned | tail | E[relief] bound | realized | hold-rate | bound holds |",
          "|---|---|---|---|---|---|---|"]
    for f in frontier:
        L.append(f"| {f['arm']} | {f['planned_relief']:.4f} | {f['tail_relief']:.4f} | "
                 f"{f['expected_bound']:.4f} | {f['realized_mean_relief']:.4f} | "
                 f"{100*f['hold_rate']:.0f}% | {'yes' if f['bound_holds'] else 'NO'} |")
    L += ["\n_The frontier is the point: a bigger buffer raises the no-harm hold-rate (soundness) but "
          "LOWERS delivered relief, because relieving moves get blocked. Over-buffering withholds grid "
          "relief, which is itself a grid harm -- so the buffer must be SIZED to the target eps, not "
          "maximized. Delivery becomes auditable ex-post (the certificate); the (1-eps) bound makes the "
          "prospective delivery shortfall statable._\n"]
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noises", default="0,0.3")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--rho", type=float, default=0.237)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--shape", type=float, default=1.0)
    ap.add_argument("--gamma-frac", type=float, default=1.0)
    ap.add_argument("--arms", choices=["frontier", "reserve-arms"], default="frontier",
                    help="frontier = sweep Gamma fraction (soundness<->delivery frontier); "
                         "reserve-arms = compare off/sqrtk/dro_corr")
    ap.add_argument("--gamma-fracs", default="0,0.25,0.5,0.75,1.0",
                    help="Gamma correlation fractions swept in frontier mode")
    ap.add_argument("--nodes-per-region", type=int, default=4)
    ap.add_argument("--pods", type=int, default=160)
    ap.add_argument("--timeslots", type=int, default=24)
    ap.add_argument("--max-timeslots", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--forecast-seed", type=int, default=100)
    ap.add_argument("--run-name", default="delivery_certificate")
    args = ap.parse_args()
    import logging
    logging.disable(logging.CRITICAL)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
