#!/usr/bin/env python3
"""T5 — Forecast robustness sweep (RQ3): break, regret-inertness, and robust recovery.

Decisions are made on a noised carbon forecast (independent draw per trial); the no-harm
certificate is verified on the REALIZED signals (footprints re-materialised at the chosen
sites/slots). This sweep maps the no-harm rate, realized carbon, and stress relief vs:
  * forecast noise x regret margin  (robust=0)  -> shows the break + that regret is inert;
  * forecast noise x robust buffer  (regret=0)  -> shows the robust guard recovering carbon.
The robust buffer is a cumulative worst-case carbon-forecast-exposure guard (see engine).

Example:
    python scripts/forecast_regret_sweep.py --noises 0,0.15,0.3,0.5 --regrets 0,0.5,1,2 \
        --robust-buffers 0,0.5,1,1.5 --seeds 0,1,2,3,4 --nodes-per-region 4 --pods 160
"""
from __future__ import annotations

import argparse
import csv
import math
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


def _ci95(vals) -> float:
    a = np.asarray(vals, dtype=float)
    if a.size < 2:
        return 0.0
    return 1.96 * a.std(ddof=1) / math.sqrt(a.size)


def run(args) -> Path:
    out_root = REPO_ROOT / "experiments" / "flexibility" / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)
    config_file = REPO_ROOT / "pkg" / "carbon-aware" / "infra-workload-config.yaml"
    base_config, gen_nodes, gen_ts = load_generator_dependencies(REPO_ROOT, config_file)

    noises = _floats(args.noises)
    regrets = _floats(args.regrets)
    robusts = _floats(args.robust_buffers)
    seeds = _ints(args.seeds)
    fleet = args.nodes_per_region * N_REGIONS
    tidy: List[Dict] = []

    case_paths = {}
    for seed in seeds:
        case = out_root / "inputs" / f"s{seed}"
        case_paths[seed] = _generate_case_inputs(
            base_config=base_config, generate_nodes_file=gen_nodes, generate_timeslot_files=gen_ts,
            input_dir=case, pod_count=args.pods, seed=seed, timeslots=args.timeslots,
            config_file=config_file, deadline_flex_hours=args.horizon, nodes_per_region=args.nodes_per_region,
        )

    # One knob at a time: (regret, robust) with at most one non-zero (plus the 0,0 cell).
    combos = sorted({(r, 0.0) for r in regrets} | {(0.0, b) for b in robusts})

    for noise in noises:
        for (regret, robust) in combos:
            for seed in seeds:
                paths = case_paths[seed]
                pc = PilotConfig(
                    repo_root=REPO_ROOT, nodes_file=paths["nodes_file"], workloads_dir=paths["workloads_dir"],
                    forecasts_file=TIMEALIGNED / "forecasts.json", config_file=config_file,
                    output_dir=out_root / "cases" / f"n{noise}_r{regret}_b{robust}_s{seed}",
                    max_timeslots=args.max_timeslots, max_pods=None, scenario="heatwave-drought",
                    lever_mode="both", regret_margin=regret, robust_buffer=robust,
                    forecast_noise=noise, forecast_seed=args.forecast_seed + seed,
                    grid_signal_csv=TIMEALIGNED / "grid_residual_region_slot.csv", wue_csv=TIMEALIGNED / "wue_region_slot.csv",
                )
                res = run_no_harm_flexibility_pilot(pc)
                flex = {r["method_key"]: r for r in res["summary_rows"]}["no_harm_flex"]
                tidy.append({
                    "noise": noise, "regret": regret, "robust": robust, "seed": seed,
                    "no_harm": int(bool(flex["no_harm_certificate"])),
                    "carbon_delta_pct": flex["carbon_delta_pct"],
                    "scarcity_delta_pct": flex["scarcity_delta_pct"],
                    "stress_kwh_avoided": flex["stress_kwh_avoided"],
                    "repairs_applied": flex["repairs_applied"],
                })
            cell = [t for t in tidy if t["noise"] == noise and t["regret"] == regret and t["robust"] == robust]
            print(f"  noise={noise:<4} regret={regret:<4} robust={robust:<4}: "
                  f"no-harm={100*np.mean([t['no_harm'] for t in cell]):5.0f}%  "
                  f"carbon%={np.mean([t['carbon_delta_pct'] for t in cell]):+.2f}  "
                  f"stress={np.mean([t['stress_kwh_avoided'] for t in cell]):.4f}", flush=True)

    tidy_path = out_root / "tidy.csv"
    with tidy_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tidy[0].keys())); w.writeheader(); w.writerows(tidy)

    agg: List[Dict] = []
    for (noise, regret, robust) in sorted({(t["noise"], t["regret"], t["robust"]) for t in tidy}):
        cell = [t for t in tidy if t["noise"] == noise and t["regret"] == regret and t["robust"] == robust]
        row = {"noise": noise, "regret": regret, "robust": robust, "n": len(cell)}
        for m in ("no_harm", "carbon_delta_pct", "scarcity_delta_pct", "stress_kwh_avoided", "repairs_applied"):
            vals = [t[m] for t in cell]
            row[f"{m}_mean"] = float(np.mean(vals)); row[f"{m}_ci95"] = _ci95(vals)
        agg.append(row)
    agg_path = out_root / "aggregate.csv"
    with agg_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0].keys())); w.writeheader(); w.writerows(agg)

    _figure(agg, noises, regrets, robusts, out_root / "forecast_robust.png", fleet)
    _report(agg, noises, regrets, robusts, out_root / "evidence_report.md", seeds, fleet)
    print(f"\noutput_dir={out_root}\naggregate={agg_path}")
    return out_root


def _cell(agg, noise, regret, robust, col):
    for r in agg:
        if r["noise"] == noise and r["regret"] == regret and r["robust"] == robust:
            return r.get(col, float("nan"))
    return float("nan")


def _hm(ax, noises, cols, getter, title, xlabel, cmap="RdYlGn", vmin=0, vmax=100, fmt="{:.0f}"):
    grid = np.array([[getter(n, c) for c in cols] for n in noises])
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([str(c) for c in cols])
    ax.set_yticks(range(len(noises))); ax.set_yticklabels([str(n) for n in noises])
    ax.set_xlabel(xlabel); ax.set_ylabel("forecast noise"); ax.set_title(title)
    for i in range(len(noises)):
        for j in range(len(cols)):
            ax.text(j, i, fmt.format(grid[i, j]), ha="center", va="center", fontsize=8)
    return im


def _figure(agg, noises, regrets, robusts, path, fleet) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    im0 = _hm(axes[0], noises, regrets, lambda n, r: 100 * _cell(agg, n, r, 0.0, "no_harm_mean"),
              "No-harm rate (%) vs regret margin\n(regret is inert vs forecast error)", "regret margin")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = _hm(axes[1], noises, robusts, lambda n, b: 100 * _cell(agg, n, 0.0, b, "no_harm_mean"),
              "No-harm rate (%) vs robust buffer\n(robust guard recovers the guarantee)", "robust buffer")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    im2 = _hm(axes[2], noises, robusts, lambda n, b: _cell(agg, n, 0.0, b, "carbon_delta_pct_mean"),
              "Realized carbon Δ% vs robust buffer\n(<=0 = no carbon harm)", "robust buffer",
              cmap="RdYlGn_r", vmin=-5, vmax=5, fmt="{:+.1f}")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    fig.suptitle(f"RQ3 — forecast error: break, regret-inertness, robust recovery ({fleet}-node, 2018 EU)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140); plt.close(fig)


def _report(agg, noises, regrets, robusts, path, seeds, fleet) -> None:
    def tbl(title, cols, getter, fmt):
        rows = [f"\n## {title}\n", "| noise \\ | " + " | ".join(str(c) for c in cols) + " |",
                "|" + "---|" * (len(cols) + 1)]
        for n in noises:
            rows.append(f"| {n} | " + " | ".join(fmt(getter(n, c)) for c in cols) + " |")
        return rows
    L = [f"# T5 — Forecast robustness (RQ3), {fleet}-node fleet\n",
         f"Seeds: {seeds} · lever=both · heatwave-drought · time-aligned 2018 EU · decide on noised "
         "forecast (independent draw per trial), verify no-harm on realized signals.\n",
         "**(1) Break + regret inertness.** No-harm certificate rate vs (noise, regret margin), robust=0:"]
    L += tbl("No-harm rate (%) vs regret margin", regrets,
             lambda n, r: 100 * _cell(agg, n, r, 0.0, "no_harm_mean"), lambda v: f"{v:.0f}")
    L += ["\n**(2) Robust recovery.** Robust buffer (cumulative worst-case carbon-exposure guard), regret=0:"]
    L += tbl("No-harm rate (%) vs robust buffer", robusts,
             lambda n, b: 100 * _cell(agg, n, 0.0, b, "no_harm_mean"), lambda v: f"{v:.0f}")
    L += tbl("Realized carbon Δ% vs robust buffer (<=0 = no carbon harm)", robusts,
             lambda n, b: _cell(agg, n, 0.0, b, "carbon_delta_pct_mean"), lambda v: f"{v:+.2f}")
    L += tbl("Stress avoided (kWh) vs robust buffer (the cost of robustness)", robusts,
             lambda n, b: _cell(agg, n, 0.0, b, "stress_kwh_avoided_mean"), lambda v: f"{v:.4f}")
    L += ["\n_Figure: `forecast_robust.png`. The regret margin is inert (it only buffers footprint-",
          "increasing moves); the robust buffer drives realized carbon to <=0 (recovers the guarantee)",
          "but trades away stress relief — a tunable robustness/flexibility frontier. The break itself",
          "is the optimizer's curse (selecting cleanest-looking slots biases realized carbon upward),",
          "which is the core argument for deciding on OBSERVED signals rather than carbon forecasts._\n"]
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noises", default="0,0.15,0.3,0.5")
    ap.add_argument("--regrets", default="0,0.5,1,2")
    ap.add_argument("--robust-buffers", default="0,0.5,1,1.5")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--nodes-per-region", type=int, default=4)
    ap.add_argument("--pods", type=int, default=160)
    ap.add_argument("--timeslots", type=int, default=24)
    ap.add_argument("--max-timeslots", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--forecast-seed", type=int, default=100)
    ap.add_argument("--run-name", default="t5_forecast_robust")
    args = ap.parse_args()
    import logging
    logging.disable(logging.CRITICAL)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
