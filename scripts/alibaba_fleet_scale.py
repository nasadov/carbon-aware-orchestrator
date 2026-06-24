#!/usr/bin/env python3
"""Fleet-scale sweep on the REAL Alibaba GPU v2020 trace (RQ1 headline + MW-extrapolation basis).

Replaces the synthetic-generator fleet-scale sweep (fleet_scale_sweep.py), whose CPU fleet ran at
~1% utilization, with the real Alibaba GPU testbed: real job sizes + measured utilization + a scarce
binding resource (8 GPUs/node) so every fleet size is realistically utilized (~constant GPU
oversubscription, held fixed as the fleet grows so the scaling slope reflects scale, not load drift).

For each fleet size (nodes-per-region x 4 EU regions of V100 DGX-1 nodes) it scales the pod count to
hold oversubscription constant, runs the no-harm pilot for several seeds, and reports how the no-harm
certificate and the stress-aware advantage over a same-budget search control behave as the fleet
grows. Emits tidy + aggregate CSVs, a scale-invariance digest (relief per GPU-node -> MW basis), a
multi-panel figure, and a markdown report. lever=both, heatwave-drought, perfect foresight.

Example:
    python scripts/alibaba_fleet_scale.py --nodes-per-region 1,2,4,8 \
        --seeds 0,1,2,3,4 --run-name t19_alibaba_fleet_scale
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import alibaba_common as ac  # noqa: E402
from carbon_aware.no_harm_flexibility import run_no_harm_flexibility_pilot  # noqa: E402

METHODS = ["no_harm_flex", "no_harm_search_control", "carbon", "water_scarcity", "waterwise", "packing"]
METRICS = ["no_harm", "carbon_delta_pct", "scarcity_delta_pct", "stress_kwh_avoided",
           "weighted_stress_kwh", "repairs_applied", "placed_pods", "unplaced_pods"]


def run(args) -> Path:
    out_root = ac.REPO_ROOT / "experiments" / "flexibility" / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)
    signals = (ac.REPO_ROOT / args.signals_dir).resolve() if args.signals_dir else ac.TIMEALIGNED

    npr_list = ac.ints(args.nodes_per_region)
    seeds = ac.ints(args.seeds)
    pods_list = ac.ints(args.pods) if args.pods else [ac.pods_for_fleet(npr, args.oversub) for npr in npr_list]
    if len(npr_list) != len(pods_list):
        raise SystemExit("--nodes-per-region and --pods must have equal length (paired by fleet size)")

    tidy_rows: List[Dict] = []
    trace_rows: List[Dict] = []

    for npr, pods in zip(npr_list, pods_list):
        fleet = npr * ac.N_REGIONS
        nodes_file, n_nodes, n_gpus = ac.build_fleet(out_root / "fleets" / f"f{fleet}", npr)
        for seed in seeds:
            case = out_root / "inputs" / f"f{fleet}_s{seed}"
            w = ac.build_window(case, target_pods=pods, seed=seed)
            oversub = w["gpu_demand"] / n_gpus if n_gpus else float("nan")
            pc = ac.make_pilot_config(
                nodes_file, w["workloads_dir"], out_root / "cases" / f"f{fleet}_s{seed}",
                signals=signals, max_timeslots=args.max_timeslots, scenario=args.scenario,
                lever_mode="both",
            )
            res = run_no_harm_flexibility_pilot(pc)
            rows = {r["method_key"]: r for r in res["summary_rows"]}
            for m in METHODS:
                r = rows.get(m, {})
                tidy_rows.append({
                    "fleet": fleet, "gpus": n_gpus, "pods": pods, "seed": seed, "method": m,
                    "no_harm": int(bool(r.get("no_harm_certificate", False))),
                    "carbon_delta_pct": r.get("carbon_delta_pct", float("nan")),
                    "scarcity_delta_pct": r.get("scarcity_delta_pct", float("nan")),
                    "stress_kwh_avoided": r.get("stress_kwh_avoided", float("nan")),
                    "weighted_stress_kwh": r.get("weighted_stress_kwh", float("nan")),
                    "repairs_applied": r.get("repairs_applied", 0),
                    "placed_pods": r.get("placed_pods", float("nan")),
                    "unplaced_pods": r.get("unplaced_pods", float("nan")),
                })
            # Per-instance signal advantage: stress-aware flex vs same-budget control (lower residual wins).
            flex_ws = rows["no_harm_flex"]["weighted_stress_kwh"]
            ctrl_ws = rows["no_harm_search_control"]["weighted_stress_kwh"]
            tidy_rows.append({
                "fleet": fleet, "gpus": n_gpus, "pods": pods, "seed": seed, "method": "flex_vs_control",
                "no_harm": int(flex_ws < ctrl_ws - 1e-12),
                "carbon_delta_pct": float("nan"), "scarcity_delta_pct": float("nan"),
                "stress_kwh_avoided": ctrl_ws - flex_ws,
                "weighted_stress_kwh": (ctrl_ws - flex_ws) / ctrl_ws * 100.0 if ctrl_ws > 1e-12 else 0.0,
                "repairs_applied": rows["no_harm_flex"]["repairs_applied"],
                "placed_pods": float("nan"), "unplaced_pods": float("nan"),
            })
            trace_rows.append({
                "fleet": fleet, "gpus": n_gpus, "pods": pods, "seed": seed,
                "n_pods": w["n_pods"], "n_flexible": w["n_flexible"], "flexible_pct": w["flexible_pct"],
                "n_gpu": w["n_gpu"], "gpu_demand": w["gpu_demand"], "gpu_oversub": oversub,
                "mean_gpu_util": w["mean_gpu_util"], "placed": rows["no_harm_flex"]["placed_pods"],
                "stress_avoided": rows["no_harm_flex"]["stress_kwh_avoided"],
            })
            print(f"  fleet={fleet:>3} ({n_gpus} GPU) pods={pods} seed={seed}: "
                  f"flex no_harm={rows['no_harm_flex']['no_harm_certificate']} "
                  f"carbon%={rows['no_harm_flex']['carbon_delta_pct']:.2f} "
                  f"scar%={rows['no_harm_flex']['scarcity_delta_pct']:.3f} "
                  f"stress_avoid={rows['no_harm_flex']['stress_kwh_avoided']:.3f} "
                  f"oversub={oversub:.2f} flex>ctrl={'Y' if flex_ws < ctrl_ws - 1e-12 else 'n'} "
                  f"carbon_nh={rows['carbon']['no_harm_certificate']} water_nh={rows['water_scarcity']['no_harm_certificate']}",
                  flush=True)

    fleets = sorted({r["fleet"] for r in tidy_rows})
    _write_csv(out_root / "tidy.csv", tidy_rows)
    _write_csv(out_root / "trace_characterization.csv", trace_rows)
    agg_rows = _aggregate(tidy_rows, fleets)
    _write_csv(out_root / "aggregate.csv", agg_rows)
    trace_agg = _aggregate_trace(trace_rows, fleets)
    digest = _scale_digest(agg_rows, trace_agg, fleets)
    (out_root / "scale_digest.json").write_text(json.dumps(digest, indent=2))

    label = f"signals={signals.name} · scenario={args.scenario} · V100 DGX-1 GPU fleet"
    _write_figure(agg_rows, trace_agg, fleets, out_root / "fleet_scale.png", label)
    _write_report(agg_rows, trace_agg, digest, fleets, out_root / "evidence_report.md", seeds, label)
    print(f"\noutput_dir={out_root}\ndigest={out_root/'scale_digest.json'}")
    return out_root


# ---------------------------------------------------------------------------- aggregation
def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _aggregate(tidy_rows, fleets) -> List[Dict]:
    agg_methods = METHODS + ["flex_vs_control"]
    agg_rows: List[Dict] = []
    for fleet in fleets:
        for m in agg_methods:
            sub = [r for r in tidy_rows if r["fleet"] == fleet and r["method"] == m]
            row = {"fleet": fleet, "method": m, "n": len(sub)}
            for metric in METRICS:
                vals = [r[metric] for r in sub]
                row[f"{metric}_mean"] = ac.mean(vals)
                row[f"{metric}_ci95"] = ac.ci95(vals)
            agg_rows.append(row)
    return agg_rows


def _aggregate_trace(trace_rows, fleets) -> Dict[int, Dict]:
    out = {}
    cols = ["flexible_pct", "gpu_oversub", "mean_gpu_util", "n_pods", "placed", "stress_avoided", "gpus"]
    for fleet in fleets:
        sub = [r for r in trace_rows if r["fleet"] == fleet]
        out[fleet] = {c: ac.mean([r[c] for r in sub]) for c in cols}
        out[fleet]["gpus"] = sub[0]["gpus"] if sub else float("nan")
    return out


def _get(agg_rows, fleet, method, col):
    for r in agg_rows:
        if r["fleet"] == fleet and r["method"] == method:
            return r.get(col, float("nan"))
    return float("nan")


def _scale_digest(agg_rows, trace_agg, fleets) -> Dict:
    """Scale-invariance check + per-GPU relief intensity (the bridge to a MW extrapolation):
    if stress relief scales ~linearly with fleet GPUs, relief-per-GPU is stable and can be
    multiplied up to a MW-class GPU fleet (paired with the Chen & Zheng grid-value range)."""
    per_gpu = []
    for f in fleets:
        relief = _get(agg_rows, f, "no_harm_flex", "stress_kwh_avoided_mean")
        gpus = trace_agg[f]["gpus"]
        per_gpu.append({"fleet": f, "gpus": gpus, "stress_avoided_kwh": relief,
                        "stress_avoided_per_gpu_kwh": relief / gpus if gpus else float("nan"),
                        "no_harm_rate": _get(agg_rows, f, "no_harm_flex", "no_harm_mean"),
                        "flexible_pct": trace_agg[f]["flexible_pct"],
                        "gpu_oversub": trace_agg[f]["gpu_oversub"]})
    pg = [p["stress_avoided_per_gpu_kwh"] for p in per_gpu if not np.isnan(p["stress_avoided_per_gpu_kwh"])]
    # DGX-1 V100 ~ 3.5 kW system; a 1 MW GPU hall ~ 1000/3.5 ~ 286 nodes ~ 2286 GPUs.
    gpus_per_mw = 1000.0 / (ac.GPU_NODE["max_w"] / 1000.0 + ac.GPUS_PER_NODE * ac.GPU_NODE["gpu_power_w"] / 1000.0)
    gpus_per_mw *= ac.GPUS_PER_NODE
    mean_per_gpu = float(np.mean(pg)) if pg else float("nan")
    return {
        "per_gpu_relief": per_gpu,
        "mean_stress_avoided_per_gpu_kwh": mean_per_gpu,
        "cv_per_gpu": float(np.std(pg) / np.mean(pg)) if pg and np.mean(pg) else float("nan"),
        "gpus_per_mw_estimate": gpus_per_mw,
        "extrapolated_mw_window_relief_kwh": mean_per_gpu * gpus_per_mw if not np.isnan(mean_per_gpu) else float("nan"),
        "note": ("Relief-per-GPU stability (low CV) supports a scale-invariant extrapolation; the MW "
                 "window-relief is per the sweep's scheduling window, to be paired with Chen & Zheng's "
                 "3-21% grid-value range, NOT reported as a literal large-fleet simulation."),
    }


# ---------------------------------------------------------------------------- figure + report
def _write_figure(agg_rows, trace_agg, fleets, path: Path, label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array(fleets, dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    for m, lab in [("no_harm_flex", "no-harm flex"), ("carbon", "carbon-only"),
                   ("water_scarcity", "water-only"), ("waterwise", "WaterWise")]:
        y = [100 * _get(agg_rows, f, m, "no_harm_mean") for f in fleets]
        ax.plot(x, y, marker="o", label=lab)
    ax.set_title("No-harm certificate rate vs fleet size")
    ax.set_xlabel("fleet size (DGX-1 nodes)"); ax.set_ylabel("% configs certified no-harm")
    ax.set_ylim(-5, 105); ax.set_xscale("log", base=2); ax.set_xticks(x); ax.set_xticklabels([int(v) for v in x])
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    win = [100 * _get(agg_rows, f, "flex_vs_control", "no_harm_mean") for f in fleets]
    adv = [_get(agg_rows, f, "flex_vs_control", "weighted_stress_kwh_mean") for f in fleets]
    ax.plot(x, win, marker="o", color="C2", label="flex wins vs control (%)")
    ax.plot(x, adv, marker="s", color="C3", label="weighted-stress relief gap (%)")
    ax.set_title("Stress-aware signal advantage vs fleet size")
    ax.set_xlabel("fleet size (DGX-1 nodes)"); ax.set_ylabel("%")
    ax.set_xscale("log", base=2); ax.set_xticks(x); ax.set_xticklabels([int(v) for v in x])
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for m_col, lab, c in [("carbon_delta_pct", "carbon Δ%", "C0"), ("scarcity_delta_pct", "scarcity Δ%", "C1")]:
        y = np.array([_get(agg_rows, f, "no_harm_flex", f"{m_col}_mean") for f in fleets])
        e = np.array([_get(agg_rows, f, "no_harm_flex", f"{m_col}_ci95") for f in fleets])
        ax.errorbar(x, y, yerr=e, marker="o", capsize=3, label=lab, color=c)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("No-harm flex co-benefits vs fleet size")
    ax.set_xlabel("fleet size (DGX-1 nodes)"); ax.set_ylabel("Δ vs packing (%)")
    ax.set_xscale("log", base=2); ax.set_xticks(x); ax.set_xticklabels([int(v) for v in x])
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    relief = np.array([_get(agg_rows, f, "no_harm_flex", "stress_kwh_avoided_mean") for f in fleets])
    err = np.array([_get(agg_rows, f, "no_harm_flex", "stress_kwh_avoided_ci95") for f in fleets])
    ax.errorbar(x, relief, yerr=err, marker="o", capsize=3, color="C2", label="grid-stress avoided (kWh)")
    ax2 = ax.twinx()
    ov = [trace_agg[f]["gpu_oversub"] for f in fleets]
    fx = [trace_agg[f]["flexible_pct"] for f in fleets]
    ax2.plot(x, ov, marker="^", ls="--", color="C4", label="GPU oversub (held ~const)")
    ax2.plot(x, fx, marker="v", ls=":", color="C5", label="flexible %")
    ax.set_title("Grid-stress relief scales with fleet; load factor held constant")
    ax.set_xlabel("fleet size (DGX-1 nodes)"); ax.set_ylabel("grid-stress avoided (kWh)")
    ax2.set_ylabel("ratio / %")
    ax.set_xscale("log", base=2); ax.set_xticks(x); ax.set_xticklabels([int(v) for v in x])
    lines, labs = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labs + lb2, fontsize=7, loc="upper left"); ax.grid(alpha=0.3)

    fig.suptitle(f"No-Harm Flexibility Envelope — fleet-scale on real Alibaba GPU v2020 (lever=both; {label})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _write_report(agg_rows, trace_agg, digest, fleets, path: Path, seeds, label) -> None:
    L = [f"# Fleet-scale sweep — real Alibaba GPU v2020 (RQ1 headline + MW basis)\n",
         f"Seeds: {seeds} · {ac.N_REGIONS} EU regions of V100 DGX-1 nodes · lever=both · {label} · "
         "perfect foresight. All values mean ± 95% CI across seeds.\n",
         "Testbed realism: real job sizes + measured utilization (R2); pod count scaled with fleet to "
         "hold GPU oversubscription ~constant, so the scaling slope reflects scale, not load drift.\n",
         "## Testbed characterization (per fleet size)\n",
         "| fleet (nodes) | GPUs | pods | flexible % | GPU oversub | mean GPU util | placed |",
         "|---|---|---|---|---|---|---|"]
    for f in fleets:
        t = trace_agg[f]
        L.append(f"| {f} | {int(t['gpus'])} | {t['n_pods']:.0f} | {t['flexible_pct']:.0f}% | "
                 f"{t['gpu_oversub']:.2f} | {t['mean_gpu_util']:.2f} | {t['placed']:.0f} |")
    L += ["\n## No-harm certificate rate (the guarantee)\n",
          "| fleet | no-harm flex | carbon-only | water-only | WaterWise |",
          "|---|---|---|---|---|"]
    for f in fleets:
        L.append(f"| {f} | {100*_get(agg_rows,f,'no_harm_flex','no_harm_mean'):.0f}% "
                 f"| {100*_get(agg_rows,f,'carbon','no_harm_mean'):.0f}% "
                 f"| {100*_get(agg_rows,f,'water_scarcity','no_harm_mean'):.0f}% "
                 f"| {100*_get(agg_rows,f,'waterwise','no_harm_mean'):.0f}% |")
    L += ["\n## No-harm flex: co-benefits + stress relief\n",
          "| fleet | carbon Δ% | scarcity Δ% | stress avoided (kWh) | repairs |",
          "|---|---|---|---|---|"]
    for f in fleets:
        L.append(
            f"| {f} | {_get(agg_rows,f,'no_harm_flex','carbon_delta_pct_mean'):.2f}±{_get(agg_rows,f,'no_harm_flex','carbon_delta_pct_ci95'):.2f} "
            f"| {_get(agg_rows,f,'no_harm_flex','scarcity_delta_pct_mean'):.3f}±{_get(agg_rows,f,'no_harm_flex','scarcity_delta_pct_ci95'):.3f} "
            f"| {_get(agg_rows,f,'no_harm_flex','stress_kwh_avoided_mean'):.3f}±{_get(agg_rows,f,'no_harm_flex','stress_kwh_avoided_ci95'):.3f} "
            f"| {_get(agg_rows,f,'no_harm_flex','repairs_applied_mean'):.1f} |")
    L += ["\n## Signal advantage: stress-aware flex vs same-budget search control\n",
          "| fleet | flex-wins rate | weighted-stress relief gap % | abs. gap (kWh) |",
          "|---|---|---|---|"]
    for f in fleets:
        L.append(
            f"| {f} | {100*_get(agg_rows,f,'flex_vs_control','no_harm_mean'):.0f}% "
            f"| {_get(agg_rows,f,'flex_vs_control','weighted_stress_kwh_mean'):.2f} "
            f"| {_get(agg_rows,f,'flex_vs_control','stress_kwh_avoided_mean'):.3f} |")
    L += ["\n## Scale invariance -> MW basis\n",
          f"Relief per GPU: **{digest['mean_stress_avoided_per_gpu_kwh']:.4f} kWh** "
          f"(CV {digest['cv_per_gpu']:.2f} across fleet sizes). "
          f"At ~{digest['gpus_per_mw_estimate']:.0f} GPUs/MW (V100 DGX-1 system power), a MW-class GPU hall "
          f"relieves ~**{digest['extrapolated_mw_window_relief_kwh']:.1f} kWh** of grid stress per "
          f"scheduling window — to be paired with the Chen & Zheng 3–21% grid-value range, not reported "
          f"as a literal large-fleet simulation.\n",
          "_Figure: `fleet_scale.png`. Digest: `scale_digest.json`._\n"]
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes-per-region", default="1,2,4,8")
    ap.add_argument("--pods", default="", help="optional explicit pod counts (paired with sizes); "
                                               "default holds GPU oversubscription ~constant")
    ap.add_argument("--oversub", type=float, default=1.06, help="target GPU oversubscription for auto pods")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--max-timeslots", type=int, default=48)
    ap.add_argument("--scenario", default="heatwave-drought", choices=["heatwave-drought", "observed-winter"])
    ap.add_argument("--signals-dir", default=None, help="time-aligned signal dir (default: 2018 stress year)")
    ap.add_argument("--run-name", default="t19_alibaba_fleet_scale")
    args = ap.parse_args()
    import logging
    logging.disable(logging.CRITICAL)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
