#!/usr/bin/env python3
"""T25.1 — Carbon signal-fidelity: does deciding on the residual-load CI *proxy* yield a FALSE
no-harm certificate when checked against real generation-mix CI?

Per (fleet size, seed) we run the no-harm pilot three ways on the Alibaba GPU testbed:
  * proxy : decide + verify on the residual-load-scaled proxy CI (the current headline signal).
  * split : decide on the proxy, VERIFY the certificate on real generation-mix CI
            (verify_forecasts_file) -- the same schedule, re-scored on reality.
  * real  : decide + verify on real generation-mix CI (the defensible primary).

Headline metric: the fraction of proxy-certified schedules that FAIL no-harm when re-scored on real
CI (false-certificate rate), and the real-carbon delta of the proxy-decided schedule. If `real`
certifies with a genuine carbon cut while `split` fails, the proxy is an unfaithful decision signal.

Run: python scripts/signal_fidelity_carbon.py --nodes-per-region 1,2,4 --seeds 0,1,2 --run-name t25_carbon_fidelity
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import alibaba_common as ac  # noqa: E402
from carbon_aware.no_harm_flexibility import run_no_harm_flexibility_pilot  # noqa: E402

REALCI = ac.REPO_ROOT / "pkg" / "carbon-aware" / "data" / "timealigned_realci"
METHODS = ["no_harm_flex", "carbon", "water_scarcity", "waterwise"]


def run(args) -> Path:
    out = ac.REPO_ROOT / "experiments" / "flexibility" / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    npr_list = ac.ints(args.nodes_per_region)
    seeds = ac.ints(args.seeds)
    rows: List[Dict] = []

    for npr in npr_list:
        fleet = npr * ac.N_REGIONS
        pods = ac.pods_for_fleet(npr, args.oversub)
        nodes_file, n_nodes, n_gpus = ac.build_fleet(out / "fleets" / f"f{fleet}", npr)
        for seed in seeds:
            w = ac.build_window(out / "inputs" / f"f{fleet}_s{seed}", target_pods=pods, seed=seed)
            base = ac.make_pilot_config(nodes_file, w["workloads_dir"], out / "cases" / f"f{fleet}_s{seed}_proxy",
                                        signals=ac.TIMEALIGNED, max_timeslots=args.max_timeslots)
            configs = {
                "proxy": base,
                "split": replace(base, output_dir=out / "cases" / f"f{fleet}_s{seed}_split",
                                 verify_forecasts_file=REALCI / "forecasts.json"),
                "real": ac.make_pilot_config(nodes_file, w["workloads_dir"], out / "cases" / f"f{fleet}_s{seed}_real",
                                             signals=REALCI, max_timeslots=args.max_timeslots),
            }
            res = {}
            for mode, pc in configs.items():
                r = {x["method_key"]: x for x in run_no_harm_flexibility_pilot(pc)["summary_rows"]}
                res[mode] = r
                for m in METHODS:
                    rows.append({"fleet": fleet, "seed": seed, "mode": mode, "method": m,
                                 "no_harm": int(bool(r[m]["no_harm_certificate"])),
                                 "carbon_delta_pct": r[m]["carbon_delta_pct"],
                                 "scarcity_delta_pct": r[m]["scarcity_delta_pct"],
                                 "stress_kwh_avoided": r[m]["stress_kwh_avoided"]})
            f = lambda mode: res[mode]["no_harm_flex"]
            print(f"  fleet={fleet:>3} seed={seed}: "
                  f"proxy[cert={f('proxy')['no_harm_certificate']!s:5} C%={f('proxy')['carbon_delta_pct']:6.2f}] "
                  f"split[cert={f('split')['no_harm_certificate']!s:5} C%={f('split')['carbon_delta_pct']:6.2f}] "
                  f"real[cert={f('real')['no_harm_certificate']!s:5} C%={f('real')['carbon_delta_pct']:6.2f}]", flush=True)

    _write_csv(out / "tidy.csv", rows)
    _report(out / "evidence_report.md", rows, seeds)
    _figure(out / "carbon_fidelity.png", rows)
    print(f"\noutput_dir={out}")
    return out


def _write_csv(path, rows):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def _flex(rows, mode):
    return [r for r in rows if r["method"] == "no_harm_flex" and r["mode"] == mode]


def _report(path, rows, seeds):
    proxy, split, real = _flex(rows, "proxy"), _flex(rows, "split"), _flex(rows, "real")
    n = len(proxy)
    # false-certificate: proxy says no-harm, split (real-CI re-score of the same schedule) says NOT.
    false_cert = sum(1 for p, s in zip(proxy, split) if p["no_harm"] and not s["no_harm"])
    L = [f"# Carbon signal-fidelity (T25.1) — Alibaba GPU, {len(seeds)} seeds {seeds}\n",
         "Decide on the residual-load CI proxy, then re-score the certificate on real generation-mix CI.\n",
         "| metric | value |", "|---|---|",
         f"| configs (fleet×seed) | {n} |",
         f"| proxy no-harm rate | {ac.mean([p['no_harm'] for p in proxy])*100:.0f}% |",
         f"| **false-certificate rate** (proxy-certified, fails on real CI) | **{100*false_cert/max(n,1):.0f}%** |",
         f"| proxy carbon Δ% (claimed) | {ac.mean([p['carbon_delta_pct'] for p in proxy]):.2f} ± {ac.ci95([p['carbon_delta_pct'] for p in proxy]):.2f} |",
         f"| **split carbon Δ% (proxy schedule on REAL CI)** | **{ac.mean([s['carbon_delta_pct'] for s in split]):.2f} ± {ac.ci95([s['carbon_delta_pct'] for s in split]):.2f}** |",
         f"| real no-harm rate (decide on real CI) | {ac.mean([r['no_harm'] for r in real])*100:.0f}% |",
         f"| real carbon Δ% (decide on real CI) | {ac.mean([r['carbon_delta_pct'] for r in real]):.2f} ± {ac.ci95([r['carbon_delta_pct'] for r in real]):.2f} |",
         "\n**Reading:** if the false-certificate rate is high and the split carbon Δ% is ≥0 while the real "
         "Δ% is clearly negative, the residual-load proxy is an unfaithful decision signal for the carbon "
         "certificate — deciding (and verifying) on real generation-mix CI is required.\n"]
    path.write_text("\n".join(L), encoding="utf-8")


def _figure(path, rows):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    modes = ["proxy", "split", "real"]
    labels = ["proxy\n(decide+verify proxy)", "split\n(decide proxy / verify REAL)", "real\n(decide+verify real)"]
    means = [ac.mean([r["carbon_delta_pct"] for r in _flex(rows, m)]) for m in modes]
    errs = [ac.ci95([r["carbon_delta_pct"] for r in _flex(rows, m)]) for m in modes]
    cert = [ac.mean([r["no_harm"] for r in _flex(rows, m)]) for m in modes]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    colors = ["#4C72B0", "#C44E52", "#55A868"]
    bars = ax.bar(labels, means, yerr=errs, capsize=4, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    for b, c, m in zip(bars, cert, means):
        ax.annotate(f"cert {c*100:.0f}%", (b.get_x() + b.get_width() / 2, m),
                    ha="center", va="bottom" if m >= 0 else "top", fontsize=9, fontweight="bold")
    ax.set_ylabel("carbon Δ% vs packing (real-CI accounting where applicable)")
    ax.set_title("Carbon signal-fidelity: the residual-load proxy can issue a false certificate")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes-per-region", default="1,2,4")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--oversub", type=float, default=1.06)
    ap.add_argument("--max-timeslots", type=int, default=48)
    ap.add_argument("--run-name", default="t25_carbon_fidelity")
    args = ap.parse_args()
    import logging; logging.disable(logging.CRITICAL)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
