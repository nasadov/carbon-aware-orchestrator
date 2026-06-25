#!/usr/bin/env python3
"""T24.1 — Scarcity signal-fidelity: does deciding on COUNTRY-level AWARE CFs yield a FALSE no-harm
certificate when checked at BASIN resolution?

Per (fleet size, seed) we run the no-harm pilot three ways on the Alibaba GPU testbed:
  * country : decide + verify on country-level AWARE CFs (the current headline scarcity signal).
  * split   : decide on country CFs, VERIFY the certificate at BASIN resolution (verify_config_file)
              -- the same schedule, re-scored on the watershed where scarcity actually bites.
  * basin   : decide + verify at basin resolution (the faithful signal).

Spatial analog of the carbon proxy result (signal_fidelity_carbon.py): country CFs misrank regions
(e.g. Milan/Po looks scarce but is the least scarce), so a country-decided schedule can route water
INTO a genuinely scarce basin and fail no-harm there. Headline: the basin false-certificate rate.

Run: python scripts/signal_fidelity_basin.py --nodes-per-region 1,2,4 --seeds 0,1,2 --run-name t24_basin_fidelity
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import alibaba_common as ac  # noqa: E402
from carbon_aware.no_harm_flexibility import run_no_harm_flexibility_pilot  # noqa: E402

METHODS = ["no_harm_flex", "carbon", "water_scarcity", "waterwise"]


def _write_basin_config(run_name: str) -> Path:
    """Temp config (in the config dir so data/ paths resolve) with basin scarcity resolution."""
    doc = yaml.safe_load(ac.CONFIG_FILE.read_text())
    doc.setdefault("water", {})["operational_scarcity_spatial_resolution"] = "basin"
    path = ac.CONFIG_FILE.parent / f"_basin_config_{run_name}.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def run(args) -> Path:
    out = ac.REPO_ROOT / "experiments" / "flexibility" / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    basin_cfg = _write_basin_config(args.run_name)
    npr_list = ac.ints(args.nodes_per_region)
    seeds = ac.ints(args.seeds)
    rows: List[Dict] = []
    try:
        for npr in npr_list:
            fleet = npr * ac.N_REGIONS
            pods = ac.pods_for_fleet(npr, args.oversub)
            nodes_file, n_nodes, n_gpus = ac.build_fleet(out / "fleets" / f"f{fleet}", npr)
            for seed in seeds:
                w = ac.build_window(out / "inputs" / f"f{fleet}_s{seed}", target_pods=pods, seed=seed)
                country = ac.make_pilot_config(nodes_file, w["workloads_dir"], out / "cases" / f"f{fleet}_s{seed}_country",
                                               signals=ac.TIMEALIGNED, max_timeslots=args.max_timeslots)
                configs = {
                    "country": country,
                    "split": replace(country, output_dir=out / "cases" / f"f{fleet}_s{seed}_split",
                                     verify_config_file=basin_cfg),
                    "basin": replace(country, output_dir=out / "cases" / f"f{fleet}_s{seed}_basin",
                                     config_file=basin_cfg),
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
                      f"country[cert={f('country')['no_harm_certificate']!s:5} W%={f('country')['scarcity_delta_pct']:6.2f}] "
                      f"split[cert={f('split')['no_harm_certificate']!s:5} W%={f('split')['scarcity_delta_pct']:6.2f}] "
                      f"basin[cert={f('basin')['no_harm_certificate']!s:5} W%={f('basin')['scarcity_delta_pct']:6.2f}]", flush=True)
    finally:
        basin_cfg.unlink(missing_ok=True)

    _write_csv(out / "tidy.csv", rows)
    _report(out / "evidence_report.md", rows, seeds)
    _figure(out / "basin_fidelity.png", rows)
    print(f"\noutput_dir={out}")
    return out


def _write_csv(path, rows):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def _flex(rows, mode):
    return [r for r in rows if r["method"] == "no_harm_flex" and r["mode"] == mode]


def _report(path, rows, seeds):
    country, split, basin = _flex(rows, "country"), _flex(rows, "split"), _flex(rows, "basin")
    n = len(country)
    false_cert = sum(1 for c, s in zip(country, split) if c["no_harm"] and not s["no_harm"])
    L = [f"# Scarcity signal-fidelity (T24.1) — Alibaba GPU, {len(seeds)} seeds {seeds}\n",
         "Decide on country-level AWARE CFs, then re-score the certificate at basin (watershed) resolution.\n",
         "| metric | value |", "|---|---|",
         f"| configs (fleet×seed) | {n} |",
         f"| country no-harm rate | {ac.mean([c['no_harm'] for c in country])*100:.0f}% |",
         f"| **basin false-certificate rate** (country-certified, fails at basin) | **{100*false_cert/max(n,1):.0f}%** |",
         f"| country scarcity Δ% (claimed) | {ac.mean([c['scarcity_delta_pct'] for c in country]):.2f} ± {ac.ci95([c['scarcity_delta_pct'] for c in country]):.2f} |",
         f"| **split scarcity Δ% (country schedule at BASIN res)** | **{ac.mean([s['scarcity_delta_pct'] for s in split]):.2f} ± {ac.ci95([s['scarcity_delta_pct'] for s in split]):.2f}** |",
         f"| basin no-harm rate (decide at basin) | {ac.mean([b['no_harm'] for b in basin])*100:.0f}% |",
         f"| basin scarcity Δ% (decide at basin) | {ac.mean([b['scarcity_delta_pct'] for b in basin]):.2f} ± {ac.ci95([b['scarcity_delta_pct'] for b in basin]):.2f} |",
         "\n**Reading:** a high basin false-certificate rate + split scarcity Δ% ≥ 0 means country-level "
         "scarcity accounting (as WaterWise et al. use) misranks watersheds badly enough to certify a "
         "schedule that is harmful where scarcity physically bites — basin resolution is required.\n"]
    path.write_text("\n".join(L), encoding="utf-8")


def _figure(path, rows):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    modes = ["country", "split", "basin"]
    labels = ["country\n(decide+verify country)", "split\n(decide country / verify BASIN)", "basin\n(decide+verify basin)"]
    means = [ac.mean([r["scarcity_delta_pct"] for r in _flex(rows, m)]) for m in modes]
    errs = [ac.ci95([r["scarcity_delta_pct"] for r in _flex(rows, m)]) for m in modes]
    cert = [ac.mean([r["no_harm"] for r in _flex(rows, m)]) for m in modes]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    bars = ax.bar(labels, means, yerr=errs, capsize=4, color=["#4C72B0", "#C44E52", "#55A868"])
    ax.axhline(0, color="k", lw=0.8)
    for b, c, m in zip(bars, cert, means):
        ax.annotate(f"cert {c*100:.0f}%", (b.get_x() + b.get_width() / 2, m),
                    ha="center", va="bottom" if m >= 0 else "top", fontsize=9, fontweight="bold")
    ax.set_ylabel("scarcity-water Δ% vs packing (basin accounting where applicable)")
    ax.set_title("Scarcity signal-fidelity: country CFs can issue a false certificate")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes-per-region", default="1,2,4")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--oversub", type=float, default=1.06)
    ap.add_argument("--max-timeslots", type=int, default=48)
    ap.add_argument("--run-name", default="t24_basin_fidelity")
    args = ap.parse_args()
    import logging; logging.disable(logging.CRITICAL)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
