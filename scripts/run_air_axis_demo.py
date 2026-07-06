#!/usr/bin/env python3
"""Air-quality damage axis demo: one more inequality, riding the cost-axis machinery.

Motivation (2026): DOE 202(c)/PJM emergency orders curtail large data centers by switching
them to on-site backup generators with pollution limits waived; "The Unpaid Toll"
(arXiv:2412.06288) prices NoVA data-center genset permits at $220--300M/yr of health damages
at just 10% of permitted hours. Grid-relief flexibility as practiced can CAUSE air-quality
harm. This demo adds the grid-side term as a certificate axis: health-damage-weighted air
pollution (NOx/SO2/PM2.5) per realized kWh, exactly parallel to carbon -- so a schedule can
certify "grid relief without the genset/coal pathway."

Mechanics: damage intensity (EUR/kWh) per (region, slot) = generation-mix share x per-fuel
emission factor (g/kWh) x EEA-class damage cost (EUR/kg). Structurally a price, so it rides
the engine's cost axis (price_csv/cost_enabled/cost_guard) with ZERO engine changes:

  * MEASURE pass (cost_enabled=True): every method's air-damage delta vs the packing
    baseline is reported -- a method can pass the 3-axis certificate yet RAISE air damage.
  * GUARD pass (cost_guard=True): the envelope additionally enforces air damage <= baseline.

Numbers are demo-grade: EMEP/EEA-guidebook-class per-fuel EFs and ETC/ATNI-class EU-average
damage costs with low/base/high brackets (see source_note columns); the exhibit's point is
the axis's pluggability and who leaks it, not the third digit of the damage factor. The
static annual mix mirrors the radiation axis's region table; a slot-resolved mix is the
future refinement.

Emits pkg/carbon-aware/data/air_quality/{air_technology_factors,air_damage_weights,
air_region_slot}.csv and experiments/air_axis/air_axis_demo.json.

Run:
  PYTHONPATH=pkg/carbon-aware/server-python python scripts/run_air_axis_demo.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict

REPO = Path(__file__).resolve().parents[1]
SERVER = REPO / "pkg" / "carbon-aware" / "server-python"
for p in (str(SERVER), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_water_basin_certification as wb  # noqa: E402
from carbon_aware.no_harm_flexibility import run_no_harm_flexibility_pilot  # noqa: E402

DATA = REPO / "pkg" / "carbon-aware" / "data" / "air_quality"
MIX = REPO / "pkg" / "carbon-aware" / "data" / "radiation" / "region_generation_mix.csv"
OUT = REPO / "experiments" / "air_axis"
N_SLOTS = 48

# Per-fuel electricity emission factors, g pollutant / kWh_el (low/base/high).
# EMEP/EEA air-pollutant-emission-inventory-guidebook-class ranges for the ~2018 EU plant
# fleet (post-LCP-BREF FGD/SCR retrofit levels); demo-grade, bracketed, not plant-resolved.
EF = {
    # fuel: {pollutant: (low, base, high)}
    "coal":    {"so2": (0.4, 0.8, 1.6), "nox": (0.5, 0.8, 1.4), "pm25": (0.03, 0.06, 0.12)},
    "gas":     {"so2": (0.001, 0.005, 0.01), "nox": (0.15, 0.3, 0.6), "pm25": (0.003, 0.008, 0.02)},
    "oil":     {"so2": (0.8, 1.5, 3.0), "nox": (0.5, 1.0, 1.8), "pm25": (0.05, 0.10, 0.20)},
    "biomass": {"so2": (0.02, 0.05, 0.10), "nox": (0.3, 0.5, 0.9), "pm25": (0.05, 0.10, 0.20)},
    # "other" is a mixed thermal residual; price it like gas (conservative nonzero).
    "other":   {"so2": (0.001, 0.005, 0.01), "nox": (0.15, 0.3, 0.6), "pm25": (0.003, 0.008, 0.02)},
    # nuclear/hydro/wind/solar/geothermal: operational air-pollutant emissions ~0 at this resolution.
}

# EU-average marginal damage costs, EUR/kg pollutant (low/base/high ~ VOLY..VSL bases),
# ETC/ATNI 2020 / EEA-2024-update-class values; country-resolved costs are the refinement.
DMG = {"so2": (11.0, 20.0, 34.0), "nox": (9.0, 18.0, 39.0), "pm25": (25.0, 40.0, 75.0)}

WINDOWS = [("azure", "azure_packing_2020_d0_s42_200", "Azure d0 (slack CPU)"),
           ("alibaba", "alibaba_gpu_2020_w1587_s43_200", "Alibaba w1587 (saturated GPU)")]
METHODS = ["packing", "carbon", "water_scarcity", "waterwise", "no_harm_flex"]


def build_tables() -> Dict[str, Dict[str, float]]:
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / "air_technology_factors.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_type", "pollutant", "g_per_kwh_low", "g_per_kwh_base", "g_per_kwh_high",
                    "source_note"])
        for fuel, polls in EF.items():
            for poll, (lo, ba, hi) in polls.items():
                w.writerow([fuel, poll, lo, ba, hi,
                            "EMEP/EEA guidebook-class EU-2018 plant-fleet range; demo-grade"])
    with (DATA / "air_damage_weights.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pollutant", "eur_per_kg_low", "eur_per_kg_base", "eur_per_kg_high", "source_note"])
        for poll, (lo, ba, hi) in DMG.items():
            w.writerow([poll, lo, ba, hi, "ETC/ATNI-2020 / EEA-2024-update-class EU average; demo-grade"])

    # per-region damage intensity from the committed generation-mix table
    intens: Dict[str, Dict[str, float]] = {}
    with MIX.open() as f:
        for row in csv.DictReader(f):
            region = row["region"]
            vals = {}
            for tier, idx in (("low", 0), ("base", 1), ("high", 2)):
                total = 0.0
                for fuel, polls in EF.items():
                    share = float(row.get(fuel, 0.0) or 0.0)
                    if share <= 0.0:
                        continue
                    for poll, br in polls.items():
                        total += share * (br[idx] / 1000.0) * DMG[poll][idx]
                vals[tier] = total
            intens[region] = vals

    with (DATA / "air_region_slot.csv").open("w", newline="") as f:
        w = csv.writer(f)
        # price_eur_per_kwh is the engine's cost-axis payload column; here it carries
        # air-quality damage intensity in EUR/kWh (base tier).
        w.writerow(["region", "slot_index", "price_eur_per_kwh",
                    "air_damage_low_eur_per_kwh", "air_damage_high_eur_per_kwh", "source"])
        for region, vals in intens.items():
            for slot in range(N_SLOTS):
                w.writerow([region, slot, f"{vals['base']:.6f}", f"{vals['low']:.6f}",
                            f"{vals['high']:.6f}", "mix x EMEP/EEA-class EF x ETC/ATNI-class damage"])
    return intens


def _summ(res):
    return {r["method_key"]: r for r in res["summary_rows"]}


def run_window(testbed: str, window: str, seed: int) -> dict:
    run_dir = OUT / window
    run_dir.mkdir(parents=True, exist_ok=True)
    if testbed == "alibaba":
        pc, n_nodes, n_pods = wb._build_gpu_config(window, run_dir)
    else:
        pc, n_nodes, n_pods = wb._build_azure_config(window, run_dir, seed=seed)
    air_csv = DATA / "air_region_slot.csv"

    measure = _summ(run_no_harm_flexibility_pilot(replace(
        pc, cost_enabled=True, price_csv=air_csv, output_dir=run_dir / "measure")))
    guard = _summ(run_no_harm_flexibility_pilot(replace(
        pc, cost_guard=True, price_csv=air_csv, output_dir=run_dir / "guard")))

    def row(summ, m):
        r = summ.get(m, {})
        return {
            "carbon_delta_pct": r.get("carbon_delta_pct"),
            "scarcity_delta_pct": r.get("scarcity_delta_pct"),
            "air_damage_eur": r.get("operational_cost_eur"),
            "air_damage_delta_pct": r.get("cost_delta_pct"),
            "certified_3axis": r.get("no_harm_certificate"),
            "certified_with_air": r.get("no_harm_certificate_with_cost"),
            "unplaced": r.get("unplaced_pods"),
        }

    return {
        "window": window, "n_pods": n_pods, "n_nodes": n_nodes,
        "measure": {m: row(measure, m) for m in METHODS if m in measure},
        "guard_envelope": row(guard, "no_harm_flex"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    intens = build_tables()
    print("[air intensity EUR/kWh, base tier]",
          {r: round(v["base"], 5) for r, v in intens.items()})

    windows = [run_window(tb, wname, args.seed) for tb, wname, _ in WINDOWS]
    out = {
        "axis": "air-quality damage (NOx+SO2+PM2.5, damage-weighted EUR/kWh), via cost-axis machinery",
        "region_intensity_eur_per_kwh": intens,
        "windows": windows,
    }
    (OUT / "air_axis_demo.json").write_text(json.dumps(out, indent=2))
    for wrow in windows:
        print(f"\n[{wrow['window']}]")
        for m, r in wrow["measure"].items():
            print(f"  {m:>15}: dC {r['carbon_delta_pct']}% dW {r['scarcity_delta_pct']}% "
                  f"dAir {r['air_damage_delta_pct']}% cert3={r['certified_3axis']}")
        print(f"  guarded envelope (air<=B): {wrow['guard_envelope']}")
    print(f"\n[done] -> {OUT/'air_axis_demo.json'}")


if __name__ == "__main__":
    main()
