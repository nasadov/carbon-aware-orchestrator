#!/usr/bin/env python3
"""R3 / reviewer MC5 (Q5): per-basin recompute of the CROSS-REGIME water (feeds Table 6).

The aggregate water guard burden-shifts (it can lower the sum while raising one watershed). The 6
real-trace windows that feed Table 5 are already recomputed per-basin in
experiments/water_basin/basin_certification.json. This script adds the two CROSS-REGIME regimes that
feed Table 6 (tab:crossregime) and were NOT yet recomputed per-basin:

  * headroom (favorable CPU, COUNTRY scarcity)  -- DE/FR/ES/IT-NO, measured CI, low utilization.
    Methodologically identical to the Azure real-trace windows (same regions/signals/config) but at
    headroom utilization; cited Table-6 point ~ -9.8C / -9.9W (country). We recompute its water at
    BASIN resolution and under the per-basin guard (July basin CFs).
  * strong / drought (favorable EU drought, BASIN scarcity) -- SE/PL/ES/DE, Aug-2022, the engine
    already runs basin-resolution; cited Table-6 point -21C / -41W. We add the PER-BASIN delta
    (Delta W_b <= 0 per watershed) and the per-basin guard (August basin CFs), which the engine's
    AGGREGATE-basin guard does not enforce.

It reuses the EXACT per-basin primitives from run_water_basin_certification.py (no engine edit):
_operational_water_by_basin, _characterize, _per_basin_guarded_member.

Run:
  PYTHONPATH=pkg/carbon-aware/server-python python scripts/r3_perbasin_crossregime.py
"""
from __future__ import annotations

import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SERVER = REPO_ROOT / "pkg" / "carbon-aware" / "server-python"
for p in (str(SERVER), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import azure_common as ac  # noqa: E402
import run_water_basin_certification as wb  # noqa: E402  (reuse the per-basin primitives)
from carbon_aware.no_harm_flexibility import (  # noqa: E402
    PilotConfig, load_pods, run_no_harm_flexibility_pilot,
)

REALCI = REPO_ROOT / "pkg" / "carbon-aware" / "data" / "timealigned_realci2"  # coherent revamp build (Option A + in-window EWIF + continuous cooling)
STRONG_SIGNALS = REPO_ROOT / "pkg" / "carbon-aware" / "data" / "timealigned_strong_real2022_v2"  # coherent Aug-2022 build
BASE_CONFIG = REPO_ROOT / "pkg" / "carbon-aware" / "infra-workload-config.yaml"
STRONG_CONFIG = REPO_ROOT / "pkg" / "carbon-aware" / "infra-workload-config-strong.yaml"
OUT_ROOT = REPO_ROOT / "experiments" / "r3_perbasin_tables" / "crossregime"

BASIN_CF_CSV = wb.BASIN_CF_CSV
COUNTRY_CF_CSV = wb.COUNTRY_CF_CSV
FRONTIER = ["no_harm_flex", "no_harm_search_control"]
METHODS = ["packing"] + FRONTIER + ["carbon", "water_scarcity", "waterwise"]
DROUGHT_CF_THRESHOLD = wb.DROUGHT_CF_THRESHOLD


def _cf_for_month(csv_path: Path, month_col: str, key_col: str = "region") -> Dict[str, float]:
    out: Dict[str, float] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            k = (row.get(key_col) or "").strip().upper()
            if k:
                try:
                    out[k] = float(row.get(month_col))
                except (TypeError, ValueError):
                    pass
    return out


def _country_for_region(region: str) -> str:
    if region == "IT-NO":
        return "IT"
    return region.split("-", 1)[0].upper() if "-" in region else region.upper()


def _per_basin_block(out_dir: Path, regions: List[str], cf_basin: Dict[str, float],
                     cf_country: Dict[str, float], summary: Dict[str, dict],
                     regime: str = "") -> dict:
    """Per-basin re-characterization (Option A: direct@basin + indirect@country) + per-basin guard for
    one regime's engine run, mirroring run_water_basin_certification.run_window. 'basin_agg' is the
    Option-A certificate axis; 'country_agg' is the OLD all-country mischarge (counterfactual contrast)."""
    base_comp = wb._components_by_basin(out_dir / "placements_packing.csv")
    base_basin = wb._scarcity_optionA(base_comp, cf_basin, cf_country)   # Option-A certificate axis
    wb._reconcile(base_comp, base_basin, regime or "crossregime", "packing")
    base_country = wb._scarcity_allcountry(base_comp, cf_country)        # old all-country mischarge

    methods_out: Dict[str, dict] = {}
    for m in METHODS:
        pcsv = out_dir / f"placements_{m}.csv"
        if not pcsv.exists():
            continue
        comp = wb._components_by_basin(pcsv)
        s_basin = wb._scarcity_optionA(comp, cf_basin, cf_country)
        wb._reconcile(comp, s_basin, regime or "crossregime", m)
        s_country = wb._scarcity_allcountry(comp, cf_country)
        basin_delta = {b: s_basin.get(b, 0.0) - base_basin.get(b, 0.0) for b in regions}
        agg_basin = sum(s_basin.values()); agg_basin_base = sum(base_basin.values())
        agg_country = sum(s_country.values()); agg_country_base = sum(base_country.values())
        tol = 1e-6
        worst_basin = max(regions, key=lambda b: basin_delta[b])
        methods_out[m] = {
            "engine_no_harm_certificate": bool(summary[m]["no_harm_certificate"]) if m in summary else None,
            "engine_carbon_delta_pct": summary[m]["carbon_delta_pct"] if m in summary else None,
            "engine_scarcity_delta_pct": summary[m]["scarcity_delta_pct"] if m in summary else None,
            "country_agg_delta_pct": 100.0 * (agg_country - agg_country_base) / agg_country_base if agg_country_base else 0.0,
            "basin_agg_delta_pct": 100.0 * (agg_basin - agg_basin_base) / agg_basin_base if agg_basin_base else 0.0,
            "basin_per_basin_delta": {b: basin_delta[b] for b in regions},
            "basin_per_basin_delta_pct": {
                b: (100.0 * basin_delta[b] / base_basin[b] if base_basin.get(b) else 0.0) for b in regions
            },
            "basin_per_basin_cert": all(d <= tol for d in basin_delta.values()),
            "basin_worst_basin": worst_basin,
            "basin_worst_delta_pct": (100.0 * basin_delta[worst_basin] / base_basin[worst_basin]
                                      if base_basin.get(worst_basin) else 0.0),
        }

    guarded: Dict[str, dict] = {}
    for m in FRONTIER:
        if (out_dir / f"placements_{m}.csv").exists():
            guarded[m] = wb._per_basin_guarded_member(
                out_dir / "placements_packing.csv", out_dir / f"placements_{m}.csv",
                regions, cf_basin, cf_country,
            )
    drought = {
        "threshold": DROUGHT_CF_THRESHOLD,
        "country_fires": [b for b in regions if cf_country.get(b, 0.0) >= DROUGHT_CF_THRESHOLD],
        "basin_fires": [b for b in regions if cf_basin.get(b, 0.0) >= DROUGHT_CF_THRESHOLD],
    }
    return {"methods": methods_out, "per_basin_guarded": guarded, "drought": drought,
            "cf_basin": cf_basin, "cf_country": cf_country, "regions": regions}


def run_headroom(window: str, util: float, *, seed: int) -> dict:
    """Favorable CPU, COUNTRY scarcity (the Table-6 headroom regime). Same regions/signals/config as the
    Azure real-trace windows but provisioned to a headroom utilization. July basin CFs."""
    regions = ["DE", "FR", "ES", "IT-NO"]
    cf_basin_all = _cf_for_month(BASIN_CF_CSV, "jul_cf")
    cf_country_all = _cf_for_month(COUNTRY_CF_CSV, "jul_cf", key_col="country_code")
    cf_basin = {r: cf_basin_all[r] for r in regions}
    cf_country = {r: cf_country_all[_country_for_region(r)] for r in regions}

    run_dir = OUT_ROOT / f"headroom_{window}_u{util:.2f}"
    run_dir.mkdir(parents=True, exist_ok=True)
    src = (REPO_ROOT / "experiments" / "real_traces" / window).resolve()
    canonical = src / "canonical_trace_workload.csv"
    stats = ac.build_window_from_canonical(canonical, run_dir, seed=seed, cpu_util=ac.conv.CPU_UTIL_MEAN)
    workloads = stats["workloads_dir"]
    peak = ac.workload_peak_cores(workloads)
    # Headroom must be a genuinely SLACKER fleet than the canonical real-trace one. The shared
    # provision_for_util() rounds nodes/region, and at this trace's peak, utils 0.40 and 0.55 both
    # round to the SAME 1-node/region fleet -- the old "headroom regime" was bit-identical to the
    # d0 real-trace window (bug found 2026-07-01). Provision with ceil at the requested util and,
    # if quantization still lands on the canonical fleet, take the next slacker point; the MEASURED
    # utilization is reported alongside the requested one (the fleet is node-quantized).
    cores = int(ac.CPU_NODE["cpu_cores"])
    canonical_nodes = ac.provision_for_util(peak, 0.55)  # the canonical real-trace fleet
    per_region = max(1, math.ceil(peak / util / cores / ac.N_REGIONS))
    if per_region <= canonical_nodes:
        per_region = canonical_nodes + 1
    assert per_region != canonical_nodes, "headroom fleet must differ from the canonical real-trace fleet"
    measured_util = peak / (per_region * cores * ac.N_REGIONS)
    nodes_file, n_nodes, _cap = ac.build_fleet(run_dir / "fleet", per_region)
    pc = ac.make_pilot_config(nodes_file, workloads, run_dir / "out", signals=REALCI,
                              max_timeslots=48, scenario="heatwave-drought", lever_mode="both")
    res = run_no_harm_flexibility_pilot(pc)
    summary = {r["method_key"]: r for r in res["summary_rows"]}
    block = _per_basin_block(pc.output_dir, regions, cf_basin, cf_country, summary)
    block.update({"regime": "headroom (favorable CPU, country)", "window": window,
                  "util_requested": util, "util_measured": round(measured_util, 3),
                  "n_nodes": n_nodes, "n_pods": len(load_pods(workloads)), "cf_month": "jul"})
    return block


def run_strong() -> dict:
    """Favorable EU drought, BASIN scarcity (the Table-6 drought regime). v3 = German-basin fleet:
    SE/PL/ES + Germany split into its three AWARE watersheds (DE=Rhine-Main, DE-BER=Spree-Elbe,
    DE-MUC=Isar-Danube), Aug-2022. DE-BER/DE-MUC share DE's grid signals (one DE-LU bidding zone:
    intra-German moves are carbon/price-neutral) but carry real Berlin/Munich weather and their own
    basin CFs (2.54x intra-DE Aug gradient; country resolution sees 1.00x). Sub-national rows enter
    via the default-off basin_cf_csv override (engine-flag-check gated 2026-07-07)."""
    regions = ["SE", "PL", "ES", "DE", "DE-BER", "DE-MUC"]
    v3_signals = REPO_ROOT / "pkg" / "carbon-aware" / "data" / "timealigned_strong_real2022_v3"
    v3_config = REPO_ROOT / "pkg" / "carbon-aware" / "infra-workload-config-strong-v3.yaml"
    v3_basin_csv = REPO_ROOT / "pkg" / "carbon-aware" / "data" / "water" / "aware20_basin_nonagri_factors_v3.csv"
    cf_basin_all = _cf_for_month(v3_basin_csv, "aug_cf")
    cf_country_all = _cf_for_month(COUNTRY_CF_CSV, "aug_cf", key_col="country_code")
    cf_basin = {r: cf_basin_all[r] for r in regions}
    cf_country = {r: cf_country_all[_country_for_region(r)] for r in regions}

    run_dir = OUT_ROOT / "strong_v3"
    run_dir.mkdir(parents=True, exist_ok=True)
    nodes_file = REPO_ROOT / "experiments" / "strong_scenario" / "fleet_v3" / "nodes.yaml"
    workloads = REPO_ROOT / "experiments" / "strong_scenario" / "fleet" / "workloads"
    pc = PilotConfig(
        repo_root=REPO_ROOT, nodes_file=nodes_file, workloads_dir=workloads,
        forecasts_file=v3_signals / "forecasts.json", config_file=v3_config,
        output_dir=run_dir / "out", max_timeslots=24, max_pods=400,
        scenario="heatwave-drought", lever_mode="both",
        scenario_month="aug",  # Aug-2022 window -> engine characterizes scarcity on the Aug CFs natively
        grid_signal_csv=v3_signals / "grid_residual_region_slot.csv",
        wue_csv=v3_signals / "wue_region_slot.csv",
        # in-window Aug-2022 off-site water (same Energy-Charts mix as the strong CI)
        ewif_csv=v3_signals / "ewif_region_slot.csv",
        basin_cf_csv=v3_basin_csv,
    )
    pc.output_dir.mkdir(parents=True, exist_ok=True)
    res = run_no_harm_flexibility_pilot(pc)
    summary = {r["method_key"]: r for r in res["summary_rows"]}
    block = _per_basin_block(pc.output_dir, regions, cf_basin, cf_country, summary)
    block.update({"regime": "strong/drought v3 (favorable EU, German basins)",
                  "window": "strong_v3_SE_PL_ES_DE_DEBER_DEMUC",
                  "n_nodes": 6, "n_pods": len(load_pods(workloads)), "cf_month": "aug"})
    return block


def main() -> int:
    logging.disable(logging.CRITICAL)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results: List[dict] = []

    print("\n##### cross-regime per-basin: HEADROOM (favorable CPU, country) #####", flush=True)
    # Headroom operating point: d0 at util 0.40 (where the fleet has slack and the certified co-benefit
    # emerges -- the Table-6 'headroom' regime). We report the per-basin recompute of its water.
    hr = run_headroom("azure_packing_2020_d0_s42_200", 0.40, seed=42)
    results.append(hr)
    for m in FRONTIER:
        mm = hr["methods"][m]; g = hr["per_basin_guarded"][m]
        print(f"  {m:24s} engine W%={mm['engine_scarcity_delta_pct']:+7.2f}  "
              f"country aggΔ={mm['country_agg_delta_pct']:+7.2f}%  basin aggΔ={mm['basin_agg_delta_pct']:+7.2f}%  "
              f"pb cert(engine)={mm['basin_per_basin_cert']!s:5}  worst={mm['basin_worst_basin']}({mm['basin_worst_delta_pct']:+.2f}%)  "
              f"|| GUARD basin aggΔ={g['agg_delta_pct']:+7.2f}% pb={g['per_basin_cert']!s:5} ({g['moves_accepted']}/{g['moves_total']})")

    print("\n##### cross-regime per-basin: STRONG/DROUGHT (favorable EU, basin) #####", flush=True)
    st = run_strong()
    results.append(st)
    for m in FRONTIER:
        mm = st["methods"][m]; g = st["per_basin_guarded"][m]
        print(f"  {m:24s} engine W%={mm['engine_scarcity_delta_pct']:+7.2f}  "
              f"basin aggΔ={mm['basin_agg_delta_pct']:+7.2f}%  "
              f"pb cert(engine)={mm['basin_per_basin_cert']!s:5}  worst={mm['basin_worst_basin']}({mm['basin_worst_delta_pct']:+.2f}%)  "
              f"|| GUARD basin aggΔ={g['agg_delta_pct']:+7.2f}% pb={g['per_basin_cert']!s:5} ({g['moves_accepted']}/{g['moves_total']})")

    digest = OUT_ROOT / "crossregime_perbasin.json"
    digest.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
