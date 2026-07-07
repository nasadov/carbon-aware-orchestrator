#!/usr/bin/env python3
"""G1 pilot: intra-German per-basin certification on the Aug-2022 strong window.

Fleet SE / PL / DE(Frankfurt, Rhine-Main) / DE-BER(Berlin, Spree-Elbe) / DE-MUC(Munich,
Isar-Danube): the strong-scenario fleet with Germany split into its three AWARE basins
(2.5x intra-DE gradient, Elbe-Spree the scarce pole). DE-BER/DE-MUC share DE's grid signals
(one DE-LU bidding zone) but carry real Aug-2022 Berlin/Munich weather and their own basin CFs,
so intra-German moves are carbon- and price-neutral and differ ONLY in watershed + cooling —
country-resolution water accounting is blind to them by construction.

Scenario/config built by the G1 assembly step (data/timealigned_strong_real2022_v3, *_g1pilot.csv/yaml,
experiments/g1_german_pilot/fleet). Reuses the exact per-basin primitives via
r3_perbasin_crossregime (which wraps run_water_basin_certification). No engine changes.

Run:
  PYTHONPATH=pkg/carbon-aware/server-python python scripts/pilot_g1_german_basins.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SERVER = REPO_ROOT / "pkg" / "carbon-aware" / "server-python"
for p in (str(SERVER), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import r3_perbasin_crossregime as r3  # noqa: E402  (per-basin helpers + METHODS/FRONTIER)
from carbon_aware.no_harm_flexibility import (  # noqa: E402
    PilotConfig, load_pods, run_no_harm_flexibility_pilot,
)

SIGNALS = REPO_ROOT / "pkg" / "carbon-aware" / "data" / "timealigned_strong_real2022_v3"
CONFIG = REPO_ROOT / "pkg" / "carbon-aware" / "infra-workload-config-strong-v3.yaml"
BASIN_CSV = REPO_ROOT / "pkg" / "carbon-aware" / "data" / "water" / "aware20_basin_nonagri_factors_v3.csv"
OUT = REPO_ROOT / "experiments" / "g1_german_pilot"
REGIONS = ["SE", "PL", "ES", "DE", "DE-BER", "DE-MUC"]
DE_BASINS = ["DE", "DE-BER", "DE-MUC"]


def main() -> int:
    logging.disable(logging.CRITICAL)
    cf_basin_all = r3._cf_for_month(BASIN_CSV, "aug_cf")
    cf_country_all = r3._cf_for_month(r3.COUNTRY_CF_CSV, "aug_cf", key_col="country_code")
    cf_basin = {r: cf_basin_all[r] for r in REGIONS}
    cf_country = {r: cf_country_all[r3._country_for_region(r)] for r in REGIONS}
    print("aug basin CFs:   ", {k: round(v, 2) for k, v in cf_basin.items()})
    print("aug country CFs: ", {k: round(v, 2) for k, v in cf_country.items()})
    print(f"intra-DE basin gradient (aug): "
          f"{max(cf_basin[b] for b in DE_BASINS) / min(cf_basin[b] for b in DE_BASINS):.2f}x "
          f"(country accounting sees 1.00x)")

    run_dir = OUT / "out"
    run_dir.mkdir(parents=True, exist_ok=True)
    pc = PilotConfig(
        repo_root=REPO_ROOT,
        nodes_file=OUT / "fleet" / "nodes.yaml",
        workloads_dir=REPO_ROOT / "experiments" / "strong_scenario" / "fleet" / "workloads",
        forecasts_file=SIGNALS / "forecasts.json",
        config_file=CONFIG,
        output_dir=run_dir,
        max_timeslots=24, max_pods=400,
        scenario="heatwave-drought", lever_mode="both",
        scenario_month="aug",
        grid_signal_csv=SIGNALS / "grid_residual_region_slot.csv",
        wue_csv=SIGNALS / "wue_region_slot.csv",
        ewif_csv=SIGNALS / "ewif_region_slot.csv",
        # sub-national German basins: the default-off engine override (engine-flag-check gated),
        # replacing this pilot's original _water_data_path monkeypatch
        basin_cf_csv=BASIN_CSV,
    )
    res = run_no_harm_flexibility_pilot(pc)
    summary = {r["method_key"]: r for r in res["summary_rows"]}
    block = r3._per_basin_block(run_dir, REGIONS, cf_basin, cf_country, summary,
                                regime="g1_german_pilot")
    block.update({
        "regime": "G1 German-basin pilot (strong Aug-2022; DE split into Rhine-Main/Spree-Elbe/Danube)",
        "window": "g1_SE_PL_DE_DEBER_DEMUC",
        "n_nodes": len(REGIONS),
        "n_pods": len(load_pods(pc.workloads_dir)),
        "cf_month": "aug",
    })

    print(f"\n{'method':24} {'cert':>5} {'C%':>8} {'W%(eng)':>8} {'ctryΔ%':>8} {'basinΔ%':>8} "
          f"{'pbCert':>6} {'worst':>8} | per-DE-basin Δ%")
    for m, mm in block["methods"].items():
        de_deltas = {b: round(mm["basin_per_basin_delta_pct"][b], 2) for b in DE_BASINS}
        cd = mm["engine_carbon_delta_pct"]
        wd = mm["engine_scarcity_delta_pct"]
        print(f"{m:24} {str(mm['engine_no_harm_certificate']):>5} "
              f"{(f'{cd:+.2f}' if cd is not None else 'n/a'):>8} "
              f"{(f'{wd:+.2f}' if wd is not None else 'n/a'):>8} "
              f"{mm['country_agg_delta_pct']:+8.2f} {mm['basin_agg_delta_pct']:+8.2f} "
              f"{str(mm['basin_per_basin_cert']):>6} {mm['basin_worst_basin']:>8} | {de_deltas}")
    for m, g in block["per_basin_guarded"].items():
        print(f"GUARD {m:18} basin aggΔ={g['agg_delta_pct']:+.2f}%  pbCert={g['per_basin_cert']}  "
              f"moves {g['moves_accepted']}/{g['moves_total']}")

    digest = OUT / "g1_perbasin.json"
    digest.write_text(json.dumps(block, indent=2), encoding="utf-8")
    print(f"\nwrote {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
