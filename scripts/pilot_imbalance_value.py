#!/usr/bin/env python3
"""Bankable-relief pilot: re-denominate the certified schedules' load shifts in settled money.

The paper's grid-relief axis is a residual-load proxy (kWh moved out of stress hours) -- not
bankable. This pilot values the SAME certified schedules (zero engine modifications; the cost
axis is MEASURE-only, so placements are bit-identical to the paper's) under two price books:

  * day-ahead   -- the in-repo Energy-Charts day-ahead table (price_region_slot.csv), all regions.
                   Labeled "lower-bound proxy": imbalance prices are spikier than day-ahead.
  * reBAP-DE    -- Germany's real uniform imbalance price (reBAP, 15-min, EUR/MWh) downloaded from
                   netztransparenz.de for 2018-07-25/26 UTC, hourly-averaged onto the engine's 48
                   hourly slots (coarsening documented in the report); FR/ES/IT-NO/PL/SE keep
                   day-ahead as a labeled proxy. Under TenneT/German-TSO passive-balancing rules a
                   helping deviation settles at the imbalance price, so the envelope's NEGATIVE
                   cost delta vs the packing baseline reads as settled deviation value.

Headline metric (per window, envelope):
  value_ratio = |cost_delta_eur| / (|carbon_delta_kg|/1000 * carbon_price_eur_per_t)
at carbon prices EUR 16/t (mid-2018 EU ETS) and EUR 80/t (current era, ~EUR 81/t July 2026).

Also reported: EUR per MWh shifted, naive EUR/MW-yr extrapolation (48h windows -- no annualization
claim), and a partial two-zone check (what share of shifted energy is DE<->DE time shift, where the
reBAP nets both legs consistently, vs cross-zone, where the non-DE leg is only day-ahead-proxied).

Emits experiments/imbalance_value/imbalance_value.json.

Run:
  PYTHONPATH=pkg/carbon-aware/server-python python scripts/pilot_imbalance_value.py [--light]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SERVER = REPO_ROOT / "pkg" / "carbon-aware" / "server-python"
for p in (str(SERVER), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_water_basin_certification as wb  # reuse the realci2 fleet/config builders  # noqa: E402
from carbon_aware.no_harm_flexibility import run_no_harm_flexibility_pilot  # noqa: E402

OUT = REPO_ROOT / "experiments" / "imbalance_value"
DATA = OUT / "data"
PRICE_DA = wb.REALCI / "price_region_slot.csv"                # day-ahead, all regions
PRICE_REBAP = DATA / "price_region_slot_rebapDE.csv"          # DE=hourly reBAP, rest=day-ahead
REBAP_15MIN = DATA / "rebap_DE_2018-07-25_26_utc_15min.csv"   # raw 15-min series (provenance)

CARBON_PRICES = {"ets_2018_eur16": 16.0, "ets_current_eur80": 80.0}
METHODS = ["packing", "no_harm_flex", "no_harm_search_control", "carbon", "waterwise",
           "water_scarcity"]
WINDOW_HOURS = 48.0
HOURS_PER_YEAR = 8760.0


def _summ(res) -> Dict[str, dict]:
    return {r["method_key"]: r for r in res["summary_rows"]}


def _read_placements(out_dir: Path, method: str) -> Dict[str, dict]:
    path = out_dir / f"placements_{method}.csv"
    if not path.exists():
        return {}
    with path.open() as f:
        return {r["pod_id"]: r for r in csv.DictReader(f)}


def _rebap_hourly() -> List[float]:
    """48 hourly means of the 15-min reBAP (EUR/MWh), aligned to slot_index 0..47."""
    by_hour: Dict[str, List[float]] = {}
    with REBAP_15MIN.open() as f:
        for r in csv.DictReader(f):
            by_hour.setdefault(r["utc_timestamp"][:13], []).append(float(r["rebap_eur_per_mwh"]))
    hours = sorted(by_hour)
    assert len(hours) == 48 and all(len(by_hour[h]) == 4 for h in hours)
    return [statistics.mean(by_hour[h]) for h in hours]


def _shift_analysis(out_dir: Path, method: str, rebap_hourly: List[float]) -> Optional[dict]:
    """Compare a method's placements to packing's: moved energy, region legs, two-zone classes."""
    base = _read_placements(out_dir, "packing")
    other = _read_placements(out_dir, method)
    if not base or not other:
        return None
    shortage_thr = sorted(rebap_hourly)[36]  # 75th percentile of the window's hourly reBAP
    moved_kwh = 0.0
    legs = {"DE->DE": 0.0, "DE->other": 0.0, "other->DE": 0.0, "other->other": 0.0}
    de_dest_shortage_kwh = 0.0  # energy the method ADDS to DE hours priced in the top reBAP quartile
    for pod_id, b in base.items():
        o = other.get(pod_id)
        if o is None:
            continue
        if b["region"] == o["region"] and b["start_slot"] == o["start_slot"]:
            continue
        kwh = float(o["operational_energy_kwh"])  # energy of the shifted pod under the method
        moved_kwh += kwh
        src = "DE" if b["region"] == "DE" else "other"
        dst = "DE" if o["region"] == "DE" else "other"
        legs[f"{src}->{dst}"] += kwh
        if dst == "DE":
            slot = int(o["start_slot"])
            dur = max(int(float(o["duration"])), 1)
            slots = [s for s in range(slot, slot + dur) if 0 <= s < 48]
            if slots and statistics.mean(rebap_hourly[s] for s in slots) >= shortage_thr:
                de_dest_shortage_kwh += kwh
    return {
        "moved_energy_kwh": moved_kwh,
        "moved_energy_by_leg_kwh": legs,
        "de_dest_energy_in_top_quartile_rebap_hours_kwh": de_dest_shortage_kwh,
        "shortage_threshold_eur_per_mwh": shortage_thr,
    }


def _method_row(s: dict, carbon_prices: Dict[str, float],
                packing: Optional[dict] = None) -> dict:
    row = {
        "carbon_delta_kg": s.get("carbon_delta_kg"),
        "carbon_delta_pct": s.get("carbon_delta_pct"),
        "scarcity_delta_pct": s.get("scarcity_delta_pct"),
        "operational_energy_kwh": s.get("operational_energy_kwh"),
        "operational_cost_eur": s.get("operational_cost_eur"),
        "cost_delta_eur": s.get("cost_delta_eur"),
        "cost_delta_pct": s.get("cost_delta_pct"),
        "weighted_stress_kwh_avoided": s.get("weighted_stress_kwh_avoided"),
        "stress_kwh_avoided": s.get("stress_kwh_avoided"),
        "moved_common_pods": s.get("moved_common_pods"),
        "no_harm_certificate": bool(s.get("no_harm_certificate")),
        "no_harm_certificate_with_cost": bool(s.get("no_harm_certificate_with_cost")),
    }
    cd = s.get("cost_delta_eur")
    ck = s.get("carbon_delta_kg")
    if cd is not None and ck is not None:
        row["carbon_value_eur"] = {k: abs(ck) / 1000.0 * p for k, p in carbon_prices.items()}
        row["value_ratio"] = {
            k: (abs(cd) / v if v > 1e-12 else None) for k, v in row["carbon_value_eur"].items()
        }
        row["deltas_sign_note"] = (
            f"cost_delta {'saves' if cd < 0 else 'COSTS'} money; "
            f"carbon_delta {'reduces' if (ck or 0) < 0 else 'INCREASES'} carbon"
        )
        # Honesty decomposition: how much of the EUR delta is just USING LESS ENERGY (a smaller
        # bill -- not a balancing deviation payment) vs REALLOCATING energy across prices (the
        # part a passive-balancing settlement could actually pay)?
        if packing is not None:
            e_pack = packing.get("operational_energy_kwh") or 0.0
            c_pack = packing.get("operational_cost_eur") or 0.0
            e_this = s.get("operational_energy_kwh") or 0.0
            if e_pack > 1e-12:
                mean_price = c_pack / e_pack
                energy_effect = (e_this - e_pack) * mean_price
                row["cost_delta_decomposition_eur"] = {
                    "energy_effect": energy_effect,
                    "reallocation_effect": cd - energy_effect,
                    "note": "energy_effect = dE * packing mean price (smaller bill from using "
                            "less energy); reallocation_effect = residual (the only part a "
                            "passive-balancing deviation settlement could pay)",
                }
                realloc = cd - energy_effect
                row["value_ratio_reallocation_only"] = {
                    k: (abs(realloc) / v if v > 1e-12 else None)
                    for k, v in row["carbon_value_eur"].items()
                }
    return row


def run_window(testbed: str, window: str, seed: int, rebap_hourly: List[float]) -> dict:
    run_dir = OUT / window
    run_dir.mkdir(parents=True, exist_ok=True)
    if testbed == "alibaba":
        pc, n_nodes, n_pods = wb._build_gpu_config(window, run_dir)
    else:
        pc, n_nodes, n_pods = wb._build_azure_config(window, run_dir, seed=seed)

    variants = {
        "day_ahead": (PRICE_DA, run_dir / "out_da"),
        "rebap_de": (PRICE_REBAP, run_dir / "out_rebap"),
    }
    out = {"testbed": testbed, "window": window, "n_nodes": n_nodes, "n_pods": n_pods,
           "variants": {}}
    consistency: Dict[str, Dict[str, float]] = {}
    for vname, (price_csv, vdir) in variants.items():
        res = run_no_harm_flexibility_pilot(
            replace(pc, output_dir=vdir, cost_enabled=True, price_csv=price_csv))
        summ = _summ(res)
        vout = {"price_csv": str(price_csv.relative_to(REPO_ROOT)), "methods": {}}
        pack_summ = summ.get("packing")
        for m in METHODS:
            if m in summ:
                vout["methods"][m] = _method_row(summ[m], CARBON_PRICES, packing=pack_summ)
                consistency.setdefault(m, {})[vname] = summ[m].get("carbon_delta_kg")
        # shift decomposition (placements are identical across variants; compute once per variant
        # anyway -- it is cheap -- and assert identity below via carbon deltas)
        for m in ("no_harm_flex", "carbon", "waterwise"):
            sa = _shift_analysis(vdir, m, rebap_hourly)
            if sa and m in vout["methods"]:
                mv = vout["methods"][m]
                mv["shift_analysis"] = sa
                cd = mv.get("cost_delta_eur")
                if cd is not None and sa["moved_energy_kwh"] > 1e-9:
                    mv["eur_per_mwh_shifted"] = cd / (sa["moved_energy_kwh"] / 1000.0)
        # naive extrapolation for the envelope
        env = vout["methods"].get("no_harm_flex")
        pack = vout["methods"].get("packing")
        if env and pack and env.get("cost_delta_eur") is not None:
            avg_mw = (pack["operational_energy_kwh"] or 0.0) / WINDOW_HOURS / 1000.0
            annual = -env["cost_delta_eur"] * (HOURS_PER_YEAR / WINDOW_HOURS)
            env["naive_extrapolation"] = {
                "fleet_avg_it_power_mw": avg_mw,
                "eur_saved_per_window": -env["cost_delta_eur"],
                "naive_eur_per_year": annual,
                "naive_eur_per_mw_yr": (annual / avg_mw) if avg_mw > 1e-9 else None,
                "caveat": "48h window scaled by 8760/48; single calm July-2018 regime; "
                          "NOT an annualization claim",
            }
        out["variants"][vname] = vout
    # cross-variant consistency: cost axis is MEASURE-only, schedules must be identical
    for m, vals in consistency.items():
        got = set(round(v, 12) for v in vals.values() if v is not None)
        assert len(got) <= 1, f"schedule changed across price variants for {m}: {vals}"
    out["schedule_invariance_check"] = "carbon_delta_kg identical across price variants (PASS)"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--light", action="store_true")
    args = ap.parse_args()
    logging.disable(logging.CRITICAL)
    OUT.mkdir(parents=True, exist_ok=True)
    rebap_hourly = _rebap_hourly()
    if args.light:
        plan = [("azure", "azure_packing_2020_d0_s42_200", 42),
                ("alibaba", "alibaba_gpu_2020_w1587_s43_200", 43)]
    else:
        plan = ([("alibaba", w, 42) for w in wb.ALIBABA_WINDOWS]
                + [("azure", w, s) for (w, s) in wb.AZURE_WINDOWS])
    results: List[dict] = []
    for testbed, window, seed in plan:
        print(f"\n##### imbalance value: {testbed} / {window} #####", flush=True)
        r = run_window(testbed, window, seed, rebap_hourly)
        results.append(r)
        for vname, v in r["variants"].items():
            env = v["methods"].get("no_harm_flex", {})
            vr = env.get("value_ratio") or {}
            print(f"  [{vname:9s}] envelope: cost_delta={env.get('cost_delta_eur'):+.4f} EUR  "
                  f"carbon_delta={env.get('carbon_delta_kg'):+.4f} kg  "
                  f"ratio@16={vr.get('ets_2018_eur16')}  ratio@80={vr.get('ets_current_eur80')}",
                  flush=True)
    doc = {
        "design": {
            "schedules": "bit-identical to the paper's certified runs (cost axis MEASURE-only)",
            "price_variants": {
                "day_ahead": "Energy-Charts day-ahead, all regions; lower-bound proxy for "
                             "imbalance settlement (imbalance prices are spikier)",
                "rebap_de": "DE = netztransparenz.de reBAP (real German imbalance price, 15-min, "
                            "hourly-averaged onto the 48 engine slots); other regions keep "
                            "day-ahead as a labeled proxy",
            },
            "value_ratio": "|cost_delta_eur| / (|carbon_delta_kg|/1000 * carbon_price)",
            "carbon_prices_eur_per_t": CARBON_PRICES,
        },
        "rebap_hourly_eur_per_mwh": rebap_hourly,
        "windows": results,
    }
    (OUT / "imbalance_value.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'imbalance_value.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
