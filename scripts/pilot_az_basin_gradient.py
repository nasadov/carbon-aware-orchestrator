#!/usr/bin/env python3
"""Extreme-gradient basin pilot: add an Arizona-class region (AZPS/Phoenix) to the US EIA-930 testbed.

Follow-up-paper pilot question: when the site set spans a REAL order-of-magnitude basin-scarcity
gradient (Phoenix Salt-Gila basin at the AWARE cap ~100 vs the Columbia at ~2), (a) how large do the
certified per-basin scarcity-water savings get, and (b) does the envelope's UNIQUENESS survive -- do
carbon-greedy / water-greedy / WaterWise also certify there (the magnitude-vs-uniqueness trade-off
seen in the strong EU Aug-2022 regime)?

ZERO ENGINE MODIFICATIONS. The pilot:
  1. Builds a 4-region US signal set (ERCO, CISO, BPAT, AZPS; 2018-07-23T00Z, 48 h) into
     pkg/carbon-aware/data/timealigned_us2018_az/ by REUSING scripts/build_timealigned_signals.py's
     EIA-930 machinery (runtime-extended with the AZPS site + a hot-climate cooling profile; the
     committed builder file and existing signal dirs are untouched). Carbon = measured EIA-930 mix x
     IPCC AR5 lifecycle EFs; EWIF = the same mix x Macknick 2012 factors; weather = Open-Meteo
     archive for Phoenix; the EIA-930 bulk CSV is reused from the committed cache (offline).
  2. Validates the rebuild: the ERCO/CISO/BPAT rows must match pkg/carbon-aware/data/timealigned_us2018.
  3. Supplies the AWARE scarcity CF for AZPS by the SAME runtime-override pattern as
     experiments/generalize_us/run_us_baselines.py (monkeypatched _water_data_path; canonical water
     files untouched). CF_AZPS(jul) = 100.0, the AWARE 2.0 state-level US-AZ July value from the
     committed aware20_country_nonagri_factors.csv (Arizona is pinned at 99.2-100.0 in EVERY month,
     so any basin choice within the state sits at/near the AWARE cap; the Phoenix AMA / Salt-Gila
     basin is at the cap). A CF_AZPS=50 sensitivity run bounds the conclusion from below.
  4. Runs, per seed (42/43/44): the engine pilot (packing baseline, carbon-greedy, water-greedy,
     WaterWise w=0.5, envelope, footprint control) on (A) the 4-region fleet with the AGGREGATE
     water guard, (B) the 4-region fleet with per_basin_scarcity_guard=True, and (C) the 3-region
     ERCO/CISO/BPAT control fleet (the banked testbed) on the same workload draw.
  5. Re-evaluates EVERY method's water certificate at PER-BASIN resolution from the engine's own
     emitted per-placement scarcity_water (the run_water_basin_certification.py pattern), and
     reports the realized per-kWh scarcity-weighted intensity gradient across sites.

Outputs: experiments/az_gradient/az_gradient.json (+ per-run engine artifacts under
experiments/az_gradient/runs/), console tables.

Run:
    PYTHONPATH=pkg/carbon-aware/server-python python scripts/pilot_az_basin_gradient.py
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SERVER = REPO_ROOT / "pkg" / "carbon-aware" / "server-python"
for _p in (str(SERVER), str(SCRIPT_DIR), str(SCRIPT_DIR / "real_traces")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402

import build_timealigned_signals as bts  # noqa: E402  (reused, runtime-extended; file untouched)
import carbon_aware.no_harm_flexibility as nhf  # noqa: E402
from carbon_aware.no_harm_flexibility import PilotConfig, run_no_harm_flexibility_pilot  # noqa: E402

# ----------------------------------------------------------------------------- constants
SIGNALS_DIR = REPO_ROOT / "pkg" / "carbon-aware" / "data" / "timealigned_us2018_az"
REFERENCE_SIGNALS = REPO_ROOT / "pkg" / "carbon-aware" / "data" / "timealigned_us2018"
EIA_CACHE_SRC = (REPO_ROOT / "pkg" / "carbon-aware" / "data" / "timealigned_us2018_v2"
                 / "cache" / "EIA930_BALANCE_2018_Jul_Dec.csv")
GENERALIZE_OVERRIDES = REPO_ROOT / "experiments" / "generalize_us" / "water_overrides"
CANONICAL_COUNTRY_CSV = (REPO_ROOT / "pkg" / "carbon-aware" / "data" / "water"
                         / "aware20_country_nonagri_factors.csv")
OUT_ROOT = REPO_ROOT / "experiments" / "az_gradient"
OVERRIDES = OUT_ROOT / "water_overrides"
CONFIG_FILE = REPO_ROOT / "pkg" / "carbon-aware" / "infra-workload-config.yaml"

WINDOW_START = "2018-07-23T00:00:00Z"   # the committed US-testbed window (fuel breakdown era)
SLOTS = 48
US3_REGIONS = ["ERCO", "CISO", "BPAT"]
US4_REGIONS = ["ERCO", "CISO", "BPAT", "AZPS"]
SEEDS = [42, 43, 44]
CF_MONTH_COL = "jul_cf"                 # scenario_month default "jul" == the window's month
METHODS = ["packing", "carbon", "water_scarcity", "waterwise",
           "no_harm_search_control", "no_harm_flex"]
TOL = 1e-6

# AZPS = Arizona Public Service (Phoenix metro; the Mesa/Phoenix DC market). EIA-930 respondent has
# full demand + fuel-mix coverage for the window (Palo Verde nuclear + coal + gas + solar).
AZ_SITE = dict(lat=33.4484, lon=-112.0740, ci_base=300.0, eia_respondent="AZPS")
# Hot-climate hybrid-economized cooling profile for Phoenix, mirroring the committed ES/ERCO/CISO
# hotter-climate rows in region_cooling_profiles.csv (same architecture as every other region --
# only the weather differs; the committed CSV is not edited, the profile is injected at runtime).
AZ_COOLING_PROFILE = dict(dry_mode_wue=0.040, evap_multiplier=0.45, act_T=24.0, act_Tw=16.0,
                          activation_rule="threshold", activation_logic="any",
                          profile_class="hybrid_economized")
# AWARE 2.0 CF row for AZPS: the committed state-level US-AZ monthly values (jul = 100.0, the cap).
AZ_CF_SOURCE_CODE = "US-AZ"
AZ_CF_SENSITIVITY_JUL = 50.0

EU_WATER_BENCHMARK_PCT = -10.5  # EU testbed certified fleet water headline (paper 3)


# ----------------------------------------------------------------------------- step 1: signals
def build_signals(force: bool = False) -> None:
    """Build the 4-region US signal set by reusing the committed EIA-930 builder, runtime-extended
    with the AZPS site + cooling profile. Writes ONLY to timealigned_us2018_az/ (a new dir)."""
    if SIGNALS_DIR.exists() and (SIGNALS_DIR / "forecasts.json").exists() and not force:
        print(f"[signals] {SIGNALS_DIR.name} already built; skipping (use --rebuild-signals to force)")
        return
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = SIGNALS_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    bulk_dst = cache_dir / EIA_CACHE_SRC.name
    if not bulk_dst.exists():
        if not EIA_CACHE_SRC.exists():
            raise SystemExit(f"missing committed EIA-930 bulk cache: {EIA_CACHE_SRC}")
        shutil.copy2(EIA_CACHE_SRC, bulk_dst)
        print(f"[signals] reused committed EIA-930 bulk cache -> {bulk_dst.relative_to(REPO_ROOT)}")

    # Runtime extension of the committed builder: +AZPS site, +Phoenix cooling profile.
    bts.US_SITES["AZPS"] = dict(AZ_SITE)
    _orig_lcp = bts.load_cooling_profile

    def _lcp(reg):
        if str(reg).upper() == "AZPS":
            return dict(AZ_COOLING_PROFILE)
        return _orig_lcp(reg)

    bts.load_cooling_profile = _lcp
    try:
        args = argparse.Namespace(carbon_source="real", eia930_cache_dir=str(cache_dir),
                                  start=WINDOW_START)
        start = pd.Timestamp(WINDOW_START)
        hours = pd.date_range(start, periods=SLOTS, freq="h", tz="UTC")
        rc = bts._build_eia930(args, start, SLOTS, hours, SIGNALS_DIR)
        if rc != 0:
            raise SystemExit("signal build failed")
    finally:
        bts.load_cooling_profile = _orig_lcp
        bts.US_SITES.pop("AZPS", None)


def validate_signals() -> Dict[str, float]:
    """The rebuilt ERCO/CISO/BPAT rows must reproduce the committed timealigned_us2018 tables.
    Returns max abs diffs per artifact (weather is refetched from Open-Meteo, so tiny reanalysis
    revisions are possible; anything beyond that fails loudly)."""
    diffs: Dict[str, float] = {}
    ref_fc = json.loads((REFERENCE_SIGNALS / "forecasts.json").read_text())
    new_fc = json.loads((SIGNALS_DIR / "forecasts.json").read_text())
    ci_diff = 0.0
    for reg in US3_REGIONS:
        a = [e["carbonIntensity"] for e in ref_fc[reg]["forecast"]]
        b = [e["carbonIntensity"] for e in new_fc[reg]["forecast"]]
        ci_diff = max(ci_diff, max(abs(x - y) for x, y in zip(a, b)))
    diffs["carbon_gCO2_per_kwh"] = ci_diff
    for fname, cols, tol in (
        ("grid_residual_region_slot.csv", ["residual_load_mw"], 1e-6),
        ("wue_region_slot.csv", ["wet_bulb_c", "direct_wue_l_per_kwh", "pue"], 0.5),
        ("ewif_us_region_slot.csv", ["ewif_l_per_kwh"], 1e-6),
    ):
        ref = pd.read_csv(REFERENCE_SIGNALS / fname)
        new = pd.read_csv(SIGNALS_DIR / fname)
        m = 0.0
        for reg in US3_REGIONS:
            r = ref[ref.region == reg].sort_values("slot_index").reset_index()
            n = new[new.region == reg].sort_values("slot_index").reset_index()
            for c in cols:
                m = max(m, float((r[c] - n[c]).abs().max()))
        diffs[fname] = m
        if m > tol:
            raise SystemExit(f"[validate] {fname}: rebuilt ERCO/CISO/BPAT deviate from the committed "
                             f"table by {m:g} (> {tol}); investigate before trusting the AZ build")
    if ci_diff > 1e-6:
        raise SystemExit(f"[validate] carbon deviates by {ci_diff:g} g/kWh")
    print(f"[validate] rebuilt ERCO/CISO/BPAT match timealigned_us2018 "
          f"(max diffs: {', '.join(f'{k}={v:g}' for k, v in diffs.items())})")
    return diffs


# ----------------------------------------------------------------------------- step 2: overrides
def write_overrides() -> Dict[str, Path]:
    """AWARE-CF + EWIF runtime overrides (run_us_baselines.py pattern; canonical files untouched).
    The AWARE 'country' table is what the engine keys US regions by (region code == country code ==
    basin proxy, exactly as the banked ERCO/CISO/BPAT rows do); we append the AZPS row."""
    OVERRIDES.mkdir(parents=True, exist_ok=True)

    # AZPS CF row = committed AWARE 2.0 state-level US-AZ monthly values.
    with CANONICAL_COUNTRY_CSV.open() as fh:
        rows = list(csv.DictReader(fh))
        fields = rows[0].keys() if rows else []
    az_src = next(r for r in rows if r["country_code"] == AZ_CF_SOURCE_CODE)

    def _az_row(jul_override: float | None = None) -> dict:
        row = dict(az_src)
        row["country_name"] = ("US-AZPS Phoenix metro proxy (AWARE2.0 US-AZ state-level; "
                               "Salt-Gila/Middle Gila basin at the AWARE cap)")
        row["country_code"] = "AZPS"
        if jul_override is not None:
            row["jul_cf"] = f"{jul_override:g}"
            row["country_name"] += f" [SENSITIVITY jul_cf={jul_override:g}]"
        return row

    src_aware = GENERALIZE_OVERRIDES / "aware20_country_nonagri_factors.csv"
    if not src_aware.exists():
        raise SystemExit(f"missing banked US override table: {src_aware}")
    base_txt = src_aware.read_text()
    out: Dict[str, Path] = {}
    for tag, jul in (("base", None), ("cf50", AZ_CF_SENSITIVITY_JUL)):
        dst = OVERRIDES / f"aware20_country_nonagri_factors_{tag}.csv"
        with dst.open("w", newline="") as fh:
            fh.write(base_txt if base_txt.endswith("\n") else base_txt + "\n")
            w = csv.DictWriter(fh, fieldnames=list(fields))
            w.writerow(_az_row(jul))
        out[f"aware_{tag}"] = dst

    # EWIF override = the new build's 4-region in-window table (feeds demand_mw + indirect water).
    ewif_dst = OVERRIDES / "ewif_region_slot.csv"
    shutil.copy2(SIGNALS_DIR / "ewif_us_region_slot.csv", ewif_dst)
    out["ewif"] = ewif_dst
    print(f"[overrides] wrote {', '.join(p.relative_to(REPO_ROOT).as_posix() for p in out.values())}")
    return out


# Switchable redirect (one patch installed once; targets swapped per run).
_REDIRECT: Dict[str, Path] = {}
_ORIG_WDP = nhf._water_data_path


def _patched_wdp(repo_root, filename):
    target = _REDIRECT.get(filename)
    if target is not None and Path(target).exists():
        return Path(target)
    return _ORIG_WDP(repo_root, filename)


nhf._water_data_path = _patched_wdp


def set_water_overrides(aware_csv: Path, ewif_csv: Path) -> None:
    _REDIRECT["aware20_country_nonagri_factors.csv"] = aware_csv
    _REDIRECT["ewif_region_slot.csv"] = ewif_csv


# ----------------------------------------------------------------------------- step 3: fleet+runs
def write_fleet(path: Path, regions: List[str], nodes_per_region: int = 4,
                counts: Dict[str, int] | None = None) -> int:
    """Same V100 DGX-1 node spec as the banked US runs (run_alibaba_no_harm.GPU_NODE).
    `counts` overrides the per-region node count (slack-probe fleets)."""
    import yaml
    from run_alibaba_no_harm import GPU_NODE
    g = GPU_NODE
    path.parent.mkdir(parents=True, exist_ok=True)
    docs, n = [], 0
    for region in regions:
        for k in range((counts or {}).get(region, nodes_per_region)):
            n += 1
            docs.append({
                "apiVersion": "v1", "kind": "Node",
                "metadata": {
                    "name": f"gpu-{region.lower()}-{k}",
                    "labels": {"topology.kubernetes.io/region": region,
                               "hardware.carbon/subcategory": "Server"},
                    "annotations": {
                        "hardware.carbon/embodied_emissions": str(g["chassis_embodied_kg"]),
                        "hardware.carbon/lifetime_years": str(g["lifetime_years"]),
                        "hardware.power/idle_watts": str(g["idle_w"]),
                        "hardware.power/active_watts": str((g["idle_w"] + g["max_w"]) / 2),
                        "hardware.power/max_watts": str(g["max_w"]),
                        "hardware.gpu/count": str(g["gpu_count"]),
                        "hardware.gpu/power_watts": str(g["gpu_power_w"]),
                        "hardware.gpu/idle_watts": str(g["gpu_idle_w"]),
                        "hardware.gpu/embodied_emissions": str(g["gpu_embodied_kg"]),
                        "hardware.gpu/embodied_water": str(g["gpu_embodied_water_l"]),
                        "hardware.gpu/type": g["gpu_type"],
                    },
                },
                "status": {"allocatable": {"cpu": str(g["cpu_cores"]), "memory": f"{g['mem_gi']}Gi"}},
            })
    with path.open("w") as fh:
        for d in docs:
            yaml.safe_dump(d, fh, sort_keys=False)
            fh.write("---\n")
    return n


def make_config(nodes_file: Path, workloads_dir: Path, out_dir: Path, **kw) -> PilotConfig:
    return PilotConfig(
        repo_root=REPO_ROOT, nodes_file=nodes_file, workloads_dir=workloads_dir,
        forecasts_file=SIGNALS_DIR / "forecasts.json", config_file=CONFIG_FILE,
        output_dir=out_dir, max_timeslots=SLOTS, max_pods=None,
        scenario="heatwave-drought", lever_mode="both",
        grid_signal_csv=SIGNALS_DIR / "grid_residual_region_slot.csv",
        wue_csv=SIGNALS_DIR / "wue_region_slot.csv",
        ewif_csv=SIGNALS_DIR / "ewif_us_region_slot.csv",
        **kw,
    )


# ----------------------------------------------------------------------------- step 4: analysis
def per_basin_water(placements_csv: Path) -> Dict[str, float]:
    """Engine-emitted scarcity_water (the certificate axis) summed per basin (== region here)."""
    scar: Dict[str, float] = defaultdict(float)
    with placements_csv.open() as fh:
        for row in csv.DictReader(fh):
            scar[row["region"].upper()] += float(row.get("scarcity_water", 0.0) or 0.0)
    return dict(scar)


def analyze_run(out_dir: Path, summary_rows: List[dict], regions: List[str]) -> dict:
    """Per-method: engine certificate + per-basin scarcity deltas vs the same run's packing."""
    summary = {r["method_key"]: r for r in summary_rows}
    base = per_basin_water(out_dir / "placements_packing.csv")
    per_method: Dict[str, dict] = {}
    for m in METHODS:
        pcsv = out_dir / f"placements_{m}.csv"
        if not pcsv.exists() or m not in summary:
            continue
        cur = per_basin_water(pcsv)
        delta = {b: cur.get(b, 0.0) - base.get(b, 0.0) for b in regions}
        delta_pct = {b: (100.0 * delta[b] / base[b] if base.get(b) else 0.0) for b in regions}
        agg_base = sum(base.values())
        agg = sum(cur.get(b, 0.0) for b in regions)
        worst = max(regions, key=lambda b: delta[b])
        s = summary[m]
        per_method[m] = {
            "engine_cert": bool(s["no_harm_certificate"]),
            "carbon_delta_pct": float(s["carbon_delta_pct"]),
            "scarcity_delta_pct": float(s["scarcity_delta_pct"]),
            "slo_non_decrease": bool(s["slo_non_decrease"]),
            "placed": int(s["placed_pods"]), "unplaced": int(s["unplaced_pods"]),
            "agg_delta_pct": 100.0 * (agg - agg_base) / agg_base if agg_base else 0.0,
            "per_basin_delta_pct": delta_pct,
            "per_basin_cert": all(d <= TOL for d in delta.values()),
            "worst_basin": worst, "worst_basin_delta_pct": delta_pct[worst],
        }
    return {"base_per_basin_scarcity": base, "methods": per_method}


def realized_gradient(cf_by_region: Dict[str, float], regions: List[str]) -> dict:
    """Per-kWh scarcity-weighted intensity per region from the built signals + AWARE CFs:
    direct = mean(direct_wue)*CF_basin, indirect = mean(EWIF)*CF_basin (region==basin==BA here)."""
    wue = pd.read_csv(SIGNALS_DIR / "wue_region_slot.csv")
    ewif = pd.read_csv(SIGNALS_DIR / "ewif_us_region_slot.csv")
    fc = json.loads((SIGNALS_DIR / "forecasts.json").read_text())
    out = {}
    for reg in regions:
        w = wue[wue.region == reg]["direct_wue_l_per_kwh"]
        e = ewif[ewif.region == reg]["ewif_l_per_kwh"]
        ci = [x["carbonIntensity"] for x in fc[reg]["forecast"]]
        cf = cf_by_region[reg]
        out[reg] = {
            "aware_cf_jul": cf,
            "mean_direct_wue_l_per_kwh": round(float(w.mean()), 4),
            "max_direct_wue_l_per_kwh": round(float(w.max()), 4),
            "mean_ewif_l_per_kwh": round(float(e.mean()), 4),
            "mean_ci_g_per_kwh": round(sum(ci) / len(ci), 1),
            "direct_scarcity_leq_per_kwh": round(float(w.mean()) * cf, 3),
            "indirect_scarcity_leq_per_kwh": round(float(e.mean()) * cf, 3),
            "total_scarcity_leq_per_kwh": round((float(w.mean()) + float(e.mean())) * cf, 3),
        }
    lo = min(regions, key=lambda r: out[r]["total_scarcity_leq_per_kwh"])
    hi = max(regions, key=lambda r: out[r]["total_scarcity_leq_per_kwh"])
    out["_gradient"] = {
        "lowest_region": lo, "highest_region": hi,
        "total_ratio_hi_over_lo": round(out[hi]["total_scarcity_leq_per_kwh"]
                                        / out[lo]["total_scarcity_leq_per_kwh"], 1),
        "direct_ratio_AZPS_over_BPAT": round(out["AZPS"]["direct_scarcity_leq_per_kwh"]
                                             / out["BPAT"]["direct_scarcity_leq_per_kwh"], 1)
        if "AZPS" in out and out.get("BPAT", {}).get("direct_scarcity_leq_per_kwh") else None,
    }
    return out


def _cf_map(aware_csv: Path, regions: List[str]) -> Dict[str, float]:
    with aware_csv.open() as fh:
        rows = {r["country_code"]: r for r in csv.DictReader(fh)}
    return {reg: float(rows[reg][CF_MONTH_COL]) for reg in regions}


# ----------------------------------------------------------------------------- driver
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild-signals", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--pods", type=int, default=200)
    ap.add_argument("--nodes-per-region", type=int, default=4)
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--probe-only", action="store_true",
                    help="run ONLY the slack probes (seed 42): (a) same 4-region fleet at lower "
                         "utilization (120 pods), (b) BPAT-heavy fleet 4/4/8/4 at 200 pods; tests "
                         "whether the certified-magnitude cap is slack-limited. Writes "
                         "az_gradient_probe.json (main digest untouched).")
    args = ap.parse_args()
    logging.disable(logging.CRITICAL)

    build_signals(force=args.rebuild_signals)
    signal_diffs = validate_signals()
    paths = write_overrides()
    import alibaba_common as ac  # after sys.path setup; loads the ~1 GB trace once per process

    runs_root = OUT_ROOT / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    fleet4 = runs_root / "fleet_us4" / "nodes.yaml"
    fleet3 = runs_root / "fleet_us3" / "nodes.yaml"
    n4 = write_fleet(fleet4, US4_REGIONS, args.nodes_per_region)
    n3 = write_fleet(fleet3, US3_REGIONS, args.nodes_per_region)

    if args.probe_only:
        s0 = args.seeds[0]
        fleet_b8 = runs_root / "fleet_us4_bpat8" / "nodes.yaml"
        n_b8 = write_fleet(fleet_b8, US4_REGIONS, args.nodes_per_region, counts={"BPAT": 8})
        probe_results: Dict[str, dict] = {}
        probe_plan = [
            # (run_key, fleet, n_nodes, pods, per_basin_guard)
            (f"us4_lowutil_seed{s0}", fleet4, n4, 120, False),
            (f"us4_lowutil_pbg_seed{s0}", fleet4, n4, 120, True),
            (f"us4_bpat8_seed{s0}", fleet_b8, n_b8, args.pods, False),
            (f"us4_bpat8_pbg_seed{s0}", fleet_b8, n_b8, args.pods, True),
        ]
        for run_key, fleet, n_nodes, pods, pbg in probe_plan:
            win = ac.build_window(runs_root / f"inputs_seed{s0}_p{pods}", target_pods=pods, seed=s0)
            set_water_overrides(paths["aware_base"], paths["ewif"])
            cfg = make_config(fleet, win["workloads_dir"], runs_root / run_key / "out")
            if pbg:
                cfg = replace(cfg, per_basin_scarcity_guard=True)
            print(f"\n##### PROBE {run_key}: pods={pods}, nodes={n_nodes}, "
                  f"per_basin_guard={pbg} #####", flush=True)
            res = run_no_harm_flexibility_pilot(cfg)
            analysis = analyze_run(cfg.output_dir, res["summary_rows"], US4_REGIONS)
            analysis.update({"seed": s0, "pods": pods, "n_nodes": n_nodes,
                             "per_basin_guard": pbg, "aware_variant": "aware_base",
                             "flexible_pct": round(win["flexible_pct"], 1)})
            probe_results[run_key] = analysis
            for m, r in analysis["methods"].items():
                print(f"{m:>24} cert={r['engine_cert']!s:5} pbCert={r['per_basin_cert']!s:5} "
                      f"C={r['carbon_delta_pct']:+.2f}% aggW={r['agg_delta_pct']:+.2f}% "
                      f"AZPS={r['per_basin_delta_pct'].get('AZPS', 0.0):+.2f}%")
        probe_json = OUT_ROOT / "az_gradient_probe.json"
        probe_json.write_text(json.dumps(
            {"probe": "slack probes: lower utilization (120 pods) and BPAT-heavy fleet (4/4/8/4)",
             "runs": probe_results}, indent=2), encoding="utf-8")
        print(f"\nwrote {probe_json.relative_to(REPO_ROOT)}")
        return 0

    cf_base = _cf_map(paths["aware_base"], US4_REGIONS)
    cf_50 = _cf_map(paths["aware_cf50"], US4_REGIONS)
    gradient = realized_gradient(cf_base, US4_REGIONS)

    results: Dict[str, dict] = {}
    plan = []  # (run_key, seed, fleet, regions, aware_variant, per_basin_guard)
    for seed in args.seeds:
        plan.append((f"us4_seed{seed}", seed, fleet4, US4_REGIONS, "aware_base", False))
        plan.append((f"us4_pbg_seed{seed}", seed, fleet4, US4_REGIONS, "aware_base", True))
        plan.append((f"us3_seed{seed}", seed, fleet3, US3_REGIONS, "aware_base", False))
    if not args.skip_sensitivity:
        s0 = args.seeds[0]
        plan.append((f"us4_cf50_seed{s0}", s0, fleet4, US4_REGIONS, "aware_cf50", False))
        plan.append((f"us4_pbg_cf50_seed{s0}", s0, fleet4, US4_REGIONS, "aware_cf50", True))

    windows: Dict[int, dict] = {}
    for run_key, seed, fleet, regions, aware_tag, pbg in plan:
        if seed not in windows:
            windows[seed] = ac.build_window(runs_root / f"inputs_seed{seed}",
                                            target_pods=args.pods, seed=seed)
        win = windows[seed]
        set_water_overrides(paths[aware_tag], paths["ewif"])
        cfg = make_config(fleet, win["workloads_dir"], runs_root / run_key / "out")
        if pbg:
            cfg = replace(cfg, per_basin_scarcity_guard=True)
        print(f"\n##### {run_key}: {len(regions)} regions, seed={seed}, "
              f"per_basin_guard={pbg}, aware={aware_tag} #####", flush=True)
        res = run_no_harm_flexibility_pilot(cfg)
        analysis = analyze_run(cfg.output_dir, res["summary_rows"], regions)
        analysis.update({"seed": seed, "regions": regions, "per_basin_guard": pbg,
                         "aware_variant": aware_tag, "n_pods": win["n_pods"],
                         "flexible_pct": round(win["flexible_pct"], 1),
                         "n_nodes": n4 if regions == US4_REGIONS else n3})
        results[run_key] = analysis
        hdr = (f"{'method':>24} {'cert':>5} {'pbCert':>6} {'carbon%':>9} {'aggW%':>9} "
               f"{'worstBasin':>10} {'worst%':>8}")
        print(hdr)
        print("-" * len(hdr))
        for m, r in analysis["methods"].items():
            print(f"{m:>24} {str(r['engine_cert']):>5} {str(r['per_basin_cert']):>6} "
                  f"{r['carbon_delta_pct']:>+9.2f} {r['agg_delta_pct']:>+9.2f} "
                  f"{r['worst_basin']:>10} {r['worst_basin_delta_pct']:>+8.2f}")
        if "AZPS" in regions:
            env_key = "no_harm_flex"
            az = analysis["methods"][env_key]["per_basin_delta_pct"].get("AZPS")
            print(f"  -> envelope AZPS-basin scarcity delta: {az:+.2f}%")

    digest = {
        "pilot": "extreme-gradient basins: AZPS (Phoenix) added to the ERCO/CISO/BPAT US testbed",
        "window": {"start": WINDOW_START, "slots": SLOTS},
        "signals_dir": str(SIGNALS_DIR.relative_to(REPO_ROOT)),
        "signal_validation_max_abs_diffs": signal_diffs,
        "aware_cf_jul": {"base": cf_base, "cf50_sensitivity": cf_50},
        "eu_water_benchmark_pct": EU_WATER_BENCHMARK_PCT,
        "realized_gradient": gradient,
        "runs": results,
        "provenance": {
            "grid_mix_and_demand": "EIA-930 Hourly Grid Monitor bulk BALANCE CSV (public, no key): "
                                   "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/"
                                   "EIA930_BALANCE_2018_Jul_Dec.csv (committed repo cache reused)",
            "weather": "Open-Meteo archive API (CC-BY 4.0): https://archive-api.open-meteo.com/v1/"
                       "archive (Phoenix 33.4484,-112.0740; ERA5-family reanalysis)",
            "carbon_efs": "IPCC AR5 WG3 Annex III lifecycle EFs (committed builder table)",
            "water_factors": "Macknick et al. 2012 (NREL) consumption factors (committed builder table)",
            "aware_cf": "AWARE 2.0 CFs_nonagri (WULCA), Zenodo 16332127; AZPS = committed US-AZ "
                        "state-level monthly row (jul=100.0), NOT a native basin polygon sample "
                        "(Arizona state CF is 99.2-100.0 in every month, so the state-level value "
                        "is a tight proxy for any AZ basin; cf50 run bounds the conclusion below)",
            "workload": "Alibaba GPU cluster trace v2020 (committed raw tables), stratified 200-pod "
                        "draws per seed via scripts/alibaba_common.build_window",
            "parametrized_elements": "AZPS cooling profile (hot-climate hybrid-economized row "
                                     "mirroring committed ES/ERCO/CISO parameters) -- PROVISIONAL; "
                                     "AZPS AWARE CF uses the state-level (not basin-polygon) value",
        },
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_json = OUT_ROOT / "az_gradient.json"
    out_json.write_text(json.dumps(digest, indent=2), encoding="utf-8")
    print(f"\nwrote {out_json.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
