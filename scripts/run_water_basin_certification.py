#!/usr/bin/env python3
"""Per-basin water certification on the six real-trace windows (Option A water model).

Water model (Option A, post-revamp). Each placement's scarcity-weighted OPERATIONAL water is
characterized at the resolution where each flow physically occurs (ISO 14046 / AWARE 2.0):

    scarcity_op = direct_l * CF_basin(DC watershed)   # on-site cooling, withdrawn at the DC's watershed
                + indirect_l * CF_country(generation)  # power-plant water, coarse domestic-grid proxy

Embodied (manufacturing) water is reported but excluded from this guarded axis (its fab watershed is
not a degree of freedom of the scheduling action). The engine emits exactly this quantity per placement
in the ``scarcity_water`` column of placements_<method>.csv, so THIS SCRIPT READS THAT COLUMN as the
source of truth for the certificate and only re-derives the direct/indirect decomposition for the
paper's contrast (asserting the re-derivation matches the engine to floating tolerance).

What it does:
  (1) Re-evaluates the no-harm WATER certificate at PER-BASIN resolution: Delta W_b <= 0 for EVERY
      watershed b (not just the aggregate sum), on the engine's own per-placement scarcity_water.
  (2) Does it for BOTH certified frontier members -- the relief-priority envelope (no_harm_flex) and
      the footprint-priority member (no_harm_search_control) -- plus carbon-greedy and WaterWise for
      contrast, on the Alibaba GPU windows (w1575/w1587/w1600) and the Azure windows (d0/d2/d4),
      deciding on MEASURED CI (timealigned_realci).
  (3) Quantifies the on-site mischarge the revamp fixes: what the OLD all-country characterization
      (direct+indirect both at the country CF) overstated on-site water by, vs Option A. At basin
      resolution the DC-watershed CFs are far below the country means (e.g. IT Po ~1.83 vs country ~41),
      so the binary drought guard (CF>=20) never fires -- it was a country-aggregation artifact, now
      subsumed by the per-basin Delta W_b <= 0 guard.

NO ENGINE EDIT. This is a faithful re-characterization using only committed data + the engine's own
emitted per-placement water; it never overwrites timealigned*/.

Run:
  PYTHONPATH=pkg/carbon-aware/server-python python scripts/run_water_basin_certification.py
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SERVER = REPO_ROOT / "pkg" / "carbon-aware" / "server-python"
for p in (str(SERVER), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import azure_common as ac  # noqa: E402
from run_alibaba_no_harm import write_gpu_fleet  # noqa: E402
from carbon_aware.no_harm_flexibility import (  # noqa: E402
    PilotConfig, load_pods, run_no_harm_flexibility_pilot,
)

REALCI = REPO_ROOT / "pkg" / "carbon-aware" / "data" / "timealigned_realci2"  # coherent revamp build (Option A + in-window EWIF + continuous cooling)
CONFIG_FILE = REPO_ROOT / "pkg" / "carbon-aware" / "infra-workload-config.yaml"
BASIN_CF_CSV = REPO_ROOT / "pkg" / "carbon-aware" / "data" / "water" / "aware20_basin_nonagri_factors.csv"
COUNTRY_CF_CSV = REPO_ROOT / "pkg" / "carbon-aware" / "data" / "water" / "aware20_country_nonagri_factors.csv"
OUT_ROOT = REPO_ROOT / "experiments" / "water_basin"

# July is the heatwave-drought scenario month the engine stamps (jul_cf); we re-characterize on the
# SAME month so the decomposition matches the engine exactly.
CF_MONTH_COL = "jul_cf"
DROUGHT_CF_THRESHOLD = 20.0  # legacy PilotConfig.drought_cf_threshold (now inert at basin resolution)
RECONCILE_TOL = 1e-6        # max |re-derived Option-A scarcity - engine scarcity_water| we tolerate

ALIBABA_WINDOWS = [
    "alibaba_gpu_2020_w1575_s42_200",
    "alibaba_gpu_2020_w1587_s43_200",
    "alibaba_gpu_2020_w1600_s44_200",
]
AZURE_WINDOWS = [
    ("azure_packing_2020_d0_s42_200", 42),
    ("azure_packing_2020_d2_s43_200", 43),
    ("azure_packing_2020_d4_s44_200", 44),
]
# The two certified frontier members + the two harm-prone baselines for the per-basin contrast.
FRONTIER = ["no_harm_flex", "no_harm_search_control"]
BASELINES = ["carbon", "water_scarcity", "waterwise"]
METHODS = ["packing"] + FRONTIER + BASELINES


def _region_to_basin(region: str) -> str:
    """Basin label = the watershed proxy for the region's DC metro. One region == one watershed here
    (each DC sits in a single native AWARE watershed by point-in-polygon over the metro proxy)."""
    return region.upper()


def _safe_float(value, default: float = 1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _basin_cf() -> Dict[str, float]:
    """region -> July basin (watershed) AWARE CF, from the committed native-geospatial table."""
    out: Dict[str, float] = {}
    with BASIN_CF_CSV.open() as f:
        for row in csv.DictReader(f):
            reg = (row.get("region") or "").strip().upper()
            if reg:
                out[reg] = _safe_float(row.get(CF_MONTH_COL))
    return out


def _country_cf() -> Dict[str, float]:
    """country_code -> July country AWARE CF (the indirect/generation resolution, and the old
    on-site mischarge resolution)."""
    out: Dict[str, float] = {}
    with COUNTRY_CF_CSV.open() as f:
        for row in csv.DictReader(f):
            cc = (row.get("country_code") or "").strip().upper()
            if cc:
                out[cc] = _safe_float(row.get(CF_MONTH_COL))
    return out


def _country_for_region(region: str) -> str:
    if region == "IT-NO":
        return "IT"
    return region.split("-", 1)[0].upper() if "-" in region else region.upper()


def _components_by_basin(placements_csv: Path) -> Dict[str, Dict[str, float]]:
    """Per basin: summed direct litres, indirect litres, and the engine's emitted scarcity_water
    (Option A: direct*CF_basin + indirect*CF_country, operational, embodied-excluded) for one method."""
    direct: Dict[str, float] = defaultdict(float)
    indirect: Dict[str, float] = defaultdict(float)
    scar_engine: Dict[str, float] = defaultdict(float)
    with placements_csv.open() as f:
        for row in csv.DictReader(f):
            b = _region_to_basin(row["region"])
            direct[b] += float(row["direct_water_l"])
            indirect[b] += float(row["indirect_water_l"])
            scar_engine[b] += float(row.get("scarcity_water", 0.0) or 0.0)
    basins = set(direct) | set(indirect) | set(scar_engine)
    return {b: {"direct_l": direct[b], "indirect_l": indirect[b], "scar_engine": scar_engine[b]}
            for b in basins}


def _scarcity_optionA(comp: Dict[str, Dict[str, float]], cf_basin: Dict[str, float],
                      cf_country: Dict[str, float]) -> Dict[str, float]:
    """Option A per-basin scarcity: direct*CF_basin + indirect*CF_country (the certificate axis)."""
    return {b: d["direct_l"] * cf_basin.get(b, 1.0) + d["indirect_l"] * cf_country.get(b, 1.0)
            for b, d in comp.items()}


def _scarcity_allcountry(comp: Dict[str, Dict[str, float]], cf_country: Dict[str, float]) -> Dict[str, float]:
    """The OLD mischarge: both on-site AND power-plant water charged at the country CF (overstates the
    DC watershed). Reported only to quantify how much the basin correction moves the on-site charge."""
    return {b: (d["direct_l"] + d["indirect_l"]) * cf_country.get(b, 1.0) for b, d in comp.items()}


def _reconcile(comp: Dict[str, Dict[str, float]], scar_A: Dict[str, float], window: str, method: str) -> None:
    """Safeguard: the script's re-derived Option-A scarcity must equal the engine's emitted
    scarcity_water per basin (the engine is the source of truth). Catches any CF-resolution drift
    between this script and the engine -- exactly the class of bug the revamp targets."""
    for b, d in comp.items():
        diff = abs(scar_A.get(b, 0.0) - d["scar_engine"])
        scale = max(1.0, abs(d["scar_engine"]))
        if diff / scale > RECONCILE_TOL:
            raise SystemExit(
                f"[reconcile] {window}/{method} basin {b}: re-derived Option-A scarcity "
                f"{scar_A.get(b, 0.0):.6f} != engine scarcity_water {d['scar_engine']:.6f} "
                f"(diff {diff:.3e}). Engine CF resolution and this script disagree.")


def _load_placements(placements_csv: Path) -> Dict[str, dict]:
    """pod_id -> {basin, direct_l, indirect_l, carbon_kg} for one method's schedule."""
    out: Dict[str, dict] = {}
    with placements_csv.open() as f:
        for row in csv.DictReader(f):
            out[row["pod_id"]] = {
                "basin": _region_to_basin(row["region"]),
                "direct_l": float(row["direct_water_l"]),
                "indirect_l": float(row["indirect_water_l"]),
                "carbon_kg": float(row["carbon_kg"]),
            }
    return out


def _wb(direct_l: float, indirect_l: float, b: str,
        cf_basin: Dict[str, float], cf_country: Dict[str, float]) -> float:
    """Option-A scarcity for a basin's (direct, indirect) litres."""
    return direct_l * cf_basin.get(b, 1.0) + indirect_l * cf_country.get(b, 1.0)


def _per_basin_guarded_member(base_csv: Path, member_csv: Path, regions: List[str],
                              cf_basin: Dict[str, float], cf_country: Dict[str, float]) -> dict:
    """Per-basin-CERTIFIED frontier member, demonstrated WITHOUT an engine edit.

    The engine's repair loop guards the AGGREGATE Option-A water sum, so a member could in principle
    shift water INTO a basin while the aggregate falls. Here we replay the member's accepted per-pod
    moves under the STRONGER per-basin guard: starting from packing B, adopt a pod's new placement ONLY
    if doing so keeps EVERY basin's Option-A scarcity (direct*CF_basin + indirect*CF_country) <= B's
    (Delta W_b <= 0 for all b). Moves are considered in carbon-savings order; the per-basin water vector
    (split into direct/indirect litres so each flow is characterized at its own resolution) is maintained
    incrementally -- exactly the acceptance rule a decision-time per-basin guard would apply.

    SCOPE. Joint resource capacity of the accepted subset is owned by the production per-basin guard
    inside the engine's repair loop; this post-hoc replay is the no-engine-edit demonstration that such a
    guard certifies per-basin and quantifies its (typically smaller) reported co-benefit.
    """
    base = _load_placements(base_csv)
    member = _load_placements(member_csv)
    common = [p for p in base if p in member]

    base_d = {b: 0.0 for b in regions}
    base_i = {b: 0.0 for b in regions}
    base_carbon = 0.0
    for p in common:
        base_d[base[p]["basin"]] += base[p]["direct_l"]
        base_i[base[p]["basin"]] += base[p]["indirect_l"]
        base_carbon += base[p]["carbon_kg"]
    base_scar = {b: _wb(base_d[b], base_i[b], b, cf_basin, cf_country) for b in regions}

    moves = [p for p in common
             if member[p]["basin"] != base[p]["basin"]
             or abs(member[p]["direct_l"] - base[p]["direct_l"]) > 1e-12
             or abs(member[p]["indirect_l"] - base[p]["indirect_l"]) > 1e-12
             or abs(member[p]["carbon_kg"] - base[p]["carbon_kg"]) > 1e-12]
    moves.sort(key=lambda p: member[p]["carbon_kg"] - base[p]["carbon_kg"])

    cur_d = dict(base_d)
    cur_i = dict(base_i)
    accepted, rejected = 0, 0
    cur_carbon = base_carbon
    for p in moves:
        src_b, dst_b = base[p]["basin"], member[p]["basin"]
        td, ti = dict(cur_d), dict(cur_i)
        td[src_b] -= base[p]["direct_l"]; ti[src_b] -= base[p]["indirect_l"]
        td[dst_b] += member[p]["direct_l"]; ti[dst_b] += member[p]["indirect_l"]
        ok = all(_wb(td[b], ti[b], b, cf_basin, cf_country) <= base_scar[b] + 1e-9 for b in regions)
        if ok:
            cur_d, cur_i = td, ti
            cur_carbon += member[p]["carbon_kg"] - base[p]["carbon_kg"]
            accepted += 1
        else:
            rejected += 1

    cur_scar = {b: _wb(cur_d[b], cur_i[b], b, cf_basin, cf_country) for b in regions}
    agg, agg_base = sum(cur_scar.values()), sum(base_scar.values())
    delta = {b: cur_scar[b] - base_scar[b] for b in regions}
    return {
        "moves_total": len(moves), "moves_accepted": accepted, "moves_rejected": rejected,
        "per_basin_cert": all(delta[b] <= 1e-6 for b in regions),
        "agg_delta_pct": 100.0 * (agg - agg_base) / agg_base if agg_base else 0.0,
        "carbon_delta_kg": cur_carbon - base_carbon,
        "per_basin_delta": delta,
        "per_basin_delta_pct": {
            b: (100.0 * delta[b] / base_scar[b] if base_scar.get(b) else 0.0) for b in regions
        },
    }


# ----------------------------------------------------------------------------- pilot configs
def _build_gpu_config(window: str, run_dir: Path) -> Tuple[PilotConfig, int, int]:
    window_dir = (REPO_ROOT / "experiments" / "real_traces" / window).resolve()
    workloads = window_dir / "workloads"
    if not workloads.exists():
        raise SystemExit(f"no workloads/ under {window_dir}")
    fleet_dir = run_dir / "fleet"
    fleet_dir.mkdir(parents=True, exist_ok=True)
    nodes_file = fleet_dir / "nodes.yaml"
    n_nodes = write_gpu_fleet(nodes_file, 4)
    pc = PilotConfig(
        repo_root=REPO_ROOT, nodes_file=nodes_file, workloads_dir=workloads,
        forecasts_file=REALCI / "forecasts.json", config_file=CONFIG_FILE,
        output_dir=run_dir / "out", max_timeslots=48, max_pods=None,
        scenario="heatwave-drought", lever_mode="both",
        grid_signal_csv=REALCI / "grid_residual_region_slot.csv",
        wue_csv=REALCI / "wue_region_slot.csv",
        ewif_csv=((REALCI / "ewif_region_slot.csv") if (REALCI / "ewif_region_slot.csv").exists() else None),
    )
    return pc, n_nodes, len(load_pods(workloads))


def _build_azure_config(window: str, run_dir: Path, *, seed: int,
                        target_util: float = 0.55) -> Tuple[PilotConfig, int, int]:
    src = (REPO_ROOT / "experiments" / "real_traces" / window).resolve()
    canonical = src / "canonical_trace_workload.csv"
    if not canonical.exists():
        raise SystemExit(f"no canonical_trace_workload.csv under {src}")
    stats = ac.build_window_from_canonical(canonical, run_dir, seed=seed, cpu_util=ac.conv.CPU_UTIL_MEAN)
    workloads = stats["workloads_dir"]
    peak = ac.workload_peak_cores(workloads)
    per_region = ac.provision_for_util(peak, target_util)
    nodes_file, n_nodes, _cap = ac.build_fleet(run_dir / "fleet", per_region)
    pc = ac.make_pilot_config(
        nodes_file, workloads, run_dir / "out", signals=REALCI,
        max_timeslots=48, scenario="heatwave-drought", lever_mode="both",
    )
    return pc, n_nodes, len(load_pods(workloads))


# ----------------------------------------------------------------------------- per-window driver
def run_window(testbed: str, window: str, *, seed: int) -> dict:
    cf_basin_raw = _basin_cf()
    cf_country_raw = _country_cf()
    regions = ["DE", "FR", "ES", "IT-NO"]
    cf_country = {r: cf_country_raw[_country_for_region(r)] for r in regions}
    cf_basin = {r: cf_basin_raw[r] for r in regions}

    run_dir = OUT_ROOT / window
    run_dir.mkdir(parents=True, exist_ok=True)
    if testbed == "alibaba":
        pc, n_nodes, n_pods = _build_gpu_config(window, run_dir)
    else:
        pc, n_nodes, n_pods = _build_azure_config(window, run_dir, seed=seed)

    res = run_no_harm_flexibility_pilot(pc)
    summary = {r["method_key"]: r for r in res["summary_rows"]}
    out_dir = pc.output_dir

    base_country_agg = summary["packing"]["scarcity_water"]  # engine's aggregate (Option A) for B

    base_comp = _components_by_basin(out_dir / "placements_packing.csv")
    base_A = _scarcity_optionA(base_comp, cf_basin, cf_country)
    _reconcile(base_comp, base_A, window, "packing")
    base_allcountry = _scarcity_allcountry(base_comp, cf_country)

    methods_out: Dict[str, dict] = {}
    for m in METHODS:
        pcsv = out_dir / f"placements_{m}.csv"
        if not pcsv.exists():
            continue
        comp = _components_by_basin(pcsv)
        s_A = _scarcity_optionA(comp, cf_basin, cf_country)        # Option A: direct@basin + indirect@country
        _reconcile(comp, s_A, window, m)                           # must match the engine's scarcity_water
        s_allcountry = _scarcity_allcountry(comp, cf_country)      # old all-country mischarge (counterfactual)

        basin_delta = {b: s_A.get(b, 0.0) - base_A.get(b, 0.0) for b in regions}
        allcountry_delta = {b: s_allcountry.get(b, 0.0) - base_allcountry.get(b, 0.0) for b in regions}

        agg_A = sum(s_A.values()); agg_A_base = sum(base_A.values())
        agg_ac = sum(s_allcountry.values()); agg_ac_base = sum(base_allcountry.values())

        tol = 1e-6
        per_basin_cert = all(d <= tol for d in basin_delta.values())
        worst_basin = max(regions, key=lambda b: basin_delta[b])

        methods_out[m] = {
            "engine_no_harm_certificate": bool(summary[m]["no_harm_certificate"]),
            "engine_carbon_delta_pct": summary[m]["carbon_delta_pct"],
            "engine_scarcity_delta_pct": summary[m]["scarcity_delta_pct"],
            # Option A (the certificate's water axis), per-basin and aggregate:
            "scarcity_agg": agg_A,
            "agg_delta_pct": 100.0 * (agg_A - agg_A_base) / agg_A_base if agg_A_base else 0.0,
            "per_basin_delta": {b: basin_delta[b] for b in regions},
            "per_basin_delta_pct": {
                b: (100.0 * basin_delta[b] / base_A[b] if base_A.get(b) else 0.0) for b in regions
            },
            "per_basin_cert": per_basin_cert,
            "worst_basin": worst_basin,
            "worst_delta": basin_delta[worst_basin],
            "worst_delta_pct": (100.0 * basin_delta[worst_basin] / base_A[worst_basin]
                                if base_A.get(worst_basin) else 0.0),
            # old all-country mischarge (counterfactual, to quantify the on-site correction):
            "allcountry_agg": agg_ac,
            "allcountry_agg_delta_pct": 100.0 * (agg_ac - agg_ac_base) / agg_ac_base if agg_ac_base else 0.0,
            "allcountry_per_basin_delta": {b: allcountry_delta[b] for b in regions},
        }

    guarded: Dict[str, dict] = {}
    for m in FRONTIER:
        if (out_dir / f"placements_{m}.csv").exists():
            guarded[m] = _per_basin_guarded_member(
                out_dir / "placements_packing.csv", out_dir / f"placements_{m}.csv",
                regions, cf_basin, cf_country,
            )

    # On-site mischarge magnitude: per basin, country CF / basin CF (how much the old model overstated
    # the DC watershed). And the (now inert) drought-guard firing at each resolution.
    onsite_overstatement = {b: (cf_country[b] / cf_basin[b] if cf_basin[b] else float("nan")) for b in regions}
    drought = {
        "threshold": DROUGHT_CF_THRESHOLD,
        "country_cf": cf_country,
        "basin_cf": cf_basin,
        "country_fires_basins": [b for b in regions if cf_country[b] >= DROUGHT_CF_THRESHOLD],
        "basin_fires_basins": [b for b in regions if cf_basin[b] >= DROUGHT_CF_THRESHOLD],
    }

    return {
        "testbed": testbed, "window": window, "n_nodes": n_nodes, "n_pods": n_pods,
        "decide_signals": "timealigned_realci", "cf_month": CF_MONTH_COL,
        "water_model": "optionA: direct@basin + indirect@country (operational, embodied-excluded)",
        "regions": regions, "cf_country": cf_country, "cf_basin": cf_basin,
        "onsite_overstatement_country_over_basin": onsite_overstatement,
        "base_scarcity_agg_engine": base_country_agg,
        "base_per_basin": base_A, "base_allcountry_per_basin": base_allcountry,
        "methods": methods_out, "per_basin_guarded": guarded, "drought": drought,
    }


def main() -> int:
    logging.disable(logging.CRITICAL)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    plan = ([("alibaba", w, 42) for w in ALIBABA_WINDOWS]
            + [("azure", w, s) for (w, s) in AZURE_WINDOWS])
    results: List[dict] = []
    for testbed, window, seed in plan:
        print(f"\n##### per-basin water cert (Option A): {testbed} / {window} #####", flush=True)
        r = run_window(testbed, window, seed=seed)
        results.append(r)
        env = r["methods"]["no_harm_flex"]
        fp = r["methods"]["no_harm_search_control"]
        ov = r["onsite_overstatement_country_over_basin"]
        print(f"  basin CF  {min(r['cf_basin'].values()):.2f}->{max(r['cf_basin'].values()):.2f}"
              f"  | on-site overstatement (country/basin) max={max(ov.values()):.1f}x")
        for name, m in (("relief (no_harm_flex)", env), ("footprint (search_control)", fp)):
            print(f"  {name:30s} aggΔ={m['agg_delta_pct']:+7.3f}%  "
                  f"per-basin cert={m['per_basin_cert']!s:5}  "
                  f"worst basin={m['worst_basin']} ({m['worst_delta_pct']:+.3f}%)")
        for name, key in (("relief", "no_harm_flex"), ("footprint", "no_harm_search_control")):
            g = r["per_basin_guarded"][key]
            print(f"  [per-basin GUARD] {name:9s} accepted {g['moves_accepted']}/{g['moves_total']} moves  "
                  f"aggΔ={g['agg_delta_pct']:+7.3f}%  carbonΔ={g['carbon_delta_kg']:+.4f}kg  "
                  f"per-basin cert={g['per_basin_cert']!s:5}")
        print(f"  drought guard fires (country): {r['drought']['country_fires_basins']}   "
              f"(basin): {r['drought']['basin_fires_basins']}")

    digest = OUT_ROOT / "basin_certification.json"
    digest.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
