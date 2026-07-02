#!/usr/bin/env python3
"""D-A -- Live Counterfactual-Harm Replay on the KWOK real-Kubernetes control plane.

THE KILLER DEMONSTRATION (paper-three Tier-1-C). On one REAL grid-stress + drought
window (the committed headroom CPU fleet, real-measured-mix CI / OPSD residual /
Open-Meteo wet-bulb signals), a PUBLISHED carbon-aware scheduler a real operator would
deploy -- Carbon-Intelligent Computing (CIC; Google's production lineage) and the
WaterWise weight sweep -- binds a schedule that pushes a water-stressed basin further
into scarcity (above its baseline AND, at the engine's drought resolution, past the
AWARE drought threshold). The No-Harm Envelope, given the SAME window / SAME signals /
SAME cluster, REFUSES exactly those moves and CERTIFIES. The prevented harm is
quantified in scarcity-liters (per AWARE 2.0 watershed), gCO2, and admitted pods, and --
crucially -- water is evaluated at PER-BASIN (watershed) resolution using the committed
AWARE 2.0 native-geospatial basin CFs, so the prevented harm is a genuine watershed
harm, not a country-aggregation artifact.

What is and is NOT claimed (honesty -- KWOK has no kubelet, nothing executes):
  * CLAIMED: realizability (every placement binds on the real apiserver + default
    kube-scheduler), the certificate's REFUSAL of the harmful moves on a real control
    plane, and the prevented harm recomputed ex-post from the BOUND placements + public
    signals (auditable, not asserted by the scheduler).
  * NOT claimed: measured energy/water (no workload runs). The harm magnitudes are the
    engine's accounting on realized signals, identical to the committed baselines digest.

Resolutions reported (both, transparently):
  * COUNTRY  -- the engine's shipped default (aware20_country_nonagri_factors.csv). The
    Milan/Po (IT-NO) and Madrid (ES) basins read as country aggregates whose CF exceeds
    the AWARE drought threshold (20), so the engine's drought guard fires. This is the
    as-shipped headline -- reported, but flagged as country-inflated (memo 08).
  * BASIN    -- AWARE 2.0 native watershed CFs (aware20_basin_nonagri_factors.csv),
    injected by redirecting the engine's hardcoded country-AWARE path (no engine edit;
    the basin file already ships). De-confounded: the harm SURVIVES (carbon-greedy still
    raises every stressed basin above B), proving it is a real watershed harm.
  * BASIN + PER-BASIN GUARD -- memo-08 P1, emulated with the EXISTING drought guard at
    drought_cf_threshold=0 (every basin flagged -> the per-move drought_delta>0 reject
    becomes a per-move per-basin non-degradation guard). Default (threshold=20) unchanged
    and bit-identical. This is the strictly-per-watershed envelope.

Live control plane: kwokctl cluster `carbon` (/tmp/kwok.kubeconfig), 4 Ready fake nodes
(node-0-fr-iot/node-1-es-smartphone/node-2-it-no-laptop/node-3-de-server) == the headroom
fleet's OWN nodes. Reuses the d5_kwok harness primitives (T1 bind, T2 per-slot feasibility,
T3 real default kube-scheduler at the busiest slot, dropped/unbindable pods).

Usage:
    PYTHONPATH=pkg/carbon-aware/server-python python scripts/run_counterfactual_harm_replay.py
    PYTHONPATH=pkg/carbon-aware/server-python python scripts/run_counterfactual_harm_replay.py --skip-kwok
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace as dc_replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SERVER = REPO / "pkg" / "carbon-aware" / "server-python"
D5 = REPO / "experiments" / "d5_kwok"
for p in (str(SERVER), str(REPO / "scripts"), str(D5)):
    if p not in sys.path:
        sys.path.insert(0, p)

import carbon_aware.no_harm_flexibility as nhf  # noqa: E402
from carbon_aware.no_harm_flexibility import (  # noqa: E402
    PilotConfig,
    write_placements_csv,
    build_action_signals,
)

TIMEALIGNED_REALCI = REPO / "pkg" / "carbon-aware" / "data" / "timealigned_realci2"
CONFIG_FILE = REPO / "pkg" / "carbon-aware" / "infra-workload-config.yaml"
OUT = REPO / "experiments" / "demo_counterfactual"
KUBECONFIG = "/tmp/kwok.kubeconfig"

# AWARE drought threshold the engine uses to flag a basin as drought-stressed (CF >= this).
DROUGHT_THRESHOLD = 20.0

_ORIG_WATER_PATH = nhf._water_data_path


def patch_basin():
    """Redirect the engine's hardcoded country-AWARE lookup to the AWARE 2.0 native
    basin (watershed) CF file. No engine edit; the file already ships."""
    def _patched(repo_root, filename):
        if filename == "aware20_country_nonagri_factors.csv":
            filename = "aware20_basin_nonagri_factors.csv"
        return _ORIG_WATER_PATH(repo_root, filename)
    nhf._water_data_path = _patched


def unpatch():
    nhf._water_data_path = _ORIG_WATER_PATH


def headroom_config(out_dir: Path, drought_cf_threshold: float = DROUGHT_THRESHOLD) -> PilotConfig:
    return PilotConfig(
        repo_root=REPO,
        nodes_file=REPO / "pkg" / "carbon-aware" / "nodes.yaml",
        workloads_dir=REPO / "pkg" / "carbon-aware" / "workloads",
        forecasts_file=TIMEALIGNED_REALCI / "forecasts.json",
        config_file=CONFIG_FILE,
        output_dir=out_dir,
        max_timeslots=24,
        max_pods=80,
        scenario="heatwave-drought",
        lever_mode="both",
        grid_signal_csv=TIMEALIGNED_REALCI / "grid_residual_region_slot.csv",
        wue_csv=TIMEALIGNED_REALCI / "wue_region_slot.csv",
        drought_cf_threshold=drought_cf_threshold,
    )


# --------------------------------------------------------------------- accounting
def per_basin_scarcity(result):
    """Scarcity-characterized water (already CF-weighted at the active resolution) per
    basin == per DC region/watershed."""
    out = {}
    for pl in result.placements:
        region = (getattr(pl.candidate.flavour, "region", "") or "").upper()
        out[region] = out.get(region, 0.0) + pl.candidate.footprint.scarcity_characterized_water
    return out


def per_basin_carbon_kg(result):
    out = {}
    for pl in result.placements:
        region = (getattr(pl.candidate.flavour, "region", "") or "").upper()
        out[region] = out.get(region, 0.0) + pl.candidate.footprint.total_carbon_kg
    return out


def scarcity_cf_by_region(signals):
    """The active-resolution AWARE CF per region (constant over slots in this scenario)."""
    out = {}
    for (region, _slot), sig in signals.items():
        out[region] = sig.scarcity_cf
    return out


def field_dict(rows, methods, field):
    return {m: rows[m].get(field) for m in methods if m in rows}


def build_field(resolution, waterwise_weights=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Run the full published-baseline field on the headroom fleet at one resolution.
    Returns (results_by_method, summary_rows_by_method, signals)."""
    logging.disable(logging.CRITICAL)
    from published_baselines import run_all_baselines
    cfg = headroom_config(OUT / f"field_{resolution}")
    res = run_all_baselines(cfg, waterwise_weights=waterwise_weights)
    by_method = {r.method_key: r for r in res["results"]}
    rows = {r["method_key"]: r for r in res["summary_rows"]}
    return by_method, rows, res["signals"], cfg


def perbasin_guarded_envelope(resolution_signals_cfg):
    """memo-08 P1 per-basin-guarded envelope, emulated with drought_cf_threshold=0 on the
    EXISTING engine machinery (basin CFs already injected by the caller). Returns
    (packing, env_pb, row, signals)."""
    from carbon_aware.no_harm_flexibility import (
        load_flavours_for_pilot, load_pods, build_greedy_schedule,
        repair_schedule_no_harm, _rematerialize_under_realized, summarize_against_reference,
    )
    cfg = headroom_config(OUT / "field_basin_perbasinguard", drought_cf_threshold=0.0)
    flavours = load_flavours_for_pilot(cfg)
    pods = load_pods(cfg.workloads_dir, max_pods=cfg.max_pods)
    signals = build_action_signals(flavours, cfg)
    packing, _, _ = build_greedy_schedule(method_key="packing", pods=pods, flavours=flavours, config=cfg)
    env_pb = repair_schedule_no_harm(
        method_key="no_harm_flex_perbasin", baseline=packing, pods=pods, flavours=flavours,
        signals=signals, config=cfg, score_mode="combined")
    for r in (packing, env_pb):
        _rematerialize_under_realized(r, flavours, cfg)
    row = summarize_against_reference(env_pb, packing, signals)
    return packing, env_pb, row, signals, cfg


def harm_block(name, by_method, rows, signals, culprit_key="carbon"):
    """Quantify the prevented harm vs baseline B (packing) for one resolution.
    culprit = a published scheduler; envelope = no_harm_flex."""
    pack = by_method["packing"]
    culprit = by_method[culprit_key]
    env = by_method["no_harm_flex"]
    sc_pack = per_basin_scarcity(pack)
    sc_culprit = per_basin_scarcity(culprit)
    sc_env = per_basin_scarcity(env)
    cf = scarcity_cf_by_region(signals)
    regions = sorted(sc_pack)
    per_basin = []
    for reg in regions:
        d_culprit = sc_culprit.get(reg, 0.0) - sc_pack[reg]
        d_env = sc_env.get(reg, 0.0) - sc_pack[reg]
        per_basin.append({
            "basin": reg,
            "aware_cf": round(cf.get(reg, 0.0), 3),
            "over_drought_threshold": cf.get(reg, 0.0) >= DROUGHT_THRESHOLD,
            "B_scarcity_L": round(sc_pack[reg], 3),
            f"{culprit_key}_scarcity_L": round(sc_culprit.get(reg, 0.0), 3),
            f"{culprit_key}_delta_L": round(d_culprit, 3),
            "envelope_scarcity_L": round(sc_env.get(reg, 0.0), 3),
            "envelope_delta_L": round(d_env, 3),
            # prevented harm in this basin = how much the culprit raised it above B that the envelope did not
            "prevented_scarcity_L": round(max(d_culprit, 0.0) - max(d_env, 0.0), 3),
        })
    car_pack = per_basin_carbon_kg(pack)
    car_culprit = per_basin_carbon_kg(culprit)
    return {
        "resolution": name,
        "culprit": culprit_key,
        "B_scarcity_total_L": round(sum(sc_pack.values()), 3),
        f"{culprit_key}_scarcity_total_L": round(sum(sc_culprit.values()), 3),
        "envelope_scarcity_total_L": round(sum(sc_env.values()), 3),
        f"{culprit_key}_scarcity_delta_pct": round(rows[culprit_key]["scarcity_delta_pct"], 3),
        "envelope_scarcity_delta_pct": round(rows["no_harm_flex"]["scarcity_delta_pct"], 3),
        f"{culprit_key}_carbon_delta_pct": round(rows[culprit_key]["carbon_delta_pct"], 3),
        "envelope_carbon_delta_pct": round(rows["no_harm_flex"]["carbon_delta_pct"], 3),
        f"{culprit_key}_certified": bool(rows[culprit_key]["no_harm_certificate"]),
        "envelope_certified": bool(rows["no_harm_flex"]["no_harm_certificate"]),
        f"{culprit_key}_drought_scarcity_L": round(rows[culprit_key].get("drought_scarcity_water", 0.0), 3),
        "B_drought_scarcity_L": round(rows["packing"].get("drought_scarcity_water", 0.0), 3),
        "envelope_drought_scarcity_L": round(rows["no_harm_flex"].get("drought_scarcity_water", 0.0), 3),
        # prevented total scarcity (sum of per-basin prevented harm)
        "prevented_scarcity_total_L": round(sum(b["prevented_scarcity_L"] for b in per_basin), 3),
        "prevented_scarcity_pct_of_B": round(
            100.0 * sum(b["prevented_scarcity_L"] for b in per_basin) / max(sum(sc_pack.values()), 1e-9), 2),
        "per_basin": per_basin,
        # carbon trade the culprit makes to cause the water harm
        "culprit_carbon_kg": round(sum(car_culprit.values()), 3),
        "B_carbon_kg": round(sum(car_pack.values()), 3),
        # dropped pods (SLO harm axis)
        f"{culprit_key}_placed_pods": int(rows[culprit_key]["placed_pods"]),
        f"{culprit_key}_unplaced_pods": int(rows[culprit_key]["unplaced_pods"]),
        "envelope_placed_pods": int(rows["no_harm_flex"]["placed_pods"]),
        "B_placed_pods": int(rows["packing"]["placed_pods"]),
    }


# --------------------------------------------------------------------- KWOK replay
def kwok_replay(schedule_csv: Path, label: str):
    """Replay one schedule on the live control plane using the d5_kwok harness primitives.
    Returns realizability metrics incl. dropped/unbindable pods at the busiest slot."""
    import kwok_refresh_no_harm as kw
    from kubernetes import client, config as kcfg
    kcfg.load_kube_config(config_file=KUBECONFIG)
    v1 = client.CoreV1Api()
    sched = kw.load_placements(schedule_csv)
    want_nodes = kw.referenced_nodes(sched)
    created = kw.register_nodes(v1, want_nodes)
    cap = kw.node_allocatable(v1, want_nodes)
    viol, peak = kw.t2_feasibility(sched, cap)
    t1 = kw.t1_integration(v1, sched)
    t3 = kw.t3_real_scheduler_busiest(v1, sched, cap, want_nodes)
    kw.clear_pods(v1)
    for name in created:
        try:
            v1.delete_node(name)
        except Exception:
            pass
    dropped_t3 = t3["active_pods"] - t3["real_scheduler_placed"]
    return {
        "label": label,
        "schedule_csv": str(schedule_csv),
        "pods": len(sched),
        "nodes": sorted(want_nodes),
        "real_allocatable": cap,
        "T1_bound": t1["bound"], "T1_rejected": t1["rejected"],
        "T1_bind_p50_ms": t1["bind_p50_ms"], "T1_bind_p95_ms": t1["bind_p95_ms"],
        "T2_capacity_violations": viol, "T2_peak_node_util_frac": peak,
        "T3_busiest_slot": t3["busiest_slot"], "T3_active_pods": t3["active_pods"],
        "T3_real_scheduler_placed": t3["real_scheduler_placed"],
        "T3_dropped_unschedulable": dropped_t3,
        "T3_total_active_cpu": t3["total_active_cpu"], "T3_cluster_cpu": t3["cluster_cpu"],
    }


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-kwok", action="store_true", help="skip the live control-plane replay")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    digest = {"window": "headroom CPU fleet (DE/FR/ES/IT-NO), real-CI / OPSD residual / Open-Meteo wet-bulb",
              "baseline_B": "packing (deadline-blind, earliest-feasible)",
              "drought_threshold_cf": DROUGHT_THRESHOLD}

    # ---- COUNTRY resolution (engine default, as-shipped headline) ----
    unpatch()
    by_c, rows_c, sig_c, cfg_c = build_field("country")
    digest["country"] = harm_block("country", by_c, rows_c, sig_c, culprit_key="carbon")
    # emit placement CSVs (country) for the live replay + audit
    write_placements_csv(OUT / "placements_country_packing.csv", by_c["packing"], sig_c)
    write_placements_csv(OUT / "placements_country_carbon_greedy.csv", by_c["carbon"], sig_c)
    write_placements_csv(OUT / "placements_country_envelope.csv", by_c["no_harm_flex"], sig_c)

    # ---- BASIN resolution (de-confounded; harm must survive) ----
    patch_basin()
    by_b, rows_b, sig_b, cfg_b = build_field("basin")
    digest["basin"] = harm_block("basin", by_b, rows_b, sig_b, culprit_key="carbon")
    write_placements_csv(OUT / "placements_basin_packing.csv", by_b["packing"], sig_b)
    write_placements_csv(OUT / "placements_basin_carbon_greedy.csv", by_b["carbon"], sig_b)
    write_placements_csv(OUT / "placements_basin_envelope.csv", by_b["no_harm_flex"], sig_b)

    # ---- BASIN + PER-BASIN GUARD (memo-08 P1, emulated via threshold=0) ----
    pack_pb, env_pb, row_pb, sig_pb, cfg_pb = perbasin_guarded_envelope(None)
    sc_pack = per_basin_scarcity(pack_pb)
    sc_env = per_basin_scarcity(env_pb)
    cf_pb = scarcity_cf_by_region(sig_pb)
    pb_rows = []
    worst = 0.0
    for reg in sorted(sc_pack):
        d = sc_env.get(reg, 0.0) - sc_pack[reg]
        worst = max(worst, d)
        pb_rows.append({"basin": reg, "aware_cf": round(cf_pb.get(reg, 0.0), 3),
                        "B_scarcity_L": round(sc_pack[reg], 3),
                        "env_perbasin_scarcity_L": round(sc_env.get(reg, 0.0), 3),
                        "delta_L": round(d, 3)})
    digest["basin_perbasin_guard"] = {
        "note": "memo-08 P1 emulated via existing drought guard at drought_cf_threshold=0 (no engine edit; default=20 unchanged & bit-identical)",
        "certified": bool(row_pb["no_harm_certificate"]),
        "scarcity_delta_pct": round(row_pb["scarcity_delta_pct"], 3),
        "carbon_delta_pct": round(row_pb["carbon_delta_pct"], 3),
        "repairs_applied": int(row_pb["repairs_applied"]),
        "worst_per_basin_rise_L": round(worst, 4),
        "per_basin": pb_rows,
    }
    write_placements_csv(OUT / "placements_basin_envelope_perbasinguard.csv", env_pb, sig_pb)
    unpatch()

    # ---- live KWOK replay: country placements (the as-shipped headline) ----
    if not args.skip_kwok:
        digest["kwok_replay"] = {
            "cluster": "carbon (kwokctl; real apiserver/etcd/default-scheduler; KWOK fake kubelet)",
            "claim": "realizability + certificate refusal on a REAL control plane; NO measured energy (no kubelet)",
            "carbon_greedy_culprit": kwok_replay(OUT / "placements_country_carbon_greedy.csv", "carbon_greedy (CULPRIT)"),
            "envelope": kwok_replay(OUT / "placements_country_envelope.csv", "no_harm_envelope"),
            "baseline_B": kwok_replay(OUT / "placements_country_packing.csv", "packing (B)"),
        }

    (OUT / "harm_replay_digest.json").write_text(json.dumps(digest, indent=2))
    print(json.dumps(digest, indent=2))
    print(f"\n-> {OUT / 'harm_replay_digest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
