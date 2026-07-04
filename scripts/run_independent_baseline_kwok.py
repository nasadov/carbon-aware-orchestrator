#!/usr/bin/env python3
"""P3.3 -- Independent baseline B_ind: stock kube-scheduler placement via the KWOK
control plane, evaluated with OUR footprint accounting.

Answers the self-grading / baseline-ownership critique (paper-three Achilles heel):
every baseline in the paper is one the authors implemented, so a skeptic can argue the
no-harm certificate is easy against baselines we chose. B_ind removes that objection:
the node placement is chosen by the REAL default kube-scheduler (kwokctl cluster `carbon`,
/tmp/kwok.kubeconfig) -- software we did not write -- and then scored with the paper's
own footprint accounting on the SAME window / SAME signals as the counterfactual replay.

Testbed (one): the headroom edge fleet -- node-0-fr-iot (2c) / node-1-es-smartphone (4c) /
node-2-it-no-laptop (8c) / node-3-de-server (16c), realci2 signals -- i.e. the SAME window
as run_counterfactual_harm_replay.py, so B_ind slots directly beside tab:counterfactual.

Independent-baseline construction (isolates the stock scheduler's ONE decision -- the node):
  The stock kube-scheduler is purely spatial (no time concept). To compare like-for-like we
  hold the TEMPORAL schedule fixed to our packing baseline B (each pod runs in B's start slot)
  and let the real default-scheduler make ONLY the node choice. We replay the pods slot by
  slot on the live control plane: at each slot we expire pods whose active window has ended
  (freeing real node capacity), submit that slot's pods to `default-scheduler` pinned to the
  fleet's own nodes, and read back the node the stock scheduler bound each pod to. A pod the
  stock scheduler leaves Pending (spatial fragmentation) is B_ind-unplaced (SLO harm).
  B_ind therefore differs from packing B ONLY in the node dimension -- exactly, and only,
  what kube-scheduler decides.

Evaluation (our accounting, idle-once occupancy-ordered, realized signals; basin = AWARE 2.0
native watershed CFs, matching the paper's per-basin water headline; country reported too):
  * B_ind vs packing B      -> is our self-implemented B a strawman? (does the independent
                               scheduler do BETTER, so certifying against B is cheap?)
  * envelope vs B_ind        -> does the paper's certified envelope hold no-harm (dC<=0, dW<=0,
                               no dropped pods) against the baseline we did NOT implement?
  * envelope-from-B_ind      -> the envelope repair started FROM B_ind: relief achievable from
                               the independent baseline (certifies vs B_ind by construction).

NOT claimed: measured energy/water (KWOK has no kubelet). Claimed: the node placement is the
real stock scheduler's, realized on a real control plane, scored ex-post from public signals.

Usage:
    PYTHONPATH=pkg/carbon-aware/server-python python scripts/run_independent_baseline_kwok.py
    PYTHONPATH=pkg/carbon-aware/server-python python scripts/run_independent_baseline_kwok.py --resolution country
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SERVER = REPO / "pkg" / "carbon-aware" / "server-python"
D5 = REPO / "experiments" / "d5_kwok"
for p in (str(SERVER), str(REPO / "scripts"), str(D5)):
    if p not in sys.path:
        sys.path.insert(0, p)

import carbon_aware.no_harm_flexibility as nhf  # noqa: E402
from carbon_aware.no_harm_flexibility import (  # noqa: E402
    Placement,
    ScheduleResult,
    PilotConfig,
    build_action_signals,
    build_greedy_schedule,
    build_timeslots,
    classify_pod,
    find_ranked_candidates,
    load_flavours_for_pilot,
    load_pods,
    order_pods_for_pilot,
    repair_schedule_no_harm,
    summarize_against_reference,
    write_placements_csv,
    _init_resources,
    _rematerialize_under_realized,
)

TIMEALIGNED_REALCI = REPO / "pkg" / "carbon-aware" / "data" / "timealigned_realci2"
CONFIG_FILE = REPO / "pkg" / "carbon-aware" / "infra-workload-config.yaml"
OUT = REPO / "experiments" / "demo_counterfactual" / "independent_baseline"
KUBECONFIG = "/tmp/kwok.kubeconfig"
NS = "default"
DROUGHT_THRESHOLD = 20.0

_ORIG_WATER_PATH = nhf._water_data_path


def patch_basin():
    """Redirect the engine's hardcoded country-AWARE lookup to AWARE 2.0 native basin
    (watershed) CFs. No engine edit; the file already ships (mirrors the counterfactual)."""
    def _patched(repo_root, filename):
        if filename == "aware20_country_nonagri_factors.csv":
            filename = "aware20_basin_nonagri_factors.csv"
        return _ORIG_WATER_PATH(repo_root, filename)
    nhf._water_data_path = _patched


def unpatch():
    nhf._water_data_path = _ORIG_WATER_PATH


def headroom_config(out_dir: Path) -> PilotConfig:
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
        drought_cf_threshold=DROUGHT_THRESHOLD,
    )


# --------------------------------------------------------------- kube-scheduler pass
def kube_scheduler_node_choice(pack: ScheduleResult, fleet_nodes):
    """Replay packing's temporal schedule on the LIVE control plane, letting the REAL
    default kube-scheduler choose the node for each pod. Returns
    (assignment {pod_id: node_id}, target_slot {pod_id: slot}, unplaced_ids)."""
    import kwok_refresh_no_harm as kw
    from kubernetes import client, config as kcfg
    kcfg.load_kube_config(config_file=KUBECONFIG)
    v1 = client.CoreV1Api()
    fleet_nodes = sorted(fleet_nodes)
    kw.register_nodes(v1, set(fleet_nodes))   # no-op if the fleet nodes already exist (they do)
    kw.clear_pods(v1)

    # Each pod's temporal footprint = packing's chosen start slot + the pod's duration.
    target_slot = {}
    dur = {}
    by_slot = defaultdict(list)
    for pl in pack.placements:
        pid = pl.pod.id
        s = int(pl.candidate.timeslot.id)
        d = max(1, int(round(float(pl.pod.duration))))
        target_slot[pid] = s
        dur[pid] = d
        by_slot[s].append(pl.pod)

    live = {}          # pod_id -> end_slot (exclusive)
    assignment = {}
    unplaced = set()
    for s in sorted(by_slot):
        # Expire pods whose active window has ended -> free real node capacity.
        expired = [pid for pid, end in live.items() if end <= s]
        for pid in expired:
            try:
                v1.delete_namespaced_pod(pid.lower(), NS)
            except Exception:
                pass
            del live[pid]
        if expired:
            time.sleep(1.0)  # let the deletions settle before the scheduler scores this slot

        arrivals = by_slot[s]
        want = {p.id.lower() for p in arrivals}
        for p in arrivals:
            v1.create_namespaced_pod(
                NS, kw.pod(p.id, p.cpuRequest, p.ramRequest, "default-scheduler", pin_nodes=fleet_nodes))
        deadline = time.time() + 90
        bound = {}
        while time.time() < deadline:
            items = v1.list_namespaced_pod(NS, label_selector=f"{kw.POD_LABEL}=1").items
            bound = {it.metadata.name: it.spec.node_name
                     for it in items if it.spec.node_name and it.metadata.name in want}
            if len(bound) >= len(want):
                break
            time.sleep(0.5)
        for p in arrivals:
            node = bound.get(p.id.lower())
            if node:
                assignment[p.id] = node
                live[p.id] = s + dur[p.id]
            else:
                unplaced.add(p.id)
                try:
                    v1.delete_namespaced_pod(p.id.lower(), NS)  # drop the Pending pod
                except Exception:
                    pass
    kw.clear_pods(v1)
    return assignment, target_slot, unplaced


# --------------------------------------------------------------- accounting helpers
def build_schedule_from_assignment(method_key, assignment, target_slot, unplaced_ids,
                                   pods, flavours, cfg) -> ScheduleResult:
    """Materialize an externally chosen {pod -> (node, slot)} placement through OUR
    footprint accounting (idle-once occupancy-ordered, realized signals), so it is
    directly comparable to packing/envelope."""
    timeslots = build_timeslots(cfg.max_timeslots)
    full_cpu, full_ram, full_gpu = _init_resources(flavours, cfg.max_timeslots)
    placements, unplaced = [], []
    for pod in order_pods_for_pilot(pods):
        if pod.id in unplaced_ids or pod.id not in assignment:
            unplaced.append(pod)
            continue
        node = assignment[pod.id]
        slot = target_slot[pod.id]
        ranked = find_ranked_candidates(
            pod=pod, flavours=list(flavours), timeslots=timeslots,
            leftover_cpu=full_cpu, leftover_ram=full_ram, max_time_slots=cfg.max_timeslots,
            objective_mode="carbon", water_metric="scarcity", leftover_gpu=full_gpu)
        match = next((c for c in ranked if c.flavour.id == node and c.timeslot.id == slot), None)
        if match is None:
            unplaced.append(pod)
            continue
        placements.append(Placement(pod=pod, candidate=match,
                                    flexibility_class=classify_pod(pod, cfg.flexibility_slack_hours)))
    res = ScheduleResult(method_key=method_key, placements=placements,
                         unplaced_pods=unplaced, elapsed_seconds=0.0)
    _rematerialize_under_realized(res, flavours, cfg)  # idle-once, occupancy-ordered, realized
    return res


def per_basin_scarcity(result):
    out = {}
    for pl in result.placements:
        region = (getattr(pl.candidate.flavour, "region", "") or "").upper()
        out[region] = out.get(region, 0.0) + pl.candidate.footprint.scarcity_characterized_water
    return out


def capacity_check(result: ScheduleResult, flavours) -> int:
    """Assert no (node, slot) instantaneous CPU load exceeds real allocatable. Returns
    the number of violations (should be 0 -> the materialization matched every pod)."""
    cap = {f.id: float(f.totalCpu) for f in flavours}
    load = defaultdict(float)
    for pl in result.placements:
        nid = pl.candidate.flavour.id
        s = pl.candidate.timeslot.id
        for off in range(max(1, int(round(float(pl.pod.duration))))):
            load[(nid, s + off)] += float(pl.pod.cpuRequest)
    return sum(1 for (nid, _), used in load.items() if used > cap.get(nid, 0.0) + 1e-6)


def row(summary, keys):
    return {k: (round(summary[k], 4) if isinstance(summary.get(k), float) else summary.get(k)) for k in keys}


# --------------------------------------------------------------- main
def run_one_resolution(resolution: str, kube_assignment) -> dict:
    """Evaluate the ONE stock-scheduler placement (kube_assignment, computed once) with our
    footprint accounting at the given water resolution. The physical B_ind placement (node +
    slot per pod) is resolution-independent; only the CF weighting differs."""
    logging.disable(logging.CRITICAL)
    assignment, target_slot, unplaced_ids = kube_assignment
    cfg = headroom_config(OUT / f"field_{resolution}")
    flavours = load_flavours_for_pilot(cfg)
    pods = load_pods(cfg.workloads_dir, max_pods=cfg.max_pods)
    signals = build_action_signals(flavours, cfg)
    fleet_nodes = sorted({f.id for f in flavours})

    # --- our baselines: packing B + envelope (repair of B) ---
    packing, _, _ = build_greedy_schedule(method_key="packing", pods=pods, flavours=flavours, config=cfg)
    carbon, _, _ = build_greedy_schedule(method_key="carbon", pods=pods, flavours=flavours, config=cfg)
    envelope = repair_schedule_no_harm(
        method_key="no_harm_flex", baseline=packing, pods=pods, flavours=flavours,
        signals=signals, config=cfg, score_mode="combined")
    for r in (packing, carbon, envelope):
        _rematerialize_under_realized(r, flavours, cfg)

    # --- independent baseline: the fixed stock kube-scheduler placement, scored at this resolution ---
    b_ind = build_schedule_from_assignment(
        "kube_scheduler_B_ind", assignment, target_slot, unplaced_ids, pods, flavours, cfg)
    envelope_from_bind = repair_schedule_no_harm(
        method_key="no_harm_flex_from_Bind", baseline=b_ind, pods=pods, flavours=flavours,
        signals=signals, config=cfg, score_mode="combined")
    _rematerialize_under_realized(envelope_from_bind, flavours, cfg)

    cap_viol = capacity_check(b_ind, flavours)

    # how many pods did the stock scheduler place on a DIFFERENT node than packing?
    pack_node = {pl.pod.id: pl.candidate.flavour.id for pl in packing.placements}
    node_diff = sum(1 for pid, nid in assignment.items() if pack_node.get(pid) != nid)

    # --- summaries (all with OUR accounting on realized signals) ---
    bind_vs_pack = summarize_against_reference(b_ind, packing, signals)
    env_vs_bind = summarize_against_reference(envelope, b_ind, signals)
    env_vs_pack = summarize_against_reference(envelope, packing, signals)
    carbon_vs_pack = summarize_against_reference(carbon, packing, signals)
    envind_vs_bind = summarize_against_reference(envelope_from_bind, b_ind, signals)

    # per-basin water, B_ind vs packing (does the stock scheduler harm a watershed?)
    sc_pack = per_basin_scarcity(packing)
    sc_bind = per_basin_scarcity(b_ind)
    per_basin = [{"basin": reg, "B_scarcity_L": round(sc_pack.get(reg, 0.0), 3),
                  "Bind_scarcity_L": round(sc_bind.get(reg, 0.0), 3),
                  "Bind_delta_L": round(sc_bind.get(reg, 0.0) - sc_pack.get(reg, 0.0), 3)}
                 for reg in sorted(set(sc_pack) | set(sc_bind))]

    keys = ["no_harm_certificate", "carbon_delta_pct", "scarcity_delta_pct",
            "carbon_delta_kg", "scarcity_delta", "placed_pods", "unplaced_pods",
            "moved_common_pods", "weighted_stress_kwh_avoided"]

    write_placements_csv(OUT / f"placements_{resolution}_Bind_kube_scheduler.csv", b_ind, signals)
    write_placements_csv(OUT / f"placements_{resolution}_packing.csv", packing, signals)
    write_placements_csv(OUT / f"placements_{resolution}_envelope.csv", envelope, signals)

    return {
        "resolution": resolution,
        "fleet_nodes": fleet_nodes,
        "n_pods": len(pods),
        "B_ind_source": "real default kube-scheduler (kwokctl `carbon`), node choice only; temporal schedule = packing B",
        "B_ind_nodes_differ_from_packing": node_diff,
        "B_ind_placed": len(b_ind.placements),
        "B_ind_unplaced": len(b_ind.unplaced_pods),
        "B_ind_capacity_violations_in_our_accounting": cap_viol,
        "totals": {
            "packing_carbon_kg": round(nhf.schedule_totals(packing, signals)["carbon_kg"], 4),
            "packing_scarcity_L": round(nhf.schedule_totals(packing, signals)["scarcity_water"], 4),
            "B_ind_carbon_kg": round(nhf.schedule_totals(b_ind, signals)["carbon_kg"], 4),
            "B_ind_scarcity_L": round(nhf.schedule_totals(b_ind, signals)["scarcity_water"], 4),
            "envelope_carbon_kg": round(nhf.schedule_totals(envelope, signals)["carbon_kg"], 4),
            "envelope_scarcity_L": round(nhf.schedule_totals(envelope, signals)["scarcity_water"], 4),
        },
        "B_ind_vs_packing": row(bind_vs_pack, keys),
        "carbon_greedy_vs_packing": row(carbon_vs_pack, keys),
        "envelope_vs_packing": row(env_vs_pack, keys),
        "envelope_vs_B_ind": row(env_vs_bind, keys),
        "envelope_from_B_ind_vs_B_ind": row(envind_vs_bind, keys),
        "per_basin_water_Bind_vs_packing": per_basin,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", choices=["basin", "country", "both"], default="both")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    digest = {"testbed": "headroom edge fleet (FR-IoT/ES-smartphone/IT-NO-laptop/DE-server), realci2 signals "
                         "(same window as run_counterfactual_harm_replay.py)",
              "control_plane": "kwokctl cluster `carbon` (/tmp/kwok.kubeconfig): real apiserver/etcd/default-scheduler; KWOK fake kubelet",
              "claim": "the No-Harm Envelope certifies against B_ind, a placement produced by the stock "
                       "kube-scheduler -- software the authors did not implement",
              "not_claimed": "measured energy/water (no kubelet); footprints are our accounting on realized public signals"}

    # Run the stock kube-scheduler ONCE (its placement is resolution-independent) so basin and
    # country score the SAME physical B_ind, isolating scheduler nondeterminism from resolution.
    logging.disable(logging.CRITICAL)
    unpatch()
    cfg0 = headroom_config(OUT / "field_sched")
    flavours0 = load_flavours_for_pilot(cfg0)
    pods0 = load_pods(cfg0.workloads_dir, max_pods=cfg0.max_pods)
    packing0, _, _ = build_greedy_schedule(method_key="packing", pods=pods0, flavours=flavours0, config=cfg0)
    fleet_nodes0 = sorted({f.id for f in flavours0})
    logging.disable(logging.NOTSET)
    kube_assignment = kube_scheduler_node_choice(packing0, fleet_nodes0)
    logging.disable(logging.CRITICAL)
    digest["B_ind_placement"] = {
        "assigned": len(kube_assignment[0]), "unplaced": sorted(kube_assignment[2]),
        "note": "one real default-scheduler run on the live control plane; scored at both resolutions below",
    }

    if args.resolution in ("basin", "both"):
        patch_basin()
        digest["basin"] = run_one_resolution("basin", kube_assignment)
        unpatch()
    if args.resolution in ("country", "both"):
        unpatch()
        digest["country"] = run_one_resolution("country", kube_assignment)

    (OUT / "independent_baseline_digest.json").write_text(json.dumps(digest, indent=2))
    print(json.dumps(digest, indent=2))
    print(f"\n-> {OUT / 'independent_baseline_digest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
