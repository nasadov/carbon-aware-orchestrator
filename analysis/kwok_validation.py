#!/usr/bin/env python3
"""KWOK control-plane validation for TotEm (temporally correct).

Replays a frozen simulator snapshot (200-pod workload, 4-region testbed) on a REAL
Kubernetes control plane (kwokctl cluster `carbon`: real kube-apiserver, etcd, default
kube-scheduler; KWOK fakes the kubelet). Kubernetes scheduling is instantaneous, while
TotEm is spatio-temporal, so the validation is done PER TIMESLOT.

  T1 Integration + latency: every pod the simulator places is admissible and bindable
     through the real API onto its assigned node, reaching Running, no rejection; report
     real control-plane bind latency.
  T2 Per-timeslot feasibility: at every timeslot, the sum of CPU of pods ACTIVE on a node
     (start_slot <= t < start_slot+duration) never exceeds that node's REAL allocatable
     capacity -> the simulator's schedule is a valid time-series on the real nodes.
  T3 Real-scheduler feasibility (fair, instantaneous): at the busiest timeslot, submit the
     active pods to the genuine default kube-scheduler on the real nodes and confirm it
     schedules them -> the per-timeslot placements are realizable by a real scheduler.

No workloads execute (KWOK has no kubelet); emissions stay post-hoc. The claim is
realizability + integration + per-slot feasibility, not measured energy.
"""

from __future__ import annotations

import csv
import glob
import time
from collections import defaultdict
from pathlib import Path

from kubernetes import client, config

REPO = Path(__file__).resolve().parents[1]
SNAP = REPO / "experiments/resubmission_baseline_matrix_sweep_v5/matrix_20260515_111056/seed_42_pods_200"
KUBECONFIG = "/tmp/kwok.kubeconfig"
NS = "default"


def load_placements(glob_pat: str, csv_name: str) -> dict[str, dict]:
    hits = glob.glob(str(SNAP / glob_pat))
    csv_path = next(iter(glob.glob(str(Path(hits[0]) / csv_name))), None)
    out = {}
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            out[r["pod_id"]] = dict(node=r["node_id"], cpu=float(r["cpu_request"]),
                                    ram=float(r["ram_request"]), start=int(float(r["start_slot"])),
                                    dur=max(1, int(float(r["duration"]))))
    return out


def node_allocatable(v1) -> dict[str, float]:
    cap = {}
    for n in v1.list_node().items:
        c = n.status.allocatable["cpu"]
        cap[n.metadata.name] = float(c[:-1]) / 1000 if c.endswith("m") else float(c)
    return cap


def pod(name, cpu, ram, sched):
    return client.V1Pod(
        metadata=client.V1ObjectMeta(name=name.lower(), labels={"val": "kwok"}),
        spec=client.V1PodSpec(scheduler_name=sched, containers=[client.V1Container(
            name="c", image="fake", resources=client.V1ResourceRequirements(
                requests={"cpu": f"{int(round(cpu*1000))}m", "memory": f"{int(round(ram))}Mi"}))]))


def clear_pods(v1):
    v1.delete_collection_namespaced_pod(NS, label_selector="val=kwok")
    for _ in range(40):
        if not v1.list_namespaced_pod(NS, label_selector="val=kwok").items:
            return
        time.sleep(0.5)


def t2_feasibility(placements, cap):
    """Per-timeslot per-node CPU vs real allocatable. Returns (violations, peak_util_frac)."""
    horizon = max(p["start"] + p["dur"] for p in placements.values())
    peak = {n: 0.0 for n in cap}
    violations = 0
    for t in range(horizon):
        load = defaultdict(float)
        for p in placements.values():
            if p["start"] <= t < p["start"] + p["dur"]:
                load[p["node"]] += p["cpu"]
        for n, used in load.items():
            peak[n] = max(peak[n], used)
            if used > cap[n] + 1e-6:
                violations += 1
    return violations, {n: round(peak[n] / cap[n], 2) for n in cap}


def t1_integration(v1, totem):
    clear_pods(v1)
    lat, rejected = [], 0
    for pid, p in totem.items():
        t0 = time.perf_counter()
        try:
            v1.create_namespaced_pod(NS, pod(pid, p["cpu"], p["ram"], "totem"))
            v1.create_namespaced_binding(NS, client.V1Binding(
                metadata=client.V1ObjectMeta(name=pid.lower()),
                target=client.V1ObjectReference(api_version="v1", kind="Node", name=p["node"])),
                _preload_content=False)
        except Exception:
            rejected += 1
            continue
        lat.append((time.perf_counter() - t0) * 1000)
    time.sleep(2)
    running = sum(1 for it in v1.list_namespaced_pod(NS, label_selector="val=kwok").items
                  if it.status.phase == "Running")
    lat.sort()
    return dict(submitted=len(totem), running=running, rejected=rejected,
                bind_p50_ms=round(lat[len(lat)//2], 1), bind_p95_ms=round(lat[int(len(lat)*0.95)], 1))


def t3_real_scheduler_busiest(v1, placements, cap):
    horizon = max(p["start"] + p["dur"] for p in placements.values())
    best_t, best_load = 0, -1.0
    for t in range(horizon):
        load = sum(p["cpu"] for p in placements.values() if p["start"] <= t < p["start"] + p["dur"])
        if load > best_load:
            best_load, best_t = load, t
    active = {pid: p for pid, p in placements.items() if p["start"] <= best_t < p["start"] + p["dur"]}
    clear_pods(v1)
    for pid, p in active.items():
        v1.create_namespaced_pod(NS, pod(pid, p["cpu"], p["ram"], "default-scheduler"))
    deadline = time.time() + 45
    scheduled = {}
    while time.time() < deadline:
        items = v1.list_namespaced_pod(NS, label_selector="val=kwok").items
        scheduled = {it.metadata.name: it.spec.node_name for it in items if it.spec.node_name}
        if len(scheduled) >= len(active):
            break
        time.sleep(1)
    by_region = defaultdict(int)
    for nd in scheduled.values():
        by_region[nd.split("-")[2]] += 1
    return dict(busiest_slot=best_t, active_pods=len(active), real_scheduler_placed=len(scheduled),
                total_active_cpu=round(best_load, 1), cluster_cpu=round(sum(cap.values()), 1),
                by_region=dict(by_region))


def main():
    config.load_kube_config(config_file=KUBECONFIG)
    v1 = client.CoreV1Api()
    cap = node_allocatable(v1)
    totem = load_placements("heuristic_proportional_*", "heuristic_prop_placements_session.csv")
    vanilla = load_placements("vanilla_most_allocated_*", "vanilla_placement_session.csv")
    print(f"nodes: {cap}")
    print(f"snapshot: TotEm placed {len(totem)}, Vanilla placed {len(vanilla)}")

    print("\n== T2 per-timeslot feasibility vs REAL node allocatable ==")
    for name, pl in (("TotEm", totem), ("Vanilla", vanilla)):
        viol, peak = t2_feasibility(pl, cap)
        print(f"  {name:8} capacity_violations={viol}  peak_node_util_frac={peak}")

    print("\n== T1 integration + latency (TotEm placements via real API) ==")
    for k, v in t1_integration(v1, totem).items():
        print(f"  {k}: {v}")

    print("\n== T3 real default kube-scheduler at busiest timeslot ==")
    for k, v in t3_real_scheduler_busiest(v1, totem, cap).items():
        print(f"  {k}: {v}")
    clear_pods(v1)


if __name__ == "__main__":
    main()
