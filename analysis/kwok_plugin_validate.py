#!/usr/bin/env python3
"""Compare FogAtlas-bound placements (driven by kubectl-carbon + gRPC service) to the
simulator. Run AFTER the scheduling gates are removed. Maps each live pod back to its
Deployment name (= simulator pod_id) and checks node agreement with the heuristic sim CSV.
"""
import csv, glob, time
from collections import Counter
from pathlib import Path
from kubernetes import client, config

REPO = Path(__file__).resolve().parents[1]
SNAP = REPO / "experiments/resubmission_baseline_matrix_sweep_v5/matrix_20260515_111056/seed_42_pods_200"
config.load_kube_config(config_file="/tmp/kwok.kubeconfig")
v1 = client.CoreV1Api()

sim = {}
hcsv = glob.glob(str(SNAP / "heuristic_proportional_*" / "heuristic_prop_placements_session.csv"))[0]
for r in csv.DictReader(open(hcsv)):
    sim[r["pod_id"]] = r["node_id"]

def depl_name(pod_name: str) -> str:
    return "-".join(pod_name.split("-")[:-2])  # strip replicaset + pod hash

prev, stable = -1, 0
for _ in range(45):
    items = v1.list_namespaced_pod("default").items
    bound = {p.metadata.name: p.spec.node_name for p in items if p.spec.node_name}
    gated = sum(1 for p in items if (p.spec.scheduling_gates or []))
    if len(bound) == prev and len(bound) > 0:
        stable += 1
    else:
        stable = 0
    prev = len(bound)
    print(f"bound={len(bound)} gated={gated}", flush=True)
    if stable >= 3:
        break
    time.sleep(4)

agree = total = 0
byreg = Counter()
for pname, node in bound.items():
    d = depl_name(pname)
    byreg[node.split("-")[2]] += 1
    if d in sim:
        total += 1
        if sim[d] == node:
            agree += 1
print("\n=== kubectl-carbon -> gRPC -> FogAtlas: real-cluster bindings ===")
print(f"bound pods: {len(bound)}")
print(f"by region: {dict(byreg)}")
print(f"node agreement with simulator (matched pods): {agree}/{total} = {round(100*agree/total,1) if total else 0}%")
