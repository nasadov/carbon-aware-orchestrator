#!/usr/bin/env python3
import argparse
import os
import sys
import yaml
import json
import subprocess
import re

HEADER = [
    "pod_id",
    "node_id",
    "start_slot",
    "duration",
    "cpu_request",
    "ram_request",
]


def parse_cpu_to_cores(cpu_str: str) -> float:
    if not cpu_str:
        return 0.0
    s = str(cpu_str).strip().lower()
    try:
        if s.endswith("m"):
            return float(s[:-1]) / 1000.0
        return float(s)
    except Exception:
        return 0.0


def parse_mem_to_mb(mem_str: str) -> float:
    if not mem_str:
        return 0.0
    s = str(mem_str).strip()
    try:
        if s.endswith("Ki"):
            return float(s[:-2]) / 1024.0
        if s.endswith("Mi"):
            return float(s[:-2])
        if s.endswith("Gi"):
            return float(s[:-2]) * 1024.0
        if s.endswith("Ti"):
            return float(s[:-2]) * 1024.0 * 1024.0
        # Fallback: assume MB
        return float(re.sub(r"[^0-9.]+", "", s))
    except Exception:
        return 0.0


def extract_duration_hours(labels: dict) -> float:
    duration_label = labels.get("duration") or labels.get("duration_hours") or ""
    m = re.search(r"duration-(\d+)h", str(duration_label))
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return 0.0
    # Fallback: try numbers in label
    m2 = re.search(r"(\d+(?:\.\d+)?)h", str(duration_label))
    if m2:
        try:
            return float(m2.group(1))
        except Exception:
            return 0.0
    return 0.0


def kubectl_get_pod_node_map(namespace: str) -> dict:
    """Fetch all pods once and build a map from ms_name (label 'name') to nodeName."""
    mapping = {}
    try:
        cmd = [
            "kubectl",
            "get",
            "pods",
            "-n",
            namespace,
            "-o",
            "json",
        ]
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        data = json.loads(proc.stdout)
        for item in data.get("items", []):
            meta = item.get("metadata", {})
            labels = meta.get("labels", {})
            ms_name = labels.get("name", "")
            if not ms_name:
                continue
            spec = item.get("spec", {})
            node_name = spec.get("nodeName", "") or ""
            if node_name:
                mapping[ms_name] = node_name
    except Exception:
        pass
    return mapping


def build_rows(workloads_dir: str, namespace: str, node_map_path: str = "") -> list:
    # Find timeslot files
    files = [
        os.path.join(workloads_dir, f)
        for f in os.listdir(workloads_dir)
        if f.startswith("timeslot_") and f.endswith(".yaml")
    ]
    def slot_index(p):
        m = re.search(r"timeslot_(\d+)\.yaml", os.path.basename(p))
        return int(m.group(1)) if m else 0
    files.sort(key=slot_index)

    rows = []

    # Load pod->node mapping from file if provided; otherwise snapshot once
    pod_node_map = {}
    if node_map_path and os.path.exists(node_map_path):
        try:
            with open(node_map_path, "r") as fh:
                pod_node_map = json.load(fh)
        except Exception:
            pod_node_map = {}
    if not pod_node_map:
        pod_node_map = kubectl_get_pod_node_map(namespace)

    for fpath in files:
        start_slot = slot_index(fpath)
        try:
            with open(fpath, "r") as fh:
                content = fh.read()
        except Exception:
            continue
        # Parse multi-doc yaml
        try:
            docs = list(yaml.safe_load_all(content))
        except Exception:
            continue

        for doc in docs:
            if not isinstance(doc, dict):
                continue
            if doc.get("kind") != "Deployment":
                continue

            metadata = doc.get("metadata", {})
            full_name = metadata.get("name") or metadata.get("labels", {}).get("app") or ""
            labels = metadata.get("labels", {})
            duration_h = extract_duration_hours(labels)

            # Microservice base name (mNNN) is used in pod labels
            ms_name = doc.get("spec", {}).get("selector", {}).get("matchLabels", {}).get("name", "")

            # Resource requests
            tpl = doc.get("spec", {}).get("template", {})
            conts = tpl.get("spec", {}).get("containers", [])
            cpu_req = 0.0
            mem_req = 0.0
            if conts:
                reqs = conts[0].get("resources", {}).get("requests", {})
                cpu_req = parse_cpu_to_cores(reqs.get("cpu", ""))
                mem_req = parse_mem_to_mb(reqs.get("memory", ""))

            # Node assignment from pre-fetched map
            node_id = pod_node_map.get(ms_name, "") if ms_name else ""

            rows.append([
                full_name,
                node_id,
                float(start_slot),
                float(duration_h),
                float(cpu_req),
                float(mem_req),
            ])
    return rows


def main():
    ap = argparse.ArgumentParser(description="Generate vanilla_placement_session.csv from workloads and kubectl assignments")
    ap.add_argument("--workloads-dir", required=True, help="Path to workloads-vanilla directory")
    ap.add_argument("--namespace", default="default", help="Kubernetes namespace")
    ap.add_argument("--experiment-dir", required=True, help="Experiment directory to write outputs to")
    ap.add_argument("--node-map", default="", help="Optional JSON file with {ms_name: nodeName} captured during run")
    args = ap.parse_args()

    os.makedirs(args.experiment_dir, exist_ok=True)
    rows = build_rows(args.workloads_dir, args.namespace, args.node_map)

    out_csv = os.path.join(args.experiment_dir, "vanilla_placement_session.csv")
    with open(out_csv, "w") as fh:
        fh.write(",".join(HEADER) + "\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")

    print(f"Wrote {len(rows)} rows to {out_csv}")

if __name__ == "__main__":
    main()
