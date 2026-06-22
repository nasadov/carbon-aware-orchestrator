#!/usr/bin/env python3
"""Convert the Alibaba Cluster Trace GPU v2020 (pai_task_table) into the repository's
Kubernetes-style timeslot workload format, with GPU requests.

Schema (no header): job_name, task_name, inst_num, status, start_time, end_time,
plan_cpu (%, 100=1 vCPU), plan_mem (GB), plan_gpu (%, 100=1 GPU; fractional), gpu_type.

Firm/flexible tier from task role (evictability/latency): interactive roles (JupyterTask,
TensorboardTask) are FIRM (slack 0); batch training/compute roles (tensorflow, worker, ps,
PyTorchWorker, evaluator, ...) are FLEXIBLE (deferrable; slack in {4,12,24,48} h). GPU requests
are fractional and written as the `trace/gpu_request` annotation; deadlines are synthetic.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

COLS = ["job_name", "task_name", "inst_num", "status", "start_time", "end_time",
        "plan_cpu", "plan_mem", "plan_gpu", "gpu_type"]
FIRM_ROLES = {"JupyterTask", "TensorboardTask"}  # interactive/latency-sensitive -> firm
DURATION_BUCKETS = [1, 3, 6]
SLACK_OPTIONS = [4, 12, 24, 48]

DEPLOYMENT_TEMPLATE = {
    "apiVersion": "apps/v1", "kind": "Deployment",
    "metadata": {"name": "", "namespace": "default", "labels": {}, "annotations": {}},
    "spec": {"replicas": 1, "selector": {"matchLabels": {}},
             "template": {"metadata": {"labels": {}},
                          "spec": {"schedulerName": "fogatlas",
                                   "containers": [{"name": "fake-container", "image": "fake-image",
                                                   "resources": {"requests": {}}}]}}},
}


def quantize_duration(hours: float) -> int:
    if hours < 2.0:
        return 1
    if hours < 4.5:
        return 3
    return 6


def fmt(h: int) -> str:
    return f"{int(h)}h"


def load_window(path: Path, window_start_s: float, arrival_window_slots: int) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLS)
    for c in ("start_time", "end_time", "plan_cpu", "plan_mem", "plan_gpu"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["plan_gpu"] = df["plan_gpu"].fillna(0.0)
    df = df.dropna(subset=["start_time", "end_time", "plan_cpu", "plan_mem"])
    df = df[(df["end_time"] > df["start_time"])]
    df["arrival_slot"] = ((df["start_time"] - window_start_s) / 3600.0).apply(math.floor)
    df = df[(df["arrival_slot"] >= 0) & (df["arrival_slot"] < arrival_window_slots)]
    return df


def stratified_sample(df: pd.DataFrame, target: int, seed: int, slots: int) -> pd.DataFrame:
    rng = random.Random(seed)
    target = min(target, len(df))
    counts = df["arrival_slot"].value_counts().sort_index()
    alloc = (counts / counts.sum() * target).apply(math.floor).astype(int)
    rem = target - int(alloc.sum())
    if rem > 0:
        frac = (counts / counts.sum() * target - alloc).sort_values(ascending=False)
        for s in frac.index[:rem]:
            alloc.loc[s] += 1
    parts = []
    for slot, n in alloc.items():
        if n <= 0:
            continue
        g = df[df["arrival_slot"] == slot]
        parts.append(g if len(g) <= n else g.sample(n=n, random_state=rng.randint(0, 2**31 - 1)))
    out = pd.concat(parts, ignore_index=True) if parts else df.head(0)
    return out.sort_values(["arrival_slot", "start_time"]).reset_index(drop=True)


def assign_fields(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    out = df.copy()
    out["duration_slots"] = ((out["end_time"] - out["start_time"]) / 3600.0).clip(lower=1.0).apply(quantize_duration)
    out["tier"] = out["task_name"].apply(lambda r: "firm" if str(r) in FIRM_ROLES else "flexible")
    rng = random.Random(seed + 7)
    flex_slacks = [SLACK_OPTIONS[i % len(SLACK_OPTIONS)] for i in range(len(out))]
    rng.shuffle(flex_slacks)
    out["slack"] = [flex_slacks[i] if out["tier"].iloc[i] == "flexible" else 0 for i in range(len(out))]
    out["deadline_slots"] = out["duration_slots"].astype(int) + pd.Series(out["slack"].values, index=out.index).astype(int)
    out["cpu_cores"] = (out["plan_cpu"] / 100.0).clip(lower=0.05)
    out["mem_gb"] = out["plan_mem"].clip(lower=1.0).round().astype(int)
    out["gpu_req"] = (out["plan_gpu"] / 100.0).clip(lower=0.0)
    out["pod_id"] = [f"m{i:04d}" for i in range(len(out))]
    return out


def make_deployment(row: pd.Series) -> dict:
    import copy
    dep = copy.deepcopy(DEPLOYMENT_TEMPLATE)
    pid = str(row["pod_id"])
    dur, dl = int(row["duration_slots"]), int(row["deadline_slots"])
    name = f"{pid}-duration-{fmt(dur)}-deadline-{fmt(dl)}"
    dep["metadata"]["name"] = name
    dep["metadata"]["labels"] = {"app": name, "duration": f"duration-{fmt(dur)}", "deadline": f"deadline-{fmt(dl)}"}
    dep["metadata"]["annotations"] = {
        "trace/source": "alibaba_gpu_2020",
        "trace/task_name": str(row["task_name"]),
        "trace/tier": str(row["tier"]),
        "trace/gpu_request": f"{float(row['gpu_req']):.4f}",
        "trace/gpu_type": str(row["gpu_type"]),
        "trace/deadline_slack_hours": str(int(row["slack"])),
        "trace/deadline_note": "synthetic_duration_plus_slack; firm=role-interactive, flexible=role-batch-training",
    }
    dep["spec"]["selector"]["matchLabels"]["name"] = pid
    dep["spec"]["template"]["metadata"]["labels"]["name"] = pid
    req = dep["spec"]["template"]["spec"]["containers"][0]["resources"]["requests"]
    req["cpu"] = f"{int(round(row['cpu_cores'] * 1000))}m"
    req["memory"] = f"{int(row['mem_gb'])}Gi"
    return dep


def write_workloads(df: pd.DataFrame, root: Path, slots: int) -> None:
    wl = root / "workloads"
    if wl.exists():
        shutil.rmtree(wl)
    wl.mkdir(parents=True, exist_ok=True)
    for slot in range(slots):
        grp = df[df["arrival_slot"] == slot]
        with (wl / f"timeslot_{slot}.yaml").open("w") as fh:
            for _, row in grp.iterrows():
                yaml.safe_dump(make_deployment(row), fh, sort_keys=False)
                fh.write("---\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="external/real-traces/raw/alibaba-gpu-2020/pai_task_table.csv")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--window-start-hours", type=float, default=240.0, help="window start (trace hours; default day 10)")
    ap.add_argument("--arrival-window-slots", type=int, default=12)
    ap.add_argument("--target-pods", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    csv = Path(args.csv).resolve()
    out = Path(args.output_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    window_start_s = args.window_start_hours * 3600.0

    df = load_window(csv, window_start_s, args.arrival_window_slots)
    if df.empty:
        raise SystemExit(f"no tasks in window starting {args.window_start_hours}h; pick another --window-start-hours")
    sampled = stratified_sample(df, args.target_pods, args.seed, args.arrival_window_slots)
    canonical = assign_fields(sampled, args.seed)
    canonical.to_csv(out / "canonical_trace_workload.csv", index=False)
    write_workloads(canonical, out, args.arrival_window_slots)

    tier_counts = {k: int(v) for k, v in canonical["tier"].value_counts().to_dict().items()}
    manifest = {
        "source": "alibaba_cluster_trace_gpu_v2020",
        "source_url": "https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020",
        "table": "pai_task_table",
        "window_start_hours": args.window_start_hours,
        "arrival_window_slots": args.arrival_window_slots,
        "target_pods": args.target_pods,
        "selected_pods": int(len(canonical)),
        "seed": args.seed,
        "resource_mapping": "plan_cpu%/100=cores (millicores), plan_mem=GiB, plan_gpu%/100=fractional GPU (trace/gpu_request)",
        "duration_mapping": "(end-start) quantized to {1,3,6} h",
        "tier_mapping": f"firm = task role in {sorted(FIRM_ROLES)} (interactive); flexible = batch/training roles, slack in {SLACK_OPTIONS} h",
        "tier_counts": tier_counts,
        "gpu_request_stats": {
            "frac_with_gpu": round(float((canonical["gpu_req"] > 0).mean()), 3),
            "mean_gpu": round(float(canonical["gpu_req"].mean()), 3),
            "max_gpu": round(float(canonical["gpu_req"].max()), 3),
        },
        "license": "CC-BY-4.0; see alibaba/clusterdata",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote Alibaba GPU workload: {out}")
    print(f"  tiers={tier_counts}  gpu_stats={manifest['gpu_request_stats']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
