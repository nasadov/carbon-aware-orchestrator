#!/usr/bin/env python3
"""Convert Alibaba GPU v2020 -> repo timeslot workloads, with MEASURED utilization (closes R2).

Joins pai_sensor_table (measured cpu_usage, gpu_wrk_util) to pai_task_table (plan_* requests), so
each pod carries its utilization-as-fraction-of-request -> the engine drives DYNAMIC power from
actual usage (idle/allocation stay request-based). Tier from task role (interactive Jupyter/
Tensorboard = firm; training = flexible). Durations seconds->hours, ROUNDED (min 1); only >=1h jobs
become schedulable pods (sub-hour tasks are ~6.5% of GPU-hours -> documented limitation / appendix
background). Period-accurate hardware = V100 (the trace's actual cards). Annotations consumed by the
loader: trace/gpu_request, trace/tier, trace/cpu_util_ratio, trace/gpu_util_ratio.
"""
from __future__ import annotations
import argparse, copy, math, random, shutil
from pathlib import Path
import pandas as pd
import yaml

TASK_COLS = ["job_name","task_name","inst_num","status","start_time","end_time",
             "plan_cpu","plan_mem","plan_gpu","gpu_type"]
SENSOR_COLS = ["job_name","task_name","worker_name","inst_id","machine","gpu_name","cpu_usage",
               "gpu_wrk_util","avg_mem","max_mem","avg_gpu_wrk_mem","max_gpu_wrk_mem",
               "read","write","read_count","write_count"]
FIRM_ROLES = {"JupyterTask", "TensorboardTask"}   # interactive/latency-sensitive -> firm
# Deferral slack (hours) for flexible jobs, grounded in the carbon-aware/grid-flexibility literature:
# grid benefit saturates after only a few hours (Chen & Zheng 2026), the production envelope is
# within-day ~24h (Google Carbon-Intelligent Computing, Radovanovic 2023), and DR events are a few
# hours (EPRI DCFlex). So we draw a distribution SKEWED to a few hours, CAPPED at 24h (no 48h tail).
SLACK_OPTIONS = [3, 6, 12, 24]
SLACK_WEIGHTS = [0.35, 0.30, 0.20, 0.15]   # mean ~9h, max 24h
# Trace-wide MEAN measured utilization-as-fraction-of-request (from the full sensor x task join);
# used to impute tasks lacking sensor data. Mean (not median, which is ~0 due to many idle ps-type
# tasks) so aggregate energy is unbiased. Sensor coverage is ~76-82% in the logged region (h>=672).
CPU_GLOBAL_UTIL = 0.40
GPU_GLOBAL_UTIL = 0.145
DEPLOYMENT_TEMPLATE = {
    "apiVersion": "apps/v1", "kind": "Deployment",
    "metadata": {"name": "", "namespace": "default", "labels": {}, "annotations": {}},
    "spec": {"replicas": 1, "selector": {"matchLabels": {}},
             "template": {"metadata": {"labels": {}},
                          "spec": {"schedulerName": "fogatlas",
                                   "containers": [{"name": "fake-container", "image": "fake-image",
                                                   "resources": {"requests": {}}}]}}},
}


def load_sensor_util(sensor_csv: Path) -> pd.DataFrame:
    """Per-(job,task) mean measured usage from the sensor table (1GB; read only needed cols)."""
    s = pd.read_csv(sensor_csv, header=None, names=SENSOR_COLS,
                    usecols=["job_name","task_name","cpu_usage","gpu_wrk_util"])
    for c in ("cpu_usage", "gpu_wrk_util"):
        s[c] = pd.to_numeric(s[c], errors="coerce")
    return s.groupby(["job_name","task_name"]).agg(
        gpu_wrk_util=("gpu_wrk_util","mean"), cpu_usage=("cpu_usage","mean")).reset_index()


def load_window(task_csv: Path, window_start_s: float, arrival_slots: int, horizon_h: int) -> pd.DataFrame:
    df = pd.read_csv(task_csv, header=None, names=TASK_COLS)
    for c in ("start_time","end_time","plan_cpu","plan_mem","plan_gpu"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["start_time","plan_cpu","plan_mem"])
    # censored (no end_time) -> clip to window end (treat as running through the window)
    window_end_s = window_start_s + horizon_h * 3600.0
    df["end_eff"] = df["end_time"].fillna(window_end_s).clip(upper=window_end_s)
    df["arrival_slot"] = ((df["start_time"] - window_start_s) / 3600.0).apply(math.floor)
    df = df[(df["arrival_slot"] >= 0) & (df["arrival_slot"] < arrival_slots)]
    # in-window duration, rounded to whole hours (min 1); keep only >=1h schedulable jobs
    df["dur_h"] = ((df["end_eff"] - df["start_time"]) / 3600.0).clip(lower=0)
    df["duration_slots"] = df["dur_h"].round().clip(lower=1).astype(int)
    df = df[df["dur_h"] >= 0.5]   # drop genuinely sub-hour (rounds to <1h); ~6.5% of GPU-hours
    df["duration_slots"] = df["duration_slots"].clip(upper=horizon_h)
    return df


def attach_util(df: pd.DataFrame, util: pd.DataFrame) -> pd.DataFrame:
    m = df.merge(util, on=["job_name","task_name"], how="left")
    # utilization as a fraction of the request, capped [0,1]; impute missing from the medians.
    m["cpu_util_ratio"] = (m["cpu_usage"] / m["plan_cpu"]).clip(lower=0, upper=1)
    gpu = m["plan_gpu"] > 0
    m["gpu_util_ratio"] = pd.Series(1.0, index=m.index)
    m.loc[gpu, "gpu_util_ratio"] = (m.loc[gpu, "gpu_wrk_util"] / m.loc[gpu, "plan_gpu"]).clip(lower=0, upper=1)
    n_meas_c = int(m["cpu_util_ratio"].notna().sum()); n_imp_c = int(m["cpu_util_ratio"].isna().sum())
    n_imp_g = int(m.loc[gpu, "gpu_util_ratio"].isna().sum())
    # impute tasks lacking sensor data with the trace-wide mean (unbiased for aggregate energy)
    m["cpu_util_ratio"] = m["cpu_util_ratio"].fillna(CPU_GLOBAL_UTIL)
    m["gpu_util_ratio"] = m["gpu_util_ratio"].fillna(GPU_GLOBAL_UTIL)
    print(f"  util: measured {n_meas_c}/{len(m)} tasks ({100*n_meas_c/max(len(m),1):.0f}%); "
          f"imputed cpu {n_imp_c}, gpu {n_imp_g} (global means cpu={CPU_GLOBAL_UTIL}, gpu={GPU_GLOBAL_UTIL})")
    return m


def stratified_sample(df: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
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
        g = df[df["arrival_slot"] == slot]
        parts.append(g if len(g) <= n else g.sample(n=n, random_state=rng.randint(0, 2**31-1)))
    out = pd.concat(parts, ignore_index=True) if parts else df.head(0)
    return out.sort_values(["arrival_slot","start_time"]).reset_index(drop=True)


def assign_fields(df: pd.DataFrame, seed: int, fixed_slack: int = 0) -> pd.DataFrame:
    out = df.copy()
    out["tier"] = out["task_name"].apply(lambda r: "firm" if str(r) in FIRM_ROLES else "flexible")
    rng = random.Random(seed + 7)
    # fixed_slack>0 -> every flexible job gets exactly that slack (for the slack-sensitivity sweep);
    # else draw the literature-grounded skewed distribution.
    def _slack(i):
        if out["tier"].iloc[i] != "flexible":
            return 0
        return fixed_slack if fixed_slack > 0 else rng.choices(SLACK_OPTIONS, weights=SLACK_WEIGHTS)[0]
    out["slack"] = [_slack(i) for i in range(len(out))]
    out["deadline_slots"] = out["duration_slots"].astype(int) + pd.Series(out["slack"].values, index=out.index).astype(int)
    out["cpu_cores"] = (out["plan_cpu"] / 100.0).clip(lower=0.05)
    out["mem_gb"] = out["plan_mem"].clip(lower=1.0).round().astype(int)
    out["gpu_req"] = (out["plan_gpu"].fillna(0.0) / 100.0).clip(lower=0.0)
    out["pod_id"] = [f"m{i:04d}" for i in range(len(out))]
    return out


def make_deployment(row: pd.Series) -> dict:
    dep = copy.deepcopy(DEPLOYMENT_TEMPLATE)
    pid = str(row["pod_id"]); dur, dl = int(row["duration_slots"]), int(row["deadline_slots"])
    name = f"{pid}-duration-{dur}h-deadline-{dl}h"
    dep["metadata"]["name"] = name
    dep["metadata"]["labels"] = {"app": name, "duration": f"duration-{dur}h", "deadline": f"deadline-{dl}h"}
    dep["metadata"]["annotations"] = {
        "trace/source": "alibaba_gpu_2020",
        "trace/task_name": str(row["task_name"]),
        "trace/tier": str(row["tier"]),
        "trace/gpu_request": f"{float(row['gpu_req']):.4f}",
        "trace/gpu_type": str(row["gpu_type"]),
        "trace/cpu_util_ratio": f"{float(row['cpu_util_ratio']):.4f}",
        "trace/gpu_util_ratio": f"{float(row['gpu_util_ratio']):.4f}",
        "trace/deadline_slack_hours": str(int(row["slack"])),
        "trace/note": "measured-util (gpu_wrk_util/cpu_usage) drives dynamic power; V100-class node",
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
    base = "external/real-traces/raw/alibaba-gpu-2020"
    ap.add_argument("--task-csv", default=f"{base}/pai_task_table.csv")
    ap.add_argument("--sensor-csv", default=f"{base}/pai_sensor_table.csv")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--window-start-hours", type=float, default=672.0,
                    help="start in the sensor-logged region (h>=672 has ~76-82%% measured coverage)")
    ap.add_argument("--arrival-window-slots", type=int, default=24)
    ap.add_argument("--horizon-hours", type=int, default=48)
    ap.add_argument("--target-pods", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fixed-slack", type=int, default=0,
                    help="if >0, all flexible jobs get this slack (hours) for the slack-sensitivity sweep")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    out = (repo / args.output_root).resolve(); out.mkdir(parents=True, exist_ok=True)
    w0 = args.window_start_hours * 3600.0

    print("loading task window ..."); df = load_window(repo / args.task_csv, w0, args.arrival_window_slots, args.horizon_hours)
    print(f"  window tasks (>=1h, arrival in [0,{args.arrival_window_slots})): {len(df):,}")
    print("loading sensor utilization ..."); util = load_sensor_util(repo / args.sensor_csv)
    df = attach_util(df, util)
    df = stratified_sample(df, args.target_pods, args.seed)
    df = assign_fields(df, args.seed, fixed_slack=args.fixed_slack)
    write_workloads(df, out, args.arrival_window_slots)
    nflex = int((df["tier"] == "flexible").sum()); ngpu = int((df["gpu_req"] > 0).sum())
    print(f"\nwrote {len(df)} pods to {out}/workloads ({args.arrival_window_slots} slots)")
    print(f"  flexible={nflex} ({100*nflex/max(len(df),1):.0f}%)  gpu-requesting={ngpu}  "
          f"mean gpu_util_ratio={df.loc[df['gpu_req']>0,'gpu_util_ratio'].mean():.2f}  "
          f"mean cpu_util_ratio={df['cpu_util_ratio'].mean():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
