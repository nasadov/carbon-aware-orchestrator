#!/usr/bin/env python3
"""
Convert Azure Trace for Packing 2020 VM requests into the repository's
Kubernetes-style timeslot workload format.

The converter intentionally keeps the scheduling harness unchanged. It writes:

  canonical_trace_workload.csv
  trace_diagnostics.csv
  manifest.json
  workloads/timeslot_*.yaml
  workloads-vanilla/timeslot_*.yaml

The Azure trace has real arrivals, lifetimes, priorities, and normalized VM
type resources. It does not have user deadlines, so this converter records that
deadlines are synthetic and generated as duration + slack.

Coherence with the Alibaba GPU testbed (so the two are comparable):
  * Energy: Packing 2020 is an *allocation* trace with no per-VM CPU-usage telemetry, so each
    pod is imputed the trace-population mean utilization (trace/cpu_util_ratio) and the engine
    drives DYNAMIC power from usage, not request -- the same population-mean imputation the
    Alibaba converter uses for its sensor-missing tasks (no more request=100% energy).
  * Slack: drawn from a literature-grounded distribution skewed to a few hours and CAPPED at 24 h
    (no 48 h tail past the signal horizon), matching the Alibaba SLACK distribution.
  * Tier: firm/flexible from the trace priority (evictability) -- an observable operator label.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


DEPLOYMENT_TEMPLATE = {
    "apiVersion": "apps/v1",
    "kind": "Deployment",
    "metadata": {
        "name": "",
        "namespace": "default",
        "labels": {"app": "", "duration": "", "deadline": ""},
        "annotations": {},
    },
    "spec": {
        "replicas": 1,
        "selector": {"matchLabels": {"name": ""}},
        "template": {
            "metadata": {"labels": {"name": ""}},
            "spec": {
                "schedulerName": "fogatlas",
                "containers": [
                    {
                        "name": "fake-container",
                        "image": "fake-image",
                        "resources": {"requests": {"memory": "", "cpu": ""}},
                    }
                ],
            },
        },
    },
}


CPU_OPTIONS = ["100m", "250m", "500m", "1000m", "2000m"]
MEM_OPTIONS = ["128Mi", "256Mi", "512Mi", "1Gi", "2Gi"]
DURATION_OPTIONS = [1, 3, 6]
# Deferral slack (hours) for the FLEXIBLE (deferrable) class. ALIGNED with the Alibaba GPU
# converter and the carbon-flexibility literature so the two testbeds make the same deferrability
# assumption: grid benefit saturates after a few hours (Chen & Zheng 2026), the production envelope
# is within-day ~24 h (Google Carbon-Intelligent Computing, Radovanovic 2023), DR events are a few
# hours (EPRI DCFlex). So we draw a distribution SKEWED to a few hours, CAPPED at 24 h (no 48 h tail
# that would defer past the signal horizon). Mean ~9 h, matching the Alibaba SLACK distribution.
SLACK_OPTIONS = [3, 6, 12, 24]
SLACK_WEIGHTS = [0.35, 0.30, 0.20, 0.15]
# Imputed CPU utilization as a fraction of the request. The Azure Packing 2020 trace is an
# *allocation* trace: it records VM requests/lifetimes/priority but NO per-VM CPU-usage telemetry.
# We therefore impute the trace-population mean utilization and write it as trace/cpu_util_ratio so
# the engine drives DYNAMIC power from usage (not request) -- the same population-mean imputation
# rule the Alibaba converter applies to its sensor-missing tasks (CPU_GLOBAL_UTIL). 0.25 is
# representative of measured average VM CPU utilization in Azure's fleet (most VMs average well
# below 50%; fleet averages ~20-30% per Cortez et al., "Resource Central", SOSP 2017).
# NOTE: verify the exact figure against the cited source before camera-ready; overridable via --cpu-util.
CPU_UTIL_MEAN = 0.25
DEFAULT_ARRIVAL_WINDOW_SLOTS = 12
EXECUTION_HORIZON_SLOTS = 48


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def format_hours(hours: int) -> str:
    return f"{int(hours)}h"


def quantile_options(values: pd.Series, options: list[str | int]) -> list[str | int]:
    if values.empty:
        return []
    ranks = values.rank(method="first", pct=True)
    n = len(options)
    indices = (ranks * n).astype(int).clip(upper=n - 1)
    return [options[int(idx)] for idx in indices]


def clean_trace_id(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def load_window_rows(sqlite_path: Path, window_day: int, window_days: int) -> pd.DataFrame:
    query = """
    WITH type_avg AS (
      SELECT vmTypeId, AVG(core) AS core, AVG(memory) AS memory
      FROM vmType
      GROUP BY vmTypeId
    )
    SELECT
      v.vmId,
      v.tenantId,
      v.vmTypeId,
      v.priority,
      v.starttime,
      v.endtime,
      t.core,
      t.memory
    FROM vm v
    JOIN type_avg t ON v.vmTypeId = t.vmTypeId
    WHERE v.starttime >= ?
      AND v.starttime < ?
      AND v.endtime IS NOT NULL
      AND v.endtime > v.starttime
    ORDER BY v.starttime, v.vmId
    """
    with sqlite3.connect(str(sqlite_path)) as conn:
        df = pd.read_sql_query(query, conn, params=(window_day, window_day + window_days))
    return df


def stratified_sample_by_hour(
    df: pd.DataFrame,
    window_day: int,
    target_pods: int,
    seed: int,
    arrival_window_slots: int,
) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"No Azure trace rows found for window starting at day {window_day}")

    rng = random.Random(seed)
    df = df.copy()
    df["arrival_slot"] = ((df["starttime"] - float(window_day)) * 24.0).apply(math.floor).astype(int)
    df = df[(df["arrival_slot"] >= 0) & (df["arrival_slot"] < arrival_window_slots)]
    if df.empty:
        raise ValueError(f"No rows remain in arrival window starting at day {window_day}")

    target = min(target_pods, len(df))
    counts = df["arrival_slot"].value_counts().sort_index()
    raw_alloc = counts / counts.sum() * target
    alloc = raw_alloc.apply(math.floor).astype(int)
    remainder = target - int(alloc.sum())
    if remainder > 0:
        fractional = (raw_alloc - alloc).sort_values(ascending=False)
        for slot in fractional.index[:remainder]:
            alloc.loc[slot] += 1

    parts = []
    for slot, n in alloc.items():
        if n <= 0:
            continue
        group = df[df["arrival_slot"] == slot]
        if len(group) <= n:
            parts.append(group)
            continue
        parts.append(group.sample(n=n, random_state=rng.randint(0, 2**31 - 1)))

    sampled = pd.concat(parts, ignore_index=True)
    if len(sampled) < target:
        missing = target - len(sampled)
        already = set(sampled.index)
        rest = df.drop(index=list(already), errors="ignore")
        if not rest.empty:
            sampled = pd.concat(
                [sampled, rest.sample(n=min(missing, len(rest)), random_state=rng.randint(0, 2**31 - 1))],
                ignore_index=True,
            )

    return sampled.sort_values(["arrival_slot", "starttime", "vmId"]).reset_index(drop=True)


def assign_workload_fields(
    df: pd.DataFrame,
    seed: int,
    slack_options: list[int],
    flexible_priorities: set[int],
    slack_weights: list[float] | None = None,
    cpu_util: float = CPU_UTIL_MEAN,
) -> pd.DataFrame:
    out = df.copy()
    duration_hours_raw = ((out["endtime"] - out["starttime"]) * 24.0).clip(lower=1.0)
    out["raw_duration_hours"] = duration_hours_raw
    out["duration_slots"] = quantile_options(duration_hours_raw, DURATION_OPTIONS)
    out["cpu_request"] = quantile_options(out["core"], CPU_OPTIONS)
    out["memory_request"] = quantile_options(out["memory"], MEM_OPTIONS)

    # Firm/flexible tier from the trace priority (evictability). The deferrable
    # (flexible) class is the evictable/low-priority VMs; the firm class is
    # latency-sensitive and gets zero slack (deadline == duration -> protected by
    # the slack-based classifier, no engine change required). This makes the
    # flexible share an observable property of the real trace, not a synthetic knob.
    out["tier"] = [
        "flexible" if int(p) in flexible_priorities else "firm" for p in out["priority"]
    ]
    # Slack for flexible jobs: draw the literature-grounded SKEWED distribution (capped at the
    # max slack option, default 24 h) when weights are given -- identical in spirit to the Alibaba
    # converter -- else fall back to the legacy round-robin. Firm jobs get 0 (deadline == duration).
    rng = random.Random(seed + 10007)
    if slack_weights:
        slacks = [
            rng.choices(slack_options, weights=slack_weights)[0] if out["tier"].iloc[i] == "flexible" else 0
            for i in range(len(out))
        ]
    else:
        flex_slacks = [slack_options[i % len(slack_options)] for i in range(len(out))]
        rng.shuffle(flex_slacks)
        slacks = [flex_slacks[i] if out["tier"].iloc[i] == "flexible" else 0 for i in range(len(out))]
    out["deadline_slack_slots"] = slacks
    out["deadline_slots"] = out["duration_slots"].astype(int) + pd.Series(slacks, index=out.index).astype(int)
    # Imputed measured-utilization fraction (no per-VM telemetry in Packing 2020). Written as
    # trace/cpu_util_ratio so the engine scales DYNAMIC power by usage, matching the Alibaba testbed.
    out["cpu_util_ratio"] = float(cpu_util)
    out["pod_id"] = [f"m{i:04d}" for i in range(len(out))]
    out["trace_source"] = "azure_packing_2020"
    out["source_id"] = out["vmId"].apply(clean_trace_id)
    return out


def make_deployment(row: pd.Series, vanilla: bool) -> dict:
    dep = deepcopy(DEPLOYMENT_TEMPLATE)
    if vanilla:
        dep["spec"]["template"]["spec"].pop("schedulerName", None)

    pod_id = str(row["pod_id"])
    duration = int(row["duration_slots"])
    deadline = int(row["deadline_slots"])
    duration_str = format_hours(duration)
    deadline_str = format_hours(deadline)
    full_name = f"{pod_id}-duration-{duration_str}-deadline-{deadline_str}"

    dep["metadata"]["name"] = full_name
    dep["metadata"]["labels"]["app"] = full_name
    dep["metadata"]["labels"]["duration"] = f"duration-{duration_str}"
    dep["metadata"]["labels"]["deadline"] = f"deadline-{deadline_str}"
    dep["metadata"]["annotations"] = {
        "trace/source": str(row["trace_source"]),
        "trace/source_id": str(row["source_id"]),
        "trace/vm_type_id": str(row["vmTypeId"]),
        "trace/priority": str(row["priority"]),
        "trace/tier": str(row["tier"]),
        "trace/deadline_slack_hours": str(int(row["deadline_slack_slots"])),
        "trace/cpu_util_ratio": f"{float(row.get('cpu_util_ratio', CPU_UTIL_MEAN)):.4f}",
        "trace/raw_start_day": f"{float(row['starttime']):.8f}",
        "trace/raw_end_day": f"{float(row['endtime']):.8f}",
        "trace/raw_duration_hours": f"{float(row['raw_duration_hours']):.4f}",
        "trace/raw_core_fraction": f"{float(row['core']):.8f}",
        "trace/raw_memory_fraction": f"{float(row['memory']):.8f}",
        "trace/deadline_note": "synthetic_duration_plus_slack",
    }
    dep["spec"]["selector"]["matchLabels"]["name"] = pod_id
    dep["spec"]["template"]["metadata"]["labels"]["name"] = pod_id
    container = dep["spec"]["template"]["spec"]["containers"][0]
    container["resources"]["requests"]["cpu"] = str(row["cpu_request"])
    container["resources"]["requests"]["memory"] = str(row["memory_request"])
    return dep


def write_yaml_docs(path: Path, docs: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for doc in docs:
            yaml.safe_dump(doc, handle, sort_keys=False)
            handle.write("---\n")


def write_workloads(df: pd.DataFrame, output_root: Path, arrival_window_slots: int) -> None:
    workloads = output_root / "workloads"
    workloads_vanilla = output_root / "workloads-vanilla"
    for path in (workloads, workloads_vanilla):
        if path.exists():
            shutil.rmtree(path)
    workloads.mkdir(parents=True, exist_ok=True)
    workloads_vanilla.mkdir(parents=True, exist_ok=True)

    for slot in range(arrival_window_slots):
        group = df[df["arrival_slot"] == slot]
        write_yaml_docs(workloads / f"timeslot_{slot}.yaml", [make_deployment(row, False) for _, row in group.iterrows()])
        write_yaml_docs(workloads_vanilla / f"timeslot_{slot}.yaml", [make_deployment(row, True) for _, row in group.iterrows()])


def write_diagnostics(
    source_df: pd.DataFrame,
    sampled: pd.DataFrame,
    output_root: Path,
    arrival_window_slots: int,
) -> None:
    rows = []
    for slot in range(arrival_window_slots):
        rows.append(
            {
                "arrival_slot": slot,
                "source_rows": int((source_df["arrival_slot"] == slot).sum()) if "arrival_slot" in source_df else "",
                "selected_rows": int((sampled["arrival_slot"] == slot).sum()),
            }
        )
    pd.DataFrame(rows).to_csv(output_root / "trace_diagnostics.csv", index=False)

    resource_diag = pd.DataFrame(
        {
            "metric": [
                "selected_pods",
                "raw_core_min",
                "raw_core_median",
                "raw_core_max",
                "raw_memory_min",
                "raw_memory_median",
                "raw_memory_max",
                "raw_duration_hours_min",
                "raw_duration_hours_median",
                "raw_duration_hours_max",
            ],
            "value": [
                len(sampled),
                sampled["core"].min(),
                sampled["core"].median(),
                sampled["core"].max(),
                sampled["memory"].min(),
                sampled["memory"].median(),
                sampled["memory"].max(),
                sampled["raw_duration_hours"].min(),
                sampled["raw_duration_hours"].median(),
                sampled["raw_duration_hours"].max(),
            ],
        }
    )
    resource_diag.to_csv(output_root / "trace_resource_diagnostics.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite-path", required=True)
    parser.add_argument("--allow-horizon-overflow", action="store_true", help="Permit arrival+duration/deadline past the nominal horizon (matches the original Azure run; the scheduler executes slots beyond 24).")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--window-day", type=int, default=0)
    parser.add_argument("--window-days", type=int, default=1)
    parser.add_argument(
        "--arrival-window-slots",
        type=int,
        default=DEFAULT_ARRIVAL_WINDOW_SLOTS,
        help="Number of hourly arrival slots sampled from each trace day; execution still uses the fixed 24-slot horizon.",
    )
    parser.add_argument("--target-pods", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--slack-options",
        type=parse_int_list,
        default=SLACK_OPTIONS,
        help="deferral-slack tiers (hours) drawn for the flexible class; default 3,6,12,24 (capped at 24)",
    )
    parser.add_argument(
        "--cpu-util",
        type=float,
        default=CPU_UTIL_MEAN,
        help="imputed CPU utilization fraction of request (Packing 2020 has no per-VM usage telemetry); "
        f"default {CPU_UTIL_MEAN} (Azure fleet average, Cortez et al. SOSP 2017)",
    )
    parser.add_argument(
        "--flexible-priorities",
        type=parse_int_list,
        default=[1],
        help="trace priority values mapped to the flexible (deferrable/evictable) tier; others "
        "are firm (slack 0). Default 1 = the LOW-priority/evictable class (Azure Packing 2020 "
        "schema: priority 0=high/firm, 1=low/evictable; 'low-priority work may be evicted early').",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sqlite_path = Path(args.sqlite_path).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.arrival_window_slots <= 0 or args.arrival_window_slots > EXECUTION_HORIZON_SLOTS:
        raise ValueError(
            f"--arrival-window-slots must be between 1 and {EXECUTION_HORIZON_SLOTS}"
        )

    source = load_window_rows(sqlite_path, args.window_day, args.window_days)
    source["arrival_slot"] = ((source["starttime"] - float(args.window_day)) * 24.0).apply(math.floor).astype(int)
    sampled = stratified_sample_by_hour(
        source,
        window_day=args.window_day,
        target_pods=args.target_pods,
        seed=args.seed,
        arrival_window_slots=args.arrival_window_slots,
    )
    slack_opts = list(args.slack_options)
    slack_wts = SLACK_WEIGHTS if slack_opts == SLACK_OPTIONS else None
    canonical = assign_workload_fields(
        sampled,
        seed=args.seed,
        slack_options=slack_opts,
        flexible_priorities=set(args.flexible_priorities),
        slack_weights=slack_wts,
        cpu_util=args.cpu_util,
    )
    earliest_finish = canonical["arrival_slot"] + canonical["duration_slots"]
    if (not args.allow_horizon_overflow) and (earliest_finish > EXECUTION_HORIZON_SLOTS).any():
        count = int((earliest_finish > EXECUTION_HORIZON_SLOTS).sum())
        raise ValueError(
            f"{count} sampled pods cannot finish within the {EXECUTION_HORIZON_SLOTS}-slot execution horizon"
        )
    absolute_deadline = canonical["arrival_slot"] + canonical["deadline_slots"]
    if (not args.allow_horizon_overflow) and (absolute_deadline > EXECUTION_HORIZON_SLOTS).any():
        count = int((absolute_deadline > EXECUTION_HORIZON_SLOTS).sum())
        raise ValueError(
            f"{count} sampled pods have deadlines beyond the {EXECUTION_HORIZON_SLOTS}-slot execution horizon"
        )
    canonical.to_csv(output_root / "canonical_trace_workload.csv", index=False)
    write_workloads(canonical, output_root, arrival_window_slots=args.arrival_window_slots)
    write_diagnostics(source, canonical, output_root, arrival_window_slots=args.arrival_window_slots)

    manifest = {
        "source": "azure_packing_2020",
        "source_url": "https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2/azurevmallocation_dataset2020/AzurePackingTraceV1.zip",
        "sqlite_path": str(sqlite_path),
        "window_day": args.window_day,
        "window_days": args.window_days,
        "arrival_window_slots": args.arrival_window_slots,
        "execution_horizon_slots": EXECUTION_HORIZON_SLOTS,
        "target_pods": args.target_pods,
        "selected_pods": int(len(canonical)),
        "seed": args.seed,
        "resource_mapping": "rank-preserving quantile mapping to Kubernetes request classes",
        "duration_mapping": "rank-preserving quantile mapping to 1h, 3h, 6h",
        "deadline_mapping": (
            "firm/flexible tier from trace priority (evictability): flexible (deferrable) "
            "class = priority in {flexible_priorities}; firm class gets slack 0 (deadline == "
            "duration). Flexible class gets synthetic duration + slack drawn from a literature-"
            f"grounded skewed distribution over {slack_opts} h (capped at 24 h; aligned with the "
            "Alibaba converter)."
        ),
        "flexible_priorities": list(args.flexible_priorities),
        "slack_options_hours": slack_opts,
        "slack_weights": slack_wts,
        "cpu_util_ratio_imputed": args.cpu_util,
        "cpu_util_note": (
            "Packing 2020 has no per-VM CPU-usage telemetry; every pod is imputed the trace-"
            "population mean CPU utilization (written as trace/cpu_util_ratio, scales dynamic "
            "power) -- the same population-mean rule the Alibaba converter applies to sensor-"
            "missing tasks. Value ~Azure fleet average (Cortez et al., Resource Central, SOSP 2017)."
        ),
        "tier_counts": {k: int(v) for k, v in canonical["tier"].value_counts().to_dict().items()},
        "tier_mapping_note": (
            "Azure Packing 2020 schema: priority 0=high/firm, 1=low/evictable ('low-priority "
            "work may be evicted early'). We map the low-priority/evictable class (priority 1) to "
            "the deferrable/flexible tier; high-priority (0) is firm (slack 0). Confirmed against "
            "the AzurePublicDataset AzureTracesForPacking2020 schema doc."
        ),
        "license": "CC-BY-4.0; see Azure/AzurePublicDataset",
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote Azure trace workload: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
