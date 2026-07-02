#!/usr/bin/env python3
"""Shared Azure Packing 2020 testbed builder for the no-harm pilots (CPU/RAM general-cloud fleet).

The Azure counterpart to ``alibaba_common.py``. It builds a REALISTIC, utilization-calibrated CPU
fleet running the real Azure Packing 2020 VM trace, so the Azure (general-cloud, CPU/RAM) and
Alibaba (AI-cluster, GPU) testbeds make the SAME assumptions and are honestly comparable:

  * Energy: each pod carries an imputed measured-utilization fraction (trace/cpu_util_ratio) so the
    engine drives DYNAMIC power from usage, not from a request=100% upper bound. Packing 2020 is an
    allocation trace with no per-VM telemetry, so we use the trace-population mean -- the same
    population-mean imputation the Alibaba converter applies to its sensor-missing tasks.
  * Nodes: a realistic mid-range 2-socket server (32 cores, 128 GiB; idle ~28% of peak per
    SPECpower/LBNL), NOT the synthetic generator's oversized 192-core nodes (which produced a ~1%
    utilization artifact). The node COUNT is provisioned to a realistic target utilization.
  * Slack: capped at 24 h, drawn from the same skewed distribution as Alibaba (set in the converter).
  * Signals/regions/PilotConfig: shared with the Alibaba testbed (DE/FR/ES/IT-NO, time-aligned
    CI + WUE) so only the workload class and the binding resource differ.

We rebuild the workloads from each window's ``canonical_trace_workload.csv`` (the converter's
reproducible per-pod intermediate) so the corrected energy/slack model takes effect without needing
the upstream SQLite. No engine edits; the existing converter functions are the single source of truth.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
TRACES = SCRIPT_DIR / "real_traces"
SERVER = REPO_ROOT / "pkg" / "carbon-aware" / "server-python"
for _p in (str(SERVER), str(SCRIPT_DIR), str(TRACES)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import convert_azure_packing_trace as conv  # noqa: E402  (reuse the audited converter functions)
# Shared, stable testbed scaffolding (identical regions/signals/PilotConfig/stats for both testbeds).
from alibaba_common import (  # noqa: E402
    CONFIG_FILE, N_REGIONS, REGIONS, TIMEALIGNED, ci95, floats, ints, make_pilot_config, mean,
)

# Realistic mid-range CPU server (cite SPECpower / LBNL): idle ~28% of peak = healthy dynamic range.
# Embodied water is left at the node default (0): for a CPU server it is dominated by chip/board fab
# and is comparatively negligible beside the GPU/HBM fab water modelled on the Alibaba nodes; the
# engine has no node-level embodied-water annotation, so this is also the conservative choice.
CPU_NODE = dict(cpu_cores=32, mem_gi=128, idle_w=110.0, active_w=250.0, max_w=400.0,
                embodied_kg=1200.0, lifetime_years=5)

__all__ = [
    "CPU_NODE", "REGIONS", "N_REGIONS", "TIMEALIGNED", "CONFIG_FILE", "make_pilot_config",
    "mean", "ci95", "ints", "floats",
    "write_cpu_fleet", "build_window_from_canonical", "workload_peak_cores",
    "provision_for_util", "build_fleet",
]


# ----------------------------------------------------------------------------- fleet
def write_cpu_fleet(path: Path, nodes_per_region: int, spec: dict = CPU_NODE) -> int:
    """Write the realistic CPU fleet (nodes_per_region per region); returns total node count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    docs, n = [], 0
    for region in REGIONS:
        for k in range(nodes_per_region):
            n += 1
            docs.append({
                "apiVersion": "v1", "kind": "Node",
                "metadata": {"name": f"cpu-{region.lower()}-{k}",
                             "labels": {"topology.kubernetes.io/region": region,
                                        "hardware.carbon/subcategory": "Server"},
                             "annotations": {
                                 "hardware.carbon/embodied_emissions": str(spec["embodied_kg"]),
                                 "hardware.carbon/lifetime_years": str(spec["lifetime_years"]),
                                 "hardware.power/idle_watts": str(spec["idle_w"]),
                                 "hardware.power/active_watts": str(spec["active_w"]),
                                 "hardware.power/max_watts": str(spec["max_w"]),
                             }},
                "status": {"allocatable": {"cpu": str(spec["cpu_cores"]),
                                           "memory": f"{spec['mem_gi']}Gi"}},
            })
    with path.open("w") as fh:
        for d in docs:
            yaml.safe_dump(d, fh, sort_keys=False); fh.write("---\n")
    return n


def build_fleet(fleet_dir: Path, nodes_per_region: int) -> Tuple[Path, int, int]:
    """Write the CPU fleet; returns (nodes_file, n_nodes, total_cpu_cores)."""
    fleet_dir.mkdir(parents=True, exist_ok=True)
    nodes_file = fleet_dir / "nodes.yaml"
    n_nodes = write_cpu_fleet(nodes_file, nodes_per_region)
    return nodes_file, n_nodes, n_nodes * int(CPU_NODE["cpu_cores"])


# ----------------------------------------------------------------------------- workloads
def build_window_from_canonical(canonical_csv: Path, out_dir: Path, *, seed: int,
                                fixed_slack: int = 0, cpu_util: float = conv.CPU_UTIL_MEAN) -> Dict:
    """Regenerate timeslot workloads from a window's canonical CSV with the corrected energy/slack
    model (imputed utilization + 24 h-capped skewed slack). Returns trace-characterization stats."""
    df = pd.read_csv(canonical_csv)
    if "arrival_slot" not in df.columns:
        raise SystemExit(f"{canonical_csv} lacks arrival_slot (re-run the converter)")
    arrival_slots = int(df["arrival_slot"].max()) + 1
    slack_opts = list(conv.SLACK_OPTIONS) if fixed_slack <= 0 else [int(fixed_slack)]
    slack_wts = conv.SLACK_WEIGHTS if fixed_slack <= 0 else None
    canonical = conv.assign_workload_fields(
        df, seed=seed, slack_options=slack_opts, flexible_priorities={1},
        slack_weights=slack_wts, cpu_util=cpu_util,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    conv.write_workloads(canonical, out_dir, arrival_window_slots=arrival_slots)
    n = len(canonical)
    n_flex = int((canonical["tier"] == "flexible").sum())
    return {
        "workloads_dir": out_dir / "workloads", "n_pods": n, "n_flexible": n_flex,
        "flexible_pct": 100.0 * n_flex / max(n, 1), "cpu_util_ratio": float(cpu_util),
        "arrival_slots": arrival_slots,
    }


# ----------------------------------------------------------------------------- provisioning
def workload_peak_cores(workloads: Path) -> float:
    """Peak concurrent CPU-core demand across arrival slots (lower bound on required capacity)."""
    import glob
    import re
    from collections import defaultdict
    per_slot: Dict[int, float] = defaultdict(float)
    for f in sorted(glob.glob(str(workloads / "timeslot_*.yaml"))):
        slot = int(re.search(r"timeslot_(\d+)", f).group(1))
        for d in (x for x in yaml.safe_load_all(open(f)) if x):
            dur = int(re.search(r"duration-(\d+)h", d["metadata"]["name"]).group(1))
            req = d["spec"]["template"]["spec"]["containers"][0]["resources"]["requests"]
            c = req.get("cpu", "0")
            cores = float(c[:-1]) / 1000 if c.endswith("m") else float(c)
            for k in range(dur):
                per_slot[slot + k] += cores
    return max(per_slot.values()) if per_slot else 0.0


def provision_for_util(peak_cores: float, target_util: float) -> int:
    """Nodes-per-region that places the workload's peak concurrent cores at ~target utilization."""
    return max(1, round(peak_cores / target_util / int(CPU_NODE["cpu_cores"]) / N_REGIONS))
