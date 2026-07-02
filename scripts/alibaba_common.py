#!/usr/bin/env python3
"""Shared Alibaba GPU v2020 testbed builder for the no-harm sweeps (fleet-scale, lever, WaterWise).

Background. The synthetic generator produced a CPU fleet at ~1% utilization (tiny generated pods on
192-core nodes with no binding resource), which makes absolute magnitudes and the fleet-scaling slope
artifacts of the generator, not the workload. This module replaces that generator for the *general*
experiments with the REAL Alibaba GPU v2020 trace: real job sizes (median 6 vCPU + 0.5 GPU), measured
utilization driving dynamic power (closes R2), and a scarce binding resource (8 GPUs/node) so the
fleet is realistically utilized (~106% GPU-allocated, oversubscribed) rather than ~1%.

It loads the (expensive, ~1 GB) sensor table and the task window ONCE per process and re-samples per
(size, seed), and re-exports the period-accurate V100 DGX-1 fleet builder from run_alibaba_no_harm so
every sweep — fleet-scale, lever ablation, WaterWise contrast — shares one identical testbed. No
edits to the engine or to the existing (synthetic) sweep scripts; those stay in place as reference.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
TRACES = SCRIPT_DIR / "real_traces"
SERVER = REPO_ROOT / "pkg" / "carbon-aware" / "server-python"
for _p in (str(SERVER), str(SCRIPT_DIR), str(TRACES)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import convert_alibaba_gpu_v2020 as conv  # noqa: E402  (reuse the audited converter functions)
from run_alibaba_no_harm import GPU_NODE, REGIONS, write_gpu_fleet  # noqa: E402  (one shared fleet)
from carbon_aware.no_harm_flexibility import PilotConfig  # noqa: E402

TIMEALIGNED = REPO_ROOT / "pkg" / "carbon-aware" / "data" / "timealigned"
CONFIG_FILE = REPO_ROOT / "pkg" / "carbon-aware" / "infra-workload-config.yaml"
RAW = REPO_ROOT / "external" / "real-traces" / "raw" / "alibaba-gpu-2020"
TASK_CSV = RAW / "pai_task_table.csv"
SENSOR_CSV = RAW / "pai_sensor_table.csv"

N_REGIONS = len(REGIONS)
GPUS_PER_NODE = int(GPU_NODE["gpu_count"])

# Process-level caches so a multi-(size, seed) sweep pays the trace/sensor load exactly once.
_UTIL_CACHE = None
_WINDOW_CACHE: Dict[Tuple, "object"] = {}


# ----------------------------------------------------------------------------- trace loading
def _load_util():
    global _UTIL_CACHE
    if _UTIL_CACHE is None:
        if not SENSOR_CSV.exists():
            raise SystemExit(f"missing sensor table: {SENSOR_CSV}")
        print("[alibaba] loading measured-utilization sensor table (~1 GB, once per run) ...", flush=True)
        _UTIL_CACHE = conv.load_sensor_util(SENSOR_CSV)
    return _UTIL_CACHE


def _load_window(window_start_hours: float, arrival_slots: int, horizon_hours: int):
    """Task window with measured util attached; cached per window geometry."""
    key = (window_start_hours, arrival_slots, horizon_hours)
    if key not in _WINDOW_CACHE:
        if not TASK_CSV.exists():
            raise SystemExit(f"missing task table: {TASK_CSV}")
        w0 = window_start_hours * 3600.0
        df = conv.load_window(TASK_CSV, w0, arrival_slots, horizon_hours)
        df = conv.attach_util(df, _load_util())
        _WINDOW_CACHE[key] = df
        print(f"[alibaba] window h{window_start_hours:g} ({arrival_slots} arrival slots, "
              f"{horizon_hours}h horizon): {len(df):,} schedulable (>=1h) tasks", flush=True)
    return _WINDOW_CACHE[key]


# ----------------------------------------------------------------------------- builders
def build_window(out_dir: Path, target_pods: int, seed: int, *, window_start_hours: float = 672.0,
                 arrival_slots: int = 24, horizon_hours: int = 48, fixed_slack: int = 0) -> Dict:
    """Stratified-sample `target_pods` from the cached Alibaba window (seed) and write timeslot
    workloads under `out_dir/workloads`. Returns trace-characterization stats for reporting."""
    df = _load_window(window_start_hours, arrival_slots, horizon_hours)
    s = conv.stratified_sample(df, target_pods, seed)
    s = conv.assign_fields(s, seed, fixed_slack=fixed_slack)
    out_dir.mkdir(parents=True, exist_ok=True)
    conv.write_workloads(s, out_dir, arrival_slots)
    n = len(s)
    n_flex = int((s["tier"] == "flexible").sum())
    n_gpu = int((s["gpu_req"] > 0).sum())
    gpu_demand = float(s["gpu_req"].sum())
    mean_gpu_util = float(s.loc[s["gpu_req"] > 0, "gpu_util_ratio"].mean()) if n_gpu else float("nan")
    return {
        "workloads_dir": out_dir / "workloads", "n_pods": n, "n_flexible": n_flex, "n_gpu": n_gpu,
        "flexible_pct": 100.0 * n_flex / max(n, 1), "gpu_demand": gpu_demand,
        "mean_gpu_util": mean_gpu_util,
    }


def build_fleet(fleet_dir: Path, nodes_per_region: int) -> Tuple[Path, int, int]:
    """Write the V100 DGX-1 GPU fleet; returns (nodes_file, n_nodes, total_gpus)."""
    fleet_dir.mkdir(parents=True, exist_ok=True)
    nodes_file = fleet_dir / "nodes.yaml"
    n_nodes = write_gpu_fleet(nodes_file, nodes_per_region)
    return nodes_file, n_nodes, n_nodes * GPUS_PER_NODE


def make_pilot_config(nodes_file: Path, workloads_dir: Path, output_dir: Path, *,
                      signals: Path = TIMEALIGNED, max_timeslots: int = 48,
                      scenario: str = "heatwave-drought", lever_mode: str = "both",
                      waterwise_carbon_weight=None) -> PilotConfig:
    kw = dict(
        repo_root=REPO_ROOT, nodes_file=nodes_file, workloads_dir=workloads_dir,
        forecasts_file=signals / "forecasts.json", config_file=CONFIG_FILE,
        output_dir=output_dir, max_timeslots=max_timeslots, max_pods=None,
        scenario=scenario, lever_mode=lever_mode,
        grid_signal_csv=signals / "grid_residual_region_slot.csv",
        wue_csv=signals / "wue_region_slot.csv",
    )
    # In-window off-site water: use the coherent EWIF emitted alongside CI/WUE (same Energy-Charts
    # mix) when present. Falls back to the engine default only for legacy dirs that lack it.
    ewif_path = signals / "ewif_region_slot.csv"
    if ewif_path.exists():
        kw["ewif_csv"] = ewif_path
    if waterwise_carbon_weight is not None:
        kw["waterwise_carbon_weight"] = waterwise_carbon_weight
    return PilotConfig(**kw)


def pods_for_fleet(nodes_per_region: int, oversubscription: float = 1.06,
                   mean_gpu_req: float = 0.68) -> int:
    """Pod count that holds GPU oversubscription ~constant as the fleet scales, so the
    fleet-scaling slope reflects scale, not a drifting load factor. total_gpu * over / mean_req."""
    total_gpus = nodes_per_region * N_REGIONS * GPUS_PER_NODE
    return max(1, int(round(total_gpus * oversubscription / mean_gpu_req)))


# ----------------------------------------------------------------------------- tiny stats
def ints(s: str) -> List[int]:
    return [int(x) for x in str(s).split(",") if str(x).strip()]


def floats(s: str) -> List[float]:
    return [float(x) for x in str(s).split(",") if str(x).strip()]


def mean(values: Sequence[float]) -> float:
    a = np.asarray([x for x in values if not (isinstance(x, float) and math.isnan(x))], dtype=float)
    return float(a.mean()) if a.size else float("nan")


def ci95(values: Sequence[float]) -> float:
    """Half-width of the 95% CI of the mean (Student-t, small-n correct; 0 for n<2).
    The earlier normal approximation (z=1.96) understated the half-width ~40% at n=5
    (t(0.975,4)=2.776) -- fixed 2026-07-01; all seeded tables re-derive with this."""
    a = np.asarray([x for x in values if not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if a.size < 2:
        return 0.0
    from scipy.stats import t as _t
    return float(_t.ppf(0.975, a.size - 1) * a.std(ddof=1) / math.sqrt(a.size))
