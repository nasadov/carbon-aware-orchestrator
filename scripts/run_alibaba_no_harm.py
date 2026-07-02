#!/usr/bin/env python3
"""T12-P7b — run the no-harm pilot on a real Alibaba GPU (PAI) trace window.

Generates a regionful GPU fleet (DE/FR/ES/IT-NO with per-site cooling kappa and V100-class
DGX-1 nodes, period-accurate for the v2020 trace; idle-inclusive GPU power + measured utilization),
points the no-harm pilot at a converted Alibaba GPU window, and reports the no-harm certificate,
carbon/water deltas (operational + embodied, GPU-aware), grid-stress relief, and the firm/flexible
share (from trace task roles). Reported beside the Azure (CPU/RAM) result as the generalization.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SERVER = REPO_ROOT / "pkg" / "carbon-aware" / "server-python"
for p in (str(SERVER), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from carbon_aware.no_harm_flexibility import PilotConfig, load_pods, run_no_harm_flexibility_pilot  # noqa: E402

TIMEALIGNED = REPO_ROOT / "pkg" / "carbon-aware" / "data" / "timealigned"
REGIONS = ["DE", "FR", "ES", "IT-NO"]
# V100-class GPU node (DGX-1 V100), PERIOD-ACCURATE for the Alibaba GPU v2020 trace: the trace's
# gpu_type is V100/P100/T4 (NO A100), so V100 is the realistic card. Magnitudes:
#   * per-GPU board: NVIDIA V100 SXM2 ~300 W peak / ~50 W idle (datasheet); a held-but-idle GPU still
#     draws idle -> with the idle-inclusive power model, low-util jobs aren't free. Embodied ~120 kg
#     CO2e (estimated from the A100 cradle-to-grave LCA [arXiv:2509.00093] scaled to V100 die+HBM2;
#     LEAST-CERTAIN, no dedicated V100 LCA); per-GPU embodied water ~2000 L (fab/HBM estimate).
#   * host (DGX-1 V100, non-GPU): 2x Xeon (40 cores) + ~512 GiB DDR -> ~1500 kgCO2e embodied; host
#     power ~0.7 kW idle / ~1.1 kW max. The 8 GPUs add on top -> ~3.5 kW system (DGX-1 datasheet max).
# Energy is driven by MEASURED utilization (gpu_wrk_util/cpu_usage) per pod, not requests (R2).
GPU_NODE = dict(gpu_count=8, gpu_power_w=300.0, gpu_idle_w=50.0, gpu_embodied_kg=120.0,
                gpu_embodied_water_l=2000.0, gpu_type="V100", chassis_embodied_kg=1500.0,
                lifetime_years=4, idle_w=700.0, max_w=1100.0, cpu_cores=40, mem_gi=512)


def write_gpu_fleet(path: Path, nodes_per_region: int) -> int:
    docs = []
    n = 0
    g = GPU_NODE
    for region in REGIONS:
        for k in range(nodes_per_region):
            n += 1
            docs.append({
                "apiVersion": "v1", "kind": "Node",
                "metadata": {
                    "name": f"gpu-{region.lower()}-{k}",
                    "labels": {"topology.kubernetes.io/region": region,
                               "hardware.carbon/subcategory": "Server"},
                    "annotations": {
                        "hardware.carbon/embodied_emissions": str(g["chassis_embodied_kg"]),  # kg, non-GPU
                        "hardware.carbon/lifetime_years": str(g["lifetime_years"]),
                        "hardware.power/idle_watts": str(g["idle_w"]),
                        "hardware.power/active_watts": str((g["idle_w"] + g["max_w"]) / 2),
                        "hardware.power/max_watts": str(g["max_w"]),
                        "hardware.gpu/count": str(g["gpu_count"]),
                        "hardware.gpu/power_watts": str(g["gpu_power_w"]),
                        "hardware.gpu/idle_watts": str(g["gpu_idle_w"]),
                        "hardware.gpu/embodied_emissions": str(g["gpu_embodied_kg"]),  # kg per GPU
                        "hardware.gpu/embodied_water": str(g["gpu_embodied_water_l"]),  # L per GPU
                        "hardware.gpu/type": g["gpu_type"],
                    },
                },
                "status": {"allocatable": {"cpu": str(g["cpu_cores"]), "memory": f"{g['mem_gi']}Gi"}},
            })
    import yaml
    with path.open("w") as fh:
        for d in docs:
            yaml.safe_dump(d, fh, sort_keys=False)
            fh.write("---\n")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="experiments/real_traces/alibaba_gpu_2020_d10_s42_200")
    ap.add_argument("--nodes-per-region", type=int, default=4)
    ap.add_argument("--max-timeslots", type=int, default=48)
    ap.add_argument("--lever-mode", default="both")
    ap.add_argument("--scenario", default="heatwave-drought")
    ap.add_argument("--run-name", default="t12_alibaba")
    ap.add_argument("--signals-dir", default=None,
                    help="time-aligned signals dir (default = proxy data/timealigned; pass data/timealigned_realci for measured CI)")
    args = ap.parse_args()
    import logging
    logging.disable(logging.CRITICAL)
    signals = (REPO_ROOT / args.signals_dir).resolve() if args.signals_dir else TIMEALIGNED

    window = (REPO_ROOT / args.window).resolve()
    workloads = window / "workloads"
    if not workloads.exists():
        raise SystemExit(f"no workloads/ under {window} (run real_traces/convert_alibaba_gpu_v2020.py first)")

    run_dir = REPO_ROOT / "experiments" / "flexibility" / args.run_name
    fleet_dir = run_dir / "fleet"
    fleet_dir.mkdir(parents=True, exist_ok=True)
    nodes_file = fleet_dir / "nodes.yaml"
    n_nodes = write_gpu_fleet(nodes_file, args.nodes_per_region)

    config_file = REPO_ROOT / "pkg" / "carbon-aware" / "infra-workload-config.yaml"
    pc = PilotConfig(
        repo_root=REPO_ROOT, nodes_file=nodes_file, workloads_dir=workloads,
        forecasts_file=signals / "forecasts.json", config_file=config_file,
        output_dir=run_dir / "out", max_timeslots=args.max_timeslots, max_pods=None,
        scenario=args.scenario, lever_mode=args.lever_mode,
        grid_signal_csv=signals / "grid_residual_region_slot.csv", wue_csv=signals / "wue_region_slot.csv",
        # In-window off-site water (same Energy-Charts mix as CI/WUE) when present in the signals dir.
        ewif_csv=((signals / "ewif_region_slot.csv") if (signals / "ewif_region_slot.csv").exists() else None),
    )
    pods = load_pods(workloads)
    n_gpu_pods = sum(1 for p in pods if getattr(p, "gpuRequest", 0) > 0)
    res = run_no_harm_flexibility_pilot(pc)
    rows = {r["method_key"]: r for r in res["summary_rows"]}
    flex = rows["no_harm_flex"]
    n_flex, n_prot = flex["flexible_pods"], flex["protected_pods"]
    total = n_flex + n_prot

    print(f"\n=== T12 Alibaba GPU 2020 no-harm result ({window.name}) ===")
    print(f"pods={len(pods)} (gpu-requesting={n_gpu_pods})  placed={flex['placed_pods']} unplaced={flex['unplaced_pods']}  "
          f"fleet={n_nodes} GPU nodes ({GPU_NODE['gpu_count']} x {GPU_NODE['gpu_type']} each)  max_ts={args.max_timeslots}")
    print(f"flexible share (task-role tier): {n_flex}/{total} = {100*n_flex/max(total,1):.1f}% (firm/inference={n_prot})")
    print(f"no_harm flex: certificate={flex['no_harm_certificate']}  carbon%={flex['carbon_delta_pct']:.2f}  "
          f"scarcity%={flex['scarcity_delta_pct']:.3f}  stress_avoided={flex['stress_kwh_avoided']:.4f} kWh  repairs={flex['repairs_applied']}")
    for m in ("carbon", "water_scarcity", "no_harm_search_control"):
        r = rows[m]
        print(f"  {m:>22}: cert={r['no_harm_certificate']}  carbon%={r['carbon_delta_pct']:.2f}  scarcity%={r['scarcity_delta_pct']:.3f}")
    digest = {"window": window.name, "gpu_nodes": n_nodes, "pods": len(pods), "gpu_pods": n_gpu_pods,
              "flexible_share_pct": round(100 * n_flex / max(total, 1), 2),
              "no_harm_flex": {k: flex[k] for k in ("no_harm_certificate", "carbon_delta_pct", "scarcity_delta_pct",
                                                    "stress_kwh_avoided", "repairs_applied")},
              "baselines": {m: {k: rows[m][k] for k in ("no_harm_certificate", "carbon_delta_pct", "scarcity_delta_pct")}
                            for m in ("carbon", "water_scarcity", "no_harm_search_control")}}
    (run_dir / "out" / "t12_digest.json").write_text(json.dumps(digest, indent=2))
    print(f"\noutput_dir={run_dir/'out'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
