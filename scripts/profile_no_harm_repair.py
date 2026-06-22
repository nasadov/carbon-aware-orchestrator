#!/usr/bin/env python3
"""T1 profiling harness: build a representative fleet instance and profile the
no-harm pilot (the repair is the hot path). Prints cProfile cumulative-time top-N.

Usage:
    python scripts/profile_no_harm_repair.py --nodes-per-region 4 --pods 160 \
        --timeslots 24 --max-timeslots 24
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import time
from io import StringIO
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SERVER_PYTHON_ROOT = REPO_ROOT / "pkg" / "carbon-aware" / "server-python"
for p in (str(SERVER_PYTHON_ROOT), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from carbon_aware.no_harm_flexibility import PilotConfig, run_no_harm_flexibility_pilot  # noqa: E402
from no_harm_flex_matrix import _generate_case_inputs, load_generator_dependencies  # noqa: E402

TIMEALIGNED = REPO_ROOT / "pkg" / "carbon-aware" / "data" / "timealigned"


def build_config(args) -> PilotConfig:
    config_file = REPO_ROOT / "pkg" / "carbon-aware" / "infra-workload-config.yaml"
    base_config, gen_nodes, gen_ts = load_generator_dependencies(REPO_ROOT, config_file)
    run_root = REPO_ROOT / "experiments" / "flexibility" / args.run_name
    input_dir = run_root / "inputs"
    paths = _generate_case_inputs(
        base_config=base_config,
        generate_nodes_file=gen_nodes,
        generate_timeslot_files=gen_ts,
        input_dir=input_dir,
        pod_count=args.pods,
        seed=args.seed,
        timeslots=args.timeslots,
        config_file=config_file,
        deadline_flex_hours=args.horizon,
        nodes_per_region=args.nodes_per_region,
    )
    return PilotConfig(
        repo_root=REPO_ROOT,
        nodes_file=paths["nodes_file"],
        workloads_dir=paths["workloads_dir"],
        forecasts_file=TIMEALIGNED / "forecasts.json",
        config_file=config_file,
        output_dir=run_root / "case",
        max_timeslots=args.max_timeslots,
        max_pods=None,
        scenario="heatwave-drought",
        lever_mode=args.lever_mode,
        grid_signal_csv=TIMEALIGNED / "grid_residual_region_slot.csv",
        wue_csv=TIMEALIGNED / "wue_region_slot.csv",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes-per-region", type=int, default=4)
    ap.add_argument("--pods", type=int, default=160)
    ap.add_argument("--timeslots", type=int, default=24)
    ap.add_argument("--max-timeslots", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lever-mode", default="both")
    ap.add_argument("--run-name", default="t1_profile")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--no-profile", action="store_true", help="just time it, no cProfile overhead")
    args = ap.parse_args()

    import logging
    logging.disable(logging.CRITICAL)  # silence debug spam during timing

    config = build_config(args)

    if args.no_profile:
        t0 = time.perf_counter()
        res = run_no_harm_flexibility_pilot(config)
        dt = time.perf_counter() - t0
        rows = {r["method_key"]: r for r in res["summary_rows"]}
        nh = rows.get("no_harm_flex", {})
        print(f"wall_seconds={dt:.2f} repairs={nh.get('repairs_applied')} "
              f"no_harm={nh.get('no_harm_certificate')} carbon_pct={nh.get('carbon_delta_pct'):.2f} "
              f"scarcity_pct={nh.get('scarcity_delta_pct'):.2f} stress_avoided={nh.get('stress_kwh_avoided'):.4f}")
        return 0

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    res = run_no_harm_flexibility_pilot(config)
    pr.disable()
    dt = time.perf_counter() - t0

    rows = {r["method_key"]: r for r in res["summary_rows"]}
    nh = rows.get("no_harm_flex", {})
    print(f"=== wall_seconds={dt:.2f} nodes_per_region={args.nodes_per_region} pods={args.pods} "
          f"max_timeslots={args.max_timeslots} lever={args.lever_mode} ===")
    print(f"repairs_applied={nh.get('repairs_applied')} no_harm={nh.get('no_harm_certificate')} "
          f"carbon_pct={nh.get('carbon_delta_pct'):.4f} scarcity_pct={nh.get('scarcity_delta_pct'):.4f} "
          f"stress_avoided={nh.get('stress_kwh_avoided'):.6f}")
    print()
    buf = StringIO()
    st = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    st.print_stats(args.top)
    print(buf.getvalue())
    buf2 = StringIO()
    st2 = pstats.Stats(pr, stream=buf2).sort_stats("tottime")
    st2.print_stats(args.top)
    print("=== by total (self) time ===")
    print(buf2.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
