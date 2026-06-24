#!/usr/bin/env python3
"""T3 — run the no-harm pilot on a real Azure Packing 2020 trace window.

Generates a regionful fleet (with per-site cooling kappa via the time-aligned WUE
table), points the no-harm pilot at a converted Azure window, and reports the
no-harm certificate, carbon/scarcity deltas, stress relief, and the flexible share
(firm/flexible tier derived from the trace priority by the converter).
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
from no_harm_flex_matrix import _generate_case_inputs, load_generator_dependencies  # noqa: E402

TIMEALIGNED = REPO_ROOT / "pkg" / "carbon-aware" / "data" / "timealigned"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="experiments/real_traces/azure_packing_2020_d0_s42_200",
                    help="converted Azure window dir (contains workloads/)")
    ap.add_argument("--nodes-per-region", type=int, default=8)
    ap.add_argument("--max-timeslots", type=int, default=48)
    ap.add_argument("--lever-mode", default="both")
    ap.add_argument("--scenario", default="heatwave-drought")
    ap.add_argument("--flex-slack-hours", type=float, default=2.0)
    ap.add_argument("--run-name", default="t3_azure")
    ap.add_argument("--fleet-seed", type=int, default=0)
    args = ap.parse_args()

    import logging
    logging.disable(logging.CRITICAL)

    window = (REPO_ROOT / args.window).resolve()
    workloads = window / "workloads"
    if not workloads.exists():
        raise SystemExit(f"no workloads/ under {window}")

    # Fleet: generate a regionful nodes.yaml (we ignore the generator's synthetic
    # workloads and point the pilot at the Azure window instead).
    config_file = REPO_ROOT / "pkg" / "carbon-aware" / "infra-workload-config.yaml"
    base_config, gen_nodes, gen_ts = load_generator_dependencies(REPO_ROOT, config_file)
    fleet_dir = REPO_ROOT / "experiments" / "flexibility" / args.run_name / "fleet"
    paths = _generate_case_inputs(
        base_config=base_config, generate_nodes_file=gen_nodes, generate_timeslot_files=gen_ts,
        input_dir=fleet_dir, pod_count=4, seed=args.fleet_seed, timeslots=12,
        config_file=config_file, deadline_flex_hours=24, nodes_per_region=args.nodes_per_region,
        server_only=True,
    )

    pc = PilotConfig(
        repo_root=REPO_ROOT, nodes_file=paths["nodes_file"], workloads_dir=workloads,
        forecasts_file=TIMEALIGNED / "forecasts.json", config_file=config_file,
        output_dir=REPO_ROOT / "experiments" / "flexibility" / args.run_name / "out",
        max_timeslots=args.max_timeslots, max_pods=None, scenario=args.scenario,
        flexibility_slack_hours=args.flex_slack_hours, lever_mode=args.lever_mode,
        grid_signal_csv=TIMEALIGNED / "grid_residual_region_slot.csv",
        wue_csv=TIMEALIGNED / "wue_region_slot.csv",
    )

    pods = load_pods(workloads)
    res = run_no_harm_flexibility_pilot(pc)
    rows = {r["method_key"]: r for r in res["summary_rows"]}
    flex = rows["no_harm_flex"]
    n_flex = flex["flexible_pods"]
    n_prot = flex["protected_pods"]
    total = n_flex + n_prot

    print(f"\n=== T3 Azure Packing 2020 no-harm result ({window.name}) ===")
    print(f"pods loaded={len(pods)}  placed={flex['placed_pods']}  unplaced={flex['unplaced_pods']}  "
          f"nodes/region={args.nodes_per_region} ({args.nodes_per_region*4} nodes)  max_timeslots={args.max_timeslots}")
    print(f"flexible share (priority tier): {n_flex}/{total} = {100*n_flex/max(total,1):.1f}%  (protected={n_prot})")
    print(f"no_harm flex: certificate={flex['no_harm_certificate']}  carbon%={flex['carbon_delta_pct']:.2f}  "
          f"scarcity%={flex['scarcity_delta_pct']:.3f}  stress_avoided={flex['stress_kwh_avoided']:.4f} kWh  "
          f"repairs={flex['repairs_applied']}")
    for m in ("carbon", "water_scarcity", "no_harm_search_control"):
        r = rows[m]
        print(f"  {m:>22}: certificate={r['no_harm_certificate']}  carbon%={r['carbon_delta_pct']:.2f}  "
              f"scarcity%={r['scarcity_delta_pct']:.3f}  stress_avoided={r['stress_kwh_avoided']:.4f}")
    print(f"\noutput_dir={pc.output_dir}\ncertificate={pc.output_dir/'no_harm_certificate.json'}")

    digest = {
        "window": window.name, "nodes": args.nodes_per_region * 4, "max_timeslots": args.max_timeslots,
        "pods_loaded": len(pods), "placed": flex["placed_pods"], "unplaced": flex["unplaced_pods"],
        "flexible_share_pct": round(100 * n_flex / max(total, 1), 2),
        "flexible_pods": int(n_flex), "protected_pods": int(n_prot),
        "no_harm_flex": {k: flex[k] for k in ("no_harm_certificate", "carbon_delta_pct", "scarcity_delta_pct",
                                              "stress_kwh_avoided", "repairs_applied", "rejected_moves")},
        "baselines": {m: {k: rows[m][k] for k in ("no_harm_certificate", "carbon_delta_pct", "scarcity_delta_pct")}
                      for m in ("carbon", "water_scarcity", "no_harm_search_control")},
    }
    (pc.output_dir / "t3_digest.json").write_text(json.dumps(digest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
