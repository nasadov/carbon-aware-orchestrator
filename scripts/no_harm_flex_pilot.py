#!/usr/bin/env python3
"""Run the no-harm flexibility-envelope pilot."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SERVER_PYTHON_ROOT = REPO_ROOT / "pkg" / "carbon-aware" / "server-python"
if str(SERVER_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_PYTHON_ROOT))

from carbon_aware.no_harm_flexibility import PilotConfig, run_no_harm_flexibility_pilot  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="no_harm_flex_pilot")
    parser.add_argument("--nodes-file", default="pkg/carbon-aware/nodes.yaml")
    parser.add_argument("--workloads-dir", default="pkg/carbon-aware/workloads")
    parser.add_argument("--forecasts-file", default="pkg/carbon-aware/server-python/all_forecasts.json")
    parser.add_argument("--config-file", default="pkg/carbon-aware/infra-workload-config.yaml")
    parser.add_argument("--max-timeslots", type=int, default=24)
    parser.add_argument("--max-pods", type=int, default=80)
    parser.add_argument("--flexibility-slack-hours", type=float, default=2.0)
    parser.add_argument("--stress-quantile", type=float, default=0.75)
    parser.add_argument("--headroom-quantile", type=float, default=0.25)
    parser.add_argument("--drought-cf-threshold", type=float, default=20.0)
    parser.add_argument("--scenario", choices=["observed-winter", "heatwave-drought"], default="heatwave-drought")
    parser.add_argument("--regret-margin", type=float, default=0.0)
    parser.add_argument("--max-repairs", type=int, default=None)
    parser.add_argument("--grid-signal-csv", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s:%(message)s",
    )
    output_dir = REPO_ROOT / "experiments" / "flexibility" / args.run_name
    config = PilotConfig(
        repo_root=REPO_ROOT,
        nodes_file=REPO_ROOT / args.nodes_file,
        workloads_dir=REPO_ROOT / args.workloads_dir,
        forecasts_file=REPO_ROOT / args.forecasts_file,
        config_file=REPO_ROOT / args.config_file,
        output_dir=output_dir,
        max_timeslots=args.max_timeslots,
        max_pods=args.max_pods,
        flexibility_slack_hours=args.flexibility_slack_hours,
        stress_quantile=args.stress_quantile,
        headroom_quantile=args.headroom_quantile,
        drought_cf_threshold=args.drought_cf_threshold,
        scenario=args.scenario,
        regret_margin=args.regret_margin,
        max_repairs=args.max_repairs,
        grid_signal_csv=REPO_ROOT / args.grid_signal_csv if args.grid_signal_csv else None,
    )
    result = run_no_harm_flexibility_pilot(config)
    print(f"output_dir={result['output_dir']}")
    print("summary_csv=" + str(output_dir / "summary.csv"))
    print("certificate_json=" + str(output_dir / "no_harm_certificate.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
