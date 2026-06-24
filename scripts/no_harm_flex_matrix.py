#!/usr/bin/env python3
"""Generate first evidence tables for the no-harm flexibility direction."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SERVER_PYTHON_ROOT = REPO_ROOT / "pkg" / "carbon-aware" / "server-python"
CARBON_AWARE_ROOT = REPO_ROOT / "pkg" / "carbon-aware"
for path in (SERVER_PYTHON_ROOT, CARBON_AWARE_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from carbon_aware.no_harm_flexibility import PilotConfig, run_no_harm_flexibility_pilot  # noqa: E402
from water_sweep_impl import load_generator_dependencies, parse_int_series  # noqa: E402


def parse_csv_series(value: str) -> List[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required")
    return values


def parse_float_series(value: str) -> List[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required")
    return values


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="no_harm_flex_first_evidence")
    parser.add_argument("--pod-counts", type=parse_int_series, default="40,80,120")
    parser.add_argument("--seeds", type=parse_int_series, default="44,45,46")
    parser.add_argument("--scenarios", type=parse_csv_series, default="observed-winter,heatwave-drought")
    parser.add_argument("--timeslots", type=int, default=12)
    parser.add_argument("--max-timeslots", type=int, default=24)
    parser.add_argument("--nodes-per-region", type=int, default=1, help="scale the fleet: nodes per region")
    parser.add_argument("--horizon-hours", type=parse_int_series, default="1", help="deferral-horizon sweep (slack hours per job)")
    parser.add_argument("--stress-quantile", type=float, default=0.75)
    parser.add_argument("--headroom-quantile", type=float, default=0.25)
    parser.add_argument("--flexibility-slack-hours", type=float, default=2.0)
    parser.add_argument("--drought-cf-threshold", type=float, default=20.0)
    parser.add_argument("--regret-margin", type=float, default=0.0)
    parser.add_argument("--regret-margins", type=parse_float_series, default=None, help="sweep regret margin (worst-case footprint buffer)")
    parser.add_argument("--forecast-noises", type=parse_float_series, default=None, help="RQ3: sweep carbon-forecast error stdev (decide on forecast, verify on realized)")
    parser.add_argument("--forecast-seed", type=int, default=0)
    parser.add_argument("--lever-modes", type=parse_csv_series, default="both")
    parser.add_argument("--config-file", default="pkg/carbon-aware/infra-workload-config.yaml")
    parser.add_argument("--forecasts-file", default="pkg/carbon-aware/server-python/all_forecasts.json")
    parser.add_argument("--grid-signal-csv", default="pkg/carbon-aware/data/grid/opsd_residual_load_region_slot.csv")
    parser.add_argument("--wue-csv", default=None, help="override WUE table (e.g. time-aligned real window)")
    parser.add_argument("--skip-generation", action="store_true")
    return parser.parse_args(argv)


def _generate_case_inputs(
    *,
    base_config: Dict[str, Any],
    generate_nodes_file,
    generate_timeslot_files,
    input_dir: Path,
    pod_count: int,
    seed: int,
    timeslots: int,
    config_file: Path,
    deadline_flex_hours: Optional[int] = None,
    nodes_per_region: int = 1,
    server_only: bool = False,
) -> Dict[str, Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    nodes_file = input_dir / "nodes.yaml"
    workloads_dir = input_dir / "workloads"
    vanilla_workloads_dir = input_dir / "workloads-vanilla"

    nodes_cfg = copy.deepcopy(base_config.get("nodes", {}))
    nodes_cfg["_base_dir"] = config_file.parent
    nodes_cfg["filename"] = str(nodes_file)
    nodes_cfg["random_seed"] = seed
    if server_only:
        # Paper-3 is a DATA-CENTER grid-flexibility study, and PUE/WUE are facility metrics:
        # they only make sense where a cooling facility exists. So the testbed is multi-region
        # DC servers (the edge classes IoT/Smartphone/Laptop are dropped), while per-region
        # cooling kappa / PUE / WUE diversity is preserved. See cooling_physics_note.md.
        _regions = list(nodes_cfg.get("regions", {}).get("region_counts", {}).keys())
        nodes_cfg["explicit_assignments"] = {r: {"Server": 1} for r in _regions}
        nodes_cfg["hardware_assignment"] = {"hardware_counts": {"Server": 1}}
        nodes_cfg["hardware_assignment_method"] = "exact_counts"
    if nodes_per_region and nodes_per_region > 1:
        # Scale the fleet: M nodes per region (same region->hardware mapping).
        m = int(nodes_per_region)
        region_counts = nodes_cfg.get("regions", {}).get("region_counts", {})
        for region in region_counts:
            region_counts[region] *= m
        for region, hw in nodes_cfg.get("explicit_assignments", {}).items():
            for h in hw:
                hw[h] *= m
        hw_counts = nodes_cfg.get("hardware_assignment", {}).get("hardware_counts", {})
        for h in hw_counts:
            hw_counts[h] *= m

    workload_cfg = copy.deepcopy(base_config.get("workload", {}))
    workload_cfg["_base_dir"] = config_file.parent
    workload_cfg.update(
        {
            "output_dir": str(workloads_dir),
            "vanilla_output_dir": str(vanilla_workloads_dir),
            "generation_strategy": "exact_total",
            "exact_total_pods": int(pod_count),
            "num_timeslots": int(timeslots),
            "random_seed": int(seed),
        }
    )
    if deadline_flex_hours is not None:
        # Sweep the deferral horizon: every job gets exactly this much slack.
        workload_cfg["deadline_flexibility_hours"] = [int(deadline_flex_hours)]

    generate_nodes_file(nodes_cfg)
    generate_timeslot_files(workload_cfg)
    return {
        "nodes_file": nodes_file,
        "workloads_dir": workloads_dir,
        "vanilla_workloads_dir": vanilla_workloads_dir,
    }


def _summarize_method(df: pd.DataFrame, method_key: str) -> Dict[str, float]:
    subset = df[df["method_key"] == method_key]
    return {
        "n": float(len(subset)),
        "no_harm_rate": float(subset["no_harm_certificate"].mean()),
        "stress_kwh_avoided_mean": float(subset["stress_kwh_avoided"].mean()),
        "stress_kwh_avoided_median": float(subset["stress_kwh_avoided"].median()),
        "weighted_stress_kwh_mean": float(subset["weighted_stress_kwh"].mean()),
        "carbon_delta_pct_mean": float(subset["carbon_delta_pct"].mean()),
        "scarcity_delta_pct_mean": float(subset["scarcity_delta_pct"].mean()),
        "repairs_applied_mean": float(subset["repairs_applied"].mean()),
    }


def _write_grouped_summary(aggregate: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (scenario, lever_mode, method), group in aggregate.groupby(["scenario", "lever_mode", "method_key"], sort=True):
        row = {"scenario": scenario, "lever_mode": lever_mode, "method_key": method}
        row.update(_summarize_method(group, method))
        rows.append(row)
    grouped = pd.DataFrame(rows)
    grouped.to_csv(output_path, index=False)
    return grouped


def _write_report(aggregate: pd.DataFrame, grouped: pd.DataFrame, output_path: Path) -> None:
    flex = aggregate[aggregate["method_key"] == "no_harm_flex"]
    control = aggregate[aggregate["method_key"] == "no_harm_search_control"]
    carbon = aggregate[aggregate["method_key"] == "carbon"]
    water = aggregate[aggregate["method_key"] == "water_scarcity"]

    merged = flex.merge(
        control[["case_id", "scenario", "stress_kwh_avoided", "weighted_stress_kwh"]],
        on=["case_id", "scenario"],
        suffixes=("_flex", "_control"),
    )
    flex_beats_control = float((merged["weighted_stress_kwh_flex"] < merged["weighted_stress_kwh_control"]).mean())

    try:
        grouped_table = grouped.to_markdown(index=False)
    except ImportError:
        grouped_table = "```\n" + grouped.to_string(index=False) + "\n```"

    lever_rows: List[Dict[str, Any]] = []
    for lever_mode, grp in flex.groupby("lever_mode", sort=True):
        lever_rows.append(
            {
                "lever_mode": lever_mode,
                "no_harm_rate": round(float(grp["no_harm_certificate"].mean()), 3),
                "stress_kwh_avoided_mean": round(float(grp["stress_kwh_avoided"].mean()), 4),
                "carbon_delta_pct_mean": round(float(grp["carbon_delta_pct"].mean()), 2),
                "scarcity_delta_pct_mean": round(float(grp["scarcity_delta_pct"].mean()), 2),
                "repairs_applied_mean": round(float(grp["repairs_applied"].mean()), 1),
            }
        )
    lever_df = pd.DataFrame(lever_rows)
    try:
        lever_table = lever_df.to_markdown(index=False)
    except ImportError:
        lever_table = "```\n" + lever_df.to_string(index=False) + "\n```"

    flex_both = flex[flex["lever_mode"] == "both"] if "lever_mode" in flex.columns else flex
    horizon_rows: List[Dict[str, Any]] = []
    if "horizon_hours" in flex_both.columns:
        for hz, grp in flex_both.groupby("horizon_hours", sort=True):
            horizon_rows.append(
                {
                    "horizon_hours": int(hz),
                    "no_harm_rate": round(float(grp["no_harm_certificate"].mean()), 3),
                    "stress_kwh_avoided_mean": round(float(grp["stress_kwh_avoided"].mean()), 4),
                    "carbon_delta_pct_mean": round(float(grp["carbon_delta_pct"].mean()), 2),
                    "scarcity_delta_pct_mean": round(float(grp["scarcity_delta_pct"].mean()), 2),
                    "repairs_applied_mean": round(float(grp["repairs_applied"].mean()), 1),
                }
            )
    horizon_df = pd.DataFrame(horizon_rows)
    try:
        horizon_table = horizon_df.to_markdown(index=False)
    except ImportError:
        horizon_table = "```\n" + horizon_df.to_string(index=False) + "\n```"

    regret_rows: List[Dict[str, Any]] = []
    if "regret_margin" in flex_both.columns and flex_both["regret_margin"].nunique() > 1:
        for rm, grp in flex_both.groupby("regret_margin", sort=True):
            regret_rows.append(
                {
                    "regret_margin": float(rm),
                    "no_harm_rate": round(float(grp["no_harm_certificate"].mean()), 3),
                    "stress_kwh_avoided_mean": round(float(grp["stress_kwh_avoided"].mean()), 4),
                    "carbon_delta_pct_mean": round(float(grp["carbon_delta_pct"].mean()), 2),
                    "scarcity_delta_pct_mean": round(float(grp["scarcity_delta_pct"].mean()), 2),
                    "repairs_applied_mean": round(float(grp["repairs_applied"].mean()), 1),
                }
            )
    regret_df = pd.DataFrame(regret_rows)
    try:
        regret_table = regret_df.to_markdown(index=False) if regret_rows else "(single regret margin; sweep with --regret-margins)"
    except ImportError:
        regret_table = "```\n" + regret_df.to_string(index=False) + "\n```"

    forecast_rows: List[Dict[str, Any]] = []
    if "forecast_noise" in flex_both.columns and flex_both["forecast_noise"].nunique() > 1:
        for (fn, rm), grp in flex_both.groupby(["forecast_noise", "regret_margin"], sort=True):
            forecast_rows.append(
                {
                    "forecast_noise": float(fn),
                    "regret_margin": float(rm),
                    "no_harm_rate": round(float(grp["no_harm_certificate"].mean()), 3),
                    "stress_kwh_avoided_mean": round(float(grp["stress_kwh_avoided"].mean()), 4),
                    "carbon_delta_pct_mean": round(float(grp["carbon_delta_pct"].mean()), 2),
                    "scarcity_delta_pct_mean": round(float(grp["scarcity_delta_pct"].mean()), 2),
                }
            )
    forecast_df = pd.DataFrame(forecast_rows)
    try:
        forecast_table = forecast_df.to_markdown(index=False) if forecast_rows else "(single forecast noise; sweep with --forecast-noises)"
    except ImportError:
        forecast_table = "```\n" + forecast_df.to_string(index=False) + "\n```"

    if "grid_stress_share" in aggregate.columns:
        rates = aggregate.groupby("scenario")[["grid_stress_share", "drought_share"]].mean().reset_index()
        base_rate_line = "; ".join(
            f"{sc}: grid-stress {g * 100:.0f}% of region-slots, drought {d * 100:.0f}%"
            for sc, g, d in rates.itertuples(index=False, name=None)
        )
    else:
        base_rate_line = "n/a"

    lines = [
        "# No-Harm Flexibility First Evidence",
        "",
        "This is a first evidence package for the Paper 3 pivot. It is suitable for",
        "deciding whether the direction is worth developing, not yet for final claims.",
        "",
        "## Design",
        "",
        f"- Cases: {aggregate['case_id'].nunique()} workload/node inputs.",
        f"- Scenarios: {', '.join(sorted(aggregate['scenario'].unique()))}.",
        "- Grid signal: OPSD residual-load-lite (load minus wind and solar) where available.",
        "- Accounting: checked-in carbon forecast, WUE/EWIF, and AWARE-derived scarcity factors.",
        "- Baselines: packing, carbon-only, water-scarcity-only, same-budget search-control, no-harm flex.",
        "",
        "## Main Result",
        "",
        f"- No-harm flex certificates passed in {flex['no_harm_certificate'].mean() * 100:.1f}% of runs.",
        f"- Mean stress-kWh avoided by no-harm flex: {flex['stress_kwh_avoided'].mean():.6f}.",
        f"- Mean carbon change vs packing: {flex['carbon_delta_pct'].mean():.2f}%.",
        f"- Mean scarcity-water change vs packing: {flex['scarcity_delta_pct'].mean():.2f}%.",
        f"- No-harm flex had lower weighted stress exposure than same-budget search-control in {flex_beats_control * 100:.1f}% of matched runs.",
        "",
        "## Failure-Mode Evidence",
        "",
        f"- Carbon-only failed no-harm in {(1.0 - carbon['no_harm_certificate'].mean()) * 100:.1f}% of runs.",
        f"- Water-only failed no-harm in {(1.0 - water['no_harm_certificate'].mean()) * 100:.1f}% of runs.",
        "These failures are the core motivation for the no-harm certificate: single-objective",
        "optimizers often improve their own metric while violating the other account.",
        "",
        "## Lever Ablation (no-harm flex: which channel carries the envelope)",
        "",
        lever_table,
        "",
        "## Horizon Sweep (no-harm flex, both levers; RQ1 envelope vs deferral horizon)",
        "",
        horizon_table,
        "",
        "## Regret-Margin Sweep (no-harm flex, both levers; RQ3 regret-bounded action)",
        "",
        regret_table,
        "",
        "## Forecast-Error Sweep (no-harm flex, both levers; RQ3 decide-on-forecast, verify-on-realized)",
        "",
        forecast_table,
        "",
        "## Stress Base-Rates (how often the guardrails bind)",
        "",
        f"- {base_rate_line}",
        "",
        "## Grouped Means",
        "",
        grouped_table,
        "",
        "## Publication Cautions",
        "",
        "- OPSD residual-load-lite omits run-of-river because it is not consistently present in the selected OPSD file.",
        "- Carbon and water accounts are replayed from local reference tables and are not time-synchronized to the 2019 OPSD grid window.",
        "- The current cooling stress scenario is a high-WUE/AWARE July sensitivity replay, not observed site telemetry.",
        "- This is evidence that the mechanism is worth pursuing; the next paper-grade step is fully time-aligned ENTSO-E/EDO/Open-Meteo replay.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_figures(aggregate: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    methods = ["carbon", "water_scarcity", "no_harm_search_control", "no_harm_flex"]
    labels = ["Carbon", "Water", "Search control", "No-harm flex"]
    plot_df = aggregate[aggregate["method_key"].isin(methods)]

    stress = plot_df.groupby("method_key")["stress_kwh_avoided"].mean().reindex(methods)
    plt.figure(figsize=(7, 4))
    plt.bar(labels, stress.values, color=["#4B5563", "#0E7490", "#A16207", "#176B45"])
    plt.ylabel("Mean stress-kWh avoided vs packing")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "stress_avoided_by_method.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    for method, label in zip(methods, labels):
        subset = plot_df[plot_df["method_key"] == method]
        plt.scatter(subset["carbon_delta_pct"], subset["scarcity_delta_pct"], label=label, alpha=0.8)
    plt.axhline(0, color="#111827", linewidth=0.8)
    plt.axvline(0, color="#111827", linewidth=0.8)
    plt.xlabel("Carbon change vs packing (%)")
    plt.ylabel("Scarcity-water change vs packing (%)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "carbon_scarcity_tradeoff.png", dpi=180)
    plt.close()


def run_matrix(args: argparse.Namespace) -> Path:
    run_root = REPO_ROOT / "experiments" / "flexibility" / args.run_name
    input_root = run_root / "inputs"
    case_root = run_root / "cases"
    run_root.mkdir(parents=True, exist_ok=True)

    config_file = REPO_ROOT / args.config_file
    forecasts_file = REPO_ROOT / args.forecasts_file
    grid_signal_csv = REPO_ROOT / args.grid_signal_csv if args.grid_signal_csv else None
    base_config, generate_nodes_file, generate_timeslot_files = load_generator_dependencies(REPO_ROOT, config_file)

    all_rows: List[Dict[str, Any]] = []
    regret_list = args.regret_margins if args.regret_margins else [args.regret_margin]
    forecast_list = args.forecast_noises if args.forecast_noises else [0.0]
    for pod_count in args.pod_counts:
        for seed in args.seeds:
            for horizon in args.horizon_hours:
                case_id = f"{pod_count}pods_s{seed}_h{horizon}"
                paths = {
                    "nodes_file": input_root / case_id / "nodes.yaml",
                    "workloads_dir": input_root / case_id / "workloads",
                }
                if not args.skip_generation:
                    paths = _generate_case_inputs(
                        base_config=base_config,
                        generate_nodes_file=generate_nodes_file,
                        generate_timeslot_files=generate_timeslot_files,
                        input_dir=input_root / case_id,
                        pod_count=pod_count,
                        seed=seed,
                        timeslots=args.timeslots,
                        config_file=config_file,
                        deadline_flex_hours=horizon,
                        nodes_per_region=args.nodes_per_region,
                    )

                for scenario in args.scenarios:
                    for lever_mode in args.lever_modes:
                        for regret in regret_list:
                            for fnoise in forecast_list:
                                output_dir = case_root / case_id / scenario / lever_mode / f"r{regret}" / f"n{fnoise}"
                                result = run_no_harm_flexibility_pilot(
                                    PilotConfig(
                                        repo_root=REPO_ROOT,
                                        nodes_file=paths["nodes_file"],
                                        workloads_dir=paths["workloads_dir"],
                                        forecasts_file=forecasts_file,
                                        config_file=config_file,
                                        output_dir=output_dir,
                                        max_timeslots=args.max_timeslots,
                                        max_pods=None,
                                        flexibility_slack_hours=args.flexibility_slack_hours,
                                        stress_quantile=args.stress_quantile,
                                        headroom_quantile=args.headroom_quantile,
                                        drought_cf_threshold=args.drought_cf_threshold,
                                        scenario=scenario,
                                        regret_margin=regret,
                                        grid_signal_csv=grid_signal_csv,
                                        lever_mode=lever_mode,
                                        wue_csv=(REPO_ROOT / args.wue_csv) if args.wue_csv else None,
                                        forecast_noise=fnoise,
                                        forecast_seed=args.forecast_seed,
                                    )
                                )
                                base = result.get("signal_base_rates", {})
                                for row in result["summary_rows"]:
                                    enriched = dict(row)
                                    enriched.update(
                                        {
                                            "case_id": case_id,
                                            "pod_count": pod_count,
                                            "seed": seed,
                                            "horizon_hours": horizon,
                                            "scenario": scenario,
                                            "lever_mode": lever_mode,
                                            "regret_margin": regret,
                                            "forecast_noise": fnoise,
                                            "grid_stress_share": base.get("grid_stress_share"),
                                            "drought_share": base.get("drought_share"),
                                            "case_output_dir": str(output_dir.relative_to(REPO_ROOT)),
                                        }
                                    )
                                    all_rows.append(enriched)

    aggregate = pd.DataFrame(all_rows)
    aggregate_path = run_root / "aggregate_summary.csv"
    aggregate.to_csv(aggregate_path, index=False)
    grouped = _write_grouped_summary(aggregate, run_root / "grouped_summary.csv")
    _write_report(aggregate, grouped, run_root / "evidence_report.md")
    _write_figures(aggregate, run_root)

    metadata = {
        "run_name": args.run_name,
        "pod_counts": args.pod_counts,
        "seeds": args.seeds,
        "scenarios": args.scenarios,
        "timeslots": args.timeslots,
        "max_timeslots": args.max_timeslots,
        "stress_quantile": args.stress_quantile,
        "headroom_quantile": args.headroom_quantile,
        "grid_signal_csv": str(grid_signal_csv.relative_to(REPO_ROOT)) if grid_signal_csv else None,
        "row_count": len(aggregate),
    }
    (run_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return run_root


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_root = run_matrix(args)
    print(f"output_dir={run_root}")
    print(f"aggregate_summary={run_root / 'aggregate_summary.csv'}")
    print(f"evidence_report={run_root / 'evidence_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
