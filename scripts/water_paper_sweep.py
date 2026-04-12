#!/usr/bin/env python3
"""Reusable carbon-water experiment runner for paper-scale study matrices.

The runner reuses the existing workload generator and precompute entry points,
then adds the water-aware method matrix needed for the new experiments:
carbon-only heuristic, weighted carbon-water heuristic, Pareto heuristic,
epsilon-guided Pareto heuristic points, carbon MILP, and epsilon-constrained
MILP points.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from time_complexity_sweep import ProgressTracker, parse_int_series  # noqa: E402

try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


@dataclass(frozen=True)
class StudyCase:
    """Generated input bundle for one pod-count/seed combination."""

    label: str
    pod_count: int
    seed: int
    input_dir: Path
    workloads_dir: Path
    vanilla_workloads_dir: Path
    nodes_file: Path


@dataclass(frozen=True)
class MethodSpec:
    """One scheduler/method variant to run on a generated input bundle."""

    key: str
    label: str
    group: str
    algorithm: str
    heuristic_objective: str = "carbon"
    heuristic_carbon_weight: float = 1.0
    water_metric: str = "scarcity"
    water_budget: Optional[float] = None


@dataclass
class RunResult:
    """Flat result row written to the sweep summary CSV."""

    run_name: str
    case_label: str
    pod_count: int
    seed: int
    method_key: str
    method_label: str
    method_group: str
    algorithm: str
    status: str
    elapsed_seconds: float
    session_dir: str
    placement_csv: str
    performance_csv: str
    error: str = ""
    water_metric: str = "scarcity"
    water_budget: Optional[float] = None
    placed_pods: Optional[int] = None
    rows: Optional[int] = None
    carbon_kg: Optional[float] = None
    operational_carbon_kg: Optional[float] = None
    embodied_carbon_kg: Optional[float] = None
    direct_water_l: Optional[float] = None
    indirect_water_l: Optional[float] = None
    embodied_water_l: Optional[float] = None
    raw_water_l: Optional[float] = None
    scarcity_water: Optional[float] = None
    criticality_adjusted_water: Optional[float] = None
    node_distribution: str = ""
    region_distribution: str = ""


def parse_float_series(series: str) -> List[float]:
    values: List[float] = []
    for chunk in series.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.append(float(chunk))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Could not parse float from '{chunk}'") from exc
    if not values:
        raise argparse.ArgumentTypeError("At least one float value is required")
    return values


def parse_method_series(series: str) -> List[str]:
    values = [chunk.strip() for chunk in series.split(",") if chunk.strip()]
    allowed = {
        "heuristic-carbon",
        "heuristic-weighted",
        "heuristic-pareto",
        "heuristic-epsilon-pareto",
        "milp-carbon",
        "milp-epsilon",
        "vanilla",
    }
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown method(s): {', '.join(unknown)}")
    if not values:
        raise argparse.ArgumentTypeError("At least one method is required")
    return values


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def add_repo_modules_to_path(root: Path) -> Path:
    carbon_dir = root / "pkg" / "carbon-aware"
    server_python_dir = carbon_dir / "server-python"
    for path in (carbon_dir, server_python_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return server_python_dir


@contextlib.contextmanager
def temporary_env(name: str, value: str):
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def load_generator_dependencies(root: Path, config_file: Path):
    add_repo_modules_to_path(root)
    from infra_workload_gen import generate_nodes_file, generate_timeslot_files, load_config  # type: ignore

    return load_config(str(config_file)), generate_nodes_file, generate_timeslot_files


def build_cases(run_root: Path, pod_counts: Sequence[int], seeds: Sequence[int]) -> List[StudyCase]:
    cases: List[StudyCase] = []
    for pod_count in pod_counts:
        for seed in seeds:
            label = f"{pod_count}pods_s{seed}"
            input_dir = run_root / "inputs" / label
            cases.append(
                StudyCase(
                    label=label,
                    pod_count=pod_count,
                    seed=seed,
                    input_dir=input_dir,
                    workloads_dir=input_dir / "workloads",
                    vanilla_workloads_dir=input_dir / "workloads-vanilla",
                    nodes_file=input_dir / "nodes.yaml",
                )
            )
    return cases


def generate_inputs(
    case: StudyCase,
    base_config: Dict[str, Any],
    generate_nodes_file,
    generate_timeslot_files,
    *,
    config_file: Path,
    timeslots: int,
) -> None:
    """Generate nodes/workloads for a case without mutating the checked-in YAML."""
    case.input_dir.mkdir(parents=True, exist_ok=True)

    nodes_cfg = copy.deepcopy(base_config.get("nodes", {}))
    nodes_cfg["_base_dir"] = config_file.parent
    nodes_cfg["filename"] = str(case.nodes_file)
    nodes_cfg["random_seed"] = case.seed

    workload_cfg = copy.deepcopy(base_config.get("workload", {}))
    workload_cfg["_base_dir"] = config_file.parent
    workload_cfg.update(
        {
            "output_dir": str(case.workloads_dir),
            "vanilla_output_dir": str(case.vanilla_workloads_dir),
            "generation_strategy": "exact_total",
            "exact_total_pods": int(case.pod_count),
            "num_timeslots": int(timeslots),
            "random_seed": int(case.seed),
        }
    )

    generate_nodes_file(nodes_cfg)
    generate_timeslot_files(workload_cfg)


def base_methods(method_names: Sequence[str], weighted_carbon_weight: float) -> List[MethodSpec]:
    specs: List[MethodSpec] = []
    for name in method_names:
        if name == "heuristic-carbon":
            specs.append(
                MethodSpec(
                    key="heuristic_carbon",
                    label="Heuristic carbon",
                    group="heuristic",
                    algorithm="heuristic",
                )
            )
        elif name == "heuristic-weighted":
            specs.append(
                MethodSpec(
                    key=f"heuristic_wsum{int(round(weighted_carbon_weight * 100)):02d}",
                    label=f"Heuristic weighted {weighted_carbon_weight:.2f}",
                    group="heuristic",
                    algorithm="heuristic",
                    heuristic_objective="weighted-sum",
                    heuristic_carbon_weight=weighted_carbon_weight,
                )
            )
        elif name == "heuristic-pareto":
            specs.append(
                MethodSpec(
                    key="heuristic_pareto",
                    label="Heuristic Pareto",
                    group="heuristic",
                    algorithm="heuristic",
                    heuristic_objective="pareto",
                )
            )
        elif name == "milp-carbon":
            specs.append(
                MethodSpec(
                    key="milp_carbon",
                    label="MILP carbon",
                    group="milp",
                    algorithm="global-optimal",
                )
            )
        elif name == "vanilla":
            specs.append(
                MethodSpec(
                    key="vanilla",
                    label="Vanilla",
                    group="baseline",
                    algorithm="vanilla",
                )
            )
    return specs


def epsilon_methods(
    baseline_water: Optional[float],
    fractions: Sequence[float],
    water_metric: str,
    *,
    include_milp: bool = True,
    include_heuristic: bool = False,
) -> List[MethodSpec]:
    if baseline_water is None or baseline_water <= 0:
        return []

    specs: List[MethodSpec] = []
    for fraction in fractions:
        budget = baseline_water * float(fraction)
        fraction_key = str(fraction).replace(".", "p")
        if include_milp:
            specs.append(
                MethodSpec(
                    key=f"milp_eps_{water_metric}_{fraction_key}",
                    label=f"MILP eps {fraction:.2f}",
                    group="milp",
                    algorithm="global-optimal",
                    water_metric=water_metric,
                    water_budget=budget,
                )
            )
        if include_heuristic:
            specs.append(
                MethodSpec(
                    key=f"heuristic_epspareto_{water_metric}_{fraction_key}",
                    label=f"Heuristic eps Pareto {fraction:.2f}",
                    group="heuristic",
                    algorithm="heuristic",
                    heuristic_objective="epsilon-pareto",
                    water_metric=water_metric,
                    water_budget=budget,
                )
            )
    return specs


def placement_csv_for(session_dir: Path, method: MethodSpec) -> Optional[Path]:
    if method.algorithm == "global-optimal":
        candidate = session_dir / "global_optimal_placements_session.csv"
        return candidate if candidate.exists() else None
    if method.algorithm == "vanilla":
        candidate = session_dir / "vanilla_placements_session.csv"
        return candidate if candidate.exists() else None

    matches = sorted(session_dir.glob("*placements_session.csv"))
    return matches[0] if matches else None


def performance_csv_for(session_dir: Path, method: MethodSpec) -> Path:
    return session_dir / f"{method.key}_performance.csv"


def summarize_placement_csv(csv_path: Optional[Path]) -> Dict[str, Any]:
    if csv_path is None or not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path)
    if df.empty:
        return {"rows": 0, "placed_pods": 0}

    def sum_column(name: str) -> Optional[float]:
        return float(df[name].fillna(0).sum()) if name in df.columns else None

    def distribution(name: str) -> str:
        if name not in df.columns:
            return ""
        counts = df[name].fillna("").astype(str).value_counts().sort_index()
        return "; ".join(f"{idx}={count}" for idx, count in counts.items() if idx)

    return {
        "placed_pods": int(df["pod_id"].nunique()) if "pod_id" in df.columns else int(len(df)),
        "rows": int(len(df)),
        "carbon_kg": sum_column("total_carbon_emissions"),
        "operational_carbon_kg": sum_column("operational_carbon_kg"),
        "embodied_carbon_kg": sum_column("embodied_carbon_kg"),
        "direct_water_l": sum_column("direct_water_l"),
        "indirect_water_l": sum_column("indirect_water_l"),
        "embodied_water_l": sum_column("embodied_water_l"),
        "raw_water_l": sum_column("total_raw_water_l"),
        "scarcity_water": sum_column("scarcity_characterized_water"),
        "criticality_adjusted_water": sum_column("criticality_adjusted_water"),
        "node_distribution": distribution("node_id"),
        "region_distribution": distribution("region"),
    }


def run_method(
    case: StudyCase,
    method: MethodSpec,
    *,
    run_name: str,
    run_root: Path,
    forecasts_file: Path,
    config_file: Path,
    embodied_mode: str,
    operational_only: bool,
) -> RunResult:
    add_repo_modules_to_path(repo_root())
    from carbon_aware.precompute_global_optimal import run_global_optimal_precomputation  # type: ignore
    from carbon_aware.precompute_heuristic import run_heuristic_precomputation  # type: ignore
    from carbon_aware.precompute_vanilla import run_vanilla_precomputation  # type: ignore
    from carbon_aware.utils import PerformanceLogger  # type: ignore

    session_dir = run_root / "runs" / case.label / method.key
    session_dir.mkdir(parents=True, exist_ok=True)

    perf_path = performance_csv_for(session_dir, method)
    perf_logger = PerformanceLogger(method.algorithm)
    perf_logger.set_new_log_file(str(session_dir), perf_path.name)

    start = time.perf_counter()
    error = ""
    success = False

    try:
        with temporary_env("CARBON_AWARE_CONFIG_PATH", str(config_file)):
            if method.algorithm == "heuristic":
                success = run_heuristic_precomputation(
                    workloads_dir=str(case.workloads_dir),
                    nodes_file=str(case.nodes_file),
                    forecasts_file=str(forecasts_file),
                    session_log_dir=str(session_dir),
                    perf_logger=perf_logger,
                    operational_only=operational_only,
                    embodied_mode=embodied_mode,
                    heuristic_objective=method.heuristic_objective,
                    heuristic_carbon_weight=method.heuristic_carbon_weight,
                    heuristic_water_budget=method.water_budget,
                    heuristic_water_metric=method.water_metric,
                )
            elif method.algorithm == "global-optimal":
                success = run_global_optimal_precomputation(
                    workloads_dir=str(case.workloads_dir),
                    nodes_file=str(case.nodes_file),
                    forecasts_file=str(forecasts_file),
                    session_log_dir=str(session_dir),
                    perf_logger=perf_logger,
                    operational_only=operational_only,
                    embodied_mode=embodied_mode,
                    global_water_budget=method.water_budget,
                    global_water_metric=method.water_metric,
                )
            elif method.algorithm == "vanilla":
                success = run_vanilla_precomputation(
                    workloads_dir=str(case.vanilla_workloads_dir),
                    nodes_file=str(case.nodes_file),
                    forecasts_file=str(forecasts_file),
                    session_log_dir=str(session_dir),
                    perf_logger=perf_logger,
                    operational_only=operational_only,
                )
            else:  # pragma: no cover - parser prevents this
                raise ValueError(f"Unsupported algorithm: {method.algorithm}")
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        logging.exception("Method %s failed for case %s", method.key, case.label)
        error = str(exc)

    elapsed = time.perf_counter() - start
    placement_csv = placement_csv_for(session_dir, method)
    summary = summarize_placement_csv(placement_csv)

    return RunResult(
        run_name=run_name,
        case_label=case.label,
        pod_count=case.pod_count,
        seed=case.seed,
        method_key=method.key,
        method_label=method.label,
        method_group=method.group,
        algorithm=method.algorithm,
        status="success" if success else "failed",
        elapsed_seconds=elapsed,
        session_dir=str(session_dir),
        placement_csv=str(placement_csv or ""),
        performance_csv=str(perf_path if perf_path.exists() else ""),
        error=error,
        water_metric=method.water_metric,
        water_budget=method.water_budget,
        **summary,
    )


def write_summary_csv(results: Sequence[RunResult], output_path: Path) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([result.__dict__ for result in results])
    df.to_csv(output_path, index=False)
    return df


def write_breakdown_csv(results_df: pd.DataFrame, output_path: Path) -> None:
    if results_df.empty:
        return
    water_columns = ["direct_water_l", "indirect_water_l", "embodied_water_l"]
    if any(column not in results_df.columns for column in water_columns):
        return

    rows = []
    for _, row in results_df.iterrows():
        for column in water_columns:
            value = row.get(column)
            rows.append(
                {
                    "run_name": row.get("run_name"),
                    "case_label": row.get("case_label"),
                    "method_key": row.get("method_key"),
                    "method_label": row.get("method_label"),
                    "component": column.replace("_water_l", ""),
                    "water_l": value,
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def create_case_frontier_plot(
    root: Path,
    case_label: str,
    results_df: pd.DataFrame,
    figures_dir: Path,
    run_name: str,
) -> Optional[Path]:
    case_df = results_df[
        (results_df["case_label"] == case_label)
        & (results_df["status"] == "success")
        & (results_df["placement_csv"].fillna("") != "")
        & (results_df["method_group"].isin(["heuristic", "milp"]))
    ].copy()
    if case_df.empty:
        return None

    output_path = figures_dir / f"{run_name}_{case_label}_carbon_scarcity_frontier.png"
    cmd = [sys.executable, str(root / "analysis" / "water_frontier_comparison_plot.py")]
    for _, row in case_df.iterrows():
        cmd.extend(
            [
                "--run",
                f"{row['method_label']}:{row['method_group']}:{row['placement_csv']}",
            ]
        )
    cmd.extend(
        [
            "--output",
            str(output_path),
            "--title",
            f"{case_label}: Carbon-Scarcity Frontier (Raw Water = Marker Size)",
        ]
    )
    subprocess.run(cmd, cwd=root, check=True)
    return output_path


def create_frontier_gap_diagnostics(root: Path, summary_csv: Path, water_metric: str) -> Path:
    output_path = summary_csv.with_name("water_frontier_gap_diagnostics.csv")
    subprocess.run(
        [
            sys.executable,
            str(root / "analysis" / "water_frontier_gap_diagnostics.py"),
            str(summary_csv),
            "--water-metric",
            water_metric,
            "--output",
            str(output_path),
        ],
        cwd=root,
        check=True,
    )
    return output_path


def default_pod_counts_for_preset(preset: str) -> List[int]:
    if preset == "paper":
        return [34, 100, 200]
    return [34]


def default_seeds_for_preset(preset: str) -> List[int]:
    if preset == "paper":
        return [42, 43, 44]
    return [44]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=["checkpoint", "paper"],
        default="checkpoint",
        help="Default matrix size. checkpoint is intentionally small; paper expands pod counts and seeds.",
    )
    parser.add_argument("--pod-counts", type=parse_int_series, default=None)
    parser.add_argument("--seeds", type=parse_int_series, default=None)
    parser.add_argument("--timeslots", type=int, default=12)
    parser.add_argument(
        "--methods",
        type=parse_method_series,
        default="heuristic-carbon,heuristic-weighted,heuristic-pareto,heuristic-epsilon-pareto,milp-carbon,milp-epsilon",
        help="Comma-separated method list.",
    )
    parser.add_argument("--weighted-carbon-weight", type=float, default=0.50)
    parser.add_argument("--epsilon-fractions", type=parse_float_series, default="0.95,0.90,0.85")
    parser.add_argument("--epsilon-water-metric", choices=["scarcity", "raw"], default="scarcity")
    parser.add_argument("--skip-milp", action="store_true", help="Skip MILP carbon and epsilon runs.")
    parser.add_argument("--operational-only", action="store_true")
    parser.add_argument("--embodied-mode", choices=["proportional", "uniform"], default="proportional")
    parser.add_argument(
        "--config-file",
        type=Path,
        default=repo_root() / "pkg" / "carbon-aware" / "infra-workload-config.yaml",
    )
    parser.add_argument(
        "--forecasts-file",
        type=Path,
        default=repo_root() / "pkg" / "carbon-aware" / "server-python" / "all_forecasts.json",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/water_paper"))
    parser.add_argument("--figures-dir", type=Path, default=Path("experiments/figures"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    add_repo_modules_to_path(root)

    pod_counts = args.pod_counts or default_pod_counts_for_preset(args.preset)
    seeds = args.seeds or default_seeds_for_preset(args.preset)
    methods = list(args.methods)
    needs_budgeted_methods = "milp-epsilon" in methods or "heuristic-epsilon-pareto" in methods
    if args.skip_milp:
        methods = [method for method in methods if not method.startswith("milp-")]
        if "heuristic-epsilon-pareto" in methods and "heuristic-carbon" not in methods:
            methods.insert(0, "heuristic-carbon")
    elif needs_budgeted_methods and "milp-carbon" not in methods:
        methods.insert(0, "milp-carbon")

    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    figures_dir = args.figures_dir if args.figures_dir.is_absolute() else root / args.figures_dir

    run_name = args.run_name or datetime.now(timezone.utc).strftime("water_paper_%Y%m%d_%H%M%S")
    run_root = output_dir / run_name

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")

    config_file = args.config_file.resolve()
    forecasts_file = args.forecasts_file.resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    if not forecasts_file.exists():
        raise FileNotFoundError(f"Forecasts file not found: {forecasts_file}")

    base_config, generate_nodes_file, generate_timeslot_files = load_generator_dependencies(root, config_file)
    cases = build_cases(run_root, pod_counts, seeds)
    non_epsilon_methods = base_methods(
        [method for method in methods if method not in ("milp-epsilon", "heuristic-epsilon-pareto")],
        weighted_carbon_weight=max(0.0, min(1.0, args.weighted_carbon_weight)),
    )

    planned_methods = [method.key for method in non_epsilon_methods]
    if "milp-epsilon" in methods:
        planned_methods.extend([f"milp-epsilon@{fraction:.2f}" for fraction in args.epsilon_fractions])
    if "heuristic-epsilon-pareto" in methods:
        planned_methods.extend([f"heuristic-epsilon-pareto@{fraction:.2f}" for fraction in args.epsilon_fractions])

    metadata = {
        "run_name": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/water_paper_sweep.py",
        "preset": args.preset,
        "pod_counts": pod_counts,
        "seeds": seeds,
        "timeslots": int(args.timeslots),
        "methods": methods,
        "planned_methods": planned_methods,
        "epsilon_fractions": list(args.epsilon_fractions),
        "epsilon_water_metric": args.epsilon_water_metric,
        "weighted_carbon_weight": float(args.weighted_carbon_weight),
        "operational_only": bool(args.operational_only),
        "embodied_mode": args.embodied_mode,
        "config_file": str(config_file),
        "forecasts_file": str(forecasts_file),
        "output_dir": str(run_root),
        "figures_dir": str(figures_dir),
    }

    if args.dry_run:
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return 0

    if run_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Run directory already exists: {run_root}. Use --overwrite to replace it.")
        import shutil

        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=False)
    figures_dir.mkdir(parents=True, exist_ok=True)
    (run_root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    total_steps = len(cases) * max(len(non_epsilon_methods), 1)
    if "milp-epsilon" in methods and not args.skip_milp:
        total_steps += len(cases) * len(args.epsilon_fractions)
    if "heuristic-epsilon-pareto" in methods:
        total_steps += len(cases) * len(args.epsilon_fractions)

    if tqdm is not None:
        progress = tqdm(total=total_steps, desc="water-paper sweep", unit="run")
    else:
        progress = ProgressTracker(total_steps)

    results: List[RunResult] = []
    for case in cases:
        logging.info("Generating inputs for %s", case.label)
        generate_inputs(
            case,
            base_config,
            generate_nodes_file,
            generate_timeslot_files,
            config_file=config_file,
            timeslots=args.timeslots,
        )

        baseline_water: Optional[float] = None
        heuristic_carbon_baseline_water: Optional[float] = None
        for method in non_epsilon_methods:
            logging.info("Running %s on %s", method.key, case.label)
            result = run_method(
                case,
                method,
                run_name=run_name,
                run_root=run_root,
                forecasts_file=forecasts_file,
                config_file=config_file,
                embodied_mode=args.embodied_mode,
                operational_only=args.operational_only,
            )
            results.append(result)
            if method.key == "milp_carbon" and result.status == "success":
                baseline_water = (
                    result.raw_water_l
                    if args.epsilon_water_metric == "raw"
                    else result.scarcity_water
                )
            if method.key == "heuristic_carbon" and result.status == "success":
                heuristic_carbon_baseline_water = (
                    result.raw_water_l
                    if args.epsilon_water_metric == "raw"
                    else result.scarcity_water
                )

            if tqdm is not None:
                progress.update(1)
                progress.set_postfix_str(f"{case.label}:{method.key}:{result.status}")
            else:
                progress.update(status=f"{case.label}:{method.key}:{result.status}")

        if baseline_water is None and heuristic_carbon_baseline_water is not None:
            baseline_water = heuristic_carbon_baseline_water

        if ("milp-epsilon" in methods and not args.skip_milp) or "heuristic-epsilon-pareto" in methods:
            for method in epsilon_methods(
                baseline_water,
                args.epsilon_fractions,
                args.epsilon_water_metric,
                include_milp="milp-epsilon" in methods and not args.skip_milp,
                include_heuristic="heuristic-epsilon-pareto" in methods,
            ):
                logging.info(
                    "Running %s on %s with %s budget %.6f",
                    method.key,
                    case.label,
                    method.water_metric,
                    method.water_budget or 0.0,
                )
                result = run_method(
                    case,
                    method,
                    run_name=run_name,
                    run_root=run_root,
                    forecasts_file=forecasts_file,
                    config_file=config_file,
                    embodied_mode=args.embodied_mode,
                    operational_only=args.operational_only,
                )
                results.append(result)
                if tqdm is not None:
                    progress.update(1)
                    progress.set_postfix_str(f"{case.label}:{method.key}:{result.status}")
                else:
                    progress.update(status=f"{case.label}:{method.key}:{result.status}")

    progress.close()

    summary_csv = run_root / "water_paper_sweep_summary.csv"
    results_df = write_summary_csv(results, summary_csv)
    write_breakdown_csv(results_df, run_root / "water_component_breakdown.csv")
    diagnostics_csv = create_frontier_gap_diagnostics(root, summary_csv, args.epsilon_water_metric)

    generated_figures: List[str] = []
    if not args.no_plots:
        for case in cases:
            plot_path = create_case_frontier_plot(root, case.label, results_df, figures_dir, run_name)
            if plot_path:
                generated_figures.append(str(plot_path))

    metadata["generated_figures"] = generated_figures
    metadata["summary_csv"] = str(summary_csv)
    metadata["breakdown_csv"] = str(run_root / "water_component_breakdown.csv")
    metadata["diagnostics_csv"] = str(diagnostics_csv)
    (run_root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    logging.info("Water-paper sweep complete: %s", summary_csv)
    for figure in generated_figures:
        logging.info("Figure: %s", figure)

    failed = results_df[results_df["status"] != "success"] if not results_df.empty else pd.DataFrame()
    return 1 if not failed.empty else 0


if __name__ == "__main__":
    sys.exit(main())
