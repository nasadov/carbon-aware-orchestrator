# TotEm: Carbon-Aware Kubernetes Scheduling with Embodied Emissions

TotEm (Total Emissions) is a Kubernetes scheduler and experiment harness for
lifecycle-aware placement under time-varying regional environmental signals. The
repository packages a carbon-aware heuristic scheduler, an Oracle MILP
benchmark, a carbon-agnostic baseline, reproducible workload generation, and
analysis utilities for scheduler evaluation.

![TotEm overview](docs/visTotEm.png)

## What's Inside

- **TotEm (`--algorithm heuristic`)**: fast carbon-aware scheduler for multi-timeslot workloads.
- **Oracle (`--algorithm global-optimal`)**: lexicographic two-phase MILP optimizer for benchmark comparisons.
- **Carbon-agnostic baseline (`--algorithm vanilla`)**: Kubernetes-like LeastAllocated scoring with node sampling and early stopping.
- **Environmental metadata**: optional carbon and water footprint signals used by supported scheduling modes and placement summaries.
- **Reproducible data**: infrastructure/workload generator, carbon-intensity traces, and experiment harnesses.
- **Interfaces**: gRPC IDL plus a Go client prototype for Kubernetes integration.
- **Analysis**: figure generators for emissions, runtime, success rate, utilization, and sensitivity studies.

## Environmental Model & Constraints

- **Carbon signal**: time- and region-indexed carbon-intensity forecasts shared across algorithms.
- **Co-location-aware power**: \(P_\text{node} = P_\text{idle} + (P_\text{max} - P_\text{active}) \times U\), where \(U\) is aggregate CPU utilization; dynamic cost per pod scales linearly with its CPU share.
- **Embodied carbon**: grams of CO2e per node amortized over lifetime hours; allocation modes are `proportional` (default) and `uniform`. Use `--operational-only` to drop embodied emissions.
- **Water signals**: optional direct, indirect, embodied, raw, and scarcity-characterized water fields can be loaded from configured reference tables and written to enriched placement CSVs.
- **Constraints enforced**: CPU and memory across each pod's duration, earliest start from `timeslot_<X>.yaml`, deadlines, and optional prioritization of emissions-per-CPU vs totals.

## Repository Layout

- `pkg/idl/idl.proto` — gRPC interface for the placement service.
- `pkg/carbon-aware/` — scheduler implementation and data, including the generator, `nodes.yaml`, generated workloads, Python server, and Go client.
- `pkg/carbon-aware/data/water/` — optional water reference tables and preprocessing utilities.
- `analysis/` — plotting, diagnostics, and sensitivity utilities; shared defaults live in `analysis/repo_paths.py`.
- `scripts/` — runnable sweep and maintenance entry points such as `water_sweep.py`, `time_complexity_sweep.py`, `sweep_podcounts_and_precompute.sh`, and `update_config.py`.
- `scripts/archive/` — retained legacy helpers that are not part of the normal workflow.
- `tests/` — smoke tests, repository hygiene checks, placement validators, and batch experiment checks.
- `experiments/` — generated run outputs (gitignored); figures go under `experiments/figures/`.
- `docs/`, `media/`, `external/`, `workloads*` — supporting documentation, data, and generated workloads.

## Quickstart

### Requirements

- Python 3.9+ and Go 1.18+.
- Python dependencies:

```bash
pip install -r pkg/carbon-aware/server-python/requirements.txt
pip install matplotlib pandas seaborn
```

Optional development checks:

```bash
pip install -r requirements-dev.txt
make syntax-check
make lint
make test
```

### Generate Infrastructure & Workloads

Configuration lives in `pkg/carbon-aware/infra-workload-config.yaml`.
Run from the generator directory so the config is picked up:

```bash
cd pkg/carbon-aware
python3 infra_workload_gen.py
cd ../..
```

Outputs:

- `pkg/carbon-aware/nodes.yaml`
- `pkg/carbon-aware/workloads/` for TotEm and Oracle runs
- `pkg/carbon-aware/workloads-vanilla/` for baseline runs

The generator supports Poisson or exact-total pod counts, fixed seeds, and
explicit region/hardware counts.

### Run Schedulers

Pick an experiment directory and embodied allocation mode
(`--embodied-mode proportional|uniform`).

**TotEm heuristic:**

```bash
python3 pkg/carbon-aware/server-python/main.py \
  --algorithm heuristic --precompute \
  --workloads-dir pkg/carbon-aware/workloads \
  --nodes-file pkg/carbon-aware/nodes.yaml \
  --forecasts-file pkg/carbon-aware/server-python/all_forecasts.json \
  --experiment-dir experiments/totem_demo \
  --embodied-mode proportional \
  --loglevel INFO
```

**TotEm with scalarized carbon-water scoring:**

```bash
python3 pkg/carbon-aware/server-python/main.py \
  --algorithm heuristic --precompute \
  --workloads-dir pkg/carbon-aware/workloads \
  --nodes-file pkg/carbon-aware/nodes.yaml \
  --forecasts-file pkg/carbon-aware/server-python/all_forecasts.json \
  --experiment-dir experiments/totem_weighted_demo \
  --embodied-mode proportional \
  --heuristic-objective weighted-sum \
  --heuristic-carbon-weight 0.50 \
  --loglevel INFO
```

**Oracle MILP benchmark:**

```bash
python3 pkg/carbon-aware/server-python/main.py \
  --algorithm global-optimal --precompute \
  --workloads-dir pkg/carbon-aware/workloads \
  --nodes-file pkg/carbon-aware/nodes.yaml \
  --forecasts-file pkg/carbon-aware/server-python/all_forecasts.json \
  --experiment-dir experiments/oracle_demo \
  --embodied-mode proportional \
  --loglevel INFO
```

**Oracle with an environmental budget constraint:**

```bash
python3 pkg/carbon-aware/server-python/main.py \
  --algorithm global-optimal --precompute \
  --workloads-dir pkg/carbon-aware/workloads \
  --nodes-file pkg/carbon-aware/nodes.yaml \
  --forecasts-file pkg/carbon-aware/server-python/all_forecasts.json \
  --experiment-dir experiments/oracle_budget_demo \
  --embodied-mode proportional \
  --global-water-budget 500 \
  --global-water-metric scarcity \
  --loglevel INFO
```

**Carbon-agnostic baseline:**

```bash
python3 pkg/carbon-aware/server-python/main.py \
  --algorithm vanilla --precompute \
  --workloads-dir pkg/carbon-aware/workloads-vanilla \
  --nodes-file pkg/carbon-aware/nodes.yaml \
  --experiment-dir experiments/vanilla_demo \
  --loglevel INFO
```

Useful flags:

- `--operational-only` ignores embodied emissions for TotEm and Oracle.
- `--prioritize-efficiency` optimizes emissions-per-CPU instead of total emissions.
- `--heuristic-objective carbon|weighted-sum|pareto|epsilon-pareto` selects the heuristic scoring mode.
- `--heuristic-carbon-weight <0..1>` sets the carbon share for weighted scalar scoring.
- `--heuristic-water-budget <float>` sets the budget used by `epsilon-pareto`.
- `--heuristic-water-metric raw|scarcity` selects the water metric used by heuristic budget modes.
- `--global-water-budget <float>` enables the Oracle budget constraint.
- `--global-water-metric raw|scarcity` selects the Oracle budget metric.
- `--global-objective carbon|waterwise-scalarized` selects the Oracle phase-2 objective.
- `--global-scalarized-carbon-weight <0..1>` sets the carbon share for the scalarized Oracle objective.
- Solver/time-limit/gap settings live under `optimization` in `pkg/carbon-aware/infra-workload-config.yaml`.

Outputs:

- Placement CSVs `*_placements_session.csv` under the chosen experiment directory.
- Performance logs under `pkg/carbon-aware/server-python/performance_logs/`.
- Timestamped subfolders within `experiments/`.

### Run Carbon-Water Sweeps

Use `scripts/water_sweep.py` for reproducible carbon-water experiment matrices.
It reuses the workload generator and scheduler precompute paths, writes a
summary CSV, component breakdown, frontier diagnostics, plots, and provenance
metadata for each run.

Small smoke matrix:

```bash
scripts/run_water_checkpoint.sh
```

Larger study matrix:

```bash
scripts/run_water_main.sh
```

Direct dry run without executing schedulers:

```bash
python3 scripts/water_sweep.py \
  --dry-run \
  --pod-counts 34 \
  --seeds 44 \
  --methods heuristic-carbon
```

Sweep outputs are written under `experiments/water/<run_name>/`, with figures
under `experiments/figures/`. Each run writes `metadata.json` with git,
environment, config, forecast, water-reference, and generated-input hashes.

## Algorithm Details

- **TotEm heuristic**: greedy scheduler that scores feasible pod-node-timeslot pairs by operational and embodied footprint cost, orders pods by tightest window then size, scans low-carbon windows first, and uses a packing-aware tie-breaker. Supported scoring modes include carbon-only, weighted scalar scoring, Pareto-style scoring, and budget-guided repair.
- **Oracle MILP benchmark**: two-phase lexicographic objective that first maximizes placed pods, then optimizes the selected phase-2 objective. Supported backends include `cp_sat`, `highs`, `gurobi`, `cplex`, and `cbc`, configured in YAML.
- **Carbon-agnostic baseline**: Kubernetes-like Filter -> Score -> Select flow with LeastAllocated-style scoring, node sampling, and early stop; respects earliest timeslot and CPU/RAM/duration constraints but ignores carbon signals and embodied costs.

## Analysis & Visualization

Run from the repository root unless noted:

- `analysis/carbon_emissions_comparison.py`: end-to-end emissions and runtime summaries.
- `analysis/emissions_vs_pods_plot_generator.py`: emissions vs implied average cluster CPU utilization.
- `analysis/emissions_vs_pods_common_pods_plot_generator.py`: common-pod emissions comparison.
- `analysis/success_rate_plot_generator.py`: placement success vs load.
- `analysis/time_complexity_plot_generator.py`: runtime scaling plots.
- `analysis/time_complexity_fixed_axes_plot.py`: fixed-axis runtime overlays.
- `analysis/time_complexity_overlay_slices.py`: runtime overlay slices.
- `analysis/utilization_plot_generator.py`: CPU/memory utilization across experiments.
- `analysis/schedule_visualization.py`: per-timeslot placement heatmaps from `*_placements_session.csv`.
- `analysis/forecast_error_sensitivity.py`: forecast-error sensitivity.
- `analysis/embodied_sensitivity.py`: embodied-emissions sensitivity.
- `analysis/simple_carbon_heatmap.py`: quick carbon heatmaps.
- Additional water-footprint plotting and diagnostics utilities are available under `analysis/`.

Figures are written to `experiments/figures/`. Sweep artifacts and logs stay
under `experiments/`.

## Validation & Reporting

- `make syntax-check`, `make lint`, and `make test` run the configured Python verification gate.
- `tests/placement_constraint_validator.py <placements_csv>` validates earliest-timeslot, deadline, CPU, and memory constraints for a single run.
- `tests/validate_experiments_and_report.py --experiments-dir experiments --nodes-file pkg/carbon-aware/nodes.yaml --workloads-dir pkg/carbon-aware/workloads --workloads-vanilla-dir pkg/carbon-aware/workloads-vanilla` performs batch validation and writes CSV/TXT reports to `tests/reports/`.
- `analysis/fix_start_slots.py` can normalize vanilla placements when needed.

## Data & Units

- Embodied carbon values are expressed in grams CO2e per node; lifetimes are in years and amortized to hours during calculations.
- Power values are in Watts.
- Durations are in hours.
- Carbon intensity is gCO2e/kWh.
- Direct, indirect, and embodied water values are written in liters where water metadata is configured.
- Scarcity-characterized water uses the configured characterization factors.
- Default forecasts: `pkg/carbon-aware/server-python/all_forecasts.json`.
