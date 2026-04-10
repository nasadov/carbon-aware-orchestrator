# TotEm: Carbon-Aware Kubernetes Scheduling with Embodied Emissions

TotEm (Total Emissions) is a Kubernetes scheduler and experiment harness for lifecycle-aware placement under time-varying regional signals. The repository packages the original carbon-aware heuristic, an Oracle MILP benchmark, a carbon-agnostic baseline, reproducible workloads, and a newer water-aware extension that adds direct water, indirect water, embodied water, and scarcity-characterized water accounting. The current heuristic supports both carbon-only scoring and a weighted carbon-water baseline.

![TotEm overview](docs/visTotEm.png)

## What’s inside
- **TotEm (heuristic, `--algorithm heuristic`)**: fast carbon-aware scheduler for multi-timeslot workloads.
- **Oracle (MILP benchmark, `--algorithm global-optimal`)**: lexicographic two-phase optimizer for lower-bound comparisons.
- **Carbon-agnostic baseline (`--algorithm vanilla`)**: Kubernetes-like LeastAllocated scoring with node sampling and early stopping.
- **Water-aware extension**: dataset-backed water metadata loader, placement-level footprint vectors, and weighted carbon-water heuristic scoring.
- **Reproducible data**: infrastructure/workload generator, carbon-intensity traces, and experiment harnesses.
- **Interfaces**: gRPC IDL plus a Go client prototype for Kubernetes integration.
- **Analysis**: figure generators for emissions, runtime, success rate, utilization, sensitivity studies, and current water trade-off checkpoint plots.

## Environmental model & constraints
- **Carbon signal**: time- and region-indexed carbon-intensity forecasts (shared across algorithms).
- **Co-location-aware power**: \(P_\text{node} = P_\text{idle} + (P_\text{max} - P_\text{active}) \times U\) where \(U\) is aggregate CPU utilization; dynamic cost per pod scales linearly with its CPU share.
- **Embodied carbon**: grams of CO₂e per node amortized over lifetime hours; allocation modes `proportional` (default) or `uniform`. Toggle `--operational-only` to drop embodied emissions.
- **Water signals**:
  - direct water from region/slot WUE tables
  - indirect water from country-level EWIF factors
  - scarcity characterization from AWARE2.0 country factors
  - provisional embodied water by hardware class
- **Water outputs**: direct water, indirect water, embodied water, total raw water, and scarcity-characterized water are written to enriched placement CSVs for the implemented water-aware paths.
- **Constraints enforced**: CPU/Memory across each pod’s duration, earliest start from `timeslot_<X>.yaml`, deadlines, and optional prioritization of emissions-per-CPU vs totals.

## Repository layout
- `pkg/idl/idl.proto` — gRPC interface for the placement service.
- `pkg/carbon-aware/` — scheduler implementation and data (infra/workload generator, `nodes.yaml`, `workloads*/`, Python server, Go client).
- `pkg/carbon-aware/data/water/` — water reference tables, preprocessing scripts, and source notes for the current water-aware implementation.
- `analysis/` — figure generators: `carbon_emissions_comparison.py`, `emissions_vs_pods*_plot_generator.py`, `success_rate_plot_generator.py`, `time_complexity_*`, `utilization_plot_generator.py`, `schedule_visualization.py`, `forecast_error_sensitivity.py`, `embodied_sensitivity.py`, `simple_carbon_heatmap.py`, and utilities.
- `scripts/` — sweeps/time-complexity helpers (`sweep_podcounts_and_precompute.sh`, `time_complexity_sweep.py`, `update_config.py`).
- `tests/` — constraint validators and batch experiment checks.
- `experiments/` & `figures/` — generated results (gitignored).
- `docs/paper-two/` — paper source.
- `media/`, `external/`, `workloads*` — supporting data and generated workloads.

## Quickstart (offline, reproducible runs)

### Requirements
- Python 3.9+, Go 1.18+.
- Python dependencies: `pip install -r pkg/carbon-aware/server-python/requirements.txt` and, for plotting, `pip install matplotlib pandas seaborn`.

### Generate infrastructure & workloads
Configuration lives in `pkg/carbon-aware/infra-workload-config.yaml` (regions/hardware, workload scales, deadlines, solver defaults). Run from the generator directory so the config is picked up:

```bash
cd pkg/carbon-aware
python3 infra_workload_gen.py
cd ..
```

Outputs:
- `pkg/carbon-aware/nodes.yaml`
- `pkg/carbon-aware/workloads/` (TotEm/Oracle)
- `pkg/carbon-aware/workloads-vanilla/` (baseline)

The generator supports Poisson or exact-total pod counts, fixed seeds, and explicit region/hardware counts.

### Run schedulers (precompute)
Pick an experiment directory and embodied allocation mode (`--embodied-mode proportional|uniform`).

**TotEm (heuristic):**
```bash
python3 pkg/carbon-aware/server-python/main.py \
  --algorithm heuristic --precompute \
  --workloads-dir pkg/carbon-aware/workloads \
  --nodes-file pkg/carbon-aware/nodes.yaml \
  --forecasts-file pkg/carbon-aware/server-python/all_forecasts.json \
  --experiment-dir experiments/totem_demo \
  --embodied-mode proportional --loglevel INFO
```

**Weighted carbon-water heuristic baseline:**
```bash
python3 pkg/carbon-aware/server-python/main.py \
  --algorithm heuristic --precompute \
  --workloads-dir pkg/carbon-aware/workloads \
  --nodes-file pkg/carbon-aware/nodes.yaml \
  --forecasts-file pkg/carbon-aware/server-python/all_forecasts.json \
  --experiment-dir experiments/totem_water_demo \
  --embodied-mode proportional \
  --heuristic-objective weighted-sum \
  --heuristic-carbon-weight 0.50 \
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

**Oracle (MILP benchmark, lexicographic two-phase by default via `infra-workload-config.yaml`):**
```bash
python3 pkg/carbon-aware/server-python/main.py \
  --algorithm global-optimal --precompute \
  --workloads-dir pkg/carbon-aware/workloads \
  --nodes-file pkg/carbon-aware/nodes.yaml \
  --forecasts-file pkg/carbon-aware/server-python/all_forecasts.json \
  --experiment-dir experiments/oracle_demo \
  --embodied-mode proportional --loglevel INFO
```

Flags of note:
- `--operational-only` to ignore embodied emissions (TotEm/Oracle).
- `--prioritize-efficiency` to optimize emissions-per-CPU instead of totals (TotEm).
- `--heuristic-objective carbon|weighted-sum` to switch between carbon-only and weighted carbon-water heuristic scoring.
- `--heuristic-carbon-weight <0..1>` to set the carbon share in weighted-sum mode.
- Solver/time-limit/gap settings live under `optimization` in `pkg/carbon-aware/infra-workload-config.yaml` (cp_sat/highs/gurobi/cplex/cbc).

Outputs:
- Placement CSVs `*_placements_session.csv` under the chosen experiment directory.
- Performance logs under `pkg/carbon-aware/server-python/performance_logs/`.
- Timestamped subfolders within `experiments/…`.

## Algorithm details
- **TotEm (heuristic)**: Greedy scheduler that scores feasible pod–node–timeslot pairs on operational+embodied environmental cost, orders pods by tightest window then size, scans low-carbon windows first, and uses a packing-aware tie-breaker. Implemented scoring modes are `carbon` and `weighted-sum` (carbon + water). Placement outputs include carbon and water breakdown columns when water metadata is available.
- **Oracle (MILP benchmark)**: Two-phase lexicographic objective (maximize placed pods, then minimize emissions) with warm-starts and a dynamic-only fallback when activation binaries time out. Backends: cp_sat, highs, gurobi, cplex, cbc (set in the YAML). Current optimization objective remains carbon-focused.
- **Carbon-agnostic baseline**: K8s-like Filter → Score → Select with LeastAllocated-style scoring, node sampling, and early stop; respects earliest timeslot and CPU/RAM/duration constraints but ignores carbon signals and embodied costs.

## Analysis & visualization
Run from repo root unless noted:
- `analysis/carbon_emissions_comparison.py`: end-to-end emissions + runtime summaries for TotEm vs Oracle vs carbon-agnostic.
- `analysis/emissions_vs_pods_plot_generator.py` and `analysis/emissions_vs_pods_common_pods_plot_generator.py`: emissions vs implied average cluster CPU utilization (proportional/uniform variants).
- `analysis/success_rate_plot_generator.py`: placement success vs load.
- `analysis/time_complexity_plot_generator.py`, `analysis/time_complexity_fixed_axes_plot.py`, `analysis/time_complexity_overlay_slices.py`: runtime scaling overlays; feed with data from `scripts/time_complexity_sweep.py` or `scripts/sweep_podcounts_and_precompute.sh`.
- `analysis/utilization_plot_generator.py`: CPU/memory utilization across experiments.
- `analysis/schedule_visualization.py`: per-timeslot placement heatmaps from `*_placements_session.csv`.
- Sensitivity: `analysis/forecast_error_sensitivity.py`, `analysis/embodied_sensitivity.py`; quick visuals: `analysis/simple_carbon_heatmap.py`.
- Water-aware checkpoint: `analysis/water_tradeoff_checkpoint_plot.py`.

Figures are written to `figures/` (timestamped folders). Sweep artifacts and logs stay under `experiments/`.

## Validation & reporting
- `tests/check_timeslot_constraints.py <placements_csv>` validates earliest-timeslot, deadline, CPU, and memory constraints for a single run.
- `tests/validate_experiments_and_report.py --experiments-dir experiments --nodes-file pkg/carbon-aware/nodes.yaml --workloads-dir pkg/carbon-aware/workloads --workloads-vanilla-dir pkg/carbon-aware/workloads-vanilla` performs batch validation and writes CSV/TXT reports to `tests/reports/`.
- `analysis/fix_start_slots.py` can normalize vanilla placements when needed.

## Data & units
- Embodied carbon values are expressed in **grams** CO₂e per node; lifetimes are in years (amortized to hours internally).
- Power values are in Watts; durations are hours; carbon intensity is gCO₂e/kWh.
- Direct, indirect, and embodied water are currently written in **liters**; scarcity-characterized water is stored as a scaled impact value from the configured AWARE2.0 factors.
- Default forecasts: `pkg/carbon-aware/server-python/all_forecasts.json` (12–24h horizons). Swap in region-matched data as needed.
- Water reference tables live under `pkg/carbon-aware/data/water/`.

## Current water-aware status
- Implemented:
  - water metadata loading
  - placement-level footprint vectors
  - enriched placement CSVs and summaries
  - weighted carbon-water heuristic baseline
- Not implemented yet:
  - Pareto-aware heuristic
  - epsilon-constraint MILP frontier
  - literature-grade embodied-water inventory
