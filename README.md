# TotEm: Carbon-Aware Kubernetes-Style Scheduling Accounting for Embodied Emissions

TotEm (Total Emissions) is a carbon-aware scheduler for Kubernetes-style clusters that minimizes operational **and** embodied emissions using regional carbon-intensity forecasts. This repository packages the TotEm algorithm, a suite of carbon-aware and carbon-agnostic baselines, reproducible workloads, the real-trace pipeline, a real-control-plane validation harness, and the analysis pipeline used in our paper under `docs/paper-two/`.

![TotEm overview](docs/visTotEm.png)

## What's inside
- **TotEm (`--algorithm heuristic`)**: fast greedy scheduler that scores feasible pod–node–timeslot placements by operational plus embodied emissions; supports an exact operational-only ablation (`--operational-only`).
- **Baseline suite**, all run under one online-arrival, feasibility, and accounting contract:
  - `--algorithm vanilla` — Vanilla-MostAllocated, Kubernetes-like filter–score–select resource packing (carbon-blind).
  - `--algorithm green-mlfq` — GREEN-MLFQ-K8s, policy-level adaptation of GREEN (Xu et al., NSDI '25): rolling-epoch MLFQ with carbon-priority factors.
  - `--algorithm greencourier-spatial` — GreenCourier-style spatial clean-region placement (Chadha et al., 2024).
  - `--algorithm caspian-operational` — Caspian-style rolling spatio-temporal operational-carbon optimization (Bahreini et al., 2024).
  - `--algorithm global-optimal` — full-horizon MILP. Serves as the formal feasibility/accounting contract only; it requires future-arrival knowledge and is **not** an empirical baseline in the paper.
  - `--algorithm piontek-temporal`, `--algorithm wait-awhile` — additional temporal policies kept for reference.
- **Real-trace pipeline**: converter for the Azure Trace for Packing 2020 plus a windowed runner that evaluates all schedulers on trace-derived workloads.
- **Control-plane validation**: KWOK-based harness that replays schedules on a genuine Kubernetes control plane (API server, etcd, scheduler) and drives the released `kubectl-carbon` plugin and FogAtlas gRPC integration end to end.
- **Reproducible data**: infrastructure/workload generator, carbon-intensity forecasts, experiment runners, and the figure/statistics pipeline behind every number in the paper.

## Emissions & constraints model
- **Carbon signal**: hourly, region-indexed average carbon intensity (ACI) forecasts, shared by all carbon-aware algorithms.
- **Power**: two-point node model — idle baseline plus a dynamic component that scales linearly with the pod's CPU-request share; the first pod on an otherwise idle node-slot is attributed the idle draw.
- **Embodied carbon**: per-node grams CO₂e amortized uniformly over lifetime hours and attributed to pods by active time and CPU share (SCI-consistent, attributional); allocation modes `proportional` (default) or `uniform`.
- **Constraints enforced**: CPU/memory across each pod's full duration, earliest start (online arrivals), and deadlines.

## Repository layout
- `pkg/idl/idl.proto` — gRPC interface for the placement service (used by `kubectl-carbon` and FogAtlas).
- `pkg/carbon-aware/` — scheduler implementations and data: infra/workload generator, `nodes.yaml`, `workloads*/`, Python server (`server-python/`), Go client.
- `scripts/` — experiment runners: `run_resubmission_baseline_matrix.py` (density matrix), `run_capacity_sensitivity.py` (capacity sweep / scalability), `run_real_trace_baseline_matrix.py` + `real_traces/convert_azure_packing_trace.py` (Azure trace pipeline), `time_complexity_sweep.py`.
- `analysis/` — paper pipeline: `evaluation_composite_plot_generator.py` (all composite figures), `motivation_aci_figure.py` (Fig. 1), `workload_regime_sweep.py` (32-regime embodied-awareness sweep), `build_merged_matrix_n10.py` + `seed_extension_pilot_stats.py` (ten-seed cohorts and paired statistics), `kwok_validation.py` + `kwok_plugin_validate.py` (control-plane validation), `attributed_common_pod_comparison.py` (shared common-pod accounting), plus legacy plot utilities.
- `tests/` — constraint validators and batch experiment checks.
- `experiments/` — generated results (gitignored); the paper-backing runs are kept locally under the roots referenced by `analysis/evaluation_composite_plot_generator.py`.
- `docs/paper-two/` — paper source (`ResubmissionDraft/main.tex`), response letter (`FeedbackReviewers/`), and generated figures (`figures/CompositeEvaluation/`).

## Quickstart (offline, reproducible runs)

### Requirements
- Python 3.9+, Go 1.18+ (Go client only).
- Python dependencies: `pip install -r pkg/carbon-aware/server-python/requirements.txt` and, for plotting, `pip install matplotlib pandas seaborn`.

### Generate infrastructure & workloads
Configuration lives in `pkg/carbon-aware/infra-workload-config.yaml` (regions/hardware, workload scales, deadlines). Run from the generator directory so the config is picked up:

```bash
cd pkg/carbon-aware
python3 infra_workload_gen.py
cd ..
```

Outputs: `pkg/carbon-aware/nodes.yaml`, `pkg/carbon-aware/workloads/`, and `pkg/carbon-aware/workloads-vanilla/`.

### Run schedulers (precompute)
Pick an experiment directory and embodied allocation mode (`--embodied-mode proportional|uniform`).

**TotEm:**
```bash
python3 pkg/carbon-aware/server-python/main.py \
  --algorithm heuristic --precompute \
  --workloads-dir pkg/carbon-aware/workloads \
  --nodes-file pkg/carbon-aware/nodes.yaml \
  --forecasts-file pkg/carbon-aware/server-python/all_forecasts.json \
  --experiment-dir experiments/totem_demo \
  --embodied-mode proportional --loglevel INFO
```

**Any baseline** — swap `--algorithm` for `vanilla`, `green-mlfq`, `greencourier-spatial`, or `caspian-operational` (carbon-blind `vanilla` needs no forecasts file). All schedulers see only pods that have already arrived; none has future-workload knowledge.

Flags of note:
- `--operational-only` — exact TotEm-OpOnly ablation (drops the embodied term).
- `--deferral-margin / --deferral-budget / --embodied-gate` — experimental TotEm regime gates; defaults off reproduce TotEm bit-for-bit.

Outputs: placement CSVs (`*_placements_session.csv`) under the experiment directory, performance logs under `pkg/carbon-aware/server-python/performance_logs/`.

## Reproducing the paper
The evaluation pipeline, in order:

1. **Baseline matrix** (ten seeds × pod densities): `scripts/run_resubmission_baseline_matrix.py`
2. **Capacity sweep / paired scalability** (1×/2×/4×; 200→3,200 pods with 16→256 nodes): `scripts/run_capacity_sensitivity.py`
3. **32-regime embodied-awareness sweep + siting follow-up**: `analysis/workload_regime_sweep.py`
4. **Ten-seed merging and paired statistics** (all p-values in the paper): `analysis/build_merged_matrix_n10.py`, then `analysis/seed_extension_pilot_stats.py`
5. **Azure trace experiment**: `scripts/real_traces/convert_azure_packing_trace.py` (SQLite → workloads), then `scripts/run_real_trace_baseline_matrix.py` (five non-overlapping 12-hour arrival windows, 1× and 4×)
6. **Control-plane validation (KWOK)**: `analysis/kwok_validation.py` (feasibility/realizability on a real control plane) and `analysis/kwok_plugin_validate.py` (`kubectl-carbon` → gRPC → FogAtlas end to end)
7. **Figures**: `analysis/evaluation_composite_plot_generator.py` (composite Figs. 3–6) and `analysis/motivation_aci_figure.py` (Fig. 1); outputs land in `docs/paper-two/figures/CompositeEvaluation/current/`

Common-pod comparisons (same placed pod IDs across schedulers) and per-seed/per-window pairing are computed by `analysis/attributed_common_pod_comparison.py`, which all runners share.

## Validation & reporting
- `tests/check_timeslot_constraints.py <placements_csv>` validates earliest-timeslot, deadline, CPU, and memory constraints for a single run.
- `tests/validate_experiments_and_report.py --experiments-dir experiments ...` performs batch validation and writes CSV/TXT reports to `tests/reports/`.
- The KWOK harness independently confirms zero capacity violations and 100% admission/binding of placed pods through the real Kubernetes API.

## Data & units
- Embodied carbon values are **grams** CO₂e per node; lifetimes are in years (amortized to hours internally).
- Power values are in Watts; durations are hours; carbon intensity is gCO₂e/kWh.
- Default forecasts: `pkg/carbon-aware/server-python/all_forecasts.json` (24-hour horizon). Swap in region-matched data as needed.

## Paper & naming
The repository underpins `docs/paper-two/ResubmissionDraft/main.tex`: *TotEm: Carbon-Aware Kubernetes-Style Scheduling Accounting for Embodied Emissions* (under review, IEEE Transactions on Sustainable Computing). Plots and tables use the names TotEm, TotEm-OpOnly, Vanilla-MostAllocated, GREEN-MLFQ-K8s, GreenCourier-Spatial-K8s, and Caspian-style. The gRPC interface in `pkg/idl/idl.proto`, the `kubectl-carbon` plugin, and the FogAtlas scheduler integration connect TotEm to a real Kubernetes control plane.
