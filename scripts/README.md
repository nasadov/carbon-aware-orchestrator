# Scripts

Runnable experiment and maintenance entry points for the Python research codebase.

## Top-level entry points
- `water_sweep.py`: public carbon-water sweep entry point (pinned by `tests/test_repo_hygiene.py`).
- `time_complexity_sweep.py`: runtime-scaling experiment runner.
- `update_config.py`: small YAML update helper used by sweep wrappers.
- `check_python_syntax.py`: import-free syntax check used by `make syntax-check`.

## No-Harm Flexibility Envelope (paper-three) — current testbed
Two coherent real-trace testbeds that share the engine, time-aligned signals, regions (DE/FR/ES/IT-NO),
and PilotConfig — differing only in workload class and binding resource. Both drive DYNAMIC power from
(measured or imputed) utilization, not request=100%, and cap deferral slack at 24 h:
- `alibaba_common.py`: shared **Alibaba GPU** v2020 testbed builder (trace+fleet+PilotConfig); measured
  per-pod utilization from the sensor table; GPU-bound DGX-1 V100 fleet.
- `azure_common.py`: shared **Azure Packing 2020** testbed builder (CPU/RAM general cloud); realistic
  32-core fleet provisioned to a target utilization; imputes the trace-population mean CPU utilization
  (Packing 2020 is an allocation trace with no per-VM telemetry — same imputation rule as Alibaba's
  sensor-missing tasks). Mirrors `alibaba_common.py`.
- `alibaba_fleet_scale.py` / `alibaba_lever_ablation.py` / `alibaba_waterwise_contrast.py`: the headline
  sweeps on the real Alibaba GPU trace.
- `signal_fidelity_carbon.py` / `signal_fidelity_basin.py`: carbon-proxy / basin-scarcity verification.
- `run_alibaba_no_harm.py`, `run_azure_no_harm.py`: canonical real-trace no-harm pilots (the Azure
  runner rebuilds the window from its `canonical_trace_workload.csv` and provisions the realistic fleet).
- `no_harm_flex_pilot.py`, `no_harm_flex_matrix.py`, `milp_optimality_gap.py`,
  `forecast_regret_sweep.py`, `slo_binding_sensitivity.py`, `azure_alibaba_contrast.py`: supporting runs.
- `build_timealigned_signals.py`: builds the time-aligned EU signals (fixed-κ + T-PUE cooling, proxy or
  real generation-mix CI). This is the cooling/CI model the paper uses (Path A).
- `real_traces/`: trace converters (Alibaba GPU v2020, Azure Packing 2020).

## Subdirectories
- `shell/`: convenience shell wrappers (`run_water_*.sh`, `sweep_podcounts_and_precompute.sh`) around
  the Python sweep entry points.
- `archive/`: retained legacy helpers not in the normal workflow — the synthetic-generator sweeps
  (`fleet_scale_sweep.py`, `lever_ablation.py`, `waterwise_baseline_contrast.py`) superseded by the
  Alibaba (`alibaba_*`) sweeps, and the first Alibaba converter (`convert_alibaba_gpu_trace.py`)
  superseded by `real_traces/convert_alibaba_gpu_v2020.py` (measured-utilization + censoring-aware).

## Conventions
- Generated outputs go under `experiments/`; figures under `experiments/figures/`.
- Experiment run dirs are named `t<NN>_<slug>` (chronological task id), e.g. `t19_alibaba_fleet_scale`.
