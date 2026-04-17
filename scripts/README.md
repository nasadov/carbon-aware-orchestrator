# Scripts

This directory contains runnable experiment and maintenance entry points.

- `water_sweep.py`: public carbon-water sweep entry point.
- `run_water_*.sh`: convenience wrappers around `water_sweep.py`.
- `time_complexity_sweep.py`: runtime-scaling experiment runner.
- `sweep_podcounts_and_precompute.sh`: legacy pod-count sweep wrapper.
- `update_config.py`: small YAML update helper used by sweep wrappers.
- `check_python_syntax.py`: import-free syntax check used by `make syntax-check`.
- `archive/`: retained legacy helpers that are not part of the normal workflow.

Generated outputs should go under `experiments/`; generated figures should go
under `experiments/figures/`.
