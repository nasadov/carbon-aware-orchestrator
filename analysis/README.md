# Analysis Utilities

This directory contains standalone plotting, diagnostics, and sensitivity
scripts. Run them from the repository root unless a script says otherwise.

Default paths are centralized in `analysis/repo_paths.py`:

- Experiment inputs and run outputs: `experiments/`
- Figure outputs: `experiments/figures/`
- Carbon-aware package data: `pkg/carbon-aware/`

Water-frontier diagnostics used by `scripts/water_sweep.py` live here because
they operate on generated CSV outputs rather than scheduler internals.
