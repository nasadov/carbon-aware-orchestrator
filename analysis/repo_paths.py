"""Shared repository-relative paths for analysis scripts.

The analysis scripts are often launched from different worktrees and working
directories. Keeping path defaults here avoids hardcoded machine-specific paths.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
FIGURES_ROOT = EXPERIMENTS_ROOT / "figures"
PKG_ROOT = REPO_ROOT / "pkg" / "carbon-aware"
SERVER_PYTHON_DIR = PKG_ROOT / "server-python"
WORKLOAD_GENERATOR = PKG_ROOT / "infra_workload_gen.py"
WORKLOAD_CONFIG = PKG_ROOT / "infra-workload-config.yaml"
WORKLOADS_DIR = PKG_ROOT / "workloads"
WORKLOADS_VANILLA_DIR = PKG_ROOT / "workloads-vanilla"
NODES_FILE = PKG_ROOT / "nodes.yaml"
FORECASTS_FILE = SERVER_PYTHON_DIR / "all_forecasts.json"
BIN_FORECASTS_FILE = REPO_ROOT / "bin" / "all_forecasts.json"
