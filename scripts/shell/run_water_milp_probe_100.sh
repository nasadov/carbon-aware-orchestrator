#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${RUN_NAME:-water_milp_probe_100pods_$(date -u +%Y%m%d)}"
SEED="${SEED:-44}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"

timeout "$TIMEOUT_SECONDS" python3 scripts/water_sweep.py \
  --preset paper \
  --pod-counts 100 \
  --seeds "$SEED" \
  --methods milp-carbon \
  --run-name "$RUN_NAME" \
  --overwrite \
  --no-plots \
  --quiet
