#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${RUN_NAME:-water_milp_ref_34pods_s42_44_$(date -u +%Y%m%d)}"
EPSILON_FRACTIONS="${EPSILON_FRACTIONS:-0.95,0.90,0.85,0.80}"
SEEDS="${SEEDS:-42,43,44}"

python3 scripts/water_sweep.py \
  --preset checkpoint \
  --pod-counts 34 \
  --seeds "$SEEDS" \
  --epsilon-fractions "$EPSILON_FRACTIONS" \
  --target-waterwise-budget \
  --run-name "$RUN_NAME" \
  --overwrite \
  --quiet
