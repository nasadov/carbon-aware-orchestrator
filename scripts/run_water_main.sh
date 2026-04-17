#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${RUN_NAME:-water_main_$(date -u +%Y%m%d_%H%M%S)}"
EPSILON_FRACTIONS="${EPSILON_FRACTIONS:-0.95,0.90,0.85,0.80,0.75,0.70}"

python3 scripts/water_sweep.py \
  --preset paper \
  --epsilon-fractions "$EPSILON_FRACTIONS" \
  --target-waterwise-budget \
  --run-name "$RUN_NAME" \
  --overwrite \
  --quiet
