#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${RUN_NAME:-water_checkpoint_$(date -u +%Y%m%d_%H%M%S)}"
EPSILON_FRACTIONS="${EPSILON_FRACTIONS:-0.95,0.90,0.85,0.80}"

python3 scripts/water_sweep.py \
  --pod-counts "${POD_COUNTS:-34}" \
  --seeds "${SEEDS:-44}" \
  --timeslots "${TIMESLOTS:-12}" \
  --epsilon-fractions "$EPSILON_FRACTIONS" \
  --target-waterwise-budget \
  --run-name "$RUN_NAME" \
  --overwrite \
  --quiet
