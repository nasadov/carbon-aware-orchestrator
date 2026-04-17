#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${RUN_NAME:-water_ablations_$(date -u +%Y%m%d_%H%M%S)}"

python3 scripts/water_sweep.py \
  --preset paper \
  --methods heuristic-carbon,heuristic-weighted,heuristic-pareto,heuristic-epsilon-pareto \
  --skip-milp \
  --epsilon-fractions "${EPSILON_FRACTIONS:-0.95,0.90,0.85}" \
  --run-name "$RUN_NAME" \
  --overwrite \
  --quiet
