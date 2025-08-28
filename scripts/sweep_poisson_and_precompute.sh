#!/usr/bin/env bash
set -euo pipefail

# Sweep values for poisson_lambda
LAMBDA_VALUES=(4 8 10 12 16)

REPO_ROOT=/root/carbon-aware-orchestrator
CONFIG_FILE="$REPO_ROOT/pkg/carbon-aware/infra-workload-config.yaml"
GENERATOR="$REPO_ROOT/pkg/carbon-aware/infra_workload_gen.py"
SERVER_MAIN="$REPO_ROOT/pkg/carbon-aware/server-python/main.py"
SERVER_DIR="$REPO_ROOT/pkg/carbon-aware/server-python"
FORECASTS_FILE="$SERVER_DIR/all_forecasts.json"

# Save original config to restore after sweep
ORIG_TMP=$(mktemp)
cp "$CONFIG_FILE" "$ORIG_TMP"
cleanup() {
  cp "$ORIG_TMP" "$CONFIG_FILE" || true
  rm -f "$ORIG_TMP" || true
}
trap cleanup EXIT

# Function to in-place update poisson_lambda in YAML while preserving inline comments
update_lambda() {
  local new_lambda="$1"
  python3 - "$CONFIG_FILE" "$new_lambda" <<'PY'
import sys, re
path=sys.argv[1]
val=sys.argv[2]
with open(path,'r') as f:
    s=f.read()
# Replace the first occurrence of the poisson_lambda value, preserving trailing comment/content
new_s=re.sub(r'^(\s*poisson_lambda:\s*)\d+(\b.*)$', rf'\g<1>{val}\2', s, count=1, flags=re.M)
with open(path,'w') as f:
    f.write(new_s)
print(f"Updated poisson_lambda to {val} in {path}")
PY
}

# Ensure forecasts exist
if [[ ! -f "$FORECASTS_FILE" ]]; then
  echo "Forecasts file not found: $FORECASTS_FILE" >&2
  exit 1
fi

for L in "${LAMBDA_VALUES[@]}"; do
  echo "==== Processing lambda=$L ===="
  update_lambda "$L"

  # Generate infra and workloads
  (
    cd "$REPO_ROOT/pkg/carbon-aware"
    python3 "$GENERATOR" | sed -u 's/.*/[gen] &/'
  )

  # Precompute heuristic
  echo "-- heuristic precompute (lambda=$L) --"
  python3 "$SERVER_MAIN" \
    --algorithm heuristic \
    --precompute \
    --workloads-dir "$REPO_ROOT/pkg/carbon-aware/workloads" \
    --nodes-file "$REPO_ROOT/pkg/carbon-aware/nodes.yaml" \
    --forecasts-file "$FORECASTS_FILE" \
    --experiment-dir "$SERVER_DIR/experiments" \
    --loglevel INFO | sed -u 's/.*/[heuristic] &/'

  # Precompute global-optimal
  echo "-- global-optimal precompute (lambda=$L) --"
  python3 "$SERVER_MAIN" \
    --algorithm global-optimal \
    --precompute \
    --workloads-dir "$REPO_ROOT/pkg/carbon-aware/workloads" \
    --nodes-file "$REPO_ROOT/pkg/carbon-aware/nodes.yaml" \
    --forecasts-file "$FORECASTS_FILE" \
    --experiment-dir "$SERVER_DIR/experiments" \
    --loglevel INFO | sed -u 's/.*/[global] &/'

  # Tag the most recent experiment directories with lambda in a marker file
  for algo in heuristic global-optimal; do
    dir=$(ls -d "$SERVER_DIR/experiments/$algo"* 2>/dev/null | tail -n 1 || true)
    if [[ -n "$dir" ]]; then
      echo "lambda=$L" > "$dir/lambda.txt"
    fi
  done

done

echo "Sweep complete. Results in $SERVER_DIR/experiments/"
