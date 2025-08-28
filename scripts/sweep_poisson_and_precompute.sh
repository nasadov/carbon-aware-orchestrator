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

# Run identifiers and timing
RUN_START_ID=$(date +%Y%m%d_%H%M%S)
RUN_ID="$RUN_START_ID"
SUMMARY_CSV="$SERVER_DIR/experiments/precompute_timing_${RUN_START_ID}.csv"

# Save original config to restore after sweep
ORIG_TMP=$(mktemp)
cp "$CONFIG_FILE" "$ORIG_TMP"
cleanup() {
  cp "$ORIG_TMP" "$CONFIG_FILE" || true
  rm -f "$ORIG_TMP" || true
  # Stamp end time in CSV filename to avoid overwriting and identify runs
  if [[ -f "$SUMMARY_CSV" ]]; then
    local run_end_full run_end_hms
    run_end_full=$(date +%Y%m%d_%H%M%S)
    run_end_hms=$(date +%H%M%S)
    mv "$SUMMARY_CSV" "$SERVER_DIR/experiments/precompute_timing_${RUN_START_ID}-${run_end_hms}.csv" || true
  fi
}
trap cleanup EXIT

# Initialize summary CSV with header
echo "run_id,begin_time,end_time,lambda,algorithm,pods,nodes,elapsed_seconds" > "$SUMMARY_CSV"

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

# Function to count pods in workloads directory
count_pods() {
  python3 - <<'PY'
import os, re, glob
wd="/root/carbon-aware-orchestrator/pkg/carbon-aware/workloads"
files=glob.glob(os.path.join(wd, "timeslot_*.yaml"))
if not files:
    print(0)
    raise SystemExit
files.sort(key=lambda x:int(re.search(r"timeslot_(\d+)\.yaml", x).group(1)))
last=files[-1]
with open(last) as f:
    names=re.findall(r'name: m(\d+)', f.read())
print((max(map(int, names))+1) if names else 0)
PY
}

# Function to count nodes in nodes.yaml (multi-document YAML)
count_nodes() {
  python3 - <<'PY'
import yaml
p="/root/carbon-aware-orchestrator/pkg/carbon-aware/nodes.yaml"
try:
    with open(p,'r') as f:
        docs=list(yaml.safe_load_all(f))
    print(sum(1 for d in docs if isinstance(d, dict) and d.get('kind')=='Node'))
except Exception:
    print(0)
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

  # Measure pods and nodes after generation
  PODS=$(count_pods)
  NODES=$(count_nodes)

  # Precompute heuristic with timing
  echo "-- heuristic precompute (lambda=$L) --"
  begin_h="$(date +%Y-%m-%dT%H:%M:%S)"
  start_h=$(date +%s%N)
  python3 "$SERVER_MAIN" \
    --algorithm heuristic \
    --precompute \
    --workloads-dir "$REPO_ROOT/pkg/carbon-aware/workloads" \
    --nodes-file "$REPO_ROOT/pkg/carbon-aware/nodes.yaml" \
    --forecasts-file "$FORECASTS_FILE" \
    --experiment-dir "$SERVER_DIR/experiments" \
    --loglevel INFO | sed -u 's/.*/[heuristic] &/'
  end_h=$(date +%s%N)
  end_iso_h="$(date +%Y-%m-%dT%H:%M:%S)"
  elapsed_h=$(python3 - "$start_h" "$end_h" <<'PY'
import sys
s=int(sys.argv[1]); e=int(sys.argv[2])
print(f"{(e-s)/1e9:.3f}")
PY
)
  echo "$RUN_ID,$begin_h,$end_iso_h,$L,heuristic,$PODS,$NODES,$elapsed_h" >> "$SUMMARY_CSV"

  # Precompute global-optimal with timing
  echo "-- global-optimal precompute (lambda=$L) --"
  begin_g="$(date +%Y-%m-%dT%H:%M:%S)"
  start_g=$(date +%s%N)
  python3 "$SERVER_MAIN" \
    --algorithm global-optimal \
    --precompute \
    --workloads-dir "$REPO_ROOT/pkg/carbon-aware/workloads" \
    --nodes-file "$REPO_ROOT/pkg/carbon-aware/nodes.yaml" \
    --forecasts-file "$FORECASTS_FILE" \
    --experiment-dir "$SERVER_DIR/experiments" \
    --loglevel INFO | sed -u 's/.*/[global] &/'
  end_g=$(date +%s%N)
  end_iso_g="$(date +%Y-%m-%dT%H:%M:%S)"
  elapsed_g=$(python3 - "$start_g" "$end_g" <<'PY'
import sys
s=int(sys.argv[1]); e=int(sys.argv[2])
print(f"{(e-s)/1e9:.3f}")
PY
)
  echo "$RUN_ID,$begin_g,$end_iso_g,$L,global-optimal,$PODS,$NODES,$elapsed_g" >> "$SUMMARY_CSV"

  # Tag the most recent experiment directories with lambda in a marker file
  for algo in heuristic global-optimal; do
    dir=$(ls -d "$SERVER_DIR/experiments/$algo"* 2>/dev/null | tail -n 1 || true)
    if [[ -n "$dir" ]]; then
      echo "lambda=$L" > "$dir/lambda.txt"
    fi
  done

done

echo "Sweep complete. Results CSV: $(basename "$SUMMARY_CSV") in $SERVER_DIR/experiments/ (will be renamed to include end time on exit)"
