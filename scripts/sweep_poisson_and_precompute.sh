#!/usr/bin/env bash
set -euo pipefail

# Prevent concurrent runs (best-effort lock)
exec 9>/tmp/sweep_poisson_and_precompute.lock
if ! flock -n 9; then
  echo "Another sweep_poisson_and_precompute.sh run is already active. Exiting." >&2
  exit 0
fi

# Sweep values for exact total pods
POD_VALUES=(20 40 60 80 100 120 140 160 180 200)

REPO_ROOT=/root/carbon-aware-orchestrator
CONFIG_FILE="$REPO_ROOT/pkg/carbon-aware/infra-workload-config.yaml"
GENERATOR="$REPO_ROOT/pkg/carbon-aware/infra_workload_gen.py"
SERVER_MAIN="$REPO_ROOT/pkg/carbon-aware/server-python/main.py"
SERVER_DIR="$REPO_ROOT/pkg/carbon-aware/server-python"
FORECASTS_FILE="$SERVER_DIR/all_forecasts.json"

# Optional: run vanilla benchmark via external script (requires cluster)
RUN_VANILLA_BENCHMARK=${RUN_VANILLA_BENCHMARK:-false}
CARBON_BENCH_SCRIPT="/root/carbon/scripts/run_benchmark.sh"

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
echo "run_id,begin_time,end_time,target_pods,algorithm,pods,nodes,elapsed_seconds" > "$SUMMARY_CSV"

# Function to ensure generation_strategy is set to exact_total
ensure_exact_strategy() {
  python3 - "$CONFIG_FILE" <<'PY'
import sys, re
path=sys.argv[1]
with open(path,'r') as f:
    s=f.read()
if re.search(r'^(\s*generation_strategy:\s*)exact_total\b', s, flags=re.M):
    print("generation_strategy already set to exact_total")
    print("OK")
    sys.exit(0)
new_s=re.sub(r'^(\s*generation_strategy:\s*).*$','\1exact_total', s, count=1, flags=re.M)
with open(path,'w') as f:
    f.write(new_s)
print(f"Updated generation_strategy to exact_total in {path}")
print("OK")
PY
}

# Function to in-place update exact_total_pods in YAML while preserving inline comments
update_total_pods() {
  local new_total="$1"
  python3 - "$CONFIG_FILE" "$new_total" <<'PY'
import sys, re
path=sys.argv[1]
val=sys.argv[2]
with open(path,'r') as f:
    s=f.read()
# Replace the first occurrence of the exact_total_pods value, preserving trailing comment/content
new_s=re.sub(r'^(\s*exact_total_pods:\s*)\d+(\b.*)$', rf'\g<1>{val}\2', s, count=1, flags=re.M)
with open(path,'w') as f:
    f.write(new_s)
print(f"Updated exact_total_pods to {val} in {path}")
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

for P in "${POD_VALUES[@]}"; do
  echo "==== Processing target_pods=$P ===="
  ensure_exact_strategy
  update_total_pods "$P"

  # Generate infra and workloads
  (
    cd "$REPO_ROOT/pkg/carbon-aware"
    python3 "$GENERATOR" | sed -u 's/.*/[gen] &/'
  )

  # Measure pods and nodes after generation
  PODS=$(count_pods)
  NODES=$(count_nodes)

  # Precompute heuristic with timing
  echo "-- heuristic precompute (pods=$P) --"
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
  echo "$RUN_ID,$begin_h,$end_iso_h,$P,heuristic,$PODS,$NODES,$elapsed_h" >> "$SUMMARY_CSV"

  # Precompute global-optimal with timing
  echo "-- global-optimal precompute (pods=$P) --"
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
  echo "$RUN_ID,$begin_g,$end_iso_g,$P,global-optimal,$PODS,$NODES,$elapsed_g" >> "$SUMMARY_CSV"

  # Optional vanilla benchmark (cluster-based)
  if [[ "$RUN_VANILLA_BENCHMARK" == "true" ]]; then
    echo "-- vanilla benchmark (pods=$P) --"
    EXP_NAME="vanilla_${PODS}pods_${RUN_START_ID}_P${P}"
    begin_v="$(date +%Y-%m-%dT%H:%M:%S)"
    # Determine metrics collection interval: default to ceil(3600/SHRINK_FACTOR) with a minimum of 1 second
    SF="${SHRINK_FACTOR:-3600}"
    if [[ -z "${INTERVAL:-}" ]]; then
      METRICS_INTERVAL=$(( (3600 + SF - 1) / SF ))
      if [[ $METRICS_INTERVAL -lt 1 ]]; then METRICS_INTERVAL=1; fi
    else
      METRICS_INTERVAL="${INTERVAL}"
    fi
    bash "$CARBON_BENCH_SCRIPT" \
      -n "$EXP_NAME" \
      -a vanilla \
      -f "${SHRINK_FACTOR:-3600}" \
      --call-interval 3600 \
      -o "$SERVER_DIR/experiments" \
      -F "$FORECASTS_FILE" \
      -i "$METRICS_INTERVAL" \
      --auto-stop \
      --non-interactive | sed -u 's/.*/[vanilla] &/' || echo "vanilla benchmark failed"
    end_v="$(date +%Y-%m-%dT%H:%M:%S)"
    # Tag the vanilla result dir with pods
    dir=$(ls -d "$SERVER_DIR/experiments/${EXP_NAME}_vanilla" 2>/dev/null | tail -n 1 || true)
    [[ -n "$dir" ]] && echo "pods=$P" > "$dir/pods.txt"
  fi

  # Tag the most recent experiment directories with pods in a marker file
  for algo in heuristic global-optimal; do
    dir=$(ls -d "$SERVER_DIR/experiments/$algo"* 2>/dev/null | tail -n 1 || true)
    if [[ -n "$dir" ]]; then
      echo "pods=$P" > "$dir/pods.txt"
    fi
  done

done

echo "Sweep complete. Results CSV: $(basename "$SUMMARY_CSV") in $SERVER_DIR/experiments/ (will be renamed to include end time on exit)"
