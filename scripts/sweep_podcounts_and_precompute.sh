#!/usr/bin/env bash
set -euo pipefail

# Prevent concurrent runs (best-effort lock)
exec 9>/tmp/sweep_podcounts_and_precompute.lock
if ! flock -n 9; then
  echo "Another sweep_podcounts_and_precompute.sh run is already active. Exiting." >&2
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
EXPERIMENTS_ROOT="$REPO_ROOT/experiments/precompute_sweeps"

# Embodied mode selection (default: proportional). Flags: --proportional | --uniform | --both
# Partial sweep range controls (defaults)
START=20
END=200
MODES=("proportional")
OPERATIONAL_ONLY=false
IDENTICAL_PODS=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --both)
      MODES=("proportional" "uniform")
      shift
      ;;
    --uniform)
      MODES=("uniform")
      shift
      ;;
    --proportional)
      MODES=("proportional")
      shift
      ;;
    --start)
      shift
      START="$1"
      shift
      ;;
    --end)
      shift
      END="$1"
      shift
      ;;
    --operational-only)
      OPERATIONAL_ONLY=true
      shift
      ;;
    --identical-pods)
      IDENTICAL_PODS=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--proportional|--uniform|--both] [--start N --end N] [--operational-only] [--identical-pods]" >&2
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--proportional|--uniform|--both] [--start N --end N] [--operational-only] [--identical-pods]" >&2
      exit 1
      ;;
  esac
done

if [[ "$OPERATIONAL_ONLY" == true ]]; then
  MODES=("operational-only")
  echo "[sweep] Operational-only mode enabled: embodied emissions will be ignored."
fi

# Run identifiers and timing
RUN_START_ID=$(date +%Y%m%d_%H%M%S)
RUN_ID="$RUN_START_ID"
RUN_DIR="$EXPERIMENTS_ROOT/sweep_${RUN_ID}"
if [[ -e "$RUN_DIR" ]]; then
  suffix=1
  while [[ -e "$EXPERIMENTS_ROOT/sweep_${RUN_START_ID}_${suffix}" ]]; do
    ((suffix++))
  done
  RUN_ID="${RUN_START_ID}_${suffix}"
  RUN_DIR="$EXPERIMENTS_ROOT/sweep_${RUN_ID}"
fi
mkdir -p "$RUN_DIR"
SUMMARY_CSV="$RUN_DIR/precompute_timing_${RUN_ID}.csv"

# Save original config to restore after sweep
ORIG_TMP=$(mktemp)
cp "$CONFIG_FILE" "$ORIG_TMP"
cleanup() {
  cp "$ORIG_TMP" "$CONFIG_FILE" || true
  rm -f "$ORIG_TMP" || true
  # Stamp end time in CSV filename to avoid overwriting and identify runs
  if [[ -f "$SUMMARY_CSV" ]]; then
    local run_end_hms
    run_end_hms=$(date +%H%M%S)
    mv "$SUMMARY_CSV" "$RUN_DIR/precompute_timing_${RUN_ID}-${run_end_hms}.csv" || true
  fi
}
trap cleanup EXIT

# Initialize summary CSV with header
echo "run_id,begin_time,end_time,target_pods,algorithm,pods,nodes,elapsed_seconds" > "$SUMMARY_CSV"

# Apply partial sweep filter if requested
if [[ -n "$START" && -n "$END" ]]; then
  FILTERED=()
  for v in "${POD_VALUES[@]}"; do
    if (( v >= START && v <= END )); then
      FILTERED+=("$v")
    fi
  done
  if [[ ${#FILTERED[@]} -gt 0 ]]; then
    POD_VALUES=("${FILTERED[@]}")
  fi
fi

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

# Function to force identical pod characteristics (CPU, memory, duration, deadline flexibility)
force_identical_pods() {
  python3 - "$CONFIG_FILE" <<'PY'
import sys, yaml
path = sys.argv[1]
with open(path, 'r') as f:
    cfg = yaml.safe_load(f)

if not isinstance(cfg, dict):
    raise SystemExit("Invalid YAML structure: expected mapping at top level")

wl = cfg.get('workload') or {}

def midpoint(lst):
    if isinstance(lst, list) and len(lst) > 0:
        return lst[len(lst)//2]
    return None

# Collapse lists to their middle value
for key in ['cpu_options', 'mem_options', 'durations', 'deadline_flexibility_hours']:
    val = wl.get(key)
    m = midpoint(val)
    if m is not None:
        wl[key] = [m]

# Make assignment deterministic
wl['deadline_strategy'] = 'flexible'
wl['duration_assignment_method'] = 'cycle'

cfg['workload'] = wl
with open(path, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print('Configured identical pods in workload using middle values in workload.* lists')
PY
}

# Function to count pods in workloads directory
count_pods() {
  python3 - <<'PY'
import os, re, glob, yaml
wd="/root/carbon-aware-orchestrator/pkg/carbon-aware/workloads"
files=glob.glob(os.path.join(wd, "timeslot_*.yaml"))
if not files:
    print(0)
    raise SystemExit
total=0
for path in files:
    try:
        with open(path) as f:
            docs=list(yaml.safe_load_all(f))
        for d in docs:
            if isinstance(d, dict) and d.get('kind')=='Deployment':
                spec=d.get('spec',{})
                total += int(spec.get('replicas',1))
    except Exception:
        pass
print(total)
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

# If requested, enforce identical pod characteristics across the generated workload
if [[ "$IDENTICAL_PODS" == true ]]; then
  echo "[sweep] Identical pods mode enabled: forcing uniform CPU/MEM/duration/deadline (middle values)."
  force_identical_pods
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

  # Heuristic (selected embodied/operational modes)
  for MODE in "${MODES[@]}"; do
    TAG="prop"
    MODE_LABEL="$MODE"
    MODE_DESC="$MODE"
    if [[ "$MODE" == "uniform" ]]; then
      TAG="uniform"
    elif [[ "$MODE" == "operational-only" ]]; then
      TAG="op"
      MODE_LABEL="op"
      MODE_DESC="operational-only"
    fi
    echo "-- heuristic precompute ($MODE_DESC) (pods=$P) --"
    begin_h="$(date +%Y-%m-%dT%H:%M:%S)"
    start_h=$(date +%s%N)
    cmd=(
      python3 "$SERVER_MAIN"
      --algorithm heuristic
      --precompute
      --workloads-dir "$REPO_ROOT/pkg/carbon-aware/workloads"
      --nodes-file "$REPO_ROOT/pkg/carbon-aware/nodes.yaml"
      --forecasts-file "$FORECASTS_FILE"
      --experiment-dir "$RUN_DIR"
      --loglevel INFO
    )
    if [[ "$MODE" == "operational-only" ]]; then
      cmd+=(--operational-only)
    else
      cmd+=(--embodied-mode "$MODE")
    fi
    "${cmd[@]}" | sed -u "s/.*/[heuristic-$TAG] &/"
    end_h=$(date +%s%N)
    end_iso_h="$(date +%Y-%m-%dT%H:%M:%S)"
    elapsed_h=$(python3 - "$start_h" "$end_h" <<'PY'
import sys
s=int(sys.argv[1]); e=int(sys.argv[2])
print(f"{(e-s)/1e9:.3f}")
PY
)
    echo "$RUN_ID,$begin_h,$end_iso_h,$P,heuristic-$MODE_LABEL,$PODS,$NODES,$elapsed_h" >> "$SUMMARY_CSV"
  done

  # Global-optimal (selected embodied/operational modes)
  for MODE in "${MODES[@]}"; do
    TAG="prop"
    MODE_LABEL="$MODE"
    MODE_DESC="$MODE"
    if [[ "$MODE" == "uniform" ]]; then
      TAG="uniform"
    elif [[ "$MODE" == "operational-only" ]]; then
      TAG="op"
      MODE_LABEL="op"
      MODE_DESC="operational-only"
    fi
    echo "-- global-optimal precompute ($MODE_DESC) (pods=$P) --"
    begin_g="$(date +%Y-%m-%dT%H:%M:%S)"
    start_g=$(date +%s%N)
    cmd=(
      python3 "$SERVER_MAIN"
      --algorithm global-optimal
      --precompute
      --workloads-dir "$REPO_ROOT/pkg/carbon-aware/workloads"
      --nodes-file "$REPO_ROOT/pkg/carbon-aware/nodes.yaml"
      --forecasts-file "$FORECASTS_FILE"
      --experiment-dir "$RUN_DIR"
      --loglevel INFO
    )
    if [[ "$MODE" == "operational-only" ]]; then
      cmd+=(--operational-only)
    else
      cmd+=(--embodied-mode "$MODE")
    fi
    "${cmd[@]}" | sed -u "s/.*/[global-$TAG] &/"
    end_g=$(date +%s%N)
    end_iso_g="$(date +%Y-%m-%dT%H:%M:%S)"
    elapsed_g=$(python3 - "$start_g" "$end_g" <<'PY'
import sys
s=int(sys.argv[1]); e=int(sys.argv[2])
print(f"{(e-s)/1e9:.3f}")
PY
)
    echo "$RUN_ID,$begin_g,$end_iso_g,$P,global-optimal-$MODE_LABEL,$PODS,$NODES,$elapsed_g" >> "$SUMMARY_CSV"
  done

  # Vanilla
  VANILLA_DESC="vanilla"
  VANILLA_LABEL="vanilla"
  if [[ "$OPERATIONAL_ONLY" == true ]]; then
    VANILLA_DESC="vanilla (operational-only)"
    VANILLA_LABEL="vanilla-op"
  fi
  echo "-- $VANILLA_DESC (pods=$P) --"
  begin_v="$(date +%Y-%m-%dT%H:%M:%S)"
  start_v=$(date +%s%N)
  cmd=(
    python3 "$SERVER_MAIN"
    --algorithm vanilla
    --precompute
    --workloads-dir "$REPO_ROOT/pkg/carbon-aware/workloads"
    --nodes-file "$REPO_ROOT/pkg/carbon-aware/nodes.yaml"
    --forecasts-file "$FORECASTS_FILE"
    --experiment-dir "$RUN_DIR"
    --loglevel INFO
  )
  if [[ "$OPERATIONAL_ONLY" == true ]]; then
    cmd+=(--operational-only)
  fi
  VANILLA_TAG="$VANILLA_LABEL"
  "${cmd[@]}" | sed -u "s/.*/[$VANILLA_TAG] &/"
  end_v=$(date +%s%N)
  end_iso_v="$(date +%Y-%m-%dT%H:%M:%S)"
  elapsed_v=$(python3 - "$start_v" "$end_v" <<'PY'
import sys
s=int(sys.argv[1]); e=int(sys.argv[2])
print(f"{(e-s)/1e9:.3f}")
PY
)
  echo "$RUN_ID,$begin_v,$end_iso_v,$P,$VANILLA_LABEL,$PODS,$NODES,$elapsed_v" >> "$SUMMARY_CSV"

  # Tag latest per-algo with pods.txt
  for algo in heuristic global-optimal vanilla; do
    dir=$(ls -dt "$RUN_DIR/${algo}_*pods_*" 2>/dev/null | head -n 1 || true)
    if [[ -n "$dir" ]]; then
      echo "pods=$P" > "$dir/pods.txt"
    fi
  done

done

echo "Sweep complete. Results stored under $RUN_DIR (timing CSV renamed with end timestamp on exit)"
