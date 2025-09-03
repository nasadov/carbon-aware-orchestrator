#!/usr/bin/env bash
set -euo pipefail

# Prevent concurrent runs (best-effort lock)
exec 7>/tmp/vanilla_benchmark_series.lock
if ! flock -n 7; then
  echo "Another vanilla_benchmark_series.sh run is already active. Exiting." >&2
  exit 0
fi

# Vanilla benchmark series runner
# - Sweeps exact pod counts (P) from a range
# - Regenerates infra/workloads for each P
# - Runs vanilla benchmark via /root/carbon/scripts/run_benchmark.sh
# - Uses SHRINK_FACTOR=3600 and CALL_INTERVAL=3600 (simulation seconds)
#
# Usage:
#   bash scripts/vanilla_benchmark_series.sh
# Optional env vars:
#   LAMBDAS="4 8 10 12 16"   # override lambda list
#   INTERVAL=60               # metrics collection interval (seconds)

# Default sweep values for exact pod counts
# Override with: MIN_PODS, MAX_PODS, PODS_STEP, or POD_COUNTS (space-separated)
MIN_PODS=${MIN_PODS:-5}
MAX_PODS=${MAX_PODS:-200}
PODS_STEP=${PODS_STEP:-5}
if [[ -n "${POD_COUNTS:-}" ]]; then
  read -r -a PODS_VALUES <<< "$POD_COUNTS"
else
  PODS_VALUES=()
  cur=$MIN_PODS
  while [[ $cur -le $MAX_PODS ]]; do
    PODS_VALUES+=($cur)
    cur=$((cur + PODS_STEP))
  done
fi

REPO_ROOT=/root/carbon-aware-orchestrator
CONFIG_FILE="$REPO_ROOT/pkg/carbon-aware/infra-workload-config.yaml"
GENERATOR="$REPO_ROOT/pkg/carbon-aware/infra_workload_gen.py"
SERVER_DIR="$REPO_ROOT/pkg/carbon-aware/server-python"
FORECASTS_FILE="$SERVER_DIR/all_forecasts.json"
CARBON_BENCH_SCRIPT="/root/carbon/scripts/run_benchmark.sh"
PY_SUMMARY_MOD="/root/carbon-aware-orchestrator/pkg/carbon-aware/server-python/carbon_aware/placement_summary.py"
PY_PLACE_GEN="/root/carbon-aware-orchestrator/scripts/generate_vanilla_placement_session.py"

# Verify dependencies
if [[ ! -f "$CARBON_BENCH_SCRIPT" ]]; then
  echo "Error: benchmark script not found: $CARBON_BENCH_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$FORECASTS_FILE" ]]; then
  echo "Error: forecasts file not found: $FORECASTS_FILE" >&2
  exit 1
fi

# Series run identifier
RUN_START_ID=$(date +%Y%m%d_%H%M%S)

# Save original config to restore after series
ORIG_TMP=$(mktemp)
cp "$CONFIG_FILE" "$ORIG_TMP"
cleanup() {
  cp "$ORIG_TMP" "$CONFIG_FILE" || true
  rm -f "$ORIG_TMP" || true
}
trap cleanup EXIT

# Generate exact workloads with total pods = TARGET_PODS across configured timeslots
# This overwrites workloads and workloads-vanilla produced by the generator
generate_exact_workloads() {
  local target_pods="$1"
  python3 - "$CONFIG_FILE" "$target_pods" <<'PY'
import sys, os, yaml, random

def format_duration(hours: float) -> str:
    if hours == int(hours):
        return f"{int(hours)}h"
    if hours * 60 == int(hours * 60):
        minutes = int(hours * 60)
        if minutes < 60:
            return f"{minutes}m"
        h = minutes // 60
        m = minutes % 60
        return f"{h}h" if m == 0 else f"{h}h{m}m"
    return f"{hours:.1f}h"

config_path = sys.argv[1]
target = int(sys.argv[2])
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)
wcfg = cfg.get('workload', {}) if isinstance(cfg, dict) else {}
# Base directory is where the config file lives (pkg/carbon-aware)
base_dir = os.path.dirname(os.path.abspath(config_path))
output_dir_rel = wcfg.get('output_dir', 'workloads')
vanilla_output_dir_rel = wcfg.get('vanilla_output_dir', output_dir_rel + '-vanilla')
output_dir = os.path.join(base_dir, output_dir_rel)
vanilla_output_dir = os.path.join(base_dir, vanilla_output_dir_rel)
num_timeslots = int(wcfg.get('num_timeslots', 12))
base_name = str(wcfg.get('base_name', 'm'))
durations = list(wcfg.get('durations', [1,3,6]))
duration_assignment_method = str(wcfg.get('duration_assignment_method', 'random'))
deadline_strategy = str(wcfg.get('deadline_strategy', 'flexible'))
deadline_flexibility_hours = list(wcfg.get('deadline_flexibility_hours', [2,6,24]))
cpu_options = list(wcfg.get('cpu_options', ['500m','200m','100m']))
mem_options = list(wcfg.get('mem_options', ['500Mi','200Mi']))
random_seed = int(wcfg.get('random_seed', 42))

random.seed(random_seed)

os.makedirs(output_dir, exist_ok=True)
os.makedirs(vanilla_output_dir, exist_ok=True)

def deployments_for_count(count, ms_start_index):
    docs_custom, docs_vanilla = [], []
    duration_counter = 0
    for i in range(count):
        ms_name = f"{base_name}{ms_start_index + i:03d}"
        full_name = ms_name  # Will append duration/deadline labels below

        # Pick CPU/mem
        cpu_req = random.choice(cpu_options)
        mem_req = random.choice(mem_options)

        # Duration selection
        if duration_assignment_method == 'cycle':
            duration_hours = durations[duration_counter % len(durations)]
            duration_counter += 1
        else:
            duration_hours = random.choice(durations)

        # Deadline selection
        if deadline_strategy == 'tight':
            deadline_hours = duration_hours
        elif deadline_strategy == 'exact':
            deadline_hours = duration_hours * 2
        elif deadline_strategy == 'mixed':
            if random.random() < 0.3:
                deadline_hours = duration_hours
            else:
                deadline_hours = duration_hours + random.choice(deadline_flexibility_hours)
        else:
            deadline_hours = duration_hours + random.choice(deadline_flexibility_hours)

        duration_str = format_duration(float(duration_hours))
        deadline_str = format_duration(float(deadline_hours))
        full_name = f"{ms_name}-duration-{duration_str}-deadline-{deadline_str}"

        # Build deployment dicts
        def make_dep(use_custom_scheduler: bool):
            return {
                'apiVersion': 'apps/v1',
                'kind': 'Deployment',
                'metadata': {
                    'name': full_name,
                    'namespace': 'default',
                    'labels': {
                        'app': full_name,
                        'duration': f'duration-{duration_str}',
                        'deadline': f'deadline-{deadline_str}',
                    }
                },
                'spec': {
                    'replicas': 1,
                    'selector': {
                        'matchLabels': { 'name': ms_name }
                    },
                    'template': {
                        'metadata': { 'labels': { 'name': ms_name } },
                        'spec': {
                            **({'schedulerName': 'fogatlas'} if use_custom_scheduler else {}),
                            'containers': [
                                {
                                    'name': 'fake-container',
                                    'image': 'fake-image',
                                    'resources': {
                                        'requests': { 'cpu': cpu_req, 'memory': mem_req }
                                    }
                                }
                            ]
                        }
                    }
                }
            }
        docs_custom.append(make_dep(True))
        docs_vanilla.append(make_dep(False))
    return docs_custom, docs_vanilla

# Distribute exactly `target` across `num_timeslots`
counts = []
for s in range(num_timeslots):
    prev = (target * s) // num_timeslots
    curr = (target * (s + 1)) // num_timeslots
    counts.append(curr - prev)

# Write timeslot files
ms_index = 0
for slot_id, c in enumerate(counts):
    custom_docs, vanilla_docs = deployments_for_count(c, ms_index)
    ms_index += c
    p1 = os.path.join(output_dir, f"timeslot_{slot_id}.yaml")
    p2 = os.path.join(vanilla_output_dir, f"timeslot_{slot_id}.yaml")
    with open(p1, 'w') as f:
        for d in custom_docs:
            yaml.safe_dump(d, f, sort_keys=False); f.write('---\n')
    with open(p2, 'w') as f:
        for d in vanilla_docs:
            yaml.safe_dump(d, f, sort_keys=False); f.write('---\n')
    print(f"[exact] Generated timeslot {slot_id} with {c} microservices in both directories.")

print(f"[exact] Total microservices generated: {sum(counts)} across {num_timeslots} timeslots")
print(f"[exact] Wrote to: {output_dir} and {vanilla_output_dir}")
PY
}

# Helpers to count pods/nodes
count_pods() {
  python3 - <<'PY'
import os, re, glob
wd="/root/carbon-aware-orchestrator/pkg/carbon-aware/workloads"
files=glob.glob(os.path.join(wd, "timeslot_*.yaml"))
if not files:
    print(0); raise SystemExit
max_idx=-1
for path in files:
    with open(path) as f:
        names=re.findall(r'name:\s+m(\d+)', f.read())
        if names:
            max_idx=max(max_idx, max(map(int, names)))
print(max_idx+1 if max_idx>=0 else 0)
PY
}
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

# Main loop
for P in "${PODS_VALUES[@]}"; do
  echo "==== Vanilla benchmark series: pods=$P (run $RUN_START_ID) ===="

  # Generate infra and workloads
  (
    cd "$REPO_ROOT/pkg/carbon-aware"
    python3 "$GENERATOR" | sed -u 's/.*/[gen] &/'
  )

  # Recreate KWOK cluster to apply timing and node architecture, then apply nodes.yaml
  echo "[kwok] Recreating KWOK cluster to apply updated timing and nodes..."
  set +e
  make -C /root/carbon kwok | sed -u 's/.*/[kwok] &/'
  rc_kwok=$?
  set -e
  if [[ $rc_kwok -ne 0 ]]; then
    echo "[kwok] WARNING: KWOK cluster recreation failed (rc=$rc_kwok). Continuing, but vanilla node mapping may be invalid." >&2
  fi
  echo "[kwok] Applying generated nodes.yaml to cluster"
  kubectl apply -f "$REPO_ROOT/pkg/carbon-aware/nodes.yaml" | sed -u 's/.*/[kwok] &/' || echo "[kwok] WARNING: kubectl apply nodes.yaml failed" >&2

  # Overwrite workloads with exact pod count
  generate_exact_workloads "$P" | sed -u 's/.*/[gen-exact] &/'

  PODS=$(count_pods)
  NODES=$(count_nodes)
  echo "Pods: $PODS, Nodes: $NODES"

  EXP_NAME="vanilla_${PODS}pods_${RUN_START_ID}"
  EXP_DIR="$SERVER_DIR/experiments/${EXP_NAME}"

  # If this experiment dir already exists for some reason, skip to avoid duplicates
  if [[ -d "$EXP_DIR" ]]; then
    echo "[skip] Experiment directory already exists: $EXP_DIR (skipping run to avoid duplicates)"
    continue
  fi

  echo "-- Running vanilla benchmark: $EXP_NAME --"
  begin_v="$(date +%Y-%m-%dT%H:%M:%S)"
  # Run non-interactively, auto-stop collector, SHRINK_FACTOR=3600, CALL_INTERVAL=3600
  set +e
  bash "$CARBON_BENCH_SCRIPT" \
    -n "$EXP_NAME" \
    -a vanilla \
    -f "${SHRINK_FACTOR:-3600}" \
    --call-interval 3600 \
    -o "$SERVER_DIR/experiments" \
    -F "$FORECASTS_FILE" \
    -i "${INTERVAL:-60}" \
    --auto-stop \
    --non-interactive | sed -u 's/.*/[vanilla] &/'
  rc=$?
  set -e
  end_v="$(date +%Y-%m-%dT%H:%M:%S)"
  if [[ $rc -ne 0 ]]; then
    echo "WARNING: vanilla benchmark failed (rc=$rc)"
  fi

done

echo "Vanilla benchmark series complete (run $RUN_START_ID)."
