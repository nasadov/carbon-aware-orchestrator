#!/usr/bin/env python3
"""Archived converter for bind-event placement traces."""
import argparse
import csv
import json
import os
from collections import OrderedDict
import re


def read_binds(ndjson_path: str):
    binds = []
    if not os.path.exists(ndjson_path):
        return binds
    with open(ndjson_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            name = str(obj.get('name', '') or '')
            node = str(obj.get('node', '') or '')
            slot = obj.get('slot', None)
            if not name or not node or slot is None:
                continue
            binds.append((name, node, int(slot)))
    return binds


def _parse_duration_label_to_hours(label_value: str) -> float:
    """Parse duration label strings like 'duration-1h', 'duration-3h', 'duration-1h30m', 'duration-90m'.

    Returns hours as float; defaults to 1.0 if parsing fails.
    """
    if not label_value:
        return 1.0
    s = str(label_value).strip()
    m = re.match(r"^duration-(?P<h>\d+(?:\.\d+)?)h(?:(?P<m>\d+)m)?$", s, flags=re.IGNORECASE)
    if m:
        hours = float(m.group('h'))
        mins = float(m.group('m')) if m.group('m') else 0.0
        return hours + mins / 60.0
    m = re.match(r"^duration-(?P<m>\d+)m$", s, flags=re.IGNORECASE)
    if m:
        return float(m.group('m')) / 60.0
    # Fallback: try to find a plain number of hours in the string
    m = re.search(r"(\d+(?:\.\d+)?)\s*h", s, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    return 1.0


def read_workload_requests(workloads_dir: str):
    # Build a map ms_name -> (full_name, cpu, mem, duration_hours)
    import re, yaml
    ms_to_info = {}
    files = [os.path.join(workloads_dir, f) for f in os.listdir(workloads_dir) if f.startswith('timeslot_') and f.endswith('.yaml')]
    def slot_index(p):
        m = re.search(r"timeslot_(\d+)\.yaml", os.path.basename(p))
        return int(m.group(1)) if m else 0
    files.sort(key=slot_index)
    for fpath in files:
        try:
            with open(fpath, 'r') as fh:
                docs = list(yaml.safe_load_all(fh.read()))
        except Exception:
            continue
        for d in docs:
            if not isinstance(d, dict) or d.get('kind') != 'Deployment':
                continue
            meta = d.get('metadata', {})
            labels = meta.get('labels', {}) if isinstance(meta, dict) else {}
            full_name = meta.get('name') or labels.get('app') or ''
            ms_name = d.get('spec', {}).get('selector', {}).get('matchLabels', {}).get('name', '')
            conts = d.get('spec', {}).get('template', {}).get('spec', {}).get('containers', [])
            cpu = mem = ''
            duration_hours = 1.0
            if conts:
                reqs = conts[0].get('resources', {}).get('requests', {})
                cpu = str(reqs.get('cpu', '') or '')
                mem = str(reqs.get('memory', '') or '')
            duration_label = labels.get('duration', '')
            if duration_label:
                duration_hours = _parse_duration_label_to_hours(duration_label)
            if ms_name:
                ms_to_info[ms_name] = (full_name, cpu, mem, duration_hours)
    return ms_to_info


def main():
    ap = argparse.ArgumentParser(description='Generate bind-based vanilla_placement_session.csv from Scheduled events')
    ap.add_argument('--binds-ndjson', required=True, help='Path to bind_watcher NDJSON output')
    ap.add_argument('--workloads-dir', required=True, help='Path to workloads-vanilla directory')
    ap.add_argument('--experiment-dir', required=True, help='Experiment directory to write CSV')
    args = ap.parse_args()

    os.makedirs(args.experiment_dir, exist_ok=True)
    binds = read_binds(args.binds_ndjson)
    ms_to_info = read_workload_requests(args.workloads_dir)

    # Deduplicate by ms_name (first bind wins)
    chosen = OrderedDict()
    for name, node, slot in binds:
        if name not in chosen:
            chosen[name] = (node, slot)

    out_csv = os.path.join(args.experiment_dir, 'vanilla_placement_session.csv')
    with open(out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['pod_id','node_id','start_slot','duration','cpu_request','ram_request'])
        for ms_name, (node, slot) in chosen.items():
            full_name, cpu, mem, duration_hours = ms_to_info.get(ms_name, (ms_name, '', '', 1.0))
            # Use manifest-defined duration (in hours/slots) for vanilla logging
            w.writerow([full_name, node, float(slot), float(duration_hours), cpu, mem])
    print(f"Wrote {len(chosen)} rows to {out_csv}")


if __name__ == '__main__':
    main()



