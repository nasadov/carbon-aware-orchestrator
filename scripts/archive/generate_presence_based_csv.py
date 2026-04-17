#!/usr/bin/env python3
"""Archived converter for presence-based placement traces."""
import argparse
import csv
import json
import os
import re
from collections import defaultdict


def parse_presence(ndjson_path: str):
    """
    Read NDJSON presence stream rows like:
      {"t": 169..., "slot": 3, "pods": [{"name": "m012-...", "node": "node-1...", "phase": "Running"}, ...]}
    Returns: dict[(ms_name)] -> { node, first_slot, last_slot }
    first_slot is the minimum slot where the pod was observed, last_slot is the maximum.
    """
    seen = {}
    # For duration spanning slot boundaries, we keep min and max slot seen
    bounds = defaultdict(lambda: {'node': '', 'first_slot': None, 'last_slot': None})
    ms_re = re.compile(r"(m\d{3,})")
    if not os.path.exists(ndjson_path):
        return {}
    with open(ndjson_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            slot = int(obj.get('slot', 0))
            pods = obj.get('pods', [])
            if not isinstance(pods, list):
                continue
            for p in pods:
                name = str(p.get('name', '') or '')
                m = ms_re.search(name)
                if not m:
                    continue
                ms = m.group(1)
                node = str(p.get('node', '') or '')
                b = bounds[ms]
                if not b['node'] and node:
                    b['node'] = node
                if b['first_slot'] is None or slot < b['first_slot']:
                    b['first_slot'] = slot
                if b['last_slot'] is None or slot > b['last_slot']:
                    b['last_slot'] = slot
    # Normalize missing
    out = {}
    for ms, b in bounds.items():
        if b['first_slot'] is None:
            continue
        # Duration in slots = (last_slot - first_slot) + 1 observation windows
        out[ms] = b
    return out


def read_workload_requests(workloads_dir: str):
    # Build a map ms_name -> (full_name, cpu, mem)
    import yaml
    ms_to_info = {}
    files = [os.path.join(workloads_dir, f) for f in os.listdir(workloads_dir) if f.startswith('timeslot_') and f.endswith('.yaml')]
    files.sort(key=lambda p: int(re.search(r"timeslot_(\d+)\.yaml", os.path.basename(p)).group(1)))
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
            full_name = meta.get('name') or meta.get('labels', {}).get('app') or ''
            ms_name = d.get('spec', {}).get('selector', {}).get('matchLabels', {}).get('name', '')
            conts = d.get('spec', {}).get('template', {}).get('spec', {}).get('containers', [])
            cpu = mem = ''
            if conts:
                reqs = conts[0].get('resources', {}).get('requests', {})
                cpu = str(reqs.get('cpu', '') or '')
                mem = str(reqs.get('memory', '') or '')
            if ms_name:
                ms_to_info[ms_name] = (full_name, cpu, mem)
    return ms_to_info


def main():
    ap = argparse.ArgumentParser(description='Generate presence-based vanilla_placement_session_presence.csv from polled cluster state')
    ap.add_argument('--presence-ndjson', required=True, help='Path to presence NDJSON stream')
    ap.add_argument('--workloads-dir', required=True, help='Path to workloads-vanilla directory')
    ap.add_argument('--experiment-dir', required=True, help='Experiment directory to write CSV')
    args = ap.parse_args()

    os.makedirs(args.experiment_dir, exist_ok=True)
    bounds = parse_presence(args.presence_ndjson)
    ms_to_info = read_workload_requests(args.workloads_dir)

    out_csv = os.path.join(args.experiment_dir, 'vanilla_placement_session_presence.csv')
    with open(out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['pod_id','node_id','start_slot','duration','cpu_request','ram_request'])
        for ms, b in bounds.items():
            full_name, cpu, mem = ms_to_info.get(ms, (ms, '', ''))
            start_slot = float(b['first_slot'])
            # Duration = observed slot span + 1
            duration = float((b['last_slot'] - b['first_slot']) + 1)
            node = b['node'] or ''
            w.writerow([full_name, node, start_slot, duration, cpu, mem])
    print(f"Wrote {len(bounds)} rows to {out_csv}")


if __name__ == '__main__':
    main()


