#!/usr/bin/env python3
import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time


def tail_performance_log(perf_log_path: str, slot_queue: queue.Queue, stop_event: threading.Event) -> None:
    pattern = re.compile(r"Processing timeslot\s+(\d+)/(\d+)")
    while not os.path.exists(perf_log_path) and not stop_event.is_set():
        time.sleep(0.1)
    if not os.path.exists(perf_log_path):
        return
    try:
        with open(perf_log_path, 'r') as f:
            f.seek(0, os.SEEK_END)
            while not stop_event.is_set():
                pos = f.tell()
                line = f.readline()
                if not line:
                    time.sleep(0.1)
                    f.seek(pos)
                    continue
                m = pattern.search(line)
                if m:
                    try:
                        idx = int(m.group(1))
                        slot = max(0, idx - 1)
                        slot_queue.put(slot)
                    except Exception:
                        continue
    except Exception:
        return


def poll_pods_every(interval_s: float, presence_out_path: str, slot_queue: queue.Queue, stop_event: threading.Event, namespace: str = 'default') -> None:
    current_slot = 0
    def updater():
        nonlocal current_slot
        while not stop_event.is_set():
            try:
                s = slot_queue.get(timeout=0.2)
                if isinstance(s, int):
                    current_slot = s
            except queue.Empty:
                pass
    t = threading.Thread(target=updater, daemon=True)
    t.start()

    while not stop_event.is_set():
        t_ms = int(time.time() * 1000)
        try:
            out = subprocess.check_output(
                ['kubectl', 'get', 'pods', '-n', namespace, '-o', 'json'],
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except Exception:
            time.sleep(interval_s)
            continue
        try:
            data = json.loads(out)
        except Exception:
            time.sleep(interval_s)
            continue
        items = data.get('items', []) if isinstance(data, dict) else []
        rec = {
            't': t_ms,
            'slot': int(current_slot),
            'pods': []
        }
        for it in items:
            meta = it.get('metadata', {})
            spec = it.get('spec', {})
            status = it.get('status', {})
            name = str(meta.get('name', '') or '')
            phase = str(status.get('phase', '') or '')
            node = str(spec.get('nodeName', '') or '')
            # Capture only pods with our naming pattern m\d{3,}
            if not re.search(r"m\d{3,}", name):
                continue
            rec['pods'].append({'name': name, 'node': node, 'phase': phase})
        try:
            with open(presence_out_path, 'a') as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass
        time.sleep(interval_s)


def main():
    ap = argparse.ArgumentParser(description='Cluster state watcher: poll pods every 0.5s and log presence with timeslot')
    ap.add_argument('--perf-log', required=True, help='Path to performance.log to read current timeslot')
    ap.add_argument('--presence-out', required=True, help='Path to write presence NDJSON stream')
    ap.add_argument('--interval', type=float, default=0.5, help='Polling interval seconds (default 0.5)')
    ap.add_argument('--namespace', default='default', help='Kubernetes namespace (default: default)')
    args = ap.parse_args()

    stop_event = threading.Event()
    slot_queue: queue.Queue = queue.Queue()

    t_perf = threading.Thread(target=tail_performance_log, args=(args.perf_log, slot_queue, stop_event), daemon=True)
    t_perf.start()
    try:
        poll_pods_every(args.interval, args.presence_out, slot_queue, stop_event, args.namespace)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        try:
            t_perf.join(timeout=1.0)
        except Exception:
            pass


if __name__ == '__main__':
    main()


