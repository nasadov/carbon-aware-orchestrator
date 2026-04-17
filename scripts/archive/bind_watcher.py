#!/usr/bin/env python3
"""Archived Kubernetes bind-event watcher."""
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
    # Wait until file exists
    while not os.path.exists(perf_log_path) and not stop_event.is_set():
        time.sleep(0.1)
    if not os.path.exists(perf_log_path):
        return
    try:
        with open(perf_log_path, 'r') as f:
            # seek to end to follow only new lines
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
                        # convert 1-based to 0-based index
                        slot = max(0, idx - 1)
                        slot_queue.put(slot)
                    except Exception:
                        continue
    except Exception:
        # Silent exit on errors; watcher is best-effort
        return


def stream_json_lines(proc: subprocess.Popen, stop_event: threading.Event):
    buf = ''
    depth = 0
    for chunk in iter(lambda: proc.stdout.read(1), ''):
        if stop_event.is_set():
            break
        if not chunk:
            time.sleep(0.01)
            continue
        buf += chunk
        if chunk == '{':
            depth += 1
        elif chunk == '}':
            depth -= 1
        if depth <= 0 and buf.strip():
            try:
                obj = json.loads(buf)
                yield obj
            except Exception:
                pass
            buf = ''
            depth = 0


def events_watcher(bind_out_path: str, slot_queue: queue.Queue, stop_event: threading.Event) -> None:
    # Current timeslot index as communicated by performance.log tailer
    current_slot = 0
    emitted = set()  # ms_name already emitted
    ms_re = re.compile(r"(m\d{3,})")
    node_re = re.compile(r" to ([^ ]+)$")

    # Start kubectl events stream
    try:
        proc = subprocess.Popen(
            ['kubectl', 'get', 'events', '-n', 'default', '-w', '-o', 'json'],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
    except Exception:
        return

    # Thread to pull slot updates
    def apply_slot_updates():
        nonlocal current_slot
        while not stop_event.is_set():
            try:
                s = slot_queue.get(timeout=0.2)
                if isinstance(s, int):
                    current_slot = s
            except queue.Empty:
                pass

    t_slot = threading.Thread(target=apply_slot_updates, daemon=True)
    t_slot.start()

    # Consume JSON objects from the event stream
    for raw in stream_json_lines(proc, stop_event):
        if stop_event.is_set():
            break
        try:
            evt = raw.get('object') if isinstance(raw, dict) and 'object' in raw else raw
            if not isinstance(evt, dict):
                continue
            reason = str(evt.get('reason', '')).strip()
            if reason != 'Scheduled':
                continue
            inv = evt.get('involvedObject') or {}
            pod_name = str(inv.get('name', '') or '')
            mm = ms_re.search(pod_name)
            if not mm:
                continue
            ms_name = mm.group(1)
            message = str(evt.get('message', '') or evt.get('note', '') or '')
            node = ''
            mnode = node_re.search(message)
            if mnode:
                node = mnode.group(1)
            if not node:
                continue
            if ms_name in emitted:
                continue
            emitted.add(ms_name)
            now_ms = int(time.time() * 1000)
            rec = {'t': now_ms, 'name': ms_name, 'node': node, 'slot': int(current_slot)}
            try:
                with open(bind_out_path, 'a') as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception:
                pass
        except Exception:
            continue

    try:
        proc.terminate()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description='Bind-time watcher to record (ms_name,node,slot) at Scheduled events')
    ap.add_argument('--perf-log', required=True, help='Path to performance.log to read current timeslot')
    ap.add_argument('--binds-out', required=True, help='Path to write bind events NDJSON')
    args = ap.parse_args()

    stop_event = threading.Event()
    slot_queue: queue.Queue = queue.Queue()

    # Start tailer for performance.log and events watcher
    t_perf = threading.Thread(target=tail_performance_log, args=(args.perf_log, slot_queue, stop_event), daemon=True)
    t_perf.start()

    try:
        events_watcher(args.binds_out, slot_queue, stop_event)
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



