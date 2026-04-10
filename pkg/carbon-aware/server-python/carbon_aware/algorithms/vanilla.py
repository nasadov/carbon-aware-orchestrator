"""
Vanilla Kubernetes-like scheduler simulation.

This algorithm ignores carbon signals and schedules pods using a simple
Kubernetes-default-inspired scoring: choose the earliest feasible timeslot,
then pick the node with highest LeastAllocated score (prefer more free nodes),
subject to CPU/RAM feasibility across the pod's entire duration.

It writes placements to a session CSV compatible with downstream analyses.
"""
import logging
import time
from typing import Dict, List, Optional, Tuple
from collections import deque
import csv
import os

from carbon_aware.algorithms.base import SchedulingAlgorithm
from carbon_aware.models import CarbonAwarePod, CarbonAwareTimeslot, EnvironmentalFlavor
from carbon_aware.utils import is_timeslot_valid


class VanillaAlgorithm(SchedulingAlgorithm):
    """
    Vanilla scheduler simulation (resource-only, no carbon-awareness).
    """

    _placement_csv_filename = "vanilla_placement_session.csv"

    def __init__(self, perf_logger=None):
        self.experiment_logger = None
        self.perf_logger = perf_logger
        self._session_log_dir: Optional[str] = None
        self._placement_csv_file_handle = None
        self._placement_csv_writer = None
        self._placement_csv_path: Optional[str] = None
        self._workloads_dir: Optional[str] = None
        # Buffered CSV writing to reduce I/O in hot path
        self._csv_buffer: List[List[object]] = []
        self._csv_buffer_limit: int = 200
        # Hot-path arrays mirroring leftover dicts
        self._node_index: Dict[str, int] = {}
        self._cpu_array: Optional[List[List[float]]] = None
        self._ram_array: Optional[List[List[float]]] = None
        self._horizon: int = 0
        # K8s-like pruning knobs
        self.percentage_of_nodes_to_score: float = 50.0
        self.min_nodes_to_score: int = 50
        self.max_nodes_to_score: int = 1000
        self.early_feasible_break_k: int = 16

    @staticmethod
    def _ensure_dir_exists(directory_path: str) -> None:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            logging.info(f"Created directory: {directory_path}")

    def set_base_log_dir(self, base_dir: str):
        """Sets the base directory for session logs."""
        self._session_log_dir = base_dir

    def set_workloads_dir(self, workloads_dir: str):
        """Set the workloads directory for YAML file lookup (to derive earliest_timeslot)."""
        self._workloads_dir = workloads_dir
        logging.info(f"VanillaAlgorithm: workloads directory set to {workloads_dir}")

    def setup_session_placement_log(self) -> None:
        """Initialize CSV writer for placements across the session."""
        if not self._session_log_dir:
            logging.error("Session log directory not set. Cannot initialize placement CSV logging for vanilla algorithm.")
            # Fallback to a repo-local analysis directory to avoid crashing
            self._session_log_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "analysis", "vanilla_fallback_logs")
            )
            VanillaAlgorithm._ensure_dir_exists(self._session_log_dir)
            logging.warning(f"Using fallback log directory for placements: {self._session_log_dir}")

        self._placement_csv_path = os.path.join(self._session_log_dir, self._placement_csv_filename)

        # Close any previous handle safely
        if hasattr(self, "_placement_csv_file_handle") and self._placement_csv_file_handle:
            try:
                self._placement_csv_file_handle.close()
            except Exception as e:
                logging.error(f"Error closing previous placement CSV file: {e}")

        try:
            file_exists_and_not_empty = os.path.exists(self._placement_csv_path) and os.path.getsize(self._placement_csv_path) > 0

            self._placement_csv_file_handle = open(self._placement_csv_path, 'a', newline='')
            self._placement_csv_writer = csv.writer(self._placement_csv_file_handle)

            if not file_exists_and_not_empty:
                self._placement_csv_writer.writerow(["pod_id", "node_id", "start_slot", "duration", "cpu_request", "ram_request"])
                self._placement_csv_file_handle.flush()
            logging.debug(f"Vanilla placements will be logged to: {self._placement_csv_path}")

        except IOError as e:
            logging.error(f"Failed to open placement CSV file: {self._placement_csv_path}. Error: {e}")
            self._placement_csv_writer = None
            self._placement_csv_file_handle = None
            self._placement_csv_path = None

    def __del__(self):
        if hasattr(self, '_placement_csv_file_handle') and self._placement_csv_file_handle:
            try:
                # Flush buffered rows before closing
                self.flush_session_placement_log()
                self._placement_csv_file_handle.close()
                logging.debug(f"Closed placement CSV file: {self._placement_csv_path}")
            except Exception as e:
                logging.error(f"Error closing placement CSV file in __del__: {e}")

    def _write_placement_to_csv(self, pod_id: str, node_id: str, start_slot: int, duration: float,
                                 cpu_request: float = 0.0, ram_request: float = 0.0) -> None:
        if self._placement_csv_writer and self._placement_csv_file_handle:
            try:
                self._csv_buffer.append([pod_id, node_id, start_slot, duration, cpu_request, ram_request])
                if len(self._csv_buffer) >= self._csv_buffer_limit:
                    self._placement_csv_writer.writerows(self._csv_buffer)
                    self._csv_buffer.clear()
            except Exception as e:
                logging.error(f"Error writing to placement CSV for vanilla: {e}")
        else:
            logging.warning("Placement CSV writer not available for vanilla algorithm. Cannot log placement.")

    def flush_session_placement_log(self) -> None:
        if self._placement_csv_writer and self._placement_csv_file_handle and self._csv_buffer:
            try:
                self._placement_csv_writer.writerows(self._csv_buffer)
                self._csv_buffer.clear()
                self._placement_csv_file_handle.flush()
            except Exception as e:
                logging.error(f"Error flushing placement CSV for vanilla: {e}")

    def _extract_earliest_timeslot_from_yaml_files(self, pod_id: str) -> int:
        """
        Determine earliest_timeslot for a pod by scanning timeslot_*.yaml filenames
        in the workloads directory for the pod name occurrence.
        """
        import re

        workloads_dir = self._workloads_dir or "../workloads"
        if not os.path.isabs(workloads_dir):
            workloads_dir = os.path.join(os.getcwd(), workloads_dir)

        try:
            for filename in os.listdir(workloads_dir):
                if re.match(r"timeslot_(\d+)\.yaml$", filename):
                    filepath = os.path.join(workloads_dir, filename)
                    m = re.match(r"timeslot_(\d+)\.yaml$", filename)
                    ts_num = int(m.group(1)) if m else 0
                    try:
                        with open(filepath, 'r') as f:
                            content = f.read()
                            if f"name: {pod_id}" in content:
                                logging.debug(f"[vanilla] Found pod {pod_id} in {filename} → earliest_timeslot={ts_num}")
                                return ts_num
                    except Exception as e:
                        logging.warning(f"[vanilla] Error reading {filepath}: {e}")
                        continue
        except Exception as e:
            logging.warning(f"[vanilla] Error scanning workloads directory {workloads_dir}: {e}")

        logging.warning(f"[vanilla] Pod {pod_id} not found in any timeslot_X.yaml file, defaulting to earliest_timeslot=0")
        return 0

    def _set_pod_earliest_timeslot(self, pod: CarbonAwarePod) -> None:
        # Respect pre-set earliest_timeslot (from precompute) to avoid directory scans
        if hasattr(pod, '_earliest_timeslot_source') and pod._earliest_timeslot_source == 'precompute':
            pod.calculate_deadline_slot()
            return
        if hasattr(pod, 'earliest_timeslot') and pod.earliest_timeslot not in (None, 0):
            pod.calculate_deadline_slot()
            return
        pod.earliest_timeslot = self._extract_earliest_timeslot_from_yaml_files(pod.id)
        pod.calculate_deadline_slot()
        if pod.deadline_slot is not None:
            logging.debug(f"[vanilla] Pod {pod.id} earliest_timeslot={pod.earliest_timeslot}, deadline_slot={pod.deadline_slot}")

    def _initialize_leftover_arrays(
        self,
        flavours: List[EnvironmentalFlavor],
        leftover_cpu: Dict[str, Dict[int, float]],
        leftover_ram: Dict[str, Dict[int, float]],
        max_time_slots: int,
    ) -> None:
        need_init = (
            self._cpu_array is None or self._ram_array is None or
            self._horizon != max_time_slots or
            len(self._node_index) != len(flavours)
        )
        if not need_init:
            return
        self._node_index = {flv.id: idx for idx, flv in enumerate(flavours)}
        self._horizon = max_time_slots
        cpu_array: List[List[float]] = []
        ram_array: List[List[float]] = []
        for flv in flavours:
            node_cpu = [0.0] * max_time_slots
            node_ram = [0.0] * max_time_slots
            lc = leftover_cpu.get(flv.id, {})
            lr = leftover_ram.get(flv.id, {})
            for s in range(max_time_slots):
                node_cpu[s] = lc.get(s, 0.0)
                node_ram[s] = lr.get(s, 0.0)
            cpu_array.append(node_cpu)
            ram_array.append(node_ram)
        self._cpu_array = cpu_array
        self._ram_array = ram_array

    def _sliding_window_min(self, arr: List[float], window: int) -> List[float]:
        """Compute sliding window minimum for an array.
        Returns a list of length len(arr) - window + 1 containing the minimum of each window.
        """
        n = len(arr)
        if window <= 1:
            return arr[:]
        if window > n:
            return []
        d: deque[int] = deque()
        out: List[float] = []
        for i, val in enumerate(arr):
            while d and arr[d[-1]] >= val:
                d.pop()
            d.append(i)
            # Remove indices out of window
            if d[0] <= i - window:
                d.popleft()
            if i >= window - 1:
                out.append(arr[d[0]])
        return out

    def note_allocation(self, node_id: str, start_slot: int, duration_hours: float, cpu_req: float, ram_req: float) -> None:
        if self._cpu_array is None or self._ram_array is None:
            return
        idx = self._node_index.get(node_id)
        if idx is None:
            return
        duration_slots = int(duration_hours)
        end = min(start_slot + duration_slots, self._horizon)
        arr_cpu = self._cpu_array[idx]
        arr_ram = self._ram_array[idx]
        for s in range(start_slot, end):
            arr_cpu[s] -= cpu_req
            arr_ram[s] -= ram_req

    @property
    def name(self) -> str:
        return "K8s-Vanilla"

    @staticmethod
    def _least_allocated_score(available_cpu: float, total_cpu: float, available_ram: float, total_ram: float) -> float:
        """
        Approximate Kubernetes NodeResourcesFit LeastAllocated scoring.
        Higher is better: prefer nodes with larger available fractions of CPU and Memory.
        """
        cpu_ratio = (available_cpu / total_cpu) if total_cpu > 0 else 0.0
        mem_ratio = (available_ram / total_ram) if total_ram > 0 else 0.0
        # Weight CPU and memory equally
        return 0.5 * cpu_ratio + 0.5 * mem_ratio

    def find_placement(
        self,
        pod: CarbonAwarePod,
        flavours: List[EnvironmentalFlavor],
        timeslots: List[CarbonAwareTimeslot],
        leftover_cpu: Dict[str, Dict[int, float]],
        leftover_ram: Dict[str, Dict[int, float]],
        max_time_slots: int = 48
    ) -> Tuple[Optional[EnvironmentalFlavor], Optional[CarbonAwareTimeslot], float]:
        """
        Find a placement using earliest-feasible-time, LeastAllocated node selection.

        Returns (node, timeslot, 0.0) where emissions are zero for vanilla.
        """
        # Ensure earliest_timeslot is set based on YAML source
        self._set_pod_earliest_timeslot(pod)

        start_time = time.time()
        considered_options = len(flavours) * len(timeslots)

        # Prepare array views
        self._initialize_leftover_arrays(flavours, leftover_cpu, leftover_ram, max_time_slots)
        # Reset per-call window caches
        self._cpu_window_cache = {}
        self._ram_window_cache = {}
        cpu_array = self._cpu_array or []
        ram_array = self._ram_array or []

        duration_slots = int(pod.duration)
        earliest_slot = max(int(getattr(pod, 'earliest_timeslot', 0)), 0)
        deadline_limit = int(pod.deadline_slot) if getattr(pod, 'deadline_slot', None) is not None else max_time_slots
        latest_start = min(deadline_limit - duration_slots, max_time_slots - duration_slots)

        if latest_start < earliest_slot:
            if self.experiment_logger:
                execution_time = time.time() - start_time
                self.experiment_logger.record_placement(
                    pod_id=pod.id,
                    success=False,
                    execution_time=execution_time,
                    emissions=0.0,
                    considered_options=0,
                    selected_node=None,
                    selected_timeslot=None,
                )
            return None, None, float('inf')

        for ts_id in range(earliest_slot, latest_start + 1):
            ts = timeslots[ts_id]
            if not is_timeslot_valid(ts, pod):
                continue

            total_nodes = len(flavours)
            target_count = int(min(
                self.max_nodes_to_score,
                max(self.min_nodes_to_score, int((self.percentage_of_nodes_to_score / 100.0) * total_nodes))
            ))
            target_count = min(target_count, total_nodes)
            nodes_to_consider = flavours[:target_count]

            best_node = None
            best_score = -1.0
            feasible_found = 0
            req_cpu = pod.cpuRequest
            req_ram = pod.ramRequest

            for flv in nodes_to_consider:
                idx = self._node_index.get(flv.id)
                if idx is None:
                    continue
                end_slot = ts_id + duration_slots
                if end_slot > max_time_slots:
                    continue
                arr_cpu = cpu_array[idx]
                arr_ram = ram_array[idx]

                # Sliding-window feasibility: ensure min CPU/RAM across the pod window >= request
                if not hasattr(self, '_cpu_window_cache'):
                    self._cpu_window_cache = {}
                    self._ram_window_cache = {}
                cache_key = (idx, duration_slots)
                if cache_key not in self._cpu_window_cache:
                    self._cpu_window_cache[cache_key] = self._sliding_window_min(arr_cpu, duration_slots)
                    self._ram_window_cache[cache_key] = self._sliding_window_min(arr_ram, duration_slots)
                cpu_min_series = self._cpu_window_cache[cache_key]
                ram_min_series = self._ram_window_cache[cache_key]
                if ts_id >= len(cpu_min_series) or ts_id >= len(ram_min_series):
                    continue
                if cpu_min_series[ts_id] < req_cpu or ram_min_series[ts_id] < req_ram:
                    continue

                feasible_found += 1
                avail_cpu_now = min(arr_cpu[ts_id], flv.totalCpu)
                avail_ram_now = min(arr_ram[ts_id], flv.totalRam)
                score = self._least_allocated_score(
                    available_cpu=avail_cpu_now,
                    total_cpu=flv.totalCpu,
                    available_ram=avail_ram_now,
                    total_ram=flv.totalRam,
                )
                if score > best_score:
                    best_score = score
                    best_node = flv
                if feasible_found >= self.early_feasible_break_k:
                    break

            if best_node is not None:
                self._write_placement_to_csv(
                    pod_id=pod.id,
                    node_id=best_node.id,
                    start_slot=ts_id,
                    duration=pod.duration,
                    cpu_request=pod.cpuRequest,
                    ram_request=pod.ramRequest,
                )
                if self.experiment_logger:
                    execution_time = time.time() - start_time
                    self.experiment_logger.record_placement(
                        pod_id=pod.id,
                        success=True,
                        execution_time=execution_time,
                        emissions=0.0,
                        considered_options=considered_options,
                        selected_node=best_node.id,
                        selected_timeslot=ts_id,
                    )
                return best_node, ts, 0.0

        # If we reach here, no feasible placement was found
        if self.experiment_logger:
            execution_time = time.time() - start_time
            self.experiment_logger.record_placement(
                pod_id=pod.id,
                success=False,
                execution_time=execution_time,
                emissions=0.0,
                considered_options=considered_options,
                selected_node=None,
                selected_timeslot=None,
            )

        return None, None, float('inf')

