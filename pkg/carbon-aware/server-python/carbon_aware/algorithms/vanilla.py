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
import csv
import os

from carbon_aware.algorithms.base import SchedulingAlgorithm
from carbon_aware.models import CarbonAwarePod, CarbonAwareFlavour, CarbonAwareTimeslot
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
            logging.info(f"Vanilla placements will be logged to: {self._placement_csv_path}")

        except IOError as e:
            logging.error(f"Failed to open placement CSV file: {self._placement_csv_path}. Error: {e}")
            self._placement_csv_writer = None
            self._placement_csv_file_handle = None
            self._placement_csv_path = None

    def __del__(self):
        if hasattr(self, '_placement_csv_file_handle') and self._placement_csv_file_handle:
            try:
                self._placement_csv_file_handle.close()
                logging.info(f"Closed placement CSV file: {self._placement_csv_path}")
            except Exception as e:
                logging.error(f"Error closing placement CSV file in __del__: {e}")

    def _write_placement_to_csv(self, pod_id: str, node_id: str, start_slot: int, duration: float,
                                 cpu_request: float = 0.0, ram_request: float = 0.0) -> None:
        if self._placement_csv_writer and self._placement_csv_file_handle:
            try:
                self._placement_csv_writer.writerow([pod_id, node_id, start_slot, duration, cpu_request, ram_request])
                self._placement_csv_file_handle.flush()
            except Exception as e:
                logging.error(f"Error writing to placement CSV for vanilla: {e}")
        else:
            logging.warning("Placement CSV writer not available for vanilla algorithm. Cannot log placement.")

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
                                logging.info(f"[vanilla] Found pod {pod_id} in {filename} → earliest_timeslot={ts_num}")
                                return ts_num
                    except Exception as e:
                        logging.warning(f"[vanilla] Error reading {filepath}: {e}")
                        continue
        except Exception as e:
            logging.warning(f"[vanilla] Error scanning workloads directory {workloads_dir}: {e}")

        logging.warning(f"[vanilla] Pod {pod_id} not found in any timeslot_X.yaml file, defaulting to earliest_timeslot=0")
        return 0

    def _set_pod_earliest_timeslot(self, pod: CarbonAwarePod) -> None:
        pod.earliest_timeslot = self._extract_earliest_timeslot_from_yaml_files(pod.id)
        pod.calculate_deadline_slot()
        if pod.deadline_slot is not None:
            logging.info(f"[vanilla] Pod {pod.id} earliest_timeslot={pod.earliest_timeslot}, deadline_slot={pod.deadline_slot}")

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
        flavours: List[CarbonAwareFlavour],
        timeslots: List[CarbonAwareTimeslot],
        leftover_cpu: Dict[str, Dict[int, float]],
        leftover_ram: Dict[str, Dict[int, float]],
        max_time_slots: int = 48
    ) -> Tuple[Optional[CarbonAwareFlavour], Optional[CarbonAwareTimeslot], float]:
        """
        Find a placement using earliest-feasible-time, LeastAllocated node selection.

        Returns (node, timeslot, 0.0) where emissions are zero for vanilla.
        """
        # Ensure earliest_timeslot is set based on YAML source
        self._set_pod_earliest_timeslot(pod)

        start_time = time.time()
        considered_options = len(flavours) * len(timeslots)

        # Iterate timeslots in ascending order; pick the first where some node can host for full duration
        for ts in timeslots:
            if not is_timeslot_valid(ts, pod):
                continue

            # Collect feasible nodes for this start slot
            feasible_nodes: List[Tuple[CarbonAwareFlavour, float]] = []

            for flv in flavours:
                duration_feasible = True
                for slot_offset in range(int(pod.duration)):
                    current_slot = ts.id + slot_offset
                    if current_slot >= max_time_slots:
                        duration_feasible = False
                        break
                    if (leftover_cpu[flv.id][current_slot] < pod.cpuRequest or
                        leftover_ram[flv.id][current_slot] < pod.ramRequest):
                        duration_feasible = False
                        break

                if duration_feasible:
                    # Score node by LeastAllocated fraction
                    avail_cpu_now = min(
                        leftover_cpu[flv.id].get(ts.id, 0.0),
                        flv.totalCpu
                    )
                    avail_ram_now = min(
                        leftover_ram[flv.id].get(ts.id, 0.0),
                        flv.totalRam
                    )
                    score = self._least_allocated_score(
                        available_cpu=avail_cpu_now,
                        total_cpu=flv.totalCpu,
                        available_ram=avail_ram_now,
                        total_ram=flv.totalRam,
                    )
                    feasible_nodes.append((flv, score))

            if feasible_nodes:
                # Choose node with highest score
                feasible_nodes.sort(key=lambda x: x[1], reverse=True)
                chosen_node = feasible_nodes[0][0]

                # Log placement
                self._write_placement_to_csv(
                    pod_id=pod.id,
                    node_id=chosen_node.id,
                    start_slot=ts.id,
                    duration=pod.duration,
                    cpu_request=pod.cpuRequest,
                    ram_request=pod.ramRequest,
                )

                # Optionally record experiment metrics
                if self.experiment_logger:
                    execution_time = time.time() - start_time
                    self.experiment_logger.record_placement(
                        pod_id=pod.id,
                        success=True,
                        execution_time=execution_time,
                        emissions=0.0,
                        considered_options=considered_options,
                        selected_node=chosen_node.id,
                        selected_timeslot=ts.id,
                    )

                # Vanilla has no emissions model; return 0.0
                return chosen_node, ts, 0.0

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


