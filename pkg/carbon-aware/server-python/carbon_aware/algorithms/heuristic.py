"""
Carbon-aware scheduling heuristic algorithm implementation.
"""
import logging
import time
from typing import Dict, List, Optional, Tuple
import csv
import os
from datetime import datetime

from carbon_aware.algorithms.base import SchedulingAlgorithm
from carbon_aware.models import CarbonAwarePod, CarbonAwareFlavour, CarbonAwareTimeslot
from carbon_aware.utils import is_timeslot_valid, compute_emissions


class HeuristicAlgorithm(SchedulingAlgorithm):
    """
    Heuristic implementation of the carbon-aware scheduling algorithm.
    """
    _placement_csv_filename_suffix = "heuristic_placements_session.csv"

    def __init__(self, perf_logger=None):
        self.experiment_logger = None
        self.perf_logger = perf_logger
        self._session_log_dir: Optional[str] = None
        
        self._placement_csv_file_handle = None
        self._placement_csv_writer = None
        self._placement_csv_path: Optional[str] = None

    @classmethod
    def _ensure_dir_exists(cls, directory_path: str):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            logging.info(f"Created directory: {directory_path}")

    def set_base_log_dir(self, base_dir: str):
        """Sets the base directory for session logs."""
        self._session_log_dir = base_dir

    def setup_session_placement_log(self):
        """Sets up a single CSV file for logging all placements during the session."""
        if not self._session_log_dir:
            logging.error("Session log directory not set. Cannot initialize placement CSV logging for heuristic algorithm.")
            self._session_log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "analysis", "heuristic_fallback_logs"))
            HeuristicAlgorithm._ensure_dir_exists(self._session_log_dir)
            logging.warning(f"Using fallback log directory for placements: {self._session_log_dir}")

        self._placement_csv_path = os.path.join(self._session_log_dir, self._placement_csv_filename_suffix)
        
        if hasattr(self, '_placement_csv_file_handle') and self._placement_csv_file_handle:
            try:
                self._placement_csv_file_handle.close()
            except Exception as e:
                logging.error(f"Error closing previous placement CSV file: {e}")
        
        try:
            file_exists_and_not_empty = os.path.exists(self._placement_csv_path) and os.path.getsize(self._placement_csv_path) > 0
            
            self._placement_csv_file_handle = open(self._placement_csv_path, 'a', newline='')
            self._placement_csv_writer = csv.writer(self._placement_csv_file_handle)
            
            if not file_exists_and_not_empty:
                self._placement_csv_writer.writerow(["pod_id", "node_id", "start_slot", "duration", "cpu_request"])
                self._placement_csv_file_handle.flush()
            logging.info(f"Heuristic placements will be logged to: {self._placement_csv_path}")

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

    def _write_placement_to_csv(self, pod_id: str, node_id: str, start_slot: int, duration: float, cpu_request: float):
        if self._placement_csv_writer and self._placement_csv_file_handle:
            try:
                # Revert: Write cpu_request as is (assuming it's already in the desired unit or not used for proportional height)
                self._placement_csv_writer.writerow([pod_id, node_id, start_slot, duration, cpu_request])
                self._placement_csv_file_handle.flush()
                logging.debug(f"Logged placement to CSV: {pod_id}, {node_id}, {start_slot}, {duration}, {cpu_request}")
            except Exception as e:
                logging.error(f"Error writing to placement CSV for heuristic: {e}")
        else:
            logging.warning("Placement CSV writer not available for heuristic algorithm. Cannot log placement.")
    
    @property
    def name(self) -> str:
        return "Carbon-Aware-Heuristic"
    
    def find_placement(
        self,
        pod: CarbonAwarePod,
        flavours: List[CarbonAwareFlavour],
        timeslots: List[CarbonAwareTimeslot],
        leftover_cpu: Dict[str, Dict[int, float]],
        leftover_ram: Dict[str, Dict[int, float]],
        max_time_slots: int = 48
    ) -> Tuple[Optional[CarbonAwareFlavour], Optional[CarbonAwareTimeslot], float]:
        """Find the best placement for a pod using the carbon-aware heuristic."""
        start_time = time.time()
        considered_options = len(flavours) * len(timeslots)
        
        best_node, best_slot, emissions = find_best_node_and_timeslot(
            pod, flavours, timeslots, leftover_cpu, leftover_ram, max_time_slots
        )
        
        if self.experiment_logger:
            execution_time = time.time() - start_time
            success = best_node is not None and best_slot is not None
            
            self.experiment_logger.record_placement(
                pod_id=pod.id,
                success=success,
                execution_time=execution_time,
                emissions=emissions if success else 0.0,
                considered_options=considered_options,
                selected_node=best_node.id if best_node else None,
                selected_timeslot=best_slot.id if best_slot else None
            )

        if best_node and best_slot:
            self._write_placement_to_csv(
                pod_id=pod.id,
                node_id=best_node.id,
                start_slot=best_slot.id,
                duration=pod.duration,
                cpu_request=pod.cpuRequest
            )
        
        return best_node, best_slot, emissions


def find_best_node_and_timeslot(
    pod: CarbonAwarePod,
    flavours: List[CarbonAwareFlavour],
    timeslots: List[CarbonAwareTimeslot],
    leftover_cpu: Dict[str, Dict[int, float]],
    leftover_ram: Dict[str, Dict[int, float]],
    max_time_slots: int = 48
) -> Tuple[Optional[CarbonAwareFlavour], Optional[CarbonAwareTimeslot], float]:
    """
    Find the best node and timeslot for a pod that minimizes carbon emissions.
    
    The algorithm iterates through all valid combinations of nodes and timeslots,
    checking resource constraints and calculating emissions for each.
    
    Args:
        pod: The pod to place
        flavours: Available node types
        timeslots: Available scheduling timeslots
        leftover_cpu: Remaining CPU capacity per node and timeslot
        leftover_ram: Remaining RAM capacity per node and timeslot
        max_time_slots: Maximum number of timeslots to consider
        
    Returns:
        Tuple of (best_node, best_timeslot, emissions) or (None, None, inf) if no placement found
    """
    best_node = None
    best_slot = None
    minimal_emissions = float('inf')

    for ts in timeslots:
        if not is_timeslot_valid(ts, pod):
            logging.debug(f"[find_best_node_and_timeslot] Skipping timeslot={ts.id}, not valid for pod={pod.id}")
            continue

        for flv in flavours:
            duration_feasible = True
            for slot_offset in range(int(pod.duration)):
                current_slot = ts.id + slot_offset
                if current_slot >= max_time_slots:
                    duration_feasible = False
                    logging.debug(f"[find_best_node_and_timeslot] Slot {ts.id}+{slot_offset}={current_slot} exceeds tracking window for pod={pod.id}")
                    break
                
                if (leftover_cpu[flv.id][current_slot] < pod.cpuRequest or 
                    leftover_ram[flv.id][current_slot] < pod.ramRequest):
                    duration_feasible = False
                    logging.debug(
                        f"[find_best_node_and_timeslot] Slot {current_slot} on {flv.id} doesn't have enough resources for pod={pod.id}: " +
                        f"CPU {leftover_cpu[flv.id][current_slot]:.2f}/{pod.cpuRequest:.2f}, " +
                        f"RAM {leftover_ram[flv.id][current_slot]:.0f}/{pod.ramRequest:.0f}"
                    )
                    break

            if duration_feasible:
                total_emi = compute_emissions(flv, ts.id, pod)
                if total_emi < minimal_emissions:
                    minimal_emissions = total_emi
                    best_node = flv
                    best_slot = ts
                    logging.debug(
                        f"[find_best_node_and_timeslot] New best found for pod={pod.id}: "
                        f"node={best_node.id}, timeslot={best_slot.id}, emissions={minimal_emissions:.3f}"
                    )

    return best_node, best_slot, minimal_emissions