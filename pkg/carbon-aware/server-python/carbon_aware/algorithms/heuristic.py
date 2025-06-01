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
                self._placement_csv_writer.writerow(["pod_id", "node_id", "start_slot", "duration", "cpu_request", "ram_request"])
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

    def _write_placement_to_csv(self, pod_id: str, node_id: str, start_slot: int, duration: float, cpu_request: float = 0.0, ram_request: float = 0.0):
        if self._placement_csv_writer and self._placement_csv_file_handle:
            try:
                self._placement_csv_writer.writerow([pod_id, node_id, start_slot, duration, cpu_request, ram_request])
                self._placement_csv_file_handle.flush()
            except Exception as e:
                logging.error(f"Error writing to placement CSV for heuristic: {e}")
        else:
            logging.warning("Placement CSV writer not available for heuristic algorithm. Cannot log placement.")
    
    def _set_pod_earliest_timeslot(self, pod: CarbonAwarePod):
        """
        Ensure the pod has earliest_timeslot set and calculate deadline_slot.
        
        This method validates that the pod has its earliest_timeslot properly set
        (usually done during pod creation from YAML files) and calculates the
        deadline_slot relative to that earliest_timeslot.
        
        If earliest_timeslot is not set, it extracts it from the timeslot YAML file
        that contains this pod's definition.
        """
        # 🔧 FORCE re-extraction from YAML files to get correct earliest_timeslot
        # Don't trust the default value of 0 - always check the actual YAML source
        pod.earliest_timeslot = self._extract_earliest_timeslot_from_yaml_files(pod.id)
        logging.info(f"🔒 Pod {pod.id} earliest_timeslot extracted from YAML file: {pod.earliest_timeslot}")
        
        logging.info(f"🔒 Pod {pod.id} has earliest_timeslot={pod.earliest_timeslot}")
        
        # Calculate deadline_slot relative to earliest_timeslot
        pod.calculate_deadline_slot()
        if pod.deadline_slot is not None:
            logging.info(f"📅 Pod {pod.id} has deadline_slot={pod.deadline_slot} (earliest_timeslot + {pod.deadline_hours}h)")
    
    def _extract_earliest_timeslot_from_yaml_files(self, pod_id: str) -> int:
        """
        Extract the earliest timeslot by finding which timeslot_X.yaml file contains this pod.
        
        This is the CORRECT way to determine earliest timeslot - by looking at which 
        timeslot file the pod came from. If pod is in timeslot_4.yaml, then earliest_timeslot=4.
        """
        import os
        import re
        
        # Look in the workloads directory for timeslot_*.yaml files
        workloads_dir = "/root/carbon-aware-orchestrator/pkg/carbon-aware/workloads"
        
        try:
            for filename in os.listdir(workloads_dir):
                if re.match(r'timeslot_(\d+)\.yaml$', filename):
                    filepath = os.path.join(workloads_dir, filename)
                    
                    # Extract timeslot number from filename
                    match = re.match(r'timeslot_(\d+)\.yaml$', filename)
                    timeslot_num = int(match.group(1))
                    
                    # Check if this pod is defined in this file
                    try:
                        with open(filepath, 'r') as f:
                            content = f.read()
                            # Look for the pod name in deployment metadata
                            if f"name: {pod_id}" in content:
                                logging.info(f"Found pod {pod_id} in {filename} → earliest_timeslot={timeslot_num}")
                                return timeslot_num
                    except Exception as e:
                        logging.warning(f"⚠️ Error reading {filepath}: {e}")
                        continue
        except Exception as e:
            logging.warning(f"⚠️ Error scanning workloads directory {workloads_dir}: {e}")
        
        # Fallback: if not found in any timeslot file, default to 0
        logging.warning(f"⚠️ Pod {pod_id} not found in any timeslot_X.yaml file, defaulting to earliest_timeslot=0")
        return 0
    
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
        # Set earliest_timeslot based on pod ID before scheduling
        self._set_pod_earliest_timeslot(pod)
        
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
                cpu_request=pod.cpuRequest,
                ram_request=pod.ramRequest
            )
        
        return best_node, best_slot, emissions

    def find_placement_atomic(
        self,
        pod: CarbonAwarePod,
        flavours: List[CarbonAwareFlavour],
        timeslots: List[CarbonAwareTimeslot],
        persistent_state,
        max_time_slots: int = 48
    ) -> Tuple[Optional[CarbonAwareFlavour], Optional[CarbonAwareTimeslot], float]:
        """
        Find and atomically allocate the best placement for a pod.
        
        This method prevents race conditions by using atomic constraint checking
        and resource allocation within the persistent state.
        
        Args:
            pod: The pod to place
            flavours: Available node types
            timeslots: Available scheduling timeslots
            persistent_state: PersistentStateStorage instance for atomic operations
            max_time_slots: Maximum number of timeslots to consider
            
        Returns:
            Tuple of (best_node, best_timeslot, emissions) or (None, None, inf) if no placement found
        """
        # CRITICAL FIX: Set earliest_timeslot based on pod ID before scheduling
        self._set_pod_earliest_timeslot(pod)
        
        logging.info(f"[find_placement_atomic] Starting placement search for pod={pod.id}")
        logging.info(f"    Pod constraints: earliest_timeslot={pod.earliest_timeslot}, duration={pod.duration}h")
        logging.info(f"    Resource requirements: CPU={pod.cpuRequest:.3f}, RAM={pod.ramRequest:.0f}MB")
        logging.info(f"    Search space: {len(flavours)} nodes × {len(timeslots)} timeslots = {len(flavours) * len(timeslots)} combinations")
        
        candidates = []  # List of (node, timeslot, emissions) tuples
        
        # First pass: find all feasible placements and calculate their emissions
        valid_timeslots = 0
        for ts in timeslots:
            if not is_timeslot_valid(ts, pod):
                logging.debug(f"[find_placement_atomic] Timeslot {ts.id} invalid for pod={pod.id} (earliest_timeslot={pod.earliest_timeslot})")
                continue
            
            valid_timeslots += 1
            logging.debug(f"[find_placement_atomic] Timeslot {ts.id} valid for pod={pod.id}")

            for flv in flavours:
                # Check if this placement would be feasible for the entire duration
                duration_feasible = True
                resource_feasible = True
                failure_reason = ""
                
                for slot_offset in range(int(pod.duration)):
                    current_slot = ts.id + slot_offset
                    if current_slot >= max_time_slots:
                        duration_feasible = False
                        failure_reason = f"slot_{current_slot}_exceeds_window_{max_time_slots}"
                        logging.debug(f"[find_placement_atomic] ❌ Pod {pod.id} on {flv.id}: {failure_reason}")
                        break
                
                if duration_feasible:
                    # Check if resources would be available (without actually allocating yet)
                    resource_check = persistent_state.check_resources_available(
                        flv.id, ts.id, int(pod.duration), pod.cpuRequest, pod.ramRequest
                    )
                    
                    if resource_check:
                        # Calculate emissions for this candidate
                        total_emi = compute_emissions(flv, ts.id, pod)
                        candidates.append((flv, ts, total_emi))
                        logging.debug(f"[find_placement_atomic] Valid candidate: pod={pod.id}, node={flv.id}, timeslot={ts.id}, emissions={total_emi:.3f}")
                    else:
                        failure_reason = "insufficient_resources"
                        logging.debug(f"[find_placement_atomic] Pod {pod.id} on {flv.id} at slot {ts.id}: {failure_reason}")
        
        logging.info(f"    Valid timeslots: {valid_timeslots}/{len(timeslots)}")
        logging.info(f"    Feasible candidates found: {len(candidates)}")
        
        if not candidates:
            logging.warning(f"[find_placement_atomic] No feasible candidates found for pod={pod.id}")
            logging.warning(f"    Diagnosis for pod={pod.id}:")
            logging.warning(f"        - Earliest timeslot constraint: {pod.earliest_timeslot}")
            logging.warning(f"        - Duration: {pod.duration}h")
            logging.warning(f"        - Resources needed: CPU={pod.cpuRequest:.3f}, RAM={pod.ramRequest:.0f}MB")
            logging.warning(f"        - Available timeslots: {[ts.id for ts in timeslots]}")
            logging.warning(f"        - Valid timeslots after constraint: {valid_timeslots}")
            
            # Check each node's current resource availability
            for flv in flavours:
                max_cpu_available = max(persistent_state.leftover_cpu[flv.id].values())
                max_ram_available = max(persistent_state.leftover_ram[flv.id].values())
                logging.warning(f"        - Node {flv.id}: max_cpu={max_cpu_available:.3f}, max_ram={max_ram_available:.0f}MB")
            
            return None, None, float('inf')
        
        # Sort candidates by emissions (best first)
        candidates.sort(key=lambda x: x[2])
        logging.info(f"    Best candidate: node={candidates[0][0].id}, timeslot={candidates[0][1].id}, emissions={candidates[0][2]:.3f}")
        
        # Second pass: try to atomically allocate the best candidate
        allocation_attempts = 0
        for flv, ts, emissions in candidates:
            allocation_attempts += 1
            success = persistent_state.atomic_check_and_allocate(
                flv.id, ts.id, int(pod.duration), 
                pod.cpuRequest, pod.ramRequest
            )
            
            if success:
                logging.info(f"[find_placement_atomic] Successfully allocated pod={pod.id} on attempt {allocation_attempts}")
                logging.info(f"    Final placement: node={flv.id}, timeslot={ts.id}, emissions={emissions:.3f}")
                
                # Write placement to CSV (same as in find_placement method)
                self._write_placement_to_csv(
                    pod_id=pod.id,
                    node_id=flv.id,
                    start_slot=ts.id,
                    duration=pod.duration,
                    cpu_request=pod.cpuRequest,
                    ram_request=pod.ramRequest
                )
                
                return flv, ts, emissions
            else:
                logging.debug(f"[find_placement_atomic] Allocation failed for pod={pod.id} on node={flv.id}, timeslot={ts.id} (resources taken)")
        
        # No candidate could be allocated (all resources were taken by other threads)
        logging.warning(f"[find_placement_atomic] All {allocation_attempts} candidates exhausted for pod={pod.id} - resources taken by other processes")
        return None, None, float('inf')


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