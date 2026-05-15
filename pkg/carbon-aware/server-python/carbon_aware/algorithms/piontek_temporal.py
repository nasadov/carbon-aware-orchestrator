"""
Piontek-style temporal carbon-aware Kubernetes baseline.

This baseline adapts the scheduler-extender idea from Piontek et al. to the
TotEm simulator. It is intentionally temporal-only: carbon decides whether an
arrived non-critical job should be admitted in the current scheduling epoch,
while node choice remains resource-only and Kubernetes-like.
"""
import csv
import logging
import os
import time
from typing import Dict, List, Optional, Set, Tuple

from carbon_aware.algorithms.base import SchedulingAlgorithm
from carbon_aware.models import CarbonAwareFlavour, CarbonAwarePod, CarbonAwareTimeslot
from carbon_aware.utils import (
    compute_node_dynamic_coeff_watts,
    get_carbon_intensity,
    is_timeslot_valid,
)


class PiontekTemporalAlgorithm(SchedulingAlgorithm):
    """
    Temporal CO2-window admission baseline.

    The original Piontek scheduler delays non-critical Kubernetes jobs until the
    current hour is inside a low-CO2 window, unless the job is critical or stale.
    Our workloads do not carry priority classes, so all pods are treated as
    non-critical batch jobs and are forced only when their deadline would
    otherwise be missed.
    """

    _placement_csv_filename = "piontek_temporal_placements_session.csv"

    def __init__(
        self,
        perf_logger=None,
        optimal_window_hours: int = 6,
        cpu_utilization_threshold: float = 0.95,
        max_stale_hours: int = 24,
        node_score_mode: str = "most_allocated",
    ):
        self.experiment_logger = None
        self.perf_logger = perf_logger
        self._session_log_dir: Optional[str] = None
        self._workloads_dir: Optional[str] = None
        self._placement_csv_file_handle = None
        self._placement_csv_writer = None
        self._placement_csv_path: Optional[str] = None
        self._csv_buffer: List[List[object]] = []
        self._csv_buffer_limit = 200
        self.optimal_window_hours = max(1, int(optimal_window_hours))
        self.cpu_utilization_threshold = max(0.0, min(1.0, float(cpu_utilization_threshold)))
        self.max_stale_hours = max(1, int(max_stale_hours))
        self.node_score_mode = (
            node_score_mode if node_score_mode in ("most_allocated", "least_allocated") else "most_allocated"
        )
        self.iterations = 0
        self.steps = 0

    @staticmethod
    def _ensure_dir_exists(directory_path: str) -> None:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    @property
    def name(self) -> str:
        return "Piontek-Temporal-K8s"

    def set_base_log_dir(self, base_dir: str) -> None:
        self._session_log_dir = base_dir

    def set_workloads_dir(self, workloads_dir: str) -> None:
        self._workloads_dir = workloads_dir
        logging.info("PiontekTemporalAlgorithm: workloads directory set to %s", workloads_dir)

    def setup_session_placement_log(self) -> None:
        if not self._session_log_dir:
            self._session_log_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "..",
                    "..",
                    "..",
                    "analysis",
                    "piontek_temporal_fallback_logs",
                )
            )
            self._ensure_dir_exists(self._session_log_dir)
            logging.warning(
                "Session log directory not set for Piontek baseline; using %s",
                self._session_log_dir,
            )

        self._placement_csv_path = os.path.join(
            self._session_log_dir, self._placement_csv_filename
        )
        if self._placement_csv_file_handle:
            try:
                self.flush_session_placement_log()
                self._placement_csv_file_handle.close()
            except Exception as exc:
                logging.error("Error closing previous Piontek placement CSV: %s", exc)

        try:
            file_exists_and_not_empty = (
                os.path.exists(self._placement_csv_path)
                and os.path.getsize(self._placement_csv_path) > 0
            )
            self._placement_csv_file_handle = open(self._placement_csv_path, "a", newline="")
            self._placement_csv_writer = csv.writer(self._placement_csv_file_handle)
            if not file_exists_and_not_empty:
                self._placement_csv_writer.writerow(
                    [
                        "pod_id",
                        "node_id",
                        "start_slot",
                        "duration",
                        "cpu_request",
                        "ram_request",
                        "decision_emissions_mode",
                        "decision_operational_emissions_g",
                        "baseline_variant",
                        "admission_reason",
                        "co2_window_start",
                        "co2_window_end",
                        "cpu_threshold",
                        "node_score_mode",
                    ]
                )
                self._placement_csv_file_handle.flush()
            logging.info(
                "Piontek baseline placements will be logged to: %s",
                self._placement_csv_path,
            )
        except IOError as exc:
            logging.error("Failed to open Piontek placement CSV %s: %s", self._placement_csv_path, exc)
            self._placement_csv_writer = None
            self._placement_csv_file_handle = None
            self._placement_csv_path = None

    def __del__(self):
        if getattr(self, "_placement_csv_file_handle", None):
            try:
                self.flush_session_placement_log()
                self._placement_csv_file_handle.close()
            except Exception:
                pass

    def flush_session_placement_log(self) -> None:
        if self._placement_csv_writer and self._placement_csv_file_handle and self._csv_buffer:
            self._placement_csv_writer.writerows(self._csv_buffer)
            self._csv_buffer.clear()
            self._placement_csv_file_handle.flush()

    def _write_placement_to_csv(
        self,
        pod_id: str,
        node_id: str,
        start_slot: int,
        duration: float,
        cpu_request: float,
        ram_request: float,
        decision_operational_emissions_g: float,
        admission_reason: str,
        co2_window: Tuple[int, int],
    ) -> None:
        if not self._placement_csv_writer or not self._placement_csv_file_handle:
            logging.warning("Piontek placement CSV writer not available; cannot log placement.")
            return
        self._csv_buffer.append(
            [
                pod_id,
                node_id,
                start_slot,
                duration,
                cpu_request,
                ram_request,
                "operational-only",
                decision_operational_emissions_g,
                "piontek-temporal-k8s",
                admission_reason,
                co2_window[0],
                co2_window[1],
                self.cpu_utilization_threshold,
                self.node_score_mode,
            ]
        )
        if len(self._csv_buffer) >= self._csv_buffer_limit:
            self.flush_session_placement_log()

    def _set_pod_earliest_timeslot(self, pod: CarbonAwarePod) -> None:
        if getattr(pod, "_earliest_timeslot_source", None) == "precompute":
            pod.calculate_deadline_slot()
            return
        if getattr(pod, "earliest_timeslot", None) is None:
            pod.earliest_timeslot = 0
        pod.calculate_deadline_slot()

    @staticmethod
    def _duration_slots(pod: CarbonAwarePod) -> int:
        return max(1, int(pod.duration))

    def latest_start_slot(self, pod: CarbonAwarePod, max_time_slots: int) -> int:
        self._set_pod_earliest_timeslot(pod)
        duration_slots = self._duration_slots(pod)
        deadline_slot = getattr(pod, "deadline_slot", None)
        if deadline_slot is None:
            deadline_slot = max_time_slots
        return min(int(deadline_slot) - duration_slots, max_time_slots - duration_slots)

    def _cluster_carbon_series(
        self,
        flavours: List[CarbonAwareFlavour],
        max_time_slots: int,
    ) -> List[float]:
        total_cpu = sum(max(float(flv.totalCpu), 0.0) for flv in flavours)
        if total_cpu <= 0.0:
            total_cpu = float(max(len(flavours), 1))
        series = []
        for slot in range(max_time_slots):
            weighted = 0.0
            for flv in flavours:
                weight = max(float(flv.totalCpu), 0.0)
                if weight <= 0.0:
                    weight = 1.0
                weighted += weight * get_carbon_intensity(flv, slot)
            series.append(weighted / total_cpu)
        return series

    def _pod_start_carbon_score(
        self,
        carbon_series: List[float],
        start_slot: int,
        duration_slots: int,
        max_time_slots: int,
    ) -> float:
        end_slot = min(start_slot + duration_slots, max_time_slots)
        values = carbon_series[start_slot:end_slot]
        if not values:
            return float("inf")
        return sum(values) / len(values)

    def _best_co2_start_window(
        self,
        pod: CarbonAwarePod,
        current_slot: int,
        flavours: List[CarbonAwareFlavour],
        max_time_slots: int,
    ) -> Set[int]:
        """
        Return admissible start slots in the lowest-CO2 window for this pod's SLA.

        Piontek uses a 24h time frame and a 6h optimal CO2 window. Here the same
        idea is applied within the pod's own feasible start interval so deadlines
        are respected in our finite-horizon simulator.
        """
        duration_slots = self._duration_slots(pod)
        earliest_slot = max(int(getattr(pod, "earliest_timeslot", 0)), 0)
        latest_start = self.latest_start_slot(pod, max_time_slots)
        if latest_start < earliest_slot:
            return set()

        start_slots = list(range(earliest_slot, latest_start + 1))
        if not start_slots:
            return set()

        window_len = min(self.optimal_window_hours, len(start_slots))
        carbon_series = self._cluster_carbon_series(flavours, max_time_slots)

        best_start = start_slots[0]
        best_score = None
        for candidate_start in range(earliest_slot, latest_start - window_len + 2):
            window_starts = range(candidate_start, candidate_start + window_len)
            score = sum(
                self._pod_start_carbon_score(
                    carbon_series, slot, duration_slots, max_time_slots
                )
                for slot in window_starts
            ) / window_len
            key = (score, candidate_start)
            if best_score is None or key < best_score:
                best_score = key
                best_start = candidate_start

        return set(range(best_start, best_start + window_len))

    def _duration_feasible(
        self,
        flv: CarbonAwareFlavour,
        start_slot: int,
        duration_slots: int,
        pod: CarbonAwarePod,
        leftover_cpu: Dict[str, Dict[int, float]],
        leftover_ram: Dict[str, Dict[int, float]],
        max_time_slots: int,
    ) -> bool:
        if start_slot + duration_slots > max_time_slots:
            return False
        for slot in range(start_slot, start_slot + duration_slots):
            if leftover_cpu.get(flv.id, {}).get(slot, 0.0) < pod.cpuRequest:
                return False
            if leftover_ram.get(flv.id, {}).get(slot, 0.0) < pod.ramRequest:
                return False
        return True

    def _cluster_cpu_utilization_after(
        self,
        pod: CarbonAwarePod,
        start_slot: int,
        duration_slots: int,
        flavours: List[CarbonAwareFlavour],
        leftover_cpu: Dict[str, Dict[int, float]],
        max_time_slots: int,
    ) -> float:
        total_cpu = sum(max(float(flv.totalCpu), 0.0) for flv in flavours)
        if total_cpu <= 0.0:
            return 1.0
        max_util = 0.0
        for slot in range(start_slot, min(start_slot + duration_slots, max_time_slots)):
            used_cpu = sum(
                max(float(flv.totalCpu) - leftover_cpu.get(flv.id, {}).get(slot, flv.totalCpu), 0.0)
                for flv in flavours
            )
            max_util = max(max_util, (used_cpu + pod.cpuRequest) / total_cpu)
        return max_util

    @staticmethod
    def _least_allocated_score(
        available_cpu: float,
        total_cpu: float,
        available_ram: float,
        total_ram: float,
    ) -> float:
        cpu_ratio = available_cpu / total_cpu if total_cpu > 0 else 0.0
        ram_ratio = available_ram / total_ram if total_ram > 0 else 0.0
        return 0.5 * cpu_ratio + 0.5 * ram_ratio

    @staticmethod
    def _most_allocated_score(
        available_cpu: float,
        total_cpu: float,
        available_ram: float,
        total_ram: float,
    ) -> float:
        cpu_ratio = 1.0 - (available_cpu / total_cpu if total_cpu > 0 else 0.0)
        ram_ratio = 1.0 - (available_ram / total_ram if total_ram > 0 else 0.0)
        return 0.5 * cpu_ratio + 0.5 * ram_ratio

    def _node_score(
        self,
        flv: CarbonAwareFlavour,
        start_slot: int,
        duration_slots: int,
        pod: CarbonAwarePod,
        leftover_cpu: Dict[str, Dict[int, float]],
        leftover_ram: Dict[str, Dict[int, float]],
    ) -> float:
        min_cpu = min(leftover_cpu[flv.id][slot] for slot in range(start_slot, start_slot + duration_slots))
        min_ram = min(leftover_ram[flv.id][slot] for slot in range(start_slot, start_slot + duration_slots))
        after_cpu = min_cpu - pod.cpuRequest
        after_ram = min_ram - pod.ramRequest
        if self.node_score_mode == "most_allocated":
            return self._most_allocated_score(after_cpu, flv.totalCpu, after_ram, flv.totalRam)
        return self._least_allocated_score(after_cpu, flv.totalCpu, after_ram, flv.totalRam)

    def _select_resource_only_node(
        self,
        pod: CarbonAwarePod,
        start_slot: int,
        flavours: List[CarbonAwareFlavour],
        leftover_cpu: Dict[str, Dict[int, float]],
        leftover_ram: Dict[str, Dict[int, float]],
        max_time_slots: int,
    ) -> Optional[CarbonAwareFlavour]:
        duration_slots = self._duration_slots(pod)
        best_node = None
        best_key = None
        for flv in flavours:
            self.steps += 1
            if not self._duration_feasible(
                flv, start_slot, duration_slots, pod, leftover_cpu, leftover_ram, max_time_slots
            ):
                continue
            score = self._node_score(
                flv, start_slot, duration_slots, pod, leftover_cpu, leftover_ram
            )
            key = (-score, flv.id)
            if best_key is None or key < best_key:
                best_key = key
                best_node = flv
        return best_node

    def _used_cpu_before_by_slot(
        self,
        flv: CarbonAwareFlavour,
        start_slot: int,
        duration_slots: int,
        leftover_cpu: Dict[str, Dict[int, float]],
    ) -> Dict[int, float]:
        return {
            slot: max(flv.totalCpu - leftover_cpu.get(flv.id, {}).get(slot, flv.totalCpu), 0.0)
            for slot in range(start_slot, start_slot + duration_slots)
        }

    def _marginal_operational_emissions_g(
        self,
        flv: CarbonAwareFlavour,
        start_slot: int,
        pod: CarbonAwarePod,
        used_cpu_before_by_slot: Dict[int, float],
    ) -> float:
        total_g = 0.0
        total_cpu = max(flv.totalCpu, 1e-6)
        pod_cpu_ratio = pod.cpuRequest / total_cpu
        dynamic_coeff = compute_node_dynamic_coeff_watts(flv)
        for offset in range(self._duration_slots(pod)):
            slot = start_slot + offset
            delta_power_w = dynamic_coeff * pod_cpu_ratio
            if used_cpu_before_by_slot.get(slot, 0.0) <= 0.0:
                delta_power_w += flv.power.get("idle", 0.0)
            total_g += get_carbon_intensity(flv, slot) * (delta_power_w / 1000.0)
        return total_g

    def should_admit_now(
        self,
        pod: CarbonAwarePod,
        current_slot: int,
        flavours: List[CarbonAwareFlavour],
        leftover_cpu: Dict[str, Dict[int, float]],
        max_time_slots: int,
    ) -> Tuple[bool, str, Tuple[int, int]]:
        duration_slots = self._duration_slots(pod)
        latest_start = self.latest_start_slot(pod, max_time_slots)
        if latest_start < current_slot:
            return False, "expired", (-1, -1)

        waited_hours = max(current_slot - int(getattr(pod, "earliest_timeslot", 0)), 0)
        co2_window = self._best_co2_start_window(
            pod, current_slot, flavours, max_time_slots
        )
        window_bounds = (
            min(co2_window) if co2_window else -1,
            max(co2_window) if co2_window else -1,
        )

        if current_slot >= latest_start:
            return True, "forced_latest_start", window_bounds
        if waited_hours >= self.max_stale_hours:
            return True, "forced_max_stale", window_bounds

        projected_utilization = self._cluster_cpu_utilization_after(
            pod, current_slot, duration_slots, flavours, leftover_cpu, max_time_slots
        )
        if projected_utilization > self.cpu_utilization_threshold:
            return False, "delayed_cpu_threshold", window_bounds

        if current_slot in co2_window:
            return True, "inside_low_co2_window", window_bounds

        return False, "delayed_outside_low_co2_window", window_bounds

    def find_placement(
        self,
        pod: CarbonAwarePod,
        flavours: List[CarbonAwareFlavour],
        timeslots: List[CarbonAwareTimeslot],
        leftover_cpu: Dict[str, Dict[int, float]],
        leftover_ram: Dict[str, Dict[int, float]],
        max_time_slots: int = 48,
    ) -> Tuple[Optional[CarbonAwareFlavour], Optional[CarbonAwareTimeslot], float]:
        self._set_pod_earliest_timeslot(pod)
        start_time = time.time()
        current_slot = int(getattr(pod, "_current_scheduling_slot", pod.earliest_timeslot))
        if current_slot < 0 or current_slot >= max_time_slots:
            return None, None, float("inf")

        admit, reason, co2_window = self.should_admit_now(
            pod, current_slot, flavours, leftover_cpu, max_time_slots
        )
        if not admit:
            return None, None, float("inf")

        timeslot = next((ts for ts in timeslots if ts.id == current_slot), None)
        if timeslot is None or not is_timeslot_valid(timeslot, pod):
            return None, None, float("inf")

        selected_node = self._select_resource_only_node(
            pod, current_slot, flavours, leftover_cpu, leftover_ram, max_time_slots
        )
        self.iterations += 1
        if selected_node is None:
            return None, None, float("inf")

        used_cpu_before = self._used_cpu_before_by_slot(
            selected_node, current_slot, self._duration_slots(pod), leftover_cpu
        )
        emissions_g = self._marginal_operational_emissions_g(
            selected_node, current_slot, pod, used_cpu_before
        )

        self._write_placement_to_csv(
            pod_id=pod.id,
            node_id=selected_node.id,
            start_slot=current_slot,
            duration=pod.duration,
            cpu_request=pod.cpuRequest,
            ram_request=pod.ramRequest,
            decision_operational_emissions_g=emissions_g,
            admission_reason=reason,
            co2_window=co2_window,
        )

        if self.experiment_logger:
            self.experiment_logger.record_placement(
                pod_id=pod.id,
                success=True,
                execution_time=time.time() - start_time,
                emissions=emissions_g,
                considered_options=len(flavours),
                selected_node=selected_node.id,
                selected_timeslot=current_slot,
            )

        return selected_node, timeslot, emissions_g
