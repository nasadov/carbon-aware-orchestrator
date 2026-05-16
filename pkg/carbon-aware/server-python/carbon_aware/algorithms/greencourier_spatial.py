"""
GreenCourier-style spatial carbon-aware Kubernetes baseline.

GreenCourier schedules serverless functions across geographically distributed
Kubernetes clusters by scoring eligible nodes with a regional carbon-efficiency
signal. This adaptation keeps that core spatial mechanism for TotEm's fixed
batch-pod simulator:

* no temporal carbon shifting;
* no embodied emissions in the decision;
* filter nodes by CPU/RAM feasibility over the pod duration;
* choose the feasible node in the lowest-carbon region at the earliest feasible
  start slot;
* tie-break with Kubernetes-like MostAllocated resource scoring.

For long-running pods, the regional carbon score is averaged over the pod's
runtime. That is the duration-aware analogue of GreenCourier's current regional
carbon score for short serverless functions.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

from carbon_aware.algorithms.base import SchedulingAlgorithm
from carbon_aware.models import CarbonAwareFlavour, CarbonAwarePod, CarbonAwareTimeslot
from carbon_aware.utils import (
    compute_node_dynamic_coeff_watts,
    get_carbon_intensity,
    is_timeslot_valid,
)


class GreenCourierSpatialAlgorithm(SchedulingAlgorithm):
    """Spatial-only carbon-aware Kubernetes placement baseline."""

    _placement_csv_filename = "greencourier_spatial_placements_session.csv"

    def __init__(
        self,
        perf_logger=None,
        node_score_mode: str = "most_allocated",
    ) -> None:
        self.experiment_logger = None
        self.perf_logger = perf_logger
        self.node_score_mode = (
            node_score_mode
            if node_score_mode in ("most_allocated", "least_allocated")
            else "most_allocated"
        )
        self._session_log_dir: Optional[str] = None
        self._workloads_dir: Optional[str] = None
        self._placement_csv_file_handle = None
        self._placement_csv_writer = None
        self._placement_csv_path: Optional[str] = None
        self._csv_buffer: List[List[object]] = []
        self._csv_buffer_limit = 200
        self.iterations = 0
        self.steps = 0

    @property
    def name(self) -> str:
        return "GreenCourier-Spatial-K8s"

    @staticmethod
    def _ensure_dir_exists(directory_path: str) -> None:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def set_base_log_dir(self, base_dir: str) -> None:
        self._session_log_dir = base_dir

    def set_workloads_dir(self, workloads_dir: str) -> None:
        self._workloads_dir = workloads_dir
        logging.info("GreenCourierSpatialAlgorithm: workloads directory set to %s", workloads_dir)

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
                    "greencourier_spatial_fallback_logs",
                )
            )
            self._ensure_dir_exists(self._session_log_dir)
            logging.warning(
                "Session log directory not set for GreenCourier baseline; using %s",
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
                logging.error("Error closing previous GreenCourier placement CSV: %s", exc)

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
                        "region",
                        "duration_avg_carbon_intensity",
                        "resource_tiebreak_score",
                        "node_score_mode",
                    ]
                )
                self._placement_csv_file_handle.flush()
            logging.info(
                "GreenCourier baseline placements will be logged to: %s",
                self._placement_csv_path,
            )
        except IOError as exc:
            logging.error(
                "Failed to open GreenCourier placement CSV %s: %s",
                self._placement_csv_path,
                exc,
            )
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
        pod: CarbonAwarePod,
        node_id: str,
        start_slot: int,
        decision_operational_emissions_g: float,
        region: str,
        duration_avg_carbon_intensity: float,
        resource_tiebreak_score: float,
    ) -> None:
        if not self._placement_csv_writer or not self._placement_csv_file_handle:
            logging.warning("GreenCourier placement CSV writer not available; cannot log placement.")
            return
        self._csv_buffer.append(
            [
                pod.id,
                node_id,
                start_slot,
                pod.duration,
                pod.cpuRequest,
                pod.ramRequest,
                "operational-only",
                decision_operational_emissions_g,
                "greencourier-spatial-k8s",
                region,
                duration_avg_carbon_intensity,
                resource_tiebreak_score,
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

    def _resource_score(
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
        if self.node_score_mode == "least_allocated":
            return self._least_allocated_score(after_cpu, flv.totalCpu, after_ram, flv.totalRam)
        return self._most_allocated_score(after_cpu, flv.totalCpu, after_ram, flv.totalRam)

    def _duration_avg_carbon_intensity(
        self,
        flv: CarbonAwareFlavour,
        start_slot: int,
        duration_slots: int,
        max_time_slots: int,
    ) -> float:
        values = [
            get_carbon_intensity(flv, min(slot, max_time_slots - 1))
            for slot in range(start_slot, start_slot + duration_slots)
        ]
        return sum(values) / max(len(values), 1)

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
        duration_slots = self._duration_slots(pod)
        earliest_slot = max(int(getattr(pod, "earliest_timeslot", 0)), 0)
        latest_start = self.latest_start_slot(pod, max_time_slots)
        considered_options = len(flavours) * max(latest_start - earliest_slot + 1, 0)

        if latest_start < earliest_slot:
            if self.experiment_logger:
                self.experiment_logger.record_placement(
                    pod_id=pod.id,
                    success=False,
                    execution_time=time.time() - start_time,
                    emissions=0.0,
                    considered_options=0,
                    selected_node=None,
                    selected_timeslot=None,
                )
            return None, None, float("inf")

        for ts_id in range(earliest_slot, latest_start + 1):
            timeslot = next((ts for ts in timeslots if ts.id == ts_id), None)
            if timeslot is None or not is_timeslot_valid(timeslot, pod):
                continue

            best_node = None
            best_key = None
            best_ci = float("inf")
            best_resource_score = 0.0

            for flv in flavours:
                self.steps += 1
                if not self._duration_feasible(
                    flv, ts_id, duration_slots, pod, leftover_cpu, leftover_ram, max_time_slots
                ):
                    continue
                avg_ci = self._duration_avg_carbon_intensity(
                    flv, ts_id, duration_slots, max_time_slots
                )
                resource_score = self._resource_score(
                    flv, ts_id, duration_slots, pod, leftover_cpu, leftover_ram
                )
                key = (avg_ci, -resource_score, flv.id)
                if best_key is None or key < best_key:
                    best_key = key
                    best_node = flv
                    best_ci = avg_ci
                    best_resource_score = resource_score

            if best_node is None:
                continue

            self.iterations += 1
            used_cpu_before = self._used_cpu_before_by_slot(
                best_node, ts_id, duration_slots, leftover_cpu
            )
            emissions_g = self._marginal_operational_emissions_g(
                best_node, ts_id, pod, used_cpu_before
            )
            region = str(getattr(best_node, "region", "")).upper()
            self._write_placement_to_csv(
                pod=pod,
                node_id=best_node.id,
                start_slot=ts_id,
                decision_operational_emissions_g=emissions_g,
                region=region,
                duration_avg_carbon_intensity=best_ci,
                resource_tiebreak_score=best_resource_score,
            )

            if self.experiment_logger:
                self.experiment_logger.record_placement(
                    pod_id=pod.id,
                    success=True,
                    execution_time=time.time() - start_time,
                    emissions=emissions_g,
                    considered_options=considered_options,
                    selected_node=best_node.id,
                    selected_timeslot=ts_id,
                )

            return best_node, timeslot, emissions_g

        if self.experiment_logger:
            self.experiment_logger.record_placement(
                pod_id=pod.id,
                success=False,
                execution_time=time.time() - start_time,
                emissions=0.0,
                considered_options=considered_options,
                selected_node=None,
                selected_timeslot=None,
            )

        return None, None, float("inf")
