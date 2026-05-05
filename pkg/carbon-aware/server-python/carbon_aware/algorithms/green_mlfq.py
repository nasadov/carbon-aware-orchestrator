"""
GREEN-style carbon-aware MLFQ baseline.

This is a policy-level adaptation of Xu et al.'s GREEN scheduler to TotEm's
fixed-size Kubernetes pod simulator. GREEN's original system targets ML/GPU
clusters with Slurm, checkpoint/restart preemption, online energy profiling,
and scale-out decisions. Our workloads do not expose GPU counts, progress
curves, checkpoint costs, or elastic resource choices, so this implementation
preserves the parts that map cleanly:

* a rolling online scheduler with no future workload arrivals;
* a short upper queue for newly arrived jobs, capped by a fraction of cluster
  capacity;
* a lower queue ordered by GREEN's carbon-footprint priority term;
* GREEN's shifting factor, which prioritizes high-power jobs in greener hours
  and low-power jobs in dirtier hours;
* a fixed-resource degradation factor of 1.0 because scaling is unavailable.

Node placement is intentionally resource-only. GREEN is a temporal/priority
scheduler in a single-cluster setting, so giving this baseline spatial carbon
optimization would make it stronger than the original policy family.
"""

from __future__ import annotations

import csv
import logging
import os
import statistics
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from carbon_aware.algorithms.base import SchedulingAlgorithm
from carbon_aware.models import CarbonAwareFlavour, CarbonAwarePod, CarbonAwareTimeslot
from carbon_aware.utils import (
    compute_node_dynamic_coeff_watts,
    get_carbon_intensity,
    is_timeslot_valid,
)


@dataclass(frozen=True)
class GreenPriorityInfo:
    queue_name: str
    priority: float
    carbon_factor: float
    shifting_factor: float
    power_w: float
    degradation_factor: float
    cluster_carbon_intensity: float
    avg_carbon_intensity: float
    median_power_w: float
    urgent: bool = False


class GreenMLFQAlgorithm(SchedulingAlgorithm):
    """
    GREEN-inspired MLFQ scheduler for fixed Kubernetes pods.

    Smaller GREEN priority values mean higher scheduling priority, matching the
    paper's statement that larger priority values represent lower priority.
    """

    _placement_csv_filename = "green_mlfq_placements_session.csv"

    def __init__(
        self,
        perf_logger=None,
        mu: float = 2.0,
        upper_queue_capacity_fraction: float = 0.30,
        upper_queue_profile_hours: int = 1,
        node_score_mode: str = "most_allocated",
    ) -> None:
        self.experiment_logger = None
        self.perf_logger = perf_logger
        self.mu = max(1.0, float(mu))
        self.upper_queue_capacity_fraction = max(
            0.0, min(1.0, float(upper_queue_capacity_fraction))
        )
        self.upper_queue_profile_hours = max(1, int(upper_queue_profile_hours))
        self.node_score_mode = node_score_mode

        self._session_log_dir: Optional[str] = None
        self._workloads_dir: Optional[str] = None
        self._placement_csv_file_handle = None
        self._placement_csv_writer = None
        self._placement_csv_path: Optional[str] = None
        self._csv_buffer: List[List[object]] = []
        self._csv_buffer_limit = 200
        self._pending_context: Sequence[CarbonAwarePod] = []
        self._priority_context: Dict[str, GreenPriorityInfo] = {}
        self.iterations = 0
        self.steps = 0

    @property
    def name(self) -> str:
        return "GREEN-MLFQ-K8s"

    @staticmethod
    def _ensure_dir_exists(directory_path: str) -> None:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def set_base_log_dir(self, base_dir: str) -> None:
        self._session_log_dir = base_dir

    def set_workloads_dir(self, workloads_dir: str) -> None:
        self._workloads_dir = workloads_dir
        logging.info("GreenMLFQAlgorithm: workloads directory set to %s", workloads_dir)

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
                    "green_mlfq_fallback_logs",
                )
            )
            self._ensure_dir_exists(self._session_log_dir)
            logging.warning(
                "Session log directory not set for GREEN baseline; using %s",
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
                logging.error("Error closing previous GREEN placement CSV: %s", exc)

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
                        "queue_name",
                        "green_priority",
                        "carbon_factor",
                        "shifting_factor",
                        "power_w",
                        "degradation_factor",
                        "cluster_carbon_intensity",
                        "avg_carbon_intensity",
                        "median_power_w",
                        "mu",
                        "upper_queue_capacity_fraction",
                        "node_score_mode",
                    ]
                )
                self._placement_csv_file_handle.flush()
            logging.info("GREEN baseline placements will be logged to: %s", self._placement_csv_path)
        except IOError as exc:
            logging.error("Failed to open GREEN placement CSV %s: %s", self._placement_csv_path, exc)
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
        priority_info: GreenPriorityInfo,
    ) -> None:
        if not self._placement_csv_writer or not self._placement_csv_file_handle:
            logging.warning("GREEN placement CSV writer not available; cannot log placement.")
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
                "green-mlfq-k8s-fixed-resource",
                priority_info.queue_name,
                priority_info.priority,
                priority_info.carbon_factor,
                priority_info.shifting_factor,
                priority_info.power_w,
                priority_info.degradation_factor,
                priority_info.cluster_carbon_intensity,
                priority_info.avg_carbon_intensity,
                priority_info.median_power_w,
                self.mu,
                self.upper_queue_capacity_fraction,
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

    def _cluster_carbon_intensity(
        self,
        flavours: Sequence[CarbonAwareFlavour],
        slot: int,
    ) -> float:
        total_cpu = sum(max(float(flv.totalCpu), 0.0) for flv in flavours)
        if total_cpu <= 0.0:
            total_cpu = float(max(len(flavours), 1))
        weighted = 0.0
        for flv in flavours:
            weight = max(float(flv.totalCpu), 0.0)
            if weight <= 0.0:
                weight = 1.0
            weighted += weight * get_carbon_intensity(flv, slot)
        return weighted / total_cpu

    def _average_cluster_carbon_intensity(
        self,
        flavours: Sequence[CarbonAwareFlavour],
        max_time_slots: int,
    ) -> float:
        values = [
            self._cluster_carbon_intensity(flavours, slot)
            for slot in range(max_time_slots)
        ]
        return sum(values) / max(len(values), 1)

    @staticmethod
    def _median(values: Iterable[float], default: float) -> float:
        cleaned = [float(v) for v in values if float(v) > 0.0]
        return statistics.median(cleaned) if cleaned else default

    def estimate_job_power_w(
        self,
        pod: CarbonAwarePod,
        flavours: Sequence[CarbonAwareFlavour],
    ) -> float:
        """
        Estimate fixed-resource job power.

        GREEN uses measured GPU/CPU/static job power. In this simulator we do not
        have runtime power traces, so the closest available signal is the median
        node power model scaled by the pod's CPU request.
        """
        if getattr(pod, "powerConsumption", 0.0):
            return max(float(pod.powerConsumption) * 1000.0, 1e-6)

        median_cpu = self._median((flv.totalCpu for flv in flavours), default=1.0)
        median_idle = self._median((flv.power.get("idle", 0.0) for flv in flavours), default=100.0)
        median_dynamic = self._median(
            (compute_node_dynamic_coeff_watts(flv) for flv in flavours),
            default=300.0,
        )
        cpu_ratio = max(float(pod.cpuRequest) / max(median_cpu, 1e-6), 0.0)
        static_share = median_idle * min(cpu_ratio, 1.0)
        return max(static_share + median_dynamic * cpu_ratio, 1e-6)

    def _power_bounds(
        self,
        pods: Sequence[CarbonAwarePod],
        flavours: Sequence[CarbonAwareFlavour],
    ) -> Tuple[float, float, float]:
        powers = [self.estimate_job_power_w(pod, flavours) for pod in pods]
        if not powers:
            return 1.0, 1.0, 1.0
        return min(powers), max(powers), statistics.median(powers)

    def _scaled_power_term(
        self,
        power_w: float,
        min_power_w: float,
        max_power_w: float,
    ) -> float:
        if max_power_w <= min_power_w:
            return 1.0
        return ((power_w - min_power_w) / (max_power_w - min_power_w)) * (self.mu - 1.0) + 1.0

    def _shifting_factor(
        self,
        power_w: float,
        min_power_w: float,
        max_power_w: float,
        median_power_w: float,
        cluster_ci: float,
        avg_ci: float,
    ) -> float:
        scaled_power = self._scaled_power_term(power_w, min_power_w, max_power_w)
        high_power = power_w > median_power_w
        green_hour = cluster_ci <= avg_ci

        if green_hour and high_power:
            return 1.0 / scaled_power
        if green_hour and not high_power:
            return scaled_power
        if not green_hour and high_power:
            return scaled_power
        return 1.0 / scaled_power

    def _projected_carbon_factor(
        self,
        power_w: float,
        start_slot: int,
        duration_slots: int,
        flavours: Sequence[CarbonAwareFlavour],
        max_time_slots: int,
    ) -> float:
        """
        Fixed-pod analogue of GREEN's accumulated FOOTPRINT term.

        Non-preemptive pods have no accumulated runtime footprint before
        admission. We therefore use the projected operational footprint for
        running this fixed job now, which preserves GREEN's preference against
        high-carbon/high-power execution.
        """
        total_g = 0.0
        for offset in range(duration_slots):
            slot = min(start_slot + offset, max_time_slots - 1)
            total_g += self._cluster_carbon_intensity(flavours, slot) * (power_w / 1000.0)
        return total_g

    def _is_upper_queue(self, pod: CarbonAwarePod, current_slot: int) -> bool:
        arrival_slot = int(getattr(pod, "earliest_timeslot", 0))
        return current_slot - arrival_slot < self.upper_queue_profile_hours

    def _priority_info(
        self,
        pod: CarbonAwarePod,
        current_slot: int,
        pending_pods: Sequence[CarbonAwarePod],
        flavours: Sequence[CarbonAwareFlavour],
        max_time_slots: int,
        queue_name: str,
        urgent: bool = False,
    ) -> GreenPriorityInfo:
        min_power, max_power, median_power = self._power_bounds(pending_pods, flavours)
        power_w = self.estimate_job_power_w(pod, flavours)
        cluster_ci = self._cluster_carbon_intensity(flavours, current_slot)
        avg_ci = self._average_cluster_carbon_intensity(flavours, max_time_slots)
        degradation = 1.0
        carbon_factor = self._projected_carbon_factor(
            power_w=power_w,
            start_slot=current_slot,
            duration_slots=self._duration_slots(pod),
            flavours=flavours,
            max_time_slots=max_time_slots,
        ) / degradation
        shifting = self._shifting_factor(
            power_w=power_w,
            min_power_w=min_power,
            max_power_w=max_power,
            median_power_w=median_power,
            cluster_ci=cluster_ci,
            avg_ci=avg_ci,
        )
        priority = carbon_factor * shifting
        if urgent:
            priority = -1.0
        return GreenPriorityInfo(
            queue_name=queue_name,
            priority=priority,
            carbon_factor=carbon_factor,
            shifting_factor=shifting,
            power_w=power_w,
            degradation_factor=degradation,
            cluster_carbon_intensity=cluster_ci,
            avg_carbon_intensity=avg_ci,
            median_power_w=median_power,
            urgent=urgent,
        )

    def rank_pending_pods(
        self,
        pending_pods: Sequence[CarbonAwarePod],
        current_slot: int,
        flavours: Sequence[CarbonAwareFlavour],
        max_time_slots: int,
    ) -> List[CarbonAwarePod]:
        """
        Rank pods for the current active scheduling round.

        GREEN first admits urgent/active jobs, then upper-queue jobs up to theta
        capacity, then lower-queue jobs ordered by carbon priority.
        """
        self._pending_context = list(pending_pods)
        self._priority_context = {}

        urgent: List[CarbonAwarePod] = []
        upper: List[CarbonAwarePod] = []
        lower: List[CarbonAwarePod] = []

        for pod in pending_pods:
            latest_start = self.latest_start_slot(pod, max_time_slots)
            if current_slot >= latest_start:
                urgent.append(pod)
            elif self._is_upper_queue(pod, current_slot):
                upper.append(pod)
            else:
                lower.append(pod)

        urgent.sort(
            key=lambda pod: (
                self.latest_start_slot(pod, max_time_slots),
                int(getattr(pod, "earliest_timeslot", 0)),
                -float(getattr(pod, "cpuRequest", 0.0)),
                pod.id,
            )
        )
        upper.sort(key=lambda pod: (int(getattr(pod, "earliest_timeslot", 0)), pod.id))

        total_cpu = sum(max(float(flv.totalCpu), 0.0) for flv in flavours)
        upper_cpu_cap = self.upper_queue_capacity_fraction * total_cpu
        upper_selected: List[CarbonAwarePod] = []
        upper_cpu_used = 0.0
        for pod in upper:
            pod_cpu = max(float(getattr(pod, "cpuRequest", 0.0)), 0.0)
            if upper_cpu_used + pod_cpu <= upper_cpu_cap or not upper_selected:
                upper_selected.append(pod)
                upper_cpu_used += pod_cpu

        lower_infos = [
            (
                self._priority_info(
                    pod,
                    current_slot,
                    pending_pods,
                    flavours,
                    max_time_slots,
                    queue_name="lower-carbon-footprint",
                ),
                pod,
            )
            for pod in lower
        ]
        lower_infos.sort(
            key=lambda item: (
                item[0].priority,
                self.latest_start_slot(item[1], max_time_slots),
                int(getattr(item[1], "earliest_timeslot", 0)),
                item[1].id,
            )
        )

        for pod in urgent:
            self._priority_context[pod.id] = self._priority_info(
                pod,
                current_slot,
                pending_pods,
                flavours,
                max_time_slots,
                queue_name="deadline-override",
                urgent=True,
            )
        for pod in upper_selected:
            self._priority_context[pod.id] = self._priority_info(
                pod,
                current_slot,
                pending_pods,
                flavours,
                max_time_slots,
                queue_name="upper-profile-fcfs",
            )
        for info, pod in lower_infos:
            self._priority_context[pod.id] = info

        return urgent + upper_selected + [pod for _, pod in lower_infos]

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
        if self.node_score_mode == "least_allocated":
            return self._least_allocated_score(after_cpu, flv.totalCpu, after_ram, flv.totalRam)
        return self._most_allocated_score(after_cpu, flv.totalCpu, after_ram, flv.totalRam)

    def _select_resource_only_node(
        self,
        pod: CarbonAwarePod,
        start_slot: int,
        flavours: Sequence[CarbonAwareFlavour],
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
            score = self._node_score(flv, start_slot, duration_slots, pod, leftover_cpu, leftover_ram)
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

        timeslot = next((ts for ts in timeslots if ts.id == current_slot), None)
        if timeslot is None or not is_timeslot_valid(timeslot, pod):
            return None, None, float("inf")

        selected_node = self._select_resource_only_node(
            pod, current_slot, flavours, leftover_cpu, leftover_ram, max_time_slots
        )
        self.iterations += 1
        if selected_node is None:
            return None, None, float("inf")

        priority_info = self._priority_context.get(pod.id)
        if priority_info is None:
            priority_info = self._priority_info(
                pod,
                current_slot,
                self._pending_context or [pod],
                flavours,
                max_time_slots,
                queue_name="lower-carbon-footprint",
            )

        used_cpu_before = self._used_cpu_before_by_slot(
            selected_node, current_slot, self._duration_slots(pod), leftover_cpu
        )
        emissions_g = self._marginal_operational_emissions_g(
            selected_node, current_slot, pod, used_cpu_before
        )

        self._write_placement_to_csv(
            pod=pod,
            node_id=selected_node.id,
            start_slot=current_slot,
            decision_operational_emissions_g=emissions_g,
            priority_info=priority_info,
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
