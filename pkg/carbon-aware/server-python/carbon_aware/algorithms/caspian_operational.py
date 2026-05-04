"""
Caspian-style spatio-temporal operational-carbon baseline.

The precompute path uses a batch CP-SAT model over pods, nodes, and start times.
It is intentionally operational-only: dynamic and idle operational emissions
are part of the objective, while embodied emissions are left out of the
decision. Post-hoc analyses can still account for embodied emissions using the
shared placement evaluator. The online find_placement method remains as a
greedy fallback for server-mode compatibility.
"""
import csv
import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

from carbon_aware.algorithms.base import SchedulingAlgorithm
from carbon_aware.models import CarbonAwareFlavour, CarbonAwarePod, CarbonAwareTimeslot
from carbon_aware.utils import (
    compute_node_dynamic_coeff_watts,
    get_carbon_intensity,
    is_timeslot_valid,
)


class CaspianOperationalAlgorithm(SchedulingAlgorithm):
    """
    Optimizer-backed spatio-temporal operational-carbon baseline.

    The original Caspian system optimizes carbon-aware multi-cluster Kubernetes
    placement with QoS terms. This simulator baseline preserves that comparison
    axis through an operational-carbon and completion-time objective over
    feasible node/time candidates, without TotEm's embodied-carbon term.
    """

    _placement_csv_filename = "caspian_operational_placements_session.csv"

    def __init__(
        self,
        perf_logger=None,
        carbon_weight: float = 0.70,
        completion_weight: float = 0.20,
        packing_weight: float = 0.10,
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
        self.iterations = 0
        self.steps = 0
        total_weight = max(carbon_weight + completion_weight + packing_weight, 1e-9)
        self.carbon_weight = carbon_weight / total_weight
        self.completion_weight = completion_weight / total_weight
        self.packing_weight = packing_weight / total_weight

    @staticmethod
    def _ensure_dir_exists(directory_path: str) -> None:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            logging.info("Created directory: %s", directory_path)

    def set_base_log_dir(self, base_dir: str) -> None:
        self._session_log_dir = base_dir

    def set_workloads_dir(self, workloads_dir: str) -> None:
        self._workloads_dir = workloads_dir
        logging.info("CaspianOperationalAlgorithm: workloads directory set to %s", workloads_dir)

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
                    "caspian_operational_fallback_logs",
                )
            )
            self._ensure_dir_exists(self._session_log_dir)
            logging.warning(
                "Session log directory not set for Caspian baseline; using %s",
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
                logging.error("Error closing previous Caspian placement CSV: %s", exc)

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
                        "solver_status",
                        "solution_time_seconds",
                        "baseline_variant",
                    ]
                )
                self._placement_csv_file_handle.flush()
            logging.info("Caspian baseline placements will be logged to: %s", self._placement_csv_path)
        except IOError as exc:
            logging.error("Failed to open Caspian placement CSV %s: %s", self._placement_csv_path, exc)
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

    def _write_placement_to_csv(
        self,
        pod_id: str,
        node_id: str,
        start_slot: int,
        duration: float,
        cpu_request: float,
        ram_request: float,
        decision_operational_emissions_g: float,
        solver_status: str = "",
        solution_time_seconds: float = 0.0,
        baseline_variant: str = "caspian-greedy",
    ) -> None:
        if not self._placement_csv_writer or not self._placement_csv_file_handle:
            logging.warning("Caspian placement CSV writer not available; cannot log placement.")
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
                solver_status,
                solution_time_seconds,
                baseline_variant,
            ]
        )
        if len(self._csv_buffer) >= self._csv_buffer_limit:
            self.flush_session_placement_log()

    def flush_session_placement_log(self) -> None:
        if self._placement_csv_writer and self._placement_csv_file_handle and self._csv_buffer:
            self._placement_csv_writer.writerows(self._csv_buffer)
            self._csv_buffer.clear()
            self._placement_csv_file_handle.flush()

    @property
    def name(self) -> str:
        return "Caspian-Operational-Opt"

    def _extract_earliest_timeslot_from_yaml_files(self, pod_id: str) -> int:
        workloads_dir = self._workloads_dir or "../workloads"
        if not os.path.isabs(workloads_dir):
            workloads_dir = os.path.join(os.getcwd(), workloads_dir)

        try:
            for filename in os.listdir(workloads_dir):
                match = re.match(r"timeslot_(\d+)\.yaml$", filename)
                if not match:
                    continue
                filepath = os.path.join(workloads_dir, filename)
                try:
                    with open(filepath, "r") as handle:
                        if f"name: {pod_id}" in handle.read():
                            return int(match.group(1))
                except Exception as exc:
                    logging.warning("Could not read %s while finding pod origin: %s", filepath, exc)
        except Exception as exc:
            logging.warning("Could not scan workloads directory %s: %s", workloads_dir, exc)
        return 0

    def _set_pod_earliest_timeslot(self, pod: CarbonAwarePod) -> None:
        if getattr(pod, "_earliest_timeslot_source", None) == "precompute":
            pod.calculate_deadline_slot()
            return
        if getattr(pod, "earliest_timeslot", None) not in (None, 0):
            pod.calculate_deadline_slot()
            return
        pod.earliest_timeslot = self._extract_earliest_timeslot_from_yaml_files(pod.id)
        pod.calculate_deadline_slot()

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
        node_cpu = leftover_cpu.get(flv.id, {})
        node_ram = leftover_ram.get(flv.id, {})
        for slot in range(start_slot, start_slot + duration_slots):
            if node_cpu.get(slot, 0.0) < pod.cpuRequest:
                return False
            if node_ram.get(slot, 0.0) < pod.ramRequest:
                return False
        return True

    def _used_cpu_before_by_slot(
        self,
        flv: CarbonAwareFlavour,
        start_slot: int,
        duration_slots: int,
        leftover_cpu: Dict[str, Dict[int, float]],
    ) -> Dict[int, float]:
        used = {}
        node_cpu = leftover_cpu.get(flv.id, {})
        for slot in range(start_slot, start_slot + duration_slots):
            used[slot] = max(flv.totalCpu - node_cpu.get(slot, flv.totalCpu), 0.0)
        return used

    def _operational_marginal_emissions_g(
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

        for offset in range(int(pod.duration)):
            slot = start_slot + offset
            used_before = used_cpu_before_by_slot.get(slot, 0.0)
            delta_power_w = dynamic_coeff * pod_cpu_ratio
            if used_before <= 0.0 and pod_cpu_ratio > 0.0:
                delta_power_w += flv.power.get("idle", 0.0)
            total_g += get_carbon_intensity(flv, slot) * (delta_power_w / 1000.0)
        return total_g

    def _candidate_dynamic_operational_g(
        self,
        flv: CarbonAwareFlavour,
        start_slot: int,
        pod: CarbonAwarePod,
    ) -> float:
        """Return the dynamic operational carbon term for a placement candidate."""
        total_cpu = max(flv.totalCpu, 1e-6)
        pod_cpu_ratio = pod.cpuRequest / total_cpu
        dynamic_coeff = compute_node_dynamic_coeff_watts(flv)
        total_g = 0.0
        for offset in range(int(pod.duration)):
            slot = start_slot + offset
            total_g += get_carbon_intensity(flv, slot) * (
                dynamic_coeff * pod_cpu_ratio / 1000.0
            )
        return total_g

    def _candidate_solo_operational_g(
        self,
        flv: CarbonAwareFlavour,
        start_slot: int,
        pod: CarbonAwarePod,
    ) -> float:
        """Return operational emissions if this pod were alone on the node slots."""
        total_cpu = max(flv.totalCpu, 1e-6)
        pod_cpu_ratio = pod.cpuRequest / total_cpu
        dynamic_coeff = compute_node_dynamic_coeff_watts(flv)
        total_g = 0.0
        for offset in range(int(pod.duration)):
            slot = start_slot + offset
            power_w = flv.power.get("idle", 0.0) + dynamic_coeff * pod_cpu_ratio
            total_g += get_carbon_intensity(flv, slot) * (power_w / 1000.0)
        return total_g

    def _idle_operational_g(self, flv: CarbonAwareFlavour, slot: int) -> float:
        return get_carbon_intensity(flv, slot) * (flv.power.get("idle", 0.0) / 1000.0)

    @staticmethod
    def _normalize(value: float, min_value: float, max_value: float) -> float:
        span = max_value - min_value
        if span <= 1e-12:
            return 0.0
        return (value - min_value) / span

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
        duration_slots = max(1, int(pod.duration))
        earliest_slot = max(int(getattr(pod, "earliest_timeslot", 0)), 0)
        deadline_slot = getattr(pod, "deadline_slot", None)
        if deadline_slot is None:
            deadline_slot = max_time_slots
        latest_start = min(int(deadline_slot) - duration_slots, max_time_slots - duration_slots)

        if latest_start < earliest_slot:
            return None, None, float("inf")

        candidates = []
        for ts in timeslots:
            if ts.id < earliest_slot or ts.id > latest_start:
                continue
            if not is_timeslot_valid(ts, pod):
                continue

            for flv in flavours:
                self.steps += 1
                if not self._duration_feasible(
                    flv, ts.id, duration_slots, pod, leftover_cpu, leftover_ram, max_time_slots
                ):
                    continue

                used_cpu_before = self._used_cpu_before_by_slot(
                    flv, ts.id, duration_slots, leftover_cpu
                )
                operational_g = self._operational_marginal_emissions_g(
                    flv, ts.id, pod, used_cpu_before
                )

                cpu_slack_sum = 0.0
                ram_slack_sum = 0.0
                for slot in range(ts.id, ts.id + duration_slots):
                    cpu_slack_sum += max(leftover_cpu[flv.id][slot] - pod.cpuRequest, 0.0)
                    ram_slack_sum += max(leftover_ram[flv.id][slot] - pod.ramRequest, 0.0)
                cpu_norm = flv.totalCpu * duration_slots
                ram_norm = flv.totalRam * duration_slots
                pack_raw = (
                    cpu_slack_sum / max(cpu_norm, 1e-6)
                    + ram_slack_sum / max(ram_norm, 1e-6)
                )

                completion_raw = (ts.id + duration_slots - earliest_slot) / max(
                    float(deadline_slot) - earliest_slot, 1.0
                )

                candidates.append(
                    {
                        "flavour": flv,
                        "timeslot": ts,
                        "operational_g": operational_g,
                        "completion_raw": completion_raw,
                        "pack_raw": pack_raw,
                    }
                )

        self.iterations += len(candidates)
        if not candidates:
            return None, None, float("inf")

        carbon_values = [c["operational_g"] for c in candidates]
        completion_values = [c["completion_raw"] for c in candidates]
        pack_values = [c["pack_raw"] for c in candidates]
        carbon_min, carbon_max = min(carbon_values), max(carbon_values)
        completion_min, completion_max = min(completion_values), max(completion_values)
        pack_min, pack_max = min(pack_values), max(pack_values)

        best_candidate = None
        best_key = None
        for candidate in candidates:
            carbon_norm = self._normalize(candidate["operational_g"], carbon_min, carbon_max)
            completion_norm = self._normalize(
                candidate["completion_raw"], completion_min, completion_max
            )
            pack_norm = self._normalize(candidate["pack_raw"], pack_min, pack_max)
            score = (
                self.carbon_weight * carbon_norm
                + self.completion_weight * completion_norm
                + self.packing_weight * pack_norm
            )
            key = (
                score,
                candidate["operational_g"],
                candidate["completion_raw"],
                candidate["pack_raw"],
                candidate["flavour"].id,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_candidate = candidate

        if best_candidate is None:
            return None, None, float("inf")

        selected_flavour = best_candidate["flavour"]
        selected_timeslot = best_candidate["timeslot"]
        selected_emissions = float(best_candidate["operational_g"])

        self._write_placement_to_csv(
            pod_id=pod.id,
            node_id=selected_flavour.id,
            start_slot=selected_timeslot.id,
            duration=pod.duration,
            cpu_request=pod.cpuRequest,
            ram_request=pod.ramRequest,
            decision_operational_emissions_g=selected_emissions,
        )

        if self.experiment_logger:
            execution_time = time.time() - start_time
            self.experiment_logger.record_placement(
                pod_id=pod.id,
                success=True,
                execution_time=execution_time,
                emissions=selected_emissions,
                considered_options=len(flavours) * len(timeslots),
                selected_node=selected_flavour.id,
                selected_timeslot=selected_timeslot.id,
            )

        return selected_flavour, selected_timeslot, selected_emissions

    def solve_batch_optimizer(
        self,
        pods: List[CarbonAwarePod],
        flavours: List[CarbonAwareFlavour],
        max_time_slots: int = 24,
        current_slot: int = 0,
        available_cpu: Optional[Dict[str, Dict[int, float]]] = None,
        available_ram: Optional[Dict[str, Dict[int, float]]] = None,
        time_limit_seconds: float = 120.0,
        carbon_weight: float = 0.85,
        completion_weight: float = 0.15,
    ) -> Dict[str, Tuple[CarbonAwareFlavour, int, float]]:
        """
        Solve a Caspian-style placement model over currently visible pods.

        The model is intentionally operational-only:
        - binary x[p,n,t] chooses a start time and node,
        - binary y[n,s] charges node idle power once per active node-slot,
        - resource constraints enforce CPU/RAM capacity per node-slot,
        - phase 1 maximizes placed workload count,
        - phase 2 fixes that count and minimizes weighted operational carbon plus
          completion-time penalty.

        In rolling-horizon mode, callers pass current_slot and only pods that
        have already arrived. Future starts are provisional; the caller should
        commit only placements whose selected start is the current epoch.

        Returns a mapping pod_id -> (flavour, start_slot, solo_operational_g).
        """
        try:
            from ortools.sat.python import cp_model
        except ImportError as exc:
            raise RuntimeError("OR-Tools CP-SAT is required for Caspian optimizer baseline") from exc

        solve_start = time.time()
        model = cp_model.CpModel()
        scale = 100000
        carbon_weight = float(carbon_weight)
        completion_weight = float(completion_weight)
        total_weight = max(carbon_weight + completion_weight, 1e-9)
        carbon_weight /= total_weight
        completion_weight /= total_weight

        candidate_meta = {}
        candidates_by_pod: Dict[str, List[Tuple[str, int]]] = {}
        max_carbon_ref = 1e-9

        for pod in pods:
            self._set_pod_earliest_timeslot(pod)
            pod_id = pod.id
            candidates_by_pod[pod_id] = []
            duration_slots = max(1, int(pod.duration))
            earliest_slot = max(
                int(getattr(pod, "earliest_timeslot", 0)),
                int(current_slot),
                0,
            )
            deadline_slot = getattr(pod, "deadline_slot", None)
            if deadline_slot is None:
                deadline_slot = max_time_slots
            latest_start = min(
                int(deadline_slot) - duration_slots,
                max_time_slots - duration_slots,
            )
            if latest_start < earliest_slot:
                continue

            for start_slot in range(earliest_slot, latest_start + 1):
                for flv in flavours:
                    if pod.cpuRequest > flv.totalCpu or pod.ramRequest > flv.totalRam:
                        continue
                    if available_cpu is not None and available_ram is not None:
                        has_capacity = True
                        for slot in range(start_slot, start_slot + duration_slots):
                            cpu_left = available_cpu.get(flv.id, {}).get(slot, 0.0)
                            ram_left = available_ram.get(flv.id, {}).get(slot, 0.0)
                            if cpu_left < pod.cpuRequest or ram_left < pod.ramRequest:
                                has_capacity = False
                                break
                        if not has_capacity:
                            continue
                    dynamic_g = self._candidate_dynamic_operational_g(flv, start_slot, pod)
                    solo_g = self._candidate_solo_operational_g(flv, start_slot, pod)
                    completion_norm = (start_slot + duration_slots - earliest_slot) / max(
                        float(deadline_slot) - earliest_slot,
                        1.0,
                    )
                    candidate_key = (pod_id, flv.id, start_slot)
                    candidate_meta[candidate_key] = {
                        "pod": pod,
                        "flavour": flv,
                        "start_slot": start_slot,
                        "duration_slots": duration_slots,
                        "dynamic_g": dynamic_g,
                        "solo_g": solo_g,
                        "completion_norm": max(0.0, completion_norm),
                    }
                    candidates_by_pod[pod_id].append((flv.id, start_slot))
                    max_carbon_ref = max(max_carbon_ref, solo_g)

        idle_meta = {}
        for flv in flavours:
            for slot in range(max_time_slots):
                idle_g = self._idle_operational_g(flv, slot)
                idle_meta[(flv.id, slot)] = idle_g
                max_carbon_ref = max(max_carbon_ref, idle_g)

        x = {}
        unplaced = {}
        for pod in pods:
            pod_id = pod.id
            unplaced[pod_id] = model.NewBoolVar(f"unplaced_{pod_id}")
            placement_vars = []
            for flv_id, start_slot in candidates_by_pod.get(pod_id, []):
                var = model.NewBoolVar(f"x_{pod_id}_{flv_id}_{start_slot}")
                x[(pod_id, flv_id, start_slot)] = var
                placement_vars.append(var)
            model.Add(sum(placement_vars) + unplaced[pod_id] == 1)

        y = {
            (flv.id, slot): model.NewBoolVar(f"y_{flv.id}_{slot}")
            for flv in flavours
            for slot in range(max_time_slots)
        }

        cpu_terms_by_node_slot: Dict[Tuple[str, int], List[Tuple[int, object]]] = {}
        ram_terms_by_node_slot: Dict[Tuple[str, int], List[Tuple[int, object]]] = {}
        active_links_by_node_slot: Dict[Tuple[str, int], List[object]] = {}

        for (pod_id, flv_id, start_slot), var in x.items():
            meta = candidate_meta[(pod_id, flv_id, start_slot)]
            pod = meta["pod"]
            for slot in range(start_slot, start_slot + meta["duration_slots"]):
                node_slot = (flv_id, slot)
                cpu_terms_by_node_slot.setdefault(node_slot, []).append(
                    (int(round(pod.cpuRequest * 1000)), var)
                )
                ram_terms_by_node_slot.setdefault(node_slot, []).append(
                    (int(round(pod.ramRequest * 1000)), var)
                )
                active_links_by_node_slot.setdefault(node_slot, []).append(var)
                model.Add(var <= y[node_slot])

        flavour_by_id = {flv.id: flv for flv in flavours}
        for flv in flavours:
            for slot in range(max_time_slots):
                node_slot = (flv.id, slot)
                cpu_terms = cpu_terms_by_node_slot.get(node_slot, [])
                ram_terms = ram_terms_by_node_slot.get(node_slot, [])
                if not cpu_terms:
                    model.Add(y[node_slot] == 0)
                    continue
                cpu_capacity = flv.totalCpu
                ram_capacity = flv.totalRam
                if available_cpu is not None:
                    cpu_capacity = available_cpu.get(flv.id, {}).get(slot, 0.0)
                if available_ram is not None:
                    ram_capacity = available_ram.get(flv.id, {}).get(slot, 0.0)
                model.Add(
                    sum(coeff * var for coeff, var in cpu_terms)
                    <= int(round(max(cpu_capacity, 0.0) * 1000))
                )
                model.Add(
                    sum(coeff * var for coeff, var in ram_terms)
                    <= int(round(max(ram_capacity, 0.0) * 1000))
                )

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(time_limit_seconds)
        solver.parameters.num_search_workers = 8

        model.Minimize(sum(unplaced.values()))
        status1 = solver.Solve(model)
        if status1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            self.status = solver.StatusName(status1)
            self.solution_time_seconds = time.time() - solve_start
            logging.warning("Caspian optimizer phase 1 failed with status %s", self.status)
            return {}

        phase1_status = solver.StatusName(status1)
        min_unplaced = int(round(sum(solver.Value(var) for var in unplaced.values())))
        phase1_selected_keys = {key for key, var in x.items() if solver.Value(var) == 1}
        phase1_solution: Dict[str, Tuple[CarbonAwareFlavour, int, float]] = {}
        for key in phase1_selected_keys:
            pod_id, flv_id, start_slot = key
            meta = candidate_meta[key]
            phase1_solution[pod_id] = (
                flavour_by_id[flv_id],
                start_slot,
                float(meta["solo_g"]),
            )

        model.Add(sum(unplaced.values()) == min_unplaced)

        objective_terms = []
        for key, var in x.items():
            meta = candidate_meta[key]
            carbon_coeff = int(
                round(carbon_weight * scale * (meta["dynamic_g"] / max_carbon_ref))
            )
            completion_coeff = int(
                round(completion_weight * scale * meta["completion_norm"])
            )
            coeff = max(carbon_coeff + completion_coeff, 0)
            if coeff:
                objective_terms.append(coeff * var)

        for (flv_id, slot), var in y.items():
            idle_coeff = int(
                round(carbon_weight * scale * (idle_meta[(flv_id, slot)] / max_carbon_ref))
            )
            if idle_coeff:
                objective_terms.append(idle_coeff * var)

        for key, var in x.items():
            model.AddHint(var, 1 if key in phase1_selected_keys else 0)
        for pod_id, var in unplaced.items():
            model.AddHint(var, 0 if pod_id in phase1_solution else 1)
        for key, var in y.items():
            model.AddHint(var, solver.Value(var))

        model.Minimize(sum(objective_terms))
        remaining_time = max(float(time_limit_seconds) - (time.time() - solve_start), 10.0)
        solver.parameters.max_time_in_seconds = remaining_time
        status2 = solver.Solve(model)
        self.status = solver.StatusName(status2)
        self.iterations = len(x)
        self.steps = len(objective_terms)
        self.solution_time_seconds = time.time() - solve_start

        if status2 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            self.status = f"{phase1_status}_PHASE1_FALLBACK_AFTER_{self.status}"
            logging.warning(
                "Caspian optimizer phase 2 failed with status %s; using phase-1 max-placement solution",
                solver.StatusName(status2),
            )
            logging.info(
                "Caspian optimizer fallback status=%s placed=%s/%s unplaced=%s candidates=%s time=%.2fs",
                self.status,
                len(phase1_solution),
                len(pods),
                min_unplaced,
                len(x),
                self.solution_time_seconds,
            )
            return phase1_solution

        solution: Dict[str, Tuple[CarbonAwareFlavour, int, float]] = {}
        for key, var in x.items():
            if solver.Value(var) != 1:
                continue
            pod_id, flv_id, start_slot = key
            meta = candidate_meta[key]
            solution[pod_id] = (
                flavour_by_id[flv_id],
                start_slot,
                float(meta["solo_g"]),
            )

        placed = len(solution)
        logging.info(
            "Caspian optimizer status=%s placed=%s/%s unplaced=%s candidates=%s time=%.2fs",
            self.status,
            placed,
            len(pods),
            min_unplaced,
            len(x),
            self.solution_time_seconds,
        )
        return solution
