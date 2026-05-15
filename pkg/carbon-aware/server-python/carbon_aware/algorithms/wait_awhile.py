"""
Let's Wait Awhile-style temporal workload shifting baseline.

This baseline adapts the non-interrupting temporal shifting strategy from
Wiesner et al. to TotEm's fixed Kubernetes batch workload model. It is
intentionally temporal-only: carbon decides when an arrived pod should start,
while node placement after admission uses resource-only Kubernetes-style
MostAllocated scoring.
"""

import csv
import logging
import os
from typing import Dict, List, Optional, Tuple

from carbon_aware.algorithms.piontek_temporal import PiontekTemporalAlgorithm
from carbon_aware.models import CarbonAwareFlavour, CarbonAwarePod


class WaitAwhileAlgorithm(PiontekTemporalAlgorithm):
    """
    Non-interrupting temporal carbon-minimum baseline.

    Let's Wait Awhile evaluates two temporal strategies: interrupting scheduling
    and non-interrupting scheduling. TotEm pods are fixed-duration,
    non-preemptive workloads, so the transferable policy is the non-interrupting
    variant: choose the coherent feasible start window with the lowest average
    carbon intensity.
    """

    _placement_csv_filename = "wait_awhile_placements_session.csv"

    def __init__(
        self,
        perf_logger=None,
        node_score_mode: str = "most_allocated",
    ):
        super().__init__(
            perf_logger=perf_logger,
            node_score_mode=node_score_mode,
        )

    @property
    def name(self) -> str:
        return "Wait-Awhile"

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
                    "wait_awhile_fallback_logs",
                )
            )
            self._ensure_dir_exists(self._session_log_dir)
            logging.warning(
                "Session log directory not set for Wait-Awhile baseline; using %s",
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
                logging.error("Error closing previous Wait-Awhile placement CSV: %s", exc)

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
                        "selected_temporal_start",
                        "selected_temporal_score",
                        "node_score_mode",
                    ]
                )
                self._placement_csv_file_handle.flush()
            logging.info(
                "Wait-Awhile baseline placements will be logged to: %s",
                self._placement_csv_path,
            )
        except IOError as exc:
            logging.error("Failed to open Wait-Awhile placement CSV %s: %s", self._placement_csv_path, exc)
            self._placement_csv_writer = None
            self._placement_csv_file_handle = None
            self._placement_csv_path = None

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
            logging.warning("Wait-Awhile placement CSV writer not available; cannot log placement.")
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
                "wait-awhile-noninterrupting",
                admission_reason,
                co2_window[0],
                co2_window[1],
                self.node_score_mode,
            ]
        )
        if len(self._csv_buffer) >= self._csv_buffer_limit:
            self.flush_session_placement_log()

    def _best_noninterrupting_start(
        self,
        pod: CarbonAwarePod,
        current_slot: int,
        flavours: List[CarbonAwareFlavour],
        max_time_slots: int,
    ) -> Tuple[Optional[int], float]:
        duration_slots = self._duration_slots(pod)
        earliest_slot = max(
            int(getattr(pod, "earliest_timeslot", 0)),
            int(current_slot),
            0,
        )
        latest_start = self.latest_start_slot(pod, max_time_slots)
        if latest_start < earliest_slot:
            return None, float("inf")

        carbon_series = self._cluster_carbon_series(flavours, max_time_slots)
        best_start = None
        best_score = float("inf")
        for candidate_start in range(earliest_slot, latest_start + 1):
            score = self._pod_start_carbon_score(
                carbon_series,
                candidate_start,
                duration_slots,
                max_time_slots,
            )
            key = (score, candidate_start)
            if best_start is None or key < (best_score, best_start):
                best_score = score
                best_start = candidate_start
        return best_start, best_score

    def should_admit_now(
        self,
        pod: CarbonAwarePod,
        current_slot: int,
        flavours: List[CarbonAwareFlavour],
        leftover_cpu: Dict[str, Dict[int, float]],
        max_time_slots: int,
    ) -> Tuple[bool, str, Tuple[int, int]]:
        latest_start = self.latest_start_slot(pod, max_time_slots)
        if latest_start < current_slot:
            return False, "expired", (-1, -1)

        if current_slot >= latest_start:
            return True, "forced_latest_start", (current_slot, current_slot)

        best_start, best_score = self._best_noninterrupting_start(
            pod,
            current_slot,
            flavours,
            max_time_slots,
        )
        if best_start is None:
            return False, "expired", (-1, -1)

        if best_start == current_slot:
            return True, "lowest_average_carbon_window", (best_start, int(round(best_score)))

        return False, "delayed_until_lower_average_carbon", (best_start, int(round(best_score)))
