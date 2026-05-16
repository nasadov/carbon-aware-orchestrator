"""
Precomputation runner for the GreenCourier-style spatial baseline.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

from carbon_aware.algorithms.greencourier_spatial import GreenCourierSpatialAlgorithm
from carbon_aware.precompute_heuristic import _extract_pods_from_yaml, _load_nodes_from_yaml
from carbon_aware.utils import PerformanceLogger, build_timeslots, load_carbon_intensity_data


def run_greencourier_spatial_precomputation(
    workloads_dir: str,
    nodes_file: str,
    forecasts_file: str,
    session_log_dir: Optional[str] = None,
    perf_logger: Optional[PerformanceLogger] = None,
    prioritize_efficiency: bool = False,
    operational_only: bool = True,
    node_score_mode: str = "most_allocated",
) -> bool:
    """
    Run a GreenCourier-style spatial-only carbon-aware baseline.

    The baseline sees only arrived pods. It does not delay pods for cleaner
    future hours. For each pod it finds the earliest feasible start slot, then
    scores feasible nodes by the average regional carbon intensity over the pod
    duration. Ties use Kubernetes-like resource scoring.
    """
    del prioritize_efficiency, operational_only
    try:
        logging.info("Starting GreenCourier-style spatial precomputation")
        logging.info("Workloads directory: %s", workloads_dir)
        logging.info("Nodes file: %s", nodes_file)
        logging.info("Forecasts file: %s", forecasts_file)

        flavours = _load_nodes_from_yaml(nodes_file)
        if not flavours:
            logging.error("No valid nodes found in %s", nodes_file)
            return False

        if not os.path.exists(forecasts_file):
            logging.error("Forecasts file not found: %s", forecasts_file)
            return False
        carbon_forecast = load_carbon_intensity_data(forecasts_file)
        if not carbon_forecast:
            logging.error("No carbon forecasts loaded from %s", forecasts_file)
            return False

        for flv in flavours:
            region = getattr(flv, "region", "")
            if region in carbon_forecast:
                flv.forecast = carbon_forecast[region]
            else:
                fallback_region = next(iter(carbon_forecast.keys()))
                flv.forecast = carbon_forecast[fallback_region]
                logging.warning(
                    "No forecast found for region %s on node %s; using %s",
                    region,
                    flv.id,
                    fallback_region,
                )

        algorithm = GreenCourierSpatialAlgorithm(
            perf_logger=perf_logger,
            node_score_mode=node_score_mode,
        )
        algorithm.set_workloads_dir(workloads_dir)
        if session_log_dir:
            algorithm.set_base_log_dir(session_log_dir)
        algorithm.setup_session_placement_log()

        if not os.path.exists(workloads_dir):
            logging.error("Workloads directory not found: %s", workloads_dir)
            return False

        yaml_files = [
            f for f in os.listdir(workloads_dir)
            if f.startswith("timeslot_") and f.endswith(".yaml")
        ]

        def _timeslot_number(filename: str) -> int:
            match = re.match(r"timeslot_(\d+)\.yaml$", filename)
            return int(match.group(1)) if match else 0

        yaml_files.sort(key=_timeslot_number)
        if not yaml_files:
            logging.error("No timeslot_*.yaml files found in %s", workloads_dir)
            return False

        max_timeslots = 24
        leftover_cpu = {
            flv.id: {slot: flv.totalCpu for slot in range(max_timeslots)}
            for flv in flavours
        }
        leftover_ram = {
            flv.id: {slot: flv.totalRam for slot in range(max_timeslots)}
            for flv in flavours
        }
        timeslots = build_timeslots(max_timeslots)
        total_pods_processed = 0
        total_pods_placed = 0
        total_pods_failed = 0

        for file_idx, yaml_file in enumerate(yaml_files):
            file_path = os.path.join(workloads_dir, yaml_file)
            file_earliest_ts = _timeslot_number(yaml_file)
            logging.info(
                "Processing file %s/%s: %s",
                file_idx + 1,
                len(yaml_files),
                yaml_file,
            )
            pods = _extract_pods_from_yaml(file_path)
            if not pods:
                continue

            for pod in pods:
                total_pods_processed += 1
                pod.earliest_timeslot = file_earliest_ts
                setattr(pod, "_earliest_timeslot_source", "precompute")
                pod.calculate_deadline_slot()

                selected_flavour, selected_timeslot, emissions = algorithm.find_placement(
                    pod, flavours, timeslots, leftover_cpu, leftover_ram, max_timeslots
                )
                if selected_flavour is None or selected_timeslot is None:
                    total_pods_failed += 1
                    logging.info("Pod %s could not be placed", pod.id)
                    continue

                duration_slots = max(1, int(pod.duration))
                for slot in range(selected_timeslot.id, selected_timeslot.id + duration_slots):
                    if slot < max_timeslots:
                        leftover_cpu[selected_flavour.id][slot] -= pod.cpuRequest
                        leftover_ram[selected_flavour.id][slot] -= pod.ramRequest

                total_pods_placed += 1
                logging.info(
                    "Placed pod %s on %s at slot %s, emissions=%.3fg",
                    pod.id,
                    selected_flavour.id,
                    selected_timeslot.id,
                    emissions,
                )

        algorithm.flush_session_placement_log()

        logging.info("GreenCourier spatial precomputation complete")
        logging.info("Total pods processed: %s", total_pods_processed)
        logging.info(
            "Successfully placed: %s (%.1f%%)",
            total_pods_placed,
            total_pods_placed / max(total_pods_processed, 1) * 100.0,
        )
        logging.info(
            "Failed to place: %s (%.1f%%)",
            total_pods_failed,
            total_pods_failed / max(total_pods_processed, 1) * 100.0,
        )

        if session_log_dir:
            try:
                with open(os.path.join(session_log_dir, "pods.txt"), "w") as handle:
                    handle.write(f"pods={total_pods_processed}\n")
                from carbon_aware.placement_summary import auto_generate_summary_from_session_dir

                auto_generate_summary_from_session_dir(session_log_dir, "greencourier_spatial")
            except Exception as exc:
                logging.warning("Could not generate GreenCourier placement summary: %s", exc)

        return True
    except Exception as exc:
        logging.error("Fatal error in GreenCourier spatial precomputation: %s", exc)
        import traceback

        logging.error(traceback.format_exc())
        return False
