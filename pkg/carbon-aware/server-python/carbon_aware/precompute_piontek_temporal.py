"""
Precomputation runner for the Piontek-style temporal Kubernetes baseline.
"""
import logging
import os
import re
from typing import Callable, Optional

from carbon_aware.algorithms.piontek_temporal import PiontekTemporalAlgorithm
from carbon_aware.precompute_heuristic import _extract_pods_from_yaml, _load_nodes_from_yaml
from carbon_aware.utils import PerformanceLogger, build_timeslots, load_carbon_intensity_data


def run_piontek_temporal_precomputation(
    workloads_dir: str,
    nodes_file: str,
    forecasts_file: str,
    session_log_dir: Optional[str] = None,
    perf_logger: Optional[PerformanceLogger] = None,
    prioritize_efficiency: bool = False,
    operational_only: bool = True,
    node_score_mode: str = "most_allocated",
    algorithm_factory: Optional[Callable[..., PiontekTemporalAlgorithm]] = None,
    summary_algorithm_name: str = "piontek_temporal",
    display_name: str = "Piontek-style temporal Kubernetes",
) -> bool:
    """
    Run the Piontek-style temporal CO2-window baseline.

    This is a rolling online simulation. Future carbon forecasts are visible, as
    in the original paper, but future workload arrivals are not. Arrived pods are
    queued until their low-carbon admission window or until the latest start that
    still satisfies their deadline.
    """
    try:
        logging.info("Starting %s precomputation", display_name)
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
            if getattr(flv, "region", None) in carbon_forecast:
                flv.forecast = carbon_forecast[flv.region]
            else:
                fallback_region = next(iter(carbon_forecast.keys()))
                flv.forecast = carbon_forecast[fallback_region]
                logging.warning(
                    "No forecast found for region %s on node %s; using %s",
                    getattr(flv, "region", ""),
                    flv.id,
                    fallback_region,
                )

        logging.info("%s node score mode: %s", display_name, node_score_mode)
        factory = algorithm_factory or PiontekTemporalAlgorithm
        algorithm = factory(
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
        arrivals_by_slot = {slot: [] for slot in range(max_timeslots)}
        total_pods_processed = 0

        for file_idx, yaml_file in enumerate(yaml_files):
            file_path = os.path.join(workloads_dir, yaml_file)
            logging.info("Loading file %s/%s: %s", file_idx + 1, len(yaml_files), yaml_file)
            pods = _extract_pods_from_yaml(file_path)
            if not pods:
                continue

            file_earliest_ts = _timeslot_number(yaml_file)
            for pod in pods:
                pod.earliest_timeslot = file_earliest_ts
                setattr(pod, "_earliest_timeslot_source", "precompute")
                pod.calculate_deadline_slot()
                if 0 <= file_earliest_ts < max_timeslots:
                    arrivals_by_slot.setdefault(file_earliest_ts, []).append(pod)
                    total_pods_processed += 1
                else:
                    logging.warning(
                        "Skipping pod %s with out-of-horizon arrival slot %s",
                        pod.id,
                        file_earliest_ts,
                    )

        logging.info(
            "Loaded %s pods for %s rolling temporal baseline",
            total_pods_processed,
            display_name,
        )

        leftover_cpu = {
            flv.id: {slot: flv.totalCpu for slot in range(max_timeslots)}
            for flv in flavours
        }
        leftover_ram = {
            flv.id: {slot: flv.totalRam for slot in range(max_timeslots)}
            for flv in flavours
        }
        timeslots = build_timeslots(max_timeslots)
        pending_pods = []
        total_pods_placed = 0
        total_pods_failed = 0

        for current_slot in range(max_timeslots):
            arrivals = arrivals_by_slot.get(current_slot, [])
            if arrivals:
                pending_pods.extend(arrivals)
                logging.info(
                    "Timeslot %s: added %s arrivals; pending=%s",
                    current_slot,
                    len(arrivals),
                    len(pending_pods),
                )

            still_pending = []
            for pod in pending_pods:
                if algorithm.latest_start_slot(pod, max_timeslots) < current_slot:
                    total_pods_failed += 1
                    logging.info(
                        "Timeslot %s: pod %s expired before placement",
                        current_slot,
                        pod.id,
                    )
                else:
                    still_pending.append(pod)
            pending_pods = still_pending

            if not pending_pods:
                continue

            pending_pods.sort(
                key=lambda pod: (
                    algorithm.latest_start_slot(pod, max_timeslots) > current_slot,
                    int(getattr(pod, "earliest_timeslot", 0)),
                    algorithm.latest_start_slot(pod, max_timeslots),
                    -float(getattr(pod, "cpuRequest", 0.0)),
                    -float(getattr(pod, "ramRequest", 0.0)),
                    pod.id,
                )
            )

            committed_ids = set()
            for pod in list(pending_pods):
                setattr(pod, "_current_scheduling_slot", current_slot)
                selected_flavour, selected_timeslot, emissions = algorithm.find_placement(
                    pod, flavours, timeslots, leftover_cpu, leftover_ram, max_timeslots
                )
                if selected_flavour is None or selected_timeslot is None:
                    continue

                duration_slots = max(1, int(pod.duration))
                for slot in range(selected_timeslot.id, selected_timeslot.id + duration_slots):
                    if slot < max_timeslots:
                        leftover_cpu[selected_flavour.id][slot] -= pod.cpuRequest
                        leftover_ram[selected_flavour.id][slot] -= pod.ramRequest

                committed_ids.add(pod.id)
                total_pods_placed += 1
                logging.info(
                    "Timeslot %s: placed pod %s on %s, emissions=%.3fg",
                    current_slot,
                    pod.id,
                    selected_flavour.id,
                    emissions,
                )

            if committed_ids:
                pending_pods = [pod for pod in pending_pods if pod.id not in committed_ids]
                logging.info(
                    "Timeslot %s: committed %s placements; pending=%s",
                    current_slot,
                    len(committed_ids),
                    len(pending_pods),
                )

        if pending_pods:
            total_pods_failed += len(pending_pods)
            logging.info("End of horizon: %s pending pods failed", len(pending_pods))

        algorithm.flush_session_placement_log()

        logging.info("%s precomputation complete", display_name)
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

                auto_generate_summary_from_session_dir(session_log_dir, summary_algorithm_name)
            except Exception as exc:
                logging.warning("Could not generate %s placement summary: %s", display_name, exc)

        return True
    except Exception as exc:
        logging.error("Fatal error in %s precomputation: %s", display_name, exc)
        import traceback

        logging.error(traceback.format_exc())
        return False
