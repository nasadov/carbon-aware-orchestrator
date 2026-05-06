"""
Precomputation runner for the Caspian-style operational-carbon baseline.
"""
import logging
import os
import re
from typing import Optional

from carbon_aware.algorithms.caspian_operational import CaspianOperationalAlgorithm
from carbon_aware.precompute_heuristic import _extract_pods_from_yaml, _load_nodes_from_yaml
from carbon_aware.utils import PerformanceLogger, load_carbon_intensity_data


def run_caspian_operational_precomputation(
    workloads_dir: str,
    nodes_file: str,
    forecasts_file: str,
    session_log_dir: Optional[str] = None,
    perf_logger: Optional[PerformanceLogger] = None,
    prioritize_efficiency: bool = False,
    operational_only: bool = True,
) -> bool:
    """
    Run Caspian-style operational-carbon precomputation on all timeslot files.

    The operational_only argument is accepted for CLI compatibility; this
    baseline is always operational-only by design.
    """
    try:
        logging.info("Starting Caspian-style operational-carbon precomputation")
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

        algorithm = CaspianOperationalAlgorithm(perf_logger=perf_logger)
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

            match = re.match(r"timeslot_(\d+)\.yaml$", yaml_file)
            file_earliest_ts = int(match.group(1)) if match else 0

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
            "Loaded %s pods for Caspian-style optimizer baseline",
            total_pods_processed,
        )

        leftover_cpu = {flv.id: {slot: flv.totalCpu for slot in range(max_timeslots)} for flv in flavours}
        leftover_ram = {flv.id: {slot: flv.totalRam for slot in range(max_timeslots)} for flv in flavours}
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
                duration_slots = max(1, int(pod.duration))
                deadline_slot = getattr(pod, "deadline_slot", None)
                if deadline_slot is None:
                    deadline_slot = max_timeslots
                latest_start = min(
                    int(deadline_slot) - duration_slots,
                    max_timeslots - duration_slots,
                )
                if latest_start < current_slot:
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

            solution = algorithm.solve_visible_queue_lp_guided(
                pods=pending_pods,
                flavours=flavours,
                max_time_slots=max_timeslots,
                current_slot=current_slot,
                available_cpu=leftover_cpu,
                available_ram=leftover_ram,
                time_limit_seconds=30.0,
                carbon_weight=0.85,
                completion_weight=0.15,
            )

            pod_by_id = {pod.id: pod for pod in pending_pods}
            committed_ids = set()
            for pod_id, (selected_flavour, start_slot, emissions) in sorted(solution.items()):
                if start_slot != current_slot:
                    continue
                pod = pod_by_id[pod_id]
                duration_slots = max(1, int(pod.duration))

                for slot in range(start_slot, start_slot + duration_slots):
                    if slot < max_timeslots:
                        leftover_cpu[selected_flavour.id][slot] -= pod.cpuRequest
                        leftover_ram[selected_flavour.id][slot] -= pod.ramRequest

                algorithm._write_placement_to_csv(
                    pod_id=pod.id,
                    node_id=selected_flavour.id,
                    start_slot=start_slot,
                    duration=pod.duration,
                    cpu_request=pod.cpuRequest,
                    ram_request=pod.ramRequest,
                    decision_operational_emissions_g=emissions,
                    solver_status=getattr(algorithm, "status", ""),
                    solution_time_seconds=getattr(algorithm, "solution_time_seconds", 0.0),
                    baseline_variant="caspian-style",
                )
                committed_ids.add(pod_id)
                total_pods_placed += 1

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

        logging.info("Caspian precomputation complete")
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

                auto_generate_summary_from_session_dir(session_log_dir, "caspian_operational")
            except Exception as exc:
                logging.warning("Could not generate Caspian placement summary: %s", exc)

        return True
    except Exception as exc:
        logging.error("Fatal error in Caspian precomputation: %s", exc)
        import traceback

        logging.error(traceback.format_exc())
        return False
