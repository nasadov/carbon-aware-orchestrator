"""
Precomputation module for the vanilla (K8s-like) scheduler simulation.
Processes all timeslot files sequentially and saves placements to CSV.

Mirrors the interface of heuristic and global-optimal runners so it can be
invoked via main.py --precompute --algorithm vanilla.
"""
import os
import logging
import yaml
import time
import re
from typing import List, Optional

from carbon_aware.utils import (
    CarbonAwarePod, CarbonAwareFlavour, CarbonAwareTimeslot,
    parse_microservice, build_timeslots, load_carbon_intensity_data,
    PerformanceLogger,
)
from carbon_aware.algorithms.vanilla import VanillaAlgorithm
import idl_pb2


def run_vanilla_precomputation(
    workloads_dir: str,
    nodes_file: str,
    forecasts_file: str,
    session_log_dir: Optional[str] = None,
    perf_logger: Optional[PerformanceLogger] = None,
    prioritize_efficiency: bool = False,  # unused in vanilla
    operational_only: bool = False,       # unused in vanilla
) -> bool:
    """
    Run vanilla algorithm precomputation on all timeslot files.

    Args:
        workloads_dir: Directory containing timeslot_*.yaml files
        nodes_file: Path to nodes.yaml file
        forecasts_file: Path to all_forecasts.json file
        session_log_dir: Directory for logging outputs
        perf_logger: Performance logger instance
        prioritize_efficiency: Unused (present for signature compatibility)
        operational_only: Unused (present for signature compatibility)

    Returns:
        True if successful, False otherwise
    """
    try:
        logging.info("🚀 Starting vanilla precomputation mode")
        logging.info(f"📂 Workloads directory: {workloads_dir}")
        logging.info(f"📋 Nodes file: {nodes_file}")
        logging.info(f"🌡️ Forecasts file (unused): {forecasts_file}")

        # 1. Load nodes from nodes.yaml (resource capacities only)
        logging.info("📊 STEP 1: Loading nodes from nodes.yaml")
        flavours = _load_nodes_from_yaml(nodes_file)
        if not flavours:
            logging.error("❌ No valid nodes found in nodes.yaml")
            return False
        logging.info(f"✅ Loaded {len(flavours)} nodes")

        # 2. Initialize vanilla algorithm
        logging.info("🔧 STEP 2: Initializing vanilla algorithm")
        algorithm = VanillaAlgorithm(perf_logger=perf_logger)
        if hasattr(algorithm, 'set_workloads_dir'):
            algorithm.set_workloads_dir(workloads_dir)
        if session_log_dir:
            algorithm.set_base_log_dir(session_log_dir)
        algorithm.setup_session_placement_log()
        logging.info("📝 CSV placement logging initialized")

        # 3. Find all timeslot files
        logging.info("📁 STEP 3: Finding timeslot files")
        if not os.path.exists(workloads_dir):
            logging.error(f"❌ Workloads directory not found: {workloads_dir}")
            return False

        yaml_files = [f for f in os.listdir(workloads_dir) if f.startswith("timeslot_") and f.endswith(".yaml")]
        yaml_files.sort()
        if not yaml_files:
            logging.error(f"❌ No timeslot_*.yaml files found in {workloads_dir}")
            return False
        logging.info(f"✅ Found {len(yaml_files)} timeslot files")

        # 4. Initialize resource state tracking across a default 24h horizon
        leftover_cpu = {flv.id: {} for flv in flavours}
        leftover_ram = {flv.id: {} for flv in flavours}
        max_timeslots = 24
        for flv in flavours:
            for ts_id in range(max_timeslots):
                leftover_cpu[flv.id][ts_id] = flv.totalCpu
                leftover_ram[flv.id][ts_id] = flv.totalRam

        total_pods_processed = 0
        total_pods_placed = 0
        total_pods_failed = 0

        # 5. Process each timeslot file sequentially
        logging.info("🔄 STEP 4: Processing timeslot files sequentially")
        for file_idx, yaml_file in enumerate(yaml_files):
            file_path = os.path.join(workloads_dir, yaml_file)
            logging.info(f"\n📄 Processing file {file_idx + 1}/{len(yaml_files)}: {yaml_file}")
            try:
                pods = _extract_pods_from_yaml(file_path)
                logging.info(f"  📦 Loaded {len(pods)} pods from {yaml_file}")
                if not pods:
                    continue

                file_pods_placed = 0
                file_pods_failed = 0

                for pod_idx, pod in enumerate(pods):
                    total_pods_processed += 1
                    timeslots = build_timeslots(max_timeslots)

                    logging.info(f"  🔍 Pod {pod_idx + 1}/{len(pods)}: {pod.id}")
                    logging.info(f"     Resources: CPU={pod.cpuRequest:.3f}, RAM={pod.ramRequest:.0f}MB")
                    logging.info(f"     Duration: {pod.duration}h, Earliest TS: {pod.earliest_timeslot}")

                    start_time = time.time()
                    selected_flavour, selected_timeslot, _ = algorithm.find_placement(
                        pod, flavours, timeslots, leftover_cpu, leftover_ram, max_timeslots
                    )
                    _exec_ms = (time.time() - start_time) * 1000

                    if selected_flavour and selected_timeslot:
                        # Update resource availability for all covered slots
                        duration_slots = int(pod.duration)
                        for ts_offset in range(duration_slots):
                            ts_id = selected_timeslot.id + ts_offset
                            if ts_id < max_timeslots:
                                leftover_cpu[selected_flavour.id][ts_id] -= pod.cpuRequest
                                leftover_ram[selected_flavour.id][ts_id] -= pod.ramRequest

                        logging.info(f"     ✅ Placed on {selected_flavour.id} at timeslot {selected_timeslot.id}")
                        logging.info(f"     ⏱️ Execution time: {_exec_ms:.2f}ms")
                        file_pods_placed += 1
                        total_pods_placed += 1
                    else:
                        logging.warning(f"     ❌ Failed to place pod {pod.id}")
                        file_pods_failed += 1
                        total_pods_failed += 1

                logging.info(f"  📈 File summary: {file_pods_placed}/{len(pods)} pods placed")
            except Exception as e:
                logging.error(f"❌ Error processing {yaml_file}: {e}")
                import traceback
                logging.error(traceback.format_exc())
                continue

        # 6. Final summary and optional summary generation
        logging.info("\n" + "=" * 60)
        logging.info("🏁 PRECOMPUTATION COMPLETE (vanilla)")
        logging.info(f"📊 Total pods processed: {total_pods_processed}")
        logging.info(f"✅ Successfully placed: {total_pods_placed} ({total_pods_placed/max(1,total_pods_processed)*100:.1f}%)")
        logging.info(f"❌ Failed to place: {total_pods_failed} ({total_pods_failed/max(1,total_pods_processed)*100:.1f}%)")

        if session_log_dir:
            csv_path = os.path.join(session_log_dir, "vanilla_placements_session.csv")
            logging.info(f"💾 Placements saved to: {csv_path}")
            try:
                from carbon_aware.placement_summary import auto_generate_summary_from_session_dir
                summary_path = auto_generate_summary_from_session_dir(session_log_dir, "vanilla")
                if summary_path:
                    logging.info(f"📋 Placement summary generated: {summary_path}")
                else:
                    logging.warning("⚠️ Could not generate placement summary")
            except Exception as e:
                logging.warning(f"⚠️ Failed to generate placement summary: {e}")

        return True

    except Exception as e:
        logging.error(f"❌ Fatal error in precomputation (vanilla): {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False


def _extract_pods_from_yaml(yaml_file: str) -> List[CarbonAwarePod]:
    pods: List[CarbonAwarePod] = []
    try:
        with open(yaml_file, 'r') as f:
            content = f.read()
        documents = yaml.safe_load_all(content)
        for doc in documents:
            if not doc or doc.get('kind') != 'Deployment':
                continue

            metadata = doc.get('metadata', {})
            pod_name = metadata.get('name', '')
            if not pod_name:
                continue

            spec = doc.get('spec', {})
            template = spec.get('template', {})
            pod_spec = template.get('spec', {})
            containers = pod_spec.get('containers', [])
            if not containers:
                continue

            container = containers[0]
            resources = container.get('resources', {})
            requests = resources.get('requests', {})

            labels = metadata.get('labels', {})
            duration_label = labels.get('duration', 'duration-1h')
            deadline_label = labels.get('deadline', 'deadline-24h')

            # Parse hours from labels like "duration-3h", "deadline-6h"
            def _parse_hours(label_value: str, default_val: float) -> float:
                m = re.search(r"(\d+(?:\.\d+)?)h", str(label_value))
                if m:
                    try:
                        return float(m.group(1))
                    except Exception:
                        return default_val
                return default_val

            duration_hours = _parse_hours(duration_label, 1.0)
            deadline_hours = _parse_hours(deadline_label, 24.0)

            # Construct microservice proto and parse as CarbonAwarePod for consistency
            microservice = idl_pb2.Microservice()
            microservice.name = pod_name
            microservice.replicas = spec.get('replicas', 1)
            microservice.duration_hours = str(duration_hours)
            microservice.deadline_hours = str(deadline_hours)

            cpu_str = requests.get('cpu', '100m')
            microservice.cpu_required.value = cpu_str
            microservice.cpu_required.format = 'DecimalSI' if 'm' in cpu_str else 'DecimalExponent'

            mem_str = requests.get('memory', '128Mi')
            microservice.mem_required.value = mem_str
            microservice.mem_required.format = 'BinarySI' if 'i' in mem_str else 'DecimalSI'

            microservice.status = 4  # TO_DEPLOY

            try:
                pod = parse_microservice(microservice)
                pods.append(pod)
            except Exception as e:
                logging.warning(f"Failed to parse pod {pod_name}: {e}")

    except Exception as e:
        logging.error(f"Error reading YAML file {yaml_file}: {e}")

    return pods


def _load_nodes_from_yaml(nodes_file: str) -> List[CarbonAwareFlavour]:
    import re

    logging.info(f"🔄 Loading nodes infrastructure from {nodes_file}")

    if not os.path.exists(nodes_file):
        logging.error(f"❌ Nodes file not found: {nodes_file}")
        return []
    if os.path.getsize(nodes_file) == 0:
        logging.error(f"❌ Nodes file is empty: {nodes_file}")
        return []

    flavours: List[CarbonAwareFlavour] = []
    try:
        logging.info(f"📂 Reading YAML content from {nodes_file}")
        with open(nodes_file, 'r') as f:
            nodes_data = yaml.safe_load_all(f)
            for node in nodes_data:
                if not node:
                    continue

                node_id = node.get("metadata", {}).get("name", "")
                if not node_id:
                    continue

                status = node.get("status", {})
                allocatable = status.get("allocatable", {})

                # CPU cores (allow formats like "200m")
                try:
                    total_cpu = float(allocatable.get("cpu", "0"))
                except ValueError:
                    cpu_str = allocatable.get("cpu", "0")
                    if cpu_str.endswith('m'):
                        total_cpu = float(cpu_str[:-1]) / 1000
                    else:
                        total_cpu = float(cpu_str)

                # Memory in MB (handle Ki/Mi/Gi)
                ram_str = allocatable.get("memory", "0")
                ram_match = re.match(r"(\d+)([KMG]i?)?", ram_str)
                if ram_match:
                    ram_value = float(ram_match.group(1))
                    ram_unit = ram_match.group(2) if ram_match.group(2) else ""
                    if ram_unit.startswith('K'):
                        total_ram = ram_value / 1024
                    elif ram_unit.startswith('M'):
                        total_ram = ram_value
                    elif ram_unit.startswith('G'):
                        total_ram = ram_value * 1024
                    else:
                        total_ram = ram_value
                else:
                    total_ram = 0.0

                # Create flavour with default power and no forecast (unused here)
                flavour = CarbonAwareFlavour(
                    id=node_id,
                    embodiedCarbon=0.0,
                    lifetime=1.0,
                    totalCpu=total_cpu,
                    totalRam=total_ram,
                    totalStorage=0.0,
                    forecast={},
                    power={"idle": 0.0, "active": 0.0, "max": 0.0},
                )
                flavours.append(flavour)

        logging.info(f"Successfully loaded {len(flavours)} nodes from {nodes_file}")
        return flavours

    except Exception as e:
        logging.error(f"Error loading nodes from {nodes_file}: {e}")
        return []


