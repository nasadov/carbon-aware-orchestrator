"""
Precomputation module for the heuristic algorithm.
Processes all timeslot files sequentially and saves placements to CSV.
"""

import os
import logging
import yaml
import time
import re
from dataclasses import dataclass
from datetime import datetime as dt
from typing import List, Dict, Optional, Any, Tuple

# Import carbon aware data types and utilities
from carbon_aware.utils import (
    CarbonAwarePod, CarbonAwareTimeslot,
    parse_microservice, build_timeslots, load_carbon_intensity_data,
    PerformanceLogger
)
from carbon_aware.models import EnvironmentalFlavor
from carbon_aware.water_signals import attach_water_metadata
from carbon_aware.algorithms.heuristic import (
    CandidatePlacement,
    HeuristicAlgorithm,
    _candidate_water_value,
    find_ranked_candidates,
)
import idl_pb2


@dataclass
class _GreedyPlacement:
    pod: CarbonAwarePod
    candidate: CandidatePlacement


def run_heuristic_precomputation(
    workloads_dir: str,
    nodes_file: str, 
    forecasts_file: str,
    session_log_dir: Optional[str] = None,
    perf_logger: Optional[PerformanceLogger] = None,
    prioritize_efficiency: bool = False,
    operational_only: bool = False,
    embodied_mode: str = "proportional",
    heuristic_objective: str = "carbon",
    heuristic_carbon_weight: float = 1.0,
    heuristic_water_budget: Optional[float] = None,
    heuristic_water_metric: str = "scarcity",
    heuristic_budget_pressure_weight: float = 1.0,
) -> bool:
    """
    Run heuristic algorithm precomputation on all timeslot files.
    
    Args:
        workloads_dir: Directory containing timeslot_*.yaml files
        nodes_file: Path to nodes.yaml file
        forecasts_file: Path to all_forecasts.json file  
        session_log_dir: Directory for logging outputs
        perf_logger: Performance logger instance
        prioritize_efficiency: Whether to prioritize carbon efficiency
        operational_only: Whether to use operational emissions only
        heuristic_objective: Heuristic objective mode (`carbon`, `weighted-sum`, `pareto`, or `epsilon-pareto`)
        heuristic_carbon_weight: Weighted-sum carbon share in [0, 1]
        heuristic_water_budget: Soft run-level water budget for `epsilon-pareto`
        heuristic_water_metric: Water metric used by water-aware heuristic modes
        heuristic_budget_pressure_weight: Strength of adaptive per-pod budget pressure
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logging.info("🚀 Starting heuristic precomputation mode")
        logging.info(f"📂 Workloads directory: {workloads_dir}")
        logging.info(f"📋 Nodes file: {nodes_file}")
        logging.info(f"🌡️ Forecasts file: {forecasts_file}")
        
        # 1. Load nodes from nodes.yaml
        logging.info("📊 STEP 1: Loading nodes from nodes.yaml")
        flavours = _load_nodes_from_yaml(nodes_file)
        
        if not flavours:
            logging.error("❌ No valid nodes found in nodes.yaml")
            return False
            
        logging.info(f"✅ Loaded {len(flavours)} nodes")
        
        # 2. Load carbon forecast
        logging.info("🌡️ STEP 2: Loading carbon intensity forecast")
        if not os.path.exists(forecasts_file):
            logging.error(f"❌ Forecasts file not found: {forecasts_file}")
            return False
            
        carbon_forecast = load_carbon_intensity_data(forecasts_file)
        logging.info(f"✅ Loaded carbon forecast with {len(carbon_forecast)} regions")
        
        # 3. Initialize heuristic algorithm
        logging.info("🔧 STEP 3: Initializing heuristic algorithm")
        algorithm = HeuristicAlgorithm(perf_logger=perf_logger)
        
        # Set additional options if supported
        if hasattr(algorithm, 'set_prioritize_efficiency'):
            algorithm.set_prioritize_efficiency(prioritize_efficiency)
        if hasattr(algorithm, 'set_operational_only'):
            algorithm.set_operational_only(operational_only)
        if hasattr(algorithm, 'set_workloads_dir'):
            algorithm.set_workloads_dir(workloads_dir)  # Fix hardcoded path bug!
        if hasattr(algorithm, 'set_embodied_allocation_mode'):
            algorithm.set_embodied_allocation_mode(embodied_mode)
        if hasattr(algorithm, 'set_environmental_objective'):
            algorithm.set_environmental_objective(
                mode=heuristic_objective,
                carbon_weight=heuristic_carbon_weight,
                water_metric=heuristic_water_metric,
            )
        if hasattr(algorithm, 'set_water_budget'):
            algorithm.set_water_budget(
                water_budget=heuristic_water_budget,
                water_metric=heuristic_water_metric,
                total_pods=None,
                budget_pressure_weight=heuristic_budget_pressure_weight,
            )
        
        # Use provided session_log_dir as-is; main.py now includes the mode suffix
        if session_log_dir:
            algorithm.set_base_log_dir(session_log_dir)
            
        # Setup session placement logging
        algorithm.setup_session_placement_log()
        logging.info("📝 CSV placement logging initialized")
        
        # Assign carbon forecast to flavours (use 'forecast' attribute used by emissions utils)
        for flv in flavours:
            if flv.region in carbon_forecast:
                flv.forecast = carbon_forecast[flv.region]
                logging.debug(f"✓ Assigned forecast to node {flv.id} in region {flv.region}")
            else:
                logging.warning(f"⚠️ No forecast found for region {flv.region} (node {flv.id})")
        
        # 4. Find all timeslot files
        logging.info("📁 STEP 4: Finding timeslot files")
        if not os.path.exists(workloads_dir):
            logging.error(f"❌ Workloads directory not found: {workloads_dir}")
            return False
            
        yaml_files = [f for f in os.listdir(workloads_dir) 
                     if f.startswith("timeslot_") and f.endswith(".yaml")]
        yaml_files.sort()  # Process in order
        
        if not yaml_files:
            logging.error(f"❌ No timeslot_*.yaml files found in {workloads_dir}")
            return False
            
        logging.info(f"✅ Found {len(yaml_files)} timeslot files")

        total_pods_estimate = 0
        for yaml_file in yaml_files:
            total_pods_estimate += len(_extract_pods_from_yaml(os.path.join(workloads_dir, yaml_file)))
        logging.info(f"📦 Estimated total pods for heuristic budget guidance: {total_pods_estimate}")
        if hasattr(algorithm, 'set_water_budget'):
            algorithm.set_water_budget(
                water_budget=heuristic_water_budget,
                water_metric=heuristic_water_metric,
                total_pods=total_pods_estimate,
                budget_pressure_weight=heuristic_budget_pressure_weight,
            )
        
        # 5. Process each timeslot file sequentially
        logging.info("🔄 STEP 5: Processing timeslot files sequentially")
        
        total_pods_processed = 0
        total_pods_placed = 0
        total_pods_failed = 0
        
        # Initialize resource state tracking
        leftover_cpu = {flv.id: {} for flv in flavours}
        leftover_ram = {flv.id: {} for flv in flavours}
        
        # Initialize with full capacity for all timeslots (standard 24 hour horizon)
        max_timeslots = 24
        for flv in flavours:
            for ts_id in range(max_timeslots):
                leftover_cpu[flv.id][ts_id] = flv.totalCpu
                leftover_ram[flv.id][ts_id] = flv.totalRam

        if heuristic_objective == "epsilon-pareto" and heuristic_water_budget is not None:
            return _run_epsilon_pareto_repair_precomputation(
                algorithm=algorithm,
                yaml_files=yaml_files,
                workloads_dir=workloads_dir,
                flavours=flavours,
                leftover_cpu=leftover_cpu,
                leftover_ram=leftover_ram,
                max_timeslots=max_timeslots,
                session_log_dir=session_log_dir,
                embodied_mode=embodied_mode,
                operational_only=operational_only,
                heuristic_water_budget=heuristic_water_budget,
                heuristic_water_metric=heuristic_water_metric,
            )
        
        # Process each file
        for file_idx, yaml_file in enumerate(yaml_files):
            file_path = os.path.join(workloads_dir, yaml_file)
            logging.info(f"\n📄 Processing file {file_idx + 1}/{len(yaml_files)}: {yaml_file}")
            
            try:
                # Load pods from YAML file
                pods = _extract_pods_from_yaml(file_path)
                logging.info(f"  📦 Loaded {len(pods)} pods from {yaml_file}")
                
                if not pods:
                    continue
                
                # Process each pod (order by tightest window first, then larger pods)
                file_pods_placed = 0
                file_pods_failed = 0

                # Compute scheduling windows for ordering
                now_dt = dt.fromtimestamp(time.time())
                ordered_pods = []
                for pod in pods:
                    try:
                        hours_until_deadline = (pod.deadline - now_dt).total_seconds() / 3600
                    except Exception:
                        hours_until_deadline = getattr(pod, 'deadline_hours', 24.0)
                    scheduling_window = max(0.0, hours_until_deadline - pod.duration)
                    ordered_pods.append((pod, scheduling_window, hours_until_deadline))
                # EDF (smallest window first), then CPU desc, RAM desc, duration desc
                ordered_pods.sort(key=lambda x: (x[1], -x[0].cpuRequest, -x[0].ramRequest, -x[0].duration))
                
                for pod_idx, (pod, scheduling_window, hours_until_deadline) in enumerate(ordered_pods):
                    total_pods_processed += 1
                    
                    # Create timeslots for precomputation mode
                    # In precomputation, all pods should have access to the full scheduling horizon
                    # rather than being limited by wall-clock time deadlines
                    timeslots = build_timeslots(max_timeslots)
                    
                    logging.info(f"  🔍 Pod {pod_idx + 1}/{len(ordered_pods)}: {pod.id}")
                    logging.info(f"     Resources: CPU={pod.cpuRequest:.3f}, RAM={pod.ramRequest:.0f}MB")
                    logging.info(f"     Duration: {pod.duration}h, Earliest TS: {pod.earliest_timeslot}")
                    logging.info(f"     Window: {scheduling_window:.1f}h (deadline in {hours_until_deadline:.1f}h)")
                    

                    
                    # Find placement using heuristic algorithm
                    start_time = time.time()
                    
                    selected_flavour, selected_timeslot, emissions = algorithm.find_placement(
                        pod, flavours, timeslots, leftover_cpu, leftover_ram, max_timeslots
                    )
                    
                    execution_time_ms = (time.time() - start_time) * 1000
                    
                    if selected_flavour and selected_timeslot:
                        # Update resource availability
                        # Convert duration from hours to number of timeslots (each timeslot = 1h)
                        duration_timeslots = int(pod.duration)
                        
                        for ts_offset in range(duration_timeslots):
                            ts_id = selected_timeslot.id + ts_offset
                            if ts_id < max_timeslots:
                                leftover_cpu[selected_flavour.id][ts_id] -= pod.cpuRequest
                                leftover_ram[selected_flavour.id][ts_id] -= pod.ramRequest
                        
                        logging.info(f"     ✅ Placed on {selected_flavour.id} at timeslot {selected_timeslot.id}")
                        logging.info(f"     📊 Emissions: {emissions:.2f}g CO2e")
                        logging.info(f"     ⏱️ Execution time: {execution_time_ms:.2f}ms")
                        
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
        
        # 6. Final summary
        logging.info("\n" + "="*60)
        logging.info("🏁 PRECOMPUTATION COMPLETE")
        logging.info(f"📊 Total pods processed: {total_pods_processed}")
        logging.info(f"✅ Successfully placed: {total_pods_placed} ({total_pods_placed/max(1,total_pods_processed)*100:.1f}%)")
        logging.info(f"❌ Failed to place: {total_pods_failed} ({total_pods_failed/max(1,total_pods_processed)*100:.1f}%)")
        
        if session_log_dir:
            csv_path = getattr(algorithm, "_placement_csv_path", None)
            if csv_path:
                logging.info(f"💾 Placements saved to: {csv_path}")
            
            # Generate placement summary automatically
            try:
                from carbon_aware.placement_summary import auto_generate_summary_from_session_dir
                summary_path = auto_generate_summary_from_session_dir(
                    session_log_dir,
                    "heuristic",
                    workloads_dir=workloads_dir,
                )
                if summary_path:
                    logging.info(f"📋 Placement summary generated: {summary_path}")
                else:
                    logging.warning("⚠️ Could not generate placement summary")
            except Exception as e:
                logging.warning(f"⚠️ Failed to generate placement summary: {e}")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Fatal error in precomputation: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False


def _apply_candidate_resources(
    pod: CarbonAwarePod,
    candidate: CandidatePlacement,
    leftover_cpu: Dict[str, Dict[int, float]],
    leftover_ram: Dict[str, Dict[int, float]],
    *,
    release: bool = False,
) -> None:
    sign = 1.0 if release else -1.0
    for ts_offset in range(int(pod.duration)):
        ts_id = candidate.timeslot.id + ts_offset
        leftover_cpu[candidate.flavour.id][ts_id] += sign * pod.cpuRequest
        leftover_ram[candidate.flavour.id][ts_id] += sign * pod.ramRequest


def _placement_totals(placements: List[_GreedyPlacement], water_metric: str) -> Tuple[float, float, float]:
    carbon_kg = sum(placement.candidate.footprint.total_carbon_g for placement in placements) / 1000.0
    raw_water_l = sum(placement.candidate.footprint.total_raw_water_l for placement in placements)
    water_value = sum(_candidate_water_value(placement.candidate.footprint, water_metric) for placement in placements)
    return carbon_kg, raw_water_l, water_value


def _order_pods_for_heuristic(pods: List[CarbonAwarePod]) -> List[Tuple[CarbonAwarePod, float, float]]:
    now_dt = dt.fromtimestamp(time.time())
    ordered_pods = []
    for pod in pods:
        try:
            hours_until_deadline = (pod.deadline - now_dt).total_seconds() / 3600
        except Exception:
            hours_until_deadline = getattr(pod, 'deadline_hours', 24.0)
        scheduling_window = max(0.0, hours_until_deadline - pod.duration)
        ordered_pods.append((pod, scheduling_window, hours_until_deadline))
    ordered_pods.sort(key=lambda x: (x[1], -x[0].cpuRequest, -x[0].ramRequest, -x[0].duration))
    return ordered_pods


def _run_epsilon_pareto_repair_precomputation(
    *,
    algorithm: HeuristicAlgorithm,
    yaml_files: List[str],
    workloads_dir: str,
    flavours: List[EnvironmentalFlavor],
    leftover_cpu: Dict[str, Dict[int, float]],
    leftover_ram: Dict[str, Dict[int, float]],
    max_timeslots: int,
    session_log_dir: Optional[str],
    embodied_mode: str,
    operational_only: bool,
    heuristic_water_budget: float,
    heuristic_water_metric: str,
) -> bool:
    logging.info("🔧 Running epsilon-pareto heuristic as carbon-greedy schedule plus local water repair")

    all_pods: List[CarbonAwarePod] = []
    for yaml_file in yaml_files:
        all_pods.extend(_extract_pods_from_yaml(os.path.join(workloads_dir, yaml_file)))
    ordered_pods = _order_pods_for_heuristic(all_pods)

    placements: List[_GreedyPlacement] = []
    total_pods_failed = 0

    for pod_idx, (pod, scheduling_window, hours_until_deadline) in enumerate(ordered_pods):
        algorithm._set_pod_earliest_timeslot(pod)
        timeslots = build_timeslots(max_timeslots)
        ranked_candidates = find_ranked_candidates(
            pod=pod,
            flavours=flavours,
            timeslots=timeslots,
            leftover_cpu=leftover_cpu,
            leftover_ram=leftover_ram,
            max_time_slots=max_timeslots,
            operational_only=operational_only,
            embodied_allocation_mode=embodied_mode,
            objective_mode="carbon",
            carbon_weight=1.0,
            water_metric=heuristic_water_metric,
        )
        if not ranked_candidates:
            logging.warning("     ❌ Failed to place pod %s in initial carbon-greedy pass", pod.id)
            total_pods_failed += 1
            continue

        candidate = ranked_candidates[0]
        _apply_candidate_resources(pod, candidate, leftover_cpu, leftover_ram)
        placements.append(_GreedyPlacement(pod=pod, candidate=candidate))
        logging.debug(
            "Initial epsilon-pareto placement %s/%s: pod=%s node=%s slot=%s carbon=%.3f water=%.3f",
            pod_idx + 1,
            len(ordered_pods),
            pod.id,
            candidate.flavour.id,
            candidate.timeslot.id,
            candidate.footprint.total_carbon_g,
            _candidate_water_value(candidate.footprint, heuristic_water_metric),
        )

    initial_carbon_kg, initial_raw_water_l, current_water = _placement_totals(placements, heuristic_water_metric)
    logging.info(
        "Initial carbon-greedy epsilon-pareto pass: placed=%s/%s carbon=%.6fkg raw=%.6f water(%s)=%.6f budget=%.6f",
        len(placements),
        len(ordered_pods),
        initial_carbon_kg,
        initial_raw_water_l,
        heuristic_water_metric,
        current_water,
        heuristic_water_budget,
    )

    repair_iterations = 0
    max_repairs = max(len(placements) * 4, 1)
    while current_water > heuristic_water_budget + 1e-9 and repair_iterations < max_repairs:
        best_move = None
        best_score = None

        for placement_idx, placement in enumerate(placements):
            old_candidate = placement.candidate
            old_water = _candidate_water_value(old_candidate.footprint, heuristic_water_metric)
            old_carbon = old_candidate.footprint.total_carbon_g

            _apply_candidate_resources(placement.pod, old_candidate, leftover_cpu, leftover_ram, release=True)
            try:
                ranked_candidates = find_ranked_candidates(
                    pod=placement.pod,
                    flavours=flavours,
                    timeslots=build_timeslots(max_timeslots),
                    leftover_cpu=leftover_cpu,
                    leftover_ram=leftover_ram,
                    max_time_slots=max_timeslots,
                    operational_only=operational_only,
                    embodied_allocation_mode=embodied_mode,
                    objective_mode="pareto",
                    carbon_weight=1.0,
                    water_metric=heuristic_water_metric,
                )
                for candidate in ranked_candidates:
                    if candidate.flavour.id == old_candidate.flavour.id and candidate.timeslot.id == old_candidate.timeslot.id:
                        continue

                    new_water = _candidate_water_value(candidate.footprint, heuristic_water_metric)
                    water_reduction = old_water - new_water
                    if water_reduction <= 1e-12:
                        continue

                    carbon_delta = candidate.footprint.total_carbon_g - old_carbon
                    new_total_water = current_water - water_reduction
                    remaining_gap = max(new_total_water - heuristic_water_budget, 0.0)
                    overshoot = abs(new_total_water - heuristic_water_budget)
                    carbon_per_water_saved = carbon_delta / water_reduction
                    score = (
                        carbon_per_water_saved,
                        remaining_gap,
                        overshoot,
                        candidate.footprint.total_carbon_g,
                        new_water,
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best_move = (placement_idx, candidate, water_reduction, carbon_delta)
            finally:
                _apply_candidate_resources(placement.pod, old_candidate, leftover_cpu, leftover_ram)

        if best_move is None:
            logging.warning(
                "Epsilon-pareto repair stopped above budget: water(%s)=%.6f budget=%.6f",
                heuristic_water_metric,
                current_water,
                heuristic_water_budget,
            )
            break

        placement_idx, new_candidate, water_reduction, carbon_delta = best_move
        placement = placements[placement_idx]
        _apply_candidate_resources(placement.pod, placement.candidate, leftover_cpu, leftover_ram, release=True)
        _apply_candidate_resources(placement.pod, new_candidate, leftover_cpu, leftover_ram)
        placement.candidate = new_candidate
        current_water -= water_reduction
        repair_iterations += 1
        logging.info(
            "Epsilon-pareto repair %s: pod=%s node=%s slot=%s water_reduction=%.6f carbon_delta=%.6fg remaining_water=%.6f",
            repair_iterations,
            placement.pod.id,
            new_candidate.flavour.id,
            new_candidate.timeslot.id,
            water_reduction,
            carbon_delta,
            current_water,
        )

    for placement in placements:
        algorithm._write_placement_to_csv(
            pod_id=placement.pod.id,
            node_id=placement.candidate.flavour.id,
            start_slot=placement.candidate.timeslot.id,
            duration=placement.pod.duration,
            cpu_request=placement.pod.cpuRequest,
            ram_request=placement.pod.ramRequest,
            flavour=placement.candidate.flavour,
            footprint=placement.candidate.footprint,
        )

    final_carbon_kg, final_raw_water_l, final_water = _placement_totals(placements, heuristic_water_metric)
    logging.info("\n" + "="*60)
    logging.info("🏁 EPSILON-PARETO HEURISTIC PRECOMPUTATION COMPLETE")
    logging.info(f"📊 Total pods processed: {len(ordered_pods)}")
    logging.info(f"✅ Successfully placed: {len(placements)} ({len(placements)/max(1,len(ordered_pods))*100:.1f}%)")
    logging.info(f"❌ Failed to place: {total_pods_failed} ({total_pods_failed/max(1,len(ordered_pods))*100:.1f}%)")
    logging.info(
        "📊 Final totals: carbon=%.6fkg raw=%.6fL water(%s)=%.6f budget=%.6f repairs=%s",
        final_carbon_kg,
        final_raw_water_l,
        heuristic_water_metric,
        final_water,
        heuristic_water_budget,
        repair_iterations,
    )

    if session_log_dir:
        csv_path = getattr(algorithm, "_placement_csv_path", None)
        if csv_path:
            logging.info(f"💾 Placements saved to: {csv_path}")
        try:
            from carbon_aware.placement_summary import auto_generate_summary_from_session_dir
            summary_path = auto_generate_summary_from_session_dir(
                session_log_dir,
                "heuristic",
                workloads_dir=workloads_dir,
            )
            if summary_path:
                logging.info(f"📋 Placement summary generated: {summary_path}")
            else:
                logging.warning("⚠️ Could not generate placement summary")
        except Exception as e:
            logging.warning(f"⚠️ Failed to generate placement summary: {e}")

    return True


def _extract_pods_from_yaml(yaml_file: str) -> List[CarbonAwarePod]:
    """
    Extract pods from a timeslot YAML file.
    
    Args:
        yaml_file: Path to the YAML file
        
    Returns:
        List of CarbonAwarePod objects
    """
    pods = []
    
    try:
        with open(yaml_file, 'r') as f:
            content = f.read()
            
        # Parse multiple YAML documents (separated by ---)
        documents = yaml.safe_load_all(content)
        
        for doc in documents:
            if not doc or doc.get('kind') != 'Deployment':
                continue
                
            # Extract metadata
            metadata = doc.get('metadata', {})
            pod_name = metadata.get('name', '')
            
            if not pod_name:
                continue
            
            # Create a mock microservice object
            spec = doc.get('spec', {})
            template = spec.get('template', {})
            pod_spec = template.get('spec', {})
            containers = pod_spec.get('containers', [])
            
            if not containers:
                continue
                
            container = containers[0]
            resources = container.get('resources', {})
            requests = resources.get('requests', {})
            
            # Extract duration and deadline from labels at the deployment level
            labels = metadata.get('labels', {})
            duration_label = labels.get('duration', 'duration-1h')  # e.g., "duration-3h"
            deadline_label = labels.get('deadline', 'deadline-24h')  # e.g., "deadline-4h"
            
            # Parse duration (e.g., "duration-3h" -> 3.0)
            duration_match = re.search(r'duration-(\d+)h', duration_label)
            duration_hours = float(duration_match.group(1)) if duration_match else 1.0
            
            # Parse deadline (e.g., "deadline-4h" -> 4.0)  
            deadline_match = re.search(r'deadline-(\d+)h', deadline_label)
            deadline_hours = float(deadline_match.group(1)) if deadline_match else 24.0
            
            # Debug logging
            logging.debug(f"Pod {pod_name}: duration_label='{duration_label}' -> duration_hours={duration_hours}")
            logging.debug(f"Pod {pod_name}: deadline_label='{deadline_label}' -> deadline_hours={deadline_hours}")
            
            # Create microservice protobuf object
            microservice = idl_pb2.Microservice()
            microservice.name = pod_name
            microservice.replicas = spec.get('replicas', 1)
            
            # Set duration and deadline using the string attributes
            microservice.duration_hours = str(duration_hours)
            microservice.deadline_hours = str(deadline_hours)
            
            # Set CPU request
            cpu_str = requests.get('cpu', '100m')
            microservice.cpu_required.value = cpu_str
            microservice.cpu_required.format = 'DecimalSI' if 'm' in cpu_str else 'DecimalExponent'
            
            # Set memory request  
            mem_str = requests.get('memory', '128Mi')
            microservice.mem_required.value = mem_str
            microservice.mem_required.format = 'BinarySI' if 'i' in mem_str else 'DecimalSI'
            
            # Set status to TO_DEPLOY
            microservice.status = 4  # TO_DEPLOY
            
            # Parse into CarbonAwarePod
            try:
                pod = parse_microservice(microservice)
                pods.append(pod)
            except Exception as e:
                logging.warning(f"Failed to parse pod {pod_name}: {e}")
                
    except Exception as e:
        logging.error(f"Error reading YAML file {yaml_file}: {e}")
        
    return pods


def _load_nodes_from_yaml(nodes_file: str) -> List[EnvironmentalFlavor]:
    """
    Load nodes directly from nodes.yaml file.
    
    Args:
        nodes_file: Path to nodes.yaml file
    
    Returns:
        List of EnvironmentalFlavor objects
    """
    import re
    
    logging.info(f"🔄 Loading nodes infrastructure from {nodes_file}")
    
    # Check if file exists
    if not os.path.exists(nodes_file):
        logging.error(f"❌ Nodes file not found: {nodes_file}")
        return []
        
    # Check if file is empty
    if os.path.getsize(nodes_file) == 0:
        logging.error(f"❌ Nodes file is empty: {nodes_file}")
        return []
        
    flavours = []
    
    try:
        logging.info(f"📂 Reading YAML content from {nodes_file}")
        with open(nodes_file, 'r') as f:
            nodes_data = yaml.safe_load_all(f)
            for node in nodes_data:
                if not node:
                    continue
                    
                # Extract node ID
                node_id = node.get("metadata", {}).get("name", "")
                if not node_id:
                    continue
                    
                # Extract annotations
                annotations = node.get("metadata", {}).get("annotations", {})
                
                # Extract status
                status = node.get("status", {})
                allocatable = status.get("allocatable", {})
                
                # Extract CPU, RAM and set defaults
                try:
                    total_cpu = float(allocatable.get("cpu", "0"))
                except ValueError:
                    # Handle unit suffixes like '2' or '200m'
                    cpu_str = allocatable.get("cpu", "0")
                    if cpu_str.endswith('m'):
                        total_cpu = float(cpu_str[:-1]) / 1000
                    else:
                        total_cpu = float(cpu_str)
                
                # Parse RAM (convert from Ki, Mi, Gi to MB for consistency with pod requests)
                ram_str = allocatable.get("memory", "0")
                ram_match = re.match(r'(\d+)([KMG]i?)?', ram_str)
                
                if ram_match:
                    ram_value = float(ram_match.group(1))
                    ram_unit = ram_match.group(2) if ram_match.group(2) else ""
                    
                    if ram_unit.startswith('K'):
                        total_ram = ram_value / 1024  # Ki to MB
                    elif ram_unit.startswith('M'):
                        total_ram = ram_value  # Mi to MB (same)
                    elif ram_unit.startswith('G'):
                        total_ram = ram_value * 1024  # Gi to MB
                    else:
                        total_ram = ram_value  # Assume MB if no unit
                else:
                    total_ram = 0
                
                # Parse embodied carbon
                try:
                    embodied_carbon = float(annotations.get("hardware.carbon/embodied_emissions", "0")) * 1000.0  # Convert kg to grams
                except ValueError:
                    embodied_carbon = 0
                
                # Parse lifetime
                try:
                    lifetime_years = float(annotations.get("hardware.carbon/lifetime_years", "3"))
                except ValueError:
                    lifetime_years = 3
                
                # Convert lifetime from years to hours
                lifetime_hours = lifetime_years * 365 * 24
                
                # Parse power settings
                power_settings = {
                    "idle": float(annotations.get("hardware.power/idle_watts", "100")),
                    "active": float(annotations.get("hardware.power/active_watts", "200")),
                    "max": float(annotations.get("hardware.power/max_watts", "400"))
                }
                
                # Extract region for later forecasting
                region_label = node.get("metadata", {}).get("labels", {}).get("topology.kubernetes.io/region", "")
                
                hardware_subcategory = node.get("metadata", {}).get("labels", {}).get("hardware.carbon/subcategory", "")

                flavour = EnvironmentalFlavor(
                    id=node_id,
                    embodiedCarbon=embodied_carbon,
                    lifetime=lifetime_hours,  # In hours
                    totalCpu=total_cpu,
                    totalRam=total_ram,
                    totalStorage=1000 * 1024 * 1024 * 1024,  # Default 1TB
                    forecast={},  # Empty forecast, will be filled later
                    power=power_settings
                )
                attach_water_metadata(
                    flavour,
                    region=region_label,
                    hardware_subcategory=hardware_subcategory,
                    slot_count=24,
                )
                
                flavours.append(flavour)
                
                logging.debug(f"Loaded node {node_id} with {total_cpu} CPU, {total_ram} RAM, " +
                            f"region={flavour.region}, embodied={embodied_carbon}, " +
                            f"lifetime={lifetime_hours}h, power={power_settings}")
        
        logging.info(f"Successfully loaded {len(flavours)} nodes from {nodes_file}")
        return flavours
        
    except Exception as e:
        logging.error(f"Error loading nodes from {nodes_file}: {e}")
        return []
