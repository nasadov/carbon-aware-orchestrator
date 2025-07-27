"""
gRPC server implementation for the carbon-aware scheduler.
"""
from concurrent import futures
import logging
import signal
import threading
import time
from datetime import datetime as dt


import grpc
import idl_pb2
import idl_pb2_grpc
from google.protobuf import empty_pb2

# Import from our own modules
from carbon_aware.models import CarbonAwarePod, CarbonAwareFlavour, CarbonAwareTimeslot
from carbon_aware.utils import (
    parse_infrastructure, parse_microservice, build_timeslots, 
    print_resource_utilization_report, MICROSERVICE_STATUS_MAP, PerformanceLogger
)
from carbon_aware.state import PersistentStateStorage
from carbon_aware.algorithms import get_algorithm


class Algorithm:
    """
    Tracks which algorithm is in use and whether it's initialized.
    """
    def __init__(self, name: str, initialized: bool) -> None:
        self.name = name
        self.initialized = initialized


class PlacementAlgorithm(idl_pb2_grpc.PlacementAlgorithmServicer):
    """
    gRPC Service Implementation. Each method corresponds to a .proto RPC definition.
    """
    def __init__(self, algorithm_name='heuristic', experiment_logger=None, perf_logger=None, session_log_dir=None, 
                 workloads_dir=None, nodes_file=None, forecasts_file=None, prioritize_efficiency=False,
                 precomputed_solution=None, precomputation_done=False, operational_only=False):
        self.algo = Algorithm(algorithm_name, True)
        self.command_line_algorithm = algorithm_name 
        self.persistent_state = PersistentStateStorage()
        self.experiment_logger = experiment_logger
        self.perf_logger = perf_logger
        self.session_log_dir = session_log_dir # This is the single directory for the entire server session
        self.operational_only = operational_only
        
        # Global optimization parameters
        self.workloads_dir = workloads_dir
        self.nodes_file = nodes_file
        self.forecasts_file = forecasts_file
        self.prioritize_efficiency = prioritize_efficiency
        self.comprehensive_initialized = precomputation_done  # Set to True if precomputation was done
        
        # Store precomputed solution if available
        self.precomputed_solution = precomputed_solution or {}
        self.precomputation_done = precomputation_done
        
        if precomputation_done:
            logging.info(f"🎯 PlacementAlgorithm initialized with precomputed solution containing {len(self.precomputed_solution)} placements")
        else:
            logging.info(f"📋 PlacementAlgorithm initialized for algorithm: {algorithm_name}")

        # Configure perf_logger once if it's provided and session_log_dir is set
        if self.perf_logger and self.session_log_dir:
            import os
            # Performance log filename is fixed for the session
            perf_log_filename = f"{self.command_line_algorithm}_perf_session.csv"
            # The log_dir for PerformanceLogger is the session_log_dir
            self.perf_logger.set_new_log_file(log_dir=self.session_log_dir, filename=perf_log_filename)
            logging.info(f"PlacementAlgorithm: Performance logs will be appended to {self.perf_logger.log_file}")
        elif self.perf_logger:
            logging.warning("PlacementAlgorithm: PerformanceLogger provided but session_log_dir is not set. Performance logs may not be saved correctly.")

    def Init(self, request: idl_pb2.AlgorithmName, context) -> empty_pb2.Empty:
        # Store original request
        requested_name = request.name
        
        # PREVENT OVERRIDE: Always use command-line specified algorithm
        self.algo.name = self.command_line_algorithm
        
        logging.info(f"[Init] Client requested '{requested_name}' but using '{self.command_line_algorithm}' from command line")
        self.algo.initialized = True
        return empty_pb2.Empty()

    def CalculatePlacement(self, request: idl_pb2.Data, context) -> idl_pb2.Placements:
        try:
            # ======== RAW REQUEST LOGGING ========
            logging.info("=" * 80)
            logging.info(f"🚀 STARTING PLACEMENT CALCULATION - ALGORITHM: {self.algo.name}")
            logging.info("=" * 80)
            logging.info(f"📥 RAW REQUEST STRUCTURE:")
            logging.info(f"Request type: {type(request)}")
            
            # Detailed infrastructure logging
            if hasattr(request, "infrastructure") and request.infrastructure:
                logging.info(f"INFRASTRUCTURE: {len(request.infrastructure.nodes)} nodes")
                for i, node in enumerate(request.infrastructure.nodes):
                    logging.info(f"  NODE {i}: name={node.name}")
            else:
                logging.info("No infrastructure in request")
                
            # Detailed workload logging
            if hasattr(request, "workload") and request.workload:
                logging.info(f"WORKLOAD: {len(request.workload.microservices)} microservices")
                for i, ms in enumerate(request.workload.microservices):
                    logging.info(f"  MICROSERVICE {i}: name={ms.name}")

                    # Status information
                    if hasattr(ms, "status"):
                        status_name = MICROSERVICE_STATUS_MAP.get(ms.status, f"Status({ms.status})")
                        
                        # Color coding for different statuses
                        status_color = "\033[0m"  # default
                        if status_name == "RUNNING":
                            status_color = "\033[92m"  # green
                        elif status_name == "TO_DEPLOY":
                            status_color = "\033[91m"  # red
                        elif status_name == "PENDING":
                            status_color = "\033[93m"  # yellow
                            
                        logging.info(f"    - status: {status_color}{status_name}\033[0m")
                    else:
                        logging.info(f"    - status field not found")
                    
                    # Check for duration/deadline fields and their values
                    if hasattr(ms, "duration_hours"):
                        logging.info(f"    - duration: '{ms.duration_hours}' (type: {type(ms.duration_hours).__name__})")
                    else:
                        logging.info(f"    - duration field not found")
                        
                    if hasattr(ms, "deadline_hours"):
                        logging.info(f"    - deadline: '{ms.deadline_hours}' (type: {type(ms.deadline_hours).__name__})")
                    else:
                        logging.info(f"    - deadline field not found")
                    
                    # Log other important attributes
                    if hasattr(ms, "cpu_required") and ms.cpu_required:
                        logging.info(f"    - cpu_required: {ms.cpu_required.value}")
                    if hasattr(ms, "mem_required") and ms.mem_required:
                        logging.info(f"    - mem_required: {ms.mem_required.value}")
            else:
                logging.info("No workload in request")
            
            logging.info("=" * 80)
            
            # ======== INITIALIZATION ========
            if not self.algo.initialized:
                msg = "Algorithm not initialized before calling CalculatePlacement."
                logging.error(f"❌ {msg}")
                context.set_details(msg)
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                return idl_pb2.Placements()

            # ======== INFRASTRUCTURE PARSING ========
            logging.info("-" * 60)
            logging.info(f"📊 INFRASTRUCTURE PARSING")
            start_time = time.time()  # Record algorithm start time
            
            # For heuristic algorithm, read nodes from nodes.yaml instead of client request
            if self.algo.name == "heuristic":
                logging.info(f"🔄 Using nodes.yaml for heuristic algorithm: {self.nodes_file}")
                # Get a temporary heuristic instance to load nodes
                from carbon_aware.algorithms.heuristic import HeuristicAlgorithm
                temp_heuristic = HeuristicAlgorithm()
                flavours = temp_heuristic._load_nodes_from_yaml(self.nodes_file)
                logging.info(f"  ▶ Loaded {len(flavours)} nodes from nodes.yaml in {(time.time() - start_time):.2f}s")
            else:
                # For other algorithms, use client-provided infrastructure
                logging.info(f"🔄 Using client-provided infrastructure for {self.algo.name} algorithm")
                flavours = parse_infrastructure(request.infrastructure)
                logging.info(f"  ▶ Processed {len(flavours)} nodes from client in {(time.time() - start_time):.2f}s")
            
            # Check if carbon intensity data is available
            regions = set()
            for flv in flavours:
                if hasattr(flv, 'region'):
                    regions.add(flv.region)
            logging.info(f"  ▶ Node regions: {', '.join(regions)}")

            # ======== RESOURCE INITIALIZATION ========
            logging.info("-" * 60)
            logging.info(f"🧮 INITIALIZING RESOURCE TRACKING")

            if not self.persistent_state.initialized:
                logging.info("🔄 Initializing persistent resource tracking")
                self.persistent_state.initialize(flavours)
            else:
                logging.info("📊 Using persistent resource tracking from previous calls")

            # Get the current resource state
            leftover_cpu, leftover_ram = self.persistent_state.get_resources()
            max_time_slots = self.persistent_state.max_time_slots
            logging.info(f"  ▶ Using resources for {len(flavours)} nodes × {max_time_slots} timeslots")

            # ======== WORKLOAD PROCESSING ========
            logging.info("-" * 60)
            logging.info(f"🔍 PROCESSING {len(request.workload.microservices)} MICROSERVICES")
            out_placements = idl_pb2.Placements()
            
            # Get the right algorithm implementation based on name
            algorithm_instance = get_algorithm(self.algo.name)

            # Configure algorithm instance with the session directory for placement CSVs
            if self.session_log_dir:
                if hasattr(algorithm_instance, 'set_base_log_dir'):
                    algorithm_instance.set_base_log_dir(self.session_log_dir)
                    # Crucially, also start a new experiment run for this CalculatePlacement call
                    # This will set up the run-specific subdirectory and placement CSV file.
                    if hasattr(algorithm_instance, 'setup_session_placement_log'):
                        algorithm_instance.setup_session_placement_log()
                        logging.info(f"📊 Algorithm placement CSV logging configured")
                    else:
                        logging.warning(f"Algorithm {self.algo.name} does not support session placement logging")
                else:
                    logging.warning(f"Algorithm {self.algo.name} does not support base log directory setting")
            
            # Set operational_only flag if the algorithm supports it
            if hasattr(algorithm_instance, 'set_operational_only'):
                algorithm_instance.set_operational_only(self.operational_only)
                if self.operational_only:
                    logging.info(f"🔥 Algorithm configured for operational-only emissions mode")
            elif self.operational_only:
                logging.warning(f"Algorithm {self.algo.name} does not support operational-only mode")

            # Pass experiment logger to algorithm if supported
            if self.experiment_logger and hasattr(algorithm_instance, 'experiment_logger'):
                algorithm_instance.experiment_logger = self.experiment_logger
            
            placements_success = 0
            placements_failed = 0
            placements_skipped = 0
            total_emissions = 0.0

            for i, ms in enumerate(request.workload.microservices):
                ms_start_time = time.time()
                logging.info(f"  ➡️ ({i+1}/{len(request.workload.microservices)}) Processing: {ms.name}")
                    
                if hasattr(ms, "status"):
                    if ms.status != 4:
                        status_name = MICROSERVICE_STATUS_MAP.get(ms.status, f"Status({ms.status})")
                        logging.info(f"    ⏩ Skipping {ms.name} with status {status_name} - only handling TO_DEPLOY")
                        placement = self._build_fallback_placement(ms.name, f"SKIPPED_{status_name}")
                        out_placements.placements.append(placement)
                        placements_skipped += 1
                        continue

                pod = parse_microservice(ms)
                current_time = dt.fromtimestamp(time.time())
                hours_until_deadline = (pod.deadline - current_time).total_seconds() / 3600
                scheduling_window = max(0, hours_until_deadline - pod.duration)
                
                # DETAILED POD ANALYSIS LOGGING
                logging.info(f"    POD ANALYSIS for {ms.name}:")
                logging.info(f"       Basic Info: duration={pod.duration}h, deadline in {hours_until_deadline:.1f}h")
                logging.info(f"       Resources: CPU={pod.cpuRequest:.3f}, RAM={pod.ramRequest:.0f}MB")
                logging.info(f"       Scheduling window: {scheduling_window:.1f}h ({max(0, int(scheduling_window))} potential timeslots)")
                
                if hours_until_deadline <= 0:
                    logging.warning(f"    DEADLINE_EXPIRED for {ms.name}: deadline was {abs(hours_until_deadline):.1f}h ago")
                    placement = self._build_fallback_placement(ms.name, "EXPIRED_DEADLINE")
                    out_placements.placements.append(placement)
                    placements_failed += 1
                    continue

                # DYNAMIC TIMESLOT RANGE FIX: Consider both deadline and earliest_timeslot constraints
                # Pre-calculate pod's earliest_timeslot to understand its constraint
                from carbon_aware.algorithms.heuristic import HeuristicAlgorithm
                temp_heuristic = HeuristicAlgorithm()
                pod_earliest_timeslot = temp_heuristic._extract_earliest_timeslot_from_yaml_files(pod.id)
                
                # Calculate required timeslot range: max(deadline_range, earliest_timeslot + buffer)
                deadline_range = int(hours_until_deadline)
                earliest_constraint_range = pod_earliest_timeslot + int(pod.duration) + 2  # Add buffer for scheduling
                required_timeslot_range = max(deadline_range, earliest_constraint_range)
                
                logging.info(f"       TIMESLOT RANGE CALCULATION:")
                logging.info(f"          Pod earliest_timeslot: {pod_earliest_timeslot}")
                logging.info(f"          Deadline-based range: {deadline_range} timeslots")
                logging.info(f"          Earliest-constraint range: {earliest_constraint_range} timeslots")
                logging.info(f"          Required range: {required_timeslot_range} timeslots")
                
                timeslots = build_timeslots(required_timeslot_range)
                logging.info(f"       Generated {len(timeslots)} timeslots (expanded for earliest_timeslot constraint)")
                
                if scheduling_window <= 0:
                    logging.warning(f"    ZERO_SCHEDULING_WINDOW for {ms.name} - immediate start required (duration={pod.duration}h = deadline)")
                elif scheduling_window < 2:
                    logging.info(f"    LIMITED_SCHEDULING_WINDOW for {ms.name} ({scheduling_window:.1f}h window)")
                else:
                    logging.info(f"    GOOD_SCHEDULING_FLEXIBILITY for {ms.name} ({scheduling_window:.1f}h window)")

                logging.info(f"    Executing {self.algo.name} algorithm...")
                algorithm_start_time = time.time()
                
                # Use atomic placement method to prevent race conditions
                if hasattr(algorithm_instance, 'find_placement_atomic'):
                    logging.debug(f"    Using atomic placement method for {self.algo.name}")
                    best_node, best_slot, minimal_emissions = algorithm_instance.find_placement_atomic(
                        pod, flavours, timeslots, self.persistent_state, max_time_slots
                    )
                else:
                    # Fallback to non-atomic method for algorithms that don't support it
                    logging.debug(f"    Using non-atomic placement method for {self.algo.name}")
                    best_node, best_slot, minimal_emissions = algorithm_instance.find_placement(
                        pod, flavours, timeslots, leftover_cpu, leftover_ram, max_time_slots
                    )
                    
                    # If using non-atomic method, we still need to update resources separately
                    if best_node and best_slot:
                        start_slot_id = best_slot.id
                        duration_slots = int(pod.duration)
                        
                        # Try to atomically allocate (this may fail if resources were taken)
                        allocation_success = self.persistent_state.atomic_check_and_allocate(
                            best_node.id, start_slot_id, duration_slots, 
                            pod.cpuRequest, pod.ramRequest
                        )
                        
                        if not allocation_success:
                            logging.warning(f"    RESOURCE_ALLOCATION_FAILED for {ms.name} - resources taken by another thread")
                            best_node = None
                            best_slot = None
                            minimal_emissions = float('inf')
                
                algorithm_execution_time = time.time() - algorithm_start_time
                logging.info(f"    Algorithm execution time: {algorithm_execution_time:.3f}s")
                
                # DETAILED PLACEMENT RESULT LOGGING
                if best_node and best_slot:
                    logging.info(f"    PLACEMENT_SUCCESS for {ms.name}:")
                    logging.info(f"       Selected: node={best_node.id}, timeslot={best_slot.id}")
                    logging.info(f"       Emissions: {minimal_emissions/1000.0:.6f}kgCO2e ({minimal_emissions:.2f}gCO2e)")
                    logging.info(f"       Resources used: CPU={pod.cpuRequest:.3f}/{best_node.totalCpu:.2f}, RAM={pod.ramRequest:.0f}/{best_node.totalRam:.0f}MB")
                else:
                    logging.warning(f"    PLACEMENT_FAILED for {ms.name}:")
                    if minimal_emissions == float('inf'):
                        logging.warning(f"       Failure reason: NO_FEASIBLE_PLACEMENT_FOUND")
                    else:
                        logging.warning(f"       Failure reason: RESOURCE_ALLOCATION_FAILED")
                    
                    # DETAILED FAILURE ANALYSIS
                    logging.warning(f"       FAILURE_ANALYSIS for {ms.name}:")
                    logging.warning(f"           Available nodes: {len(flavours)}")
                    logging.warning(f"           Available timeslots: {len(timeslots)}")
                    logging.warning(f"           Total combinations checked: {len(flavours) * len(timeslots)}")
                    
                    # Check resource availability on each node
                    leftover_cpu, leftover_ram = self.persistent_state.get_resources()
                    for node in flavours:
                        node_has_cpu = any(leftover_cpu[node.id][slot] >= pod.cpuRequest for slot in range(max_time_slots))
                        node_has_ram = any(leftover_ram[node.id][slot] >= pod.ramRequest for slot in range(max_time_slots))
                        logging.warning(f"           Node {node.id}: sufficient_cpu={node_has_cpu}, sufficient_ram={node_has_ram}")
                        
                        # Check specific timeslots
                        feasible_slots = []
                        for ts in timeslots:
                            if ts.id < max_time_slots:
                                cpu_ok = leftover_cpu[node.id][ts.id] >= pod.cpuRequest
                                ram_ok = leftover_ram[node.id][ts.id] >= pod.ramRequest
                                if cpu_ok and ram_ok:
                                    feasible_slots.append(ts.id)
                        logging.warning(f"           Node {node.id}: feasible_timeslots={feasible_slots}")
                
                if self.experiment_logger:
                    success = best_node is not None and best_slot is not None
                    node_id = best_node.id if best_node else None
                    slot_id = best_slot.id if best_slot else None
                    considered_options = len(flavours) * len(timeslots)
                    self.experiment_logger.record_placement(
                        pod_id=pod.id,
                        success=success,
                        execution_time=algorithm_execution_time,
                        emissions=minimal_emissions if success else 0.0,
                        considered_options=considered_options,
                        selected_node=node_id,
                        selected_timeslot=slot_id
                    )

                if best_node and best_slot:
                    # Resources are already allocated by the atomic method
                    # No need to call update_resources again
                    self.persistent_state.record_placement(
                        ms.name, best_node.id, best_slot.getStart(), pod.duration
                    )
                    
                    import datetime
                    workload_start_time = best_slot.getStart()
                    end_time = workload_start_time + datetime.timedelta(hours=pod.duration)
                    
                    placement = self._build_success_placement(ms.name, best_node, best_slot, minimal_emissions, flavours)
                    total_emissions += minimal_emissions / 1000.0  # Convert from g CO₂e to kg CO₂e
                    placements_success += 1
                    
                    logging.info(f"    Final placement: {best_node.id} at {workload_start_time.strftime('%Y-%m-%d %H:%M')}")
                    logging.info(f"       Duration: {pod.duration}h, Finishes: {end_time.strftime('%Y-%m-%d %H:%M')}")
                    logging.info(f"       Emissions: {minimal_emissions/1000.0:.6f}kgCO2e ({minimal_emissions:.2f}gCO2e), Resources: CPU={pod.cpuRequest:.2f}/{best_node.totalCpu:.2f}, " +
                                f"RAM={pod.ramRequest:.0f}/{best_node.totalRam:.0f}MB")
                else:
                    placement = self._build_fallback_placement(ms.name, "NONE_FOUND")
                    placements_failed += 1
                    logging.warning(f"    Final result: No feasible placement found for {ms.name}")

                out_placements.placements.append(placement)
                logging.info(f"    Processing time: {(time.time() - ms_start_time):.3f}s")

            logging.info("=" * 60)
            logging.info(f"📋 PLACEMENT SUMMARY")
            logging.info(f"  ▶ Total microservices: {len(request.workload.microservices)}")
            logging.info(f"  ▶ Successfully placed: {placements_success}")
            logging.info(f"  ▶ Failed to place: {placements_failed}")
            logging.info(f"  ▶ Skipped (non-TO_DEPLOY): {placements_skipped}") 
            logging.info(f"  ▶ Total carbon footprint: {total_emissions:.6f}kgCO2e ({total_emissions*1000:.2f}gCO2e)")
            logging.info(f"  ▶ Total execution time: {(time.time() - start_time):.3f}s")
            logging.info("=" * 80)

            print_resource_utilization_report(flavours, leftover_cpu, leftover_ram, max_time_slots, hours_to_show=24)
            
            if self.perf_logger and self.perf_logger.log_file:
                cpu_util = 0
                mem_util = 0
                total_cpu = 0
                total_ram = 0
                used_cpu = 0  # Initialize used_cpu
                used_ram = 0  # Initialize used_ram
                
                for flv in flavours:
                    node_id = flv.id
                    total_cpu += flv.totalCpu
                    total_ram += flv.totalRam
                    if node_id in leftover_cpu and 0 in leftover_cpu[node_id]:
                        used_cpu += (flv.totalCpu - leftover_cpu[node_id][0])
                    if node_id in leftover_ram and 0 in leftover_ram[node_id]:
                        used_ram += (flv.totalRam - leftover_ram[node_id][0])
                
                if total_cpu > 0:
                    cpu_util = (used_cpu / total_cpu) * 100
                if total_ram > 0:
                    mem_util = (used_ram / total_ram) * 100
                
                microservice_names = [ms.name for ms in request.workload.microservices]
                
                algorithm_metrics = {
                    'iterations': getattr(algorithm_instance, 'iterations', 0),
                    'steps': getattr(algorithm_instance, 'steps', 0)
                }
                
                metrics_summary = self.perf_logger.log_placement_call(
                    execution_time=(time.time() - start_time),
                    algorithm_name=self.algo.name,
                    pods_total=len(request.workload.microservices),
                    pods_processed=len(request.workload.microservices) - placements_skipped,
                    pods_placed=placements_success,
                    pods_failed=placements_failed,
                    pods_skipped=placements_skipped,
                    total_emissions=total_emissions,
                    algorithm_metrics=algorithm_metrics,
                    flavours=flavours,
                    timeslots_count=max_time_slots,
                    cpu_util=cpu_util,
                    memory_util=mem_util,
                    microservices=microservice_names
                )
                
                logging.info(f"📊 Performance metrics logged to {self.perf_logger.log_file}")
                logging.info(f"📈 Call #{self.perf_logger.call_counter}: {metrics_summary['execution_time_ms']:.1f}ms, " +
                           f"placed {metrics_summary['pods_placed']}/{metrics_summary['pods_total']} pods " +
                           f"({metrics_summary['avg_placement_time_ms']:.1f}ms/pod)")
            
            return out_placements

        except Exception as e:
            logging.exception(f"❌ Uncaught exception: {e}")
            context.set_details(f"Server error: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return idl_pb2.Placements()


    def _build_success_placement(
        self, microservice_name: str,
        best_node: CarbonAwareFlavour,
        best_slot: CarbonAwareTimeslot,
        emissions: float,
        all_flavours: list[CarbonAwareFlavour]
    ) -> idl_pb2.Placement:
        """
        Build a placement proto for a successful scheduling decision.
        Include all nodes with scores (best node gets meaningful score, others get 0).
        """
        placement = idl_pb2.Placement()
        placement.microservice_name = microservice_name
        replica_score = idl_pb2.ReplicaScores()

        # Scale down from 100000 to 100 for the best node's score
        best_score = int(100 - min(emissions, 100))  # Cap at 100 to ensure non-negative score

        # Add a score entry for every node
        for flavour in all_flavours:
            score_msg = idl_pb2.Score()
            score_msg.node = flavour.id
            
            # Only the best node gets a non-zero score
            if flavour.id == best_node.id:
                score_msg.score = best_score
            else:
                score_msg.score = 0  # All other nodes get zero score
                
            replica_score.scores.append(score_msg)

        # Convert best_slot's start to epoch time
        epoch_time = int(best_slot.getStart().timestamp())
        placement.time_to_schedule = epoch_time

        placement.replica_scores.append(replica_score)

        logging.info(
            f"[_build_success_placement] microservice={microservice_name}, best_node={best_node.id}, "
            f"timeslot={best_slot.id}, best_score={best_score}, nodes_scored={len(all_flavours)}"
        )
        return placement
    
    def _build_fallback_placement(self, microservice_name: str, reason: str) -> idl_pb2.Placement:
        """
        Build a fallback placement proto if no feasible allocation is found or the
        request is invalid.
        """
        logging.warning(
            f"[_build_fallback_placement] microservice={microservice_name}, reason={reason}"
        )

        placement = idl_pb2.Placement()
        placement.microservice_name = microservice_name
        replica_score = idl_pb2.ReplicaScores()

        fallback_score = idl_pb2.Score()
        fallback_score.node = reason
        fallback_score.score = 0  # 0 indicates no preference
        replica_score.scores.append(fallback_score)

        placement.time_to_schedule = int(time.time())  # "Now"
        placement.replica_scores.append(replica_score)
        return placement


def serve(port='50051', algorithm='heuristic', experiment_logger=None, perf_logger=None, session_log_dir=None,
         workloads_dir="./workloads", nodes_file="../nodes.yaml", forecasts_file="./all_forecasts.json",
         prioritize_efficiency=False, operational_only=False):
    """
    Creates and runs the gRPC server on the specified port, registering the PlacementAlgorithm servicer.
    Implements graceful shutdown handling.
    
    Args:
        port (str): Port to listen on, defaults to '50051'
        algorithm (str): Algorithm to use ('heuristic' or 'optimal')
        experiment_logger (ExperimentLogger, optional): Logger for experiment metrics
        perf_logger (PerformanceLogger, optional): Logger for performance metrics
        session_log_dir (str, optional): Single directory for all logs of this server session.
        workloads_dir (str): Directory containing timeslot_*.yaml files
        nodes_file (str): Path to nodes.yaml
        forecasts_file (str): Path to all_forecasts.json
        prioritize_efficiency (bool): Whether to prioritize carbon efficiency per CPU
        operational_only (bool): Whether to use only operational emissions (omit embodied emissions)
    """
    shutdown_in_progress = False
    
    # MILP Precomputation for global-optimal algorithm BEFORE starting server
    if algorithm == 'global-optimal':
        logging.info("=" * 80)
        logging.info("🧮 STARTING MILP PRECOMPUTATION BEFORE SERVER STARTUP")
        logging.info("=" * 80)
        
        try:
            from carbon_aware.algorithms.global_optimal import GlobalOptimalAlgorithm
            from carbon_aware.algorithms import set_precomputed_global_optimal
            
            # Create algorithm instance for precomputation
            precompute_algorithm = GlobalOptimalAlgorithm()
            
            # Register it so the factory can reuse it
            set_precomputed_global_optimal(precompute_algorithm)
            
            # Set up session log directory if available
            if session_log_dir:
                precompute_algorithm.set_base_log_dir(session_log_dir)
                precompute_algorithm.setup_session_placement_log()
                logging.info(f"📊 CSV logging configured for precomputation")
            
            # Perform the comprehensive MILP precomputation
            logging.info(f"🔄 Starting comprehensive global optimization...")
            logging.info(f"  📂 Workloads directory: {workloads_dir}")
            logging.info(f"  📋 Nodes file: {nodes_file}")
            logging.info(f"  📊 Forecasts file: {forecasts_file}")
            
            precomputation_start_time = time.time()
            
            success = precompute_algorithm.precompute_all_workloads(
                workloads_dir=workloads_dir,
                nodes_file=nodes_file,
                forecasts_file=forecasts_file
            )
            
            precomputation_time = time.time() - precomputation_start_time
            
            if success:
                logging.info("=" * 80)
                logging.info("✅ MILP PRECOMPUTATION COMPLETED SUCCESSFULLY!")
                logging.info(f"⏱️  Total precomputation time: {precomputation_time:.2f} seconds")
                logging.info(f"📊 Global solution contains {len(precompute_algorithm.global_solution)} pod placements")
                logging.info("🎯 THE EXPERIMENT CAN NOW START - SERVER IS READY!")
                logging.info("=" * 80)
                
                # Store the precomputed solution for use by the servicer
                global_precomputed_solution = precompute_algorithm.global_solution
                global_optimization_done = True
                
            else:
                logging.error("=" * 80)
                logging.error("❌ MILP PRECOMPUTATION FAILED!")
                logging.error(f"⏱️  Time spent attempting precomputation: {precomputation_time:.2f} seconds")
                logging.error("🚫 SERVER STARTUP ABORTED")
                logging.error("=" * 80)
                return
                
        except Exception as e:
            logging.error("=" * 80)
            logging.error(f"❌ CRITICAL ERROR during MILP precomputation: {e}")
            logging.error("🚫 SERVER STARTUP ABORTED")
            logging.error("=" * 80)
            import traceback
            logging.error(traceback.format_exc())
            return
    else:
        global_precomputed_solution = {}
        global_optimization_done = False
        logging.info(f"ℹ️  Algorithm '{algorithm}' does not require MILP precomputation")
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # Pass parameters to the servicer
    servicer = PlacementAlgorithm(
        algorithm_name=algorithm, 
        experiment_logger=experiment_logger, 
        perf_logger=perf_logger, 
        session_log_dir=session_log_dir,
        workloads_dir=workloads_dir,
        nodes_file=nodes_file,
        forecasts_file=forecasts_file,
        prioritize_efficiency=prioritize_efficiency,
        precomputed_solution=global_precomputed_solution,
        precomputation_done=global_optimization_done,
        operational_only=operational_only
    )
    idl_pb2_grpc.add_PlacementAlgorithmServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:' + port)

    def graceful_shutdown(sig, frame):
        nonlocal shutdown_in_progress
        if shutdown_in_progress:
            return
        
        shutdown_in_progress = True
        logging.info("⏳ Received shutdown signal, stopping server gracefully...")
        
        if experiment_logger and hasattr(experiment_logger, 'session_started'):
            experiment_logger.end_session()
            experiment_logger.save_all_results()
            experiment_logger.generate_report()
            logging.info("📊 Experiment results saved")
        
        threading.Thread(target=server.stop, args=(5,)).start()
        
        logging.info("👋 Server shutdown initiated")
    
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)
    
    servicer.perf_logger = perf_logger
    
    server.start()
    logging.info(f"🚀 Server started, listening on port {port}")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        if not shutdown_in_progress:
            logging.info("Keyboard interrupt received")
            graceful_shutdown(signal.SIGINT, None)
    finally:
        logging.info("Server shutdown complete")