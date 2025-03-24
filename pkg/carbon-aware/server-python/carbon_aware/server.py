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
    print_resource_utilization_report, MICROSERVICE_STATUS_MAP
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
    def __init__(self, algorithm='heuristic', experiment_logger=None):
        self.algo = Algorithm(algorithm, True)
        self.command_line_algorithm = algorithm  # Remember what was specified
        self.persistent_state = PersistentStateStorage()
        self.experiment_logger = experiment_logger

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
            flavours = parse_infrastructure(request.infrastructure)
            logging.info(f"  ▶ Processed {len(flavours)} nodes in {(time.time() - start_time):.2f}s")
            
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
            algorithm = get_algorithm(self.algo.name)

            # Add these lines to start experiment session and inject logger
            if self.experiment_logger:
                # Start session if not already started
                if not hasattr(self.experiment_logger, 'session_started'):
                    self.experiment_logger.start_session(self.algo.name)
                    self.experiment_logger.session_started = True
                # Inject logger into algorithm
                algorithm.experiment_logger = self.experiment_logger
            """
            # Add timer around algorithm execution
            ms_algorithm_start_time = time.time()
            best_node, best_slot, minimal_emissions = algorithm.find_placement(
                pod, flavours, timeslots, leftover_cpu, leftover_ram, max_time_slots
            )
            ms_algorithm_time = time.time() - ms_algorithm_start_time

            # Record metrics if we're in experiment mode
            if self.experiment_logger and not hasattr(algorithm, 'experiment_logger'):
                # Direct measurement in case algorithm implementation doesn't track metrics
                success = best_node is not None and best_slot is not None
                node_id = best_node.id if best_node else None
                slot_id = best_slot.id if best_slot else None
                considered_options = len(flavours) * len(timeslots)  # Rough estimate
                self.experiment_logger.record_placement(
                    pod_id=pod.id,
                    success=success,
                    execution_time=ms_algorithm_time,
                    emissions=minimal_emissions if success else 0.0,
                    considered_options=considered_options,
                    selected_node=node_id,
                    selected_timeslot=slot_id
                )
            """
            # Summary counters
            placements_success = 0
            placements_failed = 0
            placements_skipped = 0
            total_emissions = 0.0

            for i, ms in enumerate(request.workload.microservices):
                ms_start_time = time.time()
                logging.info(f"  ➡️ ({i+1}/{len(request.workload.microservices)}) Processing: {ms.name}")
                    
                if hasattr(ms, "status"):
                    # Process only TO_DEPLOY status
                    if ms.status != 4:  # Only process TO_DEPLOY (4)
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
                
                if hours_until_deadline <= 0:
                    logging.warning(f"    ⚠️  Expired deadline for {ms.name}")
                    placement = self._build_fallback_placement(ms.name, "EXPIRED_DEADLINE")
                    out_placements.placements.append(placement)
                    placements_failed += 1
                    continue

                # Build timeslots for this pod
                timeslots = build_timeslots(hours_until_deadline)
                
                # Enhanced logging to emphasize scheduling flexibility
                logging.info(f"    ⏰ Pod {pod.id}: duration={pod.duration}h, deadline in {hours_until_deadline:.1f}h")
                logging.info(f"    🔄 Scheduling window: {scheduling_window:.1f}h ({len(timeslots)} potential timeslots)")
                
                if scheduling_window <= 0:
                    logging.warning(f"    ⚠️  No scheduling flexibility for {ms.name} - immediate start required")
                elif scheduling_window < 2:
                    logging.info(f"    ℹ️  Limited scheduling window for {ms.name}")
                else:
                    logging.info(f"    ✨ Good scheduling flexibility for {ms.name} - can optimize for carbon")

                # Find optimal placement using the selected algorithm
                best_node, best_slot, minimal_emissions = algorithm.find_placement(
                    pod, flavours, timeslots, leftover_cpu, leftover_ram, max_time_slots
                )

                # Build placement result
                if best_node and best_slot:
                    # Update resource tracking for the ENTIRE DURATION
                    start_slot_id = best_slot.id
                    duration_slots = int(pod.duration)  # Convert hours to slots
                    
                    # Update persistent state
                    self.persistent_state.update_resources(
                        best_node.id, start_slot_id, duration_slots, 
                        pod.cpuRequest, pod.ramRequest
                    )
                    self.persistent_state.record_placement(
                        ms.name, best_node.id, best_slot.getStart(), pod.duration
                    )
                    
                    # Log the reservation
                    logging.info(f"    🔒 Reserved resources for {pod.id} on {best_node.id} for slots {start_slot_id} to {start_slot_id + duration_slots - 1}")
                    
                    # Calculate when this workload will start and end
                    import datetime
                    workload_start_time = best_slot.getStart()
                    end_time = workload_start_time + datetime.timedelta(hours=pod.duration)
                    
                    placement = self._build_success_placement(ms.name, best_node, best_slot, minimal_emissions, flavours)
                    total_emissions += minimal_emissions
                    placements_success += 1
                    
                    logging.info(f"    ✅ Placed on {best_node.id} at {workload_start_time.strftime('%Y-%m-%d %H:%M')}")
                    logging.info(f"       Duration: {pod.duration}h, Finishes: {end_time.strftime('%Y-%m-%d %H:%M')}")
                    logging.info(f"       Emissions: {minimal_emissions:.2f}kgCO2e, Resources: CPU={pod.cpuRequest:.2f}/{best_node.totalCpu:.2f}, " +
                                f"RAM={pod.ramRequest:.0f}/{best_node.totalRam:.0f}MB")
                else:
                    placement = self._build_fallback_placement(ms.name, "NONE_FOUND")
                    placements_failed += 1
                    logging.warning(f"    ❌ No feasible placement found for {ms.name}")

                out_placements.placements.append(placement)
                logging.info(f"    🕒 Processing time: {(time.time() - ms_start_time):.3f}s")

            # ======== SUMMARY ========
            logging.info("=" * 60)
            logging.info(f"📋 PLACEMENT SUMMARY")
            logging.info(f"  ▶ Total microservices: {len(request.workload.microservices)}")
            logging.info(f"  ▶ Successfully placed: {placements_success}")
            logging.info(f"  ▶ Failed to place: {placements_failed}")
            logging.info(f"  ▶ Skipped (non-TO_DEPLOY): {placements_skipped}") 
            logging.info(f"  ▶ Total carbon footprint: {total_emissions:.2f}kgCO2e")
            logging.info(f"  ▶ Total execution time: {(time.time() - start_time):.3f}s")
            logging.info("=" * 80)

            print_resource_utilization_report(flavours, leftover_cpu, leftover_ram, max_time_slots, hours_to_show=24)
            
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


def serve(port='50051', algorithm='heuristic', experiment_logger=None):
    """
    Creates and runs the gRPC server on the specified port, registering the PlacementAlgorithm servicer.
    Implements graceful shutdown handling.
    
    Args:
        port (str): Port to listen on, defaults to '50051'
        algorithm (str): Algorithm to use ('heuristic' or 'optimal')
        experiment_logger (ExperimentLogger, optional): Logger for experiment metrics
    """
    shutdown_in_progress = False
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = PlacementAlgorithm(algorithm, experiment_logger)
    idl_pb2_grpc.add_PlacementAlgorithmServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:' + port)

    # Define graceful shutdown handler with experiment handling
    def graceful_shutdown(sig, frame):
        nonlocal shutdown_in_progress
        if shutdown_in_progress:
            return  # Skip if shutdown already in progress
        
        shutdown_in_progress = True
        logging.info("⏳ Received shutdown signal, stopping server gracefully...")
        
        # Add experiment results saving
        if experiment_logger and hasattr(experiment_logger, 'session_started'):
            experiment_logger.end_session()
            experiment_logger.save_all_results()
            experiment_logger.generate_report()
            logging.info("📊 Experiment results saved")
        
        # Give ongoing requests time to complete
        threading.Thread(target=server.stop, args=(5,)).start()  # 5 second timeout
        
        logging.info("👋 Server shutdown initiated")
    
    # Register signal handlers
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)
    
    server.start()
    logging.info(f"🚀 Server started, listening on port {port}")
    
    try:
        # This is a blocking call until server is terminated
        server.wait_for_termination()
    except KeyboardInterrupt:
        # Only log if not already shutting down
        if not shutdown_in_progress:
            logging.info("Keyboard interrupt received")
            graceful_shutdown(signal.SIGINT, None)
    finally:
        logging.info("Server shutdown complete")