"""
Main entry point for the Carbon-Aware Orchestrator.
"""
import argparse
import logging
import signal
import sys
import os
from datetime import datetime

from carbon_aware.server import serve
from carbon_aware.experiments import ExperimentLogger
import re
import glob

# Global variable to track placed pods
placed_pods = set()

def get_total_pod_count(workloads_dir):
    """
    Compute total pod count by summing Deployments across all timeslot files.
    This works correctly with round-robin timeslot assignment.
    """
    try:
        timeslot_files = glob.glob(os.path.join(workloads_dir, "timeslot_*.yaml"))
        if not timeslot_files:
            return None

        total = 0
        for path in timeslot_files:
            try:
                with open(path, 'r') as f:
                    content = f.read()
                # Count occurrences of kind: Deployment at start of YAML docs
                total += len(re.findall(r'^kind:\s*Deployment\b', content, flags=re.M))
            except Exception:
                continue

        if total == 0:
            return None

        logging.info(f"📊 Detected {total} total pods in workload (summed across {len(timeslot_files)} files)")
        return total
    except Exception as e:
        logging.warning(f"Could not determine pod count from workloads: {e}")
        return None

def log_experiment_summary(experiment_dir, total_pods):
    """Log overall experiment summary when server shuts down."""
    if not experiment_dir:
        return
        
    log_file = None
    for file in os.listdir(experiment_dir):
        if '_server_' in file and file.endswith('.log'):
            log_file = os.path.join(experiment_dir, file)
            break
            
    if not log_file or not os.path.exists(log_file):
        return
        
    try:
        # Extract unique placed pod names from log
        with open(log_file, 'r') as f:
            log_content = f.read()
            
        placed_pod_matches = re.findall(r'PLACEMENT_SUCCESS for ([^-]*)-', log_content)
        unique_placed_pods = set(placed_pod_matches)
        placed_count = len(unique_placed_pods)
        
        overall_success_rate = (placed_count / total_pods) * 100 if total_pods > 0 else 0
        
        # Append summary to log file
        with open(log_file, 'a') as f:
            f.write("=" * 80 + "\n")
            f.write("🏁 OVERALL EXPERIMENT SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"  ▶ Total unique pods placed: {placed_count}\n")
            f.write(f"  ▶ Total pods in workload: {total_pods}\n")
            f.write(f"  ▶ OVERALL SUCCESS RATE: {placed_count}/{total_pods} pods ({overall_success_rate:.1f}%)\n")
            f.write("=" * 80 + "\n")
            
        logging.info("=" * 80)
        logging.info("🏁 OVERALL EXPERIMENT SUMMARY")
        logging.info("=" * 80)
        logging.info(f"  ▶ Total unique pods placed: {placed_count}")
        logging.info(f"  ▶ Total pods in workload: {total_pods}")
        logging.info(f"  ▶ OVERALL SUCCESS RATE: {placed_count}/{total_pods} pods ({overall_success_rate:.1f}%)")
        logging.info("=" * 80)
        
    except Exception as e:
        logging.warning(f"Could not generate experiment summary: {e}")

def signal_handler(signum, frame, experiment_dir=None, total_pods=None):
    """Handle shutdown signals and log experiment summary."""
    logging.info("⏳ Received shutdown signal, stopping server gracefully...")
    
    if experiment_dir and total_pods:
        log_experiment_summary(experiment_dir, total_pods)
    
    logging.info("👋 Server shutdown initiated")
    sys.exit(0)

def main() -> None:
    """
    Main entrypoint. Sets up logging based on command-line args,
    sets signal handler, and runs the server.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Carbon-aware scheduling server.')
    parser.add_argument(
        '--loglevel', 
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level (default: INFO)'
    )
    parser.add_argument(
        '--port',
        default='50051',
        help='Server port (default: 50051)'
    )
    parser.add_argument(
        '--algorithm',
        default='heuristic',
        choices=['heuristic', 'global-optimal', 'vanilla', 'caspian-operational', 'piontek-temporal', 'wait-awhile', 'green-mlfq'],
        help='Scheduling algorithm to use: heuristic (TotEm), global-optimal (MILP), vanilla (K8s-like, carbon-unaware), caspian-operational (spatio-temporal operational-carbon baseline), piontek-temporal (temporal CO2-window Kubernetes baseline), wait-awhile (non-interrupting temporal carbon-shifting baseline), or green-mlfq (GREEN-style carbon-aware MLFQ baseline)'
    )
    parser.add_argument(
        '--workloads-dir',
        default='../workloads',
        help='Directory containing timeslot_*.yaml workload files. Default is "../workloads" relative to server-python.'
    )
    parser.add_argument(
        '--nodes-file',
        default='../nodes.yaml',
        help='Path to nodes.yaml file (used by global-optimal algorithm)'
    )
    parser.add_argument(
        '--forecasts-file',
        default='./all_forecasts.json',
        help='Path to all_forecasts.json file (used by global-optimal algorithm)'
    )
    parser.add_argument(
        '--experiment',
        action='store_true',
        help='Enable experiment logging mode'
    )
    parser.add_argument(
        '--experiment-dir',
        default=os.path.join(os.path.dirname(__file__), 'experiments'),
        help='Directory to store experiment results (default: server-python/experiments)'
    )
    parser.add_argument(
        '--perf-log',
        action='store_true',
        default=True,  # Enable performance logging by default
        help='Enable performance logging to CSV files (enabled by default)'
    )
    parser.add_argument(
        '--perf-log-dir',
        default='./performance_logs',
        help='Directory to store performance logs'
    )
    parser.add_argument(
        '--prioritize-efficiency',
        action='store_true',
        help='Prioritize carbon efficiency per CPU rather than total emissions for scheduling decisions'
    )
    parser.add_argument(
        '--operational-only',
        action='store_true',
        help='Use only operational emissions (omit embodied emissions) when using heuristic algorithm'
    )
    parser.add_argument(
        '--embodied-mode',
        default='proportional',
        choices=['proportional','uniform'],
        help='Embodied allocation mode for heuristic/global-optimal precompute runs'
    )
    parser.add_argument(
        '--vanilla-score-mode',
        default='most_allocated',
        choices=['most_allocated', 'least_allocated'],
        help='Resource-only scoring mode for the vanilla Kubernetes baseline'
    )
    parser.add_argument(
        '--piontek-node-score-mode',
        default='most_allocated',
        choices=['most_allocated', 'least_allocated'],
        help='Resource-only node scoring mode used after Piontek-style temporal admission'
    )
    parser.add_argument(
        '--wait-awhile-node-score-mode',
        default='most_allocated',
        choices=['most_allocated', 'least_allocated'],
        help='Resource-only node scoring mode used after Wait-Awhile temporal admission'
    )
    parser.add_argument(
        '--precompute',
        action='store_true',
        help='Run precomputation mode: process all timeslot files sequentially and save placements to CSV without starting server'
    )
    
    args = parser.parse_args()
    
    # Configure logging with specified level
    log_level = getattr(logging, args.loglevel)
    
    # Create a custom logger to handle both console and file output
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler (always present)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Determine the main log directory for this server session
    session_log_dir = None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def algorithm_mode_name() -> str:
        if args.algorithm == 'vanilla':
            return f"vanilla_{args.vanilla_score_mode}"
        if args.operational_only:
            return f"{args.algorithm}_op"
        mode_suffix = args.embodied_mode if args.algorithm in ('heuristic', 'global-optimal') else None
        return f"{args.algorithm}_{mode_suffix}" if mode_suffix else args.algorithm

    # Set up session directory for logging
    if args.experiment:
        session_type_prefix = "experiment_session"
        algo_mode = algorithm_mode_name()
        session_log_dir = os.path.join(args.experiment_dir, f"{algo_mode}_{session_type_prefix}_{timestamp}")
        os.makedirs(session_log_dir, exist_ok=True)
        logging.info(f"🧪 Experiment mode: {session_log_dir}")
    elif args.perf_log:
        # Try to get pod count for more informative directory names
        pod_count = get_total_pod_count(args.workloads_dir)
        
        if pod_count is not None:
            session_type_prefix = f"{pod_count}pods"
        else:
            session_type_prefix = "perf_log_session"  # Fallback
            
        algo_mode = algorithm_mode_name()
        session_log_dir = os.path.join(args.experiment_dir, f"{algo_mode}_{session_type_prefix}_{timestamp}")
        os.makedirs(session_log_dir, exist_ok=True)
        
        # Set up signal handler with experiment parameters
        if pod_count is not None:
            signal.signal(signal.SIGINT, lambda signum, frame: signal_handler(signum, frame, session_log_dir, pod_count))
            signal.signal(signal.SIGTERM, lambda signum, frame: signal_handler(signum, frame, session_log_dir, pod_count))
            logging.info(f"📈 Performance logging ({pod_count} pods): {session_log_dir}")
        else:
            # Basic signal handlers without experiment summary
            signal.signal(signal.SIGINT, lambda signum, frame: signal_handler(signum, frame))
            signal.signal(signal.SIGTERM, lambda signum, frame: signal_handler(signum, frame))
            logging.info(f"📈 Performance logging: {session_log_dir}")
    else:
        # Basic signal handlers for non-performance mode
        signal.signal(signal.SIGINT, lambda signum, frame: signal_handler(signum, frame))
        signal.signal(signal.SIGTERM, lambda signum, frame: signal_handler(signum, frame))
        session_log_dir = None

    # Add file handler if we have a session directory
    if session_log_dir:
        log_filename = f"{args.algorithm}_server_{timestamp}.log"
        log_filepath = os.path.join(session_log_dir, log_filename)
        
        # Create file handler
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        
        # Add file handler to logger
        logger = logging.getLogger()
        logger.addHandler(file_handler)
        
        logging.info(f"📝 Server logs will be saved to: {log_filepath}")

    # Setup experiment logger if enabled
    experiment_logger = None
    if args.experiment and session_log_dir:
        try:
            from carbon_aware.experiments import ExperimentLogger
            experiment_logger = ExperimentLogger(output_dir=session_log_dir)
            logging.info(f"📊 Experiment logging initialized")
        except ImportError:
            logging.error("Failed to import ExperimentLogger. Running without experiment logging.")
            experiment_logger = None
        except Exception as e:
            logging.error(f"Error setting up experiment logger: {e}. Running without experiment logging.")
            experiment_logger = None
    
    # Setup performance logger if enabled
    perf_logger = None
    if args.perf_log:
        try:
            from carbon_aware.utils import PerformanceLogger
            perf_logger = PerformanceLogger(args.algorithm)

            if session_log_dir:
                perf_log_filename = f"{args.algorithm}_perf_session.csv" 
                perf_logger.set_new_log_file(log_dir=session_log_dir, filename=perf_log_filename)
                logging.info(f"📈 Performance logging configured")
            else:
                logging.warning("Performance logging enabled, but no session directory established. Performance logs may not be saved as expected.")
        except ImportError:
            logging.error("Failed to import PerformanceLogger. Running without performance logging.")
            perf_logger = None
        except Exception as e:
            logging.error(f"Error setting up performance logger: {e}. Running without performance logging.")
            perf_logger = None
    
    # Check if precompute mode is requested
    if args.precompute:
        if args.algorithm not in ['heuristic', 'global-optimal', 'vanilla', 'caspian-operational', 'piontek-temporal', 'wait-awhile', 'green-mlfq']:
            logging.error(f"Precompute mode is only supported for 'heuristic', 'global-optimal', 'vanilla', 'caspian-operational', 'piontek-temporal', 'wait-awhile', and 'green-mlfq' algorithms, got: {args.algorithm}")
            sys.exit(1)
        
        logging.info(f"🧮 Running precomputation mode for {args.algorithm} algorithm")
        logging.info(f"📂 Will process all timeslot files from: {args.workloads_dir}")
        
        # Run precomputation instead of starting server
        if args.algorithm == 'heuristic':
            from carbon_aware.precompute_heuristic import run_heuristic_precomputation
            success = run_heuristic_precomputation(
                workloads_dir=args.workloads_dir,
                nodes_file=args.nodes_file,
                forecasts_file=args.forecasts_file,
                session_log_dir=session_log_dir,
                perf_logger=perf_logger,
                prioritize_efficiency=args.prioritize_efficiency,
                operational_only=args.operational_only,
                embodied_mode=args.embodied_mode
            )
        elif args.algorithm == 'global-optimal':
            from carbon_aware.precompute_global_optimal import run_global_optimal_precomputation
            success = run_global_optimal_precomputation(
                workloads_dir=args.workloads_dir,
                nodes_file=args.nodes_file,
                forecasts_file=args.forecasts_file,
                session_log_dir=session_log_dir,
                perf_logger=perf_logger,
                prioritize_efficiency=args.prioritize_efficiency,
                operational_only=args.operational_only,
                embodied_mode=args.embodied_mode
            )
        elif args.algorithm == 'vanilla':
            from carbon_aware.precompute_vanilla import run_vanilla_precomputation
            success = run_vanilla_precomputation(
                workloads_dir=args.workloads_dir,
                nodes_file=args.nodes_file,
                forecasts_file=args.forecasts_file,
                session_log_dir=session_log_dir,
                perf_logger=perf_logger,
                prioritize_efficiency=args.prioritize_efficiency,
                operational_only=args.operational_only,
                score_mode=args.vanilla_score_mode
            )
        elif args.algorithm == 'caspian-operational':
            from carbon_aware.precompute_caspian_operational import run_caspian_operational_precomputation
            success = run_caspian_operational_precomputation(
                workloads_dir=args.workloads_dir,
                nodes_file=args.nodes_file,
                forecasts_file=args.forecasts_file,
                session_log_dir=session_log_dir,
                perf_logger=perf_logger,
                prioritize_efficiency=args.prioritize_efficiency,
                operational_only=True
            )
        elif args.algorithm == 'piontek-temporal':
            from carbon_aware.precompute_piontek_temporal import run_piontek_temporal_precomputation
            success = run_piontek_temporal_precomputation(
                workloads_dir=args.workloads_dir,
                nodes_file=args.nodes_file,
                forecasts_file=args.forecasts_file,
                session_log_dir=session_log_dir,
                perf_logger=perf_logger,
                prioritize_efficiency=args.prioritize_efficiency,
                operational_only=True,
                node_score_mode=args.piontek_node_score_mode
            )
        elif args.algorithm == 'wait-awhile':
            from carbon_aware.precompute_wait_awhile import run_wait_awhile_precomputation
            success = run_wait_awhile_precomputation(
                workloads_dir=args.workloads_dir,
                nodes_file=args.nodes_file,
                forecasts_file=args.forecasts_file,
                session_log_dir=session_log_dir,
                perf_logger=perf_logger,
                prioritize_efficiency=args.prioritize_efficiency,
                operational_only=True,
                node_score_mode=args.wait_awhile_node_score_mode
            )
        elif args.algorithm == 'green-mlfq':
            from carbon_aware.precompute_green_mlfq import run_green_mlfq_precomputation
            success = run_green_mlfq_precomputation(
                workloads_dir=args.workloads_dir,
                nodes_file=args.nodes_file,
                forecasts_file=args.forecasts_file,
                session_log_dir=session_log_dir,
                perf_logger=perf_logger,
                prioritize_efficiency=args.prioritize_efficiency,
                operational_only=True
            )
        
        if success:
            logging.info("✅ Precomputation completed successfully")
            sys.exit(0)
        else:
            logging.error("❌ Precomputation failed")
            sys.exit(1)
    
    # Basic server start logging (moved after log dir setup for clarity)
    logging.info(f"Starting carbon-aware server with algorithm: {args.algorithm}")
    
    # Signal handlers are set up in the perf_log section above when needed

    # Start the server
    # Pass the session_log_dir to the server, so it can pass it to algorithms
    # for consistent logging paths for placements.
    serve(
        port=args.port, 
        algorithm=args.algorithm, 
        experiment_logger=experiment_logger, 
        perf_logger=perf_logger, 
        session_log_dir=session_log_dir,
        workloads_dir=args.workloads_dir,
        nodes_file=args.nodes_file,
        forecasts_file=args.forecasts_file,
        prioritize_efficiency=args.prioritize_efficiency,
        operational_only=args.operational_only
    )


if __name__ == "__main__":
    main()
