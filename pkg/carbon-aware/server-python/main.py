"""
Main entry point for the Carbon-Aware Orchestrator.
"""
import argparse
import logging
import signal
import os
from datetime import datetime

from carbon_aware.server import serve
from carbon_aware.experiments import ExperimentLogger

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
        choices=['heuristic', 'optimal', 'global-optimal'],
        help='Scheduling algorithm to use: heuristic (fast, local optimization), optimal (MILP, per-pod optimization), or global-optimal (MILP, considers all pods simultaneously) (default: heuristic)'
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
        default='./experiments',
        help='Directory to store experiment results'
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

    # Set up session directory for logging
    if args.experiment:
        session_type_prefix = "experiment_session"
        session_log_dir = os.path.join(args.experiment_dir, f"{args.algorithm}_{session_type_prefix}_{timestamp}")
        os.makedirs(session_log_dir, exist_ok=True)
        logging.info(f"🧪 Experiment mode: {session_log_dir}")
    elif args.perf_log:
        session_type_prefix = "perf_log_session" 
        session_log_dir = os.path.join(args.experiment_dir, f"{args.algorithm}_{session_type_prefix}_{timestamp}")
        os.makedirs(session_log_dir, exist_ok=True)
        logging.info(f"📈 Performance logging: {session_log_dir}")
    else:
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
    
    # Basic server start logging (moved after log dir setup for clarity)
    logging.info(f"Starting carbon-aware server with algorithm: {args.algorithm}")
    
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
        prioritize_efficiency=args.prioritize_efficiency
    )


if __name__ == "__main__":
    main()