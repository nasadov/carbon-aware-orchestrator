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
        '--experiment',
        action='store_true',
        help='Enable experiment logging mode'
    )
    parser.add_argument(
        '--experiment-dir',
        default='./experiments',
        help='Directory to store experiment results'
    )
    
    args = parser.parse_args()
    
    # Configure logging with specified level
    log_level = getattr(logging, args.loglevel)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Basic server start logging
    logging.info(f"Starting carbon-aware server with algorithm: {args.algorithm}")
    logging.info(f"Log level: {args.loglevel}, Port: {args.port}")
    
    # Setup experiment logger if enabled
    experiment_logger = None
    if args.experiment:
        try:
            from carbon_aware.experiments import ExperimentLogger
            
            # Create timestamp-based subdirectory for this run
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_dir = os.path.join(args.experiment_dir, f"{args.algorithm}_{timestamp}")
            os.makedirs(result_dir, exist_ok=True)
            
            experiment_logger = ExperimentLogger(output_dir=result_dir)
            logging.info(f"📊 Experiment mode enabled - metrics will be collected in {result_dir}")
        except ImportError:
            logging.error("Failed to import ExperimentLogger. Running without experiment logging.")
        except Exception as e:
            logging.error(f"Error setting up experiment logger: {e}. Running without experiment logging.")
    
    # Start the server
    serve(port=args.port, algorithm=args.algorithm, experiment_logger=experiment_logger)


if __name__ == "__main__":
    main()