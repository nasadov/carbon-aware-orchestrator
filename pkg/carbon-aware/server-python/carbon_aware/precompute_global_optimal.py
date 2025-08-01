#!/usr/bin/env python3
"""
Precomputation module for global-optimal algorithm
"""

import logging
from carbon_aware.algorithms.global_optimal import GlobalOptimalAlgorithm


def run_global_optimal_precomputation(
    workloads_dir: str,
    nodes_file: str,
    forecasts_file: str,
    session_log_dir: str = None,
    perf_logger=None,
    prioritize_efficiency: bool = False,
    operational_only: bool = False
) -> bool:
    """
    Run global-optimal algorithm in precomputation mode
    
    Args:
        workloads_dir: Directory containing timeslot_*.yaml files
        nodes_file: Path to nodes.yaml file
        forecasts_file: Path to all_forecasts.json file
        session_log_dir: Directory for session logs and CSV output
        perf_logger: Performance logger instance
        prioritize_efficiency: Whether to prioritize efficiency (unused for global-optimal)
        operational_only: Whether to use operational-only emissions
        
    Returns:
        True if precomputation succeeded, False otherwise
    """
    
    try:
        logging.info("🔧 STEP 1: Initializing global-optimal algorithm for precomputation")
        
        # Initialize global-optimal algorithm
        algorithm = GlobalOptimalAlgorithm()
        
        # Configure session logging if directory provided
        if session_log_dir:
            logging.info(f"📁 Setting log directory: {session_log_dir}")
            algorithm.set_base_log_dir(session_log_dir)
            algorithm.setup_session_placement_log()
            logging.info("📝 CSV placement logging initialized")
        
        # Set operational-only mode if requested
        if hasattr(algorithm, 'set_operational_only') and operational_only:
            algorithm.set_operational_only(operational_only)
            logging.info(f"⚙️ Operational-only mode: {operational_only}")
        
        logging.info("🔧 STEP 2: Starting comprehensive global optimization")
        logging.info(f"📂 Workloads directory: {workloads_dir}")
        logging.info(f"📋 Nodes file: {nodes_file}")
        logging.info(f"📊 Forecasts file: {forecasts_file}")
        
        # Run the precomputation
        success = algorithm.precompute_all_workloads(
            workloads_dir=workloads_dir,
            nodes_file=nodes_file,
            forecasts_file=forecasts_file
        )
        
        if success:
            logging.info("✅ Global-optimal precomputation completed successfully")
            return True
        else:
            logging.error("❌ Global-optimal precomputation failed")
            return False
            
    except Exception as e:
        logging.error(f"❌ Fatal error in global-optimal precomputation: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False