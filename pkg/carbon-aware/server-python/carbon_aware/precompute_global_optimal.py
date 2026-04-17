#!/usr/bin/env python3
"""
Precomputation module for global-optimal algorithm
"""

import logging
import os
from carbon_aware.algorithms.global_optimal import GlobalOptimalAlgorithm


def run_global_optimal_precomputation(
    workloads_dir: str,
    nodes_file: str,
    forecasts_file: str,
    session_log_dir: str = None,
    perf_logger=None,
    prioritize_efficiency: bool = False,
    operational_only: bool = False,
    embodied_mode: str = "proportional",
    global_water_budget: float = None,
    global_water_metric: str = "scarcity",
    global_objective: str = "carbon",
    global_scalarized_carbon_weight: float = 0.5,
    global_scalarized_water_metric: str = "scarcity",
    global_scalarized_ref_weight: float = 0.1,
    global_scalarized_history_window: int = 10,
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
        
        # Configure session logging if directory provided (use as-is; main.py names with mode)
        if session_log_dir:
            logging.info(f"📁 Setting log directory: {session_log_dir}")
            algorithm.set_base_log_dir(session_log_dir)
            algorithm.setup_session_placement_log()
            logging.info("📝 CSV placement logging initialized")
        
        # Set operational-only mode if requested
        if hasattr(algorithm, 'set_operational_only') and operational_only:
            algorithm.set_operational_only(operational_only)
            logging.info(f"⚙️ Operational-only mode: {operational_only}")
        if hasattr(algorithm, 'set_embodied_allocation_mode'):
            algorithm.set_embodied_allocation_mode(embodied_mode)
        if hasattr(algorithm, 'set_epsilon_constraint'):
            algorithm.set_epsilon_constraint(
                water_budget=global_water_budget,
                water_metric=global_water_metric,
            )
        if hasattr(algorithm, 'set_phase2_objective'):
            algorithm.set_phase2_objective(
                mode=global_objective,
                carbon_weight=global_scalarized_carbon_weight,
                water_metric=global_scalarized_water_metric,
                reference_weight=global_scalarized_ref_weight,
                history_window=global_scalarized_history_window,
            )

        logging.info("🔧 STEP 2: Starting comprehensive global optimization")
        logging.info(f"📂 Workloads directory: {workloads_dir}")
        logging.info(f"📋 Nodes file: {nodes_file}")
        logging.info(f"📊 Forecasts file: {forecasts_file}")
        if global_water_budget is not None:
            logging.info(
                "💧 Epsilon-constraint enabled: %s water <= %.6f",
                global_water_metric,
                float(global_water_budget),
            )
        if global_objective != "carbon":
            logging.info(
                "🌊 Global objective: %s lambda_C=%.2f water_metric=%s lambda_ref=%.3f history_window=%s",
                global_objective,
                float(global_scalarized_carbon_weight),
                global_scalarized_water_metric,
                float(global_scalarized_ref_weight),
                int(global_scalarized_history_window),
            )
        
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
