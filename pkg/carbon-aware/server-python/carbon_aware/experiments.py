"""
Experiment logging utilities for the Carbon-Aware Orchestrator.

This module contains utilities for logging experiment results, including
placement decisions, carbon impact, and performance metrics.
"""
import os
import csv
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ExperimentLogger:
    """
    Logger for carbon-aware scheduling experiments.
    
    This class handles logging of experiment results, including placement decisions,
    carbon impact metrics, and performance data. Results are saved to CSV files
    in an experiment-specific directory.
    """
    
    def __init__(self, algorithm_name: str, base_dir: str = "experiments"):
        """
        Initialize the experiment logger.
        
        Args:
            algorithm_name: Name of the algorithm being used (e.g., 'heuristic', 'global-optimal')
            base_dir: Base directory where experiment logs will be stored
        """
        self.algorithm_name = algorithm_name
        
        # Create experiment directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(
            base_dir, 
            f"{algorithm_name}_perf_log_session_{timestamp}"
        )
        os.makedirs(self.session_dir, exist_ok=True)
        
        logger.info(f"Experiment logs will be saved to: {self.session_dir}")
        
        # Initialize performance tracking
        self.start_time = time.time()
        self.total_time = 0.0
        self.per_pod_times = []
        
        # CSV file paths
        self.placements_csv_path = os.path.join(
            self.session_dir, 
            f"{algorithm_name}_placements_session.csv"
        )
        self.perf_csv_path = os.path.join(
            self.session_dir,
            f"{algorithm_name}_perf_session.csv"
        )
        
        # Initialize CSV files with headers
        self._init_csv_files()
    
    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        # Placements CSV
        with open(self.placements_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'pod_id', 'node_id', 'start_slot', 'duration', 'cpu_request',
                'memory_request', 'earliest_timeslot', 'deadline',
                'carbon_footprint', 'placement_time'
            ])
        
        # Performance CSV
        with open(self.perf_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'total_pods', 'scheduled_pods', 'total_time', 'avg_time_per_pod',
                'total_carbon', 'avg_carbon_per_pod'
            ])
    
    def log_placement(self, pod_data: Dict[str, Any], node_id: str, 
                     start_slot: int, carbon_footprint: float,
                     placement_time: float):
        """
        Log a placement decision.
        
        Args:
            pod_data: Dictionary with pod information
            node_id: ID of the node where pod was placed
            start_slot: Starting timeslot for the pod
            carbon_footprint: Estimated carbon footprint of this placement
            placement_time: Time taken to make this placement decision
        """
        with open(self.placements_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                pod_data.get('id', 'unknown'),
                node_id,
                start_slot,
                pod_data.get('duration', 0),
                pod_data.get('cpu_request', 0),
                pod_data.get('memory_request', 0),
                pod_data.get('earliest_timeslot', 0),
                pod_data.get('deadline', 0),
                carbon_footprint,
                placement_time
            ])
        
        # Track performance metrics
        self.per_pod_times.append(placement_time)
    
    def log_unscheduled_pod(self, pod_data: Dict[str, Any], reason: str):
        """
        Log information about a pod that couldn't be scheduled.
        
        Args:
            pod_data: Dictionary with pod information
            reason: Reason why the pod couldn't be scheduled
        """
        logger.warning(
            f"Pod {pod_data.get('id', 'unknown')} could not be scheduled: {reason}"
        )
    
    def finalize(self, total_pods: int, scheduled_pods: int, 
                total_carbon: float):
        """
        Finalize the experiment and write summary data.
        
        Args:
            total_pods: Total number of pods in the workload
            scheduled_pods: Number of pods successfully scheduled
            total_carbon: Total carbon footprint of all placements
        """
        self.total_time = time.time() - self.start_time
        avg_time = self.total_time / total_pods if total_pods > 0 else 0
        avg_carbon = total_carbon / scheduled_pods if scheduled_pods > 0 else 0
        
        # Write performance summary
        with open(self.perf_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                total_pods,
                scheduled_pods,
                self.total_time,
                avg_time,
                total_carbon,
                avg_carbon
            ])
        
        logger.info(f"Experiment completed in {self.total_time:.4f} seconds")
        logger.info(f"Scheduled {scheduled_pods}/{total_pods} pods")
        logger.info(f"Total carbon footprint: {total_carbon:.2f} g CO2e")
        
        return {
            'total_time': self.total_time,
            'scheduled_pods': scheduled_pods,
            'total_pods': total_pods,
            'total_carbon': total_carbon,
            'session_dir': self.session_dir
        }
