#!/usr/bin/env python3
"""
Carbon-aware scheduling algorithm comparison experiments using real workloads.

This script runs both algorithms on the same workloads and compares their performance.
"""
import argparse
import logging
import sys
import os
import time
import json
import yaml
import glob
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from carbon_aware.models import CarbonAwarePod, CarbonAwareFlavour, CarbonAwareTimeslot
from carbon_aware.experiments import ExperimentLogger
from carbon_aware.algorithms.heuristic import HeuristicAlgorithm
from carbon_aware.algorithms.optimal import OptimalAlgorithm
from carbon_aware.utils import is_timeslot_valid


def load_nodes_from_yaml(yaml_file: str) -> List[CarbonAwareFlavour]:
    """Load node flavours from a Kubernetes-formatted YAML file"""
    try:
        flavours = []
        with open(yaml_file, 'r') as f:
            all_docs = list(yaml.safe_load_all(f))
            
        # Process each document (each is a Kubernetes node)
        for doc in all_docs:
            if not doc or doc.get('kind') != 'Node':
                continue
                
            # Extract node name and metadata
            node_name = doc.get('metadata', {}).get('name', '')
            annotations = doc.get('metadata', {}).get('annotations', {})
            labels = doc.get('metadata', {}).get('labels', {})
            status = doc.get('status', {})
            
            # Extract region from labels
            region = labels.get('topology.kubernetes.io/region', '')
            
            # Extract CPU and memory from allocatable
            allocatable = status.get('allocatable', {})
            cpu_str = allocatable.get('cpu', '0')
            memory_str = allocatable.get('memory', '0')
            
            # Convert CPU to float
            try:
                total_cpu = float(cpu_str)
            except (ValueError, TypeError):
                total_cpu = 0
                
            # Convert memory to MB
            try:
                # Handle Gi format
                if isinstance(memory_str, str) and memory_str.endswith('Gi'):
                    total_ram = float(memory_str.rstrip('Gi')) * 1024  # Convert Gi to Mi
                else:
                    total_ram = float(memory_str)
            except (ValueError, TypeError):
                total_ram = 0
                
            # Extract power draw from annotations
            try:
                power_idle = float(annotations.get('hardware.power/idle_watts', 0))
                power_max = float(annotations.get('hardware.power/max_watts', 0))
            except (ValueError, TypeError):
                power_idle = 0
                power_max = 0
                
            # Create flavour
            flavours.append(CarbonAwareFlavour(
                id=node_name,
                embodiedCarbon=float(annotations.get('hardware.carbon/embodied_emissions', 0)),
                lifetime=float(annotations.get('hardware.carbon/lifetime_years', 5)),
                totalCpu=total_cpu,
                totalRam=total_ram,
                totalStorage=0,                # Required parameter
                forecast={},                   # Required parameter - empty dict for now
                power={
                    "idle": power_idle, 
                    "active": float(annotations.get('hardware.power/active_watts', (power_idle + power_max)/2)),
                    "max": power_max
                }
            ))
            
        logging.info(f"Loaded {len(flavours)} node flavours from {yaml_file}")
        return flavours
    except Exception as e:
        logging.error(f"Error loading nodes from {yaml_file}: {e}")
        return []

def load_workloads_from_directory(directory: str) -> Dict[str, List[CarbonAwarePod]]:
    """Load workloads from JSON files in a directory"""
    workloads = {}
    
    try:
        # Find all JSON files in the directory
        json_files = glob.glob(os.path.join(directory, "*.json"))
        
        for file_path in json_files:
            filename = os.path.basename(file_path)
            workload_name = os.path.splitext(filename)[0]
            
            with open(file_path, 'r') as f:
                pod_data = json.load(f)
            
            pods = []
            for pod in pod_data:
                # Convert deadline string to datetime if it exists
                deadline = None
                if 'deadline' in pod:
                    try:
                        deadline = datetime.fromisoformat(pod['deadline'])
                    except ValueError:
                        # If it's not an ISO format date, assume it's hours from now
                        try:
                            hours = float(pod['deadline'])
                            deadline = datetime.now() + timedelta(hours=hours)
                        except (ValueError, TypeError):
                            deadline = datetime.now() + timedelta(hours=24)  # Default deadline
                
                pods.append(CarbonAwarePod(
                    id=pod.get('id', ''),
                    name=pod.get('name', ''),
                    duration=pod.get('duration', 1.0),
                    deadline=deadline,
                    cpuRequest=pod.get('cpuRequest', 1.0),
                    ramRequest=pod.get('ramRequest', 1024),
                    gpuRequest=pod.get('gpuRequest', 0),
                ))
            
            workloads[workload_name] = pods
            logging.info(f"Loaded workload '{workload_name}' with {len(pods)} pods from {file_path}")
        
        return workloads
    except Exception as e:
        logging.error(f"Error loading workloads from {directory}: {e}")
        return {}


def generate_test_timeslots(num_slots=48):
    """Generate synthetic timeslots with carbon intensity for testing"""
    timeslots = []
    start_time = datetime.now()
    
    for i in range(num_slots):
        slot_time = start_time + timedelta(hours=i)
        
        # Generate carbon intensity with day/night cycle pattern
        hour = slot_time.hour
        # Lower intensity at night (when solar might be available)
        base_intensity = 600 - 200 * (0 <= hour <= 6 or 20 <= hour <= 23)
        
        timeslots.append(CarbonAwareTimeslot(
            id=i,
            timestamp=slot_time,
            durationMinutes=60,
            carbonIntensity=base_intensity
        ))
    return timeslots


def prepare_resources(flavours, num_slots=48):
    """Initialize resource tracking for CPU and RAM"""
    leftover_cpu = {}
    leftover_ram = {}
    
    for flv in flavours:
        leftover_cpu[flv.id] = {}
        leftover_ram[flv.id] = {}
        for ts_id in range(num_slots):
            leftover_cpu[flv.id][ts_id] = flv.totalCpu
            leftover_ram[flv.id][ts_id] = flv.totalRam
            
    return leftover_cpu, leftover_ram


def run_experiments(
    nodes_file: str = "/root/carbon-aware-orchestrator/pkg/carbon-aware/nodes.yaml",
    workloads_dir: str = "/root/carbon-aware-orchestrator/pkg/carbon-aware/workloads",
    algorithms: List[str] = ["heuristic", "optimal"],
    output_dir: str = "./experiment_results"
):
    """
    Run comparison experiments with real workloads and node configurations.
    
    Args:
        nodes_file: YAML file containing node configurations
        workloads_dir: Directory containing JSON workload files
        algorithms: Which algorithms to compare
        output_dir: Directory to store results
    """
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    experiment_results = {
        "metadata": {
            "timestamp": timestamp,
            "nodes_file": nodes_file,
            "workloads_dir": workloads_dir,
            "algorithms": algorithms,
        },
        "results": []
    }
    
    # Load nodes and workloads
    flavours = load_nodes_from_yaml(nodes_file)
    if not flavours:
        logging.error("No node flavours loaded, aborting experiment")
        return None
    
    workloads = load_workloads_from_directory(workloads_dir)
    if not workloads:
        logging.error("No workloads loaded, aborting experiment")
        return None
    
    # Generate timeslots
    timeslots = generate_test_timeslots(num_slots=48)
    
    # Print experiment setup
    logging.info("=" * 80)
    logging.info(f"Starting experiments with:")
    logging.info(f"- Algorithms: {', '.join(algorithms)}")
    logging.info(f"- Node flavours: {len(flavours)}")
    logging.info(f"- Workloads: {list(workloads.keys())}")
    logging.info(f"- Timeslots: {len(timeslots)}")
    logging.info("=" * 80)
    
    for workload_name, pods in workloads.items():
        logging.info(f"=== Running experiments with workload '{workload_name}' ({len(pods)} pods) ===")
        
        for algo_name in algorithms:
            # Create algorithm instance
            if algo_name == "heuristic":
                algorithm = HeuristicAlgorithm()
            elif algo_name == "optimal":
                algorithm = OptimalAlgorithm()
            else:
                logging.error(f"Unknown algorithm: {algo_name}")
                continue
            
            # Create dedicated output directory
            algo_dir = os.path.join(run_dir, f"{algo_name}_{workload_name}")
            os.makedirs(algo_dir, exist_ok=True)
            
            # Create experiment logger
            logger = ExperimentLogger(output_dir=algo_dir)
            algorithm.experiment_logger = logger
            
            # Start session
            logger.start_session(algorithm.name)
            
            # Initialize fresh resources for this run
            leftover_cpu, leftover_ram = prepare_resources(flavours, len(timeslots))
            
            # Process each pod in the workload
            logging.info(f"Processing {len(pods)} pods with {algo_name} algorithm...")
            start_time = time.time()
            
            for pod in pods:
                # Find placement using the algorithm
                best_node, best_slot, emissions = algorithm.find_placement(
                    pod, flavours, timeslots, leftover_cpu, leftover_ram
                )
                
                # If placement found, update resources
                if best_node and best_slot:
                    # Update resources
                    for slot_offset in range(int(pod.duration)):
                        current_slot = best_slot.id + slot_offset
                        if current_slot >= len(timeslots):
                            continue
                        
                        leftover_cpu[best_node.id][current_slot] -= pod.cpuRequest
                        leftover_ram[best_node.id][current_slot] -= pod.ramRequest
            
            total_time = time.time() - start_time
            logging.info(f"Completed {algo_name} in {total_time:.2f}s")
            
            # End session and record results
            logger.end_session()
            logger.save_all_results()
            
            # Save summary to overall results
            if hasattr(logger, 'session_metrics'):
                experiment_results["results"].append({
                    "algorithm": algo_name,
                    "workload": workload_name,
                    "workload_size": len(pods),
                    "summary": logger.session_metrics
                })
    
    # Save overall experiment results
    with open(os.path.join(run_dir, "experiment_summary.json"), 'w') as f:
        json.dump(experiment_results, f, indent=2)
    
    # Generate comparison charts
    generate_comparison_charts(run_dir, experiment_results)
    
    logging.info(f"✅ All experiments completed and saved to {run_dir}")
    return run_dir


def generate_comparison_charts(run_dir, experiment_results):
    """Generate comparison charts from experiment results"""
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Convert results to DataFrame
        rows = []
        for result in experiment_results["results"]:
            summary = result["summary"]
            rows.append({
                "algorithm": result["algorithm"],
                "workload": result["workload"],
                "workload_size": result["workload_size"],
                "total_pods": summary.get("total_pods", 0),
                "successful_placements": summary.get("successful_placements", 0),
                "failed_placements": summary.get("failed_placements", 0),
                "success_rate": summary.get("successful_placements", 0) / max(1, summary.get("total_pods", 0)) * 100,
                "avg_execution_time": summary.get("avg_execution_time", 0),
                "total_carbon": summary.get("total_carbon", 0),
                "peak_memory_mb": summary.get("peak_memory_mb", 0),
                "duration": summary.get("end_time", 0) - summary.get("start_time", 0)
            })
        
        df = pd.DataFrame(rows)
        
        # Create charts directory
        charts_dir = os.path.join(run_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        # Plot comparison charts for each workload
        for workload in df["workload"].unique():
            workload_df = df[df["workload"] == workload]
            
            # Create a figure with 2x2 subplots
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle(f"Algorithm Comparison for Workload: {workload}", fontsize=16)
            
            # 1. Execution time comparison
            workload_df.plot(x='algorithm', y='avg_execution_time', kind='bar', ax=axes[0, 0],
                            title='Average Execution Time per Pod',
                            ylabel='Time (seconds)')
            axes[0, 0].set_xlabel('Algorithm')
            
            # 2. Success rate comparison
            workload_df.plot(x='algorithm', y='success_rate', kind='bar', ax=axes[0, 1],
                            title='Placement Success Rate',
                            ylabel='Success Rate (%)')
            axes[0, 1].set_xlabel('Algorithm')
            
            # 3. Carbon emissions comparison
            workload_df.plot(x='algorithm', y='total_carbon', kind='bar', ax=axes[1, 0],
                            title='Total Carbon Emissions',
                            ylabel='Carbon (kgCO2e)')
            axes[1, 0].set_xlabel('Algorithm')
            
            # 4. Memory usage comparison
            workload_df.plot(x='algorithm', y='peak_memory_mb', kind='bar', ax=axes[1, 1],
                            title='Peak Memory Usage',
                            ylabel='Memory (MB)')
            axes[1, 1].set_xlabel('Algorithm')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(charts_dir, f"comparison_{workload}.png"))
        
        # Create overall comparison chart
        if len(df["workload"].unique()) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f"Overall Algorithm Performance Across All Workloads", fontsize=16)
            
            # Group by algorithm and calculate means
            algo_summary = df.groupby("algorithm").agg({
                'success_rate': 'mean',
                'avg_execution_time': 'mean',
                'total_carbon': 'sum',
                'peak_memory_mb': 'mean'
            }).reset_index()
            
            # 1. Average execution time
            algo_summary.plot(x='algorithm', y='avg_execution_time', kind='bar', ax=axes[0, 0],
                            title='Average Execution Time per Pod',
                            ylabel='Time (seconds)')
            axes[0, 0].set_xlabel('Algorithm')
            
            # 2. Average success rate
            algo_summary.plot(x='algorithm', y='success_rate', kind='bar', ax=axes[0, 1],
                            title='Average Placement Success Rate',
                            ylabel='Success Rate (%)')
            axes[0, 1].set_xlabel('Algorithm')
            
            # 3. Total carbon emissions
            algo_summary.plot(x='algorithm', y='total_carbon', kind='bar', ax=axes[1, 0],
                            title='Total Carbon Emissions (All Workloads)',
                            ylabel='Carbon (kgCO2e)')
            axes[1, 0].set_xlabel('Algorithm')
            
            # 4. Average memory usage
            algo_summary.plot(x='algorithm', y='peak_memory_mb', kind='bar', ax=axes[1, 1],
                            title='Average Peak Memory Usage',
                            ylabel='Memory (MB)')
            axes[1, 1].set_xlabel('Algorithm')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(charts_dir, "overall_comparison.png"))
        
        # Save data for further analysis
        df.to_csv(os.path.join(run_dir, "all_experiment_data.csv"), index=False)
        
        logging.info(f"📊 Comparison charts generated in {charts_dir}")
        
    except ImportError as e:
        logging.error(f"Could not generate charts: {e}")
    except Exception as e:
        logging.error(f"Error generating comparison charts: {e}")
        logging.exception(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run algorithm comparison experiments with real workloads.')
    parser.add_argument('--nodes', default='/root/carbon-aware-orchestrator/pkg/carbon-aware/nodes.yaml',
                        help='YAML file containing node configurations')
    parser.add_argument('--workloads', default='/root/carbon-aware-orchestrator/pkg/carbon-aware/workloads',
                        help='Directory containing JSON workload files')
    parser.add_argument('--algorithms', choices=['heuristic', 'optimal', 'both'], 
                        default='both', help='Which algorithms to test')
    parser.add_argument('--output-dir', default='./experiment_results',
                        help='Directory to store experiment results')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING'], 
                        default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(level=log_level, 
                       format='%(asctime)s [%(levelname)s] %(message)s',
                       datefmt='%Y-%m-%d %H:%M:%S')
    
    # Determine which algorithms to run
    algos = []
    if args.algorithms == 'both' or args.algorithms == 'heuristic':
        algos.append('heuristic')
    if args.algorithms == 'both' or args.algorithms == 'optimal':
        algos.append('optimal')
        
    # Run experiments
    run_dir = run_experiments(
        nodes_file=args.nodes,
        workloads_dir=args.workloads,
        algorithms=algos,
        output_dir=args.output_dir
    )
    
    logging.info(f"Experiment complete! Results available at: {run_dir}")