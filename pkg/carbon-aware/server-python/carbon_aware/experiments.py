"""
Experiment logging and analysis for comparing scheduling algorithms.
"""
import json
import logging
import os
import time
import datetime
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any


class ExperimentLogger:
    """
    Records metrics for algorithm comparison experiments.
    """
    def __init__(self, output_dir="./experiments"):
        self.output_dir = output_dir
        self.experiment_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = []
        self.session_metrics = {
            "algorithm": None,
            "start_time": None,
            "end_time": None,
            "total_pods": 0,
            "successful_placements": 0,
            "failed_placements": 0,
            "total_carbon": 0,
            "execution_times": [],
            "peak_memory_mb": 0,
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def start_session(self, algorithm_name: str):
        """Record the start of an algorithm session"""
        self.session_metrics["algorithm"] = algorithm_name
        self.session_metrics["start_time"] = time.time()
        # Record initial memory usage
        self.session_metrics["initial_memory_mb"] = psutil.Process().memory_info().rss / (1024 * 1024)
        logging.info(f"📊 Starting experiment session with algorithm: {algorithm_name}")
        
    def end_session(self):
        """Record the end of an algorithm session and save results"""
        self.session_metrics["end_time"] = time.time()
        duration = self.session_metrics["end_time"] - self.session_metrics["start_time"]
        
        # Calculate average execution time
        if self.session_metrics["execution_times"]:
            self.session_metrics["avg_execution_time"] = sum(self.session_metrics["execution_times"]) / len(self.session_metrics["execution_times"])
        else:
            self.session_metrics["avg_execution_time"] = 0
            
        # Get final memory usage and calculate peak
        current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        self.session_metrics["final_memory_mb"] = current_memory
        if current_memory > self.session_metrics["peak_memory_mb"]:
            self.session_metrics["peak_memory_mb"] = current_memory
            
        # Save session metrics to file
        filename = f"{self.output_dir}/experiment_{self.experiment_id}_{self.session_metrics['algorithm']}.json"
        with open(filename, 'w') as f:
            json.dump(self.session_metrics, f, indent=2)
            
        # Log summary
        logging.info(f"📊 Experiment session completed:")
        logging.info(f"   Algorithm: {self.session_metrics['algorithm']}")
        logging.info(f"   Duration: {duration:.2f}s")
        logging.info(f"   Pods placed: {self.session_metrics['successful_placements']}/{self.session_metrics['total_pods']}")
        logging.info(f"   Avg execution time per pod: {self.session_metrics['avg_execution_time']:.3f}s")
        logging.info(f"   Total carbon: {self.session_metrics['total_carbon']:.2f} kgCO2e")
        logging.info(f"   Peak memory: {self.session_metrics['peak_memory_mb']:.1f} MB")
        logging.info(f"   Results saved to: {filename}")
        
    def record_placement(self, pod_id: str, success: bool, execution_time: float, 
                        emissions: float, considered_options: int, selected_node: Optional[str] = None,
                        selected_timeslot: Optional[int] = None, solver_iterations: Optional[int] = None, 
                        solver_status: Optional[str] = None):
        """Record metrics for a single placement decision"""
        # Update session counters
        self.session_metrics["total_pods"] += 1
        if success:
            self.session_metrics["successful_placements"] += 1
            self.session_metrics["total_carbon"] += emissions
        else:
            self.session_metrics["failed_placements"] += 1
            
        self.session_metrics["execution_times"].append(execution_time)
        
        # Record memory usage
        current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        if current_memory > self.session_metrics["peak_memory_mb"]:
            self.session_metrics["peak_memory_mb"] = current_memory
            
        # Record individual placement result
        result = {
            "pod_id": pod_id,
            "algorithm": self.session_metrics["algorithm"],
            "timestamp": time.time(),
            "success": success,
            "execution_time": execution_time,
            "emissions": emissions if success else None,
            "node": selected_node,
            "timeslot": selected_timeslot,
            "considered_options": considered_options,
        }
        
        # Add solver-specific metrics for optimal algorithm
        if solver_iterations is not None:
            result["solver_iterations"] = solver_iterations
            result["solver_status"] = solver_status
            
        self.results.append(result)
        
    def save_all_results(self):
        """Save detailed results to CSV"""
        if not self.results:
            return
            
        df = pd.DataFrame(self.results)
        filename = f"{self.output_dir}/detailed_results_{self.experiment_id}.csv"
        df.to_csv(filename, index=False)
        logging.info(f"📊 Detailed results saved to {filename}")
        
    def generate_report(self):
        """Generate comparison charts and report"""
        if not self.results:
            logging.warning("No results to generate report from")
            return
            
        df = pd.DataFrame(self.results)
        
        # Create comparison plots if we have data from both algorithms
        algorithms = df['algorithm'].unique()
        if len(algorithms) > 1:
            # Setup plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 1. Execution time comparison
            time_df = df.groupby('algorithm')['execution_time'].agg(['mean', 'std', 'max'])
            time_df.plot(kind='bar', y='mean', yerr='std', ax=axes[0, 0], 
                        title='Average Execution Time per Pod', 
                        ylabel='Time (seconds)')
            axes[0, 0].set_xlabel('Algorithm')
            
            # 2. Success rate comparison
            success_df = df.groupby(['algorithm', 'success']).size().unstack(fill_value=0)
            success_rate = success_df[True] / (success_df[True] + success_df[False]) * 100
            success_rate.plot(kind='bar', ax=axes[0, 1], 
                            title='Placement Success Rate', 
                            ylabel='Success Rate (%)')
            axes[0, 1].set_xlabel('Algorithm')
            
            # 3. Carbon emissions comparison (successful placements only)
            successful = df[df['success'] == True]
            if not successful.empty:
                emissions_df = successful.groupby('algorithm')['emissions'].agg(['mean', 'sum', 'count'])
                emissions_df.plot(kind='bar', y='mean', ax=axes[1, 0], 
                                title='Average Carbon Emissions per Pod', 
                                ylabel='kgCO2e')
                axes[1, 0].set_xlabel('Algorithm')
                
                emissions_df.plot(kind='bar', y='sum', ax=axes[1, 1], 
                                title='Total Carbon Emissions', 
                                ylabel='kgCO2e')
                axes[1, 1].set_xlabel('Algorithm')
            
            plt.tight_layout()
            report_path = f"{self.output_dir}/comparison_report_{self.experiment_id}.png"
            plt.savefig(report_path)
            logging.info(f"📊 Comparison report generated: {report_path}")