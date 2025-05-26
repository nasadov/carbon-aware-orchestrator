#!/usr/bin/env python3
"""
Test script for comprehensive global optimization with carbon efficiency per CPU prioritization.

This script runs the server with the comprehensive global optimization and tests
whether the French IoT node (node-4-fr-iot) is utilized despite having fewer CPUs.

Usage:
    python test_comprehensive_scheduling.py
"""
import os
import subprocess
import time
import logging
import argparse
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def analyze_placement_csv(csv_path):
    """Analyze the placement CSV to check node utilization"""
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"Loaded placement CSV with {len(df)} rows")
        
        # Check which nodes are being used
        node_counts = df['node_id'].value_counts()
        logging.info("Node utilization:")
        for node, count in node_counts.items():
            logging.info(f"  - {node}: {count} pods")
        
        # Check if French IoT node is being used
        if 'node-4-fr-iot' in node_counts:
            logging.info(f"✅ French IoT node utilized: {node_counts['node-4-fr-iot']} pods")
        else:
            logging.info("❌ French IoT node not utilized!")
        
        # Visualize node usage
        plt.figure(figsize=(10, 6))
        sns.barplot(x=node_counts.index, y=node_counts.values)
        plt.title('Pod Allocation by Node')
        plt.ylabel('Number of Pods')
        plt.xlabel('Node ID')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the figure
        os.makedirs('figures', exist_ok=True)
        plt.savefig('figures/node_utilization.png')
        logging.info("📊 Node utilization chart saved to figures/node_utilization.png")
        
        return df
    except Exception as e:
        logging.error(f"Error analyzing placement CSV: {e}")
        return None

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Test comprehensive scheduling with carbon efficiency per CPU prioritization')
    parser.add_argument('--port', default='50051', help='Server port')
    parser.add_argument('--workloads-dir', default='./workloads', help='Directory containing timeslot_*.yaml files')
    parser.add_argument('--nodes-file', default='../nodes.yaml', help='Path to nodes.yaml')
    parser.add_argument('--forecasts-file', default='./all_forecasts.json', help='Path to all_forecasts.json')
    
    args = parser.parse_args()
    
    setup_logging()
    
    logging.info("🚀 Starting test of comprehensive global optimization")
    
    # Create output directories
    os.makedirs('experiments', exist_ok=True)
    
    # Start the server with global optimization (now the default) and prioritize efficiency
    server_cmd = [
        "python", "main.py",
        "--algorithm", "global-optimal",
        "--prioritize-efficiency",
        "--workloads-dir", args.workloads_dir,
        "--nodes-file", args.nodes_file,
        "--forecasts-file", args.forecasts_file,
        "--port", args.port,
        "--experiment"
    ]
    
    logging.info(f"Starting server with command: {' '.join(server_cmd)}")
    
    server_process = subprocess.Popen(
        server_cmd, 
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Give the server time to start and perform comprehensive optimization
    logging.info("Waiting for server to start and perform comprehensive optimization...")
    time.sleep(5)
    
    # Capture server output for 30 seconds or until it completes
    start_time = time.time()
    server_output = []
    
    try:
        while time.time() - start_time < 30 and server_process.poll() is None:
            if server_process.stdout:
                line = server_process.stdout.readline()
                if line:
                    server_output.append(line.strip())
                    print(line.strip())
            time.sleep(0.1)
    except KeyboardInterrupt:
        logging.info("Test interrupted by user")
    finally:
        # Terminate the server
        if server_process.poll() is None:
            logging.info("Terminating server process...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logging.warning("Server process did not terminate cleanly, killing...")
                server_process.kill()
    
    # Find the most recent experiment directory
    experiment_dirs = [d for d in os.listdir('experiments') if d.startswith('global-optimal_experiment_session_')]
    if experiment_dirs:
        latest_dir = sorted(experiment_dirs)[-1]
        experiment_path = os.path.join('experiments', latest_dir)
        logging.info(f"Found experiment directory: {experiment_path}")
        
        # Look for placement CSV files
        csv_files = [f for f in os.listdir(experiment_path) if f.endswith('_placements.csv')]
        if csv_files:
            placement_csv = os.path.join(experiment_path, csv_files[0])
            logging.info(f"Found placement CSV: {placement_csv}")
            
            # Analyze the placement CSV
            df = analyze_placement_csv(placement_csv)
            
            if df is not None:
                logging.info("✅ Test completed successfully")
            else:
                logging.error("❌ Failed to analyze placement CSV")
        else:
            logging.error("❌ No placement CSV files found in the experiment directory")
    else:
        logging.error("❌ No experiment directories found")

if __name__ == "__main__":
    main()
