#!/usr/bin/env python3
"""
Script to check if the earliest timeslot constraint is properly enforced in placement CSV output.
The constraint: Pods from timeslot_n.yaml should only be scheduled at or after timeslot n.
"""

import argparse
import pandas as pd
import re
import os
import sys


import glob

def build_pod_timeslot_map(workloads_dir="/root/carbon-aware-orchestrator/pkg/carbon-aware/workloads"):
    """
    Build a mapping of pod IDs to their source timeslot numbers.
    
    Args:
        workloads_dir: Directory containing timeslot_*.yaml files
    
    Returns:
        dict: Mapping of pod IDs to timeslot numbers
    """
    pod_map = {}
    
    # Get all timeslot files
    timeslot_files = sorted(glob.glob(os.path.join(workloads_dir, "timeslot_*.yaml")))
    print(f"Found {len(timeslot_files)} timeslot files in {workloads_dir}")
    
    for timeslot_file in timeslot_files:
        # Extract timeslot number from filename
        timeslot_match = re.search(r'timeslot_(\d+)\.yaml', os.path.basename(timeslot_file))
        if not timeslot_match:
            continue
            
        timeslot_number = int(timeslot_match.group(1))
        
        # Find all pod names in this file
        try:
            with open(timeslot_file, 'r') as f:
                content = f.read()
                # Look for all deployment names with pattern m\d+
                for match in re.finditer(r'name: (m\d+-[^,\s\n]+)', content):
                    pod_id = match.group(1)
                    pod_map[pod_id] = timeslot_number
                    print(f"Found pod {pod_id} in timeslot {timeslot_number}")
        except Exception as e:
            print(f"Error reading {timeslot_file}: {e}")
    
    return pod_map


def check_timeslot_constraints(csv_path):
    """
    Check if pods are scheduled at or after their source timeslot.
    
    The constraint is: Pods from timeslot_n.yaml should only be scheduled
    at or after timeslot n, regardless of the pod's ID number.
    """
    if not os.path.exists(csv_path):
        print(f"Error: File not found - {csv_path}")
        return False

    try:
        # Build mapping of pod IDs to source timeslots
        print("Building pod to timeslot mapping...")
        pod_timeslot_map = build_pod_timeslot_map()
        print(f"Found {len(pod_timeslot_map)} pods in timeslot files")
        
        # Load the CSV data
        print(f"Loading CSV from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        if 'pod_id' not in df.columns or 'start_slot' not in df.columns:
            print("Error: CSV file must contain 'pod_id' and 'start_slot' columns")
            return False

        # Convert start_slot to numeric if it's not already
        df['start_slot'] = pd.to_numeric(df['start_slot'], errors='coerce')
        
        # Check if each pod is scheduled at or after its source timeslot
        violations = []
        valid_pods = 0
        checked_pods = 0
        total_pods = len(df)
        
        # Process each pod in the CSV
        for _, row in df.iterrows():
            pod_id = row['pod_id']
            start_slot = row['start_slot']
            source_timeslot = pod_timeslot_map.get(pod_id, -1)
            
            if source_timeslot == -1:
                print(f"Warning: Pod {pod_id} not found in any timeslot file, skipping check")
                continue
                
            checked_pods += 1
            
            # Pod must be scheduled at or after its source timeslot
            if start_slot < source_timeslot:
                violations.append({
                    'pod_id': pod_id,
                    'source_timeslot': source_timeslot,
                    'actual_start_slot': start_slot
                })
            else:
                valid_pods += 1
                    
        # Print the analysis results
        print(f"\n===== TIMESLOT CONSTRAINT ANALYSIS =====")
        print(f"CSV File: {csv_path}")
        print(f"Total pods analyzed: {total_pods}")
        print(f"Pods checked against timeslot files: {checked_pods}")
        print(f"Pods with valid timeslots: {valid_pods}")
        print(f"Constraint violations: {len(violations)}")
        
        if violations:
            print("\nVIOLATIONS FOUND:")
            print("------------------")
            for v in violations:
                print(f"Pod {v['pod_id']} from timeslot_{v['source_timeslot']}.yaml scheduled at timeslot {v['actual_start_slot']}")
            return False
        else:
            print("\n✓ All pods respect the earliest timeslot constraint!")
            return True
            
    except Exception as e:
        print(f"Error analyzing CSV file: {e}")
        import traceback
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    # Get CSV path from command line argument or use default
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "/root/carbon-aware-orchestrator/pkg/carbon-aware/server-python/experiments/global-optimal_perf_log_session_20250517_121052/global_optimal_placements_session.csv"
    
    print(f"Script is running, checking file: {csv_path}")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"CSV file not found at: {csv_path}")
        sys.exit(1)
    
    try:
        # Just try to read the first few lines to confirm the file is accessible
        with open(csv_path, 'r') as f:
            first_lines = [next(f) for _ in range(5) if f.readable()]
            print("First 5 lines of the CSV:")
            for line in first_lines:
                print(line.strip())
    except Exception as e:
        print(f"Error reading file: {e}")
        
    # Run the check
    try:
        if check_timeslot_constraints(csv_path):
            print("All constraints satisfied!")
            sys.exit(0)
        else:
            print("Constraint violations found!")
            sys.exit(1)
    except Exception as e:
        print(f"Uncaught exception: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)
