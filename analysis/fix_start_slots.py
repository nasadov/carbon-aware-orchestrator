#!/usr/bin/env python3

import pandas as pd
import yaml
import os
import re
from pathlib import Path

def extract_base_pod_name(full_pod_name):
    """
    Extract base pod name by removing the last two dash-separated components.
    Example: m000-duration-1h-deadline-7h-5c5cf8d79-2g54p -> m000-duration-1h-deadline-7h
    """
    parts = full_pod_name.split('-')
    if len(parts) >= 2:
        # Remove last two components (hash1 and hash2)
        base_name = '-'.join(parts[:-2])
        return base_name
    return full_pod_name

def parse_duration_from_name(pod_name):
    """
    Extract duration in hours from pod name.
    Example: m000-duration-1h-deadline-7h -> 1.0
    """
    import re
    match = re.search(r'duration-(\d+)h', pod_name)
    if match:
        return float(match.group(1))
    return 1.0  # Default duration

def parse_cpu_from_yaml_content(yaml_content, pod_name):
    """
    Extract CPU request from YAML content for specific pod.
    Returns CPU in cores (e.g., 1000m -> 1.0, 250m -> 0.25)
    """
    try:
        # Split by deployment sections
        deployments = yaml_content.split('---')
        for deployment in deployments:
            if f'name: {pod_name}' in deployment:
                # Look for cpu: value
                import re
                cpu_match = re.search(r'cpu:\s*(\d+)m', deployment)
                if cpu_match:
                    cpu_millicores = int(cpu_match.group(1))
                    return cpu_millicores / 1000.0  # Convert to cores
                # Also check for cpu without 'm' suffix
                cpu_match = re.search(r'cpu:\s*(\d+\.?\d*)', deployment)
                if cpu_match:
                    return float(cpu_match.group(1))
        return 0.0  # Default if not found
    except Exception as e:
        print(f"Error parsing CPU for {pod_name}: {e}")
        return 0.0

def find_pod_details_in_timeslot_files(base_pod_name, workloads_dir):
    """
    Find which timeslot file contains the given pod name and extract its details.
    Returns tuple: (timeslot_number, duration, cpu_request) or (None, None, None) if not found.
    """
    for timeslot in range(12):  # timeslot_0.yaml through timeslot_11.yaml
        timeslot_file = os.path.join(workloads_dir, f'timeslot_{timeslot}.yaml')
        try:
            with open(timeslot_file, 'r') as f:
                content = f.read()
                # Look for the pod name in the metadata.name field
                if f'name: {base_pod_name}' in content:
                    # Extract duration from pod name
                    duration = parse_duration_from_name(base_pod_name)
                    # Extract CPU from YAML content
                    cpu_request = parse_cpu_from_yaml_content(content, base_pod_name)
                    return timeslot, duration, cpu_request
        except FileNotFoundError:
            print(f"Warning: {timeslot_file} not found")
            continue
    return None, None, None

def fix_csv_start_slots(csv_path, workloads_dir, output_path):
    """
    Read the CSV file, fix start_slot values and extract real duration/CPU, then save the corrected version.
    """
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Found {len(df)} pod entries")
    
    # Create a mapping from base pod names to correct details
    pod_to_details = {}
    fixed_count = 0
    not_found_count = 0
    
    for idx, row in df.iterrows():
        full_pod_name = row['pod_id']
        base_pod_name = extract_base_pod_name(full_pod_name)
        
        if base_pod_name not in pod_to_details:
            # Find which timeslot file contains this pod and extract details
            correct_timeslot, duration, cpu_request = find_pod_details_in_timeslot_files(base_pod_name, workloads_dir)
            pod_to_details[base_pod_name] = (correct_timeslot, duration, cpu_request)
            
            if correct_timeslot is not None:
                print(f"Pod {base_pod_name} found in timeslot_{correct_timeslot}.yaml - duration: {duration}h, CPU: {cpu_request} cores")
            else:
                print(f"Warning: Pod {base_pod_name} not found in any timeslot file")
                not_found_count += 1
        
        # Update the values
        correct_timeslot, duration, cpu_request = pod_to_details[base_pod_name]
        if correct_timeslot is not None:
            df.at[idx, 'start_slot'] = correct_timeslot
            df.at[idx, 'duration'] = duration
            df.at[idx, 'cpu_request'] = cpu_request
            fixed_count += 1
        else:
            # Keep original values if not found
            pass
    
    print(f"\nSummary:")
    print(f"- Fixed start_slot, duration, and CPU for {fixed_count} pods")
    print(f"- Could not find {not_found_count} pods in timeslot files")
    
    # Save the corrected CSV
    df.to_csv(output_path, index=False)
    print(f"Saved corrected CSV to: {output_path}")
    
    return df

def main():
    print("Starting fix_start_slots.py script...")
    
    # Paths
    from repo_paths import EXPERIMENTS_ROOT, WORKLOADS_VANILLA_DIR

    workloads_dir = str(WORKLOADS_VANILLA_DIR)
    experiments_root = EXPERIMENTS_ROOT

    csv_path = experiments_root / 'vanilla_placement_session.csv'
    output_path = experiments_root / 'vanilla_placement_session_fixed.csv'

    if not csv_path.exists():
        candidates = sorted(
            experiments_root.glob('**/vanilla_placement_session.csv'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if candidates:
            csv_path = candidates[0]
            output_path = csv_path.with_name('vanilla_placement_session_fixed.csv')

    
    print(f"Workloads directory: {workloads_dir}")
    print(f"Input CSV: {csv_path}")
    print(f"Output CSV: {output_path}")
    
    # Check if paths exist
    if not os.path.exists(workloads_dir):
        print(f"ERROR: Workloads directory not found: {workloads_dir}")
        return
    if not os.path.exists(csv_path):
        print(f"ERROR: Input CSV not found: {csv_path}")
        return

    # Fix the start slot values
    df_fixed = fix_csv_start_slots(str(csv_path), workloads_dir, str(output_path))
    
    # Show some statistics
    print(f"\nStart slot distribution after fixing:")
    print(df_fixed['start_slot'].value_counts().sort_index())

if __name__ == '__main__':
    main()
