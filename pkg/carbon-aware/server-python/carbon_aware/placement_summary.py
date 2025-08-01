#!/usr/bin/env python3
"""
Utility module for generating placement summary reports
"""

import pandas as pd
import os
import re
import logging
from datetime import datetime



def get_total_pods_from_workloads(workloads_dir=None):
    """
    Dynamically calculate the total number of pods by scanning the last timeslot file.
    
    Args:
        workloads_dir (str): Path to workloads directory. If None, uses relative path.
    
    Returns:
        int: Total number of pods (highest pod number + 1)
    """
    try:
        # Determine workloads directory
        if workloads_dir is None:
            # Default to relative path from placement_summary.py location
            # placement_summary.py is in carbon_aware/, workloads is in ../../workloads
            current_dir = os.path.dirname(os.path.abspath(__file__))
            workloads_dir = os.path.join(current_dir, "..", "..", "workloads")
        
        # Look for timeslot_11.yaml (the last timeslot file)
        timeslot_11_path = os.path.join(workloads_dir, "timeslot_11.yaml")
        
        if not os.path.exists(timeslot_11_path):
            logging.warning(f"⚠️ Could not find timeslot_11.yaml at {timeslot_11_path}, falling back to total=47")
            return 47
        
        # Read the file and extract pod numbers
        with open(timeslot_11_path, "r") as f:
            content = f.read()
        
        # Find all pod numbers (pattern: m followed by digits)
        pod_numbers = re.findall(r"m(\d+)", content)
        
        if not pod_numbers:
            logging.warning(f"⚠️ No pod numbers found in {timeslot_11_path}, falling back to total=47")
            return 47
        
        # Convert to integers and find the maximum
        max_pod_number = max(int(num) for num in pod_numbers)
        total_pods = max_pod_number + 1  # +1 because pod numbering starts from 0
        
        logging.info(f"📊 Dynamically calculated total pods: {total_pods} (highest pod: m{max_pod_number:03d})")
        return total_pods
        
    except Exception as e:
        logging.warning(f"⚠️ Error calculating total pods: {e}, falling back to total=47")
        return 47


def generate_placement_summary(csv_path, experiment_type, output_dir=None):
    """
    Generate a placement summary report from a placement CSV file
    
    Args:
        csv_path (str): Path to the placement CSV file
        experiment_type (str): Type of experiment ('global-optimal', 'heuristic', etc.)
        output_dir (str): Optional output directory, defaults to same dir as CSV
    
    Returns:
        str: Path to the generated summary file, or None if failed
    """
    
    try:
        # Check if CSV file exists
        if not os.path.exists(csv_path):
            logging.warning(f"📄 Placement CSV file not found: {csv_path}")
            return None
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        if df.empty:
            logging.warning(f"📄 Placement CSV file is empty: {csv_path}")
            return None
        
        # Count unique pods
        unique_pods = df['pod_id'].nunique()
        unique_pod_list = sorted(df['pod_id'].unique())
        
        # Count node distribution 
        node_distribution = df['node_id'].value_counts().sort_index()
        
        # Total pods in workloads (from our analysis - could be made dynamic)
        total_workload_pods = get_total_pods_from_workloads()
        
        # Calculate success rate
        pods_not_placed = total_workload_pods - unique_pods
        success_rate = (unique_pods / total_workload_pods) * 100
        
        # Extract experiment timestamp from CSV path
        timestamp_match = re.search(r'(\d{8}_\d{6})', csv_path)
        if timestamp_match:
            timestamp_str = timestamp_match.group(1)
            dt = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            formatted_date = dt.strftime('%Y-%m-%d %H:%M:%S')
        else:
            formatted_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Determine output path
        if output_dir is None:
            output_dir = os.path.dirname(csv_path)
        
        output_path = os.path.join(output_dir, 'placement_summary.log')
        
        # Generate the report content
        report_lines = [
            f"{experiment_type.title()} Experiment Placement Summary",
            f"Generated: {formatted_date}",
            "",
            "Experiment-Wide Pod Placement Statistics:",
            f"- Total pods in workloads: {total_workload_pods}",
            f"- Unique pods successfully placed: {unique_pods}",
            f"- Pods not placed: {pods_not_placed}",
            f"- Success rate: {unique_pods}/{total_workload_pods} = {success_rate:.1f}%",
            "",
            "Unique Successfully Placed Pods:",
            ", ".join(unique_pod_list),
            "",
            "Node Placement Distribution:",
        ]
        
        # Add node distribution
        for node, count in node_distribution.items():
            report_lines.append(f"- {node}: {count} pods")
        
        report_lines.append("")
        
        # Write the report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logging.info(f"📊 Generated placement summary: {output_path}")
        logging.info(f"   📈 Placed: {unique_pods}/{total_workload_pods} pods ({success_rate:.1f}%)")
        logging.info(f"   🏗️ Nodes used: {len(node_distribution)}")
        
        return output_path
        
    except Exception as e:
        logging.error(f"❌ Failed to generate placement summary for {csv_path}: {e}")
        return None


def auto_generate_summary_from_session_dir(session_log_dir, experiment_type):
    """
    Automatically find and generate placement summary from session directory
    
    Args:
        session_log_dir (str): Path to the session log directory
        experiment_type (str): Type of experiment
    
    Returns:
        str: Path to generated summary, or None if failed
    """
    
    if not session_log_dir or not os.path.exists(session_log_dir):
        logging.warning(f"📁 Session log directory not found: {session_log_dir}")
        return None
    
    # Look for placement CSV files in the session directory
    csv_patterns = [
        f"{experiment_type.replace('-', '_')}_placements_session.csv",
        "placements_session.csv",
        "*placements_session.csv"
    ]
    
    csv_path = None
    for pattern in csv_patterns:
        if '*' in pattern:
            # Use glob for wildcard patterns
            import glob
            matches = glob.glob(os.path.join(session_log_dir, pattern))
            if matches:
                csv_path = matches[0]  # Take first match
                break
        else:
            potential_path = os.path.join(session_log_dir, pattern)
            if os.path.exists(potential_path):
                csv_path = potential_path
                break
    
    if csv_path:
        return generate_placement_summary(csv_path, experiment_type, session_log_dir)
    else:
        logging.warning(f"📄 No placement CSV file found in {session_log_dir}")
        return None