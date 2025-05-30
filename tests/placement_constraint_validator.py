#!/usr/bin/env python3
"""
Unified Placement Constraint Validator for Carbon-Aware Orchestrator

This script validates placement CSV results against both capacity and timeslot constraints:
- Capacity constraints: Verifies node resource limits are not exceeded
- Timeslot constraints: Verifies pods are scheduled at or after their earliest allowed timeslot

Usage:
    python placement_constraint_validator.py [CSV_FILE] [--nodes-file NODES_FILE] [--workloads-dir WORKLOADS_DIR]
"""

import argparse
import pandas as pd
import yaml
import re
import os
import sys
import glob
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

def parse_memory_str(mem_str: str) -> float:
    """Convert memory string (like '2Gi', '4Gi') to MB."""
    if mem_str.endswith('Gi'):
        return float(mem_str[:-2]) * 1024
    elif mem_str.endswith('Mi'):
        return float(mem_str[:-2])
    elif mem_str.endswith('G'):
        return float(mem_str[:-1]) * 1024
    elif mem_str.endswith('M'):
        return float(mem_str[:-1])
    else:
        return float(mem_str)

def load_node_capacities(nodes_file: str) -> Dict[str, Dict[str, float]]:
    """Load node capacities from YAML file."""
    with open(nodes_file, 'r') as f:
        docs = list(yaml.safe_load_all(f))
    
    capacities = {}
    for doc in docs:
        if doc and doc.get('kind') == 'Node':
            name = doc['metadata']['name']
            cpu_capacity = float(doc['status']['capacity']['cpu'])
            mem_capacity = parse_memory_str(doc['status']['capacity']['memory'])
            region = doc['metadata']['labels']['topology.kubernetes.io/region']
            
            capacities[name] = {
                'cpu': cpu_capacity,
                'memory': mem_capacity,
                'region': region
            }
    
    return capacities

def build_pod_timeslot_map(workloads_dir: str) -> Dict[str, int]:
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
    print(f"📁 Found {len(timeslot_files)} timeslot files in {workloads_dir}")
    
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
        except Exception as e:
            print(f"⚠️  Error reading {timeslot_file}: {e}")
    
    return pod_map

def validate_capacity_constraints(df: pd.DataFrame, node_capacities: Dict[str, Dict[str, float]]) -> Tuple[bool, List[Dict]]:
    """
    Validate capacity constraints for all placements.
    
    Returns:
        Tuple of (is_valid, violations_list)
    """
    violations = []
    node_timeslot_usage = defaultdict(lambda: defaultdict(lambda: {'cpu': 0, 'memory': 0, 'pods': []}))
    
    # Calculate resource usage by node and timeslot
    for _, pod in df.iterrows():
        node = pod['node_id']
        start_slot = int(pod['start_slot'])
        duration = pod['duration']
        cpu = pod['cpu_request']
        memory = pod['ram_request']
        pod_id = pod['pod_id']
        
        # Add resource usage for each timeslot the pod runs
        for slot in range(start_slot, start_slot + int(duration)):
            node_timeslot_usage[node][slot]['cpu'] += cpu
            node_timeslot_usage[node][slot]['memory'] += memory
            node_timeslot_usage[node][slot]['pods'].append({
                'id': pod_id,
                'cpu': cpu,
                'memory': memory,
                'start': start_slot,
                'duration': duration
            })
    
    # Check for violations
    for node_id, timeslots in node_timeslot_usage.items():
        if node_id not in node_capacities:
            violations.append({
                'type': 'unknown_node',
                'node_id': node_id,
                'message': f"Node {node_id} not found in nodes file"
            })
            continue
            
        node_caps = node_capacities[node_id]
        
        for slot, usage in timeslots.items():
            # Check CPU violation
            if usage['cpu'] > node_caps['cpu']:
                violations.append({
                    'type': 'cpu_violation',
                    'node_id': node_id,
                    'timeslot': slot,
                    'used': usage['cpu'],
                    'capacity': node_caps['cpu'],
                    'excess': usage['cpu'] - node_caps['cpu'],
                    'pod_count': len(usage['pods']),
                    'pods': [p['id'] for p in usage['pods']]
                })
            
            # Check memory violation
            if usage['memory'] > node_caps['memory']:
                violations.append({
                    'type': 'memory_violation',
                    'node_id': node_id,
                    'timeslot': slot,
                    'used': usage['memory'],
                    'capacity': node_caps['memory'],
                    'excess': usage['memory'] - node_caps['memory'],
                    'pod_count': len(usage['pods']),
                    'pods': [p['id'] for p in usage['pods']]
                })
    
    return len(violations) == 0, violations

def validate_timeslot_constraints(df: pd.DataFrame, pod_timeslot_map: Dict[str, int]) -> Tuple[bool, List[Dict]]:
    """
    Validate timeslot constraints for all placements.
    
    Returns:
        Tuple of (is_valid, violations_list)
    """
    violations = []
    checked_pods = 0
    
    for _, row in df.iterrows():
        pod_id = row['pod_id']
        start_slot = row['start_slot']
        source_timeslot = pod_timeslot_map.get(pod_id)
        
        if source_timeslot is None:
            continue  # Skip pods not found in timeslot files
            
        checked_pods += 1
        
        # Pod must be scheduled at or after its source timeslot
        if start_slot < source_timeslot:
            violations.append({
                'type': 'timeslot_violation',
                'pod_id': pod_id,
                'source_timeslot': source_timeslot,
                'actual_start_slot': start_slot,
                'violation_amount': source_timeslot - start_slot
            })
    
    return len(violations) == 0, violations

def print_capacity_violations(violations: List[Dict], node_capacities: Dict[str, Dict[str, float]]):
    """Print detailed capacity violation report."""
    cpu_violations = [v for v in violations if v['type'] == 'cpu_violation']
    memory_violations = [v for v in violations if v['type'] == 'memory_violation']
    unknown_nodes = [v for v in violations if v['type'] == 'unknown_node']
    
    if unknown_nodes:
        print("\n🚫 UNKNOWN NODES:")
        for v in unknown_nodes:
            print(f"   {v['message']}")
    
    if cpu_violations:
        print(f"\n🚨 CPU VIOLATIONS ({len(cpu_violations)}):")
        for v in cpu_violations:
            print(f"   Node {v['node_id']}, Slot {v['timeslot']}: {v['used']:.2f}/{v['capacity']:.2f} CPU")
            print(f"      Excess: {v['excess']:.2f} CPU from {v['pod_count']} pods")
            print(f"      Pods: {', '.join(v['pods'][:5])}{'...' if len(v['pods']) > 5 else ''}")
    
    if memory_violations:
        print(f"\n🚨 MEMORY VIOLATIONS ({len(memory_violations)}):")
        for v in memory_violations:
            print(f"   Node {v['node_id']}, Slot {v['timeslot']}: {v['used']:.0f}/{v['capacity']:.0f} MB")
            print(f"      Excess: {v['excess']:.0f} MB from {v['pod_count']} pods")
            print(f"      Pods: {', '.join(v['pods'][:5])}{'...' if len(v['pods']) > 5 else ''}")

def print_timeslot_violations(violations: List[Dict]):
    """Print detailed timeslot violation report."""
    if violations:
        print(f"\n🚨 TIMESLOT VIOLATIONS ({len(violations)}):")
        for v in violations:
            print(f"   Pod {v['pod_id']}: scheduled at slot {v['actual_start_slot']} "
                  f"but earliest allowed is {v['source_timeslot']} "
                  f"(violation: {v['violation_amount']} slots early)")

def print_summary_stats(df: pd.DataFrame, node_capacities: Dict[str, Dict[str, float]], 
                       pod_timeslot_map: Dict[str, int]):
    """Print summary statistics about the placement."""
    print("📊 PLACEMENT SUMMARY:")
    print(f"   Total placements: {len(df)}")
    print(f"   Unique nodes used: {df['node_id'].nunique()}")
    print(f"   Available nodes: {len(node_capacities)}")
    print(f"   Timeslot range: {df['start_slot'].min()} - {df['start_slot'].max()}")
    print(f"   Pods with timeslot mapping: {len(pod_timeslot_map)}")
    
    # Node utilization
    placement_counts = df['node_id'].value_counts()
    print("\n📍 NODE UTILIZATION:")
    for node, count in placement_counts.items():
        percentage = (count / len(df)) * 100
        if node in node_capacities:
            caps = node_capacities[node]
            print(f"   {node}: {count} pods ({percentage:.1f}%) - "
                  f"{caps['cpu']:.1f} CPU, {caps['memory']:.0f} MB, {caps['region']}")
        else:
            print(f"   {node}: {count} pods ({percentage:.1f}%) - UNKNOWN NODE")

def validate_placement_constraints(csv_file: str, nodes_file: Optional[str] = None, 
                                 workloads_dir: Optional[str] = None) -> bool:
    """
    Main validation function that checks both capacity and timeslot constraints.
    
    Returns:
        True if all constraints are satisfied, False otherwise
    """
    print("=" * 80)
    print("CARBON-AWARE ORCHESTRATOR - PLACEMENT CONSTRAINT VALIDATOR")
    print("=" * 80)
    print(f"📄 Analyzing: {csv_file}")
    
    # Load placement data
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return False
    
    # Validate required columns
    required_columns = ['pod_id', 'node_id', 'start_slot', 'duration', 'cpu_request', 'ram_request']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        return False
    
    # Convert numeric columns
    df['start_slot'] = pd.to_numeric(df['start_slot'], errors='coerce')
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
    df['cpu_request'] = pd.to_numeric(df['cpu_request'], errors='coerce')
    df['ram_request'] = pd.to_numeric(df['ram_request'], errors='coerce')
    
    all_valid = True
    
    # Validate capacity constraints if nodes file provided
    if nodes_file and os.path.exists(nodes_file):
        print(f"\n🔍 VALIDATING CAPACITY CONSTRAINTS")
        print(f"📄 Nodes file: {nodes_file}")
        
        try:
            node_capacities = load_node_capacities(nodes_file)
            capacity_valid, capacity_violations = validate_capacity_constraints(df, node_capacities)
            
            if capacity_valid:
                print("✅ All capacity constraints satisfied!")
            else:
                print(f"❌ Found {len(capacity_violations)} capacity violations")
                print_capacity_violations(capacity_violations, node_capacities)
                all_valid = False
                
        except Exception as e:
            print(f"❌ Error validating capacity constraints: {e}")
            all_valid = False
    else:
        print("⏭️  Skipping capacity validation (no nodes file provided)")
        node_capacities = {}
    
    # Validate timeslot constraints if workloads dir provided
    if workloads_dir and os.path.exists(workloads_dir):
        print(f"\n🔍 VALIDATING TIMESLOT CONSTRAINTS")
        print(f"📁 Workloads directory: {workloads_dir}")
        
        try:
            pod_timeslot_map = build_pod_timeslot_map(workloads_dir)
            timeslot_valid, timeslot_violations = validate_timeslot_constraints(df, pod_timeslot_map)
            
            pods_checked = len([pod for pod in df['pod_id'] if pod in pod_timeslot_map])
            print(f"📋 Checked {pods_checked}/{len(df)} pods against timeslot files")
            
            if timeslot_valid:
                print("✅ All timeslot constraints satisfied!")
            else:
                print(f"❌ Found {len(timeslot_violations)} timeslot violations")
                print_timeslot_violations(timeslot_violations)
                all_valid = False
                
        except Exception as e:
            print(f"❌ Error validating timeslot constraints: {e}")
            all_valid = False
    else:
        print("⏭️  Skipping timeslot validation (no workloads directory provided)")
        pod_timeslot_map = {}
    
    # Print summary
    print(f"\n📊 SUMMARY")
    print_summary_stats(df, node_capacities, pod_timeslot_map)
    
    print(f"\n🎯 FINAL RESULT: {'✅ ALL CONSTRAINTS SATISFIED' if all_valid else '❌ CONSTRAINT VIOLATIONS FOUND'}")
    
    return all_valid

def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Validate placement CSV results against capacity and timeslot constraints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate both constraints
  python placement_constraint_validator.py results.csv --nodes-file nodes.yaml --workloads-dir workloads/
  
  # Validate only capacity constraints
  python placement_constraint_validator.py results.csv --nodes-file nodes.yaml
  
  # Validate only timeslot constraints
  python placement_constraint_validator.py results.csv --workloads-dir workloads/
        """
    )
    
    parser.add_argument('csv_file', 
                       help='Path to the placement CSV file to validate')
    parser.add_argument('--nodes-file', 
                       help='Path to the nodes.yaml file (for capacity validation)')
    parser.add_argument('--workloads-dir', 
                       help='Path to the workloads directory containing timeslot_*.yaml files')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.csv_file):
        print(f"❌ Error: CSV file not found: {args.csv_file}")
        sys.exit(1)
    
    # Set defaults if not provided
    if not args.nodes_file and not args.workloads_dir:
        # Default paths
        base_dir = "/root/carbon-aware-orchestrator/pkg/carbon-aware"
        args.nodes_file = f"{base_dir}/server-python/nodes.yaml"
        args.workloads_dir = f"{base_dir}/workloads"
        print(f"🔧 Using default paths:")
        print(f"   Nodes file: {args.nodes_file}")
        print(f"   Workloads dir: {args.workloads_dir}")
    
    # Run validation
    try:
        success = validate_placement_constraints(
            args.csv_file, 
            args.nodes_file, 
            args.workloads_dir
        )
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
