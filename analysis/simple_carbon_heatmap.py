#!/usr/bin/env python3
"""
Simple Carbon Footprint Heatmap Generator
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import yaml
import json
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def load_nodes_yaml(filepath):
    """Load node information from nodes.yaml"""
    nodes_data = {}
    try:
        with open(filepath, 'r') as file:
            all_docs = yaml.safe_load_all(file)
            for doc in all_docs:
                if not doc:
                    continue
                
                metadata = doc.get('metadata', {})
                name = metadata.get('name')
                
                if not name:
                    continue
                
                # Get node region
                labels = metadata.get('labels', {})
                region = labels.get('topology.kubernetes.io/region', '')
                subcategory = labels.get('hardware.carbon/subcategory', '')
                
                # Get embodied carbon data
                annotations = metadata.get('annotations', {})
                embodied_emissions = float(annotations.get('hardware.carbon/embodied_emissions', 0))
                lifetime_years = float(annotations.get('hardware.carbon/lifetime_years', 1))
                
                # Get power data
                idle_watts = float(annotations.get('hardware.power/idle_watts', 0))
                active_watts = float(annotations.get('hardware.power/active_watts', 0))
                
                # Get CPU capacity
                status = doc.get('status', {})
                allocatable = status.get('allocatable', {})
                cpu_str = allocatable.get('cpu', '1')
                
                try:
                    if isinstance(cpu_str, str):
                        if cpu_str.endswith('m'):
                            cpu_capacity = float(cpu_str[:-1]) / 1000.0
                        else:
                            cpu_capacity = float(cpu_str)
                    else:
                        cpu_capacity = float(cpu_str)
                except (ValueError, TypeError):
                    cpu_capacity = 1.0
                
                nodes_data[name] = {
                    'region': region,
                    'subcategory': subcategory,
                    'embodied_emissions': embodied_emissions,
                    'lifetime_years': lifetime_years,
                    'idle_watts': idle_watts,
                    'active_watts': active_watts,
                    'cpu_capacity': cpu_capacity
                }
                
        return nodes_data
    except Exception as e:
        logging.error(f"Error loading nodes.yaml: {e}")
        return {}

def load_carbon_intensity(filepath):
    """Load carbon intensity data"""
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
        
        carbon_data = {}
        for region, region_data in data.items():
            forecast = region_data.get('forecast', [])
            carbon_data[region] = [entry.get('carbonIntensity', 0) for entry in forecast]
        
        return carbon_data
    except Exception as e:
        logging.error(f"Error loading carbon intensity data: {e}")
        return {}

def calculate_carbon_footprint(nodes_data, carbon_intensity, placements_df=None, num_timeslots=24):
    """Calculate carbon footprint for each node and timeslot"""
    node_names = list(nodes_data.keys())
    carbon_matrix = np.zeros((len(node_names), num_timeslots))
    embodied_matrix = np.zeros((len(node_names), num_timeslots))
    operational_matrix = np.zeros((len(node_names), num_timeslots))
    
    # Track pod placements to show occupancy
    pod_occupancy = {}
    for node in node_names:
        pod_occupancy[node] = {ts: [] for ts in range(num_timeslots)}
    
    # If we have placements data, populate occupancy
    if placements_df is not None and not placements_df.empty:
        for _, row in placements_df.iterrows():
            node_id = row['node_id']
            if node_id not in pod_occupancy:
                continue
                
            start_slot = int(row['start_slot'])
            duration = int(row['duration']) 
            pod_id = row['pod_id']
            
            for ts in range(start_slot, min(start_slot + duration, num_timeslots)):
                pod_occupancy[node_id][ts].append(pod_id)
    
    # Log node configuration to understand embodied vs operational carbon
    logging.info("\nNode configuration details:")
    for node_name, node_info in nodes_data.items():
        logging.info(f"\n{node_name}:")
        logging.info(f"  - Embodied emissions: {node_info['embodied_emissions']} g CO2e (grams)")
        logging.info(f"  - Lifetime years: {node_info['lifetime_years']} years")
        logging.info(f"  - Idle watts: {node_info['idle_watts']} W")
        logging.info(f"  - Active watts: {node_info['active_watts']} W")
        # Calculate hourly amortized embodied carbon
        # Assuming embodied_emissions is in grams
        embodied_per_hour = node_info['embodied_emissions'] / (node_info['lifetime_years'] * 365 * 24)
        logging.info(f"  - Amortized embodied carbon: {embodied_per_hour:.6f} g CO2e/hour")
        
    # Calculate carbon footprint for each cell
    for node_idx, node_name in enumerate(node_names):
        node_info = nodes_data[node_name]
        region = node_info['region']
        
        # Calculate hourly amortized embodied carbon
        # Assuming embodied_emissions is in grams
        embodied_per_hour = node_info['embodied_emissions'] / (node_info['lifetime_years'] * 365 * 24)
        
        # Get region carbon intensity, or use average if not available
        region_intensity = carbon_intensity.get(region, [350] * num_timeslots)  # 350 is a reasonable default
        if len(region_intensity) < num_timeslots:
            # Pad if we don't have enough data
            region_intensity = region_intensity + [region_intensity[-1]] * (num_timeslots - len(region_intensity))
        
        for ts in range(num_timeslots):
            # Base carbon is embodied + idle
            idle_power_kw = node_info['idle_watts'] / 1000.0  # convert to kW
            carbon_intensity_value = region_intensity[ts]     # in gCO2/kWh
            
            # Carbon from idle power
            idle_carbon = idle_power_kw * carbon_intensity_value
            
            # Save embodied carbon separately
            embodied_matrix[node_idx, ts] = embodied_per_hour
            
            # Save operational carbon (idle) separately
            operational_carbon = idle_carbon
            
            # If there are pods scheduled in this timeslot, add active power
            pods_in_slot = len(pod_occupancy[node_name][ts])
            if pods_in_slot > 0:
                # Simple utilization based on number of pods vs CPU capacity
                utilization = min(pods_in_slot / node_info['cpu_capacity'], 1.0)
                active_power_kw = (node_info['active_watts'] - node_info['idle_watts']) / 1000.0 * utilization
                active_carbon = active_power_kw * carbon_intensity_value
                operational_carbon += active_carbon
            
            operational_matrix[node_idx, ts] = operational_carbon
            
            # Total carbon is embodied + operational
            carbon_matrix[node_idx, ts] = embodied_per_hour + operational_carbon
    
    return carbon_matrix, node_names, pod_occupancy, operational_matrix, embodied_matrix

def create_carbon_heatmap(carbon_matrix, node_names, pod_occupancy, output_dir, filename_prefix):
    """Create and save a heatmap visualization"""
    plt.figure(figsize=(15, 8))
    
    # Create the heatmap with values rounded to 2 decimal places
    ax = sns.heatmap(np.round(carbon_matrix, 2), cmap='YlOrRd', annot=True, fmt=".2f", linewidths=0.5, cbar_kws={'label': 'Carbon Footprint (g CO2e/hour)'})
    
    # Set axis labels and title
    ax.set_title('Carbon Footprint Heatmap (Embodied + Operational, g CO2e/hour)')
    ax.set_ylabel('Node')
    ax.set_xlabel('Timeslot')
    
    # Set y-tick labels to node names
    ax.set_yticks(np.arange(len(node_names)) + 0.5)
    ax.set_yticklabels(node_names)
    
    # Set x-tick labels to timeslot numbers
    ax.set_xticks(np.arange(carbon_matrix.shape[1]) + 0.5)
    ax.set_xticklabels(range(carbon_matrix.shape[1]))
    
    # Mark cells with pod placements
    for node_idx, node_name in enumerate(node_names):
        for ts, pods in pod_occupancy[node_name].items():
            if pods:  # If there are pods in this cell
                # Add an outline or marker
                rect = plt.Rectangle((ts, node_idx), 1, 1, fill=False, edgecolor='blue', lw=2)
                ax.add_patch(rect)
                
                # Add text showing number of pods if more than 1
                if len(pods) > 1:
                    plt.text(ts + 0.5, node_idx + 0.25, f"{len(pods)} pods", ha='center', fontsize=8)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    output_path = os.path.join(output_dir, f"{filename_prefix}_carbon_heatmap.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path



def create_operational_carbon_heatmap(operational_matrix, node_names, pod_occupancy, output_dir, filename_prefix):
    """Create and save a heatmap visualization for operational carbon only (without embodied)"""
    # Create the heatmap
    plt.figure(figsize=(15, 8))
    
    # Create the heatmap with a different colormap to distinguish from total carbon
    # Round to 2 decimal places for better readability
    ax = sns.heatmap(np.round(operational_matrix, 2), cmap='Blues', annot=True, fmt=".2f", linewidths=0.5, 
                    cbar_kws={'label': 'Operational Carbon (g CO2e/hour)'})
    
    # Set axis labels and title
    ax.set_title('Operational Carbon Footprint Heatmap (Power Consumption Only, g CO2e/hour)')
    ax.set_ylabel('Node')
    ax.set_xlabel('Timeslot')
    
    # Set y-tick labels to node names
    ax.set_yticks(np.arange(len(node_names)) + 0.5)
    ax.set_yticklabels(node_names)
    
    # Set x-tick labels to timeslot numbers
    ax.set_xticks(np.arange(operational_matrix.shape[1]) + 0.5)
    ax.set_xticklabels(range(operational_matrix.shape[1]))
    
    # Mark cells with pod placements
    for node_idx, node_name in enumerate(node_names):
        for ts, pods in pod_occupancy[node_name].items():
            if pods:  # If there are pods in this cell
                # Add an outline or marker
                rect = plt.Rectangle((ts, node_idx), 1, 1, fill=False, edgecolor='blue', lw=2)
                ax.add_patch(rect)
                
                # Add text showing number of pods if more than 1
                if len(pods) > 1:
                    plt.text(ts + 0.5, node_idx + 0.25, f"{len(pods)} pods", ha='center', fontsize=8)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    output_path = os.path.join(output_dir, f"{filename_prefix}_operational_carbon_heatmap.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path

def create_embodied_carbon_heatmap(embodied_matrix, node_names, output_dir, filename_prefix):
    """Create and save a heatmap visualization for embodied carbon only"""
    # Create the heatmap
    plt.figure(figsize=(15, 8))
    
    # Create the heatmap with a different colormap to distinguish from other plots
    # Round to 2 decimal places for better readability
    ax = sns.heatmap(np.round(embodied_matrix, 2), cmap='Greens', annot=True, fmt=".2f", linewidths=0.5, 
                    cbar_kws={'label': 'Embodied Carbon (g CO2e/hour)'})
    
    # Set axis labels and title
    ax.set_title('Embodied Carbon Footprint Heatmap (Amortized Hardware Emissions, g CO2e/hour)')
    ax.set_ylabel('Node')
    ax.set_xlabel('Timeslot')
    
    # Set y-tick labels to node names
    ax.set_yticks(np.arange(len(node_names)) + 0.5)
    ax.set_yticklabels(node_names)
    
    # Set x-tick labels to timeslot numbers
    ax.set_xticks(np.arange(embodied_matrix.shape[1]) + 0.5)
    ax.set_xticklabels(range(embodied_matrix.shape[1]))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    output_path = os.path.join(output_dir, f"{filename_prefix}_embodied_carbon_heatmap.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate carbon footprint heatmap')
    parser.add_argument('placements_csv', help='Path to placements CSV file')
    parser.add_argument('--nodes-yaml', default='/root/carbon-aware-orchestrator/pkg/carbon-aware/nodes.yaml', 
                        help='Path to nodes.yaml file')
    parser.add_argument('--carbon-data', default='/root/carbon-aware-orchestrator/bin/all_forecasts.json', 
                        help='Path to carbon intensity data')
    parser.add_argument('--timeslots', type=int, default=24, help='Number of timeslots')
    args = parser.parse_args()
    
    # Load node data
    nodes_data = load_nodes_yaml(args.nodes_yaml)
    if not nodes_data:
        logging.error("Failed to load node data")
        return 1
    
    # Load carbon intensity data
    carbon_intensity = load_carbon_intensity(args.carbon_data)
    if not carbon_intensity:
        logging.error("Failed to load carbon intensity data")
        return 1
    
    # Load placements data
    try:
        placements_df = pd.read_csv(args.placements_csv)
        logging.info(f"Loaded placements with {len(placements_df)} entries")
    except Exception as e:
        logging.error(f"Error loading placements CSV: {e}")
        return 1
    
    # Determine output directory and filename
    csv_dir = os.path.dirname(args.placements_csv)
    csv_basename = os.path.basename(csv_dir)
    output_dir = os.path.join('/root/carbon-aware-orchestrator/figures', csv_basename)
    filename_prefix = os.path.splitext(os.path.basename(args.placements_csv))[0]
    
    # Calculate carbon footprint
    carbon_matrix, node_names, pod_occupancy, operational_matrix, embodied_matrix = calculate_carbon_footprint(
        nodes_data, 
        carbon_intensity, 
        placements_df, 
        num_timeslots=args.timeslots
    )
    
    # Create and save the heatmap
    output_path = create_carbon_heatmap(carbon_matrix, node_names, pod_occupancy, output_dir, filename_prefix)
    logging.info(f"Carbon footprint heatmap saved to: {output_path}")
    
    # Create and save operational carbon heatmap (without embodied emissions)
    operational_path = create_operational_carbon_heatmap(
        operational_matrix, node_names, pod_occupancy, output_dir, filename_prefix
    )
    logging.info(f"Operational carbon heatmap saved to: {operational_path}")
    
    # Create and save embodied carbon heatmap
    embodied_path = create_embodied_carbon_heatmap(
        embodied_matrix, node_names, output_dir, filename_prefix
    )
    logging.info(f"Embodied carbon heatmap saved to: {embodied_path}")
    
    # Print total carbon footprint
    total_carbon = carbon_matrix.sum()
    total_operational_carbon = operational_matrix.sum()
    total_embodied_carbon = embodied_matrix.sum()
    
    logging.info(f"\nCarbon Footprint Summary for 24-hour period:")
    logging.info(f"- Total carbon footprint: {total_carbon:.2f} g CO2e")
    logging.info(f"- Operational carbon: {total_operational_carbon:.2f} g CO2e ({(total_operational_carbon/total_carbon*100):.1f}%)")
    logging.info(f"- Embodied carbon: {total_embodied_carbon:.2f} g CO2e ({(total_embodied_carbon/total_carbon*100):.1f}%)")
    
    # Print carbon footprint by node
    logging.info("\nCarbon footprint by node:")
    for node_idx, node_name in enumerate(node_names):
        node_carbon = carbon_matrix[node_idx].sum()
        node_operational = operational_matrix[node_idx].sum()
        node_embodied = embodied_matrix[node_idx].sum()
        logging.info(f"- {node_name}: {node_carbon:.2f} g CO2e total")
        logging.info(f"  • Operational: {node_operational:.2f} g CO2e ({(node_operational/node_carbon*100):.1f}%)")
        logging.info(f"  • Embodied: {node_embodied:.2f} g CO2e ({(node_embodied/node_carbon*100):.1f}%)")
    
    # Print top 5 highest carbon cells
    logging.info("\nTop 5 highest carbon cells (total):")
    flat_indices = carbon_matrix.flatten().argsort()[-5:][::-1]
    rows, cols = np.unravel_index(flat_indices, carbon_matrix.shape)
    for i in range(5):
        node_name = node_names[rows[i]]
        timeslot = cols[i]
        cell_carbon = carbon_matrix[rows[i], cols[i]]
        cell_operational = operational_matrix[rows[i], cols[i]]
        cell_embodied = embodied_matrix[rows[i], cols[i]]
        logging.info(f"- Node: {node_name}, Timeslot: {timeslot}, Carbon: {cell_carbon:.2f} g CO2e/hour (Operational: {cell_operational:.4f}, Embodied: {cell_embodied:.6f})")
    
    return 0

if __name__ == "__main__":
    exit(main())
