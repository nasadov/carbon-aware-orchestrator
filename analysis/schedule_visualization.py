import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import glob
import datetime
import yaml
import re
import logging
from collections import defaultdict  # Add defaultdict
from matplotlib.patches import Patch

CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "pkg", "carbon-aware", "nodes.yaml")

def load_nodes_yaml_config(nodes_yaml_path):
    """Loads node configuration directly from a nodes.yaml file with multiple Kubernetes node documents."""
    logging.info(f"Loading nodes configuration from: {nodes_yaml_path}")
    
    if not os.path.exists(nodes_yaml_path):
        logging.warning(f"Nodes YAML file not found: {nodes_yaml_path}")
        return {}
    
    try:
        with open(nodes_yaml_path, 'r') as f:
            # Handle multiple YAML documents (Kubernetes node definitions)
            all_docs = list(yaml.safe_load_all(f))
            
            # Extract node information from Kubernetes node definitions
            nodes = []
            node_cpu_capacities = {}
            
            for doc in all_docs:
                if doc and isinstance(doc, dict) and doc.get('kind') == 'Node':
                    metadata = doc.get('metadata', {})
                    labels = metadata.get('labels', {})
                    hostname = labels.get('kubernetes.io/hostname')
                    
                    if hostname:
                        nodes.append(hostname)
                        
                        # Extract CPU capacity from node status or spec
                        status = doc.get('status', {})
                        capacity = status.get('capacity', {})
                        cpu_capacity = capacity.get('cpu', '1')  # Default to 1 CPU
                        
                        # Convert CPU capacity to float (handle formats like '2' or '2000m')
                        try:
                            if isinstance(cpu_capacity, str):
                                if cpu_capacity.endswith('m'):
                                    cpu_cores = float(cpu_capacity[:-1]) / 1000.0
                                else:
                                    cpu_cores = float(cpu_capacity)
                            else:
                                cpu_cores = float(cpu_capacity)
                            node_cpu_capacities[hostname] = cpu_cores
                            logging.info(f"Node {hostname} has {cpu_cores} CPU cores")
                        except (ValueError, TypeError):
                            logging.warning(f"Could not parse CPU capacity '{cpu_capacity}' for node {hostname}, using default 1.0")
                            node_cpu_capacities[hostname] = 1.0
            
            # Build config structure
            config = {
                'nodes': sorted(nodes),
                'node_cpu_capacities': node_cpu_capacities,
                'total_timeslots': 24  # Default timeslots
            }
            
            logging.info(f"Successfully loaded nodes config. Nodes: {nodes}")
            logging.info(f"Node CPU capacities: {node_cpu_capacities}")
            return config
            
    except Exception as e:
        logging.error(f"Error parsing nodes YAML file {nodes_yaml_path}: {e}")
        return {}

def load_infra_config(config_path_str=CONFIG_FILE_PATH):
    """Loads infrastructure configuration (nodes) from a YAML file."""
    if not config_path_str:
        logging.warning("Path to infra config not provided. Cannot load node names.")
        return {}

    # Determine the absolute path to the config file
    # If config_path_str is not absolute, assume it's relative to the repository root.
    # The script itself is in analysis/, so repo_root is its parent.
    script_dir = os.path.dirname(os.path.abspath(__file__)) # analysis/
    repo_root = os.path.dirname(script_dir) # repo root

    if not os.path.isabs(config_path_str):
        # Correctly join repo_root with config_path_str which might be like 'pkg/carbon-aware/infra-workload-config.yaml'
        config_path_abs = os.path.abspath(os.path.join(repo_root, config_path_str))
    else:
        config_path_abs = os.path.normpath(config_path_str)

    logging.info(f"Attempting to load infra config from: {config_path_abs}")

    # Check if this is a nodes.yaml file (contains Kubernetes node definitions)
    if config_path_abs.endswith('nodes.yaml'):
        return load_nodes_yaml_config(config_path_abs)

    try:
        with open(config_path_abs, 'r') as f:
            config_data = yaml.safe_load(f)
        if not config_data:
            logging.warning(f"Infra config file loaded but is empty: {config_path_abs}")
            return {}
        
        # Try to get nodes.filename from the new structure
        nodes_filename = config_data.get('nodes', {}).get('filename')
        logging.info(f"Read 'nodes.filename' from {config_path_abs}: {nodes_filename}")

        # num_timeslots will not be read from this config, will rely on data/default
        parsed_num_timeslots = None 
        logging.info("'num_timeslots' will be determined by data or default, not from this config file.")

        node_names = []
        node_cpu_capacities = {}  # Store CPU capacities per node
        
        if nodes_filename:
            config_dir = os.path.dirname(config_path_abs)
            # Path to nodes.yaml is relative to the directory of infra-workload-config.yaml
            nodes_yaml_path_abs = os.path.join(config_dir, nodes_filename)
            nodes_yaml_path_abs = os.path.abspath(nodes_yaml_path_abs) # Ensure it's absolute
            logging.info(f"Attempting to load node names from: {nodes_yaml_path_abs}")
            try:
                with open(nodes_yaml_path_abs, 'r') as nf:
                    all_node_documents = yaml.safe_load_all(nf) # Use safe_load_all for multi-document YAML
                    for doc in all_node_documents:
                        if doc and isinstance(doc, dict): # Check if doc is not None and is a dict
                            metadata = doc.get('metadata', {})
                            node_name = metadata.get('name')
                            
                            # Extract CPU capacity from status.allocatable.cpu or status.capacity.cpu
                            status = doc.get('status', {})
                            allocatable = status.get('allocatable', {})
                            capacity = status.get('capacity', {})
                            
                            # Try allocatable first, then capacity if allocatable not available
                            cpu_str = allocatable.get('cpu', capacity.get('cpu', '1'))
                            
                            # Parse CPU value (handle formats like '2' or '2000m')
                            cpu_capacity = 1.0  # Default if parsing fails
                            try:
                                if isinstance(cpu_str, str):
                                    if cpu_str.endswith('m'):  # millicores
                                        cpu_capacity = float(cpu_str[:-1]) / 1000.0
                                    else:
                                        cpu_capacity = float(cpu_str)
                                else:
                                    cpu_capacity = float(cpu_str)  # Try direct conversion if not string
                            except (ValueError, TypeError):
                                logging.warning(f"Could not parse CPU capacity '{cpu_str}' for node {node_name}, using default 1.0")
                            
                            if node_name:
                                node_names.append(node_name)
                                node_cpu_capacities[node_name] = cpu_capacity
                                logging.info(f"Node {node_name} has {cpu_capacity} CPU cores")
                
                if node_names:
                    logging.info(f"Successfully loaded node names: {node_names} from {nodes_yaml_path_abs}")
                    logging.info(f"CPU capacities: {node_cpu_capacities}")
                else:
                    logging.warning(f"No node names extracted from {nodes_yaml_path_abs}. Check file content and format.")
            except FileNotFoundError:
                logging.error(f"Nodes YAML file not found: {nodes_yaml_path_abs}")
            except yaml.YAMLError as e:
                logging.error(f"Error parsing nodes YAML file {nodes_yaml_path_abs}: {e}")
        else:
            logging.warning("Path to nodes.yaml not found in infra_config.")
        
        return {
            'nodes': node_names,
            'total_timeslots': parsed_num_timeslots,
            'node_cpu_capacities': node_cpu_capacities
        }
    except FileNotFoundError:
        logging.error(f"Infra config file not found: {config_path_abs}")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing infra config YAML file {config_path_abs}: {e}")
        return {}
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading infra config {config_path_abs}: {e}")
        return {}

def load_data(csv_path, pod_cpu_requests=None):
    """Loads data from a CSV file into a pandas DataFrame.
    
    Args:
        csv_path (str): Path to the CSV file.
        pod_cpu_requests (dict, optional): A mapping from pod ID to CPU request.
            If provided, this will be used to set the cpuRequest column.
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Ensure cpuRequest is available
        if 'cpuRequest' not in df.columns:
            # Check for alternative column names like 'cpu_request'
            if 'cpu_request' in df.columns:
                logging.info(f"Found 'cpu_request' column, mapping to 'cpuRequest'")
                df['cpuRequest'] = pd.to_numeric(df['cpu_request'], errors='coerce').fillna(0.2)
            # If we have pod CPU requests from workload files, use them
            elif pod_cpu_requests and not df.empty and 'pod_id' in df.columns:
                logging.info(f"Setting cpuRequest based on workload YAML files")
                # Function to lookup CPU request for each pod ID
                def get_cpu_request(pod_id):
                    # Extract base pod ID (e.g., 'm001' from 'm001-duration-3h-deadline-4h')
                    match = re.match(r'([a-zA-Z]+\d+)', pod_id)
                    base_pod_id = match.group(1) if match else pod_id
                    return pod_cpu_requests.get(base_pod_id, 0.2)
                
                # Apply the lookup to each pod
                df['cpuRequest'] = df['pod_id'].apply(get_cpu_request)
                logging.info(f"Added cpuRequest column with data from workload files")
            else:
                logging.warning(f"cpuRequest column not found in {csv_path}, adding default value of 0.2")
                df['cpuRequest'] = 0.2  # Default CPU request if not provided
        else:
            # Make sure it's numeric
            df['cpuRequest'] = pd.to_numeric(df['cpuRequest'], errors='coerce').fillna(0.2)
            
        return df
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file at {csv_path} is empty.")
        return None
    except Exception as e:
        print(f"Error loading CSV {csv_path}: {e}")
        return None

def shorten_pod_label(pod_id_str):
    """Shortens pod labels like 'm001' to '1' for legibility."""
    match = re.fullmatch(r'([a-zA-Z]+)(\d+)', pod_id_str)
    if match:
        numerical_part = match.group(2)
        try:
            return str(int(numerical_part)) # Convert to int and back to str to remove leading zeros
        except ValueError:
            return numerical_part # Should not happen with \d+, but as a fallback
    return pod_id_str # Return original if no match

def plot_individual_pods(df, output_path, title_prefix="Pod Placement", all_node_names_from_config=None, total_timeslots_from_config=None, node_cpu_capacities=None):
    """Generates a matrix plot showing individual pod placements with pod boxes.
    
    Pod heights are drawn proportionally to their CPU requests, and node heights
    are proportionally to their CPU capacities.
    """
    DEFAULT_TIMESLOTS = 24
    
    logging.info(f"Plotting individual pods. Infra config provided: {bool(total_timeslots_from_config)}")
    logging.info(f"Value of 'total_timeslots_from_config': {total_timeslots_from_config} (type: {type(total_timeslots_from_config)})")
    logging.info(f"Node CPU capacities provided: {bool(node_cpu_capacities)}")
    
    # Debug: Check if we have CPU data
    if not df.empty:
        cpu_requests = df['cpuRequest'].unique()
        logging.info(f"Found CPU request values in data: {sorted(cpu_requests)}")
    
    # Get list of nodes to plot
    plot_nodes_data_driven = sorted(df['node_id'].unique())
    
    plot_nodes = all_node_names_from_config if all_node_names_from_config else plot_nodes_data_driven
    
    # Create a dictionary mapping node names to their y-positions based on CPU capacity
    node_y_positions = {}
    node_heights = {}
    
    # Calculate node heights and positions based on CPU capacity
    if node_cpu_capacities:
        # Make node heights proportional to CPU capacity
        # Use a base unit that makes the smallest node readable while maintaining proper proportions
        min_cpu_capacity = min(node_cpu_capacities.get(node, 1.0) for node in plot_nodes)
        base_height_unit = 0.4  # Height for the smallest node (in plot units)
        
        # Calculate heights proportionally to CPU capacity
        for node in plot_nodes:
            if node in node_cpu_capacities:
                cpu = node_cpu_capacities[node]
                # Scale relative to minimum CPU capacity to maintain proportions
                node_heights[node] = (cpu / min_cpu_capacity) * base_height_unit
                logging.info(f"Node {node}: {cpu} CPU cores -> height {node_heights[node]:.2f}")
            else:
                node_heights[node] = base_height_unit  # Default for unknown nodes
        
        # Calculate y-positions (centers of nodes, with spacing)
        current_y = 0
        node_spacing = 0.2  # Fixed spacing between nodes
        for node in plot_nodes:
            node_y_positions[node] = current_y + node_heights[node] / 2
            current_y += node_heights[node] + node_spacing
            logging.info(f"Node {node} positioned at y={node_y_positions[node]:.2f}")
    else:
        # Default equal heights if no CPU capacities provided
        current_y = 0
        for node in plot_nodes:
            node_heights[node] = 1.0
            node_y_positions[node] = current_y + 0.5  # Center of the node
            current_y += 1.3  # Node height (1.0) + spacing (0.3)
    
    # Initialize node_pod_layout at the start
    node_pod_layout = defaultdict(list)  # node -> [(start, end, pod_id, cpu_request)]
    
    # First pass: collect all pods for each node to plan layout
    if not df.empty:
        for _, row in df.iterrows():
            node = row['node_id']
            start_slot = int(row['start_slot'])
            duration = int(row['duration'])
            end_slot = start_slot + duration
            pod_id = row['pod_id']
            cpu_request = float(row['cpuRequest'])
            
            node_pod_layout[node].append((start_slot, end_slot, pod_id, cpu_request))
    
    # Sort pods by start time for each node
    for node in node_pod_layout:
        node_pod_layout[node].sort(key=lambda x: x[0])  # Sort by start_slot
    
    # Force plot time slots to be exactly 24 hours (override any dynamic calculation)
    plot_time_slots = 24
    logging.info(f"Fixed plot_time_slots to 24 hours (was going to be dynamic)")
    
    # Create figure with consistent size parameters across all algorithms
    standard_width = 20   # Standard width for all plots
    standard_height = 12  # Standard height for all plots
    
    fig, ax = plt.subplots(figsize=(standard_width, standard_height))
    
    # Add a bit more margin on both sides of the time axis
    ax.set_xlim(-0.7, plot_time_slots - 0.3)
    
    # Calculate tight ylim based on actual node boundaries
    if plot_nodes and node_y_positions and node_heights:
        # Top edge of the topmost node (smallest y-coordinate)
        top_edge = min(node_y_positions[node] - node_heights[node]/2 for node in plot_nodes)
        # Bottom edge of the bottommost node (largest y-coordinate)  
        bottom_edge = max(node_y_positions[node] + node_heights[node]/2 for node in plot_nodes)
        
        # Add minimal margins
        margin = 0.1
        min_y = top_edge - margin    # Visual top (smallest y-value)
        max_y = bottom_edge + margin # Visual bottom (largest y-value)
    else:
        # Fallback for empty data
        min_y = -1.0
        max_y = 1.0
    
    ax.set_ylim(max_y, min_y)  # Reverse y-axis to put node-0 at top
    
    # Draw node capacity indicators (light gray rectangles)
    for node in plot_nodes:
        node_y_center = node_y_positions[node]
        node_height = node_heights[node]
        
        # Draw node background indicating capacity
        for slot in range(plot_time_slots):
            node_rect = plt.Rectangle(
                (slot - 0.5, node_y_center - node_height / 2),  # bottom left (x, y)
                1.0,  # width (1 time slot)
                node_height,  # height proportional to CPU capacity
                facecolor='lightgray',
                edgecolor='gray',
                alpha=0.3,
                zorder=1  # Put this behind pods
            )
            ax.add_patch(node_rect)
    
    # Create a color map for pods
    unique_pods = sorted(list(set(pod_id for node_pods in node_pod_layout.values() for _, _, pod_id, _ in node_pods)))
    num_pods = len(unique_pods)
    
    # Use HSV colormap for better color distribution
    pod_to_color = {}
    pod_to_hatch = {}
    
    # Create color map with strategic hatching patterns for adjacent similar colors
    hsv_colors = plt.cm.hsv(np.linspace(0, 1, num_pods))
    for i, pod_id in enumerate(unique_pods):
        pod_to_color[pod_id] = hsv_colors[i]
    
    # Apply hatching patterns strategically to help distinguish adjacent similar colors
    # Define different hatch patterns to cycle through
    hatch_patterns = ['', '///', '\\\\\\', '|||', '---', '+++', 'xxx', '...']
    
    # Check each pod against its neighbors in the sorted sequence and apply hatching
    # when colors are too similar to adjacent pods
    for i, pod_id in enumerate(unique_pods):
        current_color = hsv_colors[i]
        needs_hatching = False
        
        # Check similarity with adjacent pods (previous and next in sequence)
        adjacent_indices = []
        if i > 0:  # Check previous pod
            adjacent_indices.append(i - 1)
        if i < len(unique_pods) - 1:  # Check next pod
            adjacent_indices.append(i + 1)
        
        # Also check a few more neighbors to handle clusters of similar colors
        for offset in [-2, -1, 1, 2]:
            neighbor_idx = i + offset
            if 0 <= neighbor_idx < len(unique_pods) and neighbor_idx != i:
                adjacent_indices.append(neighbor_idx)
        
        # Remove duplicates and sort
        adjacent_indices = sorted(set(adjacent_indices))
        
        # Check if current color is too similar to any adjacent colors
        for adj_idx in adjacent_indices:
            adj_color = hsv_colors[adj_idx]
            # Calculate perceptual color difference in RGB space
            color_diff = np.sqrt(np.sum((current_color[:3] - adj_color[:3])**2))
            if color_diff < 0.2:  # Threshold for "similar" colors (increased from 0.15)
                needs_hatching = True
                break
        
        # Apply hatching pattern if needed
        if needs_hatching:
            # Use a pattern based on the pod's position to ensure variety
            pattern_idx = i % len(hatch_patterns)
            if pattern_idx == 0:  # Skip empty pattern for pods that need hatching
                pattern_idx = 1
            pod_to_hatch[pod_id] = hatch_patterns[pattern_idx]
    
    # Second pass: draw pods with better space distribution
    if not df.empty:
        for _, row in df.iterrows():
            if row['node_id'] not in plot_nodes:
                continue
                
            node = row['node_id']
            node_y_center = node_y_positions[node]
            node_height = node_heights[node]
            start_slot = int(row['start_slot'])
            duration = int(row['duration'])
            pod_id = row['pod_id']
            cpu_request = float(row['cpuRequest'])
            
            # Calculate pod height proportional to its CPU request relative to node capacity
            if node_cpu_capacities and node in node_cpu_capacities:
                node_cpu_capacity = node_cpu_capacities[node]
                # Pod height should be proportional to its CPU request relative to the node's capacity
                # This ensures proper visual representation of resource utilization
                pod_height_ratio = cpu_request / node_cpu_capacity
                visible_pod_height = pod_height_ratio * node_height
                logging.debug(f"Pod {pod_id} on {node}: {cpu_request}/{node_cpu_capacity} CPU = {pod_height_ratio:.2f} ratio -> height {visible_pod_height:.3f}")
            else:
                # Fallback if no capacity info available
                visible_pod_height = cpu_request * 0.1  # Use simple scaling
            
            # Ensure minimum visibility for very small pods
            min_visible_height = 0.02
            visible_pod_height = max(visible_pod_height, min_visible_height)
            
            # Find overlapping pods at this time
            overlapping_pods = []
            for s, e, p, c in node_pod_layout[node]:
                if not (e <= start_slot or s >= start_slot + duration):  # Check for overlap
                    overlapping_pods.append((s, e, p, c))
            
            # Sort overlapping pods by start time
            overlapping_pods.sort(key=lambda x: x[0])
            
            # Find position in the stack of overlapping pods
            try:
                pod_index = next(i for i, (s, e, p, c) in enumerate(overlapping_pods) if p == pod_id)
            except StopIteration:
                pod_index = 0  # Fallback if pod not found in overlapping pods
            
            # Calculate vertical position for this pod
            total_overlapping = len(overlapping_pods)
            if total_overlapping > 1:
                # Distribute overlapping pods within the full node height
                available_height = node_height  # Use 100% of node height for pods
                vertical_spacing = available_height / total_overlapping
                pod_y_offset = (pod_index - (total_overlapping - 1) / 2) * vertical_spacing
            else:
                pod_y_offset = 0
            
            # Calculate final y position
            pod_y_bottom = node_y_center + pod_y_offset - visible_pod_height / 2
            
            # Check for capacity violation
            capacity_violation = False
            if node_cpu_capacities:
                overlapping_cpu_sum = sum(c for s, e, p, c in overlapping_pods if s <= start_slot < e)
                capacity_violation = overlapping_cpu_sum > node_cpu_capacities[node]
            
            # Draw the pod rectangle
            rect = plt.Rectangle(
                (start_slot - 0.5, pod_y_bottom),  # bottom left (x, y)
                duration,  # width (number of time slots)
                visible_pod_height,  # height with minimum visibility
                facecolor=pod_to_color.get(pod_id, 'gray'),
                edgecolor='red' if capacity_violation else 'black',
                alpha=0.8,
                linewidth=1.2 if capacity_violation else 0.8,  # Thicker red border for capacity violations
                hatch=pod_to_hatch.get(pod_id, ''),  # Add hatching for similar colored pods
                zorder=2  # Above node backgrounds
            )
            ax.add_patch(rect)
    
    # Set x-ticks for time slots - always show all 24 hours (1-24)
    x_ticks = range(24)
    x_labels = range(1, 25)  # Labels from 1 to 24
    plt.xticks(x_ticks, x_labels)
    
    # Set y-ticks to show node names
    plt.yticks([node_y_positions[node] for node in plot_nodes], plot_nodes)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add descriptive title with pod and node counts
    num_pods = len(df) if not df.empty else 0
    num_nodes = len(plot_nodes)
    plt.title(f"{title_prefix} Schedule: {num_pods} Pods placed across {num_nodes} Nodes")
    plt.xlabel("Time Slot (Hours)")
    plt.ylabel("Node")
    
    # Add legend for pods
    if not df.empty:
        legend_elements = []
        for pod_id in sorted(pod_to_color.keys()):
            color = pod_to_color[pod_id]
            hatch = pod_to_hatch.get(pod_id, '')  # Use .get() to safely access hatch
            legend_elements.append(Patch(facecolor=color, 
                                      label=pod_id,
                                      hatch=hatch,
                                      alpha=0.8))
        
        # Place legend outside the plot on the right
        plt.legend(handles=legend_elements,
                  title="Pods",
                  bbox_to_anchor=(1.05, 1),
                  loc='upper left',
                  borderaxespad=0.,
                  ncol=max(1, len(legend_elements) // 30))  # Split into columns if many pods
    
    # Adjust layout to prevent legend from being cut off
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Individual pod placement plot saved to {output_path}")

def plot_density_heatmap(df, output_path, title_prefix="Pod Density", all_node_names_from_config=None, total_timeslots_from_config=None):
    """Generates a heatmap showing pod density per node and time slot."""
    DEFAULT_TIMESLOTS = 24

    logging.info(f"Plotting density heatmap. Infra config provided: {bool(total_timeslots_from_config)}")
    logging.info(f"Value of 'total_timeslots_from_config': {total_timeslots_from_config} (type: {type(total_timeslots_from_config)})")

    df_for_plotting = None
    if df is not None and not df.empty:
        df_for_plotting = df.copy()
        try:
            if 'start_slot' in df_for_plotting.columns:
                df_for_plotting['start_slot'] = pd.to_numeric(df_for_plotting['start_slot'], errors='raise').astype(int)
            if 'duration' in df_for_plotting.columns:
                df_for_plotting['duration'] = pd.to_numeric(df_for_plotting['duration'], errors='raise').astype(int)
            if 'node_id' in df_for_plotting.columns:
                df_for_plotting['node_id'] = df_for_plotting['node_id'].astype(str)
            else:
                print(f"Warning: 'node_id' column missing in DataFrame for '{title_prefix}' density heatmap. Plot may be incomplete.")
                df_for_plotting = pd.DataFrame()
        except Exception as e:
            print(f"Warning: Error converting essential columns for '{title_prefix}' density heatmap: {e}. Plot may be affected.")
            df_for_plotting = pd.DataFrame()
    else:
        print(f"No data or empty DataFrame for density heatmap: {title_prefix}.")
        df_for_plotting = pd.DataFrame()

    all_nodes_from_config = all_node_names_from_config if all_node_names_from_config else []
    logging.info(f"Nodes from config for plot_density_heatmap: {all_nodes_from_config}")

    plot_nodes_data_driven = []
    if not df_for_plotting.empty and 'node_id' in df_for_plotting.columns:
        plot_nodes_data_driven = sorted(df_for_plotting['node_id'].unique())
    
    plot_nodes = all_node_names_from_config if all_node_names_from_config else plot_nodes_data_driven

    if not plot_nodes:
        print(f"Cannot create plot '{title_prefix}' for density heatmap: No nodes found in data or config.")
        return

    if total_timeslots_from_config is not None:
        try:
            plot_time_slots = int(total_timeslots_from_config)
            logging.info(f"Source of plot_time_slots for heatmap: infra config. Value: {plot_time_slots}")
        except (ValueError, TypeError) as e:
            logging.warning(f"Could not convert total_timeslots_from_config ('{total_timeslots_from_config}') to int for heatmap: {e}. Falling back.")
            plot_time_slots = DEFAULT_TIMESLOTS
    elif not df_for_plotting.empty and 'start_slot' in df_for_plotting.columns and 'duration' in df_for_plotting.columns:
        max_data_slot = 0
        try:
            max_data_slot = (df_for_plotting['start_slot'] + df_for_plotting['duration']).max()
            if pd.isna(max_data_slot): max_data_slot = 0
        except TypeError:
             max_data_slot = 0
             print(f"Warning: Could not calculate max_data_slot for density heatmap '{title_prefix}' due to non-numeric types.")
        plot_time_slots = max(int(max_data_slot), DEFAULT_TIMESLOTS)
    else:
        plot_time_slots = DEFAULT_TIMESLOTS
    
    if plot_time_slots <= 0:
        print(f"Warning: plot_time_slots is {plot_time_slots} for '{title_prefix}' density heatmap. Using 1 as minimum.")
        plot_time_slots = 1

    logging.info(f"Final plot_time_slots for plot_density_heatmap: {plot_time_slots}")

    density_matrix = pd.DataFrame(0, index=plot_nodes, columns=range(plot_time_slots))

    if not df_for_plotting.empty and 'start_slot' in df_for_plotting.columns and \
       'duration' in df_for_plotting.columns and 'node_id' in df_for_plotting.columns:
        for _, row in df_for_plotting.iterrows():
            if row['node_id'] not in plot_nodes:
                continue
            for t_slot in range(row['start_slot'], row['start_slot'] + row['duration']):
                if 0 <= t_slot < plot_time_slots:
                    density_matrix.loc[row['node_id'], t_slot] += 1
    elif df_for_plotting.empty:
         print(f"No data to populate density heatmap '{title_prefix}'. Plot will be an empty grid.")
    else:
        print(f"Warning: DataFrame for '{title_prefix}' density heatmap is missing required columns (start_slot, duration, node_id) for density calculation. Plot will be an empty grid.")
    
    fig, ax = plt.subplots(figsize=(max(10, plot_time_slots * 0.4), max(6, len(plot_nodes) * 0.5)))
    heatmap_cmap = plt.colormaps['viridis']
    heatmap = ax.imshow(density_matrix, cmap=heatmap_cmap, aspect='auto', interpolation='nearest')
    
    for r_idx, node_label in enumerate(plot_nodes):
        for c_idx in range(plot_time_slots):
            val = density_matrix.loc[node_label, c_idx]
            if val > 0:
                cell_color_val = heatmap_cmap(density_matrix.loc[node_label, c_idx] / max(1, density_matrix.values.max()))[0:3]
                luminance = 0.299*cell_color_val[0] + 0.587*cell_color_val[1] + 0.114*cell_color_val[2]
                text_color = 'white' if luminance < 0.5 else 'black'
                ax.text(c_idx, r_idx, str(val), va='center', ha='center', color=text_color, fontsize=8)

    ax.set_xticks(np.arange(plot_time_slots))
    ax.set_yticks(np.arange(len(plot_nodes)))
    ax.set_xticklabels(np.arange(1, plot_time_slots + 1))  # Labels from 1 to plot_time_slots
    ax.set_yticklabels(plot_nodes)
    ax.set_xticks(np.arange(plot_time_slots + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(plot_nodes) + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.5)
    ax.tick_params(which='major', bottom=False, left=False)

    plt.xlabel("Time Slot")
    plt.ylabel("Node ID")
    # Add descriptive title with pod and node counts for density heatmap
    num_pods = len(df_for_plotting) if not df_for_plotting.empty else 0
    num_nodes = len(plot_nodes)
    plt.title(f"{title_prefix} - Density Heatmap: {num_pods} Pods across {num_nodes} Nodes")
    plt.colorbar(heatmap, label="Number of Pods")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Density heatmap saved to {output_path}")

def extract_cpu_requests_from_workloads(workloads_dir):
    """Extracts CPU requests for pods from workload YAML files.
    
    Args:
        workloads_dir (str): Path to the directory containing workload YAML files.
        
    Returns:
        dict: A mapping from pod ID (e.g. 'm001') to CPU request in cores.
    """
    import os
    import yaml
    import re
    
    logging.info(f"Extracting CPU requests from workloads in: {workloads_dir}")
    
    if not os.path.exists(workloads_dir):
        logging.warning(f"Workloads directory not found: {workloads_dir}")
        return {}
    
    pod_cpu_requests = {}
    yaml_files = [f for f in os.listdir(workloads_dir) if re.match(r'timeslot_\d+\.yaml', f)]
    logging.info(f"Found {len(yaml_files)} workload YAML files")
    
    # Look for files matching the pattern timeslot_*.yaml
    for filename in sorted(yaml_files):
        filepath = os.path.join(workloads_dir, filename)
        logging.info(f"Processing workload file: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                # A single file may contain multiple YAML documents separated by '---'
                all_yamls = list(yaml.safe_load_all(f))
                logging.info(f"Found {len(all_yamls)} YAML documents in {filename}")
                
                for doc_idx, doc in enumerate(all_yamls):
                    if (doc and isinstance(doc, dict) and 
                        doc.get('kind') == 'Deployment' and 
                        'metadata' in doc and 'name' in doc['metadata']):
                        
                        # Extract the pod ID (e.g., 'm001' from 'm001-duration-3h-deadline-4h')
                        pod_name = doc['metadata']['name']
                        pod_id_match = re.match(r'([a-zA-Z]+\d+)', pod_name)
                        
                        if pod_id_match:
                            pod_id = pod_id_match.group(1)
                            logging.info(f"Processing pod {pod_id} from {pod_name} (doc #{doc_idx} in {filename})")
                            
                            # Extract CPU request
                            try:
                                containers = doc['spec']['template']['spec']['containers']
                                for container_idx, container in enumerate(containers):
                                    if 'resources' in container and 'requests' in container['resources']:
                                        cpu_request = container['resources']['requests'].get('cpu', '0')
                                        logging.info(f"Raw CPU request for pod {pod_id}, container #{container_idx}: {cpu_request}")
                                        
                                        # Convert to cores (handle formats like '1000m', '0.5', '2')
                                        if isinstance(cpu_request, str):
                                            if cpu_request.endswith('m'):
                                                cpu_cores = float(cpu_request[:-1]) / 1000.0
                                            else:
                                                cpu_cores = float(cpu_request)
                                        else:
                                            cpu_cores = float(cpu_request)
                                            
                                        # Store in the dictionary
                                        if pod_id not in pod_cpu_requests:
                                            pod_cpu_requests[pod_id] = cpu_cores
                                            logging.info(f"Extracted CPU request for pod {pod_id}: {cpu_cores} cores")
                            except (KeyError, TypeError, ValueError) as e:
                                logging.warning(f"Failed to extract CPU request for pod {pod_name}: {e}")
                        else:
                            logging.warning(f"Could not extract pod ID from name: {pod_name}")
        except Exception as e:
            logging.error(f"Error processing workload file {filepath}: {e}")
    
    if pod_cpu_requests:        
        logging.info(f"Extracted CPU requests for {len(pod_cpu_requests)} pods: {pod_cpu_requests}")
    else:
        logging.warning("No CPU requests extracted from workload files")
        
    return pod_cpu_requests

def main():
    parser = argparse.ArgumentParser(description="Visualize pod placement schedules from CSV data.")
    parser.add_argument("input_csvs", nargs='+', help="Path(s) to the input CSV file(s) containing pod placement data. Can be a glob pattern.")
    parser.add_argument("-o", "--output_dir_base", default=os.path.join(os.path.dirname(__file__), "..", "figures"), help="Base directory to save the generated plots (default: ../figures relative to script). A subdirectory will be created here.")
    parser.add_argument("-m", "--mode", choices=["individual", "density", "all"], default="all", help="Type of visualization to generate: 'individual' pods, 'density' heatmap, or 'all' (default: all).")
    parser.add_argument("-n", "--name", default=None, help="Optional name for the run/comparison. This will be used as the subdirectory name under output_dir_base and in plot titles/filenames.")
    parser.add_argument("--config_file", default=CONFIG_FILE_PATH, help=f"Path to the nodes.yaml file (default: {CONFIG_FILE_PATH}).")
    parser.add_argument("--workloads_dir", default=None, help="Path to the directory containing workload YAML files with CPU request information.")
    
    args = parser.parse_args()

    infra_config = load_infra_config(args.config_file)
    all_node_names_from_config = infra_config.get('nodes', [])
    total_timeslots_from_config = infra_config.get('total_timeslots')
    
    # Extract CPU requests from workload YAML files if a directory is provided
    pod_cpu_requests = {}
    if args.workloads_dir:
        logging.info(f"Extracting CPU requests from workload files in: {args.workloads_dir}")
        pod_cpu_requests = extract_cpu_requests_from_workloads(args.workloads_dir)
        logging.info(f"Found CPU requests for {len(pod_cpu_requests)} pods: {pod_cpu_requests}")
    else:
        logging.info("No workloads directory provided. Will use default CPU request values.")

    # Add a timestamp to the output filenames to avoid overwriting
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.name:
        plot_output_subdir_name = args.name.replace(" ", "_")
    elif len(args.input_csvs) == 1 and os.path.isfile(args.input_csvs[0]):
        csv_filename = os.path.basename(args.input_csvs[0])
        
        # Detect algorithm type from CSV filename and create appropriate directory structure
        if "global_optimal" in csv_filename.lower() or "global-optimal" in csv_filename.lower():
            algorithm_base_dir = "Global-Optimal"
            algorithm_subdir = f"Global-Optimal_{timestamp}"
            plot_output_subdir_name = os.path.join(algorithm_base_dir, algorithm_subdir)
        elif "heuristic" in csv_filename.lower():
            algorithm_base_dir = "Heuristic"
            algorithm_subdir = f"Heuristic_{timestamp}"
            plot_output_subdir_name = os.path.join(algorithm_base_dir, algorithm_subdir)
        elif "vanilla" in csv_filename.lower():
            algorithm_base_dir = "Vanilla"
            algorithm_subdir = f"Vanilla_{timestamp}"
            plot_output_subdir_name = os.path.join(algorithm_base_dir, algorithm_subdir)
        else:
            # Fallback to original logic for unknown algorithm types
            parent_dir_name = os.path.basename(os.path.dirname(args.input_csvs[0]))
            if "placements" in parent_dir_name.lower():
                plot_output_subdir_name = os.path.splitext(os.path.basename(args.input_csvs[0]))[0]
            else:
                plot_output_subdir_name = parent_dir_name if parent_dir_name else "visualization_run"
    else:
        plot_output_subdir_name = f"visualization_{timestamp}"

    final_plot_output_dir = os.path.join(args.output_dir_base, plot_output_subdir_name)
    os.makedirs(final_plot_output_dir, exist_ok=True)

    # Helper: if the user passed a directory, try to find the expected placement CSV inside it
    def _find_csv_in_dir(dir_path: str):
        if not os.path.isdir(dir_path):
            return None
        # Preferred filenames by algorithm
        candidates = [
            # Global-optimal
            'global_optimal_placements_session.csv',
            # Heuristic
            'heuristic_prop_placements_session.csv',
            'heuristic_uniform_placements_session.csv',
            'heuristic_placements_session.csv',
            # Vanilla
            'vanilla_placement_session_presence.csv',
            'vanilla_placement_session_bind.csv',
            'vanilla_placement_session_fixed.csv',
            'vanilla_placement_session.csv',
            'vanilla_placements.csv',
        ]
        for name in candidates:
            p = os.path.join(dir_path, name)
            if os.path.isfile(p):
                return p
        # Fallback: any placements-like CSV in the directory
        for p in glob.glob(os.path.join(dir_path, "*placement*.csv")):
            if os.path.isfile(p):
                return p
        for p in glob.glob(os.path.join(dir_path, "*placements*.csv")):
            if os.path.isfile(p):
                return p
        return None

    # Build the list of CSV files from inputs (files, globs, or directories)
    all_csv_files = []
    raw_inputs = []
    for pattern in args.input_csvs:
        matches = glob.glob(pattern)
        # If no glob matches, still consider the raw value (could be a directory)
        if not matches:
            raw_inputs.append(pattern)
        else:
            raw_inputs.extend(matches)

    for item in raw_inputs:
        if os.path.isdir(item):
            csv_inside = _find_csv_in_dir(item)
            if csv_inside:
                all_csv_files.append(csv_inside)
            else:
                print(f"Warning: no placement CSV found inside directory: {item}")
        elif os.path.isfile(item):
            all_csv_files.append(item)
        else:
            # Not a file or directory; skip
            pass

    if not all_csv_files:
        print(f"No CSV files found from inputs: {args.input_csvs}")
        return

    if len(all_csv_files) > 1 and args.name:
        print(f"Combining {len(all_csv_files)} CSV files for comparison: {args.name}")
        loaded_dfs = [load_data(f, pod_cpu_requests) for f in all_csv_files]
        valid_dfs = [df for df in loaded_dfs if df is not None and not df.empty]

        if not valid_dfs:
            print(f"No valid data loaded from CSV files. Exiting.")
            return
        
        combined_df = pd.concat(valid_dfs, ignore_index=True)
        
        base_filename = os.path.join(final_plot_output_dir, args.name.replace(" ", "_"))
        title_prefix = args.name

        if args.mode in ["individual", "all"]:
            # Pass node CPU capacities to the function
            node_cpu_capacities = infra_config.get('node_cpu_capacities', {})
            plot_individual_pods(combined_df, f"{base_filename}_individual.png", title_prefix, all_node_names_from_config, total_timeslots_from_config, node_cpu_capacities)
        if args.mode in ["density", "all"]:
            plot_density_heatmap(combined_df, f"{base_filename}_density.png", title_prefix, all_node_names_from_config, total_timeslots_from_config)
    else:
        for csv_file in all_csv_files:
            df = load_data(csv_file, pod_cpu_requests)
            if df is None or df.empty:
                print(f"No valid data loaded from CSV: {csv_file}. Skipping.")
                continue

            file_basename = os.path.splitext(os.path.basename(csv_file))[0]
            
            current_plot_name_prefix = file_basename
            title_prefix_for_plot = args.name if args.name else file_basename

            output_base = os.path.join(final_plot_output_dir, current_plot_name_prefix.replace(" ", "_"))

            print(f"Processing: {csv_file} into {final_plot_output_dir} with plot name: {current_plot_name_prefix}")

            if args.mode in ["individual", "all"]:
                # Pass node CPU capacities to the function
                node_cpu_capacities = infra_config.get('node_cpu_capacities', {})
                plot_individual_pods(df, f"{output_base}_individual.png", title_prefix_for_plot, all_node_names_from_config, total_timeslots_from_config, node_cpu_capacities)
            if args.mode in ["density", "all"]:
                plot_density_heatmap(df, f"{output_base}_density.png", title_prefix_for_plot, all_node_names_from_config, total_timeslots_from_config)

if __name__ == "__main__":
    # Configure logging to show INFO level messages
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(levelname)s - %(message)s',
                      datefmt='%Y-%m-%d %H:%M:%S')
    main()
