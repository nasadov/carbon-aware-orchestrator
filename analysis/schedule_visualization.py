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

CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "pkg", "carbon-aware", "infra-workload-config.yaml")

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
    if df is not None and not df.empty and 'cpuRequest' in df.columns:
        cpu_values = df['cpuRequest'].unique()
        logging.info(f"Found CPU request values in data: {sorted(cpu_values)}")
    elif df is not None and not df.empty and 'cpu_request' in df.columns:
        cpu_values = df['cpu_request'].unique()
        logging.info(f"Found cpu_request values in data: {sorted(cpu_values)}")
    else:
        logging.warning("No CPU request data found in DataFrame")

    # Data type conversion should happen early if df is not None
    if df is not None and not df.empty:
        try:
            if 'start_slot' in df.columns:
                df['start_slot'] = pd.to_numeric(df['start_slot'], errors='raise').astype(int)
            if 'duration' in df.columns:
                df['duration'] = pd.to_numeric(df['duration'], errors='raise').astype(int)
            if 'pod_id' in df.columns:
                df['pod_id'] = df['pod_id'].astype(str)
            if 'node_id' in df.columns:
                df['node_id'] = df['node_id'].astype(str)
            if 'cpuRequest' in df.columns:
                df['cpuRequest'] = pd.to_numeric(df['cpuRequest'], errors='coerce').fillna(0.2)
            elif 'cpu_request' in df.columns:
                # Handle different column name convention
                df['cpuRequest'] = pd.to_numeric(df['cpu_request'], errors='coerce').fillna(0.2)
            else:
                # Default CPU request if not provided
                logging.info(f"cpuRequest column not found in DataFrame. Using default of 0.2.")
                df['cpuRequest'] = 0.2
        except Exception as e:
            print(f"Warning: Error converting essential columns to required types for '{title_prefix}': {e}. Plot may be affected.")

    all_nodes_from_config = all_node_names_from_config if all_node_names_from_config else []
    logging.info(f"Nodes from config for plot_individual_pods: {all_nodes_from_config}")

    if df is None or df.empty:
        print(f"No data or empty DataFrame for individual pods: {title_prefix}.")
        plot_nodes = all_nodes_from_config if all_nodes_from_config else []
        plot_time_slots = total_timeslots_from_config if total_timeslots_from_config is not None else DEFAULT_TIMESLOTS
        if not plot_nodes:
            print(f"Cannot create empty plot '{title_prefix}': No node data from CSV or config.")
            return
            
        # Create node_y_positions and node_heights for empty visualization
        node_y_positions = {}
        node_heights = {}
        
        # If we have CPU capacities, use them to determine node heights
        if node_cpu_capacities:
            total_cpu = sum(node_cpu_capacities.get(node, 1.0) for node in plot_nodes)
            current_y = 0
            for node in plot_nodes:
                cpu_capacity = node_cpu_capacities.get(node, 1.0)
                node_height = cpu_capacity / total_cpu * len(plot_nodes)
                node_heights[node] = node_height
                node_y_positions[node] = current_y + node_height / 2
                current_y += node_height
        else:
            for i, node in enumerate(plot_nodes):
                node_y_positions[node] = i
                node_heights[node] = 1.0
        
        # Determine appropriate figure height
        total_height = max(8, sum(node_heights.values()) * 1.2)
        
        fig, ax = plt.subplots(figsize=(max(12, plot_time_slots * 0.5), total_height))
        ax.set_xlim(-0.5, plot_time_slots - 0.5)
        
        # Set ylim based on node positions
        min_y = min(list(node_y_positions.values()) + [0]) - max(node_heights.values(), default=1) / 2 - 0.5
        max_y = max(list(node_y_positions.values()) + [0]) + max(node_heights.values(), default=1) / 2 + 0.5
        ax.set_ylim(min_y, max_y)
        
        # Draw empty node capacity indicators
        for node in plot_nodes:
            node_y_center = node_y_positions[node]
            node_height = node_heights[node]
            
            for slot in range(plot_time_slots):
                node_rect = plt.Rectangle(
                    (slot - 0.5, node_y_center - node_height / 2),
                    1.0,
                    node_height,
                    facecolor='lightgray',
                    edgecolor='gray',
                    alpha=0.3
                )
                ax.add_patch(node_rect)
        
        # Set ticks
        ax.set_xticks(np.arange(plot_time_slots))
        ax.set_xticklabels(np.arange(plot_time_slots))
        
        # Set custom y-ticks at node centers
        y_ticks = [node_y_positions[node] for node in plot_nodes]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{node} ({node_cpu_capacities.get(node, 1.0)}CPU)" if node_cpu_capacities else node for node in plot_nodes])
        
        # Create grid lines
        ax.set_xticks(np.arange(plot_time_slots + 1) - 0.5, minor=True)
        
        node_boundaries = []
        for node in plot_nodes:
            y_center = node_y_positions[node]
            height = node_heights[node]
            node_boundaries.append(y_center - height/2)
            node_boundaries.append(y_center + height/2)
        
        ax.set_yticks(sorted(set(node_boundaries)), minor=True)
        ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.5)
        
        plt.xlabel("Time Slot")
        plt.ylabel("Node ID")
        plt.title(f"{title_prefix} - Individual Pods (CPU Proportional, No Data)")
        fig.subplots_adjust(bottom=0.2)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Empty individual pod placement plot with CPU proportional layout saved to {output_path}")
        return

    plot_nodes_data_driven = []
    if df is not None and not df.empty and 'node_id' in df.columns:
        plot_nodes_data_driven = sorted(df['node_id'].unique())
    
    plot_nodes = all_node_names_from_config if all_node_names_from_config else plot_nodes_data_driven
    
    # Create a dictionary mapping node names to their y-positions based on CPU capacity
    node_y_positions = {}
    node_heights = {}
    
    # Default CPU capacity if not provided
    default_cpu = 1.0
    
    # If node_cpu_capacities is provided, calculate y-positions based on CPU
    if node_cpu_capacities:
        # Get CPU capacities for all nodes
        capacities = [node_cpu_capacities.get(node, default_cpu) for node in plot_nodes]
        
        # Apply a square root scaling to prevent extreme differences while preserving relative sizes better
        # This makes small capacity nodes still visible while maintaining proportionality better than log
        sqrt_capacities = [max(1.0, np.sqrt(cap)) for cap in capacities]
        
        # Calculate total height based on sqrt-scaled capacities
        total_sqrt_cpu = sum(sqrt_capacities)
        
        # Initialize the current y position at 0 (top of the plot)
        current_y = 0
        
        # Minimum height for a node (ensures small nodes are still visible)
        min_height = 1.0  # Increased from 0.5 for better visibility
        
        # Add spacing between nodes (each node has some margin)
        node_spacing = 0.3  # Spacing between nodes
        
        # Assign positions and heights to each node based on their CPU capacity
        for i, node in enumerate(plot_nodes):
            cpu_capacity = node_cpu_capacities.get(node, default_cpu)
            sqrt_capacity = sqrt_capacities[i]
            
            # Node height is proportional to its sqrt-scaled CPU capacity
            # Multiply by len(plot_nodes) to maintain a reasonable total plot height
            # Scale factor increased from 1.5 to 2.0 for better visibility
            node_height = max(min_height, sqrt_capacity / total_sqrt_cpu * len(plot_nodes) * 2.0)
            node_heights[node] = node_height
            
            # Store the position (center) of the node
            node_y_positions[node] = current_y + node_height / 2
            
            # Move to the next position with added spacing
            current_y += node_height + node_spacing
            
            logging.info(f"Node {node} with {cpu_capacity} CPU has scaled height {node_height} at position {node_y_positions[node]}")
    else:
        # If no CPU capacities, use equal heights for all nodes with spacing
        node_spacing = 0.3
        for i, node in enumerate(plot_nodes):
            node_y_positions[node] = i * (1 + node_spacing)
            node_heights[node] = 1.0
    
    if total_timeslots_from_config is not None:
        try:
            plot_time_slots = int(total_timeslots_from_config)
            logging.info(f"Source of plot_time_slots: infra config. Value: {plot_time_slots}")
        except (ValueError, TypeError) as e:
            logging.warning(f"Could not convert total_timeslots_from_config ('{total_timeslots_from_config}') to int: {e}. Falling back.")
            plot_time_slots = DEFAULT_TIMESLOTS
    elif df is not None and not df.empty and 'start_slot' in df.columns and 'duration' in df.columns:
        max_data_slot = 0
        try:
            max_data_slot = (df['start_slot'] + df['duration']).max()
            if pd.isna(max_data_slot): max_data_slot = 0
        except TypeError:
            max_data_slot = 0
            print(f"Warning: Could not calculate max_data_slot due to non-numeric types for '{title_prefix}'.")
        plot_time_slots = max(int(max_data_slot), DEFAULT_TIMESLOTS)
    else:
        plot_time_slots = DEFAULT_TIMESLOTS

    if not plot_nodes:
        print(f"Cannot create plot '{title_prefix}': No nodes found in data or config.")
        return

    if plot_time_slots <= 0:
        print(f"Warning: plot_time_slots is {plot_time_slots} for '{title_prefix}'. Using 1 as minimum.")
        plot_time_slots = 1

    logging.info(f"Final plot_time_slots for plot_individual_pods: {plot_time_slots}")

    text_matrix = pd.DataFrame(index=plot_nodes, columns=range(plot_time_slots), data="")
    if not df.empty:
        for _, row in df.iterrows():
            if row['node_id'] not in plot_nodes:
                continue
            node_idx = plot_nodes.index(row['node_id'])
            for t_slot in range(row['start_slot'], row['start_slot'] + row['duration']):
                if 0 <= t_slot < plot_time_slots:
                    current_text = text_matrix.iloc[node_idx, t_slot]
                    pod_short_id = row['pod_id']
                    if current_text == "":
                        text_matrix.iloc[node_idx, t_slot] = pod_short_id
                    else:
                        if pod_short_id not in current_text.split(','):
                            text_matrix.iloc[node_idx, t_slot] += "," + pod_short_id

    # Determine plot height based on the number of nodes and their relative CPU capacities
    # For CPU-proportional visualization, we want the overall height to be reasonable
    # Increased scaling factor from 1.2 to 1.5 for better readability
    total_height = max(10, sum(node_heights.values()) * 1.5 + (len(plot_nodes) - 1) * 0.3)
    
    # Calculate width based on time slots with more space per slot for better legibility
    # Increase base width to accommodate legend on the right side
    plot_width = max(16, plot_time_slots * 0.7)  # Increased from 14 and 0.6
    
    # Create figure with improved size parameters
    fig, ax = plt.subplots(figsize=(plot_width, total_height))
    
    # Add a bit more margin on both sides of the time axis
    ax.set_xlim(-0.7, plot_time_slots - 0.3)
    
    # Calculate tight ylim based on actual node boundaries
    # Find the actual top and bottom edges of all nodes
    if plot_nodes and node_y_positions and node_heights:
        # Top edge of the topmost node (smallest y-coordinate)
        top_edge = min(node_y_positions[node] - node_heights[node]/2 for node in plot_nodes)
        # Bottom edge of the bottommost node (largest y-coordinate)  
        bottom_edge = max(node_y_positions[node] + node_heights[node]/2 for node in plot_nodes)
        
        # Add minimal margins (0.1 instead of large values)
        margin = 0.1
        min_y = top_edge - margin    # Visual top (smallest y-value)
        max_y = bottom_edge + margin # Visual bottom (largest y-value)
    else:
        # Fallback for empty data
        min_y = -1.0
        max_y = 1.0
    
    ax.set_ylim(max_y, min_y)  # Reverse y-axis to put node-0 at top
    
    logging.info(f"Plot ylim: {max_y}, {min_y} (reversed for top-to-bottom node order)")
    logging.info(f"Plot size: {plot_width} x {total_height}")

    unique_pods = df['pod_id'].unique() if df is not None and not df.empty else []
    num_unique_pods = len(unique_pods)
    
    pod_to_color = {}
    pod_to_hatch = {}  # Initialize hatching patterns mapping
    if num_unique_pods > 0:
        # Use a combination of colorful colormaps for better distinction
        if num_unique_pods <= 10:
            cmap_name = 'tab10'  # For 10 or fewer pods, use tab10 which has very distinct colors
        elif num_unique_pods <= 20:
            cmap_name = 'tab20'  # For up to 20 pods, use tab20
        else:
            # For more pods, use hsv which provides better color separation for many items
            cmap_name = 'hsv'
        
        logging.info(f"Using colormap '{cmap_name}' for {num_unique_pods} pods")
        actual_cmap = plt.colormaps[cmap_name]
        
        # Generate colors from colormap
        if cmap_name == 'hsv':
            # For HSV, generate evenly spaced points for better visual separation
            colors = [actual_cmap(i/num_unique_pods) for i in range(num_unique_pods)]
        else:
            # For tab10/tab20, use resampled method
            colors = actual_cmap.resampled(num_unique_pods if num_unique_pods > 1 else 2).colors[:num_unique_pods] if num_unique_pods > 0 else []
        
        # Create mapping from pod ID to color
        pod_to_color = {pod: colors[i] for i, pod in enumerate(unique_pods)}
        
        # Add hatching patterns for similar colored pods
        def color_distance(c1, c2):
            """Calculate Euclidean distance between two RGB colors."""
            return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1[:3], c2[:3])))
        
        def assign_hatching_patterns(pod_colors):
            """Assign hatching patterns to pods with similar colors."""
            hatch_patterns = ['', '///', '\\\\\\', '|||', '---', '+++', 'xxx', 'ooo', '...', '***']
            pod_to_hatch = {}
            color_threshold = 0.2  # Threshold for color similarity
            
            used_patterns = []
            for pod_id, color in pod_colors.items():
                # Check if this color is similar to any previous colors
                similar_found = False
                for prev_pod, prev_color in pod_colors.items():
                    if prev_pod != pod_id and prev_pod in pod_to_hatch:
                        if color_distance(color, prev_color) < color_threshold:
                            # Find a pattern not used by similar colors
                            for pattern in hatch_patterns:
                                if pattern not in [pod_to_hatch[p] for p in pod_colors.keys() 
                                                 if p in pod_to_hatch and color_distance(pod_colors[p], color) < color_threshold]:
                                    pod_to_hatch[pod_id] = pattern
                                    similar_found = True
                                    break
                            if similar_found:
                                break
                
                if not similar_found:
                    pod_to_hatch[pod_id] = ''  # No hatching for distinct colors
            
            return pod_to_hatch
        
        pod_to_hatch = assign_hatching_patterns(pod_to_color)
        
        logging.info(f"Assigned {len(pod_to_color)} colors to pods")
        logging.info(f"Assigned hatching patterns: {sum(1 for h in pod_to_hatch.values() if h)} pods have hatching")

    # Initialize a dictionary to keep track of pod positioning within each node
    # This will track all overlapping time slots for better space distribution
    node_pod_layout = defaultdict(list)  # node -> [(start, end, pod_id, cpu_request)]
    
    # First pass: collect all pods for each node to plan layout
    if not df.empty:
        for _, row in df.iterrows():
            if row['node_id'] not in plot_nodes:
                continue
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
            
            # Calculate pod height as a proportion of the node's height based on CPU request
            node_cpu_capacity = node_cpu_capacities.get(node, 1.0) if node_cpu_capacities else 1.0
            
            # Pod height proportional to its CPU request relative to node capacity
            pod_height_proportion = min(cpu_request / node_cpu_capacity, 1.0)
            pod_height = node_height * pod_height_proportion
            
            # Calculate pod positioning based on actual time overlaps and CPU capacity
            pods_in_node = node_pod_layout[node]
            
            # Find all pods that overlap in time with this pod at this specific time slot
            overlapping_pods_at_time = []
            for time_slot in range(start_slot, start_slot + duration):
                for s, e, p, c in pods_in_node:
                    if s <= time_slot < e and not any(pod[2] == p for pod in overlapping_pods_at_time):
                        overlapping_pods_at_time.append((s, e, p, c))
            
            # Sort overlapping pods by start time, then by pod_id for consistent ordering
            overlapping_pods_at_time.sort(key=lambda x: (x[0], x[2]))
            
            # Find this pod's position among the overlapping pods
            pod_index = next(i for i, (s, e, p, c) in enumerate(overlapping_pods_at_time) if p == pod_id)
            total_overlapping = len(overlapping_pods_at_time)
            
            # Calculate the total CPU demand from overlapping pods
            total_cpu_demand = sum(c for s, e, p, c in overlapping_pods_at_time)
            node_cpu_capacity = node_cpu_capacities.get(node, 1.0) if node_cpu_capacities else 1.0
            
            # If total CPU demand exceeds capacity, we have a capacity violation
            capacity_violation = total_cpu_demand > node_cpu_capacity
            
            if total_overlapping > 1:
                # Stack overlapping pods vertically to show capacity violations
                available_height = node_height * 0.9  # Use 90% of node height
                margin_top = node_height * 0.05
                
                # If there's a capacity violation, extend beyond the node bounds to show the problem
                if capacity_violation:
                    # Allow pods to extend beyond node capacity area to show violation
                    stack_height = available_height * 1.3  # 30% extra space to show violations
                    pod_slot_height = stack_height / total_overlapping
                else:
                    # Normal stacking within node bounds
                    pod_slot_height = available_height / total_overlapping
                    
                pod_y_offset = margin_top + (pod_index * pod_slot_height) + (pod_slot_height - pod_height) / 2
                
            else:
                # Single pod, center it in the node
                pod_y_offset = (node_height - pod_height) / 2
            
            pod_y_bottom = node_y_center - node_height/2 + pod_y_offset
            
            # Center position for the label
            text_y_center = pod_y_bottom + pod_height / 2
            
            # Enforce minimum pod height for better visibility
            MIN_VISIBLE_HEIGHT = 0.1  # Minimum height for any pod to ensure visibility
            visible_pod_height = max(pod_height, MIN_VISIBLE_HEIGHT)
            
            # If we've increased the height for visibility, adjust the y position
            height_adjustment = (visible_pod_height - pod_height) / 2
            adjusted_pod_y_bottom = pod_y_bottom - height_adjustment
            
            # Choose edge color and style based on capacity violation
            if capacity_violation:
                edge_color = 'red'
                line_width = 1.2
                edge_alpha = 1.0
            else:
                edge_color = 'black'
                line_width = 0.8
                edge_alpha = 0.8
            
            # Draw the pod rectangle with capacity violation indicators
            rect = plt.Rectangle(
                (start_slot - 0.5, adjusted_pod_y_bottom),  # bottom left (x, y)
                duration,  # width (number of time slots)
                visible_pod_height,  # height with minimum visibility
                facecolor=pod_to_color.get(pod_id, 'gray'),
                edgecolor=edge_color,
                alpha=0.8,  # Increased from 0.7 for better visibility
                linewidth=line_width,  # Thicker red border for capacity violations
                hatch=pod_to_hatch.get(pod_id, ''),  # Add hatching for similar colored pods
                zorder=2  # Above node backgrounds
            )
            ax.add_patch(rect)
            
            # Pod labels have been removed for cleaner visualization

    # Set x-ticks for time slots - limit to a reasonable number if there are many
    max_xticks = min(plot_time_slots, 30)  # Don't show more than 30 ticks for readability
    if plot_time_slots > max_xticks:
        # Show regular intervals
        step = max(1, plot_time_slots // max_xticks)
        xticks = np.arange(0, plot_time_slots, step)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
    else:
        # Show all slots
        ax.set_xticks(np.arange(plot_time_slots))
        ax.set_xticklabels(np.arange(plot_time_slots))
    
    # Make x-axis labels bigger and rotated if there are many time slots
    if plot_time_slots > 15:
        plt.xticks(fontsize=9, rotation=45)
    else:
        plt.xticks(fontsize=10)
        
    # Set custom y-ticks at the center of each node with enhanced formatting
    y_ticks = [node_y_positions[node] for node in plot_nodes]
    ax.set_yticks(y_ticks)
    
    # Format y-tick labels to include node name and CPU capacity with better formatting
    if node_cpu_capacities:
        node_labels = []
        for node in plot_nodes:
            cpu = node_cpu_capacities.get(node, 1.0)
            # Format CPU value to 1 decimal place if it's not an integer
            if cpu == int(cpu):
                cpu_str = f"{int(cpu)}CPU"
            else:
                cpu_str = f"{cpu:.1f}CPU"
            node_labels.append(f"{node} ({cpu_str})")
    else:
        node_labels = plot_nodes
        
    ax.set_yticklabels(node_labels, fontsize=10, fontweight='bold')
    
    # Set vertical grid lines - thinner and lighter for better visibility of pods
    ax.set_xticks(np.arange(plot_time_slots + 1) - 0.5, minor=True)
    
    # Create horizontal grid lines at node boundaries
    # We'll add horizontal lines at the top and bottom of each node
    node_boundaries = []
    for node in plot_nodes:
        y_center = node_y_positions[node]
        height = node_heights[node]
        node_boundaries.append(y_center - height/2)  # bottom
        node_boundaries.append(y_center + height/2)  # top
    
    # Sort and deduplicate boundaries
    node_boundaries = sorted(set(node_boundaries))
    ax.set_yticks(node_boundaries, minor=True)
    
    # Draw the grid with improved styling
    ax.grid(which='minor', color='lightgrey', linestyle='-', linewidth=0.5, alpha=0.7)
    ax.tick_params(which='major', bottom=True, left=True, length=4, width=1.0)

    plt.xlabel("Time Slot", fontsize=11, fontweight='bold')
    plt.ylabel("Node ID", fontsize=11, fontweight='bold')
    
    # Improved title with more informative details
    title_text = f"{title_prefix} - Individual Pods (CPU Proportional)"
    if not df.empty:
        total_pods = len(unique_pods)
        title_text += f"\n{total_pods} Pods across {len(plot_nodes)} Nodes"
    plt.title(title_text, fontsize=14, fontweight='bold', pad=10)
    
    patches = []
    patch_labels = []
    
    # Add visual explanation elements for the legend
    node_capacity_legend = plt.Rectangle((0, 0), 1, 1, facecolor='lightgray', edgecolor='gray', alpha=0.3)
    patches.append(node_capacity_legend)
    patch_labels.append("Node capacity area")
    
    # Add capacity violation indicator to legend
    violation_legend = plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='red', linewidth=1.2, alpha=0.8)
    patches.append(violation_legend)
    patch_labels.append("Capacity violation (red border)")
    
    if node_cpu_capacities:
        # Create a rectangle example showing CPU usage
        cpu_legend = plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='black', alpha=0.8)
        patches.append(cpu_legend)
        patch_labels.append("Pod height ∝ CPU request")
    
    # Sort pods for consistent legend order
    sorted_legend_pods = sorted(list(unique_pods))
    
    # Show all pods in legend - remove artificial limit for better visibility
    legend_pods = sorted_legend_pods  # Show all pods
    note_text = None  # No truncation message needed
    
    # Add pods to legend
    for pod_id_leg in legend_pods:
        if pod_id_leg in pod_to_color:
            patches.append(plt.Rectangle((0,0),1,1, 
                                       facecolor=pod_to_color[pod_id_leg], 
                                       edgecolor='black', 
                                       alpha=0.8,
                                       hatch=pod_to_hatch.get(pod_id_leg, '')))  # Include hatching in legend
            # Show only short ID for cleaner legend (no double labeling)
            short_id = shorten_pod_label(pod_id_leg)
            patch_labels.append(short_id)
    
    # Add the "more pods" note if necessary
    if note_text:
        patches.append(plt.Rectangle((0,0),1,1, fill=False, edgecolor='none'))
        patch_labels.append(note_text)
    
    if patches:
        # Calculate optimal number of columns for legend - use more columns for horizontal layout
        num_patches = len(patches)
        if num_patches <= 10:
            num_legend_cols = min(num_patches, 5)  # 1-5 columns for small numbers
        elif num_patches <= 25:
            num_legend_cols = 6  # 6 columns for medium numbers
        elif num_patches <= 50:
            num_legend_cols = 8  # 8 columns for larger numbers
        else:
            num_legend_cols = 10  # Maximum 10 columns for very large numbers
            
        # Position legend below the plot for better use of horizontal space
        fig.legend(patches, patch_labels, 
                  loc='upper center',
                  bbox_to_anchor=(0.5, -0.02),  # Position below the plot
                  ncol=num_legend_cols,
                  title='Legend',
                  fontsize=8,  # Slightly larger font for readability
                  frameon=True, 
                  fancybox=True,
                  framealpha=0.9,
                  title_fontsize=10,
                  columnspacing=0.8,  # Increase spacing between columns for clarity
                  handletextpad=0.3,  # Spacing between legend markers and text
                  handlelength=1.2,   # Legend marker length
                  markerscale=0.9)    # Legend marker size

    # Use tight layout with padding for the legend below
    plt.tight_layout()
    if patches:
        # Adjust the plot to make room for the legend below
        plt.subplots_adjust(bottom=0.15)  # Leave space at the bottom for legend
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
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
    ax.set_xticklabels(np.arange(plot_time_slots))
    ax.set_yticklabels(plot_nodes)
    ax.set_xticks(np.arange(plot_time_slots + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(plot_nodes) + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.5)
    ax.tick_params(which='major', bottom=False, left=False)

    plt.xlabel("Time Slot")
    plt.ylabel("Node ID")
    plt.title(f"{title_prefix} - Density Heatmap")
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
    parser.add_argument("--config_file", default=CONFIG_FILE_PATH, help=f"Path to the infrastructure config YAML file (default: {CONFIG_FILE_PATH}).")
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
        parent_dir_name = os.path.basename(os.path.dirname(args.input_csvs[0]))
        if "placements" in parent_dir_name.lower():
             plot_output_subdir_name = os.path.splitext(os.path.basename(args.input_csvs[0]))[0]
        else:
            plot_output_subdir_name = parent_dir_name if parent_dir_name else "visualization_run"
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_output_subdir_name = f"visualization_{timestamp}"

    final_plot_output_dir = os.path.join(args.output_dir_base, plot_output_subdir_name)
    os.makedirs(final_plot_output_dir, exist_ok=True)

    all_csv_files = []
    for pattern in args.input_csvs:
        all_csv_files.extend(glob.glob(pattern))
    
    if not all_csv_files:
        print(f"No CSV files found matching patterns: {args.input_csvs}")
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
