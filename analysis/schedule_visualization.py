import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import datetime
import yaml
import re
import logging
from collections import defaultdict

CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "pkg", "carbon-aware", "infra-workload-config.yaml")

def load_infra_config(config_path_str=CONFIG_FILE_PATH):
    """Loads infrastructure configuration (nodes) from a YAML file."""
    if not config_path_str:
        logging.warning("Path to infra config not provided. Cannot load node names.")
        return {}

    # Determine the absolute path to the config file
    script_dir = os.path.dirname(os.path.abspath(__file__)) # analysis/
    repo_root = os.path.dirname(script_dir) # repo root

    if not os.path.isabs(config_path_str):
        config_path_abs = os.path.abspath(os.path.join(repo_root, config_path_str))
    else:
        config_path_abs = os.path.normpath(config_path_str)

    logging.info(f"Attempting to load infra config from: {config_path_abs}")

    try:
        with open(config_path_abs, 'r') as f:
            config_data = yaml.safe_load(f)
        if not config_data:
            logging.warning(f"Infra config file loaded but is empty: {config_path_abs}")
            return {}
        
        nodes_filename = config_data.get('nodes', {}).get('filename')
        logging.info(f"Read 'nodes.filename' from {config_path_abs}: {nodes_filename}")

        parsed_num_timeslots = None 
        logging.info("'num_timeslots' will be determined by data or default, not from this config file.")

        node_names = []
        if nodes_filename:
            config_dir = os.path.dirname(config_path_abs)
            nodes_yaml_path_abs = os.path.abspath(os.path.join(config_dir, nodes_filename))
            logging.info(f"Attempting to load node names from: {nodes_yaml_path_abs}")
            try:
                with open(nodes_yaml_path_abs, 'r') as nf:
                    all_node_documents = yaml.safe_load_all(nf)
                    for doc in all_node_documents:
                        if doc and isinstance(doc, dict):
                            metadata = doc.get('metadata', {})
                            node_name = metadata.get('name')
                            if node_name:
                                node_names.append(node_name)
                
                if node_names:
                    logging.info(f"Successfully loaded node names: {node_names} from {nodes_yaml_path_abs}")
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
            'total_timeslots': parsed_num_timeslots 
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

def load_data(csv_path):
    """Loads data from a CSV file into a pandas DataFrame."""
    try:
        # cpu_request is not strictly needed for this version, but load it if present.
        df = pd.read_csv(csv_path)
        if 'cpu_request' in df.columns:
             df['cpu_request'] = pd.to_numeric(df['cpu_request'], errors='coerce').fillna(0)
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
    """Shortens pod labels like 'm001' to '1' or '001' to '1' for legibility."""
    match_prefix_num = re.fullmatch(r'([a-zA-Z_]+)(\d+)', pod_id_str)
    if match_prefix_num:
        numerical_part = match_prefix_num.group(2)
        try:
            return str(int(numerical_part))
        except ValueError:
            return numerical_part

    match_num_only = re.fullmatch(r'(\d+)', pod_id_str)
    if match_num_only:
        numerical_part = match_num_only.group(1) 
        try:
            return str(int(numerical_part))
        except ValueError:
            return numerical_part 
            
    return pod_id_str

def extract_pod_type(pod_id_str):
    """Extracts the type (e.g., 'm', 'w') from pod_id like 'm001' or 'worker01'."""
    if not isinstance(pod_id_str, str):
        return "?" # Return a default for non-string inputs
    match = re.match(r'([a-zA-Z]+)', pod_id_str) # Match leading letters
    if match:
        return match.group(1)
    return "?" # Default if no type prefix is found

def plot_individual_pods(df, output_path, title_prefix="Pod Placement", all_node_names_from_config=None, total_timeslots_from_config=None):
    DEFAULT_TIMESLOTS = 24
    logging.info(f"Plotting individual pods (grid view). Infra config provided: {bool(total_timeslots_from_config)}")

    #region ======== Determine plot_nodes and plot_time_slots (largely preserved) ========
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
        except Exception as e:
            print(f"Warning: Error converting essential columns to required types for '{title_prefix}': {e}. Plot may be affected.")

    all_nodes_from_config_list = all_node_names_from_config if all_node_names_from_config else []
    logging.info(f"Nodes from config for plot_individual_pods: {all_nodes_from_config_list}")

    plot_nodes_data_driven = []
    if df is not None and not df.empty and 'node_id' in df.columns:
        # Ensure node_id column is not all NaN or empty strings before calling unique()
        valid_node_ids = df['node_id'].dropna()
        if not valid_node_ids.empty:
            plot_nodes_data_driven = sorted(valid_node_ids.unique())
    
    plot_nodes = all_nodes_from_config_list if all_nodes_from_config_list else plot_nodes_data_driven
    
    if not plot_nodes: # If still no nodes, handle empty plot or return
        if df is None or df.empty: # True empty case
            print(f"No data or empty DataFrame for individual pods: {title_prefix}.")
            plot_nodes_for_empty = all_nodes_from_config_list
            plot_time_slots_for_empty = total_timeslots_from_config if total_timeslots_from_config is not None else DEFAULT_TIMESLOTS
            if not plot_nodes_for_empty:
                print(f"Cannot create empty plot '{title_prefix}': No node data from CSV or config.")
                return
            
            fig, ax = plt.subplots(figsize=(max(12, plot_time_slots_for_empty * 0.5), max(8, len(plot_nodes_for_empty) * 0.6)))
            ax.set_xticks(np.arange(plot_time_slots_for_empty))
            ax.set_yticks(np.arange(len(plot_nodes_for_empty)))
            ax.set_xticklabels(np.arange(plot_time_slots_for_empty))
            ax.set_yticklabels(plot_nodes_for_empty)
            ax.set_xticks(np.arange(plot_time_slots_for_empty + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(len(plot_nodes_for_empty) + 1) - 0.5, minor=True)
            ax.grid(which='minor', color='lightgrey', linestyle='-', linewidth=0.5)
            ax.tick_params(which='major', bottom=False, left=False)

            plt.xlabel("Time Slot")
            plt.ylabel("Node ID")
            plt.title(f"{title_prefix} - Individual Pods (No Data)")
            fig.subplots_adjust(bottom=0.1)
            plt.savefig(output_path, bbox_inches='tight')
            plt.close(fig)
            print(f"Empty individual pod placement plot saved to {output_path}")
            return
        else: 
             print(f"Cannot create plot '{title_prefix}': No nodes found in data or config, though data is present.")
             return

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
            numeric_start_slot = pd.to_numeric(df['start_slot'], errors='coerce')
            numeric_duration = pd.to_numeric(df['duration'], errors='coerce')
            valid_slots = numeric_start_slot.notna() & numeric_duration.notna()
            if valid_slots.any():
                 max_data_slot = (numeric_start_slot[valid_slots] + numeric_duration[valid_slots]).max()
            if pd.isna(max_data_slot): max_data_slot = 0
        except TypeError:
            max_data_slot = 0 
            print(f"Warning: Could not calculate max_data_slot due to non-numeric types for '{title_prefix}'.")
        plot_time_slots = max(int(max_data_slot if max_data_slot > 0 else 0), DEFAULT_TIMESLOTS)
    else:
        plot_time_slots = DEFAULT_TIMESLOTS

    if plot_time_slots <= 0:
        print(f"Warning: plot_time_slots is {plot_time_slots} for '{title_prefix}'. Using {DEFAULT_TIMESLOTS} as minimum.")
        plot_time_slots = DEFAULT_TIMESLOTS
    #endregion

    logging.info(f"Final plot_nodes for individual_pods (grid): {plot_nodes}")
    logging.info(f"Final plot_time_slots for individual_pods (grid): {plot_time_slots}")

    placement_matrix = np.full((len(plot_nodes), plot_time_slots), "", dtype=object)

    if df is not None and not df.empty:
        for _, row in df.iterrows():
            pod_id_val = str(row.get('pod_id', 'N/A'))
            node_id_val = str(row.get('node_id', 'N/A'))
            
            if node_id_val not in plot_nodes:
                continue

            try:
                r_idx = plot_nodes.index(node_id_val)
                start_slot = int(row['start_slot'])
                duration = int(row['duration'])
                pod_id_short = shorten_pod_label(pod_id_val)
                
                if not pod_id_short: # Ensure pod_id_short is never None or empty
                    pod_id_short = "?" 

                for t_slot in range(start_slot, start_slot + duration):
                    if 0 <= t_slot < plot_time_slots:
                        current_text = placement_matrix[r_idx, t_slot]
                        if current_text == "":
                            placement_matrix[r_idx, t_slot] = pod_id_short
                        else:
                            # Refined logic for appending/truncating multiple pod_ids in a cell
                            # Avoid appending if pod_id_short is already present or if both are "?"
                            if pod_id_short not in current_text.split('/') and not (pod_id_short == "?" and current_text == "?"):
                                new_combined_text = current_text + "/" + pod_id_short
                                # Case 1: Current cell is '?', new pod_id is not '?'. Replace '?' with new pod_id.
                                if current_text == "?" and pod_id_short != "?":
                                    placement_matrix[r_idx, t_slot] = pod_id_short
                                # Case 2: Current cell is not '?', new text fits. Append.
                                elif len(new_combined_text) < 10 and current_text != "?":
                                    placement_matrix[r_idx, t_slot] = new_combined_text
                                # Case 3: Text too long, needs truncation. And not already ending with ".."
                                elif not current_text.endswith(".."):
                                    first_part_current = current_text.split('/')[0]
                                    # If multiple items already, truncate to "first/.." (unless first is "?")
                                    if "/" in current_text and first_part_current != "?":
                                        placement_matrix[r_idx, t_slot] = first_part_current + "/.."
                                    # If single long item (not "?"), truncate to "abc.."
                                    elif len(current_text) > 3 and current_text != "?": 
                                        placement_matrix[r_idx, t_slot] = current_text[:3] + ".."
                                    # Else, leave as is (e.g. current_text is "?", or already "X/..", or became too long and was not "?")
            except (ValueError, TypeError, KeyError) as e:
                logging.warning(f"Skipping row due to data error for pod {pod_id_val}: {e} - Row: {row.to_dict()}")
                continue

    fig, ax = plt.subplots(figsize=(max(12, plot_time_slots * 0.5), max(8, len(plot_nodes) * 0.6)))
    
    ax.set_xlim(-0.5, plot_time_slots - 0.5)
    ax.set_ylim(len(plot_nodes) - 0.5, -0.5)

    for r, node_name in enumerate(plot_nodes):
        for c in range(plot_time_slots):
            text_val = placement_matrix[r, c]
            if text_val:
                ax.text(c, r, text_val, va='center', ha='center', fontsize=7)

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
    plt.title(f"{title_prefix} - Individual Pods (Grid View)")

    fig.subplots_adjust(bottom=0.1)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Individual pod placement (grid view) plot saved to {output_path}")

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

def main():
    parser = argparse.ArgumentParser(description="Visualize pod placement schedules from CSV data.")
    parser.add_argument("input_csvs", nargs='+', help="Path(s) to the input CSV file(s) containing pod placement data. Can be a glob pattern.")
    parser.add_argument("-o", "--output_dir_base", default=os.path.join(os.path.dirname(__file__), "..", "figures"), help="Base directory to save the generated plots (default: ../figures relative to script). A subdirectory will be created here.")
    parser.add_argument("-m", "--mode", choices=["individual", "density", "all"], default="all", help="Type of visualization to generate: 'individual' pods, 'density' heatmap, or 'all' (default: all).")
    parser.add_argument("-n", "--name", default=None, help="Optional name for the run/comparison. This will be used as the subdirectory name under output_dir_base and in plot titles/filenames.")
    parser.add_argument("--config_file", default=CONFIG_FILE_PATH, help=f"Path to the infrastructure config YAML file (default: {CONFIG_FILE_PATH}).")
    
    args = parser.parse_args()

    infra_config = load_infra_config(args.config_file)
    all_node_names_from_config = infra_config.get('nodes', [])
    total_timeslots_from_config = infra_config.get('total_timeslots')

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
        loaded_dfs = [load_data(f) for f in all_csv_files]
        valid_dfs = [df for df in loaded_dfs if df is not None and not df.empty]

        if not valid_dfs:
            print(f"No valid data loaded from CSV files. Exiting.")
            return
        
        combined_df = pd.concat(valid_dfs, ignore_index=True)
        
        base_filename = os.path.join(final_plot_output_dir, args.name.replace(" ", "_"))
        title_prefix = args.name

        if args.mode in ["individual", "all"]:
            plot_individual_pods(combined_df, f"{base_filename}_individual.png", title_prefix, 
                                 all_node_names_from_config, total_timeslots_from_config)
        if args.mode in ["density", "all"]:
            plot_density_heatmap(combined_df, f"{base_filename}_density.png", title_prefix, all_node_names_from_config, total_timeslots_from_config)
    else:
        for csv_file in all_csv_files:
            df = load_data(csv_file)

            file_basename = os.path.splitext(os.path.basename(csv_file))[0]
            
            current_plot_name_prefix = file_basename
            title_prefix_for_plot = args.name if args.name else file_basename

            output_base = os.path.join(final_plot_output_dir, current_plot_name_prefix.replace(" ", "_"))

            print(f"Processing: {csv_file} into {final_plot_output_dir} with plot name: {current_plot_name_prefix}")

            if args.mode in ["individual", "all"]:
                plot_individual_pods(df, f"{output_base}_individual.png", title_prefix_for_plot, 
                                     all_node_names_from_config, total_timeslots_from_config)
            if args.mode in ["density", "all"]:
                plot_density_heatmap(df, f"{output_base}_density.png", title_prefix_for_plot, all_node_names_from_config, total_timeslots_from_config)

if __name__ == "__main__":
    main()
