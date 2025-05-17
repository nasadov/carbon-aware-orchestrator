#!/bin/bash

#print_usage() {
  echo "Usage: $0 [-o <base_figures_directory>] [-m <mode>] [-n <n>] [-c <infra_config_path>] [-w <workloads_dir>] <input_csv_path_or_pattern_1> [input_csv_path_or_pattern_2 ...]"elper script to run the schedule_visualization.py script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/analysis/schedule_visualization.py" # Corrected path to script in analysis subdir

# Default base output directory for figures, plots will go into a subfolder here
FIGURES_BASE_DIR="$SCRIPT_DIR/figures" 
MODE="all" # Default mode: individual, density, all
NAME="" # Optional name for the run/comparison, used for subfolder name
INFRA_CONFIG_PATH="" # Optional path to the infra-workload-config.yaml
WORKLOADS_DIR="$SCRIPT_DIR/pkg/carbon-aware/workloads" # Default path to workloads directory

print_usage() {
  echo "Usage: $0 [-o <base_figures_directory>] [-m <mode>] [-n <name>] [-c <infra_config_path>] <input_csv_path_or_pattern_1> [input_csv_path_or_pattern_2 ...]"
  echo ""
  echo "Options:"
  echo "  -o <base_figures_directory>  Set the base directory for output plots (default: ./figures relative to project root)."
  echo "                             A subdirectory based on <name> or timestamp will be created here."
  echo "  -m <mode>                    Set the visualization mode: 'individual', 'density', or 'all' (default: all)."
  echo "  -n <name>                    Optional name for the run/comparison. This will be used as the subdirectory name"
  echo "                             under <base_figures_directory> and in plot titles."
  echo "  -c <infra_config_path>       Path to the infra-workload-config.yaml file."
  echo "  -w <workloads_dir>         Path to the directory containing workload YAML files with CPU requests."
  echo "  -h                           Display this help message."
  echo ""
  echo "Arguments:"
  echo "  <input_csv_path_or_pattern>  One or more paths or glob patterns for input CSV files."
  echo "                                 Example: analysis/heuristic_20230101_120000/heuristic_placements_20230101_120000.csv"
  echo "                                 Example: 'analysis/*_placements_*/\*_placements_*.csv'" # Example for new structure
  exit 1
}

# Parse options
while getopts "ho:m:n:c:w:" opt; do
  case $opt in
    h) print_usage ;;
    o) FIGURES_BASE_DIR="$OPTARG" ;;
    m) MODE="$OPTARG" ;;
    n) NAME="$OPTARG" ;;
    c) INFRA_CONFIG_PATH="$OPTARG" ;;
    w) WORKLOADS_DIR="$OPTARG" ;;
    *) print_usage ;;
  esac
done
shift "$((OPTIND -1))"

# Check if input CSVs are provided
if [ -z "$1" ]; then
  echo "Error: No input CSV path(s) provided."
  print_usage
fi

# Ensure the Python script is executable
if [ ! -x "$PYTHON_SCRIPT" ]; then
  echo "Making Python script executable: $PYTHON_SCRIPT"
  chmod +x "$PYTHON_SCRIPT"
  if [ $? -ne 0 ]; then
    echo "Error: Failed to make Python script executable. Please check permissions."
    exit 1
  fi
fi

# Construct the command
CMD=("python3" "$PYTHON_SCRIPT")

# The python script now expects --output_dir_base
if [ -n "$FIGURES_BASE_DIR" ]; then
  CMD+=("--output_dir_base" "$FIGURES_BASE_DIR")
fi

if [ -n "$MODE" ]; then
  CMD+=("-m" "$MODE")
fi

if [ -n "$NAME" ]; then
  CMD+=("-n" "$NAME") # This name will be used for the subdirectory by the python script
fi

if [ -n "$INFRA_CONFIG_PATH" ]; then
  CMD+=("--config_file" "$INFRA_CONFIG_PATH")
fi

if [ -n "$WORKLOADS_DIR" ]; then
  CMD+=("--workloads_dir" "$WORKLOADS_DIR")
fi

# Add all remaining arguments as input CSVs
CMD+=("$@")

# The Python script will create the specific subdirectory (FIGURES_BASE_DIR/NAME_OR_TIMESTAMP)
# We just need to ensure the FIGURES_BASE_DIR exists.
mkdir -p "$FIGURES_BASE_DIR"

# Run the command
echo "Running visualization script..."
echo "Command: ${CMD[@]}"

"${CMD[@]}"

if [ $? -eq 0 ]; then
  echo "Visualization script completed successfully. Plots saved in a subdirectory under $FIGURES_BASE_DIR"
else
  echo "Error running visualization script."
  exit 1
fi
