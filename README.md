# Carbon-Aware Orchestrator

This repository contains an experimental implementation of a carbon-aware orchestrator that optimizes workload placement based on both operational and embodied carbon emissions. The goal is to explore environmentally-conscious placement decisions within Kubernetes scheduling frameworks.

This work builds upon the original scheduler plugin developed by Fondazione Bruno Kessler (FBK), and remains an ongoing research project.

## Table of Contents

- [Overview](#overview)
- [The IDL](#the-idl)
- [Repository Structure](#repository-structure)
- [Carbon-Aware Algorithm](#carbon-aware-algorithm)
    - [Heuristic Algorithm](#heuristic-algorithm)
    - [MILP Global Optimizer](#milp-global-optimizer)
    - [Carbon-Agnostic (vanilla k8s-like) Baseline](#carbon-agnostic-vanilla-k8s-like-baseline)
    - [Constraints Handling](#constraints-handling)
- [Carbon-Aware Data Model](#carbon-aware-data-model)
- [Node Metadata](#node-metadata)
- [Workload Specifications](#workload-specifications)
- [Carbon Emissions Model](#carbon-emissions-model)
- [Carbon Units Documentation](docs/carbon_units.md)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Setup Environment](#setup-environment)
    - [Configure Test Infrastructure and Workloads](#configure-test-infrastructure-and-workloads)
    - [Generate Test Data](#generate-test-data)
    - [Run the Experiment](#run-the-experiment)
    - [MILP Global Optimizer](#running-milp-global-optimizer)
    - [Verify Constraint Enforcement](#verify-constraint-enforcement)
    - [Performance Metrics](#performance-metrics)
    - [Carbon Emissions Analysis](#carbon-emissions-analysis)
    - [Visualizing Results](#visualizing-results)

## Overview

The carbon-aware orchestrator considers multiple factors when placing workloads:

1. **Operational carbon emissions**: Estimates based on regional carbon intensity forecasts and power consumption models
2. **Embodied carbon emissions**: Approximations of hardware manufacturing emissions amortized over device lifetime
3. **Node characteristics**: Including hardware type, region, and resource constraints
4. **Workload characteristics**: Including duration, resource requirements, and constraints

This is experimental work and the accuracy of carbon calculations depends on the quality of input data and modeling assumptions.

## The IDL 

The interface definition is in the `./pkg/idl/idl.proto` file. It models a snapshot of a Kubernetes cluster in terms of infrastructure and workload deployment, along with placement information including scheduling time for each microservice and scores for each pod and node. This gRPC interface decouples the FogAtlas code (written in Go) from the placement algorithm implementations that can be written in various programming languages.

## Repository Structure

- `pkg/idl/`: gRPC interface definition
- `pkg/carbon-aware/`: Implementation of the carbon-aware scheduling algorithm
    - `server-python/`: Python implementation 
    - `client/`: Go client for testing
    - `infra_workload_gen.py`: Tool for generating test infrastructure and workload data
    - `performance_logs/`: Directory containing performance metrics logs
    - `nodes.yaml`: Node configuration with CPU capacities and hardware metadata
- `tests/`: Testing tools and experimental results
    - `milp_global_optimizer.py`: MILP-based global optimization algorithm
    - `experiments/`: Generated experiment results from MILP runs
    - `check_timeslot_constraints.py`: Constraint validation tools
- `analysis/`: Tools for analyzing and visualizing performance data
    - `schedule_visualization.py`: Enhanced script for visualizing pod placement schedules
    - `visualization.py`: Script for generating performance visualizations
    - `simple_carbon_heatmap.py`: Carbon emissions visualization tools
    - `carbon_emissions_comparison.py`: Comprehensive analysis comparing carbon emissions across algorithms
    - `emissions_vs_pods_plot_generator.py`: Plot total and per‑pod emissions vs. pod count (Vanilla/Heuristic/Global‑Optimal)
    - `success_rate_plot_generator.py`: Plot scheduling success rate vs. pod count (multi‑algorithm)
    - `heuristic_time_complexity_plot_generator.py`: Plot heuristic precompute time vs. pods and vs. nodes
    - `fix_start_slots.py`: Tool for fixing timeslot inconsistencies in placement data for vanilla algorithm
- `figures/`: Generated visualization outputs organized by experiment
    - `Comparison/Carbon_Emissions_Analysis_TIMESTAMP/`: Carbon emissions comparison results with automatic timestamping
    - `Algorithm_Comparison_Latest/`: Multi-algorithm performance comparisons
    - `Heuristic_Latest/`, `Global_Optimal_Latest/`: Algorithm-specific visualizations
- `docs/`: Documentation including research papers and guides
    - `schedule_visualization.py`: Script for visualizing pod placement schedules
    - `run_schedule_visualization.sh`: Helper script for running schedule visualizations
    - `VISUALIZATION_GUIDE.md`: Documentation for using visualizations

## Carbon-Aware Algorithm

The carbon-aware orchestrator currently provides two scheduling approaches, both of which are still under active development and may have limitations:

### Heuristic Algorithm

The heuristic algorithm provides reasonable placement decisions by:

1. **Carbon Score Calculation**: For each pod-node pairing, calculating total carbon emissions (operational + embodied)
2. **Constraint Filtering**: Ensuring resource requirements and timing constraints are met
3. **Greedy Selection**: Choosing the node with lowest carbon emissions for each pod

This approach is fast and practical, but may not always find the global optimum.

### MILP Global Optimizer

The Mixed Integer Linear Programming (MILP) optimizer finds globally optimal solutions by:

1. **Mathematical Modeling**: Formulating the entire scheduling problem as a linear program
2. **Global Optimization**: Finding the best solution across all pods and nodes simultaneously
3. **Constraint Satisfaction**: Guaranteeing all resource and timing constraints are respected
4. **Solver Integration**: Using CBC (Coin-OR), CPLEX, or Gurobi for optimization

The MILP optimizer supports two optimization approaches:

#### Lexicographic Optimization (Recommended)

**New feature**: The global optimizer now uses **lexicographic optimization** by default, which solves the pod placement problem in two phases:

1. **Phase 1**: Maximize the number of placed pods (ensuring feasibility first)
2. **Phase 2**: Minimize carbon emissions while maintaining the optimal placement count

This approach ensures that:
- **All feasible pods are placed** (no pods are left unplaced due to solver optimization)
- **Carbon footprint is minimized** among all optimal placement solutions
- **Academic rigor** is maintained for research publications

#### Standard Optimization

The traditional single-phase optimization that directly minimizes carbon emissions with mandatory placement constraints (`== 1` for each pod).

#### Configuration

The optimization approach can be configured in `infra-workload-config.yaml`:

```yaml
# Global optimization algorithm configuration
optimization:
  # Use lexicographic optimization: first maximize placements, then minimize emissions
  # This ensures all feasible pods are placed while optimizing carbon footprint
  use_lexicographic: true
  
  # Solver configuration
  solver:
    time_limit: 20  # seconds per phase for lexicographic, total for standard
    gap_tolerance: 0.0  # No gap tolerance - enforce strict feasibility
```

#### Running MILP Global Optimizer

```bash
# Run with lexicographic optimization (default)
cd pkg/carbon-aware/server-python
python main.py --algorithm global-optimal --workloads-dir ../workloads \
    --nodes-file ../nodes.yaml --forecasts-file all_forecasts.json \
    --experiment --experiment-dir experiments/lexicographic_test

# Or run standalone tests
cd tests
python test_lexicographic_optimization.py

# Results are saved to experiments/ directory with timestamp
```

The MILP optimizer is also available in `tests/milp_global_optimizer.py` for standalone testing and generates experiment results for comparison with heuristic approaches. Please note that this is an experimental feature and may require significant computational resources for larger problems.

### Carbon-Agnostic (vanilla k8s-like) Baseline

The carbon-agnostic baseline (vanilla k8s-like) simulates a Kubernetes-default-inspired scheduler to provide a robust, apples-to-apples comparison against carbon-aware methods.

Key properties:

1. **Filter → Score → Select**: mirrors kube-scheduler structure.
2. **Feasibility**: CPU/RAM must be available across the pod's labeled duration on a node for a chosen start slot.
3. **Scoring (LeastAllocated-like)**: prefers nodes with higher free CPU and RAM fractions (equal weights).
4. **Earliest timeslot respected**: pods from `timeslot_X.yaml` can only start at or after X.
5. **No carbon signals**: ignores carbon intensity and embodied emissions; returns 0.0 emissions in placements.

Run precompute (offline, capacity-safe):

```bash
python3 pkg/carbon-aware/server-python/main.py \
  --algorithm vanilla \
  --precompute \
  --workloads-dir pkg/carbon-aware/workloads \
  --nodes-file pkg/carbon-aware/nodes.yaml \
  --forecasts-file pkg/carbon-aware/server-python/all_forecasts.json \
  --experiment-dir pkg/carbon-aware/server-python/experiments \
  --loglevel INFO
```

Outputs: a timestamped directory under `experiments/vanilla_*` with `vanilla_placements_session.csv`.

Notes and caveats:

- Real kube-scheduler is instantaneous (no time horizon), and can consider many plugins (taints/tolerations, affinity, topology spread, volumes), priorities, and preemption. The simulator focuses on CPU/RAM feasibility with a simple LeastAllocated-like score to create a consistent baseline over the same timeslot horizon used by heuristic and MILP.
- For higher fidelity, we can incrementally add plugin-like filters (taints, affinity), spread constraints, and weighted multi-plugin scoring.

### Constraints Handling

The algorithm enforces several constraints when placing workloads:

- **Resource constraints**: Ensuring pods have enough CPU and memory
- **Deadline constraints**: Making sure workloads complete before their specified deadlines
- **Earliest timeslot constraint**: Pods from timeslot_X.yaml files are only scheduled at or after timeslot X

Both the heuristic and global optimization (MILP) algorithms properly enforce the earliest_timeslot constraint, which is a key real-world operational requirement. This implementation includes:
- Mapping pod IDs to their source files to determine the earliest valid timeslot
- Validating all potential placements against the earliest_timeslot constraint
- Special handling in the MILP formulation for the global optimal algorithm

## Carbon-Aware Data Model

### Node Metadata

Nodes include detailed hardware information through Kubernetes labels and annotations:

- **Region**: `topology.kubernetes.io/region` (e.g., "DE", "FR")
- **Hardware subcategory**: `hardware.carbon/subcategory` (e.g., "Server", "Laptop", "IoT")
- **Embodied carbon**: `hardware.carbon/embodied_emissions` (kgCO2e)
- **Lifetime**: `hardware.carbon/lifetime_years` (years)
- **Power consumption**:
    - `hardware.power/idle_watts`: Base power consumption when idle
    - `hardware.power/active_watts`: Typical power when active
    - `hardware.power/max_watts`: Maximum power at full utilization

### Workload Specifications

Workloads include resource requirements and time parameters:

- **CPU and Memory**: Standard Kubernetes resource requests
- **Duration**: How long the workload will run, specified via annotation `workload.carbon/duration_hours`

## Carbon Emissions Model

The scheduler calculates emissions for each potential node-workload pairing using:

1. **Dynamic power model**: `idle + (max-active) * CPU_usage_ratio` (measured in Watts)
2. **Operational emissions**: `power * duration * carbon_intensity` (measured in g CO2e)
3. **Embodied emissions**: `(embodied_carbon / lifetime_hours) * duration` (measured in g CO2e)

Updated (co-location aware) allocation:
- Node power at aggregate utilization U = sum_i (cpu_i / CPU_total): `P_node = P_idle + (P_max - P_active) * U`.
- Idle and embodied are paid once per (node,timeslot) when any pod is present; dynamic is linear in CPU share.
- Per-pod allocation: `dynamic_i = (P_max - P_active) * u_i`, and idle+embodied are allocated proportionally to `u_i / U` across active pods in that slot.
- The MILP objective mirrors this: dynamic terms per placement and a binary `y[node,slot]` to pay idle+embodied once per active slot.

**Important**: All embodied carbon values in configuration files are specified in whole grams (g CO2e), not kilograms. For detailed information about units and calculation methods, see the [Carbon Units Documentation](docs/carbon_units.md).

## Getting Started

### Prerequisites

- Kubernetes cluster (v1.21+)
- Go 1.18+
- Python 3.9+

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/nasadov/carbon-aware-orchestrator.git
cd carbon-aware-orchestrator

# Install required Python dependencies
pip install -r pkg/carbon-aware/server-python/requirements.txt

# Install additional dependencies for visualization
pip install matplotlib pandas seaborn
```

### Configure Test Infrastructure and Workloads

Open `infra-workload-config.yaml` and adjust the configuration:
```yaml
nodes:
    # Number of nodes to generate
    num_nodes: 8
    
    # Regions that match carbon intensity forecast regions
    regions:
        - DE
        - FR
        - ES
        - IT-NO
        
    # Method to assign regions to nodes ("cycle" or "random")
    assignment_method: cycle
    
    # Hardware subcategories configuration with power and embodied carbon profiles
    hardware_subcategories:
        IoT:
            cpu: "2"
            memory: "2Gi"
            embodied_carbon: 27.471
            lifetime: 5.2
            power:
                idle: 0.5
                active: 2.0
                max: 5.0
        # Additional hardware profiles...
    
    # Method to assign hardware types to nodes
    # - "cycle": Simple cycling through hardware types
    # - "random": Random assignment with fixed seed
    # - "shifted_cycle": Advanced cycling that ensures diverse region-hardware mappings
    hardware_assignment_method: shifted_cycle
    
    # Offset for shifted_cycle (how many positions to shift each cycle)
    hardware_cycle_offset: 1

workload:
    # Directory to output workload files
    output_dir: workloads
    
    # Number of timeslots to generate (e.g., 24 for each hour in a day)
    num_timeslots: 24
    
    # Lambda parameter for Poisson distribution (average services per timeslot)
    poisson_lambda: 4
    
    # Minimum number of services per timeslot
    min_services: 1
    
    # Available duration values (in hours)
    durations:
        - 1  # short duration
        - 3  # medium duration 
        - 6  # long duration
    
    # Method for assigning durations ("cycle" or "random")
    duration_assignment_method: cycle
    
    # Strategy for determining deadlines
    # - "tight": Deadline equals duration (no flexibility)
    # - "flexible": Duration plus variable flexibility
    # - "mixed": Mix of tight and flexible deadlines
    # - "exact": Deadline is exactly 2x duration
    deadline_strategy: flexible
    
    # Hours to add to duration for flexible deadlines
    deadline_flexibility_hours:
        - 1   # Low flexibility
        - 3   # Medium flexibility
        - 6   # High flexibility
    
    # Available CPU request options
    cpu_options:
        - 1000m
        - 500m
        - 250m
        - 100m
    
    # Available memory request options
    mem_options:
        - 1Gi
        - 512Mi
        - 256Mi
        - 128Mi

    # Strategy for generating the number of pods per timeslot
    # - "poisson": draw per-timeslot counts from a Poisson(λ)
    # - "exact_total": generate an exact total and distribute across timeslots deterministically
    generation_strategy: poisson

    # Exact total pods to generate across all timeslots (used when generation_strategy: exact_total)
    exact_total_pods: 120
```

### Generate Test Data

```bash
# Generate infrastructure and workload data based on your config
python pkg/carbon-aware/infra_workload_gen.py
```
This will create:
- `nodes.yaml`: Infrastructure definitions with hardware characteristics and regional assignments
- `workloads/timeslot_*.yaml`: Directory containing workload definitions for each timeslot

The `shifted_cycle` hardware assignment method creates different hardware-region combinations across nodes, making tests more realistic.

### Run the Experiment

```bash
# Start the Python server
cd pkg/carbon-aware/server-python
python3 -m server

# In a new terminal, run the Go client with the test data
cd /root/carbon-aware-orchestrator
go run client-test.go
```

### Verify Constraint Enforcement

To verify that the earliest timeslot constraint is properly enforced in placement results:

```bash
# Check constraints on a specific experiment result CSV file
python3 tests/check_timeslot_constraints.py /path/to/your/placements_session.csv

# Example: Check the latest heuristic experiment
python3 tests/check_timeslot_constraints.py pkg/carbon-aware/server-python/experiments/heuristic_perf_log_session_YYYYMMDD_HHMMSS/heuristic_placements_session.csv
```

The script will analyze the placements and report any constraint violations.

#### Batch validation and reporting (all experiments)

We also provide a batch validator that scans all experiment folders and writes a timestamped report under `tests/reports/` (ignored by Git):

```bash
python3 tests/validate_experiments_and_report.py \
  --experiments-dir pkg/carbon-aware/server-python/experiments \
  --nodes-file pkg/carbon-aware/nodes.yaml \
  --workloads-dir pkg/carbon-aware/workloads \
  --workloads-vanilla-dir pkg/carbon-aware/workloads-vanilla
```

Outputs:
- CSV: `tests/reports/placement_validation_report_<timestamp>.csv`
- TXT: `tests/reports/placement_validation_report_<timestamp>.txt`

### Performance Metrics

The orchestrator automatically tracks detailed performance metrics for each scheduling run, without needing any special flags (previously the `--perf-log` flag was required). These metrics are stored in CSV format in the `pkg/carbon-aware/server-python/performance_logs/` directory.

Performance tracking includes:
- Execution time (total and per-pod)
- Number of pods processed, placed, failed, and skipped
- Total and per-pod carbon emissions
- CPU and memory utilization
- Algorithm-specific metrics (iterations, steps)
- Number of scheduling options considered
- Region diversity information
- Hardware type distributions

The automatic performance tracking helps with:
- Comparing how the algorithm performs across multiple runs
- Comparing different algorithms
- Creating useful visualizations
- Understanding how placement decisions affect carbon emissions
- Tracking system performance over time
- Finding ways to improve

Each log file is named with the algorithm and timestamp (e.g., `heuristic_20250512_121534.csv`).

### Carbon Emissions Analysis

The repository includes comprehensive tools for analyzing and comparing carbon emissions across different scheduling algorithms. The main analysis tool is `analysis/carbon_emissions_comparison.py`, which provides detailed carbon footprint comparisons between heuristic, global-optimal, and vanilla scheduling approaches.

**Key Features:**
- **Unified Carbon Calculation**: All algorithms now use the same carbon emissions calculation methodology for fair comparison
- **Detailed Breakdown**: Analysis of operational vs. embodied carbon emissions
- **Per-Pod Analysis**: Individual pod carbon footprint calculations
- **Algorithm Comparison**: Direct comparison of carbon efficiency across scheduling approaches
- **Real Data Integration**: Uses actual node specifications and carbon intensity forecasts

**Running Carbon Emissions Analysis:**

```bash
# Navigate to the analysis directory
cd analysis

# Basic usage - automatically finds latest experiments
python carbon_emissions_comparison.py

# Specify custom experiment directories
python carbon_emissions_comparison.py --heuristic-dir experiments/heuristic_perf_log_session_20250617_154028 --vanilla-dir experiments/vanilla_20250620_101010

# Specify individual files for fine-grained control
python carbon_emissions_comparison.py --heuristic-perf experiments/heuristic_session.csv --global-optimal-placement experiments/global_optimal_placements.csv

# Custom output directory
python carbon_emissions_comparison.py --output-dir custom_analysis_results

# Results are saved to figures/Comparison/Carbon_Emissions_Analysis_TIMESTAMP/
```

**Command-Line Options:**
- `--heuristic-dir`, `--global-optimal-dir`, `--vanilla-dir`: Specify experiment directories
- `--heuristic-perf`, `--heuristic-placement`: Specify individual performance and placement files
- `--output-dir`: Custom output directory for results
- `--experiments-dir`: Base directory for experiment auto-discovery

**Analysis Outputs:**
- **Carbon Emissions Comparison Chart**: Visual comparison of total emissions across algorithms
- **Algorithm Efficiency Analysis**: Performance vs. carbon emissions trade-offs
- **Simple Comparison Plot**: Clean emissions per pod comparison
- **Comprehensive Summary Report**: Detailed text report with statistics and insights
- **Per-Pod Carbon Footprint**: Individual pod placement decisions and their carbon impact



### Visualizing Results

The repository includes visualization tools for analyzing performance data and pod placement schedules.

**1. Performance Metrics Visualization:**

```bash
# Navigate to the analysis directory
cd analysis

# Generate visualizations from a performance log file
python visualization.py --log-file ../pkg/carbon-aware/server-python/performance_logs/heuristic_YYYYMMDD_HHMMSS.csv
```

This script (`analysis/visualization.py`) generates various plots related to algorithm performance, such as execution time, carbon emissions, and scalability. For detailed instructions, see `analysis/VISUALIZATION_GUIDE.md`. Plots are saved in the `figures/` directory, typically within a subdirectory named after the log file or a custom comparison name.

**2. Pod Placement Schedule Visualization:**

```bash
# Basic usage
python3 analysis/schedule_visualization.py /path/to/your/placements_session.csv

# Example: Visualize the latest heuristic experiment
python3 analysis/schedule_visualization.py pkg/carbon-aware/server-python/experiments/heuristic_perf_log_session_YYYYMMDD_HHMMSS/heuristic_placements_session.csv

# Generate both individual and density plots
python3 analysis/schedule_visualization.py /path/to/placements.csv --mode all

# Custom output directory and run name
python3 analysis/schedule_visualization.py /path/to/placements.csv --output-dir custom_figures --run-name my_experiment
```

The visualization script creates:
- A matrix of pod placements across nodes and timeslots
- A heatmap of pod density
- Constraint satisfaction markers (if earliest timeslot is respected)

Visualization outputs are saved to the `figures/` directory in a timestamped folder.

**Additional Plot Generators:**

- Emissions vs Pods (total and per‑pod):
  ```bash
  python analysis/emissions_vs_pods_plot_generator.py
  ```
  Outputs saved under `figures/EmissionsVsPods/`.

- Success Rate vs Pods:
  ```bash
  python analysis/success_rate_plot_generator.py
  ```
  Outputs saved under `figures/SuccessRate/`.

- Heuristic Time Complexity (runtime vs pods and vs nodes):
  ```bash
  python analysis/heuristic_time_complexity_plot_generator.py
  ```
  Consumes precompute timing CSVs from `pkg/carbon-aware/server-python/experiments/` named like `precompute_timing_<START>-<HHMMSS>.csv`; outputs saved under `figures/TimeComplexity/`.

**3. CSV Tracking and Output Details:**

*   **CSV Tracking:**
    *   Both `HeuristicAlgorithm` and `GlobalOptimalAlgorithm` now log their pod placement decisions to CSV files.
    *   These CSVs are stored in timestamped, algorithm-specific subdirectories within the `analysis/` directory (e.g., `analysis/heuristic_YYYYMMDD_HHMMSS/heuristic_placements_YYYYMMDD_HHMMSS.csv`).
    *   Each directory also contains a symlink (e.g., `heuristic_placements.csv`) pointing to the latest run's CSV for easier access.
    *   The logged data includes `pod_id`, `node_id`, `start_slot`, `duration`, and for the global optimal algorithm, additional metrics like `cpu_request`, `ram_request`, `total_carbon_emissions`, and solver details.

*   **Helper Script (`run_schedule_visualization.sh`):**
    *   This shell script simplifies running the `schedule_visualization.py` script.
    *   It provides options to specify input CSVs (including glob patterns), output directory, visualization mode (`individual`, `density`, or `all`), and a name for the visualization run.

The comprehensive visualization toolkit allows for detailed analysis of both algorithm performance characteristics and specifics of pod placements across the cluster.



## Architecture

The carbon-aware orchestrator follows this architecture for making placement decisions:

```mermaid
flowchart TB
      subgraph Orchestration
            K["Kubernetes Scheduler Framework"] <--gRPC API--> C["Carbon-Aware Orchestrator"]
            P["Performance Tracking"] <-- Metrics --> C
            V["Visualization Tools"] <-- Analyzes --> P
      end
      
      subgraph Data Sources
            D["Carbon Intensity Data"]
            N["Node Metadata - Region - Hardware - Power Profile"]
            W["Workload Specifications - CPU/Memory - Duration"]
      end
      
      subgraph Analysis Flow
            CSV["Performance Logs (CSV)"] --> V
            V --> Figs["Visualization Outputs"]
            Figs --> Doc["Research Documents"]
      end
      
      C <--Reads--> D
      N --"Resource Info"--> C
      W --"Requirements"--> C
      
      C --"Returns Optimized Placements"--> K
      P --"Logs to CSV"--> CSV
      
      style P fill:#f9f,stroke:#333,stroke-width:2px
      style CSV fill:#bbf,stroke:#333,stroke-width:2px
      style V fill:#bfb,stroke:#333,stroke-width:2px
```

The diagram shows how performance tracking (pink) creates CSV logs (blue) that visualization tools (green) can analyze to help improve the orchestrator.

The architecture includes:
- **Performance tracking** for detailed algorithm metrics
- **Carbon emissions analysis** for comparing algorithmic carbon efficiency
- **Visualization tools** for analyzing placement decisions and performance characteristics
- **Multi-algorithm support** enabling comparison between heuristic, global-optimal, and vanilla approaches

## Contributing

Contributions to improve the carbon-aware scheduler are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgements

- Fondazione Bruno Kessler (FBK) for the original scheduler plugin
- Electricity Maps for granting access to their carbon intensity data
