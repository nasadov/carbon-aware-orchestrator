# MILP Global Optimizer - Standalone Script

## Overview

This directory contains `milp_global_optimizer.py`, a standalone script that replicates the MILP-based global optimization from the carbon-aware orchestrator. The script reads all pods from workload files, nodes from the infrastructure configuration, and carbon intensity forecasts, then solves a global optimization problem to minimize total carbon emissions.

## Features

- **Complete MILP Implementation**: Uses PuLP to solve the Mixed-Integer Linear Programming problem
- **Real Data Processing**: Reads actual Kubernetes Deployment YAML files from the workloads directory
- **Carbon-Aware Optimization**: Minimizes total carbon emissions across all placements
- **Resource Constraints**: Respects CPU and RAM capacity limits for each node and timeslot
- **Deadline Constraints**: Ensures pods finish before their specified deadlines
- **CSV Output**: Saves results in the same format as the original algorithm

## Usage

```bash
python milp_global_optimizer.py \
    --workloads-dir ../pkg/carbon-aware/workloads \
    --nodes-file ../pkg/carbon-aware/nodes.yaml \
    --forecasts-file ../pkg/carbon-aware/server-python/all_forecasts.json \
    --output global_optimization_results.csv \
    --loglevel INFO
```

## Input Files

1. **Workload Files**: `timeslot_*.yaml` files containing Kubernetes Deployment manifests
2. **Nodes Configuration**: `nodes.yaml` with node specifications and carbon/power metadata
3. **Carbon Forecasts**: `all_forecasts.json` with carbon intensity data by region

## Output

The script generates a CSV file with the following columns:
- `pod_id`: Unique identifier for each pod
- `node_id`: Target node for placement
- `start_slot`: Starting timeslot for the pod
- `duration`: Pod execution duration in hours
- `cpu_request`: CPU requirement
- `ram_request`: RAM requirement in MB
- `total_carbon_emissions`: Carbon emissions for this placement (kgCO2e)
- `solver_status`: MILP solver status (e.g., "Optimal")
- `solver_iterations`: Number of solver iterations
- `solution_time_seconds`: Time taken to solve the optimization

## Recent Test Results

**Date**: May 25, 2025  
**Test Dataset**: 47 pods from 12 timeslot files  
**Infrastructure**: 5 nodes across 4 regions (DE, FR, ES, IT-NO)  
**Result**: ✅ **SUCCESS**

### Key Metrics:
- **Pods Placed**: 47/47 (100% success rate)
- **Total Carbon Emissions**: 82.63 kgCO2e
- **Solver Status**: Optimal
- **Solution Time**: 300.49 seconds (~5 minutes)
- **Valid Placements Found**: 1,045 options
- **Resource Constraints**: 220 constraints added
- **Nodes Utilized**: 3 out of 5 available nodes

### Resource Utilization:
- `node-0-de-iot`: Peak CPU=2.00, Peak RAM=2048MB (at capacity)
- `node-4-fr-iot`: Peak CPU=2.00, Peak RAM=2048MB (at capacity)  
- `node-1-fr-smartphone`: Peak CPU=0.95, Peak RAM=4096MB (partial utilization)

### Scheduling Horizon:
- **Timeslots Used**: 0 to 21 (out of 36 available)
- **Total Scheduling Window**: 22 hours of the 36-hour horizon

## Technical Details

### MILP Formulation:
- **Decision Variables**: Binary variables `x[pod_id, node_id, timeslot_id]`
- **Objective**: Minimize `Σ(emissions × x[pod, node, slot])`
- **Constraints**:
  - Each pod placed exactly once: `Σ(x[pod, node, slot]) = 1`
  - CPU capacity: `Σ(cpu_request × x[pod, node, slot]) ≤ node_cpu_capacity`
  - RAM capacity: `Σ(ram_request × x[pod, node, slot]) ≤ node_ram_capacity`
  - Deadline compliance: `start_slot + duration ≤ deadline_slot`

### Data Processing:
- **Kubernetes YAML Parsing**: Extracts CPU/RAM requests from Deployment manifests
- **Duration/Deadline Extraction**: Parses labels like `duration-1h` and `deadline-7h`
- **Carbon Forecast Integration**: Attaches regional carbon intensity data to nodes
- **Resource State Tracking**: Maintains available CPU/RAM for each node-timeslot combination

### Error Handling:
- Validates input files and data formats
- Handles missing carbon forecast data with fallbacks
- Provides detailed logging for debugging
- Graceful handling of infeasible placements

## Dependencies

- Python 3.7+
- PuLP (Linear Programming library)
- PyYAML (YAML parsing)
- Standard libraries: csv, json, logging, argparse, datetime

## Files

- `milp_global_optimizer.py`: Main script
- `global_optimization_results.csv`: Example output file
- `README_MILP_Global_Optimizer.md`: This documentation

## Comparison with Reference Implementation

This standalone script replicates the core MILP formulation from `/pkg/carbon-aware/server-python/carbon_aware/algorithms/global_optimal.py` while providing:

1. **Standalone Operation**: No dependency on the full carbon-aware server
2. **Real Data Integration**: Processes actual workload and infrastructure files
3. **Complete Pipeline**: Handles data loading, optimization, and result output
4. **Enhanced Logging**: Detailed progress and debugging information
5. **CSV Export**: Results in analysis-ready format

The script successfully demonstrates the global optimization approach for carbon-aware container scheduling.
