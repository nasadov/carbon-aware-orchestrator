# Carbon-Aware Orchestrator

This repository contains the implementation of a carbon-aware orchestrator that optimizes workload placement based on both operational and embodied carbon emissions. It integrates with K8s scheduling through a custom scheduler plugin, allowing for environmentally-conscious placement decisions.

This work is based on the original scheduler plugin developed by Fondazione Bruno Kessler (FBK).

## Table of Contents

- [Overview](#overview)
- [The IDL](#the-idl)
- [Repository Structure](#repository-structure)
- [The Algorithms](#the-algorithms)
  - [Carbon-Aware Algorithm](#carbon-aware-algorithm)
- [Carbon-Aware Data Model](#carbon-aware-data-model)
  - [Node Metadata](#node-metadata)
  - [Workload Specifications](#workload-specifications)
- [Carbon Emissions Model](#carbon-emissions-model)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Setup Environment](#setup-environment)
  - [Configure Test Infrastructure and Workloads](#configure-test-infrastructure-and-workloads)
  - [Generate Test Data](#generate-test-data)
  - [Run the Experiment](#run-the-experiment)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

The carbon-aware orchestrator considers multiple factors when placing workloads:

1. **Operational carbon emissions**: Based on regional carbon intensity forecasts and the power consumption model of each node
2. **Embodied carbon emissions**: Accounting for hardware manufacturing emissions amortized over device lifetime
3. **Node characteristics**: Including hardware type, region, and resource constraints
4. **Workload characteristics**: Including duration, resource requirements, and constraints

## The IDL 

The definition of the interface is in the `./pkg/idl/idl.proto` file. It aims at modelling the snapshot of a k8s cluster in terms of infrastructure and workload deployed and the placement information in terms of i) scheduling time for each microservice and ii) scores for each pod and for each node. Such an IDL/GRPC interface allows to decouple the code written in FogAtlas (golang) from the code of the placement algorithms that could be written in (almost) any programming language.

**Note that the IDL definition on branch `feature/energy` is different and not backward compatible neither with the one in branch `feature/reschedule` nor with the one on branch `main`.**

## Repository Structure

- `pkg/idl/`: gRPC interface definition for communication between the scheduler plugin and placement algorithms
- `pkg/silly/`: Implementation of the carbon-aware scheduling algorithm
   - `server-python/`: Python implementation of the carbon-aware scheduler
   - `client-go/`: Go client for testing the scheduler
   - `infra_workload_gen.py`: Tool for generating test infrastructure and workload data

## The algorithms 

Currently, the following algorithms have been implemented:
* Carbon-Aware algorithm (updated version of the "Silly" algorithm)

The usage workflow is this:
1. The client initializes the algorithm calling the method `Init()`
2. The client provides the current status of the cluster to the algorithm and expects back the scores for each node and for each pod - method `CalculatePlacement()`.

### Carbon-Aware algorithm

This algorithm optimizes workload placement based on carbon emissions.

The client (`.pkg/silly/client-go`) sends the following cluster status:
* Infrastructure
   * 4 Nodes 
* Workload
   * 4 Microservices: 
      * one RUNNING
      * one TO_SCHEDULE
      * one PENDING
      * one TO_DEPLOY

and expects back:
* Placements 
    * one placement for each microservice in status TO_DEPLOY
         * scores for this microservice for each node 
         * time to schedule for this microservice   

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

1. **Dynamic power model**: `idle + (max-active) * CPU_usage_ratio`
2. **Operational emissions**: `power * duration * carbon_intensity`
3. **Embodied emissions**: `(embodied_carbon / lifetime_hours) * duration`
4. **Total emissions**: Sum of operational and embodied emissions

## Getting Started

### Prerequisites

- Kubernetes cluster (v1.18+)
- Go 1.16+ (for building)
- Python 3.8+ (for Python implementation)

### Setup Environment

    ```bash
    # Clone the repository
    git clone https://github.com/nasadov/carbon-aware-orchestrator.git
    cd carbon-aware-orchestrator
    
    # Install required Python dependencies
    pip install -r pkg/silly/server-python/requirements.txt
    ```

### Configure Test Infrastructure and Workloads

    - Open `infra-workload-config.yaml` and adjust the configuration:
       ```yaml
       nodes:
         num_nodes: 8
         regions:
            - DE
            - FR
            - ES
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
       
       workload:
         num_timeslots: 12
         poisson_lambda: 16
         durations:
            - 1  # hours
            - 3
            - 6
         cpu_options:
            - 1000m
            - 500m
            - 250m
         mem_options:
            - 1Gi
            - 512Mi
       ```

### Generate Test Data

    ```bash
    # Generate infrastructure and workload data based on your config
    python pkg/silly/infra_workload_gen.py
    ```
    This will create:
    - `nodes.yaml`: Infrastructure definitions with hardware characteristics
    - `workloads/timeslot_*.yaml`: Directory containing workload definitions for each timeslot

### Run the Experiment

    ```bash
    # Start the Python server
    cd pkg/silly/server-python
    python carbon-aware.py &
    
    # In a new terminal, run the Go client with the test data
    cd pkg/silly/client-go
    go run client.go
    ```

## Architecture

The carbon-aware orchestrator follows this architecture for making placement decisions:

```mermaid
flowchart TB
    subgraph Orchestration
        K["Kubernetes Scheduler Framework"] <--gRPC API--> C["Carbon-Aware Orchestrator"]
    end
    
    subgraph Data Sources
        D["Carbon Intensity Data"]
        N["Node Metadata - Region - Hardware - Power Profile"]
        W["Workload Specifications - CPU/Memory - Duration"]
    end
    
    C <--Reads--> D
    N --"Resource Info"--> C
    W --"Requirements"--> C
    
    C --"Returns Optimized Placements"--> K
```

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
