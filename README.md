# Carbon-Aware Kubernetes Scheduler

This repository contains the implementation of a carbon-aware Kubernetes scheduler that optimizes workload placement based on both operational and embodied carbon emissions. It integrates with K8s scheduling through a custom scheduler plugin, allowing for environmentally-conscious placement decisions.

This work is based on the original scheduler plugin developed by Fondazione Bruno Kessler (FBK).

## Overview

The carbon-aware scheduler considers multiple factors when placing workloads:

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

This algorithm optimizes workload placement based on carbon emissions. It comes with implementations in golang and python.

The client (`pkg/silly/client-go`) sends the cluster status and expects back:
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
