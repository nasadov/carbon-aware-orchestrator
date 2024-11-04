# Placement algorithms and their interface

This repository contains (i) the definition of the interface (gRPC) exposed by the placement algorithms and (ii) the implementation of different placement algorithms. A placement algorithm ingests the status of the infrastructure and the workload to be placed/deployed on it and provides back the scores (for each pod and for each node) that will be used by [a custom scheduler plugin](https://gitlab.fbk.eu/fogatlas/scheduler-plugins) inside K8s Scheduling cycle.

## The IDL 

The definition of the interface is in the `./pkg/idl/idl.proto` file. It aims at modelling the snapshot of a k8s cluster in terms of infrastructure and workload deployed and the placement information in terms of i) scheduling time for each microservice and ii) scores for each pod and for each node. Such an IDL/GRPC interface allows to decouple the code written in FogAtlas (golang) from the code of the placement algorithms that could be written in (almost) any programming language.

**Note that the IDL definition on branch `feature/energy` is different and not backward compatible neither with the one in branch `feature/reschedule` nor with the one on branch `main`.**

## The algorithms 

Currently, the following algorithms have been implemented:
* Silly algorithm (currently the only one working on `feature/energy` branch)

The usage workflow is this:
1. The client initializes the algorithm calling the method `Init()`
1. The client provides the current status of the cluster to the algorithm and expects back the scores for each node and for each pod - method `CalculatePlacement()`.

### Silly algorithm

This is just for testing purposes. It comes with two implementation (golang and python).

The client (`.pkg/silly/client-go`) sends the following cluster status:
* Infrastructure
   * 2 Regions
      * 2 Nodes 
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

## Code generation 

Code generation is needed only if the IDL has been changed.

### Golang

Check [this quick start guide](https://grpc.io/docs/languages/go/quickstart/).

Recall that gRPC uses `protocol buffer`, Google’s open source mechanism for serializing structured data. Protocol buffer data is structured in `messages`. You define gRPC services in ordinary `.proto` files, with RPC method parameters and return types specified as protocol buffer messages.

You use the protocol buffer compiler `protoc` to generate the serialization and client/server code:
```
make generate-go
```    

### Python

Check [this quick start guide](https://grpc.io/docs/languages/python/quickstart/).

First of all, you need to create a python environment and activate it:

```
# Create an environment with `conda` and activate it.
# conda-forge is needed for k8s client package
conda config --append channels conda-forge

# create it
conda create -n grpc grpcio grpcio-tools python-kubernetes python=3.10

# activate it
conda activate grpc
```
Then:
```
make generate-py
```

## Test

Open a terminal and run the server. For example in case of `silly` algorithm with golang implementation:
```
cd ./pkg/silly/server-go
go run silly.go
```
If you want to run the python implementation do:
```
cd ./pkg/silly/server-python
# Activate conda environment (see above)
conda activate grpc
python silly.py
```

Open another terminal and run the client. In case of `silly`:
```
cd ./pkg/silly/client
go run client.go
```

check the output on both terminals.

## License

Copyright 2023 Fondazione Bruno Kessler.

Licensed under the Apache License, Version 2.0 (the “License”); you may not use this
file except in compliance with the License. You may obtain a copy of the License
[here](http://www.apache.org/licenses/LICENSE-2.0).

Unless required by applicable law or agreed to in writing, software distributed under
the License is distributed on an “AS IS” BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the License for the specific language governing permissions
and limitations under the License.
