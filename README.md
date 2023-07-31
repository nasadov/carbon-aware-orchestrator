# Placement algorithms and their interface

This repository contains (i) the definition of the interface between the [FogAtlas controller](https://gitlab.fbk.eu/fogatlas/fadepl-controller) and the placement algorithms and (ii) the implementation of the different placement algorithms. A placement algorithm ingests the status of the infrastructure and the workload to be placed/deployed on it and provides back the scores (for each microservice for each node) that will be used by [a custom scheduler plugin](https://gitlab.fbk.eu/fogatlas/scheduler-plugins) inside K8s Scheduling cycle.

## The IDL 

The definition of the interface is in the `./pkg/idl/idl.proto` file. It aims at decoupling the code written for the FogAtlas controller (in golang) from the code of the placement algorithm that can be written in (almost) whatever programming language. 

The workflow is this:
1. The client initializes the algorithm calling the method `Init()`
1. The client provides the current status of the cluster to the algorithm and expects back the scores for each node and for each microservice - method `CalculatePlacement()`.

## The algorithms 

Currently, the following algorithms have been implemented:
* Silly algorithm
* Tradeoffboard algorithm

### Silly algorithm

This is jsut for test. Silly algorithm assigns scores to the nodes based on the alphabetical order of their names. We have two implementation: `server-go` and `server-python`.

The client (`.pkg/silly/client-go`) sends the following cluster status:
* Infrastructure
   * 3 Regions
      * 3 Nodes 
      * 3 ExternalEndpoints
   * 3 Links
* Workload
   * 3 Applications
      * 3 Microservices
      * 3 DataFlows
      * 3 Placements (one for each Microservice) 
         * 3 Nodes Selected (one for each replica)

and expects back:
* 3 Applications
   * 3 Placements (one for each Microservice)
      * One Replica
      * 9 Scores (one for each node that are 3 for each region)

Moreover, the application and the microservices inside each application are **ordered** so as to specify
which one should be deployed first. The order is the reversed wrt the one passed by the client to the server.

### TradeoffBoard algorithm

[TradeoffBoard algorithm](https://github.com/stfbk/CryptoAC/tree/CryptoAC_Demo_FogAtlas) is able to place the microservices of a cloud-native application according to the imposed requirements in terms of security and resource availability. Also in this case we have a client written in go (only for testing purposes) and a server (written in Python) that implements the algorithm. 

## Code generation 

### Golang

Check [this quick start guide](https://grpc.io/docs/languages/go/quickstart/).

Recall that gRPC uses `protocol buffer`, Google’s open source mechanism for serializing structured data. Protocol buffer data is structured in `messages`. You define gRPC services in ordinary `.proto` files, with RPC method parameters and return types specified as protocol buffer messages.

You use the protocol buffer compiler `protoc` to generate the serialization and client/server code. In case of `silly` algorithm:
```
protoc -I./pkg/idl --go_out=./pkg/silly/client --go_out=./pkg/silly/server-go --go-grpc_out=./pkg/silly/client --go-grpc_out=./pkg/silly/server-go  ./pkg/idl/idl.proto
```    

### Python

Check [this quick start guide](https://grpc.io/docs/languages/python/quickstart/).

In case of `silly` algorithm:

```
# Create an environment with `conda` and activate it.
# Needed for k8s client package
conda config --append channels conda-forge

# create it
conda create -n grpc grpcio grpcio-tools python-kubernetes python=3.10

# activate it
conda activate grpc

# generate the code
python -m grpc_tools.protoc -I ./pkg/idl/ --python_out=./pkg/silly/server-python --pyi_out=./pkg/silly/server-python --grpc_python_out=./pkg/silly/server-python ./pkg/idl/idl.proto
```

### Docker container

Generate a Docker image with the following command (example for `silly-go`)
```
docker build -f ./pkg/silly/silly-go/Dockerfile -t gitlab-registry.fbk.eu/fogatlas/algorithms/silly-go .
```

Push it with:
```
docker push gitlab-registry.fbk.eu/fogatlas/algorithms:silly-go
```

These commands are also inside the `Makefile`.

## Test

Open a terminal and rum the server. For example in case of silly algorithm:
```
cd ./pk/silly/silly.go
go run server.go
```

Open another terminal and run the client. In case of silly:
```
cd ./pkg/silly/client
go run client.go
```

## License

Copyright 2023 FBK CREATE-NET

Licensed under the Apache License, Version 2.0 (the “License”); you may not use this
file except in compliance with the License. You may obtain a copy of the License
[here](http://www.apache.org/licenses/LICENSE-2.0).

Unless required by applicable law or agreed to in writing, software distributed under
the License is distributed on an “AS IS” BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the License for the specific language governing permissions
and limitations under the License.
