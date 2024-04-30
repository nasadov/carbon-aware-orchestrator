#
# Copyright 2023 Fondazione Bruno Kessler.
#  
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#  
#      http://www.apache.org/licenses/LICENSE-2.0
#  
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


from concurrent import futures
import logging
import signal

import grpc
import idl_pb2
import idl_pb2_grpc
from google.protobuf import empty_pb2
from dataclasses import dataclass
from kubernetes import utils as k8sutils

@dataclass
class Algorithm:
    name: str
    initialized: bool 

algo = Algorithm('unknown',False)

class PlacementAlgorithm(idl_pb2_grpc.PlacementAlgorithmServicer):

    def Init(self, request, context):
        logging.debug(f"Init called: received: {request.name}")
        global algo
        algo.name = request.name 
        algo.initialized = True
        return empty_pb2.Empty()
        
        
    # It just return a score per node per microservice according to the order of the nodes:
    # score = (indexNode + 1)
    def CalculatePlacement(self, request, context):
        global algo
        logging.debug(f"CalculatePlacement called: received {request}")
        logging.info(f"Going to calculate the placement of with algorithm {algo.name}")
        if algo.initialized == False:
            logging.error("SillyAlgorithm not intitialized. Unable to proceed.")
            raise grpc.RpcError(grpc.StatusCode.FAILED_PRECONDITION,"silly algorithm not intitialized")
        
        inInfra = request.infrastructure
        inWorkload = request.workload
        outWorkload = idl_pb2.Workload()
        nodeList = []
        for reg in inInfra.regions:
            nodeList.extend(reg.nodes)
        
        # An example on how to convert a pd.ResourceQuantity to a resource.Quantity in python
        node = nodeList[0]
        q = k8sutils.parse_quantity(node.cpu_used.value)
        logging.debug(f"Node cpu used is: {q}")
        # and the memory ...
        q = k8sutils.parse_quantity(node.mem_used.value)
        logging.debug(f"Node mem used is: {q}")
        posMs = 0
        for ms in inWorkload.microservices:
            placement = idl_pb2.Placement()
            placement.microservice_name = ms.name
            placement.order = len(inWorkload.microservices) - posMs 
            posMs += 1
            replicaScore = idl_pb2.ReplicaScores()
            counter = 1
            for node in nodeList:
                score = idl_pb2.Score()
                score.node = node.name
                score.score = counter
                counter += 1
                replicaScore.scores.append(score)
            placement.replica_scores.append(replicaScore)
            outWorkload.placements.append(placement)  

        return outWorkload    

def serve():
    port = '50051'
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    idl_pb2_grpc.add_PlacementAlgorithmServicer_to_server(PlacementAlgorithm(), server)
    server.add_insecure_port('[::]:' + port)
    server.start()
    logging.info("Server started, listening on " + port)
    server.wait_for_termination()

def handler(signum, frame):
    logging.info('Got CTRL+C')
    exit (0)

signal.signal(signal.SIGINT, handler)

def main():
    logging.basicConfig(level=logging.DEBUG)
    serve()

if __name__ == '__main__':
    main()
    
