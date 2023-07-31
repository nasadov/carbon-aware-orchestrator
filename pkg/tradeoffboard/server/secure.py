from concurrent import futures
import logging
import signal
import argparse

import grpc
import idl_pb2
import idl_pb2_grpc
from google.protobuf import empty_pb2
from dataclasses import dataclass
from kubernetes import utils as k8sutils






# Import from TradeOffBoard the function that returns the ranked list of deployments
from tradeoffboard import rankArchitectures_FogAtlas

# Import from Controller the function that checks whether an architecture is
# feasible according to the (computational and network) resources available
from controller import checkIfArchitectureIsFeasible_FogAtlas

# Invoke TradeOffBoard's algorithm, which returns a ranked list of possible 
# deployments (intuitively, the best deployment is be the first one in the list).
# Each deployment is a list of microservice-region pairs; for instance:
# (('RM', 'cloudregion'), ('MM', 'cloudregion'), ('PR', 'onpremiseregion'), ('DM', 'onpremiseregion')).
# Then, invoke the Controller function to identify the first architecture in the
# list that is also "feasible" according to the (computational and network) resources available. 
# Finally, compute a placement and return it.
#
# The `request` parameter is the same as the one in the `CalculatePlacement` function
def getOptimalDeployment(request):
    
    # ==== ASSUMPTION ====
    # Below, we assume that we have a single application, i.e., CryptoAC
    applicationCryptoAC = request.workload.applications[0]
    infrastructure = request.infrastructure

    # 1. rank architectures
    microservices = set()
    for microservice in applicationCryptoAC.microservices:
        microservices.add(microservice.name)

    regions = set()
    for region in infrastructure.regions:
        regions.add(region.id)

    rankedArchitectures = rankArchitectures_FogAtlas(microservices, regions) 

    # 2. find first feasible architecture architectures
    for architecture in rankedArchitectures:
        checkResult = checkIfArchitectureIsFeasible_FogAtlas(applicationCryptoAC, infrastructure, architecture)
        if (checkResult != False):
            # If we got here, it means that "checkResult" contains an architecture 
            # as a list of (microservice, nodeRegionName) pairs
            outWorkload = idl_pb2.Workload()
            application = idl_pb2.Application()

            # The list of all nodes (to then assign scores to these nodes)
            nodeList = []
            for region in infrastructure.regions:
                nodeList.extend(region.nodes)

            posMs = 0
            for microserviceNodePair in checkResult:
                placement = idl_pb2.Placement()
                placement.microservice_name = microserviceNodePair[0]
                placement.order = len(checkResult) - posMs 
                posMs += 1
                placement.must_reschedule = True
                replicaScore = idl_pb2.ReplicaScores()

                # Give score to each node for the current microservice
                for node in nodeList:
                    score = idl_pb2.Score()
                    score.node = node.name
                    if (node.name == microserviceNodePair[1]):
                        score.score = 100
                    else:
                        # TODO set score to 1 since, if we set score to 0, the "score"
                        # TODO field does not appear in the JSON output workflow
                        score.score = 1
                    replicaScore.scores.append(score)
                placement.replica_scores.append(replicaScore)
                application.placements.append(placement)  

            application.name = applicationCryptoAC.name  
            application.order = 1
            outWorkload.applications.append(application)
            return outWorkload 

    return None


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
        
        
    # It just return a score per node per microservice according to the order the nodes:
    # score = (indexNode + 1)
    def CalculatePlacement(self, request, context):


        # BELOW, NEW CODE
        logging.debug(f"CalculatePlacement (CryptoAC) called: received {request}")
        placement = getOptimalDeployment(request)
        if (placement == None):
            # TODO is this the correct approach?
            raise grpc.RpcError(grpc.StatusCode.NOT_FOUND, "No architecture is feasible")
        else:
            return placement

def serve():
    
    # parge arguments and set log level and port
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default='50051',help='Port number')
    parser.add_argument('--log', choices=['debug', 'info', 'warning', 'error','critical'], default='debug', help='Logging level')
    args = parser.parse_args()
    levels = {
    'critical': logging.CRITICAL,
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
    }
    level = levels.get(args.log.lower())
    logging.basicConfig(level=level)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    idl_pb2_grpc.add_PlacementAlgorithmServicer_to_server(PlacementAlgorithm(), server)
    server.add_insecure_port('[::]:' + args.port)
    server.start()
    logging.info("Server started, listening on " + args.port)
    server.wait_for_termination()

def handler(signum, frame):
    logging.info('Got CTRL+C')
    exit (0)

signal.signal(signal.SIGINT, handler)

def main():
    serve()

if __name__ == '__main__':
    main()
    
