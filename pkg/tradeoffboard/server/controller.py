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
#


import logging
import copy

from kubernetes import utils as k8sutils

# Check feasibility of the given architecture according to the given infrastructure
# "architecture" is a list like the following
# (('RM', 'cloudregion'), ('MM', 'cloudregion'), ('PR', 'onpremiseregion'), ('DM', 'onpremiseregion'))
#
# The output is either False (error symbol) or an assignment of microservices with nodes
def checkIfArchitectureIsFeasible_FogAtlas(applicationCryptoAC, infrastructure, architecture):

    # To check whether the given architecture is feasible, below we check the
    # computational and network resources available in the infrastructure (i.e., 
    # cpu_used, mem_used, bandwidth). As we modify the values of these resources 
    # during the process, we need keep the values in temporary structures: it is 
    # simpler to manage to addition/subtraction of amount of resources. 
    nodesCPUUsed = {}
    nodesRAMUsed = {}
    for region in infrastructure.regions:
        for node in region.nodes:
            nodesCPUUsed[region.id + node.name] = k8sutils.parse_quantity(node.cpu_used.value)
            nodesRAMUsed[region.id + node.name] = k8sutils.parse_quantity(node.mem_used.value)
    
    logging.info("[CONTROLLER] Checking architecture " + str(architecture))

    # The return value, an array of microservice-node pairs
    microservicesAndNodes = []


    # ==== ASSUMPTION ====
    # Each microservice of the CryptoAC application has only one replica. If this was not
    # the case, we should modify the code below, even though perhaps not the logic (ideally,
    # we just need to understand how to iterate over microservices to check the feasibility 
    # of the deployment, and consider that requirements on CPU and memory should be moltiplied
    # for each replica --- and that each replica can be deployed into a different node)

    # Check that regions have (nodes that have) enough 
    # computational resources to host microservices
    for microservice in applicationCryptoAC.microservices:

        logging.debug("[CONTROLLER] Controlling resources for microservice " + microservice.name)

        # each "microservice" is a dict like the following
        # {
        #     "name":"PR",
        #     "region_recommended":"onpremiseregion",
        #     "cpu_required":{
        #         "value":"100m",
        #         "format":"DecimalSI"
        #     },
        #     "mem_required":{
        #         "value":"100M",
        #         "format":"DecimalSI"
        #     }
        # }

        # These are the requirements on the resources that must be 
        # available on a node to be able to deploy the microservice.
        # Each requirement is a dictionary with two fields, i.e.,
        # "value" (e.g., "100m") and "format" (e.g., "DecimalSI")
        requirementCPU = microservice.cpu_required
        requirementRAM = microservice.mem_required
        
        # These are all the regions where a microservice has to 
        # be deployed (e.g., ["cloudregion", "onpremiseregion"])
        microserviceRegions = [assignment[1] for assignment in architecture if assignment[0] == microservice.name]

        # "microserviceRegion" is the id of a region (e.g., ""cloudregion"")
        for microserviceRegion in microserviceRegions:
            
            logging.debug("[CONTROLLER] Controlling resources for region " + microserviceRegion)

            # K8s follows a "rolling update" strategy, meaning that old microservice instances 
            # are kept alive and running until new instances are ready. As such, below we must
            # consider "cpu_used" and "mem_used"  as they are, i.e., without substracting the
            # cpu and memory used by old CryptoAC microservice instances. The same applies even
            # if a microservice instance stays on the same node (except, perhaps, in the latest
            # version of K8s).

            # The list of all nodes in the given "microserviceRegion"
            # A region is a dict like the following
            # {
            #     "id":"onpremiseregion",
            #     "location":"onpremiselocation",
            #     "tier":1,
            #     "extendpoint_ids":[
            #     "E_0",
            #     "E_1",
            #     "E_2"
            #     ],
            #     "nodes": [
            #         {
            #             "name":"onpremisenode",
            #             "cpu_used":{
            #                 "value":"100m",
            #                 "format":"DecimalSI"
            #             },
            #             "mem_used":{
            #                 "value":"100M",
            #                 "format":"DecimalSI"
            #             },
            #             "cpu_cap":{
            #                 "value":"100m",
            #                 "format":"DecimalSI"
            #             },
            #             "mem_cap":{
            #                 "value":"100M",
            #                 "format":"DecimalSI"
            #             }
            #         }
            #     ]
            # }
            nodesRegion = [region.nodes for region in infrastructure.regions if region.id == microserviceRegion][0]

            # ==== ASSUMPTION ====
            # Below, we assume that each region has one node only. If this was not the case, we
            # should solve another optimization problem (similar to the Knapsack problem), i.e.,
            # find a feasible assignment of microservices to nodes in polynomial time (more details
            # on the paper).
            nodeRegion = nodesRegion[0]
            nodeRegionName = nodeRegion.name

            logging.debug("[CONTROLLER] Region " + microserviceRegion + " has 1 node named " + nodeRegionName)

            usedCPU  = nodesCPUUsed[microserviceRegion + nodeRegionName]
            capCPU  = k8sutils.parse_quantity(nodeRegion.cpu_cap.value)
            requiredCPU = k8sutils.parse_quantity(requirementCPU.value)
            usedRAM = nodesRAMUsed[microserviceRegion + nodeRegionName]
            capRAM = k8sutils.parse_quantity(nodeRegion.mem_cap.value)
            requiredRAM = k8sutils.parse_quantity(requirementRAM.value)
            availableCPU = capCPU - usedCPU
            availableRAM = capRAM - usedRAM
            
            logging.debug("[CONTROLLER] Node " + microserviceRegion + nodeRegionName + " has available CPU " + str(availableCPU) + ", required CPU is " + str(requiredCPU))
            if (availableCPU > requiredCPU and availableRAM > requiredRAM):
                nodesCPUUsed[microserviceRegion + nodeRegionName] = usedCPU + requiredCPU
                nodesRAMUsed[microserviceRegion + nodeRegionName] = usedRAM + requiredRAM
            else:
                # The given architecture is not feasible
                return False
            
            # If we got here, it means that we can place the microservice in the node
            microservicesAndNodes.append((microservice.name, nodeRegionName))
            

    # Check that links existing between regions have enough 
    # network resources to host microservices that communicate
    # A data flow is a dict like the following
    # {
    #     "name":"dfPRMM",
    #     "bandwidth_required":{
    #         "value":"100M",
    #         "format":"DecimalSI"
    #     },
    #     "latency_required":{
    #         "value":"100m",
    #         "format":"DecimalSI"
    #     },
    #     "source_id":"PR",
    #     "destination_id":"MM"
    # }

    linksBandwidthUsed = {}
    for link in infrastructure.links:
        if (link.bandwidth_used.value is None) or (link.bandwidth_used.value == ""):
            link.bandwidth_used.value = "0"
        linksBandwidthUsed[link.endpoint_a + link.endpoint_b] = k8sutils.parse_quantity(link.bandwidth_used.value)

    for data_flow in applicationCryptoAC.data_flows:

        numberOfVerticesInWalk = len(data_flow.vertices)
        logging.debug("[CONTROLLER] Controlling network resources for data flow " + data_flow.name)
        logging.debug("[CONTROLLER] Data flow " + data_flow.name + " has " + str(numberOfVerticesInWalk) + " vertices")

        # These are the requirements on the network resources that must be 
        # available on links among regions to be able to deploy the given
        # application considering the current data flow (i.e., walk)
        # Each requirement is a dictionary with two fields, i.e.,
        # "value" (e.g., "100m") and "format" (e.g., "DecimalSI")
        requirementBandwidth = data_flow.bandwidth_required
        requirementLatency = data_flow.latency_required

        # For each pair of vertices in the data flow, we sum the latency
        # between the regions hosting the services. At the end, we will
        # check that the cumulativeLatency is less than the requirementLatency
        cumulativeLatency = 0

        for i in range(numberOfVerticesInWalk - 1):
            sourceMicroservice = data_flow.vertices[i]
            destinationMicroservice = data_flow.vertices[i + 1]

            # Regions where the source and the destination microservices are deployed
            sourceMicroserviceRegions = [assignment[1] for assignment in architecture if assignment[0] == sourceMicroservice]
            destinationMicroserviceRegions = [assignment[1] for assignment in architecture if assignment[0] == destinationMicroservice]

            # We consider the worst latency that there can be between the regions
            worstLatency = "0"

            for sourceMicroserviceRegion in sourceMicroserviceRegions:
                for destinationMicroserviceRegion in destinationMicroserviceRegions:

                    logging.debug("[CONTROLLER] Controlling resources from region " + sourceMicroserviceRegion + " to region " + destinationMicroserviceRegion)

                    # If the region is the same, we assume infinite bandwidth and zero latency
                    if (sourceMicroserviceRegion != destinationMicroserviceRegion):

                        # The links between the region hosting the source microservice and the region
                        # hosting the destination microservice. Ideally, there should be one link only 
                        # between two regions
                        linksBetweenRegions = [link for link in infrastructure.links if ((link.endpoint_a == sourceMicroserviceRegion and link.endpoint_b == destinationMicroserviceRegion) or ((link.endpoint_a == destinationMicroserviceRegion and link.endpoint_b == sourceMicroserviceRegion)))]
                        if (len(linksBetweenRegions) != 1):
                            raise ValueError('No link or more than one links between regions ' + sourceMicroserviceRegion + ' and ' + destinationMicroserviceRegion + ' (' + str(len(linksBetweenRegions)) + ')')
                        
                        # A link is a dict like the following
                        # {
                        #     "endpoint_a":"onpremiseregion",
                        #     "endpoint_b":"cloudregion",
                        #     "bandwidth":{
                        #         "value":"100M",
                        #         "format":"DecimalSI"
                        #     },
                        #     "latency":{
                        #         "value":"100m",
                        #         "format":"DecimalSI"
                        #     }
                        # }
                        linkBetweenRegions = linksBetweenRegions[0]
                            
                        usedBandwidth = linksBandwidthUsed[linkBetweenRegions.endpoint_a + linkBetweenRegions.endpoint_b]
                        availableBandwidth = k8sutils.parse_quantity(linkBetweenRegions.bandwidth.value) - usedBandwidth
                        requiredBandwidth = k8sutils.parse_quantity(requirementBandwidth.value)
                        

                        if (k8sutils.parse_quantity(worstLatency) < k8sutils.parse_quantity(linkBetweenRegions.latency.value)):
                            worstLatency = linkBetweenRegions.latency.value

                        if (requiredBandwidth < availableBandwidth):
                            linksBandwidthUsed[linkBetweenRegions.endpoint_a + linkBetweenRegions.endpoint_b] = usedBandwidth + requiredBandwidth
                        else:
                            # The given architecture is not feasible
                            logging.debug("[CONTROLLER] Not enough bandwidth")
                            return False
                        
            # Cumulate latency
            cumulativeLatency = cumulativeLatency + k8sutils.parse_quantity(worstLatency)

        # At the end of the walk, check that the cumulated latency is less than the required latency
        if (cumulativeLatency > k8sutils.parse_quantity(requirementLatency.value)):
            # The given architecture is not feasible
            logging.debug("[CONTROLLER] Too much latency")
            return False


    # If we got here, it means that the architecture is feasible
    logging.info("[CONTROLLER] Found feasible architecture: " + str(microservicesAndNodes))
    return microservicesAndNodes
