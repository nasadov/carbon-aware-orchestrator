#!/usr/bin/env python3
"""
Simple test client to trigger global optimization initialization.
"""
import grpc
import logging
import sys
import os

# Add the current directory to the path to import the generated gRPC files
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import idl_pb2
import idl_pb2_grpc

def create_sample_request():
    """Create a sample placement request"""
    # Create a sample microservice
    microservice = idl_pb2.Microservice(
        name="test-service",
        replicas=1,
        cpu_required=idl_pb2.ResourceQuantity(value="100", format="m"),
        mem_required=idl_pb2.ResourceQuantity(value="128", format="Mi"),
        status=idl_pb2.MicroserviceStatus.TO_DEPLOY,
        duration_hours="1",
        deadline_hours="2"
    )
    
    # Create workload
    workload = idl_pb2.Workload(microservices=[microservice])
    
    # Create sample node
    node = idl_pb2.Node(
        name="test-node",
        cpu_used=idl_pb2.ResourceQuantity(value="0", format="m"),
        mem_used=idl_pb2.ResourceQuantity(value="0", format="Mi"),
        cpu_cap=idl_pb2.ResourceQuantity(value="1000", format="m"),
        mem_cap=idl_pb2.ResourceQuantity(value="1000", format="Mi"),
        region="us-west",
        subcategory="compute"
    )
    
    # Create infrastructure
    infrastructure = idl_pb2.Infrastructure(nodes=[node])
    
    # Create the complete request data
    request_data = idl_pb2.Data(workload=workload, infrastructure=infrastructure)
    
    return request_data

def main():
    """Main entry point"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Connect to the server
    logging.info("🔗 Connecting to server at localhost:50051...")
    channel = grpc.insecure_channel('localhost:50051')
    stub = idl_pb2_grpc.PlacementAlgorithmStub(channel)
    
    try:
        # Test connection first
        logging.info("🔧 Testing connection and initializing algorithm...")
        init_request = idl_pb2.AlgorithmName(name="global-optimal")
        stub.Init(init_request, timeout=60)
        logging.info("✅ Algorithm initialized")
        
        # Create and send a placement request
        logging.info("📝 Creating sample placement request...")
        request_data = create_sample_request()
        logging.info("📤 Sending placement request...")
        response = stub.CalculatePlacement(request_data, timeout=120)
        
        logging.info(f"✅ Received response with {len(response.placements)} placements")
        for placement in response.placements:
            logging.info(f"  - Microservice: {placement.microservice_name}")
            # Handle replica_scores properly
            if hasattr(placement, 'replica_scores') and placement.replica_scores:
                for replica_scores in placement.replica_scores:
                    for score in replica_scores.scores:
                        logging.info(f"    Node: {score.node}, Score: {score.score}")
            else:
                logging.info(f"    No replica scores available")
        
    except grpc.RpcError as e:
        logging.error(f"❌ gRPC error: {e.code()}: {e.details()}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        logging.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        channel.close()

if __name__ == "__main__":
    main()
