import logging
import time
from datetime import datetime, timedelta
from carbon_aware.algorithms import get_algorithm
from carbon_aware.models import CarbonAwarePod, CarbonAwareFlavour, CarbonAwareTimeslot

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format="%(asctime)s [%(levelname)s] %(message)s")

def test_global_algorithm():
    # Create test nodes with different carbon intensities
    flavours = [
        CarbonAwareFlavour(
            id="node1", 
            embodiedCarbon=1000.0, 
            lifetime=87600.0,
            totalCpu=4000.0,   # 4 cores
            totalRam=8192.0,   # 8GB
            totalStorage=10000.0,
            forecast={i: 100.0 if i < 12 else 200.0 for i in range(48)}  # Lower carbon in first 12 hours
        ),
        CarbonAwareFlavour(
            id="node2", 
            embodiedCarbon=1200.0, 
            lifetime=87600.0,
            totalCpu=4000.0,   
            totalRam=8192.0,   
            totalStorage=10000.0,
            forecast={i: 200.0 if i < 12 else 100.0 for i in range(48)}  # Lower carbon in last 12 hours
        )
    ]
    
    # Create 24 hour-long timeslots
    now = datetime.now()
    timeslots = []
    for i in range(24):
        start_time = now + timedelta(hours=i)
        ts = CarbonAwareTimeslot(id=i, start_time=start_time, length=1)
        timeslots.append(ts)
    
    # Available resources - nodes are empty
    leftover_cpu = {
        "node1": {i: 4000.0 for i in range(48)},
        "node2": {i: 4000.0 for i in range(48)}
    }
    
    leftover_ram = {
        "node1": {i: 8192.0 for i in range(48)},
        "node2": {i: 8192.0 for i in range(48)}
    }
    
    # Create test pods
    test_pods = [
        CarbonAwarePod(
            id="test-pod-1",
            deadline_hours=12.0,
            duration=2.0,
            powerConsumption=100.0,
            cpuRequest=500.0,  # 500m
            ramRequest=512.0,  # 512MB
            storageRequest=0
        ),
        CarbonAwarePod(
            id="test-pod-2",
            deadline_hours=15.0,
            duration=3.0,
            powerConsumption=150.0,
            cpuRequest=1000.0,
            ramRequest=1024.0,
            storageRequest=0
        )
    ]
    
    # Get the algorithm instance
    algorithm = get_algorithm("global-optimal")
    
    # Add pods to algorithm's pending queue
    algorithm.pending_pods = test_pods.copy()
    
    # Force a global optimization run
    algorithm.solve_global_optimization(
        flavours, timeslots, leftover_cpu, leftover_ram, max_time_slots=24
    )
    
    # Check results
    print("\n=== Global Optimal Algorithm Test Results ===")
    if algorithm.global_solution:
        print(f"Found {len(algorithm.global_solution)} placements:")
        for pod_id, (node_id, ts_id, emissions) in algorithm.global_solution.items():
            print(f"  Pod {pod_id}: Node={node_id}, Timeslot={ts_id}, Emissions={emissions:.2f}")
    else:
        print("No placements found!")
        
    # Try finding placement for each pod
    for i, pod in enumerate(test_pods):
        print(f"\nTesting find_placement for {pod.id}:")
        flv, ts, emissions = algorithm.find_placement(
            pod, flavours, timeslots, leftover_cpu, leftover_ram
        )
        if flv and ts:
            print(f"  ✅ Placed on {flv.id} at timeslot {ts.id}, emissions: {emissions:.2f}")
        else:
            print(f"  ❌ No placement found")

if __name__ == "__main__":
    test_global_algorithm()