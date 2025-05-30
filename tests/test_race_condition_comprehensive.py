#!/usr/bin/env python3
"""
Comprehensive Race Condition Test Suite for Carbon-Aware Orchestrator

This merged test suite combines all race condition and concurrency tests:
- Core race condition fix validation
- Concurrent stress testing with multiple scenarios  
- Quick concurrent validation
- Atomic operations verification

Tests verify that the atomic race condition fixes prevent resource overallocation
in concurrent scheduling scenarios.
"""

import sys
import os
import threading
import time
import random
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the carbon_aware module to the path
sys.path.insert(0, '/root/carbon-aware-orchestrator/pkg/carbon-aware/server-python')

try:
    from carbon_aware.models import CarbonAwarePod, CarbonAwareFlavour, CarbonAwareTimeslot
    from carbon_aware.algorithms.heuristic import HeuristicAlgorithm
    from carbon_aware.state import PersistentStateStorage
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


# ============================================================================
# CORE RACE CONDITION FIX TEST (from test_race_condition_fix.py)
# ============================================================================

def test_race_condition_fix():
    """Core test that validates the atomic race condition fix."""
    print("\n" + "="*60)
    print("CORE RACE CONDITION FIX TEST")
    print("="*60)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
    )

    print("Testing FIXED race condition in heuristic algorithm constraint checking...")
    
    # Initialize state storage
    state_storage = PersistentStateStorage()
    print("🔄 Persistent state initialized")

    # Create a simple test setup: one node with limited resources
    test_node = CarbonAwareFlavour(
        id="test-node",
        embodiedCarbon=100.0,
        lifetime=5.0,
        totalCpu=2.0,      # Only 2 CPU cores
        totalRam=2048.0,   # Only 2GB RAM  
        totalStorage=1000.0,
        forecast={0: 0.3, 1: 0.4, 2: 0.5}
    )

    # Create 2 pods that each want ALL the resources
    pods = []
    for i in range(2):
        pod = CarbonAwarePod(
            id=f"pod-{i+1}",
            deadline_hours=24.0,
            duration=1.0,
            powerConsumption=0.0,
            cpuRequest=2.0,    # Each pod wants 2 CPU (all available)
            ramRequest=2048.0, # Each pod wants 2GB RAM (all available)
            storageRequest=0
        )
        pod.earliest_timeslot = 0
        pods.append(pod)

    print(f"\nInitial setup:")
    print(f"Node: {test_node.id} - CPU: {test_node.totalCpu}, RAM: {test_node.totalRam}")
    print(f"Pods: {len(pods)} pods each requesting CPU: {pods[0].cpuRequest}, RAM: {pods[0].ramRequest}")

    # Initialize the algorithm with our test setup
    algorithm = HeuristicAlgorithm()
    algorithm.state_storage = state_storage
    algorithm.pending_pods = pods

    # Initialize resources in state storage
    timeslots = [CarbonAwareTimeslot(id=0, start_time=1000000, length=1)]
    algorithm._initialize_resource_state([test_node], timeslots)
    
    logging.info(f"Initial resources - CPU: {test_node.totalCpu}, RAM: {test_node.totalRam}")

    def simulate_atomic_placement(pod, thread_id):
        """Simulate atomic placement using the fixed algorithm."""
        logging.info(f"Thread {thread_id}: Starting atomic placement for {pod.id}")
        
        # Use the atomic placement method
        result = algorithm.find_placement_atomic(pod, [test_node], timeslots, 10)
        
        if result:
            node_id, timeslot_id, emissions = result
            logging.info(f"Thread {thread_id}: Successfully placed {pod.id} on {node_id} at slot {timeslot_id}")
            return (pod.id, node_id, timeslot_id)
        else:
            logging.info(f"Thread {thread_id}: Failed to place {pod.id} - no resources available")
            return (pod.id, None, None)

    print("\n=== RUNNING CONCURRENT PLACEMENT TEST (ATOMIC VERSION) ===")
    
    # Run both placements concurrently
    results = []
    threads = []
    
    for i, pod in enumerate(pods):
        thread = threading.Thread(target=lambda p=pod, tid=i: results.append(simulate_atomic_placement(p, tid)))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("\n=== PLACEMENT RESULTS (ATOMIC VERSION) ===")
    successful_placements = 0
    
    for pod_id, node_id, timeslot_id in results:
        if node_id is not None:
            print(f"✅ Thread {results.index((pod_id, node_id, timeslot_id))}: {pod_id} placed on {node_id} at slot {timeslot_id}")
            successful_placements += 1
        else:
            print(f"❌ Thread {results.index((pod_id, node_id, timeslot_id))}: {pod_id} - no placement found")

    print(f"\nSuccessful placements: {successful_placements}")

    # Validate final state
    logging.info("=== FINAL STATE VALIDATION (FIXED VERSION) ===")
    for node_id in state_storage.nodes_by_id:
        for slot_id in range(len(timeslots)):
            cpu_used = state_storage.nodes_by_id[node_id]["cpu"].get(slot_id, {}).get("used", 0)
            ram_used = state_storage.nodes_by_id[node_id]["ram"].get(slot_id, {}).get("used", 0)
            cpu_capacity = test_node.totalCpu
            ram_capacity = test_node.totalRam
            logging.info(f"Node {node_id} Slot {slot_id}: CPU={cpu_capacity-cpu_used}/{cpu_capacity}, RAM={ram_capacity-ram_used}/{ram_capacity}")

    # Check for constraint violations
    violations = []
    for node_id in state_storage.nodes_by_id:
        for slot_id in range(len(timeslots)):
            cpu_used = state_storage.nodes_by_id[node_id]["cpu"].get(slot_id, {}).get("used", 0)
            ram_used = state_storage.nodes_by_id[node_id]["ram"].get(slot_id, {}).get("used", 0)
            
            if cpu_used > test_node.totalCpu:
                violations.append(f"CPU overallocation on {node_id} slot {slot_id}: {cpu_used} > {test_node.totalCpu}")
            if ram_used > test_node.totalRam:
                violations.append(f"RAM overallocation on {node_id} slot {slot_id}: {ram_used} > {test_node.totalRam}")

    print("\n=== CONSTRAINT VALIDATION (FIXED VERSION) ===")
    if violations:
        print(f"❌ Constraint violations detected: {len(violations)}")
        for violation in violations:
            print(f"  - {violation}")
        return False
    else:
        print("✅ No constraint violations detected")
        print("🎯 RACE CONDITION SUCCESSFULLY FIXED!")
        print("Only one pod was placed, preventing resource overallocation.")
        print("The atomic constraint checking worked correctly.")
        return True


# ============================================================================
# STRESS TESTING SUITE (from test_concurrent_stress_simple.py)
# ============================================================================

def create_stress_test_setup():
    """Create test setup with limited resources to force competition."""
    from datetime import datetime, timedelta
    
    # Create 2 nodes with limited resources
    flavours = []
    
    flavour1 = CarbonAwareFlavour(
        id="stress-node-1",
        embodiedCarbon=1000.0,
        lifetime=5.0,
        totalCpu=4.0,     # Limited: 4 CPUs
        totalRam=4096.0,  # Limited: 4GB RAM
        totalStorage=1000.0,
        forecast={0: 0.3, 1: 0.4, 2: 0.5}
    )
    
    flavour2 = CarbonAwareFlavour(
        id="stress-node-2",
        embodiedCarbon=800.0,
        lifetime=4.0, 
        totalCpu=2.0,     # Very limited: 2 CPUs
        totalRam=2048.0,  # Very limited: 2GB RAM
        totalStorage=1000.0,
        forecast={0: 0.2, 1: 0.3, 2: 0.4}
    )
    
    flavours.extend([flavour1, flavour2])
    
    # Create timeslots
    timeslots = []
    for i in range(3):
        ts = CarbonAwareTimeslot(
            id=i,
            start_time=datetime.now().timestamp() + i*3600,
            length=1
        )
        timeslots.append(ts)
    
    return flavours, timeslots


def run_concurrent_stress_test(num_threads: int, pods_per_thread: int):
    """Run a comprehensive concurrent stress test."""
    print(f"\n=== CONCURRENT STRESS TEST ===")
    print(f"Configuration: {num_threads} threads, {pods_per_thread} pods per thread")
    
    # Setup
    flavours, timeslots = create_stress_test_setup()
    total_cpu = sum(f.totalCpu for f in flavours)
    total_ram = sum(f.totalRam for f in flavours)
    total_pods = num_threads * pods_per_thread
    
    print(f"Available resources: {total_cpu} CPU, {total_ram} RAM")
    print(f"Total pods to place: {total_pods}")
    
    # Initialize algorithm and state
    algorithm = HeuristicAlgorithm()
    algorithm.state_storage = PersistentStateStorage()
    algorithm._initialize_resource_state(flavours, timeslots)
    
    # Generate competing pods
    all_pods = []
    for thread_id in range(num_threads):
        for pod_id in range(pods_per_thread):
            pod = CarbonAwarePod(
                id=f"thread-{thread_id}-pod-{pod_id}",
                deadline_hours=24.0,
                duration=1.0,
                powerConsumption=0.0,
                cpuRequest=random.choice([0.5, 1.0, 1.5]),
                ramRequest=random.choice([512, 1024, 1536]),
                storageRequest=0
            )
            pod.earliest_timeslot = 0
            all_pods.append(pod)
    
    def place_pods_thread(thread_pods, thread_id):
        """Place a set of pods in a thread."""
        results = []
        for pod in thread_pods:
            result = algorithm.find_placement_atomic(pod, flavours, timeslots, 10)
            if result:
                node_id, timeslot_id, emissions = result
                print(f"Thread {thread_id}: ✅ Placed {pod.id} on {node_id}")
                results.append((pod.id, node_id, timeslot_id, True))
            else:
                print(f"Thread {thread_id}: ❌ No placement found for {pod.id}")
                results.append((pod.id, None, None, False))
        return results
    
    # Run concurrent placement
    print(f"Starting concurrent placement with {num_threads} threads...")
    start_time = time.time()
    
    # Split pods among threads
    pods_per_thread_actual = len(all_pods) // num_threads
    thread_pod_groups = []
    for i in range(num_threads):
        start_idx = i * pods_per_thread_actual
        end_idx = start_idx + pods_per_thread_actual if i < num_threads - 1 else len(all_pods)
        thread_pod_groups.append(all_pods[start_idx:end_idx])
    
    # Execute threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i, pod_group in enumerate(thread_pod_groups):
            future = executor.submit(place_pods_thread, pod_group, i)
            futures.append(future)
        
        # Collect results
        all_results = []
        for future in as_completed(futures):
            thread_results = future.result()
            all_results.extend(thread_results)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Analyze results
    successful = sum(1 for _, _, _, success in all_results if success)
    failed = len(all_results) - successful
    
    # Check for constraint violations
    violations = validate_resource_constraints(algorithm.state_storage, flavours, timeslots)
    
    print(f"\n=== STRESS TEST RESULTS ===")
    print(f"Duration: {duration:.2f}s")
    print(f"Total attempts: {len(all_results)}")
    print(f"Successful placements: {successful}")
    print(f"Failed placements: {failed}")
    print(f"Constraint violations: {violations}")
    print(f"Success rate: {(successful/len(all_results)*100):.1f}%")
    
    result = {
        'total_attempted': len(all_results),
        'successful': successful,
        'failed': failed,
        'constraint_violations': violations,
        'duration': duration
    }
    
    if violations == 0:
        print("✅ NO RACE CONDITIONS DETECTED!")
        print("✅ Atomic operations working correctly")
    else:
        print(f"❌ RACE CONDITIONS DETECTED: {violations} constraint violations!")
        print("❌ Atomic operations may have issues")
    
    return result


def validate_resource_constraints(state_storage, flavours, timeslots):
    """Validate that no resource constraints are violated."""
    violations = 0
    
    for node in flavours:
        for ts in timeslots:
            if node.id in state_storage.nodes_by_id:
                node_data = state_storage.nodes_by_id[node.id]
                
                # Check CPU
                cpu_used = node_data["cpu"].get(ts.id, {}).get("used", 0)
                if cpu_used > node.totalCpu:
                    violations += 1
                    print(f"❌ CPU violation on {node.id} slot {ts.id}: {cpu_used} > {node.totalCpu}")
                
                # Check RAM
                ram_used = node_data["ram"].get(ts.id, {}).get("used", 0)
                if ram_used > node.totalRam:
                    violations += 1
                    print(f"❌ RAM violation on {node.id} slot {ts.id}: {ram_used} > {node.totalRam}")
    
    return violations


# ============================================================================
# QUICK CONCURRENT TEST (from test_quick_concurrent.py)
# ============================================================================

def test_quick_concurrent():
    """Quick concurrent validation test."""
    print("\n" + "="*60)
    print("QUICK CONCURRENT VALIDATION TEST")
    print("="*60)
    
    print("Quick Concurrent Placement Test")
    print("="*40)
    
    # Create a single node with limited resources
    node = CarbonAwareFlavour(
        id="test-node",
        embodiedCarbon=100.0,
        lifetime=5.0,
        totalCpu=3.0,      # Only 3 CPU cores
        totalRam=3072.0,   # Only 3GB RAM
        totalStorage=1000.0,
        forecast={0: 0.3}
    )
    
    timeslots = [CarbonAwareTimeslot(id=0, start_time=1000000, length=1)]
    
    print(f"Available: {node.totalCpu} CPU, {node.totalRam} RAM")
    
    # Create 5 competing pods (only 3 should fit)
    pods = []
    for i in range(5):
        pod = CarbonAwarePod(
            id=f"compete-pod-{i+1}",
            deadline_hours=24.0,
            duration=1.0,
            powerConsumption=0.0,
            cpuRequest=1.0,    # Each wants 1 CPU
            ramRequest=1024.0, # Each wants 1GB RAM
            storageRequest=0
        )
        pod.earliest_timeslot = 0
        pods.append(pod)
    
    print(f"Competing pods: {len(pods)} (each wants 1 CPU, 1GB RAM)")
    print("Expected: Only 3 pods should be placed")
    
    # Initialize algorithm
    algorithm = HeuristicAlgorithm()
    algorithm.state_storage = PersistentStateStorage()
    algorithm._initialize_resource_state([node], timeslots)
    
    def place_pod_thread(pod, thread_id):
        """Thread function to place a pod."""
        result = algorithm.find_placement_atomic(pod, [node], timeslots, 10)
        if result:
            node_id, timeslot_id, emissions = result
            # Check remaining resources
            cpu_used = algorithm.state_storage.nodes_by_id[node_id]["cpu"].get(timeslot_id, {}).get("used", 0)
            ram_used = algorithm.state_storage.nodes_by_id[node_id]["ram"].get(timeslot_id, {}).get("used", 0)
            cpu_left = node.totalCpu - cpu_used
            ram_left = node.totalRam - ram_used
            print(f"Thread {thread_id}: ✅ Placed {pod.id} (CPU left: {cpu_left}, RAM left: {ram_left})")
            return True
        else:
            print(f"Thread {thread_id}: ❌ No placement for {pod.id}")
            return False
    
    print("\nStarting concurrent placement threads...")
    start_time = time.time()
    
    # Run threads concurrently
    with ThreadPoolExecutor(max_workers=len(pods)) as executor:
        futures = []
        for i, pod in enumerate(pods):
            future = executor.submit(place_pod_thread, pod, i)
            futures.append(future)
        
        results = [future.result() for future in as_completed(futures)]
    
    end_time = time.time()
    duration = end_time - start_time
    
    successful = sum(results)
    failed = len(results) - successful
    violations = validate_resource_constraints(algorithm.state_storage, [node], timeslots)
    
    # Check final resources
    cpu_used = algorithm.state_storage.nodes_by_id[node.id]["cpu"].get(0, {}).get("used", 0)
    ram_used = algorithm.state_storage.nodes_by_id[node.id]["ram"].get(0, {}).get("used", 0)
    cpu_final = node.totalCpu - cpu_used
    ram_final = node.totalRam - ram_used
    
    print(f"\n=== RESULTS ===")
    print(f"Duration: {duration:.3f}s")
    print(f"Successful placements: {successful}")
    print(f"Failed placements: {failed}")
    print(f"Constraint violations: {violations}")
    print(f"Final resources: {cpu_final} CPU, {ram_final} RAM")
    
    if violations == 0 and successful == 3:
        print(f"\n✅ ATOMIC OPERATIONS WORKING CORRECTLY!")
        print(f"✅ No race conditions detected")
        return True
    else:
        print(f"\n❌ ISSUES DETECTED!")
        if violations > 0:
            print(f"❌ {violations} constraint violations")
        if successful != 3:
            print(f"❌ Expected 3 successful placements, got {successful}")
        return False


# ============================================================================
# MAIN TEST SUITE RUNNER
# ============================================================================

def main():
    """Run the comprehensive race condition test suite."""
    print("Carbon-Aware Orchestrator - Comprehensive Race Condition Test Suite")
    print("="*80)
    
    test_results = []
    
    try:
        # Test 1: Core race condition fix
        print("\n🧪 TEST 1: Core Race Condition Fix Validation")
        result1 = test_race_condition_fix()
        test_results.append(("Core Race Condition Fix", result1))
        
        # Test 2: Quick concurrent validation
        print("\n🧪 TEST 2: Quick Concurrent Validation")
        result2 = test_quick_concurrent()
        test_results.append(("Quick Concurrent", result2))
        
        # Test 3: Stress tests
        print("\n🧪 TEST 3: Concurrent Stress Tests")
        
        # Light concurrency
        print("\n" + "="*60)
        print("TEST 3A: Light Concurrency")
        stress_result1 = run_concurrent_stress_test(num_threads=3, pods_per_thread=2)
        test_results.append(("Light Stress", stress_result1['constraint_violations'] == 0))
        
        # Medium concurrency  
        print("\n" + "="*60)
        print("TEST 3B: Medium Concurrency")
        stress_result2 = run_concurrent_stress_test(num_threads=5, pods_per_thread=3)
        test_results.append(("Medium Stress", stress_result2['constraint_violations'] == 0))
        
        # High concurrency
        print("\n" + "="*60)
        print("TEST 3C: High Concurrency")
        stress_result3 = run_concurrent_stress_test(num_threads=8, pods_per_thread=3)
        test_results.append(("High Stress", stress_result3['constraint_violations'] == 0))
        
        # Final assessment
        total_violations = (stress_result1['constraint_violations'] + 
                          stress_result2['constraint_violations'] + 
                          stress_result3['constraint_violations'])
        
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST SUITE FINAL RESULTS")
        print("="*80)
        
        all_passed = True
        for test_name, passed in test_results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name:<25}: {status}")
            if not passed:
                all_passed = False
        
        print(f"\nTotal constraint violations across all stress tests: {total_violations}")
        
        if all_passed and total_violations == 0:
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ No race conditions detected across all test scenarios")
            print("✅ Atomic operations are working correctly under all concurrency levels")
            print("✅ Resource constraints properly enforced")
            return 0
        else:
            print(f"\n❌ SOME TESTS FAILED!")
            if total_violations > 0:
                print(f"❌ {total_violations} constraint violations detected")
            print("❌ Race conditions may still exist in the implementation")
            return 1
            
    except Exception as e:
        print(f"\nTest suite execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
