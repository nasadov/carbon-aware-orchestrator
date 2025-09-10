#!/usr/bin/env python3
"""
infra_workload_gen.py

Generates test data for carbon-aware scheduling simulations:
1. Infrastructure - A single "nodes.yaml" file describing KWOK nodes with region information
   that corresponds to carbon intensity data regions.
2. Workload - One YAML file per time slot, each with multiple Deployments (microservices).

The infrastructure generator assigns regions to nodes that match the regions in carbon 
intensity forecasts, using either cyclical or random assignment with a fixed seed.

The workload generator creates microservices using a Poisson distribution for each timeslot.

Configuration is read from a infra-workload-config.yaml file.
"""

import os
import yaml
import random
import numpy as np
from typing import Dict, List, Any

# ---------------------------------------------------------------------
# Templates from Silvio
# ---------------------------------------------------------------------

NODE_TEMPLATE = {
    "apiVersion": "v1",
    "kind": "Node",
    "metadata": {
        "annotations": {
            "node.alpha.kubernetes.io/ttl": "0",
            "kwok.x-k8s.io/node": "fake"
        },
        "labels": {
            "beta.kubernetes.io/arch": "amd64",
            "beta.kubernetes.io/os": "linux",
            "kubernetes.io/arch": "amd64",
            "kubernetes.io/hostname": "",  # to fill
            "kubernetes.io/os": "linux",
            "kubernetes.io/role": "agent",
            "node-role.kubernetes.io/agent": "",
            "type": "kwok"
        },
        "name": ""  # to fill
    },
    "spec": {},
    "status": {
        "allocatable": {
            "cpu": "",   # to fill
            "memory": "",# to fill
            "pods": 110
        },
        "capacity": {
            "cpu": "",   # to fill
            "memory": "",# to fill
            "pods": 110
        },
        "nodeInfo": {
            "architecture": "amd64",
            "bootID": "",
            "containerRuntimeVersion": "",
            "kernelVersion": "",
            "kubeProxyVersion": "fake",
            "kubeletVersion": "fake",
            "machineID": "",
            "operatingSystem": "linux",
            "osImage": "",
            "systemUUID": ""
        },
        "phase": "Running"
    }
}

DEPLOYMENT_TEMPLATE = {
    "apiVersion": "apps/v1",
    "kind": "Deployment",
    "metadata": {
        "name": "",   # e.g. "m000"
        "namespace": "default",
        "labels": {
            "app": "",           # e.g. "m000"
            "duration": "",      # e.g. "duration-1h"
            "deadline": ""       # e.g. "deadline-24h"
        }
    },
    "spec": {
        "replicas": 1,
        "selector": {
            "matchLabels": {
                "name": ""        # e.g. "m000"
            }
        },
        "template": {
            "metadata": {
                "labels": {
                    "name": ""    # e.g. "m000"
                }
            },
            "spec": {
                "schedulerName": "fogatlas",
                "containers": [
                    {
                        "name": "fake-container",
                        "image": "fake-image",
                        "resources": {
                            "requests": {
                                "memory": "",  # e.g. "500Mi"
                                "cpu": ""      # e.g. "500m"
                            }
                        }
                    }
                ]
            }
        }
    }
}

# ---------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------

DEFAULT_CONFIG = {
    "nodes": {
        "filename": "nodes.yaml",
        "num_nodes": 8,
        "base_node_name": "kwok-node-0",
        "regions": ["DE", "FR", "ES", "IT-NO"],
        "assignment_method": "cycle",  # "cycle" or "random"
        "random_seed": 42,
        # New hardware subcategory configuration
        "hardware_subcategories": {
            "IoT": {
                "cpu": "2",
                "memory": "2Gi",
                "embodied_carbon": 27.471,
                "lifetime": 5.2,
                "power": {
                    "idle": 0.5,   # Watts
                    "active": 2.0,  # Watts
                    "max": 5.0      # Watts
                }
            },
            "Smartphone": {
                "cpu": "4",
                "memory": "4Gi",
                "embodied_carbon": 52.729,
                "lifetime": 3.03,
                "power": {
                    "idle": 1.0,    # Watts
                    "active": 3.0,  # Watts
                    "max": 15.0     # Watts
                }
            },
            "Laptop": {
                "cpu": "8",
                "memory": "16Gi",
                "embodied_carbon": 231.855,
                "lifetime": 4.13,
                "power": {
                    "idle": 10.0,   # Watts
                    "active": 40.0, # Watts
                    "max": 150.0    # Watts
                }
            },
            "Server": {
                "cpu": "32",
                "memory": "64Gi",
                "embodied_carbon": 1230.656,
                "lifetime": 3.87,
                "power": {
                    "idle": 100.0,  # Watts
                    "active": 200.0,# Watts
                    "max": 400.0    # Watts
                }
            }
        },
        "hardware_assignment_method": "cycle",  # "cycle" or "random"
        "hardware_cycle_offset": 1  # Offset for shifted_cycle method
    },
    "workload": {
        "output_dir": "workloads",
        "num_timeslots": 12,
        "poisson_lambda": 4,  # Average number of microservices per timeslot
        "min_services": 1,    # Minimum services per timeslot (at least 1)
        "base_name": "m",
        "durations": [1, 3, 6],  # in hours
        "duration_assignment_method": "random",  # "random" or "cycle"
        # NEW: Deadline configuration
        "deadline_strategy": "flexible",  # "tight", "flexible", "mixed", or "exact"
        "deadline_flexibility_hours": [2, 6, 24],  # Hours to add to duration for deadlines
        "cpu_options": ["500m", "200m", "100m"],
        "mem_options": ["500Mi", "200Mi"]
    }
}

# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def format_duration(hours: float) -> str:
    """
    Format a duration in hours to a string representation.
    Handles whole numbers and fractions appropriately.
    
    Args:
        hours: Duration in hours
        
    Returns:
        String representation with appropriate units
    """
    # For exact hours, use simple format
    if hours == int(hours):
        return f"{int(hours)}h"
    
    # For fractions, use decimal format
    if hours * 60 == int(hours * 60):
        # If it's a whole number of minutes, use minutes format
        minutes = int(hours * 60)
        if minutes < 60:
            return f"{minutes}m"
        else:
            h = minutes // 60
            m = minutes % 60
            if m == 0:
                return f"{h}h"
            else:
                return f"{h}h{m}m"
    
    # For other fractions, use decimal hours
    return f"{hours:.1f}h"

# ---------------------------------------------------------------------
# 1. Generate Nodes File
# ---------------------------------------------------------------------

def generate_nodes_file(config: Dict[str, Any]):
    """
    Create a yaml with nodes based on configuration.
    
    Args:
        config: Dictionary with node generation configuration
    """
    filename = config.get("filename", "nodes.yaml")
    base_node_name = config.get("base_node_name", "kwok-node-0")
    random_seed = config.get("random_seed", 42)
    
    # Handle regions configuration (new and legacy formats)
    regions_config = config.get("regions", {})
    if isinstance(regions_config, dict):
        # New format with region_counts or region_list
        assignment_method = config.get("assignment_method", "cycle")
        
        if assignment_method == "exact_counts" and "region_counts" in regions_config:
            # Use exact counts method
            region_counts = regions_config["region_counts"]
            regions_list = list(region_counts.keys())
            print(f"Using exact_counts region assignment: {region_counts}")
        else:
            # Use legacy method with region_list
            regions_list = regions_config.get("region_list", ["DE", "FR", "ES", "IT-NO"])
            region_counts = None
            print(f"Using {assignment_method} region assignment with regions: {regions_list}")
    else:
        # Legacy format: regions is directly a list
        regions_list = regions_config
        assignment_method = config.get("assignment_method", "cycle")
        region_counts = None
        print(f"Using legacy region format with {assignment_method} assignment")
    
    # Handle hardware assignment configuration (new and legacy formats)
    hardware_assignment = config.get("hardware_assignment", {})
    if isinstance(hardware_assignment, dict) and "method" in hardware_assignment:
        # New format
        hardware_assignment_method = hardware_assignment.get("method", "cycle")
        hardware_cycle_offset = hardware_assignment.get("cycle_offset", 1)
        
        if hardware_assignment_method == "exact_counts" and "hardware_counts" in hardware_assignment:
            # Use exact counts method
            hardware_counts = hardware_assignment["hardware_counts"]
            hardware_types = list(hardware_counts.keys())
            print(f"Using exact_counts hardware assignment: {hardware_counts}")
        else:
            # Use legacy method
            hardware_counts = None
    else:
        # Legacy format
        hardware_assignment_method = config.get("hardware_assignment_method", "cycle")
        hardware_cycle_offset = config.get("hardware_cycle_offset", 1)
        
        # Check if we have hardware_counts in the flat structure (current config format)
        if hardware_assignment_method == "exact_counts" and "hardware_counts" in hardware_assignment:
            hardware_counts = hardware_assignment["hardware_counts"]
            hardware_types = list(hardware_counts.keys())
            print(f"Using exact_counts hardware assignment: {hardware_counts}")
        else:
            hardware_counts = None
    
    # Get hardware subcategories configuration
    hardware_subcategories = config.get("hardware_subcategories", {
        "Server": {"cpu": "8", "memory": "16Gi", "embodied_carbon": 1230.656, "lifetime": 3.87}
    })
    
    # Convert hardware subcategories to a list for easier cycling/random selection
    if hardware_counts is None:
        hardware_types = list(hardware_subcategories.keys())
    
    # ------------------------------------------------------------------
    # NEW: Support explicit region-to-hardware assignments
    # Accept either a dict mapping region -> {hardware: count} or
    # a list of {region, hardware, count} objects.
    # When provided, this overrides region/hardware assignment methods.
    # ------------------------------------------------------------------
    explicit_cfg = config.get("explicit_assignments") or config.get("region_hardware_map")
    use_explicit = explicit_cfg is not None
    explicit_node_assignments = []
    if use_explicit:
        if isinstance(explicit_cfg, dict):
            for region, hw_map in explicit_cfg.items():
                if isinstance(hw_map, dict):
                    for hw_type, count in hw_map.items():
                        for _ in range(int(count)):
                            explicit_node_assignments.append({"region": region, "hardware": hw_type})
        elif isinstance(explicit_cfg, list):
            for item in explicit_cfg:
                region = item.get("region")
                hw_type = item.get("hardware")
                count = int(item.get("count", 1))
                for _ in range(count):
                    explicit_node_assignments.append({"region": region, "hardware": hw_type})
        else:
            raise ValueError("explicit_assignments must be dict or list of {region, hardware, count}")
        # Number of nodes is driven by explicit mapping
        num_nodes = len(explicit_node_assignments)
    
    # Calculate total number of nodes based on assignment method (unless explicit mapping provided)
    if not use_explicit:
        if assignment_method == "exact_counts" and region_counts:
            if hardware_assignment_method == "exact_counts" and hardware_counts:
                # Both exact counts - they must match
                total_regions = sum(region_counts.values())
                total_hardware = sum(hardware_counts.values())
                if total_regions != total_hardware:
                    print(f"Warning: Region count total ({total_regions}) doesn't match hardware count total ({total_hardware})")
                    print("Using the larger of the two totals")
                    num_nodes = max(total_regions, total_hardware)
                else:
                    num_nodes = total_regions
            else:
                # Only regions exact count
                num_nodes = sum(region_counts.values())
        elif hardware_assignment_method == "exact_counts" and hardware_counts:
            # Only hardware exact count
            num_nodes = sum(hardware_counts.values())
        else:
            # Legacy: use num_nodes from config
            num_nodes = config.get("num_nodes", 4)
    
    print(f"Generating {num_nodes} total nodes")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Generate node assignments
    if use_explicit:
        node_assignments = list(explicit_node_assignments)
        print("Using explicit region-to-hardware assignments")
    else:
        node_assignments = []
        if assignment_method == "exact_counts" and region_counts:
            # Create exact assignments for regions
            for region, count in region_counts.items():
                for _ in range(count):
                    node_assignments.append({"region": region})
        else:
            # Legacy region assignment
            for i in range(num_nodes):
                if assignment_method == "cycle":
                    region = regions_list[i % len(regions_list)]
                else:  # random
                    region = random.choice(regions_list)
                node_assignments.append({"region": region})
        
        # Assign hardware to nodes
        if hardware_assignment_method == "exact_counts" and hardware_counts:
            # Create exact assignments for hardware
            hardware_assignments = []
            for hw_type, count in hardware_counts.items():
                for _ in range(count):
                    hardware_assignments.append(hw_type)
            
            # If we have fewer hardware assignments than nodes, cycle through them
            while len(hardware_assignments) < len(node_assignments):
                for hw_type in hardware_types:
                    if len(hardware_assignments) >= len(node_assignments):
                        break
                    hardware_assignments.append(hw_type)
            
            # Assign hardware to each node
            for i, assignment in enumerate(node_assignments):
                if i < len(hardware_assignments):
                    assignment["hardware"] = hardware_assignments[i]
                else:
                    # Fallback to cycling if we somehow don't have enough
                    assignment["hardware"] = hardware_types[i % len(hardware_types)]
        else:
            # Legacy hardware assignment
            for i, assignment in enumerate(node_assignments):
                if hardware_assignment_method == "cycle":
                    subcategory = hardware_types[i % len(hardware_types)]
                elif hardware_assignment_method == "shifted_cycle":
                    # Calculate the region cycle and region position within the cycle
                    region_cycle = i // len(regions_list)
                    region_position = i % len(regions_list)
                    
                    # Calculate the hardware position with an offset that increases with each cycle
                    hardware_position = (region_position + (region_cycle * hardware_cycle_offset)) % len(hardware_types)
                    subcategory = hardware_types[hardware_position]
                else:  # random
                    subcategory = random.choice(hardware_types)
                
                assignment["hardware"] = subcategory
    
    # Log assignment methods
    print(f"Using {assignment_method} region assignment with seed: {random_seed}")
    print(f"Using {hardware_assignment_method} hardware assignment")
    if hardware_assignment_method == "shifted_cycle":
        print(f"Using hardware cycle offset: {hardware_cycle_offset}")
    
    all_docs = []
    for i, assignment in enumerate(node_assignments):
        # deep-copy to avoid changing the template in place
        import copy
        node_doc = copy.deepcopy(NODE_TEMPLATE)

        region = assignment["region"]
        subcategory = assignment["hardware"]
        
        # Get hardware specs for the selected subcategory
        hw_specs = hardware_subcategories[subcategory]
        cpu_per_node = hw_specs["cpu"]
        memory_per_node = hw_specs["memory"]
        
        # Add labels for region and hardware subcategory
        node_doc["metadata"]["labels"]["topology.kubernetes.io/region"] = region
        node_doc["metadata"]["labels"]["hardware.carbon/subcategory"] = subcategory
        
        # Add annotations for embodied carbon and lifetime
        if "embodied_carbon" in hw_specs:
            node_doc["metadata"]["annotations"]["hardware.carbon/embodied_emissions"] = str(hw_specs["embodied_carbon"])
        if "lifetime" in hw_specs:
            node_doc["metadata"]["annotations"]["hardware.carbon/lifetime_years"] = str(hw_specs["lifetime"])
            
        # Add power consumption annotations
        if "power" in hw_specs:
            power_data = hw_specs["power"]
            node_doc["metadata"]["annotations"]["hardware.power/idle_watts"] = str(power_data["idle"])
            node_doc["metadata"]["annotations"]["hardware.power/active_watts"] = str(power_data["active"])
            node_doc["metadata"]["annotations"]["hardware.power/max_watts"] = str(power_data["max"])
        
        # Include region and hardware type in node name for easier identification
        node_name_with_metadata = f"node-{i}-{region}-{subcategory}".lower()
        node_doc["metadata"]["name"] = node_name_with_metadata
        node_doc["metadata"]["labels"]["kubernetes.io/hostname"] = node_name_with_metadata

        # fill CPU/Memory based on the hardware subcategory
        node_doc["status"]["allocatable"]["cpu"] = cpu_per_node
        node_doc["status"]["allocatable"]["memory"] = memory_per_node
        node_doc["status"]["capacity"]["cpu"] = cpu_per_node
        node_doc["status"]["capacity"]["memory"] = memory_per_node

        all_docs.append(node_doc)

    # Track and report distribution
    region_counts_actual = {}
    hardware_counts_actual = {}
    for doc in all_docs:
        region = doc["metadata"]["labels"]["topology.kubernetes.io/region"]
        region_counts_actual[region] = region_counts_actual.get(region, 0) + 1
        
        subcategory = doc["metadata"]["labels"]["hardware.carbon/subcategory"]
        hardware_counts_actual[subcategory] = hardware_counts_actual.get(subcategory, 0) + 1
    
    # write them as one multi-document yaml
    with open(filename, "w") as f:
        for doc in all_docs:
            yaml.safe_dump(doc, f, sort_keys=False)
            f.write("---\n")

    print(f"Generated {filename} with {num_nodes} nodes")
    print(f"Region distribution: {region_counts_actual}")
    print(f"Hardware subcategory distribution: {hardware_counts_actual}")

# ---------------------------------------------------------------------
# 2. Generate Time Slot Files with Poisson Distribution
# ---------------------------------------------------------------------

def generate_timeslot_files(config: Dict[str, Any]):
    """
    Generate one file per time slot. Each file has a Poisson-distributed 
    number of Deployments with durations and deadlines in hours.
    
    Generates two sets of files:
    1. With custom scheduler ("fogatlas")
    2. With default Kubernetes scheduler (no schedulerName specified)
    """
    output_dir = config.get("output_dir", "workloads")
    vanilla_output_dir = config.get("vanilla_output_dir", output_dir + "-vanilla")
    num_timeslots = config.get("num_timeslots", 5)
    generation_strategy = config.get("generation_strategy", "poisson")
    poisson_lambda = config.get("poisson_lambda", 2)
    exact_total_pods = config.get("exact_total_pods", None)
    min_services = config.get("min_services", 1)
    base_name = config.get("base_name", "m")
    
    # Get duration options (directly as hour values)
    durations = config.get("durations", [1, 4, 12])  # Default durations in hours
    duration_assignment_method = config.get("duration_assignment_method", "random")
    
    # NEW: Get deadline configuration
    deadline_strategy = config.get("deadline_strategy", "flexible")
    deadline_flexibility_hours = config.get("deadline_flexibility_hours", [2, 6, 24])
    
    # Resource selection based on hardware capability
    cpu_by_hardware = {
        "IoT": ["100m", "200m"],
        "Smartphone": ["200m", "500m"],
        "Laptop": ["500m", "1000m"],
        "Server": ["1000m", "2000m", "4000m"]
    }
    
    mem_by_hardware = {
        "IoT": ["128Mi", "256Mi"],
        "Smartphone": ["256Mi", "512Mi"],
        "Laptop": ["512Mi", "1Gi", "2Gi"],
        "Server": ["2Gi", "4Gi", "8Gi"]
    }
    
    # Default resource options if not using hardware-specific
    cpu_options = config.get("cpu_options", ["500m", "200m", "100m"])
    mem_options = config.get("mem_options", ["500Mi", "200Mi"])
    
    # Set random seed for reproducibility
    random_seed = config.get("random_seed", 42)
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    print(f"Generation strategy: {generation_strategy}")
    if generation_strategy == "poisson":
        print(f"Using Poisson distribution (λ={poisson_lambda}) for microservices per timeslot")
    else:
        print(f"Using exact total pods across all timeslots")
        if exact_total_pods is None:
            raise ValueError("exact_total_pods must be set when generation_strategy is 'exact_total'")
        if not isinstance(exact_total_pods, int) or exact_total_pods < 0:
            raise ValueError("exact_total_pods must be a non-negative integer")
    print(f"Using {duration_assignment_method} duration assignment with options: {durations} hours")
    print(f"Using {deadline_strategy} deadline strategy")
    if deadline_strategy == "flexible" or deadline_strategy == "mixed":
        print(f"Deadline flexibility options: {deadline_flexibility_hours} hours")
    print(f"Random seed: {random_seed}")

    # Create the output directories if not exists
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vanilla_output_dir, exist_ok=True)
    print(f"Creating workload files in two directories:")
    print(f" - Custom scheduler (fogatlas): {output_dir}")
    print(f" - Default scheduler (vanilla): {vanilla_output_dir}")

    ms_counter = 0
    duration_counter = 0  # For cycling through durations if needed

    # Pre-generate service counts so we can report the total
    service_counts = []
    if generation_strategy == "poisson":
        for _ in range(num_timeslots):
            count = max(min_services, np.random.poisson(poisson_lambda))
            service_counts.append(count)
    else:
        # Deterministically distribute exact_total_pods across timeslots using fixed seed
        # Even split with remainder distributed to early slots via floor-division method
        total = int(exact_total_pods)
        for s in range(num_timeslots):
            prev = (total * s) // num_timeslots
            curr = (total * (s + 1)) // num_timeslots
            service_counts.append(curr - prev)
    
    total_services = sum(service_counts)
    print(f"Will generate {total_services} total microservices across {num_timeslots} timeslots")
    print(f"All pods will use normal naming pattern: {base_name}000, {base_name}001, etc.")

    # Two approaches to assign pods to timeslots:
    # - For Poisson: keep existing per-slot generation (random counts per slot)
    # - For exact_total: assign pod i to slot (i % num_timeslots) to keep stable mapping across totals
    if generation_strategy == "poisson":
        for slot_id in range(num_timeslots):
            # Filenames for both custom and vanilla scheduler
            timeslot_filename = os.path.join(output_dir, f"timeslot_{slot_id}.yaml")
            vanilla_timeslot_filename = os.path.join(vanilla_output_dir, f"timeslot_{slot_id}.yaml")
            microservices_per_slot = service_counts[slot_id]
            
            custom_deployments = []
            vanilla_deployments = []
            
            for _ in range(microservices_per_slot):
                # Make two new deployments (one for each scheduler)
                import copy
                custom_dep = copy.deepcopy(DEPLOYMENT_TEMPLATE)
                vanilla_dep = copy.deepcopy(DEPLOYMENT_TEMPLATE)
                
                # Remove the custom scheduler from the vanilla deployment
                # The default scheduler will be used if schedulerName is not specified
                if "schedulerName" in vanilla_dep["spec"]["template"]["spec"]:
                    del vanilla_dep["spec"]["template"]["spec"]["schedulerName"]

                # Generate the core microservice name using normal naming pattern
                ms_name = f"{base_name}{ms_counter:03d}"  # e.g. "m000", "m001", etc.
                ms_counter += 1

                # Pick CPU/mem from our lists
                cpu_req = random.choice(cpu_options)
                mem_req = random.choice(mem_options)
                
                # Get duration based on assignment method
                if duration_assignment_method == "cycle":
                    duration_hours = durations[duration_counter % len(durations)]
                    duration_counter += 1
                else:  # random
                    duration_hours = random.choice(durations)
                
                # Calculate deadline based on strategy
                if deadline_strategy == "tight":
                    # Tight deadlines - exact same as duration (no flexibility)
                    deadline_hours = duration_hours
                elif deadline_strategy == "exact":
                    # Exact deadlines - use the duration times a specific multiplier
                    deadline_hours = duration_hours * 2  # 2x the duration
                elif deadline_strategy == "mixed":
                    # Mixed strategy - sometimes tight, sometimes flexible
                    if random.random() < 0.3:  # 30% chance of tight deadline
                        deadline_hours = duration_hours
                    else:
                        flexibility = random.choice(deadline_flexibility_hours)
                        deadline_hours = duration_hours + flexibility
                else:  # flexible (default)
                    flexibility = random.choice(deadline_flexibility_hours)
                    deadline_hours = duration_hours + flexibility
                
                # Format duration and deadline as strings with units
                duration_str = format_duration(duration_hours)
                deadline_str = format_duration(deadline_hours)
                
                # Create the full name with both duration and deadline
                full_name = f"{ms_name}-duration-{duration_str}-deadline-{deadline_str}"
                
                # Apply the same configuration to both deployments
                for dep in [custom_dep, vanilla_dep]:
                    # Fill in the deployment template
                    dep["metadata"]["name"] = full_name
                    dep["metadata"]["labels"]["app"] = full_name

                    # Add duration and deadline labels 
                    dep["metadata"]["labels"]["duration"] = f"duration-{duration_str}"
                    dep["metadata"]["labels"]["deadline"] = f"deadline-{deadline_str}"

                    # Update selector and pod labels
                    dep["spec"]["selector"]["matchLabels"]["name"] = ms_name
                    dep["spec"]["template"]["metadata"]["labels"]["name"] = ms_name

                    # Fill CPU/mem
                    container = dep["spec"]["template"]["spec"]["containers"][0]
                    container["resources"]["requests"]["cpu"] = cpu_req
                    container["resources"]["requests"]["memory"] = mem_req
                
                custom_deployments.append(custom_dep)
                vanilla_deployments.append(vanilla_dep)

            # Write custom scheduler deployments
            with open(timeslot_filename, "w") as f:
                for doc in custom_deployments:
                    yaml.safe_dump(doc, f, sort_keys=False)
                    f.write("---\n")
            
            # Write vanilla scheduler deployments
            with open(vanilla_timeslot_filename, "w") as f:
                for doc in vanilla_deployments:
                    yaml.safe_dump(doc, f, sort_keys=False)
                    f.write("---\n")

            print(f"Generated timeslot {slot_id} with {microservices_per_slot} microservices in both directories.")
    else:
        # exact_total: assign pod i to timeslot (i % num_timeslots) for stable mapping across different totals
        slot_custom_deployments = [[] for _ in range(num_timeslots)]
        slot_vanilla_deployments = [[] for _ in range(num_timeslots)]

        for i in range(total_services):
            import copy
            custom_dep = copy.deepcopy(DEPLOYMENT_TEMPLATE)
            vanilla_dep = copy.deepcopy(DEPLOYMENT_TEMPLATE)

            # Remove the custom scheduler from the vanilla deployment
            if "schedulerName" in vanilla_dep["spec"]["template"]["spec"]:
                del vanilla_dep["spec"]["template"]["spec"]["schedulerName"]

            # Microservice name
            ms_name = f"{base_name}{ms_counter:03d}"
            ms_counter += 1

            # Resource requests
            cpu_req = random.choice(cpu_options)
            mem_req = random.choice(mem_options)

            # Duration assignment
            if duration_assignment_method == "cycle":
                duration_hours = durations[duration_counter % len(durations)]
                duration_counter += 1
            else:
                duration_hours = random.choice(durations)

            # Deadline assignment
            if deadline_strategy == "tight":
                deadline_hours = duration_hours
            elif deadline_strategy == "exact":
                deadline_hours = duration_hours * 2
            elif deadline_strategy == "mixed":
                if random.random() < 0.3:
                    deadline_hours = duration_hours
                else:
                    flexibility = random.choice(deadline_flexibility_hours)
                    deadline_hours = duration_hours + flexibility
            else:
                flexibility = random.choice(deadline_flexibility_hours)
                deadline_hours = duration_hours + flexibility

            duration_str = format_duration(duration_hours)
            deadline_str = format_duration(deadline_hours)
            full_name = f"{ms_name}-duration-{duration_str}-deadline-{deadline_str}"

            for dep in [custom_dep, vanilla_dep]:
                dep["metadata"]["name"] = full_name
                dep["metadata"]["labels"]["app"] = full_name
                dep["metadata"]["labels"]["duration"] = f"duration-{duration_str}"
                dep["metadata"]["labels"]["deadline"] = f"deadline-{deadline_str}"
                dep["spec"]["selector"]["matchLabels"]["name"] = ms_name
                dep["spec"]["template"]["metadata"]["labels"]["name"] = ms_name
                container = dep["spec"]["template"]["spec"]["containers"][0]
                container["resources"]["requests"]["cpu"] = cpu_req
                container["resources"]["requests"]["memory"] = mem_req

            slot_id = i % num_timeslots
            slot_custom_deployments[slot_id].append(custom_dep)
            slot_vanilla_deployments[slot_id].append(vanilla_dep)

        # Write files per timeslot using the round-robin assignment
        for slot_id in range(num_timeslots):
            timeslot_filename = os.path.join(output_dir, f"timeslot_{slot_id}.yaml")
            vanilla_timeslot_filename = os.path.join(vanilla_output_dir, f"timeslot_{slot_id}.yaml")

            with open(timeslot_filename, "w") as f:
                for doc in slot_custom_deployments[slot_id]:
                    yaml.safe_dump(doc, f, sort_keys=False)
                    f.write("---\n")

            with open(vanilla_timeslot_filename, "w") as f:
                for doc in slot_vanilla_deployments[slot_id]:
                    yaml.safe_dump(doc, f, sort_keys=False)
                    f.write("---\n")

            print(f"Generated timeslot {slot_id} with {len(slot_custom_deployments[slot_id])} microservices in both directories.")
        
    # Report duration distribution
    if duration_assignment_method == "cycle" and total_services > 0:
        duration_counts = {}
        for i, hours in enumerate(durations):
            count = (total_services + i) // len(durations) if i < (total_services % len(durations)) else total_services // len(durations)
            hours_str = format_duration(hours)
            duration_counts[hours_str] = count
        print(f"Duration distribution (cyclic): {duration_counts}")
    else:
        print(f"Duration distribution: random selection from {[format_duration(d) for d in durations]}")
        
    # Report deadline strategy
    print(f"Deadline strategy: {deadline_strategy}")
    if deadline_strategy == "tight":
        print("All services have zero scheduling flexibility (deadline = duration)")
    elif deadline_strategy == "exact":
        print("All services have deadline = 2x duration")
    elif deadline_strategy == "mixed":
        print("30% of services have zero flexibility, 70% have variable flexibility")
    else:
        flex_strings = [format_duration(h) for h in deadline_flexibility_hours]
        print(f"All services have flexible deadlines (duration + {flex_strings})")

# ---------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------

def load_config(config_file="infra-workload-config.yaml"):
    """
    Load configuration from YAML file, with defaults.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Complete configuration dictionary with defaults applied
    """
    config = DEFAULT_CONFIG.copy()
    
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = yaml.safe_load(f)
                
                # Update the defaults with user configuration
                if user_config.get("nodes"):
                    config["nodes"].update(user_config["nodes"])
                if user_config.get("workload"):
                    config["workload"].update(user_config["workload"])
                    
            print(f"Loaded configuration from {config_file}")
        else:
            print(f"Configuration file {config_file} not found, using defaults")
            
            # Create a sample config file for future use
            with open(config_file, 'w') as f:
                yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False)
            print(f"Created sample configuration file: {config_file}")
                
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Using default configuration")
        
    return config

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Load configuration
    config = load_config("infra-workload-config.yaml")
    
    # 1. Generate the KWOK nodes file
    generate_nodes_file(config["nodes"])

    # 2. Generate timeslot files with Poisson distribution
    generate_timeslot_files(config["workload"])

    print("Done generating YAML files!")

# ---------------------------------------------------------------------
# End of infra_workload_gen.py
# ---------------------------------------------------------------------