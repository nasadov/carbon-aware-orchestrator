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
        "hardware_assignment_method": "cycle"  # "cycle" or "random"
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
    num_nodes = config.get("num_nodes", 4)
    base_node_name = config.get("base_node_name", "kwok-node-0")
    regions = config.get("regions", ["DE", "FR", "ES", "IT-NO"])
    assignment_method = config.get("assignment_method", "cycle")
    random_seed = config.get("random_seed", 42)
    
    # Get hardware subcategories configuration
    hardware_subcategories = config.get("hardware_subcategories", {
        "Server": {"cpu": "8", "memory": "16Gi", "embodied_carbon": 1230.656, "lifetime": 3.87}
    })
    hardware_assignment_method = config.get("hardware_assignment_method", "cycle")
    
    # Convert hardware subcategories to a list for easier cycling/random selection
    hardware_types = list(hardware_subcategories.keys())
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Log assignment methods
    print(f"Using {assignment_method} region assignment with seed: {random_seed}")
    print(f"Using {hardware_assignment_method} hardware subcategory assignment")
    
    all_docs = []
    for i in range(num_nodes):
        # deep-copy to avoid changing the template in place
        import copy
        node_doc = copy.deepcopy(NODE_TEMPLATE)

        node_name = f"{base_node_name}{i}"
        
        # Determine region based on assignment method
        if assignment_method == "cycle":
            region = regions[i % len(regions)]
        else:  # random
            region = random.choice(regions)
        
        # Determine hardware subcategory based on assignment method
        if hardware_assignment_method == "cycle":
            subcategory = hardware_types[i % len(hardware_types)]
        else:  # random
            subcategory = random.choice(hardware_types)
        
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
        node_name_with_metadata = f"node-{i}-{region}-{subcategory}"
        node_doc["metadata"]["name"] = node_name_with_metadata
        node_doc["metadata"]["labels"]["kubernetes.io/hostname"] = node_name_with_metadata

        # fill CPU/Memory based on the hardware subcategory
        node_doc["status"]["allocatable"]["cpu"] = cpu_per_node
        node_doc["status"]["allocatable"]["memory"] = memory_per_node
        node_doc["status"]["capacity"]["cpu"] = cpu_per_node
        node_doc["status"]["capacity"]["memory"] = memory_per_node

        all_docs.append(node_doc)

    # Track and report distribution
    region_counts = {}
    hardware_counts = {}
    for doc in all_docs:
        region = doc["metadata"]["labels"]["topology.kubernetes.io/region"]
        region_counts[region] = region_counts.get(region, 0) + 1
        
        subcategory = doc["metadata"]["labels"]["hardware.carbon/subcategory"]
        hardware_counts[subcategory] = hardware_counts.get(subcategory, 0) + 1
    
    # write them as one multi-document yaml
    with open(filename, "w") as f:
        for doc in all_docs:
            yaml.safe_dump(doc, f, sort_keys=False)
            f.write("---\n")

    print(f"Generated {filename} with {num_nodes} nodes")
    print(f"Region distribution: {region_counts}")
    print(f"Hardware subcategory distribution: {hardware_counts}")

# ---------------------------------------------------------------------
# 2. Generate Time Slot Files with Poisson Distribution
# ---------------------------------------------------------------------

def generate_timeslot_files(config: Dict[str, Any]):
    """
    Generate one file per time slot. Each file has a Poisson-distributed 
    number of Deployments with durations and deadlines in hours.
    """
    output_dir = config.get("output_dir", "workloads")
    num_timeslots = config.get("num_timeslots", 5)
    poisson_lambda = config.get("poisson_lambda", 2)
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
    
    print(f"Using Poisson distribution (λ={poisson_lambda}) for microservices per timeslot")
    print(f"Using {duration_assignment_method} duration assignment with options: {durations} hours")
    print(f"Using {deadline_strategy} deadline strategy")
    if deadline_strategy == "flexible" or deadline_strategy == "mixed":
        print(f"Deadline flexibility options: {deadline_flexibility_hours} hours")
    print(f"Random seed: {random_seed}")

    # Create the output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Assign microservice names sequentially across all timeslots
    ms_counter = 0
    duration_counter = 0  # For cycling through durations if needed

    # Pre-generate service counts so we can report the total
    service_counts = []
    for _ in range(num_timeslots):
        count = max(min_services, np.random.poisson(poisson_lambda))
        service_counts.append(count)
    
    total_services = sum(service_counts)
    print(f"Will generate {total_services} total microservices across {num_timeslots} timeslots")

    for slot_id in range(num_timeslots):
        timeslot_filename = os.path.join(output_dir, f"timeslot_{slot_id}.yaml")
        microservices_per_slot = service_counts[slot_id]
        
        deployments = []
        for _ in range(microservices_per_slot):
            # Make a new deployment
            import copy
            dep_doc = copy.deepcopy(DEPLOYMENT_TEMPLATE)

            # Generate the core microservice name
            ms_name = f"{base_name}{ms_counter:03d}"  # e.g. "m000"
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
            
            # Fill in the deployment template
            dep_doc["metadata"]["name"] = full_name
            dep_doc["metadata"]["labels"]["app"] = ms_name  # Keep app label simple
            
            # Add duration and deadline labels
            dep_doc["metadata"]["labels"]["duration"] = f"duration-{duration_str}" 
            dep_doc["metadata"]["labels"]["deadline"] = f"deadline-{deadline_str}"
            
            # Update selector and pod labels
            dep_doc["spec"]["selector"]["matchLabels"]["name"] = ms_name
            dep_doc["spec"]["template"]["metadata"]["labels"]["name"] = ms_name

            # Fill CPU/mem
            container = dep_doc["spec"]["template"]["spec"]["containers"][0]
            container["resources"]["requests"]["cpu"] = cpu_req
            container["resources"]["requests"]["memory"] = mem_req

            deployments.append(dep_doc)

        # Write them as a multi-document YAML for this timeslot
        with open(timeslot_filename, "w") as f:
            for doc in deployments:
                yaml.safe_dump(doc, f, sort_keys=False)
                f.write("---\n")

        print(f"Generated {timeslot_filename} with {microservices_per_slot} microservices.")
        
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