#!/usr/bin/env python3
"""
Standalone MILP Global Optimizer Script

This script replicates the MILP-based global optimization from the carbon-aware 
global-optimal algorithm to place all pods from the workloads directory in a single
optimization. It reads pods from all workload files, nodes.yaml, and carbon intensity
forecasts, then solves the optimization problem and saves placement results in CSV format.

Example usage:
    python milp_global_optimizer.py --workloads-dir pkg/carbon-aware/server-python/workloads \\
                                  --nodes-file pkg/carbon-aware/server-python/nodes.yaml \\
                                  --forecasts-file pkg/carbon-aware/server-python/all_forecasts.json \\
                                  --output results_global_optimization.csv
"""
import argparse
import csv
import json
import logging
import os
import re
import sys
import time
import traceback
import yaml
from datetime import datetime

import pulp

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pkg', 'carbon-aware', 'server-python'))

# Import carbon-aware models and utilities
from carbon_aware.models import CarbonAwarePod, CarbonAwareFlavour, CarbonAwareTimeslot
from carbon_aware.utils import compute_emissions

# Add carbon-aware-orchestrator modules to path
carbon_aware_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              "pkg/carbon-aware/server-python")
sys.path.append(carbon_aware_path)

# Import carbon-aware modules
try:
    from carbon_aware.models import CarbonAwarePod, CarbonAwareFlavour, CarbonAwareTimeslot
    from carbon_aware.utils import compute_emissions
except ImportError:
    print("Error importing carbon-aware modules. Make sure the path is correct.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class GlobalOptimizer:
    """
    Global optimizer that places all pods from all workloads at once using MILP.
    Extracted from the global-optimal algorithm implementation.
    """
    
    CSV_HEADERS = [
        "pod_id", "node_id", "start_slot", "duration",
        "cpu_request", "ram_request", "total_carbon_emissions",
        "solver_status", "solver_iterations", "solution_time_seconds"
    ]
    
    def __init__(self, output_file):
        """
        Initialize the global optimizer with tracking structures.
        """
        self.output_file = output_file
        self.all_flavours = []
        self.all_timeslots = []
        self.pending_pods = []
        self.global_solution = {}
        self.has_solved = False
        self.iterations = 0
        self.status = None
        self.optimization_done = False
        self.last_solve_time = 0
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        logging.info(f"🔧 Initializing Global Optimizer. Results will be saved to {output_file}")

    def _extract_region_from_node_id(self, node_id):
        """Extract region code from node ID (e.g., 'node-0-de-iot' -> 'DE', 'node-3-it-no-smartphone' -> 'IT-NO')."""
        match = re.search(r'node-\d+-([a-z]{2}(?:-[a-z]{2})?)-', node_id)
        if match:
            return match.group(1).upper()  # Convert to uppercase to match JSON keys
        return "unknown"

    def _load_nodes_from_yaml(self, nodes_file):
        """Load node definitions from Kubernetes nodes.yaml."""
        try:
            with open(nodes_file, 'r') as f:
                content = f.read()
            
            flavours = []
            # Parse YAML documents (nodes are separated by ---)
            for doc in yaml.safe_load_all(content):
                if doc and doc.get('kind') == 'Node':
                    metadata = doc.get('metadata', {})
                    status = doc.get('status', {})
                    annotations = metadata.get('annotations', {})
                    labels = metadata.get('labels', {})
                    
                    node_name = metadata.get('name', '')
                    capacity = status.get('capacity', {})
                    
                    # Extract CPU (convert from string like "2" to float)
                    cpu_str = capacity.get('cpu', '0')
                    total_cpu = float(cpu_str)
                    
                    # Extract RAM (convert from string like "2Gi" to MB)
                    ram_str = capacity.get('memory', '0')
                    total_ram = self._parse_memory_to_mb(ram_str)
                    
                    # Extract region from topology label
                    region = labels.get('topology.kubernetes.io/region', 'DE')
                    
                    # Extract carbon and power data from annotations
                    embodied_carbon = float(annotations.get('hardware.carbon/embodied_emissions', '0')) / 1000.0  # Convert g to kg
                    lifetime_years = float(annotations.get('hardware.carbon/lifetime_years', '5.0'))
                    idle_watts = float(annotations.get('hardware.power/idle_watts', '100.0'))
                    active_watts = float(annotations.get('hardware.power/active_watts', '200.0'))
                    max_watts = float(annotations.get('hardware.power/max_watts', '400.0'))
                    
                    # Create flavour with empty forecast (will be filled later)
                    flv = CarbonAwareFlavour(
                        id=node_name,
                        embodiedCarbon=embodied_carbon,
                        lifetime=lifetime_years * 8760,  # Convert years to hours
                        totalCpu=total_cpu,
                        totalRam=total_ram,
                        totalStorage=1000.0,  # Default storage
                        forecast={},  # Will be populated later
                        power={
                            "idle": idle_watts,
                            "active": active_watts,
                            "max": max_watts
                        }
                    )
                    flv.region = self._extract_region_from_node_id(node_name)  # Use extracted region with proper case
                    flavours.append(flv)
                    
                    logging.info(f"  - Loaded node {node_name}: CPU={total_cpu}, RAM={total_ram}MB, region={region}")
            
            return flavours
        except Exception as e:
            logging.error(f"Failed to load nodes from {nodes_file}: {e}")
            traceback.print_exc()
            return []
    
    def _parse_memory_to_mb(self, memory_str):
        """Parse Kubernetes memory format (e.g., '2Gi', '512Mi') to MB."""
        memory_str = memory_str.strip()
        if memory_str.endswith('Gi'):
            return float(memory_str[:-2]) * 1024
        elif memory_str.endswith('Mi'):
            return float(memory_str[:-2])
        elif memory_str.endswith('Ki'):
            return float(memory_str[:-2]) / 1024
        elif memory_str.endswith('G'):
            return float(memory_str[:-1]) * 1000
        elif memory_str.endswith('M'):
            return float(memory_str[:-1])
        else:
            # Assume bytes
            return float(memory_str) / (1024 * 1024)

    def _load_carbon_forecasts(self, forecasts_file):
        """Load carbon intensity forecasts from JSON file."""
        try:
            with open(forecasts_file, 'r') as f:
                forecasts_data = json.load(f)
            
            # Create a dictionary to store forecasts by region
            carbon_forecasts = {}
            
            for region, data in forecasts_data.items():
                # Extract the forecast entries
                carbon_forecasts[region.upper()] = {}
                
                for entry in data.get('forecast', []):
                    ts_id = len(carbon_forecasts[region.upper()])  # Use sequential timeslot IDs
                    carbon_forecasts[region.upper()][ts_id] = float(entry.get('carbonIntensity', 0))
            
            logging.info(f"Loaded carbon forecasts for regions: {list(carbon_forecasts.keys())}")
            return carbon_forecasts
        except Exception as e:
            logging.error(f"Failed to load carbon forecasts from {forecasts_file}: {e}")
            logging.error(traceback.format_exc())
            return {}

    def _extract_pods_from_yaml(self, yaml_file):
        """Extract pod definitions from a Kubernetes Deployment YAML file."""
        try:
            with open(yaml_file, 'r') as f:
                content = f.read()
            
            # Extract timeslot number from filename using regex
            match = re.match(r'.*timeslot_(\d+)\.yaml$', yaml_file)
            earliest_timeslot = int(match.group(1)) if match else 0
            
            pods = []
            # Split by --- to handle multiple deployments in one file
            documents = content.split('---')
            
            for doc in documents:
                doc = doc.strip()
                if not doc:
                    continue
                
                try:
                    deployment = yaml.safe_load(doc)
                    if not deployment or deployment.get('kind') != 'Deployment':
                        continue
                    
                    # Extract pod information from Deployment
                    metadata = deployment.get('metadata', {})
                    pod_name = metadata.get('name', '')
                    labels = metadata.get('labels', {})
                    
                    # Extract duration and deadline from labels
                    duration_label = labels.get('duration', 'duration-1h')
                    deadline_label = labels.get('deadline', 'deadline-24h')
                    
                    # Parse duration (e.g., "duration-1h" -> 1.0)
                    duration_match = re.search(r'duration-(\d+(?:\.\d+)?)h', duration_label)
                    duration = float(duration_match.group(1)) if duration_match else 1.0
                    
                    # Parse deadline (e.g., "deadline-7h" -> 7.0)
                    deadline_match = re.search(r'deadline-(\d+(?:\.\d+)?)h', deadline_label)
                    deadline_hours = float(deadline_match.group(1)) if deadline_match else 24.0
                    
                    # Extract resource requests from container spec
                    spec = deployment.get('spec', {})
                    template = spec.get('template', {})
                    pod_spec = template.get('spec', {})
                    containers = pod_spec.get('containers', [])
                    
                    if not containers:
                        logging.warning(f"No containers found in deployment {pod_name}")
                        continue
                    
                    # Get resource requests from first container
                    container = containers[0]
                    resources = container.get('resources', {})
                    requests = resources.get('requests', {})
                    
                    # Parse CPU request (e.g., "1000m" -> 1.0)
                    cpu_str = requests.get('cpu', '0')
                    if cpu_str.endswith('m'):
                        cpu_request = float(cpu_str[:-1]) / 1000.0
                    else:
                        cpu_request = float(cpu_str)
                    
                    # Parse memory request (e.g., "1Gi" -> 1024 MB)
                    memory_str = requests.get('memory', '0')
                    ram_request = self._parse_memory_to_mb(memory_str)
                    
                    # Create CarbonAwarePod with all required parameters
                    pod = CarbonAwarePod(
                        id=pod_name,
                        deadline_hours=deadline_hours,
                        duration=duration,
                        powerConsumption=0.0,  # Set default value (not used in MILP)
                        cpuRequest=cpu_request,
                        ramRequest=ram_request,
                        storageRequest=0  # Set default value (not used in MILP)
                    )
                    pod.earliest_timeslot = earliest_timeslot
                    pod.deadline_slot = earliest_timeslot + deadline_hours
                    
                    pods.append(pod)
                    logging.debug(f"  - Extracted pod {pod_name}: CPU={cpu_request}, RAM={ram_request}MB, duration={duration}h, deadline={deadline_hours}h")
                    
                except yaml.YAMLError as e:
                    logging.warning(f"Error parsing YAML document in {yaml_file}: {e}")
                    continue
                except Exception as e:
                    logging.warning(f"Error processing deployment in {yaml_file}: {e}")
                    continue
            
            return pods
        except Exception as e:
            logging.error(f"Error extracting pods from {yaml_file}: {e}")
            logging.error(traceback.format_exc())
            return []

    def _initialize_resource_state(self):
        """Initialize resource tracking for CPU and RAM."""
        self.resource_state = {
            "cpu": {},
            "ram": {}
        }
        
        for flv in self.all_flavours:
            self.resource_state["cpu"][flv.id] = {}
            self.resource_state["ram"][flv.id] = {}
            
            for ts in self.all_timeslots:
                self.resource_state["cpu"][flv.id][ts.id] = flv.totalCpu
                self.resource_state["ram"][flv.id][ts.id] = flv.totalRam

    def load_data(self, workloads_dir, nodes_file, forecasts_file):
        """
        Load all necessary data for optimization.
        """
        logging.info("🚀 Starting global optimization data loading")
        
        # 1. Load all nodes from nodes.yaml
        logging.info(f"📂 STEP 1: Loading infrastructure from {nodes_file}")
        self.all_flavours = self._load_nodes_from_yaml(nodes_file)
        if not self.all_flavours:
            logging.error("❌ Failed to load nodes. Aborting optimization.")
            return False
        
        logging.info(f"✅ SUCCESS: Loaded {len(self.all_flavours)} nodes")
        
        # 2. Load carbon intensity from all_forecasts.json
        logging.info(f"📂 STEP 2: Loading carbon intensity forecasts from {forecasts_file}")
        carbon_forecasts = self._load_carbon_forecasts(forecasts_file)
        if not carbon_forecasts:
            logging.error("❌ Failed to load carbon forecasts. Aborting optimization.")
            return False
        
        logging.info(f"✅ SUCCESS: Loaded carbon forecasts for {len(carbon_forecasts)} regions")
        
        # 3. Attach carbon forecasts to flavours
        logging.info(f"🔄 STEP 3: Attaching carbon forecasts to nodes")
        for flv in self.all_flavours:
            if flv.region in carbon_forecasts:
                flv.forecast = carbon_forecasts[flv.region]
                logging.info(f"  - Assigned {len(flv.forecast)} forecast data points to node {flv.id} (region {flv.region})")
            else:
                logging.warning(f"⚠️ No carbon forecast found for region {flv.region} (node {flv.id})")
                # Use first region's forecast as fallback
                if carbon_forecasts:
                    fallback_region = next(iter(carbon_forecasts.keys()))
                    flv.forecast = carbon_forecasts[fallback_region]
                    logging.warning(f"   Using {fallback_region} as fallback for {flv.id}")
        
        logging.info(f"✅ SUCCESS: Carbon forecasts attached to all nodes")
        
        # 4. Load all pods from all timeslot_*.yaml files
        logging.info(f"📂 STEP 4: Loading pods from timeslot YAML files in {workloads_dir}")
        
        all_pods = []
        if not os.path.isdir(workloads_dir):
            logging.error(f"❌ FAILED: Workloads directory {workloads_dir} does not exist")
            return False
        
        yaml_files = [f for f in os.listdir(workloads_dir) 
                     if f.startswith("timeslot_") and f.endswith(".yaml")]
        yaml_files.sort()  # Sort to ensure chronological order
        
        if not yaml_files:
            logging.error(f"❌ FAILED: No timeslot_*.yaml files found in {workloads_dir}")
            return False
        
        logging.info(f"  - Found {len(yaml_files)} timeslot files to process: {yaml_files}")
        
        for yaml_file in sorted(yaml_files):
            file_path = os.path.join(workloads_dir, yaml_file)
            try:
                pods = self._extract_pods_from_yaml(file_path)
                all_pods.extend(pods)
                logging.info(f"  - Loaded {len(pods)} pods from {yaml_file}")
            except Exception as e:
                logging.error(f"❌ Error processing {yaml_file}: {e}")
        
        if not all_pods:
            logging.error("❌ FAILED: No pods found in any timeslot files")
            return False
        
        self.pending_pods = all_pods
        logging.info(f"✅ SUCCESS: Loaded {len(all_pods)} total pods from {len(yaml_files)} timeslot files")
        
        # 5. Create timeslots
        logging.info(f"🔄 STEP 5: Creating timeslots for scheduling horizon")
        max_timeslot_id = max(pod.earliest_timeslot for pod in all_pods) + 24  # Add 24h window for scheduling
        self.all_timeslots = []
        
        for i in range(max_timeslot_id + 1):
            self.all_timeslots.append(CarbonAwareTimeslot(id=i, start_time=time.time() + i*3600, length=1))
        
        logging.info(f"✅ SUCCESS: Created {len(self.all_timeslots)} timeslots for scheduling")
        
        # 6. Initialize resource state
        logging.info(f"🔄 STEP 6: Initializing resource state for all nodes")
        self._initialize_resource_state()
        logging.info(f"✅ SUCCESS: Initialized resource state for {len(self.all_flavours)} nodes")
        
        return True
    
    def solve_global_optimization(self):
        """
        Solve the global optimization problem with MILP.
        Considers all pods together for optimal carbon-aware scheduling.
        """
        start_time = time.time()
        logging.info(f"🧮 Starting MILP global optimization with {len(self.pending_pods)} pods")
        
        try:
            # Create the optimization problem
            prob = pulp.LpProblem("CarbonAwareScheduling", pulp.LpMinimize)
            
            # Get resource state
            leftover_cpu = self.resource_state["cpu"]
            leftover_ram = self.resource_state["ram"]
            max_time_slots = len(self.all_timeslots)
            
            # Decision variables dictionary: key=(pod.id, flavour.id, timeslot.id)
            x = {}
            
            # Track valid placement options for each pod
            placements = {}
            valid_placement_count = 0
            
            logging.info(f"🔍 STEP 1: Creating decision variables")
            logging.info(f"  - Processing {len(self.pending_pods)} pods")
            
            # Process each pod to create variables and constraints
            for pod in self.pending_pods:
                logging.info(f"  - Creating variables for pod {pod.id}")
                pod_placements = []  # Will hold all valid (flv, ts, emissions) options for this pod
                valid_timeslot_set = set()  # Track unique timeslots with valid placements
                valid_options = 0
                
                # Process each possible node-timeslot combination for this pod
                for flv in self.all_flavours:
                    # Skip if node doesn't have enough total capacity
                    if flv.totalCpu < pod.cpuRequest or flv.totalRam < pod.ramRequest:
                        continue
                    
                    for ts in self.all_timeslots:
                        # Skip if timeslot is before pod's earliest allowed timeslot
                        if ts.id < pod.earliest_timeslot:
                            continue
                        
                        # Skip if would finish after pod's deadline
                        deadline_slot = pod.deadline_slot
                        if deadline_slot is not None and ts.id + pod.duration > deadline_slot:
                            continue
                        
                        # Check if placement is feasible throughout pod's duration
                        duration_feasible = True
                        resource_issues = []
                        
                        for offset in range(int(pod.duration)):
                            current_slot = ts.id + offset
                            
                            # Check if slot is within our time horizon
                            if current_slot >= max_time_slots:
                                duration_feasible = False
                                resource_issues.append(f"Timeslot {current_slot} beyond scheduling horizon")
                                break
                            
                            # Check CPU
                            if flv.id not in leftover_cpu or current_slot not in leftover_cpu[flv.id]:
                                duration_feasible = False
                                resource_issues.append(f"No CPU data for slot {current_slot}")
                                break
                            
                            if leftover_cpu[flv.id][current_slot] < pod.cpuRequest:
                                duration_feasible = False
                                resource_issues.append(f"CPU {leftover_cpu[flv.id][current_slot]} < {pod.cpuRequest}")
                                break
                            
                            # Check RAM
                            if flv.id not in leftover_ram or current_slot not in leftover_ram[flv.id]:
                                duration_feasible = False
                                resource_issues.append(f"No RAM data for slot {current_slot}")
                                break
                            
                            if leftover_ram[flv.id][current_slot] < pod.ramRequest:
                                duration_feasible = False
                                resource_issues.append(f"RAM {leftover_ram[flv.id][current_slot]} < {pod.ramRequest}")
                                break
                        
                        if duration_feasible:
                            # Calculate emissions for this placement
                            try:
                                emissions = compute_emissions(flv, ts.id, pod)
                                logging.debug(f"     - Valid placement: pod={pod.id}, node={flv.id}, ts={ts.id}, emissions={emissions:.2f}kgCO2e")
                                
                                placement_key = (pod.id, flv.id, ts.id)
                                # Store emissions for optimization and reporting
                                pod_placements.append((flv.id, ts.id, emissions))
                                valid_options += 1
                                valid_placement_count += 1
                                valid_timeslot_set.add(ts.id)  # Track this timeslot as having valid placements
                                
                                # Create decision variable
                                x[placement_key] = pulp.LpVariable(f"x_{pod.id}_{flv.id}_{ts.id}",
                                                                 cat=pulp.LpBinary)
                            except Exception as e:
                                logging.error(f"     - Error computing emissions for pod {pod.id} on {flv.id} at ts={ts.id}: {e}")
                                logging.error(traceback.format_exc())
                        else:
                            if valid_options < 10:  # Only log first 10 invalid options to avoid log spam
                                logging.debug(f"     - Invalid placement on {flv.id} at ts={ts.id}: {resource_issues}")
                
                placements[pod.id] = pod_placements
                valid_timeslots = len(valid_timeslot_set)  # Count unique timeslots
                logging.info(f"     - Result: {valid_options} valid placements across {valid_timeslots} valid timeslots")
                
                # Constraint: Each pod must be placed exactly once
                if pod_placements:
                    valid_vars = [x[(pod.id, flv_id, ts_id)] for flv_id, ts_id, _ in pod_placements 
                                 if (pod.id, flv_id, ts_id) in x]
                    if valid_vars:
                        prob += pulp.lpSum(valid_vars) == 1
                    else:
                        logging.warning(f"⚠️ No valid decision variables found for pod {pod.id}")
                else:
                    logging.warning(f"⚠️ No valid placements found for pod {pod.id}")
            
            # If no placements were found at all, exit early
            if not x:
                logging.warning("🚫 No valid placements found for any pod! Exiting solver.")
                return False
            
            logging.info(f"✅ Found {valid_placement_count} valid placement options for {len(self.pending_pods)} pods")
            
            logging.info(f"🔍 STEP 2: Setting up objective function and constraints")
            # Objective: Minimize total carbon emissions
            logging.info(f"  - Setting objective: minimize total carbon emissions")
            prob += pulp.lpSum([emissions * x[(pod_id, flv_id, ts_id)]
                                for pod_id, pod_placements in placements.items()
                                for flv_id, ts_id, emissions in pod_placements
                                if (pod_id, flv_id, ts_id) in x])  # Only include if variable exists
            
            logging.info(f"  - Setting up resource capacity constraints")
            # Resource constraints: Don't exceed capacity at any node/timeslot
            used_cpu = {}
            used_ram = {}
            
            # Add capacity constraints for each node and timeslot
            constraint_count = 0
            for pod_id, pod_placements_list in placements.items():  # Renamed for clarity
                current_pod_obj = next((p for p in self.pending_pods if p.id == pod_id), None)
                if not current_pod_obj:
                    logging.warning(f"  - Pod {pod_id} not found in self.pending_pods during constraint setup. Skipping.")
                    continue
                
                for flv_id, ts_id, _ in pod_placements_list:  # ts_id is the starting timeslot
                    # For each timeslot in the pod's duration
                    for offset in range(int(current_pod_obj.duration)):
                        current_ts = ts_id + offset  # Use ts_id from pod_placements_list
                        if current_ts >= max_time_slots:
                            continue
                        
                        # Add CPU constraint
                        cpu_key = (flv_id, current_ts)
                        if cpu_key not in used_cpu:
                            used_cpu[cpu_key] = 0
                        used_cpu[cpu_key] += current_pod_obj.cpuRequest * x[(pod_id, flv_id, ts_id)]
                        
                        # Add RAM constraint
                        ram_key = (flv_id, current_ts)
                        if ram_key not in used_ram:
                            used_ram[ram_key] = 0
                        used_ram[ram_key] += current_pod_obj.ramRequest * x[(pod_id, flv_id, ts_id)]
            
            # Add the capacity constraints to the model
            for (flv_id, ts_id), cpu_expr in used_cpu.items():
                if flv_id in leftover_cpu and ts_id in leftover_cpu[flv_id]:
                    constraint_count += 1
                    prob += cpu_expr <= leftover_cpu[flv_id][ts_id], f"CPU_{flv_id}_{ts_id}"
            
            for (flv_id, ts_id), ram_expr in used_ram.items():
                if flv_id in leftover_ram and ts_id in leftover_ram[flv_id]:
                    constraint_count += 1
                    prob += ram_expr <= leftover_ram[flv_id][ts_id], f"RAM_{flv_id}_{ts_id}"
            
            logging.info(f"  - Added {constraint_count} resource constraints")
            
            # STEP 3: Solve the MILP
            logging.info("🔍 STEP 3: Solving the MILP problem")
            logging.info("  - Starting CBC solver with optimized settings:")
            logging.info("    • Time limit: 20 seconds (reduced from 300s for efficiency)")
            logging.info("    • Gap tolerance: 1% (early termination when near-optimal)")
            logging.info("    • Enhanced heuristics and preprocessing enabled")
            pulp_solver = pulp.PULP_CBC_CMD(
                msg=True,                      # Enable verbose output
                timeLimit=20,                  # Reduced from 300s - most solutions found early
                gapRel=0.01                   # Stop when gap between best and bound <= 1%
            )
            
            solution_start_time = time.time()
            prob.solve(pulp_solver)
            solution_time = time.time() - solution_start_time
            
            self.status = pulp.LpStatus[prob.status]
            self.iterations = pulp_solver.solution_time if hasattr(pulp_solver, 'solution_time') else 0
            self.last_solve_time = solution_time
            
            if prob.status == pulp.LpStatusOptimal:
                self.has_solved = True
                objective_value = pulp.value(prob.objective)
                logging.info(f"✅ MILP solved! Status: {self.status}, Objective: {objective_value:.2f}kgCO2e")
                logging.info(f"  - Solution time: {solution_time:.2f}s")
                
                # Extract solution
                total_placed = 0
                for placement_key, var in x.items():
                    if var.value() > 0.5:  # Binary variable is active (allow for numerical precision)
                        pod_id, flv_id, ts_id = placement_key
                        # Find the emissions value for this placement
                        pod_placements = placements.get(pod_id, [])
                        emissions = next((e for f, t, e in pod_placements if f == flv_id and t == ts_id), 0)
                        
                        self.global_solution[pod_id] = (flv_id, ts_id, emissions)
                        total_placed += 1
                
                logging.info(f"  - Placed {total_placed}/{len(self.pending_pods)} pods")
                
                # Save results to CSV
                self._save_results_to_csv()
                return True
            else:
                logging.error(f"❌ MILP solver failed! Status: {self.status}")
                logging.error(f"  - Solution time: {solution_time:.2f}s")
                return False
        
        except Exception as e:
            logging.error(f"❌ Exception during optimization: {e}")
            logging.error(traceback.format_exc())
            return False
        finally:
            total_time = time.time() - start_time
            logging.info(f"⏱️ Total optimization process time: {total_time:.2f}s")

    def _save_results_to_csv(self):
        """Save placement results to CSV file."""
        try:
            with open(self.output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_HEADERS)
                
                for pod_id, (node_id, ts_id, emissions) in sorted(self.global_solution.items()):
                    pod = next((p for p in self.pending_pods if p.id == pod_id), None)
                    if not pod:
                        continue
                    
                    writer.writerow([
                        pod_id,
                        node_id,
                        ts_id,
                        pod.duration,
                        pod.cpuRequest,
                        pod.ramRequest,
                        emissions,
                        self.status,
                        self.iterations,
                        self.last_solve_time
                    ])
            
            logging.info(f"✅ Results saved to {self.output_file}")
            return True
        except Exception as e:
            logging.error(f"❌ Error saving results to CSV: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description='Standalone MILP Global Optimizer')
    parser.add_argument(
        '--workloads-dir',
        default='pkg/carbon-aware/server-python/workloads',
        help='Directory containing timeslot_*.yaml workload files'
    )
    parser.add_argument(
        '--nodes-file',
        default='pkg/carbon-aware/nodes.yaml',
        help='Path to nodes.yaml file'
    )
    parser.add_argument(
        '--forecasts-file',
        default='pkg/carbon-aware/server-python/all_forecasts.json',
        help='Path to all_forecasts.json file'
    )
    parser.add_argument(
        '--output',
        default='./global_optimization_placements.csv',
        help='Output CSV file for placements'
    )
    parser.add_argument(
        '--loglevel', 
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level'
    )
    
    args = parser.parse_args()
    
    # Configure logging with specified level
    log_level = getattr(logging, args.loglevel)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logging.info("🚀 Starting Standalone MILP Global Optimizer")
    
    # Create optimizer instance
    optimizer = GlobalOptimizer(args.output)
    
    # Load all data
    if not optimizer.load_data(args.workloads_dir, args.nodes_file, args.forecasts_file):
        logging.error("❌ Failed to load data. Exiting.")
        return 1
    
    # Solve optimization problem
    success = optimizer.solve_global_optimization()
    
    if success:
        logging.info("✅ Global optimization completed successfully")
        return 0
    else:
        logging.error("❌ Global optimization failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
