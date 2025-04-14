"""
Carbon-aware scheduling with global optimization using MILP.
This algorithm considers all pods simultaneously to find the system-wide optimal solution.
"""
import logging
import time
import os
import yaml
from typing import Dict, List, Optional, Tuple
import pulp

from carbon_aware.algorithms.base import SchedulingAlgorithm
from carbon_aware.models import CarbonAwarePod, CarbonAwareFlavour, CarbonAwareTimeslot
from carbon_aware.utils import is_timeslot_valid, compute_emissions


class GlobalOptimalAlgorithm(SchedulingAlgorithm):
    """
    Global optimization implementation of carbon-aware scheduling using MILP.
    This algorithm considers all pods together to find a system-wide optimal solution
    that minimizes total carbon emissions across the entire workload.
    """
    
    def __init__(self):
        # Add experiment logger and tracking properties
        self.experiment_logger = None
        self.iterations = 0
        self.status = None
        self.global_solution = {}  # Will store the global solution
        self.pending_pods = []     # Pods waiting for placement
        self.has_solved = False
        self.last_solve_time = 0
        self.solve_interval = 5
        
        # New attributes for pre-computed optimization
        self.timeslot_data = {}    # Stores loaded timeslot YAML data
        self.optimization_done = False  # Flag to indicate if pre-computation is done
        self.timeslots_dir = None  # Directory containing timeslot YAML files
        self.all_flavours = []     # All available node flavours
        self.all_timeslots = []    # All available timeslots
        self.resource_state = {    # Current resource state
            "cpu": {},             # CPU availability per node and timeslot
            "ram": {}              # RAM availability per node and timeslot
        }
    
    def initialize_for_precomputation(self, timeslots_dir: str):
        """
        Initialize the algorithm for pre-computation mode using timeslot YAML files.
        
        Args:
            timeslots_dir: Directory containing timeslot_*.yaml files
        """
        self.timeslots_dir = timeslots_dir
        self.load_all_timeslot_files()
        logging.info(f"Initialized global optimizer with {len(self.timeslot_data)} timeslots from {timeslots_dir}")
    
    def load_all_timeslot_files(self):
        """Load all timeslot_*.yaml files from the specified directory"""
        if not self.timeslots_dir or not os.path.isdir(self.timeslots_dir):
            logging.error(f"Invalid timeslots directory: {self.timeslots_dir}")
            return
        
        # Find all timeslot_*.yaml files
        yaml_files = [f for f in os.listdir(self.timeslots_dir) if f.startswith("timeslot_") and f.endswith(".yaml")]
        yaml_files.sort()  # Sort to ensure chronological order
        
        for filename in yaml_files:
            try:
                filepath = os.path.join(self.timeslots_dir, filename)
                with open(filepath, 'r') as f:
                    data = yaml.safe_load(f)
                    # Extract timeslot ID from filename (e.g., timeslot_5.yaml -> 5)
                    timeslot_id = int(filename.replace("timeslot_", "").replace(".yaml", ""))
                    self.timeslot_data[timeslot_id] = data
                    logging.debug(f"Loaded timeslot {timeslot_id} from {filename}")
            except Exception as e:
                logging.error(f"Error loading {filename}: {e}")
    
    def prepare_optimization_data(self):
        """
        Convert loaded YAML data into optimization inputs (flavours, timeslots, resources).
        This prepares all the data needed for the global optimization.
        """
        if not self.timeslot_data:
            logging.error("No timeslot data loaded. Call load_all_timeslot_files first.")
            return False
        
        # Create flavours from the first timeslot data (assuming node configs are consistent)
        first_timeslot = next(iter(self.timeslot_data.values()))
        nodes_data = first_timeslot.get("nodes", [])
        
        self.all_flavours = []
        for node in nodes_data:
            flavour = CarbonAwareFlavour(
                id=node["id"],
                embodiedCarbon=node.get("embodiedCarbon", 0.0),
                lifetime=node.get("lifetime", 87600.0),  # Default 10 years in hours
                totalCpu=node.get("totalCpu", 0.0),
                totalRam=node.get("totalRam", 0.0),
                totalStorage=node.get("totalStorage", 0.0),
                forecast=self._build_forecast_for_node(node["id"])
            )
            self.all_flavours.append(flavour)
        
        # Create timeslots
        self.all_timeslots = []
        for ts_id in sorted(self.timeslot_data.keys()):
            ts_data = self.timeslot_data[ts_id]
            start_time = ts_data.get("start_time", time.time() + ts_id * 3600)  # Default: current time + hours
            ts = CarbonAwareTimeslot(id=ts_id, start_time=start_time, length=1)  # 1-hour timeslots
            self.all_timeslots.append(ts)
        
        # Initialize resource state
        for flv in self.all_flavours:
            self.resource_state["cpu"][flv.id] = {}
            self.resource_state["ram"][flv.id] = {}
            
            # For each timeslot, get the resource availability
            for ts_id, ts_data in self.timeslot_data.items():
                node_data = next((n for n in ts_data.get("nodes", []) if n["id"] == flv.id), None)
                if node_data:
                    self.resource_state["cpu"][flv.id][ts_id] = node_data.get("availableCpu", flv.totalCpu)
                    self.resource_state["ram"][flv.id][ts_id] = node_data.get("availableRam", flv.totalRam)
                else:
                    # If node not found in this timeslot, use default values
                    self.resource_state["cpu"][flv.id][ts_id] = flv.totalCpu
                    self.resource_state["ram"][flv.id][ts_id] = flv.totalRam
        
        logging.info(f"Prepared optimization data with {len(self.all_flavours)} nodes and {len(self.all_timeslots)} timeslots")
        return True
    
    def _build_forecast_for_node(self, node_id: str) -> Dict[int, float]:
        """Build a carbon forecast for a node across all timeslots"""
        forecast = {}
        for ts_id, ts_data in self.timeslot_data.items():
            node_data = next((n for n in ts_data.get("nodes", []) if n["id"] == node_id), None)
            if node_data and "carbonIntensity" in node_data:
                forecast[ts_id] = node_data["carbonIntensity"]
            else:
                forecast[ts_id] = 0.0  # Default if not specified
        return forecast
    
    def precompute_global_optimization(self, pods_list: List[CarbonAwarePod]):
        """
        Precompute the global optimization for all pods against all timeslots.
        
        Args:
            pods_list: List of pods to optimize placement for
        
        Returns:
            Success flag indicating if optimization completed successfully
        """
        if not self.prepare_optimization_data():
            return False
            
        # Store the pods
        self.pending_pods = pods_list.copy()
        
        # Run the global optimization with our prepared data
        max_timeslots = max(ts.id for ts in self.all_timeslots) + 1
        self.solve_global_optimization(
            self.all_flavours,
            self.all_timeslots,
            self.resource_state["cpu"],
            self.resource_state["ram"],
            max_timeslots
        )
        
        # Mark as done
        self.optimization_done = self.has_solved
        
        # Log results
        if self.optimization_done:
            logging.info(f"Precomputation complete! Optimized {len(self.global_solution)}/{len(pods_list)} placements")
            # Show sample of results
            if self.global_solution:
                sample = list(self.global_solution.items())[:3]  # Show first 3
                for pod_id, (node, ts, emissions) in sample:
                    logging.info(f"Sample placement: Pod {pod_id} -> Node {node}, Timeslot {ts}, Emissions {emissions:.2f}")
        else:
            logging.error("Precomputation failed!")
        
        return self.optimization_done
    
    @property
    def name(self) -> str:
        return "Carbon-Aware-GlobalOptimal"
    
    def solve_global_optimization(self, 
                                 flavours: List[CarbonAwareFlavour],
                                 timeslots: List[CarbonAwareTimeslot],
                                 leftover_cpu: Dict[str, Dict[int, float]],
                                 leftover_ram: Dict[str, Dict[int, float]],
                                 max_time_slots: int = 48):
        """
        Solve the global optimization problem for all pending pods.
        """
        if not self.pending_pods:
            logging.info("No pods pending for placement")
            return
            
        logging.info(f"Starting global optimization for {len(self.pending_pods)} pods")
        
        # Debug resource availability
        for flv in flavours:
            logging.debug(f"Node {flv.id} CPU at t0: {leftover_cpu[flv.id][0]}, RAM: {leftover_ram[flv.id][0]}")
        
        try:
            # Create optimization problem
            prob = pulp.LpProblem("Global_Carbon_Aware_Scheduling", pulp.LpMinimize)
            
            # Initialize decision variables and constraints
            placements = {}  # Will store all valid placements
            x = {}  # Decision variables
            valid_placement_count = 0
            
            # For each pod, find all valid placement options
            for pod in self.pending_pods:
                pod_placements = []
                valid_timeslots = 0
                valid_options = 0
                
                logging.debug(f"Finding placements for pod {pod.id}")
                logging.debug(f"  Requirements: CPU={pod.cpuRequest}, RAM={pod.ramRequest}, Duration={pod.duration}h")
                
                for ts in timeslots:
                    # Check if timeslot is valid for this pod
                    if not is_timeslot_valid(ts, pod):
                        continue
                        
                    valid_timeslots += 1
                    
                    for flv in flavours:
                        # Check if placement is feasible for the entire duration
                        duration_feasible = True
                        resource_issues = []
                        
                        for slot_offset in range(int(pod.duration)):
                            current_slot = ts.id + slot_offset
                            if current_slot >= max_time_slots:
                                duration_feasible = False
                                resource_issues.append(f"Slot {current_slot} exceeds max timeslots")
                                break
                            
                            if leftover_cpu[flv.id][current_slot] < pod.cpuRequest:
                                duration_feasible = False
                                resource_issues.append(f"CPU {leftover_cpu[flv.id][current_slot]} < {pod.cpuRequest}")
                                break
                                
                            if leftover_ram[flv.id][current_slot] < pod.ramRequest:
                                duration_feasible = False
                                resource_issues.append(f"RAM {leftover_ram[flv.id][current_slot]} < {pod.ramRequest}")
                                break
                        
                        if duration_feasible:
                            # Calculate emissions for this placement
                            emissions = compute_emissions(flv, ts.id, pod)
                            placement_key = (pod.id, flv.id, ts.id)
                            pod_placements.append((flv.id, ts.id, emissions))
                            valid_options += 1
                            valid_placement_count += 1
                            
                            # Create decision variable
                            x[placement_key] = pulp.LpVariable(f"x_{pod.id}_{flv.id}_{ts.id}", 
                                                              cat=pulp.LpBinary)
                        else:
                            logging.debug(f"  Invalid placement on {flv.id} at ts={ts.id}: {resource_issues}")
                
                placements[pod.id] = pod_placements
                logging.info(f"Pod {pod.id}: {valid_options} valid placements across {valid_timeslots} valid timeslots")
                
                # Constraint: Each pod must be placed exactly once
                if pod_placements:
                    prob += pulp.lpSum([x[(pod.id, flv_id, ts_id)] for flv_id, ts_id, _ in pod_placements]) == 1
                else:
                    logging.warning(f"⚠️ No valid placements found for pod {pod.id}")
            
            # If no placements were found at all, exit early
            if not x:
                logging.warning("No valid placements found for any pod!")
                return
                
            logging.info(f"Found {valid_placement_count} valid placements for {len(self.pending_pods)} pods")
            
            # Objective: Minimize total carbon emissions
            prob += pulp.lpSum([emissions * x[(pod.id, flv_id, ts_id)] 
                              for pod.id, pod_placements in placements.items()
                              for flv_id, ts_id, emissions in pod_placements])
            
            # Resource constraints: Don't exceed capacity at any node/timeslot
            used_cpu = {}
            used_ram = {}
            
            # Add capacity constraints for each node and timeslot
            for pod_id, pod_placements in placements.items():
                pod = next(p for p in self.pending_pods if p.id == pod_id)
                
                for flv_id, ts_id, _ in pod_placements:
                    # For each timeslot in the pod's duration
                    for offset in range(int(pod.duration)):
                        current_ts = ts.id + offset
                        if current_ts >= max_time_slots:
                            continue
                            
                        # Add CPU constraint
                        cpu_key = (flv_id, current_ts)
                        if cpu_key not in used_cpu:
                            used_cpu[cpu_key] = 0
                        used_cpu[cpu_key] += pod.cpuRequest * x[(pod_id, flv_id, ts_id)]
                        
                        # Add RAM constraint
                        ram_key = (flv_id, current_ts)
                        if ram_key not in used_ram:
                            used_ram[ram_key] = 0
                        used_ram[ram_key] += pod.ramRequest * x[(pod_id, flv_id, ts_id)]
            
            # Add actual capacity constraints
            for flv in flavours:
                for ts_id in range(max_time_slots):
                    cpu_key = (flv.id, ts_id)
                    ram_key = (flv.id, ts_id)
                    
                    if cpu_key in used_cpu:
                        prob += used_cpu[cpu_key] <= leftover_cpu[flv.id][ts_id]
                    if ram_key in used_ram:
                        prob += used_ram[ram_key] <= leftover_ram[flv.id][ts_id]
            
            # Solve the problem
            logging.info(f"Solving global optimization with {len(x)} decision variables")
            solver_start_time = time.time()
            
            # Use CBC solver with a time limit and verbose output
            prob.solve(pulp.PULP_CBC_CMD(msg=True, timeLimit=60))
            
            solution_time = time.time() - solver_start_time
            self.status = pulp.LpStatus[prob.status]
            self.iterations = prob.solverModel.Iterations if hasattr(prob.solverModel, 'Iterations') else 0
            
            logging.info(f"Global optimization completed in {solution_time:.3f}s with status: {self.status}")
            
            # Extract solution if optimal
            if prob.status == pulp.LpStatusOptimal:
                solution = {}
                total_emissions = 0
                
                for pod_id, pod_placements in placements.items():
                    for flv_id, ts_id, emissions in pod_placements:
                        placement_key = (pod_id, flv_id, ts_id)
                        if x[placement_key].value() is not None and x[placement_key].value() > 0.5:  # Selected placement
                            solution[pod_id] = (flv_id, ts_id, emissions)
                            total_emissions += emissions
                
                logging.info(f"Found optimal global solution with total emissions: {total_emissions:.2f}")
                logging.info(f"Placed {len(solution)}/{len(self.pending_pods)} pods")
                
                # Update the solution and remove placed pods
                self.global_solution.update(solution)
                self.pending_pods = [p for p in self.pending_pods if p.id not in solution]
                self.has_solved = True
            else:
                logging.warning(f"No optimal global solution found, status: {self.status}")
                
        except Exception as e:
            logging.exception(f"Error in global optimization: {e}")
        
        self.last_solve_time = time.time()
    
    def find_placement(
        self,
        pod: CarbonAwarePod,
        flavours: List[CarbonAwareFlavour],
        timeslots: List[CarbonAwareTimeslot],
        leftover_cpu: Dict[str, Dict[int, float]],
        leftover_ram: Dict[str, Dict[int, float]],
        max_time_slots: int = 48
    ) -> Tuple[Optional[CarbonAwareFlavour], Optional[CarbonAwareTimeslot], float]:
        """
        Return placement for a pod, solving globally if needed.
        """
        start_time = time.time()
        logging.debug(f"Finding placement for pod {pod.id}")
        
        # Check if this pod already has a solution
        if pod.id in self.global_solution:
            flv_id, ts_id, emissions = self.global_solution[pod.id]
            
            # Find the corresponding objects
            flv = next((f for f in flavours if f.id == flv_id), None)
            ts = next((t for t in timeslots if t.id == ts_id), None)
            
            if flv and ts:
                logging.info(f"Using cached placement for pod {pod.id}: {flv_id}, timeslot {ts_id}")
                if self.experiment_logger:
                    execution_time = time.time() - start_time
                    self.experiment_logger.record_placement(
                        pod_id=pod.id, success=True, execution_time=execution_time,
                        emissions=emissions, considered_options=len(self.global_solution),
                        selected_node=flv_id, selected_timeslot=ts_id,
                        solver_iterations=self.iterations, solver_status=self.status
                    )
                return flv, ts, emissions
        
        # Add this pod to pending list if not already there
        if not any(p.id == pod.id for p in self.pending_pods):
            self.pending_pods.append(pod)
            logging.info(f"Added pod {pod.id} to pending queue (now {len(self.pending_pods)} pods)")
        
        # Solve if needed
        current_time = time.time()
        if (current_time - self.last_solve_time > self.solve_interval) or len(self.pending_pods) >= 5:
            self.solve_global_optimization(flavours, timeslots, leftover_cpu, leftover_ram, max_time_slots)
        
        # Check if we now have a solution
        if pod.id in self.global_solution:
            flv_id, ts_id, emissions = self.global_solution[pod.id]
            flv = next((f for f in flavours if f.id == flv_id), None)
            ts = next((t for t in timeslots if t.id == ts_id), None)
            
            if flv and ts:
                if self.experiment_logger:
                    execution_time = time.time() - start_time
                    self.experiment_logger.record_placement(
                        pod_id=pod.id, success=True, execution_time=execution_time,
                        emissions=emissions, considered_options=len(self.global_solution),
                        selected_node=flv_id, selected_timeslot=ts_id,
                        solver_iterations=self.iterations, solver_status=self.status
                    )
                return flv, ts, emissions
        
        # No solution found
        logging.warning(f"No placement found for pod {pod.id}")
        if self.experiment_logger:
            execution_time = time.time() - start_time
            self.experiment_logger.record_placement(
                pod_id=pod.id, success=False, execution_time=execution_time,
                emissions=0.0, considered_options=0, selected_node=None, 
                selected_timeslot=None, solver_iterations=0, solver_status="No Solution"
            )
        return None, None, float('inf')