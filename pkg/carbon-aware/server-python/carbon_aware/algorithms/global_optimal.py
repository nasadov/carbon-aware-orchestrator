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
import csv
from datetime import datetime

from carbon_aware.algorithms.base import SchedulingAlgorithm
from carbon_aware.models import CarbonAwarePod, CarbonAwareFlavour, CarbonAwareTimeslot
from carbon_aware.utils import is_timeslot_valid, compute_emissions


class GlobalOptimalAlgorithm(SchedulingAlgorithm):
    """
    Global optimization implementation of carbon-aware scheduling using MILP.
    This algorithm considers all pods together to find a system-wide optimal solution
    that minimizes total carbon emissions across the entire workload.
    """
    SESSION_CSV_FILENAME = "global_optimal_placements_session.csv"  # New constant for session-wide CSV
    CSV_HEADERS = [
        "pod_id", "node_id", "start_slot", "duration",
        "cpu_request", "ram_request", "total_carbon_emissions",
        "solver_status", "solver_iterations", "solution_time_seconds"
    ]

    def __init__(self):
        # Add experiment logger and tracking properties
        self.experiment_logger = None
        self.iterations = 0
        self.status = None
        self.global_solution = {}  # Will store the global solution
        self.pending_pods = []     # Pods waiting for placement
        self.has_solved = False
        self.last_solve_time = 0
        self.solve_interval = 5  # Default: 5 seconds or 5 pods

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

        # CSV Logging attributes for session-wide logging
        self._session_log_dir: Optional[str] = None
        self._session_csv_path: Optional[str] = None  # Path to the session-wide CSV
        self._csv_writer = None
        self._csv_file_handle = None

    def _set_pod_earliest_timeslot(self, pod: CarbonAwarePod):
        """
        Set the earliest_timeslot attribute for a pod based on its ID.
        
        The pod ID format (mXXX) determines which timeslot_X.yaml file it came from:
        - m000-m004: from timeslot_0.yaml (earliest_timeslot = 0)
        - m005-m009: from timeslot_1.yaml (earliest_timeslot = 1)
        - m010-m014: from timeslot_2.yaml (earliest_timeslot = 2)
        - m015-m018: from timeslot_3.yaml (earliest_timeslot = 3)
        - m019-m023: from timeslot_5.yaml (earliest_timeslot = 5)
        - m024-m025: from timeslot_6.yaml (earliest_timeslot = 6) 
        - m026-m030: from timeslot_7.yaml (earliest_timeslot = 7)
        - m031-m033: from timeslot_8.yaml (earliest_timeslot = 8)
        - m034-m038: from timeslot_9.yaml (earliest_timeslot = 9)
        - m039-m043: from timeslot_10.yaml (earliest_timeslot = 10)
        - m044-m046: from timeslot_11.yaml (earliest_timeslot = 11)
        """
        import re
        
        # Extract the pod number from the ID (e.g., 019 from m019-duration-3h-deadline-9h)
        match = re.match(r'([a-zA-Z]+)(\d+)[-_]?', pod.id)
        if match:
            pod_num = int(match.group(2))
            # Map pods to their source file's timeslot number
            if 0 <= pod_num <= 4:
                earliest_ts = 0  # timeslot_0.yaml
            elif 5 <= pod_num <= 9:
                earliest_ts = 1  # timeslot_1.yaml
            elif 10 <= pod_num <= 14:
                earliest_ts = 2  # timeslot_2.yaml
            elif 15 <= pod_num <= 18:
                earliest_ts = 3  # timeslot_3.yaml
            elif 19 <= pod_num <= 23:
                earliest_ts = 5  # timeslot_5.yaml
            elif 24 <= pod_num <= 25:
                earliest_ts = 6  # timeslot_6.yaml
            elif 26 <= pod_num <= 30:
                earliest_ts = 7  # timeslot_7.yaml
            elif 31 <= pod_num <= 33:
                earliest_ts = 8  # timeslot_8.yaml
            elif 34 <= pod_num <= 38:
                earliest_ts = 9  # timeslot_9.yaml
            elif 39 <= pod_num <= 43:
                earliest_ts = 10  # timeslot_10.yaml
            elif 44 <= pod_num <= 46:
                earliest_ts = 11  # timeslot_11.yaml
            else:
                # If we can't determine, ensure it's within the valid range (0-23)
                earliest_ts = min(pod_num, 23)
            
            # Set the earliest_timeslot and log it
            pod.earliest_timeslot = earliest_ts
            logging.info(f"🔒 Pod {pod.id} has earliest_timeslot={pod.earliest_timeslot} (from timeslot_{earliest_ts}.yaml)")

    def __del__(self):
        self.close_csv()

    @staticmethod
    def _ensure_dir_exists(directory_path: str):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            logging.info(f"Created directory: {directory_path}")

    def set_base_log_dir(self, base_dir: str):
        """Sets the base directory for session logs."""
        self._session_log_dir = base_dir

    def setup_session_placement_log(self):
        """Initializes the session-wide CSV file for logging placements."""
        if not self._session_log_dir:
            logging.error("Session log directory not set. Cannot initialize CSV logging for global optimal algorithm.")
            # Fallback to a default local directory to prevent crashes if not set via main.py
            self._session_log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "analysis", "global_optimal_fallback_logs_session"))
            GlobalOptimalAlgorithm._ensure_dir_exists(self._session_log_dir)
            logging.warning(f"Using fallback session log directory: {self._session_log_dir}")

        self._session_csv_path = os.path.join(self._session_log_dir, self.SESSION_CSV_FILENAME)
        
        self.close_csv()  # Close any existing file handle before opening a new one or reopening

        file_exists = os.path.exists(self._session_csv_path)
        is_empty = os.path.getsize(self._session_csv_path) == 0 if file_exists else True

        try:
            # Open in append mode, create if not exists
            self._csv_file_handle = open(self._session_csv_path, 'a', newline='')
            self._csv_writer = csv.writer(self._csv_file_handle)

            if not file_exists or is_empty:
                self._csv_writer.writerow(self.CSV_HEADERS)
                logging.info(f"Initialized new placement CSV with headers: {self._session_csv_path}")
            else:
                logging.info(f"Appending to existing placement CSV: {self._session_csv_path}")
            
            self._csv_file_handle.flush()  # Ensure headers are written if it's a new file

        except IOError as e:
            logging.error(f"Failed to open or write to session CSV file: {self._session_csv_path}. Error: {e}")
            self._csv_writer = None
            self._csv_file_handle = None
            self._session_csv_path = None

    def close_csv(self):
        if self._csv_file_handle:
            try:
                self._csv_file_handle.close()
            except Exception as e:
                logging.error(f"Error closing CSV file: {e}")
            self._csv_file_handle = None
            self._csv_writer = None

    def _write_placement_to_csv(self, pod_id: str, node_id: str, start_slot: int, duration: float,
                                cpu_request: float, ram_request: float, total_carbon_emissions: float,
                                solver_status: str, solver_iterations: int, solution_time_seconds: float):
        if self._csv_writer and self._csv_file_handle:
            try:
                row = [
                    pod_id, node_id, start_slot, duration,
                    cpu_request, ram_request, total_carbon_emissions,
                    solver_status, solver_iterations, solution_time_seconds
                ]
                self._csv_writer.writerow(row)
                self._csv_file_handle.flush()  # Ensure data is written to disk
            except Exception as e:
                logging.error(f"Error writing to CSV: {e}")

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

        # Store the pods and set earliest_timeslot for each pod
        self.pending_pods = pods_list.copy()
        # Set earliest_timeslot for all pods
        for pod in self.pending_pods:
            self._set_pod_earliest_timeslot(pod)
        
        # Set earliest_timeslot for each pod before optimization
        for pod in self.pending_pods:
            self._set_pod_earliest_timeslot(pod)

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

        # Set earliest_timeslot for all pending pods
        for pod in self.pending_pods:
            self._set_pod_earliest_timeslot(pod)

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
                                for pod_id, pod_placements in placements.items()
                                for flv_id, ts_id, emissions in pod_placements])

            # Resource constraints: Don't exceed capacity at any node/timeslot
            used_cpu = {}
            used_ram = {}

            # Add capacity constraints for each node and timeslot
            for pod_id, pod_placements_list in placements.items():  # Renamed for clarity
                current_pod_obj = next((p for p in self.pending_pods if p.id == pod_id), None)
                if not current_pod_obj:
                    logging.warning(f"Pod {pod_id} not found in self.pending_pods during constraint setup. Skipping.")
                    continue

                for flv_id, ts_id, _ in pod_placements_list:  # ts_id is the starting timeslot
                    # For each timeslot in the pod's duration
                    for offset in range(int(current_pod_obj.duration)):
                        current_ts = ts_id + offset  # CORRECTED: Use ts_id from pod_placements_list
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
                total_emissions_objective = 0.0

                # Create a dictionary for quick pod lookup
                pending_pods_dict = {p.id: p for p in self.pending_pods}

                for pod_id_sol, current_pod_placements in placements.items():
                    for flv_id_sol, ts_id_sol, emissions_sol in current_pod_placements:
                        placement_key = (pod_id_sol, flv_id_sol, ts_id_sol)

                        if placement_key not in x:
                            logging.debug(f"Placement key {placement_key} not found in decision variables. Skipping.")
                            continue

                        if x[placement_key].value() is not None and x[placement_key].value() > 0.5:  # Selected placement
                            current_pod = pending_pods_dict.get(pod_id_sol)
                            if not current_pod:
                                logging.error(f"Pod {pod_id_sol} not found in pending_pods_dict. Skipping CSV write for this placement.")
                                continue

                            # Log this placement to CSV
                            self._write_placement_to_csv(
                                pod_id=current_pod.id,
                                node_id=flv_id_sol,
                                start_slot=ts_id_sol,
                                duration=current_pod.duration,
                                cpu_request=current_pod.cpuRequest,
                                ram_request=current_pod.ramRequest,
                                total_carbon_emissions=emissions_sol,  # This is per-pod carbon for this placement
                                solver_status=str(self.status),  # self.status is set after prob.solve()
                                solver_iterations=self.iterations,  # self.iterations is set after prob.solve()
                                solution_time_seconds=solution_time  # solution_time is calculated before this loop
                            )

                            solution[pod_id_sol] = (flv_id_sol, ts_id_sol, emissions_sol)
                            total_emissions_objective += emissions_sol

                logging.info(f"Found optimal global solution with total objective emissions: {total_emissions_objective:.2f}")
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
        # Set earliest_timeslot based on pod ID before scheduling
        self._set_pod_earliest_timeslot(pod)
        logging.debug(f"Finding placement for pod {pod.id}")
        
        # Set earliest_timeslot based on pod ID before scheduling
        self._set_pod_earliest_timeslot(pod)

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