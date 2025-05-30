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
        """
        Initialize the global optimal algorithm with necessary tracking and state.
        Sets up logging structures, solver state, and resource tracking.
        """
        # Configure logging for the algorithm
        logging.info("🔧 Initializing Global Optimal Algorithm")
        
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
        
        logging.info(f"✅ Global Optimal Algorithm instance created with CSV headers: {self.CSV_HEADERS}")
        logging.info(f"📋 This algorithm will track placements in a CSV file once set_base_log_dir is called")

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
        logging.info(f"📊 Setting up CSV log file at: {self._session_csv_path}")
        
        self.close_csv()  # Close any existing file handle before opening a new one or reopening

        file_exists = os.path.exists(self._session_csv_path)
        is_empty = os.path.getsize(self._session_csv_path) == 0 if file_exists else True

        try:
            # Open in append mode, create if not exists
            self._csv_file_handle = open(self._session_csv_path, 'a', newline='')
            self._csv_writer = csv.writer(self._csv_file_handle)

            if not file_exists or is_empty:
                self._csv_writer.writerow(self.CSV_HEADERS)
                logging.info(f"✅ Initialized new placement CSV with headers: {self._session_csv_path}")
                logging.info(f"📋 CSV Headers: {', '.join(self.CSV_HEADERS)}")
            else:
                logging.info(f"✅ Appending to existing placement CSV: {self._session_csv_path}")
                
                # Check if file is writable
                try:
                    # Try writing a test row and then removing it (in a temporary copy of the file)
                    import tempfile
                    import shutil
                    
                    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', newline='')
                    shutil.copy2(self._session_csv_path, temp_file.name)
                    
                    with open(temp_file.name, 'a', newline='') as test_file:
                        test_writer = csv.writer(test_file)
                        test_writer.writerow(["TEST_WRITE"] * len(self.CSV_HEADERS))
                    
                    os.unlink(temp_file.name)
                    logging.info(f"✅ Verified CSV file is writable")
                except Exception as test_e:
                    logging.error(f"⚠️ CSV file might not be writable: {test_e}")
            
            self._csv_file_handle.flush()  # Ensure headers are written if it's a new file
            logging.info(f"✅ CSV file ready for recording placements")

        except IOError as e:
            logging.error(f"❌ Failed to open or write to session CSV file: {self._session_csv_path}. Error: {e}")
            import traceback
            logging.error(traceback.format_exc())
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
        """
        Write a placement decision to the CSV file.
        
        Args:
            pod_id: ID of the pod being placed
            node_id: ID of the node selected for placement
            start_slot: Starting timeslot ID
            duration: Pod runtime duration in hours
            cpu_request: CPU request for the pod
            ram_request: RAM request for the pod in MB
            total_carbon_emissions: Total carbon emissions for this placement in kg CO2e
            solver_status: Status of the solver (e.g., "Optimal")
            solver_iterations: Number of solver iterations
            solution_time_seconds: Time taken to solve the problem in seconds
        """
        if self._csv_writer and self._csv_file_handle:
            try:
                row = [
                    pod_id, node_id, start_slot, duration,
                    cpu_request, ram_request, total_carbon_emissions,
                    solver_status, solver_iterations, solution_time_seconds
                ]
                
                # Add detailed logging
                logging.info(f"📝 Writing CSV row for pod {pod_id}:")
                logging.info(f"   - Placement: Node={node_id}, Start={start_slot}, Duration={duration}h")
                logging.info(f"   - Resources: CPU={cpu_request:.2f}, RAM={ram_request:.0f}MB")
                logging.info(f"   - Emissions: {total_carbon_emissions:.4f}kg CO2e")
                
                # Write to CSV
                self._csv_writer.writerow(row)
                self._csv_file_handle.flush()  # Ensure data is written to disk
                
                # Verify the data was written
                import os
                if hasattr(self, '_session_csv_path') and os.path.exists(self._session_csv_path):
                    file_size = os.path.getsize(self._session_csv_path)
                    logging.debug(f"   - CSV file size after write: {file_size} bytes")
                
                logging.info(f"✅ Successfully wrote placement data for pod {pod_id}")
            except Exception as e:
                import traceback
                logging.error(f"❌ Error writing to CSV: {e}")
                logging.error(traceback.format_exc())

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

    def solve_global_optimization(
        self,
        flavours: List[CarbonAwareFlavour],
        timeslots: List[CarbonAwareTimeslot],
        leftover_cpu: Dict[str, Dict[int, float]],
        leftover_ram: Dict[str, Dict[int, float]],
        max_time_slots: int = 48
    ) -> None:
        """
        Solve the global optimization problem for multiple pods at once.
        
        Args:
            flavours: List of available flavours (nodes)
            timeslots: List of available timeslots
            leftover_cpu: Available CPU resources for each node and timeslot
            leftover_ram: Available RAM resources for each node and timeslot
            max_time_slots: Maximum number of timeslots to consider
        """
        import traceback
        import time
        import os
        
        # Log detailed information about the inputs
        logging.info(f"====== GLOBAL OPTIMIZATION SOLVER START ======")
        logging.info(f"Input parameters:")
        logging.info(f"- Flavours (nodes): {len(flavours)}")
        logging.info(f"- Timeslots: {len(timeslots)} (max: {max_time_slots})")
        logging.info(f"- Pending pods: {len(self.pending_pods)}")
        
        # Log CSV status before starting optimization
        if hasattr(self, '_session_csv_path') and self._session_csv_path:
            if os.path.exists(self._session_csv_path):
                logging.info(f"CSV file exists at: {self._session_csv_path}")
                file_size = os.path.getsize(self._session_csv_path)
                logging.info(f"- Current file size: {file_size} bytes")
            else:
                logging.warning(f"CSV file path is set but file doesn't exist: {self._session_csv_path}")
        else:
            logging.warning("CSV file path not set - results may not be logged correctly")
        
        try:
            import pulp

            # Ensure we have pods to place
            if not self.pending_pods:
                logging.warning("🚫 No pending pods to place. Exiting solver.")
                return

            # Create the LP problem
            logging.info(f"🔍 STEP 1: Setting up MILP problem")
            prob = pulp.LpProblem("CarbonAwareScheduling", pulp.LpMinimize)

            # Decision variables dictionary
            x = {}

            # Placement options for each pod
            placements = {}  # pod_id -> list of (flv_id, ts_id, emissions)
            valid_placement_count = 0  # Track total number of valid placement options across all pods

            logging.info(f"🔍 STEP 2: Finding valid placement options for {len(self.pending_pods)} pods")
            # For each pod, find valid placement options
            for pod_index, pod in enumerate(self.pending_pods):
                valid_options = 0  # Track valid placements for this pod
                valid_timeslots = 0  # Track valid timeslots for this pod
                pod_placements = []  # List of valid (flv_id, ts_id, emissions) for this pod
                
                # Make sure earliest_timeslot is set
                self._set_pod_earliest_timeslot(pod)
                
                logging.info(f"  ➡️ Processing pod {pod_index+1}/{len(self.pending_pods)}: {pod.id}")
                logging.info(f"     - Requirements: CPU={pod.cpuRequest}, RAM={pod.ramRequest}, Duration={pod.duration}h")
                logging.info(f"     - Constraints: earliest_timeslot={pod.earliest_timeslot}, deadline_slot={pod.deadline_slot}")

                # For each timeslot, check if it's within the deadline
                for ts in timeslots:
                    # Skip timeslots before the pod's earliest_timeslot
                    if ts.id < pod.earliest_timeslot:
                        continue

                    # Skip timeslots that would make the pod miss its deadline
                    deadline_slot = pod.deadline_slot
                    if deadline_slot is not None and ts.id + pod.duration > deadline_slot:
                        logging.debug(f"     - Skipping timeslot {ts.id} for pod {pod.id}: beyond deadline slot {deadline_slot}")
                        continue

                    valid_timeslots += 1

                    # For each flavour (node), check if it has enough resources
                    for flv in flavours:
                        if pod.duration < 1:
                            logging.warning(f"     - Pod {pod.id} has duration < 1 hour ({pod.duration}h), setting to 1h minimum")
                            pod.duration = 1

                        # Check if the pod can fit in this flavour over its entire duration
                        resource_issues = []
                        duration_feasible = True

                        for offset in range(int(pod.duration)):
                            current_slot = ts.id + offset
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
                logging.info(f"     - Result: {valid_options} valid placements across {valid_timeslots} valid timeslots")

                # Constraint: Each pod must be placed exactly once
                if pod_placements:
                    prob += pulp.lpSum([x[(pod.id, flv_id, ts_id)] for flv_id, ts_id, _ in pod_placements]) == 1
                else:
                    logging.warning(f"⚠️ No valid placements found for pod {pod.id}")

            # If no placements were found at all, exit early
            if not x:
                logging.warning("🚫 No valid placements found for any pod! Exiting solver.")
                return

            logging.info(f"✅ Found {valid_placement_count} valid placement options for {len(self.pending_pods)} pods")

            logging.info(f"🔍 STEP 3: Setting up objective function and constraints")
            # Objective: Minimize total carbon emissions
            logging.info(f"  - Setting objective: minimize total carbon emissions")
            prob += pulp.lpSum([emissions * x[(pod_id, flv_id, ts_id)]
                                for pod_id, pod_placements in placements.items()
                                for flv_id, ts_id, emissions in pod_placements])

            logging.info(f"  - Setting up resource capacity constraints")
            # Resource constraints: Don't exceed capacity at any node/timeslot
            
            # Dictionary to collect resource usage expressions for each (node, timeslot)
            cpu_usage = {}  # (flv_id, ts_id) -> list of expressions
            ram_usage = {}  # (flv_id, ts_id) -> list of expressions

            # Collect resource usage expressions for each node-timeslot combination
            for pod_id, pod_placements_list in placements.items():
                current_pod_obj = next((p for p in self.pending_pods if p.id == pod_id), None)
                if not current_pod_obj:
                    logging.warning(f"  - Pod {pod_id} not found in self.pending_pods during constraint setup. Skipping.")
                    continue

                for flv_id, ts_id, _ in pod_placements_list:  # ts_id is the starting timeslot
                    # For each timeslot in the pod's duration
                    for offset in range(int(current_pod_obj.duration)):
                        current_ts = ts_id + offset
                        if current_ts >= max_time_slots:
                            continue

                        # Collect CPU usage expressions
                        cpu_key = (flv_id, current_ts)
                        if cpu_key not in cpu_usage:
                            cpu_usage[cpu_key] = []
                        cpu_usage[cpu_key].append(current_pod_obj.cpuRequest * x[(pod_id, flv_id, ts_id)])

                        # Collect RAM usage expressions
                        ram_key = (flv_id, current_ts)
                        if ram_key not in ram_usage:
                            ram_usage[ram_key] = []
                        ram_usage[ram_key].append(current_pod_obj.ramRequest * x[(pod_id, flv_id, ts_id)])

            # Add capacity constraints to the model using available leftover capacity as limits
            constraint_count = 0

            # Add CPU capacity constraints using leftover CPU
            for (flv_id, ts_id), usage_expressions in cpu_usage.items():
                if flv_id in leftover_cpu and ts_id in leftover_cpu[flv_id]:
                    constraint_count += 1
                    cpu_limit = leftover_cpu[flv_id][ts_id]
                    prob += pulp.lpSum(usage_expressions) <= cpu_limit, f"CPU_{flv_id}_{ts_id}"
                    logging.debug(f"     - Added CPU constraint for node {flv_id}, timeslot {ts_id}: usage <= {cpu_limit}")
                    # Additional validation logging
                    if len(usage_expressions) > 0:
                        logging.debug(f"       Constraint involves {len(usage_expressions)} pods for node {flv_id} at timeslot {ts_id}")
                else:
                    logging.warning(f"     - Missing CPU capacity data for node {flv_id}, timeslot {ts_id}")

            # Add RAM capacity constraints using leftover RAM
            for (flv_id, ts_id), usage_expressions in ram_usage.items():
                if flv_id in leftover_ram and ts_id in leftover_ram[flv_id]:
                    constraint_count += 1
                    ram_limit = leftover_ram[flv_id][ts_id]
                    prob += pulp.lpSum(usage_expressions) <= ram_limit, f"RAM_{flv_id}_{ts_id}"
                    logging.debug(f"     - Added RAM constraint for node {flv_id}, timeslot {ts_id}: usage <= {ram_limit}")
                    # Additional validation logging
                    if len(usage_expressions) > 0:
                        logging.debug(f"       Constraint involves {len(usage_expressions)} pods for node {flv_id} at timeslot {ts_id}")
                else:
                    logging.warning(f"     - Missing RAM capacity data for node {flv_id}, timeslot {ts_id}")
            
            logging.info(f"  - Created {constraint_count} resource capacity constraints")
            logging.info(f"  - Final problem size: {len(x)} variables, {len(prob.constraints)} constraints")

            # Solve the problem
            logging.info(f"🔍 STEP 4: Solving global optimization problem")
            logging.info(f"  - Starting CBC solver with constraint-enforcing settings:")
            logging.info(f"    • Time limit: 20 seconds")
            logging.info(f"    • NO gap tolerance (strict constraint enforcement)")
            logging.info(f"    • Enhanced heuristics and preprocessing enabled")
            logging.info(f"    • PRIORITY: Constraint satisfaction over solution speed")
            solver_start_time = time.time()

            # Use CBC solver with detailed logging to a temporary file
            import tempfile
            import os
            import sys
            from contextlib import redirect_stdout, redirect_stderr
            import io
            
            # Create temporary file for CBC log output
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.cbc_log', delete=False) as temp_log:
                cbc_log_file = temp_log.name
            
            # Also capture stdout/stderr as backup
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            
            try:
                logging.info(f"  - CBC detailed logs will be captured to: {cbc_log_file}")
                
                # Try multiple approaches to capture CBC output
                cbc_output_captured = False
                
                # Approach 1: Use PuLP's built-in logPath parameter (most reliable)
                try:
                    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                        solver = pulp.PULP_CBC_CMD(
                            msg=True,
                            timeLimit=20,                   # Time limit to prevent excessive computation
                            logPath=cbc_log_file,          # Use PuLP's built-in log file parameter
                            keepFiles=True                  # Keep temporary files to help with debugging
                            # NO gapRel parameter - enforce strict constraint satisfaction
                        )
                        prob.solve(solver)
                        cbc_output_captured = True
                        logging.info("CBC solver completed with logPath approach")
                        
                except Exception as e:
                    logging.warning(f"CBC logPath approach failed: {e}")
                
                # Approach 2: Fallback - try with explicit options for log verbosity
                if not cbc_output_captured:
                    try:
                        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                            solver = pulp.PULP_CBC_CMD(
                                msg=True, 
                                timeLimit=20,                   # Consistent with primary approach
                                # NO gapRel parameter - enforce strict constraint satisfaction
                                options=['log', '2', 'printingOptions', 'all']  # Maximum verbosity
                            )
                            prob.solve(solver)
                            cbc_output_captured = True
                            logging.info("CBC solver completed with verbose options approach")
                    except Exception as e:
                        logging.error(f"CBC verbose options approach failed: {e}")
                
                # Approach 3: Final fallback - basic solve with msg=True
                if not cbc_output_captured:
                    logging.warning("Using basic CBC solver without output capture")
                    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                        prob.solve(pulp.PULP_CBC_CMD(msg=True, timeLimit=20))
                        # NO gapRel parameter - enforce strict constraint satisfaction
                
                # Process captured output from log file
                if os.path.exists(cbc_log_file) and os.path.getsize(cbc_log_file) > 0:
                    try:
                        with open(cbc_log_file, 'r') as f:
                            cbc_log_content = f.read().strip()
                            
                        if cbc_log_content:
                            logging.info("=== DETAILED CBC SOLVER LOG FILE OUTPUT ===")
                            for line in cbc_log_content.split('\n'):
                                line = line.strip()
                                if line:
                                    logging.info(f"CBC: {line}")
                            logging.info("=== END DETAILED CBC SOLVER LOG FILE OUTPUT ===")
                            
                    except Exception as e:
                        logging.error(f"Failed to read CBC log file: {e}")
                
                # Process captured stdout/stderr
                captured_stdout = stdout_buffer.getvalue().strip()
                captured_stderr = stderr_buffer.getvalue().strip()
                
                if captured_stdout:
                    logging.info("=== CBC SOLVER STDOUT CAPTURE ===")
                    for line in captured_stdout.split('\n'):
                        line = line.strip()
                        if line:
                            logging.info(f"CBC: {line}")
                    logging.info("=== END CBC SOLVER STDOUT CAPTURE ===")
                    
                if captured_stderr:
                    logging.info("=== CBC SOLVER STDERR CAPTURE ===")
                    for line in captured_stderr.split('\n'):
                        line = line.strip()
                        if line:
                            logging.warning(f"CBC: {line}")
                    logging.info("=== END CBC SOLVER STDERR CAPTURE ===")
                
                # If no output was captured, note this
                if (not os.path.exists(cbc_log_file) or os.path.getsize(cbc_log_file) == 0) and \
                   not captured_stdout and not captured_stderr:
                    logging.warning("No CBC solver detailed output was captured - this may indicate CBC is running silently or output is going elsewhere")
                    
            except Exception as e:
                logging.error(f"Error in CBC solver output capture: {e}")
                # Final fallback
                prob.solve(pulp.PULP_CBC_CMD(msg=True, timeLimit=60))
                
            finally:
                # Clean up temporary log file
                try:
                    if os.path.exists(cbc_log_file):
                        os.unlink(cbc_log_file)
                except Exception as e:
                    logging.debug(f"Could not remove temporary CBC log file: {e}")

            solution_time = time.time() - solver_start_time
            self.status = pulp.LpStatus[prob.status]
            self.iterations = prob.solverModel.Iterations if hasattr(prob.solverModel, 'Iterations') else 0

            logging.info(f"  - Global optimization completed in {solution_time:.3f}s")
            logging.info(f"  - Solver status: {self.status}")
            logging.info(f"  - Solver iterations: {self.iterations}")

            # Extract solution if optimal
            logging.info(f"🔍 STEP 5: Processing solution and recording results")
            if prob.status == pulp.LpStatusOptimal:
                solution = {}
                total_emissions_objective = 0.0

                # Create a dictionary for quick pod lookup
                pending_pods_dict = {p.id: p for p in self.pending_pods}
                placements_saved_to_csv = 0

                logging.info(f"  - Found optimal solution! Extracting placements...")
                
                # CONSTRAINT VALIDATION: Verify the solution respects all constraints
                logging.info(f"🔍 STEP 5a: Validating solution constraints")
                is_valid, constraint_violations = self._validate_solution_constraints(
                    {pod_id: (flv_id, ts_id, emissions) for pod_id, pod_placements_list in placements.items()
                     for flv_id, ts_id, emissions in pod_placements_list
                     if (pod_id, flv_id, ts_id) in x and x[(pod_id, flv_id, ts_id)].value() is not None and x[(pod_id, flv_id, ts_id)].value() > 0.5},
                    flavours, timeslots, leftover_cpu, leftover_ram
                )
                
                if not is_valid:
                    logging.error(f"❌ CRITICAL: Solution violates {len(constraint_violations)} constraints!")
                    for violation in constraint_violations[:10]:  # Show first 10 violations
                        logging.error(f"  - VIOLATION: {violation}")
                    if len(constraint_violations) > 10:
                        logging.error(f"  - ... and {len(constraint_violations) - 10} more violations")
                    
                    # Log detailed constraint information for debugging
                    self._log_constraint_debugging_info(constraint_violations, {})
                    
                    logging.error("❌ REJECTING INVALID SOLUTION - treating as infeasible")
                    self.status = "Infeasible_Due_To_Constraint_Violations"
                    self.has_solved = False
                    return
                else:
                    logging.info("✅ Solution constraint validation PASSED - all constraints satisfied")
                
                for pod_id_sol, current_pod_placements in placements.items():
                    for flv_id_sol, ts_id_sol, emissions_sol in current_pod_placements:
                        placement_key = (pod_id_sol, flv_id_sol, ts_id_sol)

                        if placement_key not in x:
                            logging.debug(f"  - Placement key {placement_key} not found in decision variables. Skipping.")
                            continue

                        if x[placement_key].value() is not None and x[placement_key].value() > 0.5:  # Selected placement
                            current_pod = pending_pods_dict.get(pod_id_sol)
                            if not current_pod:
                                logging.error(f"  - ERROR: Pod {pod_id_sol} not found in pending_pods_dict. Skipping CSV write for this placement.")
                                continue

                            logging.info(f"  - Selected placement: Pod {pod_id_sol} -> Node {flv_id_sol}, Timeslot {ts_id_sol}, Emissions {emissions_sol:.2f}kg CO2e")
                            
                            # Log this placement to CSV
                            try:
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
                                placements_saved_to_csv += 1
                                logging.debug(f"  - Wrote placement for pod {current_pod.id} to CSV")
                            except Exception as e:
                                logging.error(f"  - ERROR: Failed to write placement for pod {current_pod.id} to CSV: {e}")
                                logging.error(traceback.format_exc())

                            solution[pod_id_sol] = (flv_id_sol, ts_id_sol, emissions_sol)
                            total_emissions_objective += emissions_sol

                logging.info(f"✅ Found optimal global solution with total objective emissions: {total_emissions_objective:.2f}")
                logging.info(f"✅ Placed {len(solution)}/{len(self.pending_pods)} pods")
                logging.info(f"📊 Wrote {placements_saved_to_csv} placements to CSV")

                # Update the solution and remove placed pods
                self.global_solution.update(solution)
                self.pending_pods = [p for p in self.pending_pods if p.id not in solution]
                self.has_solved = True
                
                # Verify CSV file status after writing
                if hasattr(self, '_session_csv_path') and self._session_csv_path and os.path.exists(self._session_csv_path):
                    file_size = os.path.getsize(self._session_csv_path)
                    logging.info(f"  - CSV file size after writing: {file_size} bytes")
                    if file_size == 0:
                        logging.error("  - ERROR: CSV file is empty after writing!")
                else:
                    logging.warning("  - WARNING: CSV file not found or not configured correctly")
                
            else:
                logging.warning(f"❌ No optimal global solution found, status: {self.status}")
                logging.warning(f"  - This may happen due to infeasibility, time limit, or other solver issues")
                logging.warning(f"  - Check resource constraints and problem formulation")
                logging.warning(f"  - Check CBC solver output for more details")

        except Exception as e:
            logging.error(f"❌ ERROR in global optimization: {e}")
            logging.error(traceback.format_exc())

        self.last_solve_time = time.time()
        logging.info(f"====== GLOBAL OPTIMIZATION SOLVER END ======")
        
        # Add CRITICAL log at the end for better visibility
        if self.has_solved:
            logging.critical(f"💡 GLOBAL OPTIMIZATION SUCCESS! Placed {len(self.global_solution)} pods.")
        else:
            logging.critical(f"💡 GLOBAL OPTIMIZATION FAILED! No solution found.")
            
        # Check CSV file state
        if hasattr(self, '_session_csv_path') and self._session_csv_path:
            if os.path.exists(self._session_csv_path):
                file_size = os.path.getsize(self._session_csv_path)
                if file_size > 0:
                    logging.critical(f"📊 RESULTS SAVED to {self._session_csv_path} (size: {file_size} bytes)")
                else:
                    logging.critical(f"⚠️ CSV FILE EMPTY: {self._session_csv_path}")
            else:
                logging.critical(f"⚠️ CSV FILE NOT FOUND: {self._session_csv_path}")
        else:
            logging.critical(f"⚠️ CSV FILE PATH NOT SET")

    def _validate_solution_constraints(
        self,
        solution: Dict[str, Tuple[str, int, float]],
        flavours: List[CarbonAwareFlavour],
        timeslots: List[CarbonAwareTimeslot],
        leftover_cpu: Dict[str, Dict[int, float]],
        leftover_ram: Dict[str, Dict[int, float]]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that the solution satisfies all constraints.
        
        Args:
            solution: Dictionary of pod_id -> (node_id, timeslot_id, emissions)
            flavours: List of available flavours
            timeslots: List of available timeslots
            leftover_cpu: Available CPU resources
            leftover_ram: Available RAM resources
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        # Track resource usage by node and timeslot
        cpu_usage = {}  # (node_id, timeslot_id) -> total_cpu_used
        ram_usage = {}  # (node_id, timeslot_id) -> total_ram_used
        
        # Create pod lookup
        pending_pods_dict = {p.id: p for p in self.pending_pods}
        
        for pod_id, (node_id, timeslot_id, emissions) in solution.items():
            pod = pending_pods_dict.get(pod_id)
            if not pod:
                violations.append(f"Pod {pod_id} not found in pending pods")
                continue
                
            # Check timeslot constraints
            if hasattr(pod, 'earliest_timeslot') and pod.earliest_timeslot is not None:
                if timeslot_id < pod.earliest_timeslot:
                    violations.append(f"Pod {pod_id} scheduled at timeslot {timeslot_id} before earliest allowed {pod.earliest_timeslot}")
            
            # Check deadline constraints
            if hasattr(pod, 'deadline_slot') and pod.deadline_slot is not None:
                if timeslot_id + pod.duration > pod.deadline_slot:
                    violations.append(f"Pod {pod_id} scheduled to finish at {timeslot_id + pod.duration} after deadline {pod.deadline_slot}")
            
            # Accumulate resource usage for capacity validation
            for offset in range(int(pod.duration)):
                current_slot = timeslot_id + offset
                key = (node_id, current_slot)
                
                if key not in cpu_usage:
                    cpu_usage[key] = 0
                    ram_usage[key] = 0
                    
                cpu_usage[key] += pod.cpuRequest
                ram_usage[key] += pod.ramRequest
        
        # Validate capacity constraints
        for (node_id, timeslot_id), total_cpu in cpu_usage.items():
            if node_id in leftover_cpu and timeslot_id in leftover_cpu[node_id]:
                available_cpu = leftover_cpu[node_id][timeslot_id]
                if total_cpu > available_cpu:
                    violations.append(f"CPU overallocation on node {node_id} at timeslot {timeslot_id}: {total_cpu:.2f} > {available_cpu:.2f}")
            else:
                violations.append(f"Missing CPU capacity data for node {node_id} at timeslot {timeslot_id}")
        
        for (node_id, timeslot_id), total_ram in ram_usage.items():
            if node_id in leftover_ram and timeslot_id in leftover_ram[node_id]:
                available_ram = leftover_ram[node_id][timeslot_id]
                if total_ram > available_ram:
                    violations.append(f"RAM overallocation on node {node_id} at timeslot {timeslot_id}: {total_ram:.2f} > {available_ram:.2f}")
            else:
                violations.append(f"Missing RAM capacity data for node {node_id} at timeslot {timeslot_id}")
        
        is_valid = len(violations) == 0
        return is_valid, violations

    def _log_constraint_debugging_info(
        self,
        violations: List[str],
        solution: Dict[str, Tuple[str, int, float]]
    ) -> None:
        """
        Log detailed constraint debugging information.
        
        Args:
            violations: List of constraint violations
            solution: The solution being validated
        """
        logging.error(f"🚨 CONSTRAINT VIOLATIONS DETECTED: {len(violations)} violations found")
        
        # Group violations by type
        capacity_violations = [v for v in violations if "overallocation" in v]
        timeslot_violations = [v for v in violations if "timeslot" in v or "deadline" in v]
        other_violations = [v for v in violations if v not in capacity_violations and v not in timeslot_violations]
        
        if capacity_violations:
            logging.error(f"📊 CAPACITY VIOLATIONS ({len(capacity_violations)}):")
            for violation in capacity_violations[:10]:  # Limit output
                logging.error(f"  - {violation}")
            if len(capacity_violations) > 10:
                logging.error(f"  - ... and {len(capacity_violations) - 10} more capacity violations")
        
        if timeslot_violations:
            logging.error(f"⏰ TIMESLOT VIOLATIONS ({len(timeslot_violations)}):")
            for violation in timeslot_violations:
                logging.error(f"  - {violation}")
        
        if other_violations:
            logging.error(f"❓ OTHER VIOLATIONS ({len(other_violations)}):")
            for violation in other_violations:
                logging.error(f"  - {violation}")
        
        # Log solution summary
        if solution:
            node_counts = {}
            for pod_id, (node_id, timeslot_id, emissions) in solution.items():
                node_counts[node_id] = node_counts.get(node_id, 0) + 1
            
            logging.error("📋 SOLUTION SUMMARY:")
            logging.error(f"  - Total pods placed: {len(solution)}")
            logging.error(f"  - Nodes used: {len(node_counts)}")
            for node_id, count in sorted(node_counts.items()):
                logging.error(f"    * {node_id}: {count} pods")

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
        Return placement for a pod, using the pre-computed comprehensive solution if available.
        Otherwise, fall back to incremental global optimization.
        """
        start_time = time.time()
        # Set earliest_timeslot based on pod ID before scheduling
        self._set_pod_earliest_timeslot(pod)
        logging.debug(f"Finding placement for pod {pod.id}")
        
        # Check if we have a comprehensive solution
        if self.optimization_done:
            if pod.id in self.global_solution:
                flv_id, ts_id, emissions = self.global_solution[pod.id]
                
                # Find the corresponding objects
                flv = next((f for f in flavours if f.id == flv_id), None)
                ts = next((t for t in timeslots if t.id == ts_id), None)
                
                if flv and ts:
                    logging.info(f"📌 Using comprehensive optimization placement for pod {pod.id}: {flv_id}, timeslot {ts_id}")
                    if self.experiment_logger:
                        execution_time = time.time() - start_time
                        self.experiment_logger.record_placement(
                            pod_id=pod.id, success=True, execution_time=execution_time,
                            emissions=emissions, considered_options=len(self.global_solution),
                            selected_node=flv_id, selected_timeslot=ts_id,
                            solver_iterations=self.iterations, solver_status="Comprehensive"
                        )
                    return flv, ts, emissions
                else:
                    logging.warning(f"⚠️ Found solution for pod {pod.id} but node or timeslot not available in current context")
            else:
                logging.warning(f"⚠️ No placement for pod {pod.id} in the comprehensive solution")
            
            # Pod not in comprehensive solution - this is unusual if we did comprehensive optimization
            logging.warning(f"🔍 Pod {pod.id} not found in comprehensive solution")

        # Fallback to incremental optimization or greedy approach
        logging.debug(f"🔄 Falling back to incremental placement for pod {pod.id}")
        
        # Add pod to pending list if not already there
        if pod not in self.pending_pods:
            self.pending_pods.append(pod)
        
        # Trigger incremental global optimization if enough pods or time has passed
        should_solve = (
            len(self.pending_pods) >= self.solve_interval or
            (time.time() - self.last_solve_time) > self.solve_interval
        )
        
        if should_solve:
            logging.info(f"🚀 Triggering incremental global optimization for {len(self.pending_pods)} pods")
            self.solve_global_optimization(flavours, timeslots, leftover_cpu, leftover_ram, max_time_slots)
            
            # Check if the pod is now in the solution
            if pod.id in self.global_solution:
                flv_id, ts_id, emissions = self.global_solution[pod.id]
                flv = next((f for f in flavours if f.id == flv_id), None)
                ts = next((t for t in timeslots if t.id == ts_id), None)
                
                if flv and ts:
                    logging.info(f"📌 Using incremental optimization placement for pod {pod.id}: {flv_id}, timeslot {ts_id}")
                    if self.experiment_logger:
                        execution_time = time.time() - start_time
                        self.experiment_logger.record_placement(
                            pod_id=pod.id, success=True, execution_time=execution_time,
                            emissions=emissions, considered_options=len(self.global_solution),
                            selected_node=flv_id, selected_timeslot=ts_id,
                            solver_iterations=self.iterations, solver_status="Incremental"
                        )
                    return flv, ts, emissions
        
        # Final fallback: greedy placement
        logging.warning(f"🔄 Using greedy fallback for pod {pod.id}")
        return self._greedy_placement(pod, flavours, timeslots, leftover_cpu, leftover_ram)

    def _greedy_placement(
        self,
        pod: CarbonAwarePod,
        flavours: List[CarbonAwareFlavour],
        timeslots: List[CarbonAwareTimeslot],
        leftover_cpu: Dict[str, Dict[int, float]],
        leftover_ram: Dict[str, Dict[int, float]]
    ) -> Tuple[Optional[CarbonAwareFlavour], Optional[CarbonAwareTimeslot], float]:
        """
        Greedy placement fallback when global optimization fails.
        """
        best_placement = None
        best_emissions = float('inf')
        
        for ts in timeslots:
            # Check earliest timeslot constraint
            if hasattr(pod, 'earliest_timeslot') and pod.earliest_timeslot is not None:
                if ts.id < pod.earliest_timeslot:
                    continue
            
            # Check deadline constraint
            if hasattr(pod, 'deadline_slot') and pod.deadline_slot is not None:
                if ts.id + pod.duration > pod.deadline_slot:
                    continue
            
            for flv in flavours:
                # Check resource availability for the entire duration
                can_place = True
                for offset in range(int(pod.duration)):
                    slot = ts.id + offset
                    if (flv.id not in leftover_cpu or slot not in leftover_cpu[flv.id] or
                        leftover_cpu[flv.id][slot] < pod.cpuRequest):
                        can_place = False
                        break
                    if (flv.id not in leftover_ram or slot not in leftover_ram[flv.id] or
                        leftover_ram[flv.id][slot] < pod.ramRequest):
                        can_place = False
                        break
                
                if can_place:
                    try:
                        emissions = compute_emissions(flv, ts.id, pod)
                        if emissions < best_emissions:
                            best_emissions = emissions
                            best_placement = (flv, ts, emissions)
                    except Exception as e:
                        logging.warning(f"Error computing emissions for greedy placement: {e}")
        
        if best_placement:
            flv, ts, emissions = best_placement
            logging.info(f"📌 Greedy placement for pod {pod.id}: {flv.id}, timeslot {ts.id}")
            return flv, ts, emissions
        else:
            logging.warning(f"❌ No valid placement found for pod {pod.id}")
            return None, None, 0.0

    def _load_nodes_from_yaml(self, nodes_file: str) -> List[CarbonAwareFlavour]:
        """
        Load nodes directly from nodes.yaml file.
        
        Args:
            nodes_file: Path to nodes.yaml file
        
        Returns:
            List of CarbonAwareFlavour objects
        """
        import yaml
        import re
        import os
        import traceback
        
        logging.info(f"🔄 Loading nodes infrastructure from {nodes_file}")
        
        # Check if file exists
        if not os.path.exists(nodes_file):
            logging.error(f"❌ Nodes file not found: {nodes_file}")
            return []
            
        # Check if file is empty
        if os.path.getsize(nodes_file) == 0:
            logging.error(f"❌ Nodes file is empty: {nodes_file}")
            return []
            
        flavours = []
        
        try:
            logging.info(f"📂 Reading YAML content from {nodes_file}")
            with open(nodes_file, 'r') as f:
                nodes_data = yaml.safe_load_all(f)
                for node in nodes_data:
                    if not node:
                        continue
                        
                    # Extract node ID
                    node_id = node.get("metadata", {}).get("name", "")
                    if not node_id:
                        continue
                        
                    # Extract annotations
                    annotations = node.get("metadata", {}).get("annotations", {})
                    
                    # Extract status
                    status = node.get("status", {})
                    allocatable = status.get("allocatable", {})
                    
                    # Extract CPU, RAM and set defaults
                    try:
                        total_cpu = float(allocatable.get("cpu", "0"))
                    except ValueError:
                        # Handle unit suffixes like '2' or '200m'
                        cpu_str = allocatable.get("cpu", "0")
                        if cpu_str.endswith('m'):
                            total_cpu = float(cpu_str[:-1]) / 1000
                        else:
                            total_cpu = float(cpu_str)
                    
                    # Parse RAM (convert from Ki, Mi, Gi to bytes)
                    ram_str = allocatable.get("memory", "0")
                    ram_match = re.match(r'(\d+)([KMG]i?)?', ram_str)
                    
                    if ram_match:
                        ram_value = float(ram_match.group(1))
                        ram_unit = ram_match.group(2) if ram_match.group(2) else ""
                        
                        if ram_unit.startswith('K'):
                            total_ram = ram_value * 1024
                        elif ram_unit.startswith('M'):
                            total_ram = ram_value * 1024 * 1024
                        elif ram_unit.startswith('G'):
                            total_ram = ram_value * 1024 * 1024 * 1024
                        else:
                            total_ram = ram_value
                    else:
                        total_ram = 0
                    
                    # Parse embodied carbon
                    try:
                        embodied_carbon = float(annotations.get("hardware.carbon/embodied_emissions", "0"))
                    except ValueError:
                        embodied_carbon = 0
                    
                    # Parse lifetime
                    try:
                        lifetime_years = float(annotations.get("hardware.carbon/lifetime_years", "3"))
                    except ValueError:
                        lifetime_years = 3
                    
                    # Convert lifetime from years to hours
                    lifetime_hours = lifetime_years * 365 * 24
                    
                    # Parse power settings
                    power_settings = {
                        "idle": float(annotations.get("hardware.power/idle_watts", "100")),
                        "active": float(annotations.get("hardware.power/active_watts", "200")),
                        "max": float(annotations.get("hardware.power/max_watts", "400"))
                    }
                    
                    # Extract region for later forecasting
                    region_label = node.get("metadata", {}).get("labels", {}).get("topology.kubernetes.io/region", "")
                    
                    # Create the flavor (node representation)
                    flavour = CarbonAwareFlavour(
                        id=node_id,
                        embodiedCarbon=embodied_carbon,
                        lifetime=lifetime_hours,  # In hours
                        totalCpu=total_cpu,
                        totalRam=total_ram,
                        totalStorage=1000 * 1024 * 1024 * 1024,  # Default 1TB
                        forecast={},  # Empty forecast, will be filled later
                        power=power_settings
                    )
                    
                    # Add region as an additional attribute
                    flavour.region = region_label.upper() if region_label else ""
                    
                    flavours.append(flavour)
                    
                    logging.debug(f"Loaded node {node_id} with {total_cpu} CPU, {total_ram} RAM, " +
                                f"region={flavour.region}, embodied={embodied_carbon}, " +
                                f"lifetime={lifetime_hours}h, power={power_settings}")
            
            logging.info(f"Successfully loaded {len(flavours)} nodes from {nodes_file}")
            return flavours
            
        except Exception as e:
            logging.error(f"Error loading nodes from {nodes_file}: {e}")
            return []

    def _load_carbon_forecasts(self, forecasts_file: str) -> Dict[str, Dict[int, float]]:
        """
        Load carbon intensity directly from all_forecasts.json file.
        
        Args:
            forecasts_file: Path to all_forecasts.json file
            
        Returns:
            Dict mapping region codes to Dict of {timeslot_id: carbon_intensity}
        """
        import json
        from datetime import datetime
        import os
        import traceback
        
        logging.info(f"📊 Loading carbon forecasts from {forecasts_file}")
        
        # First check if file exists
        if not os.path.exists(forecasts_file):
            logging.error(f"❌ Carbon forecast file not found: {forecasts_file}")
            return {}
            
        # Check if file is empty
        if os.path.getsize(forecasts_file) == 0:
            logging.error(f"❌ Carbon forecast file is empty: {forecasts_file}")
            return {}
            
        forecasts = {}
        
        try:
            with open(forecasts_file, 'r') as f:
                logging.info(f"📂 Reading JSON content from {forecasts_file}")
                try:
                    data = json.load(f)
                    logging.info(f"✅ Successfully parsed JSON data")
                    logging.info(f"📊 Found {len(data)} regions in carbon forecast file")
                except json.JSONDecodeError as json_err:
                    logging.error(f"❌ Invalid JSON format in {forecasts_file}: {json_err}")
                    logging.error(traceback.format_exc())
                    return {}
                
            # Establish base time for calculating timeslot IDs
            base_time = None
            
            # Check if JSON is properly formatted
            if not isinstance(data, dict):
                logging.error(f"❌ Unexpected forecast data format: expected dict, got {type(data)}")
                return {}
                
            # Debug sample of data
            sample_region = next(iter(data.keys()), None) if data else None
            if sample_region:
                logging.info(f"📊 Sample region: {sample_region}")
                if isinstance(data[sample_region], dict) and "forecast" in data[sample_region]:
                    sample_forecast = data[sample_region]["forecast"]
                    logging.info(f"📊 Sample forecast entries: {len(sample_forecast)} entries")
                    if sample_forecast and len(sample_forecast) > 0:
                        sample_entry = sample_forecast[0]
                        logging.info(f"📊 Sample forecast entry: {sample_entry}")
                
            for region, region_data in data.items():
                if not isinstance(region_data, dict) or "forecast" not in region_data or not region_data["forecast"]:
                    logging.warning(f"⚠️ Region {region} has no forecast data, skipping")
                    continue
                    
                # Find base time if not set
                if base_time is None:
                    try:
                        first_datetime_str = region_data["forecast"][0]["datetime"]
                        base_time = datetime.fromisoformat(first_datetime_str.replace("Z", "+00:00"))
                        logging.info(f"⏰ Using base time {base_time} for timeslot calculations")
                    except (KeyError, IndexError) as e:
                        logging.error(f"❌ Error extracting datetime from forecast: {e}")
                        if not region_data["forecast"]:
                            logging.error("  - Forecast array is empty")
                        elif len(region_data["forecast"]) > 0:
                            logging.error(f"  - First forecast entry: {region_data['forecast'][0]}")
                        continue
                
                # Process forecast data
                forecast_dict = {}
                valid_entries = 0
                invalid_entries = 0
                
                for entry in region_data["forecast"]:
                    try:
                        if "carbonIntensity" not in entry or "datetime" not in entry:
                            logging.debug(f"⚠️ Missing required fields in forecast entry: {entry}")
                            invalid_entries += 1
                            continue
                            
                        carbon_intensity = entry["carbonIntensity"]
                        entry_time = datetime.fromisoformat(entry["datetime"].replace("Z", "+00:00"))
                        
                        # Calculate timeslot ID (hours since base time)
                        hours_diff = int((entry_time - base_time).total_seconds() / 3600)
                        timeslot_id = hours_diff
                        
                        forecast_dict[timeslot_id] = carbon_intensity
                        valid_entries += 1
                    except Exception as entry_err:
                        logging.debug(f"⚠️ Error processing forecast entry: {entry_err}")
                        invalid_entries += 1
                
                if valid_entries > 0:
                    forecasts[region] = forecast_dict
                    logging.info(f"✅ Region {region}: loaded {valid_entries} forecast entries, skipped {invalid_entries}")
                    
                    # Log some statistics about this region's forecast
                    if len(forecast_dict) > 0:
                        min_ts = min(forecast_dict.keys())
                        max_ts = max(forecast_dict.keys())
                        min_intensity = min(forecast_dict.values())
                        max_intensity = max(forecast_dict.values())
                        avg_intensity = sum(forecast_dict.values()) / len(forecast_dict)
                        
                        logging.info(f"📊 Region {region} forecast stats:")
                        logging.info(f"  - Timeslot range: {min_ts} to {max_ts} ({max_ts - min_ts + 1} hours)")
                        logging.info(f"  - Carbon intensity range: {min_intensity:.2f} to {max_intensity:.2f} gCO2/kWh")
                        logging.info(f"  - Average carbon intensity: {avg_intensity:.2f} gCO2/kWh")
                else:
                    logging.warning(f"⚠️ No valid forecast entries for region {region}")
            
            if forecasts:
                logging.info(f"✅ Successfully loaded carbon forecasts for {len(forecasts)} regions")
                all_timeslots = set()
                for region_forecast in forecasts.values():
                    all_timeslots.update(region_forecast.keys())
                    
                logging.info(f"📊 Total timeslots across all regions: {len(all_timeslots)}")
                logging.info(f"⏱️ Timeslot range: {min(all_timeslots)} to {max(all_timeslots)}")
                return forecasts
            else:
                logging.error(f"❌ No valid forecast data found in {forecasts_file}")
                return {}
            
        except Exception as e:
            logging.error(f"❌ Error loading carbon forecasts from {forecasts_file}: {e}")
            logging.error(traceback.format_exc())
            return {}
            
    def _extract_region_from_node_id(self, node_id: str) -> str:
        """
        Extract the region code from a node ID.
        
        Args:
            node_id: Node identifier (e.g., node-0-de-iot)
            
        Returns:
            Region code (e.g., DE)
        """
        import re
        
        # Try to extract region from standard naming patterns
        match = re.search(r'-([a-z]{2})(-[a-z]+)?$', node_id.lower())
        if match:
            return match.group(1).upper()
        
        # For IT-NO style regions
        match = re.search(r'-([a-z]{2}-[a-z]{2})(-[a-z]+)?$', node_id.lower())
        if match:
            return match.group(1).upper()
            
        # Default fallback
        return "DE"
        
    def _extract_pods_from_yaml(self, yaml_file: str) -> List[CarbonAwarePod]:
        """
        Extract pod definitions from a timeslot_X.yaml file.
        
        Args:
            yaml_file: Path to timeslot_X.yaml file
            
        Returns:
            List of CarbonAwarePod objects
        """
        import yaml
        import re
        from datetime import datetime
        
        logging.debug(f"Extracting pods from {yaml_file}")
        pods = []
        
        try:
            with open(yaml_file, 'r') as f:
                # Handle multi-document YAML files (separated by ---) 
                all_documents = list(yaml.safe_load_all(f))
            
            # Extract the timeslot ID from the filename
            timeslot_id = 0
            match = re.search(r'timeslot_(\d+)\.yaml$', yaml_file)
            if match:
                timeslot_id = int(match.group(1))
            
            # Extract the timestamp if available from any document
            reference_time = datetime.now()
            
            # Process each document (Kubernetes Deployment)
            for doc in all_documents:
                if not doc or not isinstance(doc, dict):
                    continue
                    
                # Skip non-Deployment documents
                if doc.get('kind') != 'Deployment':
                    continue
                    
                # Extract pod info from Kubernetes Deployment
                metadata = doc.get('metadata', {})
                deployment_name = metadata.get('name', '')
                
                if not deployment_name:
                    continue
                
                # Parse duration and deadline from deployment name
                duration = 1.0  # Default 1 hour
                deadline = 24.0  # Default 24 hours
                
                # Try to extract from name (e.g., mXXX-duration-6h-deadline-12h)
                name_duration_match = re.search(r'-duration-(\d+)h', deployment_name)
                if name_duration_match:
                    duration = float(name_duration_match.group(1))
                
                name_deadline_match = re.search(r'-deadline-(\d+)h', deployment_name)
                if name_deadline_match:
                    deadline = float(name_deadline_match.group(1))
                
                # Extract resource requirements from container spec
                spec = doc.get('spec', {})
                template = spec.get('template', {})
                pod_spec = template.get('spec', {})
                containers = pod_spec.get('containers', [])
                
                if not containers:
                    logging.warning(f"No containers found in deployment {deployment_name}")
                    continue
                
                # Get resource requests from first container
                container = containers[0]
                resources = container.get('resources', {})
                requests = resources.get('requests', {})
                
                # Parse CPU request (e.g., "1000m" -> 1.0)
                cpu_str = requests.get('cpu', '100m')
                if cpu_str.endswith('m'):
                    cpu_request = float(cpu_str[:-1]) / 1000.0
                else:
                    cpu_request = float(cpu_str)
                
                # Parse memory request (e.g., "1Gi" -> 1024 MB)
                memory_str = requests.get('memory', '128Mi')
                if memory_str.endswith('Mi'):
                    ram_request = float(memory_str[:-2])
                elif memory_str.endswith('Gi'):
                    ram_request = float(memory_str[:-2]) * 1024
                elif memory_str.endswith('Ki'):
                    ram_request = float(memory_str[:-2]) / 1024
                else:
                    ram_request = float(memory_str)  # Assume MB
                
                # Create the pod
                pod = CarbonAwarePod(
                    id=deployment_name,
                    deadline_hours=deadline,
                    duration=duration,
                    powerConsumption=0.0,  # Will be calculated during emission computation
                    cpuRequest=cpu_request,
                    ramRequest=ram_request,
                    storageRequest=100 * 1024 * 1024,  # Default 100MB
                    reference_time=reference_time
                )
                
                # Set earliest_timeslot based on the file
                pod.earliest_timeslot = timeslot_id
                
                pods.append(pod)
                logging.debug(f"Extracted pod {deployment_name} with duration={duration}h, " +
                            f"deadline={deadline}h, CPU={cpu_request}, RAM={ram_request}")
            
            # Legacy format support - if no Deployments found, try old microservices format
            if not pods:
                for doc in all_documents:
                    if doc and isinstance(doc, dict) and "microservices" in doc:
                        # Process each microservice (pod) in legacy format
                        for ms in doc.get("microservices", []):
                            if not ms or "name" not in ms:
                                continue
                            
                            # Parse duration and deadline from name or fields
                            duration = 1.0  # Default 1 hour
                            deadline = 24.0  # Default 24 hours
                            
                            # Try to extract from name first (e.g., mXXX-duration-6h-deadline-12h)
                            name_duration_match = re.search(r'-duration-(\d+)h', ms["name"])
                            if name_duration_match:
                                duration = float(name_duration_match.group(1))
                            
                            name_deadline_match = re.search(r'-deadline-(\d+)h', ms["name"])
                            if name_deadline_match:
                                deadline = float(name_deadline_match.group(1))
                            
                            # If not in name, check for explicit fields
                            if "duration" in ms:
                                duration_str = ms["duration"]
                                if duration_str.endswith('h'):
                                    duration = float(duration_str[:-1])
                                else:
                                    duration = float(duration_str)
                            
                            if "deadline" in ms:
                                deadline_str = ms["deadline"]
                                if deadline_str.endswith('h'):
                                    deadline = float(deadline_str[:-1])
                                else:
                                    deadline = float(deadline_str)
                            
                            # Extract CPU and RAM requirements
                            cpu_request = float(ms.get("cpu", "0.1"))
                            ram_request = float(ms.get("ram", "128"))  # Default 128MB
                            
                            # Create the pod
                            pod = CarbonAwarePod(
                                id=ms["name"],
                                deadline_hours=deadline,
                                duration=duration,
                                powerConsumption=0.0,  # Will be calculated during emission computation
                                cpuRequest=cpu_request,
                                ramRequest=ram_request,
                                storageRequest=100 * 1024 * 1024,  # Default 100MB
                                reference_time=reference_time
                            )
                            
                            # Set earliest_timeslot based on the file
                            pod.earliest_timeslot = timeslot_id
                            
                            pods.append(pod)
                            logging.debug(f"Extracted pod {ms['name']} with duration={duration}h, " +
                                        f"deadline={deadline}h, CPU={cpu_request}, RAM={ram_request}")
            
            logging.info(f"Extracted {len(pods)} pods from {yaml_file}")
            return pods
            
        except Exception as e:
            logging.error(f"Error extracting pods from {yaml_file}: {e}")
            return []
            
    def _initialize_resource_state(self):
        """Initialize resource availability for all nodes and timeslots."""
        # Initialize resource state
        self.resource_state = {
            "cpu": {},
            "ram": {}
        }
        
        max_timeslot = max(ts.id for ts in self.all_timeslots) if self.all_timeslots else 24
        
        for flv in self.all_flavours:
            self.resource_state["cpu"][flv.id] = {}
            self.resource_state["ram"][flv.id] = {}
            
            # For each timeslot, initialize full capacity 
            for ts_id in range(max_timeslot + 1):
                self.resource_state["cpu"][flv.id][ts_id] = flv.totalCpu
                self.resource_state["ram"][flv.id][ts_id] = flv.totalRam

    def precompute_all_workloads(self, workloads_dir: str, nodes_file: str, forecasts_file: str):
        """
        Load all pods from all timeslot files and solve them in one comprehensive global optimization.
        
        This method:
        1. Loads all nodes from nodes.yaml
        2. Loads carbon intensity from all_forecasts.json
        3. Loads all pods from all timeslot_*.yaml files
        4. Solves the global optimization problem for all pods at once
        
        Args:
            workloads_dir: Directory containing timeslot_*.yaml files
            nodes_file: Path to nodes.yaml
            forecasts_file: Path to all_forecasts.json
        
        Returns:
            True if optimization was successful, False otherwise
        """
        import os
        import traceback
        
        logging.info(f"🚀 Starting comprehensive global optimization with all workloads")
        logging.info(f"- Workloads directory: {workloads_dir}")
        logging.info(f"- Nodes file: {nodes_file}")
        logging.info(f"- Forecasts file: {forecasts_file}")
        
        try:
            # 1. Load all nodes from nodes.yaml
            logging.info(f"📂 STEP 1: Loading infrastructure from {nodes_file}")
            self.all_flavours = self._load_nodes_from_yaml(nodes_file)
            if not self.all_flavours:
                logging.error("❌ Failed to load nodes. Aborting optimization.")
                return False
            
            logging.info(f"✅ SUCCESS: Loaded {len(self.all_flavours)} nodes")
            
            # Log node details for debugging
            for flv in self.all_flavours:
                logging.info(f"  - Node {flv.id}: {flv.totalCpu} CPU, {flv.totalRam} RAM, region={flv.region if hasattr(flv, 'region') else 'unknown'}")
        except Exception as e:
            logging.error(f"❌ FAILED: Error loading nodes from {nodes_file}: {e}")
            logging.error(traceback.format_exc())
            return False
        
        try:
            # 2. Load carbon intensity from all_forecasts.json
            logging.info(f"📂 STEP 2: Loading carbon intensity forecasts from {forecasts_file}")
            carbon_forecasts = self._load_carbon_forecasts(forecasts_file)
            if not carbon_forecasts:
                logging.error("❌ Failed to load carbon forecasts. Aborting optimization.")
                return False
                
            logging.info(f"✅ SUCCESS: Loaded carbon forecasts for {len(carbon_forecasts)} regions")
            
            # Log forecast details for debugging
            for region, forecast in carbon_forecasts.items():
                logging.info(f"  - Region {region}: {len(forecast)} timeslots of forecast data")
                if len(forecast) > 0:
                    min_value = min(forecast.values())
                    max_value = max(forecast.values())
                    logging.info(f"    - Range: {min_value:.2f} - {max_value:.2f} gCO2/kWh")
                    sample_timeslots = sorted(list(forecast.keys()))[:5]  # First 5 timeslots
                    logging.info(f"    - Sample timeslots: {sample_timeslots}")
                    for ts in sample_timeslots[:3]:  # Show first 3
                        logging.info(f"      - Timeslot {ts}: {forecast[ts]:.2f} gCO2/kWh")
        except Exception as e:
            logging.error(f"❌ FAILED: Error loading carbon forecasts from {forecasts_file}: {e}")
            logging.error(traceback.format_exc())
            return False
            
        try:
            # 3. Attach carbon forecasts to flavours
            logging.info(f"🔄 STEP 3: Attaching carbon forecasts to nodes")
            for flv in self.all_flavours:
                # Extract region from node ID if not already set
                if not hasattr(flv, 'region') or not flv.region:
                    flv.region = self._extract_region_from_node_id(flv.id)
                    logging.info(f"  - Extracted region {flv.region} for node {flv.id}")
                    
                # Assign forecast data to the node
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
        except Exception as e:
            logging.error(f"❌ FAILED: Error attaching forecasts to nodes: {e}")
            logging.error(traceback.format_exc())
            return False
        
        try:
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
            
            pods_by_file = {}  # Track pods by source file
            
            for yaml_file in sorted(yaml_files):
                file_path = os.path.join(workloads_dir, yaml_file)
                try:
                    pods = self._extract_pods_from_yaml(file_path)
                    pods_by_file[yaml_file] = pods
                    all_pods.extend(pods)
                    logging.info(f"  - Loaded {len(pods)} pods from {yaml_file}")
                    if pods:
                        for pod in pods[:3]:  # Log details of first 3 pods
                            logging.info(f"    - Pod {pod.id}: CPU={pod.cpuRequest}, RAM={pod.ramRequest}, Duration={pod.duration}h, Earliest TS={pod.earliest_timeslot}")
                except Exception as e:
                    logging.error(f"❌ Error processing {yaml_file}: {e}")
            
            if not all_pods:
                logging.error("❌ FAILED: No pods found in any timeslot files")
                return False
                
            logging.info(f"✅ SUCCESS: Loaded {len(all_pods)} total pods from {len(yaml_files)} timeslot files")
        except Exception as e:
            logging.error(f"❌ FAILED: Error loading pods from timeslot files: {e}")
            logging.error(traceback.format_exc())
            return False
            
        try:
            # 5. Create timeslots
            logging.info(f"🔄 STEP 5: Creating timeslots for scheduling horizon")
            max_timeslot_id = max(pod.earliest_timeslot for pod in all_pods) + 24  # Add 24h window for scheduling
            self.all_timeslots = []
            
            for i in range(max_timeslot_id + 1):
                self.all_timeslots.append(CarbonAwareTimeslot(id=i, start_time=time.time() + i*3600, length=1))
            
            logging.info(f"✅ SUCCESS: Created {len(self.all_timeslots)} timeslots for scheduling")
        except Exception as e:
            logging.error(f"❌ FAILED: Error creating timeslots: {e}")
            logging.error(traceback.format_exc())
            return False
            
        try:
            # 6. Initialize resource state
            logging.info(f"🔄 STEP 6: Initializing resource state for all nodes")
            self._initialize_resource_state()
            logging.info(f"✅ SUCCESS: Initialized resource state for {len(self.all_flavours)} nodes")
        except Exception as e:
            logging.error(f"❌ FAILED: Error initializing resource state: {e}")
            logging.error(traceback.format_exc())
            return False
            
        try:
            # 7. Run the global optimization with all data
            logging.info(f"🧮 STEP 7: Starting global optimization with {len(all_pods)} pods across {len(self.all_timeslots)} timeslots")
            self.pending_pods = all_pods
            
            # Ensure CSV log is set up before optimization
            if hasattr(self, 'setup_session_placement_log'):
                logging.info("📊 Setting up session placement log for recording results")
                self.setup_session_placement_log()
                
            # Solve the optimization problem
            logging.info("🔍 Starting MILP solver for global optimization problem")
            self.solve_global_optimization(
                self.all_flavours,
                self.all_timeslots,
                self.resource_state["cpu"],
                self.resource_state["ram"],
                len(self.all_timeslots)
            )
            
            # Set flags
            self.optimization_done = self.has_solved
            
            # Log results
            if self.optimization_done:
                logging.info(f"🎉 Comprehensive global optimization complete!")
                logging.info(f"✅ Successfully placed {len(self.global_solution)}/{len(all_pods)} pods")
                
                # Verify the CSV file was created and data was written
                csv_path = self._session_csv_path if hasattr(self, '_session_csv_path') else None
                if csv_path and os.path.exists(csv_path):
                    file_size = os.path.getsize(csv_path)
                    row_count = 0
                    
                    # Count rows in CSV
                    import csv
                    try:
                        with open(csv_path, 'r') as f:
                            row_count = sum(1 for _ in csv.reader(f)) - 1  # Subtract header row
                    except Exception as csv_err:
                        logging.error(f"Error counting CSV rows: {csv_err}")
                        
                    logging.info(f"📊 CSV output file: {csv_path}")
                    logging.info(f"   - File size: {file_size} bytes")
                    logging.info(f"   - Row count: {row_count} (excluding header)")
                    
                    if row_count <= 0:
                        logging.error("⚠️ CSV file appears empty (only contains header row)")
                    elif row_count != len(self.global_solution):
                        logging.error(f"⚠️ CSV row count ({row_count}) doesn't match solution count ({len(self.global_solution)})")
                else:
                    logging.error(f"⚠️ CSV file not found or not properly configured: {csv_path}")
                
                # Log pod placement statistics by node
                node_placements = {}
                for pod_id, (node_id, ts_id, emissions) in self.global_solution.items():
                    if node_id not in node_placements:
                        node_placements[node_id] = 0
                    node_placements[node_id] += 1
                    
                for node_id, count in node_placements.items():
                    logging.info(f"  - {node_id}: {count} pods placed")
                
                # Log any unplaced pods
                if len(self.global_solution) < len(all_pods):
                    unplaced_pods = [p.id for p in all_pods if p.id not in self.global_solution]
                    logging.warning(f"⚠️ {len(unplaced_pods)} pods could not be placed")
                    if len(unplaced_pods) <= 10:  # Only list if not too many
                        for pod_id in unplaced_pods:
                            logging.warning(f"  - {pod_id}")
                            
                # Show sample of results
                if self.global_solution:
                    sample = list(self.global_solution.items())[:5]  # Show first 5
                    logging.info("📊 Sample placements:")
                    for pod_id, (node_id, ts_id, emissions) in sample:
                        logging.info(f"  - Pod {pod_id} -> Node {node_id}, Timeslot {ts_id}, Emissions {emissions:.2f}kg CO2e")
                        
                return True
            else:
                logging.error("❌ Comprehensive global optimization failed!")
                logging.error(f"  - Solver status: {self.status}")
                logging.error(f"  - Solver iterations: {self.iterations}")
                return False
        except Exception as e:
            logging.error(f"❌ FAILED: Error during global optimization: {e}")
            logging.error(traceback.format_exc())
            return False