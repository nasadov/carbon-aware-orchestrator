"""
Carbon-aware scheduling optimal algorithm implementation using MILP.
"""
import logging
import time  # Proper time import
from typing import Dict, List, Optional, Tuple
import pulp

from carbon_aware.algorithms.base import SchedulingAlgorithm
from carbon_aware.models import CarbonAwarePod, CarbonAwareFlavour, CarbonAwareTimeslot
from carbon_aware.utils import is_timeslot_valid, compute_emissions


class OptimalAlgorithm(SchedulingAlgorithm):
    """
    Optimal implementation of the carbon-aware scheduling algorithm using MILP.
    The algorithm formulates the scheduling problem as a Mixed Integer Linear 
    Programming problem and solves it to find the globally optimal solution.
    """
    
    def __init__(self):
        # Add experiment logger and tracking properties
        self.experiment_logger = None
        self.iterations = 0
        self.status = None
    
    @property
    def name(self) -> str:
        return "Carbon-Aware-Optimal"
    
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
        Find the optimal placement for a pod using MILP.
        
        Args:
            pod: The pod to place
            flavours: Available node types
            timeslots: Available scheduling timeslots
            leftover_cpu: Remaining CPU capacity per node and timeslot
            leftover_ram: Remaining RAM capacity per node and timeslot
            max_time_slots: Maximum number of timeslots to consider
            
        Returns:
            Tuple of (best_node, best_timeslot, emissions) or (None, None, inf) if no placement found
        """
        # Start timing the placement decision
        overall_start_time = time.time()
        considered_options = len(flavours) * len(timeslots)
        
        try:
            # Install PuLP if not already installed
            try:
                import pulp
            except ImportError:
                logging.warning("PuLP not found. Attempting to install...")
                import subprocess
                subprocess.check_call(['pip', 'install', 'pulp'])
                import pulp
                
            # Create optimization problem
            prob = pulp.LpProblem(f"Carbon_Aware_Scheduling_{pod.id}", pulp.LpMinimize)
            
            # Filter valid timeslots for this pod
            valid_timeslots = [ts for ts in timeslots if is_timeslot_valid(ts, pod)]
            
            if not valid_timeslots:
                logging.warning(f"No valid timeslots for pod {pod.id}")
                
                # Record failure if experiment logger is available
                if self.experiment_logger:
                    execution_time = time.time() - overall_start_time
                    self.experiment_logger.record_placement(
                        pod_id=pod.id,
                        success=False,
                        execution_time=execution_time,
                        emissions=0.0,
                        considered_options=considered_options,
                        selected_node=None,
                        selected_timeslot=None,
                        solver_iterations=0,
                        solver_status="No Valid Timeslots"
                    )
                
                return None, None, float('inf')
                
            # Pre-compute emissions for all valid combinations
            emissions_cache = {}
            valid_combinations = []
            
            for ts in valid_timeslots:
                for flv in flavours:
                    # Check if placement is feasible for the entire duration
                    duration_feasible = True
                    
                    for slot_offset in range(int(pod.duration)):
                        current_slot = ts.id + slot_offset
                        if current_slot >= max_time_slots:
                            duration_feasible = False
                            break
                            
                        if (leftover_cpu[flv.id][current_slot] < pod.cpuRequest or 
                            leftover_ram[flv.id][current_slot] < pod.ramRequest):
                            duration_feasible = False
                            break
                    
                    if duration_feasible:
                        # Calculate emissions for this placement
                        emissions = compute_emissions(flv, ts.id, pod)
                        emissions_cache[(flv.id, ts.id)] = emissions
                        valid_combinations.append((flv, ts))
            
            if not valid_combinations:
                logging.warning(f"No valid placement combinations found for pod {pod.id}")
                
                # Record failure if experiment logger is available
                if self.experiment_logger:
                    execution_time = time.time() - overall_start_time
                    self.experiment_logger.record_placement(
                        pod_id=pod.id,
                        success=False,
                        execution_time=execution_time,
                        emissions=0.0,
                        considered_options=considered_options,
                        selected_node=None,
                        selected_timeslot=None,
                        solver_iterations=0,
                        solver_status="No Valid Combinations"
                    )
                
                return None, None, float('inf')
                
            # Update considered options with actual valid combinations
            considered_options = len(valid_combinations)
                
            # Create decision variables: x[node_id,timeslot_id] = 1 if pod is placed here, 0 otherwise
            x = pulp.LpVariable.dicts("placement", 
                                     [(flv.id, ts.id) for flv, ts in valid_combinations],
                                     cat=pulp.LpBinary)
            
            # Objective function: minimize total carbon emissions
            prob += pulp.lpSum([emissions_cache[(flv.id, ts.id)] * x[(flv.id, ts.id)] 
                               for flv, ts in valid_combinations])
            
            # Constraint: only one placement can be chosen
            prob += pulp.lpSum([x[(flv.id, ts.id)] for flv, ts in valid_combinations]) == 1
            
            # Solve the problem
            logging.info(f"Solving optimal placement for pod {pod.id} with {len(valid_combinations)} possible placements")
            solver_start_time = time.time()
            
            # Use CBC solver (comes with PuLP)
            prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=10))  # 10-second time limit
            
            solution_time = time.time() - solver_start_time
            self.status = pulp.LpStatus[prob.status]
            self.iterations = prob.solverModel.Iterations if hasattr(prob.solverModel, 'Iterations') else 0
            logging.info(f"Optimization for pod {pod.id} completed in {solution_time:.3f}s with status: {self.status}")
            
            # Check if a solution was found
            if prob.status == pulp.LpStatusOptimal:
                # Find which placement was chosen
                for flv, ts in valid_combinations:
                    if pulp.value(x[(flv.id, ts.id)]) > 0.5:  # Should be 1 if selected
                        emissions = emissions_cache[(flv.id, ts.id)]
                        logging.info(f"Optimal placement found for {pod.id}: node={flv.id}, timeslot={ts.id}, emissions={emissions:.2f}")
                        
                        # Record success if experiment logger is available
                        if self.experiment_logger:
                            execution_time = time.time() - overall_start_time
                            self.experiment_logger.record_placement(
                                pod_id=pod.id,
                                success=True,
                                execution_time=execution_time,
                                emissions=emissions,
                                considered_options=considered_options,
                                selected_node=flv.id,
                                selected_timeslot=ts.id,
                                solver_iterations=self.iterations,
                                solver_status=self.status
                            )
                        
                        return flv, ts, emissions
            
            # If no solution found or status is not optimal
            logging.warning(f"No optimal solution found for pod {pod.id}, status: {self.status}")
            
            # Record failure if experiment logger is available
            if self.experiment_logger:
                execution_time = time.time() - overall_start_time
                self.experiment_logger.record_placement(
                    pod_id=pod.id,
                    success=False,
                    execution_time=execution_time,
                    emissions=0.0,
                    considered_options=considered_options,
                    selected_node=None,
                    selected_timeslot=None,
                    solver_iterations=self.iterations,
                    solver_status=self.status
                )
                
            return None, None, float('inf')
            
        except Exception as e:
            logging.exception(f"Error in optimal placement algorithm for pod {pod.id}: {e}")
            
            # Record error if experiment logger is available
            if self.experiment_logger:
                execution_time = time.time() - overall_start_time
                self.experiment_logger.record_placement(
                    pod_id=pod.id,
                    success=False,
                    execution_time=execution_time,
                    emissions=0.0,
                    considered_options=considered_options,
                    selected_node=None,
                    selected_timeslot=None,
                    solver_iterations=0,
                    solver_status=f"Error: {str(e)[:50]}"
                )
                
            return None, None, float('inf')