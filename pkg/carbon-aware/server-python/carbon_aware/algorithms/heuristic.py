"""
Carbon-aware scheduling heuristic algorithm implementation.
"""
from dataclasses import dataclass
import logging
import time
from typing import Dict, List, Optional, Tuple
import csv
import os
from datetime import datetime

from carbon_aware.algorithms.base import SchedulingAlgorithm
from carbon_aware.footprints import FootprintVector, build_used_cpu_before_map, compute_footprint_vector
from carbon_aware.models import CarbonAwarePod, CarbonAwareTimeslot, EnvironmentalFlavor
from carbon_aware.water_signals import attach_water_metadata
from carbon_aware.utils import is_timeslot_valid


@dataclass
class CandidatePlacement:
    flavour: EnvironmentalFlavor
    timeslot: CarbonAwareTimeslot
    footprint: FootprintVector
    pack_score: float
    used_cpu_before: Dict[int, float]
    objective_score: float = float("inf")
    pareto_rank: int = 0
    budget_violation: float = 0.0
    water_pressure: float = 0.0
    budget_slack_after: float = float("inf")


class HeuristicAlgorithm(SchedulingAlgorithm):
    """
    Heuristic implementation of the carbon-aware scheduling algorithm.
    """
    _placement_csv_filename_suffix = "heuristic_placements_session.csv"

    def __init__(self, perf_logger=None):
        self.experiment_logger = None
        self.perf_logger = perf_logger
        self._session_log_dir: Optional[str] = None
        self._operational_only = False  # Flag for operational-only emissions mode
        self._embodied_allocation_mode = "proportional"  # or "uniform"
        self._objective_mode = "carbon"
        self._carbon_weight = 1.0
        self._water_metric = "scarcity"
        self._water_budget: Optional[float] = None
        self._water_budget_used = 0.0
        self._remaining_pods_estimate: Optional[int] = None
        self._budget_pressure_weight = 1.0
        
        self._placement_csv_file_handle = None
        self._placement_csv_writer = None
        self._placement_csv_path: Optional[str] = None

    def _load_nodes_from_yaml(self, nodes_file: str) -> List[EnvironmentalFlavor]:
        """
        Load nodes directly from nodes.yaml file.
        
        Args:
            nodes_file: Path to nodes.yaml file
        
        Returns:
            List of EnvironmentalFlavor objects
        """
        import yaml
        import re
        import os
        import traceback
        from carbon_aware.utils import load_carbon_intensity_data
        
        logging.info(f"🔄 [HEURISTIC] Loading nodes infrastructure from {nodes_file}")
        
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
                    
                    # Parse RAM (convert from Ki, Mi, Gi to MB for consistency with pod requests)
                    ram_str = allocatable.get("memory", "0")
                    ram_match = re.match(r'(\d+)([KMG]i?)?', ram_str)
                    if ram_match:
                        ram_value = float(ram_match.group(1))
                        ram_unit = ram_match.group(2) or ""
                        
                        if ram_unit == "Ki":
                            total_ram = ram_value / 1024  # Convert Ki to MB
                        elif ram_unit == "Mi":
                            total_ram = ram_value  # Already in MB
                        elif ram_unit in ["Gi", "G"]:
                            total_ram = ram_value * 1024  # Convert Gi to MB
                        else:
                            total_ram = ram_value / (1024 * 1024)  # Assume bytes, convert to MB
                    else:
                        total_ram = 0.0
                    
                    # Extract region from labels
                    labels = node.get("metadata", {}).get("labels", {})
                    region_label = labels.get("topology.kubernetes.io/region", "")
                    
                    # Extract hardware annotations for embodied carbon, lifetime, and power
                    embodied_carbon = 0.0
                    lifetime_hours = 8760.0  # Default 1 year
                    # Default power in watts (active retained for backward compatibility; not used in calculations)
                    power_settings = {"idle": 50.0, "active": 50.0, "max": 150.0}
                    
                    # Parse embodied emissions from annotations (convert kg to grams)
                    if "hardware.carbon/embodied_emissions" in annotations:
                        try:
                            embodied_carbon = float(annotations["hardware.carbon/embodied_emissions"]) * 1000.0  # Convert kg to grams
                        except ValueError:
                            embodied_carbon = 0.0
                    
                    # Parse lifetime in years and convert to hours.
                    if "hardware.carbon/lifetime_years" in annotations:
                        try:
                            lifetime_hours = float(annotations["hardware.carbon/lifetime_years"]) * 365 * 24
                        except ValueError:
                            lifetime_hours = 8760.0
                    elif "hardware.carbon/lifetime" in annotations:
                        try:
                            lifetime_hours = float(annotations["hardware.carbon/lifetime"])
                        except ValueError:
                            lifetime_hours = 8760.0
                    
                    # Parse power consumption settings from the current node annotations.
                    if "hardware.power/idle_watts" in annotations or "hardware.power/max_watts" in annotations:
                        try:
                            power_settings = {
                                "idle": float(annotations.get("hardware.power/idle_watts", power_settings["idle"])),
                                "active": float(annotations.get("hardware.power/active_watts", annotations.get("hardware.power/idle_watts", power_settings["active"]))),
                                "max": float(annotations.get("hardware.power/max_watts", power_settings["max"])),
                            }
                        except ValueError:
                            pass  # Keep defaults
                    elif "hardware.carbon/power_consumption" in annotations:
                        try:
                            power_str = annotations["hardware.carbon/power_consumption"]
                            power_parts = power_str.split(',')
                            for part in power_parts:
                                if ':' in part:
                                    key, value = part.split(':', 1)
                                    power_settings[key.strip()] = float(value.strip())
                        except (ValueError, AttributeError):
                            pass  # Keep defaults
                    
                    # Load carbon intensity data
                    carbon_data = load_carbon_intensity_data()
                    
                    # Get carbon intensity forecast for this region
                    if region_label.upper() in carbon_data:
                        forecast_dict = carbon_data[region_label.upper()]
                        logging.debug(f"Using carbon intensity data for node {node_id} (region {region_label})")
                    else:
                        # Fallback to default values if region not found in carbon data
                        forecast_dict = {i: 200.0 for i in range(24)}  # Default carbon intensity
                        logging.warning(f"No carbon data for region {region_label}, using default values for node {node_id}")
                    
                    # Create the flavor (node representation)
                    hardware_subcategory = labels.get("hardware.carbon/subcategory", "")

                    flavour = EnvironmentalFlavor(
                        id=node_id,
                        embodiedCarbon=embodied_carbon,
                        lifetime=lifetime_hours,  # In hours
                        totalCpu=total_cpu,
                        totalRam=total_ram,
                        totalStorage=1000 * 1024 * 1024 * 1024,  # Default 1TB
                        forecast=forecast_dict,
                        power=power_settings
                    )
                    attach_water_metadata(
                        flavour,
                        region=region_label,
                        hardware_subcategory=hardware_subcategory,
                        slot_count=max(len(forecast_dict), 24),
                    )
                    
                    flavours.append(flavour)
                    
                    logging.debug(f"Loaded node {node_id} with {total_cpu} CPU, {total_ram} RAM, " +
                                f"region={flavour.region}, embodied={embodied_carbon}, " +
                                f"lifetime={lifetime_hours}h, power={power_settings}")
            
            logging.info(f"✅ [HEURISTIC] Successfully loaded {len(flavours)} nodes from {nodes_file}")
            return flavours
            
        except Exception as e:
            logging.error(f"❌ [HEURISTIC] Error loading nodes from {nodes_file}: {e}")
            logging.error(traceback.format_exc())
            return []

    @classmethod
    def _ensure_dir_exists(cls, directory_path: str):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            logging.info(f"Created directory: {directory_path}")

    def set_base_log_dir(self, base_dir: str):
        """Sets the base directory for session logs."""
        self._session_log_dir = base_dir

    def setup_session_placement_log(self):
        """Sets up a single CSV file for logging all placements during the session."""
        if not self._session_log_dir:
            logging.error("Session log directory not set. Cannot initialize placement CSV logging for heuristic algorithm.")
            self._session_log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "analysis", "heuristic_fallback_logs"))
            HeuristicAlgorithm._ensure_dir_exists(self._session_log_dir)
            logging.warning(f"Using fallback log directory for placements: {self._session_log_dir}")

        # Use suffixes for operational-only and embodied allocation mode
        if self._operational_only:
            base_filename = "heuristic_op_placements_session.csv"
        else:
            suffix = "prop" if getattr(self, "_embodied_allocation_mode", "proportional") == "proportional" else "uniform"
            objective_suffix = ""
            if getattr(self, "_objective_mode", "carbon") == "weighted-sum":
                objective_suffix = f"_wsum{int(round(getattr(self, '_carbon_weight', 1.0) * 100)):02d}"
            elif getattr(self, "_objective_mode", "carbon") == "pareto":
                objective_suffix = "_pareto"
            elif getattr(self, "_objective_mode", "carbon") == "epsilon-pareto":
                objective_suffix = f"_epspareto{getattr(self, '_water_metric', 'scarcity')}"
                water_budget = getattr(self, "_water_budget", None)
                if water_budget is not None:
                    objective_suffix += str(water_budget).replace(".", "p")
            base_filename = f"heuristic_{suffix}{objective_suffix}_placements_session.csv"
        self._placement_csv_path = os.path.join(self._session_log_dir, base_filename)
        
        if hasattr(self, '_placement_csv_file_handle') and self._placement_csv_file_handle:
            try:
                self._placement_csv_file_handle.close()
            except Exception as e:
                logging.error(f"Error closing previous placement CSV file: {e}")
        
        try:
            file_exists_and_not_empty = os.path.exists(self._placement_csv_path) and os.path.getsize(self._placement_csv_path) > 0
            
            self._placement_csv_file_handle = open(self._placement_csv_path, 'a', newline='')
            self._placement_csv_writer = csv.writer(self._placement_csv_file_handle)
            
            if not file_exists_and_not_empty:
                self._placement_csv_writer.writerow([
                    "pod_id", "node_id", "start_slot", "duration", "cpu_request", "ram_request",
                    "embodied_mode", "objective_mode", "carbon_weight", "water_metric",
                    "region", "country", "operational_energy_kwh",
                    "operational_carbon_kg", "embodied_carbon_kg", "total_carbon_emissions",
                    "direct_water_l", "indirect_water_l", "embodied_water_l", "total_raw_water_l",
                    "scarcity_characterized_water", "criticality_adjusted_water"
                ])
                self._placement_csv_file_handle.flush()
            logging.info(f"Heuristic placements will be logged to: {self._placement_csv_path}")

        except IOError as e:
            logging.error(f"Failed to open placement CSV file: {self._placement_csv_path}. Error: {e}")
            self._placement_csv_writer = None
            self._placement_csv_file_handle = None
            self._placement_csv_path = None

    def __del__(self):
        if hasattr(self, '_placement_csv_file_handle') and self._placement_csv_file_handle:
            try:
                self._placement_csv_file_handle.close()
                logging.info(f"Closed placement CSV file: {self._placement_csv_path}")
            except Exception as e:
                logging.error(f"Error closing placement CSV file in __del__: {e}")

    def _write_placement_to_csv(
        self,
        pod_id: str,
        node_id: str,
        start_slot: int,
        duration: float,
        cpu_request: float = 0.0,
        ram_request: float = 0.0,
        flavour: Optional[EnvironmentalFlavor] = None,
        footprint: Optional[FootprintVector] = None,
    ):
        if self._placement_csv_writer and self._placement_csv_file_handle:
            try:
                mode = "operational-only" if self._operational_only else getattr(self, "_embodied_allocation_mode", "proportional")
                csv_fields = footprint.as_csv_fields() if footprint else FootprintVector().finalize().as_csv_fields()
                self._placement_csv_writer.writerow([
                    pod_id,
                    node_id,
                    start_slot,
                    duration,
                    cpu_request,
                    ram_request,
                    mode,
                    getattr(self, "_objective_mode", "carbon"),
                    getattr(self, "_carbon_weight", 1.0),
                    getattr(self, "_water_metric", "scarcity"),
                    getattr(flavour, "region", "") if flavour else "",
                    getattr(flavour, "country", "") if flavour else "",
                    csv_fields["operational_energy_kwh"],
                    csv_fields["operational_carbon_kg"],
                    csv_fields["embodied_carbon_kg"],
                    csv_fields["total_carbon_emissions"],
                    csv_fields["direct_water_l"],
                    csv_fields["indirect_water_l"],
                    csv_fields["embodied_water_l"],
                    csv_fields["total_raw_water_l"],
                    csv_fields["scarcity_characterized_water"],
                    csv_fields["criticality_adjusted_water"],
                ])
                self._placement_csv_file_handle.flush()
            except Exception as e:
                logging.error(f"Error writing to placement CSV for heuristic: {e}")
        else:
            logging.warning("Placement CSV writer not available for heuristic algorithm. Cannot log placement.")
    
    def _set_pod_earliest_timeslot(self, pod: CarbonAwarePod):
        """
        Ensure the pod has earliest_timeslot set and calculate deadline_slot.
        
        This method validates that the pod has its earliest_timeslot properly set
        (usually done during pod creation from YAML files) and calculates the
        deadline_slot relative to that earliest_timeslot.
        
        If earliest_timeslot is not set, it extracts it from the timeslot YAML file
        that contains this pod's definition.
        """
        # 🔧 FORCE re-extraction from YAML files to get correct earliest_timeslot
        # Don't trust the default value of 0 - always check the actual YAML source
        pod.earliest_timeslot = self._extract_earliest_timeslot_from_yaml_files(pod.id)
        logging.info(f"🔒 Pod {pod.id} earliest_timeslot extracted from YAML file: {pod.earliest_timeslot}")
        
        logging.info(f"🔒 Pod {pod.id} has earliest_timeslot={pod.earliest_timeslot}")
        
        # Calculate deadline_slot relative to earliest_timeslot
        pod.calculate_deadline_slot()
        if pod.deadline_slot is not None:
            logging.info(f"📅 Pod {pod.id} has deadline_slot={pod.deadline_slot} (earliest_timeslot + {pod.deadline_hours}h)")
    
    def _extract_earliest_timeslot_from_yaml_files(self, pod_id: str) -> int:
        """
        Extract the earliest timeslot by finding which timeslot_X.yaml file contains this pod.
        
        This is the CORRECT way to determine earliest timeslot - by looking at which 
        timeslot file the pod came from. If pod is in timeslot_4.yaml, then earliest_timeslot=4.
        """
        import os
        import re
        
        # Look in the workloads directory for timeslot_*.yaml files
        # Use the configured workloads directory, with fallback to default
        workloads_dir = getattr(self, '_workloads_dir', "../workloads")
        
        # If relative path, make it relative to the current working directory
        if not os.path.isabs(workloads_dir):
            workloads_dir = os.path.join(os.getcwd(), workloads_dir)
        

        
        try:
            for filename in os.listdir(workloads_dir):
                if re.match(r'timeslot_(\d+)\.yaml$', filename):
                    filepath = os.path.join(workloads_dir, filename)
                    
                    # Extract timeslot number from filename
                    match = re.match(r'timeslot_(\d+)\.yaml$', filename)
                    timeslot_num = int(match.group(1))
                    
                    # Check if this pod is defined in this file
                    try:
                        with open(filepath, 'r') as f:
                            content = f.read()
                            # Look for the pod name in deployment metadata
                            if f"name: {pod_id}" in content:
                                logging.info(f"Found pod {pod_id} in {filename} → earliest_timeslot={timeslot_num}")
                                return timeslot_num
                    except Exception as e:
                        logging.warning(f"⚠️ Error reading {filepath}: {e}")
                        continue
        except Exception as e:
            logging.warning(f"⚠️ Error scanning workloads directory {workloads_dir}: {e}")
        
        # Fallback: if not found in any timeslot file, default to 0
        logging.warning(f"⚠️ Pod {pod_id} not found in any timeslot_X.yaml file, defaulting to earliest_timeslot=0")
        return 0
    
    def set_operational_only(self, operational_only: bool):
        """Set whether to use operational-only emissions calculation."""
        self._operational_only = operational_only
        logging.info(f"HeuristicAlgorithm: operational_only mode set to {operational_only}")

    def set_embodied_allocation_mode(self, mode: str):
        """Set embodied allocation mode: 'proportional' (default) or 'uniform'."""
        if mode not in ("proportional", "uniform"):
            logging.warning(f"Unknown embodied allocation mode '{mode}', defaulting to 'proportional'")
            mode = "proportional"
        self._embodied_allocation_mode = mode
        logging.info(f"HeuristicAlgorithm: embodied_allocation_mode set to {self._embodied_allocation_mode}")

    def set_environmental_objective(self, mode: str = "carbon", carbon_weight: float = 1.0, water_metric: str = "scarcity"):
        """Set heuristic scoring objective. Water-aware modes use the selected water metric."""
        if mode not in ("carbon", "weighted-sum", "pareto", "epsilon-pareto"):
            logging.warning(f"Unknown heuristic objective '{mode}', defaulting to 'carbon'")
            mode = "carbon"

        try:
            carbon_weight = float(carbon_weight)
        except (TypeError, ValueError):
            carbon_weight = 1.0
        carbon_weight = min(max(carbon_weight, 0.0), 1.0)

        if water_metric not in ("scarcity", "raw"):
            logging.warning(f"Unknown heuristic water metric '{water_metric}', defaulting to 'scarcity'")
            water_metric = "scarcity"

        self._objective_mode = mode
        self._carbon_weight = carbon_weight
        self._water_metric = water_metric
        logging.info(
            "HeuristicAlgorithm: objective=%s carbon_weight=%.2f water_metric=%s",
            self._objective_mode,
            self._carbon_weight,
            self._water_metric,
        )

    def set_water_budget(
        self,
        water_budget: Optional[float] = None,
        water_metric: str = "scarcity",
        total_pods: Optional[int] = None,
        budget_pressure_weight: float = 1.0,
    ):
        """
        Configure the soft run-level water budget used by epsilon-pareto mode.

        The budget is not treated as a hard feasibility cut in the heuristic. It
        guides candidate ranking so tight budgets reduce water pressure while
        still allowing a placement if every candidate would overshoot.
        """
        if water_budget is None:
            self._water_budget = None
        else:
            try:
                self._water_budget = max(float(water_budget), 0.0)
            except (TypeError, ValueError):
                logging.warning("Invalid heuristic water budget %r; disabling budget guidance", water_budget)
                self._water_budget = None

        if water_metric not in ("scarcity", "raw"):
            logging.warning(f"Unknown heuristic water metric '{water_metric}', defaulting to 'scarcity'")
            water_metric = "scarcity"
        self._water_metric = water_metric

        try:
            self._remaining_pods_estimate = max(int(total_pods), 0) if total_pods is not None else None
        except (TypeError, ValueError):
            self._remaining_pods_estimate = None

        try:
            self._budget_pressure_weight = max(float(budget_pressure_weight), 0.0)
        except (TypeError, ValueError):
            self._budget_pressure_weight = 1.0

        self._water_budget_used = 0.0
        logging.info(
            "HeuristicAlgorithm: water_budget=%s water_metric=%s total_pods=%s pressure_weight=%.2f",
            self._water_budget,
            self._water_metric,
            self._remaining_pods_estimate,
            self._budget_pressure_weight,
        )

    def _remaining_water_budget(self) -> Optional[float]:
        if self._water_budget is None:
            return None
        return max(self._water_budget - self._water_budget_used, 0.0)

    def _remaining_pods_for_budget(self) -> Optional[int]:
        if self._remaining_pods_estimate is None:
            return None
        return max(self._remaining_pods_estimate, 1)

    def _record_budget_progress(self, footprint: Optional[FootprintVector] = None):
        if footprint is not None:
            self._water_budget_used += _candidate_water_value(footprint, getattr(self, "_water_metric", "scarcity"))
        if self._remaining_pods_estimate is not None and self._remaining_pods_estimate > 0:
            self._remaining_pods_estimate -= 1
        
    def set_workloads_dir(self, workloads_dir: str):
        """Set the workloads directory for YAML file lookup."""
        self._workloads_dir = workloads_dir
        logging.info(f"HeuristicAlgorithm: workloads directory set to {workloads_dir}")

    @property
    def name(self) -> str:
        return "Carbon-Aware-Heuristic"
    
    def find_placement(
        self,
        pod: CarbonAwarePod,
        flavours: List[EnvironmentalFlavor],
        timeslots: List[CarbonAwareTimeslot],
        leftover_cpu: Dict[str, Dict[int, float]],
        leftover_ram: Dict[str, Dict[int, float]],
        max_time_slots: int = 48
    ) -> Tuple[Optional[EnvironmentalFlavor], Optional[CarbonAwareTimeslot], float]:
        """Find the best placement for a pod using the carbon-aware heuristic."""
        # Set earliest_timeslot based on pod ID before scheduling
        self._set_pod_earliest_timeslot(pod)
        
        start_time = time.time()
        considered_options = len(flavours) * len(timeslots)
        
        best_node, best_slot, emissions = find_best_node_and_timeslot(
            pod,
            flavours,
            timeslots,
            leftover_cpu,
            leftover_ram,
            max_time_slots,
            self._operational_only,
            self._embodied_allocation_mode,
            getattr(self, "_objective_mode", "carbon"),
            getattr(self, "_carbon_weight", 1.0),
            getattr(self, "_water_metric", "scarcity"),
            self._remaining_water_budget(),
            self._remaining_pods_for_budget(),
            getattr(self, "_budget_pressure_weight", 1.0),
        )
        
        if self.experiment_logger:
            execution_time = time.time() - start_time
            success = best_node is not None and best_slot is not None
            
            self.experiment_logger.record_placement(
                pod_id=pod.id,
                success=success,
                execution_time=execution_time,
                emissions=emissions if success else 0.0,
                considered_options=considered_options,
                selected_node=best_node.id if best_node else None,
                selected_timeslot=best_slot.id if best_slot else None
            )

        if best_node and best_slot:
            used_cpu_before = build_used_cpu_before_map(
                flavour=best_node,
                start_slot=best_slot.id,
                duration_hours=pod.duration,
                leftover_cpu_by_slot=leftover_cpu[best_node.id],
            )
            footprint = compute_footprint_vector(
                flavour=best_node,
                start_slot=best_slot.id,
                pod=pod,
                used_cpu_before_by_slot=used_cpu_before,
                embodied_allocation_mode=getattr(self, "_embodied_allocation_mode", "proportional"),
                operational_only=self._operational_only,
                use_pod_power_only=self._operational_only,
            )
            self._write_placement_to_csv(
                pod_id=pod.id,
                node_id=best_node.id,
                start_slot=best_slot.id,
                duration=pod.duration,
                cpu_request=pod.cpuRequest,
                ram_request=pod.ramRequest,
                flavour=best_node,
                footprint=footprint,
            )
            self._record_budget_progress(footprint)
        else:
            self._record_budget_progress(None)
        
        return best_node, best_slot, emissions

    def find_placement_atomic(
        self,
        pod: CarbonAwarePod,
        flavours: List[EnvironmentalFlavor],
        timeslots: List[CarbonAwareTimeslot],
        persistent_state,
        max_time_slots: int = 48
    ) -> Tuple[Optional[EnvironmentalFlavor], Optional[CarbonAwareTimeslot], float]:
        """
        Find and atomically allocate the best placement for a pod.
        
        This method prevents race conditions by using atomic constraint checking
        and resource allocation within the persistent state.
        
        Args:
            pod: The pod to place
            flavours: Available node types
            timeslots: Available scheduling timeslots
            persistent_state: PersistentStateStorage instance for atomic operations
            max_time_slots: Maximum number of timeslots to consider
            
        Returns:
            Tuple of (best_node, best_timeslot, emissions) or (None, None, inf) if no placement found
        """
        # CRITICAL FIX: Set earliest_timeslot based on pod ID before scheduling
        self._set_pod_earliest_timeslot(pod)
        
        logging.info(f"[find_placement_atomic] Starting placement search for pod={pod.id}")
        logging.info(f"    Pod constraints: earliest_timeslot={pod.earliest_timeslot}, duration={pod.duration}h")
        logging.info(f"    Resource requirements: CPU={pod.cpuRequest:.3f}, RAM={pod.ramRequest:.0f}MB")
        logging.info(f"    Search space: {len(flavours)} nodes × {len(timeslots)} timeslots = {len(flavours) * len(timeslots)} combinations")
        
        candidates: List[CandidatePlacement] = []
        
        # First pass: find all feasible placements and calculate their emissions
        valid_timeslots = 0
        # Prefer lower-carbon hours first across nodes (best-effort)
        try:
            ordered_timeslots = sorted(
                timeslots,
                key=lambda t: min(flv.forecast.get(t.id, 200.0) for flv in flavours)
            )
        except Exception:
            ordered_timeslots = timeslots
        
        for ts in ordered_timeslots:
            if not is_timeslot_valid(ts, pod):
                logging.debug(f"[find_placement_atomic] Timeslot {ts.id} invalid for pod={pod.id} (earliest_timeslot={pod.earliest_timeslot})")
                continue
            
            valid_timeslots += 1
            logging.debug(f"[find_placement_atomic] Timeslot {ts.id} valid for pod={pod.id}")

            for flv in flavours:
                # Check if this placement would be feasible for the entire duration
                duration_feasible = True
                resource_feasible = True
                failure_reason = ""
                
                for slot_offset in range(int(pod.duration)):
                    current_slot = ts.id + slot_offset
                    if current_slot >= max_time_slots:
                        duration_feasible = False
                        failure_reason = f"slot_{current_slot}_exceeds_window_{max_time_slots}"
                        logging.debug(f"[find_placement_atomic] ❌ Pod {pod.id} on {flv.id}: {failure_reason}")
                        break
                
                if duration_feasible:
                    # Check if resources would be available (without actually allocating yet)
                    resource_check = persistent_state.check_resources_available(
                        flv.id, ts.id, int(pod.duration), pod.cpuRequest, pod.ramRequest
                    )
                    
                    if resource_check:
                        # Compute emissions using proportional embodied allocation (or operational-only)
                        # Build a view of already used CPU for each slot from persistent state (cores consumed)
                        used_cpu_before = {}
                        cpu_slack_sum = 0.0
                        ram_slack_sum = 0.0
                        for slot_offset in range(int(pod.duration)):
                            slot_id = ts.id + slot_offset
                            # total used = total capacity - leftover
                            total_capacity = flv.totalCpu
                            leftover_cpu_now = persistent_state.leftover_cpu[flv.id][slot_id]
                            leftover_ram_now = persistent_state.leftover_ram[flv.id][slot_id]
                            used_cpu_before[slot_id] = max(total_capacity - leftover_cpu_now, 0.0)
                            # Projected slack after placing this pod (clamped at 0)
                            cpu_slack_sum += max(leftover_cpu_now - pod.cpuRequest, 0.0)
                            ram_slack_sum += max(leftover_ram_now - pod.ramRequest, 0.0)

                        footprint = compute_footprint_vector(
                            flavour=flv,
                            start_slot=ts.id,
                            pod=pod,
                            used_cpu_before_by_slot=used_cpu_before,
                            embodied_allocation_mode=getattr(self, "_embodied_allocation_mode", "proportional"),
                            operational_only=self._operational_only,
                            use_pod_power_only=self._operational_only,
                        )

                        # Packing-aware tie-breaker: prefer tighter fit (smaller slack)
                        cpu_norm = flv.totalCpu * max(int(pod.duration), 1)
                        ram_norm = flv.totalRam * max(int(pod.duration), 1)
                        pack_score = (cpu_slack_sum / max(cpu_norm, 1e-6)) + (ram_slack_sum / max(ram_norm, 1e-6))

                        candidates.append(
                            CandidatePlacement(
                                flavour=flv,
                                timeslot=ts,
                                footprint=footprint,
                                pack_score=pack_score,
                                used_cpu_before=used_cpu_before,
                            )
                        )
                        logging.debug(
                            f"[find_placement_atomic] Valid candidate: pod={pod.id}, node={flv.id}, "
                            f"timeslot={ts.id}, carbon={footprint.total_carbon_g:.3f}, "
                            f"water={_candidate_water_value(footprint, getattr(self, '_water_metric', 'scarcity')):.3f}, "
                            f"pack={pack_score:.6f}"
                        )
                    else:
                        failure_reason = "insufficient_resources"
                        logging.debug(f"[find_placement_atomic] Pod {pod.id} on {flv.id} at slot {ts.id}: {failure_reason}")
        
        logging.info(f"    Valid timeslots: {valid_timeslots}/{len(timeslots)}")
        logging.info(f"    Feasible candidates found: {len(candidates)}")
        
        if not candidates:
            logging.warning(f"[find_placement_atomic] No feasible candidates found for pod={pod.id}")
            logging.warning(f"    Diagnosis for pod={pod.id}:")
            logging.warning(f"        - Earliest timeslot constraint: {pod.earliest_timeslot}")
            logging.warning(f"        - Duration: {pod.duration}h")
            logging.warning(f"        - Resources needed: CPU={pod.cpuRequest:.3f}, RAM={pod.ramRequest:.0f}MB")
            logging.warning(f"        - Available timeslots: {[ts.id for ts in timeslots]}")
            logging.warning(f"        - Valid timeslots after constraint: {valid_timeslots}")
            
            # Check each node's current resource availability
            for flv in flavours:
                max_cpu_available = max(persistent_state.leftover_cpu[flv.id].values())
                max_ram_available = max(persistent_state.leftover_ram[flv.id].values())
                logging.warning(f"        - Node {flv.id}: max_cpu={max_cpu_available:.3f}, max_ram={max_ram_available:.0f}MB")
            
            self._record_budget_progress(None)
            return None, None, float('inf')
        
        ranked_candidates = _rank_candidates(
            candidates,
            objective_mode=getattr(self, "_objective_mode", "carbon"),
            carbon_weight=getattr(self, "_carbon_weight", 1.0),
            water_metric=getattr(self, "_water_metric", "scarcity"),
            water_budget_remaining=self._remaining_water_budget(),
            remaining_pods=self._remaining_pods_for_budget(),
            budget_pressure_weight=getattr(self, "_budget_pressure_weight", 1.0),
        )
        best_candidate = ranked_candidates[0]
        logging.info(
            "    Best candidate: node=%s, timeslot=%s, objective=%.6f, carbon=%.3f, water=%.3f, pack=%.6f",
            best_candidate.flavour.id,
            best_candidate.timeslot.id,
            best_candidate.objective_score,
            best_candidate.footprint.total_carbon_g,
            _candidate_water_value(best_candidate.footprint, getattr(self, "_water_metric", "scarcity")),
            best_candidate.pack_score,
        )
        
        # Second pass: try to atomically allocate the best candidate
        allocation_attempts = 0
        for candidate in ranked_candidates:
            allocation_attempts += 1
            success = persistent_state.atomic_check_and_allocate(
                candidate.flavour.id, candidate.timeslot.id, int(pod.duration), 
                pod.cpuRequest, pod.ramRequest
            )
            
            if success:
                logging.info(f"[find_placement_atomic] Successfully allocated pod={pod.id} on attempt {allocation_attempts}")
                logging.info(
                    "    Final placement: node=%s, timeslot=%s, carbon=%.3f, objective=%.6f",
                    candidate.flavour.id,
                    candidate.timeslot.id,
                    candidate.footprint.total_carbon_g,
                    candidate.objective_score,
                )
                
                # Write placement to CSV (same as in find_placement method)
                self._write_placement_to_csv(
                    pod_id=pod.id,
                    node_id=candidate.flavour.id,
                    start_slot=candidate.timeslot.id,
                    duration=pod.duration,
                    cpu_request=pod.cpuRequest,
                    ram_request=pod.ramRequest,
                    flavour=candidate.flavour,
                    footprint=candidate.footprint,
                )
                self._record_budget_progress(candidate.footprint)
                
                return candidate.flavour, candidate.timeslot, candidate.footprint.total_carbon_g
            else:
                logging.debug(
                    "[find_placement_atomic] Allocation failed for pod=%s on node=%s, timeslot=%s (resources taken)",
                    pod.id,
                    candidate.flavour.id,
                    candidate.timeslot.id,
                )
        
        # No candidate could be allocated (all resources were taken by other threads)
        logging.warning(f"[find_placement_atomic] All {allocation_attempts} candidates exhausted for pod={pod.id} - resources taken by other processes")
        self._record_budget_progress(None)
        return None, None, float('inf')


def _candidate_water_value(footprint: FootprintVector, water_metric: str) -> float:
    if water_metric == "raw":
        return footprint.total_raw_water_l
    return footprint.scarcity_characterized_water


def _normalize_to_unit_interval(value: float, lower: float, upper: float) -> float:
    if upper <= lower + 1e-12:
        return 0.0
    return (value - lower) / (upper - lower)


def _candidate_environmental_tuple(candidate: CandidatePlacement, water_metric: str) -> Tuple[float, float]:
    return candidate.footprint.total_carbon_g, _candidate_water_value(candidate.footprint, water_metric)


def _dominates(lhs: CandidatePlacement, rhs: CandidatePlacement, water_metric: str, eps: float = 1e-12) -> bool:
    lhs_carbon, lhs_water = _candidate_environmental_tuple(lhs, water_metric)
    rhs_carbon, rhs_water = _candidate_environmental_tuple(rhs, water_metric)

    carbon_no_worse = lhs_carbon <= rhs_carbon + eps
    water_no_worse = lhs_water <= rhs_water + eps
    strictly_better = lhs_carbon < rhs_carbon - eps or lhs_water < rhs_water - eps
    return carbon_no_worse and water_no_worse and strictly_better


def _assign_pareto_ranks(candidates: List[CandidatePlacement], water_metric: str) -> None:
    remaining = list(candidates)
    current_rank = 0

    while remaining:
        frontier: List[CandidatePlacement] = []
        for candidate in remaining:
            if not any(_dominates(other, candidate, water_metric) for other in remaining if other is not candidate):
                frontier.append(candidate)

        for candidate in frontier:
            candidate.pareto_rank = current_rank
            candidate.objective_score = float(current_rank)

        remaining = [candidate for candidate in remaining if candidate not in frontier]
        current_rank += 1


def _rank_candidates(
    candidates: List[CandidatePlacement],
    objective_mode: str,
    carbon_weight: float,
    water_metric: str,
    water_budget_remaining: Optional[float] = None,
    remaining_pods: Optional[int] = None,
    budget_pressure_weight: float = 1.0,
) -> List[CandidatePlacement]:
    if not candidates:
        return []

    if objective_mode == "weighted-sum":
        carbon_values = [candidate.footprint.total_carbon_g for candidate in candidates]
        water_values = [_candidate_water_value(candidate.footprint, water_metric) for candidate in candidates]
        carbon_min, carbon_max = min(carbon_values), max(carbon_values)
        water_min, water_max = min(water_values), max(water_values)

        for candidate in candidates:
            carbon_norm = _normalize_to_unit_interval(candidate.footprint.total_carbon_g, carbon_min, carbon_max)
            water_norm = _normalize_to_unit_interval(_candidate_water_value(candidate.footprint, water_metric), water_min, water_max)
            candidate.objective_score = carbon_weight * carbon_norm + (1.0 - carbon_weight) * water_norm
            candidate.pareto_rank = 0
    elif objective_mode == "pareto":
        _assign_pareto_ranks(candidates, water_metric)
    elif objective_mode == "epsilon-pareto":
        _assign_pareto_ranks(candidates, water_metric)
        carbon_values = [candidate.footprint.total_carbon_g for candidate in candidates]
        carbon_min, carbon_max = min(carbon_values), max(carbon_values)
        water_values = [_candidate_water_value(candidate.footprint, water_metric) for candidate in candidates]
        water_min, water_max = min(water_values), max(water_values)

        allowance: Optional[float] = None
        if water_budget_remaining is not None:
            pods_left = max(int(remaining_pods or 1), 1)
            allowance = max(float(water_budget_remaining), 0.0) / pods_left

        for candidate in candidates:
            water_value = _candidate_water_value(candidate.footprint, water_metric)
            carbon_norm = _normalize_to_unit_interval(candidate.footprint.total_carbon_g, carbon_min, carbon_max)
            water_norm = _normalize_to_unit_interval(water_value, water_min, water_max)

            if water_budget_remaining is None:
                candidate.budget_violation = 0.0
                candidate.budget_slack_after = float("inf")
                candidate.water_pressure = water_norm
            else:
                remaining_budget = max(float(water_budget_remaining), 0.0)
                candidate.budget_slack_after = remaining_budget - water_value
                candidate.budget_violation = max(water_value - remaining_budget, 0.0)
                if allowance is not None and allowance > 1e-12:
                    candidate.water_pressure = max(water_value - allowance, 0.0) / allowance
                else:
                    candidate.water_pressure = water_norm

        pressure_values = [candidate.water_pressure for candidate in candidates]
        violation_values = [candidate.budget_violation for candidate in candidates]
        pressure_min, pressure_max = min(pressure_values), max(pressure_values)
        violation_min, violation_max = min(violation_values), max(violation_values)

        for candidate in candidates:
            carbon_norm = _normalize_to_unit_interval(candidate.footprint.total_carbon_g, carbon_min, carbon_max)
            pressure_norm = _normalize_to_unit_interval(candidate.water_pressure, pressure_min, pressure_max)
            violation_norm = _normalize_to_unit_interval(candidate.budget_violation, violation_min, violation_max)
            candidate.objective_score = carbon_norm + budget_pressure_weight * (pressure_norm + violation_norm)
    else:
        for candidate in candidates:
            candidate.objective_score = candidate.footprint.total_carbon_g
            candidate.pareto_rank = 0

    if objective_mode == "pareto":
        return sorted(
            candidates,
            key=lambda candidate: (
                candidate.pareto_rank,
                candidate.pack_score,
                candidate.timeslot.id,
                candidate.footprint.total_carbon_g,
                _candidate_water_value(candidate.footprint, water_metric),
                candidate.flavour.id,
            ),
        )

    if objective_mode == "epsilon-pareto":
        return sorted(
            candidates,
            key=lambda candidate: (
                1 if candidate.budget_violation > 1e-12 else 0,
                candidate.pareto_rank,
                candidate.objective_score,
                candidate.footprint.total_carbon_g,
                _candidate_water_value(candidate.footprint, water_metric),
                candidate.pack_score,
                candidate.timeslot.id,
                candidate.flavour.id,
            ),
        )

    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.objective_score,
            candidate.footprint.total_carbon_g,
            _candidate_water_value(candidate.footprint, water_metric),
            candidate.pack_score,
        ),
    )


def _build_feasible_candidates(
    pod: CarbonAwarePod,
    flavours: List[EnvironmentalFlavor],
    timeslots: List[CarbonAwareTimeslot],
    leftover_cpu: Dict[str, Dict[int, float]],
    leftover_ram: Dict[str, Dict[int, float]],
    max_time_slots: int = 48,
    operational_only: bool = False,
    embodied_allocation_mode: str = "proportional",
) -> List[CandidatePlacement]:
    candidates: List[CandidatePlacement] = []

    # Prefer lower-carbon hours first across nodes (best-effort)
    try:
        ordered_timeslots = sorted(
            timeslots,
            key=lambda t: min(flv.forecast.get(t.id, 200.0) for flv in flavours)
        )
    except Exception:
        ordered_timeslots = timeslots

    for ts in ordered_timeslots:
        if not is_timeslot_valid(ts, pod):
            logging.debug(f"[find_best_node_and_timeslot] Skipping timeslot={ts.id}, not valid for pod={pod.id}")
            continue

        for flv in flavours:
            duration_feasible = True
            for slot_offset in range(int(pod.duration)):
                current_slot = ts.id + slot_offset
                if current_slot >= max_time_slots:
                    duration_feasible = False
                    logging.debug(f"[find_best_node_and_timeslot] Slot {ts.id}+{slot_offset}={current_slot} exceeds tracking window for pod={pod.id}")
                    break

                if (leftover_cpu[flv.id][current_slot] < pod.cpuRequest or
                    leftover_ram[flv.id][current_slot] < pod.ramRequest):
                    duration_feasible = False
                    logging.debug(
                        f"[find_best_node_and_timeslot] Slot {current_slot} on {flv.id} doesn't have enough resources for pod={pod.id}: " +
                        f"CPU {leftover_cpu[flv.id][current_slot]:.2f}/{pod.cpuRequest:.2f}, " +
                        f"RAM {leftover_ram[flv.id][current_slot]:.0f}/{pod.ramRequest:.0f}"
                    )
                    break

            if duration_feasible:
                # Pre-compute packing slacks for tie-breaker
                cpu_slack_sum = 0.0
                ram_slack_sum = 0.0
                used_cpu_before = {}
                for slot_offset in range(int(pod.duration)):
                    slot_id = ts.id + slot_offset
                    leftover_cpu_now = leftover_cpu[flv.id][slot_id]
                    leftover_ram_now = leftover_ram[flv.id][slot_id]
                    cpu_slack_sum += max(leftover_cpu_now - pod.cpuRequest, 0.0)
                    ram_slack_sum += max(leftover_ram_now - pod.ramRequest, 0.0)
                    total_capacity = flv.totalCpu
                    used_cpu_before[slot_id] = max(total_capacity - leftover_cpu_now, 0.0)

                footprint = compute_footprint_vector(
                    flavour=flv,
                    start_slot=ts.id,
                    pod=pod,
                    used_cpu_before_by_slot=used_cpu_before,
                    embodied_allocation_mode=embodied_allocation_mode,
                    operational_only=operational_only,
                    use_pod_power_only=operational_only,
                )

                cpu_norm = flv.totalCpu * max(int(pod.duration), 1)
                ram_norm = flv.totalRam * max(int(pod.duration), 1)
                pack_score = (cpu_slack_sum / max(cpu_norm, 1e-6)) + (ram_slack_sum / max(ram_norm, 1e-6))
                candidates.append(
                    CandidatePlacement(
                        flavour=flv,
                        timeslot=ts,
                        footprint=footprint,
                        pack_score=pack_score,
                        used_cpu_before=used_cpu_before,
                    )
                )

    return candidates


def find_ranked_candidates(
    pod: CarbonAwarePod,
    flavours: List[EnvironmentalFlavor],
    timeslots: List[CarbonAwareTimeslot],
    leftover_cpu: Dict[str, Dict[int, float]],
    leftover_ram: Dict[str, Dict[int, float]],
    max_time_slots: int = 48,
    operational_only: bool = False,
    embodied_allocation_mode: str = "proportional",
    objective_mode: str = "carbon",
    carbon_weight: float = 1.0,
    water_metric: str = "scarcity",
    water_budget_remaining: Optional[float] = None,
    remaining_pods: Optional[int] = None,
    budget_pressure_weight: float = 1.0,
) -> List[CandidatePlacement]:
    candidates = _build_feasible_candidates(
        pod=pod,
        flavours=flavours,
        timeslots=timeslots,
        leftover_cpu=leftover_cpu,
        leftover_ram=leftover_ram,
        max_time_slots=max_time_slots,
        operational_only=operational_only,
        embodied_allocation_mode=embodied_allocation_mode,
    )
    return _rank_candidates(
        candidates,
        objective_mode=objective_mode,
        carbon_weight=carbon_weight,
        water_metric=water_metric,
        water_budget_remaining=water_budget_remaining,
        remaining_pods=remaining_pods,
        budget_pressure_weight=budget_pressure_weight,
    )


def find_best_node_and_timeslot(
    pod: CarbonAwarePod,
    flavours: List[EnvironmentalFlavor],
    timeslots: List[CarbonAwareTimeslot],
    leftover_cpu: Dict[str, Dict[int, float]],
    leftover_ram: Dict[str, Dict[int, float]],
    max_time_slots: int = 48,
    operational_only: bool = False,
    embodied_allocation_mode: str = "proportional",
    objective_mode: str = "carbon",
    carbon_weight: float = 1.0,
    water_metric: str = "scarcity",
    water_budget_remaining: Optional[float] = None,
    remaining_pods: Optional[int] = None,
    budget_pressure_weight: float = 1.0,
) -> Tuple[Optional[EnvironmentalFlavor], Optional[CarbonAwareTimeslot], float]:
    """
    Find the best node and timeslot for a pod using the configured heuristic objective.

    The algorithm iterates through all valid combinations of nodes and timeslots,
    checking resource constraints and calculating emissions for each.

    Args:
        pod: The pod to place
        flavours: Available node types
        timeslots: Available scheduling timeslots
        leftover_cpu: Remaining CPU capacity per node and timeslot
        leftover_ram: Remaining RAM capacity per node and timeslot
        max_time_slots: Maximum number of timeslots to consider
        operational_only: Flag to use operational-only emissions calculation
        objective_mode: `carbon`, `weighted-sum`, `pareto`, or `epsilon-pareto`
        carbon_weight: Weighted-sum carbon share in [0, 1]
        water_metric: `scarcity` or `raw`
        water_budget_remaining: Remaining run-level water budget for epsilon-pareto mode
        remaining_pods: Estimated number of pods left including the current pod
        budget_pressure_weight: Strength of per-pod water-budget pressure

    Returns:
        Tuple of (best_node, best_timeslot, chosen carbon emissions in gCO2e)
        or (None, None, inf) if no placement found
    """
    ranked_candidates = find_ranked_candidates(
        pod=pod,
        flavours=flavours,
        timeslots=timeslots,
        leftover_cpu=leftover_cpu,
        leftover_ram=leftover_ram,
        max_time_slots=max_time_slots,
        operational_only=operational_only,
        embodied_allocation_mode=embodied_allocation_mode,
        objective_mode=objective_mode,
        carbon_weight=carbon_weight,
        water_metric=water_metric,
        water_budget_remaining=water_budget_remaining,
        remaining_pods=remaining_pods,
        budget_pressure_weight=budget_pressure_weight,
    )

    if not ranked_candidates:
        return None, None, float("inf")

    best_candidate = ranked_candidates[0]
    logging.debug(
        "[find_best_node_and_timeslot] Best candidate for pod=%s: node=%s timeslot=%s objective=%.6f carbon=%.3f water=%.3f",
        pod.id,
        best_candidate.flavour.id,
        best_candidate.timeslot.id,
        best_candidate.objective_score,
        best_candidate.footprint.total_carbon_g,
        _candidate_water_value(best_candidate.footprint, water_metric),
    )

    return best_candidate.flavour, best_candidate.timeslot, best_candidate.footprint.total_carbon_g
