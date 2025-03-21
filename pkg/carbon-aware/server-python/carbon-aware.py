from concurrent import futures
import json
import logging
import signal
import threading
import sys
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import argparse
import decimal
import copy

import grpc
import idl_pb2
import idl_pb2_grpc
from google.protobuf import empty_pb2
from kubernetes import utils as k8sutils

# Define constants for microservice status
MICROSERVICE_STATUS_MAP = {
    0: "MICROSERVICESTATUS_UNSPECIFIED",
    1: "RUNNING",
    2: "PENDING", 
    3: "TO_SCHEDULE",
    4: "TO_DEPLOY"
}

class PersistentStateStorage:
    """
    Maintains resource allocation state between API calls.
    """
    def __init__(self):
        self.leftover_cpu = {}  # node_id -> {timeslot_id -> available_cpu}
        self.leftover_ram = {}  # node_id -> {timeslot_id -> available_ram}
        self.placements = {}    # microservice_id -> {node_id, start_time, duration}
        self.max_time_slots = 48
        self.initialized = False
        self.state_lock = threading.RLock()  # For thread safety
        
    def initialize(self, flavours):
        """Initialize the state with full capacity for all nodes."""
        with self.state_lock:
            self.leftover_cpu = {}
            self.leftover_ram = {}
            
            for flv in flavours:
                self.leftover_cpu[flv.id] = {}
                self.leftover_ram[flv.id] = {}
                for ts_id in range(self.max_time_slots):
                    self.leftover_cpu[flv.id][ts_id] = flv.totalCpu
                    self.leftover_ram[flv.id][ts_id] = flv.totalRam
            
            self.initialized = True
            logging.info("🔄 Persistent state initialized")
    
    def get_resources(self):
        """Get a copy of the current resource state."""
        with self.state_lock:
            return copy.deepcopy(self.leftover_cpu), copy.deepcopy(self.leftover_ram)
    
    def update_resources(self, node_id, start_slot, duration, cpu_request, ram_request):
        """Reserve resources for a pod placement."""
        with self.state_lock:
            for slot_offset in range(duration):
                current_slot = start_slot + slot_offset
                if current_slot < self.max_time_slots:
                    self.leftover_cpu[node_id][current_slot] -= cpu_request
                    self.leftover_ram[node_id][current_slot] -= ram_request
            logging.info(f"📝 Updated persistent state for node={node_id}, slots={start_slot}-{start_slot+duration-1}")
    
    def record_placement(self, ms_name, node_id, start_time, duration):
        """Record a placement decision."""
        with self.state_lock:
            self.placements[ms_name] = {
                'node_id': node_id,
                'start_time': start_time,
                'duration': duration
            }


# classes to be imported from mbmo in the future:
class CarbonAwareFlavour:
    """
    Represents a node (or flavor) with carbon and resource characteristics.
    """
    def __init__(self, id: str, embodiedCarbon: float, lifetime: float, totalCpu: float, 
                 totalRam: float, totalStorage: float, forecast: Dict[int, float],
                 power: Dict[str, float] = None):
        self.id = id
        self.embodiedCarbon = embodiedCarbon
        self.lifetime = lifetime
        self.totalCpu = totalCpu
        self.totalRam = totalRam
        self.totalStorage = totalStorage
        self.forecast = forecast
        self.power = power or {"idle": 100.0, "active": 200.0, "max": 400.0}  # Default server values

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__dict__!r})"


class CarbonAwarePod:
    def __init__(self, id: str, deadline_hours: float, duration: float,
                 powerConsumption: float, cpuRequest: float,
                 ramRequest: float, storageRequest: int,
                 reference_time: datetime = None) -> None:
        self.id = id
        self.deadline = self._processDeadline(deadline_hours, reference_time)
        self.duration = duration  # Still store internally as float hours
        self.powerConsumption = powerConsumption
        self.cpuRequest = cpuRequest
        self.ramRequest = ramRequest
        self.storageRequest = storageRequest

    def _processDeadline(self, deadline_hours: float, reference_time: datetime = None) -> datetime:
        """
        Convert a float-based 'deadline_hours' into a datetime.
        
        Args:
            deadline_hours: Hours until deadline
            reference_time: Optional reference time (uses current time if None)
            
        Returns:
            Absolute deadline as datetime
        """
        now = datetime.now() if reference_time is None else reference_time
        delta = timedelta(hours=deadline_hours)
        return now + delta


class CarbonAwareTimeslot:
    """
    Represents a discrete block of time for scheduling,
    identified by an integer 'id', and a start datetime.
    The length is specified in hours.
    """
    def __init__(self, id: int, startYear: int, startMonth: int, startDay: int,
                 startHour: int, length: int) -> None:
        self.id = id
        self.start = datetime(startYear, startMonth, startDay, startHour)
        self.length = timedelta(hours=length)

    def getEnd(self) -> datetime:
        """
        Return the end of this timeslot, i.e., start + length.
        """
        return self.start + self.length

    def getStart(self) -> datetime:
        """
        Return the start datetime of this timeslot.
        """
        return self.start


#####################################
# Helper Functions
#####################################

def is_timeslot_valid(ts: CarbonAwareTimeslot, pod: CarbonAwarePod) -> bool:
    """
    Determines whether the timeslot is valid for the given pod.
    Valid if:
      - The current time hasn't passed the timeslot's end.
      - The timeslot starts before the pod's deadline.
    """
    now = datetime.now()
    valid = (now <= ts.getEnd()) and (ts.getStart() <= pod.deadline)

    logging.debug(
        f"[is_timeslot_valid] Timeslot {ts.id}: start={ts.getStart()}, "
        f"end={ts.getEnd()}, now={now}, pod_deadline={pod.deadline}, valid={valid}"
    )
    return valid


def check_node_resource(
    flavour: CarbonAwareFlavour,
    ts: CarbonAwareTimeslot,
    pod: CarbonAwarePod,
    available_cpu: float,
    available_ram: float
) -> bool:
    """
    Instead of checking flavour.totalCpu, we now check leftover resources
    'available_cpu' and 'available_ram' for that node/timeslot.
    """
    feasible = (available_cpu >= pod.cpuRequest) and (available_ram >= pod.ramRequest)
    logging.debug(
        f"[check_node_resource] Node={flavour.id}, Timeslot={ts.id}, "
        f"available_cpu={available_cpu:.2f}, available_ram={available_ram:.2f}, "
        f"PodCPU={pod.cpuRequest:.2f}, PodRAM={pod.ramRequest:.2f}, feasible={feasible}"
    )
    return feasible


def get_node_region(node) -> str:
    """
    Determine the region for a node based on its metadata,
    specifically focusing on the topology.kubernetes.io/region label.
    
    Args:
        node: Node object with metadata
        
    Returns:
        Region code as uppercase string (e.g., "DE", "FR")
    """
    # Primary method: Check for standard Kubernetes topology.kubernetes.io/region label
    if hasattr(node, "labels") and node.labels:
        for label in node.labels:
            if label.key == "topology.kubernetes.io/region":
                logging.debug(f"Found region from topology label: {label.value}")
                return label.value.upper()
    
    # Fallback: Extract from node name if it contains a region identifier
    if hasattr(node, "name"):
        node_name = node.name.lower()
        for region_code in ["de", "fr", "es", "it-no"]:
            if f"-{region_code}" in node_name:
                logging.debug(f"Extracted region from node name: {region_code.upper()}")
                return region_code.upper()
    
    # Last resort: default region
    logging.warning(f"Could not determine region for node {node.name}, using default region")
    return "DE"  # Default fallback region


def load_carbon_intensity_data(json_file_path: str = "all_forecasts.json", regions: Optional[List[str]] = None) -> Dict[str, Dict[int, float]]:
    """
    Load carbon intensity data from the specified JSON file.
    
    Args:
        json_file_path: Path to the JSON file with carbon intensity data.
                       Defaults to "all_forecasts.json" in the current directory.
        regions: List of region codes to filter (e.g., ["DE", "FR"]). If None, load all regions.
    
    Returns:
        A nested dictionary mapping region codes to timeslot forecasts:
        {region_code: {timeslot_id: carbon_intensity}}
    """
    # Data structure to return
    carbon_data = {}
    
    try:
        # Load data from the specified JSON file
        logging.info(f"Loading carbon intensity data from {json_file_path}")
        with open(json_file_path, 'r') as f:
            raw_data = json.load(f)
        
        # Filter regions if specified
        if regions:
            region_codes = [r for r in regions if r in raw_data]
            logging.info(f"Filtering data for regions: {region_codes}")
        else:
            region_codes = list(raw_data.keys())
            logging.info(f"Processing all available regions: {region_codes}")
        
        # Base time for calculating timeslot IDs
        # Use the first datetime in the forecast to establish the reference point
        base_time = None
        
        # Process each region
        for region in region_codes:
            if region not in raw_data:
                logging.warning(f"Region {region} not found in carbon intensity data")
                continue
                
            forecast_data = raw_data[region]["forecast"]
            
            if not forecast_data:
                logging.warning(f"No forecast data for region {region}")
                continue
                
            # Find base time if not set
            if base_time is None:
                first_datetime_str = forecast_data[0]["datetime"]
                base_time = datetime.fromisoformat(first_datetime_str.replace("Z", "+00:00"))
                logging.info(f"Base time set to {base_time}")
            
            # Process forecast entries
            forecast_dict = {}
            for entry in forecast_data:
                carbon_intensity = entry["carbonIntensity"]
                entry_time = datetime.fromisoformat(entry["datetime"].replace("Z", "+00:00"))
                
                # Calculate timeslot ID (hours since base time)
                hours_diff = int((entry_time - base_time).total_seconds() / 3600)
                timeslot_id = hours_diff
                
                forecast_dict[timeslot_id] = carbon_intensity
            
            # Store the forecast for this region
            carbon_data[region] = forecast_dict
            logging.info(f"Processed {len(forecast_dict)} forecast entries for region {region}")
        
    except FileNotFoundError:
        logging.error(f"Carbon intensity data file not found: {json_file_path}")
        return {}
    except json.JSONDecodeError:
        logging.error(f"Error parsing JSON data from {json_file_path}")
        return {}
    except Exception as e:
        logging.error(f"Error loading carbon intensity data: {e}")
        return {}
    
    return carbon_data


def compute_emissions(flavour: CarbonAwareFlavour, timeslot_id: int, pod: CarbonAwarePod) -> float:
    """
    Compute the total carbon emissions for placing 'pod' on 'flavour' during timeslot 'timeslot_id'.
    
    Power consumption model:
    - Base power = idle power
    - Additional power based on CPU usage = (active-idle) * (pod.cpuRequest / flavour.totalCpu)
    
    Emissions calculation:
    - operationalEmissions = carbon_intensity * duration * power_consumption (kW)
    - embodiedEmissions = (embodiedCarbon / (365 * lifetime * 24)) * duration
    """
    # Default carbon intensity if timeslot not in forecast
    carbon_intensity = flavour.forecast.get(timeslot_id, 200.0)

    # Calculate power consumption based on CPU usage
    # Use the formula: idle + (active-idle) * cpu_usage_ratio
    idle_power = flavour.power["idle"]  # watts
    active_power = flavour.power["active"]  # watts
    max_power = flavour.power["max"]  # watts
    
    # CPU usage ratio (ensure we don't divide by zero)
    cpu_usage_ratio = pod.cpuRequest / max(flavour.totalCpu, 0.001)  # Avoid division by zero
    
    # Calculate power based on the formula: idle + (max-active) * cpu_usage_ratio
    power_consumption_watts = idle_power + (max_power - active_power) * cpu_usage_ratio
    
    # Convert watts to kilowatts for emissions calculation
    pod.powerConsumption = power_consumption_watts / 1000.0  # convert W to kW
    
    # Calculate operational emissions
    operationalEmissions = carbon_intensity * pod.duration * pod.powerConsumption

    # Embodied carbon distributed over lifetime (in hours)
    hours_in_lifetime = 365 * flavour.lifetime * 24
    if hours_in_lifetime <= 0:
        hours_in_lifetime = 1e-6
    embodied_per_hour = flavour.embodiedCarbon / hours_in_lifetime
    embodiedEmissions = embodied_per_hour * pod.duration

    total_emi = operationalEmissions + embodiedEmissions

    logging.debug(
        f"[compute_emissions] Node={flavour.id}, TimeslotID={timeslot_id}, "
        f"carbon_intensity={carbon_intensity}, duration={pod.duration}, "
        f"cpu_ratio={cpu_usage_ratio:.2f}, power={power_consumption_watts:.2f}W ({pod.powerConsumption:.3f}kW), "
        f"operationalEmi={operationalEmissions:.3f}, embodiedEmi={embodiedEmissions:.3f}, total={total_emi:.3f}"
    )
    return total_emi


def get_node_region(node) -> str:
    """
    Extract the region identifier from the node.
    This function tries multiple approaches to determine a node's region.
    """
    # 1. Check if we can directly get the region from the node object
    if hasattr(node, "region") and node.region:
        return node.region.upper()  # Ensure uppercase for consistency

    # 2. Check for standard naming format: node-order-region-subcategory
    if hasattr(node, "name") and node.name.startswith("node-"):
        parts = node.name.split("-")
        if len(parts) >= 4:  # node-0-DE-Server format
            region = parts[2].upper()
            logging.info(f"Extracted region {region} from node name format: {node.name}")
            return region

    # 3. Check for region in labels if available
    if hasattr(node, "labels"):
        for label in node.labels:
            if label.key.lower() in ["region", "topology.kubernetes.io/region"]:
                return label.value.upper()
    
    # 4. Look for region codes anywhere in the name (fallback)
    node_name_lower = node.name.lower()
    for region_code in ["de", "fr", "es", "it-no"]:
        if region_code in node_name_lower:
            return region_code.upper()
    
    # Fallback to default region
    logging.warning(f"Could not determine region for node {node.name}, using default region")
    return "DE"  # Default fallback region


def get_node_hardware_metadata(node) -> tuple[float, float, Dict[str, float]]:
    """
    Extract hardware metadata related to carbon emissions and power consumption.
    Priority order:
    1. Annotations (specific hardware values)
    2. Labels (hardware subcategory)
    3. Direct node attributes (subcategory)
    4. Default values as fallback
    """
    # Default values if metadata not found
    default_embodied_carbon = 50000.0
    default_lifetime = 4.0
    default_power = {
        "idle": 100.0,
        "active": 200.0,
        "max": 400.0
    }
    
    embodied_carbon = default_embodied_carbon
    lifetime = default_lifetime
    power = default_power.copy()
    
    # STEP 1: Check for hardware carbon metadata in annotations (HIGHEST PRIORITY)
    if hasattr(node, "annotations") and node.annotations:
        for annotation in node.annotations:
            # Embodied carbon and lifetime
            if annotation.key == "hardware.carbon/embodied_emissions":
                try:
                    embodied_carbon = float(annotation.value)
                    logging.debug(f"Found embodied carbon from annotation: {embodied_carbon}")
                except (ValueError, TypeError):
                    logging.warning(f"Invalid embodied carbon value: {annotation.value}, using default")
                
            if annotation.key == "hardware.carbon/lifetime_years":
                try:
                    lifetime = float(annotation.value)
                    logging.debug(f"Found lifetime from annotation: {lifetime}")
                except (ValueError, TypeError):
                    logging.warning(f"Invalid lifetime value: {annotation.value}, using default")
                    
            # Power consumption values
            if annotation.key == "hardware.power/idle_watts":
                try:
                    power["idle"] = float(annotation.value)
                    logging.debug(f"Found idle power from annotation: {power['idle']}W")
                except (ValueError, TypeError):
                    logging.warning(f"Invalid idle power value: {annotation.value}, using default")
                    
            if annotation.key == "hardware.power/active_watts":
                try:
                    power["active"] = float(annotation.value)
                    logging.debug(f"Found active power from annotation: {power['active']}W")
                except (ValueError, TypeError):
                    logging.warning(f"Invalid active power value: {annotation.value}, using default")
                    
            if annotation.key == "hardware.power/max_watts":
                try:
                    power["max"] = float(annotation.value)
                    logging.debug(f"Found max power from annotation: {power['max']}W")
                except (ValueError, TypeError):
                    logging.warning(f"Invalid max power value: {annotation.value}, using default")
    
    # STEP 2: Check for hardware subcategory label (MEDIUM PRIORITY)
    subcategory = None
    if (embodied_carbon == default_embodied_carbon or lifetime == default_lifetime or power == default_power) and hasattr(node, "labels") and node.labels:
        for label in node.labels:
            if label.key == "hardware.carbon/subcategory":
                subcategory = label.value
                logging.debug(f"Found hardware subcategory from label: {subcategory}")
                break
    
    # STEP 3: Check for direct subcategory attribute (LOWEST PRIORITY)
    if not subcategory and (embodied_carbon == default_embodied_carbon or lifetime == default_lifetime or power == default_power) and hasattr(node, "subcategory") and node.subcategory:
        subcategory = node.subcategory
        logging.debug(f"Using subcategory directly from node: {subcategory}")
    
    # Apply subcategory-based values for any values not set by annotations
    if subcategory:
        if subcategory == "IoT":
            if embodied_carbon == default_embodied_carbon:
                embodied_carbon = 27.471
            if lifetime == default_lifetime:
                lifetime = 5.2
            if power == default_power:  # Only replace if we haven't found any power annotations
                power = {"idle": 0.5, "active": 2.0, "max": 5.0}
        elif subcategory == "Smartphone":
            if embodied_carbon == default_embodied_carbon:
                embodied_carbon = 52.729
            if lifetime == default_lifetime:
                lifetime = 3.03
            if power == default_power:
                power = {"idle": 1.0, "active": 3.0, "max": 15.0}
        elif subcategory == "Laptop":
            if embodied_carbon == default_embodied_carbon:
                embodied_carbon = 231.855
            if lifetime == default_lifetime:
                lifetime = 4.13
            if power == default_power:
                power = {"idle": 10.0, "active": 40.0, "max": 150.0}
        elif subcategory == "Server":
            if embodied_carbon == default_embodied_carbon:
                embodied_carbon = 1230.656
            if lifetime == default_lifetime:
                lifetime = 3.87
            if power == default_power:
                power = {"idle": 100.0, "active": 200.0, "max": 400.0}
    
    # Log the final values
    logging.info(f"Node {node.name} hardware metadata: embodied_carbon={embodied_carbon}kg CO2e, "
                f"lifetime={lifetime}y, power(idle)={power['idle']}W, power(active)={power['active']}W")
    
    return embodied_carbon, lifetime, power


def parse_infrastructure(infra: idl_pb2.Infrastructure) -> list[CarbonAwareFlavour]:
    # Load carbon intensity data once
    carbon_data = load_carbon_intensity_data()
    logging.info(f"Loaded carbon intensity data for regions: {list(carbon_data.keys())}")

    flavours = []
    for node in infra.nodes:
        # 1) Parse CPU and memory to a Decimal
        cpu_cap = k8sutils.parse_quantity(node.cpu_cap.value)
        mem_cap = k8sutils.parse_quantity(node.mem_cap.value)

        # 2) Convert Decimal → float
        cpu_cores = float(cpu_cap)  # e.g. 0.1 for "100m"
        mem_bytes = float(mem_cap)  # e.g. 1073741824 for "1Gi"

        # 3) Decide how to store them
        total_cpu = cpu_cores  # store as cores
        total_ram = mem_bytes / (1024 * 1024)  # store as MB

        # 4) Extract node's region based on topology.kubernetes.io/region label
        node_region = get_node_region(node)
        
        # 5) Extract embodied carbon, lifetime, and power consumption from node metadata
        embodied_carbon, lifetime, power_consumption = get_node_hardware_metadata(node)
        
        # 6) Get carbon intensity forecast for this region
        if node_region in carbon_data:
            forecast_dict = carbon_data[node_region]
            logging.info(f"Using carbon intensity data for node {node.name} (region {node_region})")
        else:
            # Fallback to random values if region not found in carbon data
            forecast_dict = {i: random.uniform(20, 1000) for i in range(24)}
            logging.warning(f"No carbon data for region {node_region}, using random values for node {node.name}")

        flavour_obj = CarbonAwareFlavour(
            id=node.name,
            embodiedCarbon=embodied_carbon,
            lifetime=lifetime,
            totalCpu=total_cpu,
            totalRam=total_ram,
            totalStorage=1000,
            forecast=forecast_dict,
            power=power_consumption  # Add power consumption to the CarbonAwareFlavour
        )
        logging.debug(
            f"[parse_infrastructure] Created flavour for node {node.name}: "
            f"region={node_region}, embodiedCarbon={embodied_carbon}, lifetime={lifetime}, "
            f"idle_power={power_consumption['idle']}W, active_power={power_consumption['active']}W, "
            f"max_power={power_consumption['max']}W"
        )
        flavours.append(flavour_obj)

    return flavours


def parse_duration_to_hours(duration_str: str) -> float:
    """Convert duration string like '1h', '90m', '1.5h' to hours as float."""
    if not duration_str:
        return 0.0
    
    # Remove common prefixes like "duration-" or "deadline-"
    if duration_str.startswith("duration-"):
        duration_str = duration_str[9:]  # Remove "duration-" prefix
    elif duration_str.startswith("deadline-"):
        duration_str = duration_str[9:]  # Remove "deadline-" prefix
    
    # Handle direct float (backward compatibility)
    try:
        return float(duration_str)
    except ValueError:
        pass
        
    # Handle hours notation
    if duration_str.endswith('h'):
        try:
            return float(duration_str[:-1])
        except ValueError:
            pass
            
    # Handle minutes notation
    if duration_str.endswith('m'):
        try:
            return float(duration_str[:-1]) / 60.0
        except ValueError:
            pass
            
    # Handle days notation
    if duration_str.endswith('d'):
        try:
            return float(duration_str[:-1]) * 24.0
        except ValueError:
            pass
    
    # Default fallback
    logging.warning(f"Could not parse duration: {duration_str}, using default")
    return 1.0  # Default to 1 hour


def parse_microservice(ms: idl_pb2.Microservice) -> CarbonAwarePod:
    try:
        # Default values based on the pod manifest
        default_cpu = 0.25  # 250m in Kubernetes notation
        default_ram = 512   # 512MB in megabytes
        
        # Parse CPU with validation
        try:
            if (hasattr(ms, "cpu_required") and ms.cpu_required and 
                hasattr(ms.cpu_required, "value") and ms.cpu_required.value):
                cpu_req = k8sutils.parse_quantity(ms.cpu_required.value)
                float_cpu_req = float(cpu_req)
                logging.info(f"Parsed CPU for {ms.name}: {float_cpu_req} cores")
            else:
                logging.warning(f"Empty CPU requirement for {ms.name}, using default: {default_cpu}")
                float_cpu_req = default_cpu
        except (ValueError, decimal.InvalidOperation) as e:
            cpu_value = ms.cpu_required.value if hasattr(ms, "cpu_required") and ms.cpu_required else "None"
            logging.warning(f"Invalid CPU format for {ms.name}: '{cpu_value}', using default: {default_cpu}")
            float_cpu_req = default_cpu
            
        # Parse RAM with validation
        try:
            if (hasattr(ms, "mem_required") and ms.mem_required and 
                hasattr(ms.mem_required, "value") and ms.mem_required.value):
                ram_req = k8sutils.parse_quantity(ms.mem_required.value)
                float_ram_req = float(ram_req) / (1024 * 1024)  # bytes → MB
                logging.info(f"Parsed RAM for {ms.name}: {float_ram_req} MB")
            else:
                logging.warning(f"Empty RAM requirement for {ms.name}, using default: {default_ram}MB")
                float_ram_req = default_ram
        except (ValueError, decimal.InvalidOperation) as e:
            ram_value = ms.mem_required.value if hasattr(ms, "mem_required") and ms.mem_required else "None"
            logging.warning(f"Invalid RAM format for {ms.name}: '{ram_value}', using default: {default_ram}MB")
            float_ram_req = default_ram

        # APPROACH 1: Get duration and deadline from direct fields
        duration_hours = None
        deadline_hours = None

        # Get values from the IDL fields
        if hasattr(ms, "duration_hours") and ms.duration_hours:
            duration_hours = parse_duration_to_hours(ms.duration_hours)
            logging.info(f"Using duration from direct field: {duration_hours}h (from '{ms.duration_hours}')")
            
        if hasattr(ms, "deadline_hours") and ms.deadline_hours:
            deadline_hours = parse_duration_to_hours(ms.deadline_hours)
            logging.info(f"Using deadline from direct field: {deadline_hours}h (from '{ms.deadline_hours}')")
        
        # APPROACH 2: Fall back to parsing from name if needed
        if duration_hours is None:
            duration_hours = 1.0  # default
            if "-duration-" in ms.name:
                try:
                    parts = ms.name.split("-duration-")[1].split("-")[0].rstrip("h")
                    duration_hours = float(parts)
                    logging.info(f"Extracted duration from name: {duration_hours}h")
                except (ValueError, IndexError) as e:
                    logging.warning(f"Failed to parse duration from name '{ms.name}': {e}")
        
        if deadline_hours is None:
            deadline_hours = max(24.0, duration_hours * 2)  # default
            if "-deadline-" in ms.name:
                try:
                    parts = ms.name.split("-deadline-")[1].split("-")[0].rstrip("h")
                    deadline_hours = float(parts)
                    logging.info(f"Extracted deadline from name: {deadline_hours}h")
                except (ValueError, IndexError) as e:
                    logging.warning(f"Failed to parse deadline from name '{ms.name}': {e}")
        
        # Ensure deadline is at least as long as duration
        if deadline_hours < duration_hours:
            logging.warning(f"Deadline ({deadline_hours}h) shorter than duration ({duration_hours}h), adjusting to match")
            deadline_hours = duration_hours
        
        # Create CarbonAwarePod with separate duration and deadline
        pod = CarbonAwarePod(
            id=ms.name,
            deadline_hours=deadline_hours,
            duration=duration_hours,
            powerConsumption=0.0,
            cpuRequest=float_cpu_req,
            ramRequest=float_ram_req,
            storageRequest=0
        )
        
        # Enhanced logging to show the distinction
        scheduling_window = deadline_hours - duration_hours
        logging.info(f"[parse_microservice] {ms.name}: duration={duration_hours}h, " +
                    f"deadline={deadline_hours}h, scheduling window={scheduling_window}h")
        
        return pod
        
    except Exception as e:
        logging.exception(f"Error parsing microservice {ms.name}: {e}")
        return CarbonAwarePod(
            id=ms.name,
            deadline_hours=24.0,  # Default deadline
            duration=1.0,         # Default duration
            powerConsumption=0.0,
            cpuRequest=0.1,
            ramRequest=100,
            storageRequest=0
        )
    

def build_timeslots(deadline_hours: float) -> list[CarbonAwareTimeslot]:
    """
    Generate discrete timeslots from 'now' to 'now + deadline_hours'.
    Each timeslot is 1 hour long.
    """
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    max_slots = int(deadline_hours)  # e.g., 6 if deadline_hours=6
    timeslots = []
    for i in range(max_slots):
        start_dt = now + timedelta(hours=i)
        ts = CarbonAwareTimeslot(
            id=i,
            startYear=start_dt.year,
            startMonth=start_dt.month,
            startDay=start_dt.day,
            startHour=start_dt.hour,
            length=1
        )
        timeslots.append(ts)
    return timeslots


def find_best_node_and_timeslot(
    pod: CarbonAwarePod,
    flavours: list[CarbonAwareFlavour],
    timeslots: list[CarbonAwareTimeslot],
    leftover_cpu: dict[str, dict[int, float]],
    leftover_ram: dict[str, dict[int, float]],
    max_time_slots: int = 48
) -> tuple[CarbonAwareFlavour | None, CarbonAwareTimeslot | None, float]:
    best_node = None
    best_slot = None
    minimal_emissions = float('inf')

    for ts in timeslots:
        if not is_timeslot_valid(ts, pod):
            logging.debug(f"[find_best_node_and_timeslot] Skipping timeslot={ts.id}, not valid for pod={pod.id}")
            continue

        for flv in flavours:
            # Check resources for ENTIRE DURATION of the workload
            duration_feasible = True
            for slot_offset in range(int(pod.duration)):
                current_slot = ts.id + slot_offset
                if current_slot >= max_time_slots:
                    # Would run beyond our tracking window
                    duration_feasible = False
                    logging.debug(f"[find_best_node_and_timeslot] Slot {ts.id}+{slot_offset}={current_slot} exceeds tracking window for pod={pod.id}")
                    break
                
                # Check if there's enough CPU and RAM at this timeslot
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
                total_emi = compute_emissions(flv, ts.id, pod)
                if total_emi < minimal_emissions:
                    minimal_emissions = total_emi
                    best_node = flv
                    best_slot = ts
                    logging.debug(
                        f"[find_best_node_and_timeslot] New best found for pod={pod.id}: "
                        f"node={best_node.id}, timeslot={best_slot.id}, emissions={minimal_emissions:.3f}"
                    )

    return best_node, best_slot, minimal_emissions


def print_resource_utilization_report(flavours, leftover_cpu, leftover_ram, max_time_slots, hours_to_show=24):
    """
    Prints a detailed utilization report for all nodes and timeslots.
    
    Args:
        flavours: List of CarbonAwareFlavour objects
        leftover_cpu: Dictionary of remaining CPU resources by node and timeslot
        leftover_ram: Dictionary of remaining RAM resources by node and timeslot
        max_time_slots: Total number of timeslots to consider
        hours_to_show: How many hours to display in the report (default 24)
    """
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    logging.info("=" * 100)
    logging.info("📊 DETAILED RESOURCE UTILIZATION REPORT BY TIMESLOT")
    logging.info("=" * 100)
    
    # Limit the display to a reasonable number of hours
    display_slots = min(hours_to_show, max_time_slots)
    
    # Header row showing timeslots
    header = "Node         | Total     |"
    for slot in range(display_slots):
        slot_time = (now + timedelta(hours=slot)).strftime("%H:%M")
        header += f" {slot_time} |"
    logging.info(header)
    logging.info("-" * len(header))
    
    # Show data for each node
    for flv in flavours:
        node_id = flv.id
        # CPU row
        cpu_row = f"{node_id[:10]:<10} | CPU {flv.totalCpu:4.1f} |"
        for slot in range(display_slots):
            used_cpu = flv.totalCpu - leftover_cpu[node_id][slot]
            pct = (used_cpu / flv.totalCpu) * 100 if flv.totalCpu > 0 else 0
            # Mark high utilization with color indicators
            if pct > 95:
                cpu_row += f" \033[91m{used_cpu:4.1f}\033[0m |"  # Red for >95%
            elif pct > 80:
                cpu_row += f" \033[93m{used_cpu:4.1f}\033[0m |"  # Yellow for >80%
            else:
                cpu_row += f" {used_cpu:4.1f} |"
        logging.info(cpu_row)
        
        # RAM row
        ram_row = f"{' '*10} | RAM {flv.totalRam:4.0f} |"
        for slot in range(display_slots):
            used_ram = flv.totalRam - leftover_ram[node_id][slot]
            pct = (used_ram / flv.totalRam) * 100 if flv.totalRam > 0 else 0
            # Mark high utilization with color indicators
            if pct > 95:
                ram_row += f" \033[91m{used_ram:4.0f}\033[0m |"  # Red for >95%
            elif pct > 80:
                ram_row += f" \033[93m{used_ram:4.0f}\033[0m |"  # Yellow for >80%
            else:
                ram_row += f" {used_ram:4.0f} |"
        logging.info(ram_row)
        logging.info("-" * len(header))
    
    logging.info("=" * 100)


#####################################
# gRPC Service & Server Setup
#####################################

class Algorithm:
    """
    Tracks which algorithm is in use and whether it's initialized.
    """
    def __init__(self, name: str, initialized: bool) -> None:
        self.name = name
        self.initialized = initialized

algo = Algorithm("Carbon-Aware", False)

persistent_state = PersistentStateStorage()


class PlacementAlgorithm(idl_pb2_grpc.PlacementAlgorithmServicer):
    """
    gRPC Service Implementation. Each method corresponds to a .proto RPC definition.
    """

    def Init(self, request: idl_pb2.AlgorithmName, context) -> empty_pb2.Empty:
        """
        Initialization RPC: sets the algorithm name in our global store.
        """
        global algo
        algo.name = request.name
        algo.initialized = True
        logging.info(f"[Init] Algorithm set to '{algo.name}'.")
        return empty_pb2.Empty()

    def CalculatePlacement(self, request: idl_pb2.Data, context) -> idl_pb2.Placements:
        try:
            global algo
            
            # ======== RAW REQUEST LOGGING ========
            logging.info("=" * 80)
            logging.info(f"🚀 STARTING PLACEMENT CALCULATION - ALGORITHM: {algo.name}")
            logging.info("=" * 80)
            logging.info(f"📥 RAW REQUEST STRUCTURE:")
            logging.info(f"Request type: {type(request)}")
            
            # Detailed infrastructure logging
            if hasattr(request, "infrastructure") and request.infrastructure:
                logging.info(f"INFRASTRUCTURE: {len(request.infrastructure.nodes)} nodes")
                for i, node in enumerate(request.infrastructure.nodes):
                    logging.info(f"  NODE {i}: name={node.name}")
            else:
                logging.info("No infrastructure in request")
                
            # Detailed workload logging
            if hasattr(request, "workload") and request.workload:
                logging.info(f"WORKLOAD: {len(request.workload.microservices)} microservices")
                for i, ms in enumerate(request.workload.microservices):
                    logging.info(f"  MICROSERVICE {i}: name={ms.name}")

                    # Status information
                    if hasattr(ms, "status"):
                        status_name = MICROSERVICE_STATUS_MAP.get(ms.status, f"Status({ms.status})")
                        
                        # Color coding for different statuses
                        status_color = "\033[0m"  # default
                        if status_name == "RUNNING":
                            status_color = "\033[92m"  # green
                        elif status_name == "TO_DEPLOY":
                            status_color = "\033[91m"  # red
                        elif status_name == "PENDING":
                            status_color = "\033[93m"  # yellow
                            
                        logging.info(f"    - status: {status_color}{status_name}\033[0m")
                    else:
                        logging.info(f"    - status field not found")
                    
                    # Check for duration/deadline fields and their values
                    if hasattr(ms, "duration_hours"):
                        logging.info(f"    - duration: '{ms.duration_hours}' (type: {type(ms.duration_hours).__name__})")
                    else:
                        logging.info(f"    - duration field not found")
                        
                    if hasattr(ms, "deadline_hours"):
                        logging.info(f"    - deadline: '{ms.deadline_hours}' (type: {type(ms.deadline_hours).__name__})")
                    else:
                        logging.info(f"    - deadline field not found")
                    
                    # Log other important attributes
                    if hasattr(ms, "cpu_required") and ms.cpu_required:
                        logging.info(f"    - cpu_required: {ms.cpu_required.value}")
                    if hasattr(ms, "mem_required") and ms.mem_required:
                        logging.info(f"    - mem_required: {ms.mem_required.value}")
            else:
                logging.info("No workload in request")
            
            logging.info("=" * 80)
            
            # ======== INITIALIZATION ========
            if not algo.initialized:
                msg = "Algorithm not initialized before calling CalculatePlacement."
                logging.error(f"❌ {msg}")
                context.set_details(msg)
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                return idl_pb2.Placements()

            # ======== INFRASTRUCTURE PARSING ========
            logging.info("-" * 60)
            logging.info(f"📊 INFRASTRUCTURE PARSING")
            start_time = time.time()  # Record algorithm start time
            flavours = parse_infrastructure(request.infrastructure)
            logging.info(f"  ▶ Processed {len(flavours)} nodes in {(time.time() - start_time):.2f}s")
            
            # Check if carbon intensity data is available
            regions = set()
            for flv in flavours:
                if hasattr(flv, 'region'):
                    regions.add(flv.region)
            logging.info(f"  ▶ Node regions: {', '.join(regions)}")

            # ======== RESOURCE INITIALIZATION ========
            logging.info("-" * 60)
            logging.info(f"🧮 INITIALIZING RESOURCE TRACKING")

            global persistent_state
            if not persistent_state.initialized:
                logging.info("🔄 Initializing persistent resource tracking")
                persistent_state.initialize(flavours)
            else:
                logging.info("📊 Using persistent resource tracking from previous calls")

            # Get the current resource state
            leftover_cpu, leftover_ram = persistent_state.get_resources()
            max_time_slots = persistent_state.max_time_slots
            logging.info(f"  ▶ Using resources for {len(flavours)} nodes × {max_time_slots} timeslots")

            # ======== WORKLOAD PROCESSING ========
            logging.info("-" * 60)
            logging.info(f"🔍 PROCESSING {len(request.workload.microservices)} MICROSERVICES")
            out_placements = idl_pb2.Placements()
            
            # Summary counters
            placements_success = 0
            placements_failed = 0
            placements_skipped = 0
            total_emissions = 0.0

            for i, ms in enumerate(request.workload.microservices):
                ms_start_time = time.time()
                logging.info(f"  ➡️ ({i+1}/{len(request.workload.microservices)}) Processing: {ms.name}")
                    
                if hasattr(ms, "status"):
                    # Process only TO_DEPLOY status
                    if ms.status != 4:  # Only process TO_DEPLOY (4)
                        status_name = MICROSERVICE_STATUS_MAP.get(ms.status, f"Status({ms.status})")
                        logging.info(f"    ⏩ Skipping {ms.name} with status {status_name} - only handling TO_DEPLOY")
                        placement = self._build_fallback_placement(ms.name, f"SKIPPED_{status_name}")
                        out_placements.placements.append(placement)
                        placements_skipped += 1
                        continue

                pod = parse_microservice(ms)
                hours_until_deadline = (pod.deadline - datetime.now()).total_seconds() / 3600
                scheduling_window = max(0, hours_until_deadline - pod.duration)
                
                if hours_until_deadline <= 0:
                    logging.warning(f"    ⚠️  Expired deadline for {ms.name}")
                    placement = self._build_fallback_placement(ms.name, "EXPIRED_DEADLINE")
                    out_placements.placements.append(placement)
                    placements_failed += 1
                    continue

                # Build timeslots for this pod
                timeslots = build_timeslots(hours_until_deadline)
                
                # Enhanced logging to emphasize scheduling flexibility
                logging.info(f"    ⏰ Pod {pod.id}: duration={pod.duration}h, deadline in {hours_until_deadline:.1f}h")
                logging.info(f"    🔄 Scheduling window: {scheduling_window:.1f}h ({len(timeslots)} potential timeslots)")
                
                if scheduling_window <= 0:
                    logging.warning(f"    ⚠️  No scheduling flexibility for {ms.name} - immediate start required")
                elif scheduling_window < 2:
                    logging.info(f"    ℹ️  Limited scheduling window for {ms.name}")
                else:
                    logging.info(f"    ✨ Good scheduling flexibility for {ms.name} - can optimize for carbon")

                # Find optimal placement
                best_node, best_slot, minimal_emissions = find_best_node_and_timeslot(
                    pod, flavours, timeslots,
                    leftover_cpu, leftover_ram
                )

                # Build placement result
                if best_node and best_slot:
                    # Update resource tracking for the ENTIRE DURATION
                    start_slot_id = best_slot.id
                    duration_slots = int(pod.duration)  # Convert hours to slots
                    
                    # Update both the local tracking and persistent state
                    for slot_offset in range(duration_slots):
                        current_slot = start_slot_id + slot_offset
                        if current_slot < max_time_slots:  # Make sure we don't go out of bounds
                            leftover_cpu[best_node.id][current_slot] -= pod.cpuRequest
                            leftover_ram[best_node.id][current_slot] -= pod.ramRequest
                    
                    # Update persistent state
                    persistent_state.update_resources(
                        best_node.id, start_slot_id, duration_slots, 
                        pod.cpuRequest, pod.ramRequest
                    )
                    persistent_state.record_placement(
                        ms.name, best_node.id, best_slot.getStart(), pod.duration
                    )
                    
                    # Log the reservation
                    logging.info(f"    🔒 Reserved resources for {pod.id} on {best_node.id} for slots {start_slot_id} to {start_slot_id + duration_slots - 1}")
                    
                    # Calculate when this workload will start and end
                    workload_start_time = best_slot.getStart()
                    end_time = workload_start_time + timedelta(hours=pod.duration)
                    
                    placement = self._build_success_placement(ms.name, best_node, best_slot, minimal_emissions, flavours)
                    total_emissions += minimal_emissions
                    placements_success += 1
                    
                    logging.info(f"    ✅ Placed on {best_node.id} at {workload_start_time.strftime('%Y-%m-%d %H:%M')}")
                    logging.info(f"       Duration: {pod.duration}h, Finishes: {end_time.strftime('%Y-%m-%d %H:%M')}")
                    logging.info(f"       Emissions: {minimal_emissions:.2f}kgCO2e, Resources: CPU={pod.cpuRequest:.2f}/{best_node.totalCpu:.2f}, " +
                                f"RAM={pod.ramRequest:.0f}/{best_node.totalRam:.0f}MB")
                else:
                    placement = self._build_fallback_placement(ms.name, "NONE_FOUND")
                    placements_failed += 1
                    logging.warning(f"    ❌ No feasible placement found for {ms.name}")

                out_placements.placements.append(placement)
                logging.info(f"    🕒 Processing time: {(time.time() - ms_start_time):.3f}s")

            # ======== SUMMARY ========
            """logging.info("-" * 60)
            logging.info(f"📊 CURRENT NODE UTILIZATION")
            current_time = datetime.now().hour
            current_ts = current_time % 24  # Map to timeslot ID

            # Show utilization for each node at current timeslot
            for flv in flavours:
                node_id = flv.id
                # Calculate utilization percentages
                cpu_used = flv.totalCpu - leftover_cpu[node_id][current_ts]
                ram_used = flv.totalRam - leftover_ram[node_id][current_ts]
                cpu_percent = (cpu_used / flv.totalCpu) * 100 if flv.totalCpu > 0 else 0
                ram_percent = (ram_used / flv.totalRam) * 100 if flv.totalRam > 0 else 0
                
                # Format a nice utilization bar
                cpu_bar = "█" * int(cpu_percent / 10) + "░" * (10 - int(cpu_percent / 10))
                ram_bar = "█" * int(ram_percent / 10) + "░" * (10 - int(ram_percent / 10))
                
                logging.info(f"  Node {node_id}:")
                logging.info(f"    CPU: {cpu_bar} {cpu_used:.2f}/{flv.totalCpu:.2f} ({cpu_percent:.1f}%)")
                logging.info(f"    RAM: {ram_bar} {ram_used:.0f}/{flv.totalRam:.0f}MB ({ram_percent:.1f}%)")
            """
            logging.info("=" * 60)
            logging.info(f"📋 PLACEMENT SUMMARY")
            logging.info(f"  ▶ Total microservices: {len(request.workload.microservices)}")
            logging.info(f"  ▶ Successfully placed: {placements_success}")
            logging.info(f"  ▶ Failed to place: {placements_failed}")
            logging.info(f"  ▶ Skipped (non-TO_DEPLOY): {placements_skipped}") 
            logging.info(f"  ▶ Total carbon footprint: {total_emissions:.2f}kgCO2e")
            logging.info(f"  ▶ Total execution time: {(time.time() - start_time):.3f}s")
            logging.info("=" * 80)

            print_resource_utilization_report(flavours, leftover_cpu, leftover_ram, max_time_slots, hours_to_show=24)
            
            return out_placements

        except Exception as e:
            logging.exception(f"❌ Uncaught exception: {e}")
            context.set_details(f"Server error: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return idl_pb2.Placements()


    ############################
    # Private helper methods
    ############################

    def _build_success_placement(
        self, microservice_name: str,
        best_node: CarbonAwareFlavour,
        best_slot: CarbonAwareTimeslot,
        emissions: float,
        all_flavours: list[CarbonAwareFlavour]  # Add all flavours parameter
    ) -> idl_pb2.Placement:
        """
        Build a placement proto for a successful scheduling decision.
        Include all nodes with scores (best node gets meaningful score, others get 0).
        """
        placement = idl_pb2.Placement()
        placement.microservice_name = microservice_name
        replica_score = idl_pb2.ReplicaScores()

        # Scale down from 100000 to 100 for the best node's score
        best_score = int(100 - min(emissions, 100))  # Cap at 100 to ensure non-negative score

        # Add a score entry for every node
        for flavour in all_flavours:
            score_msg = idl_pb2.Score()
            score_msg.node = flavour.id
            
            # Only the best node gets a non-zero score
            if flavour.id == best_node.id:
                score_msg.score = best_score
            else:
                score_msg.score = 0  # All other nodes get zero score
                
            replica_score.scores.append(score_msg)

        # Convert best_slot's start to epoch time
        epoch_time = int(best_slot.getStart().timestamp())
        placement.time_to_schedule = epoch_time

        placement.replica_scores.append(replica_score)

        logging.info(
            f"[_build_success_placement] microservice={microservice_name}, best_node={best_node.id}, "
            f"timeslot={best_slot.id}, best_score={best_score}, nodes_scored={len(all_flavours)}"
        )
        return placement
    

    def _build_fallback_placement(self, microservice_name: str, reason: str) -> idl_pb2.Placement:
        """
        Build a fallback placement proto if no feasible allocation is found or the
        request is invalid.
        """
        logging.warning(
            f"[_build_fallback_placement] microservice={microservice_name}, reason={reason}"
        )

        placement = idl_pb2.Placement()
        placement.microservice_name = microservice_name
        replica_score = idl_pb2.ReplicaScores()

        fallback_score = idl_pb2.Score()
        fallback_score.node = reason
        fallback_score.score = 0  # 0 indicates no preference
        replica_score.scores.append(fallback_score)

        placement.time_to_schedule = int(time.time())  # "Now"
        placement.replica_scores.append(replica_score)
        return placement


#####################################
# Server Bootstrap
#####################################

def serve() -> None:
    """
    Creates and runs the gRPC server on port 50051, registering the PlacementAlgorithm servicer.
    Implements graceful shutdown handling.
    """
    port = '50051'
    shutdown_in_progress = False  # Add this flag
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    idl_pb2_grpc.add_PlacementAlgorithmServicer_to_server(PlacementAlgorithm(), server)
    server.add_insecure_port('[::]:' + port)

    # Define graceful shutdown handler
    def graceful_shutdown(sig, frame):
        nonlocal shutdown_in_progress
        if shutdown_in_progress:
            return  # Skip if shutdown already in progress
        
        shutdown_in_progress = True
        logging.info("⏳ Received shutdown signal, stopping server gracefully...")
        
        # Give ongoing requests time to complete
        threading.Thread(target=server.stop, args=(5,)).start()  # 5 second timeout
        
        logging.info("👋 Server shutdown initiated")

    # Register signal handlers
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)
    
    server.start()
    logging.info(f"🚀 Server started, listening on port {port}")
    
    try:
        # This is a blocking call until server is terminated
        server.wait_for_termination()
    except KeyboardInterrupt:
        # Only log if not already shutting down
        if not shutdown_in_progress:
            logging.info("Keyboard interrupt received")
            graceful_shutdown(signal.SIGINT, None)
    finally:
        logging.info("Server shutdown complete")


def main() -> None:
    """
    Main entrypoint. Sets up logging based on command-line args, 
    sets signal handler, and runs the server.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Carbon-aware scheduling server.')
    parser.add_argument(
        '--loglevel', 
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level (default: INFO)'
    )
    args = parser.parse_args()
    
    # Configure logging with specified level
    log_level = getattr(logging, args.loglevel)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logging.info(f"Starting carbon-aware server with log level: {args.loglevel}")
    serve()


if __name__ == "__main__":
    main()
