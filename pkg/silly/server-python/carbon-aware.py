from concurrent import futures
import json
import logging
import signal
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import grpc
import idl_pb2
import idl_pb2_grpc
from google.protobuf import empty_pb2
from kubernetes import utils as k8sutils


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
    """
    Represents the microservice to schedule, including:
    - a computed 'deadline' (datetime) from 'deadline_hours'
    - resource requests (CPU, RAM, storage)
    - powerConsumption, which can be set or computed during scheduling
    """
    def __init__(self, id: str, deadline_hours: float, duration: int,
                 powerConsumption: float, cpuRequest: float,
                 ramRequest: float, storageRequest: int) -> None:
        self.id = id
        self.deadline = self._processDeadline(deadline_hours)
        self.duration = duration
        self.powerConsumption = powerConsumption
        self.cpuRequest = cpuRequest
        self.ramRequest = ramRequest
        self.storageRequest = storageRequest

    def _processDeadline(self, deadline_hours: float) -> datetime:
        """
        Convert a float-based 'deadline_hours' into a datetime.
        """
        now = datetime.now()
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
        return node.region

    # 2. Check for region in labels if available
    if hasattr(node, "labels"):
        for label in node.labels:
            if label.key.lower() in ["region", "topology.kubernetes.io/region"]:
                return label.value
    
    # 3. Extract from node name - common patterns
    node_name_lower = node.name.lower()
    
    # Check for common region codes in the node name
    for region_code in ["de", "fr", "es", "it-no"]:
        if region_code in node_name_lower:
            return region_code.upper()
    
    # 4. Check parent region if the node is part of a region structure
    if hasattr(node, "parent") and hasattr(node.parent, "name"):
        return node.parent.name
        
    # 5. Fallback to default region or extract from node zone if available
    if hasattr(node, "zone"):
        return node.zone.upper()
        
    # Last resort: map to a default region based on node name
    logging.warning(f"Could not determine region for node {node.name}, mapping to default")
    return "DE"  # Default fallback region


def get_node_hardware_metadata(node) -> tuple[float, float, Dict[str, float]]:
    """
    Extract hardware metadata related to carbon emissions and power consumption from node annotations.
    
    Args:
        node: Node object with metadata
        
    Returns:
        Tuple of (embodied_emissions, lifetime_years, power_consumption)
        where power_consumption is a dict with 'idle', 'active', and 'max' values in watts
    """
    # Default values if metadata not found
    default_embodied_carbon = 50000.0
    default_lifetime = 4.0
    default_power = {
        "idle": 100.0,   # Default to server values
        "active": 200.0,
        "max": 400.0
    }
    
    embodied_carbon = default_embodied_carbon
    lifetime = default_lifetime
    power = default_power.copy()
    
    # Check for hardware carbon metadata in annotations
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
    
    # Check for hardware subcategory label if no direct annotations
    if (embodied_carbon == default_embodied_carbon or lifetime == default_lifetime or power == default_power) and hasattr(node, "labels") and node.labels:
        subcategory = None
        for label in node.labels:
            if label.key == "hardware.carbon/subcategory":
                subcategory = label.value
                logging.debug(f"Found hardware subcategory from label: {subcategory}")
                break
        
        # If subcategory is found but no direct annotations, use subcategory-based defaults
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
    
    nodeList = []
    for region in infra.regions:
        nodeList.extend(region.nodes)

    flavours = []
    for node in nodeList:
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


def parse_microservice(ms: idl_pb2.Microservice) -> CarbonAwarePod:
    """
    Convert gRPC Microservice to a CarbonAwarePod object.
    
    Extract duration from annotations or labels if present.
    """
    # CPU request is specified as a string (e.g. "100m") → parse to a Decimal
    cpu_req = k8sutils.parse_quantity(ms.resources.requests["cpu"])
    float_cpu_req = float(cpu_req)  # to float

    # RAM request is specified as a string (e.g. "1Gi")
    ram_req = k8sutils.parse_quantity(ms.resources.requests["memory"])
    float_ram_req = float(ram_req) / (1024 * 1024)  # bytes → MB

    # Initialize duration to None to detect if we've found a value
    duration_hours = None
    
    # Try to extract duration from annotations
    if hasattr(ms, "annotations") and ms.annotations:
        for annotation in ms.annotations:
            if annotation.key == "workload.carbon/duration_hours":
                try:
                    duration_hours = float(annotation.value)
                    logging.debug(f"Found duration from annotation: {duration_hours} hours")
                except (ValueError, TypeError):
                    logging.warning(f"Invalid duration value: {annotation.value}")
    
    # If no annotation, try to extract from duration label
    if duration_hours is None and hasattr(ms, "labels") and ms.labels:
        for label in ms.labels:
            if label.key == "duration":
                if label.value.startswith("duration-") and label.value.endswith("h"):
                    try:
                        # Parse from format "duration-Xh"
                        hours_str = label.value.replace("duration-", "").replace("h", "")
                        duration_hours = float(hours_str)
                        logging.debug(f"Extracted duration from label {label.value}: {duration_hours} hours")
                    except (ValueError, IndexError):
                        logging.warning(f"Could not parse duration from label: {label.value}")
                elif label.value.isdigit():
                    # Handle old format where duration label might just be a number
                    try:
                        duration_hours = float(label.value)
                        logging.debug(f"Extracted duration from numeric label: {duration_hours} hours")
                    except (ValueError):
                        logging.warning(f"Could not parse duration from numeric label: {label.value}")
                break
    
    # If still no duration found, use a default
    if duration_hours is None:
        duration_hours = 1.0
        logging.warning(f"No duration specified for {ms.name}, using default of {duration_hours} hours")
    
    # Create the pod with the extracted duration (deadline is set by CarbonAwarePod to now + duration)
    pod = CarbonAwarePod(
        id=ms.name,
        deadline_hours=duration_hours,  # Use duration for deadline since we don't need separate deadline
        duration=duration_hours,
        powerConsumption=0.0,  # will be computed if 0
        cpuRequest=float_cpu_req,
        ramRequest=float_ram_req,
        storageRequest=0
    )
    return pod


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
    leftover_ram: dict[str, dict[int, float]]
) -> tuple[CarbonAwareFlavour | None, CarbonAwareTimeslot | None, float]:
    best_node = None
    best_slot = None
    minimal_emissions = float('inf')

    for ts in timeslots:
        if not is_timeslot_valid(ts, pod):
            logging.debug(f"[find_best_node_and_timeslot] Skipping timeslot={ts.id}, not valid for pod={pod.id}")
            continue

        for flv in flavours:
            # Instead of flv.totalCpu, we read leftover_cpu[flv.id][ts.id]
            # We do the same for leftover RAM.
            feasible = check_node_resource(
                flv, ts, pod,
                leftover_cpu[flv.id][ts.id],
                leftover_ram[flv.id][ts.id]
            )

            if feasible:
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

algo = Algorithm("unknown", False)


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
        global algo
        if not algo.initialized:
            msg = "Algorithm not initialized before calling CalculatePlacement."
            logging.error(f"[CalculatePlacement] {msg}")
            context.set_details(msg)
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            return idl_pb2.Placements()

        logging.info(f"[CalculatePlacement] Using algorithm '{algo.name}'.")

        # 1) Parse the infrastructure into CarbonAwareFlavours
        flavours = parse_infrastructure(request.infrastructure)

        # 1.b) Initialize leftover resources for each (node, timeslot).
        # Let's say we plan a maximum of 24 timeslots.
        leftover_cpu = {}
        leftover_ram = {}
        max_time_slots = 24

        for flv in flavours:
            leftover_cpu[flv.id] = {}
            leftover_ram[flv.id] = {}
            for ts_id in range(max_time_slots):
                # Initially, leftover is the entire node capacity
                leftover_cpu[flv.id][ts_id] = flv.totalCpu
                leftover_ram[flv.id][ts_id] = flv.totalRam

        # 2) Prepare output container
        out_placements = idl_pb2.Placements()

        # 3) Iterate over microservices in the workload
        for ms in request.workload.microservices:
            pod = parse_microservice(ms)
            hours_until_deadline = (pod.deadline - datetime.now()).total_seconds() / 3600
            if hours_until_deadline <= 0:
                # Expired
                placement = self._build_fallback_placement(ms.name, "EXPIRED_DEADLINE")
                out_placements.placements.append(placement)
                continue

            timeslots = build_timeslots(hours_until_deadline)

            # 4) Find the best node & timeslot based on minimal emissions
            best_node, best_slot, minimal_emissions = find_best_node_and_timeslot(
                pod, flavours, timeslots,
                leftover_cpu,  # pass these new dictionaries
                leftover_ram
            )

            # 5) Build the final placement object
            if best_node and best_slot:
                # Once chosen, those resources must be "consumed" so that
                # subsequent pods see reduced leftover.
                leftover_cpu[best_node.id][best_slot.id] -= pod.cpuRequest
                leftover_ram[best_node.id][best_slot.id] -= pod.ramRequest

                placement = self._build_success_placement(ms.name, best_node, best_slot, minimal_emissions)
            else:
                placement = self._build_fallback_placement(ms.name, "NONE_FOUND")

            out_placements.placements.append(placement)

        logging.debug(f"[CalculatePlacement] Final placements: {out_placements}")
        return out_placements


    ############################
    # Private helper methods
    ############################

    def _build_success_placement(
        self, microservice_name: str,
        best_node: CarbonAwareFlavour,
        best_slot: CarbonAwareTimeslot,
        emissions: float
    ) -> idl_pb2.Placement:
        """
        Build a placement proto for a successful scheduling decision.
        """
        placement = idl_pb2.Placement()
        placement.microservice_name = microservice_name
        replica_score = idl_pb2.ReplicaScores()

        # Example: store an integer "score" as (100000 - emissions) for demonstration
        score_msg = idl_pb2.Score()
        score_msg.node = best_node.id
        score_msg.score = int(100000 - emissions)
        replica_score.scores.append(score_msg)

        # Convert best_slot's start to epoch time
        epoch_time = int(best_slot.getStart().timestamp())
        placement.time_to_schedule = epoch_time

        placement.replica_scores.append(replica_score)

        logging.info(
            f"[_build_success_placement] microservice={microservice_name}, node={best_node.id}, "
            f"timeslot={best_slot.id}, epoch_time={epoch_time}, totalEmissions={emissions:.2f}"
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
    """
    port = '50051'
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    idl_pb2_grpc.add_PlacementAlgorithmServicer_to_server(PlacementAlgorithm(), server)
    server.add_insecure_port('[::]:' + port)

    server.start()
    logging.info(f"[serve] Server started, listening on {port}.")
    server.wait_for_termination()


def handler(signum, frame) -> None:
    """
    Signal handler to gracefully shut down on CTRL+C.
    """
    logging.info("[handler] Caught CTRL+C, shutting down.")
    exit(0)


def main() -> None:
    """
    Main entrypoint. Sets logging level to DEBUG, sets signal handler, and runs the server.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    signal.signal(signal.SIGINT, handler)
    serve()


if __name__ == "__main__":
    main()
