"""
Utility functions for carbon-aware scheduling.
"""
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import decimal

# Import kubernetes utils for resource parsing
from kubernetes import utils as k8sutils

# Import models (should be defined in models.py)
from carbon_aware.models import CarbonAwarePod, CarbonAwareFlavour, CarbonAwareTimeslot

# Define constants for microservice status
MICROSERVICE_STATUS_MAP = {
    0: "MICROSERVICESTATUS_UNSPECIFIED",
    1: "RUNNING",
    2: "PENDING", 
    3: "TO_SCHEDULE",
    4: "TO_DEPLOY"
}

#####################################
# Carbon Intensity & Power Utilities
#####################################

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
    # Use the formula: idle + (max-active) * cpu_usage_ratio
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


def get_node_region(node) -> str:
    """
    Determine the region for a node based on its metadata,
    specifically focusing on the topology.kubernetes.io/region label.
    
    Args:
        node: Node object with metadata
        
    Returns:
        Region code as uppercase string (e.g., "DE", "FR")
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


#####################################
# Scheduling Utilities
#####################################

def is_timeslot_valid(ts: CarbonAwareTimeslot, pod: CarbonAwarePod) -> bool:
    """
    Determines whether the timeslot is valid for the given pod.
    Valid if:
      - The current time hasn't passed the timeslot's end.
      - The current time hasn't passed the start of the timeslot (no scheduling in the past).
      - The timeslot starts before the pod's deadline.
    """
    now = datetime.now()
    now_date_hour = now.replace(minute=0, second=0, microsecond=0)
    ts_date_hour = ts.getStart().replace(minute=0, second=0, microsecond=0)
    not_past_end = now <= ts.getEnd()
    not_in_past = ts_date_hour >= now_date_hour
    before_deadline = ts.getStart() <= pod.deadline
    valid = not_past_end and not_in_past and before_deadline
    
    if not valid:
        if not not_past_end:
            logging.debug(f"[is_timeslot_valid] Timeslot {ts.id} invalid: end time {ts.getEnd()} already passed current time {now}")
        if not not_in_past:
            logging.debug(f"[is_timeslot_valid] Timeslot {ts.id} invalid: start hour {ts_date_hour} is in the past (current hour: {now_date_hour})")
        if not before_deadline:
            logging.debug(f"[is_timeslot_valid] Timeslot {ts.id} invalid: starts at {ts.getStart()} which is after deadline {pod.deadline}")

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


#####################################
# Parsing Utilities
#####################################

def parse_infrastructure(infra) -> list[CarbonAwareFlavour]:
    """
    Parse infrastructure data to create CarbonAwareFlavour objects.
    
    Args:
        infra: Infrastructure protobuf object containing node information
        
    Returns:
        List of CarbonAwareFlavour objects
    """
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


def parse_microservice(ms) -> CarbonAwarePod:
    """
    Parse microservice data to create a CarbonAwarePod.
    
    Args:
        ms: Microservice protobuf object
        
    Returns:
        CarbonAwarePod object
    """
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


#####################################
# Reporting Utilities
#####################################

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
# Performance Logging
#####################################

# Performance logging for research paper metrics
class PerformanceLogger:
    """
    Logger for tracking algorithm performance metrics across multiple calls.
    Creates a separate CSV log file with detailed performance data suitable for
    generating research paper figures.
    """
    def __init__(self, algorithm_name, log_dir="performance_logs"):
        import os
        import datetime
        
        self.algorithm_name = algorithm_name
        self.log_dir = log_dir
        
        # Create the log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a unique filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{algorithm_name}_{timestamp}.csv")
        
        # Initialize the log file with headers
        with open(self.log_file, 'w') as f:
            f.write("timestamp,call_id,execution_time_ms,algorithm,pods_total,pods_processed,pods_placed,pods_failed,"
                    "pods_skipped,total_emissions_kg,per_pod_emissions_kg,avg_placement_time_ms,total_flavours,"
                    "total_timeslots,scheduling_options,cpu_utilization_pct,memory_utilization_pct,"
                    "regions,algorithm_iterations,algorithm_steps,microservices\n")
        
        self.call_counter = 0
        logging.info(f"Performance logging enabled. Writing to {self.log_file}")
    
    def log_placement_call(self, execution_time, algorithm_name, pods_total, pods_processed, pods_placed, 
                          pods_failed, pods_skipped, total_emissions, algorithm_metrics, 
                          flavours, timeslots_count, cpu_util=None, memory_util=None, microservices=None):
        """
        Log performance metrics for a single CalculatePlacement call.
        
        Args:
            execution_time: Total execution time for the call in seconds
            algorithm_name: Name of the algorithm used
            pods_total: Total number of pods in the request
            pods_processed: Number of pods actually processed (excluding skipped)
            pods_placed: Number of pods successfully placed
            pods_failed: Number of pods that failed placement
            pods_skipped: Number of pods skipped
            total_emissions: Total carbon emissions in kgCO2e
            algorithm_metrics: Dict with algorithm-specific metrics like iterations, steps
            flavours: List of flavours (nodes)
            timeslots_count: Number of timeslots considered
            cpu_util: CPU utilization percentage
            memory_util: Memory utilization percentage
            microservices: List of microservice names (for tracking what was scheduled)
        """
        import datetime
        
        # Calculate additional metrics
        total_flavours = len(flavours)
        
        # Calculate scheduling options (number of possible placement combinations)
        scheduling_options = total_flavours * timeslots_count if pods_processed > 0 else 0
        
        # Extract algorithm-specific metrics
        iterations = algorithm_metrics.get('iterations', 0)
        steps = algorithm_metrics.get('steps', 0)
        
        # Extract regions information
        regions = set()
        for flv in flavours:
            if hasattr(flv, 'region'):
                regions.add(flv.region)
        regions_str = "|".join(regions)
        
        # Calculate utilization if not provided
        if cpu_util is None or memory_util is None:
            cpu_util = -1
            memory_util = -1
        
        # Convert times to milliseconds for better readability
        execution_time_ms = execution_time * 1000
        avg_placement_time_ms = 0
        if pods_processed > 0:
            avg_placement_time_ms = execution_time_ms / pods_processed
        
        # Calculate per-pod emissions
        per_pod_emissions = 0
        if pods_placed > 0:
            per_pod_emissions = total_emissions / pods_placed
        
        # Get a compact representation of microservices
        if microservices is None:
            microservices_str = "-"
        else:
            # Limit to avoid huge CSV fields
            if len(microservices) > 5:
                microservices_str = f"{len(microservices)}_pods"
            else:
                microservices_str = "|".join(microservices)
        
        self.call_counter += 1
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp},{self.call_counter},{execution_time_ms:.2f},{algorithm_name},{pods_total},"
                    f"{pods_processed},{pods_placed},{pods_failed},{pods_skipped},{total_emissions:.6f},"
                    f"{per_pod_emissions:.6f},{avg_placement_time_ms:.2f},{total_flavours},{timeslots_count},"
                    f"{scheduling_options},{cpu_util:.2f},{memory_util:.2f},{regions_str},{iterations},"
                    f"{steps},{microservices_str}\n")
        
        logging.info(f"📊 Performance metrics logged to {self.log_file} (call #{self.call_counter})")
        
        # Return a summary for console output
        return {
            "execution_time_ms": execution_time_ms,
            "pods_placed": pods_placed,
            "pods_total": pods_total,
            "avg_placement_time_ms": avg_placement_time_ms
        }