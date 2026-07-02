from datetime import datetime, timedelta
from typing import Dict, Optional

class CarbonAwarePod:
    """Represents a workload pod with carbon-aware scheduling metadata"""
    
    def __init__(self, id: str, deadline_hours: float, duration: float,
                 powerConsumption: float, cpuRequest: float,
                 ramRequest: float, storageRequest: int,
                 reference_time: datetime = None,
                 gpuRequest: float = 0.0, tier: Optional[str] = None,
                 cpu_util_ratio: float = 1.0, gpu_util_ratio: float = 1.0) -> None:
        self.id = id
        self.deadline = self._processDeadline(deadline_hours, reference_time)
        self.deadline_hours = deadline_hours  # Store deadline hours for relative calculation
        self.duration = duration
        self.powerConsumption = powerConsumption
        self.cpuRequest = cpuRequest
        self.ramRequest = ramRequest
        self.storageRequest = storageRequest
        self.gpuRequest = gpuRequest  # GPU count requested (extended resource, nvidia.com/gpu); 0 = CPU/RAM-only pod
        self.tier = tier              # explicit firm/flexible tier (e.g. from trace job type); None -> slack-based
        # Measured utilization as a fraction of the request/allocation (from a trace's sensor
        # data, e.g. Alibaba gpu_wrk_util/cpu_usage). Scales DYNAMIC power only; allocation/idle
        # and embodied stay request-based. Default 1.0 -> request-driven (bit-identical to legacy).
        self.cpu_util_ratio = cpu_util_ratio
        self.gpu_util_ratio = gpu_util_ratio
        self.earliest_timeslot = 0  # Default: can be scheduled from timeslot 0
        self.deadline_slot = None   # Default: no specific deadline slot (calculated later if needed)

    def _processDeadline(self, deadline_hours: float, reference_time: datetime = None) -> datetime:
        now = datetime.now() if reference_time is None else reference_time
        delta = timedelta(hours=deadline_hours)
        return now + delta
    
    def calculate_deadline_slot(self):
        """
        Calculate deadline_slot as earliest_timeslot + deadline_hours.
        This ensures deadlines are relative to the pod's origin timeslot rather than from timeslot 0.
        """
        if hasattr(self, 'earliest_timeslot') and hasattr(self, 'deadline_hours'):
            self.deadline_slot = self.earliest_timeslot + self.deadline_hours
        else:
            self.deadline_slot = None

class EnvironmentalFlavor:
    """Represents a schedulable node with environmental metadata."""

    def __init__(
        self,
        id: str,
        embodiedCarbon: float,
        lifetime: float,
        totalCpu: float,
        totalRam: float,
        totalStorage: float,
        forecast: Dict[int, float],
        power: Dict[str, float] = None,
        totalGpu: float = 0,
        gpu_power_w: float = 0.0,
        gpu_idle_w: float = 0.0,  # per-GPU idle/baseline power (W); a held GPU draws this even at 0% util (V100~50). Default 0 -> legacy peak-only model.
        gpu_type: str = "",
        gpu_embodied_carbon: float = 0.0,  # per-GPU embodied carbon (gCO2e); NVIDIA H100 PCF + ACT/LLMCarbon
        gpu_embodied_water: float = 0.0,   # per-GPU embodied water (L); fab disclosures (low/base/high)
        region: str = "",
        country: str = "",
        pue: float = 1.0,
        pue_by_slot: Optional[Dict[int, float]] = None,
        embodiedWater: float = 0.0,
        wue_by_slot: Optional[Dict[int, float]] = None,
        ewif_by_slot: Optional[Dict[int, float]] = None,
        water_scarcity_direct_cf_by_slot: Optional[Dict[int, float]] = None,
        water_scarcity_indirect_cf_by_slot: Optional[Dict[int, float]] = None,
        water_scarcity_direct_cf: float = 1.0,
        water_scarcity_indirect_cf: float = 1.0,
        water_scarcity_embodied_cf: float = 1.0,
        water_criticality: float = 1.0,
        radiation_by_slot: Optional[Dict[int, float]] = None,
        radiation_intensity: float = 0.0,
    ):
        self.id = id
        self.embodiedCarbon = embodiedCarbon
        self.lifetime = lifetime
        self.totalCpu = totalCpu
        self.totalRam = totalRam
        self.totalStorage = totalStorage
        self.totalGpu = totalGpu            # GPU count on the node (extended resource, fractional-capable); 0 = no GPUs
        self.gpu_power_w = gpu_power_w       # per-GPU peak load power (W), e.g. V100~300, A100~400
        self.gpu_idle_w = gpu_idle_w        # per-GPU idle power (W); held-but-underutilized GPUs still draw this
        self.gpu_type = gpu_type
        self.gpu_embodied_carbon = gpu_embodied_carbon  # per-GPU embodied carbon (gCO2e)
        self.gpu_embodied_water = gpu_embodied_water    # per-GPU embodied water (L)
        self.forecast = forecast
        self.power = power or {"idle": 100.0, "active": 200.0, "max": 400.0}

        self.region = region
        self.country = country
        self.pue = pue
        self.pue_by_slot = pue_by_slot or {}
        self.embodiedWater = embodiedWater
        self.wue_by_slot = wue_by_slot or {}
        self.ewif_by_slot = ewif_by_slot or {}
        self.water_scarcity_direct_cf_by_slot = water_scarcity_direct_cf_by_slot or {}
        self.water_scarcity_indirect_cf_by_slot = water_scarcity_indirect_cf_by_slot or {}
        self.water_scarcity_direct_cf = water_scarcity_direct_cf
        self.water_scarcity_indirect_cf = water_scarcity_indirect_cf
        self.water_scarcity_embodied_cf = water_scarcity_embodied_cf
        self.water_criticality = water_criticality
        # Operational ionising-radiation characterization (kBq U-235 eq / kWh of facility
        # electricity), region-resolved (mix-average) like AWARE; per-slot dict optional.
        self.radiation_by_slot = radiation_by_slot or {}
        self.radiation_intensity = radiation_intensity


# Backward-compatible alias while the codebase migrates from the old name.
CarbonAwareFlavour = EnvironmentalFlavor

class CarbonAwareTimeslot:
    """Represents a scheduling time slot"""
    
    def __init__(self, id: int, startYear: int = None, startMonth: int = None,
                 startDay: int = None, startHour: int = None, length: int = 1,
                 start_time: datetime = None) -> None:
        self.id = id
        
        if start_time is not None:
            self.start = start_time
        else:
            self.start = datetime(startYear, startMonth, startDay, startHour)
            
        self.length = timedelta(hours=length)

    def getEnd(self) -> datetime:
        return self.start + self.length

    def getStart(self) -> datetime:
        return self.start
