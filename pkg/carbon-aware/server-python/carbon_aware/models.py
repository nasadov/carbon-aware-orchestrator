from datetime import datetime, timedelta
from typing import Dict, Optional

class CarbonAwarePod:
    """Represents a workload pod with carbon-aware scheduling metadata"""
    
    def __init__(self, id: str, deadline_hours: float, duration: float,
                 powerConsumption: float, cpuRequest: float,
                 ramRequest: float, storageRequest: int,
                 reference_time: datetime = None) -> None:
        self.id = id
        self.deadline = self._processDeadline(deadline_hours, reference_time)
        self.duration = duration
        self.powerConsumption = powerConsumption
        self.cpuRequest = cpuRequest
        self.ramRequest = ramRequest
        self.storageRequest = storageRequest
        self.earliest_timeslot = 0  # Default: can be scheduled from timeslot 0

    def _processDeadline(self, deadline_hours: float, reference_time: datetime = None) -> datetime:
        now = datetime.now() if reference_time is None else reference_time
        delta = timedelta(hours=deadline_hours)
        return now + delta

class CarbonAwareFlavour:
    """Represents a node with carbon-aware characteristics"""
    
    def __init__(self, id: str, embodiedCarbon: float, lifetime: float, 
                 totalCpu: float, totalRam: float, totalStorage: float, 
                 forecast: Dict[int, float], power: Dict[str, float] = None):
        self.id = id
        self.embodiedCarbon = embodiedCarbon
        self.lifetime = lifetime
        self.totalCpu = totalCpu
        self.totalRam = totalRam
        self.totalStorage = totalStorage
        self.forecast = forecast
        self.power = power or {"idle": 100.0, "active": 200.0, "max": 400.0}

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