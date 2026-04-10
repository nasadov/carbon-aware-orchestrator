from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from carbon_aware.models import CarbonAwarePod, CarbonAwareTimeslot, EnvironmentalFlavor

class SchedulingAlgorithm(ABC):
    """Base interface for carbon-aware scheduling algorithms"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Algorithm name identifier"""
        pass
    
    @abstractmethod
    def find_placement(
        self,
        pod: CarbonAwarePod,
        flavours: List[EnvironmentalFlavor],
        timeslots: List[CarbonAwareTimeslot],
        leftover_cpu: Dict[str, Dict[int, float]],
        leftover_ram: Dict[str, Dict[int, float]],
        max_time_slots: int = 48
    ) -> Tuple[Optional[EnvironmentalFlavor], Optional[CarbonAwareTimeslot], float]:
        """Find the best placement for a given pod"""
        pass
