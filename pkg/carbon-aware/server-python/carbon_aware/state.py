"""
State management for the Carbon-Aware Orchestrator.
"""
import logging
import threading
from datetime import datetime
from typing import Dict, Tuple


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
    
    def get_resources(self) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, float]]]:
        """Get current resource state."""
        with self.state_lock:
            # Return deep copies to prevent external modifications
            cpu_copy = {}
            ram_copy = {}
            for node_id in self.leftover_cpu:
                cpu_copy[node_id] = dict(self.leftover_cpu[node_id])
                ram_copy[node_id] = dict(self.leftover_ram[node_id])
            return cpu_copy, ram_copy
    
    def update_resources(self, node_id: str, start_slot: int, duration: int, 
                         cpu_request: float, ram_request: float) -> None:
        """
        Consume resources for a placement.
        
        Args:
            node_id: ID of the node
            start_slot: Starting timeslot
            duration: Duration in slots
            cpu_request: CPU to reserve
            ram_request: RAM to reserve
        """
        with self.state_lock:
            logging.debug(
                f"[update_resources] node={node_id}, start_slot={start_slot}, duration={duration}, "
                f"cpu={cpu_request:.2f}, ram={ram_request:.2f}MB"
            )
            
            # Update resources for each slot in the duration
            for offset in range(duration):
                slot_id = start_slot + offset
                if slot_id >= self.max_time_slots:
                    continue  # Skip slots beyond our tracking window
                
                # Update CPU
                if node_id in self.leftover_cpu and slot_id in self.leftover_cpu[node_id]:
                    self.leftover_cpu[node_id][slot_id] -= cpu_request
                    
                    # Ensure we don't go negative due to rounding errors
                    if self.leftover_cpu[node_id][slot_id] < 0:
                        self.leftover_cpu[node_id][slot_id] = 0
                    
                # Update RAM
                if node_id in self.leftover_ram and slot_id in self.leftover_ram[node_id]:
                    self.leftover_ram[node_id][slot_id] -= ram_request
                    
                    # Ensure we don't go negative due to rounding errors
                    if self.leftover_ram[node_id][slot_id] < 0:
                        self.leftover_ram[node_id][slot_id] = 0
    
    def record_placement(self, ms_id: str, node_id: str, start_time: datetime, duration: float) -> None:
        """
        Record a placement for tracking.
        
        Args:
            ms_id: Microservice ID
            node_id: Node ID
            start_time: Start time as datetime
            duration: Duration in hours
        """
        with self.state_lock:
            self.placements[ms_id] = {
                "node_id": node_id,
                "start_time": start_time,
                "duration": duration
            }
    
    def check_resources_available(self, node_id: str, start_slot: int, duration: int, 
                                  cpu_request: float, ram_request: float) -> bool:
        """
        Check if resources would be available for allocation without actually allocating them.
        
        This method is used to check feasibility before attempting atomic allocation.
        
        Args:
            node_id: ID of the node
            start_slot: Starting timeslot
            duration: Duration in slots
            cpu_request: CPU required
            ram_request: RAM required
            
        Returns:
            True if resources would be available, False otherwise
        """
        with self.state_lock:
            # Check if allocation would be feasible for all required slots
            for offset in range(duration):
                slot_id = start_slot + offset
                if slot_id >= self.max_time_slots:
                    return False  # Slot beyond tracking window
                
                # Check CPU availability
                if (node_id not in self.leftover_cpu or 
                    slot_id not in self.leftover_cpu[node_id] or
                    self.leftover_cpu[node_id][slot_id] < cpu_request):
                    return False  # CPU constraint violated
                
                # Check RAM availability
                if (node_id not in self.leftover_ram or 
                    slot_id not in self.leftover_ram[node_id] or
                    self.leftover_ram[node_id][slot_id] < ram_request):
                    return False  # RAM constraint violated
            
            return True  # All constraints satisfied

    def atomic_check_and_allocate(self, node_id: str, start_slot: int, duration: int, 
                                  cpu_request: float, ram_request: float) -> bool:
        """
        Atomically check resource constraints and allocate resources if feasible.
        
        This method prevents race conditions by performing constraint checking
        and resource updates within a single critical section.
        
        Args:
            node_id: ID of the node
            start_slot: Starting timeslot
            duration: Duration in slots
            cpu_request: CPU to reserve
            ram_request: RAM to reserve
            
        Returns:
            True if resources were successfully allocated, False if constraints violated
        """
        with self.state_lock:
            # First, check if allocation is feasible for all required slots
            for offset in range(duration):
                slot_id = start_slot + offset
                if slot_id >= self.max_time_slots:
                    logging.debug(
                        f"[atomic_check_and_allocate] Slot {slot_id} exceeds tracking window for "
                        f"node={node_id}, start_slot={start_slot}, duration={duration}"
                    )
                    return False  # Slot beyond tracking window
                
                # Check CPU availability
                if (node_id not in self.leftover_cpu or 
                    slot_id not in self.leftover_cpu[node_id] or
                    self.leftover_cpu[node_id][slot_id] < cpu_request):
                    
                    available_cpu = self.leftover_cpu.get(node_id, {}).get(slot_id, 0)
                    logging.debug(
                        f"[atomic_check_and_allocate] CPU constraint violation at slot {slot_id}: "
                        f"need {cpu_request:.2f}, available {available_cpu:.2f}"
                    )
                    return False  # CPU constraint violated
                
                # Check RAM availability
                if (node_id not in self.leftover_ram or 
                    slot_id not in self.leftover_ram[node_id] or
                    self.leftover_ram[node_id][slot_id] < ram_request):
                    
                    available_ram = self.leftover_ram.get(node_id, {}).get(slot_id, 0)
                    logging.debug(
                        f"[atomic_check_and_allocate] RAM constraint violation at slot {slot_id}: "
                        f"need {ram_request:.2f}, available {available_ram:.2f}"
                    )
                    return False  # RAM constraint violated
            
            # All constraints satisfied - proceed with allocation
            logging.debug(
                f"[atomic_check_and_allocate] Allocating resources: node={node_id}, "
                f"start_slot={start_slot}, duration={duration}, cpu={cpu_request:.2f}, ram={ram_request:.2f}"
            )
            
            for offset in range(duration):
                slot_id = start_slot + offset
                
                # Update CPU
                self.leftover_cpu[node_id][slot_id] -= cpu_request
                if self.leftover_cpu[node_id][slot_id] < 0:
                    self.leftover_cpu[node_id][slot_id] = 0
                
                # Update RAM
                self.leftover_ram[node_id][slot_id] -= ram_request
                if self.leftover_ram[node_id][slot_id] < 0:
                    self.leftover_ram[node_id][slot_id] = 0
            
            return True  # Successfully allocated