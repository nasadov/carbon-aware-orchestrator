"""
Carbon-aware scheduling algorithm factory.
"""
import logging
from typing import Optional

from carbon_aware.algorithms.base import SchedulingAlgorithm
from carbon_aware.algorithms.heuristic import HeuristicAlgorithm
from carbon_aware.algorithms.global_optimal import GlobalOptimalAlgorithm
from carbon_aware.algorithms.vanilla import VanillaAlgorithm

# Store reference to precomputed global optimal instance
_precomputed_global_optimal = None

def set_precomputed_global_optimal(instance: GlobalOptimalAlgorithm):
    """Set the precomputed global optimal algorithm instance to be reused."""
    global _precomputed_global_optimal
    _precomputed_global_optimal = instance
    logging.info("✅ Precomputed global optimal algorithm instance registered for reuse")

def get_algorithm(name: str) -> SchedulingAlgorithm:
    """
    Get algorithm implementation by name.
    
    Args:
        name: Algorithm name (heuristic or global-optimal)
        
    Returns:
        Algorithm implementation
        
    Raises:
        ValueError: If algorithm name is unknown
    """
    global _precomputed_global_optimal
    
    if name == "heuristic":
        return HeuristicAlgorithm()
    elif name == "global-optimal":  # Updated name
        # Return the precomputed instance if available, otherwise create new
        if _precomputed_global_optimal is not None:
            logging.info("♻️ Reusing precomputed global optimal algorithm instance")
            return _precomputed_global_optimal
        else:
            logging.info("🔧 Creating new global optimal algorithm instance (no precomputed available)")
            return GlobalOptimalAlgorithm()
    elif name == "vanilla":
        return VanillaAlgorithm()
    else:
        raise ValueError(f"Unknown algorithm: {name}. Available: heuristic, global-optimal, vanilla")