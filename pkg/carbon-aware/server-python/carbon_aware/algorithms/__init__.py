"""
Carbon-aware scheduling algorithm factory.
"""
from typing import Optional

from carbon_aware.algorithms.base import SchedulingAlgorithm
from carbon_aware.algorithms.heuristic import HeuristicAlgorithm
from carbon_aware.algorithms.optimal import OptimalAlgorithm 
from carbon_aware.algorithms.global_optimal import GlobalOptimalAlgorithm


def get_algorithm(name: str) -> SchedulingAlgorithm:
    """
    Get algorithm implementation by name.
    
    Args:
        name: Algorithm name (heuristic, optimal, or global-optimal)
        
    Returns:
        Algorithm implementation
        
    Raises:
        ValueError: If algorithm name is unknown
    """
    if name == "heuristic":
        return HeuristicAlgorithm()
    elif name == "optimal":
        return OptimalAlgorithm()
    elif name == "global-optimal":  # Updated name
        return GlobalOptimalAlgorithm()
    else:
        raise ValueError(f"Unknown algorithm: {name}")