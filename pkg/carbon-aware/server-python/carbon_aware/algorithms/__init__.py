"""
Algorithm registry and factory functions.
"""
from carbon_aware.algorithms.base import SchedulingAlgorithm
from carbon_aware.algorithms.heuristic import HeuristicAlgorithm
from carbon_aware.algorithms.optimal import OptimalAlgorithm

def get_algorithm(name: str) -> SchedulingAlgorithm:
    """Factory function to get algorithm implementation by name"""
    
    algorithms = {
        "heuristic": HeuristicAlgorithm(),
        "optimal": OptimalAlgorithm(),
    }
    
    normalized_name = name.lower()
    if normalized_name in algorithms:
        return algorithms[normalized_name]
    
    raise ValueError(f"Unknown algorithm: {name}")