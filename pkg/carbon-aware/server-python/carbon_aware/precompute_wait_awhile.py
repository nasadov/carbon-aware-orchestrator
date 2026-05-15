"""
Precomputation runner for the Let's Wait Awhile-style temporal baseline.
"""

from typing import Optional

from carbon_aware.algorithms.wait_awhile import WaitAwhileAlgorithm
from carbon_aware.precompute_piontek_temporal import run_piontek_temporal_precomputation
from carbon_aware.utils import PerformanceLogger


def run_wait_awhile_precomputation(
    workloads_dir: str,
    nodes_file: str,
    forecasts_file: str,
    session_log_dir: Optional[str] = None,
    perf_logger: Optional[PerformanceLogger] = None,
    prioritize_efficiency: bool = False,
    operational_only: bool = True,
    node_score_mode: str = "most_allocated",
) -> bool:
    """
    Run the Wait-Awhile non-interrupting temporal carbon-shifting baseline.

    Future carbon forecasts are visible, as in the original paper, but future
    workload arrivals are not. Arrived pods wait until the coherent feasible
    start window with the lowest average cluster carbon intensity, or until the
    latest deadline-safe start.
    """
    return run_piontek_temporal_precomputation(
        workloads_dir=workloads_dir,
        nodes_file=nodes_file,
        forecasts_file=forecasts_file,
        session_log_dir=session_log_dir,
        perf_logger=perf_logger,
        prioritize_efficiency=prioritize_efficiency,
        operational_only=operational_only,
        node_score_mode=node_score_mode,
        algorithm_factory=WaitAwhileAlgorithm,
        summary_algorithm_name="wait_awhile",
        display_name="Wait-Awhile non-interrupting temporal",
    )
