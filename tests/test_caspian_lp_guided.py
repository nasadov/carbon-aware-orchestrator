from datetime import datetime
from pathlib import Path
import sys


SERVER_DIR = Path(__file__).resolve().parents[1] / "pkg/carbon-aware/server-python"
sys.path.insert(0, str(SERVER_DIR))

from carbon_aware.algorithms.caspian_operational import CaspianOperationalAlgorithm
from carbon_aware.models import CarbonAwareFlavour, CarbonAwarePod


def _pod(name: str, cpu: float = 1.0) -> CarbonAwarePod:
    reference = datetime.now().replace(minute=0, second=0, microsecond=0)
    pod = CarbonAwarePod(
        id=name,
        deadline_hours=1,
        duration=1,
        powerConsumption=0.0,
        cpuRequest=cpu,
        ramRequest=256.0,
        storageRequest=0,
        reference_time=reference,
    )
    pod.earliest_timeslot = 0
    pod.deadline_slot = 1
    setattr(pod, "_earliest_timeslot_source", "precompute")
    return pod


def _node(name: str, carbon: float, cpu: float = 2.0) -> CarbonAwareFlavour:
    return CarbonAwareFlavour(
        id=name,
        embodiedCarbon=0.0,
        lifetime=1.0,
        totalCpu=cpu,
        totalRam=4096.0,
        totalStorage=0.0,
        forecast={0: carbon},
        power={"idle": 100.0, "max": 300.0},
    )


def test_caspian_lp_guided_prefers_lower_carbon_node():
    algorithm = CaspianOperationalAlgorithm()
    low = _node("low-carbon", carbon=50.0)
    high = _node("high-carbon", carbon=500.0)
    available_cpu = {low.id: {0: 2.0}, high.id: {0: 2.0}}
    available_ram = {low.id: {0: 4096.0}, high.id: {0: 4096.0}}

    solution = algorithm.solve_visible_queue_lp_guided(
        pods=[_pod("pod-a")],
        flavours=[high, low],
        max_time_slots=1,
        current_slot=0,
        available_cpu=available_cpu,
        available_ram=available_ram,
        time_limit_seconds=3.0,
    )

    assert solution["pod-a"][0] is low
    assert solution["pod-a"][1] == 0
    assert algorithm.status.startswith("LP_GUIDED")


def test_caspian_lp_guided_allocation_respects_integral_capacity():
    algorithm = CaspianOperationalAlgorithm()
    node = _node("node-a", carbon=100.0, cpu=2.0)
    available_cpu = {node.id: {0: 2.0}}
    available_ram = {node.id: {0: 4096.0}}

    solution = algorithm.solve_visible_queue_lp_guided(
        pods=[_pod("pod-a"), _pod("pod-b"), _pod("pod-c")],
        flavours=[node],
        max_time_slots=1,
        current_slot=0,
        available_cpu=available_cpu,
        available_ram=available_ram,
        time_limit_seconds=3.0,
    )

    assert len(solution) == 2
    placed_cpu = len(solution) * 1.0
    assert placed_cpu <= node.totalCpu
