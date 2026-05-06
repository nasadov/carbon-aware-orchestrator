from datetime import datetime, timedelta
from pathlib import Path
import sys


SERVER_DIR = Path(__file__).resolve().parents[1] / "pkg/carbon-aware/server-python"
sys.path.insert(0, str(SERVER_DIR))

from carbon_aware.algorithms.heuristic import find_best_node_and_timeslot
from carbon_aware.models import CarbonAwareFlavour, CarbonAwarePod, CarbonAwareTimeslot
from carbon_aware.utils import compute_marginal_emissions_for_pod_over_duration


def test_operational_only_marginal_cost_charges_idle_once():
    node = CarbonAwareFlavour(
        id="node-a",
        embodiedCarbon=0.0,
        lifetime=1.0,
        totalCpu=4.0,
        totalRam=4096.0,
        totalStorage=0.0,
        forecast={0: 100.0},
        power={"idle": 100.0, "max": 300.0},
    )

    empty_slot_g = compute_marginal_emissions_for_pod_over_duration(
        flavour=node,
        start_slot=0,
        duration_hours=1,
        pod_cpu_request=1.0,
        used_cpu_before_by_slot={0: 0.0},
        include_embodied=False,
    )
    occupied_slot_g = compute_marginal_emissions_for_pod_over_duration(
        flavour=node,
        start_slot=0,
        duration_hours=1,
        pod_cpu_request=1.0,
        used_cpu_before_by_slot={0: 1.0},
        include_embodied=False,
    )

    assert empty_slot_g == 15.0
    assert occupied_slot_g == 5.0


def test_totem_operational_only_prefers_reusing_active_node_slot():
    now_plus_one_hour = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    timeslot = CarbonAwareTimeslot(id=0, start_time=now_plus_one_hour)
    pod = CarbonAwarePod(
        id="pod-a",
        deadline_hours=4,
        duration=1,
        powerConsumption=0.0,
        cpuRequest=1.0,
        ramRequest=512.0,
        storageRequest=0,
        reference_time=now_plus_one_hour,
    )
    pod.earliest_timeslot = 0
    pod.deadline_slot = 4

    active_node = CarbonAwareFlavour(
        id="active-node",
        embodiedCarbon=0.0,
        lifetime=1.0,
        totalCpu=4.0,
        totalRam=4096.0,
        totalStorage=0.0,
        forecast={0: 100.0},
        power={"idle": 100.0, "max": 300.0},
    )
    idle_node = CarbonAwareFlavour(
        id="idle-node",
        embodiedCarbon=0.0,
        lifetime=1.0,
        totalCpu=4.0,
        totalRam=4096.0,
        totalStorage=0.0,
        forecast={0: 100.0},
        power={"idle": 100.0, "max": 300.0},
    )

    leftover_cpu = {
        "active-node": {0: 3.0},
        "idle-node": {0: 4.0},
    }
    leftover_ram = {
        "active-node": {0: 3584.0},
        "idle-node": {0: 4096.0},
    }

    node, slot, emissions = find_best_node_and_timeslot(
        pod=pod,
        flavours=[idle_node, active_node],
        timeslots=[timeslot],
        leftover_cpu=leftover_cpu,
        leftover_ram=leftover_ram,
        max_time_slots=1,
        operational_only=True,
    )

    assert node is active_node
    assert slot is timeslot
    assert emissions == 5.0
