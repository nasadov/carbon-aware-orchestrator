"""
Helpers for computing carbon and water footprint vectors for a placement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from carbon_aware.models import CarbonAwarePod, EnvironmentalFlavor
from carbon_aware.utils import compute_embodied_per_hour_g, compute_node_dynamic_coeff_watts, get_carbon_intensity


@dataclass
class FootprintVector:
    operational_energy_kwh: float = 0.0
    operational_carbon_g: float = 0.0
    embodied_carbon_g: float = 0.0
    total_carbon_g: float = 0.0
    direct_water_l: float = 0.0
    indirect_water_l: float = 0.0
    embodied_water_l: float = 0.0
    total_raw_water_l: float = 0.0
    scarcity_characterized_water: float = 0.0
    criticality_adjusted_water: float = 0.0

    def finalize(self) -> "FootprintVector":
        self.total_carbon_g = self.operational_carbon_g + self.embodied_carbon_g
        self.total_raw_water_l = self.direct_water_l + self.indirect_water_l + self.embodied_water_l
        self.criticality_adjusted_water = self.total_raw_water_l * self._criticality
        return self

    @property
    def operational_carbon_kg(self) -> float:
        return self.operational_carbon_g / 1000.0

    @property
    def embodied_carbon_kg(self) -> float:
        return self.embodied_carbon_g / 1000.0

    @property
    def total_carbon_kg(self) -> float:
        return self.total_carbon_g / 1000.0

    def as_csv_fields(self) -> Dict[str, float]:
        return {
            "operational_energy_kwh": self.operational_energy_kwh,
            "operational_carbon_kg": self.operational_carbon_kg,
            "embodied_carbon_kg": self.embodied_carbon_kg,
            "total_carbon_emissions": self.total_carbon_kg,
            "direct_water_l": self.direct_water_l,
            "indirect_water_l": self.indirect_water_l,
            "embodied_water_l": self.embodied_water_l,
            "total_raw_water_l": self.total_raw_water_l,
            "scarcity_characterized_water": self.scarcity_characterized_water,
            "criticality_adjusted_water": self.criticality_adjusted_water,
        }

    # These are set by compute_footprint_vector before finalize().
    _direct_cf: float = 1.0
    _indirect_cf: float = 1.0
    _embodied_cf: float = 1.0
    _criticality: float = 1.0


def _hours(duration_hours: float) -> int:
    return max(int(duration_hours), 0)


def _slot_value(values: Dict[int, float], slot: int, default: float = 0.0) -> float:
    if slot in values:
        return float(values[slot])
    if values:
        return float(next(iter(values.values())))
    return float(default)


def _slot_cf_value(
    flavour: EnvironmentalFlavor,
    slot: int,
    slot_attr: str,
    scalar_attr: str,
    default: float = 1.0,
) -> float:
    slot_values = getattr(flavour, slot_attr, {}) or {}
    scalar_default = float(getattr(flavour, scalar_attr, default) or default)
    return _slot_value(slot_values, slot, scalar_default)


def _compute_embodied_share(
    total_cpu_ratio_before: float,
    pod_cpu_ratio: float,
    embodied_allocation_mode: str,
) -> float:
    if pod_cpu_ratio <= 0.0:
        return 0.0

    if embodied_allocation_mode == "uniform":
        return 1.0 if total_cpu_ratio_before <= 0.0 else 0.0

    # Conserving, order-independent, utilization-based capacity share (SCI TE x TS x RS,
    # RS = request/capacity). The pod is charged its own capacity fraction of the device's
    # time-amortized embodied; concurrent pods' shares sum to <= 1 (unused capacity is
    # stranded), and the result does not depend on placement order. This replaces the
    # earlier marginal share pod/(used_before+pod), which over-counted and was order-dependent.
    return pod_cpu_ratio


def compute_embodied_water_per_hour(flavour: EnvironmentalFlavor) -> float:
    """Return the node embodied water amortized over its lifetime."""
    hours_in_lifetime = flavour.lifetime if flavour.lifetime and flavour.lifetime > 0 else 1e-6
    return flavour.embodiedWater / hours_in_lifetime


def build_used_cpu_before_map(
    flavour: EnvironmentalFlavor,
    start_slot: int,
    duration_hours: float,
    leftover_cpu_by_slot: Dict[int, float],
) -> Dict[int, float]:
    """Convert leftover CPU state into used CPU before placement for the covered slots."""
    used_cpu_before = {}
    total_capacity = max(flavour.totalCpu, 0.0)

    for offset in range(_hours(duration_hours)):
        slot = start_slot + offset
        leftover_cpu = leftover_cpu_by_slot.get(slot, total_capacity)
        used_cpu_before[slot] = max(total_capacity - leftover_cpu, 0.0)

    return used_cpu_before


def compute_footprint_vector(
    flavour: EnvironmentalFlavor,
    start_slot: int,
    pod: CarbonAwarePod,
    used_cpu_before_by_slot: Optional[Dict[int, float]] = None,
    embodied_allocation_mode: str = "proportional",
    operational_only: bool = False,
    use_pod_power_only: bool = False,
) -> FootprintVector:
    """
    Compute the full placement footprint vector for a pod.

    `use_pod_power_only=True` mirrors the legacy operational-only carbon path where
    each pod is charged idle + dynamic power independently of prior node occupancy.
    """
    total_cpu = max(flavour.totalCpu, 1e-6)
    pod_cpu_ratio = pod.cpuRequest / total_cpu
    dynamic_k = compute_node_dynamic_coeff_watts(flavour)
    embodied_carbon_per_hour = compute_embodied_per_hour_g(flavour)
    embodied_water_per_hour = compute_embodied_water_per_hour(flavour)

    # GPU extended resource (fractional-capable). Operational power adds to IT energy;
    # embodied is allocated by GPU usage with the same conserving, time-amortized rule as
    # CPU/RAM. All terms are 0 when the pod requests no GPU (gpuRequest=0) -> CPU/RAM path
    # is unchanged. Note: GPU embodied amortizes over the same device lifetime as the node.
    gpu_request = float(getattr(pod, "gpuRequest", 0) or 0)
    gpu_power_w = float(getattr(flavour, "gpu_power_w", 0) or 0)
    _gpu_lifetime = flavour.lifetime if getattr(flavour, "lifetime", 0) and flavour.lifetime > 0 else 1e-6
    gpu_embodied_carbon_per_hour = float(getattr(flavour, "gpu_embodied_carbon", 0) or 0) / _gpu_lifetime
    gpu_embodied_water_per_hour = float(getattr(flavour, "gpu_embodied_water", 0) or 0) / _gpu_lifetime

    direct_cf_scalar = float(getattr(flavour, "water_scarcity_direct_cf", 1.0) or 1.0)
    indirect_cf_scalar = float(getattr(flavour, "water_scarcity_indirect_cf", 1.0) or 1.0)
    embodied_cf = float(getattr(flavour, "water_scarcity_embodied_cf", 1.0) or 1.0)

    result = FootprintVector(
        _direct_cf=direct_cf_scalar,
        _indirect_cf=indirect_cf_scalar,
        _embodied_cf=embodied_cf,
        _criticality=float(getattr(flavour, "water_criticality", 1.0) or 1.0),
    )

    # Resolve per-slot lookup tables once. _slot_value/_slot_cf_value return the
    # scalar fallback when the by-slot dict is empty; in that (common) case we hoist
    # the constant out of the per-slot loop. Behaviour is identical to per-slot calls.
    idle_power_w = flavour.power["idle"]
    pue = float(getattr(flavour, "pue", 1.0) or 1.0)
    pue_by_slot = getattr(flavour, "pue_by_slot", {}) or {}  # temperature-dependent facility overhead
    wue_by_slot = getattr(flavour, "wue_by_slot", {}) or {}
    ewif_by_slot = getattr(flavour, "ewif_by_slot", {}) or {}
    direct_cf_by_slot = getattr(flavour, "water_scarcity_direct_cf_by_slot", {}) or {}
    indirect_cf_by_slot = getattr(flavour, "water_scarcity_indirect_cf_by_slot", {}) or {}

    for offset in range(_hours(pod.duration)):
        slot = start_slot + offset
        used_before = 0.0
        if used_cpu_before_by_slot is not None:
            used_before = used_cpu_before_by_slot.get(slot, 0.0)
        total_cpu_ratio_before = used_before / total_cpu

        if use_pod_power_only:
            delta_power_w = idle_power_w + dynamic_k * pod_cpu_ratio
        else:
            delta_power_w = 0.0
            if total_cpu_ratio_before <= 0.0 and pod_cpu_ratio > 0.0:
                delta_power_w += idle_power_w
            delta_power_w += dynamic_k * pod_cpu_ratio

        # GPU load power scales with the (fractional) GPU count requested.
        delta_power_w += gpu_request * gpu_power_w

        it_energy_kwh = delta_power_w / 1000.0
        # Facility (grid-drawn) energy = IT energy x PUE. PUE may be temperature-dependent
        # (per-slot) so hotter hours raise cooling overhead. This is what the grid, the
        # carbon account, and the power-plant water account all see.
        pue_slot = (_slot_value(pue_by_slot, slot, pue) if pue_by_slot else pue)
        facility_energy_kwh = it_energy_kwh * pue_slot
        carbon_intensity = get_carbon_intensity(flavour, slot)
        wue = _slot_value(wue_by_slot, slot, 0.0) if wue_by_slot else 0.0
        ewif = _slot_value(ewif_by_slot, slot, 0.0) if ewif_by_slot else 0.0
        direct_cf = _slot_value(direct_cf_by_slot, slot, direct_cf_scalar) if direct_cf_by_slot else direct_cf_scalar
        indirect_cf = _slot_value(indirect_cf_by_slot, slot, indirect_cf_scalar) if indirect_cf_by_slot else indirect_cf_scalar

        result.operational_energy_kwh += facility_energy_kwh
        result.operational_carbon_g += carbon_intensity * facility_energy_kwh
        direct_water = it_energy_kwh * wue          # on-site cooling water (WUE per IT kWh)
        indirect_water = facility_energy_kwh * ewif  # power-plant water for total grid draw
        result.direct_water_l += direct_water
        result.indirect_water_l += indirect_water
        result.scarcity_characterized_water += direct_water * direct_cf + indirect_water * indirect_cf

        if not operational_only:
            embodied_share = _compute_embodied_share(
                total_cpu_ratio_before=total_cpu_ratio_before,
                pod_cpu_ratio=pod_cpu_ratio,
                embodied_allocation_mode=embodied_allocation_mode,
            )
            embodied_carbon = embodied_carbon_per_hour * embodied_share
            embodied_water = embodied_water_per_hour * embodied_share
            # GPU board embodied: conserving GPU-usage share (gpuRequest GPUs of the device's
            # per-GPU embodied, time-amortized), same philosophy as the CPU/RAM term.
            if gpu_request > 0.0:
                embodied_carbon += gpu_embodied_carbon_per_hour * gpu_request
                embodied_water += gpu_embodied_water_per_hour * gpu_request
            result.embodied_carbon_g += embodied_carbon
            result.embodied_water_l += embodied_water
            result.scarcity_characterized_water += embodied_water * embodied_cf

    return result.finalize()
