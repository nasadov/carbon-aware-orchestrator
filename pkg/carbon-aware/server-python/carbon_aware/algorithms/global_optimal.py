"""
Carbon-aware scheduling with global optimization using MILP.
This algorithm considers all pods simultaneously to find the system-wide optimal solution.
"""
import logging
import time
import os
import yaml
import threading
from typing import Dict, List, Optional, Tuple
import pulp
import csv
from datetime import datetime
from pathlib import Path

from carbon_aware.algorithms.base import SchedulingAlgorithm
from carbon_aware.footprints import FootprintVector, build_used_cpu_before_map, compute_footprint_vector
from carbon_aware.models import CarbonAwarePod, CarbonAwareTimeslot, EnvironmentalFlavor
from carbon_aware.water_signals import attach_water_metadata
from carbon_aware.utils import is_timeslot_valid, compute_emissions, compute_node_dynamic_coeff_watts, get_carbon_intensity, compute_embodied_per_hour_g, compute_emissions_with_allocation


class GlobalOptimalAlgorithm(SchedulingAlgorithm):
    """
    Global optimization implementation of carbon-aware scheduling using MILP.
    This algorithm considers all pods together to find a system-wide optimal solution
    that minimizes total carbon emissions across the entire workload.
    """
    SESSION_CSV_FILENAME = "global_optimal_placements_session.csv"  # New constant for session-wide CSV
    CSV_HEADERS = [
        "pod_id", "node_id", "start_slot", "duration",
        "cpu_request", "ram_request", "region", "country", "operational_energy_kwh",
        "operational_carbon_kg", "embodied_carbon_kg", "total_carbon_emissions",
        "direct_water_l", "indirect_water_l", "embodied_water_l", "total_raw_water_l",
        "scarcity_characterized_water", "criticality_adjusted_water",
        "solver_status", "solver_iterations", "solution_time_seconds",
        "embodied_mode"
    ]

    def __init__(self):
        """
        Initialize the global optimal algorithm with necessary tracking and state.
        Sets up logging structures, solver state, and resource tracking.
        """
        # Configure logging for the algorithm
        logging.info("🔧 Initializing Global Optimal Algorithm")
        
        # Load configuration
        self.config = self._load_config()
        use_lexicographic = self.config.get('optimization', {}).get('use_lexicographic', True)
        self.use_lexicographic = use_lexicographic
        logging.info(f"🔧 Configuration loaded: use_lexicographic={use_lexicographic}")
        # Cache solver name
        solver_cfg = self.config.get('optimization', {}).get('solver', {})
        self.solver_name = solver_cfg.get('name', 'cp_sat')
        logging.info(f"🔧 Solver configured: name={self.solver_name}")
        
        # Add experiment logger and tracking properties
        self.experiment_logger = None
        self.iterations = 0
        self.status = None
        self.global_solution = {}  # Dictionary: pod_id -> (flv_id, ts_id, emissions)
        
        # State tracking for incremental vs precomputation
        self.is_precomputation_mode = False  # Flag to distinguish precomputation from incremental calls
        self.optimization_done = False       # Marks if comprehensive optimization is complete
        self.has_solved = False              # Marks if any solution has been found
        
        # Solver state and resource tracking
        self.pending_pods: List[CarbonAwarePod] = []
        self._processed_pods = set()  # Track pods that have been processed to avoid duplicates
        self.timeslot_data = {}
        self.all_flavours = []
        self.all_timeslots = []
        self.resource_state = {"cpu": {}, "ram": {}}  # For precomputation mode
        
        # CSV logging setup
        self._csv_file_handle = None
        self._csv_writer = None
        self._session_csv_path = None
        
        # Solver control and timing
        self.solve_interval = 10  # Solve every N pods or every N seconds
        self.last_solve_time = 0
        self._solving_lock = threading.Lock()  # Thread-safe solving
        
        logging.info(f"✅ Global Optimal Algorithm instance created with CSV headers: {self.CSV_HEADERS}")
        logging.info(f"📋 This algorithm will track placements in a CSV file once set_base_log_dir is called")

        # Embodied allocation mode ("proportional" or "uniform")
        self.embodied_allocation_mode = "proportional"

        # Flag: when True, ignore embodied emissions in objectives and reporting
        # and optimize using operational emissions only (idle + dynamic)
        self.operational_only = False

        # Optional epsilon-constraint settings for Phase 2.
        # When water_budget is set, Phase 2 minimizes carbon subject to
        # total water <= water_budget for the selected water metric.
        self.phase2_water_budget: Optional[float] = None
        self.phase2_water_metric: str = "scarcity"

        # Phase-2 objective mode. The default is the original carbon objective.
        # The WaterWise-style mode is a scalarized MILP baseline adapted from
        # Jiang et al.'s normalized carbon-water objective.
        self.phase2_objective_mode: str = "carbon"
        self.waterwise_carbon_weight: float = 0.5
        self.waterwise_water_metric: str = "scarcity"
        self.waterwise_reference_weight: float = 0.1
        self.waterwise_history_window: int = 10

        # Objective weighting knobs (config-driven)
        # - weight_dynamic: scales per-pod dynamic emissions (k_watts * u * intensity)
        # - weight_activation: scales per-(node,slot) activation cost (idle + embodied)
        # - weight_fallback_surrogate_activation: scales surrogate activation penalty in dynamic-only fallback
        emissions_cfg = self.config.get('optimization', {}).get('emissions', {}) if hasattr(self, 'config') else {}
        try:
            self.weight_dynamic = float(emissions_cfg.get('dynamic_weight', 1.0))
        except Exception:
            self.weight_dynamic = 1.0
        try:
            self.weight_activation = float(emissions_cfg.get('activation_weight', 1.0))
        except Exception:
            self.weight_activation = 1.0
        try:
            self.weight_fallback_surrogate_activation = float(
                emissions_cfg.get('fallback_surrogate_activation_weight', emissions_cfg.get('surrogate_activation_weight', 0.0))
            )
        except Exception:
            self.weight_fallback_surrogate_activation = 0.0
        logging.info(
            f"🔧 Emissions weights: dynamic={self.weight_dynamic}, activation={self.weight_activation}, "
            f"fallback_surrogate_activation={self.weight_fallback_surrogate_activation}"
        )

        # Candidate pruning knobs to focus MILP on greener placements
        pruning_cfg = self.config.get('optimization', {}).get('pruning', {}) if hasattr(self, 'config') else {}
        self.pruning_enable = bool(pruning_cfg.get('enable', True))
        try:
            self.pruning_top_k_per_pod = int(pruning_cfg.get('top_k_per_pod', 8))
        except Exception:
            self.pruning_top_k_per_pod = 8
        try:
            self.pruning_emissions_margin = float(pruning_cfg.get('emissions_margin', 0.15))
        except Exception:
            self.pruning_emissions_margin = 0.15
        logging.info(
            f"🔧 Pruning: enable={self.pruning_enable}, top_k_per_pod={self.pruning_top_k_per_pod}, "
            f"emissions_margin={self.pruning_emissions_margin}"
        )

    def set_embodied_allocation_mode(self, mode: str):
        if mode not in ("proportional", "uniform"):
            logging.warning(f"Unknown embodied allocation mode '{mode}', defaulting to 'proportional'")
            mode = "proportional"
        self.embodied_allocation_mode = mode
        logging.info(f"GlobalOptimalAlgorithm: embodied_allocation_mode set to {self.embodied_allocation_mode}")

    def set_operational_only(self, operational_only: bool):
        """Enable or disable operational-only optimization.

        When enabled, embodied emissions are excluded from solver objectives and
        from the per-pod emissions reported to CSV/logs. Idle and dynamic operational
        terms remain included.
        """
        self.operational_only = bool(operational_only)
        logging.info(f"GlobalOptimalAlgorithm: operational_only mode set to {self.operational_only}")

    def set_epsilon_constraint(self, water_budget: Optional[float] = None, water_metric: str = "scarcity"):
        """Configure an optional epsilon-constraint water budget for Phase 2."""
        parsed_budget: Optional[float]
        if water_budget in (None, "", "none", "None"):
            parsed_budget = None
        else:
            try:
                parsed_budget = float(water_budget)
            except (TypeError, ValueError):
                logging.warning("Invalid water budget %r; disabling epsilon-constraint", water_budget)
                parsed_budget = None

        if parsed_budget is not None and parsed_budget <= 0:
            logging.warning("Non-positive water budget %.6f; disabling epsilon-constraint", parsed_budget)
            parsed_budget = None

        if water_metric not in ("scarcity", "raw"):
            logging.warning("Unknown water metric '%s'; defaulting to 'scarcity'", water_metric)
            water_metric = "scarcity"

        self.phase2_water_budget = parsed_budget
        self.phase2_water_metric = water_metric
        if parsed_budget is None:
            logging.info("GlobalOptimalAlgorithm: epsilon-constraint disabled")
        else:
            logging.info(
                "GlobalOptimalAlgorithm: epsilon-constraint enabled with %s water budget %.6f",
                self.phase2_water_metric,
                self.phase2_water_budget,
            )

    def set_phase2_objective(
        self,
        mode: str = "carbon",
        carbon_weight: float = 0.5,
        water_metric: str = "scarcity",
        reference_weight: float = 0.1,
        history_window: int = 10,
    ):
        """Configure the Phase-2 objective used after placement maximization.

        `carbon` is the original MILP objective. `waterwise-scalarized` is a
        WaterWise-style scalarized baseline:

            lambda_C * C/C_max + lambda_W * W/W_max
            + lambda_ref * (lambda_C * C_ref + lambda_W * W_ref)

        adapted from region assignment to our pod-node-time candidate space.
        """
        if mode not in ("carbon", "waterwise-scalarized"):
            logging.warning("Unknown global objective '%s'; defaulting to carbon", mode)
            mode = "carbon"
        try:
            carbon_weight = float(carbon_weight)
        except (TypeError, ValueError):
            carbon_weight = 0.5
        try:
            reference_weight = float(reference_weight)
        except (TypeError, ValueError):
            reference_weight = 0.1
        try:
            history_window = int(history_window)
        except (TypeError, ValueError):
            history_window = 10
        if water_metric not in ("scarcity", "raw"):
            logging.warning("Unknown WaterWise water metric '%s'; defaulting to scarcity", water_metric)
            water_metric = "scarcity"

        self.phase2_objective_mode = mode
        self.waterwise_carbon_weight = min(max(carbon_weight, 0.0), 1.0)
        self.waterwise_water_metric = water_metric
        self.waterwise_reference_weight = max(reference_weight, 0.0)
        self.waterwise_history_window = max(history_window, 1)
        logging.info(
            "GlobalOptimalAlgorithm: phase2 objective=%s carbon_weight=%.2f water_metric=%s ref_weight=%.3f history_window=%s",
            self.phase2_objective_mode,
            self.waterwise_carbon_weight,
            self.waterwise_water_metric,
            self.waterwise_reference_weight,
            self.waterwise_history_window,
        )

    @staticmethod
    def _slot_signal_value(values: Dict[int, float], slot: int, default: float = 0.0) -> float:
        if slot in values:
            return float(values[slot])
        if values:
            return float(next(iter(values.values())))
        return float(default)

    @staticmethod
    def _footprint_water_value(footprint: FootprintVector, water_metric: str) -> float:
        if water_metric == "raw":
            return float(footprint.total_raw_water_l)
        return float(footprint.scarcity_characterized_water)

    def _water_intensity_signal(self, flavour: EnvironmentalFlavor, slot: int, water_metric: str) -> float:
        """Return direct+indirect water intensity per kWh for a node slot."""
        wue = self._slot_signal_value(getattr(flavour, "wue_by_slot", {}) or {}, slot, 0.0)
        ewif = self._slot_signal_value(getattr(flavour, "ewif_by_slot", {}) or {}, slot, 0.0)
        direct = wue
        indirect = float(getattr(flavour, "pue", 1.0) or 1.0) * ewif
        if water_metric == "scarcity":
            direct *= float(getattr(flavour, "water_scarcity_direct_cf", 1.0) or 1.0)
            indirect *= float(getattr(flavour, "water_scarcity_indirect_cf", 1.0) or 1.0)
        return direct + indirect

    def _build_waterwise_reference_maps(
        self,
        flavours: List[EnvironmentalFlavor],
        timeslots: List[CarbonAwareTimeslot],
        water_metric: str,
    ) -> Tuple[Dict[Tuple[str, int], float], Dict[Tuple[str, int], float]]:
        """Build normalized WaterWise-style reference signals.

        WaterWise uses region-level historical normalized carbon/water terms.
        In our offline node-time setting, the closest analogue is a rolling
        reference over recent node-slot carbon and water-intensity signals.
        """
        slot_ids = sorted(ts.id for ts in timeslots)
        slot_set = set(slot_ids)
        window = max(int(getattr(self, "waterwise_history_window", 10)), 1)
        raw_carbon: Dict[Tuple[str, int], float] = {}
        raw_water: Dict[Tuple[str, int], float] = {}

        for flavour in flavours:
            for ts in timeslots:
                current_slot = ts.id
                history_slots = [
                    slot
                    for slot in range(current_slot - window + 1, current_slot + 1)
                    if slot in slot_set
                ]
                if not history_slots:
                    history_slots = [current_slot]
                carbon_avg = sum(get_carbon_intensity(flavour, slot) for slot in history_slots) / len(history_slots)
                water_avg = sum(self._water_intensity_signal(flavour, slot, water_metric) for slot in history_slots) / len(history_slots)
                key = (flavour.id, current_slot)
                raw_carbon[key] = float(carbon_avg)
                raw_water[key] = float(water_avg)

        carbon_max = max(raw_carbon.values(), default=0.0)
        water_max = max(raw_water.values(), default=0.0)
        carbon_denom = carbon_max if carbon_max > 1e-12 else 1.0
        water_denom = water_max if water_max > 1e-12 else 1.0
        carbon_ref = {key: value / carbon_denom for key, value in raw_carbon.items()}
        water_ref = {key: value / water_denom for key, value in raw_water.items()}
        return carbon_ref, water_ref

    def _build_waterwise_scalarized_terms(
        self,
        placements,
        flavours: List[EnvironmentalFlavor],
        timeslots: List[CarbonAwareTimeslot],
        placement_vars,
        use_cpsat: bool = False,
        excluded_pods: Optional[set] = None,
    ):
        """Build WaterWise-style normalized scalarized objective terms."""
        excluded_pods = excluded_pods or set()
        flv_by_id = {flavour.id: flavour for flavour in flavours}
        pod_by_id = {pod.id: pod for pod in self.pending_pods}
        carbon_weight = float(getattr(self, "waterwise_carbon_weight", 0.5))
        water_weight = 1.0 - carbon_weight
        reference_weight = float(getattr(self, "waterwise_reference_weight", 0.1))
        water_metric = getattr(self, "waterwise_water_metric", "scarcity")
        carbon_ref, water_ref = self._build_waterwise_reference_maps(flavours, timeslots, water_metric)

        candidate_rows: Dict[str, List[Tuple[str, int, float, float, float, float]]] = {}
        for pod_id, pod_placements in placements.items():
            if pod_id in excluded_pods:
                continue
            pod_obj = pod_by_id.get(pod_id)
            if not pod_obj:
                continue
            for flv_id, ts_id, _ in pod_placements:
                flv_obj = flv_by_id.get(flv_id)
                if not flv_obj:
                    continue
                footprint = compute_footprint_vector(
                    flavour=flv_obj,
                    start_slot=ts_id,
                    pod=pod_obj,
                    used_cpu_before_by_slot=None,
                    embodied_allocation_mode=getattr(self, "embodied_allocation_mode", "proportional"),
                    operational_only=getattr(self, "operational_only", False),
                    use_pod_power_only=True,
                )
                carbon_value = float(footprint.total_carbon_g)
                water_value = self._footprint_water_value(footprint, water_metric)
                candidate_rows.setdefault(pod_id, []).append(
                    (
                        flv_id,
                        ts_id,
                        carbon_value,
                        water_value,
                        carbon_ref.get((flv_id, ts_id), 0.0),
                        water_ref.get((flv_id, ts_id), 0.0),
                    )
                )

        terms = []
        scale = 1_000_000
        for pod_id, rows in candidate_rows.items():
            carbon_max = max((row[2] for row in rows), default=0.0)
            water_max = max((row[3] for row in rows), default=0.0)
            carbon_denom = carbon_max if carbon_max > 1e-12 else 1.0
            water_denom = water_max if water_max > 1e-12 else 1.0

            for flv_id, ts_id, carbon_value, water_value, carbon_ref_value, water_ref_value in rows:
                normalized_score = (
                    carbon_weight * (carbon_value / carbon_denom)
                    + water_weight * (water_value / water_denom)
                    + reference_weight * (
                        carbon_weight * carbon_ref_value
                        + water_weight * water_ref_value
                    )
                )
                var = placement_vars[(pod_id, flv_id, ts_id)]
                if use_cpsat:
                    terms.append(int(round(normalized_score * scale)) * var)
                else:
                    terms.append(normalized_score * var)

        logging.info(
            "Built WaterWise-style scalarized objective terms: pods=%s terms=%s lambda_C=%.2f lambda_W=%.2f lambda_ref=%.3f metric=%s",
            len(candidate_rows),
            len(terms),
            carbon_weight,
            water_weight,
            reference_weight,
            water_metric,
        )
        return terms

    def _build_water_linear_terms(
        self,
        placements,
        flavours: List[EnvironmentalFlavor],
        timeslots: List[CarbonAwareTimeslot],
        placement_vars,
        activation_vars,
        max_time_slots: int,
        use_cpsat: bool = False,
        excluded_pods: Optional[set] = None,
    ):
        """Build dynamic and activation water terms for Phase 2."""
        excluded_pods = excluded_pods or set()
        flv_by_id = {f.id: f for f in flavours}
        dynamic_terms = []
        activation_terms = []

        for pod_id, pod_placements in placements.items():
            if pod_id in excluded_pods:
                continue
            pod_obj = next((p for p in self.pending_pods if p.id == pod_id), None)
            if not pod_obj:
                continue
            for flv_id, ts_id, _ in pod_placements:
                flv_obj = flv_by_id.get(flv_id)
                if not flv_obj:
                    continue
                k_watts = compute_node_dynamic_coeff_watts(flv_obj)
                total_cpu = flv_obj.totalCpu if flv_obj.totalCpu else 1e-6
                u_pod = pod_obj.cpuRequest / total_cpu
                for offset in range(int(pod_obj.duration)):
                    slot = ts_id + offset
                    if slot >= max_time_slots:
                        break
                    wue = self._slot_signal_value(getattr(flv_obj, "wue_by_slot", {}) or {}, slot, 0.0)
                    ewif = self._slot_signal_value(getattr(flv_obj, "ewif_by_slot", {}) or {}, slot, 0.0)
                    direct_component = wue
                    indirect_component = float(getattr(flv_obj, "pue", 1.0) or 1.0) * ewif
                    if self.phase2_water_metric == "scarcity":
                        direct_component *= float(getattr(flv_obj, "water_scarcity_direct_cf", 1.0) or 1.0)
                        indirect_component *= float(getattr(flv_obj, "water_scarcity_indirect_cf", 1.0) or 1.0)
                    coef = (k_watts * u_pod / 1000.0) * (direct_component + indirect_component)
                    if use_cpsat:
                        dynamic_terms.append(int(round(coef * 1000.0)) * placement_vars[(pod_id, flv_id, ts_id)])
                    else:
                        dynamic_terms.append(coef * placement_vars[(pod_id, flv_id, ts_id)])

        for flv in flavours:
            idle_w = flv.power.get("idle", 0.0)
            embodied_water_per_h = (flv.embodiedWater / flv.lifetime) if flv.lifetime else 0.0
            if getattr(self, "operational_only", False):
                embodied_water_per_h = 0.0
            if self.phase2_water_metric == "scarcity":
                embodied_water_per_h *= float(getattr(flv, "water_scarcity_embodied_cf", 1.0) or 1.0)
            for ts in timeslots:
                wue = self._slot_signal_value(getattr(flv, "wue_by_slot", {}) or {}, ts.id, 0.0)
                ewif = self._slot_signal_value(getattr(flv, "ewif_by_slot", {}) or {}, ts.id, 0.0)
                direct_component = wue
                indirect_component = float(getattr(flv, "pue", 1.0) or 1.0) * ewif
                if self.phase2_water_metric == "scarcity":
                    direct_component *= float(getattr(flv, "water_scarcity_direct_cf", 1.0) or 1.0)
                    indirect_component *= float(getattr(flv, "water_scarcity_indirect_cf", 1.0) or 1.0)
                coef = (idle_w / 1000.0) * (direct_component + indirect_component) + embodied_water_per_h
                if use_cpsat:
                    activation_terms.append(int(round(coef * 1000.0)) * activation_vars[(flv.id, ts.id)])
                else:
                    activation_terms.append(coef * activation_vars[(flv.id, ts.id)])

        return dynamic_terms, activation_terms

    def _compute_solution_footprints(
        self,
        solution_dict: Dict[str, Tuple[str, int, float]],
        pending_pods_dict: Dict[str, CarbonAwarePod],
        flavours: List[EnvironmentalFlavor],
        max_time_slots: int,
    ) -> Dict[str, FootprintVector]:
        """Compute per-pod footprint vectors consistent with the MILP occupancy model."""
        flv_by_id = {f.id: f for f in flavours}
        occupancy: Dict[Tuple[str, int], List[Tuple[str, float]]] = {}
        pod_footprints: Dict[str, FootprintVector] = {}

        for pod_id_sol, (flv_id_sol, ts_id_sol, _) in solution_dict.items():
            pod_obj = pending_pods_dict.get(pod_id_sol)
            flv_obj = flv_by_id.get(flv_id_sol)
            if not pod_obj or not flv_obj:
                continue
            pod_footprints[pod_id_sol] = FootprintVector(
                _direct_cf=float(getattr(flv_obj, "water_scarcity_direct_cf", 1.0) or 1.0),
                _indirect_cf=float(getattr(flv_obj, "water_scarcity_indirect_cf", 1.0) or 1.0),
                _embodied_cf=float(getattr(flv_obj, "water_scarcity_embodied_cf", 1.0) or 1.0),
                _criticality=float(getattr(flv_obj, "water_criticality", 1.0) or 1.0),
            )
            total_cpu = max(flv_obj.totalCpu, 1e-6)
            u_i = pod_obj.cpuRequest / total_cpu
            for offset in range(int(pod_obj.duration)):
                slot = ts_id_sol + offset
                if slot >= max_time_slots:
                    break
                occupancy.setdefault((flv_id_sol, slot), []).append((pod_id_sol, u_i))

        for (flv_id, slot), items in occupancy.items():
            flv_obj = flv_by_id.get(flv_id)
            if not flv_obj:
                continue
            intensity = get_carbon_intensity(flv_obj, slot)
            k_watts = compute_node_dynamic_coeff_watts(flv_obj)
            idle_w = flv_obj.power.get("idle", 0.0)
            embodied_carbon_per_h = 0.0 if getattr(self, "operational_only", False) else compute_embodied_per_hour_g(flv_obj)
            embodied_water_per_h = 0.0
            if not getattr(self, "operational_only", False):
                embodied_water_per_h = (flv_obj.embodiedWater / flv_obj.lifetime) if flv_obj.lifetime else 0.0
            wue = self._slot_signal_value(getattr(flv_obj, "wue_by_slot", {}) or {}, slot, 0.0)
            ewif = self._slot_signal_value(getattr(flv_obj, "ewif_by_slot", {}) or {}, slot, 0.0)
            pue = float(getattr(flv_obj, "pue", 1.0) or 1.0)

            total_u = sum(u for _, u in items)
            if total_u <= 0:
                continue

            for pod_id_sol, u_i in items:
                footprint = pod_footprints[pod_id_sol]
                share = u_i / total_u
                dynamic_energy_kwh = (k_watts * u_i) / 1000.0
                idle_energy_kwh = (idle_w / 1000.0) * share
                operational_energy_kwh = dynamic_energy_kwh + idle_energy_kwh

                footprint.operational_energy_kwh += operational_energy_kwh
                footprint.operational_carbon_g += intensity * operational_energy_kwh
                footprint.embodied_carbon_g += embodied_carbon_per_h * share
                footprint.direct_water_l += operational_energy_kwh * wue
                footprint.indirect_water_l += operational_energy_kwh * pue * ewif
                footprint.embodied_water_l += embodied_water_per_h * share

        for pod_id_sol, footprint in pod_footprints.items():
            pod_footprints[pod_id_sol] = footprint.finalize()

        return pod_footprints

    def _load_config(self) -> Dict:
        """Load configuration from infra-workload-config.yaml, honoring an env override."""
        env_path = os.environ.get("CARBON_AWARE_CONFIG_PATH")
        config_path = env_path or str(Path(__file__).resolve().parents[3] / "infra-workload-config.yaml")
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logging.info(f"✅ Configuration loaded from {config_path}")
                return config or {}
        except Exception as e:
            logging.warning(f"⚠️ Failed to load configuration from {config_path}: {e}")
            # Return default configuration
            return {
                'optimization': {
                    'use_lexicographic': True,
                    'solver': {
                        'time_limit': 20,
                        'gap_tolerance': 0.0
                    }
                }
            }

    def _set_pod_earliest_timeslot(self, pod: CarbonAwarePod):
        """
        Ensure the pod has earliest_timeslot set and calculate deadline_slot.
        
        This method validates that the pod has its earliest_timeslot properly set
        (usually done during pod creation from YAML files) and calculates the
        deadline_slot relative to that earliest_timeslot.
        
        If earliest_timeslot is not set, it extracts it from the timeslot YAML file
        that contains this pod's definition.
        """
        current_ts = getattr(pod, "earliest_timeslot", None)
        if current_ts is None or current_ts < 0:
            extracted_ts = self._extract_earliest_timeslot_from_yaml_files(pod.id)
            pod.earliest_timeslot = extracted_ts
            logging.info(f"🔒 Pod {pod.id} earliest_timeslot set from YAML lookup: {pod.earliest_timeslot}")
        else:
            logging.debug(f"🔒 Pod {pod.id} retaining existing earliest_timeslot={pod.earliest_timeslot}")

        # Calculate deadline_slot relative to earliest_timeslot
        pod.calculate_deadline_slot()
        logging.info(f"⏰ Pod {pod.id} deadline_slot calculated as {pod.deadline_slot}")

    def _extract_earliest_timeslot_from_yaml_files(self, pod_id: str) -> int:
        """
        Extract the earliest timeslot by finding which timeslot_X.yaml file contains this pod.
        
        This is the CORRECT way to determine earliest timeslot - by looking at which 
        timeslot file the pod came from. If pod is in timeslot_4.yaml, then earliest_timeslot=4.
        """
        import os
        import re
        
        # Look in the workloads directory for timeslot_*.yaml files
        workloads_dir = getattr(self, "_workloads_dir", None) or str(
            Path(__file__).resolve().parents[3] / "workloads"
        )
        
        try:
            for filename in os.listdir(workloads_dir):
                if re.match(r'timeslot_(\d+)\.yaml$', filename):
                    filepath = os.path.join(workloads_dir, filename)
                    
                    # Extract timeslot number from filename
                    match = re.match(r'timeslot_(\d+)\.yaml$', filename)
                    timeslot_num = int(match.group(1))
                    
                    # Check if this pod is defined in this file
                    try:
                        with open(filepath, 'r') as f:
                            content = f.read()
                            # Look for the pod name in deployment metadata
                            if f"name: {pod_id}" in content:
                                logging.info(f"Found pod {pod_id} in {filename} → earliest_timeslot={timeslot_num}")
                                return timeslot_num
                    except Exception as e:
                        logging.warning(f"⚠️ Error reading {filepath}: {e}")
                        continue
        except Exception as e:
            logging.warning(f"⚠️ Error scanning workloads directory {workloads_dir}: {e}")
        
        # Fallback: if not found in any timeslot file, default to 0
        logging.warning(f"⚠️ Pod {pod_id} not found in any timeslot_X.yaml file under {workloads_dir}, defaulting to earliest_timeslot=0")
        return 0

    def __del__(self):
        self.close_csv()

    @staticmethod
    def _ensure_dir_exists(directory_path: str):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            logging.info(f"Created directory: {directory_path}")

    def set_base_log_dir(self, base_dir: str):
        """Sets the base directory for session logs."""
        self._session_log_dir = base_dir

    def _get_pulp_solver(self, solver_cfg: Dict, time_limit: int, gap: float, threads: int):
        """Return a PuLP solver instance based on config and availability.
        Tries HiGHS, Gurobi, CPLEX, then CBC as a fallback.
        """
        import pulp
        name = (solver_cfg.get('name') or self.solver_name or 'cbc').lower()
        msg = bool(solver_cfg.get('msg', False))
        # Try explicit name first
        if name == 'highs':
            try:
                return pulp.apis.HiGHS_CMD(msg=msg, timeLimit=time_limit, mip_rel_gap=gap, threads=threads)
            except Exception:
                pass
        if name == 'gurobi':
            try:
                return pulp.GUROBI_CMD(msg=msg, timeLimit=time_limit, mipgap=gap, threads=threads)
            except Exception:
                pass
        if name == 'cplex':
            try:
                return pulp.CPLEX_CMD(msg=msg, timelimit=time_limit, epgap=gap, threads=threads)
            except Exception:
                pass
        if name == 'cbc':
            try:
                return pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit, gapRel=gap, threads=threads)
            except Exception:
                pass
        # Auto-detect best available
        try:
            return pulp.apis.HiGHS_CMD(msg=msg, timeLimit=time_limit, mip_rel_gap=gap, threads=threads)
        except Exception:
            try:
                return pulp.GUROBI_CMD(msg=msg, timeLimit=time_limit, mipgap=gap, threads=threads)
            except Exception:
                try:
                    return pulp.CPLEX_CMD(msg=msg, timelimit=time_limit, epgap=gap, threads=threads)
                except Exception:
                    return pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit, gapRel=gap, threads=threads)

    def _solve_phase2_with_cpsat(self, placements, flavours, timeslots, leftover_cpu, leftover_ram, max_time_slots,
                                 unplaced_phase1, placed_phase1, time_limit_sec: int, gap_tolerance: float | None = None):
        """Solve Phase 2 with OR-Tools CP-SAT. Returns (ok, solution_dict, elapsed_sec)."""
        import time
        start = time.time()
        try:
            from ortools.sat.python import cp_model
        except Exception:
            return False, {}, time.time() - start

        model = cp_model.CpModel()

        # Index helpers
        flv_by_id = {f.id: f for f in flavours}
        ts_ids = set(ts.id for ts in timeslots)

        # Decision variables: X for placements, Y for activation
        X = {}
        for pod_id, pod_placements in placements.items():
            for flv_id, ts_id, _ in pod_placements:
                X[(pod_id, flv_id, ts_id)] = model.NewBoolVar(f"x_{pod_id}_{flv_id}_{ts_id}")

        Y = {}
        for flv in flavours:
            for ts in timeslots:
                Y[(flv.id, ts.id)] = model.NewBoolVar(f"y_{flv.id}_{ts.id}")

        # Assignment constraints with fixed unplaced set
        for pod_id, pod_placements in placements.items():
            if pod_id in unplaced_phase1:
                # Force unplaced: sum X = 0
                placement_vars = [X[(pod_id, flv_id, ts_id)] for flv_id, ts_id, _ in pod_placements]
                if placement_vars:
                    model.Add(sum(placement_vars) == 0)
            else:
                placement_vars = [X[(pod_id, flv_id, ts_id)] for flv_id, ts_id, _ in pod_placements]
                if placement_vars:
                    model.Add(sum(placement_vars) == 1)

        # Capacity and activation linking
        # Build per (flv,slot) usage
        cpu_usage = {}
        ram_usage = {}
        for pod_id, pod_placements in placements.items():
            if pod_id in unplaced_phase1:
                continue
            pod_obj = next((p for p in self.pending_pods if p.id == pod_id), None)
            if not pod_obj:
                continue
            for flv_id, ts_id, _ in pod_placements:
                for offset in range(int(pod_obj.duration)):
                    slot = ts_id + offset
                    if slot >= max_time_slots:
                        break
                    key = (flv_id, slot)
                    cpu_usage.setdefault(key, []).append((pod_obj.cpuRequest, X[(pod_id, flv_id, ts_id)]))
                    ram_usage.setdefault(key, []).append((pod_obj.ramRequest, X[(pod_id, flv_id, ts_id)]))

        # Add constraints
        for (flv_id, slot), terms in cpu_usage.items():
            if flv_id in leftover_cpu and slot in leftover_cpu[flv_id]:
                cpu_limit = leftover_cpu[flv_id][slot]
                model.Add(sum(int(coeff * 1000) * var for coeff, var in terms) <= int(cpu_limit * 1000))
                # Activation linking: if any CPU used then y=1 (via big-M style)
                M = int(cpu_limit * 1000)
                model.Add(sum(int(coeff * 1000) * var for coeff, var in terms) <= M * Y[(flv_id, slot)])

        for (flv_id, slot), terms in ram_usage.items():
            if flv_id in leftover_ram and slot in leftover_ram[flv_id]:
                ram_limit = leftover_ram[flv_id][slot]
                # Keep RAM in MB; assume requests already MB
                model.Add(sum(int(coeff) * var for coeff, var in terms) <= int(ram_limit))

        # Objective: original carbon objective or WaterWise-style scalarized objective.
        if getattr(self, "phase2_objective_mode", "carbon") == "waterwise-scalarized":
            obj_terms = self._build_waterwise_scalarized_terms(
                placements=placements,
                flavours=flavours,
                timeslots=timeslots,
                placement_vars=X,
                use_cpsat=True,
                excluded_pods=set(unplaced_phase1),
            )
        else:
            obj_terms = []
            for pod_id, pod_placements in placements.items():
                if pod_id in unplaced_phase1:
                    continue
                pod_obj = next((p for p in self.pending_pods if p.id == pod_id), None)
                if not pod_obj:
                    continue
                for flv_id, ts_id, _ in pod_placements:
                    flv_obj = flv_by_id.get(flv_id)
                    if not flv_obj:
                        continue
                    k_watts = (flv_obj.power.get('max', 0.0) - flv_obj.power.get('active', 0.0))
                    total_cpu = flv_obj.totalCpu if flv_obj.totalCpu else 1e-6
                    u = pod_obj.cpuRequest / total_cpu
                    for offset in range(int(pod_obj.duration)):
                        slot = ts_id + offset
                        if slot >= max_time_slots:
                            break
                        intensity = flv_obj.forecast.get(slot, 200.0)
                        coef_milli = int(round(self.weight_dynamic * intensity * (k_watts * u) / 1000.0 * 1000))  # milli-grams
                        obj_terms.append(coef_milli * X[(pod_id, flv_id, ts_id)])

            for flv in flavours:
                idle_w = flv.power.get('idle', 0.0)
                emb_per_h = (flv.embodiedCarbon / flv.lifetime) if flv.lifetime else 0.0
                for ts in timeslots:
                    intensity = flv.forecast.get(ts.id, 200.0)
                    activation_g = intensity * (idle_w / 1000.0)
                    if not getattr(self, 'operational_only', False):
                        activation_g += emb_per_h
                    coef_milli = int(round(self.weight_activation * activation_g * 1000))
                    obj_terms.append(coef_milli * Y[(flv.id, ts.id)])

        if self.phase2_water_budget is not None:
            water_dynamic_terms, water_activation_terms = self._build_water_linear_terms(
                placements=placements,
                flavours=flavours,
                timeslots=timeslots,
                placement_vars=X,
                activation_vars=Y,
                max_time_slots=max_time_slots,
                use_cpsat=True,
                excluded_pods=set(unplaced_phase1),
            )
            water_budget_scaled = int(round(float(self.phase2_water_budget) * 1000.0))
            model.Add(sum(water_dynamic_terms) + sum(water_activation_terms) <= water_budget_scaled)

        model.Minimize(sum(obj_terms))

        # Time limit
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(time_limit_sec)
        solver.parameters.num_search_workers = 8
        # Apply relative gap early-stopping when supported by this OR-Tools version
        if gap_tolerance is not None and gap_tolerance > 0:
            try:
                # Newer OR-Tools versions
                solver.parameters.relative_gap_limit = float(gap_tolerance)
            except AttributeError:
                # Older OR-Tools: no relative-gap support; fall back to time-limit only
                import logging
                logging.warning(
                    "CP-SAT SatParameters has no 'relative_gap_limit' field; "
                    "skipping gap-based early stopping for Phase 2."
                )

        status = solver.Solve(model)
        ok = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        solution = {}
        if ok:
            for (pod_id, flv_id, ts_id), var in X.items():
                if solver.BooleanValue(var):
                    solution[pod_id] = (flv_id, ts_id, 0.0)
        return ok, solution, time.time() - start

    def setup_session_placement_log(self):
        """Initializes the session-wide CSV file for logging placements."""
        if not self._session_log_dir:
            logging.error("Session log directory not set. Cannot initialize CSV logging for global optimal algorithm.")
            # Fallback to a default local directory to prevent crashes if not set via main.py
            self._session_log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "analysis", "global_optimal_fallback_logs_session"))
            GlobalOptimalAlgorithm._ensure_dir_exists(self._session_log_dir)
            logging.warning(f"Using fallback session log directory: {self._session_log_dir}")

        self._session_csv_path = os.path.join(self._session_log_dir, self.SESSION_CSV_FILENAME)
        logging.info(f"📊 Setting up CSV log file at: {self._session_csv_path}")
        
        self.close_csv()  # Close any existing file handle before opening a new one or reopening

        file_exists = os.path.exists(self._session_csv_path)
        is_empty = os.path.getsize(self._session_csv_path) == 0 if file_exists else True

        try:
            # Open in append mode, create if not exists
            self._csv_file_handle = open(self._session_csv_path, 'a', newline='')
            self._csv_writer = csv.writer(self._csv_file_handle)

            if not file_exists or is_empty:
                self._csv_writer.writerow(self.CSV_HEADERS)
                logging.info(f"✅ Initialized new placement CSV with headers: {self._session_csv_path}")
                logging.info(f"📋 CSV Headers: {', '.join(self.CSV_HEADERS)}")
            else:
                logging.info(f"✅ Appending to existing placement CSV: {self._session_csv_path}")
                
                # Check if file is writable
                try:
                    # Try writing a test row and then removing it (in a temporary copy of the file)
                    import tempfile
                    import shutil
                    
                    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', newline='')
                    shutil.copy2(self._session_csv_path, temp_file.name)
                    
                    with open(temp_file.name, 'a', newline='') as test_file:
                        test_writer = csv.writer(test_file)
                        test_writer.writerow(["TEST_WRITE"] * len(self.CSV_HEADERS))
                    
                    os.unlink(temp_file.name)
                    logging.info(f"✅ Verified CSV file is writable")
                except Exception as test_e:
                    logging.error(f"⚠️ CSV file might not be writable: {test_e}")
            
            self._csv_file_handle.flush()  # Ensure headers are written if it's a new file
            logging.info(f"✅ CSV file ready for recording placements")

        except IOError as e:
            logging.error(f"❌ Failed to open or write to session CSV file: {self._session_csv_path}. Error: {e}")
            import traceback
            logging.error(traceback.format_exc())
            self._csv_writer = None
            self._csv_file_handle = None
            self._session_csv_path = None

    def close_csv(self):
        if self._csv_file_handle:
            try:
                self._csv_file_handle.close()
            except Exception as e:
                logging.error(f"Error closing CSV file: {e}")
            self._csv_file_handle = None
            self._csv_writer = None

    def _build_logging_footprint(
        self,
        pod: CarbonAwarePod,
        flavour: EnvironmentalFlavor,
        start_slot: int,
        used_cpu_before_by_slot: Optional[Dict[int, float]] = None,
    ) -> FootprintVector:
        return compute_footprint_vector(
            flavour=flavour,
            start_slot=start_slot,
            pod=pod,
            used_cpu_before_by_slot=used_cpu_before_by_slot,
            embodied_allocation_mode=self.embodied_allocation_mode,
            operational_only=self.operational_only,
            use_pod_power_only=self.operational_only,
        )

    def _write_placement_to_csv(self, pod_id: str, node_id: str, start_slot: int, duration: float,
                                cpu_request: float, ram_request: float, total_carbon_emissions: float,
                                solver_status: str, solver_iterations: int, solution_time_seconds: float,
                                embodied_mode: str = "proportional",
                                flavour: Optional[EnvironmentalFlavor] = None,
                                footprint: Optional[FootprintVector] = None):
        """
        Write a placement decision to the CSV file.
        
        Args:
            pod_id: ID of the pod being placed
            node_id: ID of the node selected for placement
            start_slot: Starting timeslot ID
            duration: Pod runtime duration in hours
            cpu_request: CPU request for the pod
            ram_request: RAM request for the pod in MB
            total_carbon_emissions: Total carbon emissions for this placement in kg CO2e
            solver_status: Status of the solver (e.g., "Optimal")
            solver_iterations: Number of solver iterations
            solution_time_seconds: Time taken to solve the problem in seconds
        """
        if self._csv_writer and self._csv_file_handle:
            try:
                csv_fields = footprint.as_csv_fields() if footprint else FootprintVector().finalize().as_csv_fields()
                row = [
                    pod_id, node_id, start_slot, duration,
                    cpu_request, ram_request,
                    getattr(flavour, "region", "") if flavour else "",
                    getattr(flavour, "country", "") if flavour else "",
                    csv_fields["operational_energy_kwh"],
                    csv_fields["operational_carbon_kg"],
                    csv_fields["embodied_carbon_kg"],
                    total_carbon_emissions,
                    csv_fields["direct_water_l"],
                    csv_fields["indirect_water_l"],
                    csv_fields["embodied_water_l"],
                    csv_fields["total_raw_water_l"],
                    csv_fields["scarcity_characterized_water"],
                    csv_fields["criticality_adjusted_water"],
                    solver_status, solver_iterations, solution_time_seconds,
                    embodied_mode
                ]
                
                # Add detailed logging
                logging.info(f"📝 Writing CSV row for pod {pod_id}:")
                logging.info(f"   - Placement: Node={node_id}, Start={start_slot}, Duration={duration}h")
                logging.info(f"   - Resources: CPU={cpu_request:.2f}, RAM={ram_request:.0f}MB")
                logging.info(f"   - Emissions: {total_carbon_emissions:.6f}kgCO2e ({total_carbon_emissions*1000.0:.2f}gCO2e)")
                
                # Write to CSV
                self._csv_writer.writerow(row)
                self._csv_file_handle.flush()  # Ensure data is written to disk
                
                # Verify the data was written
                import os
                if hasattr(self, '_session_csv_path') and os.path.exists(self._session_csv_path):
                    file_size = os.path.getsize(self._session_csv_path)
                    logging.debug(f"   - CSV file size after write: {file_size} bytes")
                
                logging.info(f"✅ Successfully wrote placement data for pod {pod_id}")
            except Exception as e:
                import traceback
                logging.error(f"❌ Error writing to CSV: {e}")
                logging.error(traceback.format_exc())

    def initialize_for_precomputation(self, timeslots_dir: str):
        """
        Initialize the algorithm for pre-computation mode using timeslot YAML files.

        Args:
            timeslots_dir: Directory containing timeslot_*.yaml files
        """
        self.timeslots_dir = timeslots_dir
        self.load_all_timeslot_files()
        logging.info(f"Initialized global optimizer with {len(self.timeslot_data)} timeslots from {timeslots_dir}")

    def load_all_timeslot_files(self):
        """Load all timeslot_*.yaml files from the specified directory"""
        if not self.timeslots_dir or not os.path.isdir(self.timeslots_dir):
            logging.error(f"Invalid timeslots directory: {self.timeslots_dir}")
            return

        # Find all timeslot_*.yaml files
        yaml_files = [f for f in os.listdir(self.timeslots_dir) if f.startswith("timeslot_") and f.endswith(".yaml")]
        yaml_files.sort()  # Sort to ensure chronological order

        for filename in yaml_files:
            try:
                filepath = os.path.join(self.timeslots_dir, filename)
                with open(filepath, 'r') as f:
                    data = yaml.safe_load(f)
                    # Extract timeslot ID from filename (e.g., timeslot_5.yaml -> 5)
                    timeslot_id = int(filename.replace("timeslot_", "").replace(".yaml", ""))
                    self.timeslot_data[timeslot_id] = data
                    logging.debug(f"Loaded timeslot {timeslot_id} from {filename}")
            except Exception as e:
                logging.error(f"Error loading {filename}: {e}")

    def prepare_optimization_data(self):
        """
        Convert loaded YAML data into optimization inputs (flavours, timeslots, resources).
        This prepares all the data needed for the global optimization.
        """
        if not self.timeslot_data:
            logging.error("No timeslot data loaded. Call load_all_timeslot_files first.")
            return False

        # Create flavours from the first timeslot data (assuming node configs are consistent)
        first_timeslot = next(iter(self.timeslot_data.values()))
        nodes_data = first_timeslot.get("nodes", [])

        self.all_flavours = []
        for node in nodes_data:
            flavour = EnvironmentalFlavor(
                id=node["id"],
                embodiedCarbon=node.get("embodiedCarbon", 0.0),
                lifetime=node.get("lifetime", 87600.0),  # Default 10 years in hours
                totalCpu=node.get("totalCpu", 0.0),
                totalRam=node.get("totalRam", 0.0),
                totalStorage=node.get("totalStorage", 0.0),
                forecast=self._build_forecast_for_node(node["id"])
            )
            attach_water_metadata(
                flavour,
                region=node.get("region", ""),
                hardware_subcategory=node.get("subcategory", ""),
                slot_count=max(len(flavour.forecast), 24),
            )
            self.all_flavours.append(flavour)

        # Create timeslots
        self.all_timeslots = []
        for ts_id in sorted(self.timeslot_data.keys()):
            ts_data = self.timeslot_data[ts_id]
            start_time = ts_data.get("start_time", time.time() + ts_id * 3600)  # Default: current time + hours
            ts = CarbonAwareTimeslot(id=ts_id, start_time=start_time, length=1)  # 1-hour timeslots
            self.all_timeslots.append(ts)

        # Initialize resource state
        for flv in self.all_flavours:
            self.resource_state["cpu"][flv.id] = {}
            self.resource_state["ram"][flv.id] = {}

            # For each timeslot, get the resource availability
            for ts_id, ts_data in self.timeslot_data.items():
                node_data = next((n for n in ts_data.get("nodes", []) if n["id"] == flv.id), None)
                if node_data:
                    self.resource_state["cpu"][flv.id][ts_id] = node_data.get("availableCpu", flv.totalCpu)
                    self.resource_state["ram"][flv.id][ts_id] = node_data.get("availableRam", flv.totalRam)
                else:
                    # If node not found in this timeslot, use default values
                    self.resource_state["cpu"][flv.id][ts_id] = flv.totalCpu
                    self.resource_state["ram"][flv.id][ts_id] = flv.totalRam

        logging.info(f"Prepared optimization data with {len(self.all_flavours)} nodes and {len(self.all_timeslots)} timeslots")
        return True

    def _build_forecast_for_node(self, node_id: str) -> Dict[int, float]:
        """Build a carbon forecast for a node across all timeslots"""
        forecast = {}
        for ts_id, ts_data in self.timeslot_data.items():
            node_data = next((n for n in ts_data.get("nodes", []) if n["id"] == node_id), None)
            if node_data and "carbonIntensity" in node_data:
                forecast[ts_id] = node_data["carbonIntensity"]
            else:
                forecast[ts_id] = 0.0  # Default if not specified
        return forecast

    def precompute_global_optimization(self, pods_list: List[CarbonAwarePod]):
        """
        Precompute the global optimization for all pods against all timeslots.

        Args:
            pods_list: List of pods to optimize placement for

        Returns:
            Success flag indicating if optimization completed successfully
        """
        if not self.prepare_optimization_data():
            return False

        # Store the pods and set earliest_timeslot for each pod
        self.pending_pods = pods_list.copy()
        # Set earliest_timeslot for all pods
        for pod in self.pending_pods:
            self._set_pod_earliest_timeslot(pod)
        
        # Set earliest_timeslot for each pod before optimization
        for pod in self.pending_pods:
            self._set_pod_earliest_timeslot(pod)

        # 🚨 CRITICAL: Use thread-safe lock to prevent race conditions
        if not self._solving_lock.acquire(blocking=False):
            logging.warning("🚫 Solver already running at main call site. Skipping to prevent race condition.")
            return

        try:
            # Set precomputation mode flag - CSV should be written only during precomputation
            self.is_precomputation_mode = True
            logging.info(f"🔧 Entering precomputation mode - CSV will be written for all placements")
            
            # Run the global optimization with our prepared data
            max_timeslots = max(ts.id for ts in self.all_timeslots) + 1
            self.solve_global_optimization(
                self.all_flavours,
                self.all_timeslots,
                self.resource_state["cpu"],
                self.resource_state["ram"],
                max_timeslots
            )

            # Mark as done and exit precomputation mode
            self.optimization_done = self.has_solved
            self.is_precomputation_mode = False
            logging.info(f"🔧 Exiting precomputation mode")
        finally:
            # 🚨 CRITICAL: Always release the lock
            self._solving_lock.release()
            logging.debug("🔓 Solver lock released at main call site")

        # Log results
        if self.optimization_done:
            logging.info(f"Precomputation complete! Optimized {len(self.global_solution)}/{len(pods_list)} placements")
            # Show sample of results
            if self.global_solution:
                sample = list(self.global_solution.items())[:3]  # Show first 3
                for pod_id, (node, ts, emissions) in sample:
                    logging.info(f"Sample placement: Pod {pod_id} -> Node {node}, Timeslot {ts}, Emissions {emissions/1000.0:.6f}kgCO2e ({emissions:.2f}gCO2e)")
        else:
            logging.error("Precomputation failed!")

        return self.optimization_done

    @property
    def name(self) -> str:
        return "Carbon-Aware-GlobalOptimal"

    def solve_global_optimization(
        self,
        flavours: List[EnvironmentalFlavor],
        timeslots: List[CarbonAwareTimeslot],
        leftover_cpu: Dict[str, Dict[int, float]],
        leftover_ram: Dict[str, Dict[int, float]],
        max_time_slots: int = 48
    ) -> None:
        """
        Solve the global optimization problem for multiple pods at once.
        
        Args:
            flavours: List of available flavours (nodes)
            timeslots: List of available timeslots
            leftover_cpu: Available CPU resources for each node and timeslot
            leftover_ram: Available RAM resources for each node and timeslot
            max_time_slots: Maximum number of timeslots to consider
        """
        import traceback
        import time
        import os
        
        # Log detailed information about the inputs
        logging.info(f"====== GLOBAL OPTIMIZATION SOLVER START ======")
        logging.info(f"Input parameters:")
        logging.info(f"- Flavours (nodes): {len(flavours)}")
        logging.info(f"- Timeslots: {len(timeslots)} (max: {max_time_slots})")
        logging.info(f"- Pending pods: {len(self.pending_pods)}")
        
        # Log CSV status before starting optimization
        if hasattr(self, '_session_csv_path') and self._session_csv_path:
            if os.path.exists(self._session_csv_path):
                logging.info(f"CSV file exists at: {self._session_csv_path}")
                file_size = os.path.getsize(self._session_csv_path)
                logging.info(f"- Current file size: {file_size} bytes")
            else:
                logging.warning(f"CSV file path is set but file doesn't exist: {self._session_csv_path}")
        else:
            logging.warning("CSV file path not set - results may not be logged correctly")
        
        try:
            import pulp
            
            # Filter out already processed pods to prevent duplicates
            original_pending_count = len(self.pending_pods)
            self.pending_pods = [pod for pod in self.pending_pods if pod.id not in self._processed_pods]
            filtered_count = len(self.pending_pods)
            
            if original_pending_count != filtered_count:
                logging.info(f"🔍 Filtered out {original_pending_count - filtered_count} already processed pods")
                logging.info(f"  - Original pending: {original_pending_count}, After filter: {filtered_count}")

            # Ensure we have pods to place
            if not self.pending_pods:
                logging.warning("🚫 No new pending pods to place (all already processed). Exiting solver.")
                return

            # Create the LP problem
            logging.info(f"🔍 STEP 1: Setting up MILP problem")
            prob = pulp.LpProblem("CarbonAwareScheduling", pulp.LpMinimize)

            # Decision variables dictionary
            x = {}

            # Placement options for each pod
            placements = {}  # pod_id -> list of (flv_id, ts_id, emissions)
            valid_placement_count = 0  # Track total number of valid placement options across all pods

            logging.info(f"🔍 STEP 2: Finding valid placement options for {len(self.pending_pods)} pods")
            # For each pod, find valid placement options
            for pod_index, pod in enumerate(self.pending_pods):
                valid_options = 0  # Track valid placements for this pod
                valid_timeslots = 0  # Track valid timeslots for this pod
                pod_placements = []  # List of valid (flv_id, ts_id, emissions) for this pod
                
                # Make sure earliest_timeslot is set
                self._set_pod_earliest_timeslot(pod)
                
                logging.info(f"  ➡️ Processing pod {pod_index+1}/{len(self.pending_pods)}: {pod.id}")
                logging.info(f"     - Requirements: CPU={pod.cpuRequest}, RAM={pod.ramRequest}, Duration={pod.duration}h")
                logging.info(f"     - Constraints: earliest_timeslot={pod.earliest_timeslot}, deadline_slot={pod.deadline_slot}")

                # For each timeslot, check if it's within the deadline
                for ts in timeslots:
                    # Skip timeslots before the pod's earliest_timeslot
                    if ts.id < pod.earliest_timeslot:
                        continue

                    # Skip timeslots that would make the pod miss its deadline
                    deadline_slot = pod.deadline_slot
                    if deadline_slot is not None and ts.id + pod.duration > deadline_slot:
                        logging.debug(f"     - Skipping timeslot {ts.id} for pod {pod.id}: beyond deadline slot {deadline_slot}")
                        continue

                    valid_timeslots += 1

                    # For each flavour (node), check if it has enough resources
                    for flv in flavours:
                        if pod.duration < 1:
                            logging.warning(f"     - Pod {pod.id} has duration < 1 hour ({pod.duration}h), setting to 1h minimum")
                            pod.duration = 1

                        # Check if the pod can fit in this flavour over its entire duration
                        resource_issues = []
                        duration_feasible = True

                        for offset in range(int(pod.duration)):
                            current_slot = ts.id + offset
                            if current_slot >= max_time_slots:
                                duration_feasible = False
                                resource_issues.append(f"Timeslot {current_slot} beyond scheduling horizon")
                                break

                            # Check CPU
                            if flv.id not in leftover_cpu or current_slot not in leftover_cpu[flv.id]:
                                duration_feasible = False
                                resource_issues.append(f"No CPU data for slot {current_slot}")
                                break

                            if leftover_cpu[flv.id][current_slot] < pod.cpuRequest:
                                duration_feasible = False
                                resource_issues.append(f"CPU {leftover_cpu[flv.id][current_slot]} < {pod.cpuRequest}")
                                break

                            # Check RAM
                            if flv.id not in leftover_ram or current_slot not in leftover_ram[flv.id]:
                                duration_feasible = False
                                resource_issues.append(f"No RAM data for slot {current_slot}")
                                break

                            if leftover_ram[flv.id][current_slot] < pod.ramRequest:
                                duration_feasible = False
                                resource_issues.append(f"RAM {leftover_ram[flv.id][current_slot]} < {pod.ramRequest}")
                                break

                        if duration_feasible:
                            # Calculate emissions for this placement
                            try:
                                # Use proportional embodied allocation by default for consistency
                                used_cpu_before = {}
                                for slot_offset in range(int(pod.duration)):
                                    slot_id = ts.id + slot_offset
                                    total_capacity = flv.totalCpu
                                    leftover = leftover_cpu[flv.id][slot_id]
                                    used_cpu_before[slot_id] = max(total_capacity - leftover, 0.0)
                                emissions = compute_emissions(
                                    flv, ts.id, pod,
                                    used_cpu_before_by_slot=used_cpu_before,
                                    embodied_allocation_mode="proportional",
                                )
                                logging.debug(f"     - Valid placement: pod={pod.id}, node={flv.id}, ts={ts.id}, emissions={emissions/1000.0:.6f}kgCO2e ({emissions:.2f}gCO2e)")
                                
                                placement_key = (pod.id, flv.id, ts.id)
                                # Store emissions for optimization and reporting
                                pod_placements.append((flv.id, ts.id, emissions))
                                valid_options += 1
                                valid_placement_count += 1

                                # Create decision variable
                                x[placement_key] = pulp.LpVariable(f"x_{pod.id}_{flv.id}_{ts.id}",
                                                                 cat=pulp.LpBinary)
                            except Exception as e:
                                logging.error(f"     - Error computing emissions for pod {pod.id} on {flv.id} at ts={ts.id}: {e}")
                                logging.error(traceback.format_exc())
                        else:
                            if valid_options < 10:  # Only log first 10 invalid options to avoid log spam
                                logging.debug(f"     - Invalid placement on {flv.id} at ts={ts.id}: {resource_issues}")

                # Optional pruning: keep only greenest candidates per pod to shrink MILP
                if self.pruning_enable and pod_placements:
                    # Sort by total emissions ascending (greener first)
                    sorted_opts = sorted(pod_placements, key=lambda t: t[2])
                    best = sorted_opts[0][2]
                    # Keep those within (1 + margin) of best and cap by top_k
                    margin = best * (1.0 + max(0.0, self.pruning_emissions_margin))
                    pruned = [opt for opt in sorted_opts if opt[2] <= margin]
                    if self.pruning_top_k_per_pod > 0:
                        pruned = pruned[: self.pruning_top_k_per_pod]
                    removed = max(0, len(pod_placements) - len(pruned))
                    if removed > 0:
                        logging.info(f"     - Pruned {removed} high-emission options for pod {pod.id}; kept {len(pruned)}")
                    pod_placements = pruned

                placements[pod.id] = pod_placements
                logging.info(f"     - Result: {len(pod_placements)} valid placements across {valid_timeslots} valid timeslots (after pruning if enabled)")

            # If no placements were found at all, exit early
            if not x:
                logging.warning("🚫 No valid placements found for any pod! Exiting solver.")
                return

            logging.info(f"✅ Found {valid_placement_count} valid placement options for {len(self.pending_pods)} pods")

            logging.info(f"🔍 STEP 3: Setting up multi-objective optimization (placement + emissions)")
            
            # Create slack variables for unplaced pods
            logging.info(f"  - Creating slack variables for {len(placements)} pods (multi-objective)")
            s = {}  # s[pod_id] = 1 if pod is NOT placed, 0 if placed
            for pod_id in placements:
                s[pod_id] = pulp.LpVariable(f"unplaced_{pod_id}", cat=pulp.LpBinary)
            
            # Assignment constraints: each pod must be placed exactly once or marked unplaced
            for pod_id, pod_placements in placements.items():
                if pod_placements:
                    prob += (
                        pulp.lpSum([x[(pod_id, flv_id, ts_id)] for flv_id, ts_id, _ in pod_placements]) + s[pod_id] == 1
                    ), f"ASSIGN_{pod_id}"
                else:
                    # If no feasible placements exist for this pod, force it to be unplaced
                    prob += (s[pod_id] == 1), f"ASSIGN_NOPLACE_{pod_id}"
            # Optional success-rate floor: cap number of unplaced pods
            try:
                constraints_cfg = self.config.get('optimization', {}).get('constraints', {}) if hasattr(self, 'config') else {}
                max_unplaced_fraction = constraints_cfg.get('max_unplaced_fraction', None)
                max_unplaced_abs = constraints_cfg.get('max_unplaced', None)
                if max_unplaced_fraction is not None or max_unplaced_abs is not None:
                    total_pods = len(s)
                    allowed_unplaced = None
                    if isinstance(max_unplaced_abs, int):
                        allowed_unplaced = max(0, int(max_unplaced_abs))
                    if allowed_unplaced is None and max_unplaced_fraction is not None:
                        try:
                            frac = float(max_unplaced_fraction)
                        except Exception:
                            frac = None
                        if frac is not None:
                            allowed_unplaced = max(0, int(frac * total_pods))
                    if allowed_unplaced is not None:
                        logging.info(f"  - Applying success-rate floor: sum(s) <= {allowed_unplaced} (of {total_pods})")
                        prob += pulp.lpSum([s[p] for p in s]) <= allowed_unplaced, "CAP_UNPLACED_PODS"
            except Exception as _e:
                logging.warning(f"  - Unable to apply success-rate floor constraint: {_e}")
            
            # Lexicographic two-phase solve (if enabled): Phase 1 maximize placements, Phase 2 minimize emissions
            if getattr(self, 'use_lexicographic', False):
                logging.info("🔀 Using lexicographic optimization: Phase 1 (maximize placements), Phase 2 (minimize emissions)")

                # -----------------
                # Phase 1: minimize number of unplaced pods (sum s)
                # -----------------
                prob1 = pulp.LpProblem("CarbonAwareScheduling_Phase1", pulp.LpMinimize)

                # Add assignment constraints to prob1
                for cname, c in list(prob.constraints.items()):
                    # Recreate assignment constraints for prob1 from stored definitions above
                    # Note: rebuild from s/x since constraints were added to 'prob'; reconstruct them explicitly
                    pass

                # Rebuild assignment constraints for prob1
                for pod_id, pod_placements in placements.items():
                    if pod_placements:
                        prob1 += (
                            pulp.lpSum([x[(pod_id, flv_id, ts_id)] for flv_id, ts_id, _ in pod_placements]) + s[pod_id] == 1
                        ), f"ASSIGN_{pod_id}_P1"
                    else:
                        prob1 += (s[pod_id] == 1), f"ASSIGN_NOPLACE_{pod_id}_P1"

                # Build resource capacity expressions once
                cpu_usage = {}
                ram_usage = {}
                for pod_id, pod_placements in placements.items():
                    pod_obj = next((p for p in self.pending_pods if p.id == pod_id), None)
                    if not pod_obj:
                        continue
                    for flv_id, ts_id, _ in pod_placements:
                        for offset in range(int(pod_obj.duration)):
                            slot = ts_id + offset
                            if slot >= max_time_slots:
                                break
                            cpu_key = (flv_id, slot)
                            ram_key = (flv_id, slot)
                            cpu_usage.setdefault(cpu_key, []).append(pod_obj.cpuRequest * x[(pod_id, flv_id, ts_id)])
                            ram_usage.setdefault(ram_key, []).append(pod_obj.ramRequest * x[(pod_id, flv_id, ts_id)])

                # Add capacity constraints to prob1
                for (flv_id, ts_id), usage_expressions in cpu_usage.items():
                    if flv_id in leftover_cpu and ts_id in leftover_cpu[flv_id]:
                        cpu_limit = leftover_cpu[flv_id][ts_id]
                        prob1 += pulp.lpSum(usage_expressions) <= cpu_limit, f"CPU_{flv_id}_{ts_id}_P1"
                for (flv_id, ts_id), usage_expressions in ram_usage.items():
                    if flv_id in leftover_ram and ts_id in leftover_ram[flv_id]:
                        ram_limit = leftover_ram[flv_id][ts_id]
                        prob1 += pulp.lpSum(usage_expressions) <= ram_limit, f"RAM_{flv_id}_{ts_id}_P1"

                # Objective phase 1: minimize number of unplaced pods
                prob1 += pulp.lpSum([s[pod_id] for pod_id in s])

                # Solve Phase 1
                solver_cfg = self.config.get('optimization', {}).get('solver', {}) if hasattr(self, 'config') else {}
                time_limit_cfg = solver_cfg.get('time_limit', 60)
                gap_tolerance_cfg = solver_cfg.get('gap_tolerance', 0.05)
                threads_cfg = int(solver_cfg.get('threads', 4))
                logging.info(f"  - [Phase 1] settings: time_limit={time_limit_cfg}s, gap_tolerance={gap_tolerance_cfg}, threads={threads_cfg}")
                solver_start_time = time.time()
                primary_solver = self._get_pulp_solver(solver_cfg, time_limit_cfg, gap_tolerance_cfg, threads_cfg)
                prob1.solve(primary_solver)
                phase1_time = time.time() - solver_start_time
                status1 = pulp.LpStatus[prob1.status]
                logging.info(f"  - [Phase 1] completed in {phase1_time:.3f}s, status={status1}")
                if status1 not in ("Optimal", "OptimalInfeasible"):
                    logging.warning("  - [Phase 1] did not find optimal solution; proceeding with found placement count")

                # Identify exact unplaced set from Phase 1 to shrink Phase 2 search space
                unplaced_phase1 = {pod_id for pod_id in s if s[pod_id].value() and s[pod_id].value() > 0.5}
                placed_phase1 = set(s.keys()) - unplaced_phase1
                opt_unplaced = len(unplaced_phase1)
                logging.info(f"  - [Phase 1] Optimal unplaced pods: {opt_unplaced} (placed={len(s)-opt_unplaced}/{len(s)})")

                solution_dict_phase1: Dict[str, Tuple[str, int, float]] = {}
                for pod_id in placed_phase1:
                    for flv_id, ts_id, _ in placements.get(pod_id, []):
                        var = x.get((pod_id, flv_id, ts_id))
                        if var is not None and var.value() and var.value() > 0.5:
                            solution_dict_phase1[pod_id] = (flv_id, ts_id, 0.0)
                            break

                # -----------------
                # Phase 2: minimize activated node-hours (packing) with fixed unplaced pods
                # -----------------
                prob2 = pulp.LpProblem("CarbonAwareScheduling_Phase2", pulp.LpMinimize)

                # Recreate assignment constraints for prob2
                for pod_id, pod_placements in placements.items():
                    if pod_placements:
                        prob2 += (
                            pulp.lpSum([x[(pod_id, flv_id, ts_id)] for flv_id, ts_id, _ in pod_placements]) + s[pod_id] == 1
                        ), f"ASSIGN_{pod_id}_P2"
                    else:
                        prob2 += (s[pod_id] == 1), f"ASSIGN_NOPLACE_{pod_id}_P2"

                # Build activation variables for prob2
                y = {}
                for flv in flavours:
                    for ts in timeslots:
                        y[(flv.id, ts.id)] = pulp.LpVariable(f"y_{flv.id}_{ts.id}", cat=pulp.LpBinary)

                # Capacity constraints and activation linking for prob2
                for (flv_id, ts_id), usage_expressions in cpu_usage.items():
                    if flv_id in leftover_cpu and ts_id in leftover_cpu[flv_id]:
                        cpu_limit = leftover_cpu[flv_id][ts_id]
                        prob2 += pulp.lpSum(usage_expressions) <= cpu_limit, f"CPU_{flv_id}_{ts_id}_P2"
                        M_cpu = cpu_limit if cpu_limit > 0 else 1.0
                        prob2 += pulp.lpSum(usage_expressions) <= M_cpu * y[(flv_id, ts_id)], f"ACTIVATION_CPU_LINK_{flv_id}_{ts_id}_P2"
                for (flv_id, ts_id), usage_expressions in ram_usage.items():
                    if flv_id in leftover_ram and ts_id in leftover_ram[flv_id]:
                        ram_limit = leftover_ram[flv_id][ts_id]
                        prob2 += pulp.lpSum(usage_expressions) <= ram_limit, f"RAM_{flv_id}_{ts_id}_P2"

                # Fix the exact unplaced set from Phase 1 (stronger than fixing only the count)
                for pod_id in unplaced_phase1:
                    prob2 += (s[pod_id] == 1), f"FIX_S_UNPLACED_{pod_id}"
                for pod_id in placed_phase1:
                    prob2 += (s[pod_id] == 0), f"FIX_S_PLACED_{pod_id}"

                # Objective Phase 2: original carbon objective or WaterWise-style scalarized baseline.
                if getattr(self, "phase2_objective_mode", "carbon") == "waterwise-scalarized":
                    waterwise_terms = self._build_waterwise_scalarized_terms(
                        placements=placements,
                        flavours=flavours,
                        timeslots=timeslots,
                        placement_vars=x,
                        use_cpsat=False,
                        excluded_pods=set(unplaced_phase1),
                    )
                    prob2 += pulp.lpSum(waterwise_terms)
                    logging.info("  - [Phase 2] Objective: WaterWise-style normalized scalarized carbon-water objective")
                else:
                    dynamic_terms = []
                    for pod_id, pod_placements in placements.items():
                        pod_obj = next((p for p in self.pending_pods if p.id == pod_id), None)
                        if not pod_obj:
                            continue
                        for flv_id, ts_id, _ in pod_placements:
                            flv_obj = next((f for f in flavours if f.id == flv_id), None)
                            if not flv_obj:
                                continue
                            k_watts = compute_node_dynamic_coeff_watts(flv_obj)
                            total_cpu = flv_obj.totalCpu if flv_obj.totalCpu else 1e-6
                            u_pod = pod_obj.cpuRequest / total_cpu
                            for offset in range(int(pod_obj.duration)):
                                slot = ts_id + offset
                                if slot >= max_time_slots:
                                    break
                                intensity = get_carbon_intensity(flv_obj, slot)
                                coef_g = intensity * (k_watts * u_pod / 1000.0)
                                dynamic_terms.append(self.weight_dynamic * coef_g * x[(pod_id, flv_id, ts_id)])

                    idle_embodied_terms = []
                    for flv in flavours:
                        idle_w = flv.power.get('idle', 0.0)
                        embodied_per_h = compute_embodied_per_hour_g(flv)
                        for ts in timeslots:
                            intensity = get_carbon_intensity(flv, ts.id)
                            idle_oper_g = intensity * (idle_w / 1000.0)
                            add_emb = 0.0 if getattr(self, 'operational_only', False) else embodied_per_h
                            idle_embodied_terms.append(self.weight_activation * (idle_oper_g + add_emb) * y[(flv.id, ts.id)])

                    emissions_objective = pulp.lpSum(dynamic_terms) + pulp.lpSum(idle_embodied_terms)
                    prob2 += emissions_objective
                    logging.info("  - [Phase 2] Objective: carbon emissions")

                if self.phase2_water_budget is not None:
                    water_dynamic_terms, water_activation_terms = self._build_water_linear_terms(
                        placements=placements,
                        flavours=flavours,
                        timeslots=timeslots,
                        placement_vars=x,
                        activation_vars=y,
                        max_time_slots=max_time_slots,
                        use_cpsat=False,
                        excluded_pods=set(unplaced_phase1),
                    )
                    water_expression = pulp.lpSum(water_dynamic_terms) + pulp.lpSum(water_activation_terms)
                    prob2 += water_expression <= float(self.phase2_water_budget), "PHASE2_WATER_BUDGET"
                    logging.info(
                        "  - [Phase 2] Applying epsilon-constraint: %s water <= %.6f",
                        self.phase2_water_metric,
                        float(self.phase2_water_budget),
                    )

                # Optional warm start from Phase 1 values (best-effort; supported by some solvers)
                try:
                    for pod_id in s:
                        if hasattr(s[pod_id], 'setInitialValue'):
                            s[pod_id].setInitialValue(1.0 if pod_id in unplaced_phase1 else 0.0)
                    for (pod_id, flv_id, ts_id), var in x.items():
                        if hasattr(var, 'setInitialValue') and var.value() is not None:
                            var.setInitialValue(1.0 if var.value() > 0.5 else 0.0)
                except Exception:
                    pass

                # Solve Phase 2 using selected solver; try CP-SAT when configured
                solver_cfg = self.config.get('optimization', {}).get('solver', {}) if hasattr(self, 'config') else {}
                time_limit_cfg = solver_cfg.get('time_limit', 60)
                gap_tolerance_cfg = solver_cfg.get('gap_tolerance', 0.05)
                threads_cfg = int(solver_cfg.get('threads', 4))
                solution_dict: Dict[str, Tuple[str, int, float]] = dict(solution_dict_phase1)

                if self.solver_name.lower() == 'cp_sat':
                    logging.info(f"  - [Phase 2] Using OR-Tools CP-SAT with time_limit={time_limit_cfg}s")
                    cpsat_ok, cpsat_solution_dict, solution_time = self._solve_phase2_with_cpsat(
                        placements=placements,
                        flavours=flavours,
                        timeslots=timeslots,
                        leftover_cpu=leftover_cpu,
                        leftover_ram=leftover_ram,
                        max_time_slots=max_time_slots,
                        unplaced_phase1=unplaced_phase1,
                        placed_phase1=placed_phase1,
                        time_limit_sec=time_limit_cfg,
                        gap_tolerance=gap_tolerance_cfg
                    )
                    self.status = 'Optimal' if cpsat_ok else 'Not Solved'
                    self.iterations = 0
                    logging.info(f"  - [Phase 2] completed in {solution_time:.3f}s, status={self.status}")
                    # Propagate solution dict for post-processing
                    if cpsat_ok:
                        solution_dict = cpsat_solution_dict
                else:
                    logging.info(f"  - [Phase 2] PuLP settings: time_limit={time_limit_cfg}s, gap_tolerance={gap_tolerance_cfg}, threads={threads_cfg}")
                    solver_start_time = time.time()
                    primary_solver = self._get_pulp_solver(solver_cfg, time_limit_cfg, gap_tolerance_cfg, threads_cfg)
                    prob2.solve(primary_solver)
                    solution_time = time.time() - solver_start_time
                    self.status = pulp.LpStatus[prob2.status]
                    self.iterations = prob2.solverModel.Iterations if hasattr(prob2.solverModel, 'Iterations') else 0
                    logging.info(f"  - [Phase 2] completed in {solution_time:.3f}s, status={self.status}")

                # Determine final status from Phase 2 (emissions)
                if (self.solver_name.lower() == 'cp_sat' and self.status == 'Optimal') or (self.solver_name.lower() != 'cp_sat' and prob2.status == pulp.LpStatusOptimal):
                    final_status = 'Optimal'
                else:
                    logging.warning("❌ Emissions minimization (Phase 2) failed, using Phase 1 (placements) solution")
                    final_status = status1

                # Extract placements and emissions as in standard flow, but use values from x/s/y
                logging.info(f"🔍 STEP 5: Processing solution and recording results")
                if final_status in ("Optimal", "OptimalInfeasible"):
                    solution = {}
                    total_emissions_objective = 0.0

                    pending_pods_dict = {p.id: p for p in self.pending_pods}
                    placements_saved_to_csv = 0

                    # Extract placed pods
                    # Note: for CP-SAT path, solution_dict is already returned from the solver
                    unplaced_pods = []
                    solution_dict_local: Dict[str, Tuple[str, int, float]] = dict(solution_dict)
                    if self.solver_name.lower() == 'cp_sat':
                        unplaced_pods = list(unplaced_phase1)
                        # solution_dict already built by CP-SAT path
                    else:
                        for pod_id, slack_var in s.items():
                            if slack_var.value() and slack_var.value() > 0.5:
                                unplaced_pods.append(pod_id)
                        for var_name, var in x.items():
                            if var.value() and var.value() > 0.5:
                                pod_id, flv_id, ts_id = var_name
                                solution_dict_local[pod_id] = (flv_id, ts_id, 0.0)

                    pod_footprints = self._compute_solution_footprints(
                        solution_dict=solution_dict_local,
                        pending_pods_dict=pending_pods_dict,
                        flavours=flavours,
                        max_time_slots=max_time_slots,
                    )

                    for pod_id_sol, (flv_id_sol, ts_id_sol, _) in solution_dict_local.items():
                        current_pod = pending_pods_dict.get(pod_id_sol)
                        if not current_pod:
                            continue
                        current_flavour = next((flv for flv in flavours if flv.id == flv_id_sol), None)
                        footprint = pod_footprints.get(pod_id_sol, FootprintVector().finalize())
                        emissions_sol_g = footprint.total_carbon_g
                        if self.is_precomputation_mode:
                            try:
                                self._write_placement_to_csv(
                                    pod_id=current_pod.id,
                                    node_id=flv_id_sol,
                                    start_slot=ts_id_sol,
                                    duration=current_pod.duration,
                                    cpu_request=current_pod.cpuRequest,
                                    ram_request=current_pod.ramRequest,
                                    total_carbon_emissions=emissions_sol_g / 1000.0,
                                    solver_status=str(self.status),
                                    solver_iterations=self.iterations,
                                    solution_time_seconds=solution_time,
                                    embodied_mode="proportional",
                                    flavour=current_flavour,
                                    footprint=footprint,
                                )
                                placements_saved_to_csv += 1
                            except Exception:
                                pass
                        solution[pod_id_sol] = (flv_id_sol, ts_id_sol, emissions_sol_g)
                        total_emissions_objective += emissions_sol_g

                    placement_penalty = 0.0
                    total_objective = prob2.objective.value() if prob2.objective.value() else 0.0
                    logging.info(f"✅ Lexicographic solution found: emissions={total_emissions_objective/1000.0:.6f}kgCO2e, placed={len(solution)}/{len(self.pending_pods)}")

                    self.global_solution.update(solution)
                    for pod_id in solution.keys():
                        self._processed_pods.add(pod_id)
                    self.pending_pods = [p for p in self.pending_pods if p.id not in solution]
                    self.has_solved = True
                else:
                    logging.warning(f"❌ Lexicographic Phase 2 failed, status: {self.status}")
                    if self.phase2_water_budget is not None:
                        logging.warning("🚫 Skipping dynamic-only fallback because epsilon-constraint is active")
                        self.last_solve_time = time.time()
                        logging.info(f"====== GLOBAL OPTIMIZATION (LEXICOGRAPHIC) END ======")
                        logging.critical(f"💡 GLOBAL OPTIMIZATION FAILED! No epsilon-feasible Phase 2 solution found.")
                        return
                    # Fallback: try dynamic-only emissions (drop activation variables and idle/embodied terms)
                    logging.info("🔁 Attempting Phase 2 fallback: dynamic-only objective without activation variables")
                    prob2_dyn = pulp.LpProblem("CarbonAwareScheduling_Phase2_DynamicOnly", pulp.LpMinimize)

                    # Recreate assignment constraints with fixed s from Phase 1
                    for pod_id, pod_placements in placements.items():
                        if pod_placements:
                            prob2_dyn += (
                                pulp.lpSum([x[(pod_id, flv_id, ts_id)] for flv_id, ts_id, _ in pod_placements]) + s[pod_id] == 1
                            ), f"ASSIGN_{pod_id}_P2D"
                        else:
                            prob2_dyn += (s[pod_id] == 1), f"ASSIGN_NOPLACE_{pod_id}_P2D"
                    for pod_id in unplaced_phase1:
                        prob2_dyn += (s[pod_id] == 1), f"FIX_S_UNPLACED_{pod_id}_P2D"
                    for pod_id in placed_phase1:
                        prob2_dyn += (s[pod_id] == 0), f"FIX_S_PLACED_{pod_id}_P2D"

                    # Capacity constraints only (no activation linking)
                    for (flv_id, ts_id), usage_expressions in cpu_usage.items():
                        if flv_id in leftover_cpu and ts_id in leftover_cpu[flv_id]:
                            cpu_limit = leftover_cpu[flv_id][ts_id]
                            prob2_dyn += pulp.lpSum(usage_expressions) <= cpu_limit, f"CPU_{flv_id}_{ts_id}_P2D"
                    for (flv_id, ts_id), usage_expressions in ram_usage.items():
                        if flv_id in leftover_ram and ts_id in leftover_ram[flv_id]:
                            ram_limit = leftover_ram[flv_id][ts_id]
                            prob2_dyn += pulp.lpSum(usage_expressions) <= ram_limit, f"RAM_{flv_id}_{ts_id}_P2D"

                    # Dynamic-only objective with optional surrogate activation penalty
                    dynamic_terms_only = []
                    surrogate_activation_terms = []
                    for pod_id, pod_placements in placements.items():
                        pod_obj = next((p for p in self.pending_pods if p.id == pod_id), None)
                        if not pod_obj:
                            continue
                        for flv_id, ts_id, _ in pod_placements:
                            flv_obj = next((f for f in flavours if f.id == flv_id), None)
                            if not flv_obj:
                                continue
                            k_watts = compute_node_dynamic_coeff_watts(flv_obj)
                            total_cpu = flv_obj.totalCpu if flv_obj.totalCpu else 1e-6
                            u_pod = pod_obj.cpuRequest / total_cpu
                            for offset in range(int(pod_obj.duration)):
                                slot = ts_id + offset
                                if slot >= max_time_slots:
                                    break
                                intensity = get_carbon_intensity(flv_obj, slot)
                                coef_g = intensity * (k_watts * u_pod / 1000.0)
                                dynamic_terms_only.append(self.weight_dynamic * coef_g * x[(pod_id, flv_id, ts_id)])
                                # Surrogate activation: small per-slot penalty proportional to idle+embodied to discourage scattering
                                if self.weight_fallback_surrogate_activation and self.weight_fallback_surrogate_activation > 0.0:
                                    idle_w = flv_obj.power.get('idle', 0.0)
                                    embodied_per_h = 0.0 if getattr(self, 'operational_only', False) else compute_embodied_per_hour_g(flv_obj)
                                    idle_embodied_g = intensity * (idle_w / 1000.0) + embodied_per_h
                                    surrogate_activation_terms.append(
                                        self.weight_fallback_surrogate_activation * idle_embodied_g * x[(pod_id, flv_id, ts_id)]
                                    )
                    if surrogate_activation_terms:
                        prob2_dyn += pulp.lpSum(dynamic_terms_only) + pulp.lpSum(surrogate_activation_terms)
                    else:
                        prob2_dyn += pulp.lpSum(dynamic_terms_only)

                    # Warm start again (best-effort)
                    try:
                        for pod_id in s:
                            if hasattr(s[pod_id], 'setInitialValue'):
                                s[pod_id].setInitialValue(1.0 if pod_id in unplaced_phase1 else 0.0)
                        for (pod_id, flv_id, ts_id), var in x.items():
                            if hasattr(var, 'setInitialValue') and var.value() is not None:
                                var.setInitialValue(1.0 if var.value() > 0.5 else 0.0)
                    except Exception:
                        pass

                    # Solve dynamic-only fallback
                    solver_start_time = time.time()
                    prob2_dyn.solve(primary_solver)
                    solution_time = time.time() - solver_start_time
                    status_dyn = pulp.LpStatus[prob2_dyn.status]
                    logging.info(f"  - [Phase 2 Dynamic-Only] completed in {solution_time:.3f}s, status={status_dyn}")

                    if prob2_dyn.status == pulp.LpStatusOptimal:
                        solution = {}
                        total_emissions_objective = 0.0
                        pending_pods_dict = {p.id: p for p in self.pending_pods}
                        placements_saved_to_csv = 0

                        # Extract placed pods from x
                        solution_dict = {}
                        unplaced_pods = list(unplaced_phase1)
                        for var_name, var in x.items():
                            if var.value() and var.value() > 0.5:
                                pod_id, flv_id, ts_id = var_name
                                if pod_id in placed_phase1:
                                    solution_dict[pod_id] = (flv_id, ts_id, 0.0)

                        pod_footprints = self._compute_solution_footprints(
                            solution_dict=solution_dict,
                            pending_pods_dict=pending_pods_dict,
                            flavours=flavours,
                            max_time_slots=max_time_slots,
                        )

                        for pod_id_sol, (flv_id_sol, ts_id_sol, _) in solution_dict.items():
                            current_pod = pending_pods_dict.get(pod_id_sol)
                            if not current_pod:
                                continue
                            current_flavour = next((flv for flv in flavours if flv.id == flv_id_sol), None)
                            footprint = pod_footprints.get(pod_id_sol, FootprintVector().finalize())
                            emissions_sol_g = footprint.total_carbon_g
                            if self.is_precomputation_mode:
                                try:
                                    self._write_placement_to_csv(
                                        pod_id=current_pod.id,
                                        node_id=flv_id_sol,
                                        start_slot=ts_id_sol,
                                        duration=current_pod.duration,
                                        cpu_request=current_pod.cpuRequest,
                                        ram_request=current_pod.ramRequest,
                                        total_carbon_emissions=emissions_sol_g / 1000.0,
                                        solver_status=str(status_dyn),
                                        solver_iterations=self.iterations,
                                        solution_time_seconds=solution_time,
                                        embodied_mode="proportional",
                                        flavour=current_flavour,
                                        footprint=footprint,
                                    )
                                    placements_saved_to_csv += 1
                                except Exception:
                                    pass
                            solution[pod_id_sol] = (flv_id_sol, ts_id_sol, emissions_sol_g)
                            total_emissions_objective += emissions_sol_g

                        logging.info(f"✅ Lexicographic dynamic-only solution: emissions={total_emissions_objective/1000.0:.6f}kgCO2e, placed={len(solution)}/{len(self.pending_pods)}")
                        self.global_solution.update(solution)
                        for pod_id in solution.keys():
                            self._processed_pods.add(pod_id)
                        self.pending_pods = [p for p in self.pending_pods if p.id not in solution]
                        self.has_solved = True
                    else:
                        logging.warning(f"❌ Phase 2 dynamic-only fallback failed, status: {status_dyn}")
                self.last_solve_time = time.time()
                logging.info(f"====== GLOBAL OPTIMIZATION (LEXICOGRAPHIC) END ======")
                if self.has_solved:
                    logging.critical(f"💡 GLOBAL OPTIMIZATION SUCCESS! Placed {len(self.global_solution)} pods.")
                else:
                    logging.critical(f"💡 GLOBAL OPTIMIZATION FAILED! No solution found.")
                return

            # Multi-objective: Minimize (penalty for unplaced pods + total emissions)
            # Get penalty weight from config, default to 10000 (prioritizes placement over emissions)
            penalty_for_unplaced = self.config.get('optimization', {}).get('unplaced_penalty', 10000)
            logging.info(f"  - Setting multi-objective: penalty for unplaced pods ({penalty_for_unplaced}) + minimize emissions")
            logging.info(f"    * Higher penalty = prioritize placement over emissions")
            logging.info(f"    * Lower penalty = allow more unplaced pods for better emissions")

            # Build node-aware objective with activation variables
            y = {}
            for flv in flavours:
                for ts in timeslots:
                    y[(flv.id, ts.id)] = pulp.LpVariable(f"y_{flv.id}_{ts.id}", cat=pulp.LpBinary)

            objective_terms = []
            # Penalty term
            objective_terms.append(pulp.lpSum([penalty_for_unplaced * s[pod_id] for pod_id in s]))

            if getattr(self, "phase2_objective_mode", "carbon") == "waterwise-scalarized":
                waterwise_terms = self._build_waterwise_scalarized_terms(
                    placements=placements,
                    flavours=flavours,
                    timeslots=timeslots,
                    placement_vars=x,
                    use_cpsat=False,
                )
                if waterwise_terms:
                    objective_terms.append(pulp.lpSum(waterwise_terms))
            else:
                # Dynamic emissions terms (gCO2)
                dynamic_terms = []
                for pod_id, pod_placements in placements.items():
                    pod_obj = next((p for p in self.pending_pods if p.id == pod_id), None)
                    if not pod_obj:
                        continue
                    for flv_id, ts_id, _ in pod_placements:
                        flv_obj = next((f for f in flavours if f.id == flv_id), None)
                        if not flv_obj:
                            continue
                        k_watts = compute_node_dynamic_coeff_watts(flv_obj)
                        total_cpu = flv_obj.totalCpu if flv_obj.totalCpu else 1e-6
                        u_pod = pod_obj.cpuRequest / total_cpu
                        for offset in range(int(pod_obj.duration)):
                            slot = ts_id + offset
                            if slot >= max_time_slots:
                                break
                            intensity = get_carbon_intensity(flv_obj, slot)
                            coef_g = intensity * (k_watts * u_pod / 1000.0)
                            dynamic_terms.append(self.weight_dynamic * coef_g * x[(pod_id, flv_id, ts_id)])
                if dynamic_terms:
                    objective_terms.append(pulp.lpSum(dynamic_terms))

                # Idle + embodied terms per (node,slot)
                idle_embodied_terms = []
                for flv in flavours:
                    idle_w = flv.power.get('idle', 0.0)
                    embodied_per_h = compute_embodied_per_hour_g(flv)
                    for ts in timeslots:
                        intensity = get_carbon_intensity(flv, ts.id)
                        idle_oper_g = intensity * (idle_w / 1000.0)
                        add_emb = 0.0 if getattr(self, 'operational_only', False) else embodied_per_h
                        idle_embodied_terms.append(self.weight_activation * (idle_oper_g + add_emb) * y[(flv.id, ts.id)])
                if idle_embodied_terms:
                    objective_terms.append(pulp.lpSum(idle_embodied_terms))

            prob += pulp.lpSum(objective_terms)

            if self.phase2_water_budget is not None:
                water_dynamic_terms, water_activation_terms = self._build_water_linear_terms(
                    placements=placements,
                    flavours=flavours,
                    timeslots=timeslots,
                    placement_vars=x,
                    activation_vars=y,
                    max_time_slots=max_time_slots,
                    use_cpsat=False,
                )
                water_expression = pulp.lpSum(water_dynamic_terms) + pulp.lpSum(water_activation_terms)
                prob += water_expression <= float(self.phase2_water_budget), "WATER_BUDGET"
                logging.info(
                    "  - Applying epsilon-constraint: %s water <= %.6f",
                    self.phase2_water_metric,
                    float(self.phase2_water_budget),
                )

            logging.info(f"  - Setting up resource capacity constraints and activation linking")
            # Build per-(node,timeslot) resource usage expressions for constraints and activation linking
            cpu_usage = {}
            ram_usage = {}
            for pod_id, pod_placements in placements.items():
                pod_obj = next((p for p in self.pending_pods if p.id == pod_id), None)
                if not pod_obj:
                    continue
                for flv_id, ts_id, _ in pod_placements:
                    for offset in range(int(pod_obj.duration)):
                        slot = ts_id + offset
                        if slot >= max_time_slots:
                            break
                        cpu_key = (flv_id, slot)
                        ram_key = (flv_id, slot)
                        cpu_usage.setdefault(cpu_key, []).append(pod_obj.cpuRequest * x[(pod_id, flv_id, ts_id)])
                        ram_usage.setdefault(ram_key, []).append(pod_obj.ramRequest * x[(pod_id, flv_id, ts_id)])

            constraint_count = 0
            # Add CPU capacity constraints and activation linking
            for (flv_id, ts_id), usage_expressions in cpu_usage.items():
                if flv_id in leftover_cpu and ts_id in leftover_cpu[flv_id]:
                    constraint_count += 1
                    cpu_limit = leftover_cpu[flv_id][ts_id]
                    prob += pulp.lpSum(usage_expressions) <= cpu_limit, f"CPU_{flv_id}_{ts_id}"
                    # Link activation: if any CPU used, y must be 1 (big-M on CPU)
                    M_cpu = cpu_limit if cpu_limit > 0 else 1.0
                    prob += pulp.lpSum(usage_expressions) <= M_cpu * y[(flv_id, ts_id)], f"ACTIVATION_CPU_LINK_{flv_id}_{ts_id}"
                else:
                    logging.warning(f"     - Missing CPU capacity data for node {flv_id}, timeslot {ts_id}")

            # Add RAM capacity constraints
            for (flv_id, ts_id), usage_expressions in ram_usage.items():
                if flv_id in leftover_ram and ts_id in leftover_ram[flv_id]:
                    constraint_count += 1
                    ram_limit = leftover_ram[flv_id][ts_id]
                    prob += pulp.lpSum(usage_expressions) <= ram_limit, f"RAM_{flv_id}_{ts_id}"
                else:
                    logging.warning(f"     - Missing RAM capacity data for node {flv_id}, timeslot {ts_id}")

            logging.info(f"  - Created {constraint_count} resource capacity constraints")
            logging.info(f"  - Final problem size: {len(x)} variables, {len(prob.constraints)} constraints")

            # Solve the problem
            logging.info(f"🔍 STEP 4: Solving multi-objective optimization problem")
            logging.info(f"  - Problem size: {len(x)} placement vars + {len(s)} slack vars = {len(x) + len(s)} total variables")

            # Read solver settings from config (with safe defaults)
            solver_cfg = self.config.get('optimization', {}).get('solver', {}) if hasattr(self, 'config') else {}
            time_limit_cfg = solver_cfg.get('time_limit', 60)
            gap_tolerance_cfg = solver_cfg.get('gap_tolerance', 0.05)
            threads_cfg = int(solver_cfg.get('threads', 4))

            logging.info(f"  - Solver settings: name={self.solver_name}, time_limit={time_limit_cfg}s, gap_tolerance={gap_tolerance_cfg}, threads={threads_cfg}")

            solver_start_time = time.time()
            try:
                # Primary attempt: config-driven time limit and gap tolerance
                primary_solver = self._get_pulp_solver(solver_cfg, time_limit_cfg, gap_tolerance_cfg, threads_cfg)

                # Warn if CBC is not available in this environment
                try:
                    if hasattr(primary_solver, 'available') and not primary_solver.available():
                        logging.error("  - CBC solver not available in this environment. Please install coinor-cbc.")
                except Exception:
                    pass

                prob.solve(primary_solver)
                solution_time = time.time() - solver_start_time
                logging.info(f"  - Solver completed in {solution_time:.3f}s")
            except Exception as solver_error:
                solution_time = time.time() - solver_start_time
                logging.error(f"  - Solver failed after {solution_time:.3f}s: {solver_error}")
                logging.info(f"  - Attempting fallback solver with relaxed settings...")
                try:
                    # Fallback attempt with longer time and looser gap to obtain a feasible solution
                    fallback_time_limit = max(int(time_limit_cfg * 2), 120)
                    fallback_gap = max(float(gap_tolerance_cfg), 0.2)
                    fallback_solver = self._get_pulp_solver(solver_cfg, fallback_time_limit, fallback_gap, threads_cfg)
                    prob.solve(fallback_solver)
                    solution_time = time.time() - solver_start_time
                    logging.info(f"  - Fallback solver completed in {solution_time:.3f}s")
                except Exception as fallback_error:
                    solution_time = time.time() - solver_start_time
                    logging.error(f"  - Fallback solver also failed: {fallback_error}")
                    self.status = "Solver_Error"
                    self.iterations = 0
                    return

            self.status = pulp.LpStatus[prob.status]
            self.iterations = prob.solverModel.Iterations if hasattr(prob.solverModel, 'Iterations') else 0

            logging.info(f"  - Global optimization completed in {solution_time:.3f}s")
            logging.info(f"  - Solver status: {self.status}")
            logging.info(f"  - Solver iterations: {self.iterations}")

            # Handle ambiguous results by retrying once with more relaxed parameters
            if self.status in ("Not Solved", "Undefined", "Time Limit") or (self.iterations == 0 and self.status != "Optimal"):
                logging.warning(f"  - Solver returned status '{self.status}' with {self.iterations} iterations. Retrying with relaxed settings...")
                retry_start = time.time()
                try:
                    retry_time_limit = max(int(time_limit_cfg * 2), 180)
                    retry_gap = max(float(gap_tolerance_cfg), 0.2)
                    retry_solver = pulp.PULP_CBC_CMD(
                        msg=bool(solver_cfg.get('msg', False)),
                        timeLimit=retry_time_limit,
                        gapRel=retry_gap,
                        threads=threads_cfg
                    )
                    prob.solve(retry_solver)
                    solution_time = time.time() - solver_start_time
                    self.status = pulp.LpStatus[prob.status]
                    self.iterations = prob.solverModel.Iterations if hasattr(prob.solverModel, 'Iterations') else 0
                    logging.info(f"  - Retry completed in {time.time() - retry_start:.3f}s with status: {self.status}, iterations: {self.iterations}")
                except Exception as retry_error:
                    logging.error(f"  - Retry failed: {retry_error}")

            # Extract solution if optimal
            logging.info(f"🔍 STEP 5: Processing solution and recording results")
            if prob.status == pulp.LpStatusOptimal:
                solution = {}
                total_emissions_objective = 0.0

                # Create a dictionary for quick pod lookup
                pending_pods_dict = {p.id: p for p in self.pending_pods}
                placements_saved_to_csv = 0

                logging.info(f"  - Found optimal solution! Extracting placements...")
                
                # CRITICAL FIX: Pre-validate solution before accepting it
                logging.info(f"🔍 STEP 5a: Validating solution constraints BEFORE processing")
                
                # Extract solution from decision variables
                solution_dict = {}
                unplaced_pods = []
                
                # Check slack variables to identify unplaced pods
                for pod_id, slack_var in s.items():
                    if slack_var.value() and slack_var.value() > 0.5:  # Pod is marked as unplaced
                        unplaced_pods.append(pod_id)
                
                # Extract placed pods from placement variables
                for var_name, var in x.items():
                    if var.value() and var.value() > 0.5:  # Binary variable is active
                        pod_id, flv_id, ts_id = var_name
                        solution_dict[pod_id] = (flv_id, ts_id, 0.0)  # emissions filled later
                
                logging.info(f"  - Multi-objective solution: {len(solution_dict)} pods placed, {len(unplaced_pods)} pods unplaced")
                logging.info(f"  - Placement success rate: {len(solution_dict)}/{len(self.pending_pods)} ({100*len(solution_dict)/len(self.pending_pods):.1f}%)")
                
                # Log sample of unplaced pods (if any)
                if unplaced_pods:
                    logging.warning(f"  - Unplaced pods: {unplaced_pods[:5]}{'...' if len(unplaced_pods) > 5 else ''}")
                elif len(solution_dict) == len(self.pending_pods):
                    logging.info(f"  - ✅ All pods successfully placed!")
                else:
                    logging.warning("  - Inconsistent solution: no unplaced pods reported but not all pods have placements")
                
                # Validate constraints on the extracted solution
                is_valid, constraint_violations = self._validate_solution_constraints(
                    solution_dict, flavours, timeslots, leftover_cpu, leftover_ram
                )
                
                if not is_valid:
                    logging.error(f"❌ CRITICAL: Solution violates {len(constraint_violations)} constraints!")
                    for violation in constraint_violations[:10]:  # Show first 10 violations
                        logging.error(f"  - VIOLATION: {violation}")
                    if len(constraint_violations) > 10:
                        logging.error(f"  - ... and {len(constraint_violations) - 10} more violations")
                    
                    # Log detailed constraint information for debugging
                    self._log_constraint_debugging_info(constraint_violations, solution_dict)
                    
                    logging.error("❌ REJECTING INVALID SOLUTION - treating as infeasible")
                    self.status = "Infeasible_Due_To_Constraint_Violations"
                    self.has_solved = False
                    return
                else:
                    logging.info("✅ Solution constraint validation PASSED - all constraints satisfied")
                

                
                pod_footprints = self._compute_solution_footprints(
                    solution_dict=solution_dict,
                    pending_pods_dict=pending_pods_dict,
                    flavours=flavours,
                    max_time_slots=max_time_slots,
                )

                # 3) Write CSV and construct solution
                for pod_id_sol, (flv_id_sol, ts_id_sol, _) in solution_dict.items():
                    current_pod = pending_pods_dict.get(pod_id_sol)
                    if not current_pod:
                        logging.error(f"  - ERROR: Pod {pod_id_sol} not found in pending_pods_dict. Skipping CSV write for this placement.")
                        continue
                    footprint = pod_footprints.get(pod_id_sol, FootprintVector().finalize())
                    emissions_sol_g = footprint.total_carbon_g
                    logging.info(f"  - Selected placement: Pod {pod_id_sol} -> Node {flv_id_sol}, Timeslot {ts_id_sol}, Emissions {emissions_sol_g/1000.0:.6f}kgCO2e ({emissions_sol_g:.2f}gCO2e)")

                    # Skip duplicate CSV row if exists in global solution
                    if pod_id_sol in self.global_solution:
                        existing_flv, existing_ts, _ = self.global_solution[pod_id_sol]
                        if existing_flv == flv_id_sol and existing_ts == ts_id_sol:
                            logging.debug(f"  - Pod {pod_id_sol} placement already exists in global solution, skipping CSV write")
                            continue

                    if self.is_precomputation_mode:
                        try:
                            current_flavour = next((flv for flv in flavours if flv.id == flv_id_sol), None)
                            self._write_placement_to_csv(
                                pod_id=current_pod.id,
                                node_id=flv_id_sol,
                                start_slot=ts_id_sol,
                                duration=current_pod.duration,
                                cpu_request=current_pod.cpuRequest,
                                ram_request=current_pod.ramRequest,
                                total_carbon_emissions=emissions_sol_g / 1000.0,
                                solver_status=str(self.status),
                                solver_iterations=self.iterations,
                                solution_time_seconds=solution_time,
                                embodied_mode="proportional",
                                flavour=current_flavour,
                                footprint=footprint,
                            )
                            placements_saved_to_csv += 1
                            logging.debug(f"  - Wrote placement for pod {current_pod.id} to CSV (precomputation mode)")
                        except Exception as e:
                            logging.error(f"  - ERROR: Failed to write placement for pod {current_pod.id} to CSV: {e}")
                            logging.error(traceback.format_exc())
                    else:
                        logging.debug(f"  - Skipping CSV write for pod {current_pod.id} (incremental mode)")

                    solution[pod_id_sol] = (flv_id_sol, ts_id_sol, emissions_sol_g)
                    total_emissions_objective += emissions_sol_g

                # Calculate separated objectives  
                placement_penalty = len(unplaced_pods) * penalty_for_unplaced
                total_objective = prob.objective.value() if prob.objective.value() else 0.0
                
                logging.info(f"✅ Multi-objective solution found:")
                logging.info(f"  - Total objective value: {total_objective:.2f}")
                logging.info(f"  - Placement penalty: {placement_penalty:.2f} ({len(unplaced_pods)} unplaced * {penalty_for_unplaced})")
                logging.info(f"  - Emissions objective: {total_emissions_objective/1000.0:.6f}kgCO2e ({total_emissions_objective:.2f}gCO2e)")
                logging.info(f"✅ Placed {len(solution)}/{len(self.pending_pods)} pods (success rate: {100*len(solution)/len(self.pending_pods):.1f}%)")
                logging.info(f"📊 Wrote {placements_saved_to_csv} placements to CSV")

                # Update the solution and mark pods as processed
                self.global_solution.update(solution)
                
                # Mark all successfully placed pods as processed to prevent duplicate processing
                for pod_id in solution.keys():
                    self._processed_pods.add(pod_id)
                    logging.debug(f"  - Marked pod {pod_id} as processed")
                
                # Remove placed pods from pending list
                self.pending_pods = [p for p in self.pending_pods if p.id not in solution]
                self.has_solved = True
                
                # Verify CSV file status after writing
                if hasattr(self, '_session_csv_path') and self._session_csv_path and os.path.exists(self._session_csv_path):
                    file_size = os.path.getsize(self._session_csv_path)
                    logging.info(f"  - CSV file size after writing: {file_size} bytes")
                    if file_size == 0:
                        logging.error("  - ERROR: CSV file is empty after writing!")
                else:
                    logging.warning("  - WARNING: CSV file not found or not configured correctly")
                
            else:
                logging.warning(f"❌ Multi-objective optimization failed, status: {self.status}")
                
                # With slack variables, infeasibility should be very rare
                if self.status == "Infeasible":
                    logging.error("🚨 UNEXPECTED INFEASIBILITY - Multi-objective with slack variables should be feasible!")
                    logging.error("   - This suggests a deeper problem with the formulation")
                    logging.error("   - Attempting legacy relaxation strategy as emergency fallback")
                    
                    self._handle_infeasibility(self.pending_pods, placements, leftover_cpu, leftover_ram, max_time_slots)
                    
                    success = self._solve_with_relaxation(
                        flavours, timeslots, leftover_cpu, leftover_ram, max_time_slots, 
                        placements, prob
                    )
                    
                    if success:
                        logging.info("✅ Emergency relaxation strategy succeeded - partial solution found!")
                    else:
                        logging.error("❌ Even emergency relaxation strategy failed")
                elif self.status == "Time Limit":
                    logging.warning("⏰ Solver reached time limit - try increasing timeLimit parameter")
                else:
                    logging.warning(f"  - Unexpected solver status: {self.status}")
                    logging.warning("  - Check CBC solver output for more details")

        except Exception as e:
            logging.error(f"❌ ERROR in global optimization: {e}")
            logging.error(traceback.format_exc())

        self.last_solve_time = time.time()
        logging.info(f"====== GLOBAL OPTIMIZATION SOLVER END ======")
        
        # Add CRITICAL log at the end for better visibility
        if self.has_solved:
            logging.critical(f"💡 GLOBAL OPTIMIZATION SUCCESS! Placed {len(self.global_solution)} pods.")
        else:
            logging.critical(f"💡 GLOBAL OPTIMIZATION FAILED! No solution found.")
            
        # Check CSV file state
        if hasattr(self, '_session_csv_path') and self._session_csv_path:
            if os.path.exists(self._session_csv_path):
                file_size = os.path.getsize(self._session_csv_path)
                if file_size > 0:
                    logging.critical(f"📊 RESULTS SAVED to {self._session_csv_path} (size: {file_size} bytes)")
                else:
                    logging.critical(f"⚠️ CSV FILE EMPTY: {self._session_csv_path}")
            else:
                logging.critical(f"⚠️ CSV FILE NOT FOUND: {self._session_csv_path}")
        else:
            logging.critical(f"⚠️ CSV FILE PATH NOT SET")

    def _validate_solution_constraints(
        self,
        solution: Dict[str, Tuple[str, int, float]],
        flavours: List[EnvironmentalFlavor],
        timeslots: List[CarbonAwareTimeslot],
        leftover_cpu: Dict[str, Dict[int, float]],
        leftover_ram: Dict[str, Dict[int, float]]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that the solution satisfies all constraints.
        
        Args:
            solution: Dictionary of pod_id -> (node_id, timeslot_id, emissions)
            flavours: List of available flavours
            timeslots: List of available timeslots
            leftover_cpu: Available CPU resources
            leftover_ram: Available RAM resources
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        # Track resource usage by node and timeslot
        cpu_usage = {}  # (node_id, timeslot_id) -> total_cpu_used
        ram_usage = {}  # (node_id, timeslot_id) -> total_ram_used
        
        # Create pod lookup
        pending_pods_dict = {p.id: p for p in self.pending_pods}
        
        for pod_id, (node_id, timeslot_id, emissions) in solution.items():
            pod = pending_pods_dict.get(pod_id)
            if not pod:
                violations.append(f"Pod {pod_id} not found in pending pods")
                continue
                
            # Check timeslot constraints
            if hasattr(pod, 'earliest_timeslot') and pod.earliest_timeslot is not None:
                if timeslot_id < pod.earliest_timeslot:
                    violations.append(f"Pod {pod_id} scheduled at timeslot {timeslot_id} before earliest allowed {pod.earliest_timeslot}")
            
            # Check deadline constraints
            if hasattr(pod, 'deadline_slot') and pod.deadline_slot is not None:
                if timeslot_id + pod.duration > pod.deadline_slot:
                    violations.append(f"Pod {pod_id} scheduled to finish at {timeslot_id + pod.duration} after deadline {pod.deadline_slot}")
            
            # Accumulate resource usage for capacity validation
            for offset in range(int(pod.duration)):
                current_slot = timeslot_id + offset
                key = (node_id, current_slot)
                
                if key not in cpu_usage:
                    cpu_usage[key] = 0
                    ram_usage[key] = 0
                    
                cpu_usage[key] += pod.cpuRequest
                ram_usage[key] += pod.ramRequest
        
        # Validate capacity constraints
        for (node_id, timeslot_id), total_cpu in cpu_usage.items():
            if node_id in leftover_cpu and timeslot_id in leftover_cpu[node_id]:
                available_cpu = leftover_cpu[node_id][timeslot_id]
                if total_cpu > available_cpu:
                    violations.append(f"CPU overallocation on node {node_id} at timeslot {timeslot_id}: {total_cpu:.2f} > {available_cpu:.2f}")
            else:
                violations.append(f"Missing CPU capacity data for node {node_id} at timeslot {timeslot_id}")
        
        for (node_id, timeslot_id), total_ram in ram_usage.items():
            if node_id in leftover_ram and timeslot_id in leftover_ram[node_id]:
                available_ram = leftover_ram[node_id][timeslot_id]
                if total_ram > available_ram:
                    violations.append(f"RAM overallocation on node {node_id} at timeslot {timeslot_id}: {total_ram:.2f} > {available_ram:.2f}")
            else:
                violations.append(f"Missing RAM capacity data for node {node_id} at timeslot {timeslot_id}")
        
        is_valid = len(violations) == 0
        return is_valid, violations

    def _log_constraint_debugging_info(
        self,
        violations: List[str],
        solution: Dict[str, Tuple[str, int, float]]
    ) -> None:
        """
        Log detailed constraint debugging information.
        
        Args:
            violations: List of constraint violations
            solution: The solution being validated
        """
        logging.error(f"🚨 CONSTRAINT VIOLATIONS DETECTED: {len(violations)} violations found")
        
        # Group violations by type
        capacity_violations = [v for v in violations if "overallocation" in v]
        timeslot_violations = [v for v in violations if "timeslot" in v or "deadline" in v]
        other_violations = [v for v in violations if v not in capacity_violations and v not in timeslot_violations]
        
        if capacity_violations:
            logging.error(f"📊 CAPACITY VIOLATIONS ({len(capacity_violations)}):")
            for violation in capacity_violations[:10]:  # Limit output
                logging.error(f"  - {violation}")
            if len(capacity_violations) > 10:
                logging.error(f"  - ... and {len(capacity_violations) - 10} more capacity violations")
        
        if timeslot_violations:
            logging.error(f"⏰ TIMESLOT VIOLATIONS ({len(timeslot_violations)}):")
            for violation in timeslot_violations:
                logging.error(f"  - {violation}")
        
        if other_violations:
            logging.error(f"❓ OTHER VIOLATIONS ({len(other_violations)}):")
            for violation in other_violations:
                logging.error(f"  - {violation}")
        
        # Log solution summary
        if solution:
            node_counts = {}
            for pod_id, (node_id, timeslot_id, emissions) in solution.items():
                node_counts[node_id] = node_counts.get(node_id, 0) + 1
            
            logging.error("📋 SOLUTION SUMMARY:")
            logging.error(f"  - Total pods placed: {len(solution)}")
            logging.error(f"  - Nodes used: {len(node_counts)}")
            for node_id, count in sorted(node_counts.items()):
                logging.error(f"    * {node_id}: {count} pods")

    def find_placement(
        self,
        pod: CarbonAwarePod,
        flavours: List[EnvironmentalFlavor],
        timeslots: List[CarbonAwareTimeslot],
        leftover_cpu: Dict[str, Dict[int, float]],
        leftover_ram: Dict[str, Dict[int, float]],
        max_time_slots: int = 48
    ) -> Tuple[Optional[EnvironmentalFlavor], Optional[CarbonAwareTimeslot], float]:
        """
        Return placement for a pod, using the pre-computed comprehensive solution if available.
        Otherwise, fall back to incremental global optimization.
        """
        start_time = time.time()
        # Set earliest_timeslot based on pod ID before scheduling
        self._set_pod_earliest_timeslot(pod)
        logging.debug(f"Finding placement for pod {pod.id}")
        
        # Check if we have a comprehensive solution
        if self.optimization_done:
            if pod.id in self.global_solution:
                flv_id, ts_id, emissions = self.global_solution[pod.id]
                
                # Find the corresponding objects
                flv = next((f for f in flavours if f.id == flv_id), None)
                ts = next((t for t in timeslots if t.id == ts_id), None)
                
                if flv and ts:
                    logging.info(f"📌 Using comprehensive optimization placement for pod {pod.id}: {flv_id}, timeslot {ts_id}")
                    if self.experiment_logger:
                        execution_time = time.time() - start_time
                        self.experiment_logger.record_placement(
                            pod_id=pod.id, success=True, execution_time=execution_time,
                            emissions=emissions, considered_options=len(self.global_solution),
                            selected_node=flv_id, selected_timeslot=ts_id,
                            solver_iterations=self.iterations, solver_status="Comprehensive"
                        )
                    return flv, ts, emissions
                else:
                    logging.warning(f"⚠️ Found solution for pod {pod.id} but node or timeslot not available in current context")
            else:
                logging.warning(f"⚠️ No placement for pod {pod.id} in the comprehensive solution")
            
            # Pod not in comprehensive solution - this is unusual if we did comprehensive optimization
            logging.warning(f"🔍 Pod {pod.id} not found in comprehensive solution")

        # Fallback to incremental optimization or greedy approach
        logging.debug(f"🔄 Falling back to incremental placement for pod {pod.id}")
        
        # Add pod to pending list if not already there
        if pod not in self.pending_pods:
            self.pending_pods.append(pod)
        
        # Trigger incremental global optimization if enough pods or time has passed
        should_solve = (
            len(self.pending_pods) >= self.solve_interval or
            (time.time() - self.last_solve_time) > self.solve_interval
        )
        
        if should_solve:
            # 🚨 CRITICAL: Use thread-safe lock to prevent race conditions (incremental path)
            if not self._solving_lock.acquire(blocking=False):
                logging.warning("🚫 Solver already running at incremental call site. Skipping to prevent race condition.")
            else:
                try:
                    logging.info(f"🚀 Triggering incremental global optimization for {len(self.pending_pods)} pods")
                    self.solve_global_optimization(flavours, timeslots, leftover_cpu, leftover_ram, max_time_slots)
                finally:
                    # 🚨 CRITICAL: Always release the lock
                    self._solving_lock.release()
                    logging.debug("🔓 Solver lock released at incremental call site")
            
            # Check if the pod is now in the solution
            if pod.id in self.global_solution:
                flv_id, ts_id, emissions = self.global_solution[pod.id]
                flv = next((f for f in flavours if f.id == flv_id), None)
                ts = next((t for t in timeslots if t.id == ts_id), None)
                
                if flv and ts:
                    logging.info(f"📌 Using incremental optimization placement for pod {pod.id}: {flv_id}, timeslot {ts_id}")
                    if self.experiment_logger:
                        execution_time = time.time() - start_time
                        self.experiment_logger.record_placement(
                            pod_id=pod.id, success=True, execution_time=execution_time,
                            emissions=emissions, considered_options=len(self.global_solution),
                            selected_node=flv_id, selected_timeslot=ts_id,
                            solver_iterations=self.iterations, solver_status="Incremental"
                        )
                    return flv, ts, emissions
        
        # Final fallback: greedy placement
        logging.warning(f"🔄 Using greedy fallback for pod {pod.id}")
        return self._greedy_placement(pod, flavours, timeslots, leftover_cpu, leftover_ram)

    def _greedy_placement(
        self,
        pod: CarbonAwarePod,
        flavours: List[EnvironmentalFlavor],
        timeslots: List[CarbonAwareTimeslot],
        leftover_cpu: Dict[str, Dict[int, float]],
        leftover_ram: Dict[str, Dict[int, float]]
    ) -> Tuple[Optional[EnvironmentalFlavor], Optional[CarbonAwareTimeslot], float]:
        """
        Greedy placement fallback when global optimization fails.
        """
        best_placement = None
        best_emissions = float('inf')
        
        for ts in timeslots:
            # Check earliest timeslot constraint
            if hasattr(pod, 'earliest_timeslot') and pod.earliest_timeslot is not None:
                if ts.id < pod.earliest_timeslot:
                    continue
            
            # Check deadline constraint
            if hasattr(pod, 'deadline_slot') and pod.deadline_slot is not None:
                if ts.id + pod.duration > pod.deadline_slot:
                    continue
            
            for flv in flavours:
                # Check resource availability for the entire duration
                can_place = True
                for offset in range(int(pod.duration)):
                    slot = ts.id + offset
                    if (flv.id not in leftover_cpu or slot not in leftover_cpu[flv.id] or
                        leftover_cpu[flv.id][slot] < pod.cpuRequest):
                        can_place = False
                        break
                    if (flv.id not in leftover_ram or slot not in leftover_ram[flv.id] or
                        leftover_ram[flv.id][slot] < pod.ramRequest):
                        can_place = False
                        break
                
                if can_place:
                    try:
                        used_cpu_before = {}
                        for slot_offset in range(int(pod.duration)):
                            slot_id = ts.id + slot_offset
                            total_capacity = flv.totalCpu
                            leftover = leftover_cpu[flv.id][slot_id]
                            used_cpu_before[slot_id] = max(total_capacity - leftover, 0.0)
                        emissions = compute_emissions(
                            flv, ts.id, pod,
                            used_cpu_before_by_slot=used_cpu_before,
                            embodied_allocation_mode="proportional",
                        )
                        if emissions < best_emissions:
                            best_emissions = emissions
                            best_placement = (flv, ts, emissions)
                    except Exception as e:
                        logging.warning(f"Error computing emissions for greedy placement: {e}")
        
        if best_placement:
            flv, ts, emissions = best_placement
            logging.info(f"📌 Greedy placement for pod {pod.id}: {flv.id}, timeslot {ts.id}")
            return flv, ts, emissions
        else:
            logging.warning(f"❌ No valid placement found for pod {pod.id}")
            return None, None, 0.0


    def _solve_with_relaxation(
        self,
        flavours: List[EnvironmentalFlavor],
        timeslots: List[CarbonAwareTimeslot], 
        leftover_cpu: Dict[str, Dict[int, float]],
        leftover_ram: Dict[str, Dict[int, float]],
        max_time_slots: int,
        original_placements: Dict[str, List[Tuple[str, int, float]]],
        original_prob
    ) -> bool:
        """
        Relaxation strategy: iteratively remove most constrained pods until problem becomes feasible.
        Then try to place remaining pods greedily.
        """
        logging.info("🔄 STARTING RELAXATION STRATEGY")
        
        # Make a copy of pending pods to work with
        remaining_pods = self.pending_pods.copy()
        removed_pods = []
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations and len(remaining_pods) > 0:
            iteration += 1
            logging.info(f"  🔄 Relaxation iteration {iteration}: {len(remaining_pods)} pods remaining")
            
            # Identify most constrained pods (with fewest placement options)
            pod_constraints = []
            for pod in remaining_pods:
                placement_count = len(original_placements.get(pod.id, []))
                if placement_count == 0:
                    # Pods with zero placements should be removed first
                    pod_constraints.append((pod, 0))
                else:
                    # Calculate constraint score based on placement options and deadline tightness
                    deadline_pressure = 0
                    if hasattr(pod, 'deadline_slot') and pod.deadline_slot is not None:
                        scheduling_window = pod.deadline_slot - pod.earliest_timeslot - pod.duration
                        deadline_pressure = max(0, 5 - scheduling_window)  # Higher score = tighter deadline
                    
                    constraint_score = deadline_pressure + (10 / max(1, placement_count))
                    pod_constraints.append((pod, constraint_score))
            
            # Sort by constraint score (most constrained first)
            pod_constraints.sort(key=lambda x: x[1], reverse=True)
            
            # Remove the most constrained 10% of pods (minimum 1, maximum 20)
            removal_count = max(1, min(20, len(remaining_pods) // 10))
            pods_to_remove = [pod for pod, _ in pod_constraints[:removal_count]]
            
            for pod in pods_to_remove:
                remaining_pods.remove(pod)
                removed_pods.append(pod)
                logging.info(f"    ❌ Removed highly constrained pod: {pod.id} (constraint score: {[sc for p, sc in pod_constraints if p.id == pod.id][0]:.2f})")
            
            # Try solving with reduced pod set
            success = self._try_milp_with_pods(
                remaining_pods, flavours, timeslots, leftover_cpu, leftover_ram, max_time_slots
            )
            
            if success:
                logging.info(f"  ✅ Relaxation succeeded after {iteration} iterations!")
                logging.info(f"    - Placed: {len(remaining_pods)} pods via MILP")
                logging.info(f"    - Removed: {len(removed_pods)} constrained pods")
                
                # Try to place removed pods greedily
                greedy_placed = self._try_greedy_placement_for_removed_pods(
                    removed_pods, flavours, timeslots, leftover_cpu, leftover_ram
                )
                
                logging.info(f"    - Greedy placement added: {greedy_placed} more pods")
                logging.info(f"    - Total placed: {len(self.global_solution)} / {len(self.pending_pods)} pods")
                return True
            
            if len(remaining_pods) <= 2:
                logging.warning(f"  ⚠️  Only {len(remaining_pods)} pods left, stopping relaxation")
                break
        
        logging.error(f"  ❌ Relaxation failed after {iteration} iterations")
        return False

    def _try_milp_with_pods(
        self,
        pods_subset: List[CarbonAwarePod],
        flavours: List[EnvironmentalFlavor],
        timeslots: List[CarbonAwareTimeslot],
        leftover_cpu: Dict[str, Dict[int, float]],
        leftover_ram: Dict[str, Dict[int, float]],
        max_time_slots: int
    ) -> bool:
        """
        Try solving MILP with a subset of pods.
        """
        import pulp
        import time
        
        try:
            logging.info(f"    🧮 Attempting MILP with {len(pods_subset)} pods")
            
            # Create new MILP problem
            prob = pulp.LpProblem("CarbonAwareScheduling_Relaxed", pulp.LpMinimize)
            x = {}
            placements = {}
            
            # Find valid placements for reduced pod set
            for pod in pods_subset:
                self._set_pod_earliest_timeslot(pod)
                pod_placements = []
                
                for ts in timeslots:
                    if ts.id < pod.earliest_timeslot:
                        continue
                    
                    if hasattr(pod, 'deadline_slot') and pod.deadline_slot is not None:
                        if ts.id + pod.duration > pod.deadline_slot:
                            continue
                    
                    for flv in flavours:
                        # Check resource feasibility
                        duration_feasible = True
                        for offset in range(int(pod.duration)):
                            current_slot = ts.id + offset
                            if current_slot >= max_time_slots:
                                duration_feasible = False
                                break
                            
                            if (leftover_cpu[flv.id][current_slot] < pod.cpuRequest or
                                leftover_ram[flv.id][current_slot] < pod.ramRequest):
                                duration_feasible = False
                                break
                        
                        if duration_feasible:
                            emissions = compute_emissions(flv, ts.id, pod)
                            pod_placements.append((flv.id, ts.id, emissions))
                            
                            placement_key = (pod.id, flv.id, ts.id)
                            x[placement_key] = pulp.LpVariable(f"x_{pod.id}_{flv.id}_{ts.id}", cat=pulp.LpBinary)
                
                placements[pod.id] = pod_placements
                
                # Constraint: each pod placed exactly once
                if pod_placements:
                    prob += pulp.lpSum([x[(pod.id, flv_id, ts_id)] for flv_id, ts_id, _ in pod_placements]) == 1
                else:
                    logging.warning(f"    ⚠️  Pod {pod.id} has no valid placements in relaxed problem")
                    return False
            
            # Objective: minimize emissions
            prob += pulp.lpSum([emissions * x[(pod_id, flv_id, ts_id)]
                               for pod_id, pod_placements in placements.items()
                               for flv_id, ts_id, emissions in pod_placements])
            
            # Resource constraints
            self._add_resource_constraints(prob, x, placements, pods_subset, leftover_cpu, leftover_ram, max_time_slots)
            
            # Solve
            start_time = time.time()
            prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=30))
            solve_time = time.time() - start_time
            
            status = pulp.LpStatus[prob.status]
            logging.info(f"    🔍 Relaxed MILP result: {status} in {solve_time:.2f}s")
            
            if prob.status == pulp.LpStatusOptimal:
                # Extract and store solution
                for pod_id, pod_placements in placements.items():
                    for flv_id, ts_id, emissions in pod_placements:
                        if x[(pod_id, flv_id, ts_id)].value() and x[(pod_id, flv_id, ts_id)].value() > 0.5:
                            self.global_solution[pod_id] = (flv_id, ts_id, emissions)
                            
                            # Write to CSV
                            if self.is_precomputation_mode:
                                pod_obj = next((p for p in pods_subset if p.id == pod_id), None)
                                flavour_obj = next((flv for flv in flavours if flv.id == flv_id), None)
                                if pod_obj:
                                    self._write_placement_to_csv(
                                        pod_id=pod_id, node_id=flv_id, start_slot=ts_id,
                                        duration=pod_obj.duration, cpu_request=pod_obj.cpuRequest,
                                        ram_request=pod_obj.ramRequest,
                                        total_carbon_emissions=emissions / 1000.0,
                                        solver_status="Relaxed_MILP", solver_iterations=0,
                                        solution_time_seconds=solve_time,
                                        flavour=flavour_obj,
                                        footprint=self._build_logging_footprint(pod_obj, flavour_obj, ts_id) if flavour_obj else None,
                                    )
                            break
                
                self.has_solved = True
                self.optimization_done = True
                return True
                
        except Exception as e:
            logging.error(f"    ❌ Error in relaxed MILP: {e}")
        
        return False

    def _try_greedy_placement_for_removed_pods(
        self,
        removed_pods: List[CarbonAwarePod],
        flavours: List[EnvironmentalFlavor],
        timeslots: List[CarbonAwareTimeslot],
        leftover_cpu: Dict[str, Dict[int, float]],
        leftover_ram: Dict[str, Dict[int, float]]
    ) -> int:
        """
        Try to place removed pods using greedy algorithm.
        Updates leftover resources as pods are placed.
        """
        placed_count = 0
        
        # Create working copies of resource availability
        working_cpu = {}
        working_ram = {}
        for node_id in leftover_cpu:
            working_cpu[node_id] = leftover_cpu[node_id].copy()
            working_ram[node_id] = leftover_ram[node_id].copy()
        
        # Reduce available resources by already placed pods
        for pod_id, (node_id, start_slot, _) in self.global_solution.items():
            pod = next((p for p in self.pending_pods if p.id == pod_id), None)
            if pod:
                for offset in range(int(pod.duration)):
                    slot = start_slot + offset
                    if slot in working_cpu[node_id]:
                        working_cpu[node_id][slot] -= pod.cpuRequest
                        working_ram[node_id][slot] -= pod.ramRequest
        
        logging.info(f"    🎯 Attempting greedy placement for {len(removed_pods)} removed pods")
        
        for pod in removed_pods:
            placement = self._greedy_placement(pod, flavours, timeslots, working_cpu, working_ram)
            flv, ts, emissions = placement
            
            if flv and ts:
                # Record placement
                self.global_solution[pod.id] = (flv.id, ts.id, emissions)
                
                placement_footprint = self._build_logging_footprint(
                    pod=pod,
                    flavour=flv,
                    start_slot=ts.id,
                    used_cpu_before_by_slot=build_used_cpu_before_map(
                        flavour=flv,
                        start_slot=ts.id,
                        duration_hours=pod.duration,
                        leftover_cpu_by_slot=working_cpu[flv.id],
                    ),
                )

                # Update working resources
                for offset in range(int(pod.duration)):
                    slot = ts.id + offset
                    if slot in working_cpu[flv.id]:
                        working_cpu[flv.id][slot] -= pod.cpuRequest
                        working_ram[flv.id][slot] -= pod.ramRequest
                
                # Write to CSV
                if self.is_precomputation_mode:
                    self._write_placement_to_csv(
                        pod_id=pod.id, node_id=flv.id, start_slot=ts.id,
                        duration=pod.duration, cpu_request=pod.cpuRequest,
                        ram_request=pod.ramRequest,
                        total_carbon_emissions=emissions / 1000.0,
                        solver_status="Greedy_Fallback", solver_iterations=0,
                        solution_time_seconds=0.0,
                        flavour=flv,
                        footprint=placement_footprint,
                    )
                
                placed_count += 1
                logging.info(f"      ✅ Greedy placed {pod.id} on {flv.id} at slot {ts.id}")
            else:
                logging.debug(f"      ❌ Could not place {pod.id} greedily")
        
        return placed_count

    def _add_resource_constraints(self, prob, x, placements, pods, leftover_cpu, leftover_ram, max_time_slots):
        """Helper method to add resource constraints to MILP problem"""
        import pulp
        
        cpu_usage = {}
        ram_usage = {}
        
        for pod in pods:
            pod_placements = placements.get(pod.id, [])
            for flv_id, ts_id, _ in pod_placements:
                for offset in range(int(pod.duration)):
                    current_ts = ts_id + offset
                    if current_ts >= max_time_slots:
                        continue
                    
                    cpu_key = (flv_id, current_ts)
                    ram_key = (flv_id, current_ts)
                    
                    if cpu_key not in cpu_usage:
                        cpu_usage[cpu_key] = []
                        ram_usage[ram_key] = []
                    
                    cpu_usage[cpu_key].append(pod.cpuRequest * x[(pod.id, flv_id, ts_id)])
                    ram_usage[ram_key].append(pod.ramRequest * x[(pod.id, flv_id, ts_id)])
        
        # Add constraints
        for (flv_id, ts_id), usage_exprs in cpu_usage.items():
            if flv_id in leftover_cpu and ts_id in leftover_cpu[flv_id]:
                prob += pulp.lpSum(usage_exprs) <= leftover_cpu[flv_id][ts_id]
        
        for (flv_id, ts_id), usage_exprs in ram_usage.items():
            if flv_id in leftover_ram and ts_id in leftover_ram[flv_id]:
                prob += pulp.lpSum(usage_exprs) <= leftover_ram[flv_id][ts_id]


    def _handle_infeasibility(self, pods, placements, leftover_cpu, leftover_ram, max_time_slots):
        """
        Analyze and report on why the MILP problem is infeasible.
        Provides detailed diagnostics to help understand the root cause.
        """
        logging.error("🔍 ANALYZING INFEASIBILITY...")
        
        # Analyze problem scale
        logging.error(f"📊 PROBLEM SCALE:")
        logging.error(f"  - Total pods: {len(pods)}")
        logging.error(f"  - Total nodes: {len(leftover_cpu)}")
        logging.error(f"  - Total timeslots: {max_time_slots}")
        logging.error(f"  - Total valid placements: {sum(len(p) for p in placements.values())}")
        
        # Analyze pods by constraint tightness
        constrained_pods = []
        zero_placement_pods = []
        
        for pod in pods:
            pod_placements = placements.get(pod.id, [])
            if len(pod_placements) == 0:
                zero_placement_pods.append(pod)
            elif len(pod_placements) <= 8:  # Very constrained
                constrained_pods.append((pod, len(pod_placements)))
        
        if zero_placement_pods:
            logging.error(f"❌ PODS WITH ZERO VALID PLACEMENTS ({len(zero_placement_pods)}):")
            for pod in zero_placement_pods[:5]:  # Show first 5
                logging.error(f"  - {pod.id}: CPU={pod.cpuRequest}, RAM={pod.ramRequest}, duration={pod.duration}h")
                logging.error(f"    earliest={pod.earliest_timeslot}, deadline={getattr(pod, 'deadline_slot', 'None')}")
            if len(zero_placement_pods) > 5:
                logging.error(f"  - ... and {len(zero_placement_pods) - 5} more")
        
        if constrained_pods:
            constrained_pods.sort(key=lambda x: x[1])  # Sort by placement count
            logging.error(f"⚠️ HIGHLY CONSTRAINED PODS (≤8 placements) ({len(constrained_pods)}):")
            for pod, count in constrained_pods[:5]:
                logging.error(f"  - {pod.id}: {count} placements")
                logging.error(f"    earliest={pod.earliest_timeslot}, deadline={getattr(pod, 'deadline_slot', 'None')}")
        
        # Analyze timeslot distribution
        timeslot_loads = {}
        for pod in pods:
            earliest = getattr(pod, 'earliest_timeslot', 0)
            timeslot_loads[earliest] = timeslot_loads.get(earliest, 0) + 1
        
        logging.error(f"⏰ TIMESLOT LOAD DISTRIBUTION:")
        for ts in sorted(timeslot_loads.keys()):
            if timeslot_loads[ts] > 5:  # Only show heavily loaded timeslots
                logging.error(f"  - Timeslot {ts}: {timeslot_loads[ts]} pods")
        
        # Suggest solutions
        logging.error("💡 SUGGESTED SOLUTIONS:")
        if zero_placement_pods:
            logging.error("  1. Remove or relax constraints for pods with zero placements")
        if len(constrained_pods) > len(pods) * 0.3:
            logging.error("  2. Increase node capacity or add more nodes")
        max_load_ts = max(timeslot_loads.items(), key=lambda x: x[1])
        if max_load_ts[1] > 20:
            logging.error(f"  3. Distribute pod earliest_timeslots more evenly (timeslot {max_load_ts[0]} has {max_load_ts[1]} pods)")
        logging.error("  4. Use relaxation strategy to place subset of pods")


    def _load_nodes_from_yaml(self, nodes_file: str) -> List[EnvironmentalFlavor]:
        """
        Load nodes directly from nodes.yaml file.
        
        Args:
            nodes_file: Path to nodes.yaml file
        
        Returns:
            List of EnvironmentalFlavor objects
        """
        import yaml
        import re
        import os
        import traceback
        
        logging.info(f"🔄 Loading nodes infrastructure from {nodes_file}")
        
        # Check if file exists
        if not os.path.exists(nodes_file):
            logging.error(f"❌ Nodes file not found: {nodes_file}")
            return []
            
        # Check if file is empty
        if os.path.getsize(nodes_file) == 0:
            logging.error(f"❌ Nodes file is empty: {nodes_file}")
            return []
            
        flavours = []
        
        try:
            logging.info(f"📂 Reading YAML content from {nodes_file}")
            with open(nodes_file, 'r') as f:
                nodes_data = yaml.safe_load_all(f)
                for node in nodes_data:
                    if not node:
                        continue
                        
                    # Extract node ID
                    node_id = node.get("metadata", {}).get("name", "")
                    if not node_id:
                        continue
                        
                    # Extract annotations
                    annotations = node.get("metadata", {}).get("annotations", {})
                    
                    # Extract status
                    status = node.get("status", {})
                    allocatable = status.get("allocatable", {})
                    
                    # Extract CPU, RAM and set defaults
                    try:
                        total_cpu = float(allocatable.get("cpu", "0"))
                    except ValueError:
                        # Handle unit suffixes like '2' or '200m'
                        cpu_str = allocatable.get("cpu", "0")
                        if cpu_str.endswith('m'):
                            total_cpu = float(cpu_str[:-1]) / 1000
                        else:
                            total_cpu = float(cpu_str)
                    
                    # Parse RAM (convert from Ki, Mi, Gi to MB for consistency with pod requests)
                    ram_str = allocatable.get("memory", "0")
                    ram_match = re.match(r'(\d+)([KMG]i?)?', ram_str)
                    
                    if ram_match:
                        ram_value = float(ram_match.group(1))
                        ram_unit = ram_match.group(2) if ram_match.group(2) else ""
                        
                        if ram_unit.startswith('K'):
                            total_ram = ram_value / 1024  # Ki to MB
                        elif ram_unit.startswith('M'):
                            total_ram = ram_value  # Mi to MB (same)
                        elif ram_unit.startswith('G'):
                            total_ram = ram_value * 1024  # Gi to MB
                        else:
                            total_ram = ram_value  # Assume MB if no unit
                    else:
                        total_ram = 0
                    
                    # Parse embodied carbon
                    try:
                        embodied_carbon = float(annotations.get("hardware.carbon/embodied_emissions", "0")) * 1000.0  # Convert kg to grams
                    except ValueError:
                        embodied_carbon = 0
                    
                    # Parse lifetime
                    try:
                        lifetime_years = float(annotations.get("hardware.carbon/lifetime_years", "3"))
                    except ValueError:
                        lifetime_years = 3
                    
                    # Convert lifetime from years to hours
                    lifetime_hours = lifetime_years * 365 * 24
                    
                    # Parse power settings
                    power_settings = {
                        "idle": float(annotations.get("hardware.power/idle_watts", "100")),
                        # active kept for compatibility; not used in calculations
                        "active": float(annotations.get("hardware.power/active_watts", annotations.get("hardware.power/idle_watts", "100"))),
                        "max": float(annotations.get("hardware.power/max_watts", "400"))
                    }
                    
                    # Extract region for later forecasting
                    region_label = node.get("metadata", {}).get("labels", {}).get("topology.kubernetes.io/region", "")
                    
                    hardware_subcategory = node.get("metadata", {}).get("labels", {}).get("hardware.carbon/subcategory", "")

                    flavour = EnvironmentalFlavor(
                        id=node_id,
                        embodiedCarbon=embodied_carbon,
                        lifetime=lifetime_hours,  # In hours
                        totalCpu=total_cpu,
                        totalRam=total_ram,
                        totalStorage=1000 * 1024 * 1024 * 1024,  # Default 1TB
                        forecast={},  # Empty forecast, will be filled later
                        power=power_settings
                    )
                    attach_water_metadata(
                        flavour,
                        region=region_label,
                        hardware_subcategory=hardware_subcategory,
                        slot_count=24,
                    )
                    
                    flavours.append(flavour)
                    
                    logging.debug(f"Loaded node {node_id} with {total_cpu} CPU, {total_ram} RAM, " +
                                f"region={flavour.region}, embodied={embodied_carbon}, " +
                                f"lifetime={lifetime_hours}h, power={power_settings}")
            
            logging.info(f"Successfully loaded {len(flavours)} nodes from {nodes_file}")
            return flavours
            
        except Exception as e:
            logging.error(f"Error loading nodes from {nodes_file}: {e}")
            return []

    def _load_carbon_forecasts(self, forecasts_file: str) -> Dict[str, Dict[int, float]]:
        """
        Load carbon intensity directly from all_forecasts.json file.
        
        Args:
            forecasts_file: Path to all_forecasts.json file
            
        Returns:
            Dict mapping region codes to Dict of {timeslot_id: carbon_intensity}
        """
        import json
        from datetime import datetime
        import os
        import traceback
        
        logging.info(f"📊 Loading carbon forecasts from {forecasts_file}")
        
        # First check if file exists
        if not os.path.exists(forecasts_file):
            logging.error(f"❌ Carbon forecast file not found: {forecasts_file}")
            return {}
            
        # Check if file is empty
        if os.path.getsize(forecasts_file) == 0:
            logging.error(f"❌ Carbon forecast file is empty: {forecasts_file}")
            return {}
            
        forecasts = {}
        
        try:
            with open(forecasts_file, 'r') as f:
                logging.info(f"📂 Reading JSON content from {forecasts_file}")
                try:
                    data = json.load(f)
                    logging.info(f"✅ Successfully parsed JSON data")
                    logging.info(f"📊 Found {len(data)} regions in carbon forecast file")
                except json.JSONDecodeError as json_err:
                    logging.error(f"❌ Invalid JSON format in {forecasts_file}: {json_err}")
                    logging.error(traceback.format_exc())
                    return {}
                
            # Establish base time for calculating timeslot IDs
            base_time = None
            
            # Check if JSON is properly formatted
            if not isinstance(data, dict):
                logging.error(f"❌ Unexpected forecast data format: expected dict, got {type(data)}")
                return {}
                
            # Debug sample of data
            sample_region = next(iter(data.keys()), None) if data else None
            if sample_region:
                logging.info(f"📊 Sample region: {sample_region}")
                if isinstance(data[sample_region], dict) and "forecast" in data[sample_region]:
                    sample_forecast = data[sample_region]["forecast"]
                    logging.info(f"📊 Sample forecast entries: {len(sample_forecast)} entries")
                    if sample_forecast and len(sample_forecast) > 0:
                        sample_entry = sample_forecast[0]
                        logging.info(f"📊 Sample forecast entry: {sample_entry}")
                
            for region, region_data in data.items():
                if not isinstance(region_data, dict) or "forecast" not in region_data or not region_data["forecast"]:
                    logging.warning(f"⚠️ Region {region} has no forecast data, skipping")
                    continue
                    
                # Find base time if not set
                if base_time is None:
                    try:
                        first_datetime_str = region_data["forecast"][0]["datetime"]
                        base_time = datetime.fromisoformat(first_datetime_str.replace("Z", "+00:00"))
                        logging.info(f"⏰ Using base time {base_time} for timeslot calculations")
                    except (KeyError, IndexError) as e:
                        logging.error(f"❌ Error extracting datetime from forecast: {e}")
                        if not region_data["forecast"]:
                            logging.error("  - Forecast array is empty")
                        elif len(region_data["forecast"]) > 0:
                            logging.error(f"  - First forecast entry: {region_data['forecast'][0]}")
                        continue
                
                # Process forecast data
                forecast_dict = {}
                valid_entries = 0
                invalid_entries = 0
                
                for entry in region_data["forecast"]:
                    try:
                        if "carbonIntensity" not in entry or "datetime" not in entry:
                            logging.debug(f"⚠️ Missing required fields in forecast entry: {entry}")
                            invalid_entries += 1
                            continue
                            
                        carbon_intensity = entry["carbonIntensity"]
                        entry_time = datetime.fromisoformat(entry["datetime"].replace("Z", "+00:00"))
                        
                        # Calculate timeslot ID (hours since base time)
                        hours_diff = int((entry_time - base_time).total_seconds() / 3600)
                        timeslot_id = hours_diff
                        
                        forecast_dict[timeslot_id] = carbon_intensity
                        valid_entries += 1
                    except Exception as entry_err:
                        logging.debug(f"⚠️ Error processing forecast entry: {entry_err}")
                        invalid_entries += 1
                
                if valid_entries > 0:
                    forecasts[region] = forecast_dict
                    logging.info(f"✅ Region {region}: loaded {valid_entries} forecast entries, skipped {invalid_entries}")
                    
                    # Log some statistics about this region's forecast
                    if len(forecast_dict) > 0:
                        min_ts = min(forecast_dict.keys())
                        max_ts = max(forecast_dict.keys())
                        min_intensity = min(forecast_dict.values())
                        max_intensity = max(forecast_dict.values())
                        avg_intensity = sum(forecast_dict.values()) / len(forecast_dict)
                        
                        logging.info(f"📊 Region {region} forecast stats:")
                        logging.info(f"  - Timeslot range: {min_ts} to {max_ts} ({max_ts - min_ts + 1} hours)")
                        logging.info(f"  - Carbon intensity range: {min_intensity:.2f} to {max_intensity:.2f} gCO2/kWh")
                        logging.info(f"  - Average carbon intensity: {avg_intensity:.2f} gCO2/kWh")
                else:
                    logging.warning(f"⚠️ No valid forecast entries for region {region}")
            
            if forecasts:
                logging.info(f"✅ Successfully loaded carbon forecasts for {len(forecasts)} regions")
                all_timeslots = set()
                for region_forecast in forecasts.values():
                    all_timeslots.update(region_forecast.keys())
                    
                logging.info(f"📊 Total timeslots across all regions: {len(all_timeslots)}")
                logging.info(f"⏱️ Timeslot range: {min(all_timeslots)} to {max(all_timeslots)}")
                return forecasts
            else:
                logging.error(f"❌ No valid forecast data found in {forecasts_file}")
                return {}
            
        except Exception as e:
            logging.error(f"❌ Error loading carbon forecasts from {forecasts_file}: {e}")
            logging.error(traceback.format_exc())
            return {}
            
    def _extract_region_from_node_id(self, node_id: str) -> str:
        """
        Extract the region code from a node ID.
        
        Args:
            node_id: Node identifier (e.g., node-0-de-iot)
            
        Returns:
            Region code (e.g., DE)
        """
        import re
        
        # Try to extract region from standard naming patterns
        match = re.search(r'-([a-z]{2})(-[a-z]+)?$', node_id.lower())
        if match:
            return match.group(1).upper()
        
        # For IT-NO style regions
        match = re.search(r'-([a-z]{2}-[a-z]{2})(-[a-z]+)?$', node_id.lower())
        if match:
            return match.group(1).upper()
            
        # Default fallback
        return "DE"
        
    def _extract_pods_from_yaml(self, yaml_file: str) -> List[CarbonAwarePod]:
        """
        Extract pod definitions from a timeslot_X.yaml file.
        
        Args:
            yaml_file: Path to timeslot_X.yaml file
            
        Returns:
            List of CarbonAwarePod objects
        """
        import yaml
        import re
        from datetime import datetime
        
        logging.debug(f"Extracting pods from {yaml_file}")
        pods = []
        
        try:
            with open(yaml_file, 'r') as f:
                # Handle multi-document YAML files (separated by ---) 
                all_documents = list(yaml.safe_load_all(f))
            
            # Extract the timeslot ID from the filename
            timeslot_id = 0
            match = re.search(r'timeslot_(\d+)\.yaml$', yaml_file)
            if match:
                timeslot_id = int(match.group(1))
            
            # Extract the timestamp if available from any document
            reference_time = datetime.now()
            
            # Process each document (Kubernetes Deployment)
            for doc in all_documents:
                if not doc or not isinstance(doc, dict):
                    continue
                    
                # Skip non-Deployment documents
                if doc.get('kind') != 'Deployment':
                    continue
                    
                # Extract pod info from Kubernetes Deployment
                metadata = doc.get('metadata', {})
                deployment_name = metadata.get('name', '')
                
                if not deployment_name:
                    continue
                
                # Parse duration and deadline from deployment name
                duration = 1.0  # Default 1 hour
                deadline = 24.0  # Default 24 hours
                
                # Try to extract from name (e.g., mXXX-duration-6h-deadline-12h)
                name_duration_match = re.search(r'-duration-(\d+)h', deployment_name)
                if name_duration_match:
                    duration = float(name_duration_match.group(1))
                
                name_deadline_match = re.search(r'-deadline-(\d+)h', deployment_name)
                if name_deadline_match:
                    deadline = float(name_deadline_match.group(1))
                
                # Extract resource requirements from container spec
                spec = doc.get('spec', {})
                template = spec.get('template', {})
                pod_spec = template.get('spec', {})
                containers = pod_spec.get('containers', [])
                
                if not containers:
                    logging.warning(f"No containers found in deployment {deployment_name}")
                    continue
                
                # Get resource requests from first container
                container = containers[0]
                resources = container.get('resources', {})
                requests = resources.get('requests', {})
                
                # Parse CPU request (e.g., "1000m" -> 1.0)
                cpu_str = requests.get('cpu', '100m')
                if cpu_str.endswith('m'):
                    cpu_request = float(cpu_str[:-1]) / 1000.0
                else:
                    cpu_request = float(cpu_str)
                
                # Parse memory request (e.g., "1Gi" -> 1024 MB)
                memory_str = requests.get('memory', '128Mi')
                if memory_str.endswith('Mi'):
                    ram_request = float(memory_str[:-2])
                elif memory_str.endswith('Gi'):
                    ram_request = float(memory_str[:-2]) * 1024
                elif memory_str.endswith('Ki'):
                    ram_request = float(memory_str[:-2]) / 1024
                else:
                    ram_request = float(memory_str)  # Assume MB
                
                # Create the pod
                pod = CarbonAwarePod(
                    id=deployment_name,
                    deadline_hours=deadline,
                    duration=duration,
                    powerConsumption=0.0,  # Will be calculated during emission computation
                    cpuRequest=cpu_request,
                    ramRequest=ram_request,
                    storageRequest=100 * 1024 * 1024,  # Default 100MB
                    reference_time=reference_time
                )
                
                # Set earliest_timeslot based on the file
                pod.earliest_timeslot = timeslot_id
                
                pods.append(pod)
                logging.debug(f"Extracted pod {deployment_name} with duration={duration}h, " +
                            f"deadline={deadline}h, CPU={cpu_request}, RAM={ram_request}")
            
            # Legacy format support - if no Deployments found, try old microservices format
            if not pods:
                for doc in all_documents:
                    if doc and isinstance(doc, dict) and "microservices" in doc:
                        # Process each microservice (pod) in legacy format
                        for ms in doc.get("microservices", []):
                            if not ms or "name" not in ms:
                                continue
                            
                            # Parse duration and deadline from name or fields
                            duration = 1.0  # Default 1 hour
                            deadline = 24.0  # Default 24 hours
                            
                            # Try to extract from name first (e.g., mXXX-duration-6h-deadline-12h)
                            name_duration_match = re.search(r'-duration-(\d+)h', ms["name"])
                            if name_duration_match:
                                duration = float(name_duration_match.group(1))
                            
                            name_deadline_match = re.search(r'-deadline-(\d+)h', ms["name"])
                            if name_deadline_match:
                                deadline = float(name_deadline_match.group(1))
                            
                            # If not in name, check for explicit fields
                            if "duration" in ms:
                                duration_str = ms["duration"]
                                if duration_str.endswith('h'):
                                    duration = float(duration_str[:-1])
                                else:
                                    duration = float(duration_str)
                            
                            if "deadline" in ms:
                                deadline_str = ms["deadline"]
                                if deadline_str.endswith('h'):
                                    deadline = float(deadline_str[:-1])
                                else:
                                    deadline = float(deadline_str)
                            
                            # Extract CPU and RAM requirements
                            cpu_request = float(ms.get("cpu", "0.1"))
                            ram_request = float(ms.get("ram", "128"))  # Default 128MB
                            
                            # Create the pod
                            pod = CarbonAwarePod(
                                id=ms["name"],
                                deadline_hours=deadline,
                                duration=duration,
                                powerConsumption=0.0,  # Will be calculated during emission computation
                                cpuRequest=cpu_request,
                                ramRequest=ram_request,
                                storageRequest=100 * 1024 * 1024,  # Default 100MB
                                reference_time=reference_time
                            )
                            
                            # Set earliest_timeslot based on the file
                            pod.earliest_timeslot = timeslot_id
                            
                            pods.append(pod)
                            logging.debug(f"Extracted pod {ms['name']} with duration={duration}h, " +
                                        f"deadline={deadline}h, CPU={cpu_request}, RAM={ram_request}")
            
            logging.info(f"Extracted {len(pods)} pods from {yaml_file}")
            return pods
            
        except Exception as e:
            logging.error(f"Error extracting pods from {yaml_file}: {e}")
            return []
            
    def _initialize_resource_state(self):
        """Initialize resource availability for all nodes and timeslots."""
        # Initialize resource state
        self.resource_state = {
            "cpu": {},
            "ram": {}
        }
        
        max_timeslot = max(ts.id for ts in self.all_timeslots) if self.all_timeslots else 24
        
        for flv in self.all_flavours:
            self.resource_state["cpu"][flv.id] = {}
            self.resource_state["ram"][flv.id] = {}
            
            # For each timeslot, initialize full capacity 
            for ts_id in range(max_timeslot + 1):
                self.resource_state["cpu"][flv.id][ts_id] = flv.totalCpu
                self.resource_state["ram"][flv.id][ts_id] = flv.totalRam

    def precompute_all_workloads(self, workloads_dir: str, nodes_file: str, forecasts_file: str):
        """
        Load all pods from all timeslot files and solve them in one comprehensive global optimization.
        
        This method:
        1. Loads all nodes from nodes.yaml
        2. Loads carbon intensity from all_forecasts.json
        3. Loads all pods from all timeslot_*.yaml files
        4. Solves the global optimization problem for all pods at once
        
        Args:
            workloads_dir: Directory containing timeslot_*.yaml files
            nodes_file: Path to nodes.yaml
            forecasts_file: Path to all_forecasts.json
        
        Returns:
            True if optimization was successful, False otherwise
        """
        import os
        import traceback
        
        logging.info(f"🚀 Starting comprehensive global optimization with all workloads")
        logging.info(f"- Workloads directory: {workloads_dir}")
        logging.info(f"- Nodes file: {nodes_file}")
        logging.info(f"- Forecasts file: {forecasts_file}")
        
        try:
            # 1. Load all nodes from nodes.yaml
            logging.info(f"📂 STEP 1: Loading infrastructure from {nodes_file}")
            self.all_flavours = self._load_nodes_from_yaml(nodes_file)
            if not self.all_flavours:
                logging.error("❌ Failed to load nodes. Aborting optimization.")
                return False
            
            logging.info(f"✅ SUCCESS: Loaded {len(self.all_flavours)} nodes")
            
            # Log node details for debugging
            for flv in self.all_flavours:
                logging.info(f"  - Node {flv.id}: {flv.totalCpu} CPU, {flv.totalRam} RAM, region={flv.region if hasattr(flv, 'region') else 'unknown'}")
        except Exception as e:
            logging.error(f"❌ FAILED: Error loading nodes from {nodes_file}: {e}")
            logging.error(traceback.format_exc())
            return False
        
        try:
            # 2. Load carbon intensity from all_forecasts.json
            logging.info(f"📂 STEP 2: Loading carbon intensity forecasts from {forecasts_file}")
            carbon_forecasts = self._load_carbon_forecasts(forecasts_file)
            if not carbon_forecasts:
                logging.error("❌ Failed to load carbon forecasts. Aborting optimization.")
                return False
                
            logging.info(f"✅ SUCCESS: Loaded carbon forecasts for {len(carbon_forecasts)} regions")
            
            # Log forecast details for debugging
            for region, forecast in carbon_forecasts.items():
                logging.info(f"  - Region {region}: {len(forecast)} timeslots of forecast data")
                if len(forecast) > 0:
                    min_value = min(forecast.values())
                    max_value = max(forecast.values())
                    logging.info(f"    - Range: {min_value:.2f} - {max_value:.2f} gCO2/kWh")
                    sample_timeslots = sorted(list(forecast.keys()))[:5]  # First 5 timeslots
                    logging.info(f"    - Sample timeslots: {sample_timeslots}")
                    for ts in sample_timeslots[:3]:  # Show first 3
                        logging.info(f"      - Timeslot {ts}: {forecast[ts]:.2f} gCO2/kWh")
        except Exception as e:
            logging.error(f"❌ FAILED: Error loading carbon forecasts from {forecasts_file}: {e}")
            logging.error(traceback.format_exc())
            return False
            
        try:
            # 3. Attach carbon forecasts to flavours
            logging.info(f"🔄 STEP 3: Attaching carbon forecasts to nodes")
            for flv in self.all_flavours:
                # Extract region from node ID if not already set
                if not hasattr(flv, 'region') or not flv.region:
                    flv.region = self._extract_region_from_node_id(flv.id)
                    logging.info(f"  - Extracted region {flv.region} for node {flv.id}")
                    
                # Assign forecast data to the node
                if flv.region in carbon_forecasts:
                    flv.forecast = carbon_forecasts[flv.region]
                    logging.info(f"  - Assigned {len(flv.forecast)} forecast data points to node {flv.id} (region {flv.region})")
                else:
                    logging.warning(f"⚠️ No carbon forecast found for region {flv.region} (node {flv.id})")
                    # Use first region's forecast as fallback
                    if carbon_forecasts:
                        fallback_region = next(iter(carbon_forecasts.keys()))
                        flv.forecast = carbon_forecasts[fallback_region]
                        logging.warning(f"   Using {fallback_region} as fallback for {flv.id}")
            logging.info(f"✅ SUCCESS: Carbon forecasts attached to all nodes")
        except Exception as e:
            logging.error(f"❌ FAILED: Error attaching forecasts to nodes: {e}")
            logging.error(traceback.format_exc())
            return False
        
        try:
            # 4. Load all pods from all timeslot_*.yaml files
            logging.info(f"📂 STEP 4: Loading pods from timeslot YAML files in {workloads_dir}")
            
            all_pods = []
            self._workloads_dir = workloads_dir
            if not os.path.isdir(workloads_dir):
                logging.error(f"❌ FAILED: Workloads directory {workloads_dir} does not exist")
                return False
                
            yaml_files = [f for f in os.listdir(workloads_dir) 
                         if f.startswith("timeslot_") and f.endswith(".yaml")]
            yaml_files.sort()  # Sort to ensure chronological order
            
            if not yaml_files:
                logging.error(f"❌ FAILED: No timeslot_*.yaml files found in {workloads_dir}")
                return False
                
            logging.info(f"  - Found {len(yaml_files)} timeslot files to process: {yaml_files}")
            
            pods_by_file = {}  # Track pods by source file
            
            for yaml_file in sorted(yaml_files):
                file_path = os.path.join(workloads_dir, yaml_file)
                try:
                    pods = self._extract_pods_from_yaml(file_path)
                    pods_by_file[yaml_file] = pods
                    all_pods.extend(pods)
                    logging.info(f"  - Loaded {len(pods)} pods from {yaml_file}")
                    if pods:
                        for pod in pods[:3]:  # Log details of first 3 pods
                            logging.info(f"    - Pod {pod.id}: CPU={pod.cpuRequest}, RAM={pod.ramRequest}, Duration={pod.duration}h, Earliest TS={pod.earliest_timeslot}")
                except Exception as e:
                    logging.error(f"❌ Error processing {yaml_file}: {e}")
            
            if not all_pods:
                logging.error("❌ FAILED: No pods found in any timeslot files")
                return False
                
            logging.info(f"✅ SUCCESS: Loaded {len(all_pods)} total pods from {len(yaml_files)} timeslot files")
        except Exception as e:
            logging.error(f"❌ FAILED: Error loading pods from timeslot files: {e}")
            logging.error(traceback.format_exc())
            return False
            
        try:
            # 5. Create timeslots
            logging.info(f"🔄 STEP 5: Creating timeslots for scheduling horizon")
            max_timeslot_id = max(pod.earliest_timeslot for pod in all_pods) + 24  # Add 24h window for scheduling
            self.all_timeslots = []
            
            for i in range(max_timeslot_id + 1):
                self.all_timeslots.append(CarbonAwareTimeslot(id=i, start_time=time.time() + i*3600, length=1))
            
            logging.info(f"✅ SUCCESS: Created {len(self.all_timeslots)} timeslots for scheduling")
        except Exception as e:
            logging.error(f"❌ FAILED: Error creating timeslots: {e}")
            logging.error(traceback.format_exc())
            return False
            
        try:
            # 6. Initialize resource state
            logging.info(f"🔄 STEP 6: Initializing resource state for all nodes")
            self._initialize_resource_state()
            logging.info(f"✅ SUCCESS: Initialized resource state for {len(self.all_flavours)} nodes")
        except Exception as e:
            logging.error(f"❌ FAILED: Error initializing resource state: {e}")
            logging.error(traceback.format_exc())
            return False
            
        try:
            # 7. Run the global optimization with all data
            logging.info(f"🧮 STEP 7: Starting global optimization with {len(all_pods)} pods across {len(self.all_timeslots)} timeslots")
            self.pending_pods = all_pods
            
            # Ensure CSV log is set up before optimization
            if hasattr(self, 'setup_session_placement_log'):
                logging.info("📊 Setting up session placement log for recording results")
                self.setup_session_placement_log()
                
            # Solve the optimization problem
            logging.info("🔍 Starting MILP solver for global optimization problem")
            
            # 🚨 CRITICAL: Use thread-safe lock to prevent race conditions (batch path)
            if not self._solving_lock.acquire(blocking=False):
                logging.warning("🚫 Solver already running at batch call site. Skipping to prevent race condition.")
                return False
            
            try:
                # Set precomputation mode flag - CSV should be written only during precomputation
                self.is_precomputation_mode = True
                logging.info(f"🔧 Entering precomputation mode - CSV will be written for all placements")
                
                self.solve_global_optimization(
                    self.all_flavours,
                    self.all_timeslots,
                    self.resource_state["cpu"],
                    self.resource_state["ram"],
                    len(self.all_timeslots)
                )
                
                # Exit precomputation mode
                self.is_precomputation_mode = False
                logging.info(f"🔧 Exiting precomputation mode")
            finally:
                # 🚨 CRITICAL: Always release the lock
                self._solving_lock.release()
                logging.debug("🔓 Solver lock released at batch call site")
            
            # Set flags
            self.optimization_done = self.has_solved
            
            # Log results
            if self.optimization_done:
                logging.info(f"🎉 Comprehensive global optimization complete!")
                logging.info(f"✅ Successfully placed {len(self.global_solution)}/{len(all_pods)} pods")
                
                # Verify the CSV file was created and data was written
                csv_path = self._session_csv_path if hasattr(self, '_session_csv_path') else None
                if csv_path and os.path.exists(csv_path):
                    file_size = os.path.getsize(csv_path)
                    row_count = 0
                    
                    # Count rows in CSV
                    import csv
                    try:
                        with open(csv_path, 'r') as f:
                            row_count = sum(1 for _ in csv.reader(f)) - 1  # Subtract header row
                    except Exception as csv_err:
                        logging.error(f"Error counting CSV rows: {csv_err}")
                        
                    logging.info(f"📊 CSV output file: {csv_path}")
                    logging.info(f"   - File size: {file_size} bytes")
                    logging.info(f"   - Row count: {row_count} (excluding header)")
                    
                    if row_count <= 0:
                        logging.error("⚠️ CSV file appears empty (only contains header row)")
                    elif row_count != len(self.global_solution):
                        logging.error(f"⚠️ CSV row count ({row_count}) doesn't match solution count ({len(self.global_solution)})")
                else:
                    logging.error(f"⚠️ CSV file not found or not properly configured: {csv_path}")
                
                # Log pod placement statistics by node
                node_placements = {}
                for pod_id, (node_id, ts_id, emissions) in self.global_solution.items():
                    if node_id not in node_placements:
                        node_placements[node_id] = 0
                    node_placements[node_id] += 1
                    
                for node_id, count in node_placements.items():
                    logging.info(f"  - {node_id}: {count} pods placed")
                
                # Log any unplaced pods
                if len(self.global_solution) < len(all_pods):
                    unplaced_pods = [p.id for p in all_pods if p.id not in self.global_solution]
                    logging.warning(f"⚠️ {len(unplaced_pods)} pods could not be placed")
                    if len(unplaced_pods) <= 10:  # Only list if not too many
                        for pod_id in unplaced_pods:
                            logging.warning(f"  - {pod_id}")
                            
                # Show sample of results
                if self.global_solution:
                    sample = list(self.global_solution.items())[:5]  # Show first 5
                    logging.info("📊 Sample placements:")
                    for pod_id, (node_id, ts_id, emissions) in sample:
                        logging.info(f"  - Pod {pod_id} -> Node {node_id}, Timeslot {ts_id}, Emissions {emissions/1000.0:.6f}kgCO2e ({emissions:.2f}gCO2e)")
                
                # Generate placement summary automatically
                try:
                    if hasattr(self, '_session_csv_path') and self._session_csv_path:
                        from carbon_aware.placement_summary import generate_placement_summary
                        summary_path = generate_placement_summary(
                            self._session_csv_path, 
                            "global-optimal", 
                            os.path.dirname(self._session_csv_path),
                            workloads_dir=workloads_dir,
                        )
                        if summary_path:
                            logging.info(f"📋 Placement summary generated: {summary_path}")
                        else:
                            logging.warning("⚠️ Could not generate placement summary")
                except Exception as e:
                    logging.warning(f"⚠️ Failed to generate placement summary: {e}")
                        
                return True
            else:
                logging.error("❌ Comprehensive global optimization failed!")
                logging.error(f"  - Solver status: {self.status}")
                logging.error(f"  - Solver iterations: {self.iterations}")
                return False
        except Exception as e:
            logging.error(f"❌ FAILED: Error during global optimization: {e}")
            logging.error(traceback.format_exc())
            return False
