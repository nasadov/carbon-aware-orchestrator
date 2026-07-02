from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_PYTHON_ROOT = REPO_ROOT / "pkg" / "carbon-aware" / "server-python"
if str(SERVER_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_PYTHON_ROOT))

from carbon_aware.no_harm_flexibility import (  # noqa: E402
    PilotConfig,
    build_action_signals,
    build_greedy_schedule,
    load_flavours_for_pilot,
    load_pods,
    repair_schedule_no_harm,
    run_no_harm_flexibility_pilot,
    schedule_totals,
    _rematerialize_under_realized,
    _z_score,
    _cantelli_factor,
    _budget_gamma,
    dro_per_move_buffer,
    sqrtk_per_move_buffer,
)


def _pilot_config(output_dir: Path, max_pods: int = 30) -> PilotConfig:
    return PilotConfig(
        repo_root=REPO_ROOT,
        nodes_file=REPO_ROOT / "pkg" / "carbon-aware" / "nodes.yaml",
        workloads_dir=REPO_ROOT / "pkg" / "carbon-aware" / "workloads",
        forecasts_file=REPO_ROOT / "pkg" / "carbon-aware" / "server-python" / "all_forecasts.json",
        config_file=REPO_ROOT / "pkg" / "carbon-aware" / "infra-workload-config.yaml",
        output_dir=output_dir,
        max_pods=max_pods,
        scenario="heatwave-drought",
    )


def test_heatwave_drought_signals_are_temporal_and_water_stressed(tmp_path: Path) -> None:
    config = _pilot_config(tmp_path)
    flavours = load_flavours_for_pilot(config)
    signals = build_action_signals(flavours, config)

    assert len(signals) == 4 * config.max_timeslots
    assert any(signal.grid_stress for signal in signals.values())
    assert any(signal.clean_headroom for signal in signals.values())
    # Drought guardrail retired under watershed (basin) resolution: no DC basin reaches the CF>=20
    # threshold (max ~7.7), so the binary guard is inert by design -- the per-basin Delta W_b <= 0
    # guard provides drought protection instead. (Water-pipeline revamp, Option A: direct=basin CF,
    # indirect=generation-country CF; the CF>=20 trigger was a country-aggregation artifact.)
    assert not any(signal.drought_guardrail for signal in signals.values())

    assert {signal.grid_signal_source for signal in signals.values()} == {"opsd_residual_load_lite"}

    slot_zero_residuals = {
        signal.residual_load_proxy_mw
        for signal in signals.values()
        if signal.slot == 0
    }
    assert len(slot_zero_residuals) > 1


def test_no_harm_flexibility_pilot_writes_certificate_and_improves_stress(tmp_path: Path) -> None:
    result = run_no_harm_flexibility_pilot(_pilot_config(tmp_path, max_pods=80))
    rows = {row["method_key"]: row for row in result["summary_rows"]}

    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "no_harm_certificate.json").exists()
    assert (tmp_path / "placements_no_harm_flex.csv").exists()

    flex = rows["no_harm_flex"]
    search_control = rows["no_harm_search_control"]
    packing = rows["packing"]

    assert flex["no_harm_certificate"] is True
    assert flex["placed_pods"] == packing["placed_pods"]
    assert flex["carbon_delta_kg"] <= 0.0
    assert flex["scarcity_delta"] <= 0.0
    assert flex["stress_kwh_avoided"] > 0.0
    assert flex["stress_kwh_avoided"] >= search_control["stress_kwh_avoided"] - 1e-12
    assert flex["weighted_stress_kwh"] <= search_control["weighted_stress_kwh"] + 1e-12


def test_dro_buffer_factors_and_gamma_budget() -> None:
    """The DRO buffer math (Tier-1-D): distribution-free Cantelli factor + Bertsimas-Sim Gamma budget."""
    import math

    eps = 0.05
    # Cantelli is the distribution-free one-sided factor sqrt((1-eps)/eps); ~2.65x the Gaussian z.
    assert abs(_cantelli_factor(eps) - math.sqrt((1.0 - eps) / eps)) < 1e-12
    assert _cantelli_factor(eps) > _z_score(eps)
    assert abs(_cantelli_factor(eps) / _z_score(eps) - 2.65) < 0.05

    # Gamma budget interpolates sqrt(K) (independence) <-> K (full correlation) via fraction c in [0,1].
    K = 30
    assert abs(_budget_gamma(gamma=0.0, gamma_mode="frac", k_moves=K) - math.sqrt(K)) < 1e-9
    assert abs(_budget_gamma(gamma=1.0, gamma_mode="frac", k_moves=K) - float(K)) < 1e-9
    assert math.sqrt(K) < _budget_gamma(gamma=0.5, gamma_mode="frac", k_moves=K) < float(K)
    # Out-of-range Gamma is clamped to [sqrt(K), K].
    assert _budget_gamma(gamma=2.0, gamma_mode="frac", k_moves=K) == float(K)

    # Total reserve = per_move * K = cantelli * rho * shape * Gamma (the Gamma-scaling guarantee).
    rho, shape = 0.3, 1.0
    for c in (0.0, 0.5, 1.0):
        pm = dro_per_move_buffer(rho=rho, epsilon=eps, shape=shape, k_moves=K, gamma=c, gamma_mode="frac")
        gamma = _budget_gamma(gamma=c, gamma_mode="frac", k_moves=K)
        assert abs(pm * K - _cantelli_factor(eps) * rho * shape * gamma) < 1e-9
    # dro at c=0 (independence) differs from sqrtk ONLY by the Cantelli-vs-Gaussian factor.
    pm_dro0 = dro_per_move_buffer(rho=rho, epsilon=eps, shape=shape, k_moves=K, gamma=0.0, gamma_mode="frac")
    pm_sk = sqrtk_per_move_buffer(rho=rho, epsilon=eps, shape=shape, k_moves=K)
    assert abs((pm_dro0 / pm_sk) - (_cantelli_factor(eps) / _z_score(eps))) < 1e-9


def test_dro_buffer_off_is_bit_identical(tmp_path: Path) -> None:
    """The DRO buffer is additive and flag-gated: with robust_buffer_mode left at the default 'flat'
    and no buffer set, enabling the new config fields must NOT change the schedule (bit-identical)."""
    config = _pilot_config(tmp_path, max_pods=80)
    flavours = load_flavours_for_pilot(config)
    signals = build_action_signals(flavours, config)
    pods = load_pods(config.workloads_dir, max_pods=config.max_pods)
    packing, _, _ = build_greedy_schedule(method_key="packing", pods=pods, flavours=flavours, config=config)

    base = repair_schedule_no_harm(
        method_key="no_harm_flex", baseline=packing, pods=pods, flavours=flavours,
        signals=signals, config=config, score_mode="combined",
    )
    # Default DRO config fields present but mode is still 'flat' -> identical placements.
    from dataclasses import replace as dc_replace
    cfg2 = dc_replace(config, robust_buffer_gamma=0.7, robust_buffer_gamma_mode="frac")
    same = repair_schedule_no_harm(
        method_key="no_harm_flex", baseline=packing, pods=pods, flavours=flavours,
        signals=signals, config=cfg2, score_mode="combined",
    )
    assert len(base.placements) == len(same.placements)
    a = {p.pod.id: (p.candidate.flavour.id, p.candidate.timeslot.id) for p in base.placements}
    b = {p.pod.id: (p.candidate.flavour.id, p.candidate.timeslot.id) for p in same.placements}
    assert a == b


def test_dro_buffer_mode_certifies_and_reserves_more_than_sqrtk(tmp_path: Path) -> None:
    """The DRO buffer mode runs end-to-end, keeps the ex-post certificate, and (being distribution-free
    + correlation-budgeted) reserves at least as much as sqrtk -> never MORE repairs than sqrtk."""
    config = _pilot_config(tmp_path, max_pods=80)
    flavours = load_flavours_for_pilot(config)
    signals = build_action_signals(flavours, config)
    pods = load_pods(config.workloads_dir, max_pods=config.max_pods)
    packing, _, _ = build_greedy_schedule(method_key="packing", pods=pods, flavours=flavours, config=config)

    from dataclasses import replace as dc_replace
    sqrtk_cfg = dc_replace(config, robust_buffer_mode="sqrtk", robust_buffer_rho=0.237,
                           robust_buffer_epsilon=0.05)
    dro_cfg = dc_replace(config, robust_buffer_mode="dro", robust_buffer_rho=0.237,
                         robust_buffer_epsilon=0.05, robust_buffer_gamma=1.0, robust_buffer_gamma_mode="frac")

    sqrtk = repair_schedule_no_harm(
        method_key="no_harm_flex", baseline=packing, pods=pods, flavours=flavours,
        signals=signals, config=sqrtk_cfg, score_mode="combined",
    )
    dro = repair_schedule_no_harm(
        method_key="no_harm_flex", baseline=packing, pods=pods, flavours=flavours,
        signals=signals, config=dro_cfg, score_mode="combined",
    )
    # Both certify ex-post: realized carbon/scarcity never exceed the packing baseline.
    base = schedule_totals(packing, signals)
    for sched in (sqrtk, dro):
        _rematerialize_under_realized(sched, flavours, config)
        t = schedule_totals(sched, signals)
        assert t["carbon_kg"] <= base["carbon_kg"] + 1e-6
        assert t["scarcity_water"] <= base["scarcity_water"] + 1e-6
    # DRO (Cantelli x Gamma=K) reserves more -> blocks at least as many moves as sqrtk.
    assert dro.repairs_applied <= sqrtk.repairs_applied


def test_footprint_accounting_is_idle_consistent_after_repair(tmp_path: Path) -> None:
    """Regression guard for the idle-attribution bug: node idle power must be charged exactly once
    per active node-slot, even after the repair moves pods. We assert (a) re-materialization is
    idempotent (the accounting has reached an occupancy-consistent fixed point, not stale per-pod
    footprints that lose a node's idle when its idle-bearer moves), and (b) the certificate is sound
    (the repaired+rematerialized flex never exceeds the packing baseline on carbon or scarcity)."""
    config = _pilot_config(tmp_path, max_pods=80)
    flavours = load_flavours_for_pilot(config)
    signals = build_action_signals(flavours, config)
    pods = load_pods(config.workloads_dir, max_pods=config.max_pods)

    packing, _, _ = build_greedy_schedule(method_key="packing", pods=pods, flavours=flavours, config=config)
    flex = repair_schedule_no_harm(
        method_key="no_harm_flex", baseline=packing, pods=pods, flavours=flavours,
        signals=signals, config=config, score_mode="flexibility",
    )

    _rematerialize_under_realized(flex, flavours, config)
    t1 = schedule_totals(flex, signals)
    _rematerialize_under_realized(flex, flavours, config)
    t2 = schedule_totals(flex, signals)

    # (a) idempotent => idle counted once per active node-slot (no stale per-pod idle to re-shuffle).
    assert abs(t1["carbon_kg"] - t2["carbon_kg"]) < 1e-6
    assert abs(t1["scarcity_water"] - t2["scarcity_water"]) < 1e-6
    # (b) sound certificate on correct accounting: flex must not exceed the baseline on either axis.
    base = schedule_totals(packing, signals)
    assert t1["carbon_kg"] <= base["carbon_kg"] + 1e-6
    assert t1["scarcity_water"] <= base["scarcity_water"] + 1e-6
