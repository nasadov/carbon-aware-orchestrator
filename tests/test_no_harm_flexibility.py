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


def _placement_map(result):
    return {p.pod.id: (p.candidate.flavour.id, p.candidate.timeslot.id) for p in result.placements}


def _prep(config):
    from carbon_aware.no_harm_flexibility import build_action_signals as _bas
    flavours = load_flavours_for_pilot(config)
    signals = _bas(flavours, config)
    pods = load_pods(config.workloads_dir, max_pods=config.max_pods)
    packing, _, _ = build_greedy_schedule(method_key="packing", pods=pods, flavours=flavours, config=config)
    return flavours, signals, pods, packing


# ------------------------------------------------------------------ F1-B1: regret-based move ranking
def test_regret_ranking_off_is_bit_identical(tmp_path: Path) -> None:
    """Regret ranking is flag-gated: the default (regret_ranking=False) must produce byte-identical
    placements to a config that never sets the field."""
    from dataclasses import replace as dc_replace
    config = _pilot_config(tmp_path, max_pods=80)
    flavours, signals, pods, packing = _prep(config)
    base = repair_schedule_no_harm(method_key="no_harm_flex", baseline=packing, pods=pods,
                                   flavours=flavours, signals=signals, config=config, score_mode="combined")
    same = repair_schedule_no_harm(method_key="no_harm_flex", baseline=packing, pods=pods, flavours=flavours,
                                   signals=signals, config=dc_replace(config, regret_ranking=False),
                                   score_mode="combined")
    assert _placement_map(base) == _placement_map(same)


def test_regret_ranking_preserves_certificate(tmp_path: Path) -> None:
    """Regret ranking is RANKING-ONLY: the guard is untouched, so the envelope must still certify
    (carbon<=B, scarcity<=B, no pods dropped) on correct occupancy-based accounting."""
    from dataclasses import replace as dc_replace
    config = _pilot_config(tmp_path, max_pods=80)
    flavours, signals, pods, packing = _prep(config)
    reg = repair_schedule_no_harm(method_key="no_harm_flex", baseline=packing, pods=pods, flavours=flavours,
                                  signals=signals, config=dc_replace(config, regret_ranking=True),
                                  score_mode="combined")
    _rematerialize_under_realized(reg, flavours, config)
    t = schedule_totals(reg, signals)
    base = schedule_totals(packing, signals)
    assert t["carbon_kg"] <= base["carbon_kg"] + 1e-6
    assert t["scarcity_water"] <= base["scarcity_water"] + 1e-6
    assert t["placed_pods"] >= base["placed_pods"]


# ------------------------------------------------------------------ F1-B3: portfolio selector
def test_portfolio_selector_off_is_bit_identical(tmp_path: Path) -> None:
    """With the selector off (default), the pilot emits NO no_harm_portfolio row/file and the summary
    rows are exactly those of the six base methods."""
    result = run_no_harm_flexibility_pilot(_pilot_config(tmp_path, max_pods=80))
    keys = [r["method_key"] for r in result["summary_rows"]]
    assert "no_harm_portfolio" not in keys
    assert not (tmp_path / "placements_no_harm_portfolio.csv").exists()
    assert "portfolio" not in result["certificate"]


def test_portfolio_selector_ships_best_certified(tmp_path: Path) -> None:
    """With the selector on, the shipped no_harm_portfolio row (a) carries the certificate, (b) is drawn
    from a certified method, and (c) is no worse than the envelope on the declared leading axis (carbon
    reduction here) -- the portfolio dominates the envelope by construction."""
    from dataclasses import replace as dc_replace
    config = dc_replace(_pilot_config(tmp_path, max_pods=80),
                        portfolio_selector=True, portfolio_priority="carbon,water,relief")
    result = run_no_harm_flexibility_pilot(config)
    rows = {r["method_key"]: r for r in result["summary_rows"]}
    assert "no_harm_portfolio" in rows
    port = rows["no_harm_portfolio"]
    assert port["no_harm_certificate"] is True
    src = port["portfolio_source"]
    assert rows[src]["no_harm_certificate"] is True
    # Declared leading axis = carbon reduction: portfolio >= envelope (more negative or equal delta).
    assert port["carbon_delta_pct"] <= rows["no_harm_flex"]["carbon_delta_pct"] + 1e-9
    assert (tmp_path / "placements_no_harm_portfolio.csv").exists()
    assert result["certificate"]["portfolio"]["source"] == src


# ------------------------------------------------------------------ F1-B4: epsilon non-inferiority margins
def test_epsilon_margins_off_is_bit_identical(tmp_path: Path) -> None:
    """All epsilon margins default to 0.0 -> base*(1+0)==base -> byte-identical placements + no
    epsilon_margins block in the certificate."""
    from dataclasses import replace as dc_replace
    config = _pilot_config(tmp_path, max_pods=80)
    flavours, signals, pods, packing = _prep(config)
    base = repair_schedule_no_harm(method_key="no_harm_flex", baseline=packing, pods=pods,
                                   flavours=flavours, signals=signals, config=config, score_mode="combined")
    cfg0 = dc_replace(config, epsilon_carbon=0.0, epsilon_scarcity=0.0,
                      epsilon_radiation=0.0, epsilon_cost=0.0)
    same = repair_schedule_no_harm(method_key="no_harm_flex", baseline=packing, pods=pods, flavours=flavours,
                                   signals=signals, config=cfg0, score_mode="combined")
    assert _placement_map(base) == _placement_map(same)
    result = run_no_harm_flexibility_pilot(_pilot_config(tmp_path / "pilot", max_pods=80))
    assert "epsilon_margins" not in result["certificate"]


def test_epsilon_margin_relaxes_guard_within_declared_band(tmp_path: Path) -> None:
    """A positive epsilon relaxes the guard to base*(1+eps): the repair may accept moves the strict
    guard rejects (>= as many repairs), realized carbon stays within the DECLARED band base*(1+eps),
    and the certificate records the margin vector."""
    from dataclasses import replace as dc_replace
    config = _pilot_config(tmp_path, max_pods=80)
    flavours, signals, pods, packing = _prep(config)
    base = repair_schedule_no_harm(method_key="no_harm_flex", baseline=packing, pods=pods,
                                   flavours=flavours, signals=signals, config=config, score_mode="combined")
    eps = 0.05
    relaxed = repair_schedule_no_harm(method_key="no_harm_flex", baseline=packing, pods=pods, flavours=flavours,
                                      signals=signals, config=dc_replace(config, epsilon_carbon=eps),
                                      score_mode="combined")
    assert relaxed.repairs_applied >= base.repairs_applied
    _rematerialize_under_realized(relaxed, flavours, config)
    t = schedule_totals(relaxed, signals)
    b = schedule_totals(packing, signals)
    # Non-inferiority: realized carbon within the declared band (never a blanket free-for-all).
    assert t["carbon_kg"] <= b["carbon_kg"] * (1.0 + eps) + 1e-6
    result = run_no_harm_flexibility_pilot(dc_replace(_pilot_config(tmp_path / "p", max_pods=80),
                                                      epsilon_scarcity=0.02))
    assert result["certificate"]["epsilon_margins"]["scarcity"] == 0.02


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


# ------------------------------------------------------------------ B2: guarded relief move-set / LNS
def test_relief_move_set_off_is_bit_identical(tmp_path: Path) -> None:
    """The guarded relief move-set is flag-gated: the default (relief_move_set=False) must produce
    byte-identical placements to a config that never sets the field (no post-pass runs)."""
    from dataclasses import replace as dc_replace
    config = _pilot_config(tmp_path, max_pods=80)
    flavours, signals, pods, packing = _prep(config)
    base = repair_schedule_no_harm(method_key="no_harm_flex", baseline=packing, pods=pods,
                                   flavours=flavours, signals=signals, config=config, score_mode="combined")
    same = repair_schedule_no_harm(method_key="no_harm_flex", baseline=packing, pods=pods, flavours=flavours,
                                   signals=signals, config=dc_replace(config, relief_move_set=False),
                                   score_mode="combined")
    assert _placement_map(base) == _placement_map(same)


def test_relief_move_set_preserves_certificate(tmp_path: Path) -> None:
    """The move-set commits a coordinated SET only after scoring the COMPLETED schedule with the exact
    idle-once evaluator against the full certificate, so the envelope must still certify (carbon<=B,
    scarcity<=B, no pods dropped) on correct occupancy-based accounting."""
    from dataclasses import replace as dc_replace
    config = _pilot_config(tmp_path, max_pods=80)
    flavours, signals, pods, packing = _prep(config)
    lns = repair_schedule_no_harm(method_key="no_harm_flex", baseline=packing, pods=pods, flavours=flavours,
                                  signals=signals, config=dc_replace(config, relief_move_set=True),
                                  score_mode="stress")
    _rematerialize_under_realized(lns, flavours, config)
    t = schedule_totals(lns, signals)
    base = schedule_totals(packing, signals)
    assert t["carbon_kg"] <= base["carbon_kg"] + 1e-6
    assert t["scarcity_water"] <= base["scarcity_water"] + 1e-6
    assert t["placed_pods"] >= base["placed_pods"]


def test_relief_move_set_never_worsens_weighted_stress(tmp_path: Path) -> None:
    """The pass commits a set only on STRICT weighted-stress improvement, so enabling it can never
    raise total weighted grid-stress vs the single-move greedy under the same objective."""
    from dataclasses import replace as dc_replace
    config = _pilot_config(tmp_path, max_pods=80)
    flavours, signals, pods, packing = _prep(config)
    greedy = repair_schedule_no_harm(method_key="no_harm_flex", baseline=packing, pods=pods, flavours=flavours,
                                     signals=signals, config=config, score_mode="stress")
    lns = repair_schedule_no_harm(method_key="no_harm_flex", baseline=packing, pods=pods, flavours=flavours,
                                  signals=signals, config=dc_replace(config, relief_move_set=True),
                                  score_mode="stress")
    _rematerialize_under_realized(greedy, flavours, config)
    _rematerialize_under_realized(lns, flavours, config)
    assert schedule_totals(lns, signals)["weighted_stress_kwh"] <= \
        schedule_totals(greedy, signals)["weighted_stress_kwh"] + 1e-9


# ------------------------------------------------------------------ G1: basin-CF table override
def test_basin_cf_csv_off_is_bit_identical(tmp_path: Path) -> None:
    """basin_cf_csv is flag-gated: the default (never set) must produce byte-identical placements
    and totals to a config that explicitly sets the field to None (bundled AWARE basin table)."""
    from dataclasses import replace as dc_replace
    config = _pilot_config(tmp_path / "a", max_pods=30)
    _, signals, _, packing = _prep(config)
    same_cfg = dc_replace(_pilot_config(tmp_path / "b", max_pods=30), basin_cf_csv=None)
    _, signals2, _, packing2 = _prep(same_cfg)
    assert _placement_map(packing) == _placement_map(packing2)
    assert schedule_totals(packing, signals) == schedule_totals(packing2, signals2)


def test_basin_cf_csv_override_reaches_scarcity_characterization(tmp_path: Path) -> None:
    """Pointing basin_cf_csv at a table with inflated basin CFs must raise the scarcity-water
    characterization of the same (water-blind packing) schedule: witnesses that the override
    actually reaches the engine's CF resolution instead of the bundled table. Placements are
    unchanged (packing ignores water); only the characterization moves, and only the DIRECT
    (basin-charged) component scales, so the total rises by a bounded factor > 2x."""
    import csv as _csv
    from dataclasses import replace as dc_replace
    bundled = REPO_ROOT / "pkg" / "carbon-aware" / "data" / "water" / "aware20_basin_nonagri_factors.csv"
    with bundled.open() as fh:
        rows = list(_csv.DictReader(fh))
    for r in rows:
        for k in r:
            if k.endswith("_cf"):
                r[k] = str(float(r[k]) * 100.0)
    override = tmp_path / "basin_x100.csv"
    with override.open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    cfg_a = _pilot_config(tmp_path / "a", max_pods=30)
    _, sig_a, _, pack_a = _prep(cfg_a)
    tot_a = schedule_totals(pack_a, sig_a)
    cfg_b = dc_replace(_pilot_config(tmp_path / "b", max_pods=30), basin_cf_csv=override)
    _, sig_b, _, pack_b = _prep(cfg_b)
    tot_b = schedule_totals(pack_b, sig_b)

    assert _placement_map(pack_a) == _placement_map(pack_b)
    assert tot_b["scarcity_water"] > 2.0 * tot_a["scarcity_water"]
    assert tot_b["carbon_kg"] == tot_a["carbon_kg"]


def test_basin_cf_csv_missing_override_fails_loud(tmp_path: Path) -> None:
    """A typo'd basin_cf_csv must raise, not silently re-price basins at the country CF."""
    import pytest
    from dataclasses import replace as dc_replace
    cfg = dc_replace(_pilot_config(tmp_path, max_pods=30), basin_cf_csv=tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError, match="basin_cf_csv"):
        _prep(cfg)
