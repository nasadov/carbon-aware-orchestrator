"""Water-pipeline consistency safeguards (the 'never-again' gate).

These tests pin the invariants whose violation produced the water-pipeline bugs the 2026-06 revamp
fixed, so a future edit that re-breaks one fails CI rather than silently shipping:

  * Option A CF resolution: on-site (direct) water is characterized at the DC WATERSHED CF, off-site
    (indirect/electricity) water at the GENERATION-COUNTRY CF. A regression that charges on-site water
    at the (much larger) country CF -- the original mischarge -- flips a direct_cf assertion.
  * Embodied water is OUT of the guarded scarcity axis: scarcity_characterized_water must be identical
    whether or not embodied is computed; embodied scarcity lives in its own field.
  * Cooling physics is CONTINUOUS and MONOTONE: the Gupta wet-tower WUE is monotone non-decreasing in
    wet-bulb (no parabola inversion) and direct WUE has no bang-bang jump across the activation band.
  * Any committed EWIF table that sits in a time-aligned signal dir is FULL-COVERAGE and IN-WINDOW
    (kills the stale-vintage / partial-coverage class that started the revamp).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_PYTHON_ROOT = REPO_ROOT / "pkg" / "carbon-aware" / "server-python"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for p in (SERVER_PYTHON_ROOT, SCRIPTS_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from carbon_aware.no_harm_flexibility import (  # noqa: E402
    PilotConfig, apply_pilot_scenario_to_flavours, load_flavours_for_pilot, load_pods,
    build_action_signals, build_greedy_schedule, repair_schedule_no_harm,
    _country_for_region, _exact_carbon_scarcity, _exact_node_totals,
)
from carbon_aware.footprints import compute_footprint_vector  # noqa: E402

WATER = REPO_ROOT / "pkg" / "carbon-aware" / "data" / "water"
BASIN_CSV = WATER / "aware20_basin_nonagri_factors.csv"
COUNTRY_CSV = WATER / "aware20_country_nonagri_factors.csv"
MONTH_COL = "jul_cf"  # heatwave-drought scenario month


def _pilot_config(output_dir: Path, max_pods: int = 40) -> PilotConfig:
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


def _cf_table(path: Path, key: str) -> dict:
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            k = (row.get(key) or "").strip().upper()
            try:
                out[k] = float(row[MONTH_COL])
            except (TypeError, ValueError):
                continue  # full AWARE country list carries a few blank monthly cells; skip them
    return out


def test_optionA_cf_resolution_direct_basin_indirect_country(tmp_path: Path) -> None:
    """Direct CF == DC watershed (basin) CF; indirect CF == generation-country CF. Guards against the
    original 'charge on-site water at the country CF' mischarge (e.g. IT-NO 1.83 basin vs ~41 country)."""
    config = _pilot_config(tmp_path)
    flavours = load_flavours_for_pilot(config)
    apply_pilot_scenario_to_flavours(flavours, config)

    basin = _cf_table(BASIN_CSV, "region")
    country = _cf_table(COUNTRY_CSV, "country_code")

    checked = 0
    for fl in flavours:
        region = (getattr(fl, "region", "") or "").upper()
        if region not in basin:
            continue
        cc = _country_for_region(region)
        assert fl.water_scarcity_direct_cf == pytest.approx(basin[region]), (
            f"{region}: direct CF {fl.water_scarcity_direct_cf} != basin {basin[region]} "
            f"(on-site water must use the watershed CF, not the country CF)")
        assert fl.water_scarcity_indirect_cf == pytest.approx(country[cc]), (
            f"{region}: indirect CF {fl.water_scarcity_indirect_cf} != country {cc} {country[cc]}")
        # The on-site mischarge would inflate direct CF toward the (larger) country value.
        if country.get(cc, 0) > basin[region]:
            assert fl.water_scarcity_direct_cf < country[cc]
        checked += 1
    assert checked > 0, "no region flavours exercised the Option A CF assignment"


def test_embodied_water_is_out_of_the_guarded_scarcity(tmp_path: Path) -> None:
    """scarcity_characterized_water (the guarded/ranked axis) must be identical with and without the
    embodied term; embodied scarcity is reported separately. (footprints.py #3 fix.)"""
    config = _pilot_config(tmp_path)
    flavours = load_flavours_for_pilot(config)
    apply_pilot_scenario_to_flavours(flavours, config)
    pods = load_pods(config.workloads_dir, max_pods=config.max_pods)
    assert pods and flavours

    saw_embodied = False
    for fl in flavours[:6]:
        for pod in pods[:6]:
            fp_op = compute_footprint_vector(fl, 0, pod, operational_only=True)
            fp_full = compute_footprint_vector(fl, 0, pod, operational_only=False)
            # Embodied is OUT of the guarded axis: identical operational scarcity either way.
            assert fp_op.scarcity_characterized_water == pytest.approx(
                fp_full.scarcity_characterized_water, abs=1e-9, rel=1e-9)
            # Embodied scarcity (when present) is captured in its own reported field.
            if fp_full.embodied_water_l > 0:
                saw_embodied = True
                assert fp_full.embodied_scarcity_water > 0
                assert fp_op.embodied_scarcity_water == pytest.approx(0.0, abs=1e-12)
    assert saw_embodied, "expected at least one flavour/pod with embodied water to exercise the split"


def test_cooling_is_continuous_and_monotone() -> None:
    """Gupta wet-tower WUE monotone non-decreasing in wet-bulb (no inversion); direct WUE continuous
    (no bang-bang jump) and monotone in heat for every region cooling profile."""
    import numpy as np
    import build_timealigned_signals as b

    Tw = np.linspace(0.0, 40.0, 800)
    g = b.gupta_wet_tower_wue(Tw)
    assert (np.diff(g) >= -1e-12).all(), "Gupta WUE must be monotone non-decreasing in wet-bulb"

    for reg in ("IT-NO", "SE", "DE", "ES", "FR", "PL"):
        prof = b.load_cooling_profile(reg)
        T = np.linspace(5.0, 42.0, 1600)
        Tw_co = T - 5.0  # realistic wet-bulb depression sweep across the activation band
        wue, pue, sigma = b.cooling_for_profile(prof, T, Tw_co)
        # Continuity: the old boolean gate produced a single large step; require small per-step deltas.
        assert np.abs(np.diff(wue)).max() < 0.05, f"{reg}: direct WUE has a bang-bang discontinuity"
        assert np.abs(np.diff(pue)).max() < 0.02, f"{reg}: PUE has a discontinuity at the gate"
        # Monotone non-decreasing in heat (engagement only increases water as it gets hotter).
        assert (np.diff(wue) >= -1e-9).all(), f"{reg}: direct WUE not monotone in heat"
        assert ((sigma >= 0.0) & (sigma <= 1.0)).all()


def _signal_dirs_with_ewif():
    data = REPO_ROOT / "pkg" / "carbon-aware" / "data"
    return sorted(d for d in data.glob("timealigned*") if (d / "ewif_region_slot.csv").exists())


@pytest.mark.parametrize("signals_dir", _signal_dirs_with_ewif(),
                         ids=lambda d: d.name)
def test_committed_ewif_is_full_coverage_and_in_window(signals_dir: Path) -> None:
    """Any EWIF table inside a time-aligned signal dir must cover every region x slot present in the
    grid table and carry timestamps in the SAME calendar year as the dir's CI forecasts (kills the
    stale-vintage / partial-coverage bug class). Dirs without an EWIF file are simply not parametrized."""
    ewif_path = signals_dir / "ewif_region_slot.csv"
    grid_path = signals_dir / "grid_residual_region_slot.csv"
    forecasts_path = signals_dir / "forecasts.json"

    def _rs(path):
        rows = list(csv.DictReader(path.open()))
        return {(r["region"], int(r["slot_index"])) for r in rows}, rows

    ewif_keys, ewif_rows = _rs(ewif_path)
    assert ewif_rows, f"{signals_dir.name}: empty EWIF table"
    # full coverage vs the grid table (same region x slot grid as the rest of the signals)
    if grid_path.exists():
        grid_keys, _ = _rs(grid_path)
        missing = grid_keys - ewif_keys
        assert not missing, f"{signals_dir.name}: EWIF missing {len(missing)} region/slot cells, e.g. {sorted(missing)[:3]}"
    # positive, finite l/kWh
    for r in ewif_rows:
        v = float(r["ewif_l_per_kwh"])
        assert v > 0 and v < 20.0, f"{signals_dir.name}: implausible EWIF {v} for {r['region']} slot {r['slot_index']}"
    # in-window: EWIF year matches the CI forecasts' year (not a stale vintage)
    if forecasts_path.exists():
        fc = json.loads(forecasts_path.read_text())
        any_zone = next(iter(fc.values()))
        ci_year = any_zone["forecast"][0]["datetime"][:4]
        ewif_years = {r["forecast_datetime_utc"][:4] for r in ewif_rows}
        assert ewif_years == {ci_year}, (
            f"{signals_dir.name}: EWIF years {ewif_years} != CI year {ci_year} (stale/out-of-window EWIF)")


def test_node_local_exact_guard_is_equivalent_to_full(tmp_path: Path) -> None:
    """The node-local exact commit-time guard must be EQUIVALENT to whole-schedule re-pricing:
    idle-once accounting never crosses node boundaries, so a move changes footprints only on its
    source and destination nodes. Assert (a) per-node totals sum to the global exact evaluator,
    and (b) the repair produces IDENTICAL placements under both guard modes."""
    from dataclasses import replace as dc_replace
    from collections import defaultdict

    config = _pilot_config(tmp_path, max_pods=80)
    flavours = load_flavours_for_pilot(config)
    apply_pilot_scenario_to_flavours(flavours, config)
    signals = build_action_signals(flavours, config)
    pods = load_pods(config.workloads_dir, max_pods=config.max_pods)
    packing, _, _ = build_greedy_schedule(method_key="packing", pods=pods, flavours=flavours, config=config)

    # (a) node-sum == global exact evaluator on the packing schedule
    glob = _exact_carbon_scarcity(packing.placements, flavours, config)
    assert glob is not None
    by_node = defaultdict(list)
    for pl in packing.placements:
        by_node[pl.candidate.flavour.id].append(pl)
    flavour_by_id = {f.id: f for f in flavours}
    sums = [0.0, 0.0]
    for fid, pls in by_node.items():
        tot = _exact_node_totals(pls, flavour_by_id[fid], config)
        assert tot is not None
        sums[0] += tot[0]
        sums[1] += tot[1]
    assert sums[0] == pytest.approx(glob[0], abs=1e-9)
    assert sums[1] == pytest.approx(glob[1], abs=1e-9)

    # (b) identical repair output under both guard modes
    out = {}
    for mode in ("full", "node_local"):
        cfg = dc_replace(config, exact_guard_mode=mode)
        fx = repair_schedule_no_harm(method_key="no_harm_flex", baseline=packing, pods=pods,
                                     flavours=flavours, signals=signals, config=cfg,
                                     score_mode="combined")
        out[mode] = {p.pod.id: (p.candidate.flavour.id, p.candidate.timeslot.id) for p in fx.placements}
    assert out["full"] == out["node_local"]
