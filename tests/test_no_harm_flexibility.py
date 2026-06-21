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
    load_flavours_for_pilot,
    run_no_harm_flexibility_pilot,
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
    assert any(signal.drought_guardrail for signal in signals.values())

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
