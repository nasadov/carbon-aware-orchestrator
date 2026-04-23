from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_PYTHON_ROOT = REPO_ROOT / "pkg" / "carbon-aware" / "server-python"
WATER_DATA_ROOT = REPO_ROOT / "pkg" / "carbon-aware" / "data" / "water"

if str(SERVER_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_PYTHON_ROOT))

from carbon_aware.models import EnvironmentalFlavor
from carbon_aware.water_signals import attach_water_metadata


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_embodied_water_builder_generates_weighted_scarcity_outputs() -> None:
    subprocess.run(
        [sys.executable, "build_embodied_water_reference.py"],
        cwd=WATER_DATA_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    for scenario in ("low", "base", "high"):
        output_path = WATER_DATA_ROOT / f"embodied_water_reference_{scenario}.csv"
        assert output_path.exists()
        rows = _read_rows(output_path)
        assert {row["subcategory"] for row in rows} == {"IoT", "Smartphone", "Laptop", "Server"}
        for row in rows:
            assert float(row["embodied_water_l"]) > 0.0
            assert float(row["embodied_scarcity_cf"]) > 0.0
            assert row["source_status"] == "bounded_component_model"


def test_attach_water_metadata_prefers_precomputed_embodied_scarcity_cf() -> None:
    config = {
        "defaults": {
            "pue": 1.2,
            "embodied_water": 0.0,
            "wue": 0.0,
            "ewif": 0.0,
            "direct_scarcity_cf": 1.0,
            "indirect_scarcity_cf": 1.0,
            "embodied_scarcity_cf": 1.0,
            "criticality": 1.0,
        },
        "region_to_country": {"DE": "DE"},
        "by_region": {},
        "by_country": {
            "DE": {"embodied_scarcity_cf": 2.09},
            "CN": {"embodied_scarcity_cf": 6.29},
        },
        "by_hardware_subcategory": {
            "Server": {
                "embodied_water": 1000.0,
                "manufacturing_country": "CN",
                "embodied_scarcity_cf": 4.25,
            }
        },
    }

    flavor = EnvironmentalFlavor(
        id="node-0",
        embodiedCarbon=0.0,
        lifetime=1.0,
        totalCpu=16.0,
        totalRam=16.0,
        totalStorage=1.0,
        forecast={0: 100.0},
    )

    attach_water_metadata(flavor, region="DE", hardware_subcategory="Server", config=config, slot_count=1)

    assert flavor.embodiedWater == 1000.0
    assert flavor.manufacturing_country == "CN"
    assert flavor.water_scarcity_embodied_cf == 4.25
