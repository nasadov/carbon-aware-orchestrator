from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_PYTHON_ROOT = REPO_ROOT / "pkg" / "carbon-aware" / "server-python"
CARBON_DATA_ROOT = REPO_ROOT / "pkg" / "carbon-aware" / "data" / "carbon"

if str(SERVER_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_PYTHON_ROOT))

from carbon_aware.utils import get_node_hardware_metadata


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_embodied_carbon_builder_generates_calibrated_outputs() -> None:
    subprocess.run(
        [sys.executable, "build_embodied_carbon_reference.py"],
        cwd=CARBON_DATA_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    for scenario in ("low", "base", "high"):
        output_path = CARBON_DATA_ROOT / f"embodied_carbon_reference_{scenario}.csv"
        assert output_path.exists()
        rows = _read_rows(output_path)
        assert {row["subcategory"] for row in rows} == {"IoT", "Smartphone", "Laptop", "Server"}
        for row in rows:
            assert float(row["embodied_carbon_kg"]) > 0.0
            assert float(row["uncalibrated_component_carbon_kg"]) > 0.0
            assert float(row["calibration_factor"]) > 0.0
            assert row["source_status"] in {"component_model", "calibrated_component_model"}


def test_get_node_hardware_metadata_uses_generated_embodied_carbon_reference() -> None:
    class Label:
        def __init__(self, key: str, value: str) -> None:
            self.key = key
            self.value = value

    class Node:
        def __init__(self) -> None:
            self.name = "node-0-DE-Laptop"
            self.labels = [Label("hardware.carbon/subcategory", "Laptop")]
            self.annotations = []

    embodied_carbon_g, lifetime, power = get_node_hardware_metadata(Node())

    assert embodied_carbon_g == 201.341 * 1000.0
    assert lifetime == 4.13
    assert power == {"idle": 10.0, "active": 40.0, "max": 150.0}
