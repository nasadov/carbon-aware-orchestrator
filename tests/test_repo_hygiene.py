from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_no_machine_specific_root_paths_in_active_code() -> None:
    active_roots = [
        REPO_ROOT / "analysis",
        REPO_ROOT / "scripts",
        REPO_ROOT / "tests",
        REPO_ROOT / "pkg" / "carbon-aware" / "server-python",
    ]
    offenders: list[str] = []
    for root in active_roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".sh"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            machine_root = "/root/" + "carbon-aware-orchestrator"
            if machine_root in text:
                offenders.append(path.relative_to(REPO_ROOT).as_posix())
    assert offenders == []


def test_water_sweep_dry_run_writes_provenance() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/water_sweep.py",
            "--dry-run",
            "--quiet",
            "--pod-counts",
            "34",
            "--seeds",
            "44",
            "--timeslots",
            "12",
            "--methods",
            "heuristic-carbon",
            "--run-name",
            "pytest_provenance",
        ],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = proc.stdout[proc.stdout.index("{") :]
    metadata = json.loads(payload)

    assert metadata["script"] == "scripts/water_sweep.py"
    assert metadata["config_sha256"]
    assert metadata["forecasts_sha256"]
    assert metadata["git"]["branch"]
    assert metadata["water_references"]["directory_sha256"]
    assert metadata["planned_methods"] == ["heuristic_carbon"]
