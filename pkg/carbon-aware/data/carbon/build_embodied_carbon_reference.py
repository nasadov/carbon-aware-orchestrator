"""
Build calibrated embodied-carbon reference tables for hardware subcategories.

The implementation mirrors the embodied-water data flow:
- component assumptions live in repo-owned CSVs
- subcategory templates aggregate components offline
- optional subcategory anchors calibrate the component totals
- generated outputs expose one embodied-carbon value per subcategory
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


THIS_DIR = Path(__file__).resolve().parent
COMPONENTS_CSV = THIS_DIR / "embodied_carbon_component_inventory.csv"
TEMPLATES_CSV = THIS_DIR / "embodied_carbon_subcategory_templates.csv"
ANCHORS_CSV = THIS_DIR / "embodied_carbon_anchors.csv"

SCENARIOS = ("low", "base", "high")


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_components(path: Path) -> Dict[str, Dict[str, str]]:
    components: Dict[str, Dict[str, str]] = {}
    for row in _read_csv_rows(path):
        component_id = (row.get("component_id") or "").strip()
        if component_id:
            components[component_id] = row
    return components


def _read_anchors(path: Path) -> Dict[str, Dict[str, str]]:
    anchors: Dict[str, Dict[str, str]] = {}
    for row in _read_csv_rows(path):
        subcategory = (row.get("subcategory") or "").strip()
        if subcategory:
            anchors[subcategory] = row
    return anchors


def _iter_templates(path: Path) -> Iterable[Dict[str, str]]:
    yield from _read_csv_rows(path)


def _as_optional_float(value: Optional[str]) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def _unique_join(values: Iterable[str]) -> str:
    ordered = sorted({value for value in values if value})
    return ";".join(ordered)


def _component_total(
    subcategory_rows: Iterable[Dict[str, str]],
    components: Dict[str, Dict[str, str]],
    scenario: str,
) -> float:
    field = f"raw_carbon_kg_{scenario}"
    total = 0.0
    for template in subcategory_rows:
        component = components[template["component_id"].strip()]
        quantity = float(template["quantity"])
        total += float(component[field]) * quantity
    return total


def _build_rows(
    scenario: str,
    components: Dict[str, Dict[str, str]],
    anchors: Dict[str, Dict[str, str]],
    template_rows: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    grouped_templates: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in template_rows:
        subcategory = (row.get("subcategory") or "").strip()
        if subcategory:
            grouped_templates[subcategory].append(row)

    output_rows: List[Dict[str, str]] = []
    raw_field = f"raw_carbon_kg_{scenario}"

    for subcategory in sorted(grouped_templates):
        templates = grouped_templates[subcategory]
        base_component_total = _component_total(templates, components, "base")
        scenario_component_total = _component_total(templates, components, scenario)

        anchor = anchors.get(subcategory, {})
        anchor_base = _as_optional_float(anchor.get("anchor_kgco2e_base"))
        anchor_low = _as_optional_float(anchor.get("anchor_kgco2e_low"))
        anchor_high = _as_optional_float(anchor.get("anchor_kgco2e_high"))
        calibration_factor = (anchor_base / base_component_total) if anchor_base and base_component_total > 0.0 else 1.0
        final_total = scenario_component_total * calibration_factor

        source_ids: List[str] = []
        for template in templates:
            component = components[template["component_id"].strip()]
            source_ids.extend(
                source_id.strip()
                for source_id in (component.get("source_id") or "").split(";")
                if source_id.strip()
            )
        source_ids.extend(
            source_id.strip()
            for source_id in (anchor.get("source_ids") or "").split(";")
            if source_id.strip()
        )

        source_status = "calibrated_component_model" if anchor_base is not None else "component_model"
        within_anchor_band = ""
        if anchor_low is not None and anchor_high is not None:
            within_anchor_band = "true" if anchor_low <= final_total <= anchor_high else "false"

        output_rows.append(
            {
                "subcategory": subcategory,
                "embodied_carbon_kg": f"{final_total:.6f}",
                "uncalibrated_component_carbon_kg": f"{scenario_component_total:.6f}",
                "calibration_factor": f"{calibration_factor:.6f}",
                "anchor_kgco2e_base": f"{anchor_base:.6f}" if anchor_base is not None else "",
                "anchor_kgco2e_low": f"{anchor_low:.6f}" if anchor_low is not None else "",
                "anchor_kgco2e_high": f"{anchor_high:.6f}" if anchor_high is not None else "",
                "within_anchor_band": within_anchor_band,
                "source_status": source_status,
                "anchor_source": (anchor.get("anchor_source") or "").strip(),
                "anchor_method": (anchor.get("anchor_method") or "").strip(),
                "source_ids": _unique_join(source_ids),
                "source_note": (
                    "Component-derived embodied-carbon inventory aggregated offline. "
                    "Per-subcategory base totals are calibrated to the configured "
                    "anchor when available; low/high scenarios reuse the same "
                    "calibration factor to preserve uncertainty structure."
                ),
            }
        )

    return output_rows


def _write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    fieldnames = [
        "subcategory",
        "embodied_carbon_kg",
        "uncalibrated_component_carbon_kg",
        "calibration_factor",
        "anchor_kgco2e_base",
        "anchor_kgco2e_low",
        "anchor_kgco2e_high",
        "within_anchor_band",
        "source_status",
        "anchor_source",
        "anchor_method",
        "source_ids",
        "source_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    components = _read_components(COMPONENTS_CSV)
    anchors = _read_anchors(ANCHORS_CSV)
    template_rows = list(_iter_templates(TEMPLATES_CSV))

    for scenario in SCENARIOS:
        rows = _build_rows(scenario, components, anchors, template_rows)
        _write_rows(THIS_DIR / f"embodied_carbon_reference_{scenario}.csv", rows)
        if scenario == "base":
            _write_rows(THIS_DIR / "embodied_carbon_reference.csv", rows)


if __name__ == "__main__":
    main()
