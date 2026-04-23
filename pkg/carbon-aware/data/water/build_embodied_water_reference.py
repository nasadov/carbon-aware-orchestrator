"""
Build literature-backed embodied-water reference tables for hardware subcategories.

This helper keeps the runtime simple:
- embodied-water component assumptions live in repo-owned CSVs
- AWARE factors are resolved offline
- the generated outputs expose one raw embodied-water value and one
  precomputed embodied scarcity factor per hardware subcategory
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


THIS_DIR = Path(__file__).resolve().parent
COMPONENTS_CSV = THIS_DIR / "embodied_water_component_inventory.csv"
TEMPLATES_CSV = THIS_DIR / "embodied_water_subcategory_templates.csv"
AWARE_CSV = THIS_DIR / "aware20_country_nonagri_factors.csv"

SCENARIOS = ("low", "base", "high")


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_aware_factors(path: Path) -> Dict[str, float]:
    factors: Dict[str, float] = {}
    for row in _read_csv_rows(path):
        country_code = (row.get("country_code") or "").strip().upper()
        annual_cf = row.get("annual_cf")
        if not country_code or annual_cf in (None, ""):
            continue
        factors[country_code] = float(annual_cf)
    return factors


def _read_components(path: Path) -> Dict[str, Dict[str, str]]:
    components: Dict[str, Dict[str, str]] = {}
    for row in _read_csv_rows(path):
        component_id = (row.get("component_id") or "").strip()
        if component_id:
            components[component_id] = row
    return components


def _iter_templates(path: Path) -> Iterable[Dict[str, str]]:
    yield from _read_csv_rows(path)


def _unique_join(values: Iterable[str]) -> str:
    ordered = sorted({value for value in values if value})
    return ";".join(ordered)


def _build_rows_for_scenario(
    scenario: str,
    aware_factors: Dict[str, float],
    components: Dict[str, Dict[str, str]],
    template_rows: Iterable[Dict[str, str]],
) -> List[Dict[str, str]]:
    grouped_templates: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in template_rows:
        subcategory = (row.get("subcategory") or "").strip()
        if subcategory:
            grouped_templates[subcategory].append(row)

    output_rows: List[Dict[str, str]] = []
    raw_field = f"raw_water_l_{scenario}"

    for subcategory in sorted(grouped_templates):
        total_raw = 0.0
        total_weighted_scarcity = 0.0
        raw_by_country: Dict[str, float] = defaultdict(float)
        source_ids: List[str] = []

        for template in grouped_templates[subcategory]:
            component_id = template["component_id"].strip()
            quantity = float(template["quantity"])
            component = components[component_id]

            component_raw = float(component[raw_field]) * quantity
            manufacturing_country = (component.get("manufacturing_country") or "").strip().upper()
            embodied_cf = aware_factors.get(manufacturing_country, 1.0)

            total_raw += component_raw
            total_weighted_scarcity += component_raw * embodied_cf
            raw_by_country[manufacturing_country] += component_raw
            source_ids.extend(
                source_id.strip()
                for source_id in (component.get("source_id") or "").split(";")
                if source_id.strip()
            )

        dominant_country = max(raw_by_country.items(), key=lambda item: item[1])[0] if raw_by_country else ""
        embodied_scarcity_cf = (total_weighted_scarcity / total_raw) if total_raw > 0.0 else 1.0

        output_rows.append(
            {
                "subcategory": subcategory,
                "embodied_water_l": f"{total_raw:.6f}",
                "embodied_scarcity_cf": f"{embodied_scarcity_cf:.6f}",
                "manufacturing_country": dominant_country,
                "source_status": "bounded_component_model",
                "source_ids": _unique_join(source_ids),
                "source_note": (
                    "Component-derived embodied-water inventory aggregated offline; "
                    "embodied scarcity factor is a raw-water-weighted AWARE average "
                    "over component manufacturing countries."
                ),
            }
        )

    return output_rows


def _write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    fieldnames = [
        "subcategory",
        "embodied_water_l",
        "embodied_scarcity_cf",
        "manufacturing_country",
        "source_status",
        "source_ids",
        "source_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    aware_factors = _read_aware_factors(AWARE_CSV)
    components = _read_components(COMPONENTS_CSV)
    template_rows = list(_iter_templates(TEMPLATES_CSV))

    for scenario in SCENARIOS:
        rows = _build_rows_for_scenario(scenario, aware_factors, components, template_rows)
        _write_rows(THIS_DIR / f"embodied_water_reference_{scenario}.csv", rows)
        if scenario == "base":
            _write_rows(THIS_DIR / "embodied_water_reference.csv", rows)


if __name__ == "__main__":
    main()
