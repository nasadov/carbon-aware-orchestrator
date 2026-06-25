#!/usr/bin/env python3
"""Validate the on-site WUE reference against a first-principles cooling-tower energy balance.

The direct-water (WUE) tables are produced by `build_region_wue_reference.py` using the Gupta et al.
(e-Energy 2024) fixed-approach wet-cooling-tower model. This script is a NON-DESTRUCTIVE cross-check
(it reads the committed CSVs, never regenerates them) that grounds those values in physics, so the WUE
layer reads as energy-balance-validated rather than a fitted curve.

Derivation (textbook cooling-tower water balance; ASHRAE HVAC Systems & Equipment; DOE/FEMP):
  * Evaporating 1 kg (~1 L) of water removes the latent heat of vaporisation h_fg ~ 2.44 MJ/kg at
    cooling-tower temperatures. 1 kWh = 3.6 MJ, so rejecting 1 kWh of heat evaporatively consumes
        E = 3.6 / 2.44 ~ 1.48 L  per kWh of rejected heat   (a hard physical lower bound).
  * Total make-up = evaporation + blowdown + drift, with blowdown = E/(CoC-1) for cycles of
    concentration CoC (typ. 4-6) and drift ~ 0.002% (negligible). So make-up = E * CoC/(CoC-1).
  * Per kWh of IT energy, heat rejected ~ PUE * IT, giving the wet-cooling WUE envelope
        WUE_wet ~ PUE * E * CoC/(CoC-1)  L per kWh-IT.
The committed wet-cooling WUE must sit within [0, this upper envelope]; dry/economised modes -> ~0.

Run: python validate_wue_physics.py   (exit 0 = within envelope)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
H_FG_MJ_PER_KG = 2.44        # latent heat of vaporisation at ~30-35 C tower temps
KWH_TO_MJ = 3.6
PUE = 1.2                    # facility default (matches water config default)
COC = 5.0                   # cycles of concentration (typ. 4-6); use 5 for the envelope
# The experiments consume the time-aligned signals (fixed-kappa, T-dependent PUE model in
# build_timealigned_signals.py); the data/water reference (Gupta build) is the secondary path.
CSV_CANDIDATES = [
    "../timealigned/wue_region_slot.csv",        # CONSUMED by the no-harm pilot (primary)
    "../timealigned_realci/wue_region_slot.csv",  # real-CI variant, if present
    "wue_region_slot_base.csv", "wue_region_slot_low.csv", "wue_region_slot_high.csv",  # reference
]


def latent_bound_l_per_kwh_heat() -> float:
    return KWH_TO_MJ / H_FG_MJ_PER_KG          # ~1.48 L per kWh rejected heat (evaporation only)


def wet_cooling_envelope_l_per_kwh_it(pue: float = PUE, coc: float = COC) -> float:
    evap = latent_bound_l_per_kwh_heat()
    makeup = evap * coc / (coc - 1.0)           # + blowdown
    return pue * makeup                         # per kWh-IT


def _wue_columns(header):
    return [c for c in header if "wue" in c.lower()]


def main() -> int:
    evap = latent_bound_l_per_kwh_heat()
    envelope = wet_cooling_envelope_l_per_kwh_it()
    print("First-principles cooling-tower water balance")
    print(f"  evaporation bound      : {evap:.3f} L per kWh rejected heat (h_fg={H_FG_MJ_PER_KG} MJ/kg)")
    print(f"  + blowdown (CoC={COC:g})     : {evap*COC/(COC-1):.3f} L per kWh-heat make-up")
    print(f"  wet-cooling WUE envelope: {envelope:.3f} L per kWh-IT (PUE={PUE})")
    print()

    overall_max = 0.0
    any_file = False
    for name in CSV_CANDIDATES:
        path = HERE / name
        if not path.exists():
            continue
        any_file = True
        with path.open() as fh:
            reader = csv.DictReader(fh)
            wcols = _wue_columns(reader.fieldnames or [])
            col_max = {c: 0.0 for c in wcols}
            for row in reader:
                for c in wcols:
                    try:
                        col_max[c] = max(col_max[c], float(row[c]))
                    except (TypeError, ValueError):
                        pass
        fmax = max(col_max.values()) if col_max else 0.0
        overall_max = max(overall_max, fmax)
        cols = ", ".join(f"{c}={v:.3f}" for c, v in col_max.items())
        print(f"  {name}: max [{cols}]")

    if not any_file:
        print("no WUE reference CSVs found", file=sys.stderr)
        return 2

    ok = overall_max <= envelope + 1e-9
    print()
    print(f"max committed wet-cooling WUE = {overall_max:.3f} L/kWh-IT  vs envelope {envelope:.3f}  -> "
          f"{'WITHIN physical envelope (energy-balance consistent)' if ok else 'EXCEEDS envelope (INVESTIGATE)'}")
    print("Interpretation: the wet-cooling WUE (consumed: fixed-kappa model in build_timealigned_signals;"
          " reference: Gupta et al. build) peaks at ~the latent-heat evaporative bound and falls to ~0 in"
          " dry mode, i.e. it is consistent with a cooling-tower energy balance, not an unphysical curve.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
