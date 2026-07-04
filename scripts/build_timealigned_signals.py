#!/usr/bin/env python3
"""Build time-aligned region x slot signal tables for the no-harm pilot.

Replaces the previously mismatched inputs (grid residual from 2019-2020 OPSD,
WUE/EWIF from a 2025-02 winter day, carbon from a third window) with ONE coherent
real calendar window: residual load (OPSD), wet-bulb -> direct WUE (Open-Meteo),
and a residual-scaled carbon intensity, all for the same hours of a documented
2018 EU heatwave-drought window. Scarcity stays AWARE monthly (already July).

Outputs (pkg/carbon-aware/data/timealigned/):
  grid_residual_region_slot.csv  -> residual_load_mw per (region, slot)
  wue_region_slot.csv            -> wet_bulb_c, direct_wue_l_per_kwh per (region, slot)
  forecasts.json                 -> {region: {forecast: [{datetime, carbonIntensity}]}}

The pilot/matrix consume these via --grid-signal-csv / --wue-csv / --forecasts-file.
"""
from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "pkg" / "carbon-aware" / "data" / "timealigned"
OPSD_LOCAL = "/tmp/opsd_ts_60min.csv"

# Representative city + 2018 country-average grid carbon intensity (g/kWh) per region.
SITES = {
    "DE": dict(lat=50.11, lon=8.68, ci_base=380.0,
               load="DE_load_actual_entsoe_transparency",
               wind=["DE_wind_generation_actual"], solar="DE_solar_generation_actual"),
    "FR": dict(lat=48.85, lon=2.35, ci_base=55.0,
               load="FR_load_actual_entsoe_transparency",
               wind=["FR_wind_onshore_generation_actual"], solar="FR_solar_generation_actual"),
    "ES": dict(lat=40.42, lon=-3.70, ci_base=190.0,
               load="ES_load_actual_entsoe_transparency",
               wind=["ES_wind_onshore_generation_actual"], solar="ES_solar_generation_actual"),
    "IT-NO": dict(lat=45.46, lon=9.19, ci_base=330.0,
                  load="IT_NORD_load_actual_entsoe_transparency",
                  wind=["IT_NORD_wind_onshore_generation_actual"], solar="IT_NORD_solar_generation_actual"),
    # Strong-scenario additions (Aug-2022 drought window). SE (Luleå) is the clean+abundant
    # sink: SE has NO national solar column in OPSD, so solar is None -> treated as 0 MW.
    "SE": dict(lat=65.5841, lon=22.1547, ci_base=30.0,
               load="SE_load_actual_entsoe_transparency",
               wind=["SE_wind_onshore_generation_actual"], solar=None),
    "PL": dict(lat=52.2297, lon=21.0122, ci_base=680.0,
               load="PL_load_actual_entsoe_transparency",
               wind=["PL_wind_onshore_generation_actual"], solar="PL_solar_generation_actual"),
}

# ---------------------------------------------------------------------------------------------------
# US testbed (Tier-1-E generalization; --grid eia930). Three EIA balancing authorities chosen so the
# carbon<->water geography is OPPOSITE the EU's. In the EU, the cleanest grid (FR nuclear) is also the
# most water-SCARCE, so "shift to clean" is water-neutral/positive. In the US:
#   * ERCO (ERCOT, Dallas TX) -- gas/coal-heavy -> HIGH CI; Trinity basin AWARE CF is low (~2.2 Jul).
#   * CISO (CAISO, Santa Clara CA) -- mid CI (gas+solar+hydro); Santa Clara basin is EXTREMELY scarce
#     (AWARE Jul CF ~100, the AWARE ceiling). Clean-ish hour != low-water.
#   * BPAT (Bonneville/Pacific-NW, The Dalles OR) -- ~71% reservoir hydro -> LOWEST CI (~98 g/kWh),
#     BUT reservoir-hydro is water-CONSUMPTIVE (evaporation; Macknick reservoir EWIF), so the cleanest
#     grid carries the HIGHEST off-site water factor. This is the channel that flips the spatial water
#     co-benefit's SIGN: a carbon-greedy shift toward clean BPAT RAISES off-site scarcity-water, which
#     the no-harm water guard W(S)<=W(B) correctly blocks while the envelope still certifies.
# These are NOT in the EU SITES dict; the build keys US BAs by their EIA respondent code (== region).
US_SITES = {
    "ERCO": dict(lat=32.7767, lon=-96.7970, ci_base=450.0, eia_respondent="ERCO"),
    "CISO": dict(lat=37.3541, lon=-121.9552, ci_base=280.0, eia_respondent="CISO"),
    "BPAT": dict(lat=45.5946, lon=-121.1787, ci_base=100.0, eia_respondent="BPAT"),
}


def wet_bulb(T, RH):
    RH = np.clip(RH, 1, 100)
    return (T * np.arctan(0.151977 * np.sqrt(RH + 8.313659)) + np.arctan(T + RH)
            - np.arctan(RH - 1.676331) + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH) - 4.686035)


# Step 4: climate-driven cooling model (replaces the earlier hand-assigned per-site
# kappa). Cooling architecture is NOT assigned by hand: every region runs the same
# hybrid-economized cooling, and whether evaporative assist actually fires is decided
# per slot from the real weather (dry-bulb / wet-bulb) against region-specific activation
# thresholds and operator-evidence parameters in region_cooling_profiles.csv
# (Microsoft/Google/DOE/NREL disclosures) -- the SAME model the test-fleet base table
# built by data/water/build_region_wue_reference.py consumes. Dry mode is NOT zero water:
# it carries real makeup/blowdown, so the carbon<->water trade-off is physical, not an
# artifact of which region was labelled "dry". PUE is coupled to the SAME activation: in
# dry/economizer mode facility overhead rises steeply with ambient heat (no evaporative
# help); when evaporative assist is active the rise is mild but water is spent. So in a
# heatwave a region trades carbon (dry, high PUE) against water (evaporative, low PUE) --
# the genuine engineering tension the envelope must navigate.
COOLING_PROFILES_CSV = REPO / "pkg" / "carbon-aware" / "data" / "water" / "region_cooling_profiles.csv"
COOLING_SCENARIO = "base"  # hybrid_economized; low/high are the sensitivity bounds

PUE_BASE_ECON = 1.12   # hybrid/economized design-point PUE (hyperscale)
PUE_T_REF = 20.0       # design-point dry-bulb (C) at which PUE = base
PUE_SLOPE_DRY = 0.018  # PUE rise per C above T_ref running dry/economizer (steep; no evap help)
PUE_SLOPE_EVAP = 0.006 # PUE rise per C when evaporative assist is active (mild)
PUE_CAP = 1.60


def load_cooling_profile(reg):
    """Base-scenario cooling-profile params for a region from the operator-evidence CSV.
    Uses csv.DictReader (same as build_region_wue_reference.py): the numeric/threshold
    fields all precede the comma-containing prose columns, so this is robust to the
    unquoted commas in source_basis/notes that break pandas' C parser."""
    opt = lambda v: None if v in (None, "", "None") else float(v)  # noqa: E731
    with COOLING_PROFILES_CSV.open(encoding="utf-8", newline="") as handle:
        for r in csv.DictReader(handle):
            if (str(r.get("region", "")).strip().upper() == reg.upper()
                    and str(r.get("scenario", "")).strip().lower() == COOLING_SCENARIO):
                return dict(
                    dry_mode_wue=float(r["dry_mode_wue_l_per_kwh"] or 0.0),
                    evap_multiplier=float(r["evap_multiplier"] or 0.0),
                    act_T=opt(r.get("evap_activation_dry_bulb_c")),
                    act_Tw=opt(r.get("evap_activation_wet_bulb_c")),
                    activation_rule=str(r.get("activation_rule", "threshold")).strip().lower(),
                    activation_logic=str(r.get("activation_logic", "any")).strip().lower(),
                    profile_class=str(r.get("profile_class", "")).strip(),
                )
    raise ValueError(f"no {COOLING_SCENARIO} cooling profile for region {reg} in {COOLING_PROFILES_CSV}")


# Gupta et al. (e-Energy 2024) Eq.(2) fixed-approach wet-tower WUE fit (in wet-bulb degF). It is a
# downward parabola peaking at wbf* = -b/2a ~= 81.6 F (Tw ~= 27.6 C); ABOVE the peak the raw fit turns
# DOWNWARD, which is unphysical (more makeup water is needed as the wet-bulb rises, not less). We
# therefore clamp the fit argument to [30 F validity floor, peak], so WUE is MONOTONE NON-DECREASING
# in wet-bulb and plateaus at the empirical maximum instead of inverting in hot windows (Aug-2022).
_GUPTA_A, _GUPTA_B, _GUPTA_C = -0.0001896, 0.03095, 0.4442
_GUPTA_PEAK_F = -_GUPTA_B / (2.0 * _GUPTA_A)  # ~81.64 F parabola vertex (max of the fit)
# Width (C) of the smooth evaporative-engagement transition band around each activation threshold. The
# real plant stages evaporative assist over a few degrees, not as an instantaneous switch; this makes
# WUE and PUE CONTINUOUS in weather (removes the bang-bang discontinuity) without changing the endpoints.
EVAP_TRANSITION_C = 3.0


def gupta_wet_tower_wue(Tw):
    """Gupta et al. (e-Energy 2024) Eq.(2) fixed-approach wet-tower WUE (L/kWh), monotone non-decreasing
    in wet-bulb: argument clamped to [30 F validity floor, parabola peak] so it never inverts above the
    fit vertex (~27.6 C wet-bulb)."""
    wbf = np.asarray(Tw, float) * 9.0 / 5.0 + 32.0
    m = np.clip(wbf, 30.0, _GUPTA_PEAK_F)
    return np.clip(_GUPTA_A * m ** 2 + _GUPTA_B * m + _GUPTA_C, 0.0, None)


def _smoothstep(x):
    """C1-continuous 0->1 ramp (Hermite smoothstep) on x in [0,1]; flat outside."""
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def evap_engagement(prof, T, Tw):
    """CONTINUOUS evaporative-assist engagement sigma in [0,1] (replaces the old bang-bang boolean).
    Each active weather threshold contributes a smoothstep ramp over an EVAP_TRANSITION_C-wide window
    centred on the threshold; the region's any/all logic combines them as fuzzy OR (max) / AND (min).
    Driving engagement off the SAME thresholds keeps the documented activation envelope but makes the
    transition smooth, so a marginally hotter or more-humid hour can never make WUE jump or invert."""
    T = np.asarray(T, float); Tw = np.asarray(Tw, float)
    if prof["activation_rule"] == "always":
        return np.ones(T.shape, dtype=float)
    if prof["activation_rule"] == "never":
        return np.zeros(T.shape, dtype=float)
    w = EVAP_TRANSITION_C
    ramps = []
    if prof["act_T"] is not None:
        ramps.append(_smoothstep((T - (prof["act_T"] - w / 2.0)) / w))
    if prof["act_Tw"] is not None:
        ramps.append(_smoothstep((Tw - (prof["act_Tw"] - w / 2.0)) / w))
    if not ramps:
        return np.zeros(T.shape, dtype=float)
    stacked = np.vstack(ramps)
    return stacked.min(axis=0) if prof["activation_logic"] == "all" else stacked.max(axis=0)


def cooling_for_profile(prof, T, Tw):
    """Climate-driven hybrid-economized cooling -> (direct_wue L/kWh, pue, evap_engagement sigma)
    per slot. CONTINUOUS in weather: evaporative assist engages smoothly (sigma in [0,1]), so both the
    water term and the PUE slope blend rather than switch. WUE is monotone non-decreasing in heat (via
    sigma) and humidity (via the clamped Gupta term); PUE keeps the physical dry(high)<->evap(low) slope
    tension but without a step. Matches build_region_wue_reference._compute_direct_wue for the WUE term."""
    sigma = evap_engagement(prof, T, Tw)
    wet_tower = gupta_wet_tower_wue(Tw)
    direct_wue = np.clip(prof["dry_mode_wue"] + sigma * wet_tower * prof["evap_multiplier"], 0.0, None)
    slope = (1.0 - sigma) * PUE_SLOPE_DRY + sigma * PUE_SLOPE_EVAP
    pue = np.clip(PUE_BASE_ECON + slope * np.clip(np.asarray(T, float) - PUE_T_REF, 0.0, None),
                  PUE_BASE_ECON, PUE_CAP)
    return direct_wue, pue, sigma


def weather(lat, lon, start, n):
    end = start + pd.Timedelta(hours=n - 1)
    q = urllib.parse.urlencode(dict(latitude=lat, longitude=lon, start_date=start.date().isoformat(),
                                    end_date=end.date().isoformat(),
                                    hourly="temperature_2m,relative_humidity_2m", timezone="UTC"))
    with urllib.request.urlopen("https://archive-api.open-meteo.com/v1/archive?" + q, timeout=120) as r:
        h = json.load(r)["hourly"]
    return pd.DataFrame({"ts": pd.to_datetime(h["time"], utc=True),
                         "T": np.array(h["temperature_2m"], float),
                         "RH": np.array(h["relative_humidity_2m"], float)}).set_index("ts")


# Lifecycle CO2-eq emission factors (gCO2eq/kWh) keyed by Energy-Charts production type.
# Medians from IPCC AR5 WG3 Annex III (Schlomer et al. 2014); lignite/oil/waste from the
# IPCC/UNECE ranges. Used to build a real generation-mix carbon intensity
# CI = sum(gen_type * EF_type) / sum(gen_type) -- the measured-mix alternative to the
# residual-load-scaled proxy.
LIFECYCLE_EF = {
    "Nuclear": 12.0, "Hydro Run-of-River": 24.0, "Hydro water reservoir": 24.0,
    "Hydro pumped storage": 24.0, "Biomass": 230.0, "Geothermal": 38.0,
    "Wind onshore": 11.0, "Wind offshore": 12.0, "Solar": 48.0,
    "Fossil gas": 490.0, "Fossil hard coal": 820.0, "Fossil oil": 650.0,
    "Fossil brown coal / lignite": 1054.0, "Fossil coal-derived gas": 820.0,
    "Waste": 580.0,
    # "Others"/"Other" is unclassified (in IT, predominantly thermal/fossil): use the
    # Electricity-Maps "unknown" default of ~700 gCO2eq/kWh rather than drop it (dropping
    # would implicitly assume it equals the renewables-inclusive known mix). Tiny (<1%)
    # in DE/ES/FR; ~22% in IT, where it is fossil-dominated.
    "Others": 700.0, "Other": 700.0, "Other renewables": 30.0,
}
# Per-fuel OPERATIONAL water-CONSUMPTION factors (L/kWh) keyed by Energy-Charts production type,
# from Macknick et al. 2012 (NREL) medians [macknick2012]. Used to build a real off-site
# (generation) water intensity EWIF = sum(gen_type * WF_type)/sum(gen_type) from the SAME measured
# fuel mix as the real CI -- removing the static-EWIF asymmetry (W1). Consumption (not withdrawal),
# the scarcity-relevant term. NOTE (flagged for review): reservoir-hydro evaporation is the dominant
# uncertainty -- estimates span ~0 (run-of-river) to ~17 L/kWh (gross reservoir evaporation, often
# excluded from operational accounting); we use a CONSERVATIVE operational value below.
# SENSITIVITY-TESTED (--hydro-reservoir-lkwh, 0.5/1.0/4.0 L/kWh, K=24 sweep re-run per factor:
# experiments/generalize_us/us2018_window_sweep_hydro*): every certification statistic is
# unchanged at every factor. Geothermal also high/variable.
WATER_FACTOR_L_PER_KWH = {
    "Nuclear": 2.5, "Fossil hard coal": 2.6, "Fossil brown coal / lignite": 2.6,
    "Fossil coal-derived gas": 2.6, "Fossil gas": 0.75, "Fossil oil": 1.1,
    "Biomass": 2.1, "Waste": 2.1, "Geothermal": 4.0,
    "Wind onshore": 0.004, "Wind offshore": 0.0, "Solar": 0.1,
    "Hydro Run-of-River": 0.0, "Hydro pumped storage": 0.0,
    "Hydro water reservoir": 2.0,   # CONSERVATIVE (gross evaporation est. up to ~17; flagged)
    "Others": 2.0, "Other": 2.0, "Other renewables": 0.1,
}
# Accounting/aggregate series in the Energy-Charts payload -- never part of the mix.
EC_NON_GENERATION = {
    "Load", "Residual load", "Renewable share of load", "Renewable share of generation",
    "Cross border electricity trading", "Hydro pumped storage consumption",
}
EC_COUNTRY = {"DE": "de", "FR": "fr", "ES": "es", "IT-NO": "it", "SE": "se", "PL": "pl"}


def real_ci_hourly(reg, start, hours):
    """Real hourly grid carbon intensity (gCO2eq/kWh) from the measured generation mix
    (Energy-Charts / ENTSO-E) x lifecycle emission factors, resampled onto `hours`.
    Production-based (excludes cross-border trade). IT-NO is approximated by country IT."""
    cc = EC_COUNTRY.get(reg, reg.lower())
    s0 = start if start.tzinfo else start.tz_localize("UTC")
    end = (s0 + pd.Timedelta(hours=len(hours)) + pd.Timedelta(days=1)).date().isoformat()
    url = (f"https://api.energy-charts.info/public_power?country={cc}"
           f"&start={s0.date().isoformat()}&end={end}")
    import time as _time
    import urllib.error as _uerr
    payload = None
    for _attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                payload = json.loads(r.read().decode())
            break
        except _uerr.HTTPError as e:  # back off on Energy-Charts rate limiting (HTTP 429)
            if e.code in (429, 500, 502, 503, 504) and _attempt < 5:
                _time.sleep(4 * (_attempt + 1))
                continue
            raise
    idx = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
    num = pd.Series(0.0, index=idx)
    den = pd.Series(0.0, index=idx)
    matched, skipped, skipped_mwh = [], [], 0.0
    for series in payload.get("production_types", []):
        name = series.get("name", "")
        if name in EC_NON_GENERATION:
            continue
        vals = pd.Series(series.get("data", []), index=idx).astype(float).clip(lower=0).fillna(0.0)
        ef = LIFECYCLE_EF.get(name)
        if ef is None:
            if float(vals.sum()) > 0:
                skipped.append(name); skipped_mwh += float(vals.sum())
            continue
        matched.append(name); num = num + vals * ef; den = den + vals
    den_sum = float(den.sum())
    excl = 100.0 * skipped_mwh / (den_sum + skipped_mwh) if (den_sum + skipped_mwh) > 0 else 0.0
    ci = num / den.replace(0.0, np.nan)
    ci_h = ci.resample("h").mean().reindex(hours).interpolate().bfill().ffill()
    print(f"    [real CI] {reg}->{cc}: {len(matched)} gen types, {excl:.1f}% generation excluded"
          + (f" (unmatched: {skipped})" if skipped else ""))
    return ci_h.values


def ewif_hourly(reg, start, hours):
    """Real hourly off-site (generation) water intensity EWIF (L consumed / kWh) from the SAME
    measured Energy-Charts generation mix as real_ci_hourly, x per-fuel water-CONSUMPTION factors
    (Macknick 2012). EWIF(h) = sum(gen_type(h) * WF_type) / sum(gen_type(h)) -- the mix-weighted
    consumption intensity. This removes the static/out-of-window-EWIF asymmetry (W1): the indirect
    water term now rides on the identical fuel mix, hour, and zone as the carbon term.

    Returns (ewif_L_per_kWh[hours], total_generation_MW[hours]). IT-NO -> country IT (as for CI)."""
    cc = EC_COUNTRY.get(reg, reg.lower())
    s0 = start if start.tzinfo else start.tz_localize("UTC")
    end = (s0 + pd.Timedelta(hours=len(hours)) + pd.Timedelta(days=1)).date().isoformat()
    url = (f"https://api.energy-charts.info/public_power?country={cc}"
           f"&start={s0.date().isoformat()}&end={end}")
    import time as _time
    import urllib.error as _uerr
    payload = None
    for _attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                payload = json.loads(r.read().decode())
            break
        except _uerr.HTTPError as e:  # back off on Energy-Charts rate limiting (HTTP 429)
            if e.code in (429, 500, 502, 503, 504) and _attempt < 5:
                _time.sleep(4 * (_attempt + 1))
                continue
            raise
    idx = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
    num = pd.Series(0.0, index=idx)   # sum(gen * WF)
    den = pd.Series(0.0, index=idx)   # sum(gen)  (total generation, the EWIF denominator)
    default_wf = WATER_FACTOR_L_PER_KWH["Others"]
    matched, defaulted, defaulted_mwh = [], [], 0.0
    for series in payload.get("production_types", []):
        name = series.get("name", "")
        if name in EC_NON_GENERATION:
            continue
        vals = pd.Series(series.get("data", []), index=idx).astype(float).clip(lower=0).fillna(0.0)
        wf = WATER_FACTOR_L_PER_KWH.get(name)
        if wf is None:
            # Unknown generation type: charge the conservative "Others" factor (do NOT drop it -- that
            # would understate off-site water) and flag it. Kept on the SAME denominator as the mix.
            wf = default_wf
            if float(vals.sum()) > 0:
                defaulted.append(name); defaulted_mwh += float(vals.sum())
        else:
            matched.append(name)
        num = num + vals * wf
        den = den + vals
    den_sum = float(den.sum())
    dflt = 100.0 * defaulted_mwh / den_sum if den_sum > 0 else 0.0
    ewif = num / den.replace(0.0, np.nan)
    ewif_h = ewif.resample("h").mean().reindex(hours).interpolate().bfill().ffill()
    gen_h = den.resample("h").mean().reindex(hours).interpolate().bfill().ffill()
    print(f"    [EWIF] {reg}->{cc}: {len(matched)} water-matched gen types, "
          f"{dflt:.1f}% generation on default WF"
          + (f" (defaulted: {defaulted})" if defaulted else ""))
    return ewif_h.values, gen_h.values


# --- MC3: short-run MARGINAL emission factor (SRMEF) via merit-order attribution ----------------
# Reviewer MC3: the carbon NO-HARM certificate is verified on the measured hourly AVERAGE
# generation-mix intensity, but the grid-supportive/consequential framing concerns the MARGINAL
# (price-setting) unit. A move can satisfy C(S)<=C(B) on average mix yet raise MARGINAL emissions if
# it shifts into an hour with a clean average mix but a dirty unit on the margin (e.g. gas).
#
# We derive a defensible, fully observable SRMEF per region-hour from the SAME measured generation
# mix (Energy-Charts/ENTSO-E) used for the real average CI, by MERIT-ORDER ATTRIBUTION. In a
# short-run economic dispatch the plants run in increasing variable (fuel) cost; a marginal change
# in net load is met by ramping the MOST EXPENSIVE dispatchable plant currently online -- the
# price-setting / marginal unit. We identify the marginal FUEL each hour as the highest-merit-order
# (most expensive, hence dirtiest dispatchable thermal) production type that is generating above a
# small floor, and set SRMEF(region, hour) = lifecycle EF of that fuel. This is the standard
# "marginal fuel" / dispatch-order proxy for a short-run marginal emission factor used when
# unit-level dispatch-stack data are unavailable (Hawkes 2010; Siler-Evans et al. 2012;
# Tranberg et al. 2019). Assumptions & scope are documented in experiments/mc3_marginal/RESULTS.md.
#
# MERIT_ORDER_RANK: ascending short-run variable cost (dispatched first -> last). Must-run / near-
# zero-marginal-cost sources (nuclear, run-of-river hydro, wind, solar, geothermal) rank lowest and
# are essentially NEVER marginal in a thermal grid; dispatchable thermal ranks highest, with the
# real-world EU ordering biomass/waste < lignite < hard coal < coal-gas < (CCGT) gas < oil. Higher
# rank = later in the stack = more likely to be the marginal (price-setting) unit.
MERIT_ORDER_RANK = {
    # must-run / zero-variable-cost (never the marginal dispatchable unit in practice)
    "Solar": 0, "Wind onshore": 0, "Wind offshore": 0,
    "Hydro Run-of-River": 1, "Nuclear": 2, "Geothermal": 2, "Other renewables": 1,
    # dispatchable hydro (storage) -- can be marginal but is low-carbon; rank above must-run
    "Hydro water reservoir": 3, "Hydro pumped storage": 3,
    # dispatchable thermal, ascending variable cost (these are the realistic marginal fuels)
    "Biomass": 4, "Waste": 4,
    "Fossil brown coal / lignite": 5,
    "Fossil hard coal": 6, "Fossil coal-derived gas": 6,
    "Fossil gas": 7,
    "Fossil oil": 8,
    "Others": 6, "Other": 6,   # unclassified, predominantly thermal -> mid-stack
}
# A fuel is "online and marginable" in an hour only if its generation exceeds this share of total
# generation. This is the load-bearing assumption: it filters PEAKING/RESERVE trace fuels (e.g. oil
# at ~0.3-0.9% share, coal-derived gas at ~0.5%) that are technically online but are NOT the unit
# that follows an ordinary MWh load change -- the marginal FOLLOWER for a shiftable-load move is the
# bulk dispatchable thermal plant on the margin (typically CCGT gas or hard coal in the 2018 EU
# summer). Calibrated to 2% from the measured 2018-07-25 mix: at 0.5% oil is spuriously "marginal"
# in DE 93% of hours (oil holds <1.3% share -- pure peaking); at 2% the marginal fuel resolves to
# gas/coal in every thermal grid, matching the documented EU marginal-fuel literature (Hawkes 2010;
# the marginal unit is "almost always" coal or gas in continental Europe). Conservative where it
# errs: in a near-fully-clean grid (SE) the residual thermal "Others" can become marginal, OVER-
# stating the marginal EF there -- which only makes the no-harm carbon test HARDER, never easier.
MARGINAL_MIN_SHARE = 0.02


def marginal_ci_hourly(reg, start, hours):
    """Short-run MARGINAL emission factor (gCO2eq/kWh) per hour for `reg`, by merit-order
    attribution to the price-setting fuel, resampled onto `hours`.

    Method: pull the measured generation mix (Energy-Charts/ENTSO-E -- the SAME payload as the real
    average CI), and for each hour pick the highest-merit-order (most expensive dispatchable) fuel
    generating above MARGINAL_MIN_SHARE of total generation. SRMEF(hour) = lifecycle EF of that fuel
    (LIFECYCLE_EF). If no dispatchable thermal fuel is online (a fully renewable/nuclear hour), the
    marginal unit is the highest-rank low-carbon dispatchable source present (storage hydro, else the
    cleanest must-run) -> SRMEF falls to that fuel's EF, capturing that an incremental MWh in such an
    hour is met essentially carbon-free. Production-based; IT-NO approximated by country IT."""
    cc = EC_COUNTRY.get(reg, reg.lower())
    s0 = start if start.tzinfo else start.tz_localize("UTC")
    end = (s0 + pd.Timedelta(hours=len(hours)) + pd.Timedelta(days=1)).date().isoformat()
    url = (f"https://api.energy-charts.info/public_power?country={cc}"
           f"&start={s0.date().isoformat()}&end={end}")
    import time as _time
    import urllib.error as _uerr
    payload = None
    for _attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                payload = json.loads(r.read().decode())
            break
        except _uerr.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and _attempt < 5:
                _time.sleep(4 * (_attempt + 1))
                continue
            raise
    idx = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
    # Build a per-fuel generation frame (MW), excluding accounting/aggregate series.
    gen = {}
    for series in payload.get("production_types", []):
        name = series.get("name", "")
        if name in EC_NON_GENERATION or name not in MERIT_ORDER_RANK:
            continue
        gen[name] = pd.Series(series.get("data", []), index=idx).astype(float).clip(lower=0).fillna(0.0)
    gdf = pd.DataFrame(gen)
    total = gdf.sum(axis=1).replace(0.0, np.nan)
    # Per-hour marginal fuel = max-rank fuel above the min-share floor; ties -> dirtiest (max EF).
    srmef = pd.Series(np.nan, index=idx)
    marg_fuel = pd.Series("", index=idx)
    for ts in idx:
        shares = gdf.loc[ts] / total.loc[ts] if total.loc[ts] and np.isfinite(total.loc[ts]) else gdf.loc[ts] * 0.0
        online = [f for f in gdf.columns if shares.get(f, 0.0) >= MARGINAL_MIN_SHARE]
        if not online:
            continue
        # pick highest merit-order rank, breaking ties by highest lifecycle EF (dirtier-on-margin)
        fuel = max(online, key=lambda f: (MERIT_ORDER_RANK.get(f, 0), LIFECYCLE_EF.get(f, 0.0)))
        srmef.loc[ts] = LIFECYCLE_EF.get(fuel, 0.0)
        marg_fuel.loc[ts] = fuel
    srmef_h = srmef.resample("h").mean().reindex(hours).interpolate().bfill().ffill()
    # diagnostics: dominant marginal fuel over the window
    mf = marg_fuel[marg_fuel != ""]
    dominant = mf.value_counts().idxmax() if len(mf) else "n/a"
    frac = (100.0 * mf.value_counts().max() / len(mf)) if len(mf) else 0.0
    print(f"    [marginal CI] {reg}->{cc}: SRMEF {float(srmef_h.min()):.0f}-{float(srmef_h.max()):.0f} g/kWh"
          f"; dominant marginal fuel = {dominant} ({frac:.0f}% of hours)")
    return srmef_h.values


def energy_charts_residual_hourly(reg: str, start: "pd.Timestamp", hours: "pd.DatetimeIndex",
                                   cache_dir: "Path | None" = None) -> "tuple[pd.Series, dict]":
    """Pull real residual load (MW) from Energy-Charts for the given region and window.

    Computes residual = Load − Wind onshore − Wind offshore − Solar (any absent series
    treated as 0 MW, matching OPSD treatment of SE solar). Resamples to the hourly `hours`
    index. Caches raw JSON under `cache_dir` so re-runs are offline.

    Returns:
        rs   : pd.Series aligned to `hours`, residual load in MW
        meta : dict with provenance fields (load_mean_mw, residual_mean_mw, n_raw_pts, ...)
    """
    import time as _time
    import urllib.error as _uerr

    cc = EC_COUNTRY.get(reg, reg.lower())
    s0 = start if start.tzinfo else pd.Timestamp(start, tz="UTC")
    # Fetch window: start day to (end day + 1) so we definitely cover all `hours`.
    fetch_start = s0.date().isoformat()
    fetch_end = (hours[-1] + pd.Timedelta(days=1)).date().isoformat()
    url = (f"https://api.energy-charts.info/public_power?country={cc}"
           f"&start={fetch_start}&end={fetch_end}")

    # --- cache ---
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"energy_charts_{cc}_{fetch_start}_{fetch_end}.json"
    else:
        cache_path = None

    if cache_path is not None and cache_path.exists():
        print(f"    [energy-charts residual] {reg}: loading from cache {cache_path.name}")
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        print(f"    [energy-charts residual] {reg}: fetching {url}")
        payload = None
        for attempt in range(6):
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    payload = json.loads(r.read().decode())
                break
            except _uerr.HTTPError as e:
                if e.code == 429 and attempt < 5:
                    _time.sleep(4 * (attempt + 1))
                    continue
                raise
        if payload is None:
            raise RuntimeError(f"energy-charts fetch failed for {reg} after retries")
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
            print(f"    [energy-charts residual] {reg}: cached -> {cache_path.name}")

    idx = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
    types_map = {t["name"]: t["data"] for t in payload.get("production_types", [])}

    def _series(name):
        if name not in types_map:
            return pd.Series(0.0, index=idx)
        vals = pd.Series(types_map[name], index=idx, dtype=float).fillna(0.0).clip(lower=0.0)
        return vals

    load_raw = _series("Load")
    wind_raw = _series("Wind onshore") + _series("Wind offshore")
    solar_raw = _series("Solar")
    resid_raw = (load_raw - wind_raw - solar_raw).clip(lower=0.0)

    # Resample to hourly then align to requested `hours` index.
    resid_h = resid_raw.resample("h").mean()
    load_h = load_raw.resample("h").mean()
    rs = resid_h.reindex(hours).interpolate().bfill().ffill()
    ls = load_h.reindex(hours).interpolate().bfill().ffill()

    n_raw = int(load_raw.notna().sum())
    raw_interval_min = int((idx[1] - idx[0]).total_seconds() // 60) if len(idx) > 1 else 60
    absent = [k for k in ["Wind onshore", "Wind offshore", "Solar"] if k not in types_map]
    if absent:
        print(f"    [energy-charts residual] {reg}: columns absent (treated as 0 MW): {absent}")

    meta = dict(
        source="energy_charts_load_minus_wind_solar",
        country_code=cc,
        url=url,
        raw_interval_min=raw_interval_min,
        n_raw_pts=n_raw,
        absent_columns=absent,
        load_mean_mw=round(float(ls.mean()), 1),
        residual_mean_mw=round(float(rs.mean()), 1),
        solar_absent=(len(absent) > 0 and "Solar" in absent),
    )
    return rs, meta


# ===================================================================================================
# EIA-930 US grid fetch (Tier-1-E generalization; --grid eia930). Free/public, NO API KEY: pulls the
# EIA Hourly Grid Monitor BALANCE bulk CSV (https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/
# EIA930_BALANCE_<YEAR>_<HALF>.csv), which carries, per balancing authority and hour, BOTH the demand
# and the net generation broken out by fuel (Coal, Natural Gas, Nuclear, Petroleum, Hydro+PumpedStorage,
# Solar, Wind, Other, Unknown). Fuel breakdown begins 2018-07-01 -- a clean calendar symmetry with the
# EU 2018 window. We map EIA's fuel buckets onto the SAME Energy-Charts fuel names used for the EU mix,
# so LIFECYCLE_EF / WATER_FACTOR_L_PER_KWH / MERIT_ORDER_RANK are reused unchanged.
#
# These functions mirror real_ci_hourly / marginal_ci_hourly / energy_charts_residual_hourly, returning
# the same shapes, so the rest of build (weather, AWARE, cooling, output) is grid-agnostic.
# ===================================================================================================

# EIA-930 "Net Generation (MW) from <fuel>" bucket -> Energy-Charts production-type name (so the EU
# EF/water/merit-order maps apply unchanged). Hydro+PumpedStorage are reported as one EIA bucket; we
# map it to reservoir hydro (the conservative, water-CONSUMPTIVE choice -- this is the BPAT sign-flip
# channel). "Other"/"Unknown" -> the "Others" unclassified-thermal default.
EIA_FUEL_TO_EC = {
    "Coal": "Fossil hard coal",
    "Natural Gas": "Fossil gas",
    "Nuclear": "Nuclear",
    "All Petroleum Products": "Fossil oil",
    "Hydropower and Pumped Storage": "Hydro water reservoir",
    "Solar": "Solar",
    "Wind": "Wind onshore",
    "Other Fuel Sources": "Others",
    "Unknown Fuel Sources": "Others",
}
EIA930_BULK_URL = ("https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/"
                   "EIA930_BALANCE_{year}_{half}.csv")
# In-process cache of the parsed bulk frame (keyed by year/half) so a multi-region build reads the
# 44 MB CSV once, not once per region/signal call.
_EIA930_BULK_CACHE: "dict" = {}


def _eia930_half(start: "pd.Timestamp") -> str:
    """EIA bulk files are half-yearly: Jan_Jun / Jul_Dec."""
    return "Jan_Jun" if start.month <= 6 else "Jul_Dec"


def _eia930_load_bulk(start: "pd.Timestamp", cache_dir: "Path | None") -> "pd.DataFrame":
    """Download (and cache) the EIA-930 BALANCE half-year bulk CSV covering `start`, return it parsed
    with a UTC datetime index column `utc`. Cached so re-runs are offline."""
    year = start.year
    half = _eia930_half(start)
    if (year, half) in _EIA930_BULK_CACHE:
        return _EIA930_BULK_CACHE[(year, half)]
    url = EIA930_BULK_URL.format(year=year, half=half)
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"EIA930_BALANCE_{year}_{half}.csv"
    else:
        cache_path = None
    if cache_path is not None and cache_path.exists():
        print(f"    [eia930] loading bulk from cache {cache_path.name}")
    else:
        print(f"    [eia930] fetching {url} (~44 MB, once per half-year) ...")
        import time as _time
        import urllib.error as _uerr
        ok = False
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "carbon-water-noharm/1.0"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                ok = True
                break
            except _uerr.HTTPError as e:
                if attempt < 4:
                    _time.sleep(4 * (attempt + 1))
                    continue
                raise
            except _uerr.URLError:
                if attempt < 4:
                    _time.sleep(4 * (attempt + 1))
                    continue
                raise
        if not ok:
            raise RuntimeError(f"EIA-930 bulk fetch failed for {url}")
        if cache_path is not None:
            cache_path.write_bytes(data)
            print(f"    [eia930] cached -> {cache_path.name}")
            df = pd.read_csv(cache_path, low_memory=False)
        else:
            import io
            df = pd.read_csv(io.BytesIO(data), low_memory=False)
        df["utc"] = pd.to_datetime(df["UTC Time at End of Hour"],
                                   format="%m/%d/%Y %I:%M:%S %p", utc=True, errors="coerce")
        _EIA930_BULK_CACHE[(year, half)] = df
        return df
    df = pd.read_csv(cache_path, low_memory=False)
    df["utc"] = pd.to_datetime(df["UTC Time at End of Hour"],
                               format="%m/%d/%Y %I:%M:%S %p", utc=True, errors="coerce")
    _EIA930_BULK_CACHE[(year, half)] = df
    return df


def _eia930_gen_frame(reg: str, start: "pd.Timestamp", hours: "pd.DatetimeIndex",
                      cache_dir: "Path | None") -> "tuple[pd.DataFrame, pd.Series]":
    """Per-fuel generation frame (MW, columns are Energy-Charts names) and the demand series for one
    BA over the fetch window, hourly-indexed. The fetch window spans start..hours[-1]+1d."""
    bulk = _eia930_load_bulk(start, cache_dir)
    resp = US_SITES.get(reg, {}).get("eia_respondent", reg)
    sub = bulk[bulk["Balancing Authority"] == resp].copy()
    if sub.empty:
        raise RuntimeError(f"no EIA-930 rows for balancing authority '{resp}' (region {reg})")
    s0 = start if start.tzinfo else pd.Timestamp(start, tz="UTC")
    lo = s0 - pd.Timedelta(hours=1)
    hi = hours[-1] + pd.Timedelta(days=1)
    sub = sub[(sub["utc"] >= lo) & (sub["utc"] <= hi)].sort_values("utc").set_index("utc")
    gen = {}
    for eia_bucket, ec_name in EIA_FUEL_TO_EC.items():
        col = f"Net Generation (MW) from {eia_bucket}"
        if col not in sub.columns:
            continue
        vals = pd.to_numeric(sub[col], errors="coerce").clip(lower=0.0).fillna(0.0)
        # Hydro and the two "Others" buckets can both map to one EC name -> sum them.
        gen[ec_name] = gen.get(ec_name, 0.0) + vals
    gdf = pd.DataFrame(gen)
    demand = pd.to_numeric(sub.get("Demand (MW)"), errors="coerce")
    return gdf, demand


def eia930_real_ci_hourly(reg, start, hours, cache_dir=None):
    """Real hourly average CI (gCO2eq/kWh) from the EIA-930 measured generation mix x IPCC AR5 EFs,
    resampled onto `hours`. Same method as real_ci_hourly, US mix."""
    gdf, _ = _eia930_gen_frame(reg, start, hours, cache_dir)
    ef = pd.Series({c: LIFECYCLE_EF.get(c, 700.0) for c in gdf.columns})
    den = gdf.sum(axis=1)
    num = gdf.mul(ef, axis=1).sum(axis=1)
    ci = (num / den.replace(0.0, np.nan))
    ci_h = ci.resample("h").mean().reindex(hours).interpolate().bfill().ffill()
    mix = (gdf.sum() / gdf.sum().sum() * 100).round(1).to_dict()
    print(f"    [eia930 real CI] {reg}: {float(ci_h.mean()):.0f} g/kWh mean "
          f"({float(ci_h.min()):.0f}-{float(ci_h.max()):.0f}); mix%={mix}")
    return ci_h.values


def eia930_marginal_ci_hourly(reg, start, hours, cache_dir=None):
    """Short-run MARGINAL EF (gCO2eq/kWh) per hour via merit-order attribution on the EIA-930 mix,
    resampled onto `hours`. Same MERIT_ORDER_RANK / MARGINAL_MIN_SHARE logic as marginal_ci_hourly."""
    gdf, _ = _eia930_gen_frame(reg, start, hours, cache_dir)
    cols = [c for c in gdf.columns if c in MERIT_ORDER_RANK]
    gdf = gdf[cols]
    total = gdf.sum(axis=1).replace(0.0, np.nan)
    srmef = pd.Series(np.nan, index=gdf.index)
    marg_fuel = pd.Series("", index=gdf.index)
    for ts in gdf.index:
        t = total.loc[ts]
        shares = gdf.loc[ts] / t if (t and np.isfinite(t)) else gdf.loc[ts] * 0.0
        online = [f for f in gdf.columns if shares.get(f, 0.0) >= MARGINAL_MIN_SHARE]
        if not online:
            continue
        fuel = max(online, key=lambda f: (MERIT_ORDER_RANK.get(f, 0), LIFECYCLE_EF.get(f, 0.0)))
        srmef.loc[ts] = LIFECYCLE_EF.get(fuel, 0.0)
        marg_fuel.loc[ts] = fuel
    srmef_h = srmef.resample("h").mean().reindex(hours).interpolate().bfill().ffill()
    mf = marg_fuel[marg_fuel != ""]
    dominant = mf.value_counts().idxmax() if len(mf) else "n/a"
    frac = (100.0 * mf.value_counts().max() / len(mf)) if len(mf) else 0.0
    print(f"    [eia930 marginal CI] {reg}: SRMEF {float(srmef_h.min()):.0f}-{float(srmef_h.max()):.0f} g/kWh"
          f"; dominant marginal fuel = {dominant} ({frac:.0f}% of hours)")
    return srmef_h.values


def eia930_ewif_hourly(reg, start, hours, cache_dir=None):
    """Off-site (generation) water-CONSUMPTION intensity EWIF (L/kWh) per hour from the EIA-930 mix x
    Macknick WATER_FACTOR_L_PER_KWH, resampled onto `hours`. This is the term that flips the spatial
    water co-benefit sign: a hydro-heavy clean grid (BPAT) carries a HIGH EWIF (reservoir evaporation)
    even though its CI is low. Returns (ewif_l_per_kwh array, demand_mw array)."""
    gdf, demand = _eia930_gen_frame(reg, start, hours, cache_dir)
    wf = pd.Series({c: WATER_FACTOR_L_PER_KWH.get(c, 2.0) for c in gdf.columns})
    den = gdf.sum(axis=1)
    num = gdf.mul(wf, axis=1).sum(axis=1)
    ewif = (num / den.replace(0.0, np.nan))
    ewif_h = ewif.resample("h").mean().reindex(hours).interpolate().bfill().ffill()
    dem_h = demand.resample("h").mean().reindex(hours).interpolate().bfill().ffill() if demand is not None else None
    print(f"    [eia930 EWIF] {reg}: {float(ewif_h.mean()):.3f} L/kWh mean "
          f"({float(ewif_h.min()):.3f}-{float(ewif_h.max()):.3f})")
    return ewif_h.values, (dem_h.values if dem_h is not None else np.full(len(hours), np.nan))


def eia930_residual_hourly(reg, start, hours, cache_dir=None):
    """Real residual load (MW) = Demand - Wind - Solar for one BA, resampled onto `hours`. Mirrors
    energy_charts_residual_hourly. Returns (pd.Series aligned to hours, meta dict)."""
    gdf, demand = _eia930_gen_frame(reg, start, hours, cache_dir)
    wind = gdf.get("Wind onshore", pd.Series(0.0, index=gdf.index))
    solar = gdf.get("Solar", pd.Series(0.0, index=gdf.index))
    if demand is None or demand.isna().all():
        # Fall back to net generation if demand is missing (rare): use total gen as the load proxy.
        demand = gdf.sum(axis=1)
    resid = (demand.fillna(0.0) - wind - solar).clip(lower=0.0)
    rs = resid.resample("h").mean().reindex(hours).interpolate().bfill().ffill()
    meta = dict(
        source="eia930_demand_minus_wind_solar",
        respondent=US_SITES.get(reg, {}).get("eia_respondent", reg),
        url=EIA930_BULK_URL.format(year=start.year, half=_eia930_half(start)),
        raw_interval_min=60,
        residual_mean_mw=round(float(rs.mean()), 1),
        absent_columns=[],
    )
    return rs, meta


def _assert_signal_coverage_in_window(rows, n, start, label, ts_field):
    """Build-time consistency guard: every region present in `rows` must have exactly the n slots
    0..n-1, and every timestamp must fall in the SAME calendar year as the build window. This catches,
    at the source, the two bugs that motivated the water revamp -- a partial-coverage table (e.g. 25/48
    EWIF slots) and a stale out-of-window vintage (e.g. EWIF stamped 2025-02 while CI is 2018-07)."""
    by_region = {}
    for r in rows:
        by_region.setdefault(r["region"], set()).add(int(r["slot_index"]))
    expected = set(range(n))
    for reg, slots in by_region.items():
        if slots != expected:
            missing = sorted(expected - slots)
            raise AssertionError(f"[{label}] region {reg}: slot coverage {len(slots)}/{n} "
                                 f"(missing {missing[:5]}{'...' if len(missing) > 5 else ''})")
    bad = sorted({r[ts_field][:4] for r in rows} - {str(start.year)})
    if bad:
        raise AssertionError(f"[{label}] out-of-window timestamps: years {bad} != build year {start.year}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2018-07-25T00:00:00Z", help="window start (UTC, documented 2018 heatwave)")
    ap.add_argument("--slots", type=int, default=48, help="number of hourly slots (superset; pilot reads first max_timeslots)")
    ap.add_argument("--out-dir", default=None, help="output dir (default pkg/.../data/timealigned); set to build a separate year/window without overwriting")
    ap.add_argument("--carbon-source", choices=["proxy", "real", "marginal"], default="proxy",
                    help="proxy = residual-scaled country-average ci_base (default, bit-identical); "
                         "real = measured generation-mix AVERAGE CI (Energy-Charts + IPCC AR5 lifecycle EFs); "
                         "marginal = short-run MARGINAL emission factor (SRMEF) via merit-order "
                         "attribution to the price-setting fuel (MC3 marginal-carbon re-verification)")
    ap.add_argument("--residual-source", choices=["opsd", "energy-charts"], default="opsd",
                    help="opsd = OPSD local CSV + analog-year fallback (default, bit-identical); "
                         "energy-charts = real measured residual load from Energy-Charts API "
                         "(residual = Load − Wind − Solar; cache under --residual-cache-dir).")
    ap.add_argument("--residual-cache-dir", default=None,
                    help="directory for caching raw Energy-Charts JSON responses "
                         "(default: <out-dir>/../../../experiments/strong_real2022/cache/); "
                         "only used when --residual-source=energy-charts")
    ap.add_argument("--grid", choices=["eu", "eia930"], default="eu",
                    help="eu = European SITES via OPSD/Energy-Charts (default, bit-identical); "
                         "eia930 = US balancing authorities (ERCO/CISO/BPAT) via the free EIA-930 "
                         "Hourly Grid Monitor bulk CSV (Tier-1-E generalization; --carbon-source and "
                         "--residual-source are ignored, both come from the EIA-930 measured mix). "
                         "Also emits ewif_us_region_slot.csv (off-site water from the US mix).")
    ap.add_argument("--eia930-cache-dir", default=None,
                    help="cache dir for EIA-930 bulk CSV (default: <out-dir>/cache/); only --grid eia930")
    ap.add_argument("--hydro-reservoir-lkwh", type=float,
                    default=WATER_FACTOR_L_PER_KWH["Hydro water reservoir"],
                    help="reservoir-hydro operational water-consumption factor (L/kWh). Default keeps the "
                         "committed conservative 2.0; the flagged sensitivity (estimates span ~0 run-of-river "
                         "to ~17 gross reservoir evaporation) sweeps this. A run-of-river share s in the "
                         "combined EIA hydro bucket is equivalent to scaling the factor by (1-s), so one "
                         "dial covers both the factor and the bucket-mapping question.")
    args = ap.parse_args()
    WATER_FACTOR_L_PER_KWH["Hydro water reservoir"] = args.hydro_reservoir_lkwh
    start = pd.Timestamp(args.start)
    n = args.slots
    OUT = Path(args.out_dir).resolve() if args.out_dir else (REPO / "pkg" / "carbon-aware" / "data" / "timealigned")
    OUT.mkdir(parents=True, exist_ok=True)
    hours = pd.date_range(start, periods=n, freq="h", tz="UTC")

    # ==================================================================
    # US testbed branch (--grid eia930): self-contained EIA-930 path.
    # Carbon (avg+marginal selectable), residual, EWIF all come from the
    # measured US mix; --residual-source/--carbon-source are ignored.
    # ==================================================================
    if args.grid == "eia930":
        return _build_eia930(args, start, n, hours, OUT)

    # ------------------------------------------------------------------
    # OPSD load (used only when --residual-source opsd, the default)
    # ------------------------------------------------------------------
    if args.residual_source == "opsd":
        header = pd.read_csv(OPSD_LOCAL, nrows=0).columns.tolist()
        needed = ["utc_timestamp"]
        for s in SITES.values():
            needed += [s["load"], *s["wind"]]
            if s["solar"]:  # solar is optional (e.g. SE has no national solar column in OPSD)
                needed.append(s["solar"])
        use = [c for c in needed if c in header]
        df = pd.read_csv(OPSD_LOCAL, usecols=use)
        df["ts"] = pd.to_datetime(df["utc_timestamp"], utc=True)
        df = df.set_index("ts")
    else:
        df = None  # not used in energy-charts path

    # Resolve the cache dir for energy-charts residual fetches.
    if args.residual_source == "energy-charts":
        if args.residual_cache_dir:
            ec_cache_dir = Path(args.residual_cache_dir).resolve()
        else:
            ec_cache_dir = REPO / "experiments" / "strong_real2022" / "cache"
        ec_cache_dir.mkdir(parents=True, exist_ok=True)
    else:
        ec_cache_dir = None

    grid_rows, wue_rows, ewif_rows, forecasts = [], [], [], {}
    for reg, s in SITES.items():
        # ------------------------------------------------------------------
        # Residual-load signal: OPSD/analog (default) OR real Energy-Charts
        # ------------------------------------------------------------------
        if args.residual_source == "opsd":
            assert df is not None
            if s["load"] not in df.columns:
                print(f"WARN: {reg} missing OPSD load column; skipping (pilot will use defaults)")
                continue
            wind_cols = [c for c in s["wind"] if c in df.columns]
            wind = df[wind_cols].sum(axis=1) if wind_cols else 0.0
            # Solar is optional: regions without a solar column (e.g. SE in OPSD) treat solar as 0 MW.
            if s["solar"] and s["solar"] in df.columns:
                solar = df[s["solar"]]
            else:
                if s["solar"]:
                    print(f"WARN: {reg} solar column '{s['solar']}' absent; treating solar=0")
                solar = 0.0
            resid = df[s["load"]] - wind - solar
            # Residual-load signal source year. The OPSD snapshot may not cover the requested
            # carbon/weather window (e.g. Aug-2022 vs OPSD ending 2020-09). If the requested year
            # is absent, fall back to the SAME month/day/hour in the most recent OPSD year that
            # has data for this region (an "analog year" for the diurnal residual-load shape) so
            # the grid-relief signal is physical instead of NaN. Carbon (real CI) and weather
            # (Open-Meteo) stay on the TRUE requested dates; only the residual-load shape is
            # borrowed. When the requested year IS covered this is a no-op (bit-identical).
            avail_years = sorted(resid[resid.notna()].index.year.unique())
            grid_source_year = start.year
            if start.year not in avail_years and avail_years:
                grid_source_year = max(y for y in avail_years if y <= start.year) \
                    if any(y <= start.year for y in avail_years) else max(avail_years)
            if grid_source_year != start.year:
                grid_hours = hours - pd.DateOffset(years=start.year - grid_source_year)
                grid_model = (f"residual_load_lite_load_minus_wind_solar "
                              f"(analog year {grid_source_year}; OPSD lacks {start.year})")
            else:
                grid_hours = hours
                grid_model = "residual_load_lite_load_minus_wind_solar"
            zmean = resid[resid.index.year == grid_source_year].mean()
            if not np.isfinite(zmean) or zmean == 0:
                zmean = resid.mean()
            rs = resid.reindex(grid_hours).interpolate().bfill().ffill()
            grid_source_dataset = "OPSD time_series 2020-10-06"
        else:
            # energy-charts path: real Aug-YYYY residual load from ENTSO-E via Energy-Charts API.
            rs, ec_meta = energy_charts_residual_hourly(reg, start, hours, cache_dir=ec_cache_dir)
            grid_model = (f"energy_charts_residual_load_minus_wind_solar"
                          f" (real {start.year}; Load−WindOnshore−WindOffshore−Solar"
                          f"; {ec_meta['raw_interval_min']}min→1h resample)")
            if ec_meta["absent_columns"]:
                grid_model += f" [absent treated as 0 MW: {ec_meta['absent_columns']}]"
            grid_source_dataset = (f"Energy-Charts / ENTSO-E {ec_meta['country_code']} "
                                   f"{start.date().isoformat()} via {ec_meta['url']}")
            zmean = float(rs.mean())
        w = weather(s["lat"], s["lon"], start, n).reindex(hours).interpolate().bfill().ffill()
        Tw = wet_bulb(w["T"], w["RH"]).values
        prof = load_cooling_profile(reg)
        wue, pue, sigma = cooling_for_profile(prof, w["T"].values, Tw)  # per-slot, climate-driven
        kappa = prof["profile_class"]  # metadata (e.g. "hybrid_economized"); not a behavioural switch
        if args.carbon_source == "real":
            ci = real_ci_hourly(reg, start, hours)
            ci_model = "energy_charts_generation_mix_x_ipcc_ar5_lifecycle_ef"
        elif args.carbon_source == "marginal":
            ci = marginal_ci_hourly(reg, start, hours)
            ci_model = "energy_charts_merit_order_marginal_fuel_srmef_x_ipcc_ar5_lifecycle_ef"
        else:
            ci = s["ci_base"] * np.clip(rs.values / zmean, 0.3, 2.0)
            ci_model = "residual_load_scaled_country_average"
        # Off-site water intensity (EWIF) from the SAME measured Energy-Charts mix as the real CI, so
        # the indirect-water term is in-window and mix-coherent (fixes the stale Feb-2025 static EWIF).
        ewif, gen_mw = ewif_hourly(reg, start, hours)
        fc = []
        for k in range(n):
            tss = (start + pd.Timedelta(hours=k)).strftime("%Y-%m-%dT%H:%M:%SZ")
            grid_rows.append(dict(region=reg, slot_index=k, utc_timestamp=tss,
                                  residual_load_mw=round(float(rs.values[k]), 3),
                                  signal_model=grid_model,
                                  source_dataset=grid_source_dataset))
            ewif_rows.append(dict(region=reg, slot_index=k, forecast_datetime_utc=tss,
                                  mix_datetime_utc=tss,
                                  ewif_l_per_kwh=round(float(ewif[k]), 6),
                                  ewif_scenario="base",
                                  power_consumption_total_mw=round(float(gen_mw[k]), 1),
                                  non_storage_share=1.0, inherit_share=0.0, is_estimated=False,
                                  estimation_method="",
                                  mix_notes="Energy-Charts/ENTSO-E measured fuel mix x Macknick 2012 "
                                            "consumption factors (same payload as real CI)",
                                  source_dataset="Energy-Charts / ENTSO-E generation mix + Macknick 2012 (NREL)",
                                  source_url=(f"https://api.energy-charts.info/public_power?country="
                                              f"{EC_COUNTRY.get(reg, reg.lower())}")))
            wue_rows.append(dict(region=reg, slot_index=k, weather_time_utc=tss,
                                 wet_bulb_c=round(float(Tw[k]), 3),
                                 direct_wue_l_per_kwh=round(float(wue[k]), 4),
                                 pue=round(float(pue[k]), 4), cooling_kappa=kappa,
                                 evaporative_mode_active=bool(sigma[k] > 0.5),
                                 evaporative_engagement=round(float(sigma[k]), 4),
                                 model="open-meteo archive + climate-driven hybrid-economized cooling "
                                       "(region_cooling_profiles base), T-dependent PUE"))
            fc.append(dict(datetime=tss, carbonIntensity=round(float(ci[k]), 2)))
        forecasts[reg] = dict(zone=reg, forecast=fc, ci_model=ci_model, updatedAt=start.strftime("%Y-%m-%dT%H:%M:%SZ"))

    # Consistency guard before persisting (full coverage + in-window) -- fail loud at build time.
    _assert_signal_coverage_in_window(grid_rows, n, start, "grid", "utc_timestamp")
    _assert_signal_coverage_in_window(wue_rows, n, start, "wue", "weather_time_utc")
    _assert_signal_coverage_in_window(ewif_rows, n, start, "ewif", "forecast_datetime_utc")
    pd.DataFrame(grid_rows).to_csv(OUT / "grid_residual_region_slot.csv", index=False)
    pd.DataFrame(wue_rows).to_csv(OUT / "wue_region_slot.csv", index=False)
    pd.DataFrame(ewif_rows).to_csv(OUT / "ewif_region_slot.csv", index=False)
    (OUT / "forecasts.json").write_text(json.dumps(forecasts, indent=2), encoding="utf-8")
    regions = sorted(forecasts)
    print(f"wrote time-aligned tables to {OUT}")
    print(f"window={args.start} slots={n} regions={regions}")
    for reg in regions:
        cis = [e["carbonIntensity"] for e in forecasts[reg]["forecast"]]
        rr = [r for r in wue_rows if r["region"] == reg]
        wb = [r["wet_bulb_c"] for r in rr]
        wu = [r["direct_wue_l_per_kwh"] for r in rr]
        print(f"  {reg}: kappa={rr[0]['cooling_kappa']} pue={rr[0]['pue']} | CI {min(cis):.0f}-{max(cis):.0f} g/kWh, "
              f"wet-bulb {min(wb):.1f}-{max(wb):.1f} C, WUE {min(wu):.2f}-{max(wu):.2f} L/kWh")
    return 0


def _build_eia930(args, start, n, hours, OUT) -> int:
    """US testbed builder (--grid eia930). Mirrors the EU loop but pulls carbon (avg or marginal),
    residual load, and the off-site EWIF all from the measured EIA-930 US generation mix. Emits the
    same three artifacts PLUS ewif_us_region_slot.csv (off-site/indirect water per region-slot, which
    the engine reads from data/water/ewif_region_slot.csv -- copy/symlink as needed for a run)."""
    if args.eia930_cache_dir:
        cache_dir = Path(args.eia930_cache_dir).resolve()
    else:
        cache_dir = OUT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    grid_rows, wue_rows, ewif_rows, forecasts = [], [], [], {}
    for reg, s in US_SITES.items():
        rs, gmeta = eia930_residual_hourly(reg, start, hours, cache_dir=cache_dir)
        grid_model = (f"eia930_residual_demand_minus_wind_solar (real {start.year}; "
                      f"BA={gmeta['respondent']}; 60min)")
        grid_source_dataset = f"EIA-930 Hourly Grid Monitor {gmeta['respondent']} via {gmeta['url']}"

        w = weather(s["lat"], s["lon"], start, n).reindex(hours).interpolate().bfill().ffill()
        Tw = wet_bulb(w["T"], w["RH"]).values
        prof = load_cooling_profile(reg)
        wue, pue, sigma = cooling_for_profile(prof, w["T"].values, Tw)
        kappa = prof["profile_class"]

        if args.carbon_source == "marginal":
            ci = eia930_marginal_ci_hourly(reg, start, hours, cache_dir=cache_dir)
            ci_model = "eia930_merit_order_marginal_fuel_srmef_x_ipcc_ar5_lifecycle_ef"
        else:  # "real" or "proxy" both resolve to the measured-mix average CI on the US grid
            ci = eia930_real_ci_hourly(reg, start, hours, cache_dir=cache_dir)
            ci_model = "eia930_generation_mix_x_ipcc_ar5_lifecycle_ef"

        ewif, demand_mw = eia930_ewif_hourly(reg, start, hours, cache_dir=cache_dir)

        fc = []
        for k in range(n):
            tss = (start + pd.Timedelta(hours=k)).strftime("%Y-%m-%dT%H:%M:%SZ")
            grid_rows.append(dict(region=reg, slot_index=k, utc_timestamp=tss,
                                  residual_load_mw=round(float(rs.values[k]), 3),
                                  signal_model=grid_model, source_dataset=grid_source_dataset))
            wue_rows.append(dict(region=reg, slot_index=k, weather_time_utc=tss,
                                 wet_bulb_c=round(float(Tw[k]), 3),
                                 direct_wue_l_per_kwh=round(float(wue[k]), 4),
                                 pue=round(float(pue[k]), 4), cooling_kappa=kappa,
                                 evaporative_mode_active=bool(sigma[k] > 0.5),
                                 evaporative_engagement=round(float(sigma[k]), 4),
                                 model="open-meteo archive + climate-driven hybrid-economized cooling "
                                       "(region_cooling_profiles base), T-dependent PUE"))
            dem = float(demand_mw[k]) if np.isfinite(demand_mw[k]) else float(rs.values[k])
            ewif_rows.append(dict(region=reg, slot_index=k, forecast_datetime_utc=tss,
                                  mix_datetime_utc=tss,
                                  ewif_l_per_kwh=round(float(ewif[k]), 6),
                                  ewif_scenario="base",
                                  power_consumption_total_mw=round(dem, 1),
                                  non_storage_share=1.0, inherit_share=0.0, is_estimated=False,
                                  estimation_method="",
                                  mix_notes="EIA-930 measured fuel mix x Macknick consumption factors",
                                  source_dataset="EIA-930 Hourly Grid Monitor + Macknick 2012 (NREL)",
                                  source_url=gmeta["url"]))
            fc.append(dict(datetime=tss, carbonIntensity=round(float(ci[k]), 2)))
        forecasts[reg] = dict(zone=reg, forecast=fc, ci_model=ci_model,
                              updatedAt=start.strftime("%Y-%m-%dT%H:%M:%SZ"))

    pd.DataFrame(grid_rows).to_csv(OUT / "grid_residual_region_slot.csv", index=False)
    pd.DataFrame(wue_rows).to_csv(OUT / "wue_region_slot.csv", index=False)
    pd.DataFrame(ewif_rows).to_csv(OUT / "ewif_us_region_slot.csv", index=False)
    (OUT / "forecasts.json").write_text(json.dumps(forecasts, indent=2), encoding="utf-8")
    regions = sorted(forecasts)
    print(f"wrote US (EIA-930) time-aligned tables to {OUT}")
    print(f"window={args.start} slots={n} regions={regions} carbon_source={args.carbon_source}")
    for reg in regions:
        cis = [e["carbonIntensity"] for e in forecasts[reg]["forecast"]]
        ew = [r["ewif_l_per_kwh"] for r in ewif_rows if r["region"] == reg]
        rr = [r for r in wue_rows if r["region"] == reg]
        wu = [r["direct_wue_l_per_kwh"] for r in rr]
        print(f"  {reg}: CI {min(cis):.0f}-{max(cis):.0f} g/kWh | EWIF {min(ew):.3f}-{max(ew):.3f} L/kWh "
              f"| direct-WUE {min(wu):.2f}-{max(wu):.2f} L/kWh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
