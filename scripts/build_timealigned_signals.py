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
}


def wet_bulb(T, RH):
    RH = np.clip(RH, 1, 100)
    return (T * np.arctan(0.151977 * np.sqrt(RH + 8.313659)) + np.arctan(T + RH)
            - np.arctan(RH - 1.676331) + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH) - 4.686035)


def hybrid_wue(T, Tw):
    """Wet-bulb-gated hybrid cooling direct WUE (L/kWh); same model as the PoC."""
    wue_wet = np.clip(0.55 + 0.060 * np.clip(Tw - 2, 0, None), 0.0, 3.0)
    return np.where(T < 14, 0.0, np.where(T < 22, 0.40 * wue_wet, 0.70 * wue_wet))


# Step 4: fixed per-site cooling architecture (kappa). Evaporative = low energy /
# high water; dry/closed-loop = high energy / ~zero water; hybrid in between. PUE
# and WUE are COUPLED so dry cooling is never free -- it trades carbon for water.
KAPPA = {"FR": "evaporative", "DE": "dry", "ES": "dry", "IT-NO": "hybrid"}
# Design-point (T_ref) PUE per cooling architecture: evaporative/water-cooled ~1.10
# (hyperscale, e.g. Google/Meta), air-cooled/dry ~1.30, hybrid in between (Shehabi 2016;
# Lei & Masanet 2022; ASHRAE). T-sensitivity below is the chiller/condenser efficiency
# loss with rising ambient -- steepest for dry cooling -- which couples carbon to heat.
PUE_BY_KAPPA = {"evaporative": 1.10, "hybrid": 1.16, "dry": 1.30}
PUE_T_REF = 20.0  # design-point dry-bulb (C) at which PUE = base
PUE_T_SLOPE = {"evaporative": 0.004, "hybrid": 0.010, "dry": 0.018}  # PUE rise per C above T_ref
PUE_CAP = {"evaporative": 1.25, "hybrid": 1.45, "dry": 1.70}


def pue_for_kappa(T, kappa):
    """Temperature-dependent PUE (L. Berkeley/ASHRAE): facility overhead rises with ambient
    dry-bulb as chiller/condenser efficiency drops; the slope is steepest for dry/air cooling
    and mildest for evaporative/water cooling. Flat below the design point T_ref."""
    base = PUE_BY_KAPPA[kappa]
    rise = PUE_T_SLOPE.get(kappa, 0.010) * np.clip(np.asarray(T, float) - PUE_T_REF, 0.0, None)
    return np.clip(base + rise, base, PUE_CAP.get(kappa, 1.6))


def wue_for_kappa(T, Tw, kappa):
    wet = np.clip(0.55 + 0.060 * np.clip(Tw - 2, 0, None), 0.0, 3.0)
    if kappa == "evaporative":
        return wet
    if kappa == "dry":
        return np.zeros_like(wet)
    return hybrid_wue(T, Tw)


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
# Accounting/aggregate series in the Energy-Charts payload -- never part of the mix.
EC_NON_GENERATION = {
    "Load", "Residual load", "Renewable share of load", "Renewable share of generation",
    "Cross border electricity trading", "Hydro pumped storage consumption",
}
EC_COUNTRY = {"DE": "de", "FR": "fr", "ES": "es", "IT-NO": "it"}


def real_ci_hourly(reg, start, hours):
    """Real hourly grid carbon intensity (gCO2eq/kWh) from the measured generation mix
    (Energy-Charts / ENTSO-E) x lifecycle emission factors, resampled onto `hours`.
    Production-based (excludes cross-border trade). IT-NO is approximated by country IT."""
    cc = EC_COUNTRY.get(reg, reg.lower())
    s0 = start if start.tzinfo else start.tz_localize("UTC")
    end = (s0 + pd.Timedelta(hours=len(hours)) + pd.Timedelta(days=1)).date().isoformat()
    url = (f"https://api.energy-charts.info/public_power?country={cc}"
           f"&start={s0.date().isoformat()}&end={end}")
    with urllib.request.urlopen(url, timeout=120) as r:
        payload = json.loads(r.read().decode())
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2018-07-25T00:00:00Z", help="window start (UTC, documented 2018 heatwave)")
    ap.add_argument("--slots", type=int, default=48, help="number of hourly slots (superset; pilot reads first max_timeslots)")
    ap.add_argument("--out-dir", default=None, help="output dir (default pkg/.../data/timealigned); set to build a separate year/window without overwriting")
    ap.add_argument("--carbon-source", choices=["proxy", "real"], default="proxy",
                    help="proxy = residual-scaled country-average ci_base (default, bit-identical); "
                         "real = measured generation-mix CI (Energy-Charts + IPCC AR5 lifecycle EFs)")
    args = ap.parse_args()
    start = pd.Timestamp(args.start)
    n = args.slots
    OUT = Path(args.out_dir).resolve() if args.out_dir else (REPO / "pkg" / "carbon-aware" / "data" / "timealigned")
    OUT.mkdir(parents=True, exist_ok=True)
    hours = pd.date_range(start, periods=n, freq="h", tz="UTC")

    header = pd.read_csv(OPSD_LOCAL, nrows=0).columns.tolist()
    needed = ["utc_timestamp"]
    for s in SITES.values():
        needed += [s["load"], s["solar"], *s["wind"]]
    use = [c for c in needed if c in header]
    df = pd.read_csv(OPSD_LOCAL, usecols=use)
    df["ts"] = pd.to_datetime(df["utc_timestamp"], utc=True)
    df = df.set_index("ts")

    grid_rows, wue_rows, forecasts = [], [], {}
    for reg, s in SITES.items():
        if s["load"] not in df.columns or s["solar"] not in df.columns:
            print(f"WARN: {reg} missing OPSD columns; skipping (pilot will use defaults)")
            continue
        wind_cols = [c for c in s["wind"] if c in df.columns]
        wind = df[wind_cols].sum(axis=1) if wind_cols else 0.0
        resid = df[s["load"]] - wind - df[s["solar"]]
        zmean = resid[resid.index.year == start.year].mean()
        if not np.isfinite(zmean) or zmean == 0:
            zmean = resid.mean()
        rs = resid.reindex(hours).interpolate().bfill().ffill()
        w = weather(s["lat"], s["lon"], start, n).reindex(hours).interpolate().bfill().ffill()
        Tw = wet_bulb(w["T"], w["RH"]).values
        kappa = KAPPA.get(reg, "hybrid")
        pue = pue_for_kappa(w["T"].values, kappa)  # per-slot, temperature-dependent
        wue = wue_for_kappa(w["T"].values, Tw, kappa)
        if args.carbon_source == "real":
            ci = real_ci_hourly(reg, start, hours)
            ci_model = "energy_charts_generation_mix_x_ipcc_ar5_lifecycle_ef"
        else:
            ci = s["ci_base"] * np.clip(rs.values / zmean, 0.3, 2.0)
            ci_model = "residual_load_scaled_country_average"
        fc = []
        for k in range(n):
            tss = (start + pd.Timedelta(hours=k)).strftime("%Y-%m-%dT%H:%M:%SZ")
            grid_rows.append(dict(region=reg, slot_index=k, utc_timestamp=tss,
                                  residual_load_mw=round(float(rs.values[k]), 3),
                                  signal_model="residual_load_lite_load_minus_wind_solar",
                                  source_dataset="OPSD time_series 2020-10-06"))
            wue_rows.append(dict(region=reg, slot_index=k, weather_time_utc=tss,
                                 wet_bulb_c=round(float(Tw[k]), 3),
                                 direct_wue_l_per_kwh=round(float(wue[k]), 4),
                                 pue=round(float(pue[k]), 4), cooling_kappa=kappa,
                                 model="open-meteo archive + fixed-kappa cooling, T-dependent PUE"))
            fc.append(dict(datetime=tss, carbonIntensity=round(float(ci[k]), 2)))
        forecasts[reg] = dict(zone=reg, forecast=fc, ci_model=ci_model, updatedAt=start.strftime("%Y-%m-%dT%H:%M:%SZ"))

    pd.DataFrame(grid_rows).to_csv(OUT / "grid_residual_region_slot.csv", index=False)
    pd.DataFrame(wue_rows).to_csv(OUT / "wue_region_slot.csv", index=False)
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


if __name__ == "__main__":
    raise SystemExit(main())
