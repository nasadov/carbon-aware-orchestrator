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
PUE_BY_KAPPA = {"evaporative": 1.10, "hybrid": 1.16, "dry": 1.30}


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2018-07-25T00:00:00Z", help="window start (UTC, documented 2018 heatwave)")
    ap.add_argument("--slots", type=int, default=48, help="number of hourly slots (superset; pilot reads first max_timeslots)")
    args = ap.parse_args()
    start = pd.Timestamp(args.start)
    n = args.slots
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
        pue = PUE_BY_KAPPA[kappa]
        wue = wue_for_kappa(w["T"].values, Tw, kappa)
        ci = s["ci_base"] * np.clip(rs.values / zmean, 0.3, 2.0)
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
                                 pue=pue, cooling_kappa=kappa,
                                 model="open-meteo archive + fixed-kappa cooling"))
            fc.append(dict(datetime=tss, carbonIntensity=round(float(ci[k]), 2)))
        forecasts[reg] = dict(zone=reg, forecast=fc, updatedAt=start.strftime("%Y-%m-%dT%H:%M:%SZ"))

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
