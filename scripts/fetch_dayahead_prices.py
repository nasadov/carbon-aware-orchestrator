#!/usr/bin/env python3
"""Fetch hourly DAY-AHEAD electricity prices for a time-aligned signals dir (fifth no-harm axis).

Reads the dir's forecasts.json to learn the window (start, #slots) and regions, pulls day-ahead
prices from the FREE/tokenless Energy-Charts API (https://api.energy-charts.info/price?bzn=...),
and writes price_region_slot.csv (region, slot_index, utc_timestamp, price_eur_per_kwh, bzn) into
the same dir -- the table PilotConfig.price_csv consumes when the cost axis is on.

Zone mapping notes:
  * DE was part of the joint DE-AT-LU bidding zone until 2018-09-30, DE-LU afterwards -- we try the
    period-appropriate zone first and fall back.
  * IT-NO is the IT-North bidding zone; SE (Lulea) is SE1; FR/ES/PL are their national zones.
Coverage guard: every region must cover every slot in-window, or the script fails loudly
(same never-again policy as the EWIF emit).

Run:
  python scripts/fetch_dayahead_prices.py --signals-dir pkg/carbon-aware/data/timealigned_realci2
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# Ordered zone candidates per region (first with data in-window wins).
ZONES = {
    "DE": ["DE-LU", "DE-AT-LU"],
    "FR": ["FR"],
    "ES": ["ES"],
    "IT-NO": ["IT-North"],
    "SE": ["SE1"],
    "PL": ["PL"],
}


def _fetch(bzn: str, start_iso: str, end_iso: str):
    url = f"https://api.energy-charts.info/price?bzn={bzn}&start={start_iso}&end={end_iso}"
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.loads(r.read().decode()), url
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(4 * (attempt + 1))
                continue
            if e.code in (400, 404, 422):
                return None, url  # zone/window unsupported -> try next candidate
            raise
    return None, url


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals-dir", required=True)
    args = ap.parse_args()
    sig = (REPO / args.signals_dir).resolve() if not Path(args.signals_dir).is_absolute() else Path(args.signals_dir)
    fc = json.loads((sig / "forecasts.json").read_text())
    regions = sorted(fc.keys())
    any_zone = fc[regions[0]]["forecast"]
    start = pd.Timestamp(any_zone[0]["datetime"])
    n = len(any_zone)
    hours = pd.date_range(start, periods=n, freq="h", tz="UTC")
    end = (start + pd.Timedelta(hours=n) + pd.Timedelta(days=1)).date().isoformat()
    print(f"window {start} x {n} slots; regions {regions}")

    rows = []
    for reg in regions:
        got = None
        for bzn in ZONES.get(reg, [reg]):
            payload, url = _fetch(bzn, start.date().isoformat(), end)
            if payload and payload.get("price"):
                idx = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
                ser = pd.Series(payload["price"], index=idx, dtype=float) / 1000.0  # EUR/MWh -> EUR/kWh
                ser = ser.resample("h").mean().reindex(hours)
                if ser.notna().sum() >= n:  # full in-window coverage
                    got = (bzn, ser, url)
                    break
                got = got or (bzn, ser, url)  # keep best-effort while trying next zone
        if got is None:
            raise SystemExit(f"[price] no zone returned data for {reg} (tried {ZONES.get(reg)})")
        bzn, ser, url = got
        missing = int(ser.isna().sum())
        if missing:
            raise SystemExit(f"[price] {reg}/{bzn}: {missing}/{n} slots missing in-window -- refusing partial coverage")
        for k, ts in enumerate(hours):
            rows.append(dict(region=reg, slot_index=k, utc_timestamp=ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                             price_eur_per_kwh=round(float(ser.iloc[k]), 6), bzn=bzn,
                             source="Energy-Charts day-ahead price API", source_url=url.split("&start")[0]))
        print(f"  {reg:6s} bzn={bzn:9s} price {ser.min():.4f}-{ser.max():.4f} EUR/kWh")

    out = sig / "price_region_slot.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
