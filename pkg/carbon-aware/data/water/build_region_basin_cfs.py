#!/usr/bin/env python3
"""Build BASIN-scale AWARE 2.0 water-scarcity CFs for each DC region (vs the country aggregation).

Motivation. The repo's scarcity layer uses AWARE 2.0 *country*-level non-agri CFs
(`aware20_country_nonagri_factors.csv`). Country averages mask large intra-country variation: a DC in
a wet basin of an otherwise arid country (or vice versa) gets the wrong scarcity weight. AWARE's
native unit is the watershed, so we map each region's metro DC proxy (lat/lon from
`region_weather_sites.csv`) to the AWARE basin polygon that contains it (point-in-polygon, EPSG:4326)
and read that basin's monthly + annual non-agri CFs directly.

Output `aware20_basin_nonagri_factors.csv` mirrors the country file's columns but is keyed by region,
so the scarcity loader can consume it as a basin-resolution override for the country CFs. The build
also prints the country-vs-basin contrast and the cross-region scarcity ordering (which flips), which
is the evidence behind the "country-level scarcity can mislead the certificate" sensitivity result.

Source: AWARE 2.0 native CFs (geospatial), WULCA, Zenodo record 16332127. Run rarely (build step);
the scheduler consumes only the generated CSV.

Usage:
    python build_region_basin_cfs.py [--gpkg /path/to/AWARE20_Native_CFs_geospatial.gpkg]
Requires: geopandas, shapely (a geospatial stack). Network only if --gpkg is not supplied/cached.
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITES_CSV = HERE / "region_weather_sites.csv"
COUNTRY_CSV = HERE / "aware20_country_nonagri_factors.csv"
OUT_CSV = HERE / "aware20_basin_nonagri_factors.csv"
GPKG_URL = ("https://zenodo.org/records/16332127/files/"
            "AWARE20_Native_CFs_geospatial.gpkg?download=1")
SOURCE_DATASET = "AWARE2.0 native CFs (geospatial), WULCA"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# region -> ISO country code, for the country-vs-basin contrast print and provenance.
REGION_TO_CC = {"DE": "DE", "FR": "FR", "ES": "ES", "IT-NO": "IT", "SE": "SE", "PL": "PL"}


def _load_sites():
    sites = []
    with SITES_CSV.open() as fh:
        for row in csv.DictReader(fh):
            sites.append((row["region"], row.get("site_proxy_name", ""),
                          float(row["latitude"]), float(row["longitude"])))
    return sites


def _resolve_gpkg(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    cache = Path(tempfile.gettempdir()) / "aware" / "native.gpkg"
    if cache.exists():
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading AWARE native gpkg -> {cache} ...", flush=True)
    urllib.request.urlretrieve(GPKG_URL, cache)
    return cache


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpkg", default=None, help="AWARE native geopackage (else download/cache)")
    args = ap.parse_args()
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        raise SystemExit("needs geopandas+shapely: pip install geopandas shapely pyogrio")

    sites = _load_sites()
    gpkg = _resolve_gpkg(args.gpkg)
    basins = gpd.read_file(gpkg)  # CF_Jan..CF_Dec, CF_annual_{unspecified,agri,nonagri}, geometry
    pts = gpd.GeoDataFrame(
        {"region": [s[0] for s in sites], "site": [s[1] for s in sites],
         "lat": [s[2] for s in sites], "lon": [s[3] for s in sites]},
        geometry=[Point(s[3], s[2]) for s in sites], crs="EPSG:4326")
    joined = gpd.sjoin(pts, basins.to_crs("EPSG:4326"), predicate="within", how="left")

    # ---- write the region-keyed basin CF table (mirrors the country file's columns) ----
    fields = (["region", "country_code", "site", "latitude", "longitude", "annual_cf"]
              + [f"{m.lower()}_cf" for m in MONTHS] + ["source_dataset", "source_url"])
    rows = []
    for _, r in joined.iterrows():
        if r.get("CF_annual_nonagri") is None or (isinstance(r.get("CF_annual_nonagri"), float) and r["CF_annual_nonagri"] != r["CF_annual_nonagri"]):
            raise SystemExit(f"region {r['region']} ({r['lat']},{r['lon']}) fell outside all AWARE basins")
        rec = {"region": r["region"], "country_code": REGION_TO_CC.get(r["region"], ""),
               "site": r["site"], "latitude": r["lat"], "longitude": r["lon"],
               "annual_cf": round(float(r["CF_annual_nonagri"]), 4),
               "source_dataset": SOURCE_DATASET, "source_url": GPKG_URL.split("?")[0]}
        for m in MONTHS:
            rec[f"{m.lower()}_cf"] = round(float(r[f"CF_{m}"]), 4)
        rows.append(rec)
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} region basin CFs -> {OUT_CSV.name}")

    # ---- transparency: country-vs-basin contrast + ordering flip ----
    cc = {}
    with COUNTRY_CSV.open() as fh:
        for row in csv.DictReader(fh):
            cc[row["country_code"]] = row
    print(f"\n{'region':7} {'basin_ann':>9} {'ctry_ann':>9} {'ratio':>6} | {'basin_Aug':>9} {'ctry_Aug':>8}")
    contrast = []
    for rec in rows:
        c = cc[rec["country_code"]]
        b_ann, c_ann = rec["annual_cf"], float(c["annual_cf"])
        b_aug, c_aug = rec["aug_cf"], float(c["aug_cf"])
        contrast.append((rec["region"], b_ann, c_ann, b_aug, c_aug))
        print(f"{rec['region']:7} {b_ann:9.1f} {c_ann:9.1f} {b_ann/c_ann:6.2f} | {b_aug:9.1f} {c_aug:8.1f}")
    print("\ncross-region scarcity ordering (most->least):")
    print("  country annual:", [x[0] for x in sorted(contrast, key=lambda x: -x[2])])
    print("  basin   annual:", [x[0] for x in sorted(contrast, key=lambda x: -x[1])])
    print("  country Aug:   ", [x[0] for x in sorted(contrast, key=lambda x: -x[4])])
    print("  basin   Aug:   ", [x[0] for x in sorted(contrast, key=lambda x: -x[3])])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
