#!/usr/bin/env python3
"""Build per-region ionising-radiation characterization factors for grid electricity.

radiation_region.csv[region] = sum_tech  share[region][tech] x factor[tech]

mirrors the EWIF (indirect-water) pipeline: a per-region generation mix crossed with
per-technology EF 3.1 / ecoinvent ionising-radiation factors (anchored to the EDF French
nuclear LCA, 0.681 kBq U235eq/kWh; EPJ-N 2024, ISO 14040/44). Ionising radiation is
nuclear-dominated, so the per-region factor tracks nuclear share. Emits low/base/high
scenarios for sensitivity. Region-resolved (mix-average) -- consistent with how AWARE
water scarcity is applied at basin resolution; the engine also accepts a per-slot table.

Usage:  python build_region_radiation_reference.py
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
FACTORS = HERE / "radiation_technology_factors.csv"
MIX = HERE / "region_generation_mix.csv"
OUT = HERE / "radiation_region.csv"
SCEN = ("low", "base", "high")
TECHS = ["nuclear","coal","gas","oil","biomass","hydro","wind","solar","geothermal","other"]


def load_factors():
    f = {s: {} for s in SCEN}
    inherit = set()
    with FACTORS.open() as fh:
        for row in csv.DictReader(fh):
            t = row["source_type"].strip()
            if row.get("source_id","").startswith("inherit") or not row.get("ionising_radiation_kbq_u235eq_per_kwh_base"):
                inherit.add(t); continue
            for s in SCEN:
                f[s][t] = float(row[f"ionising_radiation_kbq_u235eq_per_kwh_{s}"])
    return f, inherit


def main():
    factors, inherit = load_factors()
    rows_out = []
    with MIX.open() as fh:
        for m in csv.DictReader(fh):
            region = m["region"].strip()
            shares = {t: float(m.get(t, 0.0) or 0.0) for t in TECHS}
            # non-nuclear mix average per scenario (for 'other'/'unknown' inherit rule)
            out = {"region": region, "nuclear_share": shares.get("nuclear", 0.0)}
            for s in SCEN:
                non_nuc = [(shares[t], factors[s][t]) for t in TECHS
                           if t not in inherit and t != "nuclear" and shares[t] > 0]
                nn_avg = (sum(sh*fa for sh, fa in non_nuc) / sum(sh for sh, fa in non_nuc)) if non_nuc else 0.0
                total = 0.0
                for t in TECHS:
                    sh = shares[t]
                    if sh <= 0:
                        continue
                    fa = factors[s].get(t, nn_avg if t in inherit else 0.0)
                    total += sh * fa
                out[f"ionising_radiation_kbq_u235eq_per_kwh_{s}"] = round(total, 6)
            rows_out.append(out)

    fields = ["region","nuclear_share"] + [f"ionising_radiation_kbq_u235eq_per_kwh_{s}" for s in SCEN]
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"wrote {OUT}")
    for r in rows_out:
        print(f"  {r['region']:>6}  nuc={r['nuclear_share']:.2f}  "
              f"low={r['ionising_radiation_kbq_u235eq_per_kwh_low']:.4f}  "
              f"base={r['ionising_radiation_kbq_u235eq_per_kwh_base']:.4f}  "
              f"high={r['ionising_radiation_kbq_u235eq_per_kwh_high']:.4f}")


if __name__ == "__main__":
    main()
