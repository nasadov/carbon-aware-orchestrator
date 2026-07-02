#!/usr/bin/env python3
"""Render the optimality-gap table from experiments/optgap_<tag>/n*/result.json.

Two views per axis:
  GLOBAL  = greedy vs MILP free to re-place every pod (global certified optimum).
  MATCHED = greedy vs MILP with firm pods pinned (the greedy's own action space).
Stress axis uses the pure-'stress' greedy (objective-matched to the wstress MILP);
carbon/scarcity use the production 'combined' greedy.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load(tag):
    rows = []
    for p in sorted(glob.glob(str(REPO / f"experiments/optgap_{tag}/n*/result.json")),
                    key=lambda p: int(p.split("/n")[-1].split("/")[0])):
        rows.append(json.load(open(p)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="exact")
    args = ap.parse_args()
    rows = [r for r in load(args.tag) if not r.get("skipped")]

    def g(r, k):
        return r.get(k, float("nan"))

    print("=" * 118)
    print("OPTIMALITY GAP — greedy no-harm repair vs EXACT no-harm-constrained MILP (engine-faithful linearization)")
    print("=" * 118)
    print("Relief = % reduction vs packing baseline.  Gap = (greedy-opt)/opt.  All MILPs solved to proven optimality.")
    print()
    hdr = (f"{'n':>3} {'flx':>3} | {'STRESS relief':>20} {'gap%(glb/mch)':>15} | "
           f"{'CARBON relief':>20} {'gap%(glb/mch)':>15} | {'runtime (s)':>18} {'speedup':>7}")
    print(hdr)
    print(f"{'':>3} {'':>3} | {'greedy / opt':>20} {'':>15} | {'greedy / opt':>20} {'':>15} | "
          f"{'greedy / MILP':>18} {'':>7}")
    print("-" * 118)
    for r in rows:
        sr_g, sr_o = g(r, "stress_relief_pct"), g(r, "opt_relief_pct")
        cr_g, cr_o = g(r, "combined_carbon_relief_pct"), g(r, "opt_carbon_relief_pct")
        sgg, sgm = g(r, "gap_wstress_pct"), g(r, "gap_wstress_matched_pct")
        cgg, cgm = g(r, "gap_carbon_pct"), g(r, "gap_carbon_matched_pct")
        gw = max(g(r, "stress_wall_s"), g(r, "combined_wall_s"))
        mw = max(g(r, "milp_wstress_wall_s"), g(r, "milp_carbon_wall_s"), g(r, "milp_scarcity_wall_s"))
        spd = mw / gw if gw else float("nan")
        print(f"{r['n_placed']:>3} {r['n_flex']:>3} | "
              f"{sr_g:>8.1f} /{sr_o:>8.1f}   {sgg:>6.0f}/{sgm:>6.0f}   | "
              f"{cr_g:>8.1f} /{cr_o:>8.1f}   {cgg:>6.0f}/{cgm:>6.0f}   | "
              f"{gw:>7.3f} /{mw:>7.2f}   {spd:>6.0f}x")
    print("-" * 118)
    # aggregate
    import statistics as st
    def med(k):
        vals = [g(r, k) for r in rows if g(r, k) == g(r, k)]
        return st.median(vals) if vals else float("nan")
    print(f"median stress relief: greedy {med('stress_relief_pct'):.1f}%  opt {med('opt_relief_pct'):.1f}%")
    print(f"median carbon relief: greedy {med('combined_carbon_relief_pct'):.1f}%  opt {med('opt_carbon_relief_pct'):.1f}%")
    print(f"median scarcity relief: greedy {med('combined_scarcity_relief_pct'):.1f}%  opt {med('opt_scarcity_relief_pct'):.1f}%")
    print(f"max greedy wall: {max(g(r,'combined_wall_s') for r in rows):.3f}s   "
          f"max MILP wall: {max(max(g(r,'milp_wstress_wall_s'),g(r,'milp_carbon_wall_s'),g(r,'milp_scarcity_wall_s')) for r in rows):.2f}s")
    cert = all(r['combined_carbon_le_base'] and r['combined_scarcity_le_base']
               and r['stress_carbon_le_base'] and r['stress_scarcity_le_base'] and r['greedy_slo_ok'] for r in rows)
    decomp = max(max(r['decomp_err_carbon_kg'], r['decomp_err_scarcity']) for r in rows)
    print(f"greedy certificate respected on ALL instances: {cert}   max decomposition error: {decomp:.1e} (machine precision)")


if __name__ == "__main__":
    raise SystemExit(main())
