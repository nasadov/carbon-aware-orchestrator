#!/usr/bin/env python3
"""W_b decomposition + on-site-only burden-shift recheck (reviewer question: what share of the
charged per-basin scarcity is generation-side water at the country CF, and does the per-basin
story survive at pure watershed resolution?).

Pure bookkeeping over committed artifacts: the per-pod placement CSVs (region, direct/indirect
liters, carbon) and the CF tables in experiments/water_basin/basin_certification.json. The
per-basin-guarded member is re-derived with the identical post-hoc replay used by
scripts/run_water_basin_certification.py (_per_basin_guarded_member: adopt the aggregate member's
per-pod moves in carbon-savings order iff every basin's combined W_b stays <= B), asserting the
replayed aggregate deltas match the committed digest before reporting anything.

Reported per window:
  * generation-side share of charged scarcity (fleet and per region), packing baseline;
  * on-site-only (direct x basin-CF) worst-basin rise vs B for carbon-greedy, the
    aggregate-guarded member, and the per-basin-guarded member;
  * per-basin certificate failure margins (worst basin rise %) for the tie analysis.
"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
E = REPO / "experiments" / "water_basin"


def load_placements(window: str, method: str):
    out = {}
    for row in csv.DictReader(open(E / window / "out" / f"placements_{method}.csv")):
        out[row["pod_id"]] = dict(basin=row["region"], d=float(row["direct_water_l"]),
                                  i=float(row["indirect_water_l"]), c=float(row["carbon_kg"]))
    return out


def region_scarcity(window: str, method: str, cfb, cfc):
    out = defaultdict(lambda: [0.0, 0.0])
    tot_check = 0.0
    for row in csv.DictReader(open(E / window / "out" / f"placements_{method}.csv")):
        reg = row["region"]
        out[reg][0] += float(row["direct_water_l"]) * cfb[reg]
        out[reg][1] += float(row["indirect_water_l"]) * cfc[reg]
        tot_check += float(row["scarcity_water"])
    tot = sum(d + i for d, i in out.values())
    assert abs(tot - tot_check) / max(tot_check, 1e-9) < 0.02, (window, method)
    return out


def pbg_replay(window: str, rec: dict):
    """Replicate _per_basin_guarded_member for the relief member; return per-basin direct/indirect."""
    cfb, cfc, regions = rec["cf_basin"], rec["cf_country"], rec["regions"]
    base, member = load_placements(window, "packing"), load_placements(window, "no_harm_flex")
    common = [p for p in base if p in member]
    bd = {b: 0.0 for b in regions}; bi = {b: 0.0 for b in regions}
    for p in common:
        bd[base[p]["basin"]] += base[p]["d"]; bi[base[p]["basin"]] += base[p]["i"]
    bscar = {b: bd[b] * cfb[b] + bi[b] * cfc[b] for b in regions}
    moves = [p for p in common if member[p]["basin"] != base[p]["basin"]
             or abs(member[p]["d"] - base[p]["d"]) > 1e-12
             or abs(member[p]["i"] - base[p]["i"]) > 1e-12
             or abs(member[p]["c"] - base[p]["c"]) > 1e-12]
    moves.sort(key=lambda p: member[p]["c"] - base[p]["c"])
    cd, ci = dict(bd), dict(bi)
    for p in moves:
        sb, db_ = base[p]["basin"], member[p]["basin"]
        td, ti = dict(cd), dict(ci)
        td[sb] -= base[p]["d"]; ti[sb] -= base[p]["i"]
        td[db_] += member[p]["d"]; ti[db_] += member[p]["i"]
        if all(td[b] * cfb[b] + ti[b] * cfc[b] <= bscar[b] + 1e-9 for b in regions):
            cd, ci = td, ti
    tot_b = sum(bscar.values())
    agg_pct = 100 * (sum(cd[b] * cfb[b] + ci[b] * cfc[b] for b in regions) - tot_b) / tot_b
    committed = rec["per_basin_guarded"]["no_harm_flex"]["agg_delta_pct"]
    assert abs(agg_pct - committed) < 0.02, (window, agg_pct, committed)
    return bd, cd


def main() -> int:
    bc = {r["window"]: r for r in json.load(open(E / "basin_certification.json"))}
    gen_shares = []
    print("window                        gen-share%   on-site worst-basin rise% (c-grd / agg-member / pbg-member)")
    for w, rec in bc.items():
        cfb, cfc = rec["cf_basin"], rec["cf_country"]
        base_rs = region_scarcity(w, "packing", cfb, cfc)
        tot_d = sum(v[0] for v in base_rs.values()); tot_i = sum(v[1] for v in base_rs.values())
        share = 100 * tot_i / (tot_d + tot_i)
        gen_shares += [100 * v[1] / (v[0] + v[1]) for v in base_rs.values() if v[0] + v[1] > 0]
        def onsite_worst(method):
            cur = region_scarcity(w, method, cfb, cfc)
            return max(100 * (cur[b][0] - base_rs[b][0]) / base_rs[b][0]
                       for b in base_rs if base_rs[b][0] > 1e-9)
        bd, cd = pbg_replay(w, rec)
        pbg_worst = max(100 * (cd[b] - bd[b]) / bd[b] for b in bd if bd[b] > 1e-9)
        print(f"{w:28}  {share:6.1f}      {onsite_worst('carbon'):+7.2f} / "
              f"{onsite_worst('no_harm_flex'):+7.2f} / {pbg_worst:+7.2f}")
    print(f"\nper-region generation-share range: {min(gen_shares):.0f}--{max(gen_shares):.0f}%")
    print("\nper-basin failure margins (worst basin rise %, combined W_b) for the tie analysis:")
    for w, rec in bc.items():
        row = [f"{m}:{rec['methods'][m]['worst_delta_pct']:+.2f}" for m in
               ("carbon", "no_harm_flex", "no_harm_search_control") if m in rec["methods"]]
        print(f"  {w:28} {'  '.join(row)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
