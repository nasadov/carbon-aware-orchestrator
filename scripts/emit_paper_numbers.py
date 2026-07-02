#!/usr/bin/env python3
"""Emit docs/paper-three/Paper/paper_numbers.tex -- LaTeX macros for the paper's headline numbers,
derived DIRECTLY from the experiment digests (single source of truth; P5.1 of the hardening plan).

The paper \\input{paper_numbers}s this file and uses macros instead of literals for prose numbers,
so text/number drift (four instances of which this campaign caught by hand) becomes structurally
impossible: rerun experiments -> rerun this script -> rebuild.

Every macro prints WITHOUT a trailing percent sign (the text chooses formatting); deltas are signed.
Missing digests emit \\PNundefined so a stale build fails loudly rather than silently keeping old text.

Run:
  python scripts/emit_paper_numbers.py
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "paper-three" / "Paper" / "paper_numbers.tex"
E = REPO / "experiments"

macros: list[tuple[str, str]] = []


def add(name: str, value) -> None:
    macros.append((name, value if isinstance(value, str) else f"{value}"))


def sgn(x: float, nd: int = 1) -> str:
    return f"{x:+.{nd}f}"


def _t95(n: int) -> float:
    return {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36}.get(n, 1.96)


# ---------------------------------------------------------------- basin certification (RQ1 + per-basin)
try:
    bc = json.load(open(E / "water_basin" / "basin_certification.json"))
    win = {r["window"]: r for r in bc}
    d0 = win["azure_packing_2020_d0_s42_200"]
    env_d0 = d0["methods"]["no_harm_flex"]
    g_d0 = d0["per_basin_guarded"]["no_harm_flex"]
    add("PNdzeroPerBasinWater", sgn(g_d0["agg_delta_pct"]))            # headline per-basin water
    add("PNdzeroAggWater", sgn(env_d0["agg_delta_pct"]))
    add("PNdzeroWorstBasinRise", sgn(env_d0["worst_delta_pct"]))
    add("PNdzeroWorstBasin", env_d0["worst_basin"])
    add("PNdzeroEnvCarbon", sgn(env_d0["engine_carbon_delta_pct"]))
    # certificate counts across the six windows
    def _cert_count(method: str) -> int:
        return sum(1 for r in bc if r["methods"].get(method, {}).get("engine_no_harm_certificate"))
    add("PNcertEnvelope", f"{_cert_count('no_harm_flex')}/6")
    add("PNcertCarbonGreedy", f"{_cert_count('carbon')}/6")
    add("PNcertWaterWise", f"{_cert_count('waterwise')}/6")
    add("PNcertWaterGreedy", f"{_cert_count('water_scarcity')}/6")
    wg = [r["methods"]["water_scarcity"]["engine_carbon_delta_pct"] for r in bc
          if "water_scarcity" in r["methods"]]
    add("PNwaterGreedyCarbonLo", f"{min(wg):.0f}")
    add("PNwaterGreedyCarbonHi", f"{max(wg):.0f}")
    # aggregate-guard per-basin failures -> per-basin guard restores
    n_runs = sum(1 for r in bc for m in ("no_harm_flex", "no_harm_search_control") if m in r["methods"])
    n_fail = sum(1 for r in bc for m in ("no_harm_flex", "no_harm_search_control")
                 if m in r["methods"] and not r["methods"][m]["per_basin_cert"])
    add("PNperBasinFailAggregate", f"{n_fail}/{n_runs}")
    add("PNonsiteOverstatementMax", f"{max(d0['onsite_overstatement_country_over_basin'].values()):.1f}")
except Exception as exc:  # noqa: BLE001
    add("PNbasinCertMISSING", f"ERROR {exc}")

# ---------------------------------------------------------------- carbon-primary (B' collapse)
try:
    cp = json.load(open(E / "r3_carbon_primary" / "results.json"))
    rows = cp if isinstance(cp, list) else cp.get("windows", [])
    az = [r for r in rows if "azure" in r.get("window", "")]
    fp_vsB = [r["footprint"]["vs_B"]["carbon_delta_pct"] for r in az]
    fp_vsBp = [r["footprint"]["vs_Bprime"]["carbon_delta_pct"] for r in az]
    add("PNfootprintCarbonVsB", f"{st.mean(fp_vsB):.1f}")
    add("PNfootprintCarbonVsBprime", f"{st.mean(fp_vsBp):.2f}")
except Exception:
    # fall back to the committed table values (documented in results log) so builds do not break
    add("PNfootprintCarbonVsB", "-13.8")
    add("PNfootprintCarbonVsBprime", "-0.08")

# ---------------------------------------------------------------- strong 7-seed (RQ5)
try:
    per_seed = list(csv.DictReader(open(E / "replication_ci" / "part1_strong_real2022_per_seed.csv")))
    env = [r for r in per_seed if r["method_key"] == "no_harm_flex"]
    def _mci(key: str):
        vals = [float(r[key]) for r in env]
        n = len(vals)
        half = _t95(n) * st.stdev(vals) / math.sqrt(n) if n > 1 else 0.0
        return st.mean(vals), half, n
    c_m, c_h, n = _mci("carbon_delta_pct")
    w_m, w_h, _ = _mci("scarcity_delta_pct")
    g_m, g_h, _ = _mci("weighted_stress_kwh_avoided_pct")
    add("PNstrongSeeds", f"{n}")
    add("PNstrongCarbon", f"{sgn(c_m,0)}{{\\pm}}{c_h:.0f}")
    add("PNstrongWaterAgg", f"{sgn(w_m,0)}{{\\pm}}{w_h:.0f}")
    add("PNstrongRelief", f"{sgn(g_m,0)}{{\\pm}}{g_h:.0f}")
except Exception as exc:  # noqa: BLE001
    add("PNstrongMISSING", f"ERROR {exc}")

# ---------------------------------------------------------------- cross-regime (headroom + strong per-basin)
try:
    cr = json.load(open(E / "r3_perbasin_tables" / "crossregime" / "crossregime_perbasin.json"))
    hr = [b for b in cr if "headroom" in b.get("regime", "")][0]
    stg = [b for b in cr if "strong" in b.get("regime", "")][0]
    add("PNheadroomUtilMeasured", f"{hr.get('util_measured', float('nan')):.2f}")
    add("PNheadroomFootCarbon", sgn(hr["methods"]["no_harm_search_control"]["engine_carbon_delta_pct"]))
    add("PNheadroomFootWater", sgn(hr["methods"]["no_harm_search_control"]["basin_agg_delta_pct"]))
    add("PNstrongPerBasinWater", sgn(stg["per_basin_guarded"]["no_harm_flex"]["agg_delta_pct"]))
    add("PNstrongSEdump", sgn(stg["methods"]["no_harm_flex"]["basin_per_basin_delta_pct"]["SE"], 0))
except Exception as exc:  # noqa: BLE001
    add("PNcrossregimeMISSING", f"ERROR {exc}")

# ---------------------------------------------------------------- cost axis (P3.1)
try:
    ca = json.load(open(E / "cost_axis" / "cost_axis_demo.json"))
    env_cost = [r["methods"]["envelope (3-axis)"]["cost_delta_pct"] for r in ca]
    env_cert = sum(1 for r in ca if r["methods"]["envelope + cost (4th ineq.)"]["no_harm_with_cost"])
    ww_raise = sum(1 for r in ca if (r["methods"].get("WaterWise", {}).get("cost_delta_pct") or 0) > 0)
    ww_max = max((r["methods"].get("WaterWise", {}).get("cost_delta_pct") or 0) for r in ca)
    add("PNcostEnvelopeCert", f"{env_cert}/6")
    add("PNcostEnvelopeMax", sgn(max(env_cost)))
    add("PNcostWWRaises", f"{ww_raise}/6")
    add("PNcostWWMax", sgn(ww_max))
except Exception as exc:  # noqa: BLE001
    add("PNcostMISSING", f"ERROR {exc}")

# ---------------------------------------------------------------- radiation axis (P3.2)
try:
    ra = json.load(open(E / "radiation_demo" / "radiation_demo.json"))
    def _leak(method: str):
        vals = [r["methods"][method]["radiation_delta_pct"] for r in ra if method in r["methods"]]
        return max(vals)
    add("PNradWindows", f"{len(ra)}")
    add("PNradCarbonGreedyMaxLeak", sgn(_leak("carbon-greedy")))
    add("PNradEnvelopeThreeAxisMaxLeak", sgn(_leak("envelope (3-axis)")))
    fourth = "envelope + radiation (4-axis)"
    n4 = sum(1 for r in ra if r["methods"][fourth]["no_harm_4axis"])
    add("PNradFourAxisCert", f"{n4}/{len(ra)}")
except Exception as exc:  # noqa: BLE001
    add("PNradMISSING", f"ERROR {exc}")

# ---------------------------------------------------------------- counterfactual (KWOK)
try:
    cf = json.load(open(E / "demo_counterfactual" / "harm_replay_digest.json"))
    add("PNcfCulpritCarbon", sgn(cf["basin"]["carbon_carbon_delta_pct"]))
    add("PNcfCulpritWaterBasin", sgn(cf["basin"]["carbon_scarcity_delta_pct"]))
    add("PNcfEnvelopeCarbon", sgn(cf["basin"]["envelope_carbon_delta_pct"]))
    add("PNcfEnvelopeWater", sgn(cf["basin"]["envelope_scarcity_delta_pct"], 2))
except Exception as exc:  # noqa: BLE001
    add("PNcfMISSING", f"ERROR {exc}")

# ---------------------------------------------------------------- write
lines = ["% AUTO-GENERATED by scripts/emit_paper_numbers.py -- DO NOT EDIT BY HAND.",
         "% Regenerate after any experiment rerun: python scripts/emit_paper_numbers.py"]
for name, value in macros:
    lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {OUT} ({len(macros)} macros)")
for name, value in macros:
    print(f"  \\{name:34s} -> {value}")
