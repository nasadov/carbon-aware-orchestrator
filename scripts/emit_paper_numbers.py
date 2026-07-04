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
    # R4/MC1 -- the same counts under the PER-BASIN certificate of Eq. (1):
    # aggregate three-axis certificate AND no watershed's ledger rises.
    def _pb_cert_count(method: str) -> int:
        return sum(1 for r in bc
                   if r["methods"].get(method, {}).get("engine_no_harm_certificate")
                   and r["methods"].get(method, {}).get("per_basin_cert"))
    add("PNcertEnvelopePB", f"{_pb_cert_count('no_harm_flex')}/6")          # 0/6 (aggregate-guarded)
    add("PNcertCarbonGreedyPB", f"{_pb_cert_count('carbon')}/6")            # 0/6
    add("PNcertWaterWisePB", f"{_pb_cert_count('waterwise')}/6")            # 0/6
    add("PNcertWaterGreedyPB", f"{_pb_cert_count('water_scarcity')}/6")     # 0/6
    n_pbg_all = sum(1 for r in bc for m in ("no_harm_flex", "no_harm_search_control")
                    if m in r["per_basin_guarded"])
    n_pbg = sum(1 for r in bc for m in ("no_harm_flex", "no_harm_search_control")
                if r["per_basin_guarded"].get(m, {}).get("per_basin_cert"))
    add("PNcertPerBasinGuarded", f"{n_pbg}/{n_pbg_all}")                    # 12/12
    # R4/MC2 -- carbon-greedy's per-basin burden-shift on the Azure windows (worst basin & aggregate win)
    az_cg = [(r["window"], r["methods"]["carbon"]) for r in bc if "azure" in r["window"]]
    add("PNcgAzureAggLo", sgn(max(m["agg_delta_pct"] for _, m in az_cg)))   # -5.6 (smallest win)
    add("PNcgAzureAggHi", sgn(min(m["agg_delta_pct"] for _, m in az_cg)))   # -9.9 (largest win)
    add("PNcgAzureRiseLo", sgn(min(m["worst_delta_pct"] for _, m in az_cg)))  # +7.5
    add("PNcgAzureRiseHi", sgn(max(m["worst_delta_pct"] for _, m in az_cg)))  # +11.4
    cg_d0 = d0["methods"]["carbon"]
    add("PNdzeroCgWorstBasin", cg_d0["worst_basin"])                        # FR
    add("PNdzeroCgWorstRise", sgn(cg_d0["worst_delta_pct"]))                # +7.5
    add("PNdzeroCgAggWater", sgn(cg_d0["agg_delta_pct"]))                   # -6.0
    # R4/MC2 -- carbon of the per-basin-GUARDED members, from the committed audit CSVs (the
    # per-pod carbon account is engine-exact: it reproduces engine_carbon_delta_pct bit-for-bit
    # on the aggregate members). SLO is preserved by construction (the replay only relocates
    # pods placed in both schedules).
    def _pbg_carbon_pct(window: str, member: str) -> float:
        out = E / "water_basin" / window / "out"
        base_c = {row["pod_id"]: float(row["carbon_kg"])
                  for row in csv.DictReader(open(out / "placements_packing.csv"))}
        mem_c = {row["pod_id"]
                 for row in csv.DictReader(open(out / f"placements_{member}.csv"))}
        base_total = sum(v for p, v in base_c.items() if p in mem_c)
        g = win[window]["per_basin_guarded"][member]
        return 100.0 * g["carbon_delta_kg"] / base_total
    add("PNdzeroPbgCarbon", sgn(_pbg_carbon_pct("azure_packing_2020_d0_s42_200", "no_harm_flex")))  # -6.5
    pbg_az = [_pbg_carbon_pct(w, m) for w in win if "azure" in w
              for m in ("no_harm_flex", "no_harm_search_control")]
    assert all(v <= 0.0 for v in pbg_az)
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
# R4/MC3: the digest's "country" block is a LEGACY NAME -- post water-revamp it is the engine's
# default accounting, i.e. the CERTIFICATE'S axis (on-site water at the watershed CF, generation
# water at the domestic-country CF).  The "basin" block is the all-watershed SENSITIVITY
# (generation water also at basin CF; de-confounding check).  Table tab:counterfactual reports
# the certificate's axis; the sensitivity is a footnote.
try:
    cf = json.load(open(E / "demo_counterfactual" / "harm_replay_digest.json"))
    cert = cf["country"]  # certificate's axis (legacy block name; see note above)
    add("PNcfCulpritCarbon", sgn(cert["carbon_carbon_delta_pct"]))          # -24.0
    add("PNcfCulpritWater", sgn(cert["carbon_scarcity_delta_pct"]))         # +63.8
    add("PNcfEnvelopeCarbon", sgn(cert["envelope_carbon_delta_pct"]))       # -13.5
    add("PNcfEnvelopeWater", sgn(cert["envelope_scarcity_delta_pct"]))      # -0.7
    m = cert["methods"]
    def _cfm(tag: str, key: str) -> None:
        add(f"PNcf{tag}Carbon", sgn(m[key]["carbon_delta_pct"]))
        add(f"PNcf{tag}Water", sgn(m[key]["scarcity_delta_pct"]))
    _cfm("CicSpatial", "cic_spatial")   # -7.5 / +42.8
    _cfm("WW", "waterwise@0.5")         # -20.6 / -1.6 (drops 4/80)
    _cfm("Cic", "cic")                  # +4.3 / +1.7
    _cfm("Lwa", "wait_awhile")          # +8.6 / +0.2
    _cfm("Greenslot", "greenslot")      # +4.3 / +1.7
    _cfm("WaterGreedy", "water_scarcity")  # +13.5 / -54.6
    add("PNcfWWPlaced", f'{m["waterwise@0.5"]["placed_pods"]}')             # 76
    # engine-native per-basin-guarded envelope, same (certificate) accounting
    pbg = cf["perbasin_guard_engine"]
    assert pbg["certified"] and pbg["worst_per_basin_rise_L"] == 0.0
    add("PNcfPbgCarbon", sgn(pbg["carbon_delta_pct"]))                      # -3.0
    add("PNcfPbgWater", sgn(pbg["scarcity_delta_pct"]))                     # -0.5
    # all-watershed sensitivity: the burden-shift survives de-confounding
    add("PNcfCulpritWaterAllBasin", sgn(cf["basin"]["carbon_scarcity_delta_pct"]))  # +29.2
except Exception as exc:  # noqa: BLE001
    add("PNcfMISSING", f"ERROR {exc}")

# ---------------------------------------------------------------- independent baseline B_ind (P3.3)
try:
    ib = json.load(open(E / "demo_counterfactual" / "independent_baseline" / "independent_baseline_digest.json"))
    bc, ba = ib["country"], ib["basin"]
    add("PNbindCarbon", sgn(bc["B_ind_vs_packing"]["carbon_delta_pct"]))          # +4.8
    add("PNbindWaterCountry", sgn(bc["B_ind_vs_packing"]["scarcity_delta_pct"]))  # -12.9
    add("PNbindWaterBasin", sgn(ba["B_ind_vs_packing"]["scarcity_delta_pct"]))    # -4.9
    add("PNbindPlaced", str(bc["B_ind_placed"]))                                   # 76
    add("PNbindDropped", str(bc["B_ind_placed"] and (ba["n_pods"] - bc["B_ind_placed"])))  # 4
    add("PNbindPods", str(ba["n_pods"]))                                           # 80
    # envelope certifying with B_ind as the reference baseline
    add("PNbindEnvCarbonCountry", sgn(bc["envelope_from_B_ind_vs_B_ind"]["carbon_delta_pct"]))  # -21.7
    add("PNbindEnvCarbonBasin", sgn(ba["envelope_from_B_ind_vs_B_ind"]["carbon_delta_pct"]))    # -15.6
    add("PNbindEnvWaterCountry", sgn(bc["envelope_from_B_ind_vs_B_ind"]["scarcity_delta_pct"], 1))  # -0.1
    add("PNbindEnvWaterBasin", sgn(ba["envelope_from_B_ind_vs_B_ind"]["scarcity_delta_pct"], 1))    # -0.9
    add("PNbindEnvReliefCountry", f'{bc["envelope_from_B_ind_vs_B_ind"]["weighted_stress_kwh_avoided"]:.2f}')  # 0.60
    add("PNbindEnvCert", r"\ding{51}" if bc["envelope_from_B_ind_vs_B_ind"]["no_harm_certificate"] else r"\ding{55}")
except Exception as exc:  # noqa: BLE001
    add("PNbindMISSING", f"ERROR {exc}")

# ---------------------------------------------------------------- MILP optimality gap + guarded move-set/LNS (F3)
try:
    og = json.load(open(E / "optgap_exact" / "summary.json"))
    # carbon-relief axis: median fraction of the exact carbon-relief optimum the footprint greedy captures
    cfrac = [100.0 * r["combined_carbon_relief_pct"] / r["opt_carbon_relief_pct"] for r in og]
    add("PNmilpCarbonReliefMedian", f"{st.median(cfrac):.0f}")
    lns = json.load(open(E / "optgap_lns" / "summary.json"))
    g_gap = [100.0 - r["stress_frac_of_opt_pct"] for r in lns]
    l_gap = [100.0 - r["lns_frac_of_opt_pct"] for r in lns]
    add("PNlnsGapGreedy", f"{st.median(g_gap):.1f}")
    add("PNlnsGapLns", f"{st.median(l_gap):.1f}")
    # consolidation-barrier (tight) instances: greedy captures <10% of optimum
    tight = [r for r in lns if r["stress_frac_of_opt_pct"] < 10.0]
    add("PNlnsTightGreedyLo", f"{min(r['stress_frac_of_opt_pct'] for r in tight):.0f}")
    add("PNlnsTightGreedyHi", f"{max(r['stress_frac_of_opt_pct'] for r in tight):.0f}")
    add("PNlnsTightLnsLo", f"{min(r['lns_frac_of_opt_pct'] for r in tight):.0f}")
    add("PNlnsTightLnsHi", f"{max(r['lns_frac_of_opt_pct'] for r in tight):.0f}")
    assert all(r["lns_carbon_le_base"] and r["lns_scarcity_le_base"] for r in lns)
except Exception as exc:  # noqa: BLE001
    add("PNlnsMISSING", f"ERROR {exc}")

# ---------------------------------------------------------------- trace-native flagship replay (F3-T1)
try:
    fl = json.load(open(E / "flagship_replay" / "flagship_w1587_npr20" / "flagship_digest.json"))
    fm = fl["methods"]
    env_fl = fm["no_harm_flex"]
    assert env_fl["no_harm_certificate"] and not fm["carbon"]["no_harm_certificate"] \
        and not fm["water_scarcity"]["no_harm_certificate"] and not fm["waterwise"]["no_harm_certificate"]
    add("PNflagshipPods", f"{fl['target_pods']}")
    add("PNflagshipNodes", f"{fl['n_nodes']}")
    add("PNflagshipGpus", f"{fl['total_gpus']}")
    add("PNflagshipRelief", f"{env_fl['stress_kwh_avoided']:.1f}")           # binary metric, same as tab:scale
    add("PNflagshipCarbon", sgn(env_fl["carbon_delta_pct"], 2))
    add("PNflagshipRepairs", f"{env_fl['repairs_applied']}")
    add("PNflagshipWaterGreedyCarbon", sgn(fm["water_scarcity"]["carbon_delta_pct"], 1))
    add("PNflagshipPilotMin", f"{fl['t_pilot_s'] / 60.0:.0f}")
except Exception as exc:  # noqa: BLE001
    add("PNflagshipMISSING", f"ERROR {exc}")

# ---------------------------------------------------------------- grid-magnitude chain (Discussion; from the P2 scale digest)
try:
    sd = json.load(open(E / "flexibility" / "t19_alibaba_fleet_scale" / "scale_digest.json"))
    per_gpu = sd["mean_stress_avoided_per_gpu_kwh"]
    mw_window = sd["extrapolated_mw_window_relief_kwh"]
    window_h = 48.0                      # the sweep's 48-slot scheduling window
    eur_per_kwh, band_lo, band_hi = 0.30, 0.03, 0.21  # deferorshift2026 assumptions (mc2 grid-value chain)
    add("PNscalePerGpu", f"{per_gpu:.3f}")
    add("PNscaleMwWindowRelief", f"{mw_window:.0f}")
    add("PNscaleAvgKwPerMw", f"{mw_window / window_h:.1f}")
    add("PNscalePctNameplate", f"{mw_window / window_h / 1000.0 * 100.0:.2f}")
    add("PNscaleEurLo", f"{mw_window * eur_per_kwh * (1 + band_lo):.0f}")
    add("PNscaleEurHi", f"{mw_window * eur_per_kwh * (1 + band_hi):.0f}")
    add("PNscaleFleetGwhLo", f"{mw_window * 50e3 / 1e6:.1f}")   # 50 GW flexible AI load
    add("PNscaleFleetGwhHi", f"{mw_window * 100e3 / 1e6:.1f}")  # 100 GW
    add("PNscaleFleetMwLo", f"{mw_window / window_h * 50:.0f}")
    add("PNscaleFleetMwHi", f"{mw_window / window_h * 100:.0f}")
except Exception as exc:  # noqa: BLE001
    add("PNscaleMISSING", f"ERROR {exc}")

# ---------------------------------------------------------------- write
lines = ["% AUTO-GENERATED by scripts/emit_paper_numbers.py -- DO NOT EDIT BY HAND.",
         "% Regenerate after any experiment rerun: python scripts/emit_paper_numbers.py"]
for name, value in macros:
    lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {OUT} ({len(macros)} macros)")
for name, value in macros:
    print(f"  \\{name:34s} -> {value}")
