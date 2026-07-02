#!/usr/bin/env python3
"""MC4 — Multi-fleet rebound / coincidence simulation for the No-Harm Flexibility Envelope.

Reviewer concern MC4 (TSUSC R2): the envelope is a *single* price-taking fleet that follows its
OWN observable residual-load signal. If MANY flexible loads do the same, they coincide into the same
low-residual hours/sites and create a NEW peak — the textbook demand-response rebound / herding /
coincidence effect. The paper's model captures no multi-fleet interaction or price/quantity feedback.

This standalone, analytical simulation answers Question 6 of the referee report:
    "At what flexible-load penetration does residual-load-following stop reducing
     and start increasing the aggregate peak?"

It is deliberately simple (no scheduler engine, no carbon/water guards) — its only job is to expose
the *aggregate residual-load* feedback that the per-fleet envelope ignores by construction.

------------------------------------------------------------------------------------------------
MODEL
------------------------------------------------------------------------------------------------
Per region we have the realized residual-load series R[t] (load minus wind+solar), 48 hourly slots,
exactly the signal the envelope decides on (data/timealigned*/grid_residual_region_slot.csv).

The total *flexible* fleet load following the signal is parameterised by penetration p, expressed as
a fraction of the region's MEAN residual load:  F = p * mean(R).  This is the natural normalisation:
"p% of regional load is flexible-and-following".  This flexible block is what the herd moves around;
the rest of the grid (1 - it) is the inflexible background R[t] itself.

Each fleet behaves EXACTLY like the envelope's grid-relief lever: it removes IT energy from the
top-quartile ("high-stress") residual slots (stress_quantile=0.75 in the engine) and re-places it in
the bottom-quartile ("headroom") slots (headroom_quantile=0.25). That is the relief move the
certificate rewards (stress_kwh_avoided = energy pulled out of stress slots).

We sweep two regimes:

  (1) NAIVE / SIMULTANEOUS herd — every fleet sees the SAME unperturbed R[t], all pull from the same
      stress slots and dump into the same headroom slots. This is the worst case and what happens if
      fleets act open-loop on a published forecast. Aggregate signal:
          A[t] = R[t]  - (shifted-out at stress slots)  + (shifted-in at headroom slots)
      The whole flexible block F lands on the headroom slots at once.

  (2) SIGNAL-RESPONSIVE / SEQUENTIAL best-response — fleets act one after another; each fleet sees the
      residual ALREADY perturbed by the fleets ahead of it and targets the *current* min/max slots.
      This is the charitable case (an implicit coordinating signal: the updated residual). It shows
      how much of the rebound a price/quantity feedback would damp. We still let every fleet target the
      instantaneous trough, so herding can still occur once the troughs fill — that is the point.

For both we report, vs penetration p, the AGGREGATE peak A_max and the AGGREGATE variance Var(A),
and we locate the inversion p* where these first EXCEED the do-nothing baseline (R itself), i.e. where
"grid support" flips to "grid harm".

A closed-form p* for the naive herd is derived in RESULTS.md and printed by --derive.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import pvariance
from typing import Dict, List, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_SIGNAL = os.path.join(
    REPO_ROOT, "pkg", "carbon-aware", "data", "timealigned_realci", "grid_residual_region_slot.csv"
)
OUT_DIR = os.path.join(REPO_ROOT, "experiments", "mc4_rebound")

# Engine-faithful quartile thresholds (no_harm_flexibility.py: stress_quantile=0.75,
# headroom_quantile=0.25). A slot is "high-stress" if its residual is in the top quartile and
# "headroom" if in the bottom quartile.
STRESS_Q = 0.75
HEADROOM_Q = 0.25

# Penetration p = flexible-and-following load as a fraction of the region's MEAN residual load.
# Swept past 100% to locate the trough-herd inversion (which the closed form puts near ~60-130%).
PENETRATIONS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90, 1.00, 1.25, 1.50, 2.00]


# ------------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------------
def load_signal(path: str) -> Dict[str, List[float]]:
    by_region: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            by_region[row["region"]].append(
                (int(row["slot_index"]), float(row["residual_load_mw"]))
            )
    out: Dict[str, List[float]] = {}
    for region, pairs in by_region.items():
        pairs.sort(key=lambda x: x[0])
        out[region] = [v for _, v in pairs]
    return out


def quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = q * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


# ------------------------------------------------------------------------------------------------
# the per-fleet envelope move (grid-relief lever, signal-faithful)
# ------------------------------------------------------------------------------------------------
def classify_slots(signal: List[float]) -> Tuple[List[int], List[int]]:
    """Return (stress_slot_indices, headroom_slot_indices) for the *given* residual series."""
    s = sorted(signal)
    hi_cut = quantile(s, STRESS_Q)
    lo_cut = quantile(s, HEADROOM_Q)
    stress = [t for t, v in enumerate(signal) if v >= hi_cut - 1e-9]
    headroom = [t for t, v in enumerate(signal) if v <= lo_cut + 1e-9]
    return stress, headroom


def shift_block(signal: List[float], block_mw: float, target: str = "band") -> List[float]:
    """Apply one envelope-style relief move of total energy `block_mw`. Energy-conserving.

    The fleet pulls `block_mw` of IT load OUT of the current stress slots (always the top-quartile band,
    spread evenly — that is where the fleet's deferrable load currently sits) and re-places it INTO the
    low-residual slots. Two re-placement targetings:

      target="band"   : spread the in-shift evenly across the whole bottom-quartile *headroom band*
                        (the engine's headroom_quantile classification). Charitable: the fleet behaves
                        as if it knows the band, not just the single min.
      target="trough" : dump the entire in-shift onto the SINGLE current-minimum slot — the myopic
                        follower that chases the deepest trough of the signal it is handed. This is the
                        textbook herding move and the one that produces a secondary peak fastest.

    The instantaneous min/headroom is recomputed each call, so the sequential composition is a true
    best-response chain (each fleet sees predecessors' perturbation).
    """
    out = list(signal)
    stress, headroom = classify_slots(signal)
    if not stress or not headroom:
        return out
    per_stress = block_mw / len(stress)
    for t in stress:
        out[t] -= per_stress
    if target == "band":
        per_head = block_mw / len(headroom)
        for t in headroom:
            out[t] += per_head
    elif target == "trough":
        t_min = min(range(len(out)), key=lambda t: out[t])
        out[t_min] += block_mw
    else:
        raise ValueError(target)
    return out


# ------------------------------------------------------------------------------------------------
# multi-fleet aggregation
# ------------------------------------------------------------------------------------------------
@dataclass
class RegimeResult:
    penetration: float
    peak_mw: float
    variance: float
    peak_ratio: float  # vs baseline R
    var_ratio: float
    new_peak_slot: int
    baseline_peak_slot: int


def run_region(
    signal: List[float], n_fleets: int, mode: str
) -> List[RegimeResult]:
    base = list(signal)
    base_peak = max(base)
    base_var = pvariance(base)
    base_peak_slot = max(range(len(base)), key=lambda t: base[t])
    mean_R = sum(base) / len(base)

    results: List[RegimeResult] = []
    for p in PENETRATIONS:
        # Total flexible-and-following load = p * mean residual. Split across n_fleets equal fleets.
        total_flex = p * mean_R
        per_fleet = total_flex / n_fleets

        if mode == "naive_band":
            # Every fleet acts on the SAME unperturbed signal -> all dump into the same headroom band.
            agg = shift_block(base, total_flex, target="band")
        elif mode == "naive_trough":
            # Worst case: open-loop herd, every fleet dumps onto the SAME single global-min slot.
            agg = shift_block(base, total_flex, target="trough")
        elif mode == "responsive_trough":
            # Myopic best-response chain: each fleet chases the *current* trough, which moves as the
            # trough fills. This is the charitable "implicit coordination" version of trough-chasing —
            # the updated residual is the coordinating signal — yet herding still bites once troughs fill.
            agg = list(base)
            for _ in range(n_fleets):
                agg = shift_block(agg, per_fleet, target="trough")
        else:
            raise ValueError(mode)

        peak = max(agg)
        var = pvariance(agg)
        new_slot = max(range(len(agg)), key=lambda t: agg[t])
        results.append(
            RegimeResult(
                penetration=p,
                peak_mw=peak,
                variance=var,
                peak_ratio=peak / base_peak,
                var_ratio=(var / base_var) if base_var else float("nan"),
                new_peak_slot=new_slot,
                baseline_peak_slot=base_peak_slot,
            )
        )
    return results


def find_inversion(results: List[RegimeResult], metric: str) -> Tuple[float, float] | None:
    """Smallest penetration at which the metric first EXCEEDS baseline (ratio crosses 1.0).

    Returns (p_lower, p_inversion_interpolated) or None if never inverts in the sweep.
    """
    prev = None
    for r in results:
        ratio = r.peak_ratio if metric == "peak" else r.var_ratio
        if ratio > 1.0 + 1e-9:
            if prev is None:
                return (r.penetration, r.penetration)
            # linear interpolation in penetration to ratio==1.0
            p0, r0 = prev
            p1, r1 = r.penetration, ratio
            if r1 == r0:
                return (p0, p1)
            p_star = p0 + (1.0 - r0) * (p1 - p0) / (r1 - r0)
            return (p1, p_star)
        prev = (r.penetration, ratio)
    return None


def closed_form_pstar(signal: List[float]) -> Dict[str, float]:
    """Analytical naive-herd inversion thresholds (two targetings).

    Let F = p * mean(R) be the total flexible-and-following block, mean(R) the region's mean residual,
    R_peak the original peak, and R_trough the original minimum (deepest headroom slot).

    (A) TROUGH-TARGETING herd (target="trough"): the whole block lands on the single deepest slot,
        lifting it to R_trough + F. The herd also depresses the original peak by F/n_stress. A NEW peak
        at the (former) trough appears once it overtakes the depressed old peak:
            R_trough + F  >=  R_peak - F / n_stress
            F * (1 + 1/n_stress)  >=  R_peak - R_trough
            F_trough* = (R_peak - R_trough) / (1 + 1/n_stress)
        p_trough* = F_trough* / mean(R).  This is the sharp, realistic herding bound.

    (B) BAND-TARGETING herd (target="band"): the block is spread over the n_head headroom slots, so the
        binding slot is the HIGHEST one in the band, R_head_hi. New band peak vs depressed old peak:
            R_head_hi + F/n_head  >=  R_peak - F/n_stress
            F_band* = (R_peak - R_head_hi) / (1/n_head + 1/n_stress)
        p_band* = F_band* / mean(R).  Charitable (the band absorbs the block), so p_band* > p_trough*.
    """
    mean_R = sum(signal) / len(signal)
    R_peak = max(signal)
    R_trough = min(signal)
    stress, headroom = classify_slots(signal)
    n_head = len(headroom)
    n_stress = len(stress)
    R_head_hi = max(signal[t] for t in headroom)  # highest residual still inside headroom band
    R_stress_lo = min(signal[t] for t in stress)  # lowest residual still inside stress band

    F_trough = (R_peak - R_trough) / (1.0 + 1.0 / n_stress)
    p_trough = F_trough / mean_R

    F_band = (R_peak - R_head_hi) / (1.0 / n_head + 1.0 / n_stress)
    p_band = F_band / mean_R

    return {
        "mean_R_mw": mean_R,
        "peak_mw": R_peak,
        "trough_mw": R_trough,
        "n_headroom": float(n_head),
        "n_stress": float(n_stress),
        "R_headroom_hi_mw": R_head_hi,
        "R_stress_lo_mw": R_stress_lo,
        "F_trough_inversion_mw": F_trough,
        "p_trough_inversion": p_trough,
        "F_band_inversion_mw": F_band,
        "p_band_inversion": p_band,
    }


# ------------------------------------------------------------------------------------------------
# driver
# ------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--signal", default=DEFAULT_SIGNAL)
    ap.add_argument("--n-fleets", type=int, default=50,
                    help="number of independent fleets in the herd (responsive mode)")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--derive", action="store_true", help="print closed-form p* and exit")
    args = ap.parse_args()

    by_region = load_signal(args.signal)
    os.makedirs(args.out, exist_ok=True)

    if args.derive:
        for region, sig in by_region.items():
            cf = closed_form_pstar(sig)
            print(f"[{region:6}] trough-herd p* = {cf['p_trough_inversion']*100:5.1f}%  "
                  f"band-herd p* = {cf['p_band_inversion']*100:5.1f}%   "
                  f"(peak={cf['peak_mw']:.0f}, trough={cf['trough_mw']:.0f}, "
                  f"R_head_hi={cf['R_headroom_hi_mw']:.0f}, n_stress={cf['n_stress']:.0f})")
        return

    summary: Dict[str, dict] = {}
    rows: List[dict] = []
    modes = ("naive_band", "naive_trough", "responsive_trough")
    for region, sig in by_region.items():
        cf = closed_form_pstar(sig)
        region_block = {"closed_form": cf, "modes": {}}
        for mode in modes:
            res = run_region(sig, args.n_fleets, mode)
            peak_inv = find_inversion(res, "peak")
            var_inv = find_inversion(res, "variance")
            region_block["modes"][mode] = {
                "peak_inversion_p": (peak_inv[1] if peak_inv else None),
                "variance_inversion_p": (var_inv[1] if var_inv else None),
                "sweep": [vars(r) for r in res],
            }
            for r in res:
                rows.append({
                    "region": region, "mode": mode, "penetration": r.penetration,
                    "peak_mw": round(r.peak_mw, 1), "variance": round(r.variance, 1),
                    "peak_ratio": round(r.peak_ratio, 4), "var_ratio": round(r.var_ratio, 4),
                    "new_peak_slot": r.new_peak_slot, "baseline_peak_slot": r.baseline_peak_slot,
                })
        summary[region] = region_block

    # write artifacts
    with open(os.path.join(args.out, "sweep.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump({"n_fleets": args.n_fleets, "signal": args.signal,
                   "stress_q": STRESS_Q, "headroom_q": HEADROOM_Q,
                   "penetrations": PENETRATIONS, "regions": summary}, fh, indent=2)

    # console report
    print(f"signal: {args.signal}")
    print(f"n_fleets (responsive herd): {args.n_fleets}\n")
    for region, blk in summary.items():
        cf = blk["closed_form"]
        print(f"=== {region} ===  mean_R={cf['mean_R_mw']:.0f} MW  peak={cf['peak_mw']:.0f} MW  "
              f"trough={cf['trough_mw']:.0f} MW  n_head={cf['n_headroom']:.0f}  n_stress={cf['n_stress']:.0f}")
        print(f"  closed-form trough-herd p* = {cf['p_trough_inversion']*100:.1f}%   "
              f"band-herd p* = {cf['p_band_inversion']*100:.1f}%")
        for mode in modes:
            m = blk["modes"][mode]
            pk = m["peak_inversion_p"]
            vr = m["variance_inversion_p"]
            pk_s = f"{pk*100:.1f}%" if pk is not None else ">100% (none in sweep)"
            vr_s = f"{vr*100:.1f}%" if vr is not None else ">100% (none in sweep)"
            print(f"  [{mode:18}] sim peak-inversion p* = {pk_s:22}  variance-inversion p* = {vr_s}")
        print("    p(%)  | naive_trough peak  var  | resp_trough peak  var  | naive_band peak  var")
        nt = {r["penetration"]: r for r in blk["modes"]["naive_trough"]["sweep"]}
        rt = {r["penetration"]: r for r in blk["modes"]["responsive_trough"]["sweep"]}
        nb = {r["penetration"]: r for r in blk["modes"]["naive_band"]["sweep"]}
        for p in PENETRATIONS:
            print(f"    {p*100:5.0f} |   {nt[p]['peak_ratio']:.3f}      {nt[p]['var_ratio']:.3f} "
                  f"|   {rt[p]['peak_ratio']:.3f}     {rt[p]['var_ratio']:.3f} "
                  f"|  {nb[p]['peak_ratio']:.3f}    {nb[p]['var_ratio']:.3f}")
        print()
    print(f"wrote {os.path.join(args.out, 'sweep.csv')} and summary.json")


if __name__ == "__main__":
    main()
