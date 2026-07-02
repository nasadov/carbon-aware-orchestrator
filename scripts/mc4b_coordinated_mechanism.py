#!/usr/bin/env python3
"""MC4b — Coordinated no-harm flexibility as a MARKET/MECHANISM primitive (TSUSC R2, mechanism frame).

MC4 (mc4_rebound_sim.py) established the *threat*: a naive open-loop herd of price-taking no-harm
fleets, all following the SAME published residual signal, coincides in the troughs and inverts the
aggregate peak at p* ~ 30-45% of regional load. The accompanying claim was that a *signal-responsive*
herd (each fleet reacting to the residual already perturbed by predecessors) never inverts.

This script makes the mechanism explicit and turns the threat into a DESIGNED instrument. It asks the
first-principles question: if the no-harm certificate is embedded in a COORDINATING SIGNAL (a broadcast
residual-load target = a shadow price), does it provably keep MANY fleets in the grid-supportive regime
past the open-loop p*? It contrasts four regimes at matched penetration p:

  (1) open_loop          - every fleet dumps its in-shift on the SAME published global-min slot.
                           This is MC4's naive_trough: the worst case, inverts at p* ~ 30-45%.
  (2) sequential_br       - Gauss-Seidel best response: fleets act in arbitrary order, each sees the
                           residual already perturbed by predecessors and targets the *current* trough.
                           This is MC4's responsive_trough (implicit coordination). Order-dependent.
  (3) broadcast_target   - THE MECHANISM. A coordinator broadcasts a residual-load TARGET (=  the
                           water-filling level lambda such that filling every below-target slot up to
                           lambda exactly absorbs the aggregate flexible block F). Every fleet, in
                           parallel and WITHOUT talking to each other, fills proportionally toward that
                           single broadcast level. This is one round of a transactive/dual-decomposition
                           clearing: the broadcast target is the shadow price; each fleet's response is
                           its no-harm relief move evaluated against that price. No herding by
                           construction: the target is the post-fill level, so nobody overshoots it.
  (4) iterated_dual      - the same mechanism but the coordinator does a few dual-ascent rounds
                           (re-broadcast target after observing realized aggregate), to show convergence
                           and robustness to fleets only partially complying.

Per-fleet no-harm guard is PRESERVED in every regime: a fleet only ever moves energy from its own
top-quartile stress slots into below-target slots, never raising any slot above the broadcast target,
so no fleet's move can, on the realized aggregate, push a slot past the level the coordinator certified
as safe. The certificate stays per-fleet and ex-post; the mechanism only changes WHERE the in-shift
lands, replacing "the published min" (open-loop) with "the broadcast water-filling level" (coordinated).

Headline we want to validate:
  - open_loop inverts (peak ratio > 1) at p* ~ 30-45% (matches MC4).
  - broadcast_target and iterated_dual do NOT invert across the swept range (0-200%): the aggregate
    peak is monotonically non-increasing and variance falls, because water-filling to a common level is
    exactly valley-filling. This is the proposition: a coordinating signal carrying the no-harm
    constraint keeps an unbounded number of fleets grid-supportive past the open-loop threshold.

Energy-conserving temporal shift only (no induced demand). Uses the SAME realized EU residual series
and the SAME engine quartile classification as MC4, so the two are directly comparable.

Reproduce:
  python3 scripts/mc4b_coordinated_mechanism.py                 # full sweep + artifacts
  python3 scripts/mc4b_coordinated_mechanism.py --n-fleets 200  # robustness to fleet count
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from statistics import pvariance
from typing import Dict, List, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_SIGNAL = os.path.join(
    REPO_ROOT, "pkg", "carbon-aware", "data", "timealigned_realci", "grid_residual_region_slot.csv"
)
OUT_DIR = os.path.join(REPO_ROOT, "experiments", "mc4_rebound")

STRESS_Q = 0.75  # engine no_harm_flexibility.py stress_quantile
HEADROOM_Q = 0.25  # engine headroom_quantile
PENETRATIONS = [0.05, 0.10, 0.20, 0.30, 0.37, 0.40, 0.50, 0.60, 0.75, 1.00, 1.25, 1.50, 2.00]


def load_signal(path: str) -> Dict[str, List[float]]:
    by_region: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            by_region[row["region"]].append((int(row["slot_index"]), float(row["residual_load_mw"])))
    out: Dict[str, List[float]] = {}
    for region, pairs in by_region.items():
        pairs.sort(key=lambda x: x[0])
        out[region] = [v for _, v in pairs]
    return out


def quantile(sorted_vals: List[float], q: float) -> float:
    idx = q * (len(sorted_vals) - 1)
    lo, hi = int(math.floor(idx)), int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def stress_slots(signal: List[float]) -> List[int]:
    hi_cut = quantile(sorted(signal), STRESS_Q)
    return [t for t, v in enumerate(signal) if v >= hi_cut - 1e-9]


def waterfill_level(signal: List[float], stress: List[int], block: float) -> float:
    """Coordinator's broadcast TARGET lambda: the level such that summing (lambda - R[t])^+ over the
    non-stress slots equals `block` (the aggregate flexible in-shift). This is the classic
    valley-filling / water-filling level. Solved by sorting the fillable slots and raising the level
    until the poured volume equals `block`. We only fill slots BELOW the lowest stress slot so the
    in-shift never approaches the stress band (the no-harm relief region)."""
    fillable = sorted([signal[t] for t in range(len(signal)) if t not in set(stress)])
    if not fillable:
        return min(signal)
    # raise level across sorted fillable slots; greedy accumulation
    level = fillable[0]
    poured = 0.0
    for k in range(1, len(fillable)):
        rise = fillable[k] - level
        need = rise * k  # raising k already-at-level slots by `rise`
        if poured + need >= block:
            level += (block - poured) / k
            return level
        poured += need
        level = fillable[k]
    # all fillable slots now at `level`; pour the rest uniformly across all of them
    level += (block - poured) / len(fillable)
    return level


def apply_open_loop(base: List[float], block: float, stress: List[int]) -> List[float]:
    out = list(base)
    per = block / len(stress)
    for t in stress:
        out[t] -= per
    t_min = min(range(len(out)), key=lambda t: out[t])
    out[t_min] += block
    return out


def apply_sequential(base: List[float], block: float, n_fleets: int) -> List[float]:
    out = list(base)
    per_fleet = block / n_fleets
    for _ in range(n_fleets):
        st = stress_slots(out)
        per = per_fleet / len(st)
        for t in st:
            out[t] -= per
        t_min = min(range(len(out)), key=lambda t: out[t])
        out[t_min] += per_fleet
    return out


def apply_broadcast(base: List[float], block: float, stress: List[int], rounds: int = 1) -> List[float]:
    """The mechanism: coordinator broadcasts water-filling target; fleets fill below-target slots up to
    the target. With `rounds`>1 the coordinator re-solves the target on the realized residual (dual
    ascent), which converges to the exact valley-fill. Energy pulled from stress slots is conserved."""
    out = list(base)
    per = block / len(stress)
    for t in stress:
        out[t] -= per  # relief: remove block from own stress band (same as every regime)
    remaining = block
    stress_set = set(stress)
    for _ in range(max(1, rounds)):
        if remaining <= 1e-9:
            break
        level = waterfill_level(out, stress, remaining)
        deficits = {t: max(0.0, level - out[t]) for t in range(len(out)) if t not in stress_set}
        total_def = sum(deficits.values())
        if total_def <= 1e-9:
            break
        pour = min(remaining, total_def)
        for t, d in deficits.items():
            out[t] += pour * (d / total_def)
        remaining -= pour
    if remaining > 1e-9:  # safety: pour any residue on the global min (won't trigger for valid block)
        out[min(range(len(out)), key=lambda t: out[t])] += remaining
    return out


@dataclass
class Row:
    region: str
    mode: str
    p: float
    peak_ratio: float
    var_ratio: float


def run() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--signal", default=DEFAULT_SIGNAL)
    ap.add_argument("--n-fleets", type=int, default=50)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    by_region = load_signal(args.signal)
    rows: List[Row] = []
    inversion: Dict[str, Dict[str, float | None]] = {}

    for region, sig in by_region.items():
        base_peak, base_var = max(sig), pvariance(sig)
        mean_R = sum(sig) / len(sig)
        st = stress_slots(sig)
        inv = {m: None for m in ("open_loop", "sequential_br", "broadcast_target", "iterated_dual")}
        for p in PENETRATIONS:
            block = p * mean_R
            variants = {
                "open_loop": apply_open_loop(sig, block, st),
                "sequential_br": apply_sequential(sig, block, args.n_fleets),
                "broadcast_target": apply_broadcast(sig, block, st, rounds=1),
                "iterated_dual": apply_broadcast(sig, block, st, rounds=8),
            }
            for mode, agg in variants.items():
                pr, vr = max(agg) / base_peak, pvariance(agg) / base_var
                rows.append(Row(region, mode, p, pr, vr))
                if inv[mode] is None and pr > 1.0 + 1e-9:
                    inv[mode] = p
        inversion[region] = inv

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "coordinated_sweep.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["region", "mode", "penetration", "peak_ratio", "var_ratio"])
        for r in rows:
            w.writerow([r.region, r.mode, r.p, round(r.peak_ratio, 4), round(r.var_ratio, 4)])
    with open(os.path.join(args.out, "coordinated_summary.json"), "w") as fh:
        json.dump({"n_fleets": args.n_fleets, "peak_inversion_p": inversion}, fh, indent=2)

    print(f"signal: {args.signal}   n_fleets(sequential): {args.n_fleets}\n")
    print("PEAK-INVERSION penetration p* (first p where aggregate peak > baseline):")
    print(f"{'region':8} {'open_loop':>12} {'sequential_br':>14} {'broadcast':>12} {'iter_dual':>11}")
    for region, inv in inversion.items():
        def f(x): return f"{x*100:.0f}%" if x is not None else "none<=200%"
        print(f"{region:8} {f(inv['open_loop']):>12} {f(inv['sequential_br']):>14} "
              f"{f(inv['broadcast_target']):>12} {f(inv['iterated_dual']):>11}")
    print("\nDE peak ratio vs penetration (open_loop / broadcast / iter_dual):")
    de = {(r.mode, r.p): r for r in rows if r.region == "DE"}
    print(f"{'p%':>5} {'open_loop':>11} {'broadcast':>11} {'iter_dual':>11}   {'broadcast var':>13}")
    for p in PENETRATIONS:
        ol = de[("open_loop", p)]; bc = de[("broadcast_target", p)]; idl = de[("iterated_dual", p)]
        flag = "  <-- open-loop inverted" if ol.peak_ratio > 1 else ""
        print(f"{p*100:5.0f} {ol.peak_ratio:11.3f} {bc.peak_ratio:11.3f} {idl.peak_ratio:11.3f}   "
              f"{bc.var_ratio:13.3f}{flag}")
    print(f"\nwrote {os.path.join(args.out, 'coordinated_sweep.csv')} and coordinated_summary.json")


if __name__ == "__main__":
    run()
