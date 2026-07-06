#!/usr/bin/env python3
"""Biocreep replay: chained vs anchored comparators across drifting signal epochs.

Serial non-inferiority certification is known to ratchet harm when the comparator
drifts between trials (Everson-Stewart & Emerson 2010, "bio-creep"): certifying each
window against the PREVIOUS window's certified schedule compounds slippage that a
fixed epoch anchor forbids. Grid signals, fleet composition, and scarcity all drift,
so the non-constancy regime is the default for a deployed certificate scheme.

Setup: one canonical Azure CPU window (recurring-batch population, fixed pods), K
signal epochs sigma_0..sigma_{K-1} built by rotating the real 48-slot signal series
(carbon forecast, residual load, wet-bulb/WUE, EWIF) by ROT_SLOTS*k slots per region
-- real marginals, drifted phase. Two arms, identical engine and guard:

  * CHAINED : S_k = repair(baseline = S_{k-1} re-priced under sigma_k)   [the trap]
  * ANCHORED: A_k = repair(baseline = B    re-priced under sigma_k)      [the rule]

where B is the packing baseline of epoch 0 (the escrowed epoch anchor). Each epoch
reports both arms' carbon/scarcity totals under sigma_k, their deltas vs the anchor
under sigma_k, and the certificate vs the anchor. The chained arm certifies vs its
own declared (drifting) comparator by construction; the question is whether it stays
non-degrading vs the ANCHOR -- if not, the per-window certificates chain into
uncertified cumulative harm, which is the bio-creep figure.

Emits experiments/biocreep/biocreep_replay.json.

Run:
  PYTHONPATH=pkg/carbon-aware/server-python python scripts/biocreep_replay.py [--epochs 12] [--rot 6]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace as dc_replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SERVER = REPO / "pkg" / "carbon-aware" / "server-python"
for p in (str(SERVER), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_water_basin_certification as wb  # noqa: E402  (Azure window builder, REALCI paths)
from carbon_aware.no_harm_flexibility import (  # noqa: E402
    ScheduleResult,
    build_action_signals,
    build_greedy_schedule,
    load_flavours_for_pilot,
    load_pods,
    repair_schedule_no_harm,
    schedule_totals,
    _rematerialize_under_realized,
)

OUT = REPO / "experiments" / "biocreep"
WINDOW = "azure_packing_2020_d0_s42_200"


# ------------------------------------------------------------------ signal-epoch construction
def _rotate(seq, shift):
    n = len(seq)
    if n == 0:
        return list(seq)
    s = shift % n
    return list(seq[s:]) + list(seq[:s])


def build_epoch_signals(src_dir: Path, dst_dir: Path, shift: int) -> None:
    """Copy the signal payload with every per-region value series rotated by `shift` slots.
    Timestamps/slot indices stay fixed; values shift -> same grid, drifted regime phase."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    fj = json.loads((src_dir / "forecasts.json").read_text())
    for region, payload in fj.items():
        fc = payload.get("forecast", [])
        vals = _rotate([e["carbonIntensity"] for e in fc], shift)
        for e, v in zip(fc, vals):
            e["carbonIntensity"] = v
    (dst_dir / "forecasts.json").write_text(json.dumps(fj))

    for name in ("grid_residual_region_slot.csv", "wue_region_slot.csv", "ewif_region_slot.csv"):
        src = src_dir / name
        if not src.exists():
            continue
        with src.open() as f:
            rows = list(csv.DictReader(f))
            fields = f.seek(0) or list(csv.DictReader(f).fieldnames or [])
        keys = {"region", "slot_index", "utc_timestamp", "signal_model", "source_dataset", "basin", "country"}
        value_cols = [c for c in fields if c not in keys]
        by_region: dict = {}
        for r in rows:
            by_region.setdefault(r["region"], []).append(r)
        for region, rr in by_region.items():
            rr.sort(key=lambda r: int(r["slot_index"]))
            for col in value_cols:
                vals = _rotate([r[col] for r in rr], shift)
                for r, v in zip(rr, vals):
                    r[col] = v
        with (dst_dir / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)


def epoch_config(base_pc, epoch_dir: Path, out_dir: Path):
    ewif = epoch_dir / "ewif_region_slot.csv"
    return dc_replace(
        base_pc,
        forecasts_file=epoch_dir / "forecasts.json",
        grid_signal_csv=epoch_dir / "grid_residual_region_slot.csv",
        wue_csv=epoch_dir / "wue_region_slot.csv",
        ewif_csv=ewif if ewif.exists() else None,
        output_dir=out_dir,
    )


# ------------------------------------------------------------------ evaluation helpers
def reprice(result: ScheduleResult, flavours, config) -> ScheduleResult:
    """Honest occupancy-based re-pricing of a schedule under a (new) signal world."""
    clone = ScheduleResult(
        method_key=result.method_key,
        placements=[dc_replace(p) for p in result.placements],
        unplaced_pods=list(result.unplaced_pods),
        elapsed_seconds=result.elapsed_seconds,
    )
    _rematerialize_under_realized(clone, flavours, config)
    return clone


def totals_row(result: ScheduleResult, signals) -> dict:
    t = schedule_totals(result, signals)
    return {"carbon_kg": t["carbon_kg"], "scarcity_water": t["scarcity_water"],
            "unplaced": len(result.unplaced_pods)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--rot", type=int, default=6, help="slots of drift per epoch (48-slot cycle)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=0.005,
                    help="declared per-window per-axis tolerance for the eps arms (0.005 = 0.5%%)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    run_dir = OUT / "fleet_build"
    run_dir.mkdir(parents=True, exist_ok=True)

    base_pc, n_nodes, n_pods = wb._build_azure_config(WINDOW, run_dir, seed=args.seed)
    pods = load_pods(base_pc.workloads_dir, max_pods=base_pc.max_pods)
    print(f"[fleet] {WINDOW}: {n_pods} pods on {n_nodes} nodes")

    # epoch signal worlds
    epoch_worlds = []
    for k in range(args.epochs):
        ed = OUT / "epochs" / f"e{k:02d}"
        build_epoch_signals(wb.REALCI, ed, shift=args.rot * k)
        pc_k = epoch_config(base_pc, ed, OUT / "out" / f"e{k:02d}")
        pc_k.output_dir.mkdir(parents=True, exist_ok=True)
        fl_k = load_flavours_for_pilot(pc_k)
        sg_k = build_action_signals(fl_k, pc_k)
        epoch_worlds.append((pc_k, fl_k, sg_k))

    # epoch anchor: packing baseline declared at epoch 0
    pc0, fl0, sg0 = epoch_worlds[0]
    anchor, _, _ = build_greedy_schedule(method_key="packing", pods=pods, flavours=fl0, config=pc0)

    pc_eps = {"epsilon_carbon": args.eps, "epsilon_scarcity": args.eps}

    rows = []
    chained_prev: ScheduleResult = anchor
    chained_eps_prev: ScheduleResult = anchor
    chained_rand_prev: ScheduleResult = anchor
    for k, (pc_k, fl_k, sg_k) in enumerate(epoch_worlds):
        anchor_k = reprice(anchor, fl_k, pc_k)          # anchor honestly priced under sigma_k
        anchor_t = totals_row(anchor_k, sg_k)

        # CHAINED arm: guard vs previous certified schedule, re-priced under sigma_k
        chain_base = reprice(chained_prev, fl_k, pc_k)
        chain_base_t = totals_row(chain_base, sg_k)
        s_k = repair_schedule_no_harm(
            method_key="chained", baseline=chain_base, pods=pods,
            flavours=fl_k, signals=sg_k, config=pc_k, score_mode="combined")
        s_k = reprice(s_k, fl_k, pc_k)
        s_t = totals_row(s_k, sg_k)

        # ANCHORED arm: guard vs the epoch anchor, re-priced under sigma_k
        a_k = repair_schedule_no_harm(
            method_key="anchored", baseline=anchor_k, pods=pods,
            flavours=fl_k, signals=sg_k, config=pc_k, score_mode="combined")
        a_k = reprice(a_k, fl_k, pc_k)
        a_t = totals_row(a_k, sg_k)

        # EPS arms: identical guard with a declared per-window tolerance band on both axes.
        # Chained-eps certifies "within declared eps vs the ROLLING baseline" each window (the
        # Everson-Stewart compounding channel); anchored-eps declares the same band vs the FIXED
        # anchor, so total drift is bounded by one eps, not K of them.
        pc_k_eps = dc_replace(pc_k, **pc_eps)
        ce_base = reprice(chained_eps_prev, fl_k, pc_k_eps)
        ce_base_t = totals_row(ce_base, sg_k)
        se_k = repair_schedule_no_harm(
            method_key="chained_eps", baseline=ce_base, pods=pods,
            flavours=fl_k, signals=sg_k, config=pc_k_eps, score_mode="combined")
        se_k = reprice(se_k, fl_k, pc_k_eps)
        se_t = totals_row(se_k, sg_k)
        ae_k = repair_schedule_no_harm(
            method_key="anchored_eps", baseline=anchor_k, pods=pods,
            flavours=fl_k, signals=sg_k, config=pc_k_eps, score_mode="combined")
        ae_k = reprice(ae_k, fl_k, pc_k_eps)
        ae_t = totals_row(ae_k, sg_k)

        # ADVERSARIAL-IN-SPEC arm: a compliant-but-lazy scheduler -- random guard-passing moves
        # inside the SAME declared eps band, chained. Every window certifies "within declared eps
        # vs the rolling baseline"; the question is what the SPECIFICATION permits to compound.
        pc_k_rand = dc_replace(pc_k_eps, repair_random_seed=1000 + k)
        cr_base = reprice(chained_rand_prev, fl_k, pc_k_rand)
        cr_base_t = totals_row(cr_base, sg_k)
        sr_k = repair_schedule_no_harm(
            method_key="chained_eps_random", baseline=cr_base, pods=pods,
            flavours=fl_k, signals=sg_k, config=pc_k_rand, score_mode="guarded_random")
        sr_k = reprice(sr_k, fl_k, pc_k_rand)
        sr_t = totals_row(sr_k, sg_k)

        def pct(x, b):
            return 100.0 * (x - b) / b if b else 0.0

        row = {
            "epoch": k,
            "shift_slots": args.rot * k,
            "anchor": anchor_t,
            "chained_declared_baseline": chain_base_t,
            "chained": s_t,
            "anchored": a_t,
            "chained_eps": se_t,
            "anchored_eps": ae_t,
            "chained_vs_anchor_pct": {
                "carbon": pct(s_t["carbon_kg"], anchor_t["carbon_kg"]),
                "scarcity": pct(s_t["scarcity_water"], anchor_t["scarcity_water"]),
            },
            "anchored_vs_anchor_pct": {
                "carbon": pct(a_t["carbon_kg"], anchor_t["carbon_kg"]),
                "scarcity": pct(a_t["scarcity_water"], anchor_t["scarcity_water"]),
            },
            "chained_eps_vs_anchor_pct": {
                "carbon": pct(se_t["carbon_kg"], anchor_t["carbon_kg"]),
                "scarcity": pct(se_t["scarcity_water"], anchor_t["scarcity_water"]),
            },
            "anchored_eps_vs_anchor_pct": {
                "carbon": pct(ae_t["carbon_kg"], anchor_t["carbon_kg"]),
                "scarcity": pct(ae_t["scarcity_water"], anchor_t["scarcity_water"]),
            },
            "chained_eps_within_declared_band": (
                se_t["carbon_kg"] <= ce_base_t["carbon_kg"] * (1.0 + args.eps) + 1e-9
                and se_t["scarcity_water"] <= ce_base_t["scarcity_water"] * (1.0 + args.eps) + 1e-9),
            "chained_eps_random": sr_t,
            "chained_eps_random_vs_anchor_pct": {
                "carbon": pct(sr_t["carbon_kg"], anchor_t["carbon_kg"]),
                "scarcity": pct(sr_t["scarcity_water"], anchor_t["scarcity_water"]),
            },
            "chained_eps_random_within_declared_band": (
                sr_t["carbon_kg"] <= cr_base_t["carbon_kg"] * (1.0 + args.eps) + 1e-9
                and sr_t["scarcity_water"] <= cr_base_t["scarcity_water"] * (1.0 + args.eps) + 1e-9),
            # certificate vs the ANCHOR (the question); tau matches the engine's reject-erring zero
            "chained_certifies_vs_anchor": (
                s_t["carbon_kg"] <= anchor_t["carbon_kg"] + 1e-9
                and s_t["scarcity_water"] <= anchor_t["scarcity_water"] + 1e-9
                and s_t["unplaced"] <= anchor_t["unplaced"]),
            # per-window certificate vs its own declared baseline (holds by construction)
            "chained_certifies_vs_declared": (
                s_t["carbon_kg"] <= chain_base_t["carbon_kg"] + 1e-9
                and s_t["scarcity_water"] <= chain_base_t["scarcity_water"] + 1e-9
                and s_t["unplaced"] <= chain_base_t["unplaced"]),
            "anchored_certifies_vs_anchor": (
                a_t["carbon_kg"] <= anchor_t["carbon_kg"] + 1e-9
                and a_t["scarcity_water"] <= anchor_t["scarcity_water"] + 1e-9
                and a_t["unplaced"] <= anchor_t["unplaced"]),
        }
        rows.append(row)
        chained_prev = s_k
        chained_eps_prev = se_k
        chained_rand_prev = sr_k
        print(f"[e{k:02d}] chained: C {row['chained_vs_anchor_pct']['carbon']:+.2f}% "
              f"W {row['chained_vs_anchor_pct']['scarcity']:+.2f}% "
              f"cert(anchor)={row['chained_certifies_vs_anchor']} | "
              f"anchored: C {row['anchored_vs_anchor_pct']['carbon']:+.2f}% | "
              f"chained-eps: C {row['chained_eps_vs_anchor_pct']['carbon']:+.2f}% "
              f"W {row['chained_eps_vs_anchor_pct']['scarcity']:+.2f}% "
              f"inband={row['chained_eps_within_declared_band']} | "
              f"anchored-eps: C {row['anchored_eps_vs_anchor_pct']['carbon']:+.2f}% | "
              f"RAND-eps: C {row['chained_eps_random_vs_anchor_pct']['carbon']:+.2f}% "
              f"W {row['chained_eps_random_vs_anchor_pct']['scarcity']:+.2f}% "
              f"inband={row['chained_eps_random_within_declared_band']}")

    n_chain_fail = sum(1 for r in rows if not r["chained_certifies_vs_anchor"])
    n_anchor_fail = sum(1 for r in rows if not r["anchored_certifies_vs_anchor"])
    out = {
        "window": WINDOW, "epochs": args.epochs, "rot_slots": args.rot, "seed": args.seed,
        "eps_per_window": args.eps,
        "n_pods": n_pods, "n_nodes": n_nodes,
        "drift_model": "real 48-slot REALCI2 series, values rotated rot*k slots per epoch",
        "chained_fails_vs_anchor": n_chain_fail,
        "anchored_fails_vs_anchor": n_anchor_fail,
        "max_chained_excess_pct": {
            "carbon": max(r["chained_vs_anchor_pct"]["carbon"] for r in rows),
            "scarcity": max(r["chained_vs_anchor_pct"]["scarcity"] for r in rows),
        },
        "max_chained_eps_excess_pct": {
            "carbon": max(r["chained_eps_vs_anchor_pct"]["carbon"] for r in rows),
            "scarcity": max(r["chained_eps_vs_anchor_pct"]["scarcity"] for r in rows),
        },
        "max_anchored_eps_excess_pct": {
            "carbon": max(r["anchored_eps_vs_anchor_pct"]["carbon"] for r in rows),
            "scarcity": max(r["anchored_eps_vs_anchor_pct"]["scarcity"] for r in rows),
        },
        "max_chained_eps_random_excess_pct": {
            "carbon": max(r["chained_eps_random_vs_anchor_pct"]["carbon"] for r in rows),
            "scarcity": max(r["chained_eps_random_vs_anchor_pct"]["scarcity"] for r in rows),
        },
        "final_chained_eps_random_vs_anchor_pct": rows[-1]["chained_eps_random_vs_anchor_pct"],
        "rows": rows,
    }
    (OUT / "biocreep_replay.json").write_text(json.dumps(out, indent=2))
    print(f"\n[done] strict chained fails vs anchor {n_chain_fail}/{args.epochs}; "
          f"chained-eps max excess C {out['max_chained_eps_excess_pct']['carbon']:+.2f}% "
          f"W {out['max_chained_eps_excess_pct']['scarcity']:+.2f}%; "
          f"anchored-eps max C {out['max_anchored_eps_excess_pct']['carbon']:+.2f}%. "
          f"-> {OUT/'biocreep_replay.json'}")


if __name__ == "__main__":
    main()
