#!/usr/bin/env python3
"""Generalization figure — Azure (general cloud, CPU/RAM) vs Alibaba (AI cluster, GPU).

Reads the per-window no-harm digests written by run_azure_no_harm.py (T3) and
run_alibaba_no_harm.py (T12) and renders a 3-panel deck figure that tells the
generalization story:

  A. Deferrable (flexible) share spectrum: Azure ~21-35% (evictable minority) vs
     Alibaba ~99-100% (batch training) — the envelope brackets the whole range.
  B. No-harm guarantee holds on BOTH: the certified flex schedule's realized
     carbon & scarcity deltas are <= 0 on every window (the certificate is the point).
  C. Water<->carbon tension (water-only baseline): the naive single-objective
     water-greedy schedule inflates carbon (uncertified), and it is far worse on
     GPU clusters (GPU power x dry-cooling PUE).

The certificate (panel B) being non-trivial is what makes Azure the hard case:
only ~1-in-4 pods can move, yet no harm is still guaranteed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
FLEX = REPO_ROOT / "experiments" / "flexibility"

# (run-name dir, digest filename, short label)
AZURE = [
    ("t3_azure_packing_2020_d0_s42_200", "t3_digest.json", "az/w0"),
    ("t3_azure_packing_2020_d2_s43_200", "t3_digest.json", "az/w2"),
    ("t3_azure_packing_2020_d4_s44_200", "t3_digest.json", "az/w4"),
]
ALIBABA = [
    ("t12_alibaba", "t12_digest.json", "ali/w1575"),
    ("t12_alibaba_w1587", "t12_digest.json", "ali/w1587"),
    ("t12_alibaba_w1600", "t12_digest.json", "ali/w1600"),
]


def _load(specs) -> List[Dict]:
    out = []
    for run, fname, label in specs:
        p = FLEX / run / "out" / fname
        if not p.exists():
            raise SystemExit(f"missing digest {p} — run the corresponding sweep first")
        d = json.loads(p.read_text())
        d["_label"] = label
        out.append(d)
    return out


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    az = _load(AZURE)
    al = _load(ALIBABA)

    AZC, ALC = "#1f77b4", "#ff7f0e"  # azure blue / alibaba orange

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    plt.rcParams.update({"axes.titlesize": 11.5, "axes.labelsize": 10.5})

    # ---- Panel A: deferrable share spectrum ----
    ax = axes[0]
    labels = [d["_label"] for d in az] + [d["_label"] for d in al]
    shares = [d["flexible_share_pct"] for d in az] + [d["flexible_share_pct"] for d in al]
    colors = [AZC] * len(az) + [ALC] * len(al)
    x = np.arange(len(labels))
    ax.bar(x, shares, color=colors)
    for xi, s in zip(x, shares):
        ax.text(xi, s + 1.5, f"{s:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("deferrable (flexible) share %"); ax.set_ylim(0, 112)
    ax.set_title("A. Flexibility spectrum\ngeneral cloud (Azure) vs AI cluster (Alibaba)")
    az_m, al_m = _mean([d["flexible_share_pct"] for d in az]), _mean([d["flexible_share_pct"] for d in al])
    ax.axhline(az_m, color=AZC, ls="--", lw=1, alpha=0.7)
    ax.axhline(al_m, color=ALC, ls="--", lw=1, alpha=0.7)
    ax.grid(axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=AZC, label=f"Azure (mean {az_m:.0f}%)"),
                       Patch(color=ALC, label=f"Alibaba (mean {al_m:.0f}%)")], fontsize=8.5, loc="upper left")

    # ---- Panel B: water<->carbon tension (uncertified water-only baseline carbon penalty) ----
    # (The certification story itself is the no-harm plane, Fig. 2; here we show the tension
    #  magnitude per testbed so the two figures are complementary, not redundant.)
    ax = axes[1]
    pen = [d["baselines"]["water_scarcity"]["carbon_delta_pct"] for d in az + al]
    certs = [d["no_harm_flex"]["no_harm_certificate"] for d in az + al]
    ax.bar(x, pen, color=colors)
    top = max(pen)
    for xi, p in zip(x, pen):
        ax.text(xi, p + top * 0.02, f"+{p:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="k", lw=0.9)
    ax.set_ylim(0, top * 1.16)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("carbon inflation of naive water-only schedule (%)")
    ax.set_title("B. Water↔carbon tension (uncertified water-only baseline)\n"
                 "inflates carbon on both — magnitude is testbed-dependent")
    ax.grid(axis="y", alpha=0.3)
    note = "envelope certifies all 6 windows" if all(certs) else "check certificates"
    ax.text(0.5, 0.93, f"({note})", transform=ax.transAxes, ha="center",
            fontsize=8.5, style="italic", color="#1a6b3a")

    fig.suptitle("No-Harm Flexibility Envelope — generalization across resource classes & workload mixes",
                 fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FLEX / "azure_alibaba_contrast.png"
    fig.savefig(out, dpi=150); plt.close(fig)

    # ---- compact textual digest for the report ----
    print("=== Azure vs Alibaba generalization ===")
    print(f"Azure  flexible share: {[round(d['flexible_share_pct'],1) for d in az]}  (mean {az_m:.1f}%)")
    print(f"Alibaba flexible share: {[round(d['flexible_share_pct'],1) for d in al]}  (mean {al_m:.1f}%)")
    print(f"no-harm flex cert — Azure: {[d['no_harm_flex']['no_harm_certificate'] for d in az]}  "
          f"Alibaba: {[d['no_harm_flex']['no_harm_certificate'] for d in al]}")
    print(f"water-only carbon penalty — Azure: {[round(d['baselines']['water_scarcity']['carbon_delta_pct']) for d in az]}%  "
          f"Alibaba: {[round(d['baselines']['water_scarcity']['carbon_delta_pct']) for d in al]}%")
    print(f"\nfigure -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
