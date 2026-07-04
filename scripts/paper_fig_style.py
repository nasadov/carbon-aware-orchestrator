#!/usr/bin/env python3
"""Shared print-size style for every tsusc_draft figure (design-at-size rule).

Figures are authored at the EXACT printed size (IEEEtran 10pt journal: \\columnwidth = 252 TeX pt
= 3.487 in, \\textwidth = 516 TeX pt = 7.140 in) with absolute font sizes, and included in LaTeX
at natural width — never shrunk via a width= factor. Shrinking below authored size is what made
the pre-2026-07-03 figures illegible (fonts landed at 1.8–4.5 pt in print); do not reintroduce it.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEPLOY = [REPO / "docs" / "paper-three" / "Paper" / "figures", REPO / "experiments" / "figures"]

COLW = 3.487    # \columnwidth in inches (252 TeX pt / 72.27)
TEXTW = 7.140   # \textwidth in inches (516 TeX pt / 72.27)

# palette shared with the draft's TikZ/table colors
ENV_GREEN = "#1a9850"     # certified / envelope
ENV_ACCENT = "#534AB7"    # envaccent (draft preamble)
CARBON_BLUE = "#4575b4"   # carbon-greedy
WATER_RED = "#d73027"     # water-greedy / harm
WW_PURPLE = "#7b3294"     # WaterWise
AGG_LILAC = "#8a86d8"     # aggregate-guarded member (diagnostic level)
HARM_BG = "#d73027"       # use with alpha≈0.06 for harm bands
HOLD_BG = "#1a9850"       # use with alpha≈0.10 for certified bands


def apply() -> None:
    plt.rcParams.update({
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "pdf.fonttype": 42,   # embed TrueType (IEEE-safe)
        "ps.fonttype": 42,
        "figure.constrained_layout.use": True,
    })


def save(fig, name: str) -> None:
    """Deploy pdf+png to the paper and the experiments figure dirs at authored size."""
    for out in DEPLOY:
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{name}.pdf")
        fig.savefig(out / f"{name}.png", dpi=300)
    print(f"wrote {name}.pdf/.png at {fig.get_size_inches().round(3)} in to {len(DEPLOY)} dirs")
