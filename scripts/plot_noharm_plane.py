#!/usr/bin/env python3
"""Hero figure — the NO-HARM PLANE.

Carbon Δ (x) vs scarcity-water Δ (y) versus the lifecycle-carbon packing baseline B (origin), on the
two real-trace testbeds. The third quadrant (ΔC≤0, ΔW≤0) is the *certified no-harm* region: only a
schedule landing there has degraded no axis. We plot every method on every window; the envelope is
the only method inside the no-harm region on every window, while the single-objective and SOTA
co-optimizer baselines escape it (water-greedy buys water by inflating carbon; WaterWise can land in
the harm region too). Reads the per-window digests written by run_azure_no_harm / run_alibaba_no_harm.
"""
from __future__ import annotations
import csv
import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FLEX = REPO / "experiments" / "flexibility"
BASIN = REPO / "experiments" / "water_basin"  # corrected-pipeline 6-window certification runs
OUT = REPO / "experiments" / "figures"

AZURE_W = ["azure_packing_2020_d0_s42_200", "azure_packing_2020_d2_s43_200", "azure_packing_2020_d4_s44_200"]
ALIBABA_W = ["alibaba_gpu_2020_w1575_s42_200", "alibaba_gpu_2020_w1587_s43_200", "alibaba_gpu_2020_w1600_s44_200"]

# method_key -> (label, color, marker)
METHODS = {
    "no_harm_flex":  ("No-harm envelope (ours)", "#1a9850", "*"),
    "carbon":        ("carbon-greedy",           "#4575b4", "o"),
    "water_scarcity":("water-greedy",            "#d73027", "X"),
    "waterwise":     ("WaterWise (SOTA)",        "#7b3294", "D"),
}


def load(window_names):
    """Load per-window method records from the corrected basin-certification runs' summary.csv
    (engine source of truth: Option A water, exact-guard engine, honest WaterWise dispatch)."""
    def conv(r):
        return {"carbon_delta_pct": float(r["carbon_delta_pct"]),
                "scarcity_delta_pct": float(r["scarcity_delta_pct"]),
                "no_harm_certificate": r["no_harm_certificate"] == "True"}
    out = []
    for w in window_names:
        f = BASIN / w / "out" / "summary.csv"
        if not f.exists():
            continue
        rows = {r["method_key"]: r for r in csv.DictReader(open(f))}
        out.append({"no_harm_flex": conv(rows["no_harm_flex"]),
                    "baselines": {k: conv(rows[k]) for k in ("carbon", "water_scarcity", "waterwise")
                                  if k in rows}})
    return out


def points(digests, mkey):
    """Return [(carbonΔ, waterΔ, certified), ...] for method mkey across windows."""
    pts = []
    for d in digests:
        rec = d["no_harm_flex"] if mkey == "no_harm_flex" else d.get("baselines", {}).get(mkey)
        if not rec:
            continue
        pts.append((rec["carbon_delta_pct"], rec["scarcity_delta_pct"],
                    bool(rec["no_harm_certificate"])))
    return pts


def panel(ax, digests, title, xlim, ylim):
    from matplotlib.patches import Rectangle
    x0, x1 = xlim; y0, y1 = ylim
    # certified no-harm quadrant (x<=0, y<=0) green; carbon-harm half (x>0) faint red
    ax.add_patch(Rectangle((x0, y0), -x0, -y0, color="#1a9850", alpha=0.10, zorder=0, lw=0))
    ax.add_patch(Rectangle((0, y0), x1, y1 - y0, color="#d73027", alpha=0.05, zorder=0, lw=0))
    offchart = []
    for mkey, (label, color, marker) in METHODS.items():
        pts = points(digests, mkey)
        if not pts:
            continue
        on = [(x, y, c) for (x, y, c) in pts if x0 <= x <= x1 and y0 <= y <= y1]
        off = [(x, y, c) for (x, y, c) in pts if not (x0 <= x <= x1 and y0 <= y <= y1)]
        size = 360 if marker == "*" else 120
        if on:
            ax.scatter([p[0] for p in on], [p[1] for p in on], c=color, marker=marker, s=size,
                       edgecolors="k", linewidths=0.7, label=label, zorder=4, alpha=0.95)
            for x, y, c in on:
                if not c:
                    ax.scatter([x], [y], s=size * 1.8, facecolors="none", edgecolors="#d73027",
                               linewidths=1.9, zorder=3)
        if off:
            offchart.append((label, color, off))
    # off-chart methods (water-greedy): arrow to the right edge + summary
    for i, (label, color, off) in enumerate(offchart):
        cx = [round(p[0]) for p in off]; cy = [round(p[1]) for p in off]
        ax.annotate("", xy=(x1, y0 + 0.30 * (y1 - y0)), xytext=(x1 - 0.18 * (x1 - x0), y0 + 0.30 * (y1 - y0)),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2))
        ax.text(x1, y0 + 0.37 * (y1 - y0),
                f"{label}\n+{min(cx)}…{max(cx)}% C, never certifies",
                color=color, fontsize=8.2, ha="right", va="bottom", weight="bold")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.axhline(0, color="k", lw=1.0, zorder=2)
    ax.axvline(0, color="k", lw=1.0, zorder=2)
    ax.set_xlabel("carbon Δ vs baseline (%)   (≤ 0 = no carbon harm)")
    ax.set_ylabel("scarcity-water Δ (%)   (≤ 0 = no water harm)")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.22, zorder=1)
    ax.text(0.035, 0.035, "CERTIFIED\nNO-HARM", transform=ax.transAxes, fontsize=9.5,
            color="#13602f", weight="bold", ha="left", va="bottom")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    az, al = load(AZURE_W), load(ALIBABA_W)
    if not az or not al:
        raise SystemExit("missing digests — run scripts/run_water_basin_certification.py first")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    panel(axes[0], az, f"(a) Azure — general cloud (CPU/RAM), {len(az)} windows",
          xlim=(-30, 14), ylim=(-46, 6))
    panel(axes[1], al, f"(b) Alibaba — GPU/AI cluster, {len(al)} windows",
          xlim=(-6, 5), ylim=(-9.5, 6))
    # shared, de-duplicated legend below the panels
    import matplotlib.lines as mlines
    handles, labels = [], []
    for mkey, (label, color, marker) in METHODS.items():
        handles.append(mlines.Line2D([], [], color=color, marker=marker, linestyle="None",
                                     markersize=13 if marker == "*" else 10, markeredgecolor="k",
                                     markeredgewidth=0.7))
        labels.append(label)
    handles.append(mlines.Line2D([], [], color="#d73027", marker="o", markerfacecolor="none",
                                 markeredgewidth=1.9, linestyle="None", markersize=12))
    labels.append("uncertified (harms an axis)")
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1.0))
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"noharm_plane.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'noharm_plane.png'}")
    # textual digest
    for tag, ds in (("Azure", az), ("Alibaba", al)):
        print(f"\n{tag}:")
        for mkey, (label, *_ ) in METHODS.items():
            pts = points(ds, mkey)
            if pts:
                ncert = sum(c for *_, c in pts)
                print(f"  {label:26} cert {ncert}/{len(pts)}  "
                      f"carbonΔ {[round(p[0],1) for p in pts]}  waterΔ {[round(p[1],1) for p in pts]}")


if __name__ == "__main__":
    raise SystemExit(main())
