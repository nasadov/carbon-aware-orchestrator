#!/usr/bin/env python3
"""Hero figure — the NO-HARM PLANE (authored at print size; see paper_fig_style).

Carbon Δ (x) vs scarcity-water Δ (y) versus the lifecycle-carbon packing baseline B (origin), on the
two real-trace testbeds. The third quadrant (ΔC≤0, ΔW≤0) is the *plotted* no-harm region — but the
certificate of record (Eq. 1) also checks SLO and every single watershed, so the figure is keyed to
the RECORD, not the plot: stars are the per-basin-guarded relief-first members (the certificate of
record's 12/12), and a red ring marks any point that fails the record — including deep in-quadrant
baseline points whose aggregate win hides a basin rise or dropped work (the burden-shift of Fig. 4).
Panel (b) carries a zoom inset because the GPU cluster is certified-near-neutral — the whole story
sits within ±0.6% of the origin and is invisible at the main axes' scale.

Data: experiments/water_basin/basin_certification.json (per-window per-method record verdicts; the
same committed source of truth behind Table S1) + the per-pod placement CSVs (engine-exact carbon
account for the per-basin-guarded members, identical to scripts/emit_paper_numbers.py).

SEMANTICS ASSERTIONS: this script fails loudly if what it encodes disagrees with the paper's counts
(unguarded methods 0/6 under the record; per-basin-guarded members 12/12) — so the figure can never
again silently drift from the certificate of record.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import paper_fig_style as sty
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[1]
BASIN = REPO / "experiments" / "water_basin"  # corrected-pipeline 6-window certification runs

AZURE_W = ["azure_packing_2020_d0_s42_200", "azure_packing_2020_d2_s43_200", "azure_packing_2020_d4_s44_200"]
ALIBABA_W = ["alibaba_gpu_2020_w1575_s42_200", "alibaba_gpu_2020_w1587_s43_200", "alibaba_gpu_2020_w1600_s44_200"]

# method_key -> (label, color, marker)
METHODS = {
    "no_harm_flex":  ("No-harm envelope (ours)", sty.ENV_GREEN, "*"),
    "carbon":        ("carbon-greedy",           sty.CARBON_BLUE, "o"),
    "water_scarcity":("water-greedy",            sty.WATER_RED, "X"),
    "waterwise":     ("WaterWise (SOTA)",        sty.WW_PURPLE, "D"),
}


def _pbg_carbon_pct(window: str, member: str, rec: dict) -> float:
    """Carbon Δ% of a per-basin-guarded member, from the committed per-pod placement CSVs —
    the identical engine-exact account used by scripts/emit_paper_numbers.py (PNdzeroPbgCarbon)."""
    out = BASIN / window / "out"
    base_c = {row["pod_id"]: float(row["carbon_kg"])
              for row in csv.DictReader(open(out / "placements_packing.csv"))}
    mem_pods = {row["pod_id"] for row in csv.DictReader(open(out / f"placements_{member}.csv"))}
    base_total = sum(v for p, v in base_c.items() if p in mem_pods)
    return 100.0 * rec["carbon_delta_kg"] / base_total


def load(window_names):
    """One record per window: the per-basin-guarded relief-first member (star) + the unguarded
    baselines, every point carrying its certificate-of-record verdict (engine ∧ per-basin)."""
    win = {r["window"]: r for r in json.load(open(BASIN / "basin_certification.json"))}
    out = []
    for w in window_names:
        r = win[w]
        pbg = r["per_basin_guarded"]["no_harm_flex"]
        star = {"carbon_delta_pct": _pbg_carbon_pct(w, "no_harm_flex", pbg),
                "scarcity_delta_pct": pbg["agg_delta_pct"],
                "record_certificate": bool(pbg["per_basin_cert"])}
        baselines = {}
        for k in ("carbon", "water_scarcity", "waterwise"):
            m = r["methods"][k]
            baselines[k] = {"carbon_delta_pct": m["engine_carbon_delta_pct"],
                            "scarcity_delta_pct": m["engine_scarcity_delta_pct"],
                            "record_certificate": bool(m["engine_no_harm_certificate"]) and bool(m["per_basin_cert"])}
        out.append({"window": w, "no_harm_flex": star, "baselines": baselines,
                    "pbg_all_members_cert": all(v["per_basin_cert"]
                                                for v in r["per_basin_guarded"].values())})
    return out


def points(digests, mkey):
    pts = []
    for d in digests:
        rec = d["no_harm_flex"] if mkey == "no_harm_flex" else d.get("baselines", {}).get(mkey)
        if rec:
            pts.append((rec["carbon_delta_pct"], rec["scarcity_delta_pct"],
                        bool(rec["record_certificate"])))
    return pts


def assert_record_semantics(digests):
    """The figure must encode the paper's counts: unguarded 0/N record-certified; stars N/N;
    every per-basin-guarded member (both rankings) certified."""
    n = len(digests)
    for mkey in ("carbon", "water_scarcity", "waterwise"):
        ncert = sum(c for *_, c in points(digests, mkey))
        assert ncert == 0, f"{mkey}: {ncert}/{n} record-certified — figure/text mismatch"
    star = points(digests, "no_harm_flex")
    assert sum(c for *_, c in star) == n, "per-basin-guarded member not certified on every window"
    assert all(x <= 0.0 and y <= 0.0 for x, y, _ in star), "star outside the no-harm quadrant"
    assert all(d["pbg_all_members_cert"] for d in digests), "a per-basin-guarded member fails Eq. (1)"


def draw_points(ax, digests, xlim, ylim, marker_scale=1.0):
    """Scatter all on-chart methods; return the off-chart set per method."""
    x0, x1 = xlim; y0, y1 = ylim
    offchart = []
    for mkey, (label, color, marker) in METHODS.items():
        pts = points(digests, mkey)
        if not pts:
            continue
        on = [(x, y, c) for (x, y, c) in pts if x0 <= x <= x1 and y0 <= y <= y1]
        off = [(x, y, c) for (x, y, c) in pts if not (x0 <= x <= x1 and y0 <= y <= y1)]
        size = (150 if marker == "*" else 42) * marker_scale
        if on:
            ax.scatter([p[0] for p in on], [p[1] for p in on], c=color, marker=marker, s=size,
                       edgecolors="k", linewidths=0.5, zorder=4, alpha=0.95)
            for x, y, c in on:
                if not c:
                    ax.scatter([x], [y], s=size * 2.1, facecolors="none", edgecolors=sty.WATER_RED,
                               linewidths=1.1, zorder=3)
        if off:
            offchart.append((label, color, off))
    return offchart


def panel(ax, digests, title, xlim, ylim):
    x0, x1 = xlim; y0, y1 = ylim
    ax.add_patch(Rectangle((x0, y0), -x0, -y0, color=sty.HOLD_BG, alpha=0.10, zorder=0, lw=0))
    ax.add_patch(Rectangle((0, y0), x1, y1 - y0, color=sty.HARM_BG, alpha=0.05, zorder=0, lw=0))
    offchart = draw_points(ax, digests, xlim, ylim)
    for label, color, off in offchart:  # water-greedy: off-chart arrow + summary
        cx = [round(p[0]) for p in off]
        ax.annotate("", xy=(x1, y0 + 0.30 * (y1 - y0)), xytext=(x1 - 0.18 * (x1 - x0), y0 + 0.30 * (y1 - y0)),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4))
        ax.text(x1 - 0.02 * (x1 - x0), y0 + 0.36 * (y1 - y0),
                f"{label}\n+{min(cx)}…{max(cx)}% C,\nnever certifies",
                color=color, fontsize=6.8, ha="right", va="bottom", weight="bold")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.axhline(0, color="k", lw=0.8, zorder=2)
    ax.axvline(0, color="k", lw=0.8, zorder=2)
    ax.set_xlabel("carbon $\\Delta$ vs baseline (%)   ($\\leq$ 0 = no carbon harm)")
    ax.set_ylabel("scarcity-water $\\Delta$ (%)   ($\\leq$ 0 = no water harm)")
    ax.set_title(title)
    ax.grid(alpha=0.22, zorder=1)
    ax.text(0.03, 0.04, "CERTIFIED\nNO-HARM", transform=ax.transAxes, fontsize=7.5,
            color="#13602f", weight="bold", ha="left", va="bottom")


def annotate_burden_shift(ax, digests):
    """Point at the deepest in-quadrant ringed carbon-greedy point: an aggregate win that fails
    the record by raising a single watershed — exactly the burden-shift Fig. 4 dissects."""
    ringed = [(x, y) for x, y, c in points(digests, "carbon")
              if not c and x < 0 and y < 0]
    if not ringed:
        return
    tx, ty = min(ringed, key=lambda p: p[0] + p[1])  # deepest into the quadrant
    ax.annotate("in the quadrant, yet uncertified:\nthe sum hides a basin rise (Fig. 4)",
                xy=(tx, ty), xytext=(0.05, 0.60), textcoords="axes fraction",
                fontsize=6.8, color=sty.WATER_RED, weight="bold", ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=sty.WATER_RED, lw=1.0,
                                connectionstyle="arc3,rad=0.15"))


def main():
    sty.apply()
    az, al = load(AZURE_W), load(ALIBABA_W)
    if not az or not al:
        raise SystemExit("missing digests — run scripts/run_water_basin_certification.py first")
    assert_record_semantics(az + al)
    fig, axes = plt.subplots(1, 2, figsize=(sty.TEXTW, 2.80))
    panel(axes[0], az, f"(a) Azure — general cloud (CPU/RAM), {len(az)} windows",
          xlim=(-30, 14), ylim=(-46, 6))
    annotate_burden_shift(axes[0], az)
    panel(axes[1], al, f"(b) Alibaba — GPU/AI cluster, {len(al)} windows",
          xlim=(-6, 5), ylim=(-9.5, 6))
    # zoom inset: the near-neutral GPU cluster (everything within ±0.6% of the origin);
    # positioned clear of the WaterWise point above and the carbon-greedy point below,
    # with ticks on the right so the labels do not collide with the main axis's ticks
    axi = axes[1].inset_axes([0.055, 0.50, 0.34, 0.38], xlim=(-0.65, 0.55), ylim=(-0.65, 0.55))
    axi.add_patch(Rectangle((-0.65, -0.65), 0.65, 0.65, color=sty.HOLD_BG, alpha=0.10, zorder=0, lw=0))
    axi.axhline(0, color="k", lw=0.5); axi.axvline(0, color="k", lw=0.5)
    draw_points(axi, al, (-0.65, 0.55), (-0.65, 0.55), marker_scale=0.8)
    axi.yaxis.tick_right()
    axi.xaxis.tick_top()
    axi.tick_params(labelsize=6, pad=1.5)
    axi.set_xticks([-0.5, 0, 0.5]); axi.set_yticks([-0.5, 0, 0.5])
    axes[1].indicate_inset_zoom(axi, edgecolor="gray", lw=0.7)
    # shared, de-duplicated legend below the panels
    handles, labels = [], []
    for mkey, (label, color, marker) in METHODS.items():
        handles.append(mlines.Line2D([], [], color=color, marker=marker, linestyle="None",
                                     markersize=9 if marker == "*" else 6, markeredgecolor="k",
                                     markeredgewidth=0.5))
        labels.append(label)
    handles.append(mlines.Line2D([], [], color=sty.WATER_RED, marker="o", markerfacecolor="none",
                                 markeredgewidth=1.1, linestyle="None", markersize=7))
    labels.append("uncertified (axis, basin, or SLO)")
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.005))
    fig.get_layout_engine().set(rect=(0, 0.06, 1, 0.94))
    sty.save(fig, "noharm_plane")
    for tag, ds in (("Azure", az), ("Alibaba", al)):
        print(f"\n{tag} (certificate of record = engine ∧ per-basin):")
        for mkey, (label, *_ ) in METHODS.items():
            pts = points(ds, mkey)
            if pts:
                ncert = sum(c for *_, c in pts)
                print(f"  {label:26} record-cert {ncert}/{len(pts)}  "
                      f"carbonΔ {[round(p[0],1) for p in pts]}  waterΔ {[round(p[1],1) for p in pts]}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
