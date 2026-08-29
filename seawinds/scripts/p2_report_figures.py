#!/usr/bin/env python
"""Render the report figures into submission/report/figures/.

Print deliverable (a 6-12 pp PDF), so: light surface only, no interaction layer,
vector PDF + a PNG preview for each figure.

Palette: the documented default categorical order, slots 1-3 only
(blue / orange / aqua) — that subset is the one validated for all pairs, and no
panel here carries more than three series. Hues are used unchanged; text wears
text tokens, never a series colour. Every series is direct-labelled as well as
legended, which also discharges the relief rule for the aqua slot on a light
surface. No dual axes anywhere: where two measures have different scales they get
their own stacked panel sharing the x axis.

Figures
  fig1_pipeline        the end-to-end pipeline flowchart (graphviz)
  fig2_ablation        attributable gain per build step, speed and direction panels
  fig3_cannibalisation measured NL capture rate / negative-price hours / offshore output
  fig4_tau_sweep       realised day-ahead revenue vs bid quantile
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.environ.get("SEAWINDS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = f"{ROOT}/submission/report/figures"
os.makedirs(FIG, exist_ok=True)

# --- design tokens (documented default palette, light mode) -------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dedcd6"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"      # slots 1-3

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 8.5, "font.family": "DejaVu Sans",
    "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 160,
})


def _clean(ax, ybase=False):
    """Recessive axes: drop the box, keep a light horizontal grid."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    if ybase:
        ax.set_ylim(bottom=0)


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# ---------------------------------------------------------------- fig 1
DOT = r"""
digraph pipeline {
  rankdir=TB; splines=ortho; nodesep=0.30; ranksep=0.55; bgcolor="%(sf)s";
  node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=10
        color="%(grid)s" fillcolor="white" fontcolor="%(ink)s" margin="0.14,0.09"];
  edge [color="%(ink2)s" arrowsize=0.6 penwidth=0.9];

  subgraph cluster_in {
    label="INPUTS — coarse only (AROME is never an input)";
    fontname="Helvetica-Bold"; fontsize=10; fontcolor="%(ink2)s";
    color="%(grid)s"; style=rounded;
    R [label="Reanalysis 0.25°\nu10 v10 u100 v100\n(14-day context)"];
    H [label="ECMWF HRES 0.25°\nfcst_speed / fcst_dir\nd+1, d+7"];
    E [label="ERA5 initial state\nat context_end"];
    B [label="Static + EMODnet\nbathymetry"];
  }
  subgraph cluster_fe {
    label="FEATURES"; fontname="Helvetica-Bold"; fontsize=10;
    fontcolor="%(ink2)s"; color="%(grid)s"; style=rounded;
    F [label="fcst_u, fcst_v, fcst_speed\nlat, lon · woy_sin, woy_cos\nctx_shear, ctx_veer"];
  }
  subgraph cluster_m {
    label="MODELS — one per (lead, variable), selected by leave-year-out CV";
    fontname="Helvetica-Bold"; fontsize=10; fontcolor="%(ink2)s";
    color="%(grid)s"; style=rounded;
    M1 [label="LightGBM quantile MOS\nτ = .05/.50/.95 → SPEED" fillcolor="%(s1l)s"];
    M2 [label="LightGBM mean MOS\nu, v → DIRECTION" fillcolor="%(s1l)s"];
    M3 [label="Month × hour CLIMATOLOGY\n(fitted on the training years)" fillcolor="%(s2l)s"];
    M4 [label="Pangu-Weather ONNX\n8-member IC ensemble\n→ ensemble MEAN" fillcolor="%(s3l)s"];
  }
  subgraph cluster_uq {
    label="UNCERTAINTY"; fontname="Helvetica-Bold"; fontsize=10;
    fontcolor="%(ink2)s"; color="%(grid)s"; style=rounded;
    U1 [label="Conformal quantile\nregression (α = 0.10)"];
    U2 [label="Winkler-optimal circular arc\nd+1 speed-conditioned\nd+7/d+14 marginal"];
  }
  SEL [label="PER-SUB-DIMENSION SELECTION\nspeed d+1 ← MOS  ·  speed d+7/d+14 ← climatology\ndir d+1/d+7/d+14 ← ensemble-mean centre"
       shape=box style="rounded,filled" fillcolor="#eef3fb" penwidth=1.2];
  D  [label="Bilinear 0.25° → 1.3 km\n43,715 footprint points"];
  P  [label="predictions.csv\n4,196,640 rows" penwidth=1.4];
  W  [label="PyWake · Bastankhah-Gaussian\n12×30° sectors · Charnock TI\nα = 0.11 shear 125→170 m"];
  L  [label="Worst-year site choice\n+ layout search\n+ 60 synthetic years"];
  SJ [label="submission.json\n55 × IEA 22 MW" penwidth=1.4];
  V  [label="Day-ahead market simulation\nPinson τ* quantile bidding"];

  R -> F; H -> F; E -> M4; F -> M1; F -> M2;
  M1 -> U1; M2 -> U2; M3 -> U2; M4 -> U2;
  U1 -> SEL; U2 -> SEL; SEL -> D -> P;
  B -> W; R -> W [style=dashed]; W -> L -> SJ;
  P -> V; SJ -> V;
}
""" % dict(sf=SURFACE, ink=INK, ink2=INK2, grid=GRID,
           s1l="#eaf1fc", s2l="#fdeee7", s3l="#e7f7f1")


def fig1():
    src = f"{FIG}/fig1_pipeline.dot"
    open(src, "w").write(DOT)
    for fmt in ("pdf", "png"):
        subprocess.run(["dot", f"-T{fmt}", src, "-o", f"{FIG}/fig1_pipeline.{fmt}"],
                       check=True)
    print("  wrote fig1_pipeline.pdf / .png")


# ---------------------------------------------------------------- fig 2
STEPS = ["kit", "v1", "v2", "v3", "v5raw", "v6ens", "v9"]
STEP_NOTE = ("v1 quantile MOS + climatology   ·   v2 coverage recalibration   ·   "
             "v3 MOS retrain + climatology widening   ·   v5raw Pangu d+7 direction   ·   "
             "v6ens 8-member ensemble-mean direction centre   ·   v9 speed-conditioned d+1 arc")
SPD = {"d+1": [9.2, 9.21, 8.38, 8.38, 8.38, 8.38, 8.38],
       "d+7": [29.8, 19.86, 19.72, 19.13, 19.13, 19.13, 19.13],
       "d+14": [40.1, 19.29, 18.98, 18.94, 18.94, 18.94, 18.94]}
DIR = {"d+1": [173, 182, 182, 171, 171, 171.5, 121.5],
       "d+7": [312, 336, 336, 329, 311.8, 293.9, 293.9],
       "d+14": [334, 343, 343, 343, 342.5, 298.0, 298.0]}
COL = {"d+1": S1, "d+7": S2, "d+14": S3}


def _stagger(ax, items, gap_frac=0.055):
    """Direct-label line ends, pushing labels apart when the lines converge."""
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * gap_frac
    items = sorted(items, key=lambda t: t[0])          # (y, text, colour)
    ys = [t[0] for t in items]
    for i in range(1, len(ys)):                        # separate upward
        if ys[i] - ys[i - 1] < gap:
            ys[i] = ys[i - 1] + gap
    xend = len(STEPS) - 1
    for (yv, txt, col), yl in zip(items, ys):
        # anchor INSIDE the axes (xlim is extended by the caller) so a label can
        # never spill into the neighbouring panel
        ax.annotate(txt, xy=(xend + 0.18, yl), xycoords="data",
                    va="center", ha="left", fontsize=8.2, color=INK2,
                    fontweight="bold")
        if abs(yl - yv) > gap * 0.4:                 # leader line to the true value
            ax.plot([xend + 0.02, xend + 0.15], [yv, yl], color=GRID, lw=0.7,
                    zorder=2, clip_on=False)


def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6),
                             gridspec_kw=dict(wspace=0.30))
    x = np.arange(len(STEPS))
    for ax, data, ttl, unit in (
            (axes[0], SPD, "Wind speed — Winkler score", "m s⁻¹"),
            (axes[1], DIR, "Wind direction — circular Winkler", "degrees")):
        for k, v in data.items():
            ax.plot(x, v, "-o", color=COL[k], lw=2.0, ms=5.5,
                    mec=SURFACE, mew=1.2, label=k, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(STEPS, fontsize=8)
        ax.set_title(ttl, fontsize=9.5, loc="left", pad=6)
        ax.set_ylabel(f"score ({unit}) — lower is better", fontsize=8)
        _clean(ax)
        ax.set_xlim(-0.35, len(STEPS) + 0.95)
        ax.margins(y=0.16)
        _stagger(ax, [(v[-1], f"{k}  {v[-1]:.1f}", COL[k]) for k, v in data.items()])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="horizon", fontsize=8.2, title_fontsize=8.2,
               loc="upper right", bbox_to_anchor=(0.995, 1.10), ncol=3,
               columnspacing=1.2)
    fig.suptitle("Ablation: attributable effect of each build step on the hidden 2021 year",
                 fontsize=10.5, x=0.005, ha="left", y=1.10, color=INK)
    fig.text(0.005, -0.10, STEP_NOTE, fontsize=7, color=INK2, ha="left")
    save(fig, "fig2_ablation")


# ---------------------------------------------------------------- fig 3
YRS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
CAPR = [0.977, 1.001, 0.958, 0.837, 0.881, 0.899, 0.932]
NEGH = [3, 97, 70, 87, 321, 463, 579]
OFFS = [402, 543, 875, 911, 1320, 1734, 1790]


def fig3():
    fig, axes = plt.subplots(3, 1, figsize=(5.6, 5.4), sharex=True,
                             gridspec_kw=dict(hspace=0.28))
    a, b, c = axes
    a.plot(YRS, CAPR, "-o", color=S1, lw=2.0, ms=6, mec=SURFACE, mew=1.2, zorder=3)
    a.axhline(1.0, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)
    a.annotate("parity with the time-weighted price", (2025.35, 1.0),
               fontsize=7.2, color=INK2, va="center", ha="right")
    a.set_ylim(top=1.03)
    a.set_ylabel("capture rate")
    a.set_title("Dutch offshore wind: measured cannibalisation, 2019–2025",
                fontsize=10, loc="left", pad=8)
    _clean(a)

    b.bar(YRS, NEGH, color=S2, width=0.62, zorder=3)
    for xx, yy in zip(YRS, NEGH):
        b.annotate(f"{yy}", (xx, yy), textcoords="offset points", xytext=(0, 3),
                   ha="center", fontsize=7.2, color=INK2)
    b.set_ylabel("hours with\nnegative prices")
    _clean(b, ybase=True)

    c.bar(YRS, OFFS, color=S3, width=0.62, zorder=3)
    c.set_ylabel("mean offshore\noutput (MW)")
    c.set_xlabel("year")
    _clean(c, ybase=True)
    c.set_xticks(YRS)

    fig.text(0.005, -0.035,
             "Source: energy-charts.info (Bundesnetzagentur / SMARD), CC BY 4.0 — "
             "hourly day-ahead prices and public power, 2018–2026.",
             fontsize=7, color=INK2, ha="left")
    save(fig, "fig3_cannibalisation")


# ---------------------------------------------------------------- fig 4
def fig4():
    # market_value_final_s0, NOT market_value_2020_s0: the latter was computed on the
    # pre- layout and records it as superseded. The adjacent table in report §3.4
    # cites the final run, so a figure on the old one is a figure/table contradiction.
    p = f"{ROOT}/scripts/results/market_value_final_s0/market_value.json"
    d = json.load(open(p))
    sw = {float(k): v / 1e6 for k, v in d["tau_sweep"].items()}
    taus = sorted(sw)
    rev = [sw[t] for t in taus]
    tstar = d["headline"]["_tau_star"]
    tbest = float(d["tau_best_realised"])

    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    ax.plot(taus, rev, "-", color=S1, lw=2.2, zorder=3)
    # the two markers sit close together on a flat maximum -> place their labels on
    # opposite sides with leader lines rather than stacking them
    for t, lbl, col, off, ha in (
            # both labels go RIGHT into the empty area under/over the falling limb; a
            # left-going label at tau*~0.37 runs off the axis into the y-label (measured
            # in the rendered raster, invisible in the code)
            (tstar, f"Pinson closed form  τ* = {tstar:.2f}", S2, (0, -46), "center"),
            (tbest, f"realised optimum  τ = {tbest:.2f}", S3, (30, 12), "left")):
        y = np.interp(t, taus, rev)
        ax.plot([t], [y], "o", ms=8, color=col, mec=SURFACE, mew=1.4, zorder=4)
        ax.annotate(lbl, (t, y), textcoords="offset points", xytext=off,
                    ha=ha, va="center", fontsize=8, color=INK2, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8,
                                    shrinkA=0, shrinkB=6))
    ax.margins(y=0.16)
    ax.set_xlabel("bid quantile τ")
    ax.set_ylabel("realised revenue (M€ / year)")
    ax.set_title("Day-ahead revenue vs bid quantile (2020, out-of-sample)",
                 fontsize=10, loc="left", pad=8)
    _clean(ax)
    save(fig, "fig4_tau_sweep")


if __name__ == "__main__":
    print(f"rendering into {FIG}")
    fig1()
    fig2()
    fig3()
    fig4()
    print("done")
    sys.exit(0)
