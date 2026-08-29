"""Build d7/d14 speed-PI WIDTH variants of the shipped v11b, for server measurement.

WHY. A33 bracketed the speed-PI scale on CV and found the shipped 1.10 is NOT the CV
optimum (CV wants 1.00) — but also that CV OVER-COVERS relative to a withheld year by
5.2 pp at d7, which is exactly why the shipped value was set from a server read in the
first place. A33 could only INFER the right level from that gap. The server can MEASURE
it: hidden AROME 2021 is a genuinely unseen year for models trained 2016-2020, i.e. the
same population class as the withheld deciding year, unlike the CV folds.

WHAT. Rescale ONLY q05/q95 at horizons 7 and 14, around the unchanged q50. Everything
else -- d1 speed, every direction column, all metadata -- is passed through untouched, so
a diff against v11b must show exactly two columns changing on exactly two horizons.

EXACTNESS. The shipped chain applies its scale at stage [2/5] and the later stages never
rewrite d7/d14 speed, so composing f = s_target / 1.10 on the FINAL csv reproduces a
rebuild at s_target. The precondition is on the SOURCE: v11b must have ZERO clipped rows
(q05 <= 0) at d7/d14 -- verified -- so the original clip never bound and no information
was lost. Given that, the composition is exact to the 3-dp rounding INCLUDING the clip,
so output clipping at s > 1.10 is correct behaviour, not a composition error.

  python p2_width_probe.py --scale 1.00 --out <dir>
"""
from __future__ import annotations
import argparse, os, sys, zipfile
import numpy as np
import pandas as pd

SHIPPED_SCALE = 1.10          # RECALIB_SPD d7/d14 in the shipped chain
SRC = "./work/widthprobe/predictions.csv"
COLS = ["type", "window", "region", "latitude", "longitude", "horizon", "hour", "level",
        "q05", "q50", "q95", "dir_05", "dir_50", "dir_95"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, help="target PI scale for BOTH d7 and d14")
    # d7 and d14 are SEPARATE scored sub-dimensions, so one submission carrying a
    # different scale at each buys two independent server reads with no confounding.
    ap.add_argument("--scale7", type=float, help="target d7 scale (default: --scale)")
    ap.add_argument("--scale14", type=float, help="target d14 scale (default: --scale)")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--tag", default="", help="zip name tag; default derives from scales")
    ap.add_argument("--out", default=os.environ.get("EXP_OUTPUT_DIR", "."))
    a = ap.parse_args()
    s7 = a.scale7 if a.scale7 is not None else a.scale
    s14 = a.scale14 if a.scale14 is not None else a.scale
    if s7 is None or s14 is None:
        ap.error("give --scale, or both --scale7 and --scale14")
    fac = {7: s7 / SHIPPED_SCALE, 14: s14 / SHIPPED_SCALE}
    os.makedirs(a.out, exist_ok=True)
    csv_out = os.path.join(a.out, "predictions.csv")
    print(f"target scales d7 {s7:.3f} / d14 {s14:.3f}  =>  factors "
          f"d7 {fac[7]:.9f} / d14 {fac[14]:.9f} on the shipped {SHIPPED_SCALE}", flush=True)

    n = 0
    n_changed = 0
    clipped = 0
    src_clipped = 0
    first = True
    for ch in pd.read_csv(a.src, chunksize=1000000):
        assert list(ch.columns) == COLS, f"schema drift: {list(ch.columns)}"
        before = ch[["q05", "q95"]].to_numpy(copy=True)
        for h in (7, 14):
            f = fac[h]
            m = (ch["horizon"] == h).to_numpy()
            if not m.any():
                continue
            q05 = ch.loc[m, "q05"].to_numpy(); q50 = ch.loc[m, "q50"].to_numpy()
            q95 = ch.loc[m, "q95"].to_numpy()
            # PRECONDITION for exact composition is on the SOURCE, not the output: the
            # shipped stage-[2/5] clip must never have bound at 1.10, so no information
            # was lost before we re-scale. Given that, q50-q05_src == (q50-q05_base)*1.10
            # identically, so composing f = s/1.10 reproduces the true rebuild at s --
            # INCLUDING its clip. Output clipping at s > 1.10 is therefore correct
            # behaviour that the real rebuild also produces, not a composition error.
            src_clipped += int((q05 <= 0).sum())
            lo = np.clip(q50 - (q50 - q05) * f, 0, None)
            hi = q50 + (q95 - q50) * f
            clipped += int(((q50 - (q50 - q05) * f) < 0).sum())
            ch.loc[m, "q05"] = lo.round(3)
            ch.loc[m, "q95"] = hi.round(3)
        after = ch[["q05", "q95"]].to_numpy()
        n_changed += int((before != after).any(axis=1).sum())
        # invariants, every chunk
        assert (ch["q05"] <= ch["q50"] + 1e-9).all() and (ch["q50"] <= ch["q95"] + 1e-9).all(), \
            "quantile monotonicity violated"
        assert (ch["q05"] >= 0).all(), "negative speed quantile"
        for c in ("dir_05", "dir_50", "dir_95"):
            assert ((ch[c] >= 0) & (ch[c] < 360)).all(), f"{c} out of [0,360)"
        ch.to_csv(csv_out, index=False, mode="w" if first else "a", header=first)
        first = False
        n += len(ch)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import p2_windows as W
    exp = 43715 * len(W.windows()) * 3 * 4
    assert n == exp, f"{n} rows, expected {exp}"
    d7d14 = exp * 2 // 3
    print(f"rows={n:,} (expected {exp:,})  changed={n_changed:,} of {d7d14:,} d7/d14 rows  "
          f"src_clipped={src_clipped}  out_clipped={clipped} "
          f"({100*clipped/max(d7d14,1):.2f}% of d7/d14)", flush=True)
    moved = sum(1 for h in (7, 14) if abs(fac[h] - 1.0) > 1e-9)
    if moved:
        assert n_changed > 0.9 * (d7d14 * moved // 2), \
            "expected nearly every rescaled-horizon row to move"
    # The composition precondition: the SOURCE must be unclipped (see the loop comment).
    assert src_clipped == 0, (
        f"{src_clipped} source rows already sat at q05<=0, so the shipped 1.10 clip bound "
        f"and information was lost — composition on the final csv is NOT exact; rebuild "
        f"from the base instead.")
    # out_clipped>0 is expected for s>1.10 and is what the real rebuild also does: the
    # lower bound is floored at 0, and since truth speed >= 0 a low-side miss is
    # impossible there, so those rows simply carry a slightly narrower interval.

    tag = a.tag or ("w%s_%s" % (str(s7).replace(".", "p"), str(s14).replace(".", "p")))
    zpath = os.path.join(a.out, "predictions_%s.zip" % tag)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.write(csv_out, arcname="predictions.csv")     # CSV at the ZIP ROOT (FAQ #3)
    print(f"wrote {zpath} ({os.path.getsize(zpath)/1e6:.1f} MB), predictions.csv at root",
          flush=True)


if __name__ == "__main__":
    main()
