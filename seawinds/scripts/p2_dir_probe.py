"""Build DIRECTION arc-width variants of the held-best, for server measurement.

WHY. Circular Winkler = arc width + ~20x circular miss, the same shape as the speed
score, and width DOMINATES: of the shipped scores, width is ~65 % of dir_d1 (78.2 deg of
120.85), ~80 % of dir_d7 (234.6 of 293.92) and ~85 % of dir_d14 (254.0 of 297.97). Yet
only the d+1 arc was ever Winkler-optimized (A10/A11, and on CV at that). The d+7 arc
comes from the ensemble-rebuild stage and the d+14 arc is a 90th-percentile climatological
width -- i.e. both are COVERAGE-targeted, not score-optimal. A35/A36 measured that the
Winkler optimum sits BELOW nominal coverage (speed d+7 optimal at 88.3 %, and the d+1
speed interval was over-wide at 93.6 % and worth -0.20 to narrow), so a 90 %-targeted arc
is the same untested-default failure mode on a dimension worth 3 of the 6 sub-scores.

WHAT. Rescale ONLY dir_05/dir_95 about the unchanged dir_50, per horizon. Speed columns
and all metadata pass through untouched. Safe on this artifact because every shipped arc
is already symmetric about dir_50 (verified: 0 of 4,196,640 rows asymmetric), so
re-centering is exact rather than a silent symmetrization of asymmetric arcs.

Half-widths are clipped to 179 deg (a 360 deg arc is degenerate). At the probed scales no
row reaches that: the widest shipped arc is 270 deg (d+1), so 155 deg of half-width at
x1.15. The script asserts this rather than trusting it.

  python p2_dir_probe.py --dir 0.85,0.85,0.85 --src <held-best csv> --out <dir>
"""
from __future__ import annotations
import argparse, os, sys, zipfile
import numpy as np
import pandas as pd

COLS = ["type", "window", "region", "latitude", "longitude", "horizon", "hour", "level",
        "q05", "q50", "q95", "dir_05", "dir_50", "dir_95"]
SPEED = ["q05", "q50", "q95"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", dest="dirs", required=True, help="arc multipliers 's1,s7,s14'")
    ap.add_argument("--src", required=True, help="source predictions.csv (the held-best)")
    ap.add_argument("--out", default=os.environ.get("EXP_OUTPUT_DIR", "."))
    ap.add_argument("--tag", default="dirprobe")
    a = ap.parse_args()
    sd = [float(x) for x in a.dirs.split(",")]
    S = {1: sd[0], 7: sd[1], 14: sd[2]}
    os.makedirs(a.out, exist_ok=True)
    csv_out = os.path.join(a.out, "predictions.csv")
    print(f"direction arc multipliers {S}", flush=True)

    n = 0; first = True
    asym = 0; clipped = 0
    wa = {h: 0.0 for h in S}; wb = {h: 0.0 for h in S}; cnt = {h: 0 for h in S}
    for ch in pd.read_csv(a.src, chunksize=1000000):
        assert list(ch.columns) == COLS, f"schema drift: {list(ch.columns)}"
        spd_before = ch[SPEED].to_numpy(copy=True)
        for H, s in S.items():
            m = (ch["horizon"] == H).to_numpy()
            if not m.any():
                continue
            d05 = ch.loc[m, "dir_05"].to_numpy(); d50 = ch.loc[m, "dir_50"].to_numpy()
            d95 = ch.loc[m, "dir_95"].to_numpy()
            lo_w = (d50 - d05) % 360.0        # half-width below the centre
            hi_w = (d95 - d50) % 360.0        # half-width above
            asym += int((np.abs(lo_w - hi_w) > 0.01).sum())
            hw = ((d95 - d05) % 360.0) / 2.0
            new_hw = hw * s
            clipped += int((new_hw > 179.0).sum())
            new_hw = np.clip(new_hw, 0.0, 179.0)
            wa[H] += float((2 * hw).sum()); wb[H] += float((2 * new_hw).sum()); cnt[H] += int(m.sum())
            ch.loc[m, "dir_05"] = np.round((d50 - new_hw) % 360.0, 3) % 360.0
            ch.loc[m, "dir_95"] = np.round((d50 + new_hw) % 360.0, 3) % 360.0
        # speed must be untouched, every chunk
        assert np.array_equal(spd_before, ch[SPEED].to_numpy()), "speed column mutated"
        for c in ("dir_05", "dir_50", "dir_95"):
            v = ch[c].to_numpy()
            assert ((v >= 0) & (v < 360)).all(), f"{c} out of [0,360)"
        ch.to_csv(csv_out, index=False, mode="w" if first else "a", header=first)
        first = False; n += len(ch)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import p2_windows as W
    exp = 43715 * len(W.windows()) * 3 * 4
    assert n == exp, f"{n} rows, expected {exp}"
    # Symmetry is the precondition for exact re-centering (see module docstring).
    assert asym == 0, f"{asym} arcs were asymmetric about dir_50 — re-centering would silently symmetrize them"
    assert clipped == 0, f"{clipped} half-widths hit the 179 deg cap — the probe is no longer a pure rescale"
    for H in S:
        print(f"  d{H:<2d} mean arc {wa[H]/cnt[H]:7.3f} -> {wb[H]/cnt[H]:7.3f} deg "
              f"(ratio {wb[H]/wa[H]:.4f}, target {S[H]:.2f})", flush=True)

    z = os.path.join(a.out, f"predictions_{a.tag}.zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as f:
        f.write(csv_out, arcname="predictions.csv")     # CSV at the ZIP ROOT (FAQ #3)
    print(f"wrote {z} ({os.path.getsize(z)/1e6:.1f} MB), predictions.csv at root", flush=True)


if __name__ == "__main__":
    main()
