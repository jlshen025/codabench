"""Recalibrate PI widths in an existing predictions.csv from the server coverage signal
(operating-point lever, read off the server per Gate 2). Rescales the speed interval around
q50 and the direction arc around dir_50, per horizon. Chunked; no MOS/clim recompute.

THIS IS A SHIPPED PIPELINE STAGE, not a one-off probe. The shipped chain runs it between the
base build and the direction patches with **spd = 1.0, 1.1, 1.1**: d+1 stays at 1.0 because the
retrained CQR is already calibrated there, while d+7/d+14 climatology intervals are widened 10 %
because the server coverage read said they were under-covering. Omitting it narrows d7/d14 by
10 % — the A9 failure mode (a sharper PI that under-covers cost +3.7 Winkler), on two sub-dims
we currently lead. Hence those are the DEFAULTS here; the earlier 0.85/1.3/1.15 were one
probe's values and must not be reintroduced as defaults.

The direction scales are inert in the shipped chain: every dir_05/dir_50/dir_95 row is rewritten
downstream (leads 7/14 by fm_ens_rebuild, lead 1 by fm_dir_spdcond), so they default to 1.0.

--spd "s1,s7,s14"  speed PI multiplier per horizon (<1 narrows, >1 widens)
--dir "s1,s7,s14"  direction arc multiplier per horizon
"""
from __future__ import annotations
import argparse, os
import numpy as np, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="./work/predictions/predictions.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--spd", default=os.environ.get("RECALIB_SPD", "1.0,1.1,1.1"))
    ap.add_argument("--dir", default=os.environ.get("RECALIB_DIR", "1.0,1.0,1.0"))
    a = ap.parse_args()
    ss = [float(x) for x in a.spd.split(",")]; sd = [float(x) for x in a.dir.split(",")]
    SSP = {1: ss[0], 7: ss[1], 14: ss[2]}; SDI = {1: sd[0], 7: sd[1], 14: sd[2]}
    print(f"recalib spd={SSP} dir={SDI}")
    first = True
    n = 0
    for ch in pd.read_csv(a.inp, chunksize=1000000):
        for H in (1, 7, 14):
            m = (ch["horizon"] == H).to_numpy()
            if not m.any():
                continue
            q05 = ch.loc[m, "q05"].to_numpy(); q50 = ch.loc[m, "q50"].to_numpy(); q95 = ch.loc[m, "q95"].to_numpy()
            lo = np.clip(q50 - (q50 - q05) * SSP[H], 0, None); hi = q50 + (q95 - q50) * SSP[H]
            ch.loc[m, "q05"] = lo.round(3); ch.loc[m, "q95"] = hi.round(3)
            d05 = ch.loc[m, "dir_05"].to_numpy(); d50 = ch.loc[m, "dir_50"].to_numpy(); d95 = ch.loc[m, "dir_95"].to_numpy()
            hw = np.clip(((d95 - d05) % 360) / 2.0 * SDI[H], 0, 179.0)
            ch.loc[m, "dir_05"] = ((d50 - hw) % 360).round(3) % 360
            ch.loc[m, "dir_95"] = ((d50 + hw) % 360).round(3) % 360
        ch.to_csv(a.out, index=False, mode="w" if first else "a", header=first)
        first = False; n += len(ch)
    # row count is derived from the resolved window set, never a literal
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import p2_windows as W
    exp = 43715 * len(W.windows()) * 3 * 4
    assert n == exp, f"{n} rows, expected {exp}"
    print(f"wrote {a.out} rows={n:,}")

if __name__ == "__main__":
    main()
