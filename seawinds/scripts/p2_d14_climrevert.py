#!/usr/bin/env python
"""Build the d+14 CLIMATOLOGY-REVERT candidate.

WHY. E16 measured our shipped d+14 direction at 356.10 against a calibrated no-skill floor of
342.0 (w* = (1-alpha)*180 = 162 deg, S_zero = 180*(2-alpha)). Our Pangu ens-mean centre therefore
carries NEGATIVE information on this draw. The correct response to a component measured below the
uninformative baseline is not to re-tune its arc -- E16 showed narrowing makes it worse, because a
near-antipodal clump survives -- but to STOP USING IT and fall back to the component our own
leave-year-out CV ranked next: the monthly climatological direction (CV d+14 dir ~309), which is
year-stable by construction where the FM demonstrably was not.

WIDTH CHOICE, and why the downside is bounded. We ship a CONSTANT half-width of 162 deg, the
no-skill optimum. That gives a two-sided guarantee on a column we get exactly one read of:
  * if the climatological centre turns out uninformative on 2022, the score lands at ~342.0 --
    still better than the 356.10 we hold, and enough to move us from rank 3 to rank 2;
  * if it carries its usual skill, 162 deg is WIDE of that error distribution's optimum, and the
    ~20x miss asymmetry means landing wide is the safe side; the score falls toward ~320.
So the candidate improves the column on essentially any outcome, rather than betting on one.

SCOPE. Only the three d+14 direction columns move. Speed, metadata, row order and the d+1/d+7
blocks are copied through byte-for-byte -- five of the six scored sub-dimensions carry zero risk.
Gate with p2_hedge_check.py (the arc-only gate p2_dir_probe_check.py is the wrong instrument for a
centre change) and p2_predcsv.validate().
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

SHIP = Path("scripts/results/f22_d1arc095_s0/predictions.csv")
CLIM = Path("models_final/clim_2016_2020.npz")
OUT = Path(os.environ.get("D14_OUT", "scripts/results/f22_d14climrev"))

HALF_WIDTH = float(os.environ.get("D14_HW", "162.0"))  # 162 = no-skill optimum; retune once
                                                       # the realised exceedance is measured
# d+14 valid month per window, measured from each window's own metadata (p2_w5_verify.py)
WINDOW_MONTH = {0: 1, 1: 3, 2: 4, 3: 6, 4: 7, 5: 8, 6: 10, 7: 11}
HOUR_IDX = {0: 0, 6: 1, 12: 2, 18: 3}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clim = np.load(CLIM)
    fields = {(m, h): clim[f"{m}_{h}_dir50"] for m in set(WINDOW_MONTH.values()) for h in range(4)}
    n_pts = len(next(iter(fields.values())))
    print(f"climatology: {n_pts} points x {len(fields)} (month,hour) fields", flush=True)

    # a (window,horizon,hour) block of 43,715 points can straddle a chunk boundary
    seen: dict[tuple[int, int], int] = {}
    n_rows = n_changed = 0
    first = True

    for chunk in pd.read_csv(SHIP, chunksize=1_000_000):
        n_rows += len(chunk)
        m14 = (chunk.horizon == 14).to_numpy()
        if m14.any():
            c = chunk.loc[m14]
            centre = c.dir_50.to_numpy().astype(float).copy()
            for w in sorted(c.window.unique()):
                mo = WINDOW_MONTH[int(w)]
                for hr, hidx in HOUR_IDX.items():
                    sel = ((c.window == w) & (c.hour == hr)).to_numpy()
                    k = int(sel.sum())
                    if not k:
                        continue
                    off = seen.get((int(w), hr), 0)
                    assert off + k <= n_pts, f"w{w} h{hr}: overran footprint ({off}+{k})"
                    centre[sel] = fields[(mo, hidx)][off:off + k]
                    seen[(int(w), hr)] = off + k
                    n_changed += k
            chunk.loc[m14, "dir_50"] = centre % 360.0
            chunk.loc[m14, "dir_05"] = (centre - HALF_WIDTH) % 360.0
            chunk.loc[m14, "dir_95"] = (centre + HALF_WIDTH) % 360.0
        # NO float_format: default repr round-trips, keeping every untouched column byte-identical
        chunk.to_csv(OUT / "predictions.csv", mode="w" if first else "a",
                     header=first, index=False)
        first = False

    print(f"rows {n_rows:,} | d+14 centres replaced by climatology: {n_changed:,} "
          f"(expect 43,715 x 4 x 8 = {43715*4*8:,})")
    print(f"constant half-width {HALF_WIDTH} deg -> arc {2*HALF_WIDTH} deg")
    print(f"predicted: ~342 if climatology is uninformative on 2022, toward ~320 if it carries "
          f"its CV skill; either way below the 356.10 we hold")
    print(f"wrote {OUT/'predictions.csv'}")


if __name__ == "__main__":
    main()
