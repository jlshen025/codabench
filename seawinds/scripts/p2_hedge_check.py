#!/usr/bin/env python
"""Acceptance gate for a CENTRE-CHANGING d+14 candidate (the w5 hedge).

Why this exists rather than p2_dir_probe_check.py: that gate gives the right answer for a pure
ARC rescale, where the centre never moves, so it hard-asserts `dir_50` untouched GLOBALLY. The
w5 hedge deliberately moves `dir_50` on one window, so that gate is measuring a different
construct and would veto a correct candidate. Bypassing it would leave the candidate ungated,
so the invariants it was enforcing are restated here in the form this candidate must satisfy.

Invariants checked (any failure aborts, non-zero exit):
  1. schema, row count and the KEY block (row ORDER) byte-identical  -- load-bearing for scoring
  2. speed q05/q50/q95 byte-identical EVERYWHERE                     -- 3 sub-dims carry zero risk
  3. d+1 and d+7 direction columns byte-identical                    -- 2 more sub-dims, zero risk
  4. d+14: `dir_50` differs ONLY on the hedged window, and nowhere else
  5. d+14: the arc rescale actually fired, and half-widths stay < 180 deg
  6. every direction value in [0, 360)

Usage: python scripts/p2_hedge_check.py --src <shipped.csv> --out <candidate.csv> --window 5
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

KEY = ["type", "window", "region", "latitude", "longitude", "horizon", "hour", "level"]
SPEED = ["q05", "q50", "q95"]
DIRC = ["dir_05", "dir_50", "dir_95"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, required=True,
                    help="the ONE window whose centre may move; -1 = the centre may move in ALL "
                         "windows (a whole-column swap such as the d+14 climatology revert)")
    a = ap.parse_args()

    n = 0
    centre_moved = {"in": 0, "out": 0}
    arc = {H: {"wa": 0.0, "wb": 0.0, "n": 0, "maxd": 0.0} for H in (1, 7, 14)}
    hmax = 0.0
    dir_range_ok = True

    for A, B in zip(pd.read_csv(a.src, chunksize=1_000_000),
                    pd.read_csv(a.out, chunksize=1_000_000)):
        assert list(A.columns) == list(B.columns), "schema drift"
        assert len(A) == len(B), "row count drift"
        for c in KEY:
            assert A[c].equals(B[c]), f"row order/key changed at {c}"
        for c in SPEED:
            assert A[c].equals(B[c]), f"{c} must be untouched (speed sub-dims carry zero risk)"

        for H in (1, 7, 14):
            m = (A["horizon"] == H).to_numpy()
            if not m.any():
                continue
            if H != 14:
                for c in DIRC:
                    assert A.loc[m, c].equals(B.loc[m, c]), f"d{H} {c} must be untouched"
                continue
            # --- d+14 ---
            d50 = np.abs(A.loc[m, "dir_50"].to_numpy() - B.loc[m, "dir_50"].to_numpy())
            inw = (np.ones(int(m.sum()), bool) if a.window < 0
                   else (A.loc[m, "window"] == a.window).to_numpy())
            centre_moved["in"] += int((d50[inw] > 0).sum())
            centre_moved["out"] += int((d50[~inw] > 0).sum())
            wa = (A.loc[m, "dir_95"] - A.loc[m, "dir_05"]) % 360.0
            wb = (B.loc[m, "dir_95"] - B.loc[m, "dir_05"]) % 360.0
            arc[14]["wa"] += float(wa.sum()); arc[14]["wb"] += float(wb.sum())
            arc[14]["n"] += int(m.sum())
            arc[14]["maxd"] = max(arc[14]["maxd"], float(np.abs(wa - wb).max()))
            hmax = max(hmax, float((wb / 2.0).max()))
            for c in DIRC:
                v = B.loc[m, c].to_numpy()
                if not ((v >= 0).all() and (v < 360).all()):
                    dir_range_ok = False
        n += len(A)

    ok = True
    print(f"rows compared: {n:,}")
    scope = "ALL windows" if a.window < 0 else f"window {a.window}"
    print(f"d+14 centre moved: {centre_moved['in']:,} rows INSIDE {scope}, "
          f"{centre_moved['out']:,} rows outside")
    if centre_moved["out"] != 0:
        print("  FAIL: the centre moved outside the hedged window"); ok = False
    if centre_moved["in"] == 0:
        print("  FAIL: the hedge is a no-op — no centre moved"); ok = False
    a14 = arc[14]
    print(f"d+14 mean arc {a14['wa']/a14['n']:.4f} -> {a14['wb']/a14['n']:.4f} deg "
          f"(ratio {a14['wb']/a14['wa']:.6f}), max half-width {hmax:.4f} deg")
    if a14["maxd"] == 0.0:
        print("  FAIL: the arc rescale did not fire"); ok = False
    if hmax >= 180.0:
        print("  FAIL: a half-width reached 180 deg (arc would be the whole circle)"); ok = False
    if not dir_range_ok:
        print("  FAIL: a direction value fell outside [0,360)"); ok = False

    print("\nACCEPTED — speed + row order + d1/d7 directions byte-identical; only d+14 moved."
          if ok else "\nREJECTED — do not submit this artifact.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
