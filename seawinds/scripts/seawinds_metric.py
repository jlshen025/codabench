#!/usr/bin/env python
"""Exact reproduction of the Sea-Winds Phase-1 scoring metric.

The official leaderboard ranks by ``mean_rank`` = mean of per-dimension RANKS over
36 dimensions = {speed, direction} x {d1, d7, d14} x {stations, surface, pressure}
x {north_sea, east_china_sea}. Ranks are *across submissions* and can't be
reproduced locally, but the 36 raw per-dimension scores CAN — and they are the
real iteration signal (they post immediately; mean_rank lags up to ~24-48h).

Per-dimension metrics (lower is better), alpha = 0.1:

  SPEED  — Winkler interval score on (q05, q95):
      WS = (q95 - q05) + (2/alpha) * [ max(0, q05 - y) + max(0, y - q95) ]

  DIRECTION — circular Winkler on (dir_05, dir_95), degrees, CCW arc:
      w        = (dir_95 - dir_05) mod 360            # arc width
      inside   = ((y - dir_05) mod 360) <= w
      d_miss   = min circular distance from y to the nearer endpoint
      cWS      = w + (2/alpha) * d_miss * 1[not inside]

This module is pure (numpy) and has no I/O; the CV protocol builds the tidy
(dim, y, lo, hi) frame and calls score_breakdown().
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ALPHA = 0.1

# The 36 dimension keys, in a canonical order.
PROBLEMS = ["speed", "dir"]
GT_GROUPS = ["stations", "surface", "pressure"]
HORIZONS = [1, 7, 14]
REGIONS = ["ns", "ecs"]
DIM_KEYS = [f"{p}_{g}_d{h}_{r}"
            for p in PROBLEMS for g in GT_GROUPS for h in HORIZONS for r in REGIONS]
assert len(DIM_KEYS) == 36


def circular_distance(a, b):
    """Shorter angular distance in degrees between angles a and b (0..180)."""
    diff = np.abs(np.asarray(a, float) - np.asarray(b, float)) % 360.0
    return np.minimum(diff, 360.0 - diff)


def winkler_speed(y, q05, q95, alpha=ALPHA):
    """Elementwise speed Winkler interval score. Lower is better."""
    y = np.asarray(y, float); q05 = np.asarray(q05, float); q95 = np.asarray(q95, float)
    width = q95 - q05
    below = np.maximum(0.0, q05 - y)
    above = np.maximum(0.0, y - q95)
    return width + (2.0 / alpha) * (below + above)


def circular_winkler(y, d05, d95, alpha=ALPHA):
    """Elementwise direction circular Winkler score (degrees). Lower is better.

    The prediction interval is the COUNTER-CLOCKWISE arc from d05 to d95
    (increasing angle mod 360), so the width is (d95 - d05) mod 360.
    """
    y = np.asarray(y, float) % 360.0
    d05 = np.asarray(d05, float) % 360.0
    d95 = np.asarray(d95, float) % 360.0
    w = (d95 - d05) % 360.0
    pos = (y - d05) % 360.0          # how far CCW y is from the lower endpoint
    inside = pos <= w
    d_miss = np.minimum(circular_distance(y, d05), circular_distance(y, d95))
    return w + (2.0 / alpha) * d_miss * (~inside)


def score_rows(df):
    """Per-row score for a tidy frame.

    Required columns: 'problem' in {speed,dir}, 'y', 'lo', 'hi'
      - speed rows: lo=q05, hi=q95, y=true speed (m/s)
      - dir rows:   lo=dir_05, hi=dir_95, y=true direction (deg)
    Returns a float Series aligned to df.index.
    """
    out = np.empty(len(df), float)
    is_spd = (df["problem"].values == "speed")
    spd = df.loc[is_spd]
    dr = df.loc[~is_spd]
    if len(spd):
        out[np.where(is_spd)[0]] = winkler_speed(spd["y"], spd["lo"], spd["hi"])
    if len(dr):
        out[np.where(~is_spd)[0]] = circular_winkler(dr["y"], dr["lo"], dr["hi"])
    return pd.Series(out, index=df.index)


def score_breakdown(df):
    """Mean per-dimension score for a tidy frame.

    Required columns: 'dim' (one of DIM_KEYS), 'problem', 'y', 'lo', 'hi'.
    Returns dict {dim_key: mean_score} (only dims present in df).
    """
    s = score_rows(df)
    g = s.groupby(df["dim"]).mean()
    return g.to_dict()


def mean_of_36(breakdown):
    """Unweighted mean of the available dimension scores (a primary_score proxy).

    NOTE: not identical to the leaderboard 'primary_score' (whose exact formula is
    unknown / likely normalized), but a monotone-ish single scalar for quick A/B.
    Direction dims dominate (deg units >> m/s), so ALSO inspect speed/dir means
    and the full 36-vector separately — never select on this alone.
    """
    vals = [v for v in breakdown.values()]
    return float(np.mean(vals)) if vals else float("nan")


def summarize(breakdown):
    """Return (mean_all, mean_speed, mean_dir, n_dims) for a breakdown dict."""
    spd = [v for k, v in breakdown.items() if k.startswith("speed_")]
    dr = [v for k, v in breakdown.items() if k.startswith("dir_")]
    allv = list(breakdown.values())
    f = lambda xs: float(np.mean(xs)) if xs else float("nan")
    return dict(mean_all=f(allv), mean_speed=f(spd), mean_dir=f(dr), n_dims=len(allv))


# --------------------------------------------------------------------------- #
# Self-tests
# --------------------------------------------------------------------------- #
def _test():
    # speed: inside -> width only
    assert abs(winkler_speed([5], [3], [8])[0] - 5.0) < 1e-9
    # speed: above by 2 -> 5 + 20*2 = 45
    assert abs(winkler_speed([10], [3], [8])[0] - 45.0) < 1e-9
    # speed: below by 1 -> 5 + 20*1 = 25
    assert abs(winkler_speed([2], [3], [8])[0] - 25.0) < 1e-9

    # circular distance wrap
    assert abs(circular_distance(10, 350) - 20.0) < 1e-9
    assert abs(circular_distance(100, 30) - 70.0) < 1e-9

    # circular Winkler: y inside a wrap-around arc [350 -> 30] (w=40), y=10 inside
    assert abs(circular_winkler([10], [350], [30])[0] - 40.0) < 1e-9
    # y=100 outside arc [350->30]: w=40, d_miss=min(110,70)=70 -> 40+20*70=1440
    assert abs(circular_winkler([100], [350], [30])[0] - 1440.0) < 1e-9
    # full-circle arc ~ width 360, always inside
    v = circular_winkler([123.4], [0.0], [359.999])[0]
    assert 359.0 < v <= 360.0, v
    # symmetric +-30 around 200, truth at 200 -> width 60
    assert abs(circular_winkler([200], [170], [230])[0] - 60.0) < 1e-9

    # breakdown grouping
    df = pd.DataFrame({
        "dim": ["speed_surface_d1_ns", "speed_surface_d1_ns", "dir_stations_d7_ecs"],
        "problem": ["speed", "speed", "dir"],
        "y": [5.0, 10.0, 100.0], "lo": [3.0, 3.0, 350.0], "hi": [8.0, 8.0, 30.0],
    })
    bd = score_breakdown(df)
    assert abs(bd["speed_surface_d1_ns"] - (5.0 + 45.0) / 2) < 1e-9
    assert abs(bd["dir_stations_d7_ecs"] - 1440.0) < 1e-9
    print("seawinds_metric self-tests PASSED")
    print("  DIM_KEYS:", len(DIM_KEYS), "e.g.", DIM_KEYS[0], "...", DIM_KEYS[-1])


if __name__ == "__main__":
    _test()
