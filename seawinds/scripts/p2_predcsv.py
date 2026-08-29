"""Shared I/O + schema validation for the Phase-2 `predictions.csv` deliverable.

Every build stage reads its source and writes its result through here, so the row
order, the column set and the legality checks have exactly ONE definition. Stages
pass plain CSVs between themselves (zipping an ~850 MB CSV three times costs more
than the patch it carries); only the final artifact is zipped, with the CSV at the
zip ROOT as the organizers' FAQ #3 requires.

Row order is load-bearing: every patch stage aligns POSITIONALLY inside a horizon
mask, so the file must stay window-major -> horizon -> hour -> footprint(_FP order).
`validate` re-checks that invariant against the window set rather than a literal
row count, so a fresh inference directory with a different number of windows is
legal while a scrambled one is not.
"""
from __future__ import annotations

import os
import zipfile

import numpy as np
import pandas as pd

COLS = ["type", "window", "region", "latitude", "longitude", "horizon", "hour", "level",
        "q05", "q50", "q95", "dir_05", "dir_50", "dir_95"]
NFP = 43715              # scored footprint points
HORIZONS = (1, 7, 14)
HOURS = (0, 6, 12, 18)
DIRCOLS = ["dir_05", "dir_50", "dir_95"]


def read_predictions(path: str) -> pd.DataFrame:
    """Load a predictions frame from a .csv or from a .zip holding predictions.csv."""
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z, z.open("predictions.csv") as f:
            return pd.read_csv(f)
    return pd.read_csv(path)


def write_predictions(df: pd.DataFrame, csv_path: str, zip_path: str | None = None) -> str:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df.to_csv(csv_path, index=False)
    if zip_path:
        os.makedirs(os.path.dirname(zip_path) or ".", exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv_path, "predictions.csv")     # ROOT of the zip (FAQ #3)
    return zip_path or csv_path


def validate(df: pd.DataFrame, n_windows: int, *, strict_order: bool = True) -> dict:
    """Assert the deliverable is schema-legal for `n_windows` windows. Returns a summary."""
    assert list(df.columns) == COLS, f"column set/order wrong: {list(df.columns)}"
    exp = NFP * n_windows * len(HORIZONS) * len(HOURS)
    assert len(df) == exp, f"row count {len(df)} != {exp} (= {NFP}x{n_windows}x3x4)"

    assert (df["type"] == "grid").all(), "type must be 'grid'"
    assert (df["region"] == "north_sea").all(), "region must be 'north_sea'"
    assert (df["level"] == "125m").all(), "level must be '125m'"
    assert sorted(df["window"].unique()) == list(range(n_windows)), \
        f"window must be 0..{n_windows - 1}, got {sorted(df['window'].unique())}"
    assert sorted(df["horizon"].unique()) == list(HORIZONS), "horizon must be {1,7,14}"
    assert sorted(df["hour"].unique()) == list(HOURS), "hour must be {0,6,12,18}"

    for c in ("q05", "q50", "q95") + tuple(DIRCOLS):
        assert df[c].notna().all(), f"{c} has NaN"
        assert np.isfinite(df[c].to_numpy()).all(), f"{c} has inf"
    assert (df["q05"] >= 0).all(), "negative speed"
    assert (df["q05"] <= df["q50"]).all() and (df["q50"] <= df["q95"]).all(), "quantiles not monotone"
    d = df[DIRCOLS].to_numpy()
    assert ((d >= 0) & (d < 360)).all(), "direction outside [0,360)"

    if strict_order:
        # window-major -> horizon -> hour -> footprint, each block exactly NFP rows.
        blk = df[["window", "horizon", "hour"]].to_numpy()
        head = blk[::NFP]
        assert (blk.reshape(-1, NFP, 3) == head[:, None, :]).all(), \
            "row order broken: a (window,horizon,hour) block is not NFP contiguous rows"
        seen = [tuple(x) for x in head]
        assert len(set(seen)) == len(seen), "duplicate (window,horizon,hour) block"
        want = [(w, h, hr) for w in range(n_windows) for h in HORIZONS for hr in HOURS]
        assert seen == want, "block sequence is not window-major -> horizon -> hour"

    return {"rows": int(len(df)), "windows": int(n_windows),
            "mean_q50": float(df["q50"].mean()), "mean_dir_50": float(df["dir_50"].mean()),
            "mean_speed_width": float((df["q95"] - df["q05"]).mean())}


def compare(a: pd.DataFrame, b: pd.DataFrame, atol: float = 0.0) -> dict:
    """Per-column difference report between two predictions frames (same shape)."""
    assert len(a) == len(b), f"row counts differ: {len(a)} vs {len(b)}"
    out = {}
    for c in COLS:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if a[c].dtype.kind in "fi":
            diff = np.abs(x.astype(float) - y.astype(float))
            n = int((diff > atol).sum())
            out[c] = {"n_diff": n, "max_abs": float(diff.max()) if len(diff) else 0.0}
        else:
            out[c] = {"n_diff": int((x != y).sum()), "max_abs": None}
    out["_identical"] = all(v["n_diff"] == 0 for k, v in out.items() if not k.startswith("_"))
    return out
