#!/usr/bin/env python
"""Carve a SYNTHETIC inference directory from the training years, to rehearse the
final-evaluation rebuild before the real inference set drops on 2026-08-08.

The point is to break every assumption the real `inference/` tree happens to satisfy:
a DIFFERENT number of windows (so a surviving `range(1, 9)` fails loudly), a different
year, and different context dates. Everything else mirrors the real layout exactly —
same filenames, same columns, same dtypes, same row counts — so a pipeline that runs
here runs on the real thing.

Sources (both real data, so the rehearsal exercises real I/O paths):
  context_hres_north_sea.parquet  <- Phase-1 hres_north_sea.parquet (2019-2021, the full
                                     27-column d1/d7/d10 schema the eval windows ship)
  context_reanalysis_*.parquet    <- phase2 train/reanalysis daily .nc, 14 days x 4 hours

This directory is NOT a scoring set — there is no truth attached and no score to read.
It answers one question only: does the rebuild run end-to-end on inputs it has never
seen, and is the output schema-legal for that window count?

  python p2_make_synthetic_inference.py --out DIR [--starts 2019-03-04,2019-08-12,2020-05-18]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import xarray as xr

P2 = os.environ["SEAWINDS_PHASE2_DIR"]
P1 = os.environ["SEAWINDS_DATA_DIR"]
HRES_P1 = f"{P1}/train/hres_north_sea.parquet"
CTX_DAYS = 14


def reanalysis_context(start: pd.Timestamp) -> pd.DataFrame:
    """14 consecutive days of the 0.25 deg reanalysis, long format, exactly as the eval
    windows ship it (time, latitude, longitude, u10, v10, u100, v100)."""
    frames = []
    for k in range(CTX_DAYS):
        d = start + pd.Timedelta(days=k)
        f = f"{P2}/train/reanalysis/{d.year}/reanalysis_{d:%Y%m%d}.nc"
        assert os.path.exists(f), f"missing reanalysis day {f}"
        ds = xr.open_dataset(f)
        df = ds.to_dataframe().reset_index()
        ds.close()
        frames.append(df[["time", "latitude", "longitude", "u10", "v10", "u100", "v100"]])
    out = pd.concat(frames, ignore_index=True)
    for c in ("latitude", "longitude", "u10", "v10", "u100", "v100"):
        out[c] = out[c].astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--starts", default="2019-03-04,2019-08-12,2020-05-18",
                    help="context_start dates; their COUNT is deliberately not 8")
    a = ap.parse_args()

    starts = [pd.Timestamp(s) for s in a.starts.split(",")]
    hres = pd.read_parquet(HRES_P1)
    os.makedirs(a.out, exist_ok=True)

    for i, cs in enumerate(starts, start=1):
        ce = cs + pd.Timedelta(days=CTX_DAYS - 1)
        wdir = f"{a.out}/window_{i}"
        os.makedirs(wdir, exist_ok=True)

        md = {"id": i,
              "context_start": f"{cs:%Y-%m-%d}", "context_end": f"{ce:%Y-%m-%d}",
              "predict_start": f"{ce + pd.Timedelta(days=1):%Y-%m-%d}",
              "predict_end": f"{ce + pd.Timedelta(days=14):%Y-%m-%d}",
              "score_days": {f"d{L}": f"{ce + pd.Timedelta(days=L):%Y-%m-%d}"
                             for L in (1, 7, 14)},
              "synthetic": True}
        json.dump(md, open(f"{wdir}/metadata.json", "w"), indent=1)

        ch = hres[hres["time"] == ce].reset_index(drop=True)
        assert len(ch) == 2565, f"window {i}: HRES at {ce:%Y-%m-%d} has {len(ch)} rows, want 2565"
        ch.to_parquet(f"{wdir}/context_hres_north_sea.parquet", index=False)

        cr = reanalysis_context(cs)
        assert len(cr) == 2565 * 4 * CTX_DAYS, f"window {i}: reanalysis has {len(cr)} rows"
        cr.to_parquet(f"{wdir}/context_reanalysis_north_sea.parquet", index=False)

        print(f"window_{i}: context {cs:%Y-%m-%d}..{ce:%Y-%m-%d} -> "
              f"hres {ch.shape} reanalysis {cr.shape}", flush=True)

    print(f"\nSYNTHETIC inference dir with {len(starts)} windows (real eval ships 8) -> {a.out}")


if __name__ == "__main__":
    main()
