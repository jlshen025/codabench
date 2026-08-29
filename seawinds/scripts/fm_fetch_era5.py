"""Fetch ERA5 initial states for the FM driver (Pangu-Weather input format) from
the WeatherBench2 public GCS zarr — one global state per eval window at the
window's context_end 00 UTC. FAIR per Gate 1: init from context_end, no data
past it; ERA5 (public reanalysis) only. Login-node (internet); one window at a
time (~270 MB peak). Saves upper (5 vars × 13 levels) + surface (4 vars) npz.
"""
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, xarray as xr, pandas as pd
import p2_windows as W

P2 = os.environ["SEAWINDS_PHASE2_DIR"]; INF = W.infer_dir()
OUT = os.environ.get("FM_ERA5_INIT_DIR", "./work/fm/era5_init")
os.makedirs(OUT, exist_ok=True)
# ERA5 stores, widest coverage first. A STORE NAME IS NOT ITS COVERAGE: the legacy
# `1959-2022-...` actually ends 2021-12-31 18:00 (ledger ), so on a 2022 withheld year
# it returns nothing, the Pangu ensemble cannot run, and every direction column silently
# degrades to the --no-fm path. Selection therefore ASSERTS the calendar instead of
# trusting the filename. The 2023_01_10 store is byte-identical to the legacy one on all
# eight shipped windows (max|diff| = 0, upper and surface) -- verified, not assumed.
ZARRS = [
    # First: the store the shipped chain is verified against end-to-end .
    "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721.zarr",  # -> 2023-01-10
    # Escalation for any later withheld year. ARCO-ERA5 is byte-IDENTICAL to the above on
    # all eight shipped windows (max|diff| = 0, upper and surface -- ledger ) and carries
    # real data past 2026. Its time axis is padded out to 2050 with NaNs, which is exactly
    # why the probe below reads a VALUE instead of trusting the index.
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",    # -> present
    "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr",             # -> 2021-12-31
]
PANGU_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
UP = {"z": "geopotential", "q": "specific_humidity", "t": "temperature",
      "u": "u_component_of_wind", "v": "v_component_of_wind"}
SF = {"mslp": "mean_sea_level_pressure", "u10": "10m_u_component_of_wind",
      "v10": "10m_v_component_of_wind", "t2m": "2m_temperature"}


def open_covering(inits):
    """First store whose time axis contains EVERY requested init, on the expected grid."""
    errs = []
    for z in ZARRS:
        try:
            ds = xr.open_zarr(z, storage_options={"token": "anon"}, chunks=None)
        except Exception as e:
            errs.append(f"{z}\n      open failed: {type(e).__name__}: {e}")
            continue
        have = pd.DatetimeIndex(ds.time.values)
        missing = [t for t in inits if t not in have]
        if missing:
            errs.append(f"{z}\n      covers to {have[-1]}; missing {len(missing)} init(s), "
                        f"first {missing[0]}")
            continue
        # A different store could ship a flipped or coarser grid, which would corrupt the
        # Pangu input silently rather than error. Check before returning.
        assert tuple(ds.sizes[d] for d in ("latitude", "longitude")) == (721, 1440), \
            f"{z}: grid is {ds.sizes}, expected 721x1440"
        assert float(ds.latitude.values[0]) == 90.0, \
            f"{z}: latitude starts at {float(ds.latitude.values[0])}, expected 90.0 (north-first)"
        for v in list(UP.values()) + list(SF.values()):
            assert v in ds, f"{z}: missing required variable {v}"
        lv = {int(x) for x in ds.level.values}
        assert not [l for l in PANGU_LEVELS if l not in lv], f"{z}: missing Pangu levels"
        # An index check is NOT a coverage check: ARCO-ERA5 declares a time axis running to
        # 2050 that is mostly unpopulated, so "the timestamp is in the index" would pass for
        # a date with no data and hand back NaNs — a silent failure worse than the loud one
        # this function exists to prevent. Read one real value per requested init instead.
        bad = []
        for t in inits:
            try:
                v = float(ds[SF["u10"]].sel(time=t).isel(latitude=360, longitude=720).values)
            except Exception as e:
                bad.append(f"{t}: read failed ({type(e).__name__})"); continue
            if not np.isfinite(v):
                bad.append(f"{t}: non-finite ({v})")
        if bad:
            errs.append(f"{z}\n      time axis lists these inits but the DATA is absent: "
                        + "; ".join(bad[:3]) + (" ..." if len(bad) > 3 else ""))
            continue
        print(f"ERA5 store: {z}\n  covers {have[0]} -> {have[-1]} (data probed at every init)",
              flush=True)
        return ds
    raise SystemExit(
        "NO ERA5 STORE COVERS THE REQUESTED INITS:\n    " + "\n    ".join(errs) +
        "\n  Every configured store was tried, including ARCO-ERA5 which normally covers "
        "to the present. Check the dates in the inference metadata, then verify any new "
        "source with scripts/fm_era5_source_check.py before shipping. "
        "`--stage assemble --no-fm` is the backstop (O9).")

def main():
    t0 = time.time()
    wids = W.window_ids(INF)
    metas = {w: json.load(open(f"{INF}/window_{w}/metadata.json")) for w in wids}
    inits = {w: pd.Timestamp(metas[w]["context_end"]) for w in wids}   # 00 UTC of context_end
    # Pick the store by the calendar we actually need, before downloading anything.
    ds = open_covering(sorted(set(inits.values())))
    print(f"opened WB2 ERA5 ({dict(ds.sizes)}); lat[0]={float(ds.latitude.values[0])}", flush=True)
    for wid in wids:
        md = metas[wid]
        init = inits[wid]
        sel = ds.sel(time=init)
        up = np.stack([sel[UP[k]].sel(level=PANGU_LEVELS).values.astype(np.float32)
                       for k in ("z", "q", "t", "u", "v")])               # (5,13,721,1440)
        sf = np.stack([sel[SF[k]].values.astype(np.float32)
                       for k in ("mslp", "u10", "v10", "t2m")])           # (4,721,1440)
        dst = f"{OUT}/window_{wid}_init.npz"
        np.savez(dst, upper=up, surface=sf, init=str(init), context_end=md["context_end"],
                 pangu_levels=np.array(PANGU_LEVELS),
                 lat=ds.latitude.values.astype(np.float32), lon=ds.longitude.values.astype(np.float32))
        print(f"  window {wid} cend={md['context_end']} up{up.shape} sf{sf.shape} "
              f"-> {dst} ({time.time()-t0:.0f}s)", flush=True)
    print(f"DONE {len(wids)} windows ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
