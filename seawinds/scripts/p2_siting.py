"""Siting library: fast PyWake farm CF from the AROME footprint cache.

The cache (p2_cache_arome_footprint.py) holds AROME u125m/v125m at the 43,715
footprint pts for hours 0/6/12/18, 2016-2020. Extract any farm-centre cell's wind
series, shear 125->170m (α=0.11), run the kit's PyWake simulator (fidelity) → CF
per year. Memory-light: loads ONE year (~0.5GB) at a time. Used for centre
ranking, layout optimization, and the final submission.json.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("PHASE2_DATA_ROOT", os.environ["SEAWINDS_PHASE2_DIR"])
KIT = "${SEAWINDS_ROOT}/kit_phase2/phase_2"
sys.path.insert(0, KIT); sys.path.insert(0, KIT + "/part0_dataset_setup")
import numpy as np, pandas as pd
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
import wind_farm_simulator as wfs
import turbines_catalog as tc
import shear

# AROME footprint cache. Default = the scratch build; the endgame rebuild points this at
# the durable copy (footprint_order.parquet only — the per-year npz are training-only).
CACHE = os.environ.get("SEAWINDS_AROME_CACHE",
                       "./work/cache/arome_footprint")
DIAM = 284.0
SHEAR = shear.power_law_factor(125, 170)
HOURS = [0, 6, 12, 18]
ALL_YEARS = [2016, 2017, 2018, 2019, 2020]
_TURB = None
_ORDER = None

def turbine():
    global _TURB
    if _TURB is None:
        _TURB = tc.load_turbine("IEA_22MW")
    return _TURB

def footprint_order():
    """The 43,715-point footprint row order -- load-bearing for every predictions.csv
    block. $SEAWINDS_FOOTPRINT_ORDER points at the durable /project copy so the endgame
    rebuild does not depend on the purgeable per-year cache."""
    global _ORDER
    if _ORDER is None:
        _ORDER = pd.read_parquet(os.environ.get("SEAWINDS_FOOTPRINT_ORDER",
                                                f"{CACHE}/footprint_order.parquet"))
    return _ORDER

def nearest_idx(lat, lon):
    o = footprint_order()
    d = (o.lat.values - lat) ** 2 + (o.lon.values - lon) ** 2
    i = int(np.argmin(d))
    return i, float(o.lat.values[i]), float(o.lon.values[i])

def load_year(y):
    """(u,v,dates) for one year. u,v: (ndays,4,npts) float32. ~0.5GB — free promptly."""
    d = np.load(f"{CACHE}/arome_fp_{y}.npz")
    return d["u"], d["v"], d["dates"]

def _times_for(dates):
    t = []
    for dd in dates:
        base = pd.Timestamp(str(dd))
        t += [base + pd.Timedelta(hours=h) for h in HOURS]
    return pd.to_datetime(t)

def series_from_year(u, v, dates, idx):
    """(times, ws_hub, wd) for one cell from an already-loaded year."""
    uu = u[:, :, idx].reshape(-1); vv = v[:, :, idx].reshape(-1)
    t = _times_for(dates)
    ok = np.isfinite(uu) & np.isfinite(vv)
    ws = np.hypot(uu, vv) * SHEAR
    wd = (270.0 - np.degrees(np.arctan2(vv, uu))) % 360.0
    return t[ok], ws[ok], wd[ok]

def simulate_cf(times, ws, wd, layout_xy):
    lay = wfs.FarmLayout(x_m=layout_xy[0], y_m=layout_xy[1], turbine=turbine())
    wind = wfs.WindSeries(pd.DataFrame({"time": times, "ws": ws, "wd": wd}))
    res = wfs.simulate_year(lay, wind)
    return res.capacity_factor, res.wake_loss_fraction, res.aep_gwh

def box_grid(spacing_d=7.4, n=55, rot=0.0):
    return wfs.grid_layout(n, spacing_d=spacing_d, diameter_m=DIAM, rotation_deg=rot)

def centre_series_allyears(idx, years=ALL_YEARS):
    """dict year->(times,ws,wd) for one cell, loading each year once (memory-light)."""
    out = {}
    for y in years:
        u, v, dates = load_year(y)
        out[y] = series_from_year(u, v, dates, idx)
        del u, v
    return out

if __name__ == "__main__":
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--topk", type=int, default=25)
    ap.add_argument("--spacing", type=float, default=7.4)
    args = ap.parse_args()
    t0 = time.time()
    xy = box_grid(args.spacing)

    if args.validate:
        u, v, dates = load_year(2020)
        idx, alat, alon = nearest_idx(53.5, 1.5)
        t, ws, wd = series_from_year(u, v, dates, idx); del u, v
        cf, wk, aep = simulate_cf(t, ws, wd, xy)
        print(f"[validate] (53.5,1.5)->({alat:.3f},{alon:.3f}) cache-2020 box{args.spacing}D "
              f"CF={cf*100:.1f}% wake={wk*100:.1f}% (raw-file baseline 56.1%/6.1%)  {time.time()-t0:.0f}s")

    if args.rank:
        rm = pd.read_parquet("${SEAWINDS_ROOT}/scripts/results/siting_rscan_all_s0/resource_map.parquet")
        latmin, latmax = rm.lat.min(), rm.lat.max(); lonmin, lonmax = rm.lon.min(), rm.lon.max()
        elig = rm[(rm.depth_m > 0) & (rm.depth_m <= 50) & np.isfinite(rm.cf_all)].copy()
        elig = elig[(elig.lat > latmin + 0.09) & (elig.lat < latmax - 0.09) &
                    (elig.lon > lonmin + 0.16) & (elig.lon < lonmax - 0.16)]
        pool = pd.concat([elig.sort_values("cf_all", ascending=False).head(80),
                          elig.sort_values("cf_min_yr", ascending=False).head(80)]).drop_duplicates("point_id")
        pool["gl"] = (pool.lat * 10).round(); pool["go"] = (pool.lon * 10).round()
        pool = pool.sort_values("cf_all", ascending=False).drop_duplicates(["gl", "go"]).head(args.topk).reset_index(drop=True)
        idxs = [nearest_idx(c.lat, c.lon)[0] for _, c in pool.iterrows()]
        cf = {i: [] for i in range(len(pool))}; wake = {i: [] for i in range(len(pool))}
        for y in ALL_YEARS:                       # load each year ONCE, score all candidates
            u, v, dates = load_year(y); tt = _times_for(dates)
            for k, gi in enumerate(idxs):
                uu = u[:, :, gi].reshape(-1); vv = v[:, :, gi].reshape(-1)
                ok = np.isfinite(uu) & np.isfinite(vv)
                ws = np.hypot(uu, vv) * SHEAR; wd = (270 - np.degrees(np.arctan2(vv, uu))) % 360
                c, w, _ = simulate_cf(tt[ok], ws[ok], wd[ok], xy)
                cf[k].append(c); wake[k].append(w)
            del u, v
            print(f"  year {y} scored ({time.time()-t0:.0f}s)", flush=True)
        res = []
        for k in range(len(pool)):
            a = np.array(cf[k]); res.append((pool.lat[k], pool.lon[k], pool.depth_m[k],
                                             a.mean(), a.min(), a.std(), np.mean(wake[k])))
        res.sort(key=lambda z: -z[4])
        print(f"\n[rank] {len(pool)} centres, box{args.spacing}D, farm CF over {ALL_YEARS} (by worst-year CF):")
        for lat, lon, dep, m, mn, sd, wk in res:
            print(f"  ({lat:.2f},{lon:.2f}) d={dep:3.0f}m  CF mean={m*100:.2f}% min={mn*100:.2f}% std={sd*100:.2f} wake={wk*100:.1f}%")
        print(f"elapsed {time.time()-t0:.0f}s")
