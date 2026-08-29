"""Reproduce the Phase-2 kit siting baseline (plain grid @53.5N,1.5E → CF≈53.8%)
to validate our local PyWake scorer against the organizer's documented method.

Imports the kit's own wind_farm_simulator (fidelity), feeds AROME 125m wind at the
farm centre, shears 125→170m (α=0.11), runs simulate_year → CF / wake / AEP.
Light: one cell, configurable years. Run on login with capped threads.
"""
from __future__ import annotations
import os, sys, glob, time
os.environ.setdefault("PHASE2_DATA_ROOT", os.environ["SEAWINDS_PHASE2_DIR"])
KIT = "${SEAWINDS_ROOT}/kit_phase2/phase_2"
sys.path.insert(0, KIT); sys.path.insert(0, KIT + "/part0_dataset_setup")
import numpy as np, pandas as pd, xarray as xr
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # kit assumes numpy>=2; our stack is numpy<2 (catboost ABI)
import wind_farm_simulator as wfs
import turbines_catalog as tc
import shear
import target_loader as tl

P2 = os.environ["SEAWINDS_PHASE2_DIR"]
AROME_ROOT = f"{P2}/train/arome"
DIAM = 284.0

st = tl.load_static()

def nearest_cell(la, lo):
    d = (st.lat - la) ** 2 + (st.lon - lo) ** 2
    iy, ix = np.unravel_index(np.nanargmin(d), d.shape)
    return int(iy), int(ix), float(st.lat[iy, ix]), float(st.lon[iy, ix])

def load_wind_series(la, lo, years):
    iy, ix, alat, alon = nearest_cell(la, lo)
    T, U, V = [], [], []
    for y in years:
        for f in sorted(glob.glob(f"{AROME_ROOT}/{y}/arome_*.nc")):
            with xr.open_dataset(f) as ds:
                U.append(ds['u125m'].isel(y=iy, x=ix).values)
                V.append(ds['v125m'].isel(y=iy, x=ix).values)
                T.append(pd.to_datetime(ds['time'].values))
    t = np.concatenate(T); u = np.concatenate(U).astype(float); v = np.concatenate(V).astype(float)
    ok = np.isfinite(u) & np.isfinite(v)
    return t[ok], u[ok], v[ok], (alat, alon, iy, ix)

def score(la, lo, years, spacing_d):
    t, u, v, cell = load_wind_series(la, lo, years)
    ws125 = np.hypot(u, v)
    wd = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    ws_hub = ws125 * shear.power_law_factor(125, 170)
    turb = tc.load_turbine("IEA_22MW")
    x, y = wfs.grid_layout(55, spacing_d=spacing_d, diameter_m=DIAM)
    ok, errs = wfs.validate_layout(x, y, box_size_m=15000, max_turbines=55,
                                   min_spacing_d=5.0, diameter_m=DIAM)
    lay = wfs.FarmLayout(x_m=x, y_m=y, turbine=turb)
    wind = wfs.WindSeries(pd.DataFrame({"time": t, "ws": ws_hub, "wd": wd}))
    res = wfs.simulate_year(lay, wind)
    return dict(cell=cell, n_steps=len(t), mean_ws125=float(ws125.mean()),
                mean_wshub=float(ws_hub.mean()), spacing_d=spacing_d, valid=ok, errs=errs,
                CF=res.capacity_factor, wake=res.wake_loss_fraction, AEP=res.aep_gwh,
                rated=res.rated_capacity_mw)

if __name__ == "__main__":
    t0 = time.time()
    turb = tc.load_turbine("IEA_22MW")
    print("IEA_22MW power(15 m/s) MW =", float(turb.power(np.array([15.0]))) / 1e6)
    print("shear factor 125->170 =", shear.power_law_factor(125, 170))
    for spc in (5.0, 7.4):
        r = score(53.5, 1.5, [2020], spc)
        print(f"\n== spacing {spc}D | cell(lat,lon)=({r['cell'][0]:.3f},{r['cell'][1]:.3f}) "
              f"n_steps={r['n_steps']} valid_layout={r['valid']}")
        print(f"   mean ws125={r['mean_ws125']:.2f}  ws_hub={r['mean_wshub']:.2f} m/s")
        print(f"   CF={r['CF']*100:.1f}%  wake={r['wake']*100:.1f}%  AEP={r['AEP']:.0f} GWh  rated={r['rated']:.0f} MW")
    print(f"\nelapsed {time.time()-t0:.0f}s")
