#!/usr/bin/env python
"""Do CONTEXT-WINDOW surface-layer features improve the MOS? (leave-year-out A/B)

Surfaced by a blind adversarial review of the methodology report: the 10 m -> 125 m
level correction the MOS learns is physically governed by marine surface-layer
stability (Monin-Obukhov), which the current feature set sees only through
week-of-year. The obvious stability fields (`reanalysis_extra`: MSLP, 2 m T,
pressure-level u/v/z/t) are NOT shipped for the inference windows, so they cannot be
used by a deliverable model.

What IS shipped for every inference window is 14 days of reanalysis u10/v10 AND
u100/v100 at every coarse grid point. That gives two directly-measured surface-layer
diagnostics per point, available at inference, with no leakage (all strictly before
the issue time):

  ctx_shear    mean|V100| / mean|V10|   - the LOCAL, RECENT effective shear ratio.
                                          This is exactly the quantity that maps the
                                          HRES 10 m driver onto a 125 m target, and it
                                          absorbs stability + sea-state + roughness.
  ctx_veer     circular(mean dir@100m - mean dir@10m) - Ekman-spiral veer, which is a
                                          direct stability signature and acts on the
                                          DIRECTION mapping (our weakest sub-dimension).
  ctx_v100     mean|V100|               - local recent wind climate
  ctx_v100_sd  sd|V100|                 - recent synoptic variability
  ctx_const    |mean V100| / mean|V100| - directional constancy (regime persistence)

A/B: identical LightGBM MOS, identical CQR calibration, identical folds; the only
difference is the feature set. Leave-year-out over 2017-2020, scored on the coarse
grid with the competition's own speed-Winkler and circular-Winkler.

CV-only, zero submissions. SLURM CPU job.
"""
from __future__ import annotations

import json
import os
import sys
import time
from functools import lru_cache

import numpy as np
import pandas as pd

os.environ.setdefault("PHASE2_DATA_ROOT", os.environ["SEAWINDS_PHASE2_DIR"])
ROOT = os.environ.get("SEAWINDS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KIT = f"{ROOT}/kit_phase2/phase_2"
for sub in ("", "/part0_dataset_setup", "/part1_forecast"):
    sys.path.insert(0, KIT + sub)
sys.path.insert(0, f"{ROOT}/scripts")

import forecast_hres as fh
import seawinds_metric as M

OUT = os.environ.get("EXP_OUTPUT_DIR", ".")
DATA = os.environ["SEAWINDS_PHASE2_DIR"]
YEARS = [2017, 2018, 2019, 2020]
CTX_DAYS = 14
NEW = ["ctx_shear", "ctx_veer", "ctx_v100", "ctx_v100_sd", "ctx_const"]
BASE = list(fh.FEATURES)


@lru_cache(maxsize=4096)
def _day(datestr: str):
    """(u10,v10,u100,v100) flat arrays for one reanalysis day, or None."""
    import xarray as xr
    d = pd.Timestamp(datestr)
    p = f"{DATA}/train/reanalysis/{d.year}/reanalysis_{d:%Y%m%d}.nc"
    if not os.path.exists(p):
        return None
    ds = xr.open_dataset(p)
    out = tuple(ds[v].values.reshape(ds[v].shape[0], -1).astype("float32")
                for v in ("u10", "v10", "u100", "v100"))
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    ds.close()
    LON, LAT = np.meshgrid(lon, lat)
    return out + (LAT.ravel(), LON.ravel())


def context_features(issue: pd.Timestamp) -> pd.DataFrame | None:
    """Per-grid-point surface-layer diagnostics over the 14 days BEFORE `issue`."""
    days = [_day(str((issue - pd.Timedelta(days=k)).date())) for k in range(1, CTX_DAYS + 1)]
    days = [d for d in days if d is not None]
    if len(days) < CTX_DAYS // 2:
        return None
    u10 = np.concatenate([d[0] for d in days]);  v10 = np.concatenate([d[1] for d in days])
    u1c = np.concatenate([d[2] for d in days]);  v1c = np.concatenate([d[3] for d in days])
    lat, lon = days[0][4], days[0][5]

    s10 = np.hypot(u10, v10)
    s100 = np.hypot(u1c, v1c)
    m10, m100 = s10.mean(0), s100.mean(0)
    # resultant (vector-mean) wind at each level -> mean direction and constancy
    ru10, rv10 = u10.mean(0), v10.mean(0)
    ru1c, rv1c = u1c.mean(0), v1c.mean(0)
    d10 = (270.0 - np.degrees(np.arctan2(rv10, ru10))) % 360.0
    d100 = (270.0 - np.degrees(np.arctan2(rv1c, ru1c))) % 360.0
    veer = (d100 - d10 + 180.0) % 360.0 - 180.0          # signed, in (-180, 180]
    return pd.DataFrame({
        "lat": lat, "lon": lon,
        "ctx_shear": m100 / np.maximum(m10, 0.1),
        "ctx_veer": veer,
        "ctx_v100": m100,
        "ctx_v100_sd": s100.std(0),
        "ctx_const": np.hypot(ru1c, rv1c) / np.maximum(m100, 0.1),
    })


STEP = int(os.environ.get("SHEAR_STEP", "8"))     # days between issue dates
MAX_FIT = int(os.environ.get("SHEAR_MAX_FIT", "1500000"))   # rows per LightGBM fit


def issue_dates(year, step=STEP):
    d0 = pd.Timestamp(f"{year}-01-01")
    out, d = [], d0
    while d + pd.Timedelta(days=14) <= pd.Timestamp(f"{year}-12-31"):
        out.append(d)
        d += pd.Timedelta(days=step)
    return out


def build(year):
    """MOS table for one year with the context features merged on (lat, lon)."""
    blocks = []
    for D in issue_dates(year):
        cf = context_features(D)
        if cf is None:
            continue
        t = fh.build_hres_table([D])
        if t.empty:
            continue
        t["_k"] = t.lat.round(3).astype(str) + "_" + t.lon.round(3).astype(str)
        cf["_k"] = cf.lat.round(3).astype(str) + "_" + cf.lon.round(3).astype(str)
        t = t.merge(cf[["_k"] + NEW], on="_k", how="left").drop(columns="_k")
        t["year"] = year
        blocks.append(t)
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


def fit_eval(tr, te, feats):
    """Train quantile+mean MOS on `feats`, CQR-calibrate on a slice of tr, score te."""
    import lightgbm as lgb
    p = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, subsample=0.8,
             colsample_bytree=0.8, verbose=-1, n_jobs=-1)
    res = {}
    for L in (1, 7):
        s_tr = tr[tr.lead == L].dropna(subset=feats)
        s_te = te[te.lead == L].dropna(subset=feats)
        if s_tr.empty or s_te.empty:
            continue
        # hold out the last 20 % of training issues for the conformal step
        cut = s_tr.index[int(0.8 * len(s_tr))]
        fit, cal = s_tr.loc[:cut], s_tr.loc[cut:]
        # bound the fit size so both arms cost the same and the A/B stays cheap;
        # the SAME rows are used for both arms (rs fixed), so the comparison is paired
        if len(fit) > MAX_FIT:
            fit = fit.sample(MAX_FIT, random_state=0)
        if len(cal) > MAX_FIT // 4:
            cal = cal.sample(MAX_FIT // 4, random_state=0)
        y_fit = np.hypot(fit.u125c, fit.v125c)
        qm = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **p)
              .fit(fit[feats], y_fit) for q in (0.05, 0.5, 0.95)}
        mu = lgb.LGBMRegressor(**p).fit(fit[feats], fit.u125c)
        mv = lgb.LGBMRegressor(**p).fit(fit[feats], fit.v125c)

        # conformal widening so the interval reaches ~90 % on the calibration slice
        y_cal = np.hypot(cal.u125c, cal.v125c)
        lo, hi = qm[0.05].predict(cal[feats]), qm[0.95].predict(cal[feats])
        adj = float(np.quantile(np.maximum(lo - y_cal, y_cal - hi), 0.90))
        # direction arc: 90th percentile circular residual on the calibration slice
        du = mu.predict(cal[feats]); dv = mv.predict(cal[feats])
        dp = (270 - np.degrees(np.arctan2(dv, du))) % 360
        dt = (270 - np.degrees(np.arctan2(cal.v125c, cal.u125c))) % 360
        arc = float(np.nanpercentile(M.circular_distance(dp, dt.to_numpy()), 90))

        y = np.hypot(s_te.u125c, s_te.v125c).to_numpy()
        q05 = np.maximum(qm[0.05].predict(s_te[feats]) - adj, 0)
        q95 = qm[0.95].predict(s_te[feats]) + adj
        tu, tv = mu.predict(s_te[feats]), mv.predict(s_te[feats])
        dc = (270 - np.degrees(np.arctan2(tv, tu))) % 360
        truth_d = ((270 - np.degrees(np.arctan2(s_te.v125c, s_te.u125c))) % 360).to_numpy()
        res[f"spd_d{L}"] = float(np.mean(M.winkler_speed(y, q05, q95)))
        res[f"dir_d{L}"] = float(np.mean(
            M.circular_winkler(truth_d, (dc - arc) % 360, (dc + arc) % 360)))
    return res


def main() -> int:
    t0 = time.time()
    tabs = {}
    for y in YEARS:
        tabs[y] = build(y)
        print(f"{y}: {len(tabs[y])} rows ({time.time()-t0:.0f}s)", flush=True)
    all_df = pd.concat(tabs.values(), ignore_index=True)
    print("feature NaN share:",
          {c: round(float(all_df[c].isna().mean()), 4) for c in NEW}, flush=True)

    out = {"folds": {}, "features_new": NEW}
    for y in YEARS:
        tr = all_df[all_df.year != y].reset_index(drop=True)
        te = all_df[all_df.year == y].reset_index(drop=True)
        b = fit_eval(tr, te, BASE)
        n = fit_eval(tr, te, BASE + NEW)
        out["folds"][y] = {"base": b, "plus": n,
                           "delta": {k: round(n[k] - b[k], 3) for k in b}}
        print(f"  fold {y}: base {  {k: round(v,2) for k,v in b.items()} }", flush=True)
        print(f"           plus { {k: round(v,2) for k,v in n.items()} }", flush=True)
        print(f"           DELTA {out['folds'][y]['delta']}  (negative = better)", flush=True)

    keys = sorted(next(iter(out["folds"].values()))["base"])
    out["summary"] = {}
    for k in keys:
        db = [out["folds"][y]["base"][k] for y in YEARS]
        dn = [out["folds"][y]["plus"][k] for y in YEARS]
        d = np.array(dn) - np.array(db)
        out["summary"][k] = dict(base_mean=float(np.mean(db)), plus_mean=float(np.mean(dn)),
                                 delta_mean=float(d.mean()), delta_sd=float(d.std()),
                                 folds_improved=int((d < 0).sum()), n_folds=len(d))
    print("\n=== SUMMARY (leave-year-out; negative delta = the new features help) ===")
    for k, v in out["summary"].items():
        print(f"  {k:8s} base {v['base_mean']:8.2f} -> plus {v['plus_mean']:8.2f}   "
              f"delta {v['delta_mean']:+7.2f} +-{v['delta_sd']:5.2f}   "
              f"{v['folds_improved']}/{v['n_folds']} folds better")
    with open(os.path.join(OUT, "mos_shear_features.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[{time.time()-t0:.0f}s] wrote {OUT}/mos_shear_features.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
