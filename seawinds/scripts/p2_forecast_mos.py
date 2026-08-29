"""HRES-MOS forecast CV (d1/d7 climb) at the 43,715-pt footprint.

Pipeline (reuses the kit): quantile LightGBM MOS HRES(10m fcst)->AROME-coarse
speed q05/q50/q95 + conformal CQR; mean-MOS u,v for direction. Coarse->footprint
= BILINEAR interp (kit downscaling: LightGBM adds only ~3% over sea, so bilinear
is near-optimal on the all-sea footprint). Direction PI = dir50 +/- calibrated
circular offset per lead. Scores speed/dir d1/d7 vs the climatology floor
(17.3/16.8 spd, 321/347 dir). d14 stays climatology (measured separately).

Honest split: train MOS 2016-2018, CQR-calibrate 2019, eval 2020 (cross-year).
SLURM CPU job.
"""
from __future__ import annotations
import os, sys, json, argparse, time
os.environ.setdefault("PHASE2_DATA_ROOT", os.environ["SEAWINDS_PHASE2_DIR"])
KIT = "${SEAWINDS_ROOT}/kit_phase2/phase_2"
for sub in ("", "/part0_dataset_setup", "/part1_forecast"):
    sys.path.insert(0, KIT + sub)
sys.path.insert(0, "${SEAWINDS_ROOT}/scripts")
import numpy as np, pandas as pd
import forecast_hres as fh
import downscaling as dsc
import footprint as fp
import p2_siting as S
import seawinds_metric as M

from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
MASK = fp.footprint_mask()
YS, XS = np.where(MASK)
_LATS, _LONS = dsc._reanalysis_axes()            # ascending 45,57 (kit-consistent)
_LI = {round(float(x), 3): i for i, x in enumerate(_LATS)}
_LJ = {round(float(x), 3): j for j, x in enumerate(_LONS)}
_FP = S.footprint_order()                        # point_id order == cache order
_FP_PTS = np.stack([_FP["lat"].to_numpy(float), _FP["lon"].to_numpy(float)], axis=1)

def to_grid(df, lead, hour, col):
    g = np.full((_LATS.size, _LONS.size), np.nan, float)
    s = df[(df["lead"] == lead) & (df["hour"] == hour)]
    ii = s["lat"].round(3).map(_LI).to_numpy(); jj = s["lon"].round(3).map(_LJ).to_numpy()
    vv = s[col].to_numpy()
    ok = ~(pd.isna(ii) | pd.isna(jj))
    g[ii[ok].astype(int), jj[ok].astype(int)] = vv[ok]
    return g

def coarse_field_to_fp(gu):
    """Bilinear-interp one coarse (45,57) field to the 43,715 footprint pts (cache order)."""
    filled = gu.copy()
    if np.isnan(filled).any():
        idx = distance_transform_edt(np.isnan(filled), return_distances=False, return_indices=True)
        filled = filled[tuple(idx)]
    f = RegularGridInterpolator((_LATS, _LONS), filled, bounds_error=False, fill_value=np.nan)
    return f(_FP_PTS)

def truth_fp(u, v, dates_index, vint, hidx):
    di = dates_index.get(vint)
    if di is None:
        return None
    uu = u[di, hidx, :]; vv = v[di, hidx, :]
    ok = np.isfinite(uu) & np.isfinite(vv)
    spd = np.hypot(uu, vv); wd = (270 - np.degrees(np.arctan2(vv, uu))) % 360
    return spd, wd, ok

def issue_dates(year, step=4):
    d0 = pd.Timestamp(f"{year}-01-01"); out = []; d = d0
    while d + pd.Timedelta(days=14) <= pd.Timestamp(f"{year}-12-31"):
        out.append(d); d = d + pd.Timedelta(days=step)
    return out

def dir_offset_calib(mos_mean, calib_df, leads=(1, 7)):
    """90th-pct circular residual (deg) of MOS direction vs coarse truth, per lead."""
    pr = fh.predict_mos(mos_mean, calib_df)
    off = {}
    for L in leads:
        s = pr[pr["lead"] == L]
        dpred = (270 - np.degrees(np.arctan2(s["v_pred"].to_numpy(), s["u_pred"].to_numpy()))) % 360
        dtrue = (270 - np.degrees(np.arctan2(s["v125c"].to_numpy(), s["u125c"].to_numpy()))) % 360
        res = M.circular_distance(dpred, dtrue)
        off[L] = float(np.nanpercentile(res, 90))
    return off

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-years", default="2016,2017,2018")
    ap.add_argument("--calib-year", type=int, default=2019)
    ap.add_argument("--eval-year", type=int, default=2020)
    ap.add_argument("--dir-scale", type=float, default=1.0)
    ap.add_argument("--max-train-iss", type=int, default=0)
    ap.add_argument("--max-eval-iss", type=int, default=0)
    ap.add_argument("--save-models", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    t0 = time.time()
    ty = [int(y) for y in a.train_years.split(",")]

    tr_iss = [d for y in ty for d in issue_dates(y)]
    if a.max_train_iss:
        tr_iss = tr_iss[::max(1, len(tr_iss) // a.max_train_iss)][:a.max_train_iss]
    print(f"build train table ({len(tr_iss)} issue dates) ...", flush=True)
    tr = fh.build_hres_table(tr_iss)
    print(f"train rows={len(tr)} ({time.time()-t0:.0f}s). training MOS ...", flush=True)
    qmos = fh.train_quantile_mos(tr)
    mmos = fh.train_mos(tr)
    ca_iss = issue_dates(a.calib_year)
    if a.max_train_iss:
        ca_iss = ca_iss[:max(6, a.max_train_iss // 2)]
    ca = fh.build_hres_table(ca_iss)
    adj = fh.conformal_adjust(qmos, ca, alpha=0.10)
    doff = dir_offset_calib(mmos, ca)
    print(f"MOS trained ({time.time()-t0:.0f}s). CQR adj={ {k:round(v,2) for k,v in adj.items()} } "
          f"dir_off={ {k:round(v,1) for k,v in doff.items()} }", flush=True)

    if a.save_models:
        import joblib
        md = a.out or os.environ.get("EXP_OUTPUT_DIR", ".")
        os.makedirs(md, exist_ok=True)
        joblib.dump({"qmos": qmos, "mmos": mmos, "adj": {int(k): v for k, v in adj.items()},
                     "doff": {int(k): v for k, v in doff.items()}, "train_years": ty,
                     "calib_year": a.calib_year}, os.path.join(md, "mos_models.joblib"))
        print("saved mos_models.joblib", flush=True)

    u, v, dates = S.load_year(a.eval_year)
    didx = {int(d): i for i, d in enumerate(dates)}
    SPD = [1.0, 1.3, 1.6, 2.0]; DIRS = [1.0, 1.3, 1.6]
    sacc = {(L, i): [0.0, 0, 0, 0.0] for L in (1, 7) for i in range(len(SPD))}  # sum,n,cov,width
    dacc = {(L, j): [0.0, 0, 0] for L in (1, 7) for j in range(len(DIRS))}      # sum,n,cov
    ev_iss = issue_dates(a.eval_year)
    if a.max_eval_iss:
        ev_iss = ev_iss[:a.max_eval_iss]
    for iss in ev_iss:
        tbl = fh.build_hres_table([iss], with_truth=False)
        if tbl.empty:
            continue
        qp = fh.predict_quantile_mos(qmos, tbl, adjust=adj)
        mp = fh.predict_mos(mmos, tbl)
        qp = qp.merge(mp[["lat", "lon", "lead", "hour", "u_pred", "v_pred"]],
                      on=["lat", "lon", "lead", "hour"], how="left")
        for L in (1, 7):
            vday = iss + pd.Timedelta(days=L); vint = int(vday.strftime("%Y%m%d"))
            for hidx, H in enumerate((0, 6, 12, 18)):
                tr_ = truth_fp(u, v, didx, vint, hidx)
                if tr_ is None:
                    continue
                y_spd, y_dir, ok = tr_; y_spd = y_spd[ok]; y_dir = y_dir[ok]
                q05 = coarse_field_to_fp(to_grid(qp, L, H, "spd_q05"))[ok]
                q50 = coarse_field_to_fp(to_grid(qp, L, H, "spd_q50"))[ok]
                q95 = coarse_field_to_fp(to_grid(qp, L, H, "spd_q95"))[ok]
                fu = coarse_field_to_fp(to_grid(qp, L, H, "u_pred"))[ok]
                fv = coarse_field_to_fp(to_grid(qp, L, H, "v_pred"))[ok]
                q = np.sort(np.stack([q05, q50, q95], 1), axis=1)
                q05, q50, q95 = np.clip(q[:, 0], 0, None), q[:, 1], q[:, 2]
                d50 = (270 - np.degrees(np.arctan2(fv, fu))) % 360
                for i, ss in enumerate(SPD):
                    lo = np.clip(q50 - (q50 - q05) * ss, 0, None); hi = q50 + (q95 - q50) * ss
                    w = M.winkler_speed(y_spd, lo, hi); A = sacc[(L, i)]
                    A[0] += float(w.sum()); A[1] += w.size
                    A[2] += int(((y_spd >= lo) & (y_spd <= hi)).sum()); A[3] += float((hi - lo).sum())
                for j, ds in enumerate(DIRS):
                    hw = min(doff[L] * ds, 179.0); d05 = (d50 - hw) % 360; d95 = (d50 + hw) % 360
                    wdk = M.circular_winkler(y_dir, d05, d95); B = dacc[(L, j)]
                    B[0] += float(wdk.sum()); B[1] += wdk.size
                    pos = (y_dir - d05) % 360; B[2] += int((pos <= ((d95 - d05) % 360)).sum())
        print(f"  eval issue {iss:%Y-%m-%d} done ({time.time()-t0:.0f}s)", flush=True)

    floor = {1: (17.3, 321), 7: (16.8, 347)}
    res = {}
    print(f"\n=== HRES-MOS CV: train {ty} calib {a.calib_year} eval {a.eval_year} ===")
    print("SPEED (MOS best-scale Winkler vs clim floor):")
    for L in (1, 7):
        bi = min(range(len(SPD)), key=lambda i: sacc[(L, i)][0] / max(sacc[(L, i)][1], 1))
        A = sacc[(L, bi)]; wk = A[0] / max(A[1], 1)
        res[f"speed_d{L}"] = wk; res[f"speed_d{L}_scale"] = SPD[bi]; res[f"cov_speed_d{L}"] = A[2] / max(A[1], 1)
        allsc = {SPD[i]: round(sacc[(L, i)][0] / max(sacc[(L, i)][1], 1), 2) for i in range(len(SPD))}
        print(f"  d{L}: MOS {wk:.2f} @scale{SPD[bi]} cov{A[2]/max(A[1],1)*100:.0f}% w{A[3]/max(A[1],1):.2f} | clim {floor[L][0]} | {allsc}")
    print("DIRECTION (MOS best-scale cWinkler vs clim floor):")
    for L in (1, 7):
        bj = min(range(len(DIRS)), key=lambda j: dacc[(L, j)][0] / max(dacc[(L, j)][1], 1))
        B = dacc[(L, bj)]; wk = B[0] / max(B[1], 1)
        res[f"dir_d{L}"] = wk; res[f"dir_d{L}_scale"] = DIRS[bj]; res[f"cov_dir_d{L}"] = B[2] / max(B[1], 1)
        allsc = {DIRS[j]: round(dacc[(L, j)][0] / max(dacc[(L, j)][1], 1), 1) for j in range(len(DIRS))}
        print(f"  d{L}: MOS {wk:.1f} @scale{DIRS[bj]} cov{B[2]/max(B[1],1)*100:.0f}% | clim {floor[L][1]} | {allsc}")
    outdir = a.out or os.environ.get("EXP_OUTPUT_DIR", ".")
    os.makedirs(outdir, exist_ok=True)
    dst = os.path.join(outdir, f"mos_cv_eval{a.eval_year}.json")
    json.dump({"train_years": ty, "calib_year": a.calib_year, "eval_year": a.eval_year,
               "cqr_adj": {int(k): v for k, v in adj.items()},
               "dir_offset": {int(k): v for k, v in doff.items()}, "scores": res}, open(dst, "w"), indent=1)
    print(f"wrote {dst} ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
