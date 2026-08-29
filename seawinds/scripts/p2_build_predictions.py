"""Assemble the Phase-2 forecast submission (predictions.csv) for the 8 eval
windows, using per-dim-best models: MOS for d1(speed+dir) + d7(dir), climatology
for d7(speed) + d14(both). MOS applied to each window's context_hres → coarse →
bilinear to footprint. 4,196,640 rows (43,715 × 8 × 3 × 4).

Schema (Phase-1 grid): type,window,region,latitude,longitude,horizon,hour,level,
q05,q50,q95,dir_05,dir_50,dir_95. window=id-1 (0-indexed). SLURM (clim loads cache).
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("PHASE2_DATA_ROOT", os.environ["SEAWINDS_PHASE2_DIR"])
KIT = "${SEAWINDS_ROOT}/kit_phase2/phase_2"
for sub in ("", "/part0_dataset_setup", "/part1_forecast"):
    sys.path.insert(0, KIT + sub)
sys.path.insert(0, "${SEAWINDS_ROOT}/scripts")
import numpy as np, pandas as pd, joblib
import forecast_hres as fh
import p2_forecast_mos as MO         # coarse_field_to_fp, to_grid, _FP
import p2_forecast_cv as FC          # build_clim
import p2_windows as W               # infer_dir / windows (env-overridable window set)

P2 = os.environ["SEAWINDS_PHASE2_DIR"]; INF = W.infer_dir()
MODELS = os.environ.get("P2_MOS_MODELS",
    "${SEAWINDS_ROOT}/scripts/results/mos_retrain_1619_c20_s0/mos_models.joblib")
OUTDIR = os.environ.get("P2_BASE_OUTDIR", "./work/predictions_v3raw")
COLS = ["type", "window", "region", "latitude", "longitude", "horizon", "hour", "level",
        "q05", "q50", "q95", "dir_05", "dir_50", "dir_95"]
SPD_SCALE_D1 = 1.0           # INERT in the shipped chain — see below before changing it.
# Stage [5/5] p2_mos_ctx_ship.apply_ rewrites every lead-1 q05/q50/q95 row (in the
# --no-fm branch too), so this value never reaches the artifact; the LIVE d+1 width knob
# is p2_mos_ctx_ship.SPD_SCALE_D1, pinned to 0.90 from a server bracket (ledger ).
# The old claim that "the CQR is well-calibrated at scale 1.0" was never tested and the
# server refuted it: 1.0 shipped a ~0.20-Winkler over-wide interval at 93.6 % coverage.
HMAP = {0: 0, 6: 1, 12: 2, 18: 3}

def window_features(ch, cend):
    lat = ch.latitude.values; lon = ch.longitude.values
    blocks = []
    for L in (1, 7):
        V = cend + pd.Timedelta(days=L); woy = V.isocalendar().week
        for H in (0, 6, 12, 18):
            sp = ch[f"fcst_speed_d{L}_h{H}"].values; di = ch[f"fcst_dir_d{L}_h{H}"].values
            fu, fv = fh._uv_from_speed_dir(sp, di)
            blocks.append(pd.DataFrame({"lat": lat, "lon": lon, "lead": L, "hour": H,
                "fcst_u": fu, "fcst_v": fv, "fcst_speed": sp,
                "woy_sin": np.sin(2 * np.pi * woy / 52.0), "woy_cos": np.cos(2 * np.pi * woy / 52.0)}))
    return pd.concat(blocks, ignore_index=True)

def main():
    t0 = time.time()
    m = joblib.load(MODELS)
    qmos, mmos, adj, doff = m["qmos"], m["mmos"], m["adj"], m["doff"]
    print(f"models loaded (train {m['train_years']}). building clim ...", flush=True)
    clim = FC.load_or_build_clim([2016, 2017, 2018, 2019, 2020])
    print(f"clim built ({time.time()-t0:.0f}s)", flush=True)
    fp_lat = MO._FP["lat"].to_numpy(float).round(2)
    fp_lon = MO._FP["lon"].to_numpy(float).round(2)
    out = []
    wins = W.windows(INF)
    for wid, widx in wins:
        md = json.load(open(f"{INF}/window_{wid}/metadata.json"))
        cend = pd.Timestamp(md["context_end"])
        ch = pd.read_parquet(f"{INF}/window_{wid}/context_hres_north_sea.parquet")
        feat = window_features(ch, cend)
        qp = fh.predict_quantile_mos(qmos, feat, adjust=adj)
        mp = fh.predict_mos(mmos, feat)
        qp = qp.merge(mp[["lat", "lon", "lead", "hour", "u_pred", "v_pred"]],
                      on=["lat", "lon", "lead", "hour"], how="left")
        for H in (1, 7, 14):
            vday = cend + pd.Timedelta(days=H); mon = vday.month
            for hour in (0, 6, 12, 18):
                hidx = HMAP[hour]; cl = clim[(mon, hidx)]
                if H in (1, 7):
                    fu = MO.coarse_field_to_fp(MO.to_grid(qp, H, hour, "u_pred"))
                    fv = MO.coarse_field_to_fp(MO.to_grid(qp, H, hour, "v_pred"))
                    d50 = (270 - np.degrees(np.arctan2(fv, fu))) % 360        # MOS direction
                    hw = min(float(doff[H]), 179.0)
                    if H == 1:                                               # MOS speed (scale 1.3)
                        q05 = MO.coarse_field_to_fp(MO.to_grid(qp, H, hour, "spd_q05"))
                        q50 = MO.coarse_field_to_fp(MO.to_grid(qp, H, hour, "spd_q50"))
                        q95 = MO.coarse_field_to_fp(MO.to_grid(qp, H, hour, "spd_q95"))
                        lo = np.clip(q50 - (q50 - q05) * SPD_SCALE_D1, 0, None); hi = q50 + (q95 - q50) * SPD_SCALE_D1; md50 = q50
                    else:                                                    # d7 speed -> clim
                        lo, md50, hi = cl["q05"].copy(), cl["q50"].copy(), cl["q95"].copy()
                    dd50 = d50; dd_hw = hw
                else:                                                        # d14 -> clim both
                    lo, md50, hi = cl["q05"].copy(), cl["q50"].copy(), cl["q95"].copy()
                    dd50 = cl["dir50"]; dd_hw = np.clip(cl["dhw"], 0, 179.0)
                q = np.sort(np.stack([lo, md50, hi], 1), axis=1)
                lo, md50, hi = np.clip(q[:, 0], 0, None), q[:, 1], q[:, 2]
                d05 = (dd50 - dd_hw) % 360; d95 = (dd50 + dd_hw) % 360
                out.append(pd.DataFrame({
                    "type": "grid", "window": np.int32(widx), "region": "north_sea",
                    "latitude": fp_lat, "longitude": fp_lon, "horizon": np.int32(H),
                    "hour": np.int32(hour), "level": "125m",
                    "q05": lo.round(3), "q50": md50.round(3), "q95": hi.round(3),
                    "dir_05": (d05 % 360).round(3), "dir_50": (dd50 % 360).round(3), "dir_95": (d95 % 360).round(3)}))
        print(f"  window {wid} (id-1={widx}) done ({time.time()-t0:.0f}s)", flush=True)
    df = pd.concat(out, ignore_index=True)[COLS]
    assert len(df) == 43715 * len(wins) * 3 * 4, len(df)
    assert (df["q05"] <= df["q95"]).all(), "q05>q95"
    for c in ("dir_05", "dir_50", "dir_95"):        # round-then-mod so 359.9997 doesn't become 360.0
        df[c] = (df[c] % 360.0).round(3) % 360.0
    dd = df[["dir_05", "dir_50", "dir_95"]]
    assert ((dd >= 0) & (dd < 360)).all().all(), "dir out of [0,360)"
    assert df["q05"].notna().all() and (df["q05"] >= 0).all(), "speed NaN/negative"
    os.makedirs(OUTDIR, exist_ok=True)
    dst = f"{OUTDIR}/predictions.csv"
    df.to_csv(dst, index=False)
    sz = os.path.getsize(dst) / 1e6
    print(f"\nwrote {dst}  rows={len(df):,}  {sz:.0f}MB  ({time.time()-t0:.0f}s)")
    man = os.path.join(os.environ.get("EXP_OUTPUT_DIR", OUTDIR), "predictions_manifest.json")
    json.dump({"rows": int(len(df)), "windows": len(wins), "mb": round(sz, 1),
               "models_train_years": m["train_years"], "spd_scale_d1": SPD_SCALE_D1,
               "per_dim": {"d1": "MOS spd+dir", "d7": "clim spd + MOS dir", "d14": "clim spd+dir"}},
              open(man, "w"), indent=1)
    print(f"manifest {man}")

if __name__ == "__main__":
    main()
