"""Ship the ensemble-MEAN direction center (: footprint circular-Winkler leave-year-
out d7 -28.6, d14 -20.4 vs my shipped single-member FM-raw). Patches v5-raw's d7-dir
and d14-dir with the ens-mean center + a 2017-2020-calibrated symmetric arc.

--mode fit  : hw[lead,hour] = Q0.90(footprint resid from ens-mean center) over 2017-2020;
              + LOO decision d14: ens-mean center vs climatology center. -> ens_rebuild_fit.json
--mode apply: eval-window ens-mean dir -> patch v5-raw predictions.csv (h7 always ens-mean;
              h14 ens-mean iff fit says it beats clim) -> new CSV + zip. Mirrors
              p2_patch_v5_fmdir positional (window-major->hour->footprint _FP order).
"""
from __future__ import annotations

import os
import os, sys, json, glob, zipfile, time, argparse
os.environ.setdefault("PHASE2_DATA_ROOT", os.environ["SEAWINDS_PHASE2_DIR"])
sys.path.insert(0, "${SEAWINDS_ROOT}/scripts")
import numpy as np, pandas as pd
import p2_forecast_mos as MO
import p2_siting as S
import seawinds_metric as M
import p2_windows as W                # infer_dir / windows (env-overridable window set)
import p2_predcsv as PC              # read/write/validate the predictions frame
from p2_dir_asym_cv import build_dir_clim

ROOT = os.environ.get("SEAWINDS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIT_DIRS = [f"{ROOT}/scripts/results/fm_ens_cv2020_s0", f"{ROOT}/scripts/results/fm_ens_train_s0"]
EVAL_DIR = os.environ.get("FM_ENS_EVAL_DIR", f"{ROOT}/scripts/results/fm_ens_eval_s0")
V5ZIP = os.environ.get("ENS_REBUILD_SRC_ZIP",
                       f"{ROOT}/submission/codabench/852513__predictions_v5raw.zip")
FITJSON = os.environ.get("ENS_REBUILD_FIT", f"{ROOT}/models_final/ens_rebuild_fit.json")
HMAP = {0: 0, 6: 1, 12: 2, 18: 3}
LEADS = [7, 14]


def load_ens(dirs):
    df = pd.concat([pd.read_parquet(f) for d in dirs for f in sorted(glob.glob(f"{d}/ens_feats_*.parquet"))],
                   ignore_index=True)
    p = df["tag"].str.rsplit("_", n=1, expand=True)
    df["base"] = p[0]; df["member"] = p[1].str.replace("m", "").astype(int)
    return df


def ensmean_dir_fp(df_blh):
    """df for one (base/window, lead, hour): mean u1000,v1000 over members -> footprint dir."""
    lead = int(df_blh["lead"].iloc[0]); hour = int(df_blh["hour"].iloc[0])
    gm = df_blh.groupby(["lat", "lon"], as_index=False)[["u1000", "v1000"]].mean()
    gm["lead"] = lead; gm["hour"] = hour
    fu = MO.coarse_field_to_fp(MO.to_grid(gm, lead, hour, "u1000"))
    fv = MO.coarse_field_to_fp(MO.to_grid(gm, lead, hour, "v1000"))
    return (270.0 - np.degrees(np.arctan2(fv, fu))) % 360.0


def truth_dir_fp(year, issue, lead, hour, cache):
    if year not in cache:
        cache.clear()
        u, v, dates = S.load_year(year)
        cache[year] = (u, v, {int(d): i for i, d in enumerate(dates)})
    u, v, didx = cache[year]
    vint = int((pd.Timestamp(issue) + pd.Timedelta(days=lead)).strftime("%Y%m%d"))
    if vint not in didx:
        return None
    di = didx[vint]; hi = HMAP[hour]
    return (270.0 - np.degrees(np.arctan2(v[di, hi, :], u[di, hi, :]))) % 360.0


def fit():
    t0 = time.time()
    df = load_ens(FIT_DIRS)
    df["issue"] = df["base"].str.split("_").str[-1]; df["year"] = df["issue"].str[:4].astype(int)
    cache = {}
    # per (year,lead,hour): emean dir + truth (footprint), for hw + d14 decision
    acc = {}
    for (issue, lead, hour), g in df[df.lead.isin(LEADS)].groupby(["issue", "lead", "hour"]):
        year = int(issue[:4])
        td = truth_dir_fp(year, issue, lead, hour, cache)
        if td is None:
            continue
        em = ensmean_dir_fp(g)
        ok = np.isfinite(td) & np.isfinite(em)
        acc.setdefault((year, lead, hour), {"em": [], "tr": []})
        acc[(year, lead, hour)]["em"].append(em[ok]); acc[(year, lead, hour)]["tr"].append(td[ok])
    for k in acc:
        acc[k]["em"] = np.concatenate(acc[k]["em"]); acc[k]["tr"] = np.concatenate(acc[k]["tr"])
    del df

    # hw[lead,hour] = Q0.90 resid over ALL fit years
    hw = {}
    for lead in LEADS:
        for hour in (0, 6, 12, 18):
            r = np.concatenate([M.circular_distance(acc[k]["tr"], acc[k]["em"])
                                for k in acc if k[1] == lead and k[2] == hour])
            hw[f"{lead}_{hour}"] = float(np.percentile(r, 90))

    # ens-mean circular Winkler per lead (with the all-year hw), + d14 ship decision.
    # d14 ships ens-mean iff it beats my shipped clim d14 ( leave-year-out ref = 324.6,
    # corroborated by A7's proper-LOO 308.4); the info-submit is the final confirm.
    years = sorted(set(k[0] for k in acc))
    def wc(center, truth, hwv):
        return M.circular_winkler(truth, (center - hwv) % 360, (center + hwv) % 360).mean()
    em = {lead: float(np.mean([wc(a["em"], a["tr"], hw[f"{lead}_{k[2]}"]) for k, a in acc.items() if k[1] == lead]))
          for lead in LEADS}
    out = {"hw": hw, "fit_years": years, "emean_winkler_by_lead": em,
           "d14_clim_ref_A2": 324.6, "ship_d14_emean": bool(em[14] < 324.6)}
    json.dump(out, open(FITJSON, "w"), indent=1)
    print(f"hw: {json.dumps(hw)}")
    print(f"ens-mean circWinkler d7={em[7]:.1f} d14={em[14]:.1f} | clim-d14 ref 324.6 -> ship d14 ens-mean: {out['ship_d14_emean']}")
    print(f"fit done ({time.time()-t0:.0f}s) -> {FITJSON}")


def apply_():
    t0 = time.time()
    fitj = json.load(open(FITJSON)); hw = fitj["hw"]; ship14 = fitj["ship_d14_emean"]
    fp_lat = MO._FP["lat"].to_numpy(float).round(2); fp_lon = MO._FP["lon"].to_numpy(float).round(2)
    ev = load_ens([EVAL_DIR])
    wins = W.windows()
    fmblocks = {7: [], 14: []}
    for wid, widx in wins:
        for lead in LEADS:
            for hour in (0, 6, 12, 18):
                g = ev[(ev.base == f"window_{wid}") & (ev.lead == lead) & (ev.hour == hour)]
                assert len(g), f"no ensemble features for window_{wid} lead {lead} hour {hour} in {EVAL_DIR}"
                d50 = ensmean_dir_fp(g)
                fmblocks[lead].append(pd.DataFrame({"window": widx, "hour": hour,
                    "latitude": fp_lat, "longitude": fp_lon, "d50": d50}))
    del ev
    v = PC.read_predictions(V5ZIP)
    print(f"source loaded rows={len(v)} from {V5ZIP} ({time.time()-t0:.0f}s)")
    v_new = v.copy()
    for lead in LEADS:
        if lead == 14 and not ship14:
            print("keeping CLIM d14 (fit says ens-mean does not beat clim)"); continue
        fm = pd.concat(fmblocks[lead], ignore_index=True)
        m = (v["horizon"] == lead).to_numpy()
        vh = v[m].reset_index(drop=True)
        assert len(vh) == len(fm), (lead, len(vh), len(fm))
        assert (vh["window"].to_numpy() == fm["window"].to_numpy()).all() and (vh["hour"].to_numpy() == fm["hour"].to_numpy()).all()
        assert np.allclose(vh["latitude"].to_numpy(), fm["latitude"].to_numpy(), atol=1e-6)
        assert np.allclose(vh["longitude"].to_numpy(), fm["longitude"].to_numpy(), atol=1e-6)
        d50 = fm["d50"].to_numpy()
        hwv = np.array([hw[f"{lead}_{h}"] for h in fm["hour"].to_numpy()])
        v_new.loc[m, "dir_50"] = np.round(d50 % 360, 3) % 360.0
        v_new.loc[m, "dir_05"] = np.round((d50 - hwv) % 360, 3) % 360.0
        v_new.loc[m, "dir_95"] = np.round((d50 + hwv) % 360, 3) % 360.0
        print(f"patched d{lead}: dir_50 changed on {(v.loc[m,'dir_50'].to_numpy()!=v_new.loc[m,'dir_50'].to_numpy()).mean()*100:.0f}% rows, hw~{hwv.mean():.0f}")
    # validate: d1 dir + all speed unchanged; ranges
    m1 = (v["horizon"] == 1).to_numpy()
    for c in ("dir_05", "dir_50", "dir_95", "q05", "q50", "q95"):
        assert np.allclose(v.loc[m1, c].to_numpy(), v_new.loc[m1, c].to_numpy()), f"d1 {c} changed"
    assert np.allclose(v["q05"], v_new["q05"]) and np.allclose(v["q95"], v_new["q95"]), "speed changed"
    assert ((v_new[["dir_05", "dir_50", "dir_95"]] >= 0) & (v_new[["dir_05", "dir_50", "dir_95"]] < 360)).all().all()
    assert (v_new["q05"] <= v_new["q50"]).all() and (v_new["q50"] <= v_new["q95"]).all()
    PC.validate(v_new, len(wins))
    dst = os.environ.get("ENS_REBUILD_OUT_CSV",
                         "./work/predictions_v6ens/predictions.csv")
    zdst = os.environ.get("ENS_REBUILD_OUT_ZIP", "")
    PC.write_predictions(v_new, dst, zdst or None)
    print(f"VALIDATED. wrote {zdst or dst} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True, choices=["fit", "apply"])
    a = ap.parse_args()
    (fit if a.mode == "fit" else apply_)()
