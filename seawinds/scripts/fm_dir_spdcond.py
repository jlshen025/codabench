"""Ship the speed-conditioned Winkler-optimal direction arc (the P0 dir_d1 lever:
LOO d1-dir 145→100.8, beating rivals ~112). Narrow arc in stable high wind, very wide
in the low-wind/frontal tail; per-speed-bin width fit to MINIMISE circular Winkler.

--mode fit  : per lead {1,7,14}, LOO validate spdcond vs base(Q0.90) on the ens-mean
              center; fit all-years per-speed-bin widths + edges; ship_leads = spdcond
              beats base by >3. -> ens_spdcond_fit.json
--mode apply: eval-window ens-mean dir center + ens-mean speed -> per-bin arc -> patch
              v6ens dir_05/50/95 for the shipping leads -> predictions_v8.zip.
Center for ALL dir leads = ens-mean (d1 switches from MOS; d7/d14 keep the v6ens center,
new arc). Symmetric arc (the validated construction). Cross-year risk (A9) mitigated by
LOO + ~90% coverage + large-sample fit; info-submit confirms on 2021 (v6ens = fallback).
"""
from __future__ import annotations

import os
import os, sys, glob, json, zipfile, time, argparse
os.environ.setdefault("PHASE2_DATA_ROOT", os.environ["SEAWINDS_PHASE2_DIR"])
sys.path.insert(0, "${SEAWINDS_ROOT}/scripts")
import numpy as np, pandas as pd
import p2_forecast_mos as MO
import p2_siting as S
import seawinds_metric as M
import p2_windows as W                # infer_dir / windows (env-overridable window set)
import p2_predcsv as PC              # read/write/validate the predictions frame

ROOT = os.environ.get("SEAWINDS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIT_DIRS = [f"{ROOT}/scripts/results/fm_ens_cv2020_s0", f"{ROOT}/scripts/results/fm_ens_train_s0"]
EVAL_DIR = os.environ.get("FM_ENS_EVAL_DIR", f"{ROOT}/scripts/results/fm_ens_eval_s0")
V6ZIP = os.environ.get("SPDCOND_SRC_ZIP",
                       f"{ROOT}/submission/codabench/853160__predictions_v6ens.zip")
FITJSON = os.environ.get("ENS_SPDCOND_FIT", f"{ROOT}/models_final/ens_spdcond_fit.json")
HMAP = {0: 0, 6: 1, 12: 2, 18: 3}
LEADS = [1, 7, 14]
WGRID = np.arange(10, 185, 5.0)
NBIN = int(os.environ.get("DIR_NBIN", "10"))   # A14: 20 ties 10 on LOO with marginally
                                               # lower fold-sd; env-overridable so the
                                               # final build can take the smoother map.


def wdir(u, v):
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def load_ens(dirs):
    df = pd.concat([pd.read_parquet(f) for d in dirs for f in sorted(glob.glob(f"{d}/ens_feats_*.parquet"))],
                   ignore_index=True)
    p = df["tag"].str.rsplit("_", n=1, expand=True)
    df["base"] = p[0]; df["member"] = p[1].str.replace("m", "").astype(int)
    df["spd"] = np.hypot(df["u1000"], df["v1000"])
    return df


def emean_dir_spd_fp(g_blh):
    """(dir, spd) at footprint from ens-mean u,v and ens-mean speed, one (base,lead,hour)."""
    lead = int(g_blh["lead"].iloc[0]); hour = int(g_blh["hour"].iloc[0])
    gm = g_blh.groupby(["lat", "lon"], as_index=False).agg(u1000=("u1000", "mean"), v1000=("v1000", "mean"), spd=("spd", "mean"))
    gm["lead"] = lead; gm["hour"] = hour
    fu = MO.coarse_field_to_fp(MO.to_grid(gm, lead, hour, "u1000"))
    fv = MO.coarse_field_to_fp(MO.to_grid(gm, lead, hour, "v1000"))
    fs = MO.coarse_field_to_fp(MO.to_grid(gm, lead, hour, "spd"))
    return wdir(fu, fv), fs


def wk(truth, c, hw):
    return M.circular_winkler(truth, (c - hw) % 360, (c + hw) % 360)


def opt_w(c, tr):
    s = [wk(tr, c, w).mean() for w in WGRID]
    return float(WGRID[int(np.argmin(s))])


def fit_bins(Dc, Tr, Sp):
    edges = np.quantile(Sp, np.linspace(0, 1, NBIN + 1))
    gw = opt_w(Dc, Tr)
    widths = []
    for b in range(NBIN):
        sel = (Sp >= edges[b]) & (Sp <= edges[b + 1]) if b == NBIN - 1 else (Sp >= edges[b]) & (Sp < edges[b + 1])
        widths.append(opt_w(Dc[sel], Tr[sel]) if sel.sum() > 50 else gw)
    return edges, np.array(widths)


def apply_bins(sp, edges, widths):
    idx = np.clip(np.digitize(sp, edges[1:-1]), 0, len(widths) - 1)
    return widths[idx]


def assemble(df, lead):
    tc = {}
    def truth(year, issue, hour):
        if year not in tc:
            tc.clear(); u, v, dts = S.load_year(year); tc[year] = (u, v, {int(d): i for i, d in enumerate(dts)})
        u, v, didx = tc[year]
        vint = int((pd.Timestamp(issue) + pd.Timedelta(days=lead)).strftime("%Y%m%d"))
        if vint not in didx: return None
        return wdir(u[didx[vint], HMAP[hour], :], v[didx[vint], HMAP[hour], :])
    acc = {}
    for (issue, hour), g in df[df.lead == lead].groupby(["issue", "hour"]):
        year = int(issue[:4]); td = truth(year, issue, hour)
        if td is None: continue
        cd, cs = emean_dir_spd_fp(g)
        ok = np.isfinite(td) & np.isfinite(cd) & np.isfinite(cs)
        k = (year, hour); acc.setdefault(k, {"dir": [], "spd": [], "tr": []})
        acc[k]["dir"].append(cd[ok]); acc[k]["spd"].append(cs[ok]); acc[k]["tr"].append(td[ok])
    for k in acc:
        for kk in acc[k]: acc[k][kk] = np.concatenate(acc[k][kk])
    return acc


def fit():
    t0 = time.time()
    df = load_ens(FIT_DIRS)
    df["issue"] = df["base"].str.split("_").str[-1]
    out = {"leads": {}}
    for lead in LEADS:
        acc = assemble(df, lead)
        years = sorted({k[0] for k in acc})
        wb, ws = [], []
        for ey in years:
            tr = [k for k in acc if k[0] != ey]
            Dc = np.concatenate([acc[k]["dir"] for k in tr]); Tr = np.concatenate([acc[k]["tr"] for k in tr]); Sp = np.concatenate([acc[k]["spd"] for k in tr])
            base_hw = np.percentile(M.circular_distance(Tr, Dc), 90)
            edges, widths = fit_bins(Dc, Tr, Sp)
            for hour in (0, 6, 12, 18):
                if (ey, hour) not in acc: continue
                a = acc[(ey, hour)]
                wb.append(wk(a["tr"], a["dir"], base_hw).mean())
                ws.append(wk(a["tr"], a["dir"], apply_bins(a["spd"], edges, widths)).mean())
        Wb, Ws = float(np.mean(wb)), float(np.mean(ws))
        # fit on ALL years for the ship
        Dc = np.concatenate([acc[k]["dir"] for k in acc]); Tr = np.concatenate([acc[k]["tr"] for k in acc]); Sp = np.concatenate([acc[k]["spd"] for k in acc])
        edges, widths = fit_bins(Dc, Tr, Sp)
        ship = bool(Ws < Wb - 3.0)
        out["leads"][str(lead)] = dict(loo_base=Wb, loo_spdcond=Ws, ship=ship,
                                       edges=[float(x) for x in edges], widths=[float(x) for x in widths])
        print(f"d{lead:<2}: LOO base {Wb:6.1f} -> spdcond {Ws:6.1f}  ({Ws-Wb:+.1f})  ship={ship}  widths(lo->hi){np.round(widths,0)}", flush=True)
    json.dump(out, open(FITJSON, "w"), indent=1)
    print(f"fit done ({time.time()-t0:.0f}s) -> {FITJSON}", flush=True)


def apply_():
    t0 = time.time()
    fj = json.load(open(FITJSON))
    # GUARD : the fit's `ship` flag is a LOO test, and LOO says ship=True for d7
    # too — but d7 arc-conditioning REGRESSED on the withheld year with BOTH conditioners
    # (server 308.96 / 325.46 vs base 293.92). LOO alone must never select a lead here, so
    # the shipped leads are the intersection of the LOO flag and an explicit allow-list.
    allow = {int(x) for x in os.environ.get("SHIP_LEADS", "1").split(",") if x.strip()}
    loo_ok = [int(l) for l, d in fj["leads"].items() if d["ship"]]
    ship_leads = [l for l in loo_ok if l in allow]
    print(f"LOO says ship={loo_ok}; allow-list={sorted(allow)} -> shipping {ship_leads}", flush=True)
    if set(loo_ok) - allow:
        print(f"  BLOCKED by allow-list (cross-year-fragile): {sorted(set(loo_ok)-allow)}", flush=True)
    fp_lat = MO._FP["lat"].to_numpy(float).round(2); fp_lon = MO._FP["lon"].to_numpy(float).round(2)
    ev = load_ens([EVAL_DIR])
    wins = W.windows()
    blocks = {l: [] for l in ship_leads}
    for wid, widx in wins:
        for lead in ship_leads:
            ed = np.array(fj["leads"][str(lead)]["edges"]); ww = np.array(fj["leads"][str(lead)]["widths"])
            for hour in (0, 6, 12, 18):
                g = ev[(ev.base == f"window_{wid}") & (ev.lead == lead) & (ev.hour == hour)]
                assert len(g), f"no ensemble features for window_{wid} lead {lead} hour {hour} in {EVAL_DIR}"
                cd, cs = emean_dir_spd_fp(g); hw = apply_bins(cs, ed, ww)
                blocks[lead].append(pd.DataFrame({"window": widx, "hour": hour, "latitude": fp_lat, "longitude": fp_lon,
                    "d50": cd, "hw": hw}))
    del ev
    v = PC.read_predictions(V6ZIP)
    for lead in ship_leads:
        fm = pd.concat(blocks[lead], ignore_index=True)
        m = (v["horizon"] == lead).to_numpy(); vh = v[m].reset_index(drop=True)
        assert len(vh) == len(fm) and (vh["window"].to_numpy() == fm["window"].to_numpy()).all() and (vh["hour"].to_numpy() == fm["hour"].to_numpy()).all()
        assert np.allclose(vh["latitude"].to_numpy(), fm["latitude"].to_numpy(), atol=1e-6) and np.allclose(vh["longitude"].to_numpy(), fm["longitude"].to_numpy(), atol=1e-6)
        d50 = fm["d50"].to_numpy(); hw = np.clip(fm["hw"].to_numpy(), 0, 179.9)
        v.loc[m, "dir_50"] = np.round(d50 % 360, 3) % 360.0
        v.loc[m, "dir_05"] = np.round((d50 - hw) % 360, 3) % 360.0
        v.loc[m, "dir_95"] = np.round((d50 + hw) % 360, 3) % 360.0
        print(f"patched d{lead} dir (speed-cond arc, mean hw {hw.mean():.0f})", flush=True)
    PC.validate(v, len(wins))
    TAG = os.environ.get("BUILD_TAG", "v8")
    dst = os.environ.get("SPDCOND_OUT_CSV",
                         f"./work/predictions_{TAG}/predictions.csv")
    zdst = os.environ.get("SPDCOND_OUT_ZIP", "")
    PC.write_predictions(v, dst, zdst or None)
    print(f"VALIDATED. wrote {zdst or dst} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True, choices=["fit", "apply"])
    (fit if ap.parse_args().mode == "fit" else apply_)()
