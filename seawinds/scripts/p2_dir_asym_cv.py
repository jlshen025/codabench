"""Leave-year-out CV: ASYMMETRIC circular direction quantiles vs my SYMMETRIC arc.

Two blind, family-diverse adversarial consults independently named this as THE
untried lever: the Winkler-optimal 90% interval is [Q0.05, Q0.95] of the predictive
distribution, NOT a symmetric arc center +- halfwidth. My speed PIs already use
empirical q05/q95 (asymmetric); my DIRECTION arcs are symmetric (dir50 +- dhw).
If the per-(month,hour,point) direction residual is skewed, an asymmetric arc is
tighter at the same nominal coverage -> lower circular Winkler.

Compares, leave-year-out on 2016-2020, direction at d1/d7/d14 (climatology):
  SYM  = dir50 +- Q0.90(|signed residual|)          [current]
  ASYM = dir50 + [Q0.05(signed r), Q0.95(signed r)]  [proposed]
CV-only. d14 ships climatology; d1/d7 ship MOS/FM but the clim-dir asymmetry
signal says whether to extend the lever to those constructions.
"""
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, "${SEAWINDS_ROOT}/scripts")
import numpy as np, pandas as pd
import p2_siting as S
import seawinds_metric as M
from p2_forecast_cv import issue_dates

YEARS = [2016, 2017, 2018, 2019, 2020]
HORIZONS = [1, 7, 14]


def month_of(dates_int):
    return np.array([(d // 100) % 100 for d in dates_int], dtype=np.int32)


def signed_resid(obs_deg, center_deg):
    """circular (obs-center) wrapped to [-180,180]. Broadcasts (n,pts) vs (pts,)."""
    return ((obs_deg - center_deg + 180.0) % 360.0) - 180.0


def build_dir_clim(train_years):
    """center[(month,hidx)] = dict(dir50, dhw, q05, q95) each (npts,).
    dhw=Q0.90(|r|) (symmetric); q05,q95 = 5th/95th pct of SIGNED residual (asymmetric)."""
    Ys = {y: S.load_year(y) for y in train_years}
    mo = {y: month_of(Ys[y][2]) for y in train_years}
    out = {}
    for m in range(1, 13):
        for hidx in range(4):
            Us, Vs = [], []
            for y in train_years:
                u, v, _ = Ys[y]; mm = mo[y] == m
                if mm.any():
                    Us.append(u[mm, hidx, :]); Vs.append(v[mm, hidx, :])
            U = np.concatenate(Us); V = np.concatenate(Vs)
            wd = (270.0 - np.degrees(np.arctan2(V, U))) % 360.0
            s = np.nanmean(np.sin(np.radians(wd)), axis=0); c = np.nanmean(np.cos(np.radians(wd)), axis=0)
            dmean = np.degrees(np.arctan2(s, c)) % 360.0
            r = signed_resid(wd, dmean[None, :])                     # (nsamp, npts), [-180,180]
            dhw = np.nanpercentile(np.abs(r), 90, axis=0)
            q05 = np.nanpercentile(r, 5, axis=0)
            q95 = np.nanpercentile(r, 95, axis=0)
            out[(m, hidx)] = dict(dir50=dmean.astype(np.float32), dhw=dhw.astype(np.float32),
                                  q05=q05.astype(np.float32), q95=q95.astype(np.float32))
    del Ys
    return out


def eval_dir(clim, ydata, eval_year, H, mode):
    u, v, dates = ydata
    dindex = {int(d): i for i, d in enumerate(dates)}
    ssum = 0.0; n = 0; cov = 0; wsum = 0.0
    for iss in issue_dates(eval_year):
        vday = iss + pd.Timedelta(days=H)
        vint = int(vday.strftime("%Y%m%d"))
        if vint not in dindex:
            continue
        di = dindex[vint]; m = vday.month
        for hidx in range(4):
            uu = u[di, hidx, :]; vv = v[di, hidx, :]
            ok = np.isfinite(uu) & np.isfinite(vv)
            y_dir = ((270 - np.degrees(np.arctan2(vv, uu))) % 360)[ok]
            cl = clim[(m, hidx)]; d50 = cl["dir50"][ok]
            if mode == "sym":
                hw = np.clip(cl["dhw"][ok], 0, 180)
                d05 = (d50 - hw) % 360; d95 = (d50 + hw) % 360
            else:  # asym
                lo = np.clip(cl["q05"][ok], -180, 0); hi = np.clip(cl["q95"][ok], 0, 180)
                d05 = (d50 + lo) % 360; d95 = (d50 + hi) % 360
            wdk = M.circular_winkler(y_dir, d05, d95)
            ssum += float(wdk.sum()); n += wdk.size
            wsum += float(((d95 - d05) % 360).sum())
            pos = (y_dir - d05) % 360; cov += int((pos <= ((d95 - d05) % 360)).sum())
    return ssum / max(n, 1), cov / max(n, 1), wsum / max(n, 1)


def main():
    t0 = time.time()
    res = {(H, mode): [] for H in HORIZONS for mode in ("sym", "asym")}
    covd = {(H, mode): [] for H in HORIZONS for mode in ("sym", "asym")}
    widd = {(H, mode): [] for H in HORIZONS for mode in ("sym", "asym")}
    for ey in YEARS:
        ty = [y for y in YEARS if y != ey]
        clim = build_dir_clim(ty)
        ydata = S.load_year(ey)
        for H in HORIZONS:
            for mode in ("sym", "asym"):
                w, c, wd = eval_dir(clim, ydata, ey, H, mode)
                res[(H, mode)].append(w); covd[(H, mode)].append(c); widd[(H, mode)].append(wd)
        del ydata, clim
        print(f"[{time.time()-t0:.0f}s] fold eval={ey} done", flush=True)
    print("\n=== clim direction: SYM vs ASYM circular Winkler (leave-year-out mean+-std) ===", flush=True)
    out = {"years": YEARS, "results": []}
    for H in HORIZONS:
        a = np.array(res[(H, "sym")]); b = np.array(res[(H, "asym")])
        ca = np.mean(covd[(H, "sym")]); cb = np.mean(covd[(H, "asym")])
        wa = np.mean(widd[(H, "sym")]); wb = np.mean(widd[(H, "asym")])
        print(f"  d{H:<2}  SYM {a.mean():6.1f}+-{a.std():4.1f} (cov {ca*100:2.0f}% arc {wa:.0f}) | "
              f"ASYM {b.mean():6.1f}+-{b.std():4.1f} (cov {cb*100:2.0f}% arc {wb:.0f}) | "
              f"delta {b.mean()-a.mean():+.1f}", flush=True)
        out["results"].append(dict(horizon=H, sym_mean=float(a.mean()), sym_std=float(a.std()),
                                   asym_mean=float(b.mean()), asym_std=float(b.std()),
                                   sym_cov=float(ca), asym_cov=float(cb), sym_arc=float(wa),
                                   asym_arc=float(wb), delta=float(b.mean()-a.mean()),
                                   sym_per_year=[float(x) for x in a], asym_per_year=[float(x) for x in b]))
    print("\nADOPT asym for a horizon IF asym_mean < sym_mean beyond fold-std (skewed residual => tighter arc).", flush=True)
    outdir = os.environ.get("EXP_OUTPUT_DIR", ".")
    os.makedirs(outdir, exist_ok=True)
    json.dump(out, open(os.path.join(outdir, "dir_asym_cv.json"), "w"), indent=1)
    print(f"[{time.time()-t0:.0f}s] wrote dir_asym_cv.json", flush=True)


if __name__ == "__main__":
    main()
