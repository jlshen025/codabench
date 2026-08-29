"""Forecast CV harness (Phase-2, 6 dims). Mirrors the eval: from 2016-2020,
issue dates -> predict AROME 125m at footprint on valid days d+1/d+7/d+14,
hours 0/6/12/18. Scores speed-Winkler + circular-Winkler (server-validated
seawinds_metric) pooled per horizon = 6 dims. Weights cross-YEAR robustness
(final eval swaps year).

v1 model: CLIMATOLOGY (per month x hour speed q05/q50/q95 + circular dir median
+ 90%-half-width from the training years) — the d14 floor, auto-calibrated PIs,
and the pipeline validator. Truth + clim both from the AROME footprint cache.

SLURM job (loads year caches). --train-years / --eval-year for leave-year-out.
"""
from __future__ import annotations
import os, sys, json, argparse, time
sys.path.insert(0, "${SEAWINDS_ROOT}/scripts")
import numpy as np, pandas as pd
import p2_siting as S           # cache loaders: load_year, footprint_order, HOURS
import seawinds_metric as M     # winkler_speed, circular_winkler, circular_distance

HOURS = S.HOURS                 # [0,6,12,18]

def _month_of(dates_int):
    return np.array([(d // 100) % 100 for d in dates_int])

def build_clim(train_years):
    """Per (month, hour) footprint climatology from training years.
    Returns clim[(month,hidx)] = dict(q05,q50,q95, dir50, dhw) each (npts,)."""
    Ys = {y: S.load_year(y) for y in train_years}       # all years in RAM (~0.5GB each)
    months = {y: _month_of(Ys[y][2]) for y in train_years}
    clim = {}
    for m in range(1, 13):
        for hidx in range(4):
            Us, Vs = [], []
            for y in train_years:
                u, v, _ = Ys[y]; mm = months[y] == m
                if mm.any():
                    Us.append(u[mm, hidx, :]); Vs.append(v[mm, hidx, :])
            U = np.concatenate(Us); V = np.concatenate(Vs)      # (nsamp, npts)
            spd = np.hypot(U, V)
            q05, q50, q95 = np.nanpercentile(spd, [5, 50, 95], axis=0)
            wd = (270.0 - np.degrees(np.arctan2(V, U))) % 360.0
            s = np.nanmean(np.sin(np.radians(wd)), axis=0); c = np.nanmean(np.cos(np.radians(wd)), axis=0)
            dmean = np.degrees(np.arctan2(s, c)) % 360.0
            resid = M.circular_distance(wd, dmean[None, :])                 # (nsamp,npts) 0..180
            dhw = np.nanpercentile(resid, 90, axis=0)
            clim[(m, hidx)] = dict(q05=q05.astype(np.float32), q50=q50.astype(np.float32),
                                   q95=q95.astype(np.float32), dir50=dmean.astype(np.float32),
                                   dhw=dhw.astype(np.float32))
    del Ys
    return clim

_CLIM_KEYS = ("q05", "q50", "q95", "dir50", "dhw")


def clim_to_npz(clim, path):
    """Freeze a built climatology to one durable .npz (~40 MB)."""
    flat = {f"{m}_{h}_{k}": clim[(m, h)][k] for (m, h) in clim for k in _CLIM_KEYS}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **flat)
    return path


def clim_from_npz(path):
    z = np.load(path)
    clim = {}
    for name in z.files:
        m, h, k = name.split("_", 2)
        clim.setdefault((int(m), int(h)), {})[k] = z[name]
    assert len(clim) == 48, f"expected 12 months x 4 hours, got {len(clim)} in {path}"
    for v in clim.values():
        assert set(v) == set(_CLIM_KEYS), f"incomplete climatology entry in {path}"
    return clim


def load_or_build_clim(train_years):
    """The climatology is a FIXED function of the 2016-2020 AROME training years — it does
    NOT depend on the inference set. $SEAWINDS_CLIM_NPZ points at the durable frozen copy so
    the endgame rebuild never needs the 2.4 GB purgeable AROME footprint cache; unset, this
    falls back to building it from that cache (identical arrays, by construction)."""
    p = os.environ.get("SEAWINDS_CLIM_NPZ", "")
    if p and os.path.exists(p):
        print(f"clim: loading frozen {p}", flush=True)
        return clim_from_npz(p)
    print("clim: building from the AROME footprint cache", flush=True)
    clim = build_clim(train_years)
    if p:
        clim_to_npz(clim, p)
        print(f"clim: froze -> {p}", flush=True)
    return clim


def issue_dates(year, step=14):
    d0 = pd.Timestamp(f"{year}-01-01")
    out = []
    d = d0
    while d + pd.Timedelta(days=14) <= pd.Timestamp(f"{year}-12-31"):
        out.append(d); d = d + pd.Timedelta(days=step)
    return out

def eval_model(clim, eval_year, pi_scale=(1.0, 1.0)):
    u, v, dates = S.load_year(eval_year)
    dindex = {int(d): i for i, d in enumerate(dates)}
    ss, sd = pi_scale
    # accumulate per (horizon, problem): sum Winkler, count, coverage
    acc = {h: {"spd_sum": 0.0, "spd_n": 0, "spd_cov": 0, "spd_w": 0.0,
               "dir_sum": 0.0, "dir_n": 0, "dir_cov": 0} for h in [1, 7, 14]}
    for iss in issue_dates(eval_year):
        for H in [1, 7, 14]:
            vday = iss + pd.Timedelta(days=H)
            vint = int(vday.strftime("%Y%m%d"))
            if vint not in dindex:
                continue
            di = dindex[vint]; m = vday.month
            for hidx in range(4):
                uu = u[di, hidx, :]; vv = v[di, hidx, :]
                ok = np.isfinite(uu) & np.isfinite(vv)
                y_spd = np.hypot(uu, vv)[ok]
                y_dir = ((270 - np.degrees(np.arctan2(vv, uu))) % 360)[ok]
                cl = clim[(m, hidx)]
                q50 = cl["q50"][ok]
                lo = np.clip(q50 - (q50 - cl["q05"][ok]) * ss, 0, None)
                hi = q50 + (cl["q95"][ok] - q50) * ss
                d50 = cl["dir50"][ok]; hw = np.clip(cl["dhw"][ok] * sd, 0, 180)
                d05 = (d50 - hw) % 360; d95 = (d50 + hw) % 360
                ws = M.winkler_speed(y_spd, lo, hi); wdk = M.circular_winkler(y_dir, d05, d95)
                a = acc[H]
                a["spd_sum"] += float(ws.sum()); a["spd_n"] += ws.size
                a["spd_cov"] += int(((y_spd >= lo) & (y_spd <= hi)).sum())
                a["spd_w"] += float((hi - lo).sum())
                a["dir_sum"] += float(wdk.sum()); a["dir_n"] += wdk.size
                pos = (y_dir - d05) % 360; a["dir_cov"] += int((pos <= ((d95 - d05) % 360)).sum())
    out = {}
    for H in [1, 7, 14]:
        a = acc[H]
        out[f"speed_d{H}"] = a["spd_sum"] / max(a["spd_n"], 1)
        out[f"dir_d{H}"] = a["dir_sum"] / max(a["dir_n"], 1)
        out[f"cov_speed_d{H}"] = a["spd_cov"] / max(a["spd_n"], 1)
        out[f"cov_dir_d{H}"] = a["dir_cov"] / max(a["dir_n"], 1)
        out[f"width_speed_d{H}"] = a["spd_w"] / max(a["spd_n"], 1)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-years", default="2016,2017,2018,2019")
    ap.add_argument("--eval-year", type=int, default=2020)
    ap.add_argument("--pi-scale-spd", type=float, default=1.0)
    ap.add_argument("--pi-scale-dir", type=float, default=1.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    ty = [int(y) for y in a.train_years.split(",")]
    t0 = time.time()
    print(f"build clim from {ty} ...", flush=True)
    clim = build_clim(ty)
    print(f"clim built ({time.time()-t0:.0f}s). eval on {a.eval_year}, sweeping PI scales ...", flush=True)
    sweep = []
    for ss in [1.0, 1.25, 1.5]:
        for sd in [1.0, 1.4, 1.8, 2.2]:
            r = eval_model(clim, a.eval_year, (ss, sd))
            r["pi_scale_spd"] = ss; r["pi_scale_dir"] = sd
            sweep.append(r)
            print(f"[spd x{ss} dir x{sd}] "
                  f"spd d1/d7/d14={r['speed_d1']:.1f}/{r['speed_d7']:.1f}/{r['speed_d14']:.1f} "
                  f"(cov {r['cov_speed_d1']*100:.0f}/{r['cov_speed_d7']*100:.0f}/{r['cov_speed_d14']*100:.0f}) | "
                  f"dir d1/d7/d14={r['dir_d1']:.0f}/{r['dir_d7']:.0f}/{r['dir_d14']:.0f} "
                  f"(cov {r['cov_dir_d1']*100:.0f}/{r['cov_dir_d7']*100:.0f}/{r['cov_dir_d14']*100:.0f})", flush=True)
    # best-per-dim (min Winkler) across the sweep = the achievable climatology floor
    dims = [f"speed_d{H}" for H in (1, 7, 14)] + [f"dir_d{H}" for H in (1, 7, 14)]
    floor = {d: min(s[d] for s in sweep) for d in dims}
    print("\n=== climatology FLOOR (best PI-scale per dim) ===")
    print("  " + " ".join(f"{d}={floor[d]:.1f}" for d in dims))
    print("kit-ref public-2021 speed d1/d7/d14=9.2/29.8/40.1 ; dir=173/312/334")
    outdir = a.out or os.environ.get("EXP_OUTPUT_DIR", ".")
    os.makedirs(outdir, exist_ok=True)
    dst = os.path.join(outdir, f"clim_cv_eval{a.eval_year}.json")
    json.dump({"train_years": ty, "eval_year": a.eval_year, "model": "climatology",
               "sweep": sweep, "floor": floor}, open(dst, "w"), indent=1)
    print(f"wrote {dst} ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
