#!/usr/bin/env python
"""Financial value of the day+1 forecast — day-ahead market simulation over a full year.

Feeds the +15 BONUS ("simulate selling your predicted production on the day-ahead
market across the evaluation year, using your forecast quantiles to place optimal
quantile bids, then add long-term profitability scenarios") and the dim-3 revenue /
demand / historical-benchmark criteria.

Everything below is measured, not assumed, except the constants named in CONSTANTS.

Design (fully out-of-sample):
  truth      AROME 2020 wind at the shipped farm centre -> PyWake farm power (the same
             faithful simulator that scores the siting task).
  forecast   the SHIPPED HRES quantile-MOS bundle trained on 2016-2018 and CQR-calibrated
             on 2019 -> d+1 speed q05/q50/q95 at the site, for every day of 2020.
             2020 is therefore unseen by both the model and its calibration.
  prices     REAL Dutch day-ahead prices for the SAME year (energy-charts / SMARD,
             CC BY 4.0) — wind and price share the year, so the wind-price covariance
             (cannibalisation) is real rather than imposed.
  settlement day-ahead bid + asymmetric imbalance, per Pinson et al. (2007); the
             expected-revenue-optimal bid is the tau* = pi_down/(pi_down+pi_up) quantile.

Strategies compared: perfect foresight · tau*-quantile (ours) · q50 (ours) ·
persistence (yesterday's production) · climatology (month x hour mean).

Outputs <EXP_OUTPUT_DIR>/market_value.json  +  market_value_hourly.parquet
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("PHASE2_DATA_ROOT", os.environ["SEAWINDS_PHASE2_DIR"])
ROOT = os.environ.get("SEAWINDS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KIT = f"{ROOT}/kit_phase2/phase_2"
for sub in ("", "/part0_dataset_setup", "/part1_forecast"):
    sys.path.insert(0, KIT + sub)
sys.path.insert(0, f"{ROOT}/scripts")

import joblib
import numpy as np
import pandas as pd

import forecast_hres as fh
import p2_siting as S
import wind_farm_simulator as wfs
import bidding as B

OUT = os.environ.get("EXP_OUTPUT_DIR", ".")
YEAR = int(os.environ.get("MV_YEAR", "2020"))
ZONE = os.environ.get("MV_ZONE", "NL")
MARKET = f"{ROOT}/scripts/data/market/market_hourly.parquet"
# train 2016-2018, CQR-calibrated on 2019 -> 2020 is out-of-sample for BOTH.
BUNDLE = f"{ROOT}/scripts/results/mos_cv_e2020_full_s0/mos_models.joblib"
SUBMISSION = f"{ROOT}/submission/siting/siting_submission.json"

# ---- CONSTANTS (assumptions; every one is swept or flagged in the report) --------
PI_UP = 90.0      # EUR/MWh paid when SHORT  (bid > actual) — kit default
PI_DOWN = 30.0    # EUR/MWh paid when LONG   (actual > bid) — kit default
TAU_STAR = B.optimal_bid_quantile(PI_UP, PI_DOWN)     # = 0.25


# ---------------------------------------------------------------- farm power
def farm_power(ws125, wd, x, y, turbine):
    """Free-stream 125 m speed + direction -> wake-corrected farm power (MW)."""
    ws_hub = np.asarray(ws125, float) * S.SHEAR
    t = pd.date_range("2020-01-01", periods=len(ws_hub), freq="h")
    wind = wfs.WindSeries(pd.DataFrame({"time": t, "ws": ws_hub,
                                        "wd": np.asarray(wd, float) % 360}))
    res = wfs.simulate(wfs.FarmLayout(x_m=np.asarray(x, float),
                                      y_m=np.asarray(y, float), turbine=turbine),
                       wind, annualisation=False)
    return np.asarray(res.farm_power_mw, float), float(res.rated_capacity_mw)


def to_hourly(times, values, index):
    """Linear-interpolate a coarse-cadence series onto the hourly market index."""
    s = pd.Series(np.asarray(values, float), index=pd.DatetimeIndex(times))
    s = s[~s.index.duplicated()].sort_index()
    return s.reindex(s.index.union(index)).interpolate("time").reindex(index)


def main() -> int:
    t0 = time.time()
    sub = json.load(open(SUBMISSION))
    lat0, lon0 = sub["farm_centre_lat"], sub["farm_centre_lon"]
    x, y = np.asarray(sub["layout_x_m"], float), np.asarray(sub["layout_y_m"], float)
    turb = S.turbine()

    # ---------------- 1. TRUTH: AROME wind at the site -> farm power ----------
    idx, ilat, ilon = S.nearest_idx(lat0, lon0)
    u, v, dates = S.load_year(YEAR)
    times_t, ws_t, wd_t = S.series_from_year(u, v, dates, idx)
    ws_t = ws_t / S.SHEAR                                    # back to 125 m free stream
    del u, v
    p_true, rated = farm_power(ws_t, wd_t, x, y, turb)
    print(f"truth: {len(p_true)} AROME steps, rated {rated:.0f} MW, "
          f"CF {p_true.mean()/rated:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- 2. FORECAST: shipped d+1 quantile MOS at the site -------
    b = joblib.load(BUNDLE)
    qmos, adj = b["qmos"], b["adj"]
    hres = fh._load_hres()
    cells = hres[["latitude", "longitude"]].drop_duplicates().to_numpy()
    clat, clon = cells[np.argmin((cells[:, 0] - lat0) ** 2 + (cells[:, 1] - lon0) ** 2)]
    site = hres[(hres.latitude == clat) & (hres.longitude == clon)].sort_values("time")
    site = site[(site.time >= f"{YEAR-1}-12-30") & (site.time < f"{YEAR+1}-01-01")]
    print(f"site coarse cell ({clat:.2f},{clon:.2f}) HRES issues={len(site)}", flush=True)

    recs = []
    for _, r in site.iterrows():                 # one HRES issue per row
        V = pd.Timestamp(r["time"]) + pd.Timedelta(days=1)          # d+1 valid day
        woy = V.isocalendar().week
        for H in (0, 6, 12, 18):
            sp, di = r[f"fcst_speed_d1_h{H}"], r[f"fcst_dir_d1_h{H}"]
            if not np.isfinite(sp) or not np.isfinite(di):
                continue
            fu, fv = fh._uv_from_speed_dir(sp, di)
            recs.append(dict(time=V + pd.Timedelta(hours=H), lead=1, hour=H,
                             lat=clat, lon=clon, fcst_u=float(fu), fcst_v=float(fv),
                             fcst_speed=float(sp),
                             woy_sin=np.sin(2 * np.pi * woy / 52.0),
                             woy_cos=np.cos(2 * np.pi * woy / 52.0),
                             wd=float(di) % 360))
    fc = pd.DataFrame(recs).sort_values("time").reset_index(drop=True)
    q = fh.predict_quantile_mos(qmos, fc, adjust=adj)
    fq = pd.DataFrame({"time": fc["time"], "wd": fc["wd"],
                       "q05": q["spd_q05"].to_numpy(), "q50": q["spd_q50"].to_numpy(),
                       "q95": q["spd_q95"].to_numpy()})
    fq = fq[(fq.time >= f"{YEAR}-01-01") & (fq.time < f"{YEAR+1}-01-01")]
    print(f"forecast steps={len(fq)}  mean q50 {fq.q50.mean():.2f} m/s", flush=True)

    pq = {}
    for c in ("q05", "q50", "q95"):
        pq[c], _ = farm_power(fq[c].to_numpy(), fq["wd"].to_numpy(), x, y, turb)
    print(f"forecast power built ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- 3. hourly alignment with the real market ----------------
    mk = pd.read_parquet(MARKET)
    mk = mk[mk.zone == ZONE].copy()
    mk["time"] = pd.to_datetime(mk["time"], utc=True).dt.tz_localize(None)
    mk = mk[(mk.time >= f"{YEAR}-01-01") & (mk.time < f"{YEAR+1}-01-01")]
    H = pd.DatetimeIndex(mk["time"])

    act = to_hourly(times_t, p_true, H)
    f05 = to_hourly(fq["time"], pq["q05"], H)
    f50 = to_hourly(fq["time"], pq["q50"], H)
    f95 = to_hourly(fq["time"], pq["q95"], H)
    df = pd.DataFrame({"time": H, "price": mk["price_eur_mwh"].to_numpy(),
                       "load_mw": mk["load_mw"].to_numpy(),
                       "woff_mw": mk["wind_offshore_mw"].to_numpy(),
                       "won_mw": mk["wind_onshore_mw"].to_numpy(),
                       "actual": act.to_numpy(), "q05": f05.to_numpy(),
                       "q50": f50.to_numpy(), "q95": f95.to_numpy()}).dropna(
                           subset=["price", "actual", "q50"])
    print(f"aligned hours = {len(df)}", flush=True)

    # ---------------- 4. bidding strategies -----------------------------------
    a = df["actual"].to_numpy()
    p = df["price"].to_numpy()

    def settle(bid, pi_up, pi_dn, convention="pinson"):
        """Realised revenue (EUR) under day-ahead bid + imbalance settlement.

        pinson  (Pinson et al. 2007, eq. 4): the deviation is bought back / sold at
                the day-ahead price plus an asymmetric SPREAD, so
                    R = p*actual - pi_up*max(bid-act,0) - pi_dn*max(act-bid,0)
                and the expected-revenue-optimal bid is the tau* = pi_dn/(pi_up+pi_dn)
                quantile — independent of the price level.
        kit     the shipped kit formula pays only for the BID and additionally
                penalises the deviation (R = p*bid - ...). Its optimum is the
                (E[p]+pi_dn)/(pi_up+pi_dn) quantile, NOT the tau* its own helper
                returns; we report it only as a sensitivity.
        """
        short = np.maximum(bid - a, 0.0)
        long_ = np.maximum(a - bid, 0.0)
        base = p * a if convention == "pinson" else p * bid
        pen = pi_up * short + pi_dn * long_
        return float((base - pen).sum()), float(pen.sum()), float(short.sum()), float(long_.sum())

    strat = {
        "perfect_foresight": a.copy(),
        "quantile_tau_star": B.interpolate_quantile_bid(df.q05.to_numpy(), df.q50.to_numpy(),
                                                        df.q95.to_numpy(), TAU_STAR),
        "median_q50": df.q50.to_numpy(),
        "persistence_24h": pd.Series(a).shift(24).bfill().to_numpy(),
    }
    clim = pd.DataFrame({"m": df.time.dt.month, "h": df.time.dt.hour, "a": a})
    strat["climatology"] = clim.groupby(["m", "h"])["a"].transform("mean").to_numpy()

    # Imbalance-price scenarios. FLAT is the kit default; PROPORTIONAL scales the
    # spread with the realised price level (2020 averaged 32 EUR/MWh, so a flat
    # 90/30 EUR/MWh penalty is 3x the energy value and dominates unrealistically).
    SCEN = {"flat_90_30": (PI_UP, PI_DOWN),
            "prop_50pct_30pct": (0.5 * p.mean(), 0.3 * p.mean()),
            "flat_60_30": (60.0, 30.0), "flat_120_30": (120.0, 30.0),
            "symmetric_60_60": (60.0, 60.0)}

    res, sens = {}, {}
    for sname, (up, dn) in SCEN.items():
        tstar = B.optimal_bid_quantile(up, dn)
        bid_t = B.interpolate_quantile_bid(df.q05.to_numpy(), df.q50.to_numpy(),
                                           df.q95.to_numpy(), tstar)
        block = {}
        for k, bid in {**strat, "quantile_tau_star": bid_t}.items():
            rev, pen, sh, lo = settle(bid, up, dn)
            block[k] = dict(revenue_eur=rev, imbalance_eur=pen,
                            short_mwh=sh, long_mwh=lo,
                            eur_per_mwh=rev / (a.sum() or 1))
        block["_tau_star"] = tstar
        sens[sname] = block
        if sname == "flat_90_30":
            res = block

    # headline scenario = the proportional spread (defensible at any price level)
    head = sens["prop_50pct_30pct"]
    print(f"\n  settlement = Pinson (R = p*actual - spreads); "
          f"tau* = {head['_tau_star']:.2f}", flush=True)
    for k in ("perfect_foresight", "quantile_tau_star", "median_q50",
              "persistence_24h", "climatology"):
        b = head[k]
        print(f"  {k:20s} {b['revenue_eur']/1e6:8.2f} M EUR   imbalance "
              f"{b['imbalance_eur']/1e6:7.2f}   {b['eur_per_mwh']:6.2f} EUR/MWh", flush=True)

    # tau sweep under the headline scenario (does tau* really optimise realised revenue?)
    up_h, dn_h = 0.5 * p.mean(), 0.3 * p.mean()
    sweep = {}
    for tau in np.round(np.arange(0.05, 0.96, 0.05), 2):
        bid = B.interpolate_quantile_bid(df.q05.to_numpy(), df.q50.to_numpy(),
                                         df.q95.to_numpy(), tau)
        sweep[float(tau)] = settle(bid, up_h, dn_h)[0]
    tau_best = max(sweep, key=sweep.get)

    # the kit's own settlement convention, for transparency
    kit_sweep = {}
    for tau in np.round(np.arange(0.05, 0.96, 0.05), 2):
        bid = B.interpolate_quantile_bid(df.q05.to_numpy(), df.q50.to_numpy(),
                                         df.q95.to_numpy(), tau)
        kit_sweep[float(tau)] = settle(bid, PI_UP, PI_DOWN, convention="kit")[0]

    # ---------------- 5. capture rate / demand / historical benchmark ---------
    energy = a.sum() / 1e3                                    # MWh -> GWh (hourly steps)
    cap_price = float((a * p).sum() / a.sum())
    base_price = float(p.mean())
    cap_rate = cap_price / base_price
    # the SAME statistic for the real Dutch offshore-wind fleet in the same year
    w = df["woff_mw"].to_numpy()
    fleet_cap = float((w * p).sum() / w.sum()) if np.nansum(w) > 0 else float("nan")
    demand_twh = float(df["load_mw"].sum() / 1e6)
    # multi-year historical fleet capture rate (validation of the mechanism)
    hist = {}
    mk_all = pd.read_parquet(MARKET)
    mk_all = mk_all[mk_all.zone == ZONE].dropna(subset=["price_eur_mwh"])
    mk_all["year"] = pd.to_datetime(mk_all["time"], utc=True).dt.year
    for yy, g in mk_all.groupby("year"):
        ww = g["wind_offshore_mw"].to_numpy(float)
        pp = g["price_eur_mwh"].to_numpy(float)
        m = np.isfinite(ww) & np.isfinite(pp)
        if m.sum() < 2000 or np.nansum(ww[m]) <= 0:
            continue
        hist[int(yy)] = dict(base_price=float(pp[m].mean()),
                             offshore_capture=float((ww[m] * pp[m]).sum() / ww[m].sum()),
                             capture_rate=float((ww[m] * pp[m]).sum() / ww[m].sum()
                                                / pp[m].mean()),
                             mean_offshore_mw=float(ww[m].mean()),
                             negative_price_hours=int((pp[m] < 0).sum()))

    payload = dict(
        year=YEAR, zone=ZONE, site=dict(lat=lat0, lon=lon0),
        constants=dict(pi_up=PI_UP, pi_down=PI_DOWN, tau_star=TAU_STAR,
                       bundle=BUNDLE, price_source="energy-charts.info (SMARD/BNetzA), CC BY 4.0"),
        production=dict(rated_mw=rated, energy_gwh=energy,
                        capacity_factor=float(a.mean() / rated), hours=int(len(df))),
        market=dict(base_price=base_price, captured_price=cap_price, capture_rate=cap_rate,
                    fleet_offshore_captured_price=fleet_cap,
                    fleet_capture_rate=fleet_cap / base_price if base_price else None,
                    demand_twh=demand_twh, share_of_demand=energy / 1e3 / demand_twh,
                    negative_price_hours=int((p < 0).sum())),
        strategies=res, headline_scenario="prop_50pct_30pct", headline=head,
        tau_sweep=sweep, tau_best_realised=tau_best, kit_convention_sweep=kit_sweep,
        imbalance_sensitivity=sens, historical_capture=hist,
    )
    with open(os.path.join(OUT, "market_value.json"), "w") as f:
        json.dump(payload, f, indent=2)
    df.to_parquet(os.path.join(OUT, "market_value_hourly.parquet"), index=False)

    q_rev = head["quantile_tau_star"]["revenue_eur"]
    print(f"\nforecast VALUE vs persistence  = {(q_rev-head['persistence_24h']['revenue_eur'])/1e6:+.2f} M EUR/yr")
    print(f"forecast VALUE vs climatology  = {(q_rev-head['climatology']['revenue_eur'])/1e6:+.2f} M EUR/yr")
    print(f"gap to perfect foresight       = {(head['perfect_foresight']['revenue_eur']-q_rev)/1e6:.2f} M EUR/yr")
    print(f"quantile bid vs median bid     = {(q_rev-head['median_q50']['revenue_eur'])/1e6:+.2f} M EUR/yr")
    print(f"captured price {cap_price:.2f} vs base {base_price:.2f} EUR/MWh "
          f"(capture rate {cap_rate:.3f}); NL offshore fleet {fleet_cap:.2f}")
    print(f"energy {energy:.0f} GWh = {payload['market']['share_of_demand']*100:.2f}% of {ZONE} demand")
    print(f"tau* = {TAU_STAR:.2f}; realised-best tau = {tau_best}")
    print(f"[{time.time()-t0:.0f}s] wrote {OUT}/market_value.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
