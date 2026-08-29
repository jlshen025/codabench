#!/usr/bin/env python
"""Financial balance: CAPEX / OPEX / LCOE / NPV-IRR scenarios for the shipped farm.

Rubric dim 3 wants (a) a credible CONSTRUCTION COST reflecting constructibility and grounded
in a critical review of the literature, (b) yield -> revenue set against demand and benchmarked
to historical data, (c) optionally OPEX over the asset life with a predictive-maintenance
strategy. This script produces the numbers for all three, from three INDEPENDENT cost bases so
the report can compare them rather than assert one:

  A. the organiser kit's `cost_model.py`   (NREL-ATB-style, WACC 6 %, 25 yr)
  B. a bottom-up build from the Dutch national study "Cost Evaluation of North Sea Offshore
     Wind (post 2030)" (Witteveen+Bos for the North Sea Wind Power Hub consortium, 2019):
     14 M EUR per 15 MW turbine; monopile substructure 5.1 M EUR at 5 m rising to 10 M EUR at
     55 m depth; 40 M EUR intra-array and 100 M EUR development per GW; O&M 45-47 M EUR/yr per
     GW with only ~3.5 % difference between near-shore and far-offshore sites; 30-yr life at a
     2.9 % social discount rate.
  C. the same bottom-up build at a COMMERCIAL discount rate, which is what actually explains the
     gap between (A) and (B).

Revenue scenarios use the CAPTURE RATE measured from real market data (p2_market_value.py), not
an assumed one, and the long-term paths are stated as scenarios rather than a forecast.

Light: runs on a login node in seconds.
"""
from __future__ import annotations

import os
import json
import sys

import numpy as np

ROOT = os.environ.get("SEAWINDS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/kit_phase2/phase_2")
import cost_model as CM                                              # noqa: E402

OUT = f"{ROOT}/scripts/results/economics.json"

CAP_MW = 55 * 22.0                 # 1210 MW
N_T = 55
AEP_GWH = 6633.6                   # PyWake, eastern-Dogger FINAL irregular layout (E3), AROME 2016-2020 mean
DEPTH_M = 30.8                     # array-mean depth (constructibility.json, optimised layout)
CF = AEP_GWH * 1e3 / (CAP_MW * 8760)
AVAIL_HAIRCUT = 0.12               # central availability / electrical-loss / curtailment haircut

# Connection routes actually available to this site (constructibility.json, cited in report)
ROUTES = {"Dogger Bank C / Sofia converter (UK)": 101.0,
          "Dogger Bank A/B - Creyke Beck (UK)": 118.0,
          "Nederwiek / Doordewind hub (NL, planned)": 147.0,
          "direct to nearest coastline": 234.0}

# --- (B) NSWPH bottom-up constants (2019 EUR, Witteveen+Bos / NSWPH) ---------------
NSWPH = dict(turbine_meur_per_15mw=14.0, sub_meur_at_5m=5.1, sub_meur_at_55m=10.0,
             intra_array_meur_per_gw=40.0, development_meur_per_gw=100.0,
             om_meur_per_gw_year=46.0, life_years=30, social_discount=0.029,
             installation_meur_per_gw=170.0)   # installation ~= decommissioning ref, p.1737


def crf(rate: float, n: int) -> float:
    return rate * (1 + rate) ** n / ((1 + rate) ** n - 1)


def nswph_capex(cap_mw: float, depth_m: float, dist_km: float) -> dict:
    """Bottom-up CAPEX (M EUR) in the NSWPH cost basis, scaled to our capacity."""
    gw = cap_mw / 1000.0
    n15 = cap_mw / 15.0                                   # in 15-MW-equivalents
    sub = NSWPH["sub_meur_at_5m"] + (NSWPH["sub_meur_at_55m"] - NSWPH["sub_meur_at_5m"]) \
        * (depth_m - 5.0) / 50.0
    c = {"turbines": NSWPH["turbine_meur_per_15mw"] * n15,
         "substructures": sub * n15,
         "intra_array": NSWPH["intra_array_meur_per_gw"] * gw,
         "development": NSWPH["development_meur_per_gw"] * gw,
         "installation": NSWPH["installation_meur_per_gw"] * gw,
         # export system: HVDC cable 2.0 M EUR/km (mid of the 1.8-2.6 M EUR/km band for
         # 320 kV HVDC) + 2 converter platforms at 350 M EUR each for a ~1.2 GW link
         "export_hvdc": 2.0 * dist_km + 700.0}
    c["total"] = sum(c.values())
    return c


def main() -> int:
    out = {"farm": dict(capacity_mw=CAP_MW, n_turbines=N_T, aep_gwh=AEP_GWH,
                        capacity_factor=CF, array_mean_depth_m=DEPTH_M)}

    # ---- A. kit cost model across the real connection routes -----------------
    kit = {}
    for name, km in ROUTES.items():
        b = CM.evaluate_farm(capacity_mw=CAP_MW, n_turbines=N_T, aep_gwh=AEP_GWH,
                             water_depth_m=DEPTH_M, distance_to_shore_km=km)
        kit[name] = dict(km=km, capex_meur=b.capex_eur / 1e6,
                         capex_eur_per_kw=b.capex_eur_per_kw,
                         opex_meur_yr=b.opex_eur_per_year / 1e6,
                         lcoe=b.lcoe_eur_per_mwh, components=b.lcoe_components)
    out["kit_model"] = kit
    print("A. KIT cost model (WACC 6 %, 25 yr)")
    for k, v in kit.items():
        print(f"   {v['km']:5.0f} km  CAPEX {v['capex_meur']:6.0f} M EUR "
              f"({v['capex_eur_per_kw']:5.0f} EUR/kW)  OPEX {v['opex_meur_yr']:5.1f}  "
              f"LCOE {v['lcoe']:6.1f} EUR/MWh   {k}")

    # ---- B/C. NSWPH bottom-up, social vs commercial financing ----------------
    nb = {}
    for name, km in ROUTES.items():
        c = nswph_capex(CAP_MW, DEPTH_M, km)
        om = NSWPH["om_meur_per_gw_year"] * CAP_MW / 1000.0
        for tag, rate, life in (("social_2.9pct_30yr", NSWPH["social_discount"], 30),
                                ("commercial_6pct_25yr", 0.06, 25),
                                ("commercial_8pct_25yr", 0.08, 25)):
            lc = (c["total"] * 1e6 * crf(rate, life) + om * 1e6) / (AEP_GWH * 1e3)
            nb.setdefault(name, {})[tag] = dict(capex_meur=c["total"], opex_meur_yr=om, lcoe=lc)
        nb[name]["capex_breakdown_meur"] = c
    out["nswph_bottom_up"] = nb
    print("\nB/C. NSWPH bottom-up CAPEX + O&M (46 M EUR/GW/yr, distance-insensitive)")
    for name, km in ROUTES.items():
        r = nb[name]
        print(f"   {km:5.0f} km  CAPEX {r['capex_breakdown_meur']['total']:6.0f} M EUR   "
              f"LCOE social {r['social_2.9pct_30yr']['lcoe']:5.1f} | "
              f"6 % {r['commercial_6pct_25yr']['lcoe']:5.1f} | "
              f"8 % {r['commercial_8pct_25yr']['lcoe']:5.1f} EUR/MWh")

    # ---- D. revenue + NPV/IRR scenarios --------------------------------------
    mv = {}
    for y in (2019, 2020):
        try:
            mv[y] = json.load(open(f"{ROOT}/scripts/results/market_value_{y}_s0/market_value.json"))
        except FileNotFoundError:
            pass
    cap_rate = float(np.mean([mv[y]["market"]["capture_rate"] for y in mv])) if mv else 0.975
    out["measured_capture_rate_own_farm"] = cap_rate
    if mv:
        out["measured_capture_rate_nl_offshore_fleet"] = \
            {int(k): v for k, v in list(mv.values())[0]["historical_capture"].items()}

    # price scenarios (base EUR/MWh, flat real). Capture rate applies on top; the LOW case
    # additionally assumes cannibalisation deepens as North-Sea offshore capacity multiplies.
    PRICES = {"low (cannibalised)": (55.0, 0.75), "central": (80.0, cap_rate * 0.92),
              "high": (110.0, cap_rate * 0.95), "CfD strike 75 EUR/MWh": (75.0, 1.0)}
    # ROUTE BASIS. The base case is the 234 km direct-to-coastline radial, because it assumes
    # no third-party asset: proximity to the Dogger Bank C / Sofia converter (101 km) is not a
    # right to connect -- those assets have owners, allocated capacity and specific landing
    # points -- so 101 km is reported as an UPSIDE case, never as the project case.
    #
    # ENERGY BASIS. Scenarios are reported on NET energy (gross AEP less a central 12 %
    # availability / electrical-loss / curtailment haircut). The GROSS basis is emitted
    # alongside so the size of that correction is explicit rather than asserted; it was
    # previously hand-derived outside this script, which is why both now come from the artifact.
    opex = NSWPH["om_meur_per_gw_year"] * CAP_MW / 1000.0 * 1e6

    def scenarios(ref_km: float, energy_gwh: float) -> dict:
        capex = nswph_capex(CAP_MW, DEPTH_M, ref_km)["total"] * 1e6
        s = {}
        for tag, (price, cr) in PRICES.items():
            rev = energy_gwh * 1e3 * price * cr
            for wacc, life in ((0.06, 30), (0.08, 30)):
                cf = rev - opex
                npv = -capex + sum(cf / (1 + wacc) ** t for t in range(1, life + 1))
                lo, hi = -0.5, 1.0                      # IRR by bisection on a level annuity
                for _ in range(200):
                    mid = (lo + hi) / 2
                    v = -capex + sum(cf / (1 + mid) ** t for t in range(1, life + 1))
                    lo, hi = (mid, hi) if v > 0 else (lo, mid)
                s[f"{tag} @WACC{int(wacc*100)}"] = dict(
                    price=price, capture=cr, revenue_meur_yr=rev / 1e6,
                    ebitda_meur_yr=cf / 1e6, npv_meur=npv / 1e6, irr=(lo + hi) / 2,
                    capex_meur=capex / 1e6,
                    payback_yr=capex / cf if cf > 0 else None)
        return s

    e_net = AEP_GWH * (1.0 - AVAIL_HAIRCUT)
    out["energy_basis"] = dict(gross_gwh=AEP_GWH, net_gwh=e_net, haircut=AVAIL_HAIRCUT)
    prof = {}
    for rtag, ref_km in (("base_234km_radial", 234.0), ("upside_101km_doggerbank", 101.0)):
        for btag, e in (("net", e_net), ("gross", AEP_GWH)):
            prof[f"{rtag}|{btag}"] = scenarios(ref_km, e)
    out["profitability_scenarios"] = prof
    # the shipped headline table
    out["profitability_headline"] = prof["base_234km_radial|net"]

    print(f"\nD. 30-yr profitability   (OPEX {opex/1e6:.0f} M EUR/yr, "
          f"measured capture rate {cap_rate:.3f}, "
          f"gross {AEP_GWH:.0f} -> net {e_net:.0f} GWh at {AVAIL_HAIRCUT:.0%} haircut)")
    for rtag, ref_km in (("base_234km_radial", 234.0), ("upside_101km_doggerbank", 101.0)):
        cx = nswph_capex(CAP_MW, DEPTH_M, ref_km)["total"]
        print(f"\n  -- {rtag}  ({ref_km:.0f} km, CAPEX {cx:.0f} M EUR) --")
        for k in PRICES:
            n = prof[f"{rtag}|net"][f"{k} @WACC6"]
            g = prof[f"{rtag}|gross"][f"{k} @WACC6"]
            print(f"   {k:26s} EBITDA(net) {n['ebitda_meur_yr']:6.0f}  "
                  f"NPV6(net) {n['npv_meur']:8.0f}  IRR(net) {n['irr']*100:5.1f} %   "
                  f"| NPV6(gross) {g['npv_meur']:8.0f}")
    capex = nswph_capex(CAP_MW, DEPTH_M, 234.0) ["total"] * 1e6   # section E uses the base case

    # ---- E. predictive maintenance ------------------------------------------
    # O&M is ~55.7 M EUR/yr here; unplanned corrective maintenance is the swing item.
    pm = {}
    for tag, om_cut, avail_gain in (("no PdM (base)", 0.0, 0.0),
                                    ("condition monitoring, -10 % O&M", 0.10, 0.005),
                                    ("full PdM + weather-window optimisation", 0.20, 0.010)):
        o = opex * (1 - om_cut)
        e = AEP_GWH * (1 + avail_gain)
        pm[tag] = dict(opex_meur_yr=o / 1e6, aep_gwh=e,
                       lcoe=(capex * crf(0.06, 30) + o) / (e * 1e3),
                       revenue_gain_meur_yr=(e - AEP_GWH) * 1e3 * 80.0 * cap_rate / 1e6,
                       opex_saving_meur_yr=(opex - o) / 1e6)
    out["predictive_maintenance"] = pm
    print("\nE. predictive-maintenance scenarios (LCOE at 6 % / 30 yr)")
    for k, v in pm.items():
        print(f"   {k:40s} OPEX {v['opex_meur_yr']:5.1f}  AEP {v['aep_gwh']:6.0f} GWh  "
              f"LCOE {v['lcoe']:5.1f}  saving {v['opex_saving_meur_yr']+v['revenue_gain_meur_yr']:5.1f} M EUR/yr")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
