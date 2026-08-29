#!/usr/bin/env python
"""Fetch + cache North-Sea market history (day-ahead price, load, wind generation).

LOGIN-NODE ONLY (compute nodes have no internet). Source: api.energy-charts.info
(Bundesnetzagentur | SMARD.de, CC BY 4.0) — no token required.

Pulls, per bidding zone and year:
  * hourly day-ahead price (/price?bzn=...)
  * hourly public power (/public_power?country=...) -> Load, Wind offshore,
    Wind onshore, Solar, Residual load

Writes one tidy parquet per (zone, year) under scripts/data/market/ plus a merged
`market_hourly.parquet`. Re-running skips years already cached.

Used by:
  * the +15 BONUS — day-ahead revenue simulation with quantile bidding
  * dim-3 Financial — yield -> revenue vs DEMAND, benchmarked to historical data
    (offshore-wind capture rate / price cannibalisation, computed not assumed)
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "market"
OUT.mkdir(parents=True, exist_ok=True)

API = "https://api.energy-charts.info"
# NL = closest EPEX zone to the Dogger-Bank site (kit default); DE-LU = deepest
# offshore-wind market, used as a cross-zone robustness check.
ZONES = {"NL": "nl", "DE-LU": "de"}
YEARS = list(range(2019, 2026))
WANT = ["Load", "Residual load", "Wind offshore", "Wind onshore", "Solar"]


def _get(url: str, tries: int = 4) -> dict:
    last = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # transient API/network hiccup
            last = e
            time.sleep(3 * (k + 1))
    raise RuntimeError(f"GET failed {url}: {last}")


def fetch_price(bzn: str, year: int) -> pd.DataFrame:
    d = _get(f"{API}/price?bzn={bzn}&start={year}-01-01&end={year+1}-01-01")
    return pd.DataFrame(
        {"time": pd.to_datetime(d["unix_seconds"], unit="s", utc=True),
         "price_eur_mwh": d["price"]}
    )


def fetch_power(country: str, year: int) -> pd.DataFrame:
    d = _get(f"{API}/public_power?country={country}&start={year}-01-01&end={year+1}-01-01")
    out = {"time": pd.to_datetime(d["unix_seconds"], unit="s", utc=True)}
    for p in d["production_types"]:
        if p["name"] in WANT:
            out[p["name"].lower().replace(" ", "_") + "_mw"] = p["data"]
    return pd.DataFrame(out)


def main() -> int:
    frames = []
    for zone, country in ZONES.items():
        for year in YEARS:
            f = OUT / f"{zone}_{year}.parquet"
            if f.exists():
                frames.append(pd.read_parquet(f))
                print(f"cached  {zone} {year}", flush=True)
                continue
            try:
                pr = fetch_price(zone, year)
                pw = fetch_power(country, year)
                # public_power is 15-min in DE / hourly in NL -> hourly mean, then
                # join on the hourly price stamp.
                pw = pw.set_index("time").resample("1h").mean().reset_index()
                df = pr.merge(pw, on="time", how="left")
                df.insert(0, "zone", zone)
                df.to_parquet(f, index=False)
                frames.append(df)
                print(f"fetched {zone} {year}: {len(df)} h, "
                      f"price mean {df.price_eur_mwh.mean():.2f} EUR/MWh", flush=True)
            except Exception as e:
                print(f"SKIP {zone} {year}: {e}", flush=True)
            time.sleep(1)
    if not frames:
        print("no data")
        return 1
    all_df = pd.concat(frames, ignore_index=True).sort_values(["zone", "time"])
    all_df.to_parquet(OUT / "market_hourly.parquet", index=False)
    print(f"\nwrote {OUT/'market_hourly.parquet'}  rows={len(all_df)}")
    print(all_df.groupby("zone").agg(
        years=("time", lambda s: f"{s.dt.year.min()}-{s.dt.year.max()}"),
        n=("price_eur_mwh", "size"),
        price_mean=("price_eur_mwh", "mean"),
        load_mean=("load_mw", "mean"),
        woff_mean=("wind_offshore_mw", "mean"),
    ).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
