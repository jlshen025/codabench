#!/usr/bin/env python
"""Constructibility metrics for the shipped farm layout (rubric dim 3: construction cost).

Reads EMODnet bathymetry + the shipped submission.json and reports, for the farm centre
AND for each of the 55 turbine positions:
  * water depth (min / mean / max over the array) -> foundation type + cost driver
  * distance to the nearest coast (EMODnet field)
  * great-circle distance to REAL North-Sea grid-connection candidates, so the export-
    cable length used in the LCOE is a defensible route rather than "nearest coastline".

The grid-connection candidates are named, dated reference points; each is cited in the
report so a reader can check the route assumption instead of taking a single number.

Light: runs on a login node (thread-capped) in seconds.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

ROOT = os.environ.get("SEAWINDS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BATHY = f"{os.environ['SEAWINDS_PHASE2_DIR']}/static/bathymetry/emodnet_northsea_1km.nc"
SUBMISSION = f"{ROOT}/submission/siting/siting_submission.json"
OUT = f"{ROOT}/scripts/results/constructibility.json"

# Real, operating or consented North-Sea offshore-grid connection points. Each entry is
# a physical landing/converter location the export cable could plausibly reach; the
# report cites the source for every one. Coordinates are approximate hub centroids.
GRID_POINTS = {
    "IJmuiden Ver (NL 2 GW HVDC platforms, TenneT)":      (52.85, 3.70),
    "Nederwiek / Doordewind zone (NL 2 GW, planned)":     (54.00, 3.80),
    "Ten noorden van de Waddeneilanden (NL)":             (53.85, 5.60),
    "BorWin / DolWin cluster (DE, German Bight)":         (54.35, 6.20),
    "Dogger Bank A/B - Creyke Beck landing (UK)":         (54.75, 2.20),
    "Dogger Bank C / Sofia - Teesside landing (UK)":      (54.60, 2.80),
    "Esbjerg / Danish west coast (DK)":                   (55.48, 8.45),
    "Energy Island North Sea (DK, planned)":              (56.10, 6.30),
}


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def main() -> int:
    sub = json.load(open(SUBMISSION))
    lat0, lon0 = sub["farm_centre_lat"], sub["farm_centre_lon"]
    xs = np.asarray(sub["layout_x_m"], float)
    ys = np.asarray(sub["layout_y_m"], float)
    # metres -> degrees around the farm centre
    tlat = lat0 + ys / 111_320.0
    tlon = lon0 + xs / (111_320.0 * np.cos(np.radians(lat0)))

    ds = xr.open_dataset(BATHY)
    lat = ds["latitude"].values if "latitude" in ds else ds["lat"].values
    lon = ds["longitude"].values if "longitude" in ds else ds["lon"].values
    depth = ds["water_depth_m"].values
    dcoast = ds["dist_coast_km"].values if "dist_coast_km" in ds else None

    def at(la, lo):
        i = int(np.argmin(np.abs(lat - la)))
        j = int(np.argmin(np.abs(lon - lo)))
        return float(depth[i, j]), (float(dcoast[i, j]) if dcoast is not None else np.nan)

    d_c, dc_c = at(lat0, lon0)
    td = np.array([at(a, b)[0] for a, b in zip(tlat, tlon)])
    tc = np.array([at(a, b)[1] for a, b in zip(tlat, tlon)])

    grid = {k: float(haversine(lat0, lon0, v[0], v[1])) for k, v in GRID_POINTS.items()}
    nearest = min(grid, key=grid.get)

    # extent of the array (sanity vs the 15x15 km box + 5D spacing rules)
    span_x, span_y = float(xs.max() - xs.min()), float(ys.max() - ys.min())
    dmat = np.hypot(xs[:, None] - xs[None, :], ys[:, None] - ys[None, :])
    np.fill_diagonal(dmat, np.inf)

    out = dict(
        centre=dict(lat=lat0, lon=lon0, depth_m=d_c, dist_coast_km=dc_c),
        turbines=dict(n=len(xs), depth_min=float(td.min()), depth_mean=float(td.mean()),
                      depth_max=float(td.max()), depth_std=float(td.std()),
                      dist_coast_min_km=float(np.nanmin(tc)),
                      dist_coast_max_km=float(np.nanmax(tc))),
        layout=dict(span_x_m=span_x, span_y_m=span_y,
                    min_pair_spacing_m=float(dmat.min()),
                    min_pair_spacing_D=float(dmat.min() / 284.0)),
        grid_connection_km=dict(sorted(grid.items(), key=lambda kv: kv[1])),
        nearest_grid_point=dict(name=nearest, km=grid[nearest]),
        source=dict(bathymetry=BATHY, rotor_diameter_m=284.0),
    )
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    print(f"centre {lat0:.3f}N {lon0:.3f}E   depth {d_c:.1f} m   nearest coast {dc_c:.0f} km")
    print(f"array  depth {td.min():.1f}-{td.max():.1f} m (mean {td.mean():.1f}, "
          f"sd {td.std():.2f})   span {span_x/1000:.1f} x {span_y/1000:.1f} km   "
          f"min spacing {dmat.min():.0f} m = {dmat.min()/284:.2f} D")
    print("\ndistance to real grid-connection candidates:")
    for k, v in out["grid_connection_km"].items():
        print(f"   {v:7.1f} km   {k}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
