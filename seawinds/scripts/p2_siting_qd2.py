#!/usr/bin/env python
"""Layout optimisation v2 — fixes the two flaws that crippled `p2_siting_qd.py`.

Run 1 (siting_qd_s0) promoted a layout worth +0.209 pp mean CF, but its diagnostics
showed the search itself had failed:

  * **the mutation operator was wrong.** It jittered ALL 55 turbines at once by
    N(0, 0.8 D). With a 5 D minimum spacing inside a 15 km box the array is nearly
    close-packed, so almost every such move violated the spacing somewhere:
    only ~300 of 9,000 mutations produced a valid layout (3 %), and the archive
    held 3 cells out of a target 200.
  * **the surrogate did not rank.** The binned wind-rose proxy scored the eventual
    winner as NO BETTER than the incumbent (+0.000 %), while the full replay scored
    it +0.209 pp — a rank inversion on the only pair that mattered. A proxy that
    inverts the decisive pair is not usable (the local-proxy rule).

v2 therefore:
  1. **searches on the TRUE objective** — mean capacity factor over the full
     fine-AROME 2016-2020 replay through the validated scorer (~1 s/evaluation,
     affordable) — so no proxy fidelity question remains;
  2. **moves 1-3 turbines at a time with per-turbine repair**, which is the standard
     operator for a tightly-packed layout and lifts the valid-move rate by an order
     of magnitude;
  3. **seeds from many valid grids** (spacing x rotation sweep) plus the incumbent;
  4. keeps the MAP-Elites archive over (mean nearest-neighbour spacing, elongation)
     so the result shows whether the grid is a local or a global optimum;
  5. scores worst-year CF alongside the mean and promotes only on BOTH, then
     re-checks the organiser's `validate_layout()`.

CV-only / offline. Costs no submission slot.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.environ.get("SEAWINDS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/kit_phase2/phase_2")

import p2_siting as S
import wind_farm_simulator as wfs

OUT = os.environ.get("EXP_OUTPUT_DIR", ".")
SUB = f"{ROOT}/submission/siting/siting_submission.json"
N_T, BOX, DIAM = 55, 15000.0, 284.0
HALF, MIN_S = BOX / 2, 5.0 * DIAM
YEARS = [2016, 2017, 2018, 2019, 2020]
N_BINS = 12
N_EVAL = int(os.environ.get("QD_EVALS", "2500"))
SEED = int(os.environ.get("QD_SEED", "0"))


def pair_ok(x, y):
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(d, np.inf)
    return d.min() >= MIN_S


def valid(x, y):
    return np.abs(x).max() <= HALF and np.abs(y).max() <= HALF and pair_ok(x, y)


def descriptors(x, y):
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1).mean() / DIAM
    w = np.sort(np.linalg.eigvalsh(np.cov(np.vstack([x, y]))))
    return nn, float(min(np.sqrt(w[1] / max(w[0], 1e-9)), 3.99))


def mutate(x, y, rng, sigma):
    """Move 1-3 turbines; repair per-turbine by resampling that turbine only."""
    x, y = x.copy(), y.copy()
    for i in rng.choice(N_T, size=int(rng.integers(1, 4)), replace=False):
        ox, oy = x[i], y[i]
        for _ in range(8):                       # a few tries for THIS turbine
            nx = np.clip(ox + rng.normal(0, sigma), -HALF, HALF)
            ny = np.clip(oy + rng.normal(0, sigma), -HALF, HALF)
            d = np.hypot(np.delete(x, i) - nx, np.delete(y, i) - ny)
            if d.min() >= MIN_S:
                x[i], y[i] = nx, ny
                break
    return x, y


def main() -> int:
    t0 = time.time()
    sub = json.load(open(SUB))
    lat0, lon0 = sub["farm_centre_lat"], sub["farm_centre_lon"]
    x0 = np.asarray(sub["layout_x_m"], float)
    y0 = np.asarray(sub["layout_y_m"], float)

    idx, alat, alon = S.nearest_idx(lat0, lon0)
    series = S.centre_series_allyears(idx, YEARS)
    print(f"site ({alat:.2f},{alon:.2f})  steps/yr={len(series[2020][1])} "
          f"({time.time()-t0:.0f}s)", flush=True)

    def score(x, y):
        cfs = np.array([S.simulate_cf(*series[yy], (x, y))[0] for yy in YEARS])
        return float(cfs.mean()), float(cfs.min()), cfs

    inc_mean, inc_min, inc_cfs = score(x0, y0)
    print(f"incumbent cf_mean {inc_mean*100:.3f}  worst {inc_min*100:.3f}", flush=True)

    rng = np.random.default_rng(SEED)
    b1 = np.linspace(5.0, 16.0, N_BINS + 1)
    b2 = np.linspace(1.0, 4.0, N_BINS + 1)
    cells: dict = {}
    n_eval = [0]
    n_try = [0]

    def cell_of(d1, d2):
        return (int(np.clip(np.searchsorted(b1, d1) - 1, 0, N_BINS - 1)),
                int(np.clip(np.searchsorted(b2, d2) - 1, 0, N_BINS - 1)))

    def offer(x, y):
        n_try[0] += 1
        if not valid(x, y):
            return None
        m, w, _ = score(x, y)
        n_eval[0] += 1
        k = cell_of(*descriptors(x, y))
        cur = cells.get(k)
        if cur is None or m > cur["m"]:
            cells[k] = {"m": m, "w": w, "x": x.copy(), "y": y.copy()}
        return m

    offer(x0, y0)                                              # incumbent seeds
    for sd in np.arange(5.0, 9.51, 0.25):                      # valid-grid sweep
        for rot in (0, 15, 30, 45, 60, 75):
            gx, gy = wfs.grid_layout(N_T, spacing_d=float(sd), diameter_m=DIAM,
                                     rotation_deg=float(rot))
            if np.abs(gx).max() <= HALF and np.abs(gy).max() <= HALF:
                offer(gx, gy)
    print(f"seeded: {len(cells)} cells, {n_eval[0]} evals of {n_try[0]} tries "
          f"({time.time()-t0:.0f}s)", flush=True)

    best_hist = []
    while n_eval[0] < N_EVAL:
        p = cells[list(cells)[rng.integers(len(cells))]]
        frac = n_eval[0] / N_EVAL
        sigma = DIAM * (1.4 * (1 - frac) + 0.15)               # anneal
        offer(*mutate(p["x"], p["y"], rng, sigma))
        if n_eval[0] % 250 == 0 and (not best_hist or best_hist[-1][0] != n_eval[0]):
            bm = max(c["m"] for c in cells.values())
            best_hist.append((n_eval[0], bm))
            print(f"  evals {n_eval[0]:5d}/{N_EVAL}  tries {n_try[0]:6d}  "
                  f"cells {len(cells):3d}  best {bm*100:.3f} "
                  f"({100*(bm-inc_mean):+.3f} pp)  [{time.time()-t0:.0f}s]", flush=True)

    elites = sorted(cells.values(), key=lambda c: -c["m"])[:15]
    rows = []
    for i, e in enumerate(elites):
        ok, msgs = wfs.validate_layout(e["x"], e["y"], box_size_m=BOX,
                                       max_turbines=N_T, min_spacing_d=5.0,
                                       diameter_m=DIAM)
        d1, d2 = descriptors(e["x"], e["y"])
        rows.append(dict(cf_mean=e["m"], cf_min=e["w"], nn_spacing_D=d1, elongation=d2,
                         valid=bool(ok), validator_msgs=msgs,
                         d_cf_mean_pp=100 * (e["m"] - inc_mean),
                         d_cf_min_pp=100 * (e["w"] - inc_min),
                         x=[float(v) for v in e["x"]], y=[float(v) for v in e["y"]]))
        print(f"  elite {i:2d}  mean {e['m']*100:.3f} ({rows[-1]['d_cf_mean_pp']:+.3f} pp)"
              f"  worst {e['w']*100:.3f} ({rows[-1]['d_cf_min_pp']:+.3f} pp)"
              f"  nn {d1:.2f}D  elong {d2:.2f}  valid={ok}", flush=True)

    win = [r for r in rows if r["d_cf_mean_pp"] > 0 and r["d_cf_min_pp"] > 0 and r["valid"]]
    win.sort(key=lambda r: -r["cf_mean"])
    best = win[0] if win else None
    json.dump(dict(site=dict(lat=lat0, lon=lon0),
                   incumbent=dict(cf_mean=inc_mean, cf_min=inc_min,
                                  per_year={str(y): float(c) for y, c in zip(YEARS, inc_cfs)}),
                   search=dict(objective="full 5-year replay mean CF",
                               n_evaluations=n_eval[0], n_tries=n_try[0],
                               valid_rate=n_eval[0] / max(n_try[0], 1),
                               archive_cells=len(cells), best_history=best_hist,
                               seed=SEED),
                   elites=rows, promoted=best,
                   promotion_rule="beat incumbent on BOTH cf_mean and cf_min, and pass validate_layout"),
              open(os.path.join(OUT, "siting_qd2.json"), "w"), indent=2)

    print(f"\nincumbent {inc_mean*100:.3f} / worst {inc_min*100:.3f}")
    if best:
        print(f"PROMOTED  mean {best['cf_mean']*100:.3f} ({best['d_cf_mean_pp']:+.3f} pp)"
              f"  worst {best['cf_min']*100:.3f} ({best['d_cf_min_pp']:+.3f} pp)")
    else:
        print("NO candidate beat the incumbent on both => the regular grid stands.")
    print(f"[{time.time()-t0:.0f}s] wrote {OUT}/siting_qd2.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
