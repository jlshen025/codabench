"""Cache AROME target (u125m/v125m) at the 43,715-pt footprint for the scored
hours (0,6,12,18), 2016-2020 → a FEW large files under /scratch. Foundational
forecast asset: reused for climatology, downscaler/MOS training, and CV truth.

Per year: arome_fp_{year}.npz with u,v arrays shape (ndays, 4, 43715) float32 +
a dates array (YYYYMMDD int32). Footprint order = footprint_points.parquet order.
AROME steps are 3-hourly (00..21); scored hours 0/6/12/18 = step indices 0,2,4,6.

SLURM CPU job. Test: --years 2020 --limit 5.
"""
from __future__ import annotations
import os, sys, glob, time, argparse
os.environ.setdefault("PHASE2_DATA_ROOT", os.environ["SEAWINDS_PHASE2_DIR"])
KIT = "${SEAWINDS_ROOT}/kit_phase2/phase_2"
sys.path.insert(0, KIT); sys.path.insert(0, KIT + "/part0_dataset_setup")
import numpy as np, pandas as pd, xarray as xr
import footprint as fp

P2 = os.environ["SEAWINDS_PHASE2_DIR"]
AROME_ROOT = f"{P2}/train/arome"
HOUR_IDX = np.array([0, 2, 4, 6])   # steps for hours 0,6,12,18 (3-hourly grid)
HOURS = [0, 6, 12, 18]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2016,2017,2018,2019,2020")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="./work/cache/arome_footprint")
    args = ap.parse_args()
    years = [int(y) for y in args.years.split(",")]
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    mask = fp.footprint_mask()
    ys, xs = np.where(mask)
    fpp = pd.read_parquet(fp.FOOTPRINT_PATH)
    assert len(fpp) == len(ys), (len(fpp), len(ys))
    n = len(ys)
    # persist footprint order once (point_id, lat, lon) for downstream joins
    fpp.to_parquet(os.path.join(args.out, "footprint_order.parquet"), index=False)
    print(f"footprint n={n}, hours={HOURS}", flush=True)

    for y in years:
        files = sorted(glob.glob(f"{AROME_ROOT}/{y}/arome_*.nc"))
        if args.limit:
            files = files[:args.limit]
        nd = len(files)
        U = np.full((nd, 4, n), np.nan, np.float32)
        V = np.full((nd, 4, n), np.nan, np.float32)
        dates = np.zeros(nd, np.int32)
        for i, f in enumerate(files):
            stem = os.path.basename(f)[len("arome_"):-len(".nc")]
            dates[i] = int(stem)
            with xr.open_dataset(f) as ds:
                u = ds["u125m"].values[HOUR_IDX][:, ys, xs]   # (4, n)
                v = ds["v125m"].values[HOUR_IDX][:, ys, xs]
            U[i] = u.astype(np.float32); V[i] = v.astype(np.float32)
        dst = os.path.join(args.out, f"arome_fp_{y}.npz")
        np.savez(dst, u=U, v=V, dates=dates, hours=np.array(HOURS, np.int32))
        finite = np.isfinite(U).mean()
        print(f"  {y}: {nd} days -> {dst}  finite%={finite*100:.0f}  ({time.time()-t0:.0f}s)", flush=True)

    # tiny manifest for downstream result verification
    man = os.path.join(os.environ.get("EXP_OUTPUT_DIR", args.out), "cache_manifest.json")
    import json
    with open(man, "w") as fh:
        json.dump({"years": years, "n_footprint": int(n), "hours": HOURS,
                   "out": args.out, "elapsed_s": round(time.time()-t0, 1)}, fh)
    print(f"wrote manifest {man} ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
