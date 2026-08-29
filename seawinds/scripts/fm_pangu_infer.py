"""Pangu-Weather ONNX rollout -> FM driver features on the coarse (45x57) grid.

For each issue state (ERA5 init npz = Pangu input), roll the 24h model x14 saving
the state at steps {1,7,14} = leads d1/d7/d14 (hour 0); at each saved state roll
the 6h model x{1,2,3} for hours {6,12,18}. Extract North-Sea wind at 10m / 1000 /
925 hPa onto the coarse grid, emit a long-format parquet the MOS consumes (mirrors
kit forecast_hres.build_hres_table: lat,lon,lead,hour,fcst_u,fcst_v,fcst_speed).

Gate-1 FAIR: init is context_end 00 UTC, ERA5-only; AROME never touches inference.

Usage:
  python fm_pangu_infer.py --tags window_1[,window_2,...] --out <parquet> [--device cpu|cuda]
  python fm_pangu_infer.py --manifest <json list of {tag,init_npz,issue_date}> --out <parquet>
Init npz resolves to $FM/era5_init/<tag>_init.npz unless given in a manifest.
"""
from __future__ import annotations
import argparse, json, os, time, glob
import numpy as np, pandas as pd, xarray as xr

FM = os.environ.get("SEAWINDS_FM_DIR", "./work/fm")
WEIGHTS = os.environ.get("FM_WEIGHTS_DIR", f"{FM}/weights")
INIT_DIR = os.environ.get("FM_ERA5_INIT_DIR", f"{FM}/era5_init")
PANGU_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
LV = {p: i for i, p in enumerate(PANGU_LEVELS)}
SAVE_STEPS = {1: 1, 7: 7, 14: 14}          # lead -> #24h steps
HOURS6 = (6, 12, 18)                        # 6h-model sub-steps after each saved state
# Pangu upper var order [z,q,t,u,v]; surface [mslp,u10,v10,t2m]
IU, IV = 3, 4                               # upper u/v channel
SU10, SV10 = 1, 2                           # surface u10/v10 channel


def coarse_grid():
    p2 = os.environ["SEAWINDS_PHASE2_DIR"]
    f = sorted(glob.glob(f"{p2}/train/arome_coarse125/*/coarse_*.nc"))[0]
    ds = xr.open_dataset(f)
    lat = ds.latitude.values.astype(float); lon = ds.longitude.values.astype(float)
    ds.close()
    return lat, lon                        # lat 45 asc 51..62, lon 57 -4..10


def make_sessions(device):
    import onnxruntime as ort
    provs = (["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda"
             else ["CPUExecutionProvider"])
    so = ort.SessionOptions()
    # Pangu ONNX is memory-heavy on CPU: the mem arena + mem pattern pre-allocate
    # huge pools (OOM > 16 GB). Disable both and use BASIC opt to bound peak RAM.
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    if device != "cuda":
        so.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
    s24 = ort.InferenceSession(f"{WEIGHTS}/pangu_weather_24.onnx", sess_options=so, providers=provs)
    s6 = ort.InferenceSession(f"{WEIGHTS}/pangu_weather_6.onnx", sess_options=so, providers=provs)
    print("providers:", s24.get_providers(), flush=True)
    return s24, s6


def step(sess, up, sf):
    o = sess.run(None, {"input": up, "input_surface": sf})
    return o[0].astype(np.float32), o[1].astype(np.float32)


def to_coarse(up, sf, lat_p, lon_p, clat, clon):
    """Extract (u10,v10,u1000,v1000,u925,v925) onto the coarse grid via xarray interp.
    Pangu lon 0..360 -> [-180,180) so the North-Sea seam (-4..10) is contiguous."""
    lon_m = np.where(lon_p >= 180, lon_p - 360.0, lon_p)
    order = np.argsort(lon_m)
    lon_s = lon_m[order]
    fields = {
        "u10": sf[SU10][:, order], "v10": sf[SV10][:, order],
        "u1000": up[IU, LV[1000]][:, order], "v1000": up[IV, LV[1000]][:, order],
        "u925": up[IU, LV[925]][:, order], "v925": up[IV, LV[925]][:, order],
    }
    out = {}
    for k, arr in fields.items():
        da = xr.DataArray(arr, coords={"latitude": lat_p, "longitude": lon_s},
                          dims=["latitude", "longitude"])
        out[k] = da.interp(latitude=clat, longitude=clon, method="linear").values  # (45,57)
    return out


def rollout(s24, s6, init_npz, lat_p, lon_p, clat, clon):
    z = np.load(init_npz)
    up = z["upper"].astype(np.float32); sf = z["surface"].astype(np.float32)
    issue = str(z["init"]) if "init" in z else ""
    rows = []
    LON, LAT = np.meshgrid(clon, clat)
    flat_lat, flat_lon = LAT.ravel(), LON.ravel()

    def emit(lead, hour, up_s, sf_s):
        f = to_coarse(up_s, sf_s, lat_p, lon_p, clat, clon)
        u10, v10 = f["u10"].ravel(), f["v10"].ravel()
        u1000, v1000 = f["u1000"].ravel(), f["v1000"].ravel()
        u925, v925 = f["u925"].ravel(), f["v925"].ravel()
        rows.append(pd.DataFrame({
            "lat": flat_lat, "lon": flat_lon, "lead": lead, "hour": hour,
            "u10": u10, "v10": v10, "u1000": u1000, "v1000": v1000,
            "u925": u925, "v925": v925,
        }))

    cur_up, cur_sf = up, sf
    for st in range(1, 15):
        cur_up, cur_sf = step(s24, cur_up, cur_sf)   # +24h
        if st in SAVE_STEPS.values():
            lead = st
            emit(lead, 0, cur_up, cur_sf)
            b_up, b_sf = cur_up, cur_sf
            for k in (1, 2, 3):
                b_up, b_sf = step(s6, b_up, b_sf)    # +6h
                emit(lead, HOURS6[k - 1], b_up, b_sf)
    out = pd.concat(rows, ignore_index=True)
    out["issue_date"] = issue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--shard", default="", help="i/n -> process items[i::n] (SLURM array)")
    a = ap.parse_args()

    if a.manifest:
        items = json.load(open(a.manifest))
    else:
        items = [{"tag": t, "init_npz": f"{INIT_DIR}/{t}_init.npz"}
                 for t in a.tags.split(",") if t]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        items = items[i::n]
        print(f"shard {i}/{n}: {len(items)} items: {[it['tag'] for it in items]}", flush=True)
    clat, clon = coarse_grid()
    z0 = np.load(items[0]["init_npz"])
    lat_p, lon_p = z0["lat"].astype(float), z0["lon"].astype(float)

    s24, s6 = make_sessions(a.device)
    t0 = time.time()
    allrows = []
    for it in items:
        df = rollout(s24, s6, it["init_npz"], lat_p, lon_p, clat, clon)
        df.insert(0, "tag", it["tag"])
        allrows.append(df)
        print(f"  {it['tag']} rows={len(df)} ({time.time()-t0:.0f}s)", flush=True)
    out = pd.concat(allrows, ignore_index=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_parquet(a.out, index=False)
    import resource
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB->GB on Linux
    print(f"DONE {len(items)} tags -> {a.out} rows={len(out)} peak_rss={peak_gb:.1f}GB ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
