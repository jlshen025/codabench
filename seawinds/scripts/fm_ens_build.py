"""Build a Pangu IC-perturbed ENSEMBLE: perturb each ERA5 init npz M-1 ways +
write a manifest for fm_pangu_infer.py to roll.

Gate-1 FAIR: perturbation only adds a small, physically-scaled wind IC noise to an
ERA5 init at context_end; ERA5-only, AROME never touches inference. Standard FM-
ensemble practice (IC perturbation seeds the chaotic divergence a static forecast
cannot express).

Perturbation: one vertically-COHERENT, spatially-SMOOTH (synoptic-scale) 2D pattern
per (issue,member) added to u & v at all levels + surface u10/v10, std ~0.7 m/s.
Coherent+smooth so it projects onto growing modes (white noise would be damped ->
false-negative spread). Member 0 = control (unperturbed). rank-corr spread-skill is
scale-invariant, so the exact amplitude is not critical for the STEP-1 test.

Usage: python fm_ens_build.py --tags-file <json list of {tag,init_npz}> --members 8 --out-manifest <path>
   or  --init-glob 'era5_init_cv/cv2020_*_init.npz'
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
from scipy.ndimage import gaussian_filter

FM = os.environ.get("SEAWINDS_FM_DIR", "./work/fm")
ENS_DIR = os.environ.get("FM_ENS_INIT_DIR", f"{FM}/ens_init")
IU, IV = 3, 4       # upper u,v channel (of [z,q,t,u,v])
SU10, SV10 = 1, 2   # surface u10,v10 channel (of [mslp,u10,v10,t2m])
AMP_MS = 0.7        # perturbation std (m/s), synoptic scale
SMOOTH = 4.0        # gaussian_filter sigma in grid cells (~synoptic)


def smooth_field(rng, shape):
    n = rng.standard_normal(shape)
    n = gaussian_filter(n, sigma=SMOOTH, mode="wrap")
    n *= AMP_MS / (n.std() + 1e-9)   # renormalize post-smoothing to AMP_MS std
    return n.astype(np.float32)


def perturb(src_npz, dst_npz, seed):
    z = dict(np.load(src_npz))
    up = z["upper"].astype(np.float32).copy()      # [5,13,nlat,nlon]
    sf = z["surface"].astype(np.float32).copy()    # [4,nlat,nlon]
    rng = np.random.default_rng(seed)
    nlat, nlon = up.shape[-2], up.shape[-1]
    du = smooth_field(rng, (nlat, nlon)); dv = smooth_field(rng, (nlat, nlon))
    up[IU] += du[None, :, :]     # same coherent pattern across all 13 levels
    up[IV] += dv[None, :, :]
    sf[SU10] += du; sf[SV10] += dv
    z["upper"] = up; z["surface"] = sf
    np.savez(dst_npz, **z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-glob", default="era5_init_cv/cv2020_*_init.npz")
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--out-manifest", default=f"{FM}/ens_manifest_cv2020.json")
    a = ap.parse_args()
    os.makedirs(ENS_DIR, exist_ok=True)
    inits = sorted(glob.glob(a.init_glob if os.path.isabs(a.init_glob) else f"{FM}/{a.init_glob}"))
    assert inits, f"no inits match {a.init_glob}"
    manifest = []
    for ii, src in enumerate(inits):
        base = os.path.basename(src).replace("_init.npz", "")   # e.g. cv2020_20200115
        for m in range(a.members):
            tag = f"{base}_m{m}"
            dst = f"{ENS_DIR}/{tag}_init.npz"
            if m == 0:
                # control = symlink-free copy of the original init
                if not os.path.exists(dst):
                    z = dict(np.load(src)); np.savez(dst, **z)
            else:
                if not os.path.exists(dst):
                    perturb(src, dst, seed=1000 * m + ii)
            manifest.append({"tag": tag, "init_npz": dst})
    json.dump(manifest, open(a.out_manifest, "w"), indent=1)
    print(f"{len(inits)} issues x {a.members} members = {len(manifest)} rollouts -> {a.out_manifest}")
    print(f"perturbation: coherent smooth wind IC, std {AMP_MS} m/s, smooth sigma {SMOOTH} cells")


if __name__ == "__main__":
    main()
