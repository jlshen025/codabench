"""Frozen MotionAGFormer features for the UPDRS gait task.
Pipeline per walk: SMPL canonical -> H36M 3D joints (smpl_fk) -> resample 30fps ->
[per view] root-center x,z + orthographic 2D (x,y,conf=1) -> 81-frame clips ->
crop_scale [-1,1] -> frozen MotionAGFormer (return_rep) -> mean over valid frames
-> mean over clips -> per-walk (17,512). Saves npz (X_<view>, y, cohort, subject, walk).

Views: side=(fwd z, up -y), front=(lat x, up -y). Both root-relative (in-place).
"""
import os, sys, argparse, numpy as np, torch

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.join(_DIR, "..", "..", "code"))   # for `model.motionagformer...`
from dataio import load_cohort, UPDRS_COHORTS
from smpl_fk import SMPLH36M, resample_30fps

# The frozen encoder checkpoint — see assets/README.md for how to obtain it.
CKPT = os.environ.get("MAGF_CKPT",
                      os.path.join(_DIR, "..", "..", "assets", "motionagformer-s-h36m.pth.tr"))
CLIP = 81

MAGF_CFG = dict(n_layers=26, dim_in=3, dim_feat=64, dim_rep=512, dim_out=3, mlp_ratio=4,
                attn_drop=0.0, drop=0.0, drop_path=0.0, use_layer_scale=True,
                layer_scale_init_value=1e-5, use_adaptive_fusion=True, num_heads=8,
                qkv_bias=False, qkv_scale=None, hierarchical=False, num_joints=17,
                use_temporal_similarity=True, temporal_connection_len=1, use_tcn=False,
                graph_only=False, neighbour_num=2, n_frames=CLIP)


def build_encoder(ckpt, device):
    from model.motionagformer.MotionAGFormer import MotionAGFormer
    import torch.nn as nn
    m = MotionAGFormer(act_layer=nn.GELU, **MAGF_CFG)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    new = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    md = m.state_dict()
    keep = {k: v for k, v in new.items() if k in md and v.shape == md[k].shape}
    missing = [k for k in md if k not in keep]
    m.load_state_dict({**md, **keep}, strict=True)
    print(f"[encoder] loaded {len(keep)}/{len(md)} layers ({len(missing)} kept from init)")
    return m.to(device).eval()


def crop_scale(motion):
    """CARE-PD crop_scale: x,y bbox-normalized to [-1,1]; 3rd dim (conf) clipped. (T,17,3)."""
    res = motion.copy()
    valid = motion[motion[..., 2] != 0][:, :2]
    if len(valid) < 4:
        return np.zeros_like(motion)
    xmin, xmax = valid[:, 0].min(), valid[:, 0].max()
    ymin, ymax = valid[:, 1].min(), valid[:, 1].max()
    scale = max(xmax - xmin, ymax - ymin)
    if scale == 0:
        return np.zeros_like(motion)
    xs = (xmin + xmax - scale) / 2; ys = (ymin + ymax - scale) / 2
    res[..., :2] = (motion[..., :2] - np.array([xs, ys])) / scale
    res[..., :2] = (res[..., :2] - 0.5) * 2
    return np.clip(res, -1, 1)


def project(joints, view):
    """joints (T,17,3) canonical x=lat,y=up,z=fwd -> (T,17,3) 2D (x,y,conf=1).
    base 'side'=(fwd z, up -y), 'front'=(lat x, up -y). Suffix '_glob' keeps global
    translation (walks across frame, retains velocity); else root-relative (in-place)."""
    j = joints.copy()
    glob = view.endswith("_glob")
    base = view[:-5] if glob else view
    if not glob:
        j[:, :, 0] -= j[:, 0:1, 0]    # center lateral per frame
        j[:, :, 2] -= j[:, 0:1, 2]    # center fwd per frame (remove walk translation -> in-place)
    if base == "side":
        x2, y2 = j[:, :, 2], -j[:, :, 1]
    elif base == "front":
        x2, y2 = j[:, :, 0], -j[:, :, 1]
    elif base == "back":
        x2, y2 = -j[:, :, 0], -j[:, :, 1]    # posterior coronal (view from behind: lateral x flipped)
    else:
        raise ValueError(view)
    return np.stack([x2, y2, np.ones_like(x2)], axis=-1).astype(np.float32)


def clips_from(seq):
    """seq (T,17,3) -> (n,81,17,3), (n,81) padmask. Non-overlapping stride 81; pad short."""
    T = seq.shape[0]
    if T < CLIP:
        pad = np.zeros((CLIP - T, 17, 3), np.float32)
        c = np.concatenate([seq, pad], 0)[None]
        pm = np.concatenate([np.ones(T), np.zeros(CLIP - T)])[None]
        return c, pm.astype(np.float32)
    cs, pms, s = [], [], 0
    while T - s >= CLIP:
        cs.append(seq[s:s + CLIP]); pms.append(np.ones(CLIP, np.float32)); s += CLIP
    return np.stack(cs), np.stack(pms)


@torch.no_grad()
def encode_walk(enc, joints, views, device, max_bs=256):
    """joints (T,17,3) -> dict view -> (17,512) per-walk pooled feature."""
    out = {}
    for v in views:
        seq = project(joints, v)
        clips_arr, pm = clips_from(seq)                   # (n,81,17,3),(n,81)
        clips = np.stack([crop_scale(c) for c in clips_arr])
        feats = []
        for i in range(0, clips.shape[0], max_bs):
            xb = torch.as_tensor(clips[i:i + max_bs], dtype=torch.float32, device=device)
            rep = enc(xb, return_rep=True)                # (b,81,17,512)
            mb = torch.as_tensor(pm[i:i + max_bs], dtype=torch.float32, device=device)  # (b,81)
            num = (rep * mb[:, :, None, None]).sum(1)
            den = mb.sum(1).clamp(min=1e-6)[:, None, None]
            feats.append((num / den).cpu().numpy())       # (b,17,512)
        allf = np.concatenate(feats, 0)                   # (n_clips,17,512)
        out[v] = {"mean": allf.mean(0),                   # (17,512)
                  "std": allf.std(0) if allf.shape[0] > 1 else np.zeros_like(allf[0])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=list(UPDRS_COHORTS))
    ap.add_argument("--views", nargs="+", default=["side", "front"])
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--out", default=os.path.join(os.environ.get("CACHE_DIR", "./cache"),
                                                  "feats_magf.npz"))
    ap.add_argument("--limit", type=int, default=0, help="smoke-test: cap walks per cohort")
    ap.add_argument("--min_frames", type=int, default=30)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[extract] device={device} views={args.views} cohorts={args.cohorts}")
    smpl = SMPLH36M(device=device)
    enc = build_encoder(args.ckpt, device)

    feats = {v: [] for v in args.views}
    feats_std = {v: [] for v in args.views}
    y, coh, subj, walk = [], [], [], []
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def snapshot():
        save = {f"X_{v}": np.asarray(feats[v], np.float32) for v in args.views}
        save.update({f"X_{v}_std": np.asarray(feats_std[v], np.float32) for v in args.views})
        save.update(y=np.asarray(y, np.int64), cohort=np.asarray(coh),
                    subject=np.asarray(subj), walk=np.asarray(walk))
        return save

    import time
    for name in args.cohorts:
        n0 = len(y); t0 = time.time()
        for k, s in enumerate(load_cohort(name)):
            if args.limit and k >= args.limit:
                break
            stride = max(1, int(round(float(s["fps"]) / 30.0)))   # resample BEFORE FK (big win @150fps)
            pose, trans = s["pose"][::stride], s["trans"][::stride]
            if pose.shape[0] < args.min_frames:
                continue
            j = smpl.joints(pose, trans)                          # already ~30fps
            fw = encode_walk(enc, j, args.views, device)
            for v in args.views:
                feats[v].append(fw[v]["mean"]); feats_std[v].append(fw[v]["std"])
            y.append(s["label"]); coh.append(s["cohort"]); subj.append(s["subject"]); walk.append(s["walk"])
        print(f"  {name}: {len(y) - n0} walks ({time.time()-t0:.0f}s)", flush=True)
        np.savez_compressed(args.out, **snapshot())             # checkpoint per cohort

    save = snapshot()
    np.savez_compressed(args.out, **save)
    print(f"[extract] saved {args.out}: " + ", ".join(f"X_{v}{save['X_'+v].shape}" for v in args.views))
    print(f"  classes {np.unique(save['y'], return_counts=True)}")


if __name__ == "__main__":
    main()
