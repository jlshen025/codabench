"""Train-time AUGMENTED MotionAGFormer features to attack the cross-cohort
generalization wall (BMCLab/T-SDU-PD LODO folds stuck ~0.34). Per CARE-PD recipe,
augment the 3D joints BEFORE projection: mirror (L/R swap + negate lateral x) and
ground-plane rotation (around up-axis y, ±deg). Frozen MAGF, side_glob jointmean.

Saves per (walk x aug) row so CV can train on ALL augmentations of the training
cohorts but TEST only on the original held-out walks:
  X_side_glob (Naug,512), aug (Naug,) in {orig,mirror,rot-10,rot+10},
  walkidx (Naug,) -> index into per-walk y/cohort/subject/walk.
Out (--out, default ./cache/feats_magf_aug.npz).
"""
import os, sys, time, argparse, numpy as np, torch
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR); sys.path.insert(0, os.path.join(_DIR, "CARE-PD"))
from dataio import load_cohort, UPDRS_COHORTS
from smpl_fk import SMPLH36M
from extract_feats import project, crop_scale, clips_from, build_encoder, CKPT, CLIP

# H36M-17 L/R swap (0 pelvis,1 RHip,2 RKnee,3 RAnk,4 LHip,5 LKnee,6 LAnk,7 Sp,8 Thx,
# 9 Neck,10 Head,11 LSho,12 LElb,13 LWri,14 RSho,15 RElb,16 RWri)
MIRROR_IDX = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]


def aug_mirror(j):
    jm = j[:, MIRROR_IDX, :].copy()
    jm[..., 0] *= -1.0           # negate lateral (x)
    return jm


def aug_rot(j, deg):
    th = np.radians(deg); c, s = np.cos(th), np.sin(th)
    jr = j.copy()
    x = j[..., 0]; z = j[..., 2]
    jr[..., 0] = c * x + s * z   # rotate in ground (x,z) plane around up-axis y
    jr[..., 2] = -s * x + c * z
    return jr


def aug_speed(j, factor):
    """Temporal speed-warp: resample (T,17,3) to T'=round(T/factor) frames by linear
    interpolation along time. factor>1 = faster (fewer frames), <1 = slower. Makes the
    probe pace-invariant (gait pace is both a severity cue AND a cross-cohort confound)."""
    T = j.shape[0]
    Tn = max(2, int(round(T / factor)))
    if Tn == T:
        return j.copy()
    src = np.linspace(0.0, T - 1.0, Tn)
    lo = np.floor(src).astype(int); hi = np.minimum(lo + 1, T - 1)
    w = (src - lo)[:, None, None]
    return (j[lo] * (1 - w) + j[hi] * w).astype(j.dtype)   # (Tn,17,3)


def aug_jitter(j, sigma, seed):
    """Per-joint Gaussian position noise (m): simulates cross-cohort tracking/keypoint
    noise. Label-preserving (small). Deterministic via seed for reproducible feats."""
    rng = np.random.RandomState(seed)
    return (j + rng.normal(0, sigma, j.shape).astype(j.dtype)).astype(j.dtype)


def aug_framedrop(j, p, seed):
    """Randomly drop frames (then they're skipped): simulates missing/occluded frames &
    variable effective fps across cohorts. Keeps >=30 frames."""
    rng = np.random.RandomState(seed)
    T = j.shape[0]
    keep = rng.rand(T) >= p
    if keep.sum() < 30:
        return j.copy()
    return j[keep].copy()


def aug_crop(j, frac, seed):
    """Random contiguous time-crop to frac*T frames (>=30): simulates different captured
    walk-segments across cohorts (start/end trimming). Label-preserving for steady gait."""
    rng = np.random.RandomState(seed)
    T = j.shape[0]; L = max(30, int(round(frac * T)))
    if L >= T:
        return j.copy()
    st = rng.randint(0, T - L + 1)
    return j[st:st + L].copy()


# FRAME-DROPOUT was the big server winner (v7, +0.027). This set = v7's train-augs
# (v5 core + drop10/20) + STRONGER drop30/40 + time-crop (same temporal-subsample family)
# + mir_drop. Superset of v7 → reproduces v7 AND tests the extensions. -> feats_magf_augE.npz
AUGS = [("orig", lambda j: j), ("mirror", aug_mirror),
        ("rot-10", lambda j: aug_rot(j, -10)), ("rot+10", lambda j: aug_rot(j, 10)),
        ("spd0.9", lambda j: aug_speed(j, 0.9)), ("spd1.1", lambda j: aug_speed(j, 1.1)),
        ("mir_spd0.9", lambda j: aug_speed(aug_mirror(j), 0.9)),
        ("mir_spd1.1", lambda j: aug_speed(aug_mirror(j), 1.1)),
        ("drop10", lambda j: aug_framedrop(j, 0.10, 3)), ("drop20", lambda j: aug_framedrop(j, 0.20, 4)),
        ("drop30", lambda j: aug_framedrop(j, 0.30, 6)), ("drop40", lambda j: aug_framedrop(j, 0.40, 7)),
        ("crop85", lambda j: aug_crop(j, 0.85, 8)), ("crop70", lambda j: aug_crop(j, 0.70, 9)),
        ("mir_drop20", lambda j: aug_framedrop(aug_mirror(j), 0.20, 10))]


@torch.no_grad()
def walk_feat(enc, joints, views, device, max_bs=256):
    out = {}
    for v in views:
        seq = project(joints, v); clips_arr, pm = clips_from(seq)
        clips = np.stack([crop_scale(c) for c in clips_arr])
        feats = []
        for i in range(0, clips.shape[0], max_bs):
            xb = torch.as_tensor(clips[i:i + max_bs], dtype=torch.float32, device=device)
            rep = enc(xb, return_rep=True)
            mb = torch.as_tensor(pm[i:i + max_bs], dtype=torch.float32, device=device)
            num = (rep * mb[:, :, None, None]).sum(1); den = mb.sum(1).clamp(min=1e-6)[:, None, None]
            feats.append((num / den).cpu().numpy())          # (b,17,512)
        allf = np.concatenate(feats, 0)
        out[v] = allf.mean(0).mean(0)                         # mean over clips, then joints -> (512,)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="+", default=["side_glob"])
    ap.add_argument("--out", default=os.path.join(os.environ.get("CACHE_DIR", "./cache"),
                                                  "feats_magf_aug.npz"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min_frames", type=int, default=30)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[aug-extract] device={device} views={args.views} augs={[a for a,_ in AUGS]}", flush=True)
    smpl = SMPLH36M(device=device); enc = build_encoder(CKPT, device)

    feats = {v: [] for v in args.views}
    aug_list, walkidx = [], []
    y, coh, subj, walk = [], [], [], []
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for name in UPDRS_COHORTS:
        n0 = len(y); t0 = time.time()
        for k, sdct in enumerate(load_cohort(name)):
            if args.limit and k >= args.limit:
                break
            stride = max(1, int(round(float(sdct["fps"]) / 30.0)))
            pose = np.ascontiguousarray(sdct["pose"][::stride]); trans = np.ascontiguousarray(sdct["trans"][::stride])
            if pose.shape[0] < args.min_frames:
                continue
            j = smpl.joints(pose, trans)                      # (T,17,3) canonical
            wi = len(y)
            y.append(sdct["label"]); coh.append(sdct["cohort"]); subj.append(sdct["subject"]); walk.append(sdct["walk"])
            for aname, afn in AUGS:
                fw = walk_feat(enc, afn(j), args.views, device)
                for v in args.views:
                    feats[v].append(fw[v])
                aug_list.append(aname); walkidx.append(wi)
        print(f"  {name}: {len(y) - n0} walks x{len(AUGS)} ({time.time()-t0:.0f}s)", flush=True)
        save = {f"X_{v}": np.asarray(feats[v], np.float32) for v in args.views}
        save.update(aug=np.asarray(aug_list), walkidx=np.asarray(walkidx, np.int64),
                    y=np.asarray(y, np.int64), cohort=np.asarray(coh), subject=np.asarray(subj), walk=np.asarray(walk))
        np.savez_compressed(args.out, **save)
    print(f"[aug-extract] saved {args.out}: rows={len(aug_list)} walks={len(y)} "
          + ", ".join(f"X_{v}{np.asarray(feats[v]).shape}" for v in args.views), flush=True)


if __name__ == "__main__":
    main()
