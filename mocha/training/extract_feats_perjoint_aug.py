"""Step 1 of training: extract frozen MotionAGFormer-S features over the augmented training set.

Pools over valid frames and then over clips, keeping the 17 joint tokens -> (17, 512) per
(walk x augmentation). The head then means over joints; keeping them here means one feature file
serves both the joint-mean and the per-joint variant.

Augmentations ("augE"): the three that transferred — spatial jitter, temporal speed-warp,
frame dropout. Over the 4 labeled CARE-PD cohorts this yields 44280 rows.

Out (--out, default ./cache/feats_magfs_augE_pj.npz):
  X_side_glob (Naug, 17, 512), aug (Naug,), walkidx (Naug,) -> per-walk y/cohort/subject/walk.
"""
import os, sys, time, argparse, numpy as np, torch
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR); sys.path.insert(0, os.path.join(_DIR, "CARE-PD"))
from dataio import load_cohort, UPDRS_COHORTS
from smpl_fk import SMPLH36M
from extract_feats import project, crop_scale, clips_from, build_encoder, CKPT
from extract_feats_aug import AUGS


@torch.no_grad()
def walk_feat_pj(enc, joints, views, device, max_bs=256):
    """joints (T,17,3) -> dict view -> (17,512): pool over VALID frames then CLIPS, KEEP joints."""
    out = {}
    for v in views:
        seq = project(joints, v); clips_arr, pm = clips_from(seq)
        clips = np.stack([crop_scale(c) for c in clips_arr])
        feats = []
        for i in range(0, clips.shape[0], max_bs):
            xb = torch.as_tensor(clips[i:i + max_bs], dtype=torch.float32, device=device)
            rep = enc(xb, return_rep=True)                        # (b,81,17,512)
            mb = torch.as_tensor(pm[i:i + max_bs], dtype=torch.float32, device=device)
            num = (rep * mb[:, :, None, None]).sum(1); den = mb.sum(1).clamp(min=1e-6)[:, None, None]
            feats.append((num / den).cpu().numpy())              # (b,17,512)
        allf = np.concatenate(feats, 0)                          # (n_clips,17,512)
        out[v] = allf.mean(0)                                    # mean over clips ONLY -> (17,512)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="+", default=["side_glob"])
    ap.add_argument("--out", default=os.path.join(os.environ.get("CACHE_DIR", "./cache"),
                                                  "feats_magfs_augE_pj.npz"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min_frames", type=int, default=30)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[pj-aug-extract] device={device} views={args.views} augs={[a for a,_ in AUGS]}", flush=True)
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
            j = smpl.joints(pose, trans)                          # (T,17,3) canonical
            wi = len(y)
            y.append(sdct["label"]); coh.append(sdct["cohort"]); subj.append(sdct["subject"]); walk.append(sdct["walk"])
            for aname, afn in AUGS:
                fw = walk_feat_pj(enc, afn(j), args.views, device)
                for v in args.views:
                    feats[v].append(fw[v])
                aug_list.append(aname); walkidx.append(wi)
        print(f"  {name}: {len(y) - n0} walks x{len(AUGS)} ({time.time()-t0:.0f}s)", flush=True)
        save = {f"X_{v}": np.asarray(feats[v], np.float32) for v in args.views}
        save.update(aug=np.asarray(aug_list), walkidx=np.asarray(walkidx, np.int64),
                    y=np.asarray(y, np.int64), cohort=np.asarray(coh), subject=np.asarray(subj), walk=np.asarray(walk))
        np.savez_compressed(args.out, **save)
    print(f"[pj-aug-extract] saved {args.out}: rows={len(aug_list)} walks={len(y)} "
          + ", ".join(f"X_{v}{np.asarray(feats[v]).shape}" for v in args.views), flush=True)
    print("ALLDONE_PJ_AUG", flush=True)


if __name__ == "__main__":
    main()
