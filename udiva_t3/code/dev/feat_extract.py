"""Extract frozen DINOv2 per-participant features over the GF exo prefix [t_b-L, t_b].
A = left crop (participant_a), B = right crop (participant_b). Frames deduped across
overlapping segments. Saves /scratch/.../t3/feats/<sid>.npz.

Run on GPU via SLURM (login CPU is slow). Usage: python feat_extract.py <sid|all>
"""
import os, sys, time
os.environ["HF_HOME"] = "<cache>/huggingface"  # FORCE (shell default = ~/.cache)
os.environ["HF_HUB_OFFLINE"] = "1"        # compute nodes have no network; model pre-cached
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import numpy as np
import torch
import decord
from PIL import Image
from transformers import AutoModel
import _path  # noqa: F401
import udiva_data as U
import cv as CV

decord.bridge.set_bridge("native")
GF_DIR = U.DATA_ROOT / "audiovisual" / "exo" / "GF"
OUT_DIR = U.WORK_DIR / "feats"
MODEL = "facebook/dinov2-small"
L = 3.0          # prefix lookback seconds
N_F = 6          # frames per segment
DIM = 384
BATCH = 96

_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)   # ImageNet (DINOv2)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_model():
    model = AutoModel.from_pretrained(MODEL).to(_DEV).eval()
    return model


def crops(frame):
    """frame HxWx3 uint8 -> (A_left, B_right) PIL 224x224."""
    h, w, _ = frame.shape
    a = frame[:, : int(w * 0.60)]          # left 60% (participant_a + center)
    b = frame[:, int(w * 0.40):]           # right 60% (participant_b + center)
    pa = Image.fromarray(a).resize((224, 224))
    pb = Image.fromarray(b).resize((224, 224))
    return pa, pb


@torch.no_grad()
def encode(model, pil_list):
    out = np.zeros((len(pil_list), DIM), dtype=np.float32)
    mean, std = _MEAN.to(_DEV), _STD.to(_DEV)
    for i in range(0, len(pil_list), BATCH):
        chunk = pil_list[i:i + BATCH]
        arr = np.stack([np.asarray(im, dtype=np.float32) for im in chunk]) / 255.0  # [B,224,224,3]
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).to(_DEV)
        t = (t - mean) / std
        f = model(pixel_values=t).last_hidden_state[:, 0]   # CLS
        out[i:i + len(chunk)] = f.float().cpu().numpy()
    return out


def seg_frame_idxs(t_b, fps, nframes):
    ts = [max(0.0, t_b - L) + (i + 0.5) * (min(L, t_b)) / N_F for i in range(N_F)]
    idx = [min(nframes - 1, max(0, int(round(t * fps)))) for t in ts]
    return idx


def process_session(sid, gt, model):
    vp = GF_DIR / f"{sid}.mp4"
    vr = decord.VideoReader(str(vp), num_threads=4)
    fps = vr.get_avg_fps(); nframes = len(vr)
    segs = list(gt[sid].items())
    # union of frame idxs
    seg_idxs = {}
    union = set()
    for seg_id, seg in segs:
        idx = seg_frame_idxs(seg["t_b"], fps, nframes)
        seg_idxs[seg_id] = idx
        union.update(idx)
    union = sorted(union)
    # read + encode unique frames
    featA = {}; featB = {}
    for i in range(0, len(union), 256):
        chunk = union[i:i + 256]
        frames = vr.get_batch(chunk).asnumpy()
        pasA = []; pasB = []
        for fr in frames:
            pa, pb = crops(fr)
            pasA.append(pa); pasB.append(pb)
        fa = encode(model, pasA); fb = encode(model, pasB)
        for j, fidx in enumerate(chunk):
            featA[fidx] = fa[j]; featB[fidx] = fb[j]
    # assemble per-segment
    seg_ids = [s for s, _ in segs]
    tb = np.array([seg["t_b"] for _, seg in segs], dtype=np.float32)
    FA = np.stack([np.stack([featA[k] for k in seg_idxs[s]]) for s in seg_ids])  # [nseg,N_F,DIM]
    FB = np.stack([np.stack([featB[k] for k in seg_idxs[s]]) for s in seg_ids])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / f"{sid}.npz", seg_ids=np.array(seg_ids), t_b=tb, featA=FA, featB=FB)
    return len(seg_ids), len(union)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    gt = CV.build_all_gt()
    sids = U.list_sessions() if which == "all" else [which]
    model = load_model()
    print(f"device={_DEV} model={MODEL}")
    for sid in sids:
        t0 = time.time()
        ns, nf = process_session(sid, gt, model)
        print(f"  {sid}: {ns} segs, {nf} unique frames, {time.time()-t0:.1f}s", flush=True)
        if limit and sid == sids[limit - 1]:
            break


if __name__ == "__main__":
    main()
