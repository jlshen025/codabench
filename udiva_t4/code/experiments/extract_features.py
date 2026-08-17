"""Dense DINOv2 feature cache for UDIVA-HHOI Track 4 ego videos.

Per session, decode E1 + E2 at FPS, resize SIZE, encode each frame with a frozen
DINOv2 ViT-B/14, save a per-session npz. Index by t_b at train/eval time (frames in
the observed window [t_b-W, t_b]). E1 is worn by participant_a, E2 by participant_b.

Run: python extract_features.py <session_id>
 or under SLURM array: indexes the sorted session list by SLURM_ARRAY_TASK_ID.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os, sys, time, glob, json
import numpy as np
import torch
import timm
import decord
import torch.nn.functional as F

from udiva import io as IO, feats as FT

DATA = os.path.join(IO.DATA_ROOT, "annotated_sessions", "audiovisual")
ANN = IO.ANN_DIR
OUT = FT.FEAT_DIR                       # $UDIVA_FEAT_DIR
FPS = 4.0
SIZE = 224
MODEL = "vit_base_patch14_dinov2.lvd142m"


def sessions():
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(ANN, "*.json")))


def load_model(dev):
    m = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=SIZE)
    m.eval().to(dev)
    cfg = timm.data.resolve_data_config({}, model=m)
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(dev)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(dev)
    return m, mean, std


def encode_video(path, m, mean, std, dev, bs=96):
    vr = decord.VideoReader(path, num_threads=4, width=SIZE, height=SIZE)
    fps = vr.get_avg_fps()
    n = len(vr)
    step = max(1, int(round(fps / FPS)))
    idx = list(range(0, n, step))
    times = np.array([i / fps for i in idx], dtype=np.float32)
    feats = []
    for b in range(0, len(idx), bs):
        chunk = idx[b:b + bs]
        fr = vr.get_batch(chunk).asnumpy()                      # (b,SIZE,SIZE,3) uint8
        x = torch.from_numpy(fr).to(dev).permute(0, 3, 1, 2).float() / 255.0
        x = (x - mean) / std
        with torch.no_grad():
            f = m(x).float().cpu().numpy()
        feats.append(f)
    return np.concatenate(feats, 0).astype(np.float16), times


def main():
    if len(sys.argv) > 1:
        sess = sys.argv[1]
    else:
        tid = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        sess = sessions()[tid]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)
    m, mean, std = load_model(dev)
    res = {}
    for view, sub in [("e1", "ego/E1"), ("e2", "ego/E2")]:
        path = os.path.join(DATA, sub, f"{sess}.mp4")
        t = time.time()
        feats, times = encode_video(path, m, mean, std, dev)
        res[f"feats_{view}"] = feats
        res[f"times_{view}"] = times
        print(f"{sess} {view}: feats={feats.shape} dt={time.time()-t:.1f}s dev={dev}", flush=True)
    np.savez_compressed(os.path.join(OUT, f"{sess}.npz"), **res)
    od = os.environ.get("EXP_OUTPUT_DIR", ".")
    json.dump({"session": sess, "ok": True,
               "e1": list(res["feats_e1"].shape), "e2": list(res["feats_e2"].shape),
               "fps": FPS, "size": SIZE, "model": MODEL},
              open(os.path.join(od, "result.json"), "w"))
    print("DONE", sess, flush=True)


if __name__ == "__main__":
    main()
