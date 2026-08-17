#!/usr/bin/env python3
"""Extract VideoMAE clip features per 2s segment from exocentric GF video (SHIPPED, stage 1).
Subject attribution via 3 horizontal crops of every frame: left(participant_a),
right(participant_b), full(shared context). One VideoMAE forward per (segment, crop),
token-mean-pooled.
Output per session: <out>/<sid>.npz  {feats:[n_seg,3,hidden] f16, segs:[n_seg], tb, te}.
Resumable: skips sessions whose npz already exists.

The shipped submission (821839) used VideoMAE-LARGE (hidden=1024); the default below matches
it. Pass --model to override (base=768 was scouted, did not win).
"""
import os, sys, argparse, time, traceback
os.environ.setdefault("HF_HOME", "<cache>/huggingface")
import numpy as np
import torch
import decord
from transformers import VideoMAEImageProcessor, VideoMAEModel
from udiva import build_gt, seg_grid, SESSIONS

GF = "<datasets>/UDIVA-HHOI/development/annotated_sessions/audiovisual/exo/GF"
MID = "MCG-NJU/videomae-large-finetuned-kinetics"
NF = 16
CROPS = {"left": (0.0, 0.60), "right": (0.40, 1.0), "full": (0.0, 1.0)}  # (x0,x1) fractions

def sample_idx(tb, te, fps, n_total, nf=NF):
    a, b = tb * fps, te * fps
    idx = np.linspace(a, b - 1, nf)
    return np.clip(np.round(idx).astype(int), 0, n_total - 1)

def run(sessions, out, gf_dir=GF, batch=12, device="cuda", limit=0, mid=MID, eval_mode=False):
    os.makedirs(out, exist_ok=True)
    proc = VideoMAEImageProcessor.from_pretrained(mid)
    model = VideoMAEModel.from_pretrained(mid).to(device).eval().half()
    maxmem = 0
    for sid in sessions:
        op = f"{out}/{sid}.npz"
        if os.path.exists(op):
            print(f"[skip] {sid}", flush=True); continue
        t0 = time.time()
        try:
            vr = decord.VideoReader(f"{gf_dir}/{sid}.mp4", ctx=decord.cpu(0), num_threads=4)
            fps = vr.get_avg_fps(); n = len(vr); H, W, _ = vr[0].shape
            grid = seg_grid(n / fps) if eval_mode else \
                [(sk, tb, te) for sk, (tb, te, s) in build_gt(sid)["verbal"].items()]
            if limit: grid = grid[:limit]
            segs = [g[0] for g in grid]
            # build all (seg, crop) clip tensors lazily, run in batches
            clip_specs = []  # (seg_i, crop_i, frame_idx_array)
            for si, (sk, tb, te) in enumerate(grid):
                fi = sample_idx(tb, te, fps, n)
                for ci in range(len(CROPS)):
                    clip_specs.append((si, ci, fi))
            feats = np.zeros((len(grid), len(CROPS), model.config.hidden_size), np.float16)
            crop_names = list(CROPS)
            buf_clips, buf_keys = [], []
            def flush():
                if not buf_clips: return
                # buf_clips: list of [NF,224,224,3] uint8 -> processor
                pv = proc(buf_clips, return_tensors="pt")["pixel_values"].to(device).half()
                with torch.no_grad():
                    h = model(pv).last_hidden_state.mean(1).float().cpu().numpy()
                for (si, ci), v in zip(buf_keys, h):
                    feats[si, ci] = v.astype(np.float16)
                buf_clips.clear(); buf_keys.clear()
            for (si, ci, fi) in clip_specs:
                frames = vr.get_batch(fi).asnumpy()  # [NF,H,W,3]
                x0, x1 = CROPS[crop_names[ci]]
                cl = frames[:, :, int(x0 * W):int(x1 * W), :]
                buf_clips.append([cl[k] for k in range(cl.shape[0])])
                buf_keys.append((si, ci))
                if len(buf_clips) >= batch:
                    flush()
            flush()
            if device == "cuda":
                maxmem = max(maxmem, torch.cuda.max_memory_allocated() / 1e9)
            np.savez_compressed(op, feats=feats, segs=np.array(segs),
                                tb=np.array([g[1] for g in grid]), te=np.array([g[2] for g in grid]))
            print(f"[done] {sid} segs={len(grid)} {time.time()-t0:.1f}s feats={feats.shape} maxmem={maxmem:.2f}GB", flush=True)
        except Exception:
            print(f"[FAIL] {sid}\n{traceback.format_exc()}", flush=True)
    # write a completion marker for the caller
    res = {"sessions_done": sum(1 for s in sessions if os.path.exists(f'{out}/{s}.npz')),
           "n_requested": len(sessions), "maxmem_gb": maxmem, "model": MID, "crops": list(CROPS)}
    import json
    json.dump(res, open(f"{out}/_extract_summary.json", "w"), indent=1)
    print("SUMMARY", res, flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="all")
    ap.add_argument("--out", default="<scratch>/t1/feats")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=MID)
    ap.add_argument("--gf_dir", default=GF)
    ap.add_argument("--eval", action="store_true", help="eval mode: grid from video duration (no annotations)")
    a = ap.parse_args()
    import glob as _glob
    sess = (SESSIONS if a.sessions == "all"
            else sorted(os.path.basename(p)[:-4] for p in _glob.glob(f"{a.gf_dir}/*.mp4")) if a.sessions == "gfdir"
            else a.sessions.split(","))
    run(sess, a.out, gf_dir=a.gf_dir, batch=a.batch, device="cpu" if a.cpu else "cuda",
        limit=a.limit, mid=a.model, eval_mode=a.eval)
