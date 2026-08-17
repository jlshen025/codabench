"""Frozen per-2s-segment clip features for one video foundation model and one view.

Used for all three frozen backbones of the entry (the model id is the only difference):

  MCG-NJU/videomae-large-finetuned-kinetics    -> feats/videomae_large      (hidden 1024)
  MCG-NJU/videomae-base-finetuned-ssv2         -> feats/videomae_ssv2       (hidden  768)
  facebook/timesformer-hr-finetuned-ssv2       -> feats/timesformer_ssv2    (hidden  768)

Per segment and view, `frames` frames are sampled uniformly from the segment's time span,
passed through the model's own image processor, and the tokens of the last hidden state are
mean-pooled into one vector. Output: <out>/<sid>_<view>.npz with `seg_keys` (array of 's_XXXX')
and `feats` [n_seg, hidden]. The two views are fused later, in udiva.models_video.load_feats.

For annotated sessions the segment grid comes from the parsed annotations; for the evaluation
sessions it is the fixed 2 s duration grid. GPU job (SLURM); `--shard/--nshard` splits the
session list across an array job.
"""
import os, sys, argparse, time
import numpy as np


def seg_grid_annotated(sid):
    from udiva import data as D
    g = D.gt_session(sid)["verbal"]
    return [(sk, g[sk]["t_b"], g[sk]["t_e"]) for sk in sorted(g.keys())]


def seg_grid_duration(dur, seg_len=2.0):
    n = int(np.ceil(dur / seg_len))
    return [(f"s_{k+1:04d}", seg_len * k, min(seg_len * (k + 1), dur)) for k in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sids", required=True, help="comma list, 'all' (annotated) or 'eval'")
    ap.add_argument("--views", default="E1,E2")
    ap.add_argument("--split", default="annotated", choices=["annotated", "unannotated", "eval"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="HuggingFace model id, see the module docstring")
    ap.add_argument("--frames", type=int, default=16, help="MUST match model.config.num_frames")
    ap.add_argument("--width", type=int, default=384, help="decode width (the processor resizes)")
    ap.add_argument("--height", type=int, default=216)
    ap.add_argument("--batch", type=int, default=16, help="clips per forward pass")
    ap.add_argument("--decode_chunk", type=int, default=16, help="segments decoded per get_batch")
    ap.add_argument("--limit", type=int, default=0, help="cap #segments per video (0=all; smoke)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch, decord
    from transformers import AutoImageProcessor, AutoModel
    from udiva import data as D
    decord.bridge.set_bridge("native")

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(args.model, cache_dir=os.environ.get("HF_HUB_CACHE"))
    model = AutoModel.from_pretrained(args.model,
                                      cache_dir=os.environ.get("HF_HUB_CACHE")).eval().to(dev)
    if dev == "cuda":
        model = model.half()
    hidden = model.config.hidden_size
    print(f"model {args.model} hidden={hidden} frames={args.frames} device={dev}", flush=True)

    sids = D.all_sids() if args.sids == "all" else (
        D.eval_sids() if args.sids == "eval" else args.sids.split(","))
    if args.nshard > 1:
        sids = sids[args.shard::args.nshard]
        print(f"shard {args.shard}/{args.nshard}: {len(sids)} sessions: {sids}")
    views = args.views.split(",")
    annotated = args.split == "annotated"
    F = args.frames

    for sid in sids:
        for view in views:
            outp = os.path.join(args.out, f"{sid}_{view}.npz")
            if os.path.exists(outp):
                print("skip exists", outp)
                continue
            vp = D.video_path(sid, view, annotated=annotated, split=args.split)
            if not os.path.exists(vp):
                print("MISSING", vp)
                continue
            t0 = time.time()
            vr = decord.VideoReader(vp, num_threads=4, width=args.width, height=args.height)
            fps = vr.get_avg_fps()
            nfr = len(vr)
            dur = nfr / fps
            segs = seg_grid_annotated(sid) if annotated else seg_grid_duration(dur)
            if args.limit:
                segs = segs[:args.limit]

            # frame indices per segment: F frames spread uniformly over [t_b, t_e]
            seg_idx = []
            for sk, tb, te in segs:
                a, b = int(tb * fps), min(int(te * fps), nfr - 1)
                b = max(b, a)
                seg_idx.append(np.linspace(a, b, F).astype(int))

            feats = np.zeros((len(segs), hidden), dtype=np.float32)
            buf_clips = []
            buf_pos = []

            def flush():
                nonlocal buf_clips, buf_pos
                if not buf_clips:
                    return
                for i in range(0, len(buf_clips), args.batch):
                    chunk = buf_clips[i:i + args.batch]
                    inp = proc([list(c) for c in chunk], return_tensors="pt")
                    pv = inp["pixel_values"].to(dev)
                    if dev == "cuda":
                        pv = pv.half()
                    with torch.no_grad():
                        out = model(pixel_values=pv).last_hidden_state.float().mean(1).cpu().numpy()
                    for j, p in enumerate(buf_pos[i:i + args.batch]):
                        feats[p] = out[j]
                buf_clips = []
                buf_pos = []

            for ci in range(0, len(segs), args.decode_chunk):
                group = list(range(ci, min(ci + args.decode_chunk, len(segs))))
                flat = np.concatenate([seg_idx[p] for p in group])
                frames = vr.get_batch(flat).asnumpy()          # [len(group)*F, H, W, 3]
                for gi, p in enumerate(group):
                    buf_clips.append(frames[gi * F:(gi + 1) * F])
                    buf_pos.append(p)
                if len(buf_clips) >= args.batch * 2:
                    flush()
            flush()

            np.savez_compressed(outp, seg_keys=np.array([s[0] for s in segs]), feats=feats)
            print(f"{sid} {view}: {len(segs)} segs -> {outp}  "
                  f"({time.time()-t0:.1f}s, {dur:.0f}s video)", flush=True)


if __name__ == "__main__":
    main()
