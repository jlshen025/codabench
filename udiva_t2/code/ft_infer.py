"""Test-time inference of the fine-tuned fourth component.

Loads the checkpoint trained by `ft_video.py --train_all --save_model`, runs it on the E1+E2
clips of the given sessions over the fixed 2 s duration grid (no ground truth needed), and
saves the per-segment (subject, high-level action) probabilities as an npz
(classes | seg_index | probs). predict_submission.py --ft_oof fuses that file into the
non-verbal ensemble.

Decodes 16 frames per segment and view at 224 px on the fly (no disk clip cache). FT_MODEL must
name the backbone the checkpoint was trained with (the default of ft_video.py). GPU job."""
import os, sys, argparse, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NFRAMES = 16
RES = 224


def seg_grid_duration(dur, seg_len=2.0):
    n = int(np.ceil(dur / seg_len))
    return [(f"s_{k+1:04d}", seg_len * k, min(seg_len * (k + 1), dur)) for k in range(n)]


def norm_clip(frames_uint8):
    # frames_uint8: [16,RES,RES,3] uint8 -> torch [16,3,RES,RES] float
    import torch
    c = frames_uint8.astype(np.float32) / 255.0
    c = (c - MEAN) / STD
    c = np.transpose(c, (0, 3, 1, 2))
    return torch.from_numpy(np.ascontiguousarray(c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_pt", required=True)
    ap.add_argument("--sids", default="eval", help="'eval', 'all', or comma list")
    ap.add_argument("--split", default="eval", choices=["annotated", "unannotated", "eval"])
    ap.add_argument("--out", required=True, help="output npz path")
    ap.add_argument("--batch", type=int, default=8, help="segments/batch (2x clips through backbone)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch, decord
    from ft_video import FTVideo
    from udiva import data as D
    decord.bridge.set_bridge("native")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model_pt, map_location="cpu")
    labels = [tuple(c.split("|")) for c in ckpt["labels"]]
    print(f"loaded {args.model_pt}: {len(labels)} classes, model_name={ckpt.get('model_name')} "
          f"best_ep={ckpt.get('best_epoch')} FT_MODEL={os.environ.get('FT_MODEL')}", flush=True)
    model = FTVideo(len(labels), n_unfreeze=ckpt.get("n_unfreeze", 2), cache_dir=os.environ.get("HF_HUB_CACHE"))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(dev)
    if dev == "cuda":
        model = model.half()

    if args.sids == "eval":
        sids = D.eval_sids()
    elif args.sids == "all":
        sids = D.all_sids()
    else:
        sids = args.sids.split(",")

    seg_index = []
    prob_rows = []
    for sid in sids:
        t0 = time.time()
        vr = {}
        for v in ("E1", "E2"):
            vp = D.video_path(sid, v, split=args.split)
            vr[v] = decord.VideoReader(vp, num_threads=4, width=RES, height=RES)
        fps = vr["E1"].get_avg_fps(); nfr = min(len(vr["E1"]), len(vr["E2"])); dur = nfr / fps
        segs = seg_grid_duration(dur)
        if args.limit:
            segs = segs[:args.limit]
        # frame indices per segment (shared across views)
        seg_idx = []
        for sk, tb, te in segs:
            a, b = int(tb * fps), min(int(te * fps), nfr - 1)
            b = max(b, a)
            seg_idx.append(np.linspace(a, b, NFRAMES).astype(int))
        # process in batches of segments
        for i in range(0, len(segs), args.batch):
            grp = list(range(i, min(i + args.batch, len(segs))))
            flat = np.concatenate([seg_idx[p] for p in grp])
            e1f = vr["E1"].get_batch(flat).asnumpy(); e2f = vr["E2"].get_batch(flat).asnumpy()
            e1 = torch.stack([norm_clip(e1f[j * NFRAMES:(j + 1) * NFRAMES]) for j in range(len(grp))])
            e2 = torch.stack([norm_clip(e2f[j * NFRAMES:(j + 1) * NFRAMES]) for j in range(len(grp))])
            e1 = e1.to(dev); e2 = e2.to(dev)
            if dev == "cuda":
                e1 = e1.half(); e2 = e2.half()
            with torch.no_grad():
                logits = model(e1, e2)
                p = torch.sigmoid(logits).float().cpu().numpy()
            for j, gp in enumerate(grp):
                seg_index.append(f"{sid}|{segs[gp][0]}")
                prob_rows.append(p[j])
        print(f"{sid}: {len(segs)} segs ({time.time()-t0:.1f}s, {dur:.0f}s vid)", flush=True)

    P = np.asarray(prob_rows, np.float32)
    classes = np.array(["|".join(c) for c in labels])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, classes=classes, seg_index=np.array(seg_index), probs=P)
    print(f"wrote {args.out}: probs {P.shape}, {len(seg_index)} segments", flush=True)


if __name__ == "__main__":
    main()
