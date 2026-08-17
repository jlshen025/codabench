"""Fine-tune of the fourth component of the non-verbal ensemble.

The backbone is set by the FT_MODEL environment variable; the entry used
MCG-NJU/videomae-large-finetuned-kinetics. The fine-tune is PARTIAL: the last `n_unfreeze`
encoder blocks, the final LayerNorm and a new multi-label head are trained (25.26M of 303.92M
parameters with the default n_unfreeze=2), everything else stays frozen. No LoRA / PEFT is
used anywhere in this entry.

Objective: multi-label BCE (positive weights clipped to [1,20]) over the (subject,
high_level_action) classes present >=3 times in TRAIN -- the same 64 joint classes the frozen
logistic heads use. E1 and E2 are fused INSIDE the network by averaging the two views'
mean-pooled last_hidden_state, before the shared head (the frozen backbones instead concatenate
the two view vectors; see udiva/models_video.py).

Two modes:
  --fold i --k 7   train on 6 folds, predict the held-out one -> ft_fold<i>.npz, the leak-free
                   out-of-fold probabilities consumed by run_ft_cv.py.
  --train_all      train on all 21 annotated sessions for a fixed epoch budget (11 for the
                   entry; val==train, monitoring only) -> --save_model checkpoint, which
                   ft_infer.py then runs on the evaluation videos.

VideoMAE requires 16 frames (position embeddings are fixed at 1568 tokens), so 16 frames per
segment are decoded ONCE into a per-(session, view) uint8 cache [n_seg,16,224,224,3] under
FTCACHE and reused; augmentation is therefore a +-10% brightness/gain jitter only (no temporal
jitter, no horizontal flips -- flipping would swap left/right participant semantics).

Self-contained apart from the project's data conventions. Run --smoke on a login CPU first
(forward/backward + caching on a few segments), then one GPU job.
"""
import os, sys, json, time, argparse, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FTCACHE = "<scratch>/t2/ftcache"
FEATS_OUT = "<scratch>/t2/feats"
# Backbone of the entry; override with FT_MODEL to fine-tune a different VideoMAE checkpoint.
MODEL = os.environ.get("FT_MODEL", "MCG-NJU/videomae-large-finetuned-kinetics")
NFRAMES = 16          # fixed by VideoMAE's position embeddings (1568 tokens)
RES = 224
VIEWS = ("E1", "E2")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ----------------------------------------------------------------------------- segments / labels
def seg_grid(sid):
    """Nonverbal 2s-segment grid for an annotated session: list of (segkey, t_b, t_e)."""
    from udiva import data as D
    g = D.gt_session(sid)["nonverbal"]
    return [(sk, g[sk]["t_b"], g[sk]["t_e"]) for sk in sorted(g.keys())]


def seg_labels(sid):
    """segkey -> set of (subject, highlevel_action) tuples present in that segment."""
    from udiva import data as D
    g = D.gt_session(sid)["nonverbal"]
    out = {}
    for sk in sorted(g.keys()):
        out[sk] = set((e["subject"], e["highlevel_action"]) for e in g[sk]["events"])
    return out


def build_label_space(train_sids, min_count=3):
    """List of (subject, highlevel_action) classes appearing >= min_count times across TRAIN."""
    from collections import Counter
    cnt = Counter()
    for sid in train_sids:
        for sk, labs in seg_labels(sid).items():
            for v in labs:
                cnt[v] += 1
    labels = sorted([v for v, n in cnt.items() if n >= min_count])
    return labels


# ----------------------------------------------------------------------------- frame caching
def cache_path(sid, view):
    return os.path.join(FTCACHE, f"{sid}_{view}.npy")


def cache_session_view(sid, view, limit=0, verbose=True):
    """Decode 16 frames/segment for (sid,view) -> uint8 .npy [n_seg,16,224,224,3]. Skip if exists.
    Returns the np.memmap (read-only) of the cache."""
    import decord
    from udiva import data as D
    decord.bridge.set_bridge("native")
    outp = cache_path(sid, view)
    segs = seg_grid(sid)
    if limit:
        segs = segs[:limit]
    n = len(segs)
    if os.path.exists(outp):
        arr = np.load(outp, mmap_mode="r")
        if arr.shape[0] >= n:
            if verbose:
                print(f"cache skip {sid} {view} ({arr.shape[0]} segs)", flush=True)
            return arr
        else:
            if verbose:
                print(f"cache stale {sid} {view} ({arr.shape[0]}<{n}); rebuild", flush=True)
    vp = D.video_path(sid, view, annotated=True)
    if not os.path.exists(vp):
        raise FileNotFoundError(vp)
    t0 = time.time()
    vr = decord.VideoReader(vp, num_threads=4, width=RES, height=RES)
    fps = vr.get_avg_fps()
    nfr = len(vr)
    seg_idx = []
    for sk, tb, te in segs:
        a, b = int(tb * fps), min(int(te * fps), nfr - 1)
        b = max(b, a)
        seg_idx.append(np.linspace(a, b, NFRAMES).astype(int))
    os.makedirs(FTCACHE, exist_ok=True)
    tmp = outp + ".tmp.npy"
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8,
                                    shape=(n, NFRAMES, RES, RES, 3))
    CHUNK = 16  # segments per decord get_batch
    for ci in range(0, n, CHUNK):
        grp = list(range(ci, min(ci + CHUNK, n)))
        flat = np.concatenate([seg_idx[p] for p in grp])
        frames = vr.get_batch(flat).asnumpy()  # [len*NFRAMES, RES, RES, 3] uint8
        for gi, p in enumerate(grp):
            out[p] = frames[gi * NFRAMES:(gi + 1) * NFRAMES]
    out.flush()
    del out
    os.replace(tmp, outp)
    if verbose:
        print(f"cache build {sid} {view}: {n} segs -> {outp} ({time.time()-t0:.1f}s)", flush=True)
    return np.load(outp, mmap_mode="r")


def ensure_cache(sids, limit=0, verbose=True):
    for sid in sids:
        for v in VIEWS:
            cache_session_view(sid, v, limit=limit, verbose=verbose)


# ----------------------------------------------------------------------------- dataset
import torch
from torch.utils.data import Dataset, DataLoader


class SegDataset(Dataset):
    """One item = one segment = the E1 clip + E2 clip (both [16,3,224,224] normalized) + label vec.
    Frames loaded from per-(sid,view) uint8 memmaps; normalized to VideoMAE mean/std here.
    `jitter` applies a small brightness/gain scale (train only); frames come from the fixed
    per-segment cache, so there is no temporal jitter, and no horizontal flip is ever applied."""

    def __init__(self, sids, labels, limit=0, jitter=False):
        self.labels_space = labels
        self.li = {v: i for i, v in enumerate(labels)}
        self.jitter = jitter
        self.items = []          # (sid, seg_position, segkey)
        self.mm = {}             # (sid,view) -> memmap
        self.Y = []
        for sid in sids:
            for v in VIEWS:
                self.mm[(sid, v)] = np.load(cache_path(sid, v), mmap_mode="r")
            labs = seg_labels(sid)
            segs = seg_grid(sid)
            if limit:
                segs = segs[:limit]
            n_cached = min(self.mm[(sid, "E1")].shape[0], self.mm[(sid, "E2")].shape[0])
            for pos, (sk, tb, te) in enumerate(segs):
                if pos >= n_cached:
                    break
                y = np.zeros(len(labels), np.float32)
                for t in labs.get(sk, ()):
                    if t in self.li:
                        y[self.li[t]] = 1.0
                self.items.append((sid, pos, sk))
                self.Y.append(y)
        self.Y = np.asarray(self.Y, np.float32)

    def __len__(self):
        return len(self.items)

    def _load_view(self, sid, view, pos):
        clip = np.asarray(self.mm[(sid, view)][pos]).astype(np.float32) / 255.0  # [16,224,224,3]
        if self.jitter:
            # +-10% brightness/gain jitter -- the only augmentation used
            g = 1.0 + np.random.uniform(-0.1, 0.1)
            clip = np.clip(clip * g, 0.0, 1.0)
        clip = (clip - MEAN) / STD
        clip = np.transpose(clip, (0, 3, 1, 2))  # [16,3,224,224]
        return torch.from_numpy(np.ascontiguousarray(clip))

    def __getitem__(self, i):
        sid, pos, sk = self.items[i]
        e1 = self._load_view(sid, "E1", pos)
        e2 = self._load_view(sid, "E2", pos)
        y = torch.from_numpy(self.Y[i])
        return e1, e2, y


# ----------------------------------------------------------------------------- model
class FTVideo(torch.nn.Module):
    """VideoMAE backbone (mostly frozen) -> mean-pool tokens per view -> average E1/E2 -> head."""

    def __init__(self, n_labels, n_unfreeze=2, dropout=0.5, cache_dir=None):
        super().__init__()
        from transformers import VideoMAEModel
        self.backbone = VideoMAEModel.from_pretrained(MODEL, cache_dir=cache_dir)
        hid = self.backbone.config.hidden_size
        n_layers = self.backbone.config.num_hidden_layers

        # freeze everything, then re-enable the last n_unfreeze blocks + the final LayerNorm
        for p in self.backbone.parameters():
            p.requires_grad = False
        for i in range(n_layers - n_unfreeze, n_layers):
            for p in self.backbone.encoder.layer[i].parameters():
                p.requires_grad = True
        if getattr(self.backbone, "layernorm", None) is not None:
            for p in self.backbone.layernorm.parameters():
                p.requires_grad = True

        self.drop = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hid, n_labels)

    def _encode(self, x):
        # x: [B,16,3,224,224] -> mean-pooled tokens [B,hid]
        return self.backbone(pixel_values=x).last_hidden_state.mean(dim=1)

    def forward(self, e1, e2):
        f1 = self._encode(e1)
        f2 = self._encode(e2)
        f = 0.5 * (f1 + f2)
        return self.head(self.drop(f))

    def trainable_param_groups(self, lr_backbone, lr_head):
        bb, hd = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (hd if n.startswith("head") else bb).append(p)
        groups = []
        if bb:
            groups.append({"params": bb, "lr": lr_backbone})
        if hd:
            groups.append({"params": hd, "lr": lr_head})
        return groups


# ----------------------------------------------------------------------------- eval
def macro_ap(y_true, y_score):
    """Diag-comparable macro-AP: per-class average_precision_score over classes with >=1 positive
    in BOTH the (implicit) train fit and the val set; here we average over val classes with a
    positive in val (matches diag_feats2). Returns (macro_ap, n_classes_scored)."""
    from sklearn.metrics import average_precision_score
    aps = []
    for j in range(y_true.shape[1]):
        if y_true[:, j].sum() == 0:
            continue
        aps.append(average_precision_score(y_true[:, j], y_score[:, j]))
    return (float(np.mean(aps)) if aps else 0.0), len(aps)


@torch.no_grad()
def evaluate(model, loader, dev, amp):
    model.eval()
    ys, ps = [], []
    for e1, e2, y in loader:
        e1 = e1.to(dev, non_blocking=True)
        e2 = e2.to(dev, non_blocking=True)
        with torch.autocast(device_type=dev.split(":")[0], dtype=torch.float16, enabled=amp):
            logits = model(e1, e2)
        ps.append(torch.sigmoid(logits).float().cpu().numpy())
        ys.append(y.numpy())
    Y = np.concatenate(ys)
    P = np.concatenate(ps)
    m, nc = macro_ap(Y, P)
    return m, nc, Y, P


# ----------------------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="CPU smoke: 1 session, few segments, 2 optim steps; confirms fwd/bwd+cache.")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8, help="segments per batch (=2x clips through backbone)")
    ap.add_argument("--lr_backbone", type=float, default=2e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--n_unfreeze", type=int, default=2)
    ap.add_argument("--patience", type=int, default=4, help="early stop on val macro-AP")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=-1, help="if>=0, CV fold (cv.make_folds k=K seed=0) as val; rest train")
    ap.add_argument("--k", type=int, default=7, help="number of CV folds")
    ap.add_argument("--limit", type=int, default=0, help="cap segments/session (smoke/debug)")
    ap.add_argument("--out_npz", default=os.path.join(FEATS_OUT, "ft_proof_val.npz"))
    ap.add_argument("--train_all", action="store_true",
                    help="train on ALL sids (val=train, monitor only, no early stop) — produces a TEST-time model")
    ap.add_argument("--save_model", default="", help="path to save best model state_dict + labels (TEST inference)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from udiva import data as D
    cache_dir = os.environ.get("HF_HUB_CACHE")

    sids = D.all_sids()
    if args.smoke:
        train_sids = sids[:1]
        val_sids = sids[:1]
        args.epochs = 1
        args.limit = args.limit or 6
        args.batch = min(args.batch, 3)
        args.workers = 0
        n_steps_cap = 2
    elif args.fold >= 0:
        from udiva.cv import make_folds
        folds = make_folds(sids, k=args.k, seed=0)
        val_sids = folds[args.fold]
        train_sids = [s for s in sids if s not in val_sids]
        n_steps_cap = 0
        print(f"FOLD {args.fold}/{args.k}: val={val_sids}", flush=True)
    elif args.train_all:
        train_sids = sids
        val_sids = sids  # monitoring only (val==train ⇒ no real early stop); trains full epochs
        n_steps_cap = 0
        print(f"TRAIN-ALL: {len(sids)} sids → TEST-time model (val=train, monitor only)", flush=True)
    else:
        train_sids = sids[:18]
        val_sids = sids[18:]
        n_steps_cap = 0

    print(f"train={train_sids if args.smoke else len(train_sids)} "
          f"val={val_sids}  epochs={args.epochs} batch={args.batch} "
          f"lr_bb={args.lr_backbone} lr_head={args.lr_head} unfreeze={args.n_unfreeze} "
          f"limit={args.limit} amp={not args.no_amp}", flush=True)

    # --- labels from train
    labels = build_label_space(train_sids, min_count=3)
    print(f"label space: {len(labels)} (subject,highlevel_action) classes (>=3 in train)", flush=True)

    # --- cache frames (decode once)
    t0 = time.time()
    ensure_cache(sorted(set(train_sids) | set(val_sids)), limit=args.limit)
    print(f"cache ready ({time.time()-t0:.1f}s)", flush=True)

    # --- datasets
    ds_tr = SegDataset(train_sids, labels, limit=args.limit, jitter=not args.smoke)
    ds_va = SegDataset(val_sids, labels, limit=args.limit, jitter=False)
    print(f"train segments={len(ds_tr)}  val segments={len(ds_va)}  "
          f"train pos/class mean={ds_tr.Y.sum(0).mean():.1f}", flush=True)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       pin_memory=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=max(args.batch, 4), shuffle=False, num_workers=args.workers,
                       pin_memory=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = (not args.no_amp) and dev == "cuda"
    model = FTVideo(len(labels), n_unfreeze=args.n_unfreeze, dropout=args.dropout,
                    cache_dir=cache_dir).to(dev)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"device={dev} amp={amp}  trainable params={n_train/1e6:.2f}M / {n_total/1e6:.2f}M", flush=True)

    # --- class-balanced BCE pos_weight (fight imbalance over the large action vocab)
    pos = ds_tr.Y.sum(0)
    neg = len(ds_tr) - pos
    pw = np.clip(neg / np.maximum(pos, 1.0), 1.0, 20.0).astype(np.float32)
    pos_weight = torch.from_numpy(pw).to(dev)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.trainable_param_groups(args.lr_backbone, args.lr_head),
                            weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(dl_tr))
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups],
        total_steps=total_steps, pct_start=0.2) if not args.smoke else None
    scaler = torch.amp.GradScaler(enabled=amp)

    BASELINE = 0.22
    best = -1.0
    best_epoch = -1
    best_Y = best_P = None
    best_state = None
    bad = 0
    gstep = 0
    for ep in range(args.epochs):
        model.train()
        running = 0.0
        nb = 0
        te = time.time()
        for e1, e2, y in dl_tr:
            e1 = e1.to(dev, non_blocking=True)
            e2 = e2.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.split(":")[0], dtype=torch.float16, enabled=amp):
                logits = model(e1, e2)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()
            running += float(loss.item())
            nb += 1
            gstep += 1
            if n_steps_cap and gstep >= n_steps_cap:
                break
        tr_loss = running / max(nb, 1)
        va_map, va_nc, Y, P = evaluate(model, dl_va, dev, amp)
        lr_now = opt.param_groups[0]["lr"]
        print(f"epoch {ep:02d}  train_loss={tr_loss:.4f}  val_macroAP={va_map:.4f} "
              f"(n_cls={va_nc})  vs frozen {BASELINE:.2f}  lr={lr_now:.2e}  "
              f"({time.time()-te:.1f}s)", flush=True)
        if va_map > best:
            best = va_map
            best_epoch = ep
            best_Y, best_P = Y, P
            if args.save_model:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if not args.smoke and bad >= args.patience:
                print(f"early stop at epoch {ep} (no val gain for {bad})", flush=True)
                break
        if n_steps_cap and gstep >= n_steps_cap:
            print("smoke step cap reached", flush=True)
            break

    delta = best - BASELINE
    verdict = "BEATS" if delta > 0 else "does NOT beat"
    print(f"\n=== BEST val macro-AP = {best:.4f} @ epoch {best_epoch}  "
          f"| frozen baseline {BASELINE:.2f}  | delta {delta:+.4f}  -> {verdict} frozen ===",
          flush=True)

    # --- save best per-segment val probabilities + a result json
    if best_P is not None and not args.smoke:
        os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
        seg_index = np.array([f"{sid}|{sk}" for (sid, pos, sk) in ds_va.items])
        np.savez_compressed(args.out_npz, probs=best_P.astype(np.float32),
                            labels_true=best_Y.astype(np.int8),
                            classes=np.array(["|".join(c) for c in labels]),
                            seg_index=seg_index,
                            best_macro_ap=np.float32(best),
                            best_epoch=np.int32(best_epoch),
                            baseline=np.float32(BASELINE))
        print(f"saved best val probs -> {args.out_npz}", flush=True)
        out_dir = os.environ.get("EXP_OUTPUT_DIR", ".")
        res = {"best_val_macro_ap": best, "best_epoch": best_epoch,
               "frozen_baseline": BASELINE, "delta": delta, "beats_frozen": bool(delta > 0),
               "n_labels": len(labels), "val_sids": val_sids,
               "n_unfreeze": args.n_unfreeze,
               "epochs_run": ep + 1, "batch": args.batch,
               "lr_backbone": args.lr_backbone, "lr_head": args.lr_head}
        with open(os.path.join(out_dir, "ft_proof_result.json"), "w") as f:
            json.dump(res, f, indent=2)
        print(f"saved result json -> {os.path.join(out_dir, 'ft_proof_result.json')}", flush=True)

    if args.save_model and best_state is not None:
        torch.save({"state_dict": best_state, "labels": ["|".join(c) for c in labels],
                    "model_name": MODEL, "n_unfreeze": args.n_unfreeze,
                    "best_epoch": int(best_epoch), "best_macro_ap": float(best)}, args.save_model)
        print(f"saved TEST-time model -> {args.save_model} (best ep {best_epoch}, macroAP {best:.4f})", flush=True)

    if args.smoke:
        print("SMOKE OK: forward/backward + caching ran without error.", flush=True)


if __name__ == "__main__":
    main()
