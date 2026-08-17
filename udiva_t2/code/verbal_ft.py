"""ABLATION -- NOT part of the submitted entry.

Fine-tunes a multilingual LM (xlm-roberta-base) on per-cue utterance-type recognition and
compares it with the frozen TF-IDF baseline of the verbal channel (0.163 out-of-fold macro-AP;
frozen multilingual MiniLM embeddings reach 0.151). It improved utterance-type macro-AP to
about 0.22 in cross-validation, but it was NOT wired into predict_submission.py: the verbal
channel of the ranked entry is the TF-IDF model in udiva/models_text.py. This script is kept
because the fact sheet reports the ablation.

One example = one PARTICIPANT cue, built exactly as the project builds them
(udiva.models_text._align_cue_events / _part_cues / VerbalCueModel._text):
    input text  = "[A]"/"[B] " + previous-cue-text + " </s> " + this-cue-text
    label-set   = utterance_type strings of the ground-truth verbal events aligned to that cue
Multi-label head (BCEWithLogitsLoss) over the utterance_type classes appearing >=3x in TRAIN.
Backbone: FacebookAI/xlm-roberta-base, fully fine-tuned (the model is small; no PEFT).

Run --smoke on a login CPU first (forward/backward on a few cues), then one GPU job. Weights
download on the LOGIN node (compute nodes have no web access) into the shared HF cache.
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import os as _os
MODEL = _os.environ.get("VFT_MODEL", "FacebookAI/xlm-roberta-base")
BASELINE = 0.163          # frozen TF-IDF OOF macro-AP for utterance_type (the bar to beat)
FEATS_OUT = "<scratch>/t2/feats"


# ----------------------------------------------------------------------------- rows / labels
def build_rows(sids):
    """One row per PARTICIPANT cue, EXACTLY as udiva.models_text builds them.
    Returns list of (text, set_of_utterance_type). Uses _align_cue_events (cue -> aligned V
    events) + VerbalCueModel._text (speaker tag + prev-cue context + ' </s> ' + cue text)."""
    from udiva.models_text import _align_cue_events, VerbalCueModel
    txt = VerbalCueModel(ctx=True)._text   # bound method: (cue, prev) -> "[A] prev </s> this"
    attr = os.environ.get("VFT_ATTR", "utterance_type")  # "utterance_type" or "target"
    rows = []; keys = []
    for sid in sids:
        ace = _align_cue_events(sid)   # [(cue, [aligned V events]), ...] in cue order
        for i, (c, al) in enumerate(ace):
            prev = ace[i - 1][0] if i > 0 else None
            if attr == "target":
                lab = set(",".join(e["target_filtered"]) for e in al)
            else:
                lab = set(e["utterance_type"] for e in al)
            rows.append((txt(c, prev), lab))
            keys.append(f"{sid}|{c['start']:.3f}")
    return rows, keys


def build_label_space(train_rows, min_count=3):
    """utterance_type classes appearing >= min_count times across TRAIN cues."""
    from collections import Counter
    cnt = Counter(u for _, us in train_rows for u in us)
    return sorted([u for u, n in cnt.items() if n >= min_count])


def encode_labels(rows, labels):
    li = {u: i for i, u in enumerate(labels)}
    Y = np.zeros((len(rows), len(labels)), np.float32)
    for r, (_, us) in enumerate(rows):
        for u in us:
            j = li.get(u)
            if j is not None:
                Y[r, j] = 1.0
    return Y


# ----------------------------------------------------------------------------- dataset
import torch
from torch.utils.data import Dataset, DataLoader


class CueDataset(Dataset):
    """One item = one cue: pre-tokenized input_ids/attention_mask + multi-label target vec."""

    def __init__(self, rows, Y, tokenizer, max_len=128):
        texts = [t for t, _ in rows]
        enc = tokenizer(texts, truncation=True, max_length=max_len, padding="max_length",
                        return_tensors="np")
        self.input_ids = enc["input_ids"].astype(np.int64)
        self.attention_mask = enc["attention_mask"].astype(np.int64)
        self.Y = np.asarray(Y, np.float32)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return (torch.from_numpy(self.input_ids[i]),
                torch.from_numpy(self.attention_mask[i]),
                torch.from_numpy(self.Y[i]))


# ----------------------------------------------------------------------------- model
class VerbalFT(torch.nn.Module):
    """xlm-roberta-base (FULLY fine-tuned) -> CLS/<s> pooled -> dropout -> multi-label head."""

    def __init__(self, n_labels, dropout=0.2, cache_dir=None):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(MODEL, cache_dir=cache_dir)
        hid = self.backbone.config.hidden_size
        self.drop = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hid, n_labels)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # XLM-R has no pooler head; use the <s> (first) token of last_hidden_state.
        cls = out.last_hidden_state[:, 0]
        return self.head(self.drop(cls))


# ----------------------------------------------------------------------------- eval
def macro_ap(y_true, y_score):
    """Macro average_precision over classes with >=1 positive in the VAL set (matches the
    frozen-baseline protocol / ft_video.py). Returns (macro_ap, n_classes_scored)."""
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
    for ids, am, y in loader:
        ids = ids.to(dev, non_blocking=True)
        am = am.to(dev, non_blocking=True)
        with torch.autocast(device_type=dev.split(":")[0], dtype=torch.float16, enabled=amp):
            logits = model(ids, am)
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
                    help="CPU smoke: few cues, 1-2 optim steps; confirms fwd/bwd + loss decreases.")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--patience", type=int, default=3, help="early stop on val macro-AP")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_count", type=int, default=3, help="min TRAIN freq for a kept class")
    ap.add_argument("--fold", type=int, default=-1, help="if>=0, CV fold (cv.make_folds k=K seed=0) as val; rest train")
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--out_npz", default=os.path.join(FEATS_OUT, "vft_proof_val.npz"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from udiva import data as D
    cache_dir = os.environ.get("HF_HUB_CACHE")

    sids = D.all_sids()
    if args.smoke:
        train_sids = sids[:2]
        val_sids = sids[2:3]
        args.epochs = 1
        args.batch = min(args.batch, 4)
        args.workers = 0
        n_steps_cap = 2
        smoke_cues = 24      # cap rows so a CPU forward/backward is seconds, not minutes
    elif args.fold >= 0:
        from udiva.cv import make_folds
        val_sids = make_folds(sids, k=args.k, seed=0)[args.fold]
        train_sids = [s for s in sids if s not in val_sids]
        n_steps_cap = 0
        smoke_cues = 0
        print(f"FOLD {args.fold}/{args.k}: val={val_sids}", flush=True)
    else:
        train_sids = sids[:18]
        val_sids = sids[18:]
        n_steps_cap = 0
        smoke_cues = 0

    print(f"MODEL={MODEL}", flush=True)
    print(f"train={train_sids if args.smoke else len(train_sids)} sessions "
          f"val={val_sids}  epochs={args.epochs} batch={args.batch} lr={args.lr} "
          f"wd={args.weight_decay} dropout={args.dropout} max_len={args.max_len} "
          f"amp={not args.no_amp}", flush=True)

    # --- build cue rows (exact project conventions)
    train_rows, _ = build_rows(train_sids)
    val_rows, val_keys = build_rows(val_sids)
    if smoke_cues:
        train_rows = train_rows[:smoke_cues]
        val_rows = val_rows[:max(8, smoke_cues // 2)]
        val_keys = val_keys[:len(val_rows)]
    labels = build_label_space(train_rows, min_count=args.min_count)
    if not labels:                      # smoke: tiny subset may have no class >=min_count
        from collections import Counter
        cnt = Counter(u for _, us in train_rows for u in us)
        labels = sorted([u for u, _ in cnt.most_common(5)])
        print(f"[smoke] min_count yielded 0 classes; using top-{len(labels)} present instead",
              flush=True)
    Ytr = encode_labels(train_rows, labels)
    Yva = encode_labels(val_rows, labels)
    print(f"label space: {len(labels)} utterance_type classes (>={args.min_count} in train)  "
          f"| train cues={len(train_rows)} val cues={len(val_rows)}  "
          f"| train pos/class mean={Ytr.sum(0).mean():.1f}", flush=True)

    # --- tokenizer + datasets
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=cache_dir)
    ds_tr = CueDataset(train_rows, Ytr, tok, max_len=args.max_len)
    ds_va = CueDataset(val_rows, Yva, tok, max_len=args.max_len)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       pin_memory=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=max(args.batch, 8), shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = (not args.no_amp) and dev == "cuda"
    model = VerbalFT(len(labels), dropout=args.dropout, cache_dir=cache_dir).to(dev)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"device={dev} amp={amp}  trainable params={n_train/1e6:.2f}M / {n_total/1e6:.2f}M "
          f"(FULL fine-tune)", flush=True)

    # --- class-balanced BCE pos_weight (fight imbalance over the utterance_type vocab)
    pos = Ytr.sum(0)
    neg = len(ds_tr) - pos
    pw = np.clip(neg / np.maximum(pos, 1.0), 1.0, 20.0).astype(np.float32)
    pos_weight = torch.from_numpy(pw).to(dev)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(dl_tr))
    total_steps = steps_per_epoch * args.epochs
    sched = (torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps,
                                                 pct_start=0.1) if not args.smoke else None)
    scaler = torch.amp.GradScaler(enabled=amp)

    best = -1.0
    best_epoch = -1
    best_Y = best_P = None
    bad = 0
    gstep = 0
    smoke_losses = []
    for ep in range(args.epochs):
        model.train()
        running = 0.0
        nb = 0
        te = time.time()
        for ids, am, y in dl_tr:
            ids = ids.to(dev, non_blocking=True)
            am = am.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.split(":")[0], dtype=torch.float16, enabled=amp):
                logits = model(ids, am)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()
            running += float(loss.item())
            nb += 1
            gstep += 1
            if args.smoke:
                smoke_losses.append(float(loss.item()))
                print(f"  [smoke] step {gstep} loss={loss.item():.4f}", flush=True)
            if n_steps_cap and gstep >= n_steps_cap:
                break
        tr_loss = running / max(nb, 1)
        if not args.smoke:
            va_map, va_nc, Y, P = evaluate(model, dl_va, dev, amp)
            lr_now = opt.param_groups[0]["lr"]
            delta = va_map - BASELINE
            print(f"epoch {ep:02d}  train_loss={tr_loss:.4f}  val_macroAP={va_map:.4f} "
                  f"(n_cls={va_nc})  vs frozen-TFIDF {BASELINE:.3f} (delta {delta:+.4f})  "
                  f"lr={lr_now:.2e}  ({time.time()-te:.1f}s)", flush=True)
            if va_map > best:
                best = va_map
                best_epoch = ep
                best_Y, best_P = Y, P
                bad = 0
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"early stop at epoch {ep} (no val gain for {bad})", flush=True)
                    break
        if n_steps_cap and gstep >= n_steps_cap:
            print("smoke step cap reached", flush=True)
            break

    if args.smoke:
        ok = len(smoke_losses) >= 2 and np.isfinite(smoke_losses).all()
        trend = "decreasing" if (len(smoke_losses) >= 2 and smoke_losses[-1] < smoke_losses[0]) \
            else "not-strictly-down (ok over 1-2 steps)"
        print(f"\nSMOKE losses={['%.4f' % l for l in smoke_losses]}  ({trend})", flush=True)
        print("SMOKE OK: forward/backward ran without error." if ok else
              "SMOKE WARN: <2 steps or non-finite loss.", flush=True)
        return

    delta = best - BASELINE
    verdict = "BEATS" if delta > 0 else "does NOT beat"
    print(f"\n=== BEST val macro-AP = {best:.4f} @ epoch {best_epoch}  "
          f"| frozen-TFIDF baseline {BASELINE:.3f}  | delta {delta:+.4f}  "
          f"-> fine-tuning {verdict} frozen TF-IDF ===", flush=True)

    # --- save best per-cue val probabilities + a result json
    if best_P is not None:
        os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
        np.savez_compressed(args.out_npz, probs=best_P.astype(np.float32),
                            labels_true=best_Y.astype(np.int8),
                            classes=np.array(labels),
                            cue_keys=np.array(val_keys),
                            best_macro_ap=np.float32(best),
                            best_epoch=np.int32(best_epoch),
                            baseline=np.float32(BASELINE))
        print(f"saved best val probs -> {args.out_npz}", flush=True)
        out_dir = os.environ.get("EXP_OUTPUT_DIR", ".")
        res = {"best_val_macro_ap": best, "best_epoch": best_epoch,
               "frozen_tfidf_baseline": BASELINE, "delta": delta,
               "beats_frozen_tfidf": bool(delta > 0),
               "frozen_minilm_baseline": 0.151,
               "n_labels": len(labels), "val_sids": val_sids, "model": MODEL,
               "epochs_run": ep + 1, "batch": args.batch, "lr": args.lr,
               "weight_decay": args.weight_decay, "max_len": args.max_len,
               "dropout": args.dropout, "min_count": args.min_count}
        with open(os.path.join(out_dir, "vft_proof_result.json"), "w") as f:
            json.dump(res, f, indent=2)
        print(f"saved result json -> {os.path.join(out_dir, 'vft_proof_result.json')}", flush=True)


if __name__ == "__main__":
    main()
