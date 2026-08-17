"""DEVELOPMENT EXPERIMENT (not part of the submitted method).

Trained linear probe on frozen prefix features -> multi-label over the N_VOCAB most
frequent event tuples (the standard 'is there linearly-decodable signal' check, run after
k-NN retrieval failed). Per LOSO fold: a single torch Linear + BCEWithLogits, full-batch
Adam (lr 0.05, weight decay 1e-2, 150 steps). At prediction time the events whose sigmoid
exceeds `thr` (top `maxlen` of them) become alternatives 2-3, alternative 1 is the empty
sequence and the remaining slots are backfilled with the prior, so the probe can only add
to the K=5 hedge.

Feature source: 'video' (DINOv2, dev/feat_extract.py) or 'text' (sentence embeddings,
dev/txt_extract.py).  Usage: python dev/vidtrain.py [video|text]
"""
import numpy as np
import torch
from collections import Counter
import _path  # noqa: F401
import udiva_data as U
import sdl
import cv as CV
import predictors as P
import vidprobe as VP

PA, PB = "participant_a", "participant_b"
N_VOCAB = 40
POOL = "meanlast"


def build_vocab(gt, sids):
    c = Counter()
    for sid in sids:
        for seg_id, seg in gt[sid].items():
            g = sdl.ref_seg_to_gt(seg)
            for p in (PA, PB):
                for e in g[p]:
                    c[tuple(sdl.ev_key(e))] += 1
    vocab = [k for k, _ in c.most_common(N_VOCAB)]
    return vocab, {k: i for i, k in enumerate(vocab)}


def make_xy(feats, gt, sids, vidx):
    X, Y = [], []
    for sid in sids:
        if sid not in feats:
            continue
        F = feats[sid]
        for i, seg_id in enumerate(F["seg_ids"]):
            seg = gt[sid].get(seg_id)
            if seg is None:
                continue
            g = sdl.ref_seg_to_gt(seg)
            for p, key in ((PA, PA), (PB, PB)):
                X.append(VP.pool(F[key][i], POOL))
                y = np.zeros(len(vidx), dtype=np.float32)
                for e in g[p]:
                    j = vidx.get(tuple(sdl.ev_key(e)))
                    if j is not None:
                        y[j] = 1.0
                Y.append(y)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def train_probe(X, Y, epochs=150, wd=1e-2, lr=0.05):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.tensor(X, device=dev); Yt = torch.tensor(Y, device=dev)
    lin = torch.nn.Linear(X.shape[1], Y.shape[1]).to(dev)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(lin(Xt), Yt)
        loss.backward(); opt.step()
    return lin, dev


def evaluate(feats, gt, thr=0.35, maxlen=3):
    sids = [s for s in gt if s in feats]
    vocab, vidx = build_vocab(gt, sids)
    vseqs = [P.k2e(list(k)) for k in vocab]
    gt_flat, pred_flat = {}, {}
    for held in sids:
        train = [s for s in sids if s != held]
        Xtr, Ytr = make_xy(feats, gt, train, vidx)
        lin, dev = train_probe(Xtr, Ytr)
        cV, cNV, call = P._freqs(train)
        topNV = [P.k2e(k2) for k2, _ in cNV.most_common(2)]
        topV = [P.k2e(k2) for k2, _ in cV.most_common(2)]
        # backfill = the SUBMITTED B2 alternatives (identical base hedge for every method)
        prior = [[topNV[0]], [topNV[1]], [topV[0]], [topV[1]]]
        F = feats[held]
        for i, seg_id in enumerate(F["seg_ids"]):
            seg = gt[held].get(seg_id)
            if seg is None:
                continue
            gt_flat[(held, seg_id)] = seg
            pr = {}
            for p, key in ((PA, PA), (PB, PB)):
                x = torch.tensor(VP.pool(F[key][i], POOL)[None], device=dev)
                with torch.no_grad():
                    probs = torch.sigmoid(lin(x))[0].cpu().numpy()
                order = np.argsort(-probs)
                top = [vseqs[j] for j in order[:maxlen] if probs[j] > thr]
                alts = [[]]                          # empty
                if top:
                    alts.append([top[0]])            # model top-1
                    if len(top) > 1:
                        alts.append([list(e) for e in top])  # model top sequence
                for g in prior:                      # prior fallback
                    if len(alts) >= 5:
                        break
                    alts.append([list(e) for e in g])
                pr[p] = alts[:5]
            pred_flat[(held, seg_id)] = pr
    return sdl.score_dataset(gt_flat, pred_flat)


def rows(gt, kind="video"):
    """[(label, res), ...] over the probe threshold grid; [] if features are absent."""
    feats = VP.load_feats(kind)
    if not feats:
        return []
    return [("%s linear probe thr=%.2f" % (kind, thr), evaluate(feats, gt, thr=thr))
            for thr in (0.25, 0.35, 0.5)]


if __name__ == "__main__":
    import sys
    kind = sys.argv[1] if len(sys.argv) > 1 else "video"
    gt = CV.build_all_gt()
    rs = rows(gt, kind)
    if not rs:
        print("NO %s FEATURES — run the matching extractor in dev/ first." % kind.upper())
        sys.exit(0)
    print("reference: constant prior B2 mean4=0.4127")
    for label, r in rs:
        print("%-28s %s" % (label, CV.fmt(r)))
