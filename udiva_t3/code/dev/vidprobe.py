"""DEVELOPMENT EXPERIMENT (not part of the submitted method).

Observation-conditioned anticipation by retrieval: k-NN over frozen features of the
OBSERVED prefix. Pool each (segment, participant) prefix into one vector; for a held-out
query retrieve the k nearest TRAIN (segment, participant) cells by cosine similarity and
build the K=5 alternatives from their ground-truth futures (empty + 2 most frequent
neighbour sequences + most frequent neighbour event + prior backfill). If this beats the
constant prior, the observed prefix carries per-segment predictive signal. Pure numpy.

Feature source (both produced by the extractors in this directory):
  video (default)  DINOv2 frames of the exocentric GF view  -> dev/feat_extract.py
  text             sentence embeddings of the transcript    -> dev/txt_extract.py

Usage: python dev/vidprobe.py [video|text]
"""
import os
import sys
import numpy as np
from collections import Counter
import _path  # noqa: F401
import udiva_data as U
import sdl
import cv as CV
import predictors as P

PA, PB = "participant_a", "participant_b"
SUBDIR = {"video": "feats", "text": "txtfeats"}
FEAT_DIR = U.WORK_DIR / "feats"


def load_feats(kind="video"):
    """{sid: {'seg_ids':[...], 't_b':[...], PA:arr[nseg,Nf,D], PB:arr}}"""
    feat_dir = U.WORK_DIR / SUBDIR[kind]
    out = {}
    for sid in U.list_sessions():
        p = feat_dir / f"{sid}.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        out[sid] = {"seg_ids": [str(s) for s in d["seg_ids"]], "t_b": d["t_b"],
                    PA: d["featA"], PB: d["featB"]}
    return out


def pool(arr, mode="meanlast"):
    """arr [Nf, D] -> vector."""
    if mode == "mean":
        v = arr.mean(0)
    elif mode == "last":
        v = arr[-1]
    else:  # meanlast
        v = np.concatenate([arr.mean(0), arr[-1]])
    n = np.linalg.norm(v) + 1e-8
    return v / n


def build_bank(feats, gt, sids, mode):
    """Return X [N,D], metas [(sid,seg_id,part,gt_seq), ...] for train sids."""
    X, metas = [], []
    for sid in sids:
        if sid not in feats:
            continue
        F = feats[sid]
        for i, seg_id in enumerate(F["seg_ids"]):
            seg = gt[sid].get(seg_id)
            if seg is None:
                continue
            g = sdl.ref_seg_to_gt(seg)
            for part, key in ((PA, PA), (PB, PB)):
                X.append(pool(F[key][i], mode))
                metas.append((sid, seg_id, part, g[part]))
    return np.array(X, dtype=np.float32), metas


def neighbor_alts(gt_seqs, k_alts, global_alts):
    """gt_seqs: list of neighbor GT sequences -> K=5 alts."""
    seqc = Counter(tuple(tuple(sdl.ev_key(e)) for e in s) for s in gt_seqs)
    evc = Counter(tuple(sdl.ev_key(e)) for s in gt_seqs for e in s)
    alts = [[]]  # empty
    for sk, _ in seqc.most_common(2):
        alts.append([P.k2e(list(x)) for x in sk])
    if evc:
        alts.append([P.k2e(list(evc.most_common(1)[0][0]))])
    for g in global_alts:
        if len(alts) >= 5:
            break
        alts.append([list(e) for e in g])
    # dedup keep order
    seen, out = set(), []
    for a in alts:
        kk = tuple(tuple(sdl.ev_key(e)) for e in a)
        if kk in seen:
            continue
        seen.add(kk); out.append(a)
    return out[:5]


class KNNPredictor(P.Predictor):
    def __init__(self, Xtr, metas, k, mode, global_alts, match_role=True):
        self.Xtr = Xtr; self.metas = metas; self.k = k; self.mode = mode
        self.global_alts = global_alts; self.match_role = match_role
        self.roles = np.array([m[2] for m in metas])

    def set_query_feats(self, F):
        self.F = F

    def predict_part(self, i, part):
        q = pool(self.F[part][i], self.mode)
        sims = self.Xtr @ q
        if self.match_role:
            mask = self.roles == part
            idx = np.where(mask)[0]
            order = idx[np.argsort(-sims[idx])[: self.k]]
        else:
            order = np.argsort(-sims)[: self.k]
        seqs = [self.metas[j][3] for j in order]
        return neighbor_alts(seqs, self.k, self.global_alts)


def evaluate(feats, gt, k=20, mode="meanlast", match_role=True):
    sids = [s for s in gt if s in feats]
    gt_flat, pred_flat = {}, {}
    for held in sids:
        train = [s for s in sids if s != held]
        Xtr, metas = build_bank(feats, gt, train, mode)
        # backfill = the SUBMITTED B2 alternatives (same base hedge for every method, so the
        # comparison measures the conditioning and not the choice of fallback set)
        cV, cNV, call = P._freqs(train)
        topNV = [P.k2e(k2) for k2, _ in cNV.most_common(2)]
        topV = [P.k2e(k2) for k2, _ in cV.most_common(2)]
        global_alts = [[topNV[0]], [topNV[1]], [topV[0]], [topV[1]]]
        knn = KNNPredictor(Xtr, metas, k, mode, global_alts, match_role)
        knn.set_query_feats(feats[held])
        F = feats[held]
        for i, seg_id in enumerate(F["seg_ids"]):
            seg = gt[held].get(seg_id)
            if seg is None:
                continue
            gt_flat[(held, seg_id)] = seg
            pred_flat[(held, seg_id)] = {PA: knn.predict_part(i, PA), PB: knn.predict_part(i, PB)}
    res = sdl.score_dataset(gt_flat, pred_flat)
    return res


def rows(gt, kind="video"):
    """[(label, res), ...] over the k-NN hyperparameter grid; [] if features are absent."""
    feats = load_feats(kind)
    if not feats:
        return []
    out = []
    modes = ["mean", "last", "meanlast"] if kind == "video" else ["mean"]
    for mode in modes:
        for k in [10, 20, 40]:
            out.append(("%s k-NN k=%d pool=%s" % (kind, k, mode),
                        evaluate(feats, gt, k=k, mode=mode)))
    return out


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "video"
    gt = CV.build_all_gt()
    rs = rows(gt, kind)
    print("%s features: %d/%d sessions" % (kind, len(load_feats(kind)), len(gt)))
    if not rs:
        print("NO FEATURES — run dev/%s first."
              % ("feat_extract.py" if kind == "video" else "txt_extract.py"))
        sys.exit(0)
    print("reference: constant prior B2 mean4=0.4127")
    for label, r in rs:
        print("%-28s %s" % (label, CV.fmt(r)))
