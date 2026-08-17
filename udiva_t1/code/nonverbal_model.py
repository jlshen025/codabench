#!/usr/bin/env python3
"""Non-verbal channel -- VideoMAE feature loader + (segment, subject) instance builder
(SHIPPED helper).

Each per-segment feature is [3, hidden] over the three crops (left, right, full). Per
(segment, subject) instance the model input is the subject-relative concatenation
[own || full || other] (participant_a: left is own; participant_b: right is own; full =
shared scene). Consumed by the non-verbal MLP ensemble (nv_mlp2.train_fold) and the
non-verbal rerankers.

(The earlier per-attribute logistic-regression non-verbal baseline that lived here was
superseded by the 11-seed MLP ensemble and has been removed; only the feature/instance
builders remain.)
"""
import os
import numpy as np
from collections import defaultdict

FEAT = os.environ.get("UDIVA_FEAT", "<scratch>/t1/feats")
CROP = {"left": 0, "right": 1, "full": 2}   # index into the per-segment [3, hidden] feature


def load_feats(sid):
    d = np.load(f"{FEAT}/{sid}.npz", allow_pickle=True)
    segs = [s for s in d["segs"]]
    F = d["feats"].astype(np.float32)  # [n, 3, hidden]
    return {seg: F[i] for i, seg in enumerate(segs)}


def feat_for(F, subj):
    # subject-relative: [own, full, other]
    if subj == "participant_a":
        return np.concatenate([F[0], F[2], F[1]])
    else:
        return np.concatenate([F[1], F[2], F[0]])


def instances(gt, sid, feats=None):
    feats = feats or load_feats(sid)
    out = []
    for seg, (tb, te, s) in gt[sid]["nonverbal"].items():
        if seg not in feats: continue
        bys = defaultdict(lambda: (set(), set(), set(), set()))
        for (subj, h, l, tgt, mod) in s:
            H, L, T, M = bys[subj]; H.add(h); L.add(l); T.add(tgt); M.add(mod)
        for subj in ("participant_a", "participant_b"):
            H, L, T, M = bys.get(subj, (set(), set(), set(), set()))
            out.append((seg, subj, feat_for(feats[seg], subj), H, L, T, M))
    return out
