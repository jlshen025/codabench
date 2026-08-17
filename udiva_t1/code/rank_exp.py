#!/usr/bin/env python3
"""Non-verbal reranker -- cross-segment feature builder (SHIPPED helper).

`xseg_feats` produces per-(session, class)-RELATIVE features from a fixed base score
(rank / percentile / z / log-cardinality / temporal peak vs neighbouring segments). They are
label-free (computable at test time from the test session's own candidates) and are
concatenated onto the non-verbal reranker rows in ship2.py / ship_eval.py to fix the
cross-segment "empty-slot FP" ranking failure that segment-local features cannot.

This module previously also hosted the cross-segment RANKING EXPERIMENT: a RankNet-style
PAIRWISE reranker and a pooled-vs-per-session mAP ablation. The pairwise variant was
evaluated but NOT selected -- the shipped reranker is the POINTWISE HistGradientBoosting in
ship2.py, whose target is the binary exact-match label. That experiment code has been
removed; only the two feature builders it contributed remain.
"""
from collections import defaultdict
import numpy as np


def make_meta(rows):
    meta = {"sid": [r[0] for r in rows], "seg": [r[1] for r in rows], "subj": [r[2] for r in rows],
            "cls": [r[3] for r in rows], "tup": [r[4] for r in rows]}
    meta["sid_arr"] = np.array(meta["sid"])
    return meta


def xseg_feats(meta, base):
    """Per-(session,class)-relative features from a fixed base score (no labels -> no leakage,
    computable at test time from the test session's own candidates)."""
    n = len(base)
    seg_i = np.array([int(s.split("_")[1]) for s in meta["seg"]])
    grp = defaultdict(list)
    for r in range(n):
        grp[(meta["sid"][r], meta["cls"][r])].append(r)
    lrank = np.zeros(n); pct = np.zeros(n); z = np.zeros(n); card = np.zeros(n)
    peak = np.zeros(n); nnb = np.zeros(n)
    for key, rs in grp.items():
        rs = np.array(rs); b = base[rs]
        order = np.argsort(-b); ranks = np.empty(len(rs)); ranks[order] = np.arange(len(rs))
        sz = len(rs); mu = b.mean(); sd = b.std() + 1e-9
        seg2max = {}
        for r in rs:
            si = seg_i[r]
            if si not in seg2max or base[r] > seg2max[si]: seg2max[si] = base[r]
        for j, r in enumerate(rs):
            lrank[r] = np.log1p(ranks[j]); pct[r] = 1.0 - ranks[j] / max(sz - 1, 1)
            z[r] = (base[r] - mu) / sd; card[r] = np.log1p(sz)
            si = seg_i[r]
            nb = [seg2max[si + k] for k in (-2, -1, 1, 2) if si + k in seg2max]
            peak[r] = base[r] - (max(nb) if nb else 0.0); nnb[r] = float(len(nb))
    return np.column_stack([lrank, pct, z, card, peak, nnb])
