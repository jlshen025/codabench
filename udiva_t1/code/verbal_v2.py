#!/usr/bin/env python3
"""Verbal channel -- candidate pool + static prior (SHIPPED helper).

`build_cands` enumerates the verbal candidate tuples (utterance_type, target, modifier)
observed >= min_count times in the training sessions and their static base rate
pi(t) = count(t) / N, where N = number of (segment, present-speaker) instances in the
training split (the marginal expected frequency of t per instance). Consumed by
verbal_emb.py and the verbal rerankers.

(The earlier TF-IDF-over-candidates experiment that lived here was superseded by the e5
heads and has been removed; only the pool/prior builder remains.)
"""
from collections import Counter
from verbal_model import session_instances


def build_cands(gt, train, min_count=2):
    ct = Counter(); ninst = 0
    for sid in train:
        for seg, (tb, te, s) in gt[sid]["verbal"].items():
            for (subj, u, tg, mod) in s:
                ct[(u, tg, mod)] += 1
    # denominator: number of (seg, speaker) instances in train
    for sid in train:
        ninst += len(session_instances(gt, sid))
    cands = [t for t, c in ct.items() if c >= min_count]
    base = {t: ct[t] / max(ninst, 1) for t in cands}
    return cands, base
