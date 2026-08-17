#!/usr/bin/env python3
"""Verbal channel -- transcript instance builder (SHIPPED helper).

Provides `session_instances`, the (segment, present-speaker) instance list consumed by the
verbal base model (verbal_emb.py) and every verbal reranker. One instance per (segment,
speaker) that produced transcript text in the segment, carrying that segment's GT verbal
attributes grouped by speaker; subject = speaker parsed from the .srt.

(The earlier TF-IDF + logistic-regression verbal baseline that lived here was superseded by
the frozen multilingual-e5 heads in verbal_emb.py and has been removed.)
"""
from collections import defaultdict
from transcripts import seg_speaker_text


def session_instances(gt, sid):
    """-> list of (seg, speaker, text, {u}, {tgt}, {mod})."""
    grid = [(sk, tb, te) for sk, (tb, te, s) in gt[sid]["verbal"].items()]
    st = seg_speaker_text(sid, grid)
    inst = []
    for sk, tb, te, s in [(k, *v) for k, v in gt[sid]["verbal"].items()]:
        txt_by_spk = st.get(sk, {})
        # group GT tuples by subject
        bys = defaultdict(lambda: (set(), set(), set()))
        for (subj, u, tgt, mod) in s:
            U, T, M = bys[subj]; U.add(u); T.add(tgt); M.add(mod)
        for spk, txt in txt_by_spk.items():
            if not txt.strip(): continue
            U, T, M = bys.get(spk, (set(), set(), set()))
            inst.append((sk, spk, txt, U, T, M))
    return inst
