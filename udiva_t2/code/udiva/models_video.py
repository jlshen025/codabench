"""Non-verbal channel: 2 s-segment video features -> (subject, high-level action) posteriors
-> complete 5-tuples (subject, high-level action, low-level action, target, modifier).

The stages below are the left-hand lane of Fig. 1 of the fact sheet:

  load_feats           view fusion -- the mean-pooled clip features of E1 and E2 are
                       CONCATENATED into one segment vector (missing view -> zeros).
  fit_pair_heads       one one-vs-rest logistic head per JOINT class (subject, high-level
                       action); a segment is a multi-label example. There is no separate
                       subject head and no separate action head.
  predict_pair_probs   those heads applied to held-out / test segments.
  fuse_pair_probs      equal-weight late fusion of several backbones' posteriors, and
                       fuse_ft_probs adds the fine-tuned model's posteriors. Fusion happens
                       HERE, on the (subject, h) posteriors, BEFORE any tuple is formed.
  compose_nonverbal    expands a fused posterior into complete events using training
                       co-occurrence priors conditioned on the same (subject, h) pair.
"""
import os
import numpy as np
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from . import data as Data

FEAT_DIR = "<scratch>/t2/feats/videomae_large"


# --------------------------------------------------------------------------- features / views
def load_feats(sids, views=("E1", "E2"), feat_dir=FEAT_DIR):
    """{sid: {segkey: vector}} where vector = concat over views of the per-view segment feature.

    One <sid>_<view>.npz per session and view, as written by extract_feats.py, holding
    `seg_keys` and `feats` [n_seg, hidden]. A view missing for a session contributes a zero
    block, so the concatenated dimension is the same for every segment.
    """
    per_sid = {}
    dim = None
    for sid in sids:
        per_view = {}
        for v in views:
            p = os.path.join(feat_dir, f"{sid}_{v}.npz")
            if os.path.exists(p):
                d = np.load(p)
                per_view[v] = (list(d["seg_keys"]), d["feats"])
                dim = d["feats"].shape[1]
        per_sid[sid] = per_view

    res = {}
    for sid, per_view in per_sid.items():
        segkeys = None
        for v in views:
            if v in per_view:
                segkeys = per_view[v][0]
                break
        if segkeys is None:          # no view available for this session
            res[sid] = {}
            continue
        res[sid] = {}
        for i, sk in enumerate(segkeys):
            vecs = [per_view[v][1][i] if v in per_view else np.zeros(dim, np.float32)
                    for v in views]
            res[sid][sk] = np.concatenate(vecs)
    return res


# --------------------------------------------------------------------------- (subject, h) heads
def fit_pair_heads(train_sids, feat_dir, C=1.0, min_count=3):
    """Fit one one-vs-rest logistic head per (subject, high-level action) class.

    A class is kept when it occurs at least `min_count` times in the training events (64 classes
    over the 21 annotated sessions). Segment vectors are L2-normalised before fitting.
    Returns (labels, classifiers) aligned by index; a classifier may be None if its column ended
    up all-negative.
    """
    feats = load_feats(train_sids, feat_dir=feat_dir)
    gt = Data.load_gt(train_sids)["nonverbal"]

    X, idx = [], []
    for sid in train_sids:
        for sk, v in feats[sid].items():
            X.append(v / (np.linalg.norm(v) + 1e-9))
            idx.append((sid, sk))
    X = np.asarray(X, np.float32)

    count = Counter()
    per_segment = defaultdict(set)
    for sid in train_sids:
        for sk, blk in gt[sid].items():
            for e in blk["events"]:
                c = (e["subject"], e["highlevel_action"])
                count[c] += 1
                per_segment[(sid, sk)].add(c)

    labels = [c for c, n in count.items() if n >= min_count]
    li = {c: i for i, c in enumerate(labels)}
    Y = np.zeros((len(idx), len(labels)), np.int8)
    for r, key in enumerate(idx):
        for c in per_segment[key]:
            if c in li:
                Y[r, li[c]] = 1

    clf = []
    for j in range(len(labels)):
        if Y[:, j].sum() == 0:
            clf.append(None)
            continue
        m = LogisticRegression(C=C, max_iter=300)
        m.fit(X, Y[:, j])
        clf.append(m)
    return labels, clf


def predict_pair_probs(sids, feat_dir, labels, clf):
    """{(sid, segkey): {(subject, h): probability}} for the given sessions."""
    feats = load_feats(sids, feat_dir=feat_dir)
    out = {}
    for sid in sids:
        for sk, v in feats[sid].items():
            v = v / (np.linalg.norm(v) + 1e-9)
            out[(sid, sk)] = {labels[j]: float(m.predict_proba(v[None])[0, 1])
                              for j, m in enumerate(clf) if m is not None}
    return out


# --------------------------------------------------------------------------- late fusion
def fuse_pair_probs(train_sids, test_sids, dev_root, test_root, dir_weights):
    """Late-fuse several frozen backbones at the (subject, h) posterior level.

    For every feature directory a separate head set is fitted on the dev features and applied to
    the test features; the posteriors are accumulated as  S(rho,h) = sum_b w_b * P_b(rho,h | x).
    With the submitted configuration all w_b are 1, so S is a sum rather than a mean -- since
    mAP only ranks predictions, the constant factor is irrelevant.

    dir_weights = [(subdirectory name, weight), ...].
    """
    accum = {}
    for d, w in dir_weights:
        labels, clf = fit_pair_heads(train_sids, f"{dev_root}/{d}")
        probs = predict_pair_probs(test_sids, f"{test_root}/{d}", labels, clf)
        for key, row in probs.items():
            acc = accum.setdefault(key, {})
            for c, p in row.items():
                acc[c] = acc.get(c, 0.0) + w * p
        print(f"  fused component {d} (w={w})", flush=True)
    return accum


def fuse_ft_probs(accum, npz_path, weight=1.0):
    """Add the fine-tuned model's (subject, h) posteriors to `accum`, in place.

    `npz_path` is written by ft_infer.py and holds classes ('subject|highlevel_action'),
    seg_index ('sid|segkey') and probs [n_seg, n_classes].
    """
    d = np.load(npz_path, allow_pickle=True)
    classes = [tuple(c.split("|")) for c in d["classes"]]
    n = 0
    for r, si in enumerate(d["seg_index"]):
        sid, sk = si.split("|")
        acc = accum.setdefault((sid, sk), {})
        for j, c in enumerate(classes):
            acc[c] = acc.get(c, 0.0) + weight * float(d["probs"][r, j])
        n += 1
    print(f"  fused fine-tuned component {npz_path} (w={weight}) over {n} segments", flush=True)
    return accum


# --------------------------------------------------------------------------- tuple composition
def compose_nonverbal(train_sids, test_sids, pair_probs, kh=12, n_ll=3, n_tgt=4, ph=0.02,
                      grid_by_sid=None):
    """Expand fused (subject, h) posteriors into complete non-verbal events.

    Priors are training co-occurrence counts conditioned on the SAME pair (subject, h):
    P(low-level | subject, h), P(target | subject, h) and P(modifier | subject, h). For each of
    the `kh` highest-scoring pairs with S >= `ph`, the `n_ll` most frequent low-level actions and
    the `n_tgt` most frequent targets are emitted, with

        confidence = S(rho,h) * (0.5 + P(l | rho,h)) * (0.5 + P(tau | rho,h))

    and the modifier taken as arg max P(m | rho,h). The prior factors are deliberately shrunk
    (0.5 + P instead of P), so the confidence is a ranking score, not a calibrated probability.
    """
    gt_tr = Data.load_gt(train_sids)["nonverbal"]
    cooc_ll = defaultdict(Counter)   # (subject, h) -> low-level action counts
    cooc_t = defaultdict(Counter)    # (subject, h) -> target counts
    cooc_m = defaultdict(Counter)    # (subject, h) -> modifier counts
    for sid in train_sids:
        for sk, blk in gt_tr[sid].items():
            for e in blk["events"]:
                k = (e["subject"], e["highlevel_action"])
                cooc_ll[k][e["lowlevel_action"]] += 1
                cooc_t[k][e["target"]] += 1
                cooc_m[k][e["modifier"]] += 1

    pred = {"nonverbal": {}}
    for sid in test_sids:
        pred["nonverbal"][sid] = {}
        seg_iter = grid_by_sid[sid] if grid_by_sid else Data.gt_session(sid)["nonverbal"]
        for sk in seg_iter:
            seg = {}
            P = pair_probs.get((sid, sk), {})
            for (subj, h), pv in sorted(P.items(), key=lambda x: -x[1])[:kh]:
                if pv < ph:
                    continue
                k = (subj, h)
                ll_total = sum(cooc_ll[k].values()) or 1
                t_total = sum(cooc_t[k].values()) or 1
                md = (cooc_m[k].most_common(1) or [("none", 1)])[0][0]
                for ll, lc in (cooc_ll[k].most_common(n_ll) or [("look_at", 1)]):
                    for tg, tc in (cooc_t[k].most_common(n_tgt) or [("none", 1)]):
                        sc = float(pv * (0.5 + lc / ll_total) * (0.5 + tc / t_total))
                        tup = (subj, h, ll, tg, md)
                        if seg.get(tup, 0) < sc:      # keep the best score per distinct tuple
                            seg[tup] = sc
            pred["nonverbal"][sid][sk] = {"events": [
                {"subject": t[0], "highlevel_action": t[1], "lowlevel_action": t[2],
                 "target": t[3], "modifier": t[4], "score": s} for t, s in seg.items()]}
    return pred
