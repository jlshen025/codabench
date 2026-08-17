"""Leave-sessions-out CV: pooled out-of-fold mAP (stable for small N=21).

Protocol: partition sessions into K folds; for each fold, a predict_fn trains on the
other sessions and predicts the held-out fold; pool all OOF predictions; compute ONE mAP
over all sessions (mirrors the eval = one mAP over the whole set). Repeat with seeds for std.
"""
import random
import numpy as np
from . import data as Data
from . import metric as M


def make_folds(sids, k=7, seed=0):
    sids = list(sids)
    rng = random.Random(seed)
    rng.shuffle(sids)
    return [sids[i::k] for i in range(k)]


def run_oof(predict_fn, sids=None, k=7, seed=0, score_kwargs=None):
    """predict_fn(train_sids, test_sids) -> predictions dict {'verbal':{sid:{seg:{events}}},'nonverbal':...}.
    Returns (score_dict, pooled_pred, gt)."""
    sids = list(sids or Data.all_sids())
    folds = make_folds(sids, k=k, seed=seed)
    pooled = {"verbal": {}, "nonverbal": {}}
    for f in folds:
        train = [s for s in sids if s not in f]
        pred = predict_fn(train, f)
        for st in ("verbal", "nonverbal"):
            pooled[st].update(pred.get(st, {}))
    gt = Data.load_gt(sids)
    sc = M.score(pooled, gt, **(score_kwargs or {}))
    return sc, pooled, gt


def repeated_oof(predict_fn, sids=None, k=7, seeds=(0, 1, 2), score_kwargs=None):
    """Return dict with mean/std of mAP across seeds + per-seed scores."""
    maps, vs, hs = [], [], []
    per = []
    for sd in seeds:
        sc, _, _ = run_oof(predict_fn, sids=sids, k=k, seed=sd, score_kwargs=score_kwargs)
        maps.append(sc["mAP"]); vs.append(sc["mAP_verbal"]); hs.append(sc["mAP_nonverbal"])
        per.append(sc)
    return {"mAP_mean": float(np.mean(maps)), "mAP_std": float(np.std(maps)),
            "mAP_verbal_mean": float(np.mean(vs)), "mAP_nonverbal_mean": float(np.mean(hs)),
            "per_seed": maps, "n_seeds": len(seeds)}
