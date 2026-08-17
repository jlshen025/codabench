"""Greedy selection of the K=5 alternative set (constant prior) to maximize LOSO-CV mean4.

The ALTS' content is rebuilt from each fold's train freqs (generalizes); only the
COMPOSITION (which builders) is the hyperparameter selected on CV. Low capacity
(<=5 from a small pool) -> low adaptive-overfit risk.
"""
from collections import Counter
import _path  # noqa: F401
import udiva_data as U
import sdl
import cv as CV
import predictors as P

PA, PB = "participant_a", "participant_b"


def fold_freqs(train):
    return P._freqs(train)


def build_pool(cV, cNV, call):
    """name -> sequence (alt). Rebuilt per fold."""
    topAll = [P.k2e(k) for k, _ in call.most_common(5)]
    topNV = [P.k2e(k) for k, _ in cNV.most_common(5)]
    topV = [P.k2e(k) for k, _ in cV.most_common(5)]
    pool = {"empty": []}
    for i in range(3):
        if i < len(topAll):
            pool[f"all{i+1}"] = [topAll[i]]
        if i < len(topV):
            pool[f"V{i+1}"] = [topV[i]]
        if i < len(topNV):
            pool[f"NV{i+1}"] = [topNV[i]]
    pool["NV12"] = topNV[:2]
    pool["NV123"] = topNV[:3]
    pool["NV1_V1"] = [topNV[0], topV[0]] if topV and topNV else topNV[:1]
    pool["V1_NV1"] = [topV[0], topNV[0]] if topV and topNV else topNV[:1]
    pool["NV12_V1"] = (topNV[:2] + [topV[0]]) if topV else topNV[:2]
    return pool


class PoolHedge(P.Predictor):
    def __init__(self, alts):
        self.alts = alts

    def predict(self, sid, seg):
        a = [list(x) for x in self.alts]
        return {PA: a, PB: a}


def make_factory(names):
    def fac(train, gt):
        cV, cNV, call = fold_freqs(train)
        pool = build_pool(cV, cNV, call)
        alts = [pool[n] for n in names if n in pool]
        return PoolHedge(alts)
    return fac


def cv_mean4(names, gt):
    res = CV.evaluate(make_factory(names), gt)
    return res


def greedy(gt, kmax=5):
    # candidate builder names (from a representative fold's pool)
    cV, cNV, call = fold_freqs(list(gt.keys()))
    names_all = list(build_pool(cV, cNV, call).keys())
    chosen = []
    history = []
    while len(chosen) < kmax:
        best = None
        for n in names_all:
            if n in chosen:
                continue
            res = cv_mean4(chosen + [n], gt)
            if best is None or res["mean4"] > best[1]["mean4"]:
                best = (n, res)
        chosen.append(best[0])
        history.append((list(chosen), best[1]))
        print("add %-8s -> %s" % (best[0], CV.fmt(best[1])))
    return chosen, history


if __name__ == "__main__":
    gt = CV.build_all_gt()
    print("greedy K=5 alt selection (maximize CV mean4):")
    chosen, hist = greedy(gt, kmax=5)
    print("\nBEST 5-alt set:", chosen)
    # show the per-subtask of the final
    print("final:", CV.fmt(hist[-1][1]))
