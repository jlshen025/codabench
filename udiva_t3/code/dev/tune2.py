"""Richer constant-prior tuning: mine the most-common ORDERED sequences (overall,
per-type, per-participant) from train GT and greedy-select K=5 alts maximizing CV mean4.
Aim: lift the 'full' subtask (predicting multi-event sequences) above the singleton prior.
"""
from collections import Counter
import _path  # noqa: F401
import udiva_data as U
import sdl
import cv as CV
import predictors as P

PA, PB = "participant_a", "participant_b"


def mine(train, gt):
    """Frequency tables from train GT segments."""
    seqAll, seqA, seqB = Counter(), Counter(), Counter()
    seqV, seqNV = Counter(), Counter()
    singAll, singV, singNV = Counter(), Counter(), Counter()
    for sid in train:
        for seg_id, seg in gt[sid].items():
            g = sdl.ref_seg_to_gt(seg)
            for rho, sc in ((PA, seqA), (PB, seqB)):
                seq = g[rho]
                sk = tuple(tuple(sdl.ev_key(e)) for e in seq)
                seqAll[sk] += 1
                sc[sk] += 1
                vk = tuple(tuple(sdl.ev_key(e)) for e in seq if len(e) == 2)
                nk = tuple(tuple(sdl.ev_key(e)) for e in seq if len(e) == 3)
                seqV[vk] += 1
                seqNV[nk] += 1
                for e in seq:
                    k = tuple(sdl.ev_key(e))
                    singAll[k] += 1
                    (singV if len(e) == 2 else singNV)[k] += 1
    return dict(seqAll=seqAll, seqA=seqA, seqB=seqB, seqV=seqV, seqNV=seqNV,
                singAll=singAll, singV=singV, singNV=singNV)


def sk2seq(sk):
    return [P.k2e(list(k)) for k in sk]


def build_pool(m):
    pool = {"empty": []}
    for i, (sk, _) in enumerate(m["seqAll"].most_common(6)):
        pool[f"seq{i+1}"] = sk2seq(sk)
    for i, (sk, _) in enumerate(m["seqNV"].most_common(4)):
        pool[f"seqNV{i+1}"] = sk2seq(sk)
    for i, (sk, _) in enumerate(m["seqV"].most_common(4)):
        pool[f"seqV{i+1}"] = sk2seq(sk)
    for i, (k, _) in enumerate(m["singAll"].most_common(4)):
        pool[f"s{i+1}"] = [P.k2e(list(k))]
    for i, (k, _) in enumerate(m["singV"].most_common(3)):
        pool[f"sV{i+1}"] = [P.k2e(list(k))]
    for i, (k, _) in enumerate(m["singNV"].most_common(3)):
        pool[f"sNV{i+1}"] = [P.k2e(list(k))]
    # drop empties / dups
    seen = {}
    out = {}
    for name, seq in pool.items():
        key = tuple(tuple(sdl.ev_key(e)) for e in seq)
        if key in seen:
            continue
        seen[key] = name
        out[name] = seq
    return out


class PoolHedge(P.Predictor):
    def __init__(self, alts):
        self.alts = alts

    def predict(self, sid, seg):
        a = [[list(e) for e in s] for s in self.alts]
        return {PA: a, PB: a}


def make_factory(names):
    def fac(train, gt):
        pool = build_pool(mine(train, gt))
        alts = [pool[n] for n in names if n in pool]
        return PoolHedge(alts)
    return fac


def greedy(gt, kmax=5):
    names_all = list(build_pool(mine(list(gt.keys()), gt)).keys())
    print("pool size:", len(names_all))
    chosen = []
    last = None
    while len(chosen) < kmax:
        best = None
        for n in names_all:
            if n in chosen:
                continue
            res = CV.evaluate(make_factory(chosen + [n]), gt)
            if best is None or res["mean4"] > best[1]["mean4"]:
                best = (n, res)
        chosen.append(best[0])
        last = best[1]
        print("add %-8s -> %s" % (best[0], CV.fmt(best[1])))
    return chosen, last


if __name__ == "__main__":
    gt = CV.build_all_gt()
    chosen, last = greedy(gt, kmax=5)
    print("\nBEST:", chosen)
    # print the actual sequences
    pool = build_pool(mine(list(gt.keys()), gt))
    for n in chosen:
        print("  %-8s = %s" % (n, pool[n]))
