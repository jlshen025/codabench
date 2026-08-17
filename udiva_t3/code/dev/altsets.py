"""DEVELOPMENT EXPERIMENT (input-free variants of the submitted method).

Metric-exploit variants of the K=5 alternative set: longer (multi-event) hypotheses and
mixed verbal+non-verbal hypotheses, versus the submitted B2 set.

Every variant is a CONSTANT predictor (no observed input); only the COMPOSITION of the
K=5 set is varied. Content is always rebuilt from the training fold's frequencies, so
LOSO CV stays honest; the composition is the only hyperparameter, and it is selected on
CV (Gate: 5 slots out of a ~12-element pool = very low capacity).

Building blocks per fold (train sessions only):
  NV1..NV4  most frequent non-verbal event tuples (singleton hypotheses)
  V1,V2     most frequent verbal event tuples
  S1..S3    most frequent ORDERED ground-truth window sequences (any type, length >= 2)
  SNV1..2   most frequent ORDERED non-verbal-only window sequences (length >= 2)

Usage: python dev/altsets.py
"""
from collections import Counter

import _path  # noqa: F401
import udiva_data as U
import sdl
import cv as CV
import predictors as P

PA, PB = "participant_a", "participant_b"


def mine(train, gt):
    """Ordered-sequence and singleton frequency tables from the training fold."""
    seqAll, seqNV, singV, singNV = Counter(), Counter(), Counter(), Counter()
    for sid in train:
        for seg in gt[sid].values():
            g = sdl.ref_seg_to_gt(seg)
            for rho in (PA, PB):
                seq = g[rho]
                if len(seq) >= 2:
                    seqAll[tuple(tuple(sdl.ev_key(e)) for e in seq)] += 1
                    nv = [e for e in seq if len(e) == 3]
                    if len(nv) >= 2:
                        seqNV[tuple(tuple(sdl.ev_key(e)) for e in nv)] += 1
    for sid in train:                     # singletons from the raw annotated events
        for a in U.load_raw(sid):
            if a["subject"] not in (PA, PB):
                continue
            e = U.event_tuple(a)
            (singV if len(e) == 2 else singNV)[tuple(sdl.ev_key(e))] += 1
    return seqAll, seqNV, singV, singNV


def pool(train, gt):
    seqAll, seqNV, singV, singNV = mine(train, gt)
    nv = [P.k2e(k) for k, _ in singNV.most_common(4)]
    v = [P.k2e(k) for k, _ in singV.most_common(2)]
    p = {"empty": []}
    for i, e in enumerate(nv):
        p["NV%d" % (i + 1)] = [e]
    for i, e in enumerate(v):
        p["V%d" % (i + 1)] = [e]
    p["NV12"] = nv[:2]
    p["NV123"] = nv[:3]
    p["V12"] = v[:2]
    p["NV1V1"] = [nv[0], v[0]]
    p["NV12V1"] = nv[:2] + [v[0]]
    for i, (sk, _) in enumerate(seqAll.most_common(3)):
        p["S%d" % (i + 1)] = [P.k2e(list(k)) for k in sk]
    for i, (sk, _) in enumerate(seqNV.most_common(2)):
        p["SNV%d" % (i + 1)] = [P.k2e(list(k)) for k in sk]
    return p


# name -> the K=5 composition (pool keys). First = the submitted set.
SETS = {
    "B2 SUBMITTED [empty,NV1,NV2,V1,V2]": ["empty", "NV1", "NV2", "V1", "V2"],
    "B4 non-verbal-heavy [empty,NV1-4]": ["empty", "NV1", "NV2", "NV3", "NV4"],
    "B2 without the empty hypothesis": ["NV1", "NV2", "NV3", "V1", "V2"],
    "mixed: +1 mixed V+NV hypothesis": ["empty", "NV1", "V1", "NV1V1", "NV12"],
    "mixed: 2 mixed hypotheses": ["empty", "NV1V1", "NV12V1", "V12", "NV1"],
    "longer: length-2/3 non-verbal": ["empty", "NV12", "NV123", "V12", "NV1"],
    "longer: mined ordered sequences": ["empty", "S1", "S2", "S3", "NV1"],
    "longer: mined non-verbal sequences": ["empty", "SNV1", "SNV2", "NV1", "V1"],
}


def make_factory(names):
    def fac(train, gt):
        pl = pool(train, gt)
        return P.ConstAlts([pl[n] for n in names if n in pl])
    return fac


def rows(gt):
    return [(label, CV.evaluate(make_factory(names), gt)) for label, names in SETS.items()]


if __name__ == "__main__":
    gt = CV.build_all_gt()
    print("K=5 alternative-set variants — LOSO, n=%d segments"
          % sum(len(v) for v in gt.values()))
    for label, res in rows(gt):
        print("%-38s %s" % (label, CV.fmt(res)))
    print("\ncomposition when fit on all 21 sessions:")
    pl = pool(U.list_sessions(), gt)
    for label, names in SETS.items():
        print("  %-38s %s" % (label, [pl[n] for n in names]))
