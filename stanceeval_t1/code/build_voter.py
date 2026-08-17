#!/usr/bin/env python
"""build_voter.py — add a voter to the champion ensemble and project the None count.

Architecture (unchanged from the champion):
    tie  = mean(voters) + 1e-3 * ce          # ce = the fine-tuned pipeline, ARBITER only
    pred = argmax(tie), then None-count projected down to exactly --n rows

The None projection is now the SMALL knob, not the big one: the 08-03 server sweep on
this base gave N=14 -> 0.891160, N=11 -> 0.893190, N=8 -> 0.895000, monotone downward,
because Favg2 excludes None as a CLASS and our None ranking runs at 43% precision
against a 56.8% break-even. So a new voter is judged at a LOW n, where the ensemble's
F/A decisions -- the thing a voter actually changes -- are what the score is reading.

Gating a candidate voter (both axes required, this project has paid for each):
  * DECORRELATION from the incumbents -- a clone adds nothing (a 0.901-agreeing voter
    once dropped a 0.865 ensemble to 0.853).
  * STANDALONE STRENGTH on the target distribution -- the best-decorrelated voter ever
    built here still dragged, because decorrelation without strength is noise.
  * Enter SEPARATELY. Averaging cross-lab voters into one member, or substituting them
    for incumbents, destroyed most of the +0.0065 cross-lab gain.

Usage:
    python build_voter.py --add magis --n 0 --label wd_v7magis_n0
    python build_voter.py --add magis solA --n 0 --label wd_v8_n0
    python build_voter.py --n 0 --label wd_base_n0          # incumbent, no new voter
"""
import argparse
import collections
import os
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(HERE, "_eval")
R = os.path.join(HERE, "results", "eval_deploy_all", "_llm")
ID2 = {0: "Against", 1: "Favor", 2: "None"}

CHAMP = ["luna", "g55", "lunaFS", "g55FS", "opus", "son45"]


def pr(k):
    return np.load(os.path.join(F, "frontier_%s_test.npz" % k))["proba"].astype(float)


def arbiter():
    return (0.15 * np.load(os.path.join(R, "enc_ce.npz"))["proba"]
            + 0.85 * np.load(os.path.join(R, "dep_qens.npz"))["proba"]).astype(float)


def project(tie, n):
    """Keep the n strongest-margin None rows; everything else takes its best stance."""
    base = tie.argmax(1)
    out = base.copy()
    nidx = np.where(base == 2)[0]
    if len(nidx):
        marg = tie[nidx, 2] - np.max(tie[nidx][:, :2], axis=1)
        keep = set(nidx[np.argsort(-marg)[:n]].tolist())
        for i in nidx:
            if i not in keep:
                out[i] = int(np.argmax(tie[i, :2]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", nargs="*", default=[], help="extra voter tags (frontier_<tag>_test.npz)")
    ap.add_argument("--drop", nargs="*", default=[], help="incumbent tags to remove")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--eps", type=float, default=1e-3,
                    help="arbiter weight. NOT a continuous axis: with k one-hot voters the "
                         "vote granularity is 1/k, so every eps in (0, 1/k) yields IDENTICAL "
                         "labels (measured: 1e-4..0.17 all agree at k=6). Only eps=0 and "
                         "eps>=1/k change anything.")
    ap.add_argument("--weights", nargs="*", type=float, default=None,
                    help="per-voter weights in CHAMP order (then --add). Flat by default. "
                         "The leave-one-out sweep measured each seat's marginal value: "
                         "luna .0182, lunaFS .0181, son45 .0152, g55 .0120, g55FS .0120 — "
                         "so weights proportional to those are a data-grounded hypothesis, "
                         "not a free parameter fished from CV.")
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    tags = [t for t in CHAMP if t not in args.drop] + list(args.add)
    V = [pr(t) for t in tags]
    if args.weights:
        if len(args.weights) != len(tags):
            raise SystemExit("--weights needs %d values for %s" % (len(tags), tags))
        w = np.asarray(args.weights, float)
        w = w / w.sum()
        tie = np.tensordot(w, np.stack(V, 0), axes=1) + args.eps * arbiter()
    else:
        tie = np.mean(V, 0) + args.eps * arbiter()
    out = project(tie, args.n)

    champ_tie = np.mean([pr(t) for t in CHAMP], 0) + 1e-3 * arbiter()
    champ = project(champ_tie, 8)      # the current server-best, N=8 -> 0.895000

    print("%-18s voters=%d %s  n=%d" % (args.label, len(tags), tags, args.n))
    print("  dist:", {ID2[k]: v for k, v in sorted(collections.Counter(out.tolist()).items())})
    print("  rows differing from the N=8 server-best: %d" % int((out != champ).sum()))
    for t in args.add:
        p = pr(t).argmax(1)
        agree = [(c, float((p == pr(c).argmax(1)).mean())) for c in CHAMP]
        print("  %-8s dist=%s  agreement vs incumbents: %s" % (
            t, dict(collections.Counter([ID2[i] for i in p])),
            " ".join("%s=%.2f" % (c, a) for c, a in agree)))

    txt = "\n".join(ID2[i] for i in out) + "\n"
    zp = os.path.join(F, args.label + ".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("predictions.txt", txt)
    print("  wrote", zp)


if __name__ == "__main__":
    main()
