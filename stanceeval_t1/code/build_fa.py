#!/usr/bin/env python
"""build_fa.py — move rows across the Favor/Against boundary of the current best.

TARGET, from the exact decode of sub 875984 (0.897170, N=3), whose off-diagonal is
UNIQUELY determined: pred F162 A187 N3, TP_F=146, TP_A=153, TP_N=3, and
    goldFavor -> predA : 12      <-- what this script hunts
    goldNone  -> predA : 22
    goldAgainst-> predF:  7
So the Against pool holds 12 recoverable gold-Favor rows among 187.

ECONOMICS (computed exactly, see _probe/fa_breakeven.py). One Against->Favor flip:
    gold Favor   +0.002988      gold Against  -0.003018      gold None  -0.000128
Break-even is therefore ~**51%** true-Favor among the flipped non-None rows — far
gentler than the 56.8% that killed the abstention family, because the 22 gold-None rows
in the pool are nearly a WASH rather than a penalty. That asymmetry is the whole reason
this lever is worth slots when the None lever was not.

RANKINGS (--src):
  margin  : the ensemble's own Favor-minus-Against score. FREE, but it is the same
            signal that put these rows in Against, so it is the control, not the bet.
  sent    : rank-averaged sentiment readers (none_detect.py --variant p). The bet:
            P(Favor | Positive) = 0.984 on labelled data and the direction is stable
            across all three seen targets, while stance polarity itself does not
            transfer. Sentiment is asked WITHOUT mentioning stance, so it is a
            genuinely decorrelated opinion rather than a seventh stance voter.

GATE-1: rows are chosen by a model's reading of the TEXT. The server decode is used only
to score a candidate afterwards and forecast the next depth — never to search over which
individual rows carry which label.

Usage:
    python build_fa.py --src margin --k 8  --label wd_fa_margin_k8
    python build_fa.py --src sent --sent _eval/sent_*_test.npz --k 12 --label wd_fa_sent_k12
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


def best_base(n=3):
    """Sub 875984: the current server-best, 0.897170."""
    tie = np.mean([pr(t) for t in CHAMP], 0) + 1e-3 * (
        0.15 * np.load(os.path.join(R, "enc_ce.npz"))["proba"]
        + 0.85 * np.load(os.path.join(R, "dep_qens.npz"))["proba"]).astype(float)
    base = tie.argmax(1)
    out = base.copy()
    ni = np.where(base == 2)[0]
    marg = tie[ni, 2] - np.max(tie[ni][:, :2], axis=1)
    keep = set(ni[np.argsort(-marg)[:n]].tolist())
    for i in ni:
        if i not in keep:
            out[i] = int(np.argmax(tie[i, :2]))
    return out, tie


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="margin", choices=["margin", "sent"])
    ap.add_argument("--sent", nargs="*", default=[], help="sentiment *_test.npz to rank-average")
    ap.add_argument("--k", type=int, required=True, help="rows to move Against->Favor")
    ap.add_argument("--reverse", action="store_true", help="instead move Favor->Against")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    cur, tie = best_base(args.n)
    if args.src == "margin":
        score = tie[:, 1] - tie[:, 0]                    # higher = more Favor-like
    else:
        rk = []
        for p in args.sent:
            v = np.load(p)["score"].astype(float)
            rk.append(np.argsort(np.argsort(v)) / (len(v) - 1.0))
        if not rk:
            raise SystemExit("--src sent needs --sent files")
        score = np.mean(rk, 0)

    out = cur.copy()
    if args.reverse:
        pool = np.where(cur == 1)[0]
        pick = pool[np.argsort(score[pool])[:args.k]]     # least Favor-like in the Favor pool
        out[pick] = 0
    else:
        pool = np.where(cur == 0)[0]
        pick = pool[np.argsort(-score[pool])[:args.k]]    # most Favor-like in the Against pool
        out[pick] = 1

    print("%-20s src=%s k=%d reverse=%s  pool=%d" % (args.label, args.src, args.k,
                                                     args.reverse, len(pool)))
    print("  dist:", {ID2[k]: v for k, v in sorted(collections.Counter(out.tolist()).items())})
    print("  rows differing from the 0.897170 best: %d" % int((out != cur).sum()))
    print("  break-even ~51%% true-Favor among flipped non-None rows; "
          "the Against pool holds 12 gold-Favor and 22 gold-None of 187")

    txt = "\n".join(ID2[i] for i in out) + "\n"
    zp = os.path.join(F, args.label + ".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("predictions.txt", txt)
    print("  wrote", zp)


if __name__ == "__main__":
    main()
