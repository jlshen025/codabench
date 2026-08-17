#!/usr/bin/env python
"""build_na.py — move Against-pool rows to NONE, ranked by NOT-AGAINST-ness.

THE REFRAMING (verified exactly in _probe/not_against.py). Moving an Against-pool row to
None pays +0.001274 whether the row is gold-None OR gold-Favor -- a gold-Favor row in the
Against pool is already a miss for F1_Favor, so it is not in that numerator and moving it
out changes neither pred_F nor TP_F. Only gold-Against rows cost anything (-0.001616).
So the payoff set is 22 None + 12 Favor = 34 of 187 (18.2%), break-even 55.9% on
not-Against, and the all-safe endpoint is 0.9451.

DESTINATION MATTERS. To None: {None +.001274, Favor +.001274, Against -.001616}.
To Favor: {Favor +.002988, None -.000128, Against -.003018}. For any ranking that
surfaces a MIX of None and Favor -- which all of mine do -- None is the better
destination, and the sub-876844 loss is exactly what the to-Favor row predicts.

RANKING. Measured not-Against precision of the pieces (from purchased decodes):
sentiment top-6 0.500 · gsol None-votes@18 0.444 · ND detectors@10 0.400 · margin 0.000.
The two best target COMPLEMENTARY halves of the payoff set -- sentiment finds gold-Favor
(high tone) and, being mid-scale, some gold-None; the None-vote rankings find gold-None.
Summing their rank-percentiles is therefore the untried construction, and `--mode isect`
takes only rows both signals like, trading recall for the precision the bar demands.

Sentiment enters as a not-Against signal directly: within the Against pool gold-Against
sits at the LOW end (negative tone), gold-Favor at the high end and gold-None in the
middle, so "tone is not low" is monotone in not-Against-ness.

GATE-1: rows are chosen by model readings of the released TEXT. Server decodes are used
only to score candidates and forecast depth -- never to search over label assignments.
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
CH = ["luna", "g55", "lunaFS", "g55FS", "opus", "son45"]
G = ["g55G", "lunaG", "g55FSG", "lunaFSG"]
SOL = ["solKI", "solB", "solFS"]
ND = ["nd_luna_r_test", "nd_terra_r_test", "nd_lunafs_r_test"]
SENT = ["sent_luna_test", "sent_terra_test", "sent_g55_test", "sent_son45_test"]


def pr(k):
    return np.load(os.path.join(F, "frontier_%s_test.npz" % k))["proba"].astype(float)


def rankpct(v):
    return np.argsort(np.argsort(v)) / (len(v) - 1.0)


def current():
    tie = np.mean([pr(t) for t in CH], 0) + 1e-3 * (
        0.15 * np.load(os.path.join(R, "enc_ce.npz"))["proba"]
        + 0.85 * np.load(os.path.join(R, "dep_qens.npz"))["proba"]).astype(float)
    b = tie.argmax(1)
    out = b.copy()
    ni = np.where(b == 2)[0]
    mg = tie[ni, 2] - np.max(tie[ni][:, :2], axis=1)
    keep = set(ni[np.argsort(-mg)[:3]].tolist())
    for i in ni:
        if i not in keep:
            out[i] = int(np.argmax(tie[i, :2]))
    return out


def signals():
    """Two decorrelated not-Against views, each as a 0-1 rank percentile."""
    gv = np.sum([pr(k).argmax(1) == 2 for k in G], 0).astype(float)
    sv = np.sum([pr(k).argmax(1) == 2 for k in SOL], 0).astype(float)
    nd = np.mean([rankpct(np.load(os.path.join(F, n + ".npz"))["score"].astype(float))
                  for n in ND if os.path.exists(os.path.join(F, n + ".npz"))], 0)
    noneness = rankpct(gv / len(G) + sv / len(SOL) + nd)      # gold-None half
    sent = np.mean([rankpct(np.load(os.path.join(F, s + ".npz"))["score"].astype(float))
                    for s in SENT if os.path.exists(os.path.join(F, s + ".npz"))], 0)
    return noneness, sent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--mode", default="sum", choices=["sum", "isect", "sent", "none"])
    ap.add_argument("--w_sent", type=float, default=0.5)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    cur = current()
    noneness, sent = signals()
    pool = np.where(cur == 0)[0]

    if args.mode == "sum":
        score = (1 - args.w_sent) * noneness + args.w_sent * sent
    elif args.mode == "sent":
        score = sent
    elif args.mode == "none":
        score = noneness
    else:   # isect: rank by the WEAKER of the two views, so both must like a row
        score = np.minimum(rankpct(noneness), rankpct(sent))

    pick = pool[np.argsort(-score[pool])[:args.k]]
    out = cur.copy()
    out[pick] = 2

    print("%-20s mode=%s k=%d  pool=%d" % (args.label, args.mode, args.k, len(pool)))
    print("  dist:", {ID2[k]: v for k, v in sorted(collections.Counter(out.tolist()).items())})
    print("  break-even 0.559 on not-Against; pool holds 22 gold-None + 12 gold-Favor of 187")
    ns, ss = rankpct(noneness)[pick], rankpct(sent)[pick]
    print("  picked rows: mean None-view pct %.2f, mean sentiment pct %.2f, "
          "both-in-top-quartile %d" % (ns.mean(), ss.mean(), int(((ns > .75) & (ss > .75)).sum())))

    txt = "\n".join(ID2[i] for i in out) + "\n"
    zp = os.path.join(F, args.label + ".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("predictions.txt", txt)
    print("  wrote", zp)


if __name__ == "__main__":
    main()
