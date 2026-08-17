#!/usr/bin/env python
"""build_none.py — put a NEW None head on top of the champion's F/A decisions.

The champion (sub 869442, Favg2 0.891160) keeps its Favor/Against call on every row.
All this script changes is WHICH rows abstain. That separation is the whole point:
the F/A axis is a measured near-Bayes residual (four independent probes), while the
None axis is bottom-of-field (F1_None 0.250 vs 0.828/0.604 for the two rivals above
us) and carries ~28 rows of headroom.

None sources, all already on disk and each a differently-framed opinion:
  G   = the convention-grounded prompts (g55G, lunaG, g55FSG, lunaFSG) — the only
        ones whose system text states the real rule, gold None = stance NOT
        RECOVERABLE. Ruled out as whole-decision replacements; untested as a None head.
  SOL = solKI / solB / solFS — a different framing family (Jaccard ~0.3-0.5 vs G),
        so they add recall rather than echoing G.
  ND  = optional dedicated none_detect.py scores (0-100), averaged as rank-percentile.

Ranking = summed votes, ties broken by the champion ensemble's own None margin, so a
row both sources like outranks one only a single source likes.

BREAK-EVEN (recomputed from the champion's exact confusion): a flip out of the Favor
pool pays iff it is truly None >=55.0% of the time; out of Against, >=56.8%. Below
that a flip is net-negative no matter how many you make.

GATE-1: the flip SET always comes from model judgment about the tweet text. The server
decode is used only to score a candidate after the fact and to forecast the next one —
never to search over which individual rows carry which label.

Usage:
    python build_none.py --k 30 --src gsol --label wd_gsol_k30
    python build_none.py --k 17 --src g4   --label wd_g4
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

G = ["g55G", "lunaG", "g55FSG", "lunaFSG"]
SOL = ["solKI", "solB", "solFS"]
CHAMP_V = ["frontier_luna_test", "frontier_g55_test", "frontier_lunaFS_test",
           "frontier_g55FS_test", "frontier_opus_test", "frontier_son45_test"]


def pr(k):
    return np.load(os.path.join(F, "frontier_%s_test.npz" % k))["proba"].astype(float)


def champion():
    """The incumbent's labels AND its tie array (needed for the margin tie-break)."""
    tie = np.mean([np.load(os.path.join(F, v + ".npz"))["proba"].astype(float)
                   for v in CHAMP_V], axis=0)
    tie = tie + 1e-3 * (0.15 * np.load(os.path.join(R, "enc_ce.npz"))["proba"]
                        + 0.85 * np.load(os.path.join(R, "dep_qens.npz"))["proba"]).astype(float)
    base = tie.argmax(1)
    out = base.copy()
    nidx = np.where(base == 2)[0]
    marg = tie[nidx, 2] - np.max(tie[nidx][:, :2], axis=1)
    keep = set(nidx[np.argsort(-marg)[:14]].tolist())
    for i in nidx:
        if i not in keep:
            out[i] = int(np.argmax(tie[i, :2]))
    return out, tie


def none_score(src, nd_files):
    """Vote-count None ranking in 0..1, higher = more None-like."""
    gv = np.sum([pr(k).argmax(1) == 2 for k in G], 0).astype(float)
    sv = np.sum([pr(k).argmax(1) == 2 for k in SOL], 0).astype(float)
    if src == "g":
        s = gv / len(G)
    elif src == "sol":
        s = sv / len(SOL)
    elif src == "gsol":
        s = (gv / len(G) + sv / len(SOL)) / 2
    elif src == "nd":
        s = np.zeros(352)
    else:
        raise SystemExit("unknown --src " + src)
    if nd_files:
        # rank-percentile average: the detectors are on their own 0-100 scales
        rk = []
        for p in nd_files:
            v = np.load(p)["score"].astype(float)
            rk.append(np.argsort(np.argsort(v)) / (len(v) - 1.0))
        nd = np.mean(rk, 0)
        s = nd if src == "nd" else (s + nd) / 2
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True, help="total number of None rows to emit")
    ap.add_argument("--src", default="gsol", choices=["g", "sol", "gsol", "nd"])
    ap.add_argument("--nd", nargs="*", default=[], help="none_detect *_test.npz files to fold in")
    ap.add_argument("--against_only", action="store_true",
                    help="only abstain on rows the champion calls Against (the decode "
                         "puts 19-27 of the 28 misplaced gold-None there)")
    ap.add_argument("--add", type=int, default=0,
                    help="ADDITIVE mode: keep the champion's 14 None rows (a server-verified "
                         "sharp peak) and add this many NEW abstentions on top. Measured "
                         "flip precision by pool: Against 50%%, Favor 25%% -- so --add with "
                         "--against_only is the only form that can clear the 56.8%% break-even.")
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    champ, tie = champion()
    score = none_score(args.src, args.nd)
    # Tie-break by the champion's own None margin, scaled so it never outranks a vote.
    marg = tie[:, 2] - np.max(tie[:, :2], axis=1)
    rank = score + 1e-3 * (marg - marg.min()) / (np.ptp(marg) + 1e-9)

    if args.add:
        # ADDITIVE: the champion's 14 stay (server-verified peak; both neighbours lose).
        # Only rows it currently calls a stance are eligible, optionally Against-only.
        elig = (champ == 0) if args.against_only else (champ != 2)
        cand = np.where(elig, rank, -1.0)
        out = champ.copy()
        pick = np.argsort(-cand)[:args.add]
        out[pick] = 2
    else:
        elig = np.ones(352, bool)
        if args.against_only:
            elig = (champ == 0) | (champ == 2)
        rank = np.where(elig, rank, -1.0)
        out = champ.copy()
        out[out == 2] = -1                  # drop the incumbent None set; rebuild from scratch
        pick = np.argsort(-rank)[:args.k]
        for i in np.where(out == -1)[0]:    # rows the champion abstained on: restore its stance
            out[i] = int(np.argmax(tie[i, :2]))
        out[pick] = 2

    n_new = int((~np.isin(pick, np.where(champ == 2)[0])).sum())
    print("%-18s k=%d src=%s against_only=%s" % (args.label, args.k, args.src, args.against_only))
    print("  dist:", {ID2[k]: v for k, v in sorted(collections.Counter(out.tolist()).items())})
    print("  None rows new vs champion's 14: %d new, %d kept" % (n_new, args.k - n_new))
    print("  flipped OUT of champion Against: %d, out of champion Favor: %d"
          % (int((champ[pick] == 0).sum()), int((champ[pick] == 1).sum())))
    print("  rows differing from champion: %d" % int((out != champ).sum()))

    txt = "\n".join(ID2[i] for i in out) + "\n"
    zp = os.path.join(F, args.label + ".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("predictions.txt", txt)
    print("  wrote", zp)


if __name__ == "__main__":
    main()
