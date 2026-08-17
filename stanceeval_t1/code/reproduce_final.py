#!/usr/bin/env python
"""Reproduce the FINAL StanceEval-2026 Track-1 entry from on-disk voter arrays.

Submission 869442, blind-test Favg2 = 0.891160 (2nd of 29, evaluation phase 29242).

Architecture
------------
    tie   = mean(luna, g55, lunaFS, g55FS, opus-4-8, sonnet-4.5) + 1e-3 * ce
    base  = argmax(tie)
    pred  = None-count projection of base to exactly N=14 None rows
            (keep the 14 largest None-margin rows; flip the rest to their
             strongest committed class)

`ce` is the seen-target fine-tuned pipeline (0.15 * MTL-encoder-ensemble +
0.85 * Qwen2.5-14B-LoRA prompt-ensemble). Its 1e-3 weight means it can only
break exact ties among the six voters -- it is the ARBITER, not a voter. Alone
on this test it scores 0.7762; as arbiter it beats every alternative (0.8788
for a cross-lab arbiter, 0.8758 few-shot, 0.8722 encoder).

Verified 2026-07-30: this script reproduces submission/869442__wd_last_swap45.zip
BIT-EXACTLY (352/352 labels).

Usage
-----
    python reproduce_final.py            # verify against the submitted zip
    python reproduce_final.py --out X.zip  # rebuild the submission zip

Regenerating the voter arrays from scratch (needs network + API keys) is
documented in REPRODUCE.md; this script starts from the cached arrays so the
result is reproducible without API access.
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
N_NONE = 14
EPS = 1e-3

# The six voters of the final ensemble, in order.
VOTERS = [
    ("frontier_luna_test",   "gpt-5.6-luna, zero-shot"),
    ("frontier_g55_test",    "gpt-5.5, zero-shot"),
    ("frontier_lunaFS_test", "gpt-5.6-luna, few-shot k=12 (batch retrieval)"),
    ("frontier_g55FS_test",  "gpt-5.5, few-shot k=12 (batch retrieval)"),
    ("frontier_opus_test",   "claude-opus-4-8, zero-shot  [cross-lab]"),
    ("frontier_son45_test",  "claude-sonnet-4.5, zero-shot [cross-lab]"),
]


def proba(name):
    return np.load(os.path.join(F, name + ".npz"))["proba"].astype(float)


def arbiter():
    """Seen-target fine-tuned pipeline: 0.15 * encoders + 0.85 * Qwen-14B LoRA."""
    enc = np.load(os.path.join(R, "enc_ce.npz"))["proba"]
    qens = np.load(os.path.join(R, "dep_qens.npz"))["proba"]
    return (0.15 * enc + 0.85 * qens).astype(float)


def build(n_none=N_NONE):
    tie = np.mean([proba(v) for v, _ in VOTERS], axis=0) + EPS * arbiter()
    base = tie.argmax(1)
    nidx = np.where(base == 2)[0]
    margin = tie[nidx, 2] - np.max(tie[nidx][:, :2], axis=1)
    keep = set(nidx[np.argsort(-margin)[:n_none]].tolist())
    out = base.copy()
    for i in nidx:
        if i not in keep:
            out[i] = int(np.argmax(tie[i, :2]))
    return [ID2[i] for i in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="", help="write a FLAT submission zip here")
    ap.add_argument("--n", type=int, default=N_NONE)
    ap.add_argument(
        "--verify",
        # submission/codabench/ mirrors each zip exactly as it was submitted.
        default=os.path.join(HERE, "..", "submission", "codabench",
                             "869442__wd_last_swap45.zip"),
    )
    a = ap.parse_args()

    pred = build(a.n)
    print(f"built {len(pred)} labels  dist={dict(collections.Counter(pred))}")

    if os.path.exists(a.verify):
        with zipfile.ZipFile(a.verify) as z:
            gold = [l.strip() for l in z.read("predictions.txt").decode().splitlines() if l.strip()]
        diff = sum(x != y for x, y in zip(pred, gold))
        print(f"vs submitted {os.path.basename(a.verify)}: "
              f"{'EXACT MATCH' if diff == 0 and len(pred) == len(gold) else f'{diff} differences'}")
    else:
        print(f"[skip verify] {a.verify} not found")

    if a.out:
        with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("predictions.txt", "\n".join(pred) + "\n")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
