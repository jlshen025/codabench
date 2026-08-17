"""
optpoint_test.py — honest nested-CV re-test of an OPERATING-POINT lever
(per-class additive offsets) on the CURRENT Qwen-ens held-best blend.

Scope-correction (2026-07-16): the 07-01 threshold rule-out was measured on the
enc+ds 86.34 system; this re-runs it on enc(4×MTL) ⊕ Qwen-ens(ar+lat,ep3).

Lever: add [0, tF, tN] to the blended proba before argmax (Against fixed=0 since
argmax is shift-invariant; Favg2 excludes None, but tN shifts None↔{Fav,Agn}
boundaries which DO move Favg2 via None false-positives/negatives).

Protocol (Gate-2 honest): repeated StratifiedKFold; the operating point is tuned
ONLY on inner-train indices and scored on the held-out fold, pooled. Reports:
  - baseline (blend weight only, no offset)          == held-best 89.78
  - nested (w, tF, tN) jointly tuned                  == honest value of the lever
  - offsets-only at fixed deploy w=0.15               == marginal at the deploy point
  - naive leaky full-fit max                          == overfit upper bound
Promote ONLY a gauge-robust nested win over 89.78.
"""
import argparse
import numpy as np
import blend_ens as BE
import stance_lib as S
from sklearn.model_selection import StratifiedKFold


def blend2(enc, llm, w):
    return w * enc + (1 - w) * llm


def apply_off(proba, tF, tN):
    p = proba.copy()
    p[:, 1] += tF
    p[:, 2] += tN
    return p


def favg2(y, proba):
    return S.compute_metrics(y, proba.argmax(1))["Favg2"]


def nested(enc, llm, y, tgt, wgrid, tgrid, use_off=True, fixed_w=None, folds=5, repeats=3):
    strat = np.array([f"{a}|{b}" for a, b in zip(y, tgt)])
    scores = []
    for r in range(repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=100 + r)
        pred = np.zeros(len(y), dtype=int)
        for tr, va in skf.split(np.zeros(len(y)), strat):
            best = (-1.0, None)
            ws = [fixed_w] if fixed_w is not None else wgrid
            tgs = tgrid if use_off else [0.0]
            for w in ws:
                pb = blend2(enc, llm, w)
                for tF in tgs:
                    for tN in tgs:
                        f = favg2(y[tr], apply_off(pb, tF, tN)[tr])
                        if f > best[0]:
                            best = (f, (w, tF, tN))
            w, tF, tN = best[1]
            pred[va] = apply_off(blend2(enc, llm, w), tF, tN)[va].argmax(1)
        scores.append(S.compute_metrics(y, pred)["Favg2"] * 100)
    return float(np.mean(scores)), float(np.std(scores))


def naive_max(enc, llm, y, wgrid, tgrid, fixed_w=None):
    best = (-1.0, None)
    ws = [fixed_w] if fixed_w is not None else wgrid
    for w in ws:
        pb = blend2(enc, llm, w)
        for tF in tgrid:
            for tN in tgrid:
                f = favg2(y, apply_off(pb, tF, tN))
                if f > best[0]:
                    best = (f, (w, tF, tN))
    return best[0] * 100, best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_bases", default="qwen14,q14lat")
    ap.add_argument("--seeds", default="42,1")
    ap.add_argument("--ep", type=int, default=3)
    ap.add_argument("--enc_dirs", default="", help="comma sep override for enc OOF dirs (else the 4 MTL default)")
    args = ap.parse_args()
    bases = args.llm_bases.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.enc_dirs:
        P, y, t = [], None, None
        for d in args.enc_dirs.split(","):
            z = np.load(f"{BE.L}/{d}/oof_proba.npz", allow_pickle=True)
            P.append(z["proba"].astype(float))
            if y is None:
                y, t = z["y_true"].astype(int), z["target"].astype(str)
        enc = np.mean(P, 0)
    else:
        enc, y, t = BE.enc_comp()
    llm = BE.llm_ens(bases, seeds, args.ep)

    wgrid = [i / 100 for i in range(0, 101, 5)]
    tgrid = [round(x, 3) for x in np.arange(-0.15, 0.1501, 0.03)]

    base_m, base_s = nested(enc, llm, y, t, wgrid, tgrid, use_off=False)
    off_m, off_s = nested(enc, llm, y, t, wgrid, tgrid, use_off=True)
    offw_m, offw_s = nested(enc, llm, y, t, wgrid, tgrid, use_off=True, fixed_w=0.15)
    nm, npar = naive_max(enc, llm, y, wgrid, tgrid)

    print(f"components: enc Favg2={favg2(y, enc)*100:.2f}  llm Favg2={favg2(y, llm)*100:.2f}")
    print(f"[baseline  w-only, no offset ] nested-CV Favg2 = {base_m:.2f} ± {base_s:.2f}")
    print(f"[lever     w + offsets tuned ] nested-CV Favg2 = {off_m:.2f} ± {off_s:.2f}   Δ={off_m-base_m:+.2f}")
    print(f"[lever     offsets @ w=0.15  ] nested-CV Favg2 = {offw_m:.2f} ± {offw_s:.2f}   Δ={offw_m-base_m:+.2f}")
    print(f"[naive leaky full-fit max    ] Favg2 = {nm:.2f}  at (w,tF,tN)={npar}  (overfit upper bound)")
    print(f"\nHELD-BEST to beat: 89.78 | offset lever honest Δ = {off_m-89.78:+.2f} "
          f"({'PROMOTE' if off_m-base_s > 89.78 else 'RULED-OUT (within/below noise)'})")


if __name__ == "__main__":
    main()
