"""
blend_ens.py — reproducible selector for the CURRENT held-best:
    enc(4×MTL) ⊕ Qwen-14B-LoRA PROMPT-ENSEMBLE.

The Qwen prompt-ensemble is the mean over one or more verbalizer prompts
(each seed-averaged at a chosen epoch): ar_letter=`qwen14`, lat_letter=`q14lat`,
digit=`q14dig`. Prompt diversity helps via AVERAGING (the members are ~0.96-corr
but their mean decorrelates the errors).

Selection is HONEST (Gate-2): the enc↔LLM blend weight is chosen by repeated
nested StratifiedKFold on dev (weight tuned on inner folds, scored on the held-out
fold, pooled) — an unbiased estimate of generalization to a fresh SEEN split.
Reports nested-CV mean±std (the number to trust), full-dev-fit (what you'd deploy),
naive grid-max (overfit upper bound), and per-target.

Usage:
  python blend_ens.py --llm_bases qwen14,q14lat --ep 3         # held-best (89.78)
  python blend_ens.py --llm_bases qwen14,q14lat,q14dig --ep 3  # + digit prompt
"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
L = os.path.join(HERE, "results")


def npz(p):
    return np.load(p, allow_pickle=True)


def enc_comp():
    """Mean of the 4 MTL encoder OOFs (marbertv2 + araberttw, seeds 1,42)."""
    P, y, t = [], None, None
    for m in ["marbertv2", "araberttw"]:
        for s in [1, 42]:
            d = npz(f"{L}/mtl_{m}_dev_s{s}/oof_proba.npz")
            P.append(d["proba"].astype(float))
            if y is None:
                y, t = d["y_true"].astype(int), d["target"].astype(str)
    return np.mean(P, 0), y, t


def base_seedavg(base, seeds, ep):
    """Seed-average a single LLM base at a fixed epoch."""
    P = []
    for s in seeds:
        d = npz(f"{L}/lora_{base}_s{s}/oof_proba.npz")
        key = f"proba_ep{ep}" if ep and f"proba_ep{ep}" in d.files else "proba"
        P.append(d[key].astype(float))
    return np.mean(P, 0)


def llm_ens(bases, seeds, ep):
    """Prompt-ensemble = mean over bases of each base's seed-average."""
    return np.mean([base_seedavg(b, seeds, ep) for b in bases], 0)


def favg2(proba, y, mask=None):
    yy, pp = (y, proba) if mask is None else (y[mask], proba[mask])
    return S.compute_metrics(yy, pp.argmax(1))["Favg2"] * 100


def favg3(proba, y):
    return S.compute_metrics(y, proba.argmax(1))["Favg3"] * 100


def per_target(proba, y, tgt):
    return {t.split()[0]: round(favg2(proba, y, tgt == t), 1) for t in sorted(set(tgt.tolist()))}


def blend2(enc, llm, w_enc):
    return w_enc * enc + (1 - w_enc) * llm


def nested_cv(enc, llm, y, tgt, folds=5, repeats=3, grid=None):
    """Repeated nested CV over the 1-D enc-weight grid."""
    if grid is None:
        grid = [i / 100 for i in range(0, 101, 5)]
    strat = np.array([f"{a}|{b}" for a, b in zip(y, tgt)])
    scores = []
    for r in range(repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=100 + r)
        pred = np.zeros(len(y), dtype=int)
        for tr, va in skf.split(np.zeros(len(y)), strat):
            best = (-1, None)
            for w in grid:
                f = S.compute_metrics(y[tr], blend2(enc, llm, w)[tr].argmax(1))["Favg2"]
                if f > best[0]:
                    best = (f, w)
            pred[va] = blend2(enc, llm, best[1])[va].argmax(1)
        scores.append(S.compute_metrics(y, pred)["Favg2"] * 100)
    return float(np.mean(scores)), float(np.std(scores))


def full_fit(enc, llm, y, grid=None):
    if grid is None:
        grid = [i / 100 for i in range(0, 101, 1)]
    best = (-1, None)
    for w in grid:
        f = S.compute_metrics(y, blend2(enc, llm, w).argmax(1))["Favg2"]
        if f > best[0]:
            best = (f, w)
    return best[0] * 100, best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_bases", default="qwen14,q14lat")
    ap.add_argument("--seeds", default="42,1")
    ap.add_argument("--ep", type=int, default=3)
    args = ap.parse_args()
    bases = args.llm_bases.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    enc, y, tgt = enc_comp()

    print(f"=== components (ep{args.ep}, seeds {seeds}) ===")
    print(f"  {'enc(4MTL)':18s} Favg2={favg2(enc, y):.2f}  Favg3={favg3(enc, y):.2f}  {per_target(enc, y, tgt)}")
    for b in bases:
        # per-epoch seed-avg curve for this base
        d0 = npz(f"{L}/lora_{b}_s{seeds[0]}/oof_proba.npz")
        eps = sorted(int(k.split("ep")[1]) for k in d0.files if k.startswith("proba_ep"))
        curve = {e: round(favg2(base_seedavg(b, seeds, e), y), 2) for e in eps}
        p = base_seedavg(b, seeds, args.ep)
        print(f"  {b:18s} Favg2={favg2(p, y):.2f}  Favg3={favg3(p, y):.2f}  {per_target(p, y, tgt)}  seedavg-curve={curve}")

    llm = llm_ens(bases, seeds, args.ep)
    print(f"\n  {'LLM-ENS(mean)':18s} Favg2={favg2(llm, y):.2f}  Favg3={favg3(llm, y):.2f}  {per_target(llm, y, tgt)}")

    # pairwise correlation of the bases' argmax (diversity check)
    if len(bases) > 1:
        preds = {b: base_seedavg(b, seeds, args.ep).argmax(1) for b in bases}
        print("  pred-agreement between bases:")
        for i in range(len(bases)):
            for j in range(i + 1, len(bases)):
                agr = float(np.mean(preds[bases[i]] == preds[bases[j]]))
                print(f"    {bases[i]} vs {bases[j]}: {agr:.3f}")

    cv_m, cv_s = nested_cv(enc, llm, y, tgt)
    ff, fw = full_fit(enc, llm, y)
    naive_m, naive_w = full_fit(enc, llm, y, grid=[i / 100 for i in range(0, 101, 1)])
    print(f"\n=== enc ⊕ LLM-ENS blend ===")
    print(f"  nested-CV Favg2 = {cv_m:.2f} ± {cv_s:.2f}   (the number to TRUST)")
    print(f"  full-dev-fit    = {ff:.2f}  at w_enc={fw:.2f} (DEPLOY weight)")
    p_deploy = blend2(enc, llm, fw)
    print(f"  full-fit per-target = {per_target(p_deploy, y, tgt)}  Favg3={favg3(p_deploy, y):.2f}")
    print(f"\n  HELD-BEST to beat: 89.78   |   Δ nested-CV = {cv_m - 89.78:+.2f}")


if __name__ == "__main__":
    main()
