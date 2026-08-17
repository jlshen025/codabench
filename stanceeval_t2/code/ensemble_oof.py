"""
ensemble_oof.py — Prototype soft-vote ensembles from saved OOF probabilities (no retrain).

Sources are named by run-dir prefix "<tag>_<token>" (e.g. base_araberttw, mtl_marbertv2).
For each source we seed-average all matching runs, then score singles and chosen combos.
Instance order is deterministic per cv → aligned across runs (asserted via y_true).
"""
import os, glob, numpy as np, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LBL = {"base_araberttw": "AraBERT", "base_marbertv2": "MARBERT", "base_camelbert": "CAMeL",
       "mtl_araberttw": "AraBERT·mtl", "mtl_marbertv2": "MARBERT·mtl"}


def load(prefix, cv):
    files = sorted(glob.glob(os.path.join(ROOT, f"{prefix}_{cv}_s*", "oof_proba.npz")))
    if not files:
        return None
    P, y = [], None
    for f in files:
        d = np.load(f, allow_pickle=True); P.append(d["proba"])
        y = d["y_true"] if y is None else y
        assert np.array_equal(y, d["y_true"]), f"order mismatch {f}"
    return np.mean(P, 0), y


def score(proba, y, tag):
    m = S.compute_metrics(y, proba.argmax(1))
    print(f"  {tag:32s} Favg2={m['Favg2']*100:6.2f}  Favg3={m['Favg3']*100:6.2f}")
    return m["Favg2"] * 100


COMBOS = [
    ("base_araberttw", "base_marbertv2"),
    ("mtl_araberttw", "mtl_marbertv2"),
    ("base_araberttw", "mtl_araberttw"),
    ("base_marbertv2", "mtl_marbertv2"),
    ("base_araberttw", "base_marbertv2", "mtl_araberttw", "mtl_marbertv2"),
    ("base_araberttw", "base_marbertv2", "base_camelbert", "mtl_araberttw", "mtl_marbertv2"),
]

for cv in ["dev", "loto"]:
    print(f"\n===== cv={cv} =====")
    src = {}
    yref = None
    for pfx in LBL:
        r = load(pfx, cv)
        if r is None:
            continue
        src[pfx], yref = r[0], r[1]
        score(src[pfx], yref, LBL[pfx] + " (single, seed-avg)")
    print("  --- soft-vote ensembles ---")
    best = (0, None)
    for combo in COMBOS:
        if all(p in src for p in combo):
            avg = np.mean([src[p] for p in combo], 0)
            f = score(avg, yref, "+".join(LBL[p] for p in combo))
            if f > best[0]:
                best = (f, combo)
    print(f"  >> BEST {cv}: {best[0]:.2f} = {'+'.join(LBL[p] for p in best[1])}")
