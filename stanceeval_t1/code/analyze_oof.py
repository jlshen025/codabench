"""
analyze_oof.py — Post-hoc analysis of saved OOF probabilities (no GPU).
Quantifies the headroom from per-class decision-threshold tuning for Favg2.

Reports argmax baseline, then an ORACLE grid-search over additive class offsets
(optimistic upper bound, tuned on the same data) so we can decide whether honest
(nested) threshold tuning is worth pursuing. Order: [Against, Favor, None].
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S


def grid_best(proba, y, og=np.arange(-0.6, 0.61, 0.04)):
    base = S.compute_metrics(y, proba.argmax(1))
    best = (base["Favg2"], (0.0, 0.0, 0.0), base)
    # fix Favor offset = 0 (only relative offsets matter); search Against & None
    for oa in og:
        for on in og:
            pred = (proba + np.array([oa, 0.0, on])).argmax(1)
            m = S.compute_metrics(y, pred)
            if m["Favg2"] > best[0]:
                best = (m["Favg2"], (round(oa, 3), 0.0, round(on, 3)), m)
    return base, best


def main(path):
    d = np.load(path, allow_pickle=True)
    proba, y, tgt = d["proba"], d["y_true"], d["target"].astype(str)
    base, (bf2, off, bm) = grid_best(proba, y)
    print(f"\n=== {os.path.relpath(path)} (N={len(y)}) ===")
    print(f"argmax    : Favg2={base['Favg2']*100:.2f} Favg3={base['Favg3']*100:.2f} "
          f"F_fav={base['F_favor']*100:.1f} F_agn={base['F_against']*100:.1f} F_non={base['F_none']*100:.1f}")
    print(f"ORACLE thr: Favg2={bf2*100:.2f} (+{(bf2-base['Favg2'])*100:.2f})  offsets[Agn,Fav,Non]={off}  "
          f"F_fav={bm['F_favor']*100:.1f} F_agn={bm['F_against']*100:.1f} F_non={bm['F_none']*100:.1f}")
    # pred-distribution under argmax
    pa = proba.argmax(1)
    import collections
    print("argmax pred dist:", {S.ID2LABEL[k]: int(v) for k, v in sorted(collections.Counter(pa).items())})
    print("gold dist       :", {S.ID2LABEL[k]: int(v) for k, v in sorted(collections.Counter(y.tolist()).items())})


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
