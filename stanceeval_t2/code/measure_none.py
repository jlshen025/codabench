"""Measure the None-hedge on a GOLD-labeled sol-grounded npz (proba/pred_id/target/y_true).
Favg2 excludes None, so a true-Favor/Against tweet predicted None = pure recall loss.
Reports per target: true vs predicted None rate, hedge false-negatives, and the Favg2
UPPER BOUND if every hedged None were reassigned to its gold stance (oracle) — the headroom."""
import numpy as np, sys
import stance_lib as S
ID2L = {0: "Against", 1: "Favor", 2: "None"}

def favg2(y, p):
    return S.compute_metrics(np.asarray(y), np.asarray(p))["Favg2"] * 100

for path in sys.argv[1:]:
    z = np.load(path, allow_pickle=True)
    if "y_true" not in z:
        print(f"{path}: no y_true, skip"); continue
    y = z["y_true"]; pred = z["pred_id"]; tgt = z["target"].astype(str)
    print(f"\n==== {path}  (n={len(y)}) ====")
    for t in ["ALL"] + list(dict.fromkeys(tgt.tolist())):
        m = np.ones(len(y), bool) if t == "ALL" else (tgt == t)
        yy, pp = y[m], pred[m]
        n = m.sum()
        true_none = (yy == 2).mean() * 100
        pred_none = (pp == 2).mean() * 100
        hedge_fn = int(((yy != 2) & (pp == 2)).sum())      # true stance -> pred None (recall loss)
        false_none = int(((yy == 2) & (pp != 2)).sum())    # true None -> pred stance
        base = favg2(yy, pp)
        # oracle: reassign every pred-None back to its GOLD label (upper bound of perfect None-suppression)
        orc = pp.copy(); orc[pp == 2] = yy[pp == 2]
        oracle = favg2(yy, orc)
        # realistic: force pred-None -> the tweet's non-None argmax (needs soft proba; one-hot has none -> skip)
        print(f"  {t:22s} n={n:4d}  trueNone={true_none:5.1f}%  predNone={pred_none:5.1f}%  "
              f"hedgeFN(stance->None)={hedge_fn:3d}  falseNone(None->stance)={false_none:3d}  "
              f"Favg2={base:5.2f}  oracleNoneFix={oracle:5.2f}  (+{oracle-base:.2f})")
