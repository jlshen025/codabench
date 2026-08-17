"""
blend_cv.py — HONEST blend evaluation for the enc + LLM ensemble (Gate-2 compliant).

The naive "search weights on all of dev, report the max" overfits dev. Instead:
  * LLM components are SEED-AVERAGED; the epoch is chosen from the seed-averaged
    dev-Favg2 curve (one coarse hyperparameter, clear overfit-after-ep2 pattern).
  * Blend WEIGHTS are selected by NESTED CV: repeated stratified K-fold on dev,
    weights tuned on the inner (train) folds, scored on the held-out fold, pooled.
    -> an unbiased estimate of how the tuned blend generalizes to a fresh SEEN split.
Reports, per component set: nested-CV Favg2 mean±std (the number to trust) AND the
full-dev-fit weights+score (what you'd deploy) AND the naive grid-max (overfit upper bound).
"""
import os, sys, itertools, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
L = os.path.join(HERE, "results")
LL = os.path.join(HERE, "_llm")


def npz(p):
    return np.load(p, allow_pickle=True)


def enc_comp():
    P, y, t = [], None, None
    for m in ["marbertv2", "araberttw"]:
        for s in [1, 42]:
            d = npz(f"{L}/mtl_{m}_dev_s{s}/oof_proba.npz")
            P.append(d["proba"].astype(float))
            if y is None:
                y, t = d["y_true"].astype(int), d["target"].astype(str)
    return np.mean(P, 0), y, t


def llm_comp(base, seeds, epoch=None):
    P = []
    for s in seeds:
        d = npz(f"{L}/lora_{base}_dev_s{s}/oof_proba.npz")
        key = f"proba_ep{epoch}" if epoch and f"proba_ep{epoch}" in d.files else "proba"
        P.append(d[key].astype(float))
    return np.mean(P, 0)


def favg2(proba, y, mask=None):
    yy, pp = (y, proba) if mask is None else (y[mask], proba[mask])
    return S.compute_metrics(yy, pp.argmax(1))["Favg2"] * 100


def per_target(proba, y, tgt):
    return {t.split()[0]: round(favg2(proba, y, tgt == t), 1) for t in sorted(set(tgt.tolist()))}


def choose_epoch(base, seeds, y):
    d = npz(f"{L}/lora_{base}_dev_s{seeds[0]}/oof_proba.npz")
    eps = sorted(int(k.split("ep")[1]) for k in d.files if k.startswith("proba_ep"))
    res = {ep: round(favg2(llm_comp(base, seeds, ep), y), 3) for ep in eps}  # real epochs only
    best = max(res, key=res.get) if res else None
    return best, res


def simplex(n, K):
    def rec(n, K):
        if n == 1:
            yield (K,); return
        for i in range(K + 1):
            for rest in rec(n - 1, K - i):
                yield (i,) + rest
    for c in rec(n, K):
        yield tuple(x / K for x in c)


def blend(comps, w):
    return sum(wi * c for wi, c in zip(w, comps))


def fit_weights(comps, y, mask=None, K=20):
    idx = np.arange(len(y)) if mask is None else mask
    best = (-1, None)
    for w in simplex(len(comps), K):
        f = S.compute_metrics(y[idx], blend(comps, w)[idx].argmax(1))["Favg2"]
        if f > best[0]:
            best = (f, w)
    return best[0] * 100, best[1]


def nested_cv(comps, y, tgt, K=10, folds=5, repeats=3):
    strat = np.array([f"{a}|{b}" for a, b in zip(y, tgt)])
    grid = list(simplex(len(comps), K))
    scores = []
    for r in range(repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=100 + r)
        pred = np.zeros(len(y), dtype=int)
        for tr, va in skf.split(np.zeros(len(y)), strat):
            best = (-1, None)
            for w in grid:
                f = S.compute_metrics(y[tr], blend(comps, w)[tr].argmax(1))["Favg2"]
                if f > best[0]:
                    best = (f, w)
            pred[va] = blend(comps, best[1])[va].argmax(1)
        scores.append(S.compute_metrics(y, pred)["Favg2"] * 100)
    return float(np.mean(scores)), float(np.std(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,1")
    ap.add_argument("--allam_ep", type=int, default=0, help="0=auto from seed-avg curve")
    ap.add_argument("--qwen_ep", type=int, default=0)
    ap.add_argument("--fit_K", type=int, default=20)
    ap.add_argument("--cv_K", type=int, default=10)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    enc, y, tgt = enc_comp()
    ds = npz(f"{LL}/deepseek_dev.npz")["proba"].astype(float)
    ae, ac = choose_epoch("allam", seeds, y)
    qe, qc = choose_epoch("qwen", seeds, y)
    allam_ep = args.allam_ep or ae
    qwen_ep = args.qwen_ep or qe
    allam = llm_comp("allam", seeds, allam_ep)
    qwen = llm_comp("qwen", seeds, qwen_ep)

    print(f"ALLaM seed-avg epoch curve: {ac} -> use ep{allam_ep}")
    print(f"Qwen  seed-avg epoch curve: {qc} -> use ep{qwen_ep}")
    print("\nSOLO components:")
    for name, p in [("enc(4MTL)", enc), ("deepseek", ds),
                    (f"allam(avg ep{allam_ep})", allam), (f"qwen(avg ep{qwen_ep})", qwen),
                    ("enc+ds.3 [HELD-BEST]", 0.7 * enc + 0.3 * ds)]:
        print(f"  {name:24s} Favg2={favg2(p, y):.2f}  {per_target(p, y, tgt)}")

    sets = {
        "enc+allam": [enc, allam],
        "enc+ds+allam": [enc, ds, allam],
        "enc+allam+qwen": [enc, allam, qwen],
        "enc+ds+allam+qwen": [enc, ds, allam, qwen],
    }
    print("\nHONEST nested-CV (weights tuned inner, scored outer) vs full-fit vs naive-max:")
    print(f"{'set':22s} {'nestedCV(mean±std)':20s} {'full-fit':>9s}  weights(full-fit)")
    results = {}
    for name, comps in sets.items():
        cv_m, cv_s = nested_cv(comps, y, tgt, K=args.cv_K)
        ff, fw = fit_weights(comps, y, K=args.fit_K)
        results[name] = (cv_m, cv_s, ff, fw)
        print(f"{name:22s} {cv_m:6.2f} ± {cv_s:4.2f}       {ff:6.2f}  {tuple(round(w,2) for w in fw)}")
    # per-target of the best full-fit set
    best_name = max(results, key=lambda k: results[k][0])   # by honest nested-CV
    fw = results[best_name][3]
    p = blend(sets[best_name], fw)
    print(f"\nBEST by nested-CV: {best_name}  nestedCV={results[best_name][0]:.2f}  "
          f"full-fit={results[best_name][2]:.2f}  per-target={per_target(p, y, tgt)}")
    print(f"HELD-BEST to beat: 86.34   |   Δ nested-CV = {results[best_name][0]-86.34:+.2f}")


if __name__ == "__main__":
    main()
