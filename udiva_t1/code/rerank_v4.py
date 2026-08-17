#!/usr/bin/env python3
"""rerank v4: v3 features + NSEEDS=11 (shippable nonverbal ens) + GBM reranker
(sklearn HistGradientBoostingClassifier, non-linear) compared to LR and base. Reuses v3
dump functions (full pool + temporal-neighbor + speech-token feats).

ROLE: dev CV-ablation harness that established the HGB reranker + 11-seed ensemble later shipped
via ship2.py. Its `eval_ch` is imported (transitively) by rerank_v7/v8 but is NOT called on the
shipped path; nothing here runs at ship time."""
import os, json, time
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from udiva import build_all_gt, compute_map, SESSIONS
from nonverbal_model import load_feats, instances as nv_instances
from verbal_model import session_instances
from verbal_emb import embed_all
import rerank as R
import rerank_v3 as R3
R3.NSEEDS = 11  # 11-seed nonverbal ensemble (match held-best, shippable)

def eval_ch(gt, cat, rows, heldK, Ns):
    meta = {"sid": [r[0] for r in rows], "seg": [r[1] for r in rows], "subj": [r[2] for r in rows],
            "cls": [r[3] for r in rows], "tup": [r[4] for r in rows]}
    X = np.array([r[5] for r in rows], float); y = np.array([r[6] for r in rows], int)
    bconf = np.array([r[7] for r in rows], float); sid = np.array(meta["sid"])
    scores = {"base": bconf}
    for name in ("lr", "hgb"):
        sc = np.zeros(len(rows)); rng = np.random.default_rng(0)
        for test in R.FOLDS:
            trm = np.where(sid != test)[0]; tem = np.where(sid == test)[0]
            ytr = y[trm]; pos = trm[ytr == 1]; neg = trm[ytr == 0]
            if len(pos) == 0: continue
            sel = np.concatenate([pos, rng.choice(neg, min(len(neg), 20 * len(pos)), replace=False)])
            if name == "lr":
                mu = X[sel].mean(0); sdv = X[sel].std(0) + 1e-9
                clf = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced").fit((X[sel] - mu) / sdv, y[sel])
                sc[tem] = clf.decision_function((X[tem] - mu) / sdv)
            else:
                sw = compute_sample_weight("balanced", y[sel])
                clf = HistGradientBoostingClassifier(max_iter=250, max_depth=4, learning_rate=0.06,
                                                     l2_regularization=1.0, min_samples_leaf=30)
                clf.fit(X[sel], y[sel], sample_weight=sw)
                sc[tem] = clf.predict_proba(X[tem])[:, 1]
        scores[name] = sc
    ngt = sum(len(gt[s][cat][seg][2]) for s in R.FOLDS for seg in gt[s][cat])
    out = {"pool_recall": float(y.sum() / max(ngt, 1))}
    methods = [("base_topK", "base", "topk", heldK, 1)]
    for sn in ("base", "lr", "hgb"):
        methods.append((f"{sn}_top1cls", sn, "topNcls", 0, 1))
        for n in Ns: methods.append((f"{sn}_top{n}cls", sn, "topNcls", 0, n))
    for name, sn, mode, K, N in methods:
        fold = []
        for test in R.FOLDS:
            idxs = np.where(sid == test)[0]
            pred = R.decode(idxs, scores[sn], meta, "topk" if mode == "topk" else "topNcls", K=K, N=N)
            fold.append(compute_map({cat: {test: pred.get(test, {})}}, gt, sessions=[test], class_set="gt")[cat])
        out[name] = {"mean": float(np.mean(fold)), "std": float(np.std(fold))}
    return out

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gt = build_all_gt(); fc = {s: load_feats(s) for s in SESSIONS}
    inst_by = {s: session_instances(gt, s) for s in SESSIONS}
    G = [(s, x[0], x[1], x[2], x[3], x[4], x[5]) for s in SESSIONS for x in inst_by[s]]
    E = embed_all(G, R.VMODEL, dev); sid_of = np.array([g[0] for g in G])
    nv_inst_by = {s: nv_instances(gt, s, fc[s]) for s in SESSIONS}
    t0 = time.time(); vrows, nrows = [], []
    for test in R.FOLDS:
        vrows += R3.dump_verbal(gt, test, G, E, sid_of)
        nrows += R3.dump_nonverbal(gt, test, nv_inst_by, dev)
    res = {"verbal": eval_ch(gt, "verbal", vrows, 40, [2]),
           "nonverbal": eval_ch(gt, "nonverbal", nrows, 80, [2, 3])}
    def best(ch): return max(v["mean"] for k, v in res[ch].items() if k != "pool_recall")
    res["overall_best"] = 0.5 * (best("verbal") + best("nonverbal")); res["secs"] = time.time() - t0
    res["nseeds"] = R3.NSEEDS
    out = os.environ.get("EXP_OUTPUT_DIR", "."); os.makedirs(out, exist_ok=True)
    json.dump(res, open(f"{out}/result.json", "w"), indent=1)
    for ch in ("verbal", "nonverbal"):
        print(f"[{ch}] pool_recall={res[ch]['pool_recall']:.3f}")
        for k in sorted(res[ch], key=lambda z: -res[ch][z]['mean'] if z != 'pool_recall' else 1):
            if k != 'pool_recall': print(f"   {k:16s}: {res[ch][k]['mean']:.4f}")
    print(f"OVERALL_BEST={res['overall_best']:.4f} nseeds={R3.NSEEDS} {res['secs']:.0f}s")

if __name__ == "__main__":
    main()
