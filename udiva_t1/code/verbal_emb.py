#!/usr/bin/env python3
"""Verbal base model (SHIPPED): multilingual-e5 embeddings + per-attribute LR + candidate
prior-blend. Embeds (seg,speaker) texts ONCE, then LOSO grid over blend a, K.

ROLE on the shipped path: `embed_all` (frozen e5 encoder), `fit_heads` (one class-balanced
logistic-regression head per attribute value u/target/modifier) and `proba_rows` (marginals) are
THE verbal base model used by every verbal reranker; main() is the a/K selection grid that chose
blend a=0.5."""
import os, json, argparse, time, warnings
import numpy as np
from collections import Counter
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from udiva import build_all_gt, compute_map, SESSIONS
from verbal_model import session_instances
from verbal_v2 import build_cands

def embed_all(G, model_name, dev):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name, device=dev)
    txt = ["query: " + g[3] for g in G]
    E = m.encode(txt, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    return np.asarray(E, np.float32)

def fit_heads(E, idx, G, cands, C):
    uvoc = sorted({t[0] for t in cands}); tvoc = sorted({t[1] for t in cands}); mvoc = sorted({t[2] for t in cands})
    X = E[idx]
    Us = [G[i][4] for i in idx]; Ts = [G[i][5] for i in idx]; Ms = [G[i][6] for i in idx]
    clf = {"u": {}, "t": {}, "m": {}}
    for name, labs, voc in [("u", Us, uvoc), ("t", Ts, tvoc), ("m", Ms, mvoc)]:
        for c in voc:
            y = np.array([1 if c in s else 0 for s in labs])
            if y.sum() == 0 or y.sum() == len(y):
                clf[name][c] = float(y.mean())
            else:
                lr = LogisticRegression(C=C, max_iter=300, solver="liblinear", class_weight="balanced")
                lr.fit(X, y); clf[name][c] = lr
    return clf

def proba_rows(clf, E, idx):
    X = E[idx]; out = {}
    for name in ("u", "t", "m"):
        out[name] = {c: (np.full(len(X), v) if isinstance(v, float) else v.predict_proba(X)[:, 1])
                     for c, v in clf[name].items()}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--C", type=float, default=4.0); ap.add_argument("--min_count", type=int, default=2)
    ap.add_argument("--out", default=os.environ.get("EXP_OUTPUT_DIR", "."))
    a = ap.parse_args()
    import torch; dev = "cuda" if torch.cuda.is_available() else "cpu"
    gt = build_all_gt()
    inst_by = {s: session_instances(gt, s) for s in SESSIONS}
    G = [(s, x[0], x[1], x[2], x[3], x[4], x[5]) for s in SESSIONS for x in inst_by[s]]  # sid,seg,spk,text,U,T,M
    t0 = time.time(); E = embed_all(G, a.model, dev); temb = time.time() - t0
    sid_of = np.array([g[0] for g in G])
    GRID = [(b, K) for b in (1.0, 0.7, 0.5, 0.3) for K in (20, 40)]
    per_cfg = {f"a{b}_K{K}": {} for b, K in GRID}
    for test in SESSIONS:
        tr = np.where(sid_of != test)[0]; te = np.where(sid_of == test)[0]
        train = [s for s in SESSIONS if s != test]
        cands, base = build_cands(gt, train, a.min_count)
        clf = fit_heads(E, tr, G, cands, a.C)
        P = proba_rows(clf, E, te)
        logbr = {t: np.log(base[t] + 1e-9) for t in cands}
        # predictions
        for b, K in GRID:
            pred = {"verbal": {test: {}}, "nonverbal": {test: {}}}
            for seg in gt[test]["verbal"]:
                pred["verbal"][test][seg] = []
            for r, i in enumerate(te):
                seg = G[i][1]; spk = G[i][2]
                scored = []
                for (u, tg, mod) in cands:
                    pu = P["u"][u][r] if u in P["u"] else 1e-9
                    pt = P["t"][tg][r] if tg in P["t"] else 1e-9
                    pm = P["m"][mod][r] if mod in P["m"] else 1e-9
                    ls = b * np.log(pu * pt * pm + 1e-9) + (1 - b) * logbr[(u, tg, mod)]
                    scored.append((ls, (spk, u, tg, mod)))
                scored.sort(key=lambda x: -x[0])
                for ls, tup in scored[:K]:
                    pred["verbal"][test][seg].append((tup, float(ls)))
            per_cfg[f"a{b}_K{K}"][test] = compute_map(pred, gt, sessions=[test], class_set="gt")["verbal"]
    res = {"v2_best": 0.0427, "tfidf_v1": 0.0242, "emb_secs": temb, "device": dev, "model": a.model, "grid": {}}
    best = (None, -1)
    for cfg, pf in per_cfg.items():
        v = np.array(list(pf.values())); res["grid"][cfg] = {"mean": float(v.mean()), "std": float(v.std())}
        if v.mean() > best[1]: best = (cfg, float(v.mean()))
    res["best_cfg"], res["best_map"] = best
    os.makedirs(a.out, exist_ok=True); json.dump(res, open(f"{a.out}/result.json", "w"), indent=1)
    for cfg in sorted(res["grid"], key=lambda c: -res["grid"][c]["mean"]):
        print(f"  {cfg}: {res['grid'][cfg]['mean']:.4f} ± {res['grid'][cfg]['std']:.4f}")
    print(f"BEST {best[0]}={best[1]:.4f} (v2 0.0427) emb {temb:.0f}s on {dev}")

if __name__ == "__main__":
    main()
