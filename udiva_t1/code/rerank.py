#!/usr/bin/env python3
"""RERANKER lineage root (rerank.py -> v3 -> v4 -> v7 -> v8; each ADDS features, they are
COMPLEMENTARY not alternative). ROLE on the shipped path: provides the shared config constants
(POOL, B=8, VA=0.5, NA=0.3, VCREG, VMIN, NMIN, ND*, NSEEDS, VMODEL, EPS, FOLDS) and re-exports
integ_tp.time_table/taus_of used across the lineage. This file's own dump_verbal/dump_nonverbal/
eval_channel + main() are the v1 LR-reranker CV ablation (superseded by v3+; kept for reference).

RERANKER: ranking is the bottleneck
(oracle 0.59/0.62 vs held-best 0.05). Stage-1 = current marginal heads (e5 verbal,
VMAE-5seed-ens nonverbal) -> per-(seg,subj) candidate POOL (top-150 by static base_conf)
with features. Stage-2 = LR over candidate features predicting EXACT-match. Decode:
top-1 per (seg,subj,class) [verbal 98% single] / count-aware top-N [nonverbal].
Ablation per channel:
  base_topK   : score=static base_conf, top-K per (seg,subj)        -> reproduces held-best
  base_top1cls: score=static base_conf, top-1 per (seg,subj,class)  -> decode-only effect
  rr_top1cls  : score=reranker,         top-1 per (seg,subj,class)  -> +reranker
  rr_topNcls  : score=reranker,         top-N per (seg,subj,class)  -> count-aware (nv)
OUTER LOSO; stage-1 scores are per-fold OOF (model trained on the other 20)."""
import os, json, time
import numpy as np
import torch
from collections import Counter
from sklearn.linear_model import LogisticRegression
from udiva import build_all_gt, compute_map, SESSIONS
from nonverbal_model import load_feats, instances as nv_instances
from verbal_model import session_instances
from verbal_emb import embed_all, fit_heads, proba_rows
from verbal_v2 import build_cands
import nv_mlp2 as NV
from integ_tp import time_table, taus_of

POOL = int(os.environ.get("UDIVA_POOL", "100000")); B = 8; VA = 0.5; NA = 0.3  # POOL huge = full candidate set (unthrottle pool_recall)
VCREG = 4.0; VMIN = 2; NMIN = 2; NDH = 512; NPP = 0.3; NEP = 80; NSEEDS = 5
VMODEL = "intfloat/multilingual-e5-base"; EPS = 1e-9
_NF = int(os.environ.get("UDIVA_NFOLDS", "0"))
FOLDS = SESSIONS[:_NF] if _NF else SESSIONS

def dump_verbal(gt, test, G, E, sid_of):
    train = [s for s in SESSIONS if s != test]
    cands, base = build_cands(gt, train, VMIN)
    tr = np.where(np.isin(sid_of, train))[0]
    clf = fit_heads(E, tr, G, cands, VCREG)
    te = np.where(sid_of == test)[0]; P = proba_rows(clf, E, te)
    tab = time_table(gt, train, "verbal", cands, B); tx = taus_of(gt, test, "verbal")
    rows = []
    for r, i in enumerate(te):
        seg, spk = G[i][1], G[i][2]; bb = min(int(tx[seg] * B), B - 1); tau = tx[seg]
        gtset = gt[test]["verbal"][seg][2]
        items = []
        for (u, tg, mod) in cands:
            pu = P["u"][u][r] if u in P["u"] else EPS
            pt = P["t"][tg][r] if tg in P["t"] else EPS
            pm = P["m"][mod][r] if mod in P["m"] else EPS
            pri = tab[(u, tg, mod)][bb]; ms = pu * pt * pm
            conf = (ms + EPS) ** VA * (base[(u, tg, mod)] + EPS) ** (1 - VA)
            logs = (np.log(pu + EPS), np.log(pt + EPS), np.log(pm + EPS),
                    np.log(base[(u, tg, mod)] + EPS), np.log(pri + EPS))
            items.append((conf, (spk, u, tg, mod), u, logs))
        items.sort(key=lambda z: -z[0]); pool = items[:POOL]
        byc = {}
        for conf, tup, cls, logs in pool: byc.setdefault(cls, []).append(conf)
        for c in byc: byc[c].sort(reverse=True)
        for rank, (conf, tup, cls, logs) in enumerate(pool):
            same = byc[cls]; nsame = len(same); second = same[1] if nsame > 1 else 0.0
            feat = [*logs, tau, float(rank), float(nsame), float(conf - second), float(conf)]
            rows.append((test, seg, tup[0], cls, tup, feat, 1 if tup in gtset else 0, conf))
    return rows

def dump_nonverbal(gt, test, nv_inst_by, dev):
    train = [s for s in SESSIONS if s != test]
    cands, base = NV.build_candidates(gt, train, NMIN); vocab = NV.vocabs_from_cands(cands)
    tri = [x for s in train for x in nv_inst_by[s]]; Xtr, Ytr = NV.encode(tri, vocab)
    models = []
    for s in range(NSEEDS):
        m, mu, sd = NV.train_fold(Xtr, Ytr, vocab, dev, NDH, NPP, NEP, seed=s); models.append(m)
    idx = {k: {c: i for i, c in enumerate(v)} for k, v in vocab.items()}
    hi = np.array([idx["h"][t[0]] for t in cands]); li = np.array([idx["l"][t[1]] for t in cands])
    ti = np.array([idx["t"][t[2]] for t in cands]); mi = np.array([idx["m"][t[3]] for t in cands])
    nbr = np.array([base[t] for t in cands])
    tab = time_table(gt, train, "nonverbal", cands, B); tx = taus_of(gt, test, "nonverbal")
    inst = nv_inst_by[test]
    X = (np.stack([z[2] for z in inst]) - mu) / sd; Xt = torch.tensor(X, device=dev)
    Pavg = {k: np.zeros((len(inst), len(vocab[k])), np.float32) for k in vocab}
    for m in models:
        m.eval()
        with torch.no_grad():
            o = m(Xt)
            for k in vocab: Pavg[k] += torch.sigmoid(o[k]).cpu().numpy()
    for k in vocab: Pavg[k] /= len(models)
    ms_all = Pavg["h"][:, hi] * Pavg["l"][:, li] * Pavg["t"][:, ti] * Pavg["m"][:, mi]
    lbr = np.log(nbr + EPS)
    rows = []
    for r, (seg, subj, _, H, L, T, M) in enumerate(inst):
        bb = min(int(tx[seg] * B), B - 1); tau = tx[seg]
        gtset = gt[test]["nonverbal"][seg][2]
        pri = np.array([tab[t][bb] for t in cands])
        ms = ms_all[r]; conf = (ms + EPS) ** NA * (nbr + EPS) ** (1 - NA)
        order = np.argsort(-conf)[:POOL]
        lph = np.log(Pavg["h"][r][hi] + EPS); lpl = np.log(Pavg["l"][r][li] + EPS)
        lpt = np.log(Pavg["t"][r][ti] + EPS); lpm = np.log(Pavg["m"][r][mi] + EPS)
        lpri = np.log(pri + EPS)
        byc = {}
        for j in order: byc.setdefault(cands[j][0], []).append(conf[j])
        for c in byc: byc[c].sort(reverse=True)
        for rank, j in enumerate(order):
            cls = cands[j][0]; same = byc[cls]; second = same[1] if len(same) > 1 else 0.0
            tup = (subj, *cands[j])
            feat = [lph[j], lpl[j], lpt[j], lpm[j], lbr[j], lpri[j], tau, float(rank),
                    float(len(same)), float(conf[j] - second), float(conf[j])]
            rows.append((test, seg, subj, cls, tup, feat, 1 if tup in gtset else 0, float(conf[j])))
    return rows

def decode(idxs, score, meta, mode, K=40, N=1):
    groups = {}
    for r in idxs:
        sid, seg, subj, cls, tup = meta["sid"][r], meta["seg"][r], meta["subj"][r], meta["cls"][r], meta["tup"][r]
        key = (sid, seg, subj) if mode == "topk" else (sid, seg, subj, cls)
        groups.setdefault(key, []).append((score[r], tup))
    pred = {}
    for key, lst in groups.items():
        lst.sort(key=lambda z: -z[0]); sid, seg = key[0], key[1]
        keep = lst[:K] if mode == "topk" else lst[:N]
        pred.setdefault(sid, {}).setdefault(seg, [])
        for sc, tup in keep: pred[sid][seg].append((tup, float(sc)))
    return pred

def eval_channel(gt, cat, rows, heldK, Ns):
    meta = {"sid": [r[0] for r in rows], "seg": [r[1] for r in rows], "subj": [r[2] for r in rows],
            "cls": [r[3] for r in rows], "tup": [r[4] for r in rows]}
    X = np.array([r[5] for r in rows], float); y = np.array([r[6] for r in rows], int)
    bconf = np.array([r[7] for r in rows], float)
    sid_arr = np.array(meta["sid"])
    rr = np.zeros(len(rows)); rng = np.random.default_rng(0)
    for test in FOLDS:
        trm = np.where(sid_arr != test)[0]; tem = np.where(sid_arr == test)[0]
        ytr = y[trm]
        pos = trm[ytr == 1]; neg = trm[ytr == 0]
        if len(pos) == 0: continue
        nkeep = min(len(neg), 20 * len(pos))           # negative subsampling (keeps LR fast on full pool)
        sel = np.concatenate([pos, rng.choice(neg, nkeep, replace=False)])
        mu = X[sel].mean(0); sdv = X[sel].std(0) + 1e-9
        lr = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")
        lr.fit((X[sel] - mu) / sdv, y[sel])
        rr[tem] = lr.decision_function((X[tem] - mu) / sdv)
    # pool recall
    ngt = sum(len(gt[s][cat][seg][2]) for s in FOLDS for seg in gt[s][cat])
    poolrec = y.sum() / max(ngt, 1)
    out = {"pool_recall": float(poolrec)}
    methods = [("base_topK", bconf, "topk", heldK, 1), ("base_top1cls", bconf, "top1cls", 0, 1),
               ("rr_top1cls", rr, "top1cls", 0, 1)] + [(f"rr_top{n}cls", rr, "topNcls", 0, n) for n in Ns]
    for name, score, mode, K, N in methods:
        fold = []
        for test in FOLDS:
            idxs = np.where(sid_arr == test)[0]
            pred = decode(idxs, score, meta, "topk" if mode == "topk" else "topNcls", K=K, N=N)
            fold.append(compute_map({cat: {test: pred.get(test, {})}}, gt, sessions=[test], class_set="gt")[cat])
        out[name] = {"mean": float(np.mean(fold)), "std": float(np.std(fold))}
    return out

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gt = build_all_gt(); fc = {s: load_feats(s) for s in SESSIONS}
    inst_by = {s: session_instances(gt, s) for s in SESSIONS}
    G = [(s, x[0], x[1], x[2], x[3], x[4], x[5]) for s in SESSIONS for x in inst_by[s]]
    E = embed_all(G, VMODEL, dev); sid_of = np.array([g[0] for g in G])
    nv_inst_by = {s: nv_instances(gt, s, fc[s]) for s in SESSIONS}
    t0 = time.time(); vrows, nrows = [], []
    for test in FOLDS:
        vrows += dump_verbal(gt, test, G, E, sid_of)
        nrows += dump_nonverbal(gt, test, nv_inst_by, dev)
    res = {"verbal": eval_channel(gt, "verbal", vrows, 40, [2]),
           "nonverbal": eval_channel(gt, "nonverbal", nrows, 80, [2, 3])}
    # overall = best decode per channel
    def best(ch): return max(v["mean"] for k, v in res[ch].items() if k != "pool_recall")
    res["overall_best"] = 0.5 * (best("verbal") + best("nonverbal"))
    res["secs"] = time.time() - t0
    out = os.environ.get("EXP_OUTPUT_DIR", "."); os.makedirs(out, exist_ok=True)
    json.dump(res, open(f"{out}/result.json", "w"), indent=1)
    for ch in ("verbal", "nonverbal"):
        print(f"[{ch}] pool_recall={res[ch]['pool_recall']:.3f}")
        for k in sorted(res[ch], key=lambda z: -res[ch][z]["mean"] if z != "pool_recall" else 1):
            if k != "pool_recall": print(f"   {k:14s}: {res[ch][k]['mean']:.4f}±{res[ch][k]['std']:.4f}")
    print(f"OVERALL_BEST={res['overall_best']:.4f} (held-best 0.0545) {res['secs']:.0f}s")

if __name__ == "__main__":
    main()
