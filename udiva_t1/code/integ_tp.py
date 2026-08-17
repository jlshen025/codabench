#!/usr/bin/env python3
"""Integrated A/B: TIME-CONDITIONED prior vs STATIC, held-best models, model trained
ONCE per fold and reused for every (prior,a) cell. nonverbal 5-seed ens (speed).
Priors: static, time_B6, time_B8 (lam=3). Small a-grid per channel. Reports overall
per (prior, va, na). Goal: does swapping base[t] -> base[t|tau] lift integrated CV
above the 0.0545 held-best?

ROLE on the shipped path: `taus_of` (normalized session position tau=i/(n-1)) and `time_table`
(the position-BINNED Dirichlet-smoothed prior pi(t|bin), B bins, lam=3) are THE session-position
conditioning; they are re-exported by rerank.py and enter the shipped rerankers as the log-prior
+ raw-tau features. main() here is the CV A/B that chose to feed them to the reranker."""
import os, json, time
import numpy as np
import torch
from collections import Counter
from udiva import build_all_gt, compute_map, SESSIONS
from nonverbal_model import load_feats, instances as nv_instances
from verbal_model import session_instances
from verbal_emb import embed_all, fit_heads, proba_rows
from verbal_v2 import build_cands
import nv_mlp2 as NV

VMODEL = "intfloat/multilingual-e5-base"; VCREG = 4.0; VMIN = 2
NSEEDS = 5; NDH = 512; NP = 0.3; NEP = 80; NMIN = 2
VK = 40; NK = 80
VA = [0.4, 0.5]; NA = [0.2, 0.3]
BINS = [6, 8]; LAM = 3.0

def taus_of(gt, sid, cat):
    segs = sorted(gt[sid][cat].keys()); n = len(segs)
    return {seg: (i / (n - 1) if n > 1 else 0.5) for i, seg in enumerate(segs)}

def time_table(gt, train, cat, cands, B):
    gct = Counter(); ninst = 0; binct = [Counter() for _ in range(B)]; binn = [0] * B
    for sid in train:
        tx = taus_of(gt, sid, cat)
        for seg, (tb, te, s) in gt[sid][cat].items():
            b = min(int(tx[seg] * B), B - 1); binn[b] += 2; ninst += 2
            for tup in s:
                key = tup[1:]; gct[key] += 1; binct[b][key] += 1
    gbase = {t: gct.get(t, 0) / max(ninst, 1) for t in cands}
    return {t: np.array([(binct[b][t] + LAM * gbase[t]) / (binn[b] + LAM) for b in range(B)]) for t in cands}

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gt = build_all_gt(); fc = {s: load_feats(s) for s in SESSIONS}
    inst_by = {s: session_instances(gt, s) for s in SESSIONS}
    G = [(s, x[0], x[1], x[2], x[3], x[4], x[5]) for s in SESSIONS for x in inst_by[s]]
    E = embed_all(G, VMODEL, dev); sid_of = np.array([g[0] for g in G])
    nv_inst_by = {s: nv_instances(gt, s, fc[s]) for s in SESSIONS}
    out = os.environ.get("EXP_OUTPUT_DIR", "."); t0 = time.time()

    priors = ["static"] + [f"time_B{B}" for B in BINS]
    # accumulate per-fold mAP: keyed by (channel, prior, a) -> list over folds
    accV = {(p, a): [] for p in priors for a in VA}
    accN = {(p, a): [] for p in priors for a in NA}

    for test in SESSIONS:
        train = [s for s in SESSIONS if s != test]
        # ---------- verbal ----------
        vcands, vbase = build_cands(gt, train, VMIN)
        tr = np.where(np.isin(sid_of, train))[0]
        clf = fit_heads(E, tr, G, vcands, VCREG)
        te = np.where(sid_of == test)[0]; P = proba_rows(clf, E, te)
        vtp = {f"time_B{B}": time_table(gt, train, "verbal", vcands, B) for B in BINS}
        vtx = taus_of(gt, test, "verbal")
        # precompute per-row marginal products per candidate
        for prior in priors:
            B = int(prior.split("B")[1]) if prior != "static" else 0
            tab = vtp.get(prior)
            for a in VA:
                pred = {test: {seg: [] for seg in gt[test]["verbal"]}}
                for r, i in enumerate(te):
                    seg, spk = G[i][1], G[i][2]
                    bb = min(int(vtx[seg] * B), B - 1) if tab is not None else 0
                    scored = []
                    for (u, tg, mod) in vcands:
                        pu = P["u"][u][r] if u in P["u"] else 1e-9
                        pt = P["t"][tg][r] if tg in P["t"] else 1e-9
                        pm = P["m"][mod][r] if mod in P["m"] else 1e-9
                        pri = tab[(u, tg, mod)][bb] if tab is not None else vbase[(u, tg, mod)]
                        conf = (pu * pt * pm + 1e-9) ** a * (pri + 1e-9) ** (1 - a)
                        scored.append((conf, (spk, u, tg, mod)))
                    scored.sort(key=lambda x: -x[0])
                    pred[test][seg] += [(t, float(c)) for c, t in scored[:VK]]
                accV[(prior, a)].append(
                    compute_map({"verbal": pred, "nonverbal": {test: {}}}, gt, sessions=[test], class_set="gt")["verbal"])
        # ---------- nonverbal ----------
        ncands, nbase = NV.build_candidates(gt, train, NMIN)
        vocab = NV.vocabs_from_cands(ncands)
        tri = [x for s in train for x in nv_inst_by[s]]
        Xtr, Ytr = NV.encode(tri, vocab)
        models = []
        for s in range(NSEEDS):
            m, mu, sd = NV.train_fold(Xtr, Ytr, vocab, dev, NDH, NP, NEP, seed=s); models.append(m)
        idx = {k: {c: i for i, c in enumerate(v)} for k, v in vocab.items()}
        hi = np.array([idx["h"][t[0]] for t in ncands]); li = np.array([idx["l"][t[1]] for t in ncands])
        ti = np.array([idx["t"][t[2]] for t in ncands]); mi = np.array([idx["m"][t[3]] for t in ncands])
        nbr = np.array([nbase[t] for t in ncands])
        ntp = {f"time_B{B}": time_table(gt, train, "nonverbal", ncands, B) for B in BINS}
        ntx = taus_of(gt, test, "nonverbal")
        inst = nv_inst_by[test]
        X = (np.stack([z[2] for z in inst]) - mu) / sd; Xt = torch.tensor(X, device=dev)
        Pavg = {k: np.zeros((len(inst), len(vocab[k])), np.float32) for k in vocab}
        for m in models:
            m.eval()
            with torch.no_grad():
                o = m(Xt)
                for k in vocab: Pavg[k] += torch.sigmoid(o[k]).cpu().numpy()
        for k in vocab: Pavg[k] /= len(models)
        ms_all = (Pavg["h"][:, hi] * Pavg["l"][:, li] * Pavg["t"][:, ti] * Pavg["m"][:, mi])  # [ninst, ncands]
        for prior in priors:
            B = int(prior.split("B")[1]) if prior != "static" else 0
            tab = ntp.get(prior)
            # per-instance prior vector
            for a in NA:
                pred = {test: {seg: [] for seg in gt[test]["nonverbal"]}}
                for r, (seg, subj, _, H, L, T, M) in enumerate(inst):
                    if tab is not None:
                        bb = min(int(ntx[seg] * B), B - 1)
                        pri = np.array([tab[t][bb] for t in ncands])
                    else:
                        pri = nbr
                    conf = (ms_all[r] + 1e-9) ** a * (pri + 1e-9) ** (1 - a)
                    top = np.argpartition(-conf, min(NK, len(conf) - 1))[:NK]
                    for j in top: pred[test][seg].append(((subj, *ncands[j]), float(conf[j])))
                accN[(prior, a)].append(
                    compute_map({"verbal": {test: {}}, "nonverbal": pred}, gt, sessions=[test], class_set="gt")["nonverbal"])

    res = {"verbal": {}, "nonverbal": {}, "overall": {}}
    for (p, a), v in accV.items():
        res["verbal"][f"{p}_a{a}"] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
    for (p, a), v in accN.items():
        res["nonverbal"][f"{p}_a{a}"] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
    # best per prior, and overall = best_v(prior) + best_nv(prior)
    for p in priors:
        bv = max(res["verbal"][f"{p}_a{a}"]["mean"] for a in VA)
        bn = max(res["nonverbal"][f"{p}_a{a}"]["mean"] for a in NA)
        res["overall"][p] = {"verbal": bv, "nonverbal": bn, "overall": 0.5 * (bv + bn)}
    res["secs"] = time.time() - t0; res["seeds"] = NSEEDS
    os.makedirs(out, exist_ok=True); json.dump(res, open(f"{out}/result.json", "w"), indent=1)
    for p in priors:
        o = res["overall"][p]
        print(f"{p:10s}: overall={o['overall']:.4f} (v={o['verbal']:.4f} nv={o['nonverbal']:.4f})")
    print(f"held-best static ref=0.0545; {res['secs']:.0f}s seeds={NSEEDS}")

if __name__ == "__main__":
    main()
