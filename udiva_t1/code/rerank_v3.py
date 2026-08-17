#!/usr/bin/env python3
"""rerank v3 = rerank.py (full pool) + extra candidate features:
  temporal-neighbor class persistence (model class-marginal at seg t-1,t+1, same subject)
  speech token count in segment (log1p) — verbal localization + nonverbal talk-vs-silent.
Reuses eval_channel/decode/time_table from rerank.py (feature-agnostic). Same ablation.

ROLE on the shipped path: `dump_nonverbal` here is THE non-verbal reranker-row builder used by
ship2.py/ship_eval.py (base marginals + static/time priors + neighbour + speech features, then
rank_exp.xseg_feats is concatenated). `speech_tokens`/`neigh` are reused by rerank_v7. main() is
a dev CV-ablation harness."""
import os, json, time
import numpy as np
import torch
from udiva import build_all_gt, compute_map, SESSIONS
from nonverbal_model import load_feats, instances as nv_instances
from verbal_model import session_instances
from verbal_emb import embed_all, fit_heads, proba_rows
from verbal_v2 import build_cands
import nv_mlp2 as NV
from transcripts import seg_speaker_text
from rerank import (time_table, taus_of, eval_channel, FOLDS, POOL, B, VA, NA,
                    VCREG, VMIN, NMIN, NDH, NPP, NEP, NSEEDS, VMODEL, EPS)

def speech_tokens(sid, gt, cat):
    grid = [(sk, tb, te) for sk, (tb, te, s) in gt[sid][cat].items()]
    st = seg_speaker_text(sid, grid)
    return {seg: {sp: len(t.split()) for sp, t in d.items()} for seg, d in st.items()}

def neigh(seg, k): return f"s_{int(seg[2:]) + k:04d}"

def dump_verbal(gt, test, G, E, sid_of, train=None):
    if train is None: train = [s for s in SESSIONS if s != test]
    cands, base = build_cands(gt, train, VMIN)
    tr = np.where(np.isin(sid_of, train))[0]; clf = fit_heads(E, tr, G, cands, VCREG)
    te = np.where(sid_of == test)[0]; P = proba_rows(clf, E, te)
    tab = time_table(gt, train, "verbal", cands, B); tx = taus_of(gt, test, "verbal")
    locr = {(G[i][1], G[i][2]): r for r, i in enumerate(te)}
    stok = speech_tokens(test, gt, "verbal")
    rows = []
    for r, i in enumerate(te):
        seg, spk = G[i][1], G[i][2]; bb = min(int(tx[seg] * B), B - 1); tau = tx[seg]
        gtset = gt[test]["verbal"][seg][2]; ntok = float(np.log1p(stok.get(seg, {}).get(spk, 0)))
        items = []
        for (u, tg, mod) in cands:
            pu = P["u"][u][r] if u in P["u"] else EPS
            pt = P["t"][tg][r] if tg in P["t"] else EPS
            pm = P["m"][mod][r] if mod in P["m"] else EPS
            pri = tab[(u, tg, mod)][bb]; ms = pu * pt * pm
            conf = (ms + EPS) ** VA * (base[(u, tg, mod)] + EPS) ** (1 - VA)
            nbv = []
            for k in (-1, 1):
                r2 = locr.get((neigh(seg, k), spk))
                if r2 is not None and u in P["u"]: nbv.append(P["u"][u][r2])
            nb_max = max(nbv) if nbv else 0.0; nb_mean = float(np.mean(nbv)) if nbv else 0.0
            logs = (np.log(pu + EPS), np.log(pt + EPS), np.log(pm + EPS),
                    np.log(base[(u, tg, mod)] + EPS), np.log(pri + EPS))
            items.append((conf, (spk, u, tg, mod), u, logs, nb_max, nb_mean))
        items.sort(key=lambda z: -z[0]); pool = items[:POOL]
        byc = {}
        for it in pool: byc.setdefault(it[2], []).append(it[0])
        for c in byc: byc[c].sort(reverse=True)
        for rank, (conf, tup, cls, logs, nb_max, nb_mean) in enumerate(pool):
            same = byc[cls]; second = same[1] if len(same) > 1 else 0.0
            feat = [*logs, tau, float(rank), float(len(same)), float(conf - second),
                    float(conf), nb_max, nb_mean, ntok]
            rows.append((test, seg, tup[0], cls, tup, feat, 1 if tup in gtset else 0, conf))
    return rows

def dump_nonverbal(gt, test, nv_inst_by, dev, train=None):
    if train is None: train = [s for s in SESSIONS if s != test]
    cands, base = NV.build_candidates(gt, train, NMIN); vocab = NV.vocabs_from_cands(cands)
    tri = [x for s in train for x in nv_inst_by[s]]; Xtr, Ytr = NV.encode(tri, vocab)
    models = []
    for s in range(NSEEDS):
        m, mu, sd = NV.train_fold(Xtr, Ytr, vocab, dev, NDH, NPP, NEP, seed=s); models.append(m)
    idx = {k: {c: i for i, c in enumerate(v)} for k, v in vocab.items()}
    hi = np.array([idx["h"][t[0]] for t in cands]); li = np.array([idx["l"][t[1]] for t in cands])
    ti = np.array([idx["t"][t[2]] for t in cands]); mi = np.array([idx["m"][t[3]] for t in cands])
    nbr = np.array([base[t] for t in cands]); lbr = np.log(nbr + EPS)
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
    locr = {(seg, subj): r for r, (seg, subj, _, H, L, T, M) in enumerate(inst)}
    stok = speech_tokens(test, gt, "nonverbal")
    rows = []
    for r, (seg, subj, _, H, L, T, M) in enumerate(inst):
        bb = min(int(tx[seg] * B), B - 1); tau = tx[seg]
        gtset = gt[test]["nonverbal"][seg][2]; ntok = float(np.log1p(stok.get(seg, {}).get(subj, 0)))
        pri = np.array([tab[t][bb] for t in cands]); ms = ms_all[r]
        conf = (ms + EPS) ** NA * (nbr + EPS) ** (1 - NA); order = np.argsort(-conf)[:POOL]
        lph = np.log(Pavg["h"][r][hi] + EPS); lpl = np.log(Pavg["l"][r][li] + EPS)
        lpt = np.log(Pavg["t"][r][ti] + EPS); lpm = np.log(Pavg["m"][r][mi] + EPS); lpri = np.log(pri + EPS)
        nbh = np.zeros(len(cands)); cnt = 0
        for k in (-1, 1):
            r2 = locr.get((neigh(seg, k), subj))
            if r2 is not None: nbh += Pavg["h"][r2][hi]; cnt += 1
        nb_mean = nbh / max(cnt, 1)
        byc = {}
        for j in order: byc.setdefault(cands[j][0], []).append(conf[j])
        for c in byc: byc[c].sort(reverse=True)
        for rank, j in enumerate(order):
            cls = cands[j][0]; same = byc[cls]; second = same[1] if len(same) > 1 else 0.0
            tup = (subj, *cands[j])
            feat = [lph[j], lpl[j], lpt[j], lpm[j], lbr[j], lpri[j], tau, float(rank),
                    float(len(same)), float(conf[j] - second), float(conf[j]), float(nb_mean[j]), ntok]
            rows.append((test, seg, subj, cls, tup, feat, 1 if tup in gtset else 0, float(conf[j])))
    return rows

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
    def best(ch): return max(v["mean"] for k, v in res[ch].items() if k != "pool_recall")
    res["overall_best"] = 0.5 * (best("verbal") + best("nonverbal")); res["secs"] = time.time() - t0
    out = os.environ.get("EXP_OUTPUT_DIR", "."); os.makedirs(out, exist_ok=True)
    json.dump(res, open(f"{out}/result.json", "w"), indent=1)
    for ch in ("verbal", "nonverbal"):
        print(f"[{ch}] pool_recall={res[ch]['pool_recall']:.3f}")
        for k in sorted(res[ch], key=lambda z: -res[ch][z]["mean"] if z != "pool_recall" else 1):
            if k != "pool_recall": print(f"   {k:14s}: {res[ch][k]['mean']:.4f}±{res[ch][k]['std']:.4f}")
    print(f"OVERALL_BEST={res['overall_best']:.4f} (+feats: nbr-persist + speech-tok) {res['secs']:.0f}s")

if __name__ == "__main__":
    main()
