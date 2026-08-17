#!/usr/bin/env python3
"""rerank v7: VERBAL-VISUAL-TARGET lever. Verbal's hard attr = TARGET (which brick),
visually grounded -> unrecoverable from text (probe-confirmed). Inject a VIDEO->target
signal into the verbal reranker: per fold, train a segment-video (full-crop VideoMAE-large,
1024d) -> target multilabel LR head (labels = targets present from verbal+nonverbal GT); add
feature = P_video(candidate.target | seg) to each verbal candidate. ONE new feature (low overfit
risk vs v6). Nonverbal dump = v3. HGB eval from v4. Reuses VideoMAE feats (no new extraction).

ROLE on the shipped path: `dump_verbal_v7` here is THE verbal reranker-row builder used by
ship2.py/ship_eval.py (v3 verbal features + the video->target feature); `train_vtarget` builds the
full-crop video->target head. rerank_v8.add_audio then appends the audio features. main() is a dev
CV-ablation harness."""
import os, json, time
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from udiva import build_all_gt, compute_map, SESSIONS
from nonverbal_model import load_feats, instances as nv_instances
from verbal_model import session_instances
from verbal_emb import embed_all, fit_heads, proba_rows
from verbal_v2 import build_cands
import rerank as R
import rerank_v3 as R3
from rerank_v3 import speech_tokens, neigh
from rerank_v4 import eval_ch
R3.NSEEDS = 11

def train_vtarget(gt, train, fc, cand_targets):
    X, segt = [], []
    for sid in train:
        for seg, (tb, te, s) in gt[sid]["nonverbal"].items():
            if seg not in fc[sid]: continue
            X.append(fc[sid][seg][2])  # full-crop scene feat
            tgs = set(t[2] for t in gt[sid]["verbal"][seg][2]) | set(t[3] for t in gt[sid]["nonverbal"][seg][2])
            segt.append(tgs)
    X = np.array(X, np.float32); sc = StandardScaler().fit(X); Xs = sc.transform(X)
    clf = {}
    for tg in cand_targets:
        y = np.array([1 if tg in s else 0 for s in segt])
        clf[tg] = float(y.mean()) if (y.sum() == 0 or y.sum() == len(y)) else \
            LogisticRegression(C=1.0, max_iter=200, class_weight="balanced").fit(Xs, y)
    return clf, sc

def dump_verbal_v7(gt, test, G, E, sid_of, fc, train=None):
    if train is None: train = [s for s in SESSIONS if s != test]
    cands, base = build_cands(gt, train, R3.VMIN)
    cand_targets = sorted({t[1] for t in cands})            # verbal cand=(u,tg,mod) -> target=idx1
    tr = np.where(np.isin(sid_of, train))[0]; clf = fit_heads(E, tr, G, cands, R3.VCREG)
    te = np.where(sid_of == test)[0]; P = proba_rows(clf, E, te)
    tab = R3.time_table(gt, train, "verbal", cands, R3.B); tx = R3.taus_of(gt, test, "verbal")
    vt_clf, vt_sc = train_vtarget(gt, train, fc, cand_targets)
    locr = {(G[i][1], G[i][2]): r for r, i in enumerate(te)}
    stok = speech_tokens(test, gt, "verbal")
    rows = []
    for r, i in enumerate(te):
        seg, spk = G[i][1], G[i][2]; bb = min(int(tx[seg] * R3.B), R3.B - 1); tau = tx[seg]
        gtset = gt[test]["verbal"][seg][2]; ntok = float(np.log1p(stok.get(seg, {}).get(spk, 0)))
        if seg in fc[test]:
            xs = vt_sc.transform(fc[test][seg][2][None].astype(np.float32))
            vtp = {tg: (vt_clf[tg] if isinstance(vt_clf[tg], float) else float(vt_clf[tg].predict_proba(xs)[0, 1]))
                   for tg in cand_targets}
        else:
            vtp = {}
        items = []
        for (u, tg, mod) in cands:
            pu = P["u"][u][r] if u in P["u"] else R3.EPS
            pt = P["t"][tg][r] if tg in P["t"] else R3.EPS
            pm = P["m"][mod][r] if mod in P["m"] else R3.EPS
            pri = tab[(u, tg, mod)][bb]; ms = pu * pt * pm
            conf = (ms + R3.EPS) ** R3.VA * (base[(u, tg, mod)] + R3.EPS) ** (1 - R3.VA)
            nbv = []
            for k in (-1, 1):
                r2 = locr.get((neigh(seg, k), spk))
                if r2 is not None and u in P["u"]: nbv.append(P["u"][u][r2])
            nb_max = max(nbv) if nbv else 0.0; nb_mean = float(np.mean(nbv)) if nbv else 0.0
            logs = (np.log(pu + R3.EPS), np.log(pt + R3.EPS), np.log(pm + R3.EPS),
                    np.log(base[(u, tg, mod)] + R3.EPS), np.log(pri + R3.EPS))
            items.append((conf, (spk, u, tg, mod), u, logs, nb_max, nb_mean, float(vtp.get(tg, 0.0))))
        items.sort(key=lambda z: -z[0]); pool = items[:R3.POOL]
        byc = {}
        for it in pool: byc.setdefault(it[2], []).append(it[0])
        for c in byc: byc[c].sort(reverse=True)
        for rank, (conf, tup, cls, logs, nb_max, nb_mean, vtg) in enumerate(pool):
            same = byc[cls]; second = same[1] if len(same) > 1 else 0.0
            feat = [*logs, tau, float(rank), float(len(same)), float(conf - second), float(conf),
                    nb_max, nb_mean, ntok, vtg]
            rows.append((test, seg, tup[0], cls, tup, feat, 1 if tup in gtset else 0, conf))
    return rows

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gt = build_all_gt(); fc = {s: load_feats(s) for s in SESSIONS}
    inst_by = {s: session_instances(gt, s) for s in SESSIONS}
    G = [(s, x[0], x[1], x[2], x[3], x[4], x[5]) for s in SESSIONS for x in inst_by[s]]
    E = embed_all(G, R.VMODEL, dev); sid_of = np.array([g[0] for g in G])
    nv_inst_by = {s: nv_instances(gt, s, fc[s]) for s in SESSIONS}
    t0 = time.time(); vrows, nrows = [], []
    for test in R.FOLDS:
        vrows += dump_verbal_v7(gt, test, G, E, sid_of, fc)
        nrows += R3.dump_nonverbal(gt, test, nv_inst_by, dev)
    res = {"verbal": eval_ch(gt, "verbal", vrows, 40, [2]),
           "nonverbal": eval_ch(gt, "nonverbal", nrows, 80, [2, 3])}
    def best(ch): return max(v["mean"] for k, v in res[ch].items() if k != "pool_recall")
    res["overall_best"] = 0.5 * (best("verbal") + best("nonverbal")); res["secs"] = time.time() - t0
    out = os.environ.get("EXP_OUTPUT_DIR", "."); os.makedirs(out, exist_ok=True)
    json.dump(res, open(f"{out}/result.json", "w"), indent=1)
    for ch in ("verbal", "nonverbal"):
        print(f"[{ch}] pool_recall={res[ch]['pool_recall']:.3f}")
        for k in sorted(res[ch], key=lambda z: -res[ch][z]['mean'] if z != 'pool_recall' else 1):
            if k != 'pool_recall': print(f"   {k:16s}: {res[ch][k]['mean']:.4f}")
    print(f"OVERALL_BEST={res['overall_best']:.4f} (verbal hgb_top2 ref 0.0551; +video-target feat) {res['secs']:.0f}s")

if __name__ == "__main__":
    main()
