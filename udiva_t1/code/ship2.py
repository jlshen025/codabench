#!/usr/bin/env python3
"""SHIPPED reranker helpers + CV/dev harness (sibling of ship_eval.py).

Defines the reranker functions reused by ship_eval.py:
  train_hgb   -- 3-seed (seeds {0,1,2}) POINTWISE HistGradientBoosting; target y = 1 iff the
                 candidate tuple exactly matches a GT event in the segment, else 0;
                 class-balanced sample weights + 20x negative subsampling per seed.
  score_hgb   -- final confidence = arithmetic MEAN of the 3 seeds' predicted P(match) (all in
                 [0,1], no standardization).
  decode_emit -- emit ALL pooled candidates with their HGB confidence (mAP is threshold-free).
Config: verbal reranker = rerank_v7 rows (video->target) + rerank_v8 audio; non-verbal reranker
= rerank_v3 rows + rank_exp cross-segment feats.
  reranker training rows = LOSO within `train` (out-of-fold base scores);
  target prediction      = base trained on all `train`, reranker applied.
mode cv  : target = 2 starting-kit sessions -> self-mAP sanity + format validation.
mode pack: write the recognition.json zip.
Reuses dump_verbal_v7 (rerank_v7) + load_audio/add_audio (rerank_v8) + R3.dump_nonverbal
(rerank_v3) + xseg_feats/make_meta (rank_exp) + pack."""
import os, json, argparse, time
import numpy as np, torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from udiva import build_all_gt, compute_map, SESSIONS
from nonverbal_model import load_feats, instances as nv_instances
from verbal_model import session_instances
from verbal_emb import embed_all
import rerank as R
import rerank_v3 as R3
from rerank_v7 import dump_verbal_v7
from rerank_v8 import load_audio, add_audio
from rank_exp import xseg_feats, make_meta
import pack
R3.NSEEDS = 11
HGB_SEEDS = (0, 1, 2)


def train_hgb(X, y, seeds=HGB_SEEDS):
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]; clfs = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        sel = np.concatenate([pos, rng.choice(neg, min(len(neg), 20 * len(pos)), replace=False)])
        sw = compute_sample_weight("balanced", y[sel])
        clf = HistGradientBoostingClassifier(max_iter=250, max_depth=4, learning_rate=0.06,
                                             l2_regularization=1.0, min_samples_leaf=30, random_state=sd)
        clf.fit(X[sel], y[sel], sample_weight=sw); clfs.append(clf)
    return clfs


def score_hgb(clfs, X):
    return np.mean([c.predict_proba(X)[:, 1] for c in clfs], axis=0)


def decode_emit(rows, score, cap="all"):
    """EMIT decode -> (tuple, hgb_score) events. cap='all' = emit-all (max CV, E13 0.0673/0.0214);
    'segsubj:K' = top-K per (sid,seg,subj); 'sesscls:K' = top-K per (sid,cls) — size-safe fallbacks.
    decode_cap_v1: all > segsubj:40 0.0635/0.0201 (390K ev, 26x smaller) > smaller. Caps DO cost CV."""
    from collections import defaultdict
    n = len(rows)
    if cap == "all":
        keep = range(n)
    else:
        mode, K = cap.split(":"); K = int(K); grp = defaultdict(list)
        for i, r in enumerate(rows):
            grp[(r[0], r[1], r[2]) if mode == "segsubj" else (r[0], r[3])].append(i)
        keep = [i for idxs in grp.values() for i in sorted(idxs, key=lambda j: -score[j])[:K]]
    pred = {}
    for i in keep:
        r = rows[i]; pred.setdefault(r[0], {}).setdefault(r[1], []).append((r[4], float(score[i])))
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="001080,181182")
    ap.add_argument("--mode", choices=["cv", "pack"], default="cv")
    ap.add_argument("--zip", default="submission/staging/ship_e13.zip")
    ap.add_argument("--out", default=os.environ.get("EXP_OUTPUT_DIR", "."))
    ap.add_argument("--decode", default="all", help="all | segsubj:K | sesscls:K (size-safe cap; all=max CV)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    targets = a.target.split(","); train = [s for s in SESSIONS if s not in targets]
    t0 = time.time()
    gt = build_all_gt(); fc = {s: load_feats(s) for s in SESSIONS}
    aud = load_audio(); naf = len(next(iter(aud[SESSIONS[0]].values())))
    inst_by = {s: session_instances(gt, s) for s in SESSIONS}
    G = [(s, x[0], x[1], x[2], x[3], x[4], x[5]) for s in SESSIONS for x in inst_by[s]]
    E = embed_all(G, R.VMODEL, dev); sid_of = np.array([g[0] for g in G])
    nv_inst_by = {s: nv_instances(gt, s, fc[s]) for s in SESSIONS}

    # ---- VERBAL: hgb_cur on v7 (video-target) + audio feats, emit-all ----
    vtr = []
    for s in train:                                 # reranker training rows: LOSO within train
        vtr += add_audio(dump_verbal_v7(gt, s, G, E, sid_of, fc, train=[x for x in train if x != s]), aud, naf)
    Xv = np.array([r[5] for r in vtr], float); yv = np.array([r[6] for r in vtr], int)
    vclfs = train_hgb(Xv, yv)
    vtgt = []
    for t in targets:                               # predict targets: base trained on all train
        vtgt += add_audio(dump_verbal_v7(gt, t, G, E, sid_of, fc, train=train), aud, naf)
    vsc = score_hgb(vclfs, np.array([r[5] for r in vtgt], float))
    vpred = decode_emit(vtgt, vsc, a.decode)

    # ---- NONVERBAL: hgb_xseg on v3 + cross-segment feats, emit-all ----
    ntr = []
    for s in train:
        ntr += R3.dump_nonverbal(gt, s, nv_inst_by, dev, train=[x for x in train if x != s])
    mtr = make_meta(ntr)
    Xn = np.concatenate([np.array([r[5] for r in ntr], float),
                         xseg_feats(mtr, np.array([r[7] for r in ntr], float))], axis=1)
    yn = np.array([r[6] for r in ntr], int)
    nclfs = train_hgb(Xn, yn)
    ntgt = []
    for t in targets:
        ntgt += R3.dump_nonverbal(gt, t, nv_inst_by, dev, train=train)
    mtt = make_meta(ntgt)
    Xnt = np.concatenate([np.array([r[5] for r in ntgt], float),
                          xseg_feats(mtt, np.array([r[7] for r in ntgt], float))], axis=1)
    nsc = score_hgb(nclfs, Xnt)
    npred = decode_emit(ntgt, nsc, a.decode)

    pred = {"verbal": vpred, "nonverbal": npred}
    res = {}
    if a.mode == "cv":
        m = compute_map(pred, gt, sessions=targets, class_set="gt"); res["self_cv"] = dict(m)
        print(f"SHIP2 self-CV target={targets} overall={m['overall']:.4f} v={m['verbal']:.4f} nv={m['nonverbal']:.4f}")
    grids = {sid: {seg: (tb, te) for seg, (tb, te, s) in gt[sid]["verbal"].items()} for sid in targets}
    sub = pack.to_submission(pred, grids, conf_field="score")
    path, arc = pack.write_zip(sub, a.zip, json_name="recognition.json")
    nev = sum(len(sub[c][sid][seg]["events"]) for c in ("verbal", "nonverbal") for sid in targets for seg in sub[c][sid])
    res.update({"zip": path, "arc": arc, "decode": a.decode, "n_events": nev, "n_vrows": len(vtgt),
                "n_nrows": len(ntgt), "size_kb": round(os.path.getsize(path) / 1024, 1), "secs": time.time() - t0})
    os.makedirs(a.out, exist_ok=True); json.dump(res, open(f"{a.out}/result.json", "w"), indent=1)
    print(f"PACKED {path} (arc={arc}) n_events={nev} size={res['size_kb']}KB; {res['secs']:.0f}s")


if __name__ == "__main__":
    main()
