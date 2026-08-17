#!/usr/bin/env python3
"""SHIPPED ENTRY POINT -- end-to-end build of submission 821839.

Trains every component on the 21 annotated dev sessions and predicts the 7 EVAL sessions,
writing recognition.json at the zip root. The EVAL 2s grid is derived from GF video duration
(validated identical to the annotation grid on dev); EVAL segments carry empty label sets
(training is dev-only).

Sequential pipeline (stage -> function @ module):
  1. GT + 2s grid       udiva.build_all_gt / seg_grid
  2. features           extract_feats.py (VideoMAE 3-crop) + extract_audio.py   [run beforehand]
  3. verbal base        verbal_emb.embed_all/fit_heads (e5 + per-attribute LR) + priors
  4. non-verbal base    nv_mlp2.train_fold x11 seeds (per-attribute MLP ensemble) + priors
  5. reranker rows      verbal:     rerank_v7.dump_verbal_v7 (base + video->target)
                                    -> rerank_v8.add_audio   (append 13 audio feats)
                        non-verbal: rerank_v3.dump_nonverbal (base + neighbour/speech)
                                    -> concat rank_exp.xseg_feats (cross-segment feats)
  6. reranker           ship2.train_hgb / score_hgb  (3-seed pointwise HGB, mean P(match))
  7. decode + pack      ship2.decode_emit (emit ALL) -> pack.to_submission / write_zip

Outputs the zip plus a self-check (event counts, size, per-session segment counts).
ship2.py is the CV/dev sibling (self-mAP + format check on held-out dev sessions)."""
import os, json, argparse, time
import numpy as np, torch
import decord
from udiva import build_all_gt, seg_grid, SESSIONS
from nonverbal_model import instances as nv_instances
from verbal_model import session_instances
from verbal_emb import embed_all
import rerank as R
import rerank_v3 as R3
from rerank_v7 import dump_verbal_v7
from rerank_v8 import add_audio
from rank_exp import xseg_feats, make_meta
from ship2 import train_hgb, score_hgb, decode_emit       # validated E13 helpers (E15)
import pack
R3.NSEEDS = 11

EVAL_SIDS = ["005013", "020025", "027113", "035040", "041083", "044156", "066067"]
DEV_FEAT = "<scratch>/t1/feats_large"
EVAL_FEAT = "<scratch>/t1/feats_eval_large"
DEV_AUDIO = "<scratch>/t1/audio_feats"
EVAL_AUDIO = "<scratch>/t1/audio_eval"
EVAL_GF = "<datasets>/UDIVA-HHOI/evaluation/audiovisual/exo/GF"


def load_npz(sid, d):
    z = np.load(f"{d}/{sid}.npz", allow_pickle=True)
    F = z["feats"].astype(np.float32)
    return {seg: F[i] for i, seg in enumerate(z["segs"])}


def load_audio_map(sid):
    d = DEV_AUDIO if sid in SESSIONS else EVAL_AUDIO
    z = np.load(f"{d}/{sid}.npz", allow_pickle=True)
    return {seg: z["feats"][i] for i, seg in enumerate(z["segs"])}


def build_eval_gt(sid):
    vr = decord.VideoReader(f"{EVAL_GF}/{sid}.mp4")
    grid = seg_grid(len(vr) / vr.get_avg_fps())
    base = {seg: (tb, te) for (seg, tb, te) in grid}
    return {"verbal": {k: (v[0], v[1], set()) for k, v in base.items()},
            "nonverbal": {k: (v[0], v[1], set()) for k, v in base.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode", default="all", help="all | segsubj:K | sesscls:K")
    ap.add_argument("--zip", default="submission/staging/eval_e13.zip")
    ap.add_argument("--out", default=os.environ.get("EXP_OUTPUT_DIR", "."))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    targets = EVAL_SIDS; train = list(SESSIONS); alls = train + targets
    gt = build_all_gt()
    for sid in targets:
        gt[sid] = build_eval_gt(sid)
    fc = {s: load_npz(s, DEV_FEAT if s in SESSIONS else EVAL_FEAT) for s in alls}
    aud = {s: load_audio_map(s) for s in alls}
    naf = len(next(iter(aud[alls[0]].values())))
    # sanity: eval feat/audio dims + segment coverage
    fdim = {s: next(iter(fc[s].values())).shape for s in alls}
    assert len({d[1:] for d in fdim.values()}) == 1, f"feat dim mismatch {set(fdim.values())}"
    inst_by = {s: session_instances(gt, s) for s in alls}
    G = [(s, x[0], x[1], x[2], x[3], x[4], x[5]) for s in alls for x in inst_by[s]]
    E = embed_all(G, R.VMODEL, dev); sid_of = np.array([g[0] for g in G])
    nv_inst_by = {s: nv_instances(gt, s, fc[s]) for s in alls}

    # ---- VERBAL: hgb_cur on v7+audio, emit-all ----
    vtr = []
    for s in train:
        vtr += add_audio(dump_verbal_v7(gt, s, G, E, sid_of, fc, train=[x for x in train if x != s]), aud, naf)
    vclfs = train_hgb(np.array([r[5] for r in vtr], float), np.array([r[6] for r in vtr], int))
    vtgt = []
    for t in targets:
        vtgt += add_audio(dump_verbal_v7(gt, t, G, E, sid_of, fc, train=train), aud, naf)
    vsc = score_hgb(vclfs, np.array([r[5] for r in vtgt], float))
    vpred = decode_emit(vtgt, vsc, a.decode)

    # ---- NONVERBAL: hgb_xseg, emit-all ----
    ntr = []
    for s in train:
        ntr += R3.dump_nonverbal(gt, s, nv_inst_by, dev, train=[x for x in train if x != s])
    mtr = make_meta(ntr)
    Xn = np.concatenate([np.array([r[5] for r in ntr], float),
                         xseg_feats(mtr, np.array([r[7] for r in ntr], float))], axis=1)
    nclfs = train_hgb(Xn, np.array([r[6] for r in ntr], int))
    ntgt = []
    for t in targets:
        ntgt += R3.dump_nonverbal(gt, t, nv_inst_by, dev, train=train)
    mtt = make_meta(ntgt)
    Xnt = np.concatenate([np.array([r[5] for r in ntgt], float),
                          xseg_feats(mtt, np.array([r[7] for r in ntgt], float))], axis=1)
    nsc = score_hgb(nclfs, Xnt)
    npred = decode_emit(ntgt, nsc, a.decode)

    pred = {"verbal": vpred, "nonverbal": npred}
    grids = {sid: {seg: (tb, te) for seg, (tb, te, s) in gt[sid]["verbal"].items()} for sid in targets}
    sub = pack.to_submission(pred, grids, conf_field="score")
    path, arc = pack.write_zip(sub, a.zip, json_name="recognition.json")
    nev = sum(len(sub[c][sid][seg]["events"]) for c in ("verbal", "nonverbal") for sid in targets for seg in sub[c][sid])
    res = {"zip": path, "arc": arc, "decode": a.decode, "n_events": nev,
           "n_vrows": len(vtgt), "n_nrows": len(ntgt), "size_kb": round(os.path.getsize(path) / 1024, 1),
           "per_session_segs": {sid: len(grids[sid]) for sid in targets},
           "feat_dim": list(next(iter(fdim.values()))), "naf": naf, "secs": time.time() - t0}
    os.makedirs(a.out, exist_ok=True); json.dump(res, open(f"{a.out}/result.json", "w"), indent=1)
    print(f"PACKED {path} (arc={arc}) decode={a.decode} n_events={nev} size={res['size_kb']}KB "
          f"vrows={len(vtgt)} nrows={len(ntgt)} feat_dim={res['feat_dim']} naf={naf}; {res['secs']:.0f}s")


if __name__ == "__main__":
    main()
