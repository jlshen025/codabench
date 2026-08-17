#!/usr/bin/env python3
"""rerank v8: + AUDIO (13 hand-crafted per-segment feats from exo mp4: rms/centroid/bw/
rolloff/zcr/onset/5-mfcc) appended to the reranker features. Audio = NEW orthogonal signal
(action-sound transients + speech activity) -> less overfit-prone than feature reshuffling.
HGB eval. Tests lift over 0.062.

ROLE on the shipped path: `add_audio` (append the 13 audio feats) and `load_audio` are reused by
ship2.py/ship_eval.py; the SHIPPED verbal reranker uses these audio feats (non-verbal does not).
main() is a dev CV-ablation harness that swept which channel(s) benefit."""
import os, json, time
import numpy as np, torch
from udiva import build_all_gt, SESSIONS
from nonverbal_model import load_feats, instances as nv_instances
from verbal_model import session_instances
from verbal_emb import embed_all
import rerank as R
import rerank_v3 as R3
from rerank_v4 import eval_ch
from rerank_v7 import dump_verbal_v7
R3.NSEEDS = 11
AUDIO_DIR = os.environ.get("UDIVA_AUDIO_OUT", "<scratch>/t1/audio_feats")

def load_audio():
    aud = {}
    for sid in SESSIONS:
        d = np.load(f"{AUDIO_DIR}/{sid}.npz", allow_pickle=True)
        aud[sid] = {seg: d["feats"][i] for i, seg in enumerate(d["segs"])}
    return aud

def add_audio(rows, aud, naf):
    z = [0.0] * naf; out = []
    for (sid, seg, subj, cls, tup, feat, lab, conf) in rows:
        af = aud.get(sid, {}).get(seg)
        out.append((sid, seg, subj, cls, tup, feat + (list(af) if af is not None else z), lab, conf))
    return out

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gt = build_all_gt(); fc = {s: load_feats(s) for s in SESSIONS}; aud = load_audio()
    naf = len(next(iter(aud[SESSIONS[0]].values())))
    inst_by = {s: session_instances(gt, s) for s in SESSIONS}
    G = [(s, x[0], x[1], x[2], x[3], x[4], x[5]) for s in SESSIONS for x in inst_by[s]]
    E = embed_all(G, R.VMODEL, dev); sid_of = np.array([g[0] for g in G])
    nv_inst_by = {s: nv_instances(gt, s, fc[s]) for s in SESSIONS}
    t0 = time.time(); vrows, nrows = [], []
    for test in R.FOLDS:
        vrows += dump_verbal_v7(gt, test, G, E, sid_of, fc)
        nrows += R3.dump_nonverbal(gt, test, nv_inst_by, dev)
    ch = os.environ.get("UDIVA_AUDIO_CH", "both")   # both|verbal|nonverbal|none
    if ch in ("both", "verbal"): vrows = add_audio(vrows, aud, naf)
    if ch in ("both", "nonverbal"): nrows = add_audio(nrows, aud, naf)
    res = {"verbal": eval_ch(gt, "verbal", vrows, 40, [2]),
           "nonverbal": eval_ch(gt, "nonverbal", nrows, 80, [2, 3])}
    def best(ch): return max(v["mean"] for k, v in res[ch].items() if k != "pool_recall")
    res["overall_best"] = 0.5 * (best("verbal") + best("nonverbal")); res["secs"] = time.time() - t0; res["naf"] = naf
    out = os.environ.get("EXP_OUTPUT_DIR", "."); os.makedirs(out, exist_ok=True)
    json.dump(res, open(f"{out}/result.json", "w"), indent=1)
    for ch in ("verbal", "nonverbal"):
        print(f"[{ch}] pool_recall={res[ch]['pool_recall']:.3f}")
        for k in sorted(res[ch], key=lambda z: -res[ch][z]['mean'] if z != 'pool_recall' else 1):
            if k != 'pool_recall': print(f"   {k:16s}: {res[ch][k]['mean']:.4f}")
    print(f"OVERALL_BEST={res['overall_best']:.4f} (+audio {naf}d; ref v7 0.0625) {res['secs']:.0f}s")

if __name__ == "__main__":
    main()
