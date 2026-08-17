"""
ensemble_llm.py — Track-2 analysis: encoder LOTO OOF vs LLM zero-shot, and their soft-vote.

Both are produced in LOTO-order (targets in first-appearance order, rows in train order),
so they align 1:1 (asserted via y_true). Reports per-target + pooled Favg2 for the encoder,
the LLM, and weighted ensembles (LLM one-hot). Pooled Favg2 mirrors the official metric.
"""
import os, glob, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
_DEF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_llm", "deepseek_train.npz")
LLM_NPZ = sys.argv[1] if len(sys.argv) > 1 else _DEF


def load_enc_loto(prefixes=("base_araberttw", "base_marbertv2")):
    P, y, t = [], None, None
    for pfx in prefixes:
        fs = sorted(glob.glob(f"{ROOT}/{pfx}_loto_s*/oof_proba.npz"))
        sp = [np.load(f, allow_pickle=True) for f in fs]
        P.append(np.mean([s["proba"] for s in sp], 0))
        y = sp[0]["y_true"]; t = sp[0]["target"].astype(str)
    return np.mean(P, 0), y, t


def main():
    d = np.load(LLM_NPZ, allow_pickle=True)
    lp, ly, lt = d["proba"], d["y_true"], d["target"].astype(str)
    enc, ey, et = load_enc_loto()
    assert np.array_equal(ly, ey), "LLM/encoder order mismatch (y_true)"
    assert (lt == et).all(), "LLM/encoder target order mismatch"
    targets = list(dict.fromkeys(lt.tolist()))

    def rep(tag, proba):
        per = {}
        for tt in targets:
            m = lt == tt
            per[tt[:10]] = round(S.compute_metrics(ly[m], proba[m].argmax(1))["Favg2"] * 100, 1)
        pooled = S.compute_metrics(ly, proba.argmax(1))
        print(f"{tag:24s} pooled={pooled['Favg2']*100:5.2f} (F3 {pooled['Favg3']*100:5.1f}) | {per}")

    print(f"N={len(ly)} targets={targets}")
    rep("encoder(ara+marb)", enc)
    rep("LLM(deepseek)", lp)
    for w in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        rep(f"ENC+LLM w_llm={w}", (1 - w) * enc + w * lp)


if __name__ == "__main__":
    main()
