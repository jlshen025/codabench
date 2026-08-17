"""
tune_weight.py — Honest (non-overfit) Track-2 ensemble-weight selection via NESTED LOTO.

The single-LOTO-tuned w (peeking at all 3 targets) overestimates. Here, for each held-out
target T we pick w* on the OTHER two targets and apply it to T — w is never tuned on the
target it's scored on. Reports: global-tuned (optimistic), NESTED (honest), per-target w*,
and a fixed-w curve (robustness). Encoder = base ara+marb LOTO seed-avg; LLM = deepseek one-hot.
"""
import os, glob, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LLM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_llm", "deepseek_train.npz")
GRID = np.round(np.arange(0, 0.66, 0.05), 3)


def load_enc(prefixes=("base_araberttw", "base_marbertv2")):
    P, y, t = [], None, None
    for pfx in prefixes:
        sp = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(f"{ROOT}/{pfx}_loto_s*/oof_proba.npz"))]
        P.append(np.mean([s["proba"] for s in sp], 0)); y = sp[0]["y_true"]; t = sp[0]["target"].astype(str)
    return np.mean(P, 0), y, t


def f2(y, enc, lp, w, mask=None):
    m = np.ones(len(y), bool) if mask is None else mask
    return S.compute_metrics(y[m], ((1 - w) * enc[m] + w * lp[m]).argmax(1))["Favg2"] * 100


def main():
    enc, y, t = load_enc()
    d = np.load(LLM, allow_pickle=True); lp = d["proba"]
    assert np.array_equal(y, d["y_true"]), "align"
    targets = list(dict.fromkeys(t.tolist()))

    gscores = [f2(y, enc, lp, w) for w in GRID]
    gw = GRID[int(np.argmax(gscores))]
    print(f"global-tuned (OPTIMISTIC): w={gw} Favg2={max(gscores):.2f}")

    nested = np.zeros(len(y), int); wstars = {}
    for T in targets:
        mask = t == T; other = ~mask
        sc = [f2(y, enc, lp, w, other) for w in GRID]
        wT = GRID[int(np.argmax(sc))]; wstars[T[:10]] = float(wT)
        nested[mask] = ((1 - wT) * enc[mask] + wT * lp[mask]).argmax(1)
    print(f"NESTED (honest): Favg2={S.compute_metrics(y, nested)['Favg2']*100:.2f}  per-target w*={wstars}")

    print("fixed-w curve:", {float(w): round(f2(y, enc, lp, w), 2) for w in [0, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]})


if __name__ == "__main__":
    main()
