"""
eval_blend.py — combine dev-OOF probabilities (encoders + LLM voters) and score Favg2.

Held-best reproduction (no args):
  enc-only (4× MTL: {marbertv2,araberttw}×{s1,s42})      -> ~85.09
  enc ⊕ deepseek (w=0.3)                                  -> ~86.34  (current held-best)

With --llm_npz PATH (a new candidate LLM dev-OOF, e.g. a LoRA run's oof_proba.npz):
  * solo Favg2 of the new LLM
  * enc ⊕ new_llm   over a weight grid (replaces deepseek)
  * enc ⊕ deepseek ⊕ new_llm   3-way grid
All proba are dev-CSV row aligned (verified via y_true).
"""
import os, sys, glob, argparse, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LLM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_llm")
ENC_DIRS = ["mtl_marbertv2_dev_s1", "mtl_marbertv2_dev_s42",
            "mtl_araberttw_dev_s1", "mtl_araberttw_dev_s42"]


def load(path):
    d = np.load(path, allow_pickle=True)
    return d["proba"].astype(np.float64), d["y_true"].astype(int), d["target"].astype(str)


def enc_ensemble():
    P, y0, t0 = [], None, None
    for name in ENC_DIRS:
        p, y, t = load(os.path.join(ROOT, name, "oof_proba.npz"))
        if y0 is None:
            y0, t0 = y, t
        else:
            assert (y == y0).all(), f"{name} y_true misaligned"
        P.append(p)
    return np.mean(P, 0), y0, t0


def score(proba, y, tgt, tag):
    m = S.compute_metrics(y, proba.argmax(1))
    per = {t: round(S.compute_metrics(y[tgt == t], proba[tgt == t].argmax(1))["Favg2"] * 100, 2)
           for t in sorted(set(tgt.tolist()))}
    print(f"  {tag:36s} Favg2={m['Favg2']*100:.2f} Favg3={m['Favg3']*100:.2f} "
          f"Acc={m['Acc']*100:.2f} | per-target={per}")
    return m["Favg2"] * 100


def blend(a, b, w):  # (1-w)*a + w*b
    return (1 - w) * a + w * b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_npz", default="")
    ap.add_argument("--grid", default="0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5")
    args = ap.parse_args()

    enc, y, tgt = enc_ensemble()
    print("=== baselines ===")
    score(enc, y, tgt, "enc-only (4×MTL)")
    ds = None
    ds_path = os.path.join(LLM, "deepseek_dev.npz")
    if os.path.exists(ds_path):
        ds, yd, _ = load(ds_path)
        assert (yd == y).all(), "deepseek y_true misaligned"
        best = max(((w, score(blend(enc, ds, w), y, tgt, f"enc ⊕ deepseek w={w}"))
                    for w in [0.3]), key=lambda x: x[1])

    if not args.llm_npz:
        return
    grid = [float(x) for x in args.grid.split(",")]
    new, yn, tn = load(args.llm_npz)
    assert (yn == y).all(), "llm_npz y_true misaligned with encoder OOF"
    print(f"\n=== NEW LLM: {args.llm_npz} ===")
    score(new, y, tgt, "new_llm solo")
    print("--- enc ⊕ new_llm (replaces deepseek) ---")
    b2 = max(((w, score(blend(enc, new, w), y, tgt, f"enc ⊕ new_llm w={w}")) for w in grid),
             key=lambda x: x[1])
    print(f"  BEST enc⊕new_llm: w={b2[0]} Favg2={b2[1]:.2f}")
    if ds is not None:
        print("--- enc ⊕ deepseek(0.3) ⊕ new_llm (3-way) ---")
        base = blend(enc, ds, 0.3)
        b3 = max(((w, score(blend(base, new, w), y, tgt, f"(enc⊕ds) ⊕ new_llm w={w}")) for w in grid),
                 key=lambda x: x[1])
        print(f"  BEST 3-way: w_new={b3[0]} Favg2={b3[1]:.2f}")
        # also a free 2-var search enc + a*ds + b*new (normalized)
        best = (-1, None)
        for a in np.arange(0, 0.51, 0.05):
            for bb in np.arange(0, 0.51, 0.05):
                if a + bb >= 1:
                    continue
                pr = (1 - a - bb) * enc + a * ds + bb * new
                f = S.compute_metrics(y, pr.argmax(1))["Favg2"] * 100
                if f > best[0]:
                    best = (f, (round(a, 2), round(bb, 2)))
        print(f"  BEST free (w_enc,w_ds,w_new): {best[1]} -> Favg2={best[0]:.2f}")


if __name__ == "__main__":
    main()
