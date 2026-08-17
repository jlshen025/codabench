"""
blend_eval.py — general N-way soft-vote blend evaluator on the 3 UNSEEN-target draws.

Voters passed as  name=devnpz,lotonpz  (same interface as analyze_voters.py).
ENC = seed-avg(t2_base_ara, t2_base_marb) softmax proba (dev + loto OOF).
Blend = w_enc*ENC + sum_i w_i*voter_i  (weights sum to 1; matches predict_test.py).

Reports (Gate-2 honest — mean over 3 distinct unseen-target draws, never one):
  1. SINGLES: per-voter Favg2 on WomenEmp-dev | Covid-LOTO | Digital-LOTO | mean3
  2. DIVERSITY: pairwise error-decorrelation on the pooled 3-draw preds
     (low corr / high "one-right" = ensemble value; opus: diversity>count)
  3. GRID: search weights (step 0.1) over ENC+all voters, top configs by mean3
  4. ANCHOR: the current held-best config's mean3 for reference
"""
import os, glob, sys, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
COV, DIG = "Covid Vaccine", "Digital Transformation"


def enc_seedavg(prefixes, cv):
    P = []; y = None; t = None
    for pfx in prefixes:
        fs = sorted(glob.glob(f"{ROOT}/{pfx}_{cv}_s*/oof_proba.npz"))
        assert fs, f"no files {pfx}_{cv}"
        sp = [np.load(f, allow_pickle=True) for f in fs]
        P.append(np.mean([s["proba"] for s in sp], 0))
        yy = sp[0]["y_true"]; y = yy if y is None else y
        assert np.array_equal(y, yy), f"y mismatch {pfx}"
        if "target" in sp[0].files:
            t = sp[0]["target"].astype(str)
    return np.mean(P, 0), y, t


def load_llm(npz):
    d = np.load(npz, allow_pickle=True)
    return d["proba"], d["y_true"], (d["target"].astype(str) if "target" in d.files else None)


def favg2(y, proba, mask=None):
    m = np.ones(len(y), bool) if mask is None else mask
    return S.compute_metrics(y[m], proba[m].argmax(1))["Favg2"] * 100


def mean3(names, W, dev, lo, yd, yl, tl):
    """W = dict name->weight. Returns (dev, covid, digital, mean3)."""
    pd_ = sum(W[n] * dev[n] for n in names)
    pl_ = sum(W[n] * lo[n] for n in names)
    d = favg2(yd, pd_); c = favg2(yl, pl_, tl == COV); g = favg2(yl, pl_, tl == DIG)
    return d, c, g, (d + c + g) / 3.0


def main():
    voters = {}
    order = []
    for a in sys.argv[1:]:
        name, paths = a.split("=")
        dp, lp = paths.split(",")
        voters[name] = (dp, lp); order.append(name)

    ENC = ("t2_base_ara", "t2_base_marb")
    enc_dev, yd, _ = enc_seedavg(ENC, "dev")
    enc_lo, yl, tl = enc_seedavg(ENC, "loto")
    dev = {"ENC": enc_dev}; lo = {"ENC": enc_lo}
    for name in order:
        dp, lp = voters[name]
        p, yy, _ = load_llm(dp); assert np.array_equal(yy, yd), f"{name} dev y mismatch"
        dev[name] = p
        p, yy, tt = load_llm(lp); assert np.array_equal(yy, yl), f"{name} loto y mismatch"
        assert tt is None or (tt == tl).all(), f"{name} loto target mismatch"
        lo[name] = p
    names = ["ENC"] + order

    print("=" * 82)
    print("SINGLES (Favg2)         WomenEmp | Covid | Digital | mean3")
    for n in names:
        d = favg2(yd, dev[n]); c = favg2(yl, lo[n], tl == COV); g = favg2(yl, lo[n], tl == DIG)
        print(f"  {n:10s}   {d:6.2f} | {c:6.2f} | {g:6.2f} | {(d+c+g)/3:6.2f}")

    # pooled 3-draw correctness per voter (argmax pred vs y), for diversity
    print("=" * 82)
    print("DIVERSITY (pooled 3 unseen draws): pair | disagree% | both-wrong% | one-right% | corr(correct)")
    # build pooled pred/correct per voter
    pool_y = np.concatenate([yd, yl, yl])  # dev + covid-part + digital-part? -> use full loto once
    # Simpler: evaluate on dev (WomenEmp) + full loto (Covid+Digital) concatenated = all 3 draws once each
    yy_pool = np.concatenate([yd, yl])
    corr = {}
    preds = {}
    for n in names:
        pdev = dev[n].argmax(1); plo = lo[n].argmax(1)
        preds[n] = np.concatenate([pdev, plo])
    okv = {n: (preds[n] == yy_pool).astype(int) for n in names}
    for a, b in itertools.combinations(names, 2):
        disagree = np.mean(preds[a] != preds[b]) * 100
        bothwrong = np.mean((okv[a] == 0) & (okv[b] == 0)) * 100
        oneright = np.mean(okv[a] != okv[b]) * 100
        c = np.corrcoef(okv[a], okv[b])[0, 1]
        print(f"  {a[:6]:6s}~{b[:6]:6s} | {disagree:6.1f}   | {bothwrong:6.1f}     | {oneright:6.1f}    | {c:+.3f}")

    print("=" * 82)
    print("GRID search weights (step 0.1) over ENC + %d voters, top-12 by mean3" % len(order))
    grid_names = names
    K = len(grid_names)
    best = []
    for combo in itertools.product(range(0, 11), repeat=K):
        if sum(combo) != 10:
            continue
        W = {n: combo[i] / 10.0 for i, n in enumerate(grid_names)}
        d, c, g, m = mean3(grid_names, W, dev, lo, yd, yl, tl)
        best.append((m, d, c, g, tuple(round(W[n], 1) for n in grid_names)))
    best.sort(reverse=True)
    print("   " + "  ".join(f"w_{n[:5]}" for n in grid_names) + " |  dev   covid  digit | mean3")
    seen = set()
    shown = 0
    for m, d, c, g, w in best:
        if w in seen:
            continue
        seen.add(w); shown += 1
        print("    " + "   ".join(f"{x:.1f}" for x in w) + f"  | {d:5.1f} {c:5.1f} {g:5.1f} | {m:5.2f}")
        if shown >= 12:
            break

    print("=" * 82)
    # anchor: current held-best ENC.3 + claude.3 + gpt5.4 (only if those voters present)
    if "claude" in dev and "gpt5" in dev:
        W = {n: 0.0 for n in names}; W["ENC"] = 0.3; W["claude"] = 0.3; W["gpt5"] = 0.4
        d, c, g, m = mean3(names, W, dev, lo, yd, yl, tl)
        print(f"ANCHOR held-best ENC.3+claude.3+gpt5.4: dev={d:.1f} covid={c:.1f} digit={g:.1f} mean3={m:.2f}")


if __name__ == "__main__":
    main()
