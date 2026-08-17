"""
analyze_voters.py — Multi-voter ensemble analysis on the NEW Track-2 data.

Three UNSEEN-target evaluations (Gate-2 honest: NEVER tune to a single one):
  * WomenEmp  = dev_track_2 (1400)  -> encoder cv=dev proba + LLM dev npz
  * Covid/Digital = LOTO on train_track_2 (2721) -> encoder cv=loto OOF + LLM loto npz

Encoder = seed-avg(t2_base_ara, t2_base_marb) softmax proba (the NEW-data runs).
LLM voters: any of deepseek / claude / ... , passed as name=devnpz:lotonpz.
Ensemble = linear proba blend: (1-sum w_i)*enc + sum_i w_i*llm_i  (matches predict_test.py).

Reports singles, 2-way, 3-way at a weight grid, per-target on LOTO, and the KEY
Gate-2 view: a fixed config's Favg2 on all 3 unseen draws + mean (robust-across, not dev-tuned).
"""
import os, glob, sys, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


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


def per_target(y, t, proba):
    return {tt[:9]: round(favg2(y, proba, t == tt), 1) for tt in dict.fromkeys(t.tolist())}


def main():
    # voters: "name=devnpz,lotonpz"
    voters = {}
    for a in sys.argv[1:]:
        name, paths = a.split("=")
        dev_p, loto_p = paths.split(",")
        voters[name] = (dev_p, loto_p)

    ENC = ("t2_base_ara", "t2_base_marb")
    enc_dev, yd, _ = enc_seedavg(ENC, "dev")
    enc_lo, yl, tl = enc_seedavg(ENC, "loto")

    dev = {"ENC": enc_dev}; lo = {"ENC": enc_lo}
    for name, (dp, lp) in voters.items():
        if os.path.exists(dp):
            p, yy, _ = load_llm(dp); assert np.array_equal(yy, yd), f"{name} dev y mismatch"
            dev[name] = p
        if os.path.exists(lp):
            p, yy, tt, = load_llm(lp)
            assert np.array_equal(yy, yl), f"{name} loto y mismatch"
            assert tt is None or (tt == tl).all(), f"{name} loto target mismatch"
            lo[name] = p

    print("=" * 78)
    print("SINGLES  (Favg2)                     WomenEmp-dev | Covid/Digital-LOTO (per-target)")
    for k in dev:
        d2 = favg2(yd, dev[k])
        l2 = favg2(yl, lo[k]) if k in lo else float("nan")
        pt = per_target(yl, tl, lo[k]) if k in lo else {}
        print(f"  {k:14s} dev={d2:6.2f}   loto={l2:6.2f}  {pt}")

    llm_names = [k for k in dev if k != "ENC"]

    print("=" * 78)
    print("2-WAY  ENC + one LLM  (w = LLM weight)")
    for name in llm_names:
        print(f"  -- ENC + {name} --")
        for w in [0.2, 0.3, 0.4, 0.5, 0.6]:
            dd = favg2(yd, (1 - w) * dev["ENC"] + w * dev[name])
            row = f"     w={w}: dev={dd:6.2f}"
            if name in lo:
                ll = favg2(yl, (1 - w) * lo["ENC"] + w * lo[name])
                row += f"  loto={ll:6.2f}  mean3={_mean3(yd,tl,yl,(1-w)*dev['ENC']+w*dev[name],(1-w)*lo['ENC']+w*lo[name])}"
            print(row)

    loto_llms = [k for k in llm_names if k in lo]
    if len(llm_names) >= 2 and len(loto_llms) >= 2:
        print("=" * 78)
        print("2-WAY  LLM + LLM  (no encoder; a=weight of first)")
        a_name, b_name = llm_names[0], llm_names[1]
        for a in [0.3, 0.4, 0.5, 0.6, 0.7]:
            dd = favg2(yd, a * dev[a_name] + (1 - a) * dev[b_name])
            row = f"  {a_name}*{a}+{b_name}*{round(1-a,2)}: dev={dd:6.2f}"
            if a_name in lo and b_name in lo:
                ll = favg2(yl, a * lo[a_name] + (1 - a) * lo[b_name])
                row += f"  loto={ll:6.2f}"
            print(row)

        print("=" * 78)
        print("3-WAY  ENC + LLM1 + LLM2   grid (w_enc, w1, w2) summing to 1")
        best = (0, None)
        rows = []
        for we in [0.2, 0.3, 0.4, 0.5, 0.6]:
            for w1 in np.round(np.arange(0, 1 - we + 1e-9, 0.1), 2):
                w2 = round(1 - we - w1, 2)
                if w2 < -1e-9:
                    continue
                pd_ = we * dev["ENC"] + w1 * dev[a_name] + w2 * dev[b_name]
                pl_ = we * lo["ENC"] + w1 * lo[a_name] + w2 * lo[b_name]
                dd = favg2(yd, pd_); ll = favg2(yl, pl_)
                m3 = _mean3_val(yd, tl, yl, pd_, pl_)
                rows.append((m3, we, w1, w2, dd, ll))
                if m3 > best[0]:
                    best = (m3, (we, w1, w2, dd, ll))
        rows.sort(reverse=True)
        print("  top-8 by mean-of-3-unseen-draws (Gate-2 robust selection):")
        print("   w_enc w_%s w_%s |  dev   covid  digit | mean3" % (a_name[:4], b_name[:4]))
        for m3, we, w1, w2, dd, ll in rows[:8]:
            cov = favg2(yl, we * lo["ENC"] + w1 * lo[a_name] + w2 * lo[b_name], tl == "Covid Vaccine")
            dig = favg2(yl, we * lo["ENC"] + w1 * lo[a_name] + w2 * lo[b_name], tl == "Digital Transformation")
            print(f"    {we:.1f}  {w1:.1f}  {w2:.1f}  | {dd:5.1f} {cov:5.1f} {dig:5.1f} | {m3:5.2f}")

    print("=" * 78)
    print("HELD-BEST anchor: ENC + deepseek w0.4")
    if "deepseek" in dev:
        pd_ = 0.6 * dev["ENC"] + 0.4 * dev["deepseek"]
        print(f"   dev={favg2(yd, pd_):.2f}", end="")
        if "deepseek" in lo:
            pl_ = 0.6 * lo["ENC"] + 0.4 * lo["deepseek"]
            print(f"  loto={favg2(yl, pl_):.2f}  per-target={per_target(yl, tl, pl_)}", end="")
        print()


def _mean3(yd, tl, yl, pd_, pl_):
    return round(_mean3_val(yd, tl, yl, pd_, pl_), 2)


def _mean3_val(yd, tl, yl, pd_, pl_):
    dev2 = favg2(yd, pd_)
    cov = favg2(yl, pl_, tl == "Covid Vaccine")
    dig = favg2(yl, pl_, tl == "Digital Transformation")
    return (dev2 + cov + dig) / 3.0


if __name__ == "__main__":
    main()
