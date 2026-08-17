"""
eval_configA.py — reproducible A/B eval of the held-best 5-way blend (config A) with
swappable q14 / Fanar LoRA npzs. Lets me compare synth-aug LoRAs against the frozen
held-best config WITHOUT re-tuning weights (Gate-2 honest).

Config A (held-best, 07-05): ENC 0.2 | claude 0.1 | gpt5 0.3 | q14 0.3 | fnr 0.1  → mean3 87.87 / min3 81.60
ENC = seed-avg(t2_base_ara, t2_base_marb) dev+loto OOF (auto).
Voters passed as npz paths; defaults = the current held-best.

Usage:
  python eval_configA.py                                  # reproduce held-best
  python eval_configA.py --q14_dev A.npz --q14_loto B.npz --fnr_dev C.npz --fnr_loto D.npz
Reports per-draw Favg2 (+ per-class F for the aim residual), mean3, min3, and LoRA SINGLES.
"""
import os, sys, glob, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_llm")
COV, DIG = "Covid Vaccine", "Digital Transformation"

# frozen held-best config A weights
WA = {"ENC": 0.2, "claude": 0.1, "gpt5": 0.3, "q14": 0.3, "fnr": 0.1}


def enc_seedavg(cv):
    P = []; y = None; t = None
    for pfx in ("t2_base_ara", "t2_base_marb"):
        fs = sorted(glob.glob(f"{ROOT}/{pfx}_{cv}_s*/oof_proba.npz"))
        assert fs, f"no ENC files {pfx}_{cv}"
        sp = [np.load(f, allow_pickle=True) for f in fs]
        P.append(np.mean([s["proba"] for s in sp], 0))
        y = sp[0]["y_true"]
        if "target" in sp[0].files:
            t = sp[0]["target"].astype(str)
    return np.mean(P, 0), y, t


def loadp(npz):
    return np.load(npz, allow_pickle=True)["proba"]


def fav(y, proba, mask=None):
    m = np.ones(len(y), bool) if mask is None else mask
    return S.compute_metrics(y[m], proba[m].argmax(1))["Favg2"] * 100


def perclass(y, proba, mask):
    mm = S.compute_metrics(y[mask], proba[mask].argmax(1))
    return mm["F_favor"] * 100, mm["F_against"] * 100, mm["F_none"] * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q14_dev", default=f"{LL}/q14sa_ep1_devt2.npz")
    ap.add_argument("--q14_loto", default=f"{LL}/q14sa_ep1_loto.npz")
    ap.add_argument("--fnr_dev", default=f"{LL}/fnr_sa_ep23_devt2.npz")
    ap.add_argument("--fnr_loto", default=f"{LL}/fnr_sa_ep23_loto.npz")
    ap.add_argument("--label", default="held-best")
    args = ap.parse_args()

    enc_dev, yd, _ = enc_seedavg("dev")
    enc_lo, yl, tl = enc_seedavg("loto")
    dev = {"ENC": enc_dev, "claude": loadp(f"{LL}/claude_grounded_devt2.npz"),
           "gpt5": loadp(f"{LL}/gpt5_grounded_devt2.npz"),
           "q14": loadp(args.q14_dev), "fnr": loadp(args.fnr_dev)}
    lo = {"ENC": enc_lo, "claude": loadp(f"{LL}/claude_grounded_loto.npz"),
          "gpt5": loadp(f"{LL}/gpt5_grounded_loto.npz"),
          "q14": loadp(args.q14_loto), "fnr": loadp(args.fnr_loto)}
    for n in dev:
        assert dev[n].shape == (1400, 3), f"{n} dev shape {dev[n].shape}"
        assert lo[n].shape == (2721, 3), f"{n} loto shape {lo[n].shape}"
    cov_m, dig_m = tl == COV, tl == DIG

    print(f"### {args.label}  |  q14={os.path.basename(args.q14_loto)}  fnr={os.path.basename(args.fnr_loto)}")
    print("SINGLES (Favg2)      WomenEmp | Covid  | Digital| mean3 |  Dig[F_fav/F_agn/F_non]")
    for n in ["ENC", "claude", "gpt5", "q14", "fnr"]:
        d = fav(yd, dev[n]); c = fav(yl, lo[n], cov_m); g = fav(yl, lo[n], dig_m)
        ff, fa, fn = perclass(yl, lo[n], dig_m)
        print(f"  {n:8s}  {d:6.2f}  | {c:6.2f} | {g:6.2f} | {(d+c+g)/3:5.2f} | {ff:5.1f}/{fa:5.1f}/{fn:5.1f}")

    pd_ = sum(WA[n] * dev[n] for n in WA)
    pl_ = sum(WA[n] * lo[n] for n in WA)
    d = fav(yd, pd_); c = fav(yl, pl_, cov_m); g = fav(yl, pl_, dig_m)
    ff, fa, fn = perclass(yl, pl_, dig_m)
    m3, mn3 = (d + c + g) / 3.0, min(d, c, g)
    print(f"\nCONFIG A (ENC.2/cl.1/g5.3/q14.3/fnr.1):")
    print(f"  dev={d:.2f}  Covid={c:.2f}  Digital={g:.2f}  |  mean3={m3:.2f}  min3={mn3:.2f}  |  Dig[F_fav/F_agn/F_non]={ff:.1f}/{fa:.1f}/{fn:.1f}")
    print(f"  >>> vs HELD-BEST 87.87/81.60 : mean3 {m3-87.87:+.2f}  min3 {mn3-81.60:+.2f}")


if __name__ == "__main__":
    main()
