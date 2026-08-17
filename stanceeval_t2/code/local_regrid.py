"""local_regrid.py — grid the LOCAL blend {ENC, q14, fnr [, cand]} on OOF (dev + Covid/Digital LOTO).
Post-fence (claude/gpt5 dropped): find best mean3 / best min3 weights. Optional --cand_{dev,loto}
adds a 4th voter (a fresh relay grounded voter) to test OOF-decorrelation gain over the local base.
ENC = train-only t2_base seed-avg OOF (held-out, matches CV); q14/fnr = held OOF npz."""
import os, sys, glob, argparse, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_llm")
COV, DIG = "Covid Vaccine", "Digital Transformation"


def enc_seedavg(cv):
    P = []; y = None; t = None
    for pfx in ("t2_base_ara", "t2_base_marb"):
        fs = sorted(glob.glob(f"{ROOT}/{pfx}_{cv}_s*/oof_proba.npz"))
        assert fs, f"no ENC {pfx}_{cv}"
        sp = [np.load(f, allow_pickle=True) for f in fs]
        P.append(np.mean([s["proba"] for s in sp], 0)); y = sp[0]["y_true"]
        if "target" in sp[0].files:
            t = sp[0]["target"].astype(str)
    return np.mean(P, 0), y, t


def loadp(p):
    return np.load(p, allow_pickle=True)["proba"].astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q14_dev", default=f"{LL}/q14sa_ep1_devt2.npz")
    ap.add_argument("--q14_loto", default=f"{LL}/q14sa_ep1_loto.npz")
    ap.add_argument("--fnr_dev", default=f"{LL}/fnr_sa_ep23_devt2.npz")
    ap.add_argument("--fnr_loto", default=f"{LL}/fnr_sa_ep23_loto.npz")
    ap.add_argument("--cand_dev", default="")
    ap.add_argument("--cand_loto", default="")
    ap.add_argument("--cand_name", default="cand")
    ap.add_argument("--step", type=float, default=0.05)
    a = ap.parse_args()

    enc_d, yd, _ = enc_seedavg("dev")
    enc_l, yl, tl = enc_seedavg("loto")
    names = ["ENC", "q14", "fnr"]
    dev = {"ENC": enc_d, "q14": loadp(a.q14_dev), "fnr": loadp(a.fnr_dev)}
    lo = {"ENC": enc_l, "q14": loadp(a.q14_loto), "fnr": loadp(a.fnr_loto)}
    if a.cand_dev:
        names.append(a.cand_name)
        dev[a.cand_name] = loadp(a.cand_dev); lo[a.cand_name] = loadp(a.cand_loto)
    for n in names:
        assert dev[n].shape == (1400, 3), f"{n} dev {dev[n].shape}"
        assert lo[n].shape == (2721, 3), f"{n} loto {lo[n].shape}"
    cov_m, dig_m = tl == COV, tl == DIG

    def fav(y, p, m=None):
        mm = np.ones(len(y), bool) if m is None else m
        return S.compute_metrics(y[mm], p[mm].argmax(1))["Favg2"] * 100

    def score(W):
        pd_ = sum(W[n] * dev[n] for n in names); pl_ = sum(W[n] * lo[n] for n in names)
        d = fav(yd, pd_); c = fav(yl, pl_, cov_m); g = fav(yl, pl_, dig_m)
        return d, c, g, (d + c + g) / 3.0, min(d, c, g)

    print("SINGLES   WomenEmp | Covid  | Digital| mean3")
    for n in names:
        d = fav(yd, dev[n]); c = fav(yl, lo[n], cov_m); g = fav(yl, lo[n], dig_m)
        print(f"  {n:6s}  {d:6.2f}  | {c:6.2f} | {g:6.2f} | {(d+c+g)/3:6.2f}")

    grid = np.round(np.arange(0, 1.0 + 1e-9, a.step), 4)
    best_m3 = best_mn = None
    for combo in itertools.product(grid, repeat=len(names)):
        if abs(sum(combo) - 1.0) > 1e-6:
            continue
        W = dict(zip(names, combo))
        d, c, g, m3, mn3 = score(W)
        if best_m3 is None or m3 > best_m3[0] or (m3 == best_m3[0] and mn3 > best_m3[1]):
            best_m3 = (m3, mn3, dict(W), d, c, g)
        if best_mn is None or mn3 > best_mn[1] or (mn3 == best_mn[1] and m3 > best_mn[0]):
            best_mn = (m3, mn3, dict(W), d, c, g)

    def show(tag, b):
        w = {k: round(v, 2) for k, v in b[2].items() if v > 0}
        print(f"{tag}: mean3={b[0]:.2f} min3={b[1]:.2f} W={w} (dev {b[3]:.2f}/Cov {b[4]:.2f}/Dig {b[5]:.2f})")
    print()
    show("BEST-mean3", best_m3)
    show("BEST-min3 ", best_mn)


if __name__ == "__main__":
    main()
