"""
assemble_lora.py — turn raw LoRA fold outputs into seed-avg, epoch-selected dev+loto npzs
in the SAME format eval_configA.py / blend_eval.py consume (proba,y_true,target).

For a model family (q14 / fnr):
  * dev  npz  = seed-avg over the dev-fold dirs of mean(proba_ep{E}) -> (1400,3)
  * loto npz  = 2721-row array assembled by TARGET from the LOTO folds:
        Covid rows    <- hcov-fold seed-avg mean(proba_ep{E})
        Digital rows  <- hdig-fold seed-avg mean(proba_ep{E})
    aligned to the ENC loto row order (t2_base_ara_loto_s42) and y_true-checked.

Epoch policy mirrors the held-best: q14 -> "1", fnr -> "2,3".
"""
import os, sys, glob, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_llm")
COV, DIG = "Covid Vaccine", "Digital Transformation"


def seedavg_epochs(dirs, epochs):
    """mean over seeds of [ mean over epochs of proba_ep{e} ]."""
    P = []
    y = None; t = None
    for d in dirs:
        f = os.path.join(d, "oof_proba.npz")
        z = np.load(f, allow_pickle=True)
        pe = np.mean([z[f"proba_ep{e}"] for e in epochs], 0)
        P.append(pe)
        y = z["y_true"]; t = z["target"].astype(str)
    return np.mean(P, 0), y, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="e.g. saug_q14 / saug_fnr")
    ap.add_argument("--epochs", required=True, help="comma epochs to avg, e.g. 1 or 2,3")
    ap.add_argument("--seeds", default="42,1")
    ap.add_argument("--out_prefix", required=True, help="e.g. _llm/q14aug -> _llm/q14aug_devt2.npz + _llm/q14aug_loto.npz")
    args = ap.parse_args()
    E = [int(x) for x in args.epochs.split(",")]
    S_ = args.seeds.split(",")

    # dev
    dev_dirs = [f"{ROOT}/{args.tag}_dev_s{s}" for s in S_]
    dev_p, dev_y, dev_t = seedavg_epochs(dev_dirs, E)
    assert dev_p.shape == (1400, 3), dev_p.shape
    np.savez(f"{args.out_prefix}_devt2.npz", proba=dev_p.astype(np.float32), y_true=dev_y, target=dev_t)

    # loto: align to ENC loto order
    enc = np.load(f"{ROOT}/t2_base_ara_loto_s42/oof_proba.npz", allow_pickle=True)
    enc_t = enc["target"].astype(str); enc_y = enc["y_true"]
    cov_p, cov_y, _ = seedavg_epochs([f"{ROOT}/{args.tag}_hcov_s{s}" for s in S_], E)
    dig_p, dig_y, _ = seedavg_epochs([f"{ROOT}/{args.tag}_hdig_s{s}" for s in S_], E)
    out = np.zeros((len(enc_t), 3), np.float32)
    cm, dm = enc_t == COV, enc_t == DIG
    assert cm.sum() == len(cov_p), f"Covid count {cm.sum()} vs fold {len(cov_p)}"
    assert dm.sum() == len(dig_p), f"Digital count {dm.sum()} vs fold {len(dig_p)}"
    assert np.array_equal(enc_y[cm], cov_y), "Covid y_true misaligned vs ENC"
    assert np.array_equal(enc_y[dm], dig_y), "Digital y_true misaligned vs ENC"
    out[cm] = cov_p; out[dm] = dig_p
    np.savez(f"{args.out_prefix}_loto.npz", proba=out, y_true=enc_y, target=enc_t)
    print(f"[assembled] {args.tag} epochs={E} seeds={S_}")
    print(f"  dev  -> {args.out_prefix}_devt2.npz  (1400,3)")
    print(f"  loto -> {args.out_prefix}_loto.npz   ({len(enc_t)},3)  Covid={cm.sum()} Digital={dm.sum()}")


if __name__ == "__main__":
    main()
