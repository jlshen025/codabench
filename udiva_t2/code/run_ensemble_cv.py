"""Leave-sessions-out cross-validation of the SUBMITTED configuration (the CV number in the
fact sheet, mAP ~= 0.0137).

The evaluation mirrors the entry exactly: for each fold, (subject, high-level action) heads are
fitted on the training sessions of every frozen backbone and applied to the held-out sessions;
those posteriors are fused with equal weights together with the fine-tuned model's out-of-fold
posteriors (ft_video.py --fold i, which must have been run with the SAME k and seed); the fused
posteriors are expanded by compose_nonverbal, and the verbal channel is fitted per fold. All
out-of-fold predictions are pooled and scored once with udiva/metric.py, as the challenge scores
one mAP over the whole set.

Example:
  python run_ensemble_cv.py --feat_root $T2/feats \\
      --backbones videomae_large,videomae_ssv2,timesformer_ssv2 \\
      --ft_glob "$T2/feats/ftL_fold*.npz" --k 7 --seed 0
"""
import argparse, glob
import numpy as np
from udiva import cv, data as D
from udiva.models_text import VerbalCueModel
from udiva.models_video import compose_nonverbal, fit_pair_heads, predict_pair_probs


def frozen_oof(sids, feat_dir, k, seed):
    """{(sid, segkey): {(subject, h): prob}} predicted while the session was held out."""
    oof = {}
    for i, fold in enumerate(cv.make_folds(sids, k=k, seed=seed)):
        train = [s for s in sids if s not in fold]
        labels, clf = fit_pair_heads(train, feat_dir)
        oof.update(predict_pair_probs(fold, feat_dir, labels, clf))
        print(f"    fold {i}: {len(fold)} sessions held out", flush=True)
    return oof


def ft_oof(pattern):
    """Out-of-fold posteriors of the fine-tuned component (ft_video.py --fold i)."""
    oof = {}
    files = sorted(glob.glob(pattern))
    for f in files:
        d = np.load(f, allow_pickle=True)
        classes = [tuple(c.split("|")) for c in d["classes"]]
        for r, si in enumerate(d["seg_index"]):
            sid, sk = si.split("|")
            oof[(sid, sk)] = {classes[j]: float(d["probs"][r, j]) for j in range(len(classes))}
    print(f"  fine-tuned component: {len(files)} fold files, {len(oof)} segments", flush=True)
    return oof


def fuse(components, weights):
    """S(rho,h) = sum_b w_b * P_b(rho,h | x), the equal-weight late fusion of the entry."""
    accum = {}
    for comp, w in zip(components, weights):
        for key, row in comp.items():
            acc = accum.setdefault(key, {})
            for c, p in row.items():
                acc[c] = acc.get(c, 0.0) + w * p
    return accum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_root", default="<scratch>/t2/feats")
    ap.add_argument("--backbones", default="videomae_large,videomae_ssv2,timesformer_ssv2")
    ap.add_argument("--ft_glob", default="", help="ft_video.py --fold out_npz files; empty = "
                                                  "frozen-only ensemble")
    ap.add_argument("--weights", default="", help="comma list, one per component "
                                                  "(backbones then fine-tune); default all 1")
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0, help="MUST match the folds the fine-tuned "
                                                        "component was trained with")
    ap.add_argument("--kh", type=int, default=14)
    ap.add_argument("--ph", type=float, default=0.02)
    args = ap.parse_args()

    sids = D.all_sids()
    components = []
    for b in args.backbones.split(","):
        print(f"  frozen backbone {b}", flush=True)
        components.append(frozen_oof(sids, f"{args.feat_root}/{b}", args.k, args.seed))
    if args.ft_glob:
        components.append(ft_oof(args.ft_glob))
    weights = ([float(w) for w in args.weights.split(",")] if args.weights
               else [1.0] * len(components))
    assert len(weights) == len(components), "one weight per component"
    fused = fuse(components, weights)

    def predict_full(train, test):
        verbal = VerbalCueModel(ku=8, kt=12, pu_thresh=0.02, pt_thresh=0.02).fit(train)
        vp = verbal.predict(test)
        npd = compose_nonverbal(train, test, fused, kh=args.kh, ph=args.ph)
        return {"verbal": vp["verbal"], "nonverbal": npd["nonverbal"]}

    sc, _, _ = cv.run_oof(predict_full, sids=sids, k=args.k, seed=args.seed)
    print(f"\nCV over {len(sids)} sessions (k={args.k}, seed={args.seed}), "
          f"{len(components)} fused components, weights {weights}:")
    print(f"  mAP = {sc['mAP']:.4f}   (verbal {sc['mAP_verbal']:.4f} / "
          f"non-verbal {sc['mAP_nonverbal']:.4f})")


if __name__ == "__main__":
    main()
