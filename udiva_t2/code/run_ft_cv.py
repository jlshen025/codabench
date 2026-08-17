"""Leave-sessions-out cross-validation of the full pipeline (the number quoted in the fact sheet).

The fine-tuned component is represented by its out-of-fold probabilities: ft_video.py --fold i
writes ft_fold<i>.npz, in which every session is predicted while held out, so composing them
here is leak-free. Non-verbal = those (subject, high-level action) probabilities expanded by
udiva.models_video.compose_nonverbal with per-fold priors; verbal = the cue-level TF-IDF model.
Scoring uses our re-implementation of the official mAP (udiva/metric.py).

The submitted entry additionally fuses three frozen backbones into the non-verbal posteriors
(see predict_submission.py); this script evaluates the fine-tuned component alone.
"""
import glob
import numpy as np
from udiva import cv
from udiva.models_text import VerbalCueModel
from udiva.models_video import compose_nonverbal

T2 = "<scratch>/t2/feats"


def load_ft_oof(folddir=T2, pattern="ft_fold*.npz"):
    """{(sid, segkey): {(subject, high-level action): probability}} pooled over the fold files."""
    oof = {}
    files = sorted(glob.glob(f"{folddir}/{pattern}"))
    for f in files:
        d = np.load(f, allow_pickle=True)
        classes = [tuple(c.split("|")) for c in d["classes"]]
        for r, si in enumerate(d["seg_index"]):
            sid, sk = si.split("|")
            oof[(sid, sk)] = {classes[j]: float(d["probs"][r, j]) for j in range(len(classes))}
    return oof, files


if __name__ == "__main__":
    oof, files = load_ft_oof()
    print(f"loaded {len(files)} fold files, {len(oof)} segments with out-of-fold probabilities")
    print(f"sessions covered: {len(set(sid for sid, sk in oof))}/21")

    def predict_full(train, test):
        vp = VerbalCueModel(ku=8, kt=12, pu_thresh=0.02, pt_thresh=0.02).fit(train).predict(test)
        npd = compose_nonverbal(train, test, oof)
        return {"verbal": vp["verbal"], "nonverbal": npd["nonverbal"]}

    res = cv.repeated_oof(predict_full, k=7, seeds=[0])
    print(f"FT-PIPELINE mAP={res['mAP_mean']:.4f} | verbal={res['mAP_verbal_mean']:.4f} "
          f"nonverbal={res['mAP_nonverbal_mean']:.4f}  (frozen baseline was 0.0109/v0.0068/h0.0149)")
