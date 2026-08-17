"""Cross-view probe: does the PARTNER's egocentric view predict this participant's future?

Third and last angle on the visual modality (after own-view appearance in diag_features.py
and own-view motion proxies in motion_scout.py). The camera wearer's own ego view mostly
shows their hands and the table; the partner's view is the one that actually contains the
participant's face/torso, so if any visual signal predicts a participant's next non-verbal
action, it should be strongest here.

Protocol identical to motion_scout.py: anchored grid, DINOv2 window features pooled over
[t_b-W, t_b], group-held-out split (every 4th session), multinomial logistic regression,
top-1/top-5 against the training-frequency prior. The only change is the view lookup:
participant_a is described by E2 (worn by B) and vice versa.

NOTE: this file re-implements the protocol of the recorded cross-view run; the original
one-off script was not preserved. The recorded result was top1 0.176 vs prior 0.176 and
top5 0.498 vs prior 0.493 -- i.e. exactly at the prior, like the other two visual angles.

Requires the DINOv2 cache: python experiments/extract_features.py (per session).
Run: python experiments/crossview_scout.py
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os, json
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from udiva import io as IO, feats as FT

OTHER = {"participant_a": "participant_b", "participant_b": "participant_a"}
W = 2.0


def crossview_feat(sess, p, t_b):
    """DINOv2 window feature of the PARTNER's ego view."""
    return FT.window_feat(sess, OTHER[p], t_b, W=W, pool="mean_last")


def main():
    sessions = IO.list_sessions()
    assert FT.have_features(sessions), "DINOv2 cache missing - run extract_features.py first"
    ds = IO.build_cv_dataset_anchored(sessions, delta=0.25, min_gap=2.0)
    X, Y, grp = [], [], []
    for s, segs in ds.items():
        for sid, seg in segs.items():
            for p in ("participant_a", "participant_b"):
                nv = [e for e in seg["participants"][p]["events"] if len(e) == 3]
                if nv:
                    X.append(crossview_feat(s, p, seg["t_b"]))
                    Y.append(nv[0][0] + ":" + nv[0][1])
                    grp.append(s)
    X = np.stack(X).astype(np.float32); Y = np.array(Y); grp = np.array(grp)
    held = set(sessions[::4]); tr = ~np.isin(grp, list(held)); te = np.isin(grp, list(held))
    c = LogisticRegression(max_iter=300, C=0.5).fit(X[tr], Y[tr]); cls = c.classes_
    pb = c.predict_proba(X[te]); order = np.argsort(-pb, 1)[:, :5]
    top1 = float((cls[np.argmax(pb, 1)] == Y[te]).mean())
    top5 = float(np.mean([Y[te][i] in cls[order[i]] for i in range(te.sum())]))
    pri1 = float((Y[te] == Counter(Y[tr]).most_common(1)[0][0]).mean())
    pri5 = float(np.isin(Y[te], [l for l, _ in Counter(Y[tr]).most_common(5)]).mean())
    print(f"CROSS-VIEW nonverbal first-action ({len(cls)} cls, test n={te.sum()}): "
          f"logreg top1={top1:.3f} top5={top5:.3f} | prior top1={pri1:.3f} top5={pri5:.3f}",
          flush=True)
    od = os.environ.get("EXP_OUTPUT_DIR", ".")
    json.dump({"top1": top1, "top5": top5, "prior_top1": pri1, "prior_top5": pri5},
              open(os.path.join(od, "result.json"), "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
