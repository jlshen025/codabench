"""Scout the motion-video family CHEAPLY: derive motion-proxy features from the cached
DINOv2 window (temporal diff / std / frame-to-frame change) and test if they predict the
first nonverbal action better than the prior. Appearance gave 0; this checks if change does."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os, json
os.environ["OMP_NUM_THREADS"] = "8"; os.environ["OPENBLAS_NUM_THREADS"] = "8"; os.environ["MKL_NUM_THREADS"] = "8"
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from udiva import io as IO, feats as FT

VIEW = {"participant_a": "e1", "participant_b": "e2"}


def motion_feat(sess, p, t_b, W=2.0):
    feats, times = FT.load_session(sess)[VIEW[p]]
    m = (times > t_b - W) & (times <= t_b)
    sub = feats[m]
    if len(sub) < 2:
        idx = max(0, int(np.searchsorted(times, t_b)) - 1)
        sub = feats[max(0, idx - 1):idx + 1]
        if len(sub) < 2:
            return np.zeros(feats.shape[1] * 3, np.float32)
    diff = sub[-1] - sub[0]
    std = sub.std(0)
    fdiff = np.abs(np.diff(sub, axis=0)).mean(0)
    v = np.concatenate([diff, std, fdiff]).astype(np.float32)
    n = np.linalg.norm(v) + 1e-8
    return v / n


sessions = IO.list_sessions()
ds = IO.build_cv_dataset_anchored(sessions, delta=0.25, min_gap=2.0)
X = []; Y = []; grp = []
for s, segs in ds.items():
    for sid, seg in segs.items():
        for p in ("participant_a", "participant_b"):
            evs = seg["participants"][p]["events"]
            nv = [e for e in evs if len(e) == 3]
            if nv:
                X.append(motion_feat(s, p, seg["t_b"])); Y.append(nv[0][0] + ":" + nv[0][1]); grp.append(s)
X = np.stack(X).astype(np.float32); Y = np.array(Y); grp = np.array(grp)
held = set(sessions[::4]); tr = ~np.isin(grp, list(held)); te = np.isin(grp, list(held))
c = LogisticRegression(max_iter=300, C=0.5).fit(X[tr], Y[tr]); cls = c.classes_
pb = c.predict_proba(X[te]); order = np.argsort(-pb, 1)[:, :5]
top1 = (cls[np.argmax(pb, 1)] == Y[te]).mean()
top5 = np.mean([Y[te][i] in cls[order[i]] for i in range(te.sum())])
pri1 = (Y[te] == Counter(Y[tr]).most_common(1)[0][0]).mean()
pri5 = np.isin(Y[te], [l for l, _ in Counter(Y[tr]).most_common(5)]).mean()
print(f"MOTION-PROXY nonverbal first-action ({len(cls)} cls, test n={te.sum()}): "
      f"logreg top1={top1:.3f} top5={top5:.3f} | prior top1={pri1:.3f} top5={pri5:.3f}", flush=True)
od = os.environ.get("EXP_OUTPUT_DIR", "/tmp")
json.dump({"top1": float(top1), "top5": float(top5), "prior_top1": float(pri1), "prior_top5": float(pri5)},
          open(os.path.join(od, "result.json"), "w"))
print("DONE", flush=True)
