"""Does bge-m3 recent transcript predict the next utterance type / verbal activity?
Gate for the transcript-content verbal lever."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
os.environ["OMP_NUM_THREADS"] = "8"; os.environ["OPENBLAS_NUM_THREADS"] = "8"; os.environ["MKL_NUM_THREADS"] = "8"
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from udiva import io as IO

from udiva import feats as FT
TE = FT.TEXT_DIR                          # $UDIVA_TEXT_DIR
SPK = {"participant_a": 0, "participant_b": 1}
_tc = {}


def text_window(sess, p, t_b, W=4.0):
    if sess not in _tc:
        d = np.load(os.path.join(TE, f"{sess}.npz"))
        _tc[sess] = (d["emb"].astype(np.float32), d["starts"], d["spk"])
    emb, starts, spk = _tc[sess]
    rec = (starts > t_b - W) & (starts < t_b)
    me = rec & (spk == SPK[p]); other = rec & (spk != SPK[p]) & (spk != 2)
    dim = emb.shape[1] if emb.size else 1024
    def pool(mask):
        return emb[mask].mean(0) if mask.any() else np.zeros(dim, np.float32)
    return np.concatenate([pool(me), pool(other), (emb[rec][-1] if rec.any() else np.zeros(dim, np.float32))])


sessions = IO.list_sessions()
ds = IO.build_cv_dataset_anchored(sessions, delta=0.25, min_gap=2.0)
X = []; Yv = []; act = []; grp = []
for s, segs in ds.items():
    for sid, seg in segs.items():
        for p in ("participant_a", "participant_b"):
            evs = seg["participants"][p]["events"]
            verb = [e for e in evs if len(e) == 2]
            X.append(text_window(s, p, seg["t_b"]))
            act.append(1 if verb else 0)
            Yv.append(verb[0][0] if verb else "NONE")
            grp.append(s)
X = np.stack(X).astype(np.float32); Yv = np.array(Yv); act = np.array(act); grp = np.array(grp)
held = set(sessions[::4]); tr = ~np.isin(grp, list(held)); te = np.isin(grp, list(held))
print(f"n={len(Yv)} test={te.sum()}", flush=True)

# (1) predict verbal-active (binary)
cb = LogisticRegression(max_iter=300, C=0.5).fit(X[tr], act[tr])
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(act[te], cb.predict_proba(X[te])[:, 1])
base = act[te].mean()
print(f"VERBAL-ACTIVE: logreg AUC={auc:.3f} (base rate={base:.3f}, acc-prior={max(base,1-base):.3f}, logreg-acc={(cb.predict(X[te])==act[te]).mean():.3f})", flush=True)

# (2) predict first verbal type on verbal-active cells
va_tr = tr & (act == 1); va_te = te & (act == 1)
c = LogisticRegression(max_iter=400, C=0.5).fit(X[va_tr], Yv[va_tr]); cls = c.classes_
pb = c.predict_proba(X[va_te]); order = np.argsort(-pb, 1)[:, :5]
top1 = (cls[np.argmax(pb, 1)] == Yv[va_te]).mean()
top5 = np.mean([Yv[va_te][i] in cls[order[i]] for i in range(va_te.sum())])
pri1 = (Yv[va_te] == Counter(Yv[va_tr]).most_common(1)[0][0]).mean()
pri5 = np.isin(Yv[va_te], [l for l, _ in Counter(Yv[va_tr]).most_common(5)]).mean()
print(f"FIRST-VERBAL-TYPE (verbal-active only, {len(cls)} types): logreg top1={top1:.3f} top5={top5:.3f} | prior top1={pri1:.3f} top5={pri5:.3f}", flush=True)
import json
od = os.environ.get("EXP_OUTPUT_DIR", "/tmp")
json.dump({"verbal_active_auc": float(auc), "verbal_active_base": float(base),
           "first_type_logreg_top1": float(top1), "first_type_logreg_top5": float(top5),
           "first_type_prior_top1": float(pri1), "first_type_prior_top5": float(pri5),
           "n_classes": int(len(cls))}, open(os.path.join(od, "result.json"), "w"))
print("DONE", flush=True)
