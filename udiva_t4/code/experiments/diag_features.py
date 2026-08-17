"""Diagnostic: do DINOv2 ego features predict the next event? Group-held-out linear probe."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
os.environ["OMP_NUM_THREADS"] = "8"; os.environ["OPENBLAS_NUM_THREADS"] = "8"; os.environ["MKL_NUM_THREADS"] = "8"
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from udiva import io as IO, feats as FT

sessions = IO.list_sessions()
ds = IO.build_cv_dataset(sessions, stride=2.0)
X = []; Yfull = []; Y3 = []; grp = []
for s, segs in ds.items():
    for sid, seg in segs.items():
        for p in ("participant_a", "participant_b"):
            evs = seg["participants"][p]["events"]
            X.append(FT.window_feat(s, p, seg["t_b"], W=2.0, pool="mean"))
            if not evs:
                Yfull.append("EMPTY"); Y3.append("EMPTY")
            elif len(evs[0]) == 2:
                Yfull.append("VERBAL"); Y3.append("VERBAL")
            else:
                Yfull.append(evs[0][0] + ":" + evs[0][1]); Y3.append("NV")
            grp.append(s)
X = np.stack(X).astype(np.float32); Yfull = np.array(Yfull); Y3 = np.array(Y3); grp = np.array(grp)
held = set(sessions[::4]); tr = ~np.isin(grp, list(held)); te = np.isin(grp, list(held))
print(f"n={len(Yfull)} test_n={te.sum()} heldsess={len(held)}", flush=True)

# 3-class type
c3 = LogisticRegression(max_iter=300, C=0.5).fit(X[tr], Y3[tr])
pr3 = Counter(Y3[tr]).most_common(1)[0][0]
print(f"TYPE 3-class: logreg={ (c3.predict(X[te])==Y3[te]).mean():.3f}  prior={ (Y3[te]==pr3).mean():.3f}", flush=True)

# first-event multiclass top1/top5
c = LogisticRegression(max_iter=300, C=0.5).fit(X[tr], Yfull[tr])
proba = c.predict_proba(X[te]); cls = c.classes_
pred = cls[np.argmax(proba, 1)]
order = np.argsort(-proba, 1)[:, :5]
top5 = np.mean([Yfull[te][i] in cls[order[i]] for i in range(te.sum())])
prior = (Yfull[te] == Counter(Yfull[tr]).most_common(1)[0][0]).mean()
prior5 = np.isin(Yfull[te], [l for l, _ in Counter(Yfull[tr]).most_common(5)]).mean()
print(f"FIRST-EVENT {len(cls)}-class: logreg top1={(pred==Yfull[te]).mean():.3f} top5={top5:.3f}  | prior top1={prior:.3f} top5={prior5:.3f}", flush=True)
print("DONE", flush=True)
