"""LOSO CV for the DINOv2 FeatureKNN predictors vs the static GreedyK5 baseline.
Run after feature extraction completes. python run_knn_cv.py
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import time
from udiva import io as IO, cv as CV, metric as M, feats as FT
from udiva.baselines import GreedyK5Predictor
from udiva.models import FeatureKNN

sessions = IO.list_sessions()
assert FT.have_features(sessions), "features missing — run extract_features first"
ds = IO.build_cv_dataset(sessions, stride=2.0)


def show(name, r):
    print(f"  {name:30s} " + " ".join(f"{k}={r[k]:.3f}" for k in M.SUBTASKS) + f"  MEAN={r['mean']:.4f}")


print("=== baseline ===")
show("GreedyK5 static", CV.leave_session_out(lambda: GreedyK5Predictor(max_cells=2500), ds))
print("=== FeatureKNN variants (LOSO) ===")
for W in (1.0, 2.0):
    for pool in ("mean", "mean_last"):
        t = time.time()
        show(f"KNN W={W} {pool}",
             CV.leave_session_out(lambda: FeatureKNN(W=W, pool=pool), ds))
        print(f"     ({time.time()-t:.0f}s)")
print("=== Hybrid KNN + static seed ===")
for ms in (1, 2, 3):
    show(f"KNN+static seed={ms}",
         CV.leave_session_out(lambda: FeatureKNN(W=2.0, pool="mean_last", mix_static=ms), ds))
