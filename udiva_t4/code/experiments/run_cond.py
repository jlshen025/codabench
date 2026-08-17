"""Conditional verbal-type hedge (bge-m3) vs the hedge-pack base, anchored LOSO."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
os.environ["OMP_NUM_THREADS"] = "8"; os.environ["OPENBLAS_NUM_THREADS"] = "8"; os.environ["MKL_NUM_THREADS"] = "8"
from udiva import io as IO, cv as CV, metric as M
from udiva.baselines import GreedyK5Predictor, rich_pool
from udiva.models import CondVerbalHedge

sessions = IO.list_sessions()
ds = IO.build_cv_dataset_anchored(sessions, delta=0.25, min_gap=2.0)
pf = lambda d: rich_pool(d, n_v=12, n_nv=10, mix=5)


def show(n, r):
    print(f"  {n:24s} " + " ".join(f"{k}={r[k]:.3f}" for k in M.SUBTASKS) + f" MEAN={r['mean']:.4f}", flush=True)


show("hedge-pack base", CV.leave_session_out(lambda: GreedyK5Predictor(max_cells=2500, pool_fn=pf), ds))
for nt in (2, 3):
    for W in (4.0, 8.0):
        show(f"cond-verbal nt={nt} W={W}",
             CV.leave_session_out(lambda nt=nt, W=W: CondVerbalHedge(n_types=nt, W=W), ds))
print("DONE", flush=True)
