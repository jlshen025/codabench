"""Verbal conditioning with dialogue-structure features vs bge-m3 vs combined, anchored LOSO.
Top lever from cross-model consults: turn-taking/timing/adjacency beats embeddings on small data."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
os.environ["OMP_NUM_THREADS"] = "8"; os.environ["OPENBLAS_NUM_THREADS"] = "8"; os.environ["MKL_NUM_THREADS"] = "8"
import json
from udiva import io as IO, cv as CV, metric as M
from udiva.baselines import GreedyK5Predictor, rich_pool
from udiva.models import CondVerbalHedge

sessions = IO.list_sessions()
ds = IO.build_cv_dataset_anchored(sessions, delta=0.25, min_gap=2.0)
pf = lambda d: rich_pool(d, n_v=12, n_nv=10, mix=5)
res = {}


def show(n, r):
    res[n] = {k: round(r[k], 4) for k in list(M.SUBTASKS) + ["mean"]}
    print(f"  {n:26s} " + " ".join(f"{k}={r[k]:.3f}" for k in M.SUBTASKS) + f" MEAN={r['mean']:.4f}", flush=True)


show("base hedge-pack", CV.leave_session_out(lambda: GreedyK5Predictor(max_cells=2500, pool_fn=pf), ds))
for feats in (("bge",), ("dialogue",), ("bge", "dialogue")):
    for nt in (3, 4):
        show(f"cond {'+'.join(feats)} nt={nt}",
             CV.leave_session_out(lambda feats=feats, nt=nt: CondVerbalHedge(n_types=nt, feats=feats), ds))
od = os.environ.get("EXP_OUTPUT_DIR", "/tmp")
json.dump(res, open(os.path.join(od, "result.json"), "w"), indent=1)
print("DONE", flush=True)
