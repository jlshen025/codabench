"""Session-phase-conditioned hedge-pack: bucket cells by normalized time-in-session
(a phase proxy: early=read/discuss, late=assemble/verify) and fit a separate hedge-pack
per phase. Last consult-flagged genuinely-new lever, targets nonverbal/full."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os, json
os.environ["OMP_NUM_THREADS"] = "8"; os.environ["OPENBLAS_NUM_THREADS"] = "8"; os.environ["MKL_NUM_THREADS"] = "8"
from udiva import io as IO, cv as CV, metric as M
from udiva.baselines import GreedyK5Predictor, rich_pool

sessions = IO.list_sessions()
ds = IO.build_cv_dataset_anchored(sessions, delta=0.25, min_gap=2.0)
pf = lambda d: rich_pool(d, n_v=12, n_nv=10, mix=5)
_DUR = {s: IO.session_duration(IO.load_annotations(s)) for s in sessions}
res = {}


def show(n, r):
    res[n] = {k: round(r[k], 4) for k in list(M.SUBTASKS) + ["mean"]}
    print(f"  {n:20s} " + " ".join(f"{k}={r[k]:.3f}" for k in M.SUBTASKS) + f" MEAN={r['mean']:.4f}", flush=True)


class PhaseHedge:
    def __init__(self, nb=2, max_cells=2500):
        self.nb = nb; self.max_cells = max_cells; self.models = {}

    def _b(self, sess, t_b):
        return min(self.nb - 1, int(self.nb * t_b / max(_DUR[sess], 1e-6)))

    def fit(self, train):
        from collections import defaultdict
        buckets = defaultdict(dict)
        for s, segs in train.items():
            for sid, seg in segs.items():
                buckets[self._b(s, seg["t_b"])].setdefault(s, {})[sid] = seg
        for b, sub in buckets.items():
            g = GreedyK5Predictor(max_cells=self.max_cells, pool_fn=pf); g.fit(sub)
            self.models[b] = g.alts
        self._fallback = next(iter(self.models.values()))

    def predict(self, s, seg, p):
        return self.models.get(self._b(s, seg["t_b"]), self._fallback)


show("base", CV.leave_session_out(lambda: GreedyK5Predictor(max_cells=2500, pool_fn=pf), ds))
for nb in (2, 3, 4):
    show(f"phase nb={nb}", CV.leave_session_out(lambda nb=nb: PhaseHedge(nb=nb), ds))
od = os.environ.get("EXP_OUTPUT_DIR", "/tmp")
json.dump(res, open(os.path.join(od, "result.json"), "w"), indent=1)
print("DONE", flush=True)
