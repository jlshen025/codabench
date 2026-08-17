"""Development experiment: transcript-conditioned verbal hedge vs. the static hedge-pack.

This is a REJECTED development model, not the submitted system. It replaces the verbal
event types of the static hedge-pack template by the top-n utterance types predicted per
cell from the recent transcript (bge-m3 embeddings + logistic regression), and scores it
on the two protocols used for selection: anchored LOSO over the 21 development sessions,
and the exact organizer development grid (fit on the 19 non-dev sessions).

Every number is measured at run time; nothing is hardcoded. Note for readers comparing
against the fact sheet: the 0.4501 anchored-LOSO value recorded during development came
from an EARLIER variant of `CondVerbalHedge` that fed raw concatenated bge-m3 embeddings
to the logistic regression. The version in `udiva/models.py` (the one shipped here)
standardizes the features first, and measures 0.4411 — i.e. BELOW the static pack's
0.4441. The +0.006/-0.003 swing from a single preprocessing choice is why this family was
judged to be inside the noise of the 21-session protocol and was not selected. See the
fact sheet, Sec. II-F.

Requires the precomputed bge-m3 transcript embeddings: run `python embed_transcripts.py`
first (see udiva/feats.py:TEXT_DIR).

Usage: python run_devval.py [--save heldbest_cond.pkl]
"""
import argparse
import os
import pickle

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

from udiva import io as IO, cv as CV, metric as M
from udiva.baselines import GreedyK5Predictor, rich_pool
from udiva.models import CondVerbalHedge

POOL = dict(n_v=12, n_nv=10, mix=5)
DEV_SESSIONS = ("001080", "181182")


def show(n, r):
    print(f"  {n:24s} " + " ".join(f"{k}={r[k]:.3f}" for k in M.SUBTASKS)
          + f" MEAN={r['mean']:.4f}", flush=True)
    return r["mean"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="heldbest_cond.pkl")
    args = ap.parse_args()

    sessions = IO.list_sessions()
    ds = IO.build_cv_dataset_anchored(sessions, delta=0.25, min_gap=2.0)
    ref = IO.load_dev_reference()
    ds_nondev = {s: ds[s] for s in sessions if s not in DEV_SESSIONS}
    pf = lambda d: rich_pool(d, **POOL)

    print("=== anchored LOSO over the 21 development sessions ===", flush=True)
    cv_static = show("static hedge-pack", CV.leave_session_out(
        lambda: GreedyK5Predictor(max_cells=2500, pool_fn=pf), ds))
    cv_cond = show("cond-verbal nt3 W4", CV.leave_session_out(
        lambda: CondVerbalHedge(n_types=3, W=4.0), ds))
    print(f"  DELTA (cond - static) = {cv_cond - cv_static:+.4f}", flush=True)

    print("=== exact organizer development grid (fit on 19 non-dev sessions) ===", flush=True)
    gs = GreedyK5Predictor(max_cells=2500, pool_fn=pf); gs.fit(ds_nondev)
    dev_static = show("static hedge-pack", M.score_dataset(
        ref, {s: CV.predict_session(gs, s, ref[s]) for s in ref}))
    gc = CondVerbalHedge(n_types=3, W=4.0); gc.fit(ds_nondev)
    dev_cond = show("cond-verbal nt3 W4", M.score_dataset(
        ref, {s: CV.predict_session(gc, s, ref[s]) for s in ref}))
    print(f"  DELTA (cond - static) = {dev_cond - dev_static:+.4f}", flush=True)
    print(f"  SELECTED: {'static' if cv_cond <= cv_static or dev_cond <= dev_static else 'cond'}"
          " hedge-pack (the conditional model must win on BOTH protocols to be selected)",
          flush=True)

    # fit on all 21 and save, so the rejected model can be inspected/rebuilt
    final = CondVerbalHedge(n_types=3, W=4.0); final.fit(ds)
    pickle.dump({"clf": final.clf, "scaler": final.scaler, "classes": final.classes,
                 "base_alts": final.base_alts, "n_types": 3, "W": 4.0,
                 "cv_anchored_loso": cv_cond, "cv_dev_organizer": dev_cond},
                open(args.save, "wb"))
    print(f"saved {args.save}; base_alts={final.base_alts}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
