"""Leave-session-out CV harness for UDIVA-HHOI Track 4.

A Predictor implements:
  fit(train_ds)              -> None    (learn priors from training sessions)
  predict(session, seg_meta) -> events  (single seq OR list of <=5 alt seqs)
where seg_meta = {'t_b','t_e','participants':{...}} (GT events are NOT read by predictors).

leave_session_out builds a held-out prediction dataset fold by fold and scores it
with the SDL 4-subtask metric. Returns mean subtask scores across all held-out cells.
"""
import copy
from . import metric as M
from . import io as IO


def _blank_seg(seg):
    """Copy a segment's structure but with empty events (what a predictor sees)."""
    return {"t_b": seg["t_b"], "t_e": seg["t_e"],
            "participants": {p: {"events": []} for p in seg["participants"]}}


def predict_session(predictor, session, segs):
    out = {}
    for sid, seg in segs.items():
        blank = _blank_seg(seg)
        pseg = {"t_b": seg["t_b"], "t_e": seg["t_e"], "participants": {}}
        for p in seg["participants"]:
            pred = predictor.predict(session, blank, p)
            pseg["participants"][p] = {"events": pred}
        out[sid] = pseg
    return out


def leave_session_out(predictor_factory, ds, sessions=None, verbose=False):
    sessions = sessions or sorted(ds.keys())
    # accumulate per-subtask sums over all held-out cells (pooled, not macro-by-session)
    full_ref = {}
    full_pred = {}
    for held in sessions:
        train_ds = {s: ds[s] for s in sessions if s != held}
        predictor = predictor_factory()
        predictor.fit(train_ds)
        pred_segs = predict_session(predictor, held, ds[held])
        full_ref[held] = ds[held]
        full_pred[held] = pred_segs
        if verbose:
            r = M.score_dataset({held: ds[held]}, {held: pred_segs})
            print(f"  fold {held}: " + " ".join(f"{k}={r[k]:.3f}" for k in M.SUBTASKS))
    return M.score_dataset(full_ref, full_pred)


def evaluate_static(predictor, ds, sessions=None):
    """Score a pre-fit (input-free) predictor on the whole ds (no refitting)."""
    sessions = sessions or sorted(ds.keys())
    ref = {s: ds[s] for s in sessions}
    pred = {s: predict_session(predictor, s, ds[s]) for s in sessions}
    return M.score_dataset(ref, pred)
