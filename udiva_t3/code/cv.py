"""Leave-one-session-out CV harness for UDIVA-HHOI Track 3.

A predictor is an object with .predict(sid, seg) -> {participant: [alt_seq, ...]}.
A predictor_factory(train_sids, gt_all) -> predictor (fit on the train sessions only).

VALIDATION PROTOCOL (the one that produces the reported result; see README):
  * 21 leave-one-session-out folds over the 21 annotated development sessions. The
    predictor is refit on the 20 training sessions of each fold; every scored segment
    comes from the held-out session.
  * Evaluation timestamps: for the two sessions that the official starting kit ships an
    anticipation grid for (001080, 181182) that grid is used VERBATIM. For the other 19
    sessions no official grid exists, so a grid is generated: non-overlapping 2 s windows
    (t_b = 4.0, 6.0, 8.0, ... ; t_e = t_b + 2 s) up to the last annotated event, keeping
    the windows that contain at least one annotated event of at least one participant
    (`udiva_data.make_grid`). Totals: 2835 segments = 5670 (segment, participant) cells,
    of which 127 segments come from the two official grids.
  * Ground truth per window is parsed from the raw annotations with the rule verified
    against the official reference (`udiva_data`): event included iff t_b < start <= t_e.
  * Scoring: `sdl.py` (our re-implementation of the official normalized SDL), best-of-K
    per participant and per subtask, mean over participants, then over segments; `mean4`
    is the unweighted mean of the four subtask columns. `mean4_foldstd` is the
    fold-to-fold population std of mean4 = the noise floor of this protocol.

`python cv.py` reproduces the reported result; `python cv.py protocols` reports the
sensitivity of the result to the grid choices (stride, event-containing filter).
"""
import json
import statistics
import udiva_data as U
import sdl

OFFICIAL = ["001080", "181182"]
PA, PB = "participant_a", "participant_b"


def build_all_gt(stride=2.0, require_events=True, use_official=True):
    """{sid: {seg_id: reference-style segment}} for all 21 labeled sessions.

    Defaults = the reported protocol (official grid where it exists, else generated
    stride-2 s event-containing windows).
    """
    gt = {}
    ref = json.load(open(U.ANT_REF))["anticipation"]
    for sid in U.list_sessions():
        if use_official and sid in ref:
            grid = [(k, v["t_b"], v["t_e"]) for k, v in ref[sid].items()]
        else:
            grid = U.make_grid(sid, stride=stride, require_events=require_events)
        gt[sid] = U.build_gt(sid, grid)
    return gt


def flatten(gt, sids):
    out = {}
    for sid in sids:
        for seg_id, seg in gt[sid].items():
            out[(sid, seg_id)] = seg
    return out


def folds_loso(sids):
    return [([s for s in sids if s != h], [h]) for h in sids]


def evaluate(predictor_factory, gt=None, folds=None, subtasks=sdl.SUBTASKS, verbose=False):
    if gt is None:
        gt = build_all_gt()
    sids = list(gt.keys())
    if folds is None:
        folds = folds_loso(sids)
    all_pred = {}
    per_fold = []
    for train, held in folds:
        pred = predictor_factory(train, gt)
        held_pred = {}
        for sid in held:
            for seg_id, seg in gt[sid].items():
                held_pred[(sid, seg_id)] = pred.predict(sid, seg)
        all_pred.update(held_pred)
        gt_held = flatten(gt, held)
        r = sdl.score_dataset(gt_held, held_pred, subtasks)
        per_fold.append(r)
        if verbose:
            print("  held %s: mean4=%.4f (n=%d)" % (",".join(held), r["mean4"], r["n_seg"]))
    gt_flat = flatten(gt, sids)
    res = sdl.score_dataset(gt_flat, all_pred, subtasks)
    # fold-to-fold std of mean4 = noise floor proxy
    res["mean4_foldstd"] = statistics.pstdev([f["mean4"] for f in per_fold]) if len(per_fold) > 1 else 0.0
    res["per_fold"] = per_fold
    return res


def fmt(res, subtasks=sdl.SUBTASKS):
    cols = " ".join("%s=%.4f" % (s, res[s]) for s in subtasks)
    return "%s | mean4=%.4f (foldstd=%.4f) n=%d" % (cols, res["mean4"], res.get("mean4_foldstd", 0), res["n_seg"])


def cols(res):
    return "%6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %5d" % (
        res["next"], res["verbal"], res["nonverbal"], res["full"],
        res["mean4"], res.get("mean4_foldstd", 0.0), res["n_seg"])


HEADER = "%-46s %6s %6s %6s %6s %6s %6s %5s" % (
    "", "next", "verbal", "nonvb", "full", "mean4", "fstd", "nseg")


def main_final():
    """Reproduce the reported validation result of the submitted predictor."""
    import predictors as P
    gt = build_all_gt()
    ref = json.load(open(U.ANT_REF))["anticipation"]
    n_off = sum(len(gt[s]) for s in gt if s in ref)
    tot = sum(len(v) for v in gt.values())
    print("PROTOCOL: leave-one-session-out over %d annotated sessions" % len(gt))
    print("  segments: %d (%d from the 2 official starting-kit grids, %d generated"
          " stride-2s event-containing windows)" % (tot, n_off, tot - n_off))
    print("  participant cells: %d;  scorer: sdl.py (4 subtasks, best-of-K=5)\n" % (2 * tot))
    print(HEADER)
    for name, fac in [("empty prediction (floor)", P.empty_factory),
                      ("B2 = SUBMITTED predictor (825331)", P.b2_factory),
                      ("B4 = non-verbal-heavy variant (837289)", P.b4_factory)]:
        print("%-46s %s" % (name, cols(evaluate(fac, gt))))


def main_protocols():
    """Sensitivity of the reported result to the grid construction choices."""
    import predictors as P
    print("Grid sensitivity (LOSO, same folds; only the segment grid changes)\n")
    print(HEADER)
    for label, kw in [
        ("REPORTED: official(2) + stride2 event-containing", dict(stride=2.0, require_events=True)),
        ("stride 3 s event-containing", dict(stride=3.0, require_events=True)),
        ("stride 4 s event-containing (official density)", dict(stride=4.0, require_events=True)),
        ("stride 2 s, ALL windows (no event filter)", dict(stride=2.0, require_events=False)),
        ("stride 4 s, ALL windows (no event filter)", dict(stride=4.0, require_events=False)),
    ]:
        gt = build_all_gt(**kw)
        for nm, fac in (("B2", P.b2_factory), ("B4", P.b4_factory)):
            print("%-46s %s" % ("%s [%s]" % (label, nm), cols(evaluate(fac, gt))))


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "final"
    if mode == "protocols":
        main_protocols()
    else:
        main_final()
