"""Full reproduction report for the submitted hedge-pack (fact-sheet numbers).

Regenerates, from the development annotations alone, every quantity reported in the
fact sheet: grid statistics, candidate-pool anatomy, the five submitted alternatives and
their per-subtask roles, anchored-LOSO scores, the pool hyper-parameter sweep, the exact
organizer development-grid results, the hill-climb refinement check, and the verbal
oracle analysis.

Usage:  python run_report.py [--out report.json] [--quick]
        --quick skips the LOSO sweeps (keeps the single-fit sections).
Runtime: ~15 min single-core CPU (~30 s with --quick). Deterministic, no RNG.
"""
import argparse
import json
import time

from udiva import io as IO, cv as CV, metric as M
from udiva.baselines import (EmptyPredictor, GreedyK5Predictor, iter_cells, rich_pool,
                             seq_key)

DEV_SESSIONS = ("001080", "181182")          # the 2 sessions of the organizer dev grid
SHIPPED = dict(n_v=12, n_nv=10, mix=5)       # pool config of the submitted pack
MAX_CELLS = 2500

R = {}


def pf(n_v, n_nv, mix, tags=None):
    return lambda d: rich_pool(d, n_v=n_v, n_nv=n_nv, mix=mix, tags=tags)


def show(name, r):
    print(f"  {name:34s} " + " ".join(f"{k}={r[k]:.4f}" for k in M.SUBTASKS)
          + f" MEAN={r['mean']:.4f}", flush=True)
    return {k: round(r[k], 4) for k in list(M.SUBTASKS) + ["mean"]}


def dev_grid_score(predictor, ref):
    return M.score_dataset(ref, {s: CV.predict_session(predictor, s, ref[s]) for s in ref})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report.json")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    sessions = IO.list_sessions()
    ds = IO.build_cv_dataset_anchored(sessions, delta=0.25, min_gap=2.0)
    ref = IO.load_dev_reference()
    nondev = [s for s in sessions if s not in DEV_SESSIONS]
    ds_nondev = {s: ds[s] for s in nondev}
    cells = list(iter_cells(ds))

    # ---------------- 1. validation grid ----------------
    print("=== 1. anchored validation grid ===", flush=True)
    lens = [len(c) for c in cells]
    R["grid"] = {
        "protocol": "event-anchored, t_b = onset - 0.25 s, min 2.0 s apart, horizon 2.0 s",
        "n_sessions": len(sessions), "n_segments": sum(len(v) for v in ds.values()),
        "n_cells": len(cells), "empty_cell_frac": round(sum(l == 0 for l in lens) / len(lens), 4),
        "mean_cell_len": round(sum(lens) / len(lens), 3),
        "median_cell_len": sorted(lens)[len(lens) // 2],
        "dev_grid_n_segments": sum(len(v) for v in ref.values()),
        "dev_grid_n_cells": 2 * sum(len(v) for v in ref.values()),
    }
    print("  " + json.dumps(R["grid"]), flush=True)

    # ---------------- 2. candidate pool anatomy ----------------
    print("=== 2. candidate pool (fit on all 21 sessions) ===", flush=True)
    tags = []
    pool = rich_pool(ds, tags=tags, **SHIPPED)
    from collections import Counter
    fam = Counter(tags)
    ln = Counter(len(s) for s in pool)
    n_verbal_ev = sum(1 for s in pool for e in s if len(e) == 2)
    R["pool"] = {
        "config": SHIPPED, "n_candidates_total": len(pool),
        "by_family": dict(fam), "by_length": {str(k): v for k, v in sorted(ln.items())},
        "max_length": max(ln), "frac_len_le_2": round(sum(v for k, v in ln.items() if k <= 2) / len(pool), 4),
        "modal_verbal_target": pool[1][0][1] if len(pool) > 1 else None,
        "n_events_verbal": n_verbal_ev,
    }
    # the retained modes themselves
    R["pool"]["verbal_modes"] = [s[0] for s, t in zip(pool, tags) if t == "verbal_singleton"]
    R["pool"]["nonverbal_modes"] = [s[0] for s, t in zip(pool, tags) if t == "nonverbal_singleton"]
    print("  " + json.dumps({k: R["pool"][k] for k in
                             ("n_candidates_total", "by_family", "by_length", "max_length")}), flush=True)

    # ---------------- 3. the submitted pack + per-alternative roles ----------------
    print("=== 3. submitted pack (greedy, fit on all 21 sessions) ===", flush=True)
    g = GreedyK5Predictor(max_cells=MAX_CELLS, pool_fn=pf(**SHIPPED))
    g.fit(ds)
    alts = g.alts
    R["pack"] = {"alts": alts}
    try:
        shipped = json.load(open("heldbest_hedgepack.json"))["alts"]
        R["pack"]["matches_submitted_json"] = ([seq_key(a) for a in alts]
                                               == [seq_key(a) for a in shipped])
    except FileNotFoundError:
        R["pack"]["matches_submitted_json"] = None
    print("  reproduces submitted heldbest_hedgepack.json:",
          R["pack"]["matches_submitted_json"], flush=True)

    # role of each alternative: solo score, and the loss if it is dropped from the pack
    full = {st: 0.0 for st in M.SUBTASKS}
    solo = [{st: 0.0 for st in M.SUBTASKS} for _ in alts]
    drop = [{st: 0.0 for st in M.SUBTASKS} for _ in alts]
    for gt in cells:
        for st in M.SUBTASKS:
            sc = [M.best_of_k(gt, [a], st) for a in alts]
            full[st] += max(sc)
            for i in range(len(alts)):
                solo[i][st] += sc[i]
                rest = [sc[j] for j in range(len(alts)) if j != i]
                drop[i][st] += max(rest) if rest else 0.0
    n = len(cells)
    R["pack"]["per_alt"] = [
        {"index": i, "events": alts[i],
         "solo": {st: round(solo[i][st] / n, 4) for st in M.SUBTASKS},
         "loss_if_dropped": {st: round((full[st] - drop[i][st]) / n, 4) for st in M.SUBTASKS}}
        for i in range(len(alts))]
    for e in R["pack"]["per_alt"]:
        print(f"  alt{e['index']} solo=" + ",".join(f"{k}:{e['solo'][k]:.3f}" for k in M.SUBTASKS)
              + "  drop-loss=" + ",".join(f"{k}:{e['loss_if_dropped'][k]:+.3f}" for k in M.SUBTASKS),
              flush=True)

    # ---------------- 4. exact organizer development grid ----------------
    print("=== 4. organizer development grid (127 segments, 254 cells) ===", flush=True)
    g19 = GreedyK5Predictor(max_cells=MAX_CELLS, pool_fn=pf(**SHIPPED)); g19.fit(ds_nondev)
    R["dev_grid"] = {
        "held_out_fit19": show("fit on 19 non-dev -> dev grid", dev_grid_score(g19, ref)),
        "submitted_pack_fit21": show("submitted pack -> dev grid", dev_grid_score(g, ref)),
    }
    best_flat, best_r = None, None
    for a in alts:
        rr = M.score_dataset(ref, {s: {k: {"t_b": v["t_b"], "t_e": v["t_e"],
                                           "participants": {p: {"events": a} for p in v["participants"]}}
                                       for k, v in ref[s].items()} for s in ref})
        if best_r is None or rr["mean"] > best_r["mean"]:
            best_flat, best_r = a, rr
    R["dev_grid"]["flat_K1_best_single"] = show("flat K=1 (best single alt)", best_r)
    R["dev_grid"]["flat_K1_sequence"] = best_flat

    # ---------------- 5. hill-climb refinement ----------------
    print("=== 5. hill-climb swap refinement (local-optimality check) ===", flush=True)
    gr = GreedyK5Predictor(max_cells=MAX_CELLS, pool_fn=pf(**SHIPPED), refine=True)
    gr.fit(ds)
    same = [seq_key(a) for a in gr.alts] == [seq_key(a) for a in alts]
    R["refine"] = {"fit21_pack_unchanged": same, "fit21_refined_alts": gr.alts}
    print(f"  refine=True on all 21 sessions -> pack unchanged: {same}", flush=True)

    # ---------------- 6. verbal oracle analysis ----------------
    print("=== 6. verbal oracle (headroom of knowing the utterance types) ===", flush=True)
    vtgt = Counter()
    for evs in iter_cells(ds):
        for e in evs:
            if len(e) == 2:
                vtgt[tuple(e[1]) if isinstance(e[1], list) else e[1]] += 1
    mvt = vtgt.most_common(1)[0][0]
    mvt = list(mvt) if isinstance(mvt, tuple) else mvt
    o_type = o_pack = 0.0
    for gt in cells:
        oracle = [[e[0], mvt] for e in gt if len(e) == 2]      # true types+count+order, modal target
        o_type += M.best_of_k(gt, [oracle], "verbal")
        o_pack += M.best_of_k(gt, alts, "verbal")
    R["oracle"] = {"verbal_oracle_types_modal_target": round(o_type / n, 4),
                   "verbal_submitted_pack": round(o_pack / n, 4),
                   "modal_verbal_target": mvt, "n_cells": n}
    print("  " + json.dumps(R["oracle"]), flush=True)

    # ---------------- 7. LOSO (primary protocol) + hyper-parameter sweep ----------------
    if not args.quick:
        print("=== 7. anchored leave-one-session-out CV (21 folds) ===", flush=True)
        R["loso"] = {}
        for name, cfg in [("nv8_nnv8_mix4", dict(n_v=8, n_nv=8, mix=4)),
                          ("nv12_nnv10_mix5 (SUBMITTED)", SHIPPED),
                          ("nv16_nnv12_mix6", dict(n_v=16, n_nv=12, mix=6))]:
            R["loso"][name] = show("LOSO " + name, CV.leave_session_out(
                lambda c=cfg: GreedyK5Predictor(max_cells=MAX_CELLS, pool_fn=pf(**c)), ds))
        R["loso"]["observed_pool_only (no rich pool)"] = show(
            "LOSO observed-pool only", CV.leave_session_out(
                lambda: GreedyK5Predictor(max_cells=MAX_CELLS), ds))
        R["loso"]["nv12_nnv10_mix5 + refine"] = show(
            "LOSO submitted + refine", CV.leave_session_out(
                lambda: GreedyK5Predictor(max_cells=MAX_CELLS, pool_fn=pf(**SHIPPED), refine=True), ds))
        R["loso"]["empty_predictor"] = show("LOSO always-empty", CV.leave_session_out(
            EmptyPredictor, ds))

    R["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(R, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out} ({R['runtime_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
