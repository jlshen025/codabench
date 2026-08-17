"""Rebuild the SUBMITTED hedge-pack from the development annotations, and write it to
`heldbest_hedgepack.json` -- the file `make_test_sub_fixed.py` packs into the submission.

Steps, in order:
  1. pool hyper-parameter selection: anchored-LOSO over the 21 development sessions for
     each candidate-pool configuration (this is how (n_v, n_nv, mix) was chosen);
  2. held-out check on the exact organizer development grid (fit on the 19 non-dev
     sessions, score the released 127-segment reference);
  3. per-fold consistency of the hedge-pack against the plain observed-sequence pool;
  4. FINAL FIT on all 21 sessions with the selected configuration -> the shipped 5-set,
     written to heldbest_hedgepack.json (and compared against any existing copy);
  5. hill-climb swap refinement of the final pack (local-optimality certificate).

Deterministic end to end (no RNG, no seeds): re-running reproduces the packaged
heldbest_hedgepack.json exactly, and hence the submitted anticipation.json byte for byte.

Usage: python run_hedge2.py [--out heldbest_hedgepack.json] [--config nv12_nnv10_mix5]
       --config pins the pool configuration and skips the selection sweep (~10 min faster).
Runtime: ~10 min single-core CPU for the full sweep.
"""
import argparse
import json
import os

from udiva import io as IO, cv as CV, metric as M
from udiva.baselines import GreedyK5Predictor, rich_pool, seq_key

CONFIGS = {"nv8_nnv8_mix4": dict(n_v=8, n_nv=8, mix=4),
           "nv12_nnv10_mix5": dict(n_v=12, n_nv=10, mix=5),      # selected -> submitted
           "nv16_nnv12_mix6": dict(n_v=16, n_nv=12, mix=6)}
DEV_SESSIONS = ("001080", "181182")
MAX_CELLS = 2500


def pf(cfg):
    return lambda d: rich_pool(d, **cfg)


def show(n, r):
    print(f"  {n:26s} " + " ".join(f"{k}={r[k]:.3f}" for k in M.SUBTASKS)
          + f" MEAN={r['mean']:.4f}", flush=True)
    return r["mean"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "heldbest_hedgepack.json"))
    ap.add_argument("--config", choices=sorted(CONFIGS), default=None)
    args = ap.parse_args()

    sessions = IO.list_sessions()
    ds = IO.build_cv_dataset_anchored(sessions, delta=0.25, min_gap=2.0)
    ref = IO.load_dev_reference()
    ds_nondev = {s: ds[s] for s in sessions if s not in DEV_SESSIONS}

    # ---- 1. pool hyper-parameter selection (anchored LOSO) ----
    if args.config:
        name, cv_loso = args.config, None
        print(f"=== pool config pinned to {name} (selection sweep skipped) ===", flush=True)
    else:
        print("=== 1. pool config selection: anchored LOSO over 21 sessions ===", flush=True)
        scores = {}
        for nm, cfg in CONFIGS.items():
            scores[nm] = show("LOSO " + nm, CV.leave_session_out(
                lambda c=cfg: GreedyK5Predictor(max_cells=MAX_CELLS, pool_fn=pf(c)), ds))
        # highest LOSO mean; ties (16/12/6 scores exactly as 12/10/5) go to the smaller pool
        order = list(CONFIGS)
        name = max(order, key=lambda nm: (round(scores[nm], 6), -order.index(nm)))
        cv_loso = scores[name]
        print(f"  selected: {name} (anchored-LOSO mean {cv_loso:.4f})", flush=True)
    cfg = CONFIGS[name]

    # ---- 2. held-out organizer development grid ----
    print("=== 2. organizer development grid (fit on the 19 non-dev sessions) ===", flush=True)
    g_obs = GreedyK5Predictor(max_cells=MAX_CELLS); g_obs.fit(ds_nondev)
    show("dev-org observed pool", M.score_dataset(
        ref, {s: CV.predict_session(g_obs, s, ref[s]) for s in ref}))
    g19 = GreedyK5Predictor(max_cells=MAX_CELLS, pool_fn=pf(cfg)); g19.fit(ds_nondev)
    cv_dev = show("dev-org hedge-pack", M.score_dataset(
        ref, {s: CV.predict_session(g19, s, ref[s]) for s in ref}))

    # ---- 3. per-fold consistency vs the plain observed-sequence pool ----
    print("=== 3. per-fold consistency (hedge-pack vs observed pool) ===", flush=True)
    wins, deltas = 0, []
    for held in sessions:
        tr = {s: ds[s] for s in sessions if s != held}
        gs = GreedyK5Predictor(max_cells=MAX_CELLS); gs.fit(tr)
        gh = GreedyK5Predictor(max_cells=MAX_CELLS, pool_fn=pf(cfg)); gh.fit(tr)
        rs = M.score_dataset({held: ds[held]}, {held: CV.predict_session(gs, held, ds[held])})["mean"]
        rh = M.score_dataset({held: ds[held]}, {held: CV.predict_session(gh, held, ds[held])})["mean"]
        deltas.append(rh - rs); wins += rh > rs
    print(f"  hedge-pack wins {wins}/{len(sessions)} folds; "
          f"mean d={sum(deltas)/len(deltas):+.4f} min={min(deltas):+.4f}", flush=True)

    # ---- 4. final fit on all 21 sessions -> the shipped pack ----
    print("=== 4. final fit on all 21 sessions -> hedge-pack ===", flush=True)
    g = GreedyK5Predictor(max_cells=MAX_CELLS, pool_fn=pf(cfg)); g.fit(ds)
    for i, a in enumerate(g.alts):
        print(f"  alt{i}: {json.dumps(a)}", flush=True)
    if cv_loso is None:
        cv_loso = CV.leave_session_out(
            lambda: GreedyK5Predictor(max_cells=MAX_CELLS, pool_fn=pf(cfg)), ds)["mean"]

    prev = None
    if os.path.exists(args.out):
        prev = json.load(open(args.out)).get("alts")
    payload = {"config": f"rich_pool nv{cfg['n_v']} nnv{cfg['n_nv']} mix{cfg['mix']} anchored",
               "cv_anchored_loso": round(cv_loso, 4),
               "cv_dev_organizer": round(cv_dev, 4),
               "alts": g.alts}
    json.dump(payload, open(args.out, "w"), indent=1)
    print(f"  wrote {args.out}", flush=True)
    if prev is not None:
        match = [seq_key(a) for a in prev] == [seq_key(a) for a in g.alts]
        print(f"  matches the previously packaged pack: {match}", flush=True)

    # ---- 5. local-optimality certificate ----
    print("=== 5. hill-climb swap refinement of the final pack ===", flush=True)
    gr = GreedyK5Predictor(max_cells=MAX_CELLS, pool_fn=pf(cfg), refine=True); gr.fit(ds)
    unchanged = [seq_key(a) for a in gr.alts] == [seq_key(a) for a in g.alts]
    print(f"  refinement left the pack unchanged: {unchanged} "
          f"(greedy solution is a local optimum of the selection objective)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
