"""Build the submission zip: this script produced the ranked entry (submission 829988).

It runs the two channels of Fig. 1 of the fact sheet on the evaluation sessions and packs the
result as a reference.json-shaped JSON:

  non-verbal   for each frozen feature directory, (subject, high-level action) logistic heads
               are fitted on all 21 annotated sessions and applied to the evaluation segments;
               the posteriors of the three frozen backbones and of the fine-tuned model
               (--ft_oof, produced by ft_infer.py) are fused with equal weights, and only then
               expanded into complete tuples with training co-occurrence priors.
  verbal       a cue-level TF-IDF model (utterance-type heads + target heads) fitted on the
               same 21 sessions and applied to the evaluation transcripts.

The two channels are independent and are written to the two separate top-level keys of one JSON.
The segment grid comes from the organisers' template when one is available (--grid_json),
otherwise from the segment keys of the feature files (a fixed 2 s duration grid).

Exact command used for the ranked entry: see the README.
"""
import os, sys, glob, json, zipfile, argparse
import numpy as np
from udiva import data as D
from udiva.models_text import VerbalCueModel
from udiva.models_video import (compose_nonverbal, fit_pair_heads, fuse_ft_probs,
                                fuse_pair_probs, predict_pair_probs)
from udiva.metric import VKEYS, NKEYS

DEV_FEAT = "<scratch>/t2/feats/videomae_large"


def grid_from_segkeys(segkeys, seg_len=2.0):
    """Fixed 2 s grid derived from the feature files' segment keys (s_0001 -> [0, 2) ...)."""
    g = {}
    for sk in segkeys:
        k = int(sk.split("_")[1]) - 1
        g[sk] = {"t_b": seg_len * k, "t_e": seg_len * (k + 1)}
    return g


def build_json(verbal_pred, nonverbal_pred, sids, grids):
    """Mirror of starting_kit/recognition/reference.json, with a confidence on every event.

    The confidence is written under both 'score' and 'confidence' because the key expected by
    the scorer was not documented; the two values are always identical.
    """
    out = {"verbal": {}, "nonverbal": {}}
    for stream, keys, pred in (("verbal", VKEYS, verbal_pred),
                               ("nonverbal", NKEYS, nonverbal_pred)):
        for sid in sids:
            out[stream][sid] = {}
            for sk, gi in grids[sid].items():
                evs = []
                for e in pred.get(stream, {}).get(sid, {}).get(sk, {"events": []})["events"]:
                    sc = float(e.get("score", 1.0))
                    ev = {k: e[k] for k in keys}
                    ev["score"] = sc
                    ev["confidence"] = sc
                    evs.append(ev)
                out[stream][sid][sk] = {"t_b": gi["t_b"], "t_e": gi["t_e"], "events": evs}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", default=DEV_FEAT,
                    help="evaluation feature dir used for the fallback grid (and, without "
                         "--ens_feats, as the single non-verbal feature source)")
    ap.add_argument("--trans_dir", default=D.TRANS_DIR, help="evaluation transcripts dir")
    ap.add_argument("--sids", default="", help="comma list; default = all sids found in feat_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid_json", default="", help="organisers' template (reference.json-shaped): "
                    "use its EXACT sids + segment keys + t_b/t_e as the prediction grid. "
                    "Falls back to the feature-derived duration grid when empty.")
    ap.add_argument("--ens_feats", default="", help="non-verbal ensemble: 'dir:weight,dir:weight' "
                    "over per-backbone feature dirs. Empty = single --feat_dir.")
    ap.add_argument("--dev_feat_root", default="<scratch>/t2/feats",
                    help="root holding the per-backbone DEV feature dirs (heads are fitted there)")
    ap.add_argument("--test_feat_root", default="",
                    help="root holding the per-backbone EVAL feature dirs (default: dev root)")
    ap.add_argument("--ft_oof", default="", help="npz of fine-tuned (subject, h) probabilities "
                    "written by ft_infer.py, fused into the non-verbal ensemble")
    ap.add_argument("--ft_weight", type=float, default=1.0)
    ap.add_argument("--names", default="predictions.json,recognition.json,answer.json,reference.json",
                    help="file name(s) the identical JSON is written under inside the zip")
    args = ap.parse_args()

    train_sids = D.all_sids()

    # ---------------------------------------------------------------- prediction grid
    grids = {}
    if args.grid_json:
        tmpl = json.load(open(args.grid_json))
        sids_set = set()
        for st in tmpl.values():
            sids_set.update(st.keys())      # union over streams: never drop an expected session
        test_sids = sorted(sids_set)
        for sid in test_sids:
            g = {}
            for st in tmpl.values():
                for sk, gi in st.get(sid, {}).items():
                    g[sk] = {"t_b": float(gi["t_b"]), "t_e": float(gi["t_e"])}
            grids[sid] = g
        print(f"GRID from template {args.grid_json}: {len(test_sids)} sids, "
              f"{sum(len(g) for g in grids.values())} segments")
    else:
        if args.sids:
            test_sids = args.sids.split(",")
        else:
            test_sids = sorted(set(os.path.basename(f).rsplit("_", 1)[0]
                                   for f in glob.glob(f"{args.feat_dir}/*_E1.npz")))
        for sid in test_sids:
            d = np.load(f"{args.feat_dir}/{sid}_E1.npz", allow_pickle=True)
            grids[sid] = grid_from_segkeys(list(d["seg_keys"]))
    print(f"train={len(train_sids)} dev sessions; test={len(test_sids)} sessions: {test_sids[:5]}...")

    # ---------------------------------------------------------------- verbal channel
    vm = VerbalCueModel(ku=8, kt=12, pu_thresh=0.02, pt_thresh=0.02).fit(train_sids)
    vpred = vm.predict(test_sids, grid_by_sid=grids, trans_dir=args.trans_dir)

    # ---------------------------------------------------------------- non-verbal channel
    if args.ens_feats:
        dir_weights = [(p.split(":")[0], float(p.split(":")[1])) for p in args.ens_feats.split(",")]
        test_root = args.test_feat_root or args.dev_feat_root
        print(f"ENSEMBLE non-verbal: {dir_weights} "
              f"(dev_root={args.dev_feat_root} test_root={test_root})")
        pair_probs = fuse_pair_probs(train_sids, test_sids, args.dev_feat_root, test_root,
                                     dir_weights)
        if args.ft_oof:
            fuse_ft_probs(pair_probs, args.ft_oof, args.ft_weight)
    else:
        labels, clf = fit_pair_heads(train_sids, DEV_FEAT)
        pair_probs = predict_pair_probs(test_sids, args.feat_dir, labels, clf)
    npred = compose_nonverbal(train_sids, test_sids, pair_probs, kh=14, ph=0.02, grid_by_sid=grids)

    # ---------------------------------------------------------------- pack
    sub = build_json(vpred, npred, test_sids, grids)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = json.dumps(sub)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for nm in args.names.split(","):
            z.writestr(nm, payload)
    nev = sum(len(b["events"]) for st in sub.values() for s in st.values() for b in s.values())
    print(f"wrote {args.out} ({len(payload)/1e6:.1f}MB/copy) | {len(test_sids)} sessions, "
          f"{nev} total events")


if __name__ == "__main__":
    main()
