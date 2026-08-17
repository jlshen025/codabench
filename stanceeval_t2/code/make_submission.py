"""
make_submission.py — Build a Codabench predictions file from saved OOF/test probabilities.

For dev-phase CALIBRATION: ensembles the saved dev OOF probabilities (held-out preds on
dev.csv) of the given source runs, argmax -> labels in dev.csv ORDER, and writes BOTH
candidate formats (the competition-page .txt one-per-line, and the offline-script id,stance
CSV) each zipped FLAT. Verifies row-alignment vs dev.csv gold and prints local Favg2.

Usage:
  python make_submission.py --cv dev --sources mtl_araberttw mtl_marbertv2 --name track1_dev
"""
import os, sys, glob, json, zipfile, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SUB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submission")


def ens_dev(sources, cv):
    P, y = [], None
    for pfx in sources:
        files = sorted(glob.glob(os.path.join(ROOT, f"{pfx}_{cv}_s*", "oof_proba.npz")))
        assert files, f"no runs for {pfx}_{cv}"
        sp = [np.load(f, allow_pickle=True) for f in files]
        for d in sp:
            assert y is None or np.array_equal(y, d["y_true"]), "order mismatch"
            y = d["y_true"]
        P.append(np.mean([d["proba"] for d in sp], 0))
    return np.mean(P, 0), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", default="dev")
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    os.makedirs(SUB, exist_ok=True)

    dev = S.load_labeled(S.DEV_CSV)  # dev.csv order, with IDs in 'ID'
    proba, y = ens_dev(args.sources, args.cv)
    assert len(proba) == len(dev), f"len mismatch {len(proba)} vs {len(dev)}"
    assert np.array_equal(y, dev["label"].to_numpy()), "OOF y_true != dev.csv order — alignment broken!"
    pred_ids = proba.argmax(1)
    pred_lbl = [S.ID2LABEL[i] for i in pred_ids]
    m = S.compute_metrics(dev["label"].to_numpy(), pred_ids)
    print(f"local dev Favg2={m['Favg2']*100:.2f} Favg3={m['Favg3']*100:.2f}  "
          f"pred dist={ {S.ID2LABEL[k]: int((pred_ids==k).sum()) for k in range(3)} }")

    # Format A: one-label-per-line .txt (competition page), dev.csv order
    txt = os.path.join(SUB, "predictions.txt")
    open(txt, "w").write("\n".join(pred_lbl) + "\n")
    zipA = os.path.join(SUB, f"{args.name}_txt.zip")
    with zipfile.ZipFile(zipA, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(txt, "predictions.txt")

    # Format B: id,stance CSV (offline eval script), zipped flat
    ids = dev["ID"].astype(str).tolist()
    csv = os.path.join(SUB, "submission.csv")
    with open(csv, "w") as f:
        f.write("id,stance\n")
        for i, l in zip(ids, pred_lbl):
            f.write(f"{i},{l}\n")
    zipB = os.path.join(SUB, f"{args.name}_csv.zip")
    with zipfile.ZipFile(zipB, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv, "submission.csv")

    print(f"wrote:\n  A (txt): {zipA}\n  B (csv): {zipB}")


if __name__ == "__main__":
    main()
