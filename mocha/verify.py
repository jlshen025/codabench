"""End-to-end check that this repository reproduces the submitted entry.

Assembles the runnable tree (code/ + weights/ + assets/) into a temporary directory exactly as it
was packaged for the evaluation server, imports that tree's own `run.py`, and calls
`predict(data)` the way the server calls it: one nested dict `data[subject_id][walk_id]` holding
the whole set at once. The probe is built from the RELEASED CARE-PD cohorts, so no hidden data is
needed, and no label is ever read.

Asset checksums are checked first, so a wrong encoder or SMPL file is reported as such instead of
being blamed on the model.

    python verify.py --data /path/to/CARE-PD/Canonicalized_SMPL_pickles

Needs only torch and numpy. Uses a GPU when one is visible; CPU works and is slower.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COHORTS = ["3DGait", "PD-GaM", "BMCLab", "T-SDU-PD"]

# md5 of every binary the entry loads. The head is ours and lives in this repository; the other
# three are third-party and must be fetched — see assets/README.md.
MD5 = {
    "weights/head_jm_zscore_focal.npz":     "e47fb3f12f29e3569507fa546d34d426",
    "assets/motionagformer-s-h36m.pth.tr":  "2c0d06b6a60610f080661c46c6f5f231",
    "assets/J_regressor_h36m_correct.npy":  "57176924f79106f64dcafd64d4a1df6d",
    "assets/smpl_neutral_clean.npz":        "44a55a168cd56fbeaa1d28e9f2ee2389",
}


def check_assets():
    ok = True
    for rel, want in MD5.items():
        path = os.path.join(HERE, rel)
        if not os.path.isfile(path):
            print(f"[verify] MISSING {rel} — see assets/README.md")
            ok = False
            continue
        got = hashlib.md5(open(path, "rb").read()).hexdigest()
        if got != want:
            print(f"[verify] MD5 MISMATCH {rel}: expected {want}, got {got}")
            ok = False
    if ok:
        print(f"[verify] all {len(MD5)} binaries present and matching their checksums")
    return ok


def assemble(workdir):
    """Flatten code/ + weights/ + assets/ the way the submitted archive was laid out."""
    shutil.copytree(os.path.join(HERE, "code"), workdir, dirs_exist_ok=True)
    for rel in MD5:
        shutil.copy(os.path.join(HERE, rel), os.path.join(workdir, os.path.basename(rel)))


def build_probe(data_dir, target_subjects=35, max_walks=11):
    """A subject-grouped probe in the hidden test's exact input format, labels dropped."""
    probe = {}
    quota = max(1, target_subjects // len(COHORTS))
    for cohort in COHORTS:
        path = os.path.join(data_dir, f"{cohort}_canonical.pkl")
        if not os.path.isfile(path):
            print(f"[verify] missing {path} — skipping {cohort}")
            continue
        with open(path, "rb") as f:
            raw = pickle.load(f)
        taken = 0
        for sid, walks in raw.items():
            if taken >= quota or not isinstance(walks, dict):
                continue
            entry = {str(wid): dict(pose=np.asarray(w["pose"], np.float32),
                                    trans=np.asarray(w["trans"], np.float32),
                                    beta=np.zeros((1, 10), np.float32),
                                    fps=int(w["fps"]))
                     for wid, w in list(walks.items())[:max_walks]}
            if entry:
                probe[str(sid)] = entry
                taken += 1
    return probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="", help="CARE-PD Canonicalized_SMPL_pickles directory")
    ap.add_argument("--subjects", type=int, default=35)
    args = ap.parse_args()

    if not check_assets():
        sys.exit(1)
    if not args.data:
        print("[verify] no --data given: checksum-only mode, skipping the run.")
        return

    probe = build_probe(args.data, target_subjects=args.subjects)
    n_walks = sum(len(w) for w in probe.values())
    if not n_walks:
        print("[verify] FAILED: probe is empty — check --data")
        sys.exit(1)
    print(f"[verify] probe: {len(probe)} subjects / {n_walks} walks (labels withheld)")

    workdir = tempfile.mkdtemp(prefix="mocha_verify_")
    try:
        assemble(workdir)
        print(f"[verify] config: {json.load(open(os.path.join(workdir, 'config.json')))}")
        sys.path.insert(0, workdir)
        spec = importlib.util.spec_from_file_location("mocha_run", os.path.join(workdir, "run.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["mocha_run"] = mod
        t0 = time.time()
        spec.loader.exec_module(mod)
        out = mod.predict(probe)
        elapsed = time.time() - t0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    # The evaluation harness requires one in-range label for every walk it handed in.
    assert set(out) == set(probe), "predict() dropped or invented subjects"
    for sid in probe:
        assert set(out[sid]) == set(probe[sid]), f"predict() dropped walks for subject {sid}"
        for wid, lab in out[sid].items():
            assert int(lab) in (0, 1, 2, 3), f"label out of range for {sid}/{wid}: {lab}"

    flat = np.array([int(out[s][w]) for s in sorted(out) for w in sorted(out[s])])
    dist = np.bincount(flat, minlength=4).tolist()
    print(f"[verify] predict() returned {len(flat)} labels in {elapsed:.1f}s "
          f"({elapsed / n_walks * 372:.0f}s projected for the 372-walk test; the limit is 3600s)")
    print(f"[verify] predicted class distribution [0,1,2,3] = {dist}")
    print("[verify] PASS")


if __name__ == "__main__":
    main()
