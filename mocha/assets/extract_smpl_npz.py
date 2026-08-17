"""Convert SMPL_NEUTRAL.pkl to a chumpy-free .npz usable at train and eval time.

The original SMPL pickle stores several arrays as chumpy objects, and chumpy does not build
against numpy 2.x. This reads the pickle with a stand-in class for anything chumpy, digs the
underlying ndarray out of each, and saves the six arrays the forward kinematics needs.

The SMPL model itself is licensed by the Max Planck Institute and is NOT redistributed with this
repository — see README.md in this directory for how to obtain SMPL_NEUTRAL.pkl.

Usage:
    python extract_smpl_npz.py --pkl /path/to/SMPL_NEUTRAL.pkl --out smpl_neutral_clean.npz
"""
import argparse
import hashlib
import pickle

import numpy as np
import scipy.sparse

NEEDED = ["v_template", "shapedirs", "posedirs", "J_regressor", "kintree_table", "weights", "f"]


class _Passthrough:
    """Stand-in for any chumpy class; captures the pickled state so the ndarray can be recovered."""

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"_state": state})

    def __setitem__(self, k, v):  # some chumpy reduce paths call this
        pass


class _FakeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if "chumpy" in module:
            return _Passthrough
        return super().find_class(module, name)


def to_array(v):
    if isinstance(v, np.ndarray):
        return v
    if scipy.sparse.issparse(v):
        return np.asarray(v.todense())
    if isinstance(v, _Passthrough):
        d = v.__dict__
        for key in ("x", "_data", "data"):
            if key in d and isinstance(d[key], np.ndarray):
                return d[key]
        cands = [(a.size, a) for a in d.values() if isinstance(a, np.ndarray)]
        if cands:
            return max(cands, key=lambda t: t[0])[1]
        raise ValueError(f"no ndarray inside chumpy object, keys={list(d.keys())}")
    return np.asarray(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="path to SMPL_NEUTRAL.pkl")
    ap.add_argument("--out", default="smpl_neutral_clean.npz")
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        d = _FakeUnpickler(f, encoding="latin1").load()

    clean = {}
    for k in NEEDED:
        if k not in d:
            print(f"  MISSING {k}")
            continue
        clean[k] = np.asarray(to_array(d[k]))
        print(f"  {k}: {clean[k].shape} {clean[k].dtype}")

    njoints = clean["posedirs"].shape[2] // 9 if clean["posedirs"].ndim == 3 else None
    assert njoints == 23, f"expected 23 SMPL body joints in posedirs, got {njoints}"

    np.savez(args.out, **clean)
    print(f"saved {args.out}  md5={hashlib.md5(open(args.out,'rb').read()).hexdigest()}")
    print("expected md5 = 44a55a168cd56fbeaa1d28e9f2ee2389")


if __name__ == "__main__":
    main()
