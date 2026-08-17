"""Load CARE-PD canonical pickles into flat arrays for the UPDRS gait task."""
import os
import pickle
import numpy as np

# Point CAREPD_DIR at the CARE-PD Canonicalized_SMPL_pickles directory.
DATA_DIR = os.environ.get("CAREPD_DIR", "./CARE-PD/Canonicalized_SMPL_pickles")
UPDRS_COHORTS = ['3DGait', 'PD-GaM', 'BMCLab', 'T-SDU-PD']  # only these carry UPDRS_GAIT
ALL_COHORTS = ['3DGait', 'PD-GaM', 'BMCLab', 'T-SDU-PD', 'DNE', 'E-LC', 'KUL-DT-T', 'T-LTC', 'T-SDU']


def load_cohort(name, data_dir=DATA_DIR, labeled_only=True):
    """Yield dicts: {cohort, subject, walk, pose, trans, fps, label}."""
    with open(os.path.join(data_dir, f"{name}_canonical.pkl"), "rb") as f:
        d = pickle.load(f)
    for sid, walks in d.items():
        if not isinstance(walks, dict):
            continue
        for wid, e in walks.items():
            lab = e.get('UPDRS_GAIT', None)
            if labeled_only and lab is None:
                continue
            yield {
                'cohort': name,
                'subject': f"{name}/{sid}",
                'walk': wid,
                'pose': np.asarray(e['pose'], dtype=np.float32),
                'trans': np.asarray(e['trans'], dtype=np.float32),
                'fps': e.get('fps', 30),
                'label': (int(lab) if lab is not None else -1),
            }


def load_features(cohorts=UPDRS_COHORTS, cache=None, data_dir=DATA_DIR, verbose=True):
    """Return dict with X(N,F) y(N) cohort(N) subject(N) walk(N) names.
    Caches to `cache` .npz (features only; cheap to recompute)."""
    from gait_features import extract_features, feature_names
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if verbose:
            print(f"[dataio] loaded cached features {cache}: X{z['X'].shape}")
        return {k: z[k] for k in z.files}
    X, y, coh, subj, walk = [], [], [], [], []
    for name in cohorts:
        n0 = len(y)
        for s in load_cohort(name, data_dir=data_dir):
            vec, _ = extract_features(s['pose'], s['trans'], s['fps'])
            X.append(vec); y.append(s['label']); coh.append(s['cohort'])
            subj.append(s['subject']); walk.append(s['walk'])
        if verbose:
            print(f"[dataio] {name}: {len(y) - n0} walks")
    out = {
        'X': np.asarray(X, dtype=np.float64),
        'y': np.asarray(y, dtype=np.int64),
        'cohort': np.asarray(coh),
        'subject': np.asarray(subj),
        'walk': np.asarray(walk),
        'names': np.asarray(feature_names()),
    }
    if cache:
        os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
        np.savez_compressed(cache, **out)
        if verbose:
            print(f"[dataio] cached -> {cache}")
    return out
