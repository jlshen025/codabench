"""Load dense DINOv2 features and pool the observed window [t_b-W, t_b] per cell.

E1 features -> participant_a, E2 -> participant_b (each participant's own ego view).
"""
import os
import numpy as np

# Caches written by experiments/extract_features.py and experiments/embed_transcripts.py.
FEAT_DIR = os.environ.get("UDIVA_FEAT_DIR", "features")
VIEW = {"participant_a": "e1", "participant_b": "e2"}
_CACHE = {}


def have_features(sessions):
    return all(os.path.exists(os.path.join(FEAT_DIR, f"{s}.npz")) for s in sessions)


def load_session(sess):
    if sess not in _CACHE:
        d = np.load(os.path.join(FEAT_DIR, f"{sess}.npz"))
        _CACHE[sess] = {
            "e1": (d["feats_e1"].astype(np.float32), d["times_e1"].astype(np.float32)),
            "e2": (d["feats_e2"].astype(np.float32), d["times_e2"].astype(np.float32)),
        }
    return _CACHE[sess]


TEXT_DIR = os.environ.get("UDIVA_TEXT_DIR", "text_emb")
SPK_IDX = {"participant_a": 0, "participant_b": 1}
_TXT = {}


def load_text(sess):
    if sess not in _TXT:
        d = np.load(os.path.join(TEXT_DIR, f"{sess}.npz"))
        _TXT[sess] = (d["emb"].astype(np.float32), d["starts"].astype(np.float32), d["spk"])
    return _TXT[sess]


def text_window_feat(sess, participant, t_b, W=4.0):
    """Concat of [mean self recent, mean other recent, last recent utterance] bge-m3 emb."""
    emb, starts, spk = load_text(sess)
    dim = emb.shape[1] if emb.size else 1024
    rec = (starts > t_b - W) & (starts < t_b)
    me = rec & (spk == SPK_IDX[participant])
    other = rec & (spk != SPK_IDX[participant]) & (spk != 2)

    def pool(m):
        return emb[m].mean(0) if m.any() else np.zeros(dim, np.float32)
    last = emb[rec][-1] if rec.any() else np.zeros(dim, np.float32)
    return np.concatenate([pool(me), pool(other), last]).astype(np.float32)


def dialogue_feat(sess, participant, t_b, W=6.0):
    """Turn-taking / timing / adjacency-pair structural features (count-based, robust on
    small data) for predicting the participant's next verbal event."""
    from . import io as IO
    utts = IO.load_transcript(sess)
    other = "participant_b" if participant == "participant_a" else "participant_a"
    obs = [u for u in utts if u[0] < t_b]
    rec = [u for u in obs if u[0] > t_b - W]
    su = [u for u in rec if u[2] == participant]
    ou = [u for u in rec if u[2] == other]
    last = max(obs, key=lambda u: u[0]) if obs else None
    qwords = ("qué", "cómo", "dónde", "cuándo", "por qué", "quién", "cuál", "cuánto", "verdad")

    def isq(u):
        t = u[3].lower()
        return 1.0 if ("?" in u[3] or "¿" in u[3] or any(t.startswith(w) for w in qwords)) else 0.0
    f = [
        float(len(su)), float(len(ou)),
        t_b - max([u[0] for u in su], default=t_b - W),
        t_b - max([u[0] for u in ou], default=t_b - W),
        1.0 if (su and ou and su[-1][0] > ou[-1][0]) else (0.0 if ou else 1.0),
        1.0 if any(u[0] < t_b < u[1] for u in su) else 0.0,
        1.0 if any(u[0] < t_b < u[1] for u in ou) else 0.0,
        isq(last) if last else 0.0,
        1.0 if (last and last[2] == participant) else 0.0,
        1.0 if (last and last[2] == other) else 0.0,
        float(len(last[3].split())) if last else 0.0,
        (t_b - last[1]) if last else W,
        sum(u[1] - u[0] for u in su), sum(u[1] - u[0] for u in ou),
    ]
    return np.array(f, dtype=np.float32)


def window_feat(sess, participant, t_b, W=1.5, pool="mean_last"):
    feats, times = load_session(sess)[VIEW[participant]]
    mask = (times > t_b - W) & (times <= t_b)
    if mask.any():
        sub = feats[mask]
        mean = sub.mean(0)
        last = sub[-1]
    else:
        idx = int(np.searchsorted(times, t_b)) - 1
        idx = max(0, min(idx, len(feats) - 1))
        mean = last = feats[idx]
    if pool == "mean":
        v = mean
    elif pool == "last":
        v = last
    else:  # mean_last
        v = np.concatenate([mean, last])
    n = np.linalg.norm(v) + 1e-8
    return (v / n).astype(np.float32)
