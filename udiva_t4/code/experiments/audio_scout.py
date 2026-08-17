"""Scout the AUDIO modality — the one genuinely-NEW input (video dead, text saturated).
Does observed-window [t_b-W, t_b] EGO audio predict the future events better than the prior?

Four diagnostics (mirror transcript_diag.py / motion_scout.py: anchored grid, sessions[::4]
held split, top1/top5 vs prior):
  (1) verbal-active   (binary; AUC + acc vs majority prior)
  (2) nonverbal-active(binary; manipulation-sound -> action hypothesis)
  (3) first verbal TYPE     on verbal-active cells   (the high-headroom subtask; oracle types ~doubles verbal)
  (4) first nonverbal ACTION on nonverbal-active cells

Own ego view (A->E1, B->E2) + PARTNER view (turn-taking cue). Hand-crafted librosa
features (rms/zcr/spectral/mfcc/onset), StandardScaler+logreg. Audio decoded via ffmpeg
subprocess pipe (audioread absent). SCOUT_MAX_SESS>0 limits sessions for a quick smoke.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os, subprocess, json
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
import numpy as np
import librosa
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from udiva import io as IO

SR = 16000
W = float(os.environ.get("AUDIO_W", "2.0"))
EGO = os.path.join(IO.DATA_ROOT, "annotated_sessions", "audiovisual", "ego")
VIEW = {"participant_a": "E1", "participant_b": "E2"}
OTHER = {"participant_a": "participant_b", "participant_b": "participant_a"}
MAXS = int(os.environ.get("SCOUT_MAX_SESS", "0"))  # 0 = all
FEATDIM = 41  # per view (see win_feats)

_cache = {}


def load_audio(sess, view):
    key = (sess, view)
    if key not in _cache:
        path = os.path.join(EGO, view, f"{sess}.mp4")
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
            capture_output=True, check=True).stdout
        _cache[key] = np.frombuffer(out, np.float32)
    return _cache[key]


def win_feats(sess, view, t_b):
    try:
        y = load_audio(sess, view)
        a = max(0, int((t_b - W) * SR)); b = min(len(y), int(t_b * SR))
        seg = np.ascontiguousarray(y[a:b])
        if len(seg) < int(0.2 * SR):
            return np.zeros(FEATDIM, np.float32)
        rms = librosa.feature.rms(y=seg)[0]
        zcr = librosa.feature.zero_crossing_rate(seg)[0]
        cen = librosa.feature.spectral_centroid(y=seg, sr=SR)[0]
        bw = librosa.feature.spectral_bandwidth(y=seg, sr=SR)[0]
        rol = librosa.feature.spectral_rolloff(y=seg, sr=SR)[0]
        fla = librosa.feature.spectral_flatness(y=seg)[0]
        mfcc = librosa.feature.mfcc(y=seg, sr=SR, n_mfcc=13)
        onset = librosa.onset.onset_strength(y=seg, sr=SR)

        def ms(x):
            return [float(np.mean(x)), float(np.std(x))]
        feat = [float(np.mean(rms)), float(np.std(rms)), float(np.max(rms))]
        feat += ms(zcr) + ms(cen) + ms(bw) + ms(rol) + ms(fla)
        feat += list(mfcc.mean(1)) + list(mfcc.std(1))
        feat += [float(np.mean(onset)), float(np.max(onset))]
        v = np.array(feat, np.float32)
        v[~np.isfinite(v)] = 0.0
        assert v.shape[0] == FEATDIM, v.shape
        return v
    except Exception as e:
        print(f"  [warn] win_feats {sess}/{view}@{t_b}: {e}", flush=True)
        return np.zeros(FEATDIM, np.float32)


def cell_feat(sess, p, t_b):
    return np.concatenate([win_feats(sess, VIEW[p], t_b),
                           win_feats(sess, VIEW[OTHER[p]], t_b)])


def diag_top(Xtr, Xte, ytr, yte, k=5, C=0.5):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=600, C=C).fit(sc.transform(Xtr), ytr)
    cls = clf.classes_
    pb = clf.predict_proba(sc.transform(Xte))
    order = np.argsort(-pb, 1)[:, :k]
    top1 = float((cls[np.argmax(pb, 1)] == yte).mean())
    top5 = float(np.mean([yte[i] in cls[order[i]] for i in range(len(yte))]))
    pri_top = [l for l, _ in Counter(ytr).most_common(k)]
    pri1 = float((yte == Counter(ytr).most_common(1)[0][0]).mean())
    pri5 = float(np.isin(yte, pri_top).mean())
    return dict(n_classes=len(cls), top1=top1, top5=top5, prior_top1=pri1, prior_top5=pri5)


def diag_bin(Xtr, Xte, ytr, yte, C=0.5):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=600, C=C).fit(sc.transform(Xtr), ytr)
    pb = clf.predict_proba(sc.transform(Xte))[:, 1]
    auc = float(roc_auc_score(yte, pb)) if len(set(yte)) > 1 else float("nan")
    base = float(yte.mean())
    acc = float((clf.predict(sc.transform(Xte)) == yte).mean())
    return dict(auc=auc, base_rate=base, prior_acc=max(base, 1 - base), logreg_acc=acc)


def main():
    sessions = IO.list_sessions()
    if MAXS:
        sessions = sessions[:MAXS]
    ds = IO.build_cv_dataset_anchored(sessions, delta=0.25, min_gap=2.0)
    X = []; vact = []; nact = []; vtype = []; naction = []; grp = []
    for s, segs in ds.items():
        for sid, seg in segs.items():
            for p in ("participant_a", "participant_b"):
                evs = seg["participants"][p]["events"]
                verb = [e for e in evs if len(e) == 2]
                nv = [e for e in evs if len(e) == 3]
                X.append(cell_feat(s, p, seg["t_b"]))
                vact.append(1 if verb else 0)
                nact.append(1 if nv else 0)
                vtype.append(verb[0][0] if verb else "NONE")
                naction.append(nv[0][0] + ":" + nv[0][1] if nv else "NONE")
                grp.append(s)
        print(f"  done session {s} (cells so far {len(X)})", flush=True)
    X = np.stack(X).astype(np.float32)
    vact = np.array(vact); nact = np.array(nact)
    vtype = np.array(vtype); naction = np.array(naction); grp = np.array(grp)
    held = set(sessions[::4]); tr = ~np.isin(grp, list(held)); te = np.isin(grp, list(held))
    print(f"n_cells={len(X)} feat_dim={X.shape[1]} test_cells={te.sum()} "
          f"verbal_active_rate={vact.mean():.3f} nonverbal_active_rate={nact.mean():.3f}", flush=True)

    res = {"n_cells": int(len(X)), "feat_dim": int(X.shape[1]), "test_cells": int(te.sum()),
           "W": W, "n_sessions": len(sessions)}

    res["verbal_active"] = diag_bin(X[tr], X[te], vact[tr], vact[te])
    print("VERBAL-ACTIVE:", res["verbal_active"], flush=True)
    res["nonverbal_active"] = diag_bin(X[tr], X[te], nact[tr], nact[te])
    print("NONVERBAL-ACTIVE:", res["nonverbal_active"], flush=True)

    va_tr = tr & (vact == 1); va_te = te & (vact == 1)
    res["first_verbal_type"] = diag_top(X[va_tr], X[va_te], vtype[va_tr], vtype[va_te])
    print("FIRST-VERBAL-TYPE:", res["first_verbal_type"], flush=True)

    na_tr = tr & (nact == 1); na_te = te & (nact == 1)
    res["first_nonverbal_action"] = diag_top(X[na_tr], X[na_te], naction[na_tr], naction[na_te])
    print("FIRST-NONVERBAL-ACTION:", res["first_nonverbal_action"], flush=True)

    od = os.environ.get("EXP_OUTPUT_DIR", "/tmp")
    json.dump(res, open(os.path.join(od, "result.json"), "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
