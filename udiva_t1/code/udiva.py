#!/usr/bin/env python3
"""UDIVA-HHOI Track 1 core library: faithful GT builder + mAP metric + CV.

VERIFIED: build_gt() reproduces starting_kit/recognition/reference.json 258/258
segments exactly for both known sessions (001080, 181182). See verify_gt.py.

Mapping (reverse-engineered & verified):
  - 2.0s stride grid: window i = [2i, 2(i+1)); last = [2(N-1), maxend], N=ceil(maxend/2).
  - event -> segment by OVERLAP: start < t_e and end > t_b.
  - act='V' -> verbal tuple   (subject, utterance_type, target, modifier)
  - act='NV'-> nonverbal tuple(subject, highlevel_action, lowlevel_action, target, modifier)
  - target = ",".join(target_filtered)
  - per-segment GT = SET of tuples (deduped)
Metric (from competition API):
  - class = utterance_type (verbal) / high_level_action (nonverbal)
  - AP per class via all-point (Pascal-VOC) interpolation; TP requires EXACT all-attr match,
    one-to-one within the same segment; ranked by confidence.
  - mAP computed separately for verbal & nonverbal, then averaged.
"""
import json, math, os
import numpy as np

DATA = "<datasets>/UDIVA-HHOI/development/annotated_sessions"
ANN = f"{DATA}/annotations"

SESSIONS = sorted([f[:-5] for f in os.listdir(ANN) if f.endswith(".json")])  # 21 dev sessions

# ---------------------------------------------------------------- GT building
def load_raw(sid):
    d = json.load(open(f"{ANN}/{sid}.json"))
    return d["annotations"] if isinstance(d, dict) else d

def _target(ev):
    tf = ev.get("target_filtered") or ["none"]
    return ",".join(tf) if tf else "none"

def vtuple(ev):  # verbal
    return (ev["subject"], ev.get("utterance_type", "none"), _target(ev), ev.get("modifier", "none"))

def ntuple(ev):  # nonverbal
    return (ev["subject"], ev.get("high_level_action", "none"), ev.get("low_level_action", "none"),
            _target(ev), ev.get("modifier", "none"))

def seg_grid(maxend):
    n = math.ceil(maxend / 2.0)
    return [(f"s_{i+1:04d}", 2.0 * i, min(2.0 * (i + 1), maxend)) for i in range(n)]

def build_gt(sid, exclude_unintentional=True):
    """Return {'verbal':{seg:(tb,te,set)}, 'nonverbal':{seg:(tb,te,set)}}."""
    ann = load_raw(sid)
    maxend = max(e["end"] for e in ann)
    grid = seg_grid(maxend)
    out = {"verbal": {}, "nonverbal": {}}
    for cat, act, tup in [("verbal", "V", vtuple), ("nonverbal", "NV", ntuple)]:
        evs = [e for e in ann if e.get("act") == act]
        if exclude_unintentional and act == "NV":
            evs = [e for e in evs if e.get("high_level_action") != "unintentional"]
        for sk, tb, te in grid:
            s = set(tup(e) for e in evs if e["start"] < te and e["end"] > tb)
            out[cat][sk] = (tb, te, s)
    return out

def build_all_gt(sessions=None, **kw):
    sessions = sessions or SESSIONS
    return {sid: build_gt(sid, **kw) for sid in sessions}

# class index within each tuple type
VCLASS = 1   # utterance_type
NCLASS = 1   # high_level_action

def gt_class(cat, t):
    return t[VCLASS] if cat == "verbal" else t[NCLASS]

# ---------------------------------------------------------------- metric
def average_precision(confs, istp, n_gt):
    """All-point (VOC) AP. confs: array, istp: bool array (already per-prediction TP), n_gt int."""
    if n_gt == 0:
        return None
    if len(confs) == 0:
        return 0.0
    confs = np.asarray(confs, float); istp = np.asarray(istp, float)
    order = np.argsort(-confs, kind="mergesort")  # stable
    tp = np.cumsum(istp[order]); fp = np.cumsum(1 - istp[order])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-12)
    mrec = np.concatenate([[0.0], rec, [rec[-1]]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

def compute_map(pred, gt, sessions=None, class_set="gt", return_per_class=False):
    """pred[cat][sid][seg] = list of (tuple, confidence). gt = build_all_gt output.
    class_set: 'gt' (classes present in GT) or 'union' (GT u pred classes).
    Returns {'verbal':mAP,'nonverbal':mAP,'overall':mean}."""
    sessions = sessions or list(gt.keys())
    res = {}
    per_class = {"verbal": {}, "nonverbal": {}}
    for cat in ("verbal", "nonverbal"):
        # gather GT tuples by class; and per-(sid,seg) remaining GT sets by class
        gt_by_class = {}            # class -> n_gt
        gt_seg = {}                 # (sid,seg) -> set of GT tuples
        for sid in sessions:
            for seg, (tb, te, s) in gt[sid][cat].items():
                gt_seg[(sid, seg)] = set(s)
                for t in s:
                    gt_by_class[gt_class(cat, t)] = gt_by_class.get(gt_class(cat, t), 0) + 1
        # gather predictions by class
        pred_by_class = {}          # class -> list of (conf, sid, seg, tuple)
        for sid in sessions:
            ps = pred.get(cat, {}).get(sid, {})
            for seg, lst in ps.items():
                # dedup per segment: keep max conf per tuple
                best = {}
                for t, c in lst:
                    if t not in best or c > best[t]:
                        best[t] = c
                for t, c in best.items():
                    pred_by_class.setdefault(gt_class(cat, t), []).append((c, sid, seg, t))
        classes = set(gt_by_class) if class_set == "gt" else set(gt_by_class) | set(pred_by_class)
        aps = {}
        for c in classes:
            preds = sorted(pred_by_class.get(c, []), key=lambda x: -x[0])
            confs, istp = [], []
            used = {}  # (sid,seg) -> set matched
            for conf, sid, seg, t in preds:
                confs.append(conf)
                key = (sid, seg)
                gtset = gt_seg.get(key, set())
                u = used.setdefault(key, set())
                if t in gtset and t not in u and gt_class(cat, t) == c:
                    u.add(t); istp.append(1)
                else:
                    istp.append(0)
            ap = average_precision(confs, istp, gt_by_class.get(c, 0))
            if ap is not None:
                aps[c] = ap
        per_class[cat] = aps
        res[cat] = float(np.mean(list(aps.values()))) if aps else 0.0
    res["overall"] = 0.5 * (res["verbal"] + res["nonverbal"])
    if return_per_class:
        res["per_class"] = per_class
    return res

# ---------------------------------------------------------------- helpers
def gt_as_pred(gt, sessions=None, conf=1.0):
    """Turn GT into a perfect prediction (sanity: mAP should be 1.0 with class_set='gt')."""
    sessions = sessions or list(gt.keys())
    pred = {"verbal": {}, "nonverbal": {}}
    for cat in ("verbal", "nonverbal"):
        for sid in sessions:
            pred[cat][sid] = {seg: [(t, conf) for t in s] for seg, (tb, te, s) in gt[sid][cat].items()}
    return pred

def empty_pred(gt, sessions=None):
    sessions = sessions or list(gt.keys())
    return {cat: {sid: {seg: [] for seg in gt[sid][cat]} for sid in sessions} for cat in ("verbal", "nonverbal")}

if __name__ == "__main__":
    print("sessions:", len(SESSIONS), SESSIONS)
    gt = build_all_gt()
    nseg = sum(len(gt[s]["verbal"]) for s in SESSIONS)
    nv = sum(len(x[2]) for s in SESSIONS for x in gt[s]["verbal"].values())
    nn = sum(len(x[2]) for s in SESSIONS for x in gt[s]["nonverbal"].values())
    print(f"total segments={nseg}  verbal GT tuples={nv}  nonverbal GT tuples={nn}")
