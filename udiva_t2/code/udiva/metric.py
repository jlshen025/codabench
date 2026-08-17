"""Official UDIVA-HHOI Track-2 recognition mAP (re-implementation).

Metric (from competition Description):
  * AP class = PRIMARY attribute: utterance_type u (verbal), high_level_action h (nonverbal).
  * TP = predicted event matches (one-to-one, WITHIN a segment) an unmatched GT event with ALL
    required attrs equal. Verbal=(subject,utterance_type,target,modifier);
    Nonverbal=(subject,highlevel_action,lowlevel_action,target,modifier).
  * Predictions ranked by confidence (global per class); AP = all-point (VOC) interpolation.
  * mAP^v = mean_u AP_u ; mAP^h = mean_h AP_h ; final mAP = (mAP^v + mAP^h)/2.

Averaging over classes PRESENT IN GT (standard VOC/COCO convention). `average='all_vocab'`
divides instead by the full provided vocabulary (diagnostic alternative).

Predictions/GT structure mirrors reference.json:
  {'verbal': {sid: {segkey: {'events':[{...attrs..., score:float}]}}}, 'nonverbal': {...}}
GT events need no score. Pred events need a confidence under `score_key` (default 'score').
"""
import numpy as np
from collections import defaultdict

VKEYS = ("subject", "utterance_type", "target", "modifier")
NKEYS = ("subject", "highlevel_action", "lowlevel_action", "target", "modifier")
PRIMARY = {"verbal": "utterance_type", "nonverbal": "highlevel_action"}
ATTRS = {"verbal": VKEYS, "nonverbal": NKEYS}


def voc_ap(rec, prec):
    """All-point interpolated AP (PASCAL VOC). rec/prec are arrays in confidence order."""
    if len(rec) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], rec, [rec[-1]]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _drop_unintentional(events, stream):
    if stream != "nonverbal":
        return events
    return [e for e in events if e.get("highlevel_action") != "unintentional"]


def _stream_ap(pred_stream, gt_stream, stream, score_key, drop_unintentional):
    """Return dict {class -> AP} over classes present in GT, plus n_pos per class."""
    keys = ATTRS[stream]
    prim = PRIMARY[stream]
    # GT: per (sid,seg) multiset of full tuples, and per-class positive counts
    gt_seg = {}            # (sid,seg) -> {tuple: remaining_count}
    n_pos = defaultdict(int)
    for sid, segs in gt_stream.items():
        for seg, blk in segs.items():
            evs = _drop_unintentional(blk["events"], stream) if drop_unintentional else blk["events"]
            d = defaultdict(int)
            for e in evs:
                tup = tuple(e[k] for k in keys)
                d[tup] += 1
                n_pos[e[prim]] += 1
            if d:
                gt_seg[(sid, seg)] = d
    # Predictions: list per class of (score, sid, seg, tuple)
    preds = defaultdict(list)
    for sid, segs in pred_stream.items():
        for seg, blk in segs.items():
            evs = _drop_unintentional(blk["events"], stream) if drop_unintentional else blk["events"]
            for e in evs:
                tup = tuple(e[k] for k in keys)
                preds[e[prim]].append((float(e.get(score_key, 0.0)), sid, seg, tup))
    # AP per class present in GT
    aps = {}
    matched = defaultdict(int)  # (sid,seg,tuple) -> matched count, reset per class implicitly
    for c, npos in n_pos.items():
        if npos == 0:
            continue
        plist = sorted(preds.get(c, []), key=lambda x: -x[0])
        used = defaultdict(int)
        tp = np.zeros(len(plist)); fp = np.zeros(len(plist))
        for i, (sc, sid, seg, tup) in enumerate(plist):
            avail = gt_seg.get((sid, seg), {}).get(tup, 0)
            if used[(sid, seg, tup)] < avail:
                used[(sid, seg, tup)] += 1
                tp[i] = 1
            else:
                fp[i] = 1
        if len(plist) == 0:
            aps[c] = 0.0
            continue
        tpc = np.cumsum(tp); fpc = np.cumsum(fp)
        rec = tpc / npos
        prec = tpc / np.maximum(tpc + fpc, 1e-9)
        aps[c] = voc_ap(rec, prec)
    return aps, dict(n_pos)


def score(pred, gt, score_key="score", drop_unintentional=True, average="gt_present",
          full_vocab=None):
    """Return dict with mAP, mAP_verbal, mAP_nonverbal, and per-class AP."""
    out = {}
    per_stream = {}
    for stream in ("verbal", "nonverbal"):
        aps, n_pos = _stream_ap(pred.get(stream, {}), gt.get(stream, {}), stream,
                                score_key, drop_unintentional)
        if average == "all_vocab" and full_vocab is not None:
            denom = len(full_vocab[stream])
            m = sum(aps.values()) / denom if denom else 0.0
        else:
            m = float(np.mean(list(aps.values()))) if aps else 0.0
        per_stream[stream] = m
        out[f"ap_{stream}"] = aps
        out[f"npos_{stream}"] = n_pos
    out["mAP_verbal"] = per_stream["verbal"]
    out["mAP_nonverbal"] = per_stream["nonverbal"]
    out["mAP"] = 0.5 * (per_stream["verbal"] + per_stream["nonverbal"])
    return out
