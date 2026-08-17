#!/usr/bin/env python3
"""Pack internal predictions -> submission JSON (mirrors reference.json) -> zip.

Internal pred format:  pred[cat][sid][seg] = list of (tuple, confidence)
  verbal tuple   = (subject, utterance_type, target, modifier)
  nonverbal tuple= (subject, highlevel_action, lowlevel_action, target, modifier)
grids: {sid: {seg: (t_b, t_e)}}  (from build_gt / the eval reference)

Submission JSON shape (per reference.json) with an added confidence field:
  {"verbal":{sid:{seg:{"t_b":..,"t_e":..,"events":[{...attrs.., <conf_field>:c}]}}},
   "nonverbal":{...}}
The exact conf_field name / json filename / inner dir are CONFIGURABLE (calibrate on DEV).
"""
import json, os, zipfile

VKEYS = ["subject", "utterance_type", "target", "modifier"]
NKEYS = ["subject", "highlevel_action", "lowlevel_action", "target", "modifier"]

def event_dict(cat, t, conf, conf_field="score"):
    keys = VKEYS if cat == "verbal" else NKEYS
    d = {k: v for k, v in zip(keys, t)}
    if conf_field:
        d[conf_field] = conf
    return d

def grids_from_gt(gt):
    return {sid: {seg: (tb, te) for seg, (tb, te, s) in gt[sid]["verbal"].items()} for sid in gt}

def to_submission(pred, grids, conf_field="score", include_conf=True):
    sub = {"verbal": {}, "nonverbal": {}}
    for cat in ("verbal", "nonverbal"):
        for sid, segs in grids.items():
            sub[cat].setdefault(sid, {})
            plist = pred.get(cat, {}).get(sid, {})
            for seg, (tb, te) in segs.items():
                # dedup per segment keep max conf
                best = {}
                for t, c in plist.get(seg, []):
                    if t not in best or c > best[t]:
                        best[t] = c
                evs = [event_dict(cat, t, c, conf_field if include_conf else None)
                       for t, c in sorted(best.items(), key=lambda x: -x[1])]
                sub[cat][sid][seg] = {"t_b": tb, "t_e": te, "events": evs}
    return sub

def write_zip(sub, zip_path, json_name="recognition.json", inner_dir=""):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    arc = (inner_dir.rstrip("/") + "/" + json_name) if inner_dir else json_name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(arc, json.dumps(sub))
    return zip_path, arc

def perfect_submission_zip(gt, zip_path, **kw):
    """GT-as-pred (confidence 1.0) -> zip. For the DEV format/metric calibration probe."""
    from udiva import gt_as_pred
    grids = grids_from_gt(gt)
    sub = to_submission(gt_as_pred(gt, conf=1.0), grids, **{k: v for k, v in kw.items()
                                                            if k in ("conf_field", "include_conf")})
    fname = kw.get("json_name", "recognition.json")
    return write_zip(sub, zip_path, json_name=fname, inner_dir=kw.get("inner_dir", ""))
