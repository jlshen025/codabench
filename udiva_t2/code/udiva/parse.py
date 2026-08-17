"""Canonical parser: raw 'spotting' annotations -> segment 'recognition' reference format.

Rules (reverse-engineered to EXACTLY reproduce starting_kit/recognition/reference.json
for sessions 001080 and 181182):
  * Segment grid: fixed 2.0s windows [2k, 2k+2), 0-based k -> key s_{k+1:04d}.
  * Assignment: an event with [start,end] is placed in EVERY segment its interval OVERLAPS
    (overlap = end > 2k and start < 2k+2).
  * Verbal (act='V')  -> {subject, utterance_type, target, modifier}
    Nonverbal (act='NV') -> {subject, highlevel_action, lowlevel_action, target, modifier}
  * target = ",".join(target_filtered)  (comma, no space; multi-target kept as one event).
"""
import json, os
from collections import Counter

SEG_LEN = 2.0
VKEYS = ("subject", "utterance_type", "target", "modifier")
NKEYS = ("subject", "highlevel_action", "lowlevel_action", "target", "modifier")


def seg_key(idx):
    return f"s_{idx + 1:04d}"


def _target(e):
    return ",".join(e["target_filtered"])


def parse_session(raw_events, n_segments=None, seg_len=SEG_LEN):
    """raw_events: list of raw annotation event dicts. Returns
    {'verbal': {segkey:{t_b,t_e,events}}, 'nonverbal': {...}}.
    If n_segments given, emits exactly that many segments (trailing empties included)."""
    seg_ev = {"verbal": {}, "nonverbal": {}}
    maxk = -1
    for e in sorted(raw_events, key=lambda x: (x["start"], x["end"])):
        s, en = float(e["start"]), float(e["end"])
        k0 = int(s // seg_len)
        k1 = int(en // seg_len)
        for k in range(k0, k1 + 1):
            tb, te = seg_len * k, seg_len * (k + 1)
            if not (en > tb and s < te):
                continue
            if e["act"] == "V":
                ev = {"subject": e["subject"], "utterance_type": e["utterance_type"],
                      "target": _target(e), "modifier": e["modifier"]}
                stream = "verbal"
            elif e["act"] == "NV":
                ev = {"subject": e["subject"], "highlevel_action": e["high_level_action"],
                      "lowlevel_action": e["low_level_action"], "target": _target(e),
                      "modifier": e["modifier"]}
                stream = "nonverbal"
            else:
                continue
            seg_ev[stream].setdefault(k, []).append(ev)
            maxk = max(maxk, k)
    N = n_segments if n_segments is not None else maxk + 1
    out = {"verbal": {}, "nonverbal": {}}
    for stream in ("verbal", "nonverbal"):
        for k in range(N):
            out[stream][seg_key(k)] = {"t_b": seg_len * k, "t_e": seg_len * (k + 1),
                                       "events": seg_ev[stream].get(k, [])}
    return out


def load_raw(annotations_dir, sid):
    return json.load(open(os.path.join(annotations_dir, f"{sid}.json")))["annotations"]


def ev_multiset(events, keys):
    return Counter(tuple(e[k] for k in keys) for e in events)
