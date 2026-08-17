"""UDIVA-HHOI Track 3 (Exocentric Event Anticipation) — data layer.

Core facts (reverse-engineered & verified against the official starting-kit
anticipation/reference.json on sessions 001080 & 181182):

GT parsing rule for a segment (t_b, t_e] (t_e = t_b + 2.0):
  * include a raw annotation event iff  t_b < event.start <= t_e   (excl lower, incl upper)
  * group by `subject` (participant_a / participant_b)
  * within a participant, order by (event.start, event.end) ascending
  * tuple:  act=='V'  -> [utterance_type, target]            (2-tuple, verbal)
            act=='NV' -> [high_level_action, low_level_action, target]  (3-tuple, non-verbal)
  * target = target_filtered, collapsed: len==1 -> the string; len>1 -> the list

The evaluator infers verbal/non-verbal from tuple LENGTH (2 vs 3). target may be a
str or list[str].

CONFIGURE THESE PATHS (environment variables, or edit the defaults below):

  UDIVA_HHOI_ROOT   the decrypted challenge dataset root, i.e. the directory that holds
                    `development/` and `evaluation/`.        [required]
  UDIVA_T3_WORK     scratch directory for the development-only feature caches written by
                    `dev/feat_extract.py` and `dev/txt_extract.py`. Not needed to
                    reproduce the submission.                [optional]

Nothing else is machine-specific; the submitted pipeline reads only the dataset and
writes only its output zip.
"""
import json, os, glob
from pathlib import Path

HHOI_ROOT = Path(os.environ.get("UDIVA_HHOI_ROOT",
                                "<datasets>/UDIVA-HHOI"))
DATA_ROOT = HHOI_ROOT / "development" / "annotated_sessions"
ANN_DIR   = DATA_ROOT / "annotations"
ANT_REF   = DATA_ROOT / "starting_kit" / "anticipation" / "reference.json"
META_DIR  = DATA_ROOT / "metadata"
TRANS_DIR = DATA_ROOT / "transcripts"
# the released test query grid + submission schema (filled in place by pack.fill_template)
ANT_TEMPLATE = HHOI_ROOT / "evaluation" / "eval_data" / "anticipation_template.json"
WORK_DIR  = Path(os.environ.get("UDIVA_T3_WORK", "<scratch>/t3"))

HORIZON = 2.0
PARTICIPANTS = ["participant_a", "participant_b"]


def list_sessions():
    """All 21 labeled (spotting) session ids that have a raw annotation file."""
    sids = []
    for p in sorted(ANN_DIR.glob("*.json")):
        sids.append(p.stem)
    return sids


_RAW_CACHE = {}


def load_raw(sid):
    """Raw annotation event list for a session, sorted by (start, original idx). Memoized."""
    if sid in _RAW_CACHE:
        return _RAW_CACHE[sid]
    obj = json.load(open(ANN_DIR / f"{sid}.json"))
    anns = obj["annotations"] if isinstance(obj, dict) else obj
    # attach original index for stable ordering
    for i, a in enumerate(anns):
        a["_idx"] = i
    _RAW_CACHE[sid] = anns
    return anns


def collapse_target(tf):
    if not isinstance(tf, list):
        return tf
    if len(tf) == 1:
        return tf[0]
    return list(tf)


def event_tuple(a):
    tgt = collapse_target(a.get("target_filtered", ["none"]))
    if a["act"] == "V":
        return [a["utterance_type"], tgt]
    else:  # NV (or anything else treated as non-verbal 3-tuple)
        return [a["high_level_action"], a["low_level_action"], tgt]


def gt_for_segment(raw, t_b, t_e):
    """Return {participant: [event_tuple, ...]} for one [t_b,t_e) window."""
    out = {p: [] for p in PARTICIPANTS}
    inwin = [a for a in raw if t_b < a["start"] <= t_e]
    inwin.sort(key=lambda a: (a["start"], a["end"]))
    for a in inwin:
        subj = a["subject"]
        if subj not in out:
            continue  # ignore SUPERVISOR etc.
        out[subj].append(event_tuple(a))
    return out


def build_gt(sid, segments):
    """segments: list of (seg_id, t_b, t_e). Returns reference-style dict for the session."""
    raw = load_raw(sid)
    seg_out = {}
    for seg_id, t_b, t_e in segments:
        g = gt_for_segment(raw, t_b, t_e)
        seg_out[seg_id] = {
            "t_b": t_b, "t_e": t_e,
            "participants": {p: {"events": g[p]} for p in PARTICIPANTS},
        }
    return seg_out


def official_grid(sid):
    """(seg_id, t_b, t_e) grid from the official anticipation reference (only 2 sessions)."""
    ref = json.load(open(ANT_REF))["anticipation"]
    if sid not in ref:
        return None
    return [(k, v["t_b"], v["t_e"]) for k, v in ref[sid].items()]


def make_grid(sid, stride=None, t_start=4.0, min_gap=2.0, require_events=True):
    """Generate a segment grid for CV. Non-overlapping 2s windows.

    Default: walk the session, place a window every `min_gap`+HORIZON? No — mirror the
    observed irregular eval grid loosely: step by HORIZON (2s) from t_start, keep
    windows that contain >=1 event for >=1 participant (require_events).
    """
    raw = load_raw(sid)
    if not raw:
        return []
    t_max = max(a["end"] for a in raw)
    step = stride if stride else HORIZON
    segs = []
    t = t_start
    i = 1
    while t + HORIZON <= t_max:
        t_b, t_e = round(t, 3), round(t + HORIZON, 3)
        if require_events:
            has = any(t_b <= a["start"] < t_e and a["subject"] in PARTICIPANTS for a in raw)
            if not has:
                t += step
                continue
        segs.append((f"s_{i:04d}", t_b, t_e))
        i += 1
        t += step
    return segs


# ----------------------------------------------------------------------------
def _verify():
    """Reproduce the official anticipation reference for the 2 provided sessions."""
    ref = json.load(open(ANT_REF))["anticipation"]
    ok = True
    for sid in ref:
        grid = official_grid(sid)
        got = build_gt(sid, grid)
        exp = ref[sid]
        for seg_id in exp:
            for p in PARTICIPANTS:
                e = exp[seg_id]["participants"][p]["events"]
                g = got[seg_id]["participants"][p]["events"]
                if e != g:
                    ok = False
                    print(f"MISMATCH {sid}/{seg_id}/{p}:\n  exp={e}\n  got={g}")
        print(f"  {sid}: {len(exp)} segments checked")
    print("VERIFY:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    print("sessions:", list_sessions())
    print("n sessions:", len(list_sessions()))
    _verify()
